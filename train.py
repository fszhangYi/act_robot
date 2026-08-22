#!/usr/bin/env python3
"""Train an ACT policy on pre-converted HDF5 episodes.

Reads dataset_info.json written by convert_episodes.py to get the correct
train/val split. Normalisation statistics are computed from training episodes
only, then saved alongside the checkpoint for use during inference.

Usage:
    python act_robot/train.py \\
        --data-dir /data/act_v1 \\
        --ckpt-dir /data/ckpt_v1 \\
        --num-epochs 2000 \\
        --batch-size 64 \\
        --chunk-size 10 \\
        --lr 1e-5 \\
        --kl-weight 10
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from copy import deepcopy
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make detr, policy, dataset importable from this directory
sys.path.insert(0, str(Path(__file__).parent))

from dataset import EpisodicDataset, SAM2EpisodicDataset, get_norm_stats
from policy import (
    ACTPolicy,
    ACTSAM2Policy,
    ACTSAM2CVAEPolicy,
    THROUGHPUT_METRIC_KEYS,
)


# TensorBoard — writer is created in main() so logs land in ckpt_dir/runs/
from torch.utils.tensorboard import SummaryWriter


def _seed_worker(worker_id: int) -> None:
    """保证 DataLoader worker 内的 numpy/random 可复现。"""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def forward_pass(batch, policy):
    # First batch tensor is either image_data (ACTPolicy) or sam2_feat (ACTSAM2Policy).
    # Both policies take (qpos, visual_input, action, is_pad), so the call is uniform.
    visual_data, qpos_data, action_data, is_pad = batch
    visual_data = visual_data.cuda(non_blocking=True)
    qpos_data = qpos_data.cuda(non_blocking=True)
    action_data = action_data.cuda(non_blocking=True)
    is_pad = is_pad.cuda(non_blocking=True)

    output = policy(qpos_data, visual_data, action_data, is_pad)
    return output


def save_optimizer(ckpt_dir, completed_epoch, optimizer, scheduler=None):
    """Save optimizer (and optional scheduler) state alongside checkpoints.

    Args:
        ckpt_dir: Output checkpoint directory.
        completed_epoch: Number of completed epochs (1-indexed).  e.g. 200
            means epochs 0..199 are done, and the next epoch to train is 200.
        optimizer: The optimizer instance.
        scheduler: Optional LR scheduler.
    """
    data = {
        'epoch': completed_epoch,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None:
        data['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(data, os.path.join(ckpt_dir, 'optimizer.pt'))


def _read_gpu_smi() -> dict[str, float]:
    """Query nvidia-smi for basic GPU metrics.  Prints error once on first failure."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=power.draw,temperature.gpu,utilization.gpu,'
             'fan.speed,memory.used',
             '--format=csv,noheader,nounits', '-i', '0'],
            timeout=5,
        )
        parts = out.decode().strip().split(',')
        keys = ['power_w', 'temp_c', 'util_pct', 'fan_pct', 'mem_used_mb']
        values = []
        for p in parts:
            p = p.strip()
            if p in ('[N/A]', 'N/A', '[Not Supported]', ''):
                values.append(0.0)
            else:
                values.append(float(p))
        return dict(zip(keys, values))
    except Exception as e:
        _read_gpu_smi._failed = getattr(_read_gpu_smi, '_failed', False)
        if not _read_gpu_smi._failed:
            _read_gpu_smi._failed = True
            print(f'[WARN] GPU monitoring unavailable: {e}')
        return {}


def _read_sys_metrics(ckpt_dir: str) -> dict[str, float]:
    """CPU memory + disk free (Linux).  Empty on failure."""
    result: dict[str, float] = {}
    try:
        import shutil
        usage = shutil.disk_usage(ckpt_dir)
        result['disk_free_gb'] = usage.free / (1024 ** 3)
    except Exception:
        pass
    try:
        with open('/proc/meminfo') as f:
            lines = f.read()
        for line in lines.splitlines():
            if line.startswith('MemAvailable:'):
                kb = int(line.split()[1])
                result['ram_avail_gb'] = kb / (1024 * 1024)
                break
    except Exception:
        pass
    return result

class TrainerCallback:
    def on_epoch_end(self, epoch, val_loss, best_loss, model, **kwargs):
        pass

class EarlyStoppingCallback(TrainerCallback):
    """根据验证 loss 早停。

    在 callback 内部自行维护 best_loss，避免调用方先把 min_val_loss
    更新成当前 val_loss 后再传入，导致 `val < best - threshold` 几乎永假、
    counter 永不复位、大约 patience 个 epoch 后必定停训的问题。
    """

    def __init__(self, patience=100, threshold=0.002):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best_loss = float('inf')
        self.stop_training = False

    def on_epoch_end(self, epoch, val_loss, best_loss=None, model=None, **kwargs):
        # best_loss 参数保留以兼容接口，实际以 self.best_loss 为准
        print(f'ES: epoch {epoch + 1}, val={val_loss:.6f}, '
              f'best={self.best_loss:.6f}, counter={self.counter}')
        if val_loss < self.best_loss - self.threshold:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
                print(f'Early stopping at epoch {epoch + 1} '
                      f'(patience={self.patience}, best_val={self.best_loss:.6f})')


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    policy: ACTPolicy,
    num_epochs: int,
    ckpt_dir: str,
    seed: int,
    writer: SummaryWriter,
    grad_clip: float = 0.0,
    use_cosine: bool = False,
    min_lr: float = 1e-6,
    resume_from: str | None = None,
    callbacks:list[TrainerCallback]=None,
) -> None:
    if callbacks is None:
        callbacks = []
    optimizer = policy.configure_optimizers()

    # ------------------------------------------------------------------
    # Resume logic
    # ------------------------------------------------------------------
    start_epoch = 0
    train_history: list[dict] = []
    val_history: list[dict] = []
    min_val_loss = float('inf')
    best_state_dict = None

    if resume_from is not None:
        resume_dir = str(Path(resume_from).parent)
        print(f'Loading model weights from: {resume_from}')
        state_dict = torch.load(resume_from, map_location='cpu')
        policy.load_state_dict(state_dict)

        # (b) Load optimizer state (and scheduler if present)
        opt_path = os.path.join(resume_dir, 'optimizer.pt')
        if os.path.exists(opt_path):
            opt_ckpt = torch.load(opt_path, map_location='cpu')
            try:
                optimizer.load_state_dict(opt_ckpt['optimizer_state_dict'])
                start_epoch = int(opt_ckpt.get('epoch', 0))
                saved_lr = optimizer.param_groups[0]['lr']
                print(f'Optimizer state loaded.  Saved LR={saved_lr}  '
                      f'resuming from epoch {start_epoch + 1}')
            except Exception as e:
                print(f'WARNING: Failed to load optimizer state: {e}')
                print('Starting optimizer from scratch.')
                start_epoch = 0
        else:
            print('WARNING: optimizer.pt not found beside checkpoint; '
                  'starting optimizer from scratch.')
            start_epoch = 0

        # (c) Restore training history
        history_path = os.path.join(resume_dir, 'train_history.json')
        if os.path.exists(history_path):
            try:
                with open(history_path) as f:
                    hist = json.load(f)
                train_history = hist.get('train', [])
                val_history = hist.get('val', [])
                if train_history:
                    print(f'Training history loaded: {len(train_history)} epochs')
            except Exception as e:
                print(f'WARNING: Could not load train_history.json: {e}')
        # (d) Restore best model
        best_path = os.path.join(resume_dir, 'policy_best.ckpt')
        if os.path.exists(best_path):
            try:
                best_state_dict = torch.load(best_path, map_location='cpu')
                if val_history:
                    min_val_loss = min(e['loss'] for e in val_history)
                print(f'Best model restored from source.  best_val={min_val_loss:.6f}')
            except Exception as e:
                print(f'WARNING: Could not load policy_best.ckpt: {e}')
        else:
            print('WARNING: policy_best.ckpt not found in source directory; '
                  'best will be re-discovered.')

        # Persist best to new ckpt_dir immediately (guard against overwrite by
        # a worse model re-discovered through fresh val pass)
        if best_state_dict is not None:
            torch.save(best_state_dict, os.path.join(ckpt_dir, 'policy_best.ckpt'))

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    scheduler = None
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=min_lr,
        )
        # Fast-forward the scheduler to match start_epoch
        for _ in range(start_epoch):
            scheduler.step()
        if start_epoch > 0:
            print(f'CosineAnnealingLR advanced to epoch {start_epoch} '
                  f'(T_max={num_epochs}, current LR={scheduler.get_last_lr()[0]:.2e})')


    # ------------------------------------------------------------------
    # Guard: already finished?
    # ------------------------------------------------------------------
    if start_epoch >= num_epochs:
        print(f'Training already completed ({start_epoch} >= {num_epochs}).  '
              f'Increase --num-epochs to continue.')
        # Still persist history if we have it
        if train_history or val_history:
            with open(os.path.join(ckpt_dir, 'train_history.json'), 'w') as f:
                json.dump({'train': train_history, 'val': val_history}, f)
        return

    print(f'Training epochs {start_epoch + 1} → {num_epochs}')

    # 从 resume 恢复的 best_val 同步到 early-stop，避免把历史最优当成「未改进」
    for cb in callbacks:
        if isinstance(cb, EarlyStoppingCallback) and min_val_loss < float('inf'):
            cb.best_loss = min_val_loss

    for epoch in tqdm(range(start_epoch, num_epochs)):
        # ---- training（先训再 val，避免「val → 可能停 → 本轮根本不训」）----
        policy.train()
        optimizer.zero_grad()
        batch_dicts: list[dict] = []
        log_every_n_batch = 20
        tokens_since_log = 0
        encoder_tokens_since_log = 0
        decoder_tokens_since_log = 0
        cvae_tokens_since_log = 0
        samples_since_log = 0
        loss_since_log = 0.0
        n_batches_since_log = 0
        batch_start_time = time.perf_counter()
        # 本 epoch 起点；循环内更新为当前 batch 的全局 step
        n_train_batches = len(train_loader)
        global_step = epoch * n_train_batches

        for idx, batch in enumerate(train_loader):
            fwd = forward_pass(batch, policy)
            fwd['loss'].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            batch_dicts.append({k: v.detach() for k, v in fwd.items()
                                if k not in THROUGHPUT_METRIC_KEYS})

            def _metric_int(key: str) -> int:
                val = fwd.get(key)
                if val is None:
                    return 0
                if torch.is_tensor(val):
                    return int(val.item())
                return int(val)

            tokens_in_batch = _metric_int('num_tokens')
            if tokens_in_batch == 0:
                # Legacy fallback when policy omits throughput fields (e.g. CNNMLP).
                tokens_in_batch = int(
                    batch[0].shape[0] * getattr(policy.model, 'num_queries', 0)
                )

            batch_size = int(batch[0].shape[0])
            tokens_since_log += tokens_in_batch
            encoder_tokens_since_log += _metric_int('num_encoder_tokens')
            decoder_tokens_since_log += _metric_int('num_decoder_tokens')
            cvae_tokens_since_log += _metric_int('num_cvae_encoder_tokens')
            samples_since_log += batch_size
            loss_since_log += float(fwd['loss'].item())
            n_batches_since_log += 1
            global_step = epoch * n_train_batches + idx

            # 满窗口或本 epoch 最后一批都落盘，避免短 epoch / 尾部窗口丢失
            is_log_boundary = ((idx + 1) % log_every_n_batch == 0
                               or (idx + 1) == n_train_batches)
            if is_log_boundary and n_batches_since_log > 0:
                elapsed = time.perf_counter() - batch_start_time
                if elapsed > 0:
                    writer.add_scalar('throughput/token_per_sec',
                                     tokens_since_log / elapsed, global_step)
                    if encoder_tokens_since_log:
                        writer.add_scalar('throughput/encoder_token_per_sec',
                                         encoder_tokens_since_log / elapsed, global_step)
                    if decoder_tokens_since_log:
                        writer.add_scalar('throughput/decoder_token_per_sec',
                                         decoder_tokens_since_log / elapsed, global_step)
                    if cvae_tokens_since_log:
                        writer.add_scalar('throughput/cvae_encoder_token_per_sec',
                                         cvae_tokens_since_log / elapsed, global_step)
                    writer.add_scalar('throughput/sample_per_sec',
                                     samples_since_log / elapsed, global_step)
                # 窗口内平均 loss，而非只记最后一 batch（原先会高估噪声）
                writer.add_scalar('Loss/train',
                                 loss_since_log / n_batches_since_log, global_step)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)

                tokens_since_log = 0
                encoder_tokens_since_log = 0
                decoder_tokens_since_log = 0
                cvae_tokens_since_log = 0
                loss_since_log = 0.0
                n_batches_since_log = 0
                batch_start_time = time.perf_counter()

        if scheduler is not None:
            scheduler.step()
        train_summary = {k: torch.stack([d[k] for d in batch_dicts]).mean().item()
                         for k in batch_dicts[0]}
        if scheduler is not None:
            train_summary['lr'] = scheduler.get_last_lr()[0]
        train_history.append(train_summary)

        # ---- validation（对本轮刚更新的权重评估）----
        policy.eval()
        with torch.inference_mode():
            epoch_dicts = [forward_pass(b, policy) for b in val_loader]
        val_summary = {k: torch.stack([d[k] for d in epoch_dicts]).mean().item()
                       for k in epoch_dicts[0] if k not in THROUGHPUT_METRIC_KEYS}
        val_history.append(val_summary)
        epoch_val_loss = val_summary['loss']
        if epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            best_state_dict = deepcopy(policy.state_dict())
            torch.save(best_state_dict, os.path.join(ckpt_dir, 'policy_best.ckpt'))

        # ---- callbacks / early stopping（在本轮 train+val 之后决定是否停）----
        stop = False
        for cb in callbacks:
            cb.on_epoch_end(epoch, epoch_val_loss, min_val_loss, policy)
            if getattr(cb, 'stop_training', False):
                stop = True

        # Save training curves + last checkpoint every epoch（停训前也落盘）
        with open(os.path.join(ckpt_dir, 'train_history.json'), 'w') as f:
            json.dump({'train': train_history, 'val': val_history}, f)
        torch.save(policy.state_dict(), os.path.join(ckpt_dir, 'policy_last.ckpt'))
        save_optimizer(ckpt_dir, epoch + 1, optimizer, scheduler)

        # epoch 级标量：与本 epoch 最后一个 batch 的 global_step 对齐
        writer.add_scalar('Loss/train_epoch', train_summary['loss'], global_step)
        writer.add_scalar('Loss/val', epoch_val_loss, global_step)
        writer.add_scalar('Loss/best_val', min_val_loss, global_step)
        for key in ('l1', 'kl', 'l2', 'cum_l1'):
            if key in train_summary:
                writer.add_scalar(f'Loss/{key}', train_summary[key], global_step)
            if key in val_summary:
                writer.add_scalar(f'Loss/val_{key}', val_summary[key], global_step)

        gpu = _read_gpu_smi()
        if gpu:
            writer.add_scalar('GPU/power_w', gpu['power_w'], global_step)
            writer.add_scalar('GPU/temp_c', gpu['temp_c'], global_step)
            writer.add_scalar('GPU/util_pct', gpu['util_pct'], global_step)
            writer.add_scalar('GPU/fan_pct', gpu['fan_pct'], global_step)
            writer.add_scalar('GPU/mem_used_gb', gpu['mem_used_mb'] / 1024, global_step)

        sys_metrics = _read_sys_metrics(ckpt_dir)
        if sys_metrics:
            for k, v in sys_metrics.items():
                writer.add_scalar(f'System/{k}', v, global_step)

        if (epoch + 1) % 100 == 0 or epoch == start_epoch:
            print(f'epoch {epoch+1:4d}/{num_epochs}  '
                  f'train={train_summary["loss"]:.4f}  '
                  f'val={epoch_val_loss:.4f}  '
                  f'(best_val={min_val_loss:.4f})')

        if (epoch + 1) % 200 == 0:
            torch.save(policy.state_dict(),
                       os.path.join(ckpt_dir, f'policy_epoch_{epoch+1}_seed_{seed}.ckpt'))
            save_optimizer(ckpt_dir, epoch + 1, optimizer, scheduler)

        if stop:
            break

    # ---- end of training ----
    print(f'\nTraining done.  best_val={min_val_loss:.6f}')
    print(f'Best checkpoint: {os.path.join(ckpt_dir, "policy_best.ckpt")}')
    print(f'Optimizer state: {os.path.join(ckpt_dir, "optimizer.pt")}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', required=True, help='Directory with episode_*.hdf5 + dataset_info.json')
    parser.add_argument('--ckpt-dir', required=True, help='Output checkpoint directory')
    parser.add_argument('--num-epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--chunk-size', type=int, default=10, help='ACT action chunk length (num_queries)')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--kl-weight', type=float, default=10.0)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--dim-feedforward', type=int, default=3200)
    parser.add_argument('--enc-layers', type=int, default=4)
    parser.add_argument('--dec-layers', type=int, default=7)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--camera-names', nargs='+', default=['wrist'],
                        choices=['wrist', 'rear_left', 'chest', 'top', 'wrist_2'],
                        help='Cameras to use for training (default: wrist). Must match what '
                             'convert_episodes.py wrote into the HDF5 + the order serve.py '
                             'expects on the wire.')
    parser.add_argument('--action-space', default='joint',
                        choices=['joint', 'cartesian_abs', 'cartesian'],
                        help='Action representation used when converting data (default: joint)')
    parser.add_argument('--use-sam2-features', action='store_true',
                        help='Train the deterministic SAM2Grasp ACT head from pre-extracted '
                             'F_t (data dir must come from extract_sam2_features.py).')
    parser.add_argument('--action-repr', choices=['absolute', 'delta'], default='absolute',
                        help='SAM2 mode only. `delta` predicts action - qpos[t] (bounded near '
                             'zero by construction); `absolute` predicts the raw target.')

    parser.add_argument('--pool-size', type=int, default=0,
                        help='SAM2 mode only. AdaptiveAvgPool the F_t map to (pool x pool) '
                             "before the transformer. 0 = no pooling (use SAM2's native 64x64). "
                             '16 is recommended to balance qpos:F_t token ratio against ACT baseline.')
    parser.add_argument('--use-cvae', action='store_true',
                        help='SAM2 mode only. Use the paper-aligned CVAE variant (ACTSAM2CVAEPolicy) '
                             'instead of the deterministic ACTSAM2Policy. The CVAE provides the '
                             'KL-regularisation that prevents OOD chunk[0] hallucinations.')
    parser.add_argument('--cumulative-loss-weight', type=float, default=0.0,
                        help='SAM2 CVAE + cartesian(delta) mode. Weight on the cumulative-trajectory '
                             'consistency loss (cumsum of pose deltas over the chunk). >0 penalises '
                             'systematic delta under-prediction (mode collapse). Try 1.0.')
    parser.add_argument('--grad-clip', type=float, default=0.0,
                        help='Max global grad norm (0 = no clipping). 1.0 is a sane default.')
    parser.add_argument('--cosine-lr', action='store_true',
                        help='Cosine-anneal LR from --lr down to --min-lr over num_epochs.')
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--resume-from', default=None,
                        help='Path to a policy checkpoint (e.g. policy_last.ckpt) to resume '
                             'training from.  The optimizer.pt file in the same directory is '
                             'loaded automatically if present.')
    parser.add_argument('--early-stop-patience', type=int, default=100,
                        help='验证 loss 连续多少个 epoch 无显著下降则停训。'
                             '0 = 关闭 early stopping。ACT 常需上千 epoch，默认 100。')
    parser.add_argument('--early-stop-threshold', type=float, default=0.002,
                        help='判定「有改进」的最小 val loss 降幅（仅 early-stop-patience>0 时生效）')

    args = parser.parse_args()

    # Early validation: fail fast if the resume checkpoint doesn't exist
    if args.resume_from is not None:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f'Resume checkpoint not found: {args.resume_from}')
        print(f'Resume mode: checkpoint={args.resume_from}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # TensorBoard writer — logs go into ckpt_dir/runs/
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'runs'))

    # Load dataset split info
    info_path = os.path.join(data_dir, 'dataset_info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f'dataset_info.json not found in {data_dir}. Run convert_episodes.py first.')
    with open(info_path) as f:
        info = json.load(f)
    rx_unwrapped = bool(info.get('rx_unwrapped', False))

    train_indices = info['train_indices']
    val_indices = info['val_indices']
    max_episode_len = info['max_episode_len']
    print(f'Dataset: {info["num_total"]} episodes  |  train={len(train_indices)}  val={len(val_indices)}')
    print(f'max_episode_len={max_episode_len}  stride={info.get("stride", 1)}')


    # Normalization stats from training set only
    print('Computing normalization statistics from training set...')
    norm_stats = get_norm_stats(data_dir, train_indices, chunk_size=args.chunk_size)
    state_dim = norm_stats['state_dim']
    print(f'state_dim={state_dim}')
    print(f'action_mean={norm_stats["action_mean"]}')
    print(f'action_std ={norm_stats["action_std"]}')
    print(f'delta_mean ={norm_stats["delta_mean"]}')
    print(f'delta_std  ={norm_stats["delta_std"]}')

    with open(os.path.join(ckpt_dir, 'dataset_stats.pkl'), 'wb') as f:
        pickle.dump(norm_stats, f)

    camera_names = args.camera_names
    if args.use_sam2_features:
        train_dataset = SAM2EpisodicDataset(train_indices, data_dir, norm_stats,
                                            max_episode_len, action_repr=args.action_repr,
                                            chunk_size=args.chunk_size)
        val_dataset = SAM2EpisodicDataset(val_indices, data_dir, norm_stats,
                                          max_episode_len, action_repr=args.action_repr,
                                          chunk_size=args.chunk_size)
    else:
        train_dataset = EpisodicDataset(train_indices, data_dir, camera_names, norm_stats,
                                        max_episode_len, chunk_size=args.chunk_size)
        val_dataset = EpisodicDataset(val_indices, data_dir, camera_names, norm_stats,
                                      max_episode_len, chunk_size=args.chunk_size)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(2, args.num_workers) if args.num_workers > 0 else 0,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
    )

    if args.use_sam2_features:
        policy_config = {
            'lr': args.lr,
            'weight_decay': 1e-4,
            'num_queries': args.chunk_size,
            'hidden_dim': args.hidden_dim,
            'dim_feedforward': args.dim_feedforward,
            'enc_layers': args.enc_layers,
            'dec_layers': args.dec_layers,
            'nheads': args.nheads,
            'sam2_feat_dim': 256,
            'state_dim': state_dim,
            'action_dim': state_dim,
            'pool_size': args.pool_size if args.pool_size > 0 else None,
            'action_repr': args.action_repr,
            'use_cvae': args.use_cvae,
            'kl_weight': args.kl_weight if args.use_cvae else 0.0,
            'rx_unwrapped': rx_unwrapped,
            'cumulative_loss_weight': args.cumulative_loss_weight,
        }
        if args.use_cvae:
            if args.action_repr != 'absolute':
                raise SystemExit('--use-cvae is paper-aligned; only --action-repr absolute is supported with CVAE.')
            policy = ACTSAM2CVAEPolicy(policy_config)
        else:
            policy = ACTSAM2Policy(policy_config)
    else:
        policy_config = {
            'lr': args.lr,
            'lr_backbone': 1e-5,
            'num_queries': args.chunk_size,
            'kl_weight': args.kl_weight,
            'hidden_dim': args.hidden_dim,
            'dim_feedforward': args.dim_feedforward,
            'enc_layers': args.enc_layers,
            'dec_layers': args.dec_layers,
            'nheads': args.nheads,
            'backbone': 'resnet18',
            'camera_names': camera_names,
            'state_dim': state_dim,
            'rx_unwrapped': rx_unwrapped,
        }
        policy = ACTPolicy(policy_config)

    # Save config for reference
    with open(os.path.join(ckpt_dir, 'policy_config.json'), 'w') as f:
        json.dump({**policy_config, 'chunk_size': args.chunk_size,
                   'num_epochs': args.num_epochs, 'batch_size': args.batch_size,
                   'seed': args.seed, 'action_space': args.action_space,
                   'use_sam2_features': args.use_sam2_features}, f, indent=2)

    policy.cuda()

    callbacks: list[TrainerCallback] = []
    if args.early_stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            patience=args.early_stop_patience,
            threshold=args.early_stop_threshold,
        ))
        print(f'Early stopping: patience={args.early_stop_patience}  '
              f'threshold={args.early_stop_threshold}')
    else:
        print('Early stopping: disabled (--early-stop-patience 0)')

    train(train_loader, val_loader, policy, args.num_epochs, ckpt_dir, args.seed,
          writer,
          grad_clip=args.grad_clip, use_cosine=args.cosine_lr, min_lr=args.min_lr,
          resume_from=args.resume_from,
          callbacks=callbacks)
    writer.close()


if __name__ == '__main__':
    main()
    