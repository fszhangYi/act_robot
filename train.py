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
from policy import ACTPolicy, ACTSAM2Policy, ACTSAM2CVAEPolicy


# Tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

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
    return policy(qpos_data, visual_data, action_data, is_pad)


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
    """Query nvidia-smi for GPU 0 crash-diagnostic metrics.  Empty on failure."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=power.draw,temperature.gpu,utilization.gpu,fan.speed,'
             'memory.used,ecc.errors.corrected.volatile.total,'
             'ecc.errors.uncorrected.volatile.total,clocks_throttle_reasons.active',
             '--format=csv,noheader,nounits'],
            timeout=5,
        )
        parts = out.decode().strip().split(',')
        keys = ['power_w', 'temp_c', 'util_pct', 'fan_pct', 'mem_used_mb',
                'ecc_corrected', 'ecc_uncorrected', 'throttle']
        values = []
        for p in parts:
            p = p.strip()
            if p in ('[N/A]', 'N/A', '[Not Supported]', ''):
                values.append(0.0)
            else:
                values.append(float(p))
        return dict(zip(keys, values))
    except Exception:
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


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    policy: ACTPolicy,
    num_epochs: int,
    ckpt_dir: str,
    seed: int,
    grad_clip: float = 0.0,
    use_cosine: bool = False,
    min_lr: float = 1e-6,
    resume_from: str | None = None,
) -> None:
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

    for epoch in tqdm(range(start_epoch, num_epochs)):
        # ---- validation ----
        policy.eval()
        with torch.inference_mode():
            epoch_dicts = [forward_pass(b, policy) for b in val_loader]
        val_summary = {k: torch.stack([d[k] for d in epoch_dicts]).mean().item()
                       for k in epoch_dicts[0]}
        val_history.append(val_summary)
        epoch_val_loss = val_summary['loss']
        if epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            best_state_dict = deepcopy(policy.state_dict())
            torch.save(best_state_dict, os.path.join(ckpt_dir, 'policy_best.ckpt'))

        # ---- training ----
        policy.train()
        optimizer.zero_grad()
        batch_dicts: list[dict] = []
        t0 = time.time()
        for idx, batch in enumerate(train_loader):
            t_data = time.time()
            print(f"Batch {idx}: Data loading took {t_data-t0:.3f}s")
            t0 = time.time()
            fwd = forward_pass(batch, policy)
            fwd['loss'].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            # ------- every N batch record once -----
            global_step = epoch * len(train_loader) + idx

            if idx % 20 == 0:
                writer.add_scalar("Loss/train", fwd['loss'].item(), global_step)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step)
                gpu = _read_gpu_smi()
                if gpu:
                    writer.add_scalar('GPU/power_w',        gpu['power_w'],        global_step)
                    writer.add_scalar('GPU/temp_c',          gpu['temp_c'],          global_step)
                    writer.add_scalar('GPU/util_pct',        gpu['util_pct'],        global_step)
                    writer.add_scalar('GPU/fan_pct',         gpu['fan_pct'],         global_step)
                    writer.add_scalar('GPU/mem_used_gb',     gpu['mem_used_mb'] / 1024, global_step)
            
            batch_dicts.append({k: v.detach() for k, v in fwd.items()})
        if scheduler is not None:
            scheduler.step()
        train_summary = {k: torch.stack([d[k] for d in batch_dicts]).mean().item()
                         for k in batch_dicts[0]}
        if scheduler is not None:
            train_summary['lr'] = scheduler.get_last_lr()[0]
        train_history.append(train_summary)

        # Save training curves + last checkpoint every epoch (safe against interrupts)
        with open(os.path.join(ckpt_dir, 'train_history.json'), 'w') as f:
            json.dump({'train': train_history, 'val': val_history}, f)
        torch.save(policy.state_dict(), os.path.join(ckpt_dir, 'policy_last.ckpt'))
        save_optimizer(ckpt_dir, epoch + 1, optimizer, scheduler)

        # ---- per-epoch TensorBoard logging (slow-changing metrics) ----
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        gpu = _read_gpu_smi()
        if gpu:
            writer.add_scalar('GPU/ecc_corrected',   gpu['ecc_corrected'],   epoch)
            writer.add_scalar('GPU/ecc_uncorrected', gpu['ecc_uncorrected'], epoch)
            writer.add_scalar('GPU/throttle',        gpu['throttle'],        epoch)
        sys_metrics = _read_sys_metrics(ckpt_dir)
        if sys_metrics:
            for k, v in sys_metrics.items():
                writer.add_scalar(f'System/{k}', v, epoch)

        if (epoch + 1) % 100 == 0 or epoch == start_epoch:
            print(f'epoch {epoch+1:4d}/{num_epochs}  '
                  f'train={train_summary["loss"]:.4f}  '
                  f'val={epoch_val_loss:.4f}  '
                  f'(best_val={min_val_loss:.4f})')

        if (epoch + 1) % 200 == 0:
            torch.save(policy.state_dict(),
                       os.path.join(ckpt_dir, f'policy_epoch_{epoch+1}_seed_{seed}.ckpt'))
            save_optimizer(ckpt_dir, epoch + 1, optimizer, scheduler)

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
                                            max_episode_len, action_repr=args.action_repr)
        val_dataset = SAM2EpisodicDataset(val_indices, data_dir, norm_stats,
                                          max_episode_len, action_repr=args.action_repr)
    else:
        train_dataset = EpisodicDataset(train_indices, data_dir, camera_names, norm_stats, max_episode_len)
        val_dataset = EpisodicDataset(val_indices, data_dir, camera_names, norm_stats, max_episode_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

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

    train(train_loader, val_loader, policy, args.num_epochs, ckpt_dir, args.seed,
          grad_clip=args.grad_clip, use_cosine=args.cosine_lr, min_lr=args.min_lr,
          resume_from=args.resume_from)


if __name__ == '__main__':
    main()
    