#!/usr/bin/env python3
"""Short end-to-end train throughput / peak-RSS probe for hyperparam search.

Runs a few train + val steps (not full epochs) for each (batch, workers, cache)
combo and reports sample/sec + peak process-tree RSS.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import EpisodicDataset, get_norm_stats  # noqa: E402
from policy import ACTPolicy  # noqa: E402
from train import _seed_worker, forward_pass  # noqa: E402


def peak_rss_gb(proc: psutil.Process) -> float:
    total = proc.memory_info().rss
    for c in proc.children(recursive=True):
        try:
            total += c.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / (1024 ** 3)


def build_loaders(data_dir, info, norm_stats, camera_names, chunk_size,
                  batch_size, num_workers, hdf5_cache_size, seed):
    train_ds = EpisodicDataset(
        info['train_indices'], data_dir, camera_names, norm_stats,
        info['max_episode_len'], chunk_size=chunk_size,
        hdf5_cache_size=hdf5_cache_size,
    )
    val_ds = EpisodicDataset(
        info['val_indices'], data_dir, camera_names, norm_stats,
        info['max_episode_len'], chunk_size=chunk_size,
        hdf5_cache_size=hdf5_cache_size,
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=min(2, num_workers) if num_workers > 0 else 0,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )
    return train_loader, val_loader


def run_combo(args, info, norm_stats, policy_config, combo) -> dict:
    batch_size, num_workers, hdf5_cache = combo
    tag = f'B={batch_size} W={num_workers} C={hdf5_cache}'
    print(f'\n=== {tag} ===', flush=True)

    # Fresh process stats
    proc = psutil.Process(os.getpid())
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        train_loader, val_loader = build_loaders(
            args.data_dir, info, norm_stats, args.camera_names,
            args.chunk_size, batch_size, num_workers, hdf5_cache, args.seed,
        )
        policy = ACTPolicy(policy_config)
        policy.cuda()
        optimizer = policy.configure_optimizers()

        # Warmup: build workers + first batch
        policy.train()
        it = iter(train_loader)
        batch = next(it)
        fwd = forward_pass(batch, policy)
        fwd['loss'].backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        # Timed train steps
        n_steps = args.train_steps
        samples = 0
        t0 = time.perf_counter()
        peak = peak_rss_gb(proc)
        for i in range(n_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            fwd = forward_pass(batch, policy)
            fwd['loss'].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            samples += int(batch[0].shape[0])
            if (i + 1) % 5 == 0:
                peak = max(peak, peak_rss_gb(proc))
        torch.cuda.synchronize()
        train_elapsed = time.perf_counter() - t0
        peak = max(peak, peak_rss_gb(proc))

        # Timed val pass (limited batches)
        policy.eval()
        val_it = iter(val_loader)
        val_batches = 0
        val_samples = 0
        t1 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.val_steps):
                try:
                    batch = next(val_it)
                except StopIteration:
                    break
                forward_pass(batch, policy)
                val_batches += 1
                val_samples += int(batch[0].shape[0])
        torch.cuda.synchronize()
        val_elapsed = time.perf_counter() - t1
        peak = max(peak, peak_rss_gb(proc))

        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()

        result = {
            'ok': True,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'hdf5_cache_size': hdf5_cache,
            'train_steps': n_steps,
            'train_samples': samples,
            'train_sec': train_elapsed,
            'sample_per_sec': samples / train_elapsed if train_elapsed > 0 else 0,
            'sec_per_step': train_elapsed / n_steps,
            'val_batches': val_batches,
            'val_samples': val_samples,
            'val_sec': val_elapsed,
            'val_sample_per_sec': val_samples / val_elapsed if val_elapsed > 0 else 0,
            'peak_rss_gb': peak,
            'peak_gpu_gb': gpu_mem,
            'ru_maxrss_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
        }
        print(
            f'  train {result["sample_per_sec"]:.2f} samp/s  '
            f'({result["sec_per_step"]:.3f} s/step)  '
            f'val {result["val_sample_per_sec"]:.2f} samp/s  '
            f'peak_RSS={peak:.1f}GB  peak_GPU={gpu_mem:.1f}GB',
            flush=True,
        )

        # Tear down loaders so persistent workers die before next combo
        del train_loader, val_loader, policy, optimizer, it, val_it
        torch.cuda.empty_cache()
        return result

    except torch.cuda.OutOfMemoryError as e:
        print(f'  CUDA OOM: {e}', flush=True)
        torch.cuda.empty_cache()
        return {
            'ok': False, 'error': 'cuda_oom',
            'batch_size': batch_size, 'num_workers': num_workers,
            'hdf5_cache_size': hdf5_cache,
            'peak_rss_gb': peak_rss_gb(proc),
        }
    except Exception as e:
        print(f'  FAIL: {type(e).__name__}: {e}', flush=True)
        torch.cuda.empty_cache()
        return {
            'ok': False, 'error': f'{type(e).__name__}: {e}',
            'batch_size': batch_size, 'num_workers': num_workers,
            'hdf5_cache_size': hdf5_cache,
            'peak_rss_gb': peak_rss_gb(proc),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--camera-names', nargs='+', default=['chest', 'top', 'wrist_2'])
    parser.add_argument('--chunk-size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--kl-weight', type=float, default=10.0)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--dim-feedforward', type=int, default=3200)
    parser.add_argument('--enc-layers', type=int, default=4)
    parser.add_argument('--dec-layers', type=int, default=7)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--train-steps', type=int, default=30,
                        help='Timed train steps after 1 warmup step')
    parser.add_argument('--val-steps', type=int, default=10)
    parser.add_argument('--out', type=Path, default=Path('/tmp/bench_train_throughput.json'))
    args = parser.parse_args()

    info = json.loads(Path(args.data_dir, 'dataset_info.json').read_text())
    print('Computing norm stats (train set)...', flush=True)
    norm_stats = get_norm_stats(args.data_dir, info['train_indices'], chunk_size=args.chunk_size)
    state_dim = norm_stats['state_dim']
    rx_unwrapped = bool(info.get('rx_unwrapped', False))

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
        'camera_names': args.camera_names,
        'state_dim': state_dim,
        'rx_unwrapped': rx_unwrapped,
    }

    # (batch, workers, cache) — sweep order: find safe GPU batch first, then workers
    combos = [
        # baseline from user
        (16, 16, 16),
        (16, 8, 16),
        (16, 8, 8),
        (16, 4, 16),
        (16, 4, 8),
        (16, 2, 16),
        (16, 2, 8),
        # larger batch (GPU-bound hope)
        (32, 8, 16),
        (32, 4, 16),
        (32, 8, 8),
        (48, 8, 16),
        (48, 4, 16),
        (64, 8, 16),
        (64, 4, 16),
        # extreme workers at best batch
        (16, 12, 8),
        (32, 12, 8),
        (32, 16, 8),
    ]

    results = []
    for combo in combos:
        r = run_combo(args, info, norm_stats, policy_config, combo)
        results.append(r)
        args.out.write_text(json.dumps(results, indent=2))

    ok = [r for r in results if r.get('ok')]
    print('\n========== RANKING (by train sample/sec) ==========')
    for r in sorted(ok, key=lambda x: -x['sample_per_sec']):
        print(
            f"  B={r['batch_size']:2d} W={r['num_workers']:2d} C={r['hdf5_cache_size']:2d}  "
            f"{r['sample_per_sec']:6.2f} samp/s  "
            f"RSS={r['peak_rss_gb']:5.1f}GB  GPU={r['peak_gpu_gb']:4.1f}GB"
        )
    fail = [r for r in results if not r.get('ok')]
    if fail:
        print('\nFAILED:')
        for r in fail:
            print(f"  B={r['batch_size']} W={r['num_workers']} C={r['hdf5_cache_size']}: {r.get('error')}")
    print(f'\nWrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
