#!/usr/bin/env python3
"""Clean re-bench of top (B,W,C) combos after GPU is empty."""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

ROOT = Path('/root/autodl-tmp/act_robot')
sys.path.insert(0, str(ROOT))
from dataset import EpisodicDataset, get_norm_stats
from policy import ACTPolicy
from train import _seed_worker, forward_pass

DATA = ROOT / 'data/home/ubuntu/act/convert_out_stride2/convert_out_stride2'
STATS = Path('/tmp/bench_norm_stats.pkl')
CAMS = ['chest', 'top', 'wrist_2']


def peak_rss(proc: psutil.Process) -> float:
    total = proc.memory_info().rss
    for c in proc.children(recursive=True):
        try:
            total += c.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / (1024 ** 3)


def main() -> None:
    info = json.loads((DATA / 'dataset_info.json').read_text())
    if STATS.exists():
        norm = pickle.loads(STATS.read_bytes())
        print('loaded cached norm stats', flush=True)
    else:
        print('computing norm stats...', flush=True)
        norm = get_norm_stats(str(DATA), info['train_indices'], chunk_size=10)
        STATS.write_bytes(pickle.dumps(norm))

    cfg = dict(
        lr=1e-5, lr_backbone=1e-5, num_queries=10, kl_weight=10.0,
        hidden_dim=512, dim_feedforward=3200, enc_layers=4, dec_layers=7,
        nheads=8, backbone='resnet18', camera_names=CAMS,
        state_dim=norm['state_dim'],
        rx_unwrapped=bool(info.get('rx_unwrapped', False)),
    )

    def run(B: int, W: int, C: int, steps: int = 40) -> dict:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        proc = psutil.Process()
        train_ds = EpisodicDataset(
            info['train_indices'], str(DATA), CAMS, norm,
            info['max_episode_len'], chunk_size=10, hdf5_cache_size=C,
        )
        gen = torch.Generator()
        gen.manual_seed(0)
        loader = DataLoader(
            train_ds, batch_size=B, shuffle=True, num_workers=W,
            pin_memory=True, persistent_workers=W > 0,
            worker_init_fn=_seed_worker if W > 0 else None, generator=gen,
        )
        policy = ACTPolicy(cfg)
        policy.cuda()
        opt = policy.configure_optimizers()
        policy.train()
        it = iter(loader)
        b = next(it)
        fwd = forward_pass(b, policy)
        fwd['loss'].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()

        samples = 0
        peak = peak_rss(proc)
        t0 = time.perf_counter()
        for i in range(steps):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            fwd = forward_pass(b, policy)
            fwd['loss'].backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            samples += int(b[0].shape[0])
            if (i + 1) % 10 == 0:
                peak = max(peak, peak_rss(proc))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = max(peak, peak_rss(proc))
        gpu = torch.cuda.max_memory_allocated() / (1024 ** 3)
        sps = samples / elapsed
        print(
            f'B={B:2d} W={W:2d} C={C:2d}  {sps:6.2f} samp/s  '
            f'{elapsed / steps:.3f}s/step  RSS={peak:.1f}GB  GPU={gpu:.1f}GB',
            flush=True,
        )
        del loader, policy, opt, it, train_ds
        torch.cuda.empty_cache()
        return dict(B=B, W=W, C=C, sps=sps, rss=peak, gpu=gpu)

    combos = [
        (16, 8, 8),
        (16, 10, 8),
        (16, 12, 8),
        (16, 16, 8),
        (16, 16, 16),
        (24, 8, 8),
        (24, 12, 8),
        (40, 8, 8),
        (40, 12, 8),
    ]
    rows = []
    for combo in combos:
        try:
            rows.append(run(*combo))
        except torch.cuda.OutOfMemoryError:
            print(f'B={combo[0]} W={combo[1]} C={combo[2]}  CUDA OOM', flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'B={combo[0]} W={combo[1]} C={combo[2]}  FAIL {e}', flush=True)
            torch.cuda.empty_cache()

    print('\nBEST:')
    for r in sorted(rows, key=lambda x: -x['sps']):
        print(
            f"  B={r['B']} W={r['W']} C={r['C']}  {r['sps']:.2f} samp/s  "
            f"RSS={r['rss']:.1f}GB GPU={r['gpu']:.1f}GB"
        )


if __name__ == '__main__':
    main()
