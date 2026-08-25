#!/usr/bin/env python3
"""Cartesian-product micro-benchmark for train throughput hyperparams.

Reads fixed train-like knobs from --base-json and a sweep grid from
--sweep-json, then ranks combos by train sample/sec (OOM eliminated).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import EpisodicDataset, get_norm_stats  # noqa: E402
from policy import ACTPolicy  # noqa: E402
from train import _seed_worker, forward_pass  # noqa: E402

SWEEP_KEYS_AFFECTING_LOADER = frozenset({'batch-size', 'num-workers', 'hdf5-cache-size'})


def peak_rss_gb(proc: psutil.Process) -> float:
    total = proc.memory_info().rss
    for c in proc.children(recursive=True):
        try:
            total += c.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / (1024 ** 3)


def _as_list(v: Any) -> list[Any]:
    if isinstance(v, list):
        return v
    return [v]


def _parse_camera_names(base: dict[str, Any]) -> list[str]:
    raw = base.get('camera-names', ['chest', 'top', 'wrist_2'])
    if isinstance(raw, str):
        return raw.split()
    return [str(x) for x in raw]


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


def run_combo(
    *,
    data_dir: str,
    info: dict,
    norm_stats: dict,
    policy_config: dict,
    camera_names: list[str],
    chunk_size: int,
    seed: int,
    grad_clip: float,
    train_steps: int,
    val_steps: int,
    combo: dict[str, Any],
) -> dict[str, Any]:
    batch_size = int(combo['batch-size'])
    num_workers = int(combo['num-workers'])
    hdf5_cache = int(combo['hdf5-cache-size'])
    tag = ' '.join(f'{k}={combo[k]}' for k in sorted(combo))
    print(f'\n=== {tag} ===', flush=True)

    proc = psutil.Process(os.getpid())
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        train_loader, val_loader = build_loaders(
            data_dir, info, norm_stats, camera_names,
            chunk_size, batch_size, num_workers, hdf5_cache, seed,
        )
        policy = ACTPolicy(policy_config)
        policy.cuda()
        optimizer = policy.configure_optimizers()

        policy.train()
        it = iter(train_loader)
        batch = next(it)
        fwd = forward_pass(batch, policy)
        fwd['loss'].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        samples = 0
        t0 = time.perf_counter()
        peak = peak_rss_gb(proc)
        for i in range(train_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            fwd = forward_pass(batch, policy)
            fwd['loss'].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            samples += int(batch[0].shape[0])
            if (i + 1) % 5 == 0:
                peak = max(peak, peak_rss_gb(proc))
        torch.cuda.synchronize()
        train_elapsed = time.perf_counter() - t0
        peak = max(peak, peak_rss_gb(proc))

        policy.eval()
        val_it = iter(val_loader)
        val_batches = 0
        val_samples = 0
        t1 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(val_steps):
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
            'combo': combo,
            'train_steps': train_steps,
            'train_samples': samples,
            'train_sec': train_elapsed,
            'sample_per_sec': samples / train_elapsed if train_elapsed > 0 else 0.0,
            'sec_per_step': train_elapsed / train_steps if train_steps else 0.0,
            'val_batches': val_batches,
            'val_samples': val_samples,
            'val_sec': val_elapsed,
            'val_sample_per_sec': val_samples / val_elapsed if val_elapsed > 0 else 0.0,
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
        del train_loader, val_loader, policy, optimizer, it, val_it
        torch.cuda.empty_cache()
        return result

    except torch.cuda.OutOfMemoryError as e:
        print(f'  CUDA OOM: {e}', flush=True)
        torch.cuda.empty_cache()
        return {
            'ok': False, 'error': 'cuda_oom', 'combo': combo,
            'peak_rss_gb': peak_rss_gb(proc),
        }
    except Exception as e:
        print(f'  FAIL: {type(e).__name__}: {e}', flush=True)
        torch.cuda.empty_cache()
        return {
            'ok': False, 'error': f'{type(e).__name__}: {e}', 'combo': combo,
            'peak_rss_gb': peak_rss_gb(proc),
        }


def expand_combos(base: dict[str, Any], sweep: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge fixed base with cartesian product of sweep lists.

    Loader knobs always present (from base if not swept).
    """
    sweep = {k.lstrip('-'): _as_list(v) for k, v in sweep.items()}
    if not sweep:
        raise SystemExit('sweep-json must contain at least one key')

    keys = sorted(sweep.keys())
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*(sweep[k] for k in keys)):
        combo = {k: values[i] for i, k in enumerate(keys)}
        # Ensure loader knobs exist for the bench harness
        for req in ('batch-size', 'num-workers', 'hdf5-cache-size'):
            if req not in combo:
                if req not in base:
                    raise SystemExit(f'missing {req} in sweep and base-json')
                combo[req] = base[req]
        combos.append(combo)
    return combos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--base-json', required=True,
                        help='JSON object of fixed train kwargs (flag names without leading --)')
    parser.add_argument('--sweep-json', required=True,
                        help='JSON object: flag -> list of candidate values')
    parser.add_argument('--train-steps', type=int, default=30)
    parser.add_argument('--val-steps', type=int, default=10)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--norm-stats-cache', type=Path, default=None,
                        help='Optional pickle cache for norm stats')
    args = parser.parse_args()

    base = json.loads(args.base_json)
    if isinstance(base, str):
        base = json.loads(base)
    sweep = json.loads(args.sweep_json)
    if isinstance(sweep, str):
        sweep = json.loads(sweep)
    base = {str(k).lstrip('-'): v for k, v in base.items()}
    sweep = {str(k).lstrip('-'): v for k, v in sweep.items()}

    combos = expand_combos(base, sweep)
    print(f'Combos to try: {len(combos)}', flush=True)
    for c in combos:
        print(f'  - {c}', flush=True)

    data_dir = args.data_dir
    info = json.loads(Path(data_dir, 'dataset_info.json').read_text())
    chunk_size = int(base.get('chunk-size', 10))
    seed = int(base.get('seed', 0))
    grad_clip = float(base.get('grad-clip', 0.0))
    camera_names = _parse_camera_names(base)

    cache = args.norm_stats_cache
    if cache and cache.is_file():
        print(f'Loading norm stats cache {cache}', flush=True)
        norm_stats = pickle.loads(cache.read_bytes())
    else:
        print('Computing norm stats (train set)...', flush=True)
        norm_stats = get_norm_stats(data_dir, info['train_indices'], chunk_size=chunk_size)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps(norm_stats))

    policy_config = {
        'lr': float(base.get('lr', 1e-5)),
        'lr_backbone': 1e-5,
        'num_queries': chunk_size,
        'kl_weight': float(base.get('kl-weight', 10.0)),
        'hidden_dim': int(base.get('hidden-dim', 512)),
        'dim_feedforward': int(base.get('dim-feedforward', 3200)),
        'enc_layers': int(base.get('enc-layers', 4)),
        'dec_layers': int(base.get('dec-layers', 7)),
        'nheads': int(base.get('nheads', 8)),
        'backbone': 'resnet18',
        'camera_names': camera_names,
        'state_dim': norm_stats['state_dim'],
        'rx_unwrapped': bool(info.get('rx_unwrapped', False)),
    }

    results: list[dict[str, Any]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for combo in combos:
        r = run_combo(
            data_dir=data_dir,
            info=info,
            norm_stats=norm_stats,
            policy_config=policy_config,
            camera_names=camera_names,
            chunk_size=chunk_size,
            seed=seed,
            grad_clip=grad_clip,
            train_steps=args.train_steps,
            val_steps=args.val_steps,
            combo=combo,
        )
        results.append(r)
        payload = {
            'ok': True,
            'data_dir': data_dir,
            'base': base,
            'sweep': sweep,
            'results': results,
            'best': None,
        }
        ok = [x for x in results if x.get('ok')]
        if ok:
            best = max(ok, key=lambda x: x['sample_per_sec'])
            # Prefer lower RSS on near-ties (~1% sps)
            top_sps = best['sample_per_sec']
            near = [x for x in ok if x['sample_per_sec'] >= top_sps * 0.99]
            best = min(near, key=lambda x: (x.get('peak_rss_gb') or 1e9, -x['sample_per_sec']))
            payload['best'] = {
                'combo': best['combo'],
                'sample_per_sec': best['sample_per_sec'],
                'peak_rss_gb': best.get('peak_rss_gb'),
                'peak_gpu_gb': best.get('peak_gpu_gb'),
            }
        args.out.write_text(json.dumps(payload, indent=2))

    ok = [r for r in results if r.get('ok')]
    print('\n========== RANKING (by train sample/sec) ==========')
    for r in sorted(ok, key=lambda x: -x['sample_per_sec']):
        combo_s = ' '.join(f'{k}={r["combo"][k]}' for k in sorted(r['combo']))
        print(
            f'  {combo_s}  {r["sample_per_sec"]:6.2f} samp/s  '
            f'RSS={r["peak_rss_gb"]:5.1f}GB  GPU={r["peak_gpu_gb"]:4.1f}GB'
        )
    fail = [r for r in results if not r.get('ok')]
    if fail:
        print('\nFAILED:')
        for r in fail:
            print(f'  {r.get("combo")}: {r.get("error")}')

    if ok:
        best_payload = json.loads(args.out.read_text())['best']
        print('\n========== BEST ==========')
        print(json.dumps(best_payload, indent=2, ensure_ascii=False))
        print('\nSuggested train flags:')
        for k, v in sorted(best_payload['combo'].items()):
            print(f'  --{k} {v}')
    print(f'\nWrote {args.out}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
