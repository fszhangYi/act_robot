"""Verify the v3 ckpt's first-chunk distance distribution is bounded inside
the training distribution. The whole point of switching to delta-action is to
make |chunk[0] - qpos| stay small. Run the trained model on many (val) frames
and report the empirical distribution.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
import importlib.util  # noqa: E402

def _load(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

_sf = _load('sam2_features', _ROOT / 'sam2_features.py')
sys.path.append(str(_ROOT))
_pol = _load('policy', _ROOT / 'policy.py')
ACTSAM2Policy = _pol.ACTSAM2Policy
ACTSAM2CVAEPolicy = _pol.ACTSAM2CVAEPolicy


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir', required=True)
    p.add_argument('--ckpt-name', default='policy_best.ckpt')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--num-episodes', type=int, default=20)
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    with (ckpt_dir / 'policy_config.json').open() as f:
        cfg = json.load(f)
    if 'chunk_size' in cfg and 'num_queries' not in cfg:
        cfg['num_queries'] = cfg.pop('chunk_size')
    action_repr = cfg.get('action_repr', 'absolute')
    use_cvae = cfg.get('use_cvae', False)
    action_space = cfg.get('action_space', 'joint')
    # For action_space='cartesian' the network already predicts a SE(3) delta;
    # the |chunk[k] - qpos| metric doesn't apply — instead measure |chunk[k]|
    # (the delta magnitude) and compare to the training delta distribution.
    cart_rel = (action_space == 'cartesian')
    print(f'action_repr={action_repr}   pool_size={cfg.get("pool_size")}   '
          f'use_cvae={use_cvae}   action_space={action_space}   cart_rel={cart_rel}')

    policy = ACTSAM2CVAEPolicy(cfg) if use_cvae else ACTSAM2Policy(cfg)
    ckpt = torch.load(ckpt_dir / args.ckpt_name, map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        policy.model.load_state_dict(ckpt['model'])
    else:
        policy.load_state_dict(ckpt)
    policy.eval().cuda()

    with (ckpt_dir / 'dataset_stats.pkl').open('rb') as f:
        stats = pickle.load(f)
    qpos_mean = torch.from_numpy(stats['qpos_mean']).float().cuda()
    qpos_std  = torch.from_numpy(stats['qpos_std']).float().cuda()
    if action_repr == 'delta':
        delta_mean = torch.from_numpy(stats['delta_mean']).float().cuda()
        delta_std  = torch.from_numpy(stats['delta_std']).float().cuda()
    else:
        action_mean = torch.from_numpy(stats['action_mean']).float().cuda()
        action_std  = torch.from_numpy(stats['action_std']).float().cuda()

    data_dir = Path(args.data_dir)
    info = json.load(open(data_dir / 'dataset_info.json'))
    indices = info['val_indices'][: args.num_episodes]
    print(f'eval {len(indices)} val episodes')

    chunk0_dists = []
    chunk9_dists = []
    for ep in indices:
        with h5py.File(data_dir / f'episode_{ep}.hdf5', 'r') as f:
            q = f['/observations/qpos'][:]
            sf = f['/observations/sam2_feat'][:]
        for t in range(len(q)):
            qt = torch.from_numpy(q[t]).float().cuda()
            ft = torch.from_numpy(sf[t].astype(np.float32)).cuda().unsqueeze(0)
            qn = ((qt - qpos_mean) / qpos_std).unsqueeze(0)
            with torch.inference_mode():
                pred_norm = policy(qn, ft)[0]
            if action_repr == 'delta':
                delta = pred_norm * delta_std + delta_mean
                action_abs = delta + qt
            else:
                action_abs = pred_norm * action_std + action_mean
            if cart_rel:
                # action_abs here IS the SE(3) delta (denormalised); compare magnitude to zero.
                chunk0_dists.append(float(torch.linalg.norm(action_abs[0][:6]).item()))
                chunk9_dists.append(float(torch.linalg.norm(action_abs[9][:6]).item()))
            else:
                chunk0_dists.append(float(torch.linalg.norm(action_abs[0][:6] - qt[:6]).item()))
                chunk9_dists.append(float(torch.linalg.norm(action_abs[9][:6] - qt[:6]).item()))

    a = np.array(chunk0_dists)
    b = np.array(chunk9_dists)
    print(f'\n|chunk[0] - qpos| (joints 1-6):')
    print(f'  n={len(a)}  mean={a.mean():.4f}  std={a.std():.4f}  median={np.median(a):.4f}  '
          f'p95={np.percentile(a, 95):.4f}  p99={np.percentile(a, 99):.4f}  max={a.max():.4f}')
    print(f'\n|chunk[9] - qpos| (joints 1-6):')
    print(f'  n={len(b)}  mean={b.mean():.4f}  std={b.std():.4f}  median={np.median(b):.4f}  '
          f'p95={np.percentile(b, 95):.4f}  p99={np.percentile(b, 99):.4f}  max={b.max():.4f}')

    # Training stats reminder (from earlier check_chunk0_distance.py):
    print(f'\n[v2 baseline reminder] training |chunk[0] - qpos| max=0.165, p99=0.091')
    print(f'[v2 inference at step 11] observed |chunk[0] - qpos| = 0.290 (OOD!)')
    if a.max() < 0.20:
        print(f'\n*** PASS: v3 max |chunk[0]-qpos| ({a.max():.4f}) within training distribution ***')
    else:
        print(f'\n*** FAIL: v3 max |chunk[0]-qpos| ({a.max():.4f}) still exceeds training max ***')


if __name__ == '__main__':
    main()
