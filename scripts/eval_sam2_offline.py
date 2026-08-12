"""Offline open-loop eval of an ACTSAM2 checkpoint.

Loads cached SAM2 features (the same ones used at training), runs them through
the trained policy, and reports per-dim MSE against the recorded ground-truth
actions. Use this BEFORE setting up serve.py + a real client — it isolates
"is the model any good?" from "is the streaming SAM2 pipeline any good?".

Usage:
    python scripts/eval_sam2_offline.py \\
        --ckpt-dir /media/znyyb/EE223AE5223AB287/act_dataset/cam100_15_wrist_sam2small_joint_ckpt_full \\
        --ckpt-name policy_epoch_200_seed_0.ckpt \\
        --data-dir /media/znyyb/EE223AE5223AB287/act_dataset/cam100_15_wrist_sam2small_joint_data_full \\
        --split val \\
        --num-episodes 10
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from policy import ACTSAM2Policy, ACTSAM2CVAEPolicy  # noqa: E402


def load_policy(ckpt_dir: Path, ckpt_name: str, device: torch.device) -> ACTSAM2Policy:
    with (ckpt_dir / 'policy_config.json').open() as f:
        cfg = json.load(f)
    if 'chunk_size' in cfg and 'num_queries' not in cfg:
        cfg['num_queries'] = cfg.pop('chunk_size')
    policy = ACTSAM2CVAEPolicy(cfg) if cfg.get('use_cvae', False) else ACTSAM2Policy(cfg)
    ckpt = torch.load(ckpt_dir / ckpt_name, map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        policy.model.load_state_dict(ckpt['model'])
    else:
        policy.load_state_dict(ckpt)
    policy.eval().to(device)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', required=True)
    parser.add_argument('--ckpt-name', default='policy_best.ckpt')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Cap episodes evaluated (None = all in split)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    data_dir = Path(args.data_dir)
    device = torch.device(args.device)

    # Load policy + stats
    policy = load_policy(ckpt_dir, args.ckpt_name, device)
    with (ckpt_dir / 'dataset_stats.pkl').open('rb') as f:
        stats = pickle.load(f)
    qpos_mean = torch.from_numpy(stats['qpos_mean']).float().to(device)
    qpos_std  = torch.from_numpy(stats['qpos_std']).float().to(device)
    action_mean = torch.from_numpy(stats['action_mean']).float().to(device)
    action_std  = torch.from_numpy(stats['action_std']).float().to(device)
    delta_mean = (torch.from_numpy(stats['delta_mean']).float().to(device)
                  if 'delta_mean' in stats else None)
    delta_std  = (torch.from_numpy(stats['delta_std']).float().to(device)
                  if 'delta_std' in stats else None)

    # Detect action_repr from policy_config so we denormalise correctly.
    with (ckpt_dir / 'policy_config.json').open() as f:
        _cfg = json.load(f)
    action_repr = _cfg.get('action_repr', 'absolute')
    print(f'action_repr={action_repr}')

    info = json.loads((data_dir / 'dataset_info.json').read_text())
    indices = info[f'{args.split}_indices']
    if args.num_episodes:
        indices = indices[: args.num_episodes]

    chunk = policy.num_queries
    all_step_sq = []   # per-timestep squared error, summed over chunk position 0
    all_chunk_sq = []  # per-chunk per-position squared error

    print(f'Eval split={args.split} episodes={len(indices)} chunk_size={chunk}')

    for idx in tqdm(indices, desc='eval'):
        with h5py.File(data_dir / f'episode_{idx}.hdf5', 'r') as f:
            qpos = f['/observations/qpos'][:]           # [T, 7]
            sam2 = f['/observations/sam2_feat'][:]      # [T, 256, 64, 64]
            action_gt = f['/action'][:]                 # [T, 7]
        T = qpos.shape[0]

        for t in range(T):
            qt = torch.from_numpy(qpos[t]).float().to(device)
            ft = torch.from_numpy(sam2[t].astype(np.float32)).to(device).unsqueeze(0)
            qn = ((qt - qpos_mean) / qpos_std).unsqueeze(0)
            with torch.inference_mode():
                pred_norm = policy(qn, ft)                  # [1, chunk, 7]
            if action_repr == 'delta':
                delta = pred_norm[0] * delta_std + delta_mean   # [chunk, 7]
                a_pred = (delta + qt).cpu().numpy()
            else:
                a_pred = (pred_norm[0] * action_std + action_mean).cpu().numpy()

            # Compare each chunk position k against GT at t+k (where GT exists)
            for k in range(chunk):
                gt_idx = t + k
                if gt_idx >= T:
                    break
                sq = (a_pred[k] - action_gt[gt_idx]) ** 2  # [7]
                all_chunk_sq.append((k, sq))
                if k == 0:
                    all_step_sq.append(sq)

    all_step_sq = np.stack(all_step_sq)                                  # [N, 7]
    all_chunk_sq_by_k = [
        np.stack([sq for k_, sq in all_chunk_sq if k_ == k])
        for k in range(chunk)
    ]

    print(f'\n[chunk position 0 — what the robot actually executes next step]')
    print(f'  total predictions: {len(all_step_sq)}')
    for d, name in enumerate(['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'gripper']):
        rmse = float(np.sqrt(all_step_sq[:, d].mean()))
        print(f'  dim[{d}] {name:8s}  RMSE={rmse:.4f}')
    overall = float(np.sqrt(all_step_sq.mean()))
    print(f'  OVERALL RMSE (all dims): {overall:.4f}')

    print(f'\n[RMSE vs chunk position — should grow with k]')
    for k in range(chunk):
        v = all_chunk_sq_by_k[k]
        if len(v) == 0:
            continue
        rmse = float(np.sqrt(v.mean()))
        print(f'  k={k:2d}  RMSE={rmse:.4f}  (n={len(v)})')


if __name__ == '__main__':
    main()
