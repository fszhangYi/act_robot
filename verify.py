#!/usr/bin/env python3
"""Verify a converted ACT dataset.

Checks:
  - HDF5 structure and dtypes
  - Action distribution per dimension (critical: rotation dims should NOT show ±2π jumps)
  - Episode length distribution
  - Dataset split counts

Usage:
    python act_robot/verify.py --data-dir /data/act_v1
    python act_robot/verify.py --data-dir /data/act_v1 --num-episodes 20
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


def verify_episode(path: str) -> dict:
    with h5py.File(path, 'r') as f:
        qpos   = f['/observations/qpos'][()]
        action = f['/action'][()]
        # images = f['/observations/images/wrist']
        images = f['/observations/images/wrist_2']
        img_shape = images.shape
        action_space = f.attrs.get('action_space', 'cartesian')
        if isinstance(action_space, bytes):
            action_space = action_space.decode()
    return {
        'T': len(qpos),
        'state_dim': qpos.shape[1],
        'action_dim': action.shape[1],
        'img_shape': img_shape[1:],  # (H, W, 3)
        'qpos': qpos,
        'action': action,
        'action_space': action_space,
    }


def print_stats(name: str, arr: np.ndarray) -> None:
    print(f'  {name}:')
    for i in range(arr.shape[1]):
        col = arr[:, i]
        print(f'    [{i}]  mean={col.mean():+.4f}  std={col.std():.4f}  '
              f'min={col.min():+.4f}  max={col.max():+.4f}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--num-episodes', type=int, default=None,
                        help='Max number of episodes to inspect (default: all)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load split info
    info_path = data_dir / 'dataset_info.json'
    if info_path.exists():
        with info_path.open() as f:
            info = json.load(f)
        print(f'dataset_info.json:')
        print(f'  stride        = {info.get("stride", "?")}')
        print(f'  num_total     = {info.get("num_total")}')
        print(f'  num_train     = {info.get("num_train")}')
        print(f'  num_val       = {info.get("num_val")}')
        print(f'  max_ep_len    = {info.get("max_episode_len")}')
        lengths = info.get('episode_lengths', [])
        if lengths:
            print(f'  ep_len  min/median/max = {min(lengths)} / {int(np.median(lengths))} / {max(lengths)}')
        print()

    # Collect all HDF5 files
    hdf5_files = sorted(data_dir.glob('episode_*.hdf5'))
    if not hdf5_files:
        print('ERROR: No episode_*.hdf5 files found.')
        return

    if args.num_episodes:
        hdf5_files = hdf5_files[:args.num_episodes]

    print(f'Inspecting {len(hdf5_files)} episodes...')

    all_actions = []
    all_qpos = []
    ep_lengths = []
    first_img_shape = None
    action_space = None

    for p in hdf5_files:
        try:
            d = verify_episode(str(p))
        except Exception as e:
            print(f'  ERROR reading {p.name}: {e}')
            continue
        all_actions.append(d['action'])
        all_qpos.append(d['qpos'])
        ep_lengths.append(d['T'])
        if first_img_shape is None:
            first_img_shape = d['img_shape']
            action_space = d['action_space']
            print(f'  state_dim={d["state_dim"]}  action_dim={d["action_dim"]}  '
                  f'img_shape={d["img_shape"]}  action_space={action_space}')

    all_actions_cat = np.concatenate(all_actions, axis=0)
    all_qpos_cat    = np.concatenate(all_qpos, axis=0)

    print(f'\nEpisode lengths:  min={min(ep_lengths)}  median={int(np.median(ep_lengths))}  max={max(ep_lengths)}')
    print(f'Total timesteps:  {len(all_actions_cat)}\n')

    print_stats('qpos', all_qpos_cat)
    print()
    print_stats('action', all_actions_cat)

    # Rotation wrap check is only meaningful for the legacy SE(3) delta mode,
    # where action dims 3-5 should be small relative rotations. For joint or
    # cartesian_abs modes the dims are absolute angles and can legitimately
    # be near ±π.
    if action_space == 'cartesian':
        print('\n--- Rotation action check (dims 3-5, SE(3) delta) ---')
        ok = True
        for i in [3, 4, 5]:
            col = all_actions_cat[:, i]
            large = np.abs(col) > 1.0
            pct = 100.0 * large.sum() / len(col)
            flag = '  *** WARNING: Euler wrap detected! ***' if pct > 1 else '  OK'
            print(f'  dim[{i}]  |x|>1.0: {large.sum()} steps ({pct:.1f}%){flag}')
            if pct > 1:
                ok = False

        if ok:
            print('\n[PASS] No Euler angle wrapping detected in rotation dims.')
        else:
            print('\n[FAIL] Euler angle wrapping detected. Re-check convert_episodes.py.')
    else:
        print(f'\n[skipped Euler-wrap check — only applies to action_space=cartesian, '
              f'got {action_space}]')


if __name__ == '__main__':
    main()
