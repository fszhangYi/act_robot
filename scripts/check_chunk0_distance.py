"""For the training set, compute |action[k][0:6] - qpos[k][0:6]| (i.e., how far
the model's chunk[0] target should land from the current qpos). This is the
quantity that should constrain model output.

Then for each training episode also compute the LARGEST single-step delta
across the whole chunk window — to find out if any training sample has a
chunk[0] that's ever 0.29 rad from qpos.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--num-episodes', type=int, default=100)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    info = json.load(open(data_dir / 'dataset_info.json'))

    indices = info['train_indices'][: args.num_episodes]

    a_minus_q_first = []           # |action[k] - qpos[k]| for every (qpos, action) pair (chunk[0])
    a_chunk_max_delta = []         # within an episode, the largest |action[k+1]-action[k]|
    starts_action_distance = []    # |action[0] - qpos[0]| at episode START
    full_chunk_drifts = []         # |action[k+9] - action[k]| (intra-chunk position drift in training)

    for ep in indices:
        with h5py.File(data_dir / f'episode_{ep}.hdf5', 'r') as f:
            q = f['/observations/qpos'][:][:, :6]  # joints only
            a = f['/action'][:][:, :6]
        T = q.shape[0]
        # every step
        for k in range(T):
            a_minus_q_first.append(float(np.linalg.norm(a[k] - q[k])))
        # chunk drift
        for k in range(T - 9):
            full_chunk_drifts.append(float(np.linalg.norm(a[k + 9] - a[k])))
        # per-step adjacent
        for k in range(T - 1):
            a_chunk_max_delta.append(float(np.linalg.norm(a[k + 1] - a[k])))
        # start
        starts_action_distance.append(float(np.linalg.norm(a[0] - q[0])))

    def stats(name, arr):
        arr = np.array(arr)
        print(f'  {name:38s}  n={len(arr):6d}  mean={arr.mean():.4f}  '
              f'std={arr.std():.4f}  median={np.median(arr):.4f}  '
              f'p95={np.percentile(arr, 95):.4f}  p99={np.percentile(arr, 99):.4f}  '
              f'max={arr.max():.4f}')

    print(f'Training stats over {len(indices)} episodes:')
    stats('|action[k] - qpos[k]| (chunk[0] dist)', a_minus_q_first)
    stats('|action[0] - qpos[0]| (episode start) ', starts_action_distance)
    stats('|action[k+1] - action[k]| (per-step)   ', a_chunk_max_delta)
    stats('|action[k+9] - action[k]| (chunk drift)', full_chunk_drifts)

    arr = np.array(a_minus_q_first)
    over_29 = (arr > 0.25).sum()
    over_15 = (arr > 0.15).sum()
    over_10 = (arr > 0.10).sum()
    print(f'\nCount of (qpos, action) pairs with |chunk[0] - qpos| above thresholds:')
    print(f'  > 0.10  : {over_10}  ({100*over_10/len(arr):.2f}%)')
    print(f'  > 0.15  : {over_15}  ({100*over_15/len(arr):.2f}%)')
    print(f'  > 0.25  : {over_29}  ({100*over_29/len(arr):.2f}%)')


if __name__ == '__main__':
    main()
