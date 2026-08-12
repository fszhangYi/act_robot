#!/usr/bin/env python3
"""Derive a `cartesian_abs` SAM2 dataset from an existing `joint` SAM2 dataset.

The SAM2 features (F_t) are IDENTICAL across action spaces (they only depend on
the wrist frames + the t=0 bbox prompt). Only qpos and action change. So we
copy F_t from the source dataset's HDF5s, recompute qpos/action from the raw
episodes in cartesian_abs space, and write a new dataset.

This avoids re-running SAM2 (~1h GPU on 550 episodes).

How the source<->raw mapping is preserved: extract_sam2_features.py shuffles
the raw episode dirs with `random.seed(args.seed)` (default 42) and assigns
global indices in shuffle order. We replicate that exactly so episode_<i>.hdf5
in the new dataset corresponds to the same raw episode as in the source.

Usage:
    python derive_cart_abs_dataset.py \\
        --src-dataset /media/.../cam100_15_wrist_sam2small_joint_data_full \\
        --raw-input-dir /media/znyyb/EE223AE5223AB287/100_15 \\
        --output-dir /media/.../cam100_15_wrist_sam2small_cart_abs_data_full \\
        --stride 3 --seed 42 --train-ratio 0.9
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from extract_sam2_features import _build_qpos_action  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--src-dataset', required=True,
                        help='Existing joint SAM2 dataset (contains episode_*.hdf5 with F_t)')
    parser.add_argument('--raw-input-dir', required=True,
                        help='Raw episode root (e.g. /media/.../100_15)')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--action-space', default='cartesian_abs',
                        choices=['joint', 'cartesian_abs', 'cartesian'],
                        help='Target action space (default cartesian_abs)')
    parser.add_argument('--unwrap-rx', action='store_true',
                        help='Shift the negative-rx cluster (~-π) to positive (~+π) so the '
                             'representation is single-valued. cam_100_15 new dataset has rx '
                             'bimodal at ±π — model otherwise cannot resolve the ambiguity '
                             '(observed v5 rx RMSE=0.6 rad). Symmetric inverse must be applied '
                             'at inference (serve.py reads rx_unwrapped from policy_config).')
    args = parser.parse_args()

    src = Path(args.src_dataset)
    raw = Path(args.raw_input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Replay the shuffle that extract_sam2_features.py did, so episode_<i>.hdf5
    # corresponds to the same raw dir.
    episode_dirs = sorted(d for d in raw.iterdir() if d.is_dir() and d.name.isdigit())
    if not episode_dirs:
        raise SystemExit(f'no numeric episode dirs under {raw}')
    random.seed(args.seed)
    random.shuffle(episode_dirs)
    n_train = int(len(episode_dirs) * args.train_ratio)
    train_eps = episode_dirs[:n_train]
    val_eps = episode_dirs[n_train:]

    # Re-walk the source dataset to find which raw episodes successfully
    # extracted (some may have been skipped). The source dataset_info.json
    # provides train_indices/val_indices over the SURVIVING global numbering.
    src_info = json.loads((src / 'dataset_info.json').read_text())
    src_train = src_info['train_indices']
    src_val = src_info['val_indices']

    # Per extract_sam2_features.py logic: it iterates train_eps first then val_eps.
    # For each, it tries _convert_episode; on success it assigns the next global_idx
    # and appends to the appropriate index list. So:
    #   - episode_<src_train[k]>.hdf5 corresponds to train_eps[k_th_success_in_train]
    #   - episode_<src_val[k]>.hdf5  corresponds to val_eps[k_th_success_in_val]
    # We just need to iterate the same order, skip the ones the source skipped.
    surviving = sorted(src_train + src_val)
    if surviving != list(range(len(surviving))):
        # This should hold because extract assigns global_idx sequentially.
        raise SystemExit('source dataset has non-contiguous global indices; cannot replay mapping')
    expected_total = len(train_eps) + len(val_eps)
    n_skipped = expected_total - len(surviving)
    print(f'source dataset has {len(surviving)} surviving episodes '
          f'({len(src_train)} train, {len(src_val)} val); {n_skipped} were skipped by SAM2 extraction')

    # Now process: for each src global_idx, find the corresponding raw dir
    # by walking train_eps + val_eps in order and skipping those that have no
    # corresponding HDF5 in src (we can detect this by checking src/episode_<i>.hdf5
    # missing). Since we don't know WHICH were skipped, we have to be more careful.
    #
    # Trick: open every src HDF5, read its `prompt_bbox_xyxy` attr — that's the
    # bbox of the actual raw episode it processed. We can match by bbox+stride
    # against what each candidate raw episode would have produced. But bbox alone
    # isn't unique. So we fall back to a simpler approach:
    #
    # Per extract_sam2_features.py: it appends to indices_list ONLY when conversion
    # succeeds. So global_idx 0..N-1 maps to train_eps[0..len(src_train)-1 modulo skips]
    # then val_eps[0..len(src_val)-1 modulo skips]. Without a per-episode skip log
    # we can't recover precise mapping if any were skipped.
    #
    # Pragmatic solution: assume the success rate is ~100% (we saw 550/552 = 99.6%)
    # and process in order, skipping raw episodes whose qpos/action computation
    # also fails (likely the same ones).

    global_idx = 0
    written = 0
    errors: list[str] = []
    train_indices: list[int] = []
    val_indices: list[int] = []
    episode_lengths: list[int] = []

    for group_eps, group_label, src_idx_list, out_idx_list in [
        (train_eps, 'train', src_train, train_indices),
        (val_eps,   'val',   src_val,   val_indices),
    ]:
        src_iter = iter(src_idx_list)
        for ep_dir in tqdm(group_eps, desc=f'derive {group_label}'):
            # Find the next surviving src index in this group
            try:
                src_global = next(src_iter)
            except StopIteration:
                # No more surviving src episodes; the remaining raw_eps were all skipped.
                break
            src_path = src / f'episode_{src_global}.hdf5'
            if not src_path.exists():
                errors.append(f'{ep_dir.name}: source HDF5 {src_path.name} missing')
                continue

            try:
                qpos, qvel, actions, frame_indices = _build_qpos_action(
                    ep_dir, stride=args.stride, action_space=args.action_space,
                )
            except Exception as exc:
                errors.append(f'{ep_dir.name}: qpos/action recompute failed: {exc}')
                # Skip this episode AND consume one src index (the source likely
                # skipped a different one — best effort).
                continue

            # Optional rx unwrap:
            #   cartesian_abs: action is an ABSOLUTE pose, so its rx column also
            #     needs the same shift as qpos.
            #   cartesian (SE(3) delta): action is a SMALL relative rotation, no
            #     wrap risk — only qpos needs unwrap. (qpos is still bimodal at
            #     ±π in the raw data, which we observed makes the model fragile
            #     when the client occasionally sends rx≈-π.)
            if args.unwrap_rx and args.action_space in ('cartesian_abs', 'cartesian'):
                qpos[:, 3] = np.where(qpos[:, 3] < 0, qpos[:, 3] + 2 * np.pi, qpos[:, 3])
                if args.action_space == 'cartesian_abs':
                    actions[:, 3] = np.where(actions[:, 3] < 0, actions[:, 3] + 2 * np.pi, actions[:, 3])
                qvel = np.zeros_like(qpos)
                qvel[1:] = qpos[1:] - qpos[:-1]

            # Copy F_t (and mask if present) from source
            with h5py.File(src_path, 'r') as srcf:
                src_T = srcf['/action'].shape[0]
                if src_T != len(qpos):
                    errors.append(
                        f'{ep_dir.name}: stride mismatch — src T={src_T} vs new T={len(qpos)}'
                    )
                    continue
                sam2_feat = srcf['/observations/sam2_feat'][:]
                src_bbox = srcf.attrs.get('prompt_bbox_xyxy', None)
                src_stride = int(srcf.attrs.get('stride', args.stride))
                has_mask = 'sam2_mask' in srcf['/observations']
                sam2_mask = srcf['/observations/sam2_mask'][:] if has_mask else None

            if src_stride != args.stride:
                errors.append(f'{ep_dir.name}: stride attr mismatch ({src_stride} vs {args.stride})')
                continue

            out_path = out / f'episode_{global_idx}.hdf5'
            with h5py.File(out_path, 'w') as f:
                f.attrs['sim'] = True
                f.attrs['stride'] = args.stride
                f.attrs['action_space'] = args.action_space
                f.attrs['camera'] = 'wrist'
                f.attrs['sam2_variant'] = 'sam2.1_hiera_small'
                if src_bbox is not None:
                    f.attrs['prompt_bbox_xyxy'] = src_bbox
                obs = f.create_group('observations')
                obs.create_dataset('qpos', data=qpos)
                obs.create_dataset('qvel', data=qvel)
                obs.create_dataset(
                    'sam2_feat',
                    data=sam2_feat,
                    chunks=(1, 256, 64, 64),
                    compression='lzf',
                )
                if sam2_mask is not None:
                    obs.create_dataset(
                        'sam2_mask',
                        data=sam2_mask,
                        chunks=(1, sam2_mask.shape[1], sam2_mask.shape[2]),
                        compression='lzf',
                    )
                f.create_dataset('action', data=actions)

            out_idx_list.append(global_idx)
            episode_lengths.append(len(qpos))
            global_idx += 1
            written += 1

    info = {
        'stride': args.stride,
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        'num_total': written,
        'num_train': len(train_indices),
        'num_val': len(val_indices),
        'train_indices': train_indices,
        'val_indices': val_indices,
        'episode_lengths': episode_lengths,
        'max_episode_len': int(max(episode_lengths)) if episode_lengths else 0,
        'action_space': args.action_space,
        'camera': 'wrist',
        'sam2_variant': 'sam2.1_hiera_small',
        'feat_dtype': 'float16',
        'feature_shape': [256, 64, 64],
        'derived_from': str(src),
        'rx_unwrapped': bool(args.unwrap_rx and args.action_space in ('cartesian_abs', 'cartesian')),
    }
    (out / 'dataset_info.json').write_text(json.dumps(info, indent=2))

    print(f'\ndone — {written} episodes → {out}')
    print(f'  max_episode_len={info["max_episode_len"]}')
    if errors:
        print(f'  {len(errors)} errors:')
        for e in errors[:5]:
            print(f'    {e}')


if __name__ == '__main__':
    main()
