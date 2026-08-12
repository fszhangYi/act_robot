#!/usr/bin/env python3
"""Stage-1 batch extraction: run frozen SAM2 once per episode, cache F_t
alongside qpos/action into HDF5 — the format consumed by SAM2EpisodicDataset
during training.

Reuses convert_episodes.py for qpos/action computation (joint / cartesian_abs
/ cartesian modes). For each raw episode:

  1. Read steps.json and bbox.json
  2. Run SAM2 video tracking on ALL raw wrist frames (no stride; SAM2 memory
     needs consecutive frames for stable tracking)
  3. Stride-subsample qpos, action, and F_t at the same indices
  4. Write to <output_dir>/episode_<i>.hdf5

Each HDF5 contains:
    /observations/qpos       [T', state_dim] float32
    /observations/qvel       [T', state_dim] float32
    /observations/sam2_feat  [T', 256, 64, 64] float16  (chunks=(1,...), lzf)
    /action                  [T', action_dim] float32
    attrs:
      sim, stride, action_space, camera='wrist',
      sam2_variant='sam2.1_hiera_small',
      prompt_bbox_xyxy=[x1,y1,x2,y2]

Smoke-test usage (3 episodes):
  python extract_sam2_features.py \\
      --input-dir  /home/znyyb/hww/vla/gongjian/cam_100_15 \\
      --output-dir /media/znyyb/EE223AE5223AB287/act_dataset/cam100_15_wrist_sam2small_joint_data \\
      --stride 3 --num-episodes 3 --save-masks
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
from convert_episodes import (  # noqa: E402
    _parse_gripper,
    compute_action,
)
from sam2_features import SAM2FeatureExtractor  # noqa: E402


def _bbox_xyxy(ep_dir: Path, frame_key: str = '000000') -> list[float]:
    bb = json.loads((ep_dir / 'bbox.json').read_text())
    rec = bb[frame_key]['rgb_wrist_1'][0]
    return [float(rec['x_min']), float(rec['y_min']),
            float(rec['x_max']), float(rec['y_max'])]


def _build_qpos_action(ep_dir: Path, stride: int, action_space: str
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Return (qpos, qvel, actions, frame_indices) at strided timesteps.

    Mirrors convert_episodes.convert_episode's qpos/action logic for the three
    supported action spaces. Returns frame_indices into the raw 0..T_total-1
    range so the SAM2 features can be sub-sampled at the same indices.
    """
    steps = json.loads((ep_dir / 'steps.json').read_text())
    obs = steps['observations']
    gripper_obs = _parse_gripper(obs['gripper_position'])
    grip_action = _parse_gripper(steps['actions']['gripper_position'])

    if action_space == 'joint':
        abs_arr = np.array(obs['joint_position'], dtype=np.float64)  # [T,6]
    else:
        abs_arr = np.array(obs['cartesian_position'], dtype=np.float64)  # [T,6]

    T_total = len(abs_arr)
    frame_indices = list(range(0, T_total, stride))
    T_sub = len(frame_indices)

    qpos = np.stack([
        np.append(abs_arr[i], gripper_obs[i]) for i in frame_indices
    ]).astype(np.float32)  # [T', 7]

    actions = np.zeros((T_sub, 7), dtype=np.float32)
    for k in range(T_sub):
        cur_idx = frame_indices[k]
        nxt_idx = frame_indices[k + 1] if k + 1 < T_sub else cur_idx
        if action_space == 'cartesian':  # SE(3) delta
            rel = compute_action(abs_arr[cur_idx], abs_arr[nxt_idx])
            actions[k] = np.append(rel, float(gripper_obs[nxt_idx])).astype(np.float32)
        else:  # joint / cartesian_abs
            actions[k] = np.append(abs_arr[nxt_idx], grip_action[nxt_idx]).astype(np.float32)

    qvel = np.zeros_like(qpos)
    qvel[1:] = qpos[1:] - qpos[:-1]
    return qpos, qvel, actions, frame_indices


def _convert_episode(
    ep_dir: Path,
    out_path: Path,
    extractor: SAM2FeatureExtractor,
    stride: int,
    action_space: str,
    save_masks: bool,
    feat_dtype: str = 'float16',
) -> int:
    """Process one raw episode dir → write one HDF5; return T'."""
    wrist_paths = sorted(ep_dir.glob('rgb_wrist_1_*.jpg'))
    if not wrist_paths:
        raise FileNotFoundError(f'no rgb_wrist_1_*.jpg in {ep_dir}')

    qpos, qvel, actions, frame_indices = _build_qpos_action(ep_dir, stride, action_space)
    if len(frame_indices) != len(qpos):
        raise RuntimeError('frame_indices vs qpos length mismatch')

    # SAM2 needs all consecutive frames for stable memory tracking.
    # We extract on every raw frame, then sub-sample at frame_indices.
    bbox = _bbox_xyxy(ep_dir)
    raw_feats, raw_masks = extractor.extract_episode(
        wrist_paths, bbox, feat_dtype=feat_dtype, return_masks=save_masks,
    )
    if raw_feats.shape[0] != len(wrist_paths):
        raise RuntimeError(
            f'expected {len(wrist_paths)} feature frames, got {raw_feats.shape[0]}'
        )

    sam2_feat = raw_feats[frame_indices]               # [T', 256, 64, 64]
    sam2_mask = raw_masks[frame_indices] if save_masks else None

    T_sub = len(frame_indices)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.attrs['sim'] = True
        f.attrs['stride'] = stride
        f.attrs['action_space'] = action_space
        f.attrs['camera'] = 'wrist'
        f.attrs['sam2_variant'] = 'sam2.1_hiera_small'
        f.attrs['prompt_bbox_xyxy'] = np.array(bbox, dtype=np.float32)

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

    return T_sub


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input-dir', required=True,
                        help='Root with numeric episode subdirs (e.g. cam_100_15/)')
    parser.add_argument('--output-dir', required=True,
                        help='Output dir for episode_*.hdf5 + dataset_info.json')
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--action-space', default='joint',
                        choices=['joint', 'cartesian_abs', 'cartesian'])
    parser.add_argument('--num-episodes', type=int, default=None,
                        help='Process only the first N episodes (smoke-test cap)')
    parser.add_argument('--save-masks', action='store_true',
                        help='Also store per-frame binary masks (LZF compressed)')
    parser.add_argument('--feat-dtype', default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument('--sam2-config', default='configs/sam2.1/sam2.1_hiera_s.yaml')
    parser.add_argument('--sam2-ckpt',
                        default='/home/znyyb/hww/vla/act_robot/sam2/checkpoints/sam2.1_hiera_small.pt')
    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(
        d for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit()
    )
    if not episode_dirs:
        raise RuntimeError(f'no numeric episode subdirs under {input_dir}')

    if args.num_episodes is not None:
        episode_dirs = episode_dirs[: args.num_episodes]

    random.shuffle(episode_dirs)
    n_train = int(len(episode_dirs) * args.train_ratio)
    train_eps = episode_dirs[:n_train]
    val_eps = episode_dirs[n_train:]
    print(f'episodes: {len(episode_dirs)}  train={len(train_eps)}  val={len(val_eps)}  '
          f'stride={args.stride}  action_space={args.action_space}')

    print('Loading SAM2...')
    extractor = SAM2FeatureExtractor(
        config_file=args.sam2_config, ckpt_path=args.sam2_ckpt,
    )

    train_indices, val_indices, episode_lengths, errors = [], [], [], []
    global_idx = 0
    for group_eps, group_label, idx_list in [
        (train_eps, 'train', train_indices),
        (val_eps,   'val',   val_indices),
    ]:
        for ep_dir in tqdm(group_eps, desc=f'extract {group_label}'):
            out_path = output_dir / f'episode_{global_idx}.hdf5'
            try:
                T = _convert_episode(
                    ep_dir, out_path, extractor,
                    stride=args.stride,
                    action_space=args.action_space,
                    save_masks=args.save_masks,
                    feat_dtype=args.feat_dtype,
                )
                idx_list.append(global_idx)
                episode_lengths.append(T)
                global_idx += 1
            except Exception as exc:
                errors.append(f'{ep_dir.name}: {exc}')
                print(f'  SKIP {ep_dir.name}: {exc}')

    info = {
        'stride': args.stride,
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        'num_total': global_idx,
        'num_train': len(train_indices),
        'num_val': len(val_indices),
        'train_indices': train_indices,
        'val_indices': val_indices,
        'episode_lengths': episode_lengths,
        'max_episode_len': int(max(episode_lengths)) if episode_lengths else 0,
        'action_space': args.action_space,
        'camera': 'wrist',
        'sam2_variant': 'sam2.1_hiera_small',
        'feat_dtype': args.feat_dtype,
        'feature_shape': [256, 64, 64],
    }
    with (output_dir / 'dataset_info.json').open('w') as f:
        json.dump(info, f, indent=2)

    print(f'\ndone — {global_idx} episodes → {output_dir}')
    print(f'  max_episode_len={info["max_episode_len"]}')
    if errors:
        print(f'  errors: {len(errors)}')
        for e in errors[:5]:
            print(f'    {e}')


if __name__ == '__main__':
    main()
