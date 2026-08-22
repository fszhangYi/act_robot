#!/usr/bin/env python3
"""Convert raw robot episodes to ACT-compatible HDF5 format.

Three action-space modes (--action-space):

  joint (default, recommended):
    qpos   = [θ1..θ6, gripper_obs]          absolute joint angles + gripper observation
    action = [θ1..θ6, gripper_target]       absolute joint angles at next stride frame
    No SE(3) maths, no Euler singularities, matches official ACT paper.

  cartesian_abs:
    qpos   = [x, y, z, rx, ry, rz, gripper_obs]
    action = [x, y, z, rx, ry, rz, gripper_target] at next stride frame (absolute)
    No SE(3) deltas — same shape as joint mode but in cartesian space.
    Safe only when rx/ry/rz never cross ±π boundary (use --unwrap-rx otherwise).

  cartesian (legacy):
    qpos   = [x, y, z, rx, ry, rz, gripper_obs]
    action = SE(3) relative pose delta + absolute gripper target
    Suffers from Euler-angle wrap near rx≈π; kept for backward compatibility.

Temporal subsampling (--stride N):
    Take every N-th frame within the (sliced) range.

Annotation-driven frame slicing (--annotation-dir DIR):
    Each episode <name> must have <DIR>/<name>.txt, a 12-line space-separated
    text file where line 1 col 2 = START FRAME and line 3 col 2 = END FRAME
    (0-indexed inclusive). Only steps[start : end+1] is converted. If start or
    end is -1, the episode is skipped (no valid annotation).

rx unwrap (--unwrap-rx):
    Cartesian-mode safety net. Shifts negative rx (~-π) by +2π so the rx
    distribution is single-peaked around ~+π+ instead of bimodal at ±π.
    For cartesian_abs both qpos[3] and action[3] are shifted; for cartesian
    (SE(3) delta) only qpos[3]. dataset_info.json records `rx_unwrapped=true`
    so train.py + serve.py can apply the symmetric inverse at inference.

Usage:
    python act_robot/convert_episodes.py \\
        --input-dir <dataset>/raw_data \\
        --output-dir /data/out \\
        --annotation-dir <dataset>/annotation \\
        --camera-names chest top wrist_2 \\
        --action-space cartesian_abs --stride 1 --unwrap-rx \\
        --num-workers 4 \\
        --filter-json data/quality_pass.json
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------

def euler_xyz_to_matrix(euler_xyz: np.ndarray) -> np.ndarray:
    """Euler XYZ (intrinsic) angles to 3×3 rotation matrix."""
    rx, ry, rz = float(euler_xyz[0]), float(euler_xyz[1]), float(euler_xyz[2])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix to Euler XYZ angles in [-π, π]."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def pose6_to_matrix(pose6: np.ndarray) -> np.ndarray:
    """[x, y, z, rx, ry, rz] → 4×4 SE(3) matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pose6[:3]
    T[:3, :3] = euler_xyz_to_matrix(pose6[3:6])
    return T


def matrix_to_pose6(T: np.ndarray) -> np.ndarray:
    """4×4 SE(3) matrix → [x, y, z, rx, ry, rz]."""
    pose = np.zeros(6, dtype=np.float64)
    pose[:3] = T[:3, 3]
    pose[3:6] = matrix_to_euler_xyz(T[:3, :3])
    return pose


def compute_action(cur_pose6: np.ndarray, nxt_pose6: np.ndarray) -> np.ndarray:
    """Pose transformation action: T_action = inv(T_cur) @ T_nxt."""
    T_cur = pose6_to_matrix(cur_pose6)
    T_nxt = pose6_to_matrix(nxt_pose6)
    T_act = np.linalg.inv(T_cur) @ T_nxt
    return matrix_to_pose6(T_act)


# ---------------------------------------------------------------------------
# Episode conversion
# ---------------------------------------------------------------------------

# Camera prefixes and resize maps shared across all episode conversions.
# Image naming convention is `<prefix>_<frame_idx>.jpg`; frame index parsed via
# the final `_<int>` token of the file stem.
_CAMERA_PREFIX: dict[str, str] = {
    'wrist':     'rgb_wrist_1',
    'rear_left': 'rgb_rear_left',
    'chest':     'rgb_chest',
    'top':       'rgb_top',
    'wrist_2':   'rgb_wrist_2',
}

# Per-camera target (H, W). None = keep native resolution.
# Repo convention is uniform 480×640 model input — cameras whose native size is
# different get resized (BILINEAR, aspect ratio NOT preserved, matching the
# original rear_left behaviour). For cameras whose native size already equals
# the target, PIL.resize((640, 480)) is effectively an identity pass.
# Native resolutions in current datasets:
#   100_15:     rgb_wrist_1 = 640×480 (W×H)
#   tonglu0602: rgb_chest / rgb_top = 1280×720;  rgb_wrist_2 = 640×480 (per
#                 user spec, though the partial sample on disk happens to be
#                 1280×720 — either way the (480, 640) target works)
_CAMERA_RESIZE: dict[str, tuple[int, int] | None] = {
    'wrist':     None,        # 100_15 rgb_wrist_1 native 640×480 — keep
    'rear_left': (480, 640),  # 100_15 rgb_rear_left native 1280×720 → squish
    'chest':     (480, 640),  # tonglu0602 native 1280×720 → squish
    'top':       (480, 640),  # tonglu0602 native 1280×720 → squish
    'wrist_2':   (480, 640),  # tonglu0602: 640×480 native is identity-resized;
                              # if a sample is actually 1280×720 it gets squished.
                              # Either way output is (480, 640, 3).
}


def _build_frame_map(episode_dir: Path, prefix: str) -> dict[int, Path]:
    """Map frame index → image path for one camera prefix."""
    frame_map: dict[int, Path] = {}
    for p in episode_dir.iterdir():
        if not p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            continue
        if not p.stem.startswith(prefix):
            continue
        try:
            idx = int(p.stem.split('_')[-1])
        except ValueError:
            continue
        frame_map[idx] = p
    return frame_map


def _load_strided_images(
    frame_map: dict[int, Path],
    frame_indices: list[int],
    resize_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load frames at the given indices (nearest-neighbour lookup) into a uint8 array."""
    available = sorted(frame_map.keys())
    images = []
    for i in frame_indices:
        nearest = min(available, key=lambda x: abs(x - i))
        img = Image.open(frame_map[nearest]).convert('RGB')
        if resize_hw is not None:
            img = img.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR)
        images.append(np.array(img, dtype=np.uint8))
    return np.stack(images)  # [T', H, W, 3]


def _parse_gripper(raw) -> np.ndarray:
    """Normalise gripper array to 1-D (T,) regardless of storage shape (T,), (T,1), or (1,T)."""
    arr = np.array(raw, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr[0]
        return arr.squeeze(-1)
    return arr


def _load_episode_filter(filter_json: Path) -> set[int]:
    """Load episode whitelist from a JSON array of integer indices."""
    if not filter_json.is_file():
        raise FileNotFoundError(f'filter-json not found: {filter_json}')
    payload = json.loads(filter_json.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise ValueError(f'filter-json must be a JSON array, got {type(payload).__name__}')
    allowed: set[int] = set()
    for item in payload:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f'filter-json entries must be integers, got {item!r}')
        allowed.add(item)
    return allowed


def _parse_annotation(annot_path: Path) -> tuple[int, int] | None:
    """Parse a 12-line space-separated annotation file.

    Convention:
      line 1, col 2 = start frame (0-indexed inclusive)
      line 3, col 2 = end frame   (0-indexed inclusive)

    Returns (start, end), or None if the file is missing / either value is -1
    (signalling "no valid annotation, skip this episode").
    """
    if not annot_path.exists():
        return None
    lines = annot_path.read_text().splitlines()
    if len(lines) < 3:
        return None
    try:
        start = int(lines[0].split()[1])
        end   = int(lines[2].split()[1])
    except (ValueError, IndexError):
        return None
    if start < 0 or end < 0 or end < start:
        return None
    return start, end


def _validate_existing_hdf5(path: Path) -> int | None:
    """Check whether *path* is a valid converted HDF5 episode.

    Returns the number of timesteps (``T_sub``) if the file is intact,
    or ``None`` if it is missing, truncated, or lacks the expected datasets.
    """
    if not path.exists():
        return None
    try:
        with h5py.File(path, 'r') as f:
            if 'action' not in f:
                return None
            T = int(f['action'].shape[0])
        return T
    except Exception:
        return None


def convert_episode(
    episode_dir: Path,
    output_path: Path,
    stride: int = 1,
    camera_names: list[str] | None = None,
    action_space: str = 'joint',
    start_idx: int | None = None,
    end_idx: int | None = None,
    unwrap_rx: bool = False,
) -> int:
    """Convert one episode directory → HDF5.

    Args:
        action_space: 'joint' / 'cartesian_abs' / 'cartesian' (legacy SE(3) delta)
        start_idx, end_idx: optional inclusive frame range [start, end] (0-indexed).
            If None, uses [0, T_total).
        unwrap_rx: shift qpos[3]<0 by +2π (cartesian_abs/cartesian only); also
            shifts action[3] for cartesian_abs. serve.py applies the inverse at
            inference time.

    Returns the number of timesteps written.
    """
    import json as _json

    if camera_names is None:
        camera_names = ['wrist']

    steps_path = episode_dir / 'steps.json'
    if not steps_path.exists():
        raise FileNotFoundError(f'Missing steps.json in {episode_dir}')

    with steps_path.open() as f:
        steps = _json.load(f)

    obs = steps['observations']
    gripper_obs = _parse_gripper(obs['gripper_position'])

    if action_space == 'joint':
        proprio = np.array(obs['joint_position'], dtype=np.float64)
    else:
        proprio = np.array(obs['cartesian_position'], dtype=np.float64)
    T_total = len(proprio)

    # Annotation-driven slicing.
    s = 0 if start_idx is None else max(0, int(start_idx))
    e = T_total - 1 if end_idx is None else min(T_total - 1, int(end_idx))
    if e < s:
        raise ValueError(f'invalid slice [{s}, {e}] for T={T_total}')
    frame_indices = list(range(s, e + 1, stride))
    T_sub = len(frame_indices)

    grip_action = _parse_gripper(steps['actions']['gripper_position'])

    qpos = np.stack([
        np.append(proprio[i], gripper_obs[i]) for i in frame_indices
    ]).astype(np.float32)

    actions = np.zeros((T_sub, 7), dtype=np.float32)
    if action_space == 'cartesian':
        # SE(3) relative delta + absolute gripper target
        for k in range(T_sub):
            cur_idx = frame_indices[k]
            nxt_idx = frame_indices[k + 1] if k + 1 < T_sub else frame_indices[k]
            rel_pose6 = compute_action(proprio[cur_idx], proprio[nxt_idx])
            actions[k] = np.append(rel_pose6, float(gripper_obs[nxt_idx])).astype(np.float32)
    else:
        # joint / cartesian_abs: action = absolute target at next stride frame
        for k in range(T_sub):
            nxt_idx = frame_indices[k + 1] if k + 1 < T_sub else frame_indices[k]
            actions[k] = np.append(proprio[nxt_idx], grip_action[nxt_idx]).astype(np.float32)

    # Load strided images for each camera. The frame indices are the actual
    # raw-frame indices (already sliced + strided), so the per-camera frame
    # lookup uses the same indices and produces aligned arrays.
    frame_maps: dict[str, dict[int, Path]] = {}
    for cam in camera_names:
        if cam not in _CAMERA_PREFIX:
            raise ValueError(f'unknown camera {cam}; supported: {list(_CAMERA_PREFIX)}')
        fmap = _build_frame_map(episode_dir, prefix=_CAMERA_PREFIX[cam])
        if not fmap:
            raise FileNotFoundError(f'No {_CAMERA_PREFIX[cam]}_*.jpg frames in {episode_dir}')
        frame_maps[cam] = fmap

    images_per_cam: dict[str, np.ndarray] = {}
    for cam in camera_names:
        images_per_cam[cam] = _load_strided_images(
            frame_maps[cam], frame_indices, resize_hw=_CAMERA_RESIZE[cam]
        )

    # Optional rx unwrap (cartesian_abs / cartesian only): shift the negative-rx
    # cluster (~-π) to positive (~+π+). serve.py applies the inverse on its
    # output, so the client still sees standard [-π, π] rx.
    if unwrap_rx and action_space in ('cartesian_abs', 'cartesian'):
        qpos[:, 3] = np.where(qpos[:, 3] < 0, qpos[:, 3] + 2 * np.pi, qpos[:, 3])
        if action_space == 'cartesian_abs':
            actions[:, 3] = np.where(actions[:, 3] < 0, actions[:, 3] + 2 * np.pi, actions[:, 3])

    qvel = np.zeros_like(qpos)
    qvel[1:] = qpos[1:] - qpos[:-1]

    with h5py.File(output_path, 'w') as f:
        f.attrs['sim'] = True
        f.attrs['stride'] = stride
        f.attrs['camera_names'] = camera_names
        f.attrs['action_space'] = action_space
        obs_grp = f.create_group('observations')
        obs_grp.create_dataset('qpos', data=qpos)
        obs_grp.create_dataset('qvel', data=qvel)
        img_grp = obs_grp.create_group('images')
        for cam in camera_names:
            img_grp.create_dataset(cam, data=images_per_cam[cam], compression='lzf')
        f.create_dataset('action', data=actions)

    return T_sub


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _run_convert_task(task: dict[str, Any]) -> dict[str, Any]:
    """ProcessPool entry point: convert one episode (or skip if already valid)."""
    ep_dir = Path(task['ep_dir'])
    out_path = Path(task['out_path'])
    try:
        if task['skip_existing']:
            T_existing = _validate_existing_hdf5(out_path)
            if T_existing is not None:
                return {
                    'status': 'skipped_existing',
                    'global_idx': task['global_idx'],
                    'group': task['group'],
                    'ep_name': ep_dir.name,
                    'T': T_existing,
                }
        T = convert_episode(
            ep_dir, out_path, stride=task['stride'],
            camera_names=task['camera_names'], action_space=task['action_space'],
            start_idx=task['start_idx'], end_idx=task['end_idx'],
            unwrap_rx=task['unwrap_rx'],
        )
        return {
            'status': 'ok',
            'global_idx': task['global_idx'],
            'group': task['group'],
            'ep_name': ep_dir.name,
            'T': T,
        }
    except Exception as exc:
        return {
            'status': 'error',
            'global_idx': task['global_idx'],
            'group': task['group'],
            'ep_name': ep_dir.name,
            'error': str(exc),
        }


def _build_convert_tasks(
    train_eps: list[Path],
    val_eps: list[Path],
    output_dir: Path,
    annot_dir: Path | None,
    skipped_no_annot: list[str],
    *,
    stride: int,
    camera_names: list[str],
    action_space: str,
    unwrap_rx: bool,
    skip_existing: bool,
) -> list[dict[str, Any]]:
    """Assign fixed episode_<idx>.hdf5 paths in train-then-val shuffle order."""
    tasks: list[dict[str, Any]] = []
    global_idx = 0
    for group_eps, group_label in [(train_eps, 'train'), (val_eps, 'val')]:
        for ep_dir in group_eps:
            start_idx, end_idx = None, None
            if annot_dir is not None:
                parsed = _parse_annotation(annot_dir / f'{ep_dir.name}.txt')
                if parsed is None:
                    skipped_no_annot.append(ep_dir.name)
                    continue
                start_idx, end_idx = parsed
            tasks.append({
                'ep_dir': str(ep_dir),
                'out_path': str(output_dir / f'episode_{global_idx}.hdf5'),
                'global_idx': global_idx,
                'group': group_label,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'stride': stride,
                'camera_names': camera_names,
                'action_space': action_space,
                'unwrap_rx': unwrap_rx,
                'skip_existing': skip_existing,
            })
            global_idx += 1
    return tasks


def _collect_task_results(
    tasks: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> tuple[list[int], list[int], list[int], list[str], list[str]]:
    """Fold worker results into dataset_info fields (processing order)."""
    by_idx = {r['global_idx']: r for r in results}
    train_indices: list[int] = []
    val_indices: list[int] = []
    episode_lengths: list[int] = []
    errors: list[str] = []
    skipped_existing: list[str] = []

    for task in tasks:
        result = by_idx.get(task['global_idx'])
        if result is None:
            continue
        if result['status'] == 'error':
            errors.append(f"{result['ep_name']}: {result['error']}")
            print(f"  SKIP {result['ep_name']}: {result['error']}")
            continue
        if result['status'] == 'skipped_existing':
            skipped_existing.append(
                f"{result['ep_name']} (idx={result['global_idx']})"
            )
        if task['group'] == 'train':
            train_indices.append(task['global_idx'])
        else:
            val_indices.append(task['global_idx'])
        episode_lengths.append(result['T'])

    return train_indices, val_indices, episode_lengths, errors, skipped_existing


def _run_tasks(
    tasks: list[dict[str, Any]],
    num_workers: int,
) -> list[dict[str, Any]]:
    """Convert all tasks sequentially (1 worker) or via ProcessPoolExecutor."""
    if not tasks:
        return []

    if num_workers <= 1:
        results: list[dict[str, Any]] = []
        for task in tqdm(tasks, desc='Converting'):
            results.append(_run_convert_task(task))
        return results

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_run_convert_task, task) for task in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Converting'):
            results.append(fut.result())
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', required=True,
                        help='Root directory containing numeric episode subdirs')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--annotation-dir', default=None,
                        help='Optional dir of <episode_name>.txt annotation files. '
                             'When provided, slice each episode to [start, end] from '
                             'line1col2 / line3col2 (0-indexed inclusive). Skip episode '
                             'if file missing or either value is -1.')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--camera-names', nargs='+', default=['wrist'],
                        choices=sorted(_CAMERA_PREFIX.keys()),
                        help=f'Cameras to include. Supported: {sorted(_CAMERA_PREFIX.keys())}')
    parser.add_argument('--action-space', default='joint',
                        choices=['joint', 'cartesian_abs', 'cartesian'])
    parser.add_argument('--unwrap-rx', action='store_true',
                        help='Shift qpos[3]<0 (and action[3] for cart_abs) by +2π so the rx '
                             'distribution is single-peaked. dataset_info.json records '
                             '`rx_unwrapped=true`; serve.py inverts at inference.')
    parser.add_argument('--filter-json', default=None,
                        help='Optional JSON array of episode indices to convert. '
                             'Only episodes whose directory name is in this whitelist are kept '
                             '(typically output of check_episode_quality.py --write-pass-json).')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip episodes whose output HDF5 already exists and is valid. '
                             'Useful for resuming an interrupted conversion run (same seed '
                             'guarantees the same train/val split and episode ordering).')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Parallel episode converters (default: 1 = sequential). '
                             'Each worker loads/resizes/writes one episode at a time; '
                             'use 4–8 on multi-core machines. Episodes share the container '
                             'CPU quota, so more workers helps most when IO-bound.')
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error('--num-workers must be >= 1')

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    annot_dir = Path(args.annotation_dir) if args.annotation_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit())
    if not episode_dirs:
        raise RuntimeError(f'No numeric episode directories found under {input_dir}')

    filter_path = Path(args.filter_json) if args.filter_json else None
    allowed_indices: set[int] | None = None
    if filter_path is not None:
        allowed_indices = _load_episode_filter(filter_path)
        before = len(episode_dirs)
        episode_dirs = [d for d in episode_dirs if int(d.name) in allowed_indices]
        print(f'Filter JSON: {filter_path}  |  kept {len(episode_dirs)}/{before} episodes')
        if not episode_dirs:
            raise RuntimeError(
                f'No episodes left after applying filter-json {filter_path}'
            )

    random.shuffle(episode_dirs)
    n_train = int(len(episode_dirs) * args.train_ratio)
    train_eps = episode_dirs[:n_train]
    val_eps = episode_dirs[n_train:]
    print(f'Episodes: {len(episode_dirs)} total  |  train={len(train_eps)}  val={len(val_eps)}  '
          f'stride={args.stride}  action_space={args.action_space}  '
          f'annotation={"yes" if annot_dir else "no"}  '
          f'filter={"yes" if filter_path else "no"}  unwrap_rx={args.unwrap_rx}  '
          f'num_workers={args.num_workers}')

    skipped_no_annot: list[str] = []
    tasks = _build_convert_tasks(
        train_eps, val_eps, output_dir, annot_dir, skipped_no_annot,
        stride=args.stride,
        camera_names=args.camera_names,
        action_space=args.action_space,
        unwrap_rx=args.unwrap_rx,
        skip_existing=args.skip_existing,
    )
    print(f'Tasks: {len(tasks)} episodes to convert '
          f'(skipped {len(skipped_no_annot)} without valid annotation)')

    results = _run_tasks(tasks, args.num_workers)
    train_indices, val_indices, episode_lengths, errors, skipped_existing = (
        _collect_task_results(tasks, results)
    )
    global_idx = len(episode_lengths)

    if skipped_no_annot:
        print(f'\n{len(skipped_no_annot)} episodes skipped (no valid annotation):')
        for n in skipped_no_annot[:10]:
            print(f'  {n}')
    if skipped_existing:
        print(f'\n{len(skipped_existing)} episodes skipped (already converted):')
        for n in skipped_existing[:10]:
            print(f'  {n}')
    if errors:
        print(f'\n{len(errors)} episodes errored:')
        for e in errors[:10]:
            print(f'  {e}')

    info = {
        'stride': args.stride,
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        'filter_json': str(filter_path) if filter_path else None,
        'num_total': global_idx,
        'num_train': len(train_indices),
        'num_val': len(val_indices),
        'train_indices': train_indices,
        'val_indices': val_indices,
        'episode_lengths': episode_lengths,
        'max_episode_len': int(max(episode_lengths)) if episode_lengths else 0,
        'action_space': args.action_space,
        'camera_names': args.camera_names,
        'num_workers': args.num_workers,
        'rx_unwrapped': bool(args.unwrap_rx and args.action_space in ('cartesian_abs', 'cartesian')),
    }
    with (output_dir / 'dataset_info.json').open('w') as f:
        json.dump(info, f, indent=2)

    print(f'\nDone.  {global_idx} episodes → {output_dir}')
    print(f'  max_episode_len={info["max_episode_len"]} (after stride={args.stride})')
    print(f'  dataset_info.json saved.')


if __name__ == '__main__':
    main()