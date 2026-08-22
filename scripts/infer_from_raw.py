#!/usr/bin/env python3
"""Run trained ACT policy on a raw episode (steps.json + camera JPEGs).

Builds qpos / images the same way as convert_episodes.py, runs inference via
serve.ACTInference, and compares predicted action chunks against ground truth.

Usage:
    python scripts/infer_from_raw.py \\
        --raw-dir data/raw \\
        --episode 3 \\
        --ckpt-dir data/ckpt_cam100_15_cart_abs_v1 \\
        --annotation-dir data/annotation/annotation \\
        --unwrap-rx \\
        --max-frames 30 \\
        --output data/infer_raw_ep3.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from convert_episodes import (  # noqa: E402
    _CAMERA_PREFIX,
    _build_frame_map,
    _load_strided_images,
    _parse_annotation,
    _parse_gripper,
)
from serve import ACTInference, apply_rx_unwrap_to_input  # noqa: E402


def _build_qpos_actions(
    steps: dict,
    frame_indices: list[int],
    action_space: str,
) -> tuple[np.ndarray, np.ndarray]:
    obs = steps['observations']
    gripper_obs = _parse_gripper(obs['gripper_position'])
    grip_action = _parse_gripper(steps['actions']['gripper_position'])
    if action_space == 'joint':
        proprio = np.array(obs['joint_position'], dtype=np.float64)
    else:
        proprio = np.array(obs['cartesian_position'], dtype=np.float64)

    qpos = np.stack([
        np.append(proprio[i], gripper_obs[i]) for i in frame_indices
    ]).astype(np.float32)

    actions = np.zeros((len(frame_indices), 7), dtype=np.float32)
    if action_space == 'cartesian':
        from convert_episodes import compute_action
        for k, cur_idx in enumerate(frame_indices):
            nxt_idx = frame_indices[k + 1] if k + 1 < len(frame_indices) else cur_idx
            rel_pose6 = compute_action(proprio[cur_idx], proprio[nxt_idx])
            actions[k] = np.append(rel_pose6, float(gripper_obs[nxt_idx])).astype(np.float32)
    else:
        for k in range(len(frame_indices)):
            nxt_idx = frame_indices[k + 1] if k + 1 < len(frame_indices) else frame_indices[k]
            actions[k] = np.append(proprio[nxt_idx], grip_action[nxt_idx]).astype(np.float32)
    return qpos, actions


def _slice_gt_chunk(
    gt_actions: np.ndarray,
    k: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align GT action chunk with model output (same padding as EpisodicDataset)."""
    valid_len = min(chunk_size, len(gt_actions) - k)
    gt_chunk = np.zeros((chunk_size, gt_actions.shape[1]), dtype=np.float32)
    if valid_len > 0:
        gt_chunk[:valid_len] = gt_actions[k:k + valid_len]
        if valid_len < chunk_size:
            gt_chunk[valid_len:] = gt_actions[k + valid_len - 1]
    is_pad = np.zeros(chunk_size, dtype=bool)
    if valid_len < chunk_size:
        is_pad[valid_len:] = True
    return gt_chunk, is_pad


def _chunk_mae(pred: np.ndarray, gt: np.ndarray, is_pad: np.ndarray) -> float:
    mask = ~is_pad
    if not mask.any():
        return float('nan')
    return float(np.abs(pred[mask] - gt[mask]).mean())


def _images_to_jpegs(images_per_cam: dict[str, np.ndarray], frame_k: int) -> list[bytes]:
    jpegs: list[bytes] = []
    for cam in images_per_cam:
        buf = io.BytesIO()
        Image.fromarray(images_per_cam[cam][frame_k]).save(buf, format='JPEG', quality=95)
        jpegs.append(buf.getvalue())
    return jpegs


def infer_episode(
    inferencer: ACTInference,
    *,
    episode: int,
    raw_dir: Path,
    annotation_dir: Path | None,
    camera_names: list[str],
    action_space: str,
    unwrap_rx: bool,
    stride: int = 1,
    max_frames: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run inference on one raw episode; return {summary, frames} payload."""
    from convert_episodes import _CAMERA_RESIZE

    ep_dir = raw_dir / str(episode)
    if not ep_dir.is_dir():
        raise FileNotFoundError(f'Episode dir not found: {ep_dir}')

    steps = json.loads((ep_dir / 'steps.json').read_text())
    obs = steps['observations']
    if action_space == 'joint':
        T_total = len(obs['joint_position'])
    else:
        T_total = len(obs['cartesian_position'])

    start, end = 0, T_total - 1
    if annotation_dir is not None:
        annot = _parse_annotation(annotation_dir / f'{episode}.txt')
        if annot is None:
            raise ValueError(f'No valid annotation for episode {episode}')
        start, end = annot

    frame_indices = list(range(start, end + 1, stride))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    if not frame_indices:
        raise ValueError(f'No frames to process for episode {episode}')

    qpos, gt_actions = _build_qpos_actions(steps, frame_indices, action_space)
    if unwrap_rx and action_space in ('cartesian_abs', 'cartesian'):
        qpos[:, 3] = np.where(qpos[:, 3] < 0, qpos[:, 3] + 2 * np.pi, qpos[:, 3])
        if action_space == 'cartesian_abs':
            gt_actions[:, 3] = np.where(
                gt_actions[:, 3] < 0, gt_actions[:, 3] + 2 * np.pi, gt_actions[:, 3],
            )

    images_per_cam: dict[str, np.ndarray] = {}
    for cam in camera_names:
        prefix = _CAMERA_PREFIX[cam]
        fmap = _build_frame_map(ep_dir, prefix)
        if not fmap:
            raise FileNotFoundError(f'No images for camera {cam} ({prefix}) in {ep_dir}')
        images_per_cam[cam] = _load_strided_images(
            fmap, frame_indices, resize_hw=_CAMERA_RESIZE[cam],
        )

    results: list[dict] = []
    chunk0_maes: list[float] = []
    chunk_maes: list[float] = []
    chunk_size = inferencer.chunk_size

    if verbose:
        print(f'Episode {episode}  frames={len(frame_indices)}  '
              f'range=[{frame_indices[0]}, {frame_indices[-1]}]')

    for k, raw_frame_idx in enumerate(frame_indices):
        q = qpos[k].copy()
        if inferencer.rx_unwrapped:
            q = apply_rx_unwrap_to_input(q)
        jpegs = _images_to_jpegs(images_per_cam, k)
        pred_chunk = inferencer.infer_chunk(jpegs, q)
        gt_chunk, is_pad = _slice_gt_chunk(gt_actions, k, chunk_size)
        mae0 = float(np.abs(pred_chunk[0] - gt_chunk[0]).mean())
        mae_chunk = _chunk_mae(pred_chunk, gt_chunk, is_pad)
        chunk0_maes.append(mae0)
        chunk_maes.append(mae_chunk)
        results.append({
            'raw_frame': int(raw_frame_idx),
            'qpos': qpos[k].tolist(),
            'pred_chunk': pred_chunk.tolist(),
            'gt_chunk': gt_chunk.tolist(),
            'is_pad': is_pad.tolist(),
            'chunk0_mae': mae0,
            'chunk_mae': mae_chunk,
        })

    ckpt_path = getattr(inferencer, '_checkpoint_path', None)
    summary = {
        'episode': episode,
        'num_frames': len(frame_indices),
        'chunk_size': chunk_size,
        'frame_range': [frame_indices[0], frame_indices[-1]],
        'action_space': action_space,
        'camera_names': camera_names,
        'checkpoint': str(ckpt_path) if ckpt_path else None,
        'mean_chunk0_mae': float(np.mean(chunk0_maes)),
        'median_chunk0_mae': float(np.median(chunk0_maes)),
        'mean_chunk_mae': float(np.nanmean(chunk_maes)),
        'median_chunk_mae': float(np.nanmedian(chunk_maes)),
    }
    if verbose:
        print(f'  mean chunk0_mae={summary["mean_chunk0_mae"]:.4f}  '
              f'mean chunk_mae={summary["mean_chunk_mae"]:.4f}')
    return {'summary': summary, 'frames': results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--episode', type=int, required=True,
                        help='Numeric episode folder name under raw-dir')
    parser.add_argument('--ckpt-dir', type=Path, required=True)
    parser.add_argument('--ckpt-name', default='policy_best.ckpt')
    parser.add_argument('--annotation-dir', type=Path, default=None)
    parser.add_argument('--camera-names', nargs='+',
                        default=['chest', 'top', 'wrist_2'])
    parser.add_argument('--action-space', default='cartesian_abs',
                        choices=['joint', 'cartesian_abs', 'cartesian'])
    parser.add_argument('--unwrap-rx', action='store_true')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Cap number of frames to infer (after slice/stride)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Optional JSON path for per-frame predictions')
    args = parser.parse_args()

    ckpt_path = args.ckpt_dir / args.ckpt_name
    stats_path = args.ckpt_dir / 'dataset_stats.pkl'
    inferencer = ACTInference(str(ckpt_path), str(stats_path))
    inferencer._checkpoint_path = str(ckpt_path)  # noqa: SLF001 — for summary metadata

    payload = infer_episode(
        inferencer,
        episode=args.episode,
        raw_dir=args.raw_dir,
        annotation_dir=args.annotation_dir,
        camera_names=args.camera_names,
        action_space=args.action_space,
        unwrap_rx=args.unwrap_rx,
        stride=args.stride,
        max_frames=args.max_frames,
        verbose=True,
    )
    summary = payload['summary']
    print(f'\nMean chunk[0] MAE: {summary["mean_chunk0_mae"]:.4f}  '
          f'median: {summary["median_chunk0_mae"]:.4f}')
    print(f'Mean full-chunk MAE (unpadded): {summary["mean_chunk_mae"]:.4f}  '
          f'median: {summary["median_chunk_mae"]:.4f}')

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
