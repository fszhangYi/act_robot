#!/usr/bin/env python3
"""Convert act_robot infer JSON → embody_model_eval EC616 episode JSON.

Maps offline inference (cartesian_abs or joint) plus raw steps.json joint
logs into the viewer schema: meta + series + frames with 8-element joint
vectors (6 arm joints in degrees + mirrored gripper pair).

IK for cartesian predictions is a placeholder (returns seed joints unchanged).

Usage:
    # Single episode
    python scripts/infer_to_embody_eval.py \\
        --infer-json data/infer_cam100_15_cart_abs_v1/episode_3.json \\
        --raw-dir data/raw \\
        --output /root/autodl-tmp/embody_model_eval/data/ec616_act/episode_3.json

    # Batch (all episode_*.json in infer dir)
    python scripts/infer_to_embody_eval.py \\
        --infer-dir data/infer_cam100_15_cart_abs_v1 \\
        --raw-dir data/raw \\
        --output-dir /root/autodl-tmp/embody_model_eval/data/ec616_act \\
        --suite ec616_act
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from convert_episodes import _parse_gripper  # noqa: E402

EC616_JOINT_NAMES = [
    'Joint1',
    'Joint2',
    'Joint3',
    'Joint4',
    'Joint5',
    'Joint6',
    'gripper_1_joint',
    'gripper_2_joint',
]

DEFAULT_GOAL_EC616 = {
    'pos': {'x': 0.42, 'y': 0.05, 'z': 0.12},
    'quat': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
    'approach': {'x': 0.0, 'y': 0.0, 'z': -1.0},
    'frame': 'world',
    'note': 'default EC616 grasp goal (placeholder)',
}


def solve_ik_cartesian_to_joint(
    cartesian6: np.ndarray,
    seed_joint6_rad: np.ndarray,
) -> np.ndarray:
    """Placeholder IK: cartesian target → 6 joint angles (rad).

    TODO: replace with EC616 inverse kinematics.
    Currently returns seed joints unchanged.
    """
    _ = cartesian6
    return np.asarray(seed_joint6_rad, dtype=np.float64).copy()


def gripper_pair_deg(grip_rad: float) -> tuple[float, float]:
    """Single gripper command (rad) → mirrored pair in degrees for URDF."""
    g = float(np.clip(np.degrees(grip_rad), 0.0, 40.0))
    return g, -g


def to_ec616_vec8(joint6_rad: np.ndarray, grip_rad: float) -> list[float]:
    """6 arm joints (rad) + 1 gripper (rad) → 8 viewer joint values (deg)."""
    j = np.degrees(np.asarray(joint6_rad, dtype=np.float64).reshape(6))
    g1, g2 = gripper_pair_deg(grip_rad)
    return [float(x) for x in j] + [g1, g2]


def _load_steps(raw_dir: Path, episode: int) -> dict[str, Any]:
    path = raw_dir / str(episode) / 'steps.json'
    if not path.is_file():
        raise FileNotFoundError(f'missing raw steps: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _next_raw_index(raw_frame: int, num_frames: int) -> int:
    return min(int(raw_frame) + 1, num_frames - 1)


def convert_infer_episode(
    infer_payload: dict[str, Any],
    *,
    raw_dir: Path,
    fps: float = 30.0,
    suite: str = 'ec616_act',
) -> dict[str, Any]:
    """Convert one infer JSON payload to embody_model_eval episode dict."""
    summary = infer_payload.get('summary') or {}
    frames_in = infer_payload.get('frames')
    if not frames_in:
        raise ValueError('infer payload has no frames')

    episode = int(summary.get('episode', infer_payload.get('episode', 0)))
    if episode <= 0:
        # infer files are episode_<id>.json; caller may pass episode via summary only
        raise ValueError('cannot determine episode id (set summary.episode)')

    action_space = str(summary.get('action_space', 'cartesian_abs'))
    steps = _load_steps(raw_dir, episode)
    obs = steps['observations']
    joint_pos = np.array(obs['joint_position'], dtype=np.float64)
    grip_obs = _parse_gripper(obs['gripper_position'])
    grip_action = _parse_gripper(steps['actions']['gripper_position'])
    t_total = len(joint_pos)

    ckpt = summary.get('checkpoint')
    policy = Path(ckpt).parent.name if ckpt else None

    states: list[list[float]] = []
    gt_actions: list[list[float]] = []
    pred_actions: list[list[float]] = []

    for fr in frames_in:
        ri = int(fr['raw_frame'])
        if ri < 0 or ri >= t_total:
            raise ValueError(f'raw_frame {ri} out of range [0, {t_total}) for episode {episode}')

        ni = _next_raw_index(ri, t_total)
        current = to_ec616_vec8(joint_pos[ri], grip_obs[ri])
        next_gt = to_ec616_vec8(joint_pos[ni], grip_action[ni])

        pred0 = np.asarray(fr['pred_chunk'][0], dtype=np.float64)
        if pred0.shape[0] < 7:
            raise ValueError(f'pred_chunk[0] must have 7 elements, got {pred0.shape[0]}')

        if action_space == 'joint':
            pred_joint6 = pred0[:6]
        else:
            pred_joint6 = solve_ik_cartesian_to_joint(pred0[:6], joint_pos[ri])

        next_pred = to_ec616_vec8(pred_joint6, pred0[6])

        states.append(current)
        gt_actions.append(next_gt)
        pred_actions.append(next_pred)

    return build_embody_payload(
        states=states,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        joint_names=EC616_JOINT_NAMES,
        fps=fps,
        robot_id='ec616',
        action_mode='absolute',
        title=f'EC616 ACT infer · {suite} · episode_{episode}',
        source=f'infer_to_embody:{suite}/episode_{episode}',
        policy=policy,
        model=action_space,
        ckpt=ckpt,
        infer_summary=summary,
    )


def build_embody_payload(
    *,
    states: list[list[float]],
    gt_actions: list[list[float]],
    pred_actions: list[list[float]],
    joint_names: list[str],
    fps: float,
    robot_id: str,
    action_mode: str,
    title: str,
    source: str,
    policy: str | None = None,
    model: str | None = None,
    ckpt: str | None = None,
    infer_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n = len(states)
    dof = len(joint_names)
    if n == 0:
        raise ValueError('empty trajectory')
    for label, rows in [('states', states), ('gt_actions', gt_actions), ('pred_actions', pred_actions)]:
        if len(rows) != n:
            raise ValueError(f'{label} length {len(rows)} != {n}')
        for i, row in enumerate(rows):
            if len(row) != dof:
                raise ValueError(f'{label}[{i}] length {len(row)} != dof {dof}')

    if action_mode == 'relative':
        next_gt = [
            [states[i][j] + gt_actions[i][j] for j in range(dof)] for i in range(n)
        ]
        next_pred = [
            [states[i][j] + pred_actions[i][j] for j in range(dof)] for i in range(n)
        ]
    else:
        next_gt = [row[:] for row in gt_actions]
        next_pred = [row[:] for row in pred_actions]

    err = [[next_pred[i][j] - next_gt[i][j] for j in range(dof)] for i in range(n)]
    err_l2 = [math.sqrt(sum(v * v for v in row)) for row in err]
    err_abs_mean = [sum(abs(err[i][j]) for i in range(n)) / n for j in range(dof)]

    frames_out: list[dict[str, Any]] = []
    for i in range(n):
        item: dict[str, Any] = {
            't': i,
            'timestamp': float(i / fps),
            'current': states[i][:],
            'action_gt': gt_actions[i][:],
            'action_pred': pred_actions[i][:],
            'next_gt': next_gt[i][:],
            'next_pred': next_pred[i][:],
            'err': err[i][:],
            'err_l2': float(err_l2[i]),
        }
        if i + 1 < n:
            item['next_obs'] = states[i + 1][:]
        else:
            item['next_obs'] = states[i][:]
        frames_out.append(item)

    meta: dict[str, Any] = {
        'title': title,
        'robot': robot_id,
        'joint_names': joint_names[:],
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_frames': int(n),
        'action_mode': action_mode,
        'mean_l2': float(sum(err_l2) / n),
        'max_l2': float(max(err_l2)),
        'per_joint_mae': {name: float(err_abs_mean[i]) for i, name in enumerate(joint_names)},
        'source': source,
        'fps': float(fps),
        'goal_pose': json.loads(json.dumps(DEFAULT_GOAL_EC616)),
    }
    if policy:
        meta['policy'] = policy
    if model:
        meta['model'] = model
    if ckpt:
        meta['ckpt'] = ckpt
    if infer_summary:
        meta['infer_summary'] = {
            k: infer_summary[k]
            for k in (
                'mean_chunk0_mae',
                'mean_chunk_mae',
                'frame_range',
                'camera_names',
                'chunk_size',
            )
            if k in infer_summary
        }

    return {
        'meta': meta,
        'series': {
            't': list(range(n)),
            'err_l2': [float(x) for x in err_l2],
            'current': [row[:] for row in states],
            'next_gt': next_gt,
            'next_pred': next_pred,
            'per_joint_err': err,
        },
        'frames': frames_out,
    }


def _infer_episode_id(path: Path) -> int:
    stem = path.stem
    if stem.startswith('episode_'):
        return int(stem.split('_', 1)[1])
    raise ValueError(f'cannot parse episode id from {path.name}')


def convert_infer_file(
    infer_path: Path,
    *,
    raw_dir: Path,
    output_path: Path,
    fps: float,
    suite: str,
) -> dict[str, Any]:
    payload = json.loads(infer_path.read_text(encoding='utf-8'))
    if 'summary' not in payload:
        payload = {'summary': {}, 'frames': payload.get('frames', payload)}
    ep = _infer_episode_id(infer_path)
    payload.setdefault('summary', {})['episode'] = ep
    out = convert_infer_episode(payload, raw_dir=raw_dir, fps=fps, suite=suite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--infer-json', type=Path, help='Single infer episode JSON')
    src.add_argument('--infer-dir', type=Path, help='Directory of episode_*.json infer outputs')

    parser.add_argument('--raw-dir', type=Path, required=True, help='data/raw root')
    parser.add_argument('--output', type=Path, help='Output path (single-episode mode)')
    parser.add_argument('--output-dir', type=Path, help='Output directory (batch mode)')
    parser.add_argument('--suite', default='ec616_act', help='Suite name for meta.source')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--limit', type=int, default=0, help='Max episodes in batch (0=all)')
    parser.add_argument(
        '--refresh-index',
        action='store_true',
        help='Run embody_model_eval/scripts/refresh_data_index.py after batch convert',
    )
    parser.add_argument(
        '--embody-root',
        type=Path,
        default=Path('/root/autodl-tmp/embody_model_eval'),
        help='embody_model_eval repo root (for --refresh-index)',
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    converted = 0
    errors: list[dict[str, Any]] = []

    if args.infer_json:
        if not args.output:
            parser.error('--output is required with --infer-json')
        try:
            out = convert_infer_file(
                args.infer_json,
                raw_dir=args.raw_dir,
                output_path=args.output,
                fps=args.fps,
                suite=args.suite,
            )
            converted = 1
            print(f'Wrote {args.output}  ({out["meta"]["n_frames"]} frames, '
                  f'mean_l2={out["meta"]["mean_l2"]:.4f})')
        except Exception as exc:
            print(f'FAILED {args.infer_json}: {exc}', file=sys.stderr)
            return 1
    else:
        if not args.output_dir:
            parser.error('--output-dir is required with --infer-dir')
        paths = sorted(args.infer_dir.glob('episode_*.json'))
        if args.limit > 0:
            paths = paths[: args.limit]
        if not paths:
            print(f'No episode_*.json under {args.infer_dir}', file=sys.stderr)
            return 1

        for i, infer_path in enumerate(paths, 1):
            ep = _infer_episode_id(infer_path)
            out_path = args.output_dir / f'episode_{ep}.json'
            try:
                out = convert_infer_file(
                    infer_path,
                    raw_dir=args.raw_dir,
                    output_path=out_path,
                    fps=args.fps,
                    suite=args.suite,
                )
                converted += 1
                print(f'[{i}/{len(paths)}] ep {ep}  {out["meta"]["n_frames"]} frames  '
                      f'mean_l2={out["meta"]["mean_l2"]:.4f}')
            except Exception as exc:
                errors.append({'episode': ep, 'file': str(infer_path), 'error': str(exc)})
                print(f'[{i}/{len(paths)}] ep {ep}  FAILED: {exc}')

        manifest = {
            'suite': args.suite,
            'infer_dir': str(args.infer_dir),
            'output_dir': str(args.output_dir),
            'num_converted': converted,
            'num_failed': len(errors),
            'elapsed_sec': time.perf_counter() - t0,
            'errors': errors,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_dir / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'\nDone: {converted}/{len(paths)}  manifest: {manifest_path}')

        if args.refresh_index:
            refresh = args.embody_root / 'scripts' / 'refresh_data_index.py'
            if refresh.is_file():
                import subprocess
                subprocess.run(
                    [sys.executable, str(refresh)],
                    cwd=str(args.embody_root),
                    check=True,
                )
                print(f'Refreshed {args.embody_root}/data/index.json')
            else:
                print(f'warn: refresh script not found: {refresh}', file=sys.stderr)

    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
