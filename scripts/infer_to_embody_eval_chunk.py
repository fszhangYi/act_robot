#!/usr/bin/env python3
"""Convert act_robot infer JSON → per-observation chunk JSONs for embody_model_eval.

For each observation frame in an infer episode, writes one JSON containing
all valid action-chunk steps (pred vs GT), e.g. episode with 120 infer frames
→ 120 files under ``<output-dir>/episode_<id>/frame_<raw_frame>.json``.

Each output file has ``frames[]`` length = valid chunk steps (≤ chunk_size;
shorter near episode end when ``is_pad`` is set).

Usage:
    # One episode → many frame JSONs
    python scripts/infer_to_embody_eval_chunk.py \\
        --infer-json data/infer_cam100_15_cart_abs_v1/episode_3.json \\
        --raw-dir data/raw \\
        --output-dir /root/autodl-tmp/embody_model_eval/data/ec616_act_chunk

    # Batch all episodes in infer dir
    python scripts/infer_to_embody_eval_chunk.py \\
        --infer-dir data/infer_cam100_15_cart_abs_v1 \\
        --raw-dir data/raw \\
        --output-dir /root/autodl-tmp/embody_model_eval/data/ec616_act_chunk \\
        --suite ec616_act_chunk
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

from convert_episodes import _parse_gripper  # noqa: E402
from ec616_ik import DEFAULT_TCP_TOOL_Z_M, solve_ik_cartesian_to_joint  # noqa: E402
from infer_to_embody_eval import (  # noqa: E402
    EC616_JOINT_NAMES,
    IkOptions,
    _infer_episode_id,
    _load_steps,
    build_embody_payload,
    to_ec616_vec8,
)


def _clamp_raw_index(raw_frame: int, num_frames: int) -> int:
    return int(np.clip(raw_frame, 0, num_frames - 1))


def _action7_to_vec8(
    action7: np.ndarray,
    *,
    action_space: str,
    seed_joint6_rad: np.ndarray,
    ik: IkOptions,
    ik_failures: list[int],
) -> list[float]:
    action = np.asarray(action7, dtype=np.float64).reshape(7)
    if action_space == 'joint':
        joint6 = action[:6]
    else:
        joint6, ok = solve_ik_cartesian_to_joint(
            action[:6],
            seed_joint6_rad,
            tcp_tool_z_m=ik.tcp_tool_z_m,
            enforce_soft_limits=ik.enforce_soft_limits,
            fallback_to_seed=ik.fallback_to_seed,
        )
        if not ok:
            ik_failures[0] += 1
    return to_ec616_vec8(joint6, action[6])


def convert_infer_frame_chunk(
    infer_frame: dict[str, Any],
    *,
    episode: int,
    action_space: str,
    joint_pos: np.ndarray,
    grip_obs: np.ndarray,
    fps: float,
    suite: str,
    ik: IkOptions,
    policy: str | None,
    ckpt: str | None,
    infer_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """One infer record (one observation) → embody payload with chunk steps."""
    ri = int(infer_frame['raw_frame'])
    t_total = len(joint_pos)
    if ri < 0 or ri >= t_total:
        raise ValueError(f'raw_frame {ri} out of range [0, {t_total}) for episode {episode}')

    pred_chunk = np.asarray(infer_frame['pred_chunk'], dtype=np.float64)
    gt_chunk = np.asarray(infer_frame['gt_chunk'], dtype=np.float64)
    is_pad = np.asarray(infer_frame.get('is_pad', []), dtype=bool)
    if pred_chunk.ndim != 2 or gt_chunk.ndim != 2:
        raise ValueError('pred_chunk / gt_chunk must be 2-D arrays')
    if pred_chunk.shape != gt_chunk.shape:
        raise ValueError(
            f'pred_chunk shape {pred_chunk.shape} != gt_chunk shape {gt_chunk.shape}',
        )
    chunk_size = pred_chunk.shape[0]
    if is_pad.size == 0:
        is_pad = np.zeros(chunk_size, dtype=bool)
    elif is_pad.size != chunk_size:
        raise ValueError(f'is_pad length {is_pad.size} != chunk_size {chunk_size}')

    valid_steps = [j for j in range(chunk_size) if not is_pad[j]]
    if not valid_steps:
        raise ValueError(f'no valid chunk steps for raw_frame {ri}')

    ik_failures = [0]
    states: list[list[float]] = []
    gt_actions: list[list[float]] = []
    pred_actions: list[list[float]] = []

    for j in valid_steps:
        state_idx = _clamp_raw_index(ri + j, t_total)
        seed = joint_pos[state_idx]
        states.append(to_ec616_vec8(seed, grip_obs[state_idx]))
        gt_actions.append(
            _action7_to_vec8(
                gt_chunk[j],
                action_space=action_space,
                seed_joint6_rad=seed,
                ik=ik,
                ik_failures=ik_failures,
            ),
        )
        pred_actions.append(
            _action7_to_vec8(
                pred_chunk[j],
                action_space=action_space,
                seed_joint6_rad=seed,
                ik=ik,
                ik_failures=ik_failures,
            ),
        )

    chunk_mae = float(np.abs(pred_chunk[valid_steps] - gt_chunk[valid_steps]).mean())
    chunk0_mae = float(np.abs(pred_chunk[valid_steps[0]] - gt_chunk[valid_steps[0]]).mean())

    return build_embody_payload(
        states=states,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        joint_names=EC616_JOINT_NAMES,
        fps=fps,
        robot_id='ec616',
        action_mode='absolute',
        title=f'EC616 ACT chunk · {suite} · episode_{episode} · frame_{ri}',
        source=f'infer_to_embody_chunk:{suite}/episode_{episode}/frame_{ri}',
        policy=policy,
        model=action_space,
        ckpt=ckpt,
        infer_summary=infer_summary,
        extra_meta={
            'layout': 'per_obs_chunk',
            'obs_raw_frame': ri,
            'chunk_size': int(chunk_size),
            'chunk_steps': len(valid_steps),
            'chunk_step_indices': valid_steps,
            'is_pad': [bool(x) for x in is_pad.tolist()],
            'chunk0_mae': chunk0_mae,
            'chunk_mae': chunk_mae,
            'infer_chunk0_mae': float(infer_frame.get('chunk0_mae', chunk0_mae)),
            'infer_chunk_mae': float(infer_frame.get('chunk_mae', chunk_mae)),
            'ik_backend': 'ec616_kin.ik_flange',
            'tcp_tool_z_m': ik.tcp_tool_z_m,
            'ik_failures': ik_failures[0],
        },
    )


def convert_infer_file_chunks(
    infer_path: Path,
    *,
    raw_dir: Path,
    output_dir: Path,
    fps: float,
    suite: str,
    ik_options: IkOptions | None = None,
) -> dict[str, Any]:
    payload = json.loads(infer_path.read_text(encoding='utf-8'))
    if 'summary' not in payload:
        payload = {'summary': {}, 'frames': payload.get('frames', payload)}
    summary = payload.get('summary') or {}
    frames_in = payload.get('frames')
    if not frames_in:
        raise ValueError('infer payload has no frames')

    episode = int(summary.get('episode', _infer_episode_id(infer_path)))
    action_space = str(summary.get('action_space', 'cartesian_abs'))
    steps = _load_steps(raw_dir, episode)
    obs = steps['observations']
    joint_pos = np.array(obs['joint_position'], dtype=np.float64)
    grip_obs = _parse_gripper(obs['gripper_position'])

    ckpt = summary.get('checkpoint')
    policy = Path(ckpt).parent.name if ckpt else None
    ik = ik_options or IkOptions()

    ep_out = output_dir / f'episode_{episode}'
    ep_out.mkdir(parents=True, exist_ok=True)

    written: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total_ik_failures = 0

    for fr in frames_in:
        ri = int(fr['raw_frame'])
        out_path = ep_out / f'frame_{ri:04d}.json'
        try:
            out = convert_infer_frame_chunk(
                fr,
                episode=episode,
                action_space=action_space,
                joint_pos=joint_pos,
                grip_obs=grip_obs,
                fps=fps,
                suite=suite,
                ik=ik,
                policy=policy,
                ckpt=ckpt,
                infer_summary=summary,
            )
            out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
            total_ik_failures += int(out['meta'].get('ik_failures', 0))
            written.append({
                'raw_frame': ri,
                'path': str(out_path),
                'chunk_steps': out['meta']['chunk_steps'],
                'mean_l2': out['meta']['mean_l2'],
            })
        except Exception as exc:
            errors.append({'raw_frame': ri, 'error': str(exc)})

    manifest = {
        'episode': episode,
        'suite': suite,
        'infer_file': str(infer_path),
        'output_dir': str(ep_out),
        'num_frames_written': len(written),
        'num_failed': len(errors),
        'total_ik_failures': total_ik_failures,
        'frames': written,
        'errors': errors,
    }
    manifest_path = ep_out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--infer-json', type=Path, help='Single infer episode JSON')
    src.add_argument('--infer-dir', type=Path, help='Directory of episode_*.json infer outputs')

    parser.add_argument('--raw-dir', type=Path, required=True, help='data/raw root')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output root directory')
    parser.add_argument('--suite', default='ec616_act_chunk', help='Suite name for meta.source')
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
    parser.add_argument(
        '--tcp-tool-z-m',
        type=float,
        default=DEFAULT_TCP_TOOL_Z_M,
        help='Flange→TCP translation along flange Z (meters); default 0.18',
    )
    parser.add_argument(
        '--ik-enforce-limits',
        action='store_true',
        help='Use TRF IK with teach soft joint limits',
    )
    args = parser.parse_args()

    ik_options = IkOptions(
        tcp_tool_z_m=args.tcp_tool_z_m,
        enforce_soft_limits=args.ik_enforce_limits,
    )

    t0 = time.perf_counter()
    episode_errors: list[dict[str, Any]] = []
    converted_episodes = 0

    if args.infer_json:
        try:
            manifest = convert_infer_file_chunks(
                args.infer_json,
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                fps=args.fps,
                suite=args.suite,
                ik_options=ik_options,
            )
            converted_episodes = 1
            print(
                f'episode {manifest["episode"]}: wrote {manifest["num_frames_written"]} frame JSONs '
                f'to {manifest["output_dir"]}  '
                f'(failed={manifest["num_failed"]}, ik_failures={manifest["total_ik_failures"]})',
            )
            if manifest['errors']:
                for err in manifest['errors']:
                    print(f'  frame {err["raw_frame"]}: {err["error"]}', file=sys.stderr)
        except Exception as exc:
            print(f'FAILED {args.infer_json}: {exc}', file=sys.stderr)
            return 1
    else:
        paths = sorted(args.infer_dir.glob('episode_*.json'))
        if args.limit > 0:
            paths = paths[: args.limit]
        if not paths:
            print(f'No episode_*.json under {args.infer_dir}', file=sys.stderr)
            return 1

        batch_manifest_eps: list[dict[str, Any]] = []
        for i, infer_path in enumerate(paths, 1):
            ep = _infer_episode_id(infer_path)
            try:
                manifest = convert_infer_file_chunks(
                    infer_path,
                    raw_dir=args.raw_dir,
                    output_dir=args.output_dir,
                    fps=args.fps,
                    suite=args.suite,
                    ik_options=ik_options,
                )
                converted_episodes += 1
                batch_manifest_eps.append(manifest)
                print(
                    f'[{i}/{len(paths)}] ep {ep}: {manifest["num_frames_written"]} frames  '
                    f'failed={manifest["num_failed"]}',
                )
            except Exception as exc:
                episode_errors.append({'episode': ep, 'file': str(infer_path), 'error': str(exc)})
                print(f'[{i}/{len(paths)}] ep {ep}  FAILED: {exc}', file=sys.stderr)

        batch_manifest = {
            'suite': args.suite,
            'infer_dir': str(args.infer_dir),
            'output_dir': str(args.output_dir),
            'num_episodes_converted': converted_episodes,
            'num_episodes_failed': len(episode_errors),
            'elapsed_sec': time.perf_counter() - t0,
            'episodes': batch_manifest_eps,
            'errors': episode_errors,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        batch_manifest_path = args.output_dir / 'manifest.json'
        batch_manifest_path.write_text(json.dumps(batch_manifest, indent=2), encoding='utf-8')
        print(f'\nDone: {converted_episodes}/{len(paths)} episodes  manifest: {batch_manifest_path}')

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

    return 1 if episode_errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
