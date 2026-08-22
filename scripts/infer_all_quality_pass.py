#!/usr/bin/env python3
"""Batch offline inference on all episodes in quality_pass.json.

Loads the policy once, runs infer_episode() per whitelist entry, and writes:
  <output-dir>/episode_<id>.json
  <output-dir>/manifest.json

Usage:
    python scripts/infer_all_quality_pass.py \\
        --filter-json data/quality_pass.json \\
        --raw-dir data/raw \\
        --annotation-dir data/annotation/annotation \\
        --ckpt-dir data/ckpt_cam100_15_cart_abs_v1 \\
        --output-dir data/infer_cam100_15_cart_abs_v1 \\
        --unwrap-rx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

sys.path.insert(0, str(_ROOT / 'scripts'))

from infer_from_raw import infer_episode  # noqa: E402
from serve import ACTInference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--filter-json', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--annotation-dir', type=Path, required=True)
    parser.add_argument('--ckpt-dir', type=Path, required=True)
    parser.add_argument('--ckpt-name', default='policy_best.ckpt')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--camera-names', nargs='+', default=['chest', 'top', 'wrist_2'])
    parser.add_argument('--action-space', default='cartesian_abs',
                        choices=['joint', 'cartesian_abs', 'cartesian'])
    parser.add_argument('--unwrap-rx', action='store_true')
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()

    episode_ids = json.loads(args.filter_json.read_text(encoding='utf-8'))
    if not isinstance(episode_ids, list):
        raise SystemExit(f'{args.filter_json} must be a JSON array')
    episode_ids = sorted(int(x) for x in episode_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.ckpt_dir / args.ckpt_name

    t_load = time.perf_counter()
    inferencer = ACTInference(str(ckpt_path), str(args.ckpt_dir / 'dataset_stats.pkl'))
    inferencer._checkpoint_path = str(ckpt_path)  # noqa: SLF001
    load_sec = time.perf_counter() - t_load
    print(f'Loaded model in {load_sec:.2f}s  |  {len(episode_ids)} episodes to infer')

    manifest_episodes: list[dict] = []
    errors: list[dict] = []
    total_frames = 0
    infer_sec_total = 0.0

    t_all = time.perf_counter()
    for i, ep in enumerate(episode_ids, 1):
        out_path = args.output_dir / f'episode_{ep}.json'
        t0 = time.perf_counter()
        try:
            payload = infer_episode(
                inferencer,
                episode=ep,
                raw_dir=args.raw_dir,
                annotation_dir=args.annotation_dir,
                camera_names=args.camera_names,
                action_space=args.action_space,
                unwrap_rx=args.unwrap_rx,
                stride=args.stride,
                verbose=True,
            )
            out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            ep_sec = time.perf_counter() - t0
            infer_sec_total += ep_sec
            s = payload['summary']
            s['infer_sec'] = ep_sec
            s['output_file'] = out_path.name
            manifest_episodes.append(s)
            total_frames += s['num_frames']
            print(f'[{i}/{len(episode_ids)}] ep {ep}  {s["num_frames"]} frames  '
                  f'{ep_sec:.1f}s  chunk0_mae={s["mean_chunk0_mae"]:.4f}')
        except Exception as exc:
            ep_sec = time.perf_counter() - t0
            errors.append({'episode': ep, 'error': str(exc), 'infer_sec': ep_sec})
            print(f'[{i}/{len(episode_ids)}] ep {ep}  FAILED: {exc}')

    total_sec = time.perf_counter() - t_all
    ok = len(manifest_episodes)
    manifest = {
        'filter_json': str(args.filter_json),
        'ckpt_dir': str(args.ckpt_dir),
        'ckpt_name': args.ckpt_name,
        'action_space': args.action_space,
        'camera_names': args.camera_names,
        'unwrap_rx': args.unwrap_rx,
        'num_episodes_requested': len(episode_ids),
        'num_episodes_ok': ok,
        'num_episodes_failed': len(errors),
        'total_frames': total_frames,
        'load_sec': load_sec,
        'infer_sec_total': infer_sec_total,
        'total_sec': total_sec,
        'ms_per_frame': (infer_sec_total / total_frames * 1000) if total_frames else None,
        'episodes': manifest_episodes,
        'errors': errors,
    }
    if manifest_episodes:
        manifest['mean_chunk0_mae'] = float(
            sum(e['mean_chunk0_mae'] for e in manifest_episodes) / ok
        )
        manifest['mean_chunk_mae'] = float(
            sum(e['mean_chunk_mae'] for e in manifest_episodes) / ok
        )

    manifest_path = args.output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print(f'\nDone: {ok}/{len(episode_ids)} episodes  |  {total_frames} frames')
    print(f'Load {load_sec:.2f}s  infer {infer_sec_total:.2f}s  total {total_sec:.2f}s')
    if total_frames:
        print(f'Throughput: {infer_sec_total/total_frames*1000:.1f} ms/frame')
    print(f'Output: {args.output_dir}/')
    print(f'Manifest: {manifest_path}')
    if errors:
        print(f'Failures: {len(errors)} — see manifest.errors')


if __name__ == '__main__':
    main()
