#!/usr/bin/env python3
"""Episode quality filter based on gripper_position in steps.json.

Rules (observations.gripper_position):
  1. Overall trend: start near 0, open to a peak, end near 0 again.
  2. Reject if >= N consecutive frames equal the stuck-sensor sentinel value.

Usage:
    python scripts/check_episode_quality.py \\
        --input-dir data/raw \\
        --write-pass-json data/quality_pass.json \\
        --write-fail-list data/quality_fail.txt \\
        --output-json data/quality_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_SENTINEL = 0.14651000499725342


def parse_gripper(raw) -> np.ndarray:
    """Normalise gripper array to 1-D (T,) regardless of storage shape."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr[0]
        return arr.squeeze(-1)
    return arr


def parse_annotation(annot_path: Path) -> tuple[int, int] | None:
    """Return (start, end) inclusive frame indices, or None if invalid."""
    if not annot_path.exists():
        return None
    lines = annot_path.read_text().splitlines()
    if len(lines) < 3:
        return None
    start = int(lines[0].split()[1])
    end = int(lines[2].split()[1])
    if start < 0 or end < 0 or end < start:
        return None
    return start, end


def check_gripper_trend(
    gripper: np.ndarray,
    *,
    zero_eps: float,
    min_peak: float,
) -> tuple[bool, str]:
    """Rule 1: closed -> open -> closed."""
    if gripper.size < 3:
        return False, 'too_short'

    start = float(gripper[0])
    end = float(gripper[-1])
    if start > zero_eps:
        return False, 'start_not_zero'
    if end > zero_eps:
        return False, 'end_not_zero'

    peak_idx = int(np.argmax(gripper))
    peak = float(gripper[peak_idx])
    if peak < min_peak:
        return False, 'no_meaningful_open'
    if peak_idx == 0 or peak_idx == gripper.size - 1:
        return False, 'peak_at_boundary'
    if peak - start < min_peak:
        return False, 'insufficient_rise'
    if peak - end < min_peak:
        return False, 'insufficient_fall'

    return True, 'ok'


def check_sentinel_streak(
    gripper: np.ndarray,
    *,
    sentinel: float,
    min_run: int,
    value_tol: float,
) -> tuple[bool, str, int]:
    """Rule 2: reject long runs stuck at the sentinel value."""
    run = 0
    max_run = 0
    for value in gripper:
        if abs(float(value) - sentinel) <= value_tol:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    if max_run >= min_run:
        return False, f'sentinel_streak_{max_run}', max_run
    return True, 'ok', max_run


@dataclass
class EpisodeQualityResult:
    episode: str
    passed: bool
    issues: list[str] = field(default_factory=list)
    frames: int = 0
    peak: float = 0.0
    peak_idx: int = -1
    max_sentinel_streak: int = 0


def evaluate_episode(
    ep_dir: Path,
    *,
    annotation_dir: Path | None,
    zero_eps: float,
    min_peak: float,
    sentinel: float,
    sentinel_min_run: int,
    value_tol: float,
) -> EpisodeQualityResult:
    steps_path = ep_dir / 'steps.json'
    if not steps_path.is_file():
        return EpisodeQualityResult(
            episode=ep_dir.name,
            passed=False,
            issues=['missing_steps_json'],
        )

    steps = json.loads(steps_path.read_text())
    gripper = parse_gripper(steps['observations']['gripper_position'])

    if annotation_dir is not None:
        bounds = parse_annotation(annotation_dir / f'{ep_dir.name}.txt')
        if bounds is None:
            return EpisodeQualityResult(
                episode=ep_dir.name,
                passed=False,
                issues=['invalid_or_missing_annotation'],
            )
        start, end = bounds
        gripper = gripper[start:end + 1]

    issues: list[str] = []
    trend_ok, trend_msg = check_gripper_trend(
        gripper, zero_eps=zero_eps, min_peak=min_peak,
    )
    if not trend_ok:
        issues.append(f'trend:{trend_msg}')

    sentinel_ok, sentinel_msg, max_run = check_sentinel_streak(
        gripper,
        sentinel=sentinel,
        min_run=sentinel_min_run,
        value_tol=value_tol,
    )
    if not sentinel_ok:
        issues.append(sentinel_msg)

    peak_idx = int(np.argmax(gripper)) if gripper.size else -1
    peak = float(gripper[peak_idx]) if gripper.size else 0.0

    return EpisodeQualityResult(
        episode=ep_dir.name,
        passed=not issues,
        issues=issues,
        frames=int(gripper.size),
        peak=peak,
        peak_idx=peak_idx,
        max_sentinel_streak=max_run,
    )


def discover_episode_dirs(input_dir: Path) -> list[Path]:
    eps = [p for p in input_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(eps, key=lambda p: int(p.name))


def write_name_list(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(names) + ('\n' if names else ''), encoding='utf-8')


def write_pass_indices_json(path: Path, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(indices, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Filter episodes by gripper quality rules.')
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Directory of raw episode folders (each with steps.json).')
    parser.add_argument('--annotation-dir', type=Path, default=None,
                        help='Optional: evaluate gripper only inside annotated frame range.')
    parser.add_argument('--output-json', type=Path, default=None,
                        help='Write full per-episode report as JSON.')
    parser.add_argument('--write-pass-json', type=Path, default=None,
                        help='Write passing episode indices as a JSON array, e.g. [0, 1, 5].')
    parser.add_argument('--write-fail-list', type=Path, default=None,
                        help='Write one failing episode name per line.')
    parser.add_argument('--zero-eps', type=float, default=1e-3,
                        help='Treat gripper values <= this as closed (default: 1e-3).')
    parser.add_argument('--min-peak', type=float, default=0.05,
                        help='Minimum peak opening required (default: 0.05).')
    parser.add_argument('--sentinel-value', type=float, default=DEFAULT_SENTINEL,
                        help='Stuck-sensor gripper reading to detect.')
    parser.add_argument('--sentinel-min-run', type=int, default=20,
                        help='Reject if sentinel repeats this many frames in a row.')
    parser.add_argument('--value-tol', type=float, default=1e-9,
                        help='Absolute tolerance when matching sentinel_value.')
    parser.add_argument('--strict', action='store_true',
                        help='Exit with code 1 if any episode fails.')
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f'input-dir not found: {args.input_dir}')

    episode_dirs = discover_episode_dirs(args.input_dir)
    if not episode_dirs:
        parser.error(f'no numeric episode directories under {args.input_dir}')

    results = [
        evaluate_episode(
            ep_dir,
            annotation_dir=args.annotation_dir,
            zero_eps=args.zero_eps,
            min_peak=args.min_peak,
            sentinel=args.sentinel_value,
            sentinel_min_run=args.sentinel_min_run,
            value_tol=args.value_tol,
        )
        for ep_dir in episode_dirs
    ]

    passed = [r.episode for r in results if r.passed]
    passed_indices = sorted(int(name) for name in passed)
    failed = [r for r in results if not r.passed]

    print(f'Episodes scanned: {len(results)}')
    print(f'  pass: {len(passed)}')
    print(f'  fail: {len(failed)}')

    if failed:
        print('\nFailed episodes:')
        for result in failed:
            print(f"  {result.episode}: {', '.join(result.issues)}")

    if args.write_pass_json:
        write_pass_indices_json(args.write_pass_json, passed_indices)
        print(f'\nWrote pass indices ({len(passed_indices)}): {args.write_pass_json}')

    if args.write_fail_list:
        write_name_list(args.write_fail_list, [r.episode for r in failed])
        print(f'Wrote fail list: {args.write_fail_list}')

    if args.output_json:
        payload = {
            'input_dir': str(args.input_dir),
            'annotation_dir': str(args.annotation_dir) if args.annotation_dir else None,
            'rules': {
                'zero_eps': args.zero_eps,
                'min_peak': args.min_peak,
                'sentinel_value': args.sentinel_value,
                'sentinel_min_run': args.sentinel_min_run,
                'value_tol': args.value_tol,
            },
            'summary': {
                'total': len(results),
                'pass': len(passed),
                'fail': len(failed),
            },
            'episodes': [asdict(r) for r in results],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        print(f'Wrote report: {args.output_json}')

    if args.strict and failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
