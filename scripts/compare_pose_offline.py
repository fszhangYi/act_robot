#!/usr/bin/env python3
"""Offline compare: next pose from (current + GT) vs (current + model).

Produces JSON + a self-contained HTML dashboard for browser viewing.
Supports:
  - simulate: synthetic 6-DoF SO-100-like trajectories (default, fast)
  - dataset: real parquet GT from mtitg/so100_lr + simulated model bias
  - model: real ACTPolicy inference on CPU/GPU (slow on CPU; few frames)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def compose_next_pose(current: np.ndarray, action: np.ndarray, mode: str) -> np.ndarray:
    """Absolute actions: next = action; relative: next = current + action."""
    current = np.asarray(current, dtype=np.float64)[:6]
    action = np.asarray(action, dtype=np.float64)[:6]
    if mode == "relative":
        return current + action
    return action.copy()


def simulate_episode(n_frames: int = 80, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * math.pi, n_frames)
    # Smooth absolute joint targets (deg-like), SO-100-ish ranges
    gt_actions = np.stack(
        [
            10 * np.sin(t),
            -90 + 15 * np.sin(t * 0.7 + 0.3),
            80 + 20 * np.cos(t * 0.9),
            50 + 18 * np.sin(t * 1.1 + 1.0),
            -10 * np.sin(t * 0.5),
            np.clip(0.5 + 0.4 * np.sin(t * 2.0), 0.0, 1.0),
        ],
        axis=1,
    )
    # Current pose lags GT target slightly (tracking)
    states = np.zeros((n_frames, 8), dtype=np.float64)
    states[0, :6] = gt_actions[0]
    states[0, 6:] = [0.0, 0.0]
    for i in range(1, n_frames):
        states[i, :6] = 0.85 * states[i - 1, :6] + 0.15 * gt_actions[i - 1]
        states[i, 6:] = [0.0, 0.0]

    # Predict ≈ GT: small smooth bias + heavily EMA-smoothed residual (no IID jitter)
    # Target MAE roughly sub-degree on joints, tiny on gripper.
    phase = rng.uniform(0.0, 2.0 * math.pi, size=6)
    amp = np.array([0.55, 0.45, 0.70, 0.50, 0.35, 0.015], dtype=np.float64)
    smooth_bias = amp * np.sin(0.35 * t[:, None] + phase[None, :])
    residual = np.zeros_like(gt_actions)
    innov = rng.normal(0.0, 0.12, size=gt_actions.shape)
    innov[:, 5] *= 0.05
    residual[0] = innov[0]
    for i in range(1, n_frames):
        residual[i] = 0.94 * residual[i - 1] + 0.06 * innov[i]
    pred_actions = gt_actions + smooth_bias + residual
    pred_actions[:, 5] = np.clip(pred_actions[:, 5], 0.0, 1.0)

    return {
        "source": "simulate",
        "fps": 30,
        "states": states,
        "gt_actions": gt_actions,
        "pred_actions": pred_actions,
    }


def load_dataset_episode(data_root: Path, episode: int) -> dict:
    path = data_root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pq.read_table(path).to_pandas()
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    gt_actions = np.stack(df["action"].to_numpy()).astype(np.float64)
    # Simulated model prediction anchored on real GT
    rng = np.random.default_rng(episode + 7)
    n = len(gt_actions)
    noise = rng.normal(0, 1.2, size=gt_actions.shape)
    bias = np.array([0.8, -0.5, 1.0, -0.7, 0.4, 0.03])
    pred = gt_actions + bias + noise
    pred[:, 5] = np.clip(pred[:, 5], 0.0, 100.0)  # gripper may be percent in some logs
    return {
        "source": f"dataset:episode_{episode:06d}",
        "fps": 30,
        "states": states,
        "gt_actions": gt_actions,
        "pred_actions": pred,
    }


def run_model_on_frames(
    model_path: Path,
    states: np.ndarray,
    max_frames: int,
    device: str,
) -> np.ndarray:
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(str(model_path))
    policy.to(device)
    policy.eval()
    policy.reset()

    n = min(max_frames, len(states))
    preds = []
    for i in range(n):
        batch = {
            "observation.state": torch.as_tensor(states[i : i + 1], dtype=torch.float32, device=device),
            # placeholder images (no torchcodec / v2.1 video path); still exercises the policy head
            "observation.images.wrist": torch.rand(1, 3, 480, 640, device=device),
            "observation.images.above": torch.rand(1, 3, 480, 640, device=device),
        }
        with torch.inference_mode():
            a = policy.select_action(batch)
        preds.append(a[0].detach().cpu().numpy())
        if (i + 1) % 5 == 0:
            print(f"  model frame {i + 1}/{n}", flush=True)
    return np.stack(preds, axis=0)


def build_compare_payload(ep: dict, action_mode: str, max_frames: int | None = None) -> dict:
    states = ep["states"]
    gt_actions = ep["gt_actions"]
    pred_actions = ep["pred_actions"]
    n = len(states)
    if max_frames is not None:
        n = min(n, max_frames)
        states = states[:n]
        gt_actions = gt_actions[:n]
        pred_actions = pred_actions[:n]

    next_gt = np.stack([compose_next_pose(states[i, :6], gt_actions[i], action_mode) for i in range(n)])
    next_pred = np.stack([compose_next_pose(states[i, :6], pred_actions[i], action_mode) for i in range(n)])
    err = next_pred - next_gt
    err_l2 = np.linalg.norm(err, axis=1)
    err_abs = np.abs(err)

    # Actual next observed state when available (dataset)
    next_obs = None
    if n >= 2:
        next_obs = np.vstack([states[1:, :6], states[-1:, :6]])

    summary = {
        "n_frames": int(n),
        "action_mode": action_mode,
        "mean_l2": float(err_l2.mean()),
        "max_l2": float(err_l2.max()),
        "per_joint_mae": {name: float(err_abs[:, i].mean()) for i, name in enumerate(JOINT_NAMES)},
        "source": ep["source"],
        "fps": ep.get("fps", 30),
    }

    frames = []
    for i in range(n):
        item = {
            "t": i,
            "timestamp": float(i / summary["fps"]),
            "current": states[i, :6].tolist(),
            "action_gt": gt_actions[i].tolist(),
            "action_pred": pred_actions[i].tolist(),
            "next_gt": next_gt[i].tolist(),
            "next_pred": next_pred[i].tolist(),
            "err": err[i].tolist(),
            "err_l2": float(err_l2[i]),
        }
        if next_obs is not None:
            item["next_obs"] = next_obs[i].tolist()
        frames.append(item)

    return {
        "meta": {
            "title": "SO-100 六轴：GT 下一时刻位姿 vs 模型下一时刻位姿",
            "joint_names": JOINT_NAMES,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **summary,
        },
        "series": {
            "t": list(range(n)),
            "err_l2": err_l2.tolist(),
            "current": states[:, :6].tolist(),
            "next_gt": next_gt.tolist(),
            "next_pred": next_pred.tolist(),
            "per_joint_err": err.tolist(),
        },
        "frames": frames,
    }


_WEB_TEMPLATE = Path("/root/autodl-tmp/embody_model_eval/index.html")


def write_html(payload: dict, out_html: Path) -> None:
    """Copy Three.js viewer template next to compare_result.json."""
    if not _WEB_TEMPLATE.is_file():
        raise FileNotFoundError(f"missing web template: {_WEB_TEMPLATE}")
    out_html.write_text(_WEB_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    # keep a hub page pointing at index
    hub = out_html.parent / "hub.html"
    hub.write_text(
        """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>Embody Model Eval</title>
<meta http-equiv="refresh" content="0; url=./index.html"/>
</head><body>
<p><a href="./index.html">打开 Embody 模型评测页</a></p>
</body></html>
""",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["simulate", "dataset", "model"], default="simulate")
    ap.add_argument("--dataset-root", default="/root/autodl-tmp/lerobot_assets/datasets/so100_lr")
    ap.add_argument("--model-path", default="/root/autodl-tmp/lerobot_assets/models/so100_lr")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=80)
    ap.add_argument("--action-mode", choices=["absolute", "relative"], default="absolute")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--out-dir",
        default="/root/autodl-tmp/embody_model_eval",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "simulate":
        ep = simulate_episode(n_frames=args.n_frames)
    elif args.mode == "dataset":
        ep = load_dataset_episode(Path(args.dataset_root), args.episode)
    else:
        # real GT from dataset + real ACT preds on first N frames
        base = load_dataset_episode(Path(args.dataset_root), args.episode)
        print(f"Running ACT on {args.n_frames} frames ({args.device})...")
        preds = run_model_on_frames(
            Path(args.model_path), base["states"], args.n_frames, args.device
        )
        n = len(preds)
        ep = {
            "source": f"model+dataset:episode_{args.episode:06d}",
            "fps": 30,
            "states": base["states"][:n],
            "gt_actions": base["gt_actions"][:n],
            "pred_actions": preds,
        }

    payload = build_compare_payload(ep, args.action_mode, max_frames=args.n_frames)
    json_path = out_dir / "compare_result.json"
    html_path = out_dir / "index.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(payload, html_path)
    print("Wrote", json_path)
    print("Wrote", html_path)
    print(
        "summary:",
        json.dumps(payload["meta"], ensure_ascii=False),
    )


if __name__ == "__main__":
    main()
