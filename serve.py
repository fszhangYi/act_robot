#!/usr/bin/env python3
"""ACT inference server.

按 ckpt 同目录 policy_config.json 的 `use_sam2_features` 自动选择线协议
（启动时会打印 `wire_protocol=...`）。以下为与代码逐字节一致的真相。

────────────────────────────────────────────────────────────────────────────
baseline_vla（use_sam2_features=False，对接 tonglu / serve_tmp VLA 布局）
────────────────────────────────────────────────────────────────────────────
  Client → Server（每帧，big-endian；无 refresh / 无 num_cams；新连接 = 新 episode）:
      4B  top_len    + top JPEG
      4B  chest_len  + chest JPEG
      4B  wrist2_len + wrist2 JPEG   → 映射到 camera_names 中的 "wrist_2"
      4B  text_len   + text UTF-8    （接收但不用于推理）
      28B robot_state (7 × float32)

  Server → Client:
      28B next_state  (7 × float32) = compose_pose(state, action)
      4B  term_flag   (uint32, 当前恒为 0)
      4B  reject_flag (uint32, 当前恒为 0)
      4B  text_len    + UTF-8 文本（当前固定 "success"）

  注意：相机 JPEG 在线上固定顺序为 top → chest → wrist2；再按
  policy_config['camera_names'] 重排后喂模型。

────────────────────────────────────────────────────────────────────────────
sam2_legacy（use_sam2_features=True，100_15 旧协议）
────────────────────────────────────────────────────────────────────────────
  Client → Server（每帧，big-endian）:
      4B  wrist_image_length (uint32)
      N B wrist JPEG bytes
      4B  left_image_length  (uint32；接收但不用于推理)
      M B rear-left JPEG bytes
      28B robot_state (7 × float32)
      4B  refresh flag (uint32；1 = 重置 episode / SAM2 状态)
      if refresh == 1:
          16B bbox xyxy (4 × float32，腕部图像素坐标；无 prompt_type 字段)

  Server → Client:
      28B next_state_absolute (7 × float32) = compose_pose(state, action)

  首帧 refresh=1：用 bbox 初始化 SAM2StreamingFeatureExtractor，提取 F_0；
  后续帧逐帧 streaming 产 F_t，喂 ACTSAM2Policy。

Usage:
    python serve.py \\
        --checkpoint /path/policy_best.ckpt \\
        --stats      /path/dataset_stats.pkl \\
        --port 5000 \\
        [--temporal-agg | --always-first]

Inference modes (mutually exclusive):
  default          chunk-replay: query every chunk_size steps, execute the
                   chunk in sequence (has discontinuities at chunk boundaries)
  --temporal-agg   query every step, exponentially-weighted average over
                   overlapping chunks (smoothest, blended)
  --always-first   query every step, always use chunk[0] (greedy 1-step
                   lookahead; eliminates boundary jumps, no blending)
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import socket
import struct
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from policy import ACTPolicy, ACTSAM2Policy, ACTSAM2CVAEPolicy
from sam2_features import SAM2StreamingFeatureExtractor


# ---------------------------------------------------------------------------
# Per-camera resize map. Mirrors convert_episodes.py — incoming JPEGs are
# resized to match what the model was trained on.
# ---------------------------------------------------------------------------
_CAMERA_RESIZE_SERVE: dict[str, tuple[int, int] | None] = {
    'wrist':     None,        # 100_15 rgb_wrist_1 native 640×480 — keep
    'rear_left': (480, 640),  # 100_15 rgb_rear_left native 1280×720 → squish
    'chest':     (480, 640),  # tonglu0602 native 1280×720 → squish
    'top':       (480, 640),  # tonglu0602 native 1280×720 → squish
    'wrist_2':   (480, 640),  # tonglu0602: 640×480 native is identity-resized;
                              # 1280×720 captures get squished. Output (480,640).
}


# ---------------------------------------------------------------------------
# SE(3) helpers  (same convention as convert_episodes.py)
# ---------------------------------------------------------------------------

def euler_xyz_to_matrix(euler_xyz: np.ndarray) -> np.ndarray:
    rx, ry, rz = float(euler_xyz[0]), float(euler_xyz[1]), float(euler_xyz[2])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def pose6_to_matrix(pose6: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pose6[:3]
    T[:3, :3] = euler_xyz_to_matrix(pose6[3:6])
    return T


def matrix_to_pose6(T: np.ndarray) -> np.ndarray:
    pose = np.zeros(6, dtype=np.float64)
    pose[:3] = T[:3, 3]
    pose[3:6] = matrix_to_euler_xyz(T[:3, :3])
    return pose


def apply_rx_unwrap_to_input(robot_state: np.ndarray) -> np.ndarray:
    """If the policy was trained on the rx-unwrapped cart_abs dataset, every
    rx value in training was shifted from [-π, 0) to [+π, +2π). At inference
    the client still sends rx ∈ [-π, π], so we apply the same shift before
    feeding qpos to the model.
    """
    out = robot_state.copy()
    if out[3] < 0:
        out[3] += 2 * np.pi
    return out


def apply_rx_unwrap_to_output(next_state: np.ndarray) -> np.ndarray:
    """Inverse of apply_rx_unwrap_to_input: if model output rx > π, wrap back
    to [-π, π] so the client sees a standard-range value.
    """
    out = next_state.copy()
    if out[3] > np.pi:
        out[3] -= 2 * np.pi
    return out


def compose_pose(current_pose: np.ndarray, action: np.ndarray,
                 action_space: str = 'joint') -> np.ndarray:
    """Apply action to current state and return the next commanded state.

    joint / cartesian_abs mode:
        action is already the absolute target — return directly.

    cartesian mode (legacy):
        action is a SE(3) relative delta; compose via T_target = T_current @ T_action.

    Gripper is clipped to [0, 1.13].
    """
    action = np.asarray(action, dtype=np.float64)

    if action_space in ('joint', 'cartesian_abs'):
        result = action.copy()
    else:
        current_pose = np.asarray(current_pose, dtype=np.float64)
        T_cur = pose6_to_matrix(current_pose)
        T_act = pose6_to_matrix(action)
        T_nxt = T_cur @ T_act
        result = np.zeros(7, dtype=np.float64)
        result[:6] = matrix_to_pose6(T_nxt)
        result[6] = action[6]

    result[6] = np.clip(result[6], 0.0, 1.13)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Client episode state
# ---------------------------------------------------------------------------

class ClientState:
    """Per-connection state. Three inference modes (chunk_replay / temporal_agg /
    always_first) — see module docstring."""

    def __init__(self, chunk_size: int, mode: str = 'chunk_replay', k: float = 0.01):
        assert mode in ('chunk_replay', 'temporal_agg', 'always_first')
        self.chunk_size = chunk_size
        self.mode = mode
        self.k = k
        self.t = 0

        if mode == 'temporal_agg':
            self.history: list[tuple[int, np.ndarray]] = []
        else:
            self.current_chunk: np.ndarray | None = None
            self.chunk_offset: int = chunk_size  # force re-query on first call

    def reset(self) -> None:
        self.t = 0
        if self.mode == 'temporal_agg':
            self.history = []
        else:
            self.current_chunk = None
            self.chunk_offset = self.chunk_size

    def needs_query(self) -> bool:
        if self.mode == 'temporal_agg' or self.mode == 'always_first':
            return True
        return self.chunk_offset >= self.chunk_size / 2 #add by huweiwen

    def add_chunk(self, chunk: np.ndarray) -> None:
        if self.mode == 'temporal_agg':
            self.history.append((self.t, chunk.copy()))
            self.history = [(t0, c) for t0, c in self.history
                            if t0 + self.chunk_size > self.t]
        else:
            self.current_chunk = chunk.copy()
            self.chunk_offset = 0

    def get_action(self) -> np.ndarray:
        if self.mode == 'temporal_agg':
            relevant = [
                (t0, chunk[self.t - t0])
                for t0, chunk in self.history
                if 0 <= self.t - t0 < self.chunk_size
            ]
            if not relevant:
                raise RuntimeError('No predictions cover the current timestep.')
            relevant.sort(key=lambda x: x[0])
            n = len(relevant)
            weights = np.exp(-self.k * np.arange(n - 1, -1, -1))
            weights /= weights.sum()
            action = sum(w * a for w, (_, a) in zip(weights, relevant))
            return action
        if self.current_chunk is None:
            raise RuntimeError('No chunk available.')
        if self.mode == 'always_first':
            return self.current_chunk[0]
        action = self.current_chunk[self.chunk_offset]
        self.chunk_offset += 1
        return action

    def step(self) -> None:
        self.t += 1


# ---------------------------------------------------------------------------
# ACT inference wrapper
# ---------------------------------------------------------------------------

class ACTInference:
    """Inference wrapper that auto-selects baseline ACT (multi-camera) or SAM2Grasp
    ACT based on the checkpoint's policy_config.json `use_sam2_features` flag."""

    def __init__(self, checkpoint_path: str, stats_path: str, device: str = 'cuda'):
        self.device = torch.device(device)

        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self.qpos_mean = torch.from_numpy(stats['qpos_mean']).float().to(self.device)
        self.qpos_std  = torch.from_numpy(stats['qpos_std']).float().to(self.device)
        self.action_mean = torch.from_numpy(stats['action_mean']).float().to(self.device)
        self.action_std  = torch.from_numpy(stats['action_std']).float().to(self.device)
        if 'delta_mean' in stats:
            self.delta_mean = torch.from_numpy(stats['delta_mean']).float().to(self.device)
            self.delta_std  = torch.from_numpy(stats['delta_std']).float().to(self.device)
        else:
            self.delta_mean = None
            self.delta_std = None

        ckpt_dir = Path(checkpoint_path).parent
        config_path = ckpt_dir / 'policy_config.json'
        if config_path.exists():
            with config_path.open() as f:
                policy_config = json.load(f)
            if 'chunk_size' in policy_config and 'num_queries' not in policy_config:
                policy_config['num_queries'] = policy_config.pop('chunk_size')
        else:
            state_dim = stats.get('state_dim', 7)
            policy_config = {
                'lr': 1e-5, 'lr_backbone': 1e-5,
                'num_queries': 10,
                'kl_weight': 10.0,
                'hidden_dim': 512,
                'dim_feedforward': 3200,
                'enc_layers': 4, 'dec_layers': 7, 'nheads': 8,
                'backbone': 'resnet18',
                'camera_names': ['wrist'],
                'state_dim': state_dim,
            }
            print('Warning: policy_config.json not found; using defaults.')

        self.use_sam2 = bool(policy_config.get('use_sam2_features', False))
        self.use_cvae = bool(policy_config.get('use_cvae', False))
        self.chunk_size = policy_config.get('num_queries', 10)
        self.action_space = policy_config.get('action_space', 'joint')
        self.action_repr = policy_config.get('action_repr', 'absolute')
        self.rx_unwrapped = bool(policy_config.get('rx_unwrapped', False))
        self.camera_names: list[str] = list(policy_config.get('camera_names', ['wrist']))
        self.expected_num_cams = len(self.camera_names)
        # Pre-compute per-camera resize tuples (matches convert_episodes.py)
        self.cam_resize: list[tuple[int, int] | None] = [
            _CAMERA_RESIZE_SERVE.get(c, None) for c in self.camera_names
        ]
        if self.action_repr == 'delta' and self.delta_mean is None:
            raise RuntimeError('policy_config says action_repr=delta but dataset_stats.pkl is '
                               'missing delta_mean/std.')

        if self.use_sam2:
            if self.use_cvae:
                self.policy = ACTSAM2CVAEPolicy(policy_config)
            else:
                self.policy = ACTSAM2Policy(policy_config)
            self.sam2 = SAM2StreamingFeatureExtractor(device=device)
        else:
            self.policy = ACTPolicy(policy_config)
            self.sam2 = None

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model' in ckpt:
            self.policy.model.load_state_dict(ckpt['model'])
        else:
            self.policy.load_state_dict(ckpt)

        self.policy.eval()
        self.policy.to(self.device)

        print(f'Loaded checkpoint: {checkpoint_path}')
        print(f'  mode={"SAM2Grasp" if self.use_sam2 else "baseline ACT"}  '
              f'chunk_size={self.chunk_size}  state_dim={policy_config.get("state_dim")}  '
              f'action_space={self.action_space}  rx_unwrapped={self.rx_unwrapped}  '
              f'cams={self.camera_names}  device={device}')

    @torch.inference_mode()
    def _infer_baseline(self, jpegs: list[bytes], qpos: np.ndarray) -> np.ndarray:
        """Multi-camera baseline ACT inference. jpegs[i] is the JPEG bytes for
        camera_names[i]; we decode + per-cam resize + stack to [1, N, 3, H, W]
        and let ACTPolicy do the per-camera ResNet18 forward + transformer.
        ACTPolicy applies ImageNet normalize internally — we only /255 here."""
        assert len(jpegs) == self.expected_num_cams, (
            f'expected {self.expected_num_cams} cams ({self.camera_names}), got {len(jpegs)}')
        per_cam_tensors: list[torch.Tensor] = []
        for i, (jpeg, resize_hw) in enumerate(zip(jpegs, self.cam_resize)):
            img = Image.open(io.BytesIO(jpeg)).convert('RGB')
            # if self.camera_names[i] == 'wrist_2':
            #     img = img.rotate(180)
            if resize_hw is not None:
                img = img.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            per_cam_tensors.append(torch.from_numpy(arr).permute(2, 0, 1) / 255.0)
        img_t = torch.stack(per_cam_tensors, dim=0).unsqueeze(0).to(self.device)   # [1, N, 3, H, W]
        qpos_t = torch.from_numpy(qpos.astype(np.float32)).to(self.device)
        qpos_norm = ((qpos_t - self.qpos_mean) / self.qpos_std).unsqueeze(0)
        action_norm = self.policy(qpos_norm, img_t)
        return (action_norm[0] * self.action_std + self.action_mean).cpu().numpy()

    @torch.inference_mode()
    def _infer_sam2(self, sam2_feat: torch.Tensor, qpos: np.ndarray) -> np.ndarray:
        qpos_t = torch.from_numpy(qpos.astype(np.float32)).to(self.device)
        qpos_norm = ((qpos_t - self.qpos_mean) / self.qpos_std).unsqueeze(0)
        feat = sam2_feat.float()
        pred_norm = self.policy(qpos_norm, feat)
        if self.action_repr == 'delta':
            delta = pred_norm[0] * self.delta_std + self.delta_mean
            action = delta + qpos_t
        else:
            action = pred_norm[0] * self.action_std + self.action_mean
        return action.cpu().numpy()

    def init_sam2_episode(self, image_bytes: bytes, bbox_xyxy: np.ndarray) -> torch.Tensor:
        assert self.sam2 is not None
        return self.sam2.init_first_frame(image_bytes, bbox_xyxy)

    def step_sam2_frame(self, image_bytes: bytes) -> torch.Tensor:
        assert self.sam2 is not None
        return self.sam2.step(image_bytes)

    def infer_chunk(self, jpegs: list[bytes] | bytes, qpos: np.ndarray,
                    sam2_feat: torch.Tensor | None = None) -> np.ndarray:
        """Single dispatch. SAM2 mode: caller supplies sam2_feat (computed via
        init_sam2_episode or step_sam2_frame); jpegs is ignored. Baseline mode:
        jpegs is a list of bytes (one per camera in camera_names order)."""
        if self.use_sam2:
            if sam2_feat is None:
                raise RuntimeError('SAM2 mode requires sam2_feat from init/step')
            return self._infer_sam2(sam2_feat, qpos)
        if isinstance(jpegs, (bytes, bytearray)):
            jpegs = [jpegs]
        return self._infer_baseline(jpegs, qpos)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = b''
    while len(buf) < n:
        pkt = conn.recv(n - len(buf))
        if not pkt:
            return None
        buf += pkt
    return buf


def _handle_sam2_client(conn, addr, inferencer, mode, log_dir):
    """Legacy SAM2 protocol: per-step [wrist_jpeg][rear_left_jpeg][state][refresh]
    + optional [bbox] when refresh=1. Unchanged from the original 100_15
    deployment to preserve client compatibility."""
    client_state = ClientState(inferencer.chunk_size, mode=mode)
    step = 0
    try:
        while True:
            step += 1

            raw = recv_exact(conn, 4)
            if raw is None: break
            img_len = struct.unpack('>I', raw)[0]
            img_data = recv_exact(conn, img_len)
            if img_data is None: break

            raw = recv_exact(conn, 4)
            if raw is None: break
            left_len = struct.unpack('>I', raw)[0]
            left_data = recv_exact(conn, left_len)
            if left_data is None: break

            raw = recv_exact(conn, 7 * 4)
            if raw is None: break
            robot_state = np.array(struct.unpack('>7f', raw), dtype=np.float32)
            if inferencer.rx_unwrapped:
                robot_state = apply_rx_unwrap_to_input(robot_state)

            raw = recv_exact(conn, 4)
            if raw is None: break
            refresh = struct.unpack('>I', raw)[0]

            sam2_feat = None
            if refresh:
                raw = recv_exact(conn, 16)
                if raw is None: break
                bbox = np.array(struct.unpack('>4f', raw), dtype=np.float32)
                client_state.reset()
                sam2_feat = inferencer.init_sam2_episode(img_data, bbox)
                print(f'[{addr}] episode reset  bbox={bbox.tolist()}')
            else:
                sam2_feat = inferencer.step_sam2_frame(img_data)

            try:
                if client_state.needs_query():
                    chunk = inferencer.infer_chunk(img_data, robot_state, sam2_feat=sam2_feat)
                    client_state.add_chunk(chunk)
                action = client_state.get_action()
                next_state = compose_pose(robot_state, action, inferencer.action_space)
                if inferencer.rx_unwrapped:
                    next_state = apply_rx_unwrap_to_output(next_state)
                client_state.step()
            except Exception as exc:
                print(f'[{addr}] inference error: {exc}')
                import traceback; traceback.print_exc()
                break

            step_dir = log_dir / f'step_{step:04d}'
            step_dir.mkdir(exist_ok=True)
            (step_dir / 'wrist.jpg').write_bytes(img_data)
            (step_dir / 'robot_state.json').write_text(json.dumps(robot_state.tolist(), indent=2))
            (step_dir / 'next_state.json').write_text(json.dumps(next_state.tolist(), indent=2))
            (step_dir / 'action.json').write_text(json.dumps(action.tolist(), indent=2))

            conn.sendall(struct.pack('>7f', *next_state))
    finally:
        pass
    return step


def _handle_baseline_client(conn, addr, inferencer, mode, log_dir):
    """Protocol matches serve_tmp.py (VLA server input/output layout).

    Per-frame input (big-endian):
        4B  regedit_len  (uint32; 0 = no register image)
            N B regedit JPEG (if regedit_len > 0, ignored)
        4B  top_len     + top JPEG
        4B  chest_len   + chest JPEG
        4B  wrist1_len  + wrist1 JPEG  (received but unused)
        4B  wrist2_len  + wrist2 JPEG  -> mapped to camera "wrist_2"
        4B  text_len    + text UTF-8     (received but unused)
        28B robot_state (7 x float32)

    Per-frame output (big-endian):
        28B next_state   (7 x float32)
        4B  term_flag    (uint32)
        4B  reject_flag  (uint32)
        4B  text_len     + processed_text UTF-8
    """
    client_state = ClientState(inferencer.chunk_size, mode=mode)
    step = 0
    try:
        while True:
            step += 1

            # --- regedit image (ignored) ---
            # raw = recv_exact(conn, 4)
            # if raw is None:
            #     break
            # regedit_len = struct.unpack('>I', raw)[0]
            # if regedit_len > 0:
            #     regedit_data = recv_exact(conn, regedit_len)
            #     if regedit_data is None:
            #         break

            # --- 4 camera JPEGs: top, chest, wrist1, wrist2 ---
            raw_jpegs: dict[str, bytes] = {}
            recv_ok = True
            for cam_key in ['top', 'chest', 'wrist2']:
                raw = recv_exact(conn, 4)
                if raw is None:
                    recv_ok = False
                    break
                jpeg_len = struct.unpack('>I', raw)[0]
                jpeg_bytes = recv_exact(conn, jpeg_len)
                if jpeg_bytes is None:
                    recv_ok = False
                    break
                raw_jpegs[cam_key] = jpeg_bytes
            if not recv_ok:
                break

            # --- text instruction (ignored) ---
            raw = recv_exact(conn, 4)
            if raw is None:
                break
            text_len = struct.unpack('>I', raw)[0]
            if text_len > 0:
                text_data = recv_exact(conn, text_len)
                if text_data is None:
                    break

            # --- robot state ---
            raw = recv_exact(conn, 7 * 4)
            if raw is None:
                break
            robot_state = np.array(struct.unpack('>7f', raw), dtype=np.float32)
            if inferencer.rx_unwrapped:
                robot_state = apply_rx_unwrap_to_input(robot_state)

            # reorder to match policy camera_names (e.g. ["chest","top","wrist_2"])
            cam_to_raw = {
                'chest': raw_jpegs['chest'],
                'top': raw_jpegs['top'],
                'wrist_2': raw_jpegs['wrist2'],
            }
            try:
                jpegs = [cam_to_raw[cam] for cam in inferencer.camera_names]
            except KeyError as exc:
                print(f'[{addr}] camera mapping failed: expected {inferencer.camera_names}, missing {exc}')
                break

            # new connection = new episode (no explicit refresh flag)
            if step == 1:
                client_state.reset()

            try:
                if client_state.needs_query():
                    chunk = inferencer.infer_chunk(jpegs, robot_state)
                    client_state.add_chunk(chunk)
                action = client_state.get_action()
                next_state = compose_pose(robot_state, action, inferencer.action_space)
                if inferencer.rx_unwrapped:
                    next_state = apply_rx_unwrap_to_output(next_state)
                client_state.step()
            except Exception as exc:
                print(f'[{addr}] inference error: {exc}')
                import traceback
                traceback.print_exc()
                break

            step_dir = log_dir / f'step_{step:04d}'
            step_dir.mkdir(exist_ok=True)
            for cam, jpeg in zip(inferencer.camera_names, jpegs):
                (step_dir / f'{cam}.jpg').write_bytes(jpeg)
            (step_dir / 'robot_state.json').write_text(json.dumps(robot_state.tolist(), indent=2))
            (step_dir / 'next_state.json').write_text(json.dumps(next_state.tolist(), indent=2))
            (step_dir / 'action.json').write_text(json.dumps(action.tolist(), indent=2))

            # output matching serve_tmp.py
            processed_text = 'success'
            processed_text_bytes = processed_text.encode('utf-8')
            term_flag = 0
            reject_flag = 0
            conn.sendall(struct.pack('>7f', *next_state))
            conn.sendall(struct.pack('>I', term_flag))
            conn.sendall(struct.pack('>I', reject_flag))
            conn.sendall(struct.pack('>I', len(processed_text_bytes)))
            conn.sendall(processed_text_bytes)
    finally:
        pass
    return step


def handle_client(conn: socket.socket, addr: tuple,
                  inferencer: ACTInference, mode: str) -> None:
    print(f'[{addr}] connected  '
          f'({"SAM2 legacy" if inferencer.use_sam2 else f"baseline multi-cam ({inferencer.camera_names})"})')

    now = datetime.now()
    date_str = now.strftime('%Y%m%d')
    time_str = now.strftime('%H%M%S')
    log_dir = Path('logs') / date_str / time_str
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        if inferencer.use_sam2:
            step = _handle_sam2_client(conn, addr, inferencer, mode, log_dir)
        else:
            step = _handle_baseline_client(conn, addr, inferencer, mode, log_dir)
    except Exception as exc:
        print(f'[{addr}] connection error: {exc}')
        step = -1
    finally:
        conn.close()
        print(f'[{addr}] disconnected  (steps={step})')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='Path to policy_best.ckpt')
    parser.add_argument('--stats', required=True, help='Path to dataset_stats.pkl')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--device', default='cuda')
    inf_mode_group = parser.add_mutually_exclusive_group()
    inf_mode_group.add_argument('--temporal-agg', action='store_true')
    inf_mode_group.add_argument('--always-first', action='store_true')
    args = parser.parse_args()

    # if args.temporal_agg:
    #     mode = 'temporal_agg'
    # elif args.always_first:
    #     mode = 'always_first'
    # else:
    #     mode = 'chunk_replay'

    mode = 'chunk_replay'

    inferencer = ACTInference(args.checkpoint, args.stats, args.device)
    mode_label = {'temporal_agg': 'temporal-agg',
                  'always_first': 'always-first (per-step, chunk[0] only)',
                  'chunk_replay': f'chunk-replay (chunk_size={inferencer.chunk_size})'}[mode]
    print(f'Inference mode: {mode_label}')

    # 启动 banner：以代码为唯一真相，避免客户端按过期 README 对不齐字节
    if inferencer.use_sam2:
        wire = 'sam2_legacy'
        layout = ('client: wrist_jpeg + rear_left_jpeg + 28B state + 4B refresh'
                  ' [+16B bbox if refresh=1]  →  server: 28B next_state')
    else:
        wire = 'baseline_vla'
        layout = ('client: top_jpeg + chest_jpeg + wrist2_jpeg + text + 28B state'
                  '  →  server: 28B next_state + term + reject + text'
                  f'  | camera_names={inferencer.camera_names}')
    print(f'wire_protocol={wire}')
    print(f'wire_layout: {layout}')

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(5)
    print(f'Listening on {args.host}:{args.port}')

    try:
        while True:
            conn, addr = server_sock.accept()
            t = threading.Thread(target=handle_client,
                                 args=(conn, addr, inferencer, mode),
                                 daemon=True)
            t.start()
    except KeyboardInterrupt:
        print('\nShutting down.')
    finally:
        server_sock.close()


if __name__ == '__main__':
    main()
