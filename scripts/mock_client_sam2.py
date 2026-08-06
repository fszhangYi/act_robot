"""End-to-end mock client for the SAM2Grasp serve.py protocol.

Spins up serve.py in a subprocess, connects via TCP, plays back N frames
from a real episode with the t=0 bbox as prompt, and prints each round-trip's
next_state.

Verifies (smoke-level):
  1. Protocol parses on both sides (refresh=1 with bbox + subsequent frames).
  2. SAM2 streaming F_t per frame.
  3. ACTSAM2Policy produces actions.
  4. compose_pose returns float32 next_state with sane shape.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def send_step(sock, jpeg_bytes: bytes, robot_state: np.ndarray, refresh: int,
              bbox_xyxy: np.ndarray | None = None) -> np.ndarray:
    """Send one step in the SAM2Grasp protocol; return next_state."""
    sock.sendall(struct.pack('>I', len(jpeg_bytes)))
    sock.sendall(jpeg_bytes)
    # rear-left placeholder (zero-length)
    sock.sendall(struct.pack('>I', 0))
    # robot state
    sock.sendall(struct.pack('>7f', *robot_state.astype(np.float32)))
    # refresh flag
    sock.sendall(struct.pack('>I', refresh))
    if refresh:
        assert bbox_xyxy is not None and bbox_xyxy.shape == (4,), \
            'must provide bbox xyxy on refresh=1'
        sock.sendall(struct.pack('>I', 0))                              # prompt_type=0 (bbox)
        sock.sendall(struct.pack('>4f', *bbox_xyxy.astype(np.float32)))
    # receive next_state
    raw = b''
    while len(raw) < 28:
        chunk = sock.recv(28 - len(raw))
        if not chunk:
            raise RuntimeError('server closed connection')
        raw += chunk
    return np.array(struct.unpack('>7f', raw), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/media/znyyb/EE223AE5223AB287/act_dataset/cam100_15_wrist_sam2small_joint_ckpt_smoke/policy_best.ckpt')
    parser.add_argument('--stats', default='/media/znyyb/EE223AE5223AB287/act_dataset/cam100_15_wrist_sam2small_joint_ckpt_smoke/dataset_stats.pkl')
    parser.add_argument('--episode-dir', default='/home/znyyb/hww/vla/gongjian/cam_100_15/000000')
    parser.add_argument('--num-steps', type=int, default=8)
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--inference-mode', choices=['chunk_replay', 'temporal_agg', 'always_first'],
                        default='chunk_replay',
                        help='Pass through to serve.py (default chunk_replay).')
    args = parser.parse_args()

    ep = Path(args.episode_dir)
    frames = sorted(ep.glob('rgb_wrist_1_*.jpg'))[: args.num_steps]
    bbox = json.loads((ep / 'bbox.json').read_text())['000000']['rgb_wrist_1'][0]
    bbox_xyxy = np.array(
        [bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']],
        dtype=np.float32,
    )

    # Get robot states for the same indices from steps.json
    steps = json.loads((ep / 'steps.json').read_text())
    joint = np.array(steps['observations']['joint_position'], dtype=np.float32)
    grip = np.array(steps['observations']['gripper_position'], dtype=np.float32).squeeze()
    qposes = [np.append(joint[i], grip[i]) for i in range(len(frames))]

    # Spin up server
    print(f'[client] starting server: checkpoint={args.checkpoint}')
    serve_args = [
        '/home/znyyb/miniconda3/envs/anygrasp/bin/python',
        '/home/znyyb/hww/vla/act_robot/serve.py',
        '--checkpoint', args.checkpoint,
        '--stats', args.stats,
        '--host', '127.0.0.1',
        '--port', str(args.port),
    ]
    if args.inference_mode == 'temporal_agg':
        serve_args.append('--temporal-agg')
    elif args.inference_mode == 'always_first':
        serve_args.append('--always-first')
    server = subprocess.Popen(serve_args, cwd='/tmp',
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    try:
        # Wait until server is listening
        time.sleep(0.5)
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((args.host, args.port))
                break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)
        else:
            raise RuntimeError('server did not start')
        print('[client] connected')

        for i, p in enumerate(frames):
            t0 = time.time()
            jpeg = p.read_bytes()
            next_state = send_step(
                sock, jpeg, qposes[i],
                refresh=1 if i == 0 else 0,
                bbox_xyxy=bbox_xyxy if i == 0 else None,
            )
            dt = (time.time() - t0) * 1000
            print(f'  t={i:2d}  qpos={qposes[i][:3].round(3).tolist()}...  '
                  f'next={next_state[:3].round(3).tolist()}...gripper={next_state[6]:.3f}  '
                  f'rtt={dt:.0f}ms')
        sock.close()
        print('[client] done')
    finally:
        # Drain server stdout (helpful on crash)
        if server.poll() is None:
            server.terminate()
            try:
                out, _ = server.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                out, _ = server.communicate()
        else:
            out, _ = server.communicate()
        print('=== server stdout ===')
        sys.stdout.write(out.decode(errors='replace'))


if __name__ == '__main__':
    main()
