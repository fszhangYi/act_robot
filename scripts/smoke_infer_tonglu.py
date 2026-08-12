"""Standalone smoke test for the tonglu0602 baseline ACT inference path.

Mimics what serve.py does end-to-end for one val frame:
  1. Load ACTPolicy + dataset_stats + policy_config (same as ACTInference)
  2. Pull val_indices[0]'s frame 0 from the HDF5 (one frame per camera as uint8 arrays)
  3. JPEG-encode each camera's image (round-tripping through PIL like a real client)
  4. Run the multi-camera _infer_baseline path (decode + resize-noop + stack + ACTPolicy)
  5. Compare predicted chunk[0] against the dataset's ground-truth action[0]

Usage:
  python scripts/smoke_infer_tonglu.py \\
    --ckpt-dir <path/to/policy_best.ckpt's dir> \\
    --data-dir <path/to/converted/data dir>
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from policy import ACTPolicy  # noqa: E402


# Mirrors convert_episodes.py / serve.py
_RESIZE = {
    'wrist':     None,
    'rear_left': (480, 640),
    'chest':     (480, 640),
    'top':       (480, 640),
    'wrist_2':   (480, 640),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--ckpt-name', default='policy_best.ckpt')
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    data_dir = Path(args.data_dir)

    cfg = json.load((ckpt_dir / 'policy_config.json').open())
    if 'chunk_size' in cfg and 'num_queries' not in cfg:
        cfg['num_queries'] = cfg.pop('chunk_size')
    cams = list(cfg.get('camera_names', ['wrist']))
    print(f'config: camera_names={cams}  action_space={cfg.get("action_space")}  '
          f'state_dim={cfg.get("state_dim")}  rx_unwrapped={cfg.get("rx_unwrapped")}  '
          f'use_sam2={cfg.get("use_sam2_features")}')

    stats = pickle.load((ckpt_dir / 'dataset_stats.pkl').open('rb'))
    qm = torch.from_numpy(stats['qpos_mean']).float()
    qs = torch.from_numpy(stats['qpos_std']).float()
    am = torch.from_numpy(stats['action_mean']).float()
    as_ = torch.from_numpy(stats['action_std']).float()

    policy = ACTPolicy(cfg)
    sd = torch.load(ckpt_dir / args.ckpt_name, map_location='cpu')
    policy.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd)
    policy.eval().cuda()
    print(f'OK: ACTPolicy built + {args.ckpt_name} loaded (no state_dict mismatch)')

    info = json.load((data_dir / 'dataset_info.json').open())
    ep = info['val_indices'][0]
    with h5py.File(data_dir / f'episode_{ep}.hdf5', 'r') as f:
        qpos0 = f['/observations/qpos'][0]
        gt_action_chunk = f['/action'][:cfg['num_queries']]
        cam_arrays = {
            cam: f[f'/observations/images/{cam}'][0]
            for cam in cams
        }

    jpegs: list[bytes] = []
    for cam in cams:
        buf = io.BytesIO()
        Image.fromarray(cam_arrays[cam]).save(buf, format='JPEG', quality=95)
        jpegs.append(buf.getvalue())
    print(f'encoded {len(jpegs)} JPEGs (sizes={[len(j) for j in jpegs]} bytes)')

    per_cam = []
    for cam, jpeg in zip(cams, jpegs):
        img = Image.open(io.BytesIO(jpeg)).convert('RGB')
        resize_hw = _RESIZE.get(cam)
        if resize_hw is not None:
            img = img.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        per_cam.append(torch.from_numpy(arr).permute(2, 0, 1) / 255.0)
    img_t = torch.stack(per_cam, dim=0).unsqueeze(0).cuda()
    print(f'img_t shape={tuple(img_t.shape)}  (expected [1, {len(cams)}, 3, 480, 640])')

    q_t = torch.from_numpy(qpos0.astype(np.float32)).cuda()
    qn = ((q_t - qm.cuda()) / qs.cuda()).unsqueeze(0)
    with torch.inference_mode():
        an = policy(qn, img_t)
    pred = (an[0].cpu() * as_ + am).numpy()

    assert pred.shape == (cfg['num_queries'], cfg['state_dim']), pred.shape
    assert not np.isnan(pred).any(), 'NaN in prediction!'

    print()
    print(f'qpos[0]       = {np.array2string(qpos0, precision=4, suppress_small=True)}')
    print(f'pred chunk[0] = {np.array2string(pred[0], precision=4, suppress_small=True)}')
    print(f'gt   action[0]= {np.array2string(gt_action_chunk[0], precision=4, suppress_small=True)}')
    chunk_mae = np.abs(pred[:len(gt_action_chunk)] - gt_action_chunk).mean(axis=0)
    print(f'chunk MAE per-dim = {np.array2string(chunk_mae, precision=4, suppress_small=True)}')
    pose_offset = float(np.linalg.norm(pred[0][:6] - qpos0[:6]))
    print(f'|chunk[0] pose - qpos pose| = {pose_offset:.4f}')
    print()
    print('=== PASS ===  multi-camera baseline inference path runs end-to-end without errors')


if __name__ == '__main__':
    main()
