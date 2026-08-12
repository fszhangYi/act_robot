"""SAM2 smoke test: init with t=0 wrist bbox, propagate through first N frames,
save mask-overlay PNGs to verify tracking.

Output: scripts/_smoke_out/{frame_idx}.png with bbox + mask overlay.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode-dir', default='/home/znyyb/hww/vla/gongjian/cam_100_15/000000')
    parser.add_argument('--num-frames', type=int, default=20)
    parser.add_argument('--ckpt', default='/home/znyyb/hww/vla/act_robot/sam2/checkpoints/sam2.1_hiera_small.pt')
    parser.add_argument('--config', default='configs/sam2.1/sam2.1_hiera_s.yaml')
    parser.add_argument('--out-dir', default='/home/znyyb/hww/vla/act_robot/scripts/_smoke_out')
    args = parser.parse_args()

    ep_dir = Path(args.episode_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load t=0 wrist bbox
    bbox_json = json.loads((ep_dir / 'bbox.json').read_text())
    bbox0 = bbox_json['000000']['rgb_wrist_1'][0]
    box_xyxy = np.array(
        [bbox0['x_min'], bbox0['y_min'], bbox0['x_max'], bbox0['y_max']],
        dtype=np.float32,
    )
    print(f'frame-0 bbox (xyxy pixel): {box_xyxy.tolist()}  label={bbox0["label"]}')

    # Stage frames in a temp dir with names SAM2 wants (00000.jpg ...).
    wrist_paths = sorted(ep_dir.glob('rgb_wrist_1_*.jpg'))[: args.num_frames]
    if not wrist_paths:
        raise FileNotFoundError(f'no rgb_wrist_1_*.jpg in {ep_dir}')
    print(f'wrist frames: {len(wrist_paths)}  resolution: {Image.open(wrist_paths[0]).size}')

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, p in enumerate(wrist_paths):
            os.symlink(p.resolve(), tmp / f'{i:05d}.jpg')

        from sam2.build_sam import build_sam2_video_predictor

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predictor = build_sam2_video_predictor(args.config, args.ckpt, device=device)

        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(tmp))
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                box=box_xyxy,
            )

            results = {}
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                # mask_logits: [num_obj, 1, H, W]
                mask = (mask_logits[0, 0] > 0).cpu().numpy()
                results[frame_idx] = mask

    print(f'propagated {len(results)} frames  mask dtype/shape: bool {results[0].shape}')

    # Render mask overlay
    for frame_idx, mask in results.items():
        img = np.array(Image.open(wrist_paths[frame_idx]).convert('RGB'))
        overlay = img.copy()
        overlay[mask] = (overlay[mask] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
        # draw bbox on frame 0
        if frame_idx == 0:
            x0, y0, x1, y1 = box_xyxy.astype(int)
            overlay[y0:y0+2, x0:x1] = (255, 0, 0)
            overlay[y1-2:y1, x0:x1] = (255, 0, 0)
            overlay[y0:y1, x0:x0+2] = (255, 0, 0)
            overlay[y0:y1, x1-2:x1] = (255, 0, 0)
        Image.fromarray(overlay).save(out_dir / f'{frame_idx:05d}.png')

        if frame_idx in (0, 5, 10, 15, args.num_frames - 1):
            area = int(mask.sum())
            print(f'  frame {frame_idx:3d}  mask_area={area:6d}  '
                  f'fraction={area / mask.size:.4f}')

    print(f'\noverlays written to {out_dir}')


if __name__ == '__main__':
    main()
