"""Probe the exact shape of pix_feat_with_mem (F_t) for one wrist frame."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

EP = Path('/home/znyyb/hww/vla/gongjian/cam_100_15/000000')
CKPT = '/home/znyyb/hww/vla/act_robot/sam2/checkpoints/sam2.1_hiera_small.pt'
CFG = 'configs/sam2.1/sam2.1_hiera_s.yaml'


def main() -> None:
    import json
    bbox = json.loads((EP / 'bbox.json').read_text())['000000']['rgb_wrist_1'][0]
    box = [bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']]

    wrist = sorted(EP.glob('rgb_wrist_1_*.jpg'))[:5]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, p in enumerate(wrist):
            os.symlink(p.resolve(), tmp / f'{i:05d}.jpg')

        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(CFG, CKPT, device='cuda')

        # Patch _track_step to capture pix_feat
        original = predictor._track_step
        captures = []
        def patched(*args, **kwargs):
            r = original(*args, **kwargs)
            current_out, sam_outputs, high_res_features, pix_feat = r
            captures.append({
                'pix_feat_shape': tuple(pix_feat.shape),
                'pix_feat_dtype': str(pix_feat.dtype),
                'high_res_count': len(high_res_features) if high_res_features else 0,
                'high_res_shapes': [tuple(t.shape) for t in (high_res_features or [])],
            })
            return r
        predictor._track_step = patched

        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(tmp))
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=1, box=box,
            )
            for frame_idx, _, _ in predictor.propagate_in_video(state):
                if frame_idx >= 3:
                    break

    for i, cap in enumerate(captures):
        print(f'capture {i}: {cap}')


if __name__ == '__main__':
    main()
