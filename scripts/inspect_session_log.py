"""Visualize a serve.py session log:
  - draw the step-1 bbox onto step-1 wrist frame
  - sample mid/late frames to see where the target actually moves
  - optionally re-run SAM2 streaming and overlay the predicted mask each frame
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def draw_bbox(img: Image.Image, bbox: np.ndarray, color=(255, 0, 0), width: int = 3,
              label: str | None = None) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = bbox.tolist()
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    if label:
        draw.text((x0 + 2, max(y0 - 14, 2)), label, fill=color)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--out-dir', default='scripts/_session_inspect')
    parser.add_argument('--sample-steps', nargs='+', type=int,
                        default=[1, 5, 20, 50, 100, 150, 191])
    parser.add_argument('--rerun-sam2', action='store_true',
                        help='Re-run streaming SAM2 on the saved frames and overlay mask')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) / log_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    step1 = log_dir / 'step_0001'
    if not step1.exists():
        raise SystemExit(f'no step_0001 in {log_dir}')
    bbox = np.load(step1 / 'bbox.npy')
    img1 = Image.open(step1 / 'wrist.jpg').convert('RGB')
    print(f'frame size: {img1.size}    step-1 bbox xyxy: {bbox.tolist()}')

    annotated = draw_bbox(img1, bbox, label=f'init bbox {bbox.astype(int).tolist()}')
    annotated.save(out_dir / 'frame_001_bbox.png')

    # Sample mid/late frames as-is (no bbox overlay since they didn't get one)
    for sid in args.sample_steps:
        sd = log_dir / f'step_{sid:04d}'
        if not sd.exists():
            continue
        im = Image.open(sd / 'wrist.jpg').convert('RGB')
        # overlay step-1 bbox for visual reference (where the target WAS at t=0)
        with_box = draw_bbox(im, bbox, color=(255, 0, 0), width=2,
                             label=f'step {sid} (red = step-1 bbox for reference)')
        with_box.save(out_dir / f'frame_{sid:03d}.png')

    # Optionally rerun SAM2 streaming and overlay predicted mask per sampled step
    if args.rerun_sam2:
        from sam2_features import SAM2StreamingFeatureExtractor
        sf = SAM2StreamingFeatureExtractor()

        all_steps = sorted(log_dir.glob('step_*'))
        max_sample = max(args.sample_steps)
        all_steps = [s for s in all_steps if int(s.name.split('_')[1]) <= max_sample]
        print(f'  re-running SAM2 over {len(all_steps)} frames…')

        masks = {}
        for sd in all_steps:
            sid = int(sd.name.split('_')[1])
            jpeg = (sd / 'wrist.jpg').read_bytes()
            try:
                if sid == 1:
                    ft = sf.init_first_frame(jpeg, bbox)
                else:
                    ft = sf.step(jpeg)
            except Exception as exc:
                print(f'    step {sid}: SAM2 step failed: {exc}')
                continue
            # mask from inference_state — last propagated frame's mask is in non_cond_frame_outputs
            if sid in args.sample_steps:
                state = sf.state
                obj0 = state['output_dict_per_obj'][0]
                pred_mask_dict = obj0['non_cond_frame_outputs'] if sid > 1 else obj0['cond_frame_outputs']
                frame_idx = sid - 1
                if frame_idx in pred_mask_dict:
                    pred_low = pred_mask_dict[frame_idx]['pred_masks']  # [1,1,H,W]
                    import torch as t
                    pm = t.nn.functional.interpolate(
                        pred_low.float(), size=(img1.size[1], img1.size[0]),
                        mode='bilinear', align_corners=False,
                    )[0, 0].cpu().numpy() > 0
                    masks[sid] = pm

        for sid, m in masks.items():
            sd = log_dir / f'step_{sid:04d}'
            im = np.array(Image.open(sd / 'wrist.jpg').convert('RGB'))
            overlay = im.copy()
            overlay[m] = (overlay[m] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
            out = Image.fromarray(overlay)
            if sid == 1:
                out = draw_bbox(out, bbox, color=(255, 0, 0), width=2)
            out.save(out_dir / f'sam2_mask_{sid:03d}.png')
            area = int(m.sum())
            print(f'    step {sid}: mask area={area}  ({100*area/m.size:.2f}% of frame)')

    print(f'output: {out_dir}')


if __name__ == '__main__':
    main()
