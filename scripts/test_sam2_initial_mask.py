"""Sanity-check what mask SAM2 produces for the step-1 bbox alone (no video,
no propagation). Tells us whether SAM2 mis-tracks because the very FIRST mask
was wrong, vs. drift over time."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
import importlib.util  # noqa: E402

def _load(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

# strip act_robot/sam2 shadowing before sam2 import
_sf = _load('sam2_features', _ROOT / 'sam2_features.py')


def overlay(img: Image.Image, mask: np.ndarray, color=(0, 255, 0)) -> Image.Image:
    arr = np.array(img)
    arr[mask] = (arr[mask] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
    return Image.fromarray(arr)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--log-dir', required=True)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    bbox = np.load(log_dir / 'step_0001' / 'bbox.npy')
    jpeg_path = log_dir / 'step_0001' / 'wrist.jpg'
    img = Image.open(jpeg_path).convert('RGB')
    print(f'image size: {img.size}   bbox xyxy: {bbox.tolist()}')

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam = build_sam2(
        'configs/sam2.1/sam2.1_hiera_s.yaml',
        '/home/znyyb/hww/vla/act_robot/sam2/checkpoints/sam2.1_hiera_small.pt',
        device='cuda',
    )
    predictor = SAM2ImagePredictor(sam)
    predictor.set_image(np.array(img))

    # 1) bbox prompt with multimask_output (get all candidate masks ranked by score)
    masks, scores, _ = predictor.predict(
        box=bbox[None, :],
        multimask_output=True,
    )
    print(f'\nmultimask_output=True: got {len(masks)} candidate masks (descending score)')
    for i, (m, s) in enumerate(zip(masks, scores)):
        area = int((m > 0).sum())
        print(f'  candidate {i}: score={s:.3f}  area={area}  ({100*area/m.size:.2f}% of frame)')

    out_dir = Path(args.out or log_dir.parent.parent / 'scripts' / '_sam2_initial_mask')
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (m, s) in enumerate(zip(masks, scores)):
        ov = overlay(img, m > 0)
        # draw bbox
        from PIL import ImageDraw
        draw = ImageDraw.Draw(ov)
        draw.rectangle(bbox.tolist(), outline=(255, 0, 0), width=2)
        draw.text((bbox[0] + 2, max(bbox[1] - 14, 2)),
                  f'cand{i} score={s:.2f}', fill=(255, 0, 0))
        ov.save(out_dir / f'cand_{i}_score{s:.3f}.png')
    print(f'\nwrote candidate overlays to {out_dir}')


if __name__ == '__main__':
    main()
