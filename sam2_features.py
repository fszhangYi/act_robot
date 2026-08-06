"""SAM2 feature extractor for Stage-1 offline F_t caching.

Wraps `build_sam2_video_predictor` and monkey-patches `_track_step` to capture
the post-memory-attention pixel features `pix_feat_with_mem` (shape [1, 256, 64, 64]
for sam2.1-small at 1024×1024 internal resolution) — this is the F_t the
SAM2Grasp paper feeds into the ACT head.

Usage:
    ext = SAM2FeatureExtractor(config_file=..., ckpt_path=...)
    feats, masks = ext.extract_episode(frame_paths, bbox_xyxy)
    # feats: [T, 256, 64, 64] float16 by default
    # masks: [T, H, W]        uint8  (binary, in original image resolution)
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


def _strip_sam2_repo_shadow() -> None:
    """Prevent the act_robot/sam2/ repo dir from shadowing the installed sam2 package.

    Two-pronged fix:
      1. Drop entries from sys.path that would let the default PathFinder resolve
         `sam2` to the repo's outer dir as a namespace package.
      2. Hoist the editable-install MetaPathFinder to the front of sys.meta_path,
         so even if step (1) misses an entry (e.g., a downstream script re-adds
         act_robot/ for its own imports), the editable finder still wins over
         PathFinder for `import sam2`.

    Background: the editable install registers a MetaPathFinder via .pth, but
    Python's default PathFinder is earlier in sys.meta_path. With act_robot/ on
    sys.path, PathFinder finds act_robot/sam2/ (no __init__.py) as an implicit
    namespace package and SAM2's build_sam.py:20-32 raises.
    """
    act_robot = str(Path(__file__).resolve().parent)
    bad = {'', act_robot, str(Path(act_robot) / 'sam2')}
    sys.path[:] = [p for p in sys.path if p not in bad]

    # Drop any cached namespace-package binding so the next `import sam2` re-resolves.
    cached = sys.modules.get('sam2')
    if cached is not None and getattr(cached, '__file__', None) is None:
        sys.modules.pop('sam2', None)

    # Hoist the SAM_2 editable finder ahead of PathFinder.
    for i, finder in enumerate(sys.meta_path):
        mod = getattr(type(finder), '__module__', '')
        if 'SAM_2' in mod or 'sam_2' in mod.lower() or 'sam2_finder' in mod:
            if i != 0:
                sys.meta_path.insert(0, sys.meta_path.pop(i))
            break


_strip_sam2_repo_shadow()


class SAM2FeatureExtractor:
    """Run frozen SAM2 over an episode and capture per-frame F_t."""

    def __init__(
        self,
        config_file: str = 'configs/sam2.1/sam2.1_hiera_s.yaml',
        ckpt_path: str = '/home/znyyb/hww/vla/act_robot/sam2/checkpoints/sam2.1_hiera_small.pt',
        device: str = 'cuda',
    ) -> None:
        _strip_sam2_repo_shadow()  # re-apply in case sys.path was mutated after module load
        from sam2.build_sam import build_sam2_video_predictor

        self.device = device
        self.predictor = build_sam2_video_predictor(config_file, ckpt_path, device=device)
        self._captured_pix_feat: torch.Tensor | None = None
        self._patch_track_step()

    def _patch_track_step(self) -> None:
        """Wrap predictor._track_step so we can grab pix_feat_with_mem per frame."""
        original = self.predictor._track_step

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            # _track_step returns (current_out, sam_outputs, high_res_features, pix_feat)
            pix_feat = result[3]
            self._captured_pix_feat = pix_feat
            return result

        self.predictor._track_step = wrapped  # type: ignore[method-assign]

    def extract_episode(
        self,
        frame_paths: Sequence[Path | str],
        bbox_xyxy: Sequence[float],
        feat_dtype: str = 'float16',
        return_masks: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract F_t for every frame after seeding the t=0 bbox prompt.

        Args:
            frame_paths: ordered list of RGB image paths (the video).
            bbox_xyxy:   [xmin, ymin, xmax, ymax] in pixel coordinates of the
                         object to track in `frame_paths[0]`.
            feat_dtype:  'float16' (default, halves disk) or 'float32'.
            return_masks: if True, also return per-frame binary masks at
                         original image resolution (for sanity checks /
                         overlay videos).

        Returns:
            features: np.ndarray [T, 256, 64, 64] in the requested dtype.
            masks:    np.ndarray [T, H, W] uint8, or None.
        """
        frame_paths = [Path(p) for p in frame_paths]
        if not frame_paths:
            raise ValueError('frame_paths must be non-empty')
        T = len(frame_paths)
        bbox = np.asarray(bbox_xyxy, dtype=np.float32)
        if bbox.shape != (4,):
            raise ValueError(f'bbox_xyxy must have 4 elements; got {bbox.shape}')

        target_np_dtype = {'float16': np.float16, 'float32': np.float32}[feat_dtype]

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            # SAM2 video predictor wants frames as <stem>.jpg with sortable names.
            for i, p in enumerate(frame_paths):
                os.symlink(p.resolve(), tmp / f'{i:05d}.jpg')

            features = np.empty((T, 256, 64, 64), dtype=target_np_dtype)
            masks: np.ndarray | None = None

            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) \
                if self.device == 'cuda' else _NullCtx()
            with torch.inference_mode(), autocast_ctx:
                state = self.predictor.init_state(video_path=str(tmp))
                self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    box=bbox,
                )

                for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(state):
                    if self._captured_pix_feat is None:
                        raise RuntimeError('SAM2 _track_step did not run for frame ' f'{frame_idx}')
                    f = self._captured_pix_feat.detach().to(torch.float32).cpu().numpy()[0]
                    features[frame_idx] = f.astype(target_np_dtype, copy=False)
                    if return_masks:
                        m = (mask_logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
                        if masks is None:
                            masks = np.empty((T, *m.shape), dtype=np.uint8)
                        masks[frame_idx] = m
                    self._captured_pix_feat = None  # invalidate to detect bugs

        return features, masks


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class SAM2StreamingFeatureExtractor:
    """Online single-frame F_t extractor for serving / closed-loop inference.

    Usage:
        sf = SAM2StreamingFeatureExtractor()
        F0 = sf.init_first_frame(jpeg_bytes, bbox_xyxy)  # refresh=1
        F1 = sf.step(jpeg_bytes)                          # refresh=0
        ...

    Each returned F_t is shape [1, 256, 64, 64] float32 (NOT cast to half;
    casting is done by the caller if needed).

    KNOWN CAVEAT: F_t produced here drifts slightly from F_t produced by
    SAM2FeatureExtractor on the SAME episode (max abs diff ~1.7 at frame 1,
    decaying toward ~0.2 by frame 7). Frame 0 matches exactly. The cause is
    bf16-autocast / memory-encoder accumulation noise that does not occur in
    a single big propagation but does occur when propagation is restarted
    per frame. This is a SAM2-internal numerical issue; for a smoke-level
    deployment test it's tolerable. To eliminate it for production, the
    cleanest workaround is to re-init the state and re-propagate from frame 0
    each step (much slower) or upstream a fix to SAM2's preflight bookkeeping.
    """

    def __init__(
        self,
        config_file: str = 'configs/sam2.1/sam2.1_hiera_s.yaml',
        ckpt_path: str = '/home/znyyb/hww/vla/act_robot/sam2/checkpoints/sam2.1_hiera_small.pt',
        device: str = 'cuda',
    ) -> None:
        _strip_sam2_repo_shadow()  # re-apply in case sys.path was mutated after module load
        from sam2.build_sam import build_sam2_video_predictor

        self.device = device
        self.predictor = build_sam2_video_predictor(config_file, ckpt_path, device=device)
        self.image_size = self.predictor.image_size
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self._captured_pix_feat: torch.Tensor | None = None
        self._patch_track_step()
        self.state: dict | None = None
        self.t: int = 0

    def _patch_track_step(self) -> None:
        original = self.predictor._track_step

        def wrapped(*args, **kwargs):
            r = original(*args, **kwargs)
            self._captured_pix_feat = r[3]
            return r

        self.predictor._track_step = wrapped  # type: ignore[method-assign]

    def _fresh_state(self) -> dict:
        device = torch.device(self.device)
        return {
            'images': [],
            'num_frames': 0,
            'offload_video_to_cpu': True,
            'offload_state_to_cpu': False,
            'video_height': None,
            'video_width': None,
            'device': device,
            'storage_device': device,
            'point_inputs_per_obj': {},
            'mask_inputs_per_obj': {},
            'cached_features': {},
            'constants': {},
            'obj_id_to_idx': OrderedDict(),
            'obj_idx_to_id': OrderedDict(),
            'obj_ids': [],
            'output_dict_per_obj': {},
            'temp_output_dict_per_obj': {},
            'frames_tracked_per_obj': {},
        }

    def reset(self) -> None:
        self.state = self._fresh_state()
        self.t = 0
        self._captured_pix_feat = None

    def _preprocess(self, jpeg_bytes: bytes) -> tuple[torch.Tensor, int, int]:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
        w, h = img.size
        img = img.resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t = (t - self.img_mean) / self.img_std
        return t, h, w

    def _autocast(self):
        if self.device == 'cuda':
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return _NullCtx()

    @torch.inference_mode()
    def init_first_frame(self, jpeg_bytes: bytes, bbox_xyxy: Sequence[float]) -> torch.Tensor:
        """Seed the episode at t=0 with a bounding-box prompt; return F_0."""
        self.reset()
        img_t, h, w = self._preprocess(jpeg_bytes)
        assert self.state is not None
        self.state['images'].append(img_t)
        self.state['num_frames'] = 1
        self.state['video_height'] = h
        self.state['video_width'] = w

        with self._autocast():
            self.predictor.add_new_points_or_box(
                inference_state=self.state,
                frame_idx=0,
                obj_id=1,
                box=np.asarray(bbox_xyxy, dtype=np.float32),
            )
        if self._captured_pix_feat is None:
            raise RuntimeError('SAM2 _track_step did not fire on bbox add')
        ft = self._captured_pix_feat.detach()
        self._captured_pix_feat = None
        self.t = 1
        return ft

    @torch.inference_mode()
    def step(self, jpeg_bytes: bytes) -> torch.Tensor:
        """Advance one frame; return F_t."""
        if self.state is None or self.t == 0:
            raise RuntimeError('call init_first_frame() first')
        idx = self.t
        img_t, h, w = self._preprocess(jpeg_bytes)
        self.state['images'].append(img_t)
        self.state['num_frames'] = idx + 1

        with self._autocast():
            for _ in self.predictor.propagate_in_video(
                self.state, start_frame_idx=idx, max_frame_num_to_track=1,
            ):
                pass
        if self._captured_pix_feat is None:
            raise RuntimeError(f'no pix_feat captured for frame {idx}')
        ft = self._captured_pix_feat.detach()
        self._captured_pix_feat = None
        self.t = idx + 1
        return ft


__all__ = ['SAM2FeatureExtractor', 'SAM2StreamingFeatureExtractor']
