"""Replay a serve.py session using OFFLINE SAM2 features.

Takes a session log (wrist.jpg + bbox.npy + robot_state.npy + action.npy per step),
re-extracts F_t using SAM2FeatureExtractor (single-shot propagate over the full
clip — the training-time path), feeds those F_t into the same ACTSAM2 model,
and compares the offline-F_t predicted next_state vs. what serve.py recorded
(which used streaming F_t).

If offline-F_t predictions look sensible while the recorded streaming-F_t
predictions are stuck/wrong, that confirms streaming F_t drift is the root cause.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent

# Do NOT add act_robot/ to sys.path — that triggers namespace shadowing of the
# editable-installed sam2 package. Load local modules via importlib instead.
import importlib.util  # noqa: E402

def _load(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod          # so dependents (e.g. `from policy import`) can find each other
    spec.loader.exec_module(mod)
    return mod

# Order matters: load sam2_features FIRST so its _strip_sam2_repo_shadow() runs
# before anything tries `import sam2`.
_sf = _load('sam2_features', _ROOT / 'sam2_features.py')
SAM2FeatureExtractor = _sf.SAM2FeatureExtractor
# detr/ subpackage is needed by policy.py; make act_robot/ importable JUST for detr.
sys.path.append(str(_ROOT))
_pol = _load('policy', _ROOT / 'policy.py')
ACTSAM2Policy = _pol.ACTSAM2Policy
_serve = _load('serve_mod', _ROOT / 'serve.py')
compose_pose = _serve.compose_pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--ckpt-dir', required=True)
    parser.add_argument('--ckpt-name', default='policy_best.ckpt')
    parser.add_argument('--num-steps', type=int, default=None,
                        help='Cap (default = all steps in log)')
    parser.add_argument('--print-every', type=int, default=10)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    ckpt_dir = Path(args.ckpt_dir)

    # Load policy + stats
    with (ckpt_dir / 'policy_config.json').open() as f:
        cfg = json.load(f)
    if 'chunk_size' in cfg and 'num_queries' not in cfg:
        cfg['num_queries'] = cfg.pop('chunk_size')
    policy = ACTSAM2Policy(cfg)
    ckpt = torch.load(ckpt_dir / args.ckpt_name, map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        policy.model.load_state_dict(ckpt['model'])
    else:
        policy.load_state_dict(ckpt)
    policy.eval().cuda()
    with (ckpt_dir / 'dataset_stats.pkl').open('rb') as f:
        stats = pickle.load(f)
    qpos_mean = torch.from_numpy(stats['qpos_mean']).float().cuda()
    qpos_std  = torch.from_numpy(stats['qpos_std']).float().cuda()
    action_mean = torch.from_numpy(stats['action_mean']).float().cuda()
    action_std  = torch.from_numpy(stats['action_std']).float().cuda()

    # Gather steps — drop any incomplete dir (e.g. last step that got truncated mid-write)
    raw_steps = sorted(log_dir.glob('step_*'))
    steps = [s for s in raw_steps if (s / 'wrist.jpg').exists() and (s / 'bbox.npy').exists()
             and (s / 'robot_state.npy').exists() and (s / 'next_state.npy').exists()]
    dropped = len(raw_steps) - len(steps)
    if args.num_steps:
        steps = steps[: args.num_steps]
    print(f'replaying {len(steps)} steps' + (f' (dropped {dropped} incomplete)' if dropped else ''))

    # Step 1 bbox
    bbox = np.load(steps[0] / 'bbox.npy')
    print(f'step-1 bbox: {bbox.tolist()}')
    if not (bbox[2] > bbox[0] and bbox[3] > bbox[1]):
        raise SystemExit('invalid bbox in step_0001/bbox.npy')

    # Offline SAM2 extraction over all frames
    print('running offline SAM2 over the full clip...')
    extractor = SAM2FeatureExtractor()
    frame_paths = [s / 'wrist.jpg' for s in steps]
    feats, _ = extractor.extract_episode(
        frame_paths, bbox, feat_dtype='float32', return_masks=False,
    )
    print(f'  feats shape: {feats.shape}  dtype: {feats.dtype}')

    chunk = policy.num_queries
    action_space = cfg.get('action_space', 'joint')

    # Chunk-replay mimic of serve.py
    current_chunk = None
    chunk_offset = chunk

    print(f'\n  k=k=step_idx, action_space={action_space}')
    print(f'  {"step":>4} | {"recorded next[:3]":>30} | {"offline next[:3]":>30} | {"diff":>8}')
    print(f'  {"-"*4} | {"-"*30} | {"-"*30} | {"-"*8}')
    for k, sd in enumerate(steps):
        rs = np.load(sd / 'robot_state.npy')
        recorded_next = np.load(sd / 'next_state.npy')

        # chunk replay: re-query every chunk steps
        if chunk_offset >= chunk:
            ft = torch.from_numpy(feats[k]).unsqueeze(0).cuda()
            qt = torch.from_numpy(rs.astype(np.float32)).cuda()
            qn = ((qt - qpos_mean) / qpos_std).unsqueeze(0)
            with torch.inference_mode():
                a_norm = policy(qn, ft)
            current_chunk = (a_norm[0] * action_std + action_mean).cpu().numpy()
            chunk_offset = 0
        action = current_chunk[chunk_offset]
        chunk_offset += 1

        offline_next = compose_pose(rs, action, action_space)

        if k % args.print_every == 0 or k < 5 or k >= len(steps) - 2:
            diff = float(np.abs(recorded_next[:3] - offline_next[:3]).max())
            print(f'  {k+1:4d} | {np.round(recorded_next[:3],3).tolist()!s:>30} | '
                  f'{np.round(offline_next[:3],3).tolist()!s:>30} | {diff:8.4f}')


if __name__ == '__main__':
    main()
