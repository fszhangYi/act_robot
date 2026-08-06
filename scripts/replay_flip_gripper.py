"""Same as replay_with_offline_sam2.py but flips qpos[6] sign before feeding
the model, to verify the gripper-convention hypothesis."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
import importlib.util  # noqa: E402

def _load(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

_sf = _load('sam2_features', _ROOT / 'sam2_features.py')
SAM2FeatureExtractor = _sf.SAM2FeatureExtractor
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
    parser.add_argument('--print-every', type=int, default=10)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    ckpt_dir = Path(args.ckpt_dir)

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

    raw_steps = sorted(log_dir.glob('step_*'))
    steps = [s for s in raw_steps if (s / 'wrist.jpg').exists() and (s / 'bbox.npy').exists()
             and (s / 'robot_state.npy').exists()]

    bbox = np.load(steps[0] / 'bbox.npy')
    print(f'step-1 bbox: {bbox.tolist()}  N={len(steps)}')

    print('extracting offline SAM2 features…')
    extractor = SAM2FeatureExtractor()
    feats, _ = extractor.extract_episode(
        [s / 'wrist.jpg' for s in steps], bbox, feat_dtype='float32', return_masks=False,
    )
    print(f'  F_t shape: {feats.shape}')

    chunk = policy.num_queries
    action_space = cfg.get('action_space', 'joint')

    current_chunk = None
    chunk_offset = chunk
    print(f'\n  {"step":>4} | gripper_in (raw → flipped) | next_state[:3,6]')
    print(f'  {"-"*4} | {"-"*30} | {"-"*40}')
    for k, sd in enumerate(steps):
        rs = np.load(sd / 'robot_state.npy').copy()
        # FLIP gripper
        rs[6] = -rs[6]

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

        next_state = compose_pose(rs, action, action_space)
        raw_gripper = -rs[6]  # original (negated back)
        if k % args.print_every == 0 or k < 5 or k >= len(steps) - 2:
            print(f'  {k+1:4d} | {raw_gripper:+.3f} → {rs[6]:+.3f}             | '
                  f'next={np.round(next_state[:3],3).tolist()} grip={next_state[6]:+.3f}')


if __name__ == '__main__':
    main()
