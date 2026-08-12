"""Compute, at every chunk-replay query boundary, the full chunk[0..9] the
model produced, and compare it to the previous chunk's [k] entries that point
at the SAME absolute future timestep.

The point: chunk(t=1)[10] and chunk(t=11)[0] both predict the absolute target
at time t=11. In a temporally-consistent model these two predictions should be
equal. The size of |chunk(t=11)[0] - chunk(t=1)[9]| measures how much the
model's plan changed in 10 steps of fresh input — i.e. how inconsistent it is.

For comparison: also print typical per-step action delta in the TRAINING data
(|action[k+1] - action[k]|) so the user can see the inference jump against the
training distribution.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import h5py

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
ACTSAM2CVAEPolicy = _pol.ACTSAM2CVAEPolicy


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--log-dir', required=True)
    p.add_argument('--ckpt-dir', required=True)
    p.add_argument('--ckpt-name', default='policy_best.ckpt')
    p.add_argument('--data-dir', required=True,
                   help='SAM2 feature dataset (for training-action-delta stats)')
    p.add_argument('--num-steps', type=int, default=30)
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    with (ckpt_dir / 'policy_config.json').open() as f:
        cfg = json.load(f)
    if 'chunk_size' in cfg and 'num_queries' not in cfg:
        cfg['num_queries'] = cfg.pop('chunk_size')
    use_cvae = cfg.get('use_cvae', False)
    policy = ACTSAM2CVAEPolicy(cfg) if use_cvae else ACTSAM2Policy(cfg)
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

    log_dir = Path(args.log_dir)
    steps = sorted(log_dir.glob('step_*'))
    steps = [s for s in steps if (s/'wrist.jpg').exists() and (s/'bbox.npy').exists()]
    steps = steps[: args.num_steps]
    print(f'replay {len(steps)} steps')

    bbox = np.load(steps[0] / 'bbox.npy')
    extractor = SAM2FeatureExtractor()
    feats, _ = extractor.extract_episode([s/'wrist.jpg' for s in steps], bbox,
                                         feat_dtype='float32', return_masks=False)

    chunk_n = policy.num_queries
    # Query at steps 1, 11, 21 (1-indexed)
    boundaries = [1, 11, 21]
    chunks = {}
    for sid in boundaries:
        if sid > len(steps): break
        rs = np.load(steps[sid - 1] / 'robot_state.npy')
        qt = torch.from_numpy(rs.astype(np.float32)).cuda()
        qn = ((qt - qpos_mean) / qpos_std).unsqueeze(0)
        ft = torch.from_numpy(feats[sid - 1]).unsqueeze(0).cuda()
        with torch.inference_mode():
            a_norm = policy(qn, ft)
        chunk = (a_norm[0] * action_std + action_mean).cpu().numpy()  # [10, 7]
        chunks[sid] = (rs, chunk)

    # Per chunk: show how chunk[k] evolves and total intra-chunk drift
    for sid, (rs, ch) in chunks.items():
        intra = np.linalg.norm(ch[-1] - ch[0])
        print(f'\n=== chunk @ step {sid}  (qpos={rs[:6].round(3).tolist()}, grip={rs[6]:+.3f}) ===')
        print(f'  intra-chunk drift |chunk[9]-chunk[0]| = {intra:.4f}')
        for k in [0, 1, 4, 8, 9]:
            d_from_now = np.linalg.norm(ch[k][:6] - rs[:6])
            print(f'  chunk[{k}]: joints={ch[k][:6].round(3).tolist()} grip={ch[k][6]:+.3f}   '
                  f'|chunk[{k}] - qpos| = {d_from_now:.4f}')

    # Compare: chunk(t=1)[9] predicts target at step 11. chunk(t=11)[0] predicts target at step 12.
    # In a consistent model: should differ by ~one stride of motion.
    if 1 in chunks and 11 in chunks:
        a = chunks[1][1][9, :6]
        b = chunks[11][1][0, :6]
        # chunk(t=1)[10] would predict target at step 11 — but chunk only has 10 entries (indices 0..9)
        # chunk(t=1)[9] is target at step 11 (chunk index k maps to time t+k+1, so chunk(1)[9]=t=11).
        # chunk(t=11)[0] is target at step 12. So they predict ADJACENT times.
        diff = np.linalg.norm(b - a)
        print(f'\n*** chunk-boundary inconsistency at step 11 ***')
        print(f'  chunk(t=1)[9]  (planned target at step 11): {a.round(3).tolist()}')
        print(f'  chunk(t=11)[0] (planned target at step 12): {b.round(3).tolist()}')
        print(f'  |b - a| = {diff:.4f}  (these predict targets 1 stride apart, should be tiny)')

    if 11 in chunks and 21 in chunks:
        a = chunks[11][1][9, :6]
        b = chunks[21][1][0, :6]
        diff = np.linalg.norm(b - a)
        print(f'\n*** chunk-boundary inconsistency at step 21 ***')
        print(f'  |chunk(t=11)[9] - chunk(t=21)[0]| = {diff:.4f}')

    # Training data per-step action delta — this is the "natural" variance scale
    data_dir = Path(args.data_dir)
    info = json.load(open(data_dir / 'dataset_info.json'))
    train_deltas = []
    for ep in info['train_indices'][:30]:
        with h5py.File(data_dir / f'episode_{ep}.hdf5', 'r') as f:
            a = f['/action'][:][:, :6]  # joints only
        for k in range(len(a) - 1):
            train_deltas.append(float(np.linalg.norm(a[k+1] - a[k])))
    arr = np.array(train_deltas)
    print(f'\n=== training |action[k+1] - action[k]| over 30 episodes (n={len(arr)}) ===')
    print(f'  mean={arr.mean():.4f}  std={arr.std():.4f}  median={np.median(arr):.4f}  '
          f'p95={np.percentile(arr,95):.4f}  p99={np.percentile(arr,99):.4f}  max={arr.max():.4f}')


if __name__ == '__main__':
    main()
