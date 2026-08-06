"""Dataset utilities for ACT training."""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodicDataset(Dataset):
    """Load pre-converted HDF5 episodes for ACT training.

    Each HDF5 file contains:
        /observations/qpos      [T, state_dim]
        /observations/qvel      [T, state_dim]
        /observations/images/wrist  [T, H, W, 3] uint8
        /action                 [T, action_dim]
    """

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_episode_len):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_episode_len = max_episode_len
        self.is_sim = True

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            start_ts = np.random.choice(episode_len)

            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = {
                cam: root[f'/observations/images/{cam}'][start_ts]
                for cam in self.camera_names
            }
            action = root['/action'][start_ts:]
            action_len = episode_len - start_ts

        padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_episode_len, dtype=bool)
        is_pad[action_len:] = True

        all_cam_images = np.stack([image_dict[cam] for cam in self.camera_names], axis=0)
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad)

        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0

        action_data = (action_data - self.norm_stats['action_mean']) / self.norm_stats['action_std']
        qpos_data = (qpos_data - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std']

        return image_data, qpos_data, action_data, is_pad


class SAM2EpisodicDataset(Dataset):
    """Load pre-extracted SAM2 features for training the deterministic ACT head.

    Each HDF5 file (written by extract_sam2_features.py) contains:
        /observations/qpos       [T, state_dim] float32
        /observations/qvel       [T, state_dim] float32
        /observations/sam2_feat  [T, 256, 64, 64] float16
        /action                  [T, action_dim] float32     (absolute targets)

    qpos is normalised with stats from train indices. The target the model
    learns is either:
      * `absolute`   — the same absolute action as stored, normalised by
                       action_mean / action_std (legacy v1/v2 mode)
      * `delta`      — `action[t+k] - qpos[t]` (relative to current qpos),
                       normalised by delta_mean / delta_std. This bounds the
                       output near zero by construction, preventing the model
                       from emitting actions outside the training distribution.
    sam2_feat is fed in unnormalised (already centred around 0 by SAM2).
    """

    def __init__(self, episode_ids, dataset_dir, norm_stats, max_episode_len,
                 action_repr: str = 'absolute'):
        super().__init__()
        assert action_repr in ('absolute', 'delta')
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.norm_stats = norm_stats
        self.max_episode_len = max_episode_len
        self.action_repr = action_repr

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(path, 'r') as root:
            T, action_dim = root['/action'].shape
            start_ts = np.random.choice(T)

            qpos = root['/observations/qpos'][start_ts]
            sam2_feat = root['/observations/sam2_feat'][start_ts]  # [256, 64, 64]
            action = root['/action'][start_ts:]
            action_len = T - start_ts

        if self.action_repr == 'delta':
            # target_k = action[start_ts + k] - qpos[start_ts]  for k in [0, action_len)
            target = action.astype(np.float32) - qpos.astype(np.float32)[None, :]
            target_mean = self.norm_stats['delta_mean']
            target_std = self.norm_stats['delta_std']
        else:
            target = action.astype(np.float32)
            target_mean = self.norm_stats['action_mean']
            target_std = self.norm_stats['action_std']

        padded_target = np.zeros((self.max_episode_len, action_dim), dtype=np.float32)
        padded_target[:action_len] = target
        is_pad = np.zeros(self.max_episode_len, dtype=bool)
        is_pad[action_len:] = True

        sam2_feat_t = torch.from_numpy(sam2_feat.astype(np.float32))   # [256, 64, 64]
        qpos_t = torch.from_numpy(qpos).float()
        target_t = torch.from_numpy(padded_target).float()
        is_pad_t = torch.from_numpy(is_pad)

        target_t = (target_t - target_mean) / target_std
        qpos_t = (qpos_t - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std']

        return sam2_feat_t, qpos_t, target_t, is_pad_t


def get_norm_stats(dataset_dir: str, episode_indices: list[int],
                   chunk_size: int = 10) -> dict:
    """Compute normalization statistics from the given episode indices only.

    Also computes delta_mean / delta_std for delta-action training, defined as
    `action[t+k] - qpos[t]` over every (t, k) pair with k in [0, chunk_size).
    The mean/std is taken per-dim across all such deltas — per-dim because
    different joints / gripper have very different motion scales.
    """
    all_qpos, all_actions = [], []
    max_episode_len = 0
    state_dim = None

    for idx in episode_indices:
        path = os.path.join(dataset_dir, f'episode_{idx}.hdf5')
        with h5py.File(path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
        if state_dim is None:
            state_dim = qpos.shape[1]
        max_episode_len = max(max_episode_len, len(qpos))
        all_qpos.append(torch.from_numpy(qpos))
        all_actions.append(torch.from_numpy(action))

    all_qpos_cat = torch.cat(all_qpos, dim=0)
    all_actions_cat = torch.cat(all_actions, dim=0)

    action_mean = all_actions_cat.mean(dim=0)
    action_std = torch.clamp(all_actions_cat.std(dim=0), min=1e-2)
    qpos_mean = all_qpos_cat.mean(dim=0)
    qpos_std = torch.clamp(all_qpos_cat.std(dim=0), min=1e-2)

    # Delta stats: for each episode, compute action[t+k] - qpos[t] for valid (t, k)
    delta_chunks = []
    for q, a in zip(all_qpos, all_actions):
        T = len(q)
        for k in range(chunk_size):
            if T - k <= 0:
                continue
            # action[k:T] - qpos[0:T-k]
            d = a[k:T] - q[:T - k]            # [T - k, state_dim]
            delta_chunks.append(d)
    deltas = torch.cat(delta_chunks, dim=0)
    delta_mean = deltas.mean(dim=0)
    delta_std = torch.clamp(deltas.std(dim=0), min=1e-2)

    return {
        'action_mean': action_mean.numpy(),
        'action_std': action_std.numpy(),
        'qpos_mean': qpos_mean.numpy(),
        'qpos_std': qpos_std.numpy(),
        'delta_mean': delta_mean.numpy(),
        'delta_std': delta_std.numpy(),
        'example_qpos': all_qpos[0].numpy(),
        'state_dim': state_dim,
        'max_episode_len': max_episode_len,
        'chunk_size_for_delta': chunk_size,
    }
