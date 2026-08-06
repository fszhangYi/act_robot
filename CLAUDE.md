# act_robot — Claude Context

## Python 环境

**所有训练 / 推理 / 数据转换都使用 `anygrasp` conda 环境**：

```
/home/znyyb/miniconda3/envs/anygrasp/bin/python
```

- torch 2.4.1+cu121, cuDNN 9.1 — 已验证 conv backward 正常
- 已安装：torch, torchvision, h5py, einops, numpy, PIL

不要用其它环境：
- `dexvla` / `dexvla01` — torch 与系统 `/usr/local/cuda-11.7/` 的 cuDNN 不兼容，需要手动覆盖 `LD_LIBRARY_PATH` 才能 backward
- `base` / `affordance` / `ai2thor_env` — 没装 torch

## 数据 / Checkpoint 路径

约定输出在 `/home/znyyb/hww/vla/gongjian/` 下，命名 `cam{stride_info}_{view}_{mode}_{data|ckpt}`，例如：
- 数据：`cam100_15_wrist_cart_abs_data/`
- 训练结果：`cam100_15_wrist_cart_abs_ckpt/`

## 三种 action_space 模式

| 模式 | qpos | action | 推理 compose_pose |
|---|---|---|---|
| `joint` | `[θ1..θ6, gripper_obs]` | `[θ1..θ6, gripper_target]` 下一 stride 帧（绝对） | 直接返回 action |
| `cartesian_abs` | `[x,y,z,rx,ry,rz, gripper_obs]` | `[x,y,z,rx,ry,rz, gripper_target]` 下一 stride 帧（绝对） | 直接返回 action |
| `cartesian` (legacy) | 同上 | SE(3) 相对位姿增量 + 绝对 gripper | `T_target = T_current @ T_action` |

`cartesian_abs` 仅在数据集 rx/ry/rz 不跨越 ±π 边界时安全（cam_100_15 数据集已验证：rx ∈ [+3.1347, +3.1416]，无 wrap 风险）。
