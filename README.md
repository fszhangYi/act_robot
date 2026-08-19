# act_robot

完全自包含的 ACT（Action Chunking with Transformers）机器人抓取训练和推理项目。

## 目录结构

```
act_robot/
├── convert_episodes.py   # 数据转换
├── train.py              # 训练
├── serve.py              # 推理服务器
├── verify.py             # 数据集验证
├── policy.py             # ACTPolicy 模型封装
├── dataset.py            # EpisodicDataset 数据集
└── detr/                 # DETR/ACT 模型架构
    ├── main.py
    ├── models/
    └── util/
```

---

## 完整流程

### Step 1：转换数据

```bash
python act_robot/convert_episodes.py \
    --input-dir /path/to/raw/cam_100_15 \
    --output-dir /data/act_v1 \
    --stride 3 \
    --train-ratio 0.9 \
    --seed 42
```

**关键参数**

| 参数 | 说明 |
|------|------|
| `--stride` | 时序降采样步长。原始数据 100Hz 推荐 `stride=3`（~33Hz）或 `stride=5`（20Hz）。步长越大，单步动作幅度越大，模型更容易学习有意义的运动。 |
| `--train-ratio` | 训练集比例，剩余为验证集 |

**Action 表示**

使用 SE(3) pose transformation，彻底解决 Euler 角环绕问题：

```
T_action = inv(T_current) @ T_next

action[0:3] = T_action[:3, 3]           # 局部坐标系平移（量值小，无 wrap）
action[3:6] = euler_xyz(T_action[:3,:3]) # 相对旋转（步长小时恒在 [-π,π] 内）
action[6]   = obs_gripper[t+1]           # 绝对 gripper 目标值
```

输出：`/data/act_v1/episode_0.hdf5` … `episode_N.hdf5` + `dataset_info.json`

---

### Step 2：验证数据

```bash
python act_robot/verify.py --data-dir /data/act_v1
```

关键检查项：action 旋转维度 `dim[3]`、`dim[4]`、`dim[5]` 中 `|x|>1.0` 的比例应 < 1%。
若出现 `WARNING: Euler wrap detected`，说明数据转换有问题，不要训练。

---

### Step 3：训练

```bash
python act_robot/train.py \
    --data-dir /data/act_v1 \
    --ckpt-dir /data/ckpt_v1 \
    --num-epochs 2000 \
    --batch-size 64 \
    --chunk-size 10 \
    --lr 1e-5 \
    --kl-weight 10
```

**标准超参数**

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--num-epochs` | 2000 | 500 episode 量级通常需要 1000~3000 epoch |
| `--batch-size` | 64 | 显存不足时降到 32 |
| `--chunk-size` | 10 | ACT action chunk 长度（stride=3 时等效 ~0.3s） |
| `--lr` | 1e-5 | 标准 ACT 学习率 |
| `--kl-weight` | 10 | CVAE KL 项权重 |

输出文件：
- `policy_best.ckpt` — val loss 最优 checkpoint
- `policy_last.ckpt` — 最后一个 epoch
- `dataset_stats.pkl` — 归一化统计（推理必须使用）
- `policy_config.json` — 模型配置（推理自动读取）
- `train_history.json` — 训练/验证 loss 曲线

---

### Step 4：启动推理服务器

```bash
# 默认模式（chunk replay）
python act_robot/serve.py \
    --checkpoint /data/ckpt_v1/policy_best.ckpt \
    --stats      /data/ckpt_v1/dataset_stats.pkl \
    --port 5000

# Temporal aggregation 模式（更平滑）
python act_robot/serve.py \
    --checkpoint /data/ckpt_v1/policy_best.ckpt \
    --stats      /data/ckpt_v1/dataset_stats.pkl \
    --port 5000 \
    --temporal-agg
```

**两种推理模式对比**

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| 默认（chunk replay） | 每 `chunk_size` 步 query 一次，顺序执行 chunk 中的动作 | 延迟敏感，计算量小 |
| `--temporal-agg` | 每步都 query，对历史 chunk 做指数加权平均（最新权重最高） | 运动更平滑，但每步都有推理开销 |

---

## 通信协议

`serve.py` 按 ckpt 的 `policy_config.json` → `use_sam2_features` 自动选协议，启动时打印
`wire_protocol=baseline_vla|sam2_legacy`。**以代码 / 启动 banner 为准**；金标准客户端见
`scripts/mock_client_sam2.py`（SAM2）与 `scripts/smoke_infer_tonglu.py`（基线离线冒烟）。

### A. `baseline_vla`（tonglu 多相机 / serve_tmp VLA 布局）

`use_sam2_features=false`。**无 refresh、无 num_cams**；新 TCP 连接 = 新 episode。

Client → Server（每帧，big-endian）：

```
4B   top_len    + top JPEG
4B   chest_len  + chest JPEG
4B   wrist2_len + wrist2 JPEG   → 映射为 camera "wrist_2"
4B   text_len   + text UTF-8    （接收但不用于推理）
28B  robot_state (7 × float32: x, y, z, rx, ry, rz, gripper)
```

Server → Client：

```
28B  next_state  (7 × float32) = compose_pose(current_state, predicted_action)
4B   term_flag   (uint32, 当前恒 0)
4B   reject_flag (uint32, 当前恒 0)
4B   text_len    + UTF-8（当前固定 "success"）
```

线上 JPEG 顺序固定为 top → chest → wrist2，再按 `camera_names` 重排后喂模型。

### B. `sam2_legacy`（100_15 SAM2Grasp）

`use_sam2_features=true`。

Client → Server（每帧，big-endian）：

```
4B   wrist_image_length (uint32)
N B  wrist JPEG bytes
4B   left_image_length  (uint32, 接收但不用于推理)
M B  rear-left JPEG bytes
28B  robot_state (7 × float32)
4B   refresh (uint32; 1 = 新 episode / 重置 SAM2)
if refresh == 1:
    16B  bbox xyxy (4 × float32, wrist 像素坐标)   # 无 prompt_type 字段
```

Server → Client：

```
28B  next_state (7 × float32) = compose_pose(current_state, predicted_action)
```

---

## 常见问题

**Q：val loss 不下降，一直在 0.4~0.5 附近**

先运行 `verify.py` 检查 action 分布。如果旋转维度 std 远大于其他维度（例如 >1.0），说明 Euler 环绕问题没有解决。

**Q：机器人运动抖动或乱跑**

检查推理侧 `next_state` 数值是否合理（正常应与 `current_state` 相差不大）。也可以检查 `dataset_stats.pkl` 中的 `action_mean/std` 是否与训练数据一致。

**Q：`stride` 怎么选？**

| 控制频率 | 推荐 stride | 等效频率 |
|----------|-------------|----------|
| 100 Hz   | 3           | ~33 Hz   |
| 100 Hz   | 5           | 20 Hz    |
| 50 Hz    | 2           | 25 Hz    |

动作幅度太小（模型预测接近全零）→ 增大 stride；动作幅度太大（运动不平滑）→ 减小 stride。

**Q：ResNet18 weights 加载报错**

下载 `resnet18-f37072fd.pth` 放到项目根目录的 `checkpoints/` 下，或设置环境让 torchvision 自动下载。

---

## 修复了什么

相比旧版 `act-main` + `act_server_new.py`：

1. **Euler 角环绕** — 改用 SE(3) pose transformation，消除 rotation dim 的 ±2π 跳变
2. **双重 ImageNet 归一化** — `act_server_new.py` 第二个 `__init__` 会在 `policy()` 之外再手动做一次 ImageNet normalization，而 `ACTPolicy.__call__` 内部已经做了，导致推理完全错误。新版只做 `/255.0`
3. **归一化统计污染** — 原版 `get_norm_stats` 用全量 episode（包含 val 集），新版只用 train 集
4. **argparse 侵入** — `detr/main.py` 的 `build_ACT_model_and_optimizer` 调用 `parse_args()` 读 sys.argv，需要 mock。改为 `parse_args([])` 后不再需要 mock
5. **高频采样** — `--stride` 参数支持时序降采样，解决动作步长过小导致模型预测偏零的问题

---

## SAM2Grasp 复现（多目标抓取场景）

针对**场景中多个相同工件、需指定抓哪一个**的问题，本项目实现了 SAM2Grasp 论文的思路：t=0 给一个 bbox prompt 指定目标，冻结 SAM2 用 memory attention 自动追踪并产出 object-centric 特征 `F_t`，喂给轻量 ACT 头预测动作 chunk。架构上去掉了 ResNet18 backbone 和 CVAE，loss 改 L2。详见 [docs/sam2grasp.pdf](docs/sam2grasp.pdf)。

### 准备 SAM2

```bash
# 装包（无需 root，绕开 python>=3.10 限制）
SAM2_BUILD_CUDA=0 /home/znyyb/miniconda3/envs/anygrasp/bin/pip install -e sam2 \
    --no-deps --no-build-isolation --ignore-requires-python

# 下载 small ckpt（~176MB）—— 默认路径与 resolve_sam2_ckpt 一致
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_small.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
# 也可: export SAM2_CKPT=/path/to/sam2.1_hiera_small.pt
# 或:   --sam2-ckpt /path/to/sam2.1_hiera_small.pt
```

### Step 1：Stage-1 离线特征抽取

```bash
python extract_sam2_features.py \
    --input-dir  /path/to/raw_episodes \
    --output-dir /path/to/sam2_data \
    --stride 3 \
    --action-space joint \
    --feat-dtype float16
```

需要每个 episode 目录里有 `bbox.json`，frame 0 的 `rgb_wrist_1` bbox 会被当成 SAM2 prompt。输出每个 episode 一个 HDF5，含 `/observations/sam2_feat`（`[T, 256, 64, 64] float16`，约 1MB/帧）、`/observations/qpos`、`/action`，外加 `dataset_info.json`（train/val split 沿用 0.9）。

### Step 2：训练 ACTSAM2

```bash
python train.py \
    --data-dir /path/to/sam2_data \
    --ckpt-dir /path/to/sam2_ckpt \
    --use-sam2-features \
    --num-epochs 300 --batch-size 2 --chunk-size 10 \
    --lr 1e-4 --grad-clip 1.0 --cosine-lr --min-lr 1e-6 \
    --action-space joint
```

> **重要**：`--grad-clip 1.0 --cosine-lr` 是必须的——v1（2000 epoch 无这两项）出现了 ep100→500 训练 loss 从 0.25 涨到 0.96 的优化崩溃。

ckpt 目录里会有 `policy_best.ckpt`（按 val loss 选）、`policy_last.ckpt`、`policy_epoch_{N}_seed_0.ckpt`（每 200 epoch）、`dataset_stats.pkl`、`policy_config.json`、`train_history.json`。`policy_config.json` 的 `use_sam2_features: true` 字段是 serve.py 自动识别模式的依据。

### Step 3：离线验证模型（推荐先做）

不需要 serve.py，直接喂训练时缓存的 F_t 进模型：

```bash
python scripts/eval_sam2_offline.py \
    --ckpt-dir /path/to/sam2_ckpt \
    --ckpt-name policy_epoch_200_seed_0.ckpt \
    --data-dir /path/to/sam2_data \
    --split val --num-episodes 10
```

输出 per-dim RMSE 和 per-chunk-position RMSE（k=0 是机器人实际下一步要执行的动作；k 越大 RMSE 应该越高，反映 chunk 末尾预测不确定性更大）。

### Step 4：起推理服务（serve.py 无需改动）

serve.py 通过读 ckpt 同目录的 `policy_config.json` 的 `use_sam2_features` 字段自动切到 SAM2 路径，加载 `SAM2StreamingFeatureExtractor`。线协议见上文 [通信协议 §B `sam2_legacy`](#b-sam2_legacy100_15-sam2grasp)；`refresh==1` 时仅多 **16B bbox**（无 `prompt_type`）。

```bash
python serve.py \
    --checkpoint /path/to/sam2_ckpt/policy_epoch_200_seed_0.ckpt \
    --stats      /path/to/sam2_ckpt/dataset_stats.pkl \
    --port 5000
```

启动后日志会打 `wire_protocol=sam2_legacy` 以及
`mode=SAM2Grasp  chunk_size=10  state_dim=7  action_space=joint`。

### Step 5：端到端 mock client 验证

`scripts/mock_client_sam2.py` 会自动起 server、按 episode 帧顺序送图（首帧带 bbox），打印每步 round-trip 时延和 next_state：

```bash
python scripts/mock_client_sam2.py \
    --checkpoint /path/to/sam2_ckpt/policy_epoch_200_seed_0.ckpt \
    --stats      /path/to/sam2_ckpt/dataset_stats.pkl \
    --episode-dir /path/to/raw_episode \
    --num-steps 30
```

### SAM2Grasp 协议扩展

相对 100_15 旧布局，**只有 `refresh==1` 时多 16 字节 bbox**；`refresh==0` 完全不变。
**不要**再发 `prompt_type`：当前实现 refresh=1 后直接读 16B bbox，多发 4B 会把 bbox 读偏。

完整字节布局见 [通信协议 §B](#b-sam2_legacy100_15-sam2grasp)。金标准客户端：`scripts/mock_client_sam2.py`。

Server 收到 refresh=1：重置 SAM2 state → 用 bbox `add_new_points_or_box` 在首帧锁定目标 → 提取 F_0 → 喂模型出 chunk。后续帧 SAM2 用 memory attention 自动追踪并产 F_t。

### 已知问题

1. **`SAM2StreamingFeatureExtractor` 与离线提取的 F_t 有数值漂移**：t=0 一致，t≥1 max abs diff 1.7→0.2（逐帧衰减）。是 SAM2 bf16 累加噪声在 `propagate_in_video` 分批调用时的副作用。训练用离线 F_t、推理用 streaming F_t 会有这个分布偏移，影响目前未定量评估。Workaround：每帧从 frame 0 全量 re-propagate（慢但保证一致性）。
2. **未修复的 bug 在 v1 ckpt 里**：v1 训练用的旧 L2 是 `.mean()` 除以全元素（含 padding），导致 `policy_best.ckpt` 是早期被 padding 比例骗的虚假 best。v1 ckpt 里 `policy_epoch_200_seed_0.ckpt` 最稳妥。v2 起的训练（README Step 2 当前版本）已经修复。

### 默认数据路径

| 类型 | 路径 |
|---|---|
| 原始 episodes | `/media/znyyb/EE223AE5223AB287/100_15/` |
| Stage-1 SAM2 特征 | `/media/znyyb/EE223AE5223AB287/act_dataset/cam100_15_wrist_sam2small_joint_data_full/` |
| v1 ckpt（无 grad-clip / cosine） | `/media/znyyb/EE223AE5223AB287/act_dataset/cam100_15_wrist_sam2small_joint_ckpt_full/` |
| v2 ckpt（带修复，short 300 epoch） | `/media/znyyb/EE223AE5223AB287/act_dataset/cam100_15_wrist_sam2small_joint_ckpt_v2/` |

---

## tonglu0602 数据集 — 多相机原始 ACT 管线

这一管线**不走 SAM2Grasp**，复用原始 ACT 路径，3 相机（`chest` / `top` / `wrist_2`）+ 绝对位姿动作 + annotation 文件帧切片。

**完整指南见 [`docs/tonglu0602.md`](docs/tonglu0602.md)**——覆盖数据布局、标注格式、转换/训练/推理命令、新的多相机线协议、rx wrap 处理、迁移到新机器的清单与故障排查。

一行速览：

```bash
# 转换
python convert_episodes.py --input-dir <root>/raw_data --output-dir <out> \
    --annotation-dir <root>/annotation --camera-names chest top wrist_2 \
    --action-space cartesian_abs --stride 1 --unwrap-rx

# 训练（冒烟）
python train.py --data-dir <out> --ckpt-dir <ckpt> \
    --num-epochs 20 --batch-size 4 --chunk-size 10 \
    --action-space cartesian_abs --camera-names chest top wrist_2 \
    --lr 1e-5 --kl-weight 10 --grad-clip 1.0

# 推理服务
python serve.py --checkpoint <ckpt>/policy_best.ckpt --stats <ckpt>/dataset_stats.pkl --port 5000
```

新线协议（`wire_protocol=baseline_vla`，与 `_handle_baseline_client` 一致）：

```
top JPEG + chest JPEG + wrist2 JPEG + text + 28B state
→ 28B next_state + term + reject + text
```

无 `refresh` / `num_cams`。详见 [通信协议 §A](#a-baseline_vlatonglu-多相机--serve_tmp-vla-布局) 与 [`docs/tonglu0602.md`](docs/tonglu0602.md)。

`serve.py` 按 ckpt 的 `policy_config.json` 自动选择 `baseline_vla` vs `sam2_legacy`；100_15 SAM2 部署保持兼容。
