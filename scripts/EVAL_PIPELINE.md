# 原始数据 → embody_model_eval 测评数据全流程

从 `data/raw` 到在 [embody_model_eval](https://github.com/fszhangYi/embody_model_eval) 中可视化对比 GT / 模型预测，按顺序执行下列脚本。

**Python 环境**（训练 / 推理 / 转换统一）：

```bash
/home/znyyb/miniconda3/envs/anygrasp/bin/python
```

本机若无该路径，可用已装 torch 的等价环境；下文命令以 `python` 代指。

---

## 流程总览

```
data/raw
    │  ① check_episode_quality.py
    ▼
quality_pass.json
    │  ② convert_episodes.py
    ▼
data/converted/*.hdf5
    │  ③ train.py
    ▼
data/ckpt_*/
    │  ④ infer_all_quality_pass.py  (或 infer_from_raw.py 单条)
    ▼
data/infer_*/
    │  ⑤ infer_to_embody_eval.py          → 每 episode 1 个 JSON（chunk 第 0 步）
    │  ⑤ infer_to_embody_eval_chunk.py    → 每观测帧 1 个 JSON（完整 10 步 chunk）
    ▼
embody_model_eval/data/<suite>/
```

更细的「过滤 + HDF5 转换」说明见 [`docs/raw_data_pipeline.md`](../docs/raw_data_pipeline.md)。

---

## 0. 目录约定

```
data/
├── raw/<episode>/steps.json     # 原始关节、图像、gripper
├── annotation/annotation/<id>.txt
├── quality_pass.json            # ① 输出
├── converted/                   # ② 输出 HDF5
├── ckpt_cam100_15_cart_abs_v1/  # ③ 输出 checkpoint
└── infer_cam100_15_cart_abs_v1/ # ④ 输出离线推理 JSON
```

embody 侧（独立仓库，默认路径）：

```
/root/autodl-tmp/embody_model_eval/data/
├── ec616_act/              # ⑤a：每 episode 一步对比
└── ec616_act_chunk/        # ⑤b：每帧 10 步 chunk 对比
    └── episode_3/
        ├── frame_0008.json
        └── ...
```

---

## ① 质量过滤

**脚本：** `scripts/check_episode_quality.py`

依据 gripper 开合轨迹与传感器卡死规则筛 episode，写出白名单。

```bash
cd /path/to/act_robot

python scripts/check_episode_quality.py \
  --input-dir data/raw \
  --annotation-dir data/annotation/annotation \
  --write-pass-json data/quality_pass.json \
  --write-fail-list data/quality_fail.txt \
  --output-json data/quality_report.json
```

| 输出 | 说明 |
|------|------|
| `quality_pass.json` | 通过 episode 序号数组，如 `[1, 3, 10, …]` |
| `quality_fail.txt` | 失败 episode 列表 |
| `quality_report.json` | 逐条 pass/fail 原因 |

---

## ② 转 HDF5 训练集

**脚本：** `convert_episodes.py`（仓库根目录）

只转换白名单内 episode；`qpos` / `action` 与 `--action-space` 一致。

```bash
python convert_episodes.py \
  --input-dir data/raw \
  --output-dir data/converted_cam100_15_cart_abs \
  --annotation-dir data/annotation/annotation \
  --camera-names chest top wrist_2 \
  --action-space cartesian_abs \
  --stride 1 \
  --unwrap-rx \
  --train-ratio 0.9 \
  --seed 42 \
  --num-workers 8 \
  --filter-json data/quality_pass.json
```

| 参数 | 说明 |
|------|------|
| `--action-space` | `joint` / `cartesian_abs` / `cartesian`（legacy） |
| `--unwrap-rx` | 笛卡尔模式下避免 rx 在 ±π 跳变 |
| `--filter-json` | 仅转换 `quality_pass.json` 中的 episode |

输出：`episode_*.hdf5` + `dataset_info.json`（train/val 划分）。

---

## ③ 训练 ACT

**脚本：** `train.py`（仓库根目录）

`--action-space` 与 `--camera-names` 须与 convert 一致。

```bash
python train.py \
  --data-dir data/converted_cam100_15_cart_abs \
  --ckpt-dir data/ckpt_cam100_15_cart_abs_v1 \
  --action-space cartesian_abs \
  --camera-names chest top wrist_2 \
  --chunk-size 10 \
  --num-epochs 2000 \
  --batch-size 64
```

输出：`policy_best.ckpt`、`dataset_stats.pkl` 等。

---

## ④ 离线推理

在 **原始 `data/raw`** 上跑模型（与线上一致：同一套 qpos / 图像预处理），得到每帧 `pred_chunk[10]` 与 `gt_chunk[10]`。

### 单 episode 调试

**脚本：** `scripts/infer_from_raw.py`

```bash
python scripts/infer_from_raw.py \
  --episode 3 \
  --raw-dir data/raw \
  --annotation-dir data/annotation/annotation \
  --ckpt-dir data/ckpt_cam100_15_cart_abs_v1 \
  --action-space cartesian_abs \
  --unwrap-rx \
  --camera-names chest top wrist_2 \
  --output data/infer_ep3_debug.json
```

### 批量（白名单全集）

**脚本：** `scripts/infer_all_quality_pass.py`

```bash
python scripts/infer_all_quality_pass.py \
  --filter-json data/quality_pass.json \
  --raw-dir data/raw \
  --annotation-dir data/annotation/annotation \
  --ckpt-dir data/ckpt_cam100_15_cart_abs_v1 \
  --output-dir data/infer_cam100_15_cart_abs_v1 \
  --action-space cartesian_abs \
  --unwrap-rx \
  --camera-names chest top wrist_2
```

| 输出 | 说明 |
|------|------|
| `episode_<id>.json` | `summary` + `frames[]`；每帧含 `raw_frame`、`pred_chunk`、`gt_chunk`、`is_pad` |
| `manifest.json` | 批量统计、失败列表 |

---

## ⑤ 转为 embody_model_eval 对比 JSON

推理结果是 **笛卡尔绝对位姿**（`cartesian_abs`）或关节空间时，需经 **EC616 IK**（`scripts/ec616_ik.py` → `ec616_kin.ik_flange`）转为 8 维关节向量（6 轴度 + 镜像夹爪），才能在 embody 的 URDF 里显示。

默认 TCP 工具偏移：法兰 Z 方向 **+0.18 m**（`--tcp-tool-z-m` 可调）。

### ⑤a 每 episode 一步对比（chunk 第 0 步）

**脚本：** `scripts/infer_to_embody_eval.py`

每个观测帧只取 `pred_chunk[0]`，与 raw 下一帧 GT 对比；**1 个 infer JSON → 1 个 embody JSON**。

```bash
# 单 episode
python scripts/infer_to_embody_eval.py \
  --infer-json data/infer_cam100_15_cart_abs_v1/episode_3.json \
  --raw-dir data/raw \
  --output /root/autodl-tmp/embody_model_eval/data/ec616_act/episode_3.json

# 批量
python scripts/infer_to_embody_eval.py \
  --infer-dir data/infer_cam100_15_cart_abs_v1 \
  --raw-dir data/raw \
  --output-dir /root/autodl-tmp/embody_model_eval/data/ec616_act \
  --suite ec616_act \
  --refresh-index
```

| 字段 | 含义 |
|------|------|
| `frames[i].current` | 观测帧 `ri` 的关节状态 |
| `frames[i].next_gt` | raw 帧 `ri+1` 的 GT 目标 |
| `frames[i].next_pred` | 模型 `pred_chunk[0]` 经 IK 后的目标 |

`--refresh-index` 会更新 `embody_model_eval/data/index.json`，供 Hub 加载 `ec616_act` 套件。

### ⑤b 每观测帧完整 chunk 对比（10 步）

**脚本：** `scripts/infer_to_embody_eval_chunk.py`

每个观测时刻输出 **独立 JSON**，内含该次推理的全部有效 chunk 步（通常 10 步；episode 末尾按 `is_pad` 截断）。

```bash
# 单 episode → 多文件：episode_<id>/frame_<raw_frame>.json
python scripts/infer_to_embody_eval_chunk.py \
  --infer-json data/infer_cam100_15_cart_abs_v1/episode_3.json \
  --raw-dir data/raw \
  --output-dir /root/autodl-tmp/embody_model_eval/data/ec616_act_chunk

# 批量全部 episode
python scripts/infer_to_embody_eval_chunk.py \
  --infer-dir data/infer_cam100_15_cart_abs_v1 \
  --raw-dir data/raw \
  --output-dir /root/autodl-tmp/embody_model_eval/data/ec616_act_chunk \
  --suite ec616_act_chunk
```

| 字段 | 含义 |
|------|------|
| `meta.obs_raw_frame` | 本次推理对应的 raw 帧号 |
| `meta.chunk_steps` | 有效步数（≤ 10） |
| `frames[j].current` | 第 `j` 步对应状态 `joint[ri+j]` |
| `frames[j].next_gt` | `gt_chunk[j]` 经 IK |
| `frames[j].next_pred` | `pred_chunk[j]` 经 IK |

**与 ⑤a 的区别：**

| | ⑤a `infer_to_embody_eval` | ⑤b `infer_to_embody_eval_chunk` |
|--|---------------------------|----------------------------------|
| 输出粒度 | 1 JSON / episode | 1 JSON / 观测帧 |
| 每文件 `frames[]` 长度 | = 推理帧数（如 180） | = chunk 步数（如 10） |
| 对比范围 | 每帧只看 chunk 第 0 步 | 完整 action chunk |

> **Hub 索引：** `refresh_data_index.py` 目前只扫描套件目录下**平铺**的 `*.json`。`ec616_act_chunk/episode_3/frame_*.json` 子目录结构需在 viewer 中**直接打开单文件**，或后续扩展索引逻辑。

---

## ⑥ 在 embody_model_eval 中查看

1. 进入 embody 仓库，启动静态服务或前端（见其 README）。
2. **⑤a**：在 Eval 页选择套件 `ec616_act`，加载 `episode_<id>.json`。
3. **⑤b**：打开 `ec616_act_chunk/episode_<id>/frame_<raw_frame>.json` 查看该时刻 10 步 rollout 对比。

JSON 需满足：`meta.robot = "ec616"`，且 `frames[]` 含 `current` / `next_gt` / `next_pred`（8 维关节，度）。

---

## 一键串联示例（cam100_15 cartesian_abs）

命名与 [`CLAUDE.md`](../CLAUDE.md) 约定一致：`cam100_15` = stride 1 + 三相机 + `cartesian_abs`。

```bash
REPO=/path/to/act_robot
EMBODY=/root/autodl-tmp/embody_model_eval
PY=python   # 或 /home/znyyb/miniconda3/envs/anygrasp/bin/python

cd "$REPO"

# ① 过滤
$PY scripts/check_episode_quality.py \
  --input-dir data/raw \
  --annotation-dir data/annotation/annotation \
  --write-pass-json data/quality_pass.json

# ② HDF5
$PY convert_episodes.py \
  --input-dir data/raw \
  --output-dir data/converted_cam100_15_cart_abs \
  --annotation-dir data/annotation/annotation \
  --camera-names chest top wrist_2 \
  --action-space cartesian_abs --stride 1 --unwrap-rx \
  --filter-json data/quality_pass.json --num-workers 8

# ③ 训练（按需调整 epoch 等）
$PY train.py \
  --data-dir data/converted_cam100_15_cart_abs \
  --ckpt-dir data/ckpt_cam100_15_cart_abs_v1 \
  --action-space cartesian_abs \
  --camera-names chest top wrist_2

# ④ 离线推理
$PY scripts/infer_all_quality_pass.py \
  --filter-json data/quality_pass.json \
  --raw-dir data/raw \
  --annotation-dir data/annotation/annotation \
  --ckpt-dir data/ckpt_cam100_15_cart_abs_v1 \
  --output-dir data/infer_cam100_15_cart_abs_v1 \
  --action-space cartesian_abs --unwrap-rx

# ⑤a embody（每 episode 一步）
$PY scripts/infer_to_embody_eval.py \
  --infer-dir data/infer_cam100_15_cart_abs_v1 \
  --raw-dir data/raw \
  --output-dir "$EMBODY/data/ec616_act" \
  --suite ec616_act --refresh-index

# ⑤b embody（每帧完整 chunk，示例只转 episode 3）
$PY scripts/infer_to_embody_eval_chunk.py \
  --infer-json data/infer_cam100_15_cart_abs_v1/episode_3.json \
  --raw-dir data/raw \
  --output-dir "$EMBODY/data/ec616_act_chunk"
```

---

## 脚本索引

| 步骤 | 脚本 | 作用 |
|------|------|------|
| ① | `scripts/check_episode_quality.py` | gripper 质量过滤 → 白名单 |
| ② | `convert_episodes.py` | raw → HDF5 |
| ③ | `train.py` | 训练 ACT |
| ④ | `scripts/infer_from_raw.py` | 单 episode 离线推理 |
| ④ | `scripts/infer_all_quality_pass.py` | 批量离线推理 |
| ⑤a | `scripts/infer_to_embody_eval.py` | infer → embody（chunk 第 0 步 / episode） |
| ⑤b | `scripts/infer_to_embody_eval_chunk.py` | infer → embody（完整 chunk / 观测帧） |
| — | `scripts/ec616_ik.py` | 笛卡尔 TCP → 关节 IK（被 ⑤ 调用） |

各脚本详细参数：`python <script> --help`。
