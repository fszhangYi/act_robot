# 原始数据处理教程：过滤 → 转换

两步入库流程：先用 **gripper 质量规则**筛 episode，再只转换白名单里的样本为 ACT 训练用 HDF5。

---

## 0. 目录准备

原始数据按 **数字 episode 目录**组织，每个目录含 `steps.json` 和相机图片：

```
data/
├── raw/                    # 原始 episode：0/ 1/ 2/ …，各有 steps.json
│   ├── 0/
│   ├── 1/
│   └── ...
└── annotation/
    └── annotation/         # 可选：每 episode 一个 <id>.txt，裁切起止帧
        ├── 0.txt
        └── 1.txt
```

`annotation` 文件格式（12 行）：**第 1 行第 2 列** = 起始帧，**第 3 行第 2 列** = 结束帧（0 起算、含首尾）。`start` 或 `end` 为 `-1` 表示无效，convert 时会跳过。

---

## 1. 质量过滤

脚本：`scripts/check_episode_quality.py`

依据 `steps.json` 里 `observations.gripper_position` 判断：

| 规则 | 说明 |
|------|------|
| 开合轨迹 | 起始≈闭合 → 中间张开到峰值 → 结束≈闭合 |
| 传感器卡死 | 连续 ≥20 帧为固定 sentinel 值则判失败 |

```bash
cd /path/to/act_robot

python scripts/check_episode_quality.py \
  --input-dir data/raw \
  --annotation-dir data/annotation/annotation \
  --write-pass-json data/quality_pass.json \
  --write-fail-list data/quality_fail.txt \
  --output-json data/quality_report.json
```

**输出：**

| 文件 | 内容 |
|------|------|
| `quality_pass.json` | 通过样本的 **episode 序号** JSON 数组，如 `[1, 2, 3, 10, …]` |
| `quality_fail.txt` | 失败 episode 名，每行一个 |
| `quality_report.json` | 每条 episode 的 pass/fail 与原因 |

终端会打印 `pass` / `fail` 数量；失败原因示例：`start_not_zero`、`no_meaningful_open`、`sentinel_stuck`。

> 有 annotation 时，规则只在标注帧段内评估 gripper；与 convert 裁切范围一致。

---

## 2. 转换为 HDF5

脚本：`convert_episodes.py`

只转换 `quality_pass.json` 白名单中的 episode（`--filter-json`）：

```bash
python convert_episodes.py \
  --input-dir data/raw \
  --output-dir data/converted \
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

**常用参数：**

| 参数 | 说明 |
|------|------|
| `--action-space` | `joint`（推荐）/ `cartesian_abs` / `cartesian`（legacy） |
| `--stride` | 时间下采样，每 N 帧取 1 帧 |
| `--unwrap-rx` | 笛卡尔模式下把负 rx 平移 +2π，避免 ±π 跳变 |
| `--num-workers` | 并行 episode 数；IO 为主时可 4–8 |
| `--skip-existing` | 中断后续跑，跳过已生成的有效 HDF5 |

**输出目录 `data/converted/`：**

```
converted/
├── episode_0.hdf5
├── episode_1.hdf5
├── ...
└── dataset_info.json    # train/val 划分、stride、filter_json 等元数据
```

`train.py` 直接读该目录：`--data-dir data/converted`。

---

## 3. 一键串联示例

```bash
# Step 1 — 过滤
python scripts/check_episode_quality.py \
  --input-dir data/raw \
  --annotation-dir data/annotation/annotation \
  --write-pass-json data/quality_pass.json \
  --output-json data/quality_report.json

# Step 2 — 转换（仅白名单）
python convert_episodes.py \
  --input-dir data/raw \
  --output-dir data/converted \
  --annotation-dir data/annotation/annotation \
  --camera-names chest top wrist_2 \
  --action-space cartesian_abs \
  --stride 1 --unwrap-rx \
  --filter-json data/quality_pass.json \
  --num-workers 8
```

---

## 4. 检查是否成功

```bash
# 白名单条数
python -c "import json; print(len(json.load(open('data/quality_pass.json'))))"

# HDF5 数量应 ≤ 白名单（无 annotation 的会被跳过）
ls data/converted/episode_*.hdf5 | wc -l

# 划分信息
python -c "import json; d=json.load(open('data/converted/dataset_info.json')); \
print(d['num_total'], 'train', d['num_train'], 'val', d['num_val'])"
```

---

## 5. 下一步

```bash
python train.py \
  --data-dir data/converted \
  --ckpt-dir data/ckpt_xxx \
  --action-space cartesian_abs \
  --camera-names chest top wrist_2
```

`--action-space` 与 `--camera-names` 须与 convert 时一致。

---

## 附：过滤规则调参

仅在数据特性特殊时修改：

```bash
python scripts/check_episode_quality.py \
  --input-dir data/raw \
  --min-peak 0.08 \              # 提高张开幅度要求
  --sentinel-min-run 30 \          # 更严格卡死判定
  --write-pass-json data/quality_pass.json
```

完整参数见 `python scripts/check_episode_quality.py --help`。
