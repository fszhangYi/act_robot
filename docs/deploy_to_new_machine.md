# 把 act_robot 部署到另一台机器

本文是「源机器打包 → 拷贝 → 新机器跑通」的一站式操作手册,目标读者是直接在新机器上执行的人(可能是你自己,或新机器上的另一个 Claude Code)。

新机器上的 Claude Code 推荐先读完这一篇,再读 [`docs/tonglu0602.md`](tonglu0602.md)(细节参数 / wire protocol / 故障排查)。

## 拷贝什么 / 不拷贝什么

| 项 | 拷贝 | 原因 |
|---|---|---|
| `convert_episodes.py` / `train.py` / `serve.py` / `policy.py` / `dataset.py` | ✅ | 核心代码 |
| `detr/` | ✅ | transformer + backbone + 各模型 |
| `checkpoints/resnet18-f37072fd.pth` | ✅ | ResNet18 预训练权重,~45MB,没它 backbone init 会去 torchvision 远程下 |
| `scripts/smoke_infer_tonglu.py` | ✅ | 推理冒烟脚本 |
| `docs/` | ✅ | 本文档 + tonglu0602 指南 |
| `README.md` / `CLAUDE.md` | ✅ | 总览 |
| `sam2_features.py` / `extract_sam2_features.py` / `derive_cart_abs_dataset.py` | ✅(顺手) | tonglu0602 基线不用,但顶层 import 不依赖 sam2 包,留着无成本 |
| `sam2/`(SAM2 仓库) | ❌ | ~大,且 tonglu0602 基线不需要 |
| `checkpoints/sam2.1_hiera_small.pt`(176MB) | ❌ | SAM2 模型权重,基线不用 |
| `logs/` | ❌ | 每机一份,不可移植 |
| `__pycache__/` / `.git/` | ❌ | 临时 |
| 任何 ckpt 目录 (`*_ckpt*` 在 /media/) | ❌ | 数据集 + 权重每机重新生成 |

## 源机器:打包

```bash
cd /home/znyyb/hww/vla
tar czf /tmp/act_robot_tonglu.tar.gz \
  --exclude='act_robot/sam2' \
  --exclude='act_robot/logs' \
  --exclude='*__pycache__*' \
  --exclude='*.pyc' \
  --exclude='act_robot/.git' \
  --exclude='act_robot/scripts/_*' \
  --exclude='act_robot/checkpoints/sam2.1_hiera_small.pt' \
  act_robot
ls -lh /tmp/act_robot_tonglu.tar.gz       # 应该 ~43MB
```

## 源机器 → 新机器:scp

```bash
# 替换 <user> / <new_host> / [<port>]
scp /tmp/act_robot_tonglu.tar.gz <user>@<new_host>:~/
# 或:
scp -P <port> /tmp/act_robot_tonglu.tar.gz <user>@<new_host>:~/
```

## 新机器:解包 + 环境

```bash
ssh <user>@<new_host>
mkdir -p ~/projects && cd ~/projects
tar xzf ~/act_robot_tonglu.tar.gz
cd act_robot

# Python 3.10 + CUDA 12.1 + torch 2.4.1 是经过验证的组合
conda create -n act python=3.10 -y
conda activate act
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install h5py einops pillow tqdm numpy

# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.cudnn.version())"
# 期望:2.4.1+cu121 True 90100
```

如果新机器是 CUDA 11.8,改 `+cu118`、`--index-url cu118`;如果是 CUDA 12.4,用 `+cu124`、`--index-url cu124`。

## 新机器:数据布局

```
/dataset/lizhiwei12/universal_grasp/tonglu0602/
├── raw_data/
│   ├── 99/        (steps.json + rgb_chest_*.jpg + rgb_top_*.jpg + rgb_wrist_2_*.jpg)
│   ├── 990/
│   └── ...
└── annotation/
    ├── 99.txt
    └── ...
```

每个 `annotation/<ep>.txt` 是 12 行空格分隔,第 1 行第 2 列 = 起始帧、第 3 行第 2 列 = 结束帧(0-indexed 闭区间)。详见 [`tonglu0602.md` §1](tonglu0602.md)。

## 新机器:跑通五步曲

### Step 1 — 转数据

```bash
cd ~/projects/act_robot
python convert_episodes.py \
  --input-dir  /dataset/lizhiwei12/universal_grasp/tonglu0602/raw_data \
  --output-dir /dataset/lizhiwei12/universal_grasp/tonglu0602_cart_abs_data \
  --annotation-dir /dataset/lizhiwei12/universal_grasp/tonglu0602/annotation \
  --camera-names chest top wrist_2 \
  --action-space cartesian_abs --stride 1 --unwrap-rx \
  --train-ratio 0.9 --seed 42
```

校验:`cat /dataset/lizhiwei12/universal_grasp/tonglu0602_cart_abs_data/dataset_info.json` 看 `rx_unwrapped: true`、`camera_names: ["chest", "top", "wrist_2"]`、`num_total > 0`。

### Step 2 — 冒烟训练(必须先做,验证 GPU + 依赖)

```bash
python train.py \
  --data-dir  /dataset/lizhiwei12/universal_grasp/tonglu0602_cart_abs_data \
  --ckpt-dir  /dataset/lizhiwei12/universal_grasp/tonglu0602_cart_abs_ckpt_smoke \
  --num-epochs 20 --batch-size 4 --chunk-size 10 \
  --action-space cartesian_abs --camera-names chest top wrist_2 \
  --lr 1e-5 --kl-weight 10 --grad-clip 1.0 --num-workers 4
```

校验:训练打印 `train=xx val=xx` 一直下降,落盘 `policy_best.ckpt / policy_config.json / dataset_stats.pkl / train_history.json`。

### Step 3 — 推理冒烟

```bash
python scripts/smoke_infer_tonglu.py \
  --ckpt-dir /dataset/lizhiwei12/universal_grasp/tonglu0602_cart_abs_ckpt_smoke \
  --data-dir /dataset/lizhiwei12/universal_grasp/tonglu0602_cart_abs_data
```

校验:打印 `img_t shape=(1, 3, 3, 480, 640)` 和 `=== PASS ===`。

### Step 4 — 全量训练

```bash
python train.py \
  --data-dir  /dataset/lizhiwei12/universal_grasp/tonglu0602_cart_abs_data \
  --ckpt-dir  /dataset/lizhiwei12/universal_grasp/tonglu0602_cart_abs_ckpt_full \
  --num-epochs 2000 --batch-size 16 --chunk-size 10 \
  --action-space cartesian_abs --camera-names chest top wrist_2 \
  --lr 1e-5 --kl-weight 10 --grad-clip 1.0 --num-workers 12
```

校验:`policy_best.ckpt` 持续被新 best 覆盖,`train_history.json` 每 100 epoch 打印,val loss 收敛到 ~0.05 以下。

### Step 5 — 推理服务

```bash
python serve.py \
  --checkpoint /dataset/lizhiwei12/universal_grasp/tonglu0602_cart_abs_ckpt_full/policy_best.ckpt \
  --stats      /dataset/lizhiwei12/universal_grasp/tonglu0602_cart_abs_ckpt_full/dataset_stats.pkl \
  --port 5000
  # 可选: --temporal-agg 或 --always-first
```

校验:启动日志看到 `mode=baseline ACT  cams=['chest', 'top', 'wrist_2']  rx_unwrapped=True`、`Listening on 0.0.0.0:5000`。

线协议详见 [`tonglu0602.md` §4](tonglu0602.md);客户端骨架(Python)也在那里。

## 常见坑

| 现象 | 原因 / 解决 |
|---|---|
| `torch.cuda.is_available() == False` | nvidia driver / CUDA 不匹配,先 `nvidia-smi` 确认 |
| `cuDNN error` 在 backward | torch 与 cuDNN 版本不兼容(`dexvla` 类老环境常见);用 conda 重建按上面装 torch 2.4.1+cu121 |
| `No <prefix>_*.jpg frames in <ep>` | jpg 文件名格式不对,或 `--camera-names` 拼写错;参考 [`tonglu0602.md` §1](tonglu0602.md) |
| `dataset_info.json` 中 `num_total=0` | 所有 episode 都被 annotation 跳过(`-1`);看打印的 skip 列表 |
| `state_dict mismatch` 加载 ckpt | `policy_config.json` 必须和训练时一致,不要改了 `--camera-names` / `--chunk-size` 再 load 老 ckpt |
| serve.py 打印 `camera mapping failed` / 客户端卡死 | 基线协议是 top→chest→wrist2 JPEG + text + 28B state（无 refresh/num_cams），回复还含 term/reject/text；对齐字节布局 |
| 训练 OOM | `--batch-size` 减半;3 相机 ResNet18 + 480×640 比单相机吃 3 倍 |

## 验证清单

部署完一并过一遍:

- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- [ ] `python -c "import sys; sys.path.insert(0,'.'); from policy import ACTPolicy"` 无 ImportError
- [ ] Step 1 完成,`dataset_info.json` 字段齐全
- [ ] Step 2 训练 20 epoch 内无 NaN、loss 下降
- [ ] Step 3 冒烟脚本打印 `=== PASS ===`
- [ ] Step 5 serve.py 启动日志相机数 / unwrap / mode 都对
- [ ] 实机闭环:发一帧,server 返回 28 字节 next_state,机械臂能动且稳定

## 完成之后建议保留的文件

```
~/projects/act_robot/                              # 整个工程
/dataset/lizhiwei12/universal_grasp/tonglu0602/   # 原始数据
/dataset/lizhiwei12/universal_grasp/tonglu0602_cart_abs_data/      # 转换后 HDF5(可重新生成,但慢)
/dataset/lizhiwei12/universal_grasp/tonglu0602_cart_abs_ckpt_full/ # 最终 ckpt(无法重新生成除非重新训练)
```

最重要的是 `*_ckpt_full/` 三件套:`policy_best.ckpt` + `dataset_stats.pkl` + `policy_config.json`。备份这三个就够任何时候重启服务。
