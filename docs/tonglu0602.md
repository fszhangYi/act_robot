# tonglu0602 数据集 — 原始 ACT 多相机训练管线

本管线在 `act_robot` 工程内**复用原始 ACT 路径**（不走 SAM2Grasp），用 3 个相机视角 + 绝对位姿动作，搭配 annotation 文件做帧切片。本文档是「上手 → 跑通 → 部署」一站式指南。其他机器上只要按本文档操作即可。

## 1. 数据布局

```
<dataset_root>/
├── raw_data/
│   ├── 99/                         # episode 名（任意数字，名字会成为 HDF5 排序键）
│   │   ├── rgb_chest_0.jpg
│   │   ├── rgb_chest_1.jpg
│   │   ├── ...
│   │   ├── rgb_top_0.jpg
│   │   ├── ...
│   │   ├── rgb_wrist_2_0.jpg
│   │   ├── ...
│   │   ├── (其它相机的 jpg 也可以放着，不被使用)
│   │   └── steps.json
│   ├── 990/
│   └── ...
└── annotation/
    ├── 99.txt                      # 名字与 raw_data 下子目录 1:1 对应
    ├── 990.txt
    └── ...
```

### 各相机原始分辨率（混合）

| 相机 | 原始分辨率 (W×H) | HDF5 存储 (H×W) |
|---|---|---|
| `rgb_wrist_2` | 640 × 480 | 480 × 640（identity resize） |
| `rgb_chest`   | 1280 × 720 | 480 × 640（squish,16:9→4:3,不保 aspect） |
| `rgb_top`     | 1280 × 720 | 480 × 640（同上） |

`convert_episodes.py` 的 `_CAMERA_RESIZE` 对三个相机统一目标 `(480, 640)`,PIL 自动按需缩放;`serve.py` 的 `_CAMERA_RESIZE_SERVE` 镜像同一份配置,所以模型在训练和推理看到的所有相机张量都是同样 `(3, 480, 640)`。客户端发上来的 JPEG 是原始分辨率即可,server 自己缩放。

### `steps.json` 必须包含的字段

```python
steps['observations']['joint_position']      # (T, 6)
steps['observations']['cartesian_position']  # (T, 6) = [x, y, z, rx, ry, rz]
steps['observations']['gripper_position']    # (T,) 或 (T, 1) 或 (1, T)
steps['actions']['gripper_position']         # (T,) 或同 obs 的形状
```

`cartesian_abs` 模式只读 `cartesian_position`；`joint` 模式读 `joint_position`。两个 gripper 字段都会用到。

### Annotation 文件格式

每个 `<episode_name>.txt` 是**空格分隔**的 12 行：

```
1 4       ← 第 1 行第 2 列 = START FRAME（0-indexed 闭区间）
2 96      ← 第 2 行第 2 列 = 中间关键帧（本管线忽略）
3 145     ← 第 3 行第 2 列 = END FRAME（0-indexed 闭区间）
4 -1
5 -1
...
12 3
```

切片规则：

- 只保留 `steps[start : end+1]` 闭区间
- `start` 或 `end` 为 `-1`：跳过该 episode（无效标注）
- 文件缺失：跳过该 episode

转换脚本会打印跳过列表，方便排查。

## 2. 数据转换

```bash
/home/znyyb/miniconda3/envs/anygrasp/bin/python convert_episodes.py \
  --input-dir       /dataset/lizhiwei12/universal_grasp/tonglu0602/raw_data \
  --output-dir      /dataset/lizhiwei12/universal_grasp/tonglu0602_chest_top_wrist2_cart_abs_data \
  --annotation-dir  /dataset/lizhiwei12/universal_grasp/tonglu0602/annotation \
  --camera-names chest top wrist_2 \
  --action-space cartesian_abs --stride 1 \
  --unwrap-rx \
  --train-ratio 0.9 --seed 42
```

### 参数要点

| flag | 含义 |
|---|---|
| `--input-dir` | 必填。`raw_data/` 目录路径 |
| `--output-dir` | 必填。输出 HDF5 + `dataset_info.json` 的目录 |
| `--annotation-dir` | 提供则按 `<ep>.txt` 切片；不提供则用整条 episode（与 100_15 老用法兼容） |
| `--camera-names` | 顺序就是 HDF5 内 `/observations/images/<cam>/` 的写入顺序，**也是 serve.py 期望客户端发送 JPEG 的顺序**。本数据集用 `chest top wrist_2` |
| `--action-space cartesian_abs` | qpos/action 都是 7 维 `[x,y,z,rx,ry,rz,gripper]`，绝对值 |
| `--unwrap-rx` | **强烈推荐**：把 `rx<0` 那簇 `+2π`，避免 ±π 边界双峰。`serve.py` 在推理时自动应用反向偏移，客户端不感知。详见 §5 |
| `--stride 1` | 不做时间下采样（每一帧都留） |
| `--train-ratio / --seed` | shuffle 后取前 90% 作 train，与 100_15 老管线相同 |

### 转换产物

```
<out_root>/tonglu0602_chest_top_wrist2_cart_abs_data/
├── episode_0.hdf5    ← attrs: sim, stride, camera_names, action_space
├── episode_1.hdf5    │
├── ...               │  /observations/qpos          (T', 7) float32
└── dataset_info.json │  /observations/qvel          (T', 7) float32
                     │  /observations/images/chest   (T', 480, 640, 3) uint8 lzf
                     │  /observations/images/top     (T', 480, 640, 3) uint8 lzf
                     │  /observations/images/wrist_2 (T', 480, 640, 3) uint8 lzf
                     └  /action                      (T', 7) float32
```

其中 `T' = end - start + 1`（即 annotation 给出的有效帧数）。

`dataset_info.json` 里会写：

```json
{
  "stride": 1,
  "train_ratio": 0.9, "seed": 42,
  "num_total": N, "num_train": ..., "num_val": ...,
  "train_indices": [...], "val_indices": [...],
  "episode_lengths": [...], "max_episode_len": ...,
  "action_space": "cartesian_abs",
  "camera_names": ["chest", "top", "wrist_2"],
  "rx_unwrapped": true
}
```

### 快速校验

```python
import h5py, json, glob
info = json.load(open('<out_root>/.../dataset_info.json'))
print(f'rx_unwrapped={info["rx_unwrapped"]}  cams={info["camera_names"]}')
f = h5py.File(glob.glob('<out_root>/.../episode_*.hdf5')[0],'r')
print({k: f[f'/observations/images/{k}'].shape for k in info['camera_names']})
print('qpos rx range:', f['/observations/qpos'][:,3].min(), f['/observations/qpos'][:,3].max())
```

期望：
- `rx_unwrapped=True`
- 每相机 shape `(T', 480, 640, 3)`
- `rx range` 若开了 unwrap 应≥ 0；不开则可能跨 0 看到负数

## 3. 训练

### 冒烟验证（确认管线在新机器上跑得通）

```bash
/home/znyyb/miniconda3/envs/anygrasp/bin/python train.py \
  --data-dir <out_root>/tonglu0602_chest_top_wrist2_cart_abs_data \
  --ckpt-dir <out_root>/tonglu0602_chest_top_wrist2_cart_abs_ckpt_smoke \
  --num-epochs 20 --batch-size 4 --chunk-size 10 \
  --lr 1e-5 --kl-weight 10 \
  --action-space cartesian_abs --camera-names chest top wrist_2 \
  --seed 0 --grad-clip 1.0 --num-workers 4
```

预期：`train loss` 在 20 epoch 内显著下降到 ~个位数（KL 项一开始很大、几个 epoch 就缩到接近 0）。落盘三件套：

- `policy_best.ckpt`、`policy_last.ckpt`
- `policy_config.json`（含 `camera_names`, `action_space`, `rx_unwrapped`）
- `dataset_stats.pkl`

### 全量训练（在新机器上跑）

500+ episode 全量时建议：

```bash
python train.py \
  --data-dir <full_data_dir> \
  --ckpt-dir <full_ckpt_dir> \
  --num-epochs 2000 --batch-size 16 --chunk-size 10 \
  --lr 1e-5 --kl-weight 10 \
  --action-space cartesian_abs --camera-names chest top wrist_2 \
  --seed 0 --grad-clip 1.0 --num-workers 12
```

要点：

- `--batch-size 16`：3 相机 ResNet18 + transformer 在 24GB GPU 上比较稳；显存余量大可加到 32
- `--num-workers 12`：3 相机 HDF5 解压在 IO 上吃力，多 workers 必要
- `--num-epochs 2000`：原始 ACT 标准 recipe（lr 1e-5 时步数要给足）
- 没开 `--cosine-lr`：1e-5 恒定 lr + grad-clip 1.0 比 cosine 更稳

监控：

```bash
tail -f <full_ckpt_dir>/train_history.json   # 每 epoch 一行
ls -lh <full_ckpt_dir>/*.ckpt                # 每 200 epoch 一个备份
```

## 4. 推理 / 部署

### 启动 server

```bash
python serve.py \
  --checkpoint <ckpt_dir>/policy_best.ckpt \
  --stats      <ckpt_dir>/dataset_stats.pkl \
  --port 5000
  # 可选: --temporal-agg 或 --always-first
```

server 自动从 `<ckpt_dir>/policy_config.json` 读取：

- `use_sam2_features=false` → 走基线 ACT 多相机路径
- `camera_names` → 决定线协议期望的相机数 + 顺序
- `action_space=cartesian_abs` → next_state 直接是预测值
- `rx_unwrapped=true` → 自动对 qpos[3] 入端 +2π、next_state[3] 出端 -2π

### 线协议（multi-camera baseline）

**每步 client → server**（big-endian）：

```
28B  robot_state   (7 × float32: x, y, z, rx, ry, rz, gripper)
 4B  refresh       (uint32; 1 = reset 当前 episode 状态)
 4B  num_cams      (uint32; 必须等于 len(camera_names))
 for cam in camera_names (顺序与 policy_config 一致):
     4B  jpeg_len  (uint32)
     N B JPEG bytes
```

**server → client**：

```
28B  next_state    (7 × float32 大端，已应用 compose_pose + rx unwrap 反偏移)
```

### Python client 参考

```python
import socket, struct, io
from PIL import Image

CAM_ORDER = ['chest', 'top', 'wrist_2']    # 必须与 policy_config['camera_names'] 一致

def query(sock, robot_state, refresh, cam_images_pil):
    # robot_state: np.ndarray shape (7,) float32; cam_images_pil: dict[str, PIL.Image]
    body  = struct.pack('>7f', *robot_state)
    body += struct.pack('>I', int(refresh))
    body += struct.pack('>I', len(CAM_ORDER))
    for cam in CAM_ORDER:
        buf = io.BytesIO()
        cam_images_pil[cam].save(buf, format='JPEG', quality=95)
        jpeg = buf.getvalue()
        body += struct.pack('>I', len(jpeg)) + jpeg
    sock.sendall(body)
    return struct.unpack('>7f', _recv_exact(sock, 28))

def _recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        pkt = sock.recv(n - len(buf))
        if not pkt: raise ConnectionError()
        buf += pkt
    return buf

# 用法：
sock = socket.create_connection(('<server_host>', 5000))
next_state = query(sock, robot_state, refresh=1, cam_images_pil=...)   # 首帧 refresh=1
for step in range(N_steps):
    next_state = query(sock, get_current_state(), refresh=0, cam_images_pil=capture_all_cams())
    drive_robot(next_state)
```

### 推理冒烟脚本

`scripts/smoke_infer_tonglu.py` 复刻 `_infer_baseline` 路径，把 val 集第一帧 round-trip 一遍：

```bash
python scripts/smoke_infer_tonglu.py \
  --ckpt-dir <ckpt_dir> \
  --data-dir <data_dir>
```

期望输出形如：

```
config: camera_names=['chest', 'top', 'wrist_2']  action_space=cartesian_abs  ...
OK: ACTPolicy built + policy_best.ckpt loaded (no state_dict mismatch)
encoded 3 JPEGs (sizes=[..., ..., ...] bytes)
img_t shape=(1, 3, 3, 480, 640)  (expected [1, 3, 3, 480, 640])

qpos[0]       = [...]
pred chunk[0] = [...]
gt   action[0]= [...]
chunk MAE per-dim = [...]
|chunk[0] pose - qpos pose| = 0.xxxx

=== PASS ===  multi-camera baseline inference path runs end-to-end without errors
```

### 三种 ACT 推理模式

由 `serve.py` 启动 flag 切换：

| flag | 行为 |
|---|---|
| 无（默认） | `chunk_replay`：每 chunk_size 步查一次,顺序回放 chunk。算力省,但 chunk 边界可能跳变 |
| `--temporal-agg` | 每步查一次,对当前覆盖窗口内所有 chunk 做指数加权融合。最平滑 |
| `--always-first` | 每步查一次,只用最新预测的 chunk[0]。无 blending、无 boundary 跳变 |

cart_abs 模式自校正能力强,默认 chunk_replay 即可；如果发现 chunk 边界突变明显,切 `--always-first`。

## 5. rx wrap（强烈推荐对全量数据开 `--unwrap-rx`）

### 问题

`cartesian_abs` 用 Euler XYZ 表达旋转。当腕部接近**竖直向下**时（机械臂抓取场景的常态），`rx` 在 ±π 边界附近会**双峰**——同一物理姿态被同时表达成 `+π` 和 `-π`（差 2π 但物理等价）。模型很难学到一致映射：100_15 cart_abs 实测 **rx RMSE 0.6 rad**,开了 unwrap 后正常 0.02 rad 量级。

### 解决方案（已内置）

转换时加 `--unwrap-rx`：

- `qpos[3] < 0` 那簇统一 `+2π`
- cart_abs 模式同步对 `action[3]` 做同样偏移；cart（SE3 delta）只动 qpos
- `dataset_info.json` 写 `rx_unwrapped: true`
- `train.py` 自动把 flag 串进 `policy_config.json`（基线 + SAM2 两路都包含）
- `serve.py` 在**输入端**对 `qpos[3] < 0` 加 2π、在**输出端**对 `next_state[3] > π` 减 2π
- **客户端始终看到标准 `[-π, π]` 的 rx**,不感知 unwrap 的存在

### 何时开 / 何时关

| 数据特征 | 建议 |
|---|---|
| 全量数据集尚未抽样过 | **直接开**——安全 default,最坏情况只是 rx 轴平移 |
| 已抽样确认 `rx` 始终在 `[0, 2π)` 或始终在 `[-2π, 0]` 单峰 | 可关 |
| `rx` 实际跨越 `0`（机械臂大幅旋转） | **关**——shift 会破坏拓扑 |

### 抽样自检（10 ep 即可）

```python
import h5py, numpy as np, glob
files = sorted(glob.glob('<data_dir>/episode_*.hdf5'))[:10]
rxs = np.concatenate([h5py.File(f,'r')['/observations/qpos'][:,3] for f in files])
print(f'rx: min={rxs.min():.3f}  max={rxs.max():.3f}  '
      f'<0 frac={(rxs<0).mean():.2%}  near±π frac={(np.abs(np.abs(rxs)-np.pi)<0.2).mean():.2%}')
```

判断：
- `near±π frac > 30%` 且 `<0 frac` 有显著比例 → **需要** `--unwrap-rx`
- `<0 frac == 0` 或 `<0 frac == 100%` → 单峰,不需要,但开了也无害
- rx 散布在 0 周围（典型机械臂大旋转） → **关掉**

## 6. 验证清单

转完 / 训完 / 起服务前过一遍：

- [ ] `dataset_info.json` 中 `action_space=='cartesian_abs'`、`camera_names=='chest','top','wrist_2'`、`rx_unwrapped` 与你的预期一致
- [ ] 任一 HDF5：每相机 `(T', 480, 640, 3) uint8`、`/action` 与 `/observations/qpos` 都是 `(T', 7) float32`
- [ ] `policy_config.json` 中 `use_sam2_features=false`、`camera_names` 列表与转换时一致
- [ ] 训练 20 epoch 内 `val loss` 单调下降无 NaN
- [ ] `scripts/smoke_infer_tonglu.py` 跑出 `=== PASS ===`、`img_t shape=(1, N, 3, 480, 640)`
- [ ] 实机闭环：服务端 log 里 `[addr] connected (baseline multi-cam (['chest','top','wrist_2']))`、`num_cams` 不匹配会在终端立刻报

## 7. 把工程拷贝到新机器

### 拷贝清单

```
act_robot/
├── convert_episodes.py        # 数据转换
├── train.py                   # 训练
├── serve.py                   # 推理服务
├── policy.py                  # ACTPolicy / SAM2 policies
├── dataset.py                 # EpisodicDataset / SAM2EpisodicDataset
├── detr/                      # transformer + backbone
├── checkpoints/
│   └── resnet18-f37072fd.pth  # ResNet18 预训练权重（必须本地有,否则 backbone init 会去下）
├── scripts/
│   └── smoke_infer_tonglu.py
├── docs/
│   └── tonglu0602.md          # 本文档
└── CLAUDE.md / README.md      # 可选
```

**不需要拷贝**：`sam2/`（SAM2 仓库）、`sam2_features.py`、`extract_sam2_features.py`、`derive_cart_abs_dataset.py` —— 这些是 SAM2Grasp 路径专用。除非新机器上也要部署 SAM2Grasp,否则可以省下。但 `serve.py` 仍 `from sam2_features import ...`,所以要么把 `sam2_features.py` 一起拷,要么把那个 import 删掉。

### Python 依赖

```bash
pip install torch==2.4.1+cu121 torchvision \
            h5py einops pillow tqdm numpy
```

要点：
- `torch 2.4.1+cu121`：经过验证 conv backward 正常
- 避开 `dexvla` / 老 cuda 11.7 环境（会有 cuDNN 兼容问题）

### 新机器上首次跑通

```bash
# 1. 转数据
python convert_episodes.py \
  --input-dir   <new_machine_path>/raw_data \
  --output-dir  <new_machine_path>/tonglu_cart_abs_data \
  --annotation-dir <new_machine_path>/annotation \
  --camera-names chest top wrist_2 \
  --action-space cartesian_abs --stride 1 --unwrap-rx \
  --train-ratio 0.9 --seed 42

# 2. 冒烟训练（确认 GPU + 依赖 OK）
python train.py \
  --data-dir <new_machine_path>/tonglu_cart_abs_data \
  --ckpt-dir <new_machine_path>/tonglu_cart_abs_ckpt_smoke \
  --num-epochs 20 --batch-size 4 --chunk-size 10 \
  --lr 1e-5 --kl-weight 10 \
  --action-space cartesian_abs --camera-names chest top wrist_2 \
  --seed 0 --grad-clip 1.0 --num-workers 4

# 3. 全量训练（参数同上 §3 全量段）

# 4. 起服务
python serve.py \
  --checkpoint <new_machine_path>/tonglu_cart_abs_ckpt/policy_best.ckpt \
  --stats      <new_machine_path>/tonglu_cart_abs_ckpt/dataset_stats.pkl \
  --port 5000
```

### 故障排查

| 现象 | 可能原因 / 排查 |
|---|---|
| `No <prefix>_*.jpg frames in <ep>` | jpg 文件名不是 `rgb_chest_<N>.jpg` 格式,改名 / 检查 `--camera-names` |
| `dataset_info.json` 中 `num_total=0` | 所有 episode 都被 annotation 跳过；`grep -c "skipped" log` |
| `state_dict mismatch` 加载 ckpt | `policy_config.json` 的 `camera_names`、`state_dim`、`hidden_dim` 必须与训练时一致;别改了配置再加载老 ckpt |
| serve.py 打印 `protocol mismatch: client sent num_cams=...` | 客户端发的相机数 ≠ ckpt 训练时的相机数；对齐顺序和数量 |
| 闭环上机预测乱跳 | 检查 client 是否对 `next_state` 中 `gripper` 维做了反归一化（serve 已 clip 到 [0, 1.13]）；rx_unwrapped 是否两边对齐 |
| GPU 余量小 | `--batch-size 8` 起步;3 相机 ResNet18 输入比单相机大 3× |

## 8. 与 SAM2Grasp 路径的关系

本管线和 SAM2Grasp 路径（`extract_sam2_features.py` + `--use-sam2-features` 训练 + serve.py SAM2 分支）**完全独立**：

- 选择由 `policy_config.json` 的 `use_sam2_features` 自动决定,无需手动切
- SAM2 路径的旧线协议（hard-coded wrist + rear_left + state + refresh + bbox）`serve.py` 仍兼容,100_15 实机部署不受影响
- 多相机基线和 SAM2Grasp 单相机+prompt **不能同时用同一个 ckpt 部署**,因为模型权重和输入形态都不同

如果你要在同一台机器上同时跑两套,起两个 `serve.py` 进程,各自指向自己的 ckpt 目录、监听不同端口。
