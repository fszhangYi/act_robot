# 训练内存估算公式

本文档说明 `train.py` 在 `EpisodicDataset` + PyTorch `DataLoader` 配置下的 **CPU / GPU 内存占用估算方法**，便于调整 `batch-size`、`num-workers`、相机数等参数时预判资源需求。

适用场景：`EpisodicDataset`（HDF5 多相机图像，每样本取单帧；`action` 按 `chunk_size` 切片）。SAM2 特征模式（`SAM2EpisodicDataset`）张量形状不同，需替换 \(M_{\text{sample}}\) 的定义。

---

## 1. 单条样本图像内存（CPU，float32）

`dataset.py` 中 `EpisodicDataset.__getitem__` 对每个相机读取 **1 帧**，归一化后形状为 `[K, 3, H, W]`：

\[
M_{\text{sample}} = K \times 3 \times H \times W \times 4 \quad \text{(bytes)}
\]

| 符号 | 含义 |
|------|------|
| \(K\) | 相机数（`--camera-names` 个数） |
| \(H, W\) | 图像高、宽（HDF5 中 `/observations/images/{cam}` 的空间尺寸） |
| 4 | float32 字节数 |

**示例**（3 相机，480×640）：

\[
M_{\text{sample}} = 3 \times 3 \times 480 \times 640 \times 4 \approx 11.1\ \text{MB}
\]

---

## 2. 单个 batch 的纯张量大小

\[
M_{\text{batch}} = B \times M_{\text{sample}} + M_{\text{aux}}
\]

| 符号 | 含义 |
|------|------|
| \(B\) | `--batch-size` |
| \(M_{\text{aux}}\) | qpos、action、is_pad 等辅助张量，通常相对 \(M_{\text{sample}}\) 可忽略 |

**示例**（\(B=32\)）：

\[
M_{\text{batch}} \approx 32 \times 11.1\ \text{MB} \approx 0.35\ \text{GB}
\]

---

## 3. DataLoader worker 的 CPU 内存

PyTorch `DataLoader` 默认 `prefetch_factor=2`，每个 worker 最多预取 2 个 batch。实际 RSS 远高于「纯张量」估算，主要因为：

- HDF5 随机读 episode 时的解码缓冲与 OS 页缓存；
- `persistent_workers=True` 时 worker 跨 epoch 常驻，缓存累积；
- Python / PyTorch 分配器碎片。

因此引入经验放大系数 \(\alpha\)：

\[
M_{\text{worker}} \approx \alpha \times \text{prefetch} \times M_{\text{batch}}
\]

| 符号 | 含义 | 典型值 |
|------|------|--------|
| \(\text{prefetch}\) | 预取 batch 数 | 2（PyTorch 默认） |
| \(\alpha_{\text{train}}\) | 训练集 shuffle + 随机 episode 采样 | **25–30** |
| \(\alpha_{\text{val}}\) | 验证集顺序读，缓存压力较小 | **~4** |

worker 数量（与 `train.py` 一致）：

\[
W_{\text{train}} = \text{num\_workers}
\]

\[
W_{\text{val}} = \min(2,\ \text{num\_workers})
\]

**所有 worker 合计：**

\[
M_{\text{workers}} =
W_{\text{train}} \times \alpha_{\text{train}} \times \text{prefetch} \times M_{\text{batch}}
+
W_{\text{val}} \times \alpha_{\text{val}} \times \text{prefetch} \times M_{\text{batch}}
\]

**示例**（`num_workers=4`，\(B=32\)，3×480×640，\(\alpha_{\text{train}}=28\)，\(\alpha_{\text{val}}=4\)）：

\[
M_{\text{workers}} \approx
4 \times 28 \times 2 \times 0.35 +
2 \times 4 \times 2 \times 0.35
\approx 78 + 5.6 \approx 84\ \text{GB}
\]

与实测 worker RSS 合计 ~87 GB 接近。

---

## 4. 总 CPU 内存

\[
M_{\text{CPU}} \approx M_{\text{main}} + M_{\text{workers}} + M_{\text{misc}}
\]

| 项 | 含义 | 典型值 |
|----|------|--------|
| \(M_{\text{main}}\) | 主进程：模型副本、当前 batch、Python 运行时 | 2–5 GB |
| \(M_{\text{workers}}\) | 上节公式 | 见参数 |
| \(M_{\text{misc}}\) | Jupyter、TensorBoard、代理等 | ~1 GB |

**示例：**

\[
M_{\text{CPU}} \approx 4 + 84 + 1 \approx 89\ \text{GB}
\]

操作系统 `buff/cache` 会额外计入「已用内存」，但多数可在压力下回收，不等同于进程 RSS 上限。

---

## 5. GPU 显存（参考）

\[
M_{\text{GPU}} \approx M_{\text{model}} + M_{\text{opt}} + M_{\text{act}} + M_{\text{batch}}^{\text{gpu}}
\]

其中 batch 相关项：

\[
M_{\text{batch}}^{\text{gpu}} = B \times K \times 3 \times H \times W \times 4 \times \beta
\]

| 符号 | 含义 |
|------|------|
| \(\beta\) | forward / backward 中间激活相对输入 batch 的倍数，ACT 常见 **3–8** |
| \(M_{\text{model}}, M_{\text{opt}}, M_{\text{act}}\) | 权重、优化器状态、激活，与 `hidden_dim`、`enc/dec layers` 等相关 |

显存需结合具体模型配置实测；同一数据 batch 下 GPU 占用通常远小于 CPU worker 合计。

---

## 6. 一行速算（改参数时用）

将 \(B, K, H, W\) 代入，\(\text{prefetch}=2\)，并沿用默认 worker 划分（\(W_{\text{train}}=4\)，\(W_{\text{val}}=2\) 当 `num_workers=4`）：

\[
\boxed{
M_{\text{CPU}} \approx
4\ \text{GB}
+
(B \times K \times H \times W \times 4 \times 10^{-9})
\times 2
\times \big( W_{\text{train}} \alpha_{\text{train}} + W_{\text{val}} \alpha_{\text{val}} \big)
}
\]

**默认 `num_workers=4` 时**可简写为：

\[
M_{\text{CPU}} \approx
4\ \text{GB}
+
(B \times K \times H \times W \times 4 \times 10^{-9})
\times 2
\times (4 \alpha_{\text{train}} + 2 \alpha_{\text{val}})
\]

建议 \(\alpha_{\text{train}} \approx 28\)，\(\alpha_{\text{val}} \approx 4\)；数据集更大或 shuffle 更随机时 \(\alpha_{\text{train}}\) 可能更高。

---

## 7. 降内存与公式中变量的对应关系

| 改法 | 公式中的影响 |
|------|----------------|
| 减小 `--num-workers` | \(W_{\text{train}}, W_{\text{val}}\) 下降，\(M_{\text{workers}}\) 近似线性减少 |
| 减小 `--batch-size` | \(M_{\text{batch}}\) 下降，worker 与 GPU batch 项同步减少 |
| `num_workers=0` | \(M_{\text{workers}} \approx 0\)，数据加载成为瓶颈 |
| 关闭 `persistent_workers` | \(\alpha\) 通常变小，但每 epoch 重启 worker 有额外开销 |
| 减少相机数 \(K\) | \(M_{\text{sample}}\) 线性减少 |
| 减小 `--hdf5-cache-size` | 每个 worker 最多同时打开的 episode 文件数（LRU）；默认 16，可避免无界缓存撑满内存 |

---

## 8. HDF5 句柄缓存（`dataset.py`）

`EpisodicDataset` / `SAM2EpisodicDataset` 对每个 DataLoader worker 使用 **有上限的 LRU** 缓存已打开的 `episode_*.hdf5`（`train.py --hdf5-cache-size`，默认 16）。设为 `0` 则每次 `__getitem__` 打开后立即关闭，内存最低、I/O 最高。

---

## 9. 相关代码位置

| 文件 | 说明 |
|------|------|
| `dataset.py` | `EpisodicDataset.__getitem__`：每样本单帧多相机 |
| `train.py` | `DataLoader(..., num_workers=..., persistent_workers=True, pin_memory=True)` |
| `train.py` | 验证 loader：`num_workers=min(2, args.num_workers)` |

---

## 10. 实测校对（参考）

在 AutoDL 754 GB 内存实例上，配置 `--batch-size 32 --num-workers 4 --camera-names chest top wrist_2`（480×640）时：

| 观测项 | 约值 |
|--------|------|
| 单个训练 `pt_data_worker` RSS | 19–22 GB |
| 验证 worker RSS | ~2.7 GB |
| 主进程 RSS | ~4 GB |
| worker 合计 | ~87 GB |
| GPU 显存（主进程） | ~24 GB |

公式估算与实测在同一数量级；精确值仍以 `ps` / `nvidia-smi` 为准。
