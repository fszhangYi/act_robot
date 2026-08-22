# TensorBoard 吞吐量监视说明

本文档记录 `train.py` / `policy.py` 中训练吞吐量（throughput）指标的**定义、历史问题与本次修改**。

---

## 1. 背景：为什么要监视吞吐量

训练循环在 TensorBoard 中记录两类吞吐指标（滑动窗口，默认每 **20 个 batch** 或 epoch 末一批落盘）：

| TB 标量 | 含义 |
|---------|------|
| `throughput/sample_per_sec` | 每秒处理的训练样本数（DataLoader 吐出的 batch 内样本之和） |
| `throughput/token_per_sec` | 每秒处理的「模型 token」总数（见下文定义） |

用途：

- 对比 `batch-size`、`num-workers`、磁盘 IO 等配置对**端到端训练速度**的影响；
- 与 `Loss/train` 联看，判断变慢是算力瓶颈还是数据加载瓶颈。

计时方式：窗口内 `time.perf_counter()` 墙钟时间，包含 forward、backward、`optimizer.step()`、grad clip；**不含** validation。

---

## 2. 修改前的问题

### 2.1 `ACTPolicy`（ResNet + 多相机）— token 少算约 3 倍

旧代码（`policy.py`）：

```python
H_feat = image.shape[-2] // 32
W_feat = image.shape[-1] // 32
num_visual_tokens = H_feat * W_feat          # 只算了单相机
num_action_tokens = self.model.num_queries
loss_dict['num_tokens'] = B * (num_action_tokens + num_visual_tokens)
```

实际数据流（`detr_vae.DETRVAE`）：

- 输入 `image` 形状为 `[B, num_cam, C, H, W]`（例如 3 相机：chest / top / wrist_2）；
- 各相机独立过 ResNet18，特征在 **width 维拼接** 后送入 transformer encoder；
- encoder 序列长度 = **2**（latent + proprio 前缀）+ **H_feat × (W_feat × num_cam)**；
- `num_queries` 属于 **decoder** 槽位，不应与单相机 spatial token 简单相加混为一谈。

以 `480×640`、3 相机、`chunk_size=10` 为例：

| 项目 | 旧计数 / 样本 | 实际 / 样本 |
|------|----------------|-------------|
| 视觉 spatial | 15×20 = 300 | 15×60 = **900** |
| encoder 前缀 | 0 | **2** |
| decoder queries | +10（混在 total 里） | 10（单独一路） |
| **合计（旧 total）** | **310** | encoder **902** + decoder **10** |

因此旧版 `token_per_sec` 在 3 相机 ACT 上**系统性偏低**，且与 transformer 真实序列不对齐。

### 2.2 `ACTSAM2Policy` / `ACTSAM2CVAEPolicy` — 几乎未计数

SAM2 路径的 policy **不返回** `num_tokens`。`train.py` 回退为：

```python
B * policy.model.num_queries   # 仅 chunk 长度，例如 8×10=80
```

而 SAM2 主 encoder 实际还有 `64×64`（或 `pool_size²`）spatial token + 2 前缀；CVAE 变体另有 action-sequence encoder（长度 `2 + num_queries`）。旧指标**严重低估**计算量。

### 2.3 `sample_per_sec` — 一直正确

`samples_since_log / elapsed` 逻辑未改；仍可作为最可靠的吞吐对比指标。

### 2.4 附带修复：`val_summary` 过滤条件

原代码 `for d in epoch_dicts if d != 'num_tokens'` 将 dict 与字符串比较，条件恒真，属于无效过滤。已改为统一的 `THROUGHPUT_METRIC_KEYS` 集合。

---

## 3. 修改后的 token 定义

与 `detr.models.transformer.Transformer` 的 4D 路径一致：

```
encoder_tokens_per_sample = encoder_prefix + spatial_h * spatial_w
decoder_tokens_per_sample = num_queries
```

- **`encoder_prefix`**：固定为 **2**（latent + proprio，与 `transformer.py` 中 `addition_input` 一致）。
- **主 ACT（ResNet）**：
  - `spatial_h = H // 32`
  - `spatial_w = (W // 32) * num_cameras`（多相机 fold 进 width）
- **SAM2**：
  - 若设置 `--pool-size N`：`spatial_h = spatial_w = N`
  - 否则取 `sam2_feat` 的空间尺寸（通常 64×64）
- **SAM2 CVAE 额外**：`cvae_encoder_tokens = 2 + num_queries`（`[CLS, qpos, action_0..K-1]`），仅训练 forward 计入。

每个 batch 上报（乘 batch size `B`）：

| 字段 | 说明 |
|------|------|
| `num_encoder_tokens` | 主 transformer encoder 序列 token 数 × B |
| `num_decoder_tokens` | decoder query 数 × B |
| `num_cvae_encoder_tokens` | （仅 CVAE）action encoder 序列 × B |
| `num_tokens` | 以上之和 × B（TB 总吞吐用） |

共享实现：`policy.count_main_transformer_tokens()`、`_batch_throughput_metrics()`。

---

## 4. 代码改动清单

### 4.1 `policy.py`

| 改动 | 说明 |
|------|------|
| `THROUGHPUT_METRIC_KEYS` | 集中声明非 loss 的吞吐字段，供 `train.py` 过滤 |
| `count_main_transformer_tokens()` | 单样本 encoder/decoder/total 计数 |
| `_batch_throughput_metrics()` | 乘 batch size，可选 CVAE encoder |
| `ACTPolicy` | `spatial_w *= num_cameras`；拆分 encoder/decoder |
| `ACTSAM2Policy` | 新增吞吐字段（按 pool 后 spatial 尺寸） |
| `ACTSAM2CVAEPolicy` | 同上 + `num_cvae_encoder_tokens` |

### 4.2 `train.py`

| 改动 | 说明 |
|------|------|
| 导入 `THROUGHPUT_METRIC_KEYS` | 统一过滤 batch_dict / val_summary |
| 滑动窗口累加 | `encoder_tokens_since_log`、`decoder_tokens_since_log`、`cvae_tokens_since_log` |
| TensorBoard 新标量 | `throughput/encoder_token_per_sec`、`throughput/decoder_token_per_sec`、`throughput/cvae_encoder_token_per_sec` |
| fallback | 仅当 policy 完全不返回 `num_tokens` 时，回退 `B * num_queries`（如 CNNMLP） |

---

## 5. TensorBoard 如何解读

当前 run（`ckpt_dir/runs/`）中可见：

```
throughput/sample_per_sec          # 首选：跨配置对比训练速度
throughput/token_per_sec           # 总 token 吞吐（新定义，可与旧 run 数值不可比）
throughput/encoder_token_per_sec   # encoder 部分
throughput/decoder_token_per_sec   # decoder 部分
throughput/cvae_encoder_token_per_sec   # 仅 SAM2 CVAE 训练时有值
```

**注意**：

1. **本次修改前后的 `token_per_sec` 不可直接对比**（定义已修正）。
2. `token_per_sec` 是序列长度代理，**不等于** FLOPs；ResNet backbone 计算未单独计入 token。
3. 多相机 ACT 下，encoder token 随相机数线性增长，总 token 上升但 `sample_per_sec` 可能下降属正常。
4. 验证阶段不记录吞吐标量。

---

## 6. 数值示例（当前 tonglu / cartesian_abs 配置）

假设：`batch_size=8`，三相机 `480×640`，`chunk_size=10`，ResNet ACT：

```
spatial_h = 15, spatial_w = 20 × 3 = 60
encoder_per_sample = 2 + 15×60 = 902
decoder_per_sample = 10
total_per_sample   = 912
num_tokens per batch = 8 × 912 = 7296
```

若 `sample_per_sec ≈ 1.5`，则 `token_per_sec ≈ 1.5 × 912 ≈ 1368`（量级参考，非实测）。

---

## 7. 相关文件

- `policy.py` — token 计数实现
- `train.py` — 滑动窗口计时与 TB 写入
- `detr/models/detr_vae.py` — 多相机 concat 与 transformer 调用
- `detr/models/transformer.py` — encoder 序列构造（+2 前缀）
- `detr/models/act_sam2.py` / `act_sam2_cvae.py` — SAM2 spatial 与 CVAE encoder

---

*文档版本：2026-08-22，对应 throughput 监视修正提交。*
