# act_robot Level-up 报告

扫描范围：`/root/autodl-tmp/act_robot` 训练 / 数据 / 推理主链路（`train.py`、`policy.py`、`dataset.py`、`convert_episodes.py`、`serve.py`、`sam2_features.py`、文档与脚本）。不含 `.venv`、权重、数据集。

排序原则：**ROI = 对训练正确性或真机成功率的收益 ÷ 改动成本**。同档内先写会直接毁掉实验或部署的问题。

---

## 总览（按 ROI 从高到低）

| 排名 | 项 | 收益 | 成本 | ROI |
|------|----|------|------|-----|
| 1 | Early stopping 永远在第 5 个 epoch 停训 | 极高 | 极低 | ★★★★★ |
| 2 | 基线 ACT 的 pad loss 仍按全元素 `.mean()` | 极高 | 极低 | ★★★★★ |
| 3 | 文档协议 ≠ `serve.py` 真实协议 | 极高 | 低 | ★★★★★ |
| 4 | SAM2 checkpoint 路径写死本机 | 高 | 极低 | ★★★★ |
| 5 | 训练数据 I/O：每步开 HDF5 + pad 到整段 episode | 高 | 中 | ★★★★ |
| 6 | SAM2 离线 F_t 与线上 streaming F_t 分布漂移 | 高 | 中高 | ★★★★ |
| 7 | 相机 16:9 被 squish 成 4:3 | 高 | 中 | ★★★ |
| 8 | `train.sh` / 环境不可复现 | 中高 | 低 | ★★★ |
| 9 | 推理默认把每帧 JPEG 落到磁盘 | 中 | 极低 | ★★★ |
| 10 | 设备写死 `.cuda()`，无 CPU/多卡抽象 | 中 | 低 | ★★★ |
| 11 | SAM2 README 多了 4B `prompt_type` | 中 | 极低 | ★★★ |
| 12 | `wrist_2` 180° 旋转被注释掉 | 中 | 低 | ★★ |
| 13 | 可视化脚本残缺（`vison.py`） | 中 | 低 | ★★ |
| 14 | 训练日志每 batch 打 Data loading | 低中 | 极低 | ★★ |
| 15 | 调试残留与死代码 | 低 | 极低 | ★ |
| 16 | DETR 遗留 hardcode / TODO | 低 | 低 | ★ |

---

## 1. Early stopping 逻辑错误：训练几乎必然停在 epoch 5

**现状**

- `EarlyStoppingCallback(patience=5, threshold=0.002)` 在 `main()` 里**一定会传入** `train()`。
- 每个 epoch 先做 val，若更好则**先更新** `min_val_loss`，再调用 callback：

```python
if epoch_val_loss < min_val_loss:
    min_val_loss = epoch_val_loss
    ...
cb.on_epoch_end(epoch, epoch_val_loss, min_val_loss, policy)
```

- callback 的复位条件是 `val_loss < best_loss - 0.002`。此时 `best_loss` 已经等于本次 `val_loss`（有提升时）或更优的历史值（无提升时）。**该不等式几乎永假，counter 从不复位。**
- 因此无论 val 是否在降，大约 **5 个 epoch 后 `stop_training=True`**，循环在 train 之前 `break`。README 写的 2000 epoch 实际上跑不完。

**为何 ROI 最高**

- 直接决定「模型有没有训起来」。改 10 行就能让后续所有实验有效。
- 这也能解释「val 看起来还行、但 epoch 很少就停了 / best 很早」一类现象。

**建议（半天内）**

1. callback 传入**更新前**的 `best_loss`，或在 callback 内部自己维护 best。
2. `patience` 默认改成 `50~200`，或做成 `--early-stop-patience 0`（0 = 关闭）。ACT 需要上千 epoch，patience=5 即使逻辑正确也过激。
3. 先 val 再决定是否停，应放在**本 epoch 训练之后**；当前是 val → 可能停 → 才 train，第 5 轮连训练都不会跑。

---

## 2. 基线 `ACTPolicy` 仍用含 padding 的 `.mean()` 当 L1

**现状**

SAM2 头已经修过（`ACTSAM2Policy` 用 `mask.sum() * A` 做分母），注释写得很清楚：旧 `.mean()` 会把高 padding 比例的 episode 的 val loss 人为压低，`policy_best.ckpt` 变成早期虚假最优。

基线 `ACTPolicy` **没改**：

```python
l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
```

分子把 pad 位置置零，分母仍是 `B * K * A`（含 pad）。`tonglu0602` 多相机原始 ACT 走的就是这条路径。

**收益**

- 与 README「v1 ckpt 不要用 policy_best」是同一类 bug；不修则 tonglu 的 `policy_best.ckpt` 不可信。
- 改动是把 SAM2 的 `denom` 公式原样拷到 `ACTPolicy`（以及 CVAE 的 L1 已修、基线未修）。

**建议**

- 三处 loss 统一：只对 `~is_pad` 求均值。
- 旧 ckpt 不要按 `policy_best` 选型；用 `policy_epoch_*` 或重训。
- 加一条单测：构造全 pad / 半 pad batch，loss 不应随 pad 比例单调变小。

---

## 3. 文档里的线协议和 `serve.py` 实现不是同一套

这是部署侧最大的「按文档接上就会挂」问题。

| 来源 | 基线多相机协议（声称） | 实际 `_handle_baseline_client` |
|------|------------------------|--------------------------------|
| `README.md` / `serve.py` 文件头 | `28B state + 4B refresh + 4B num_cams + 各相机 JPEG` → 只回 `28B next_state` | **不是这样** |
| 实现 | — | 每帧：`top / chest / wrist2` 三路 JPEG → `text` → `28B state`；**无 refresh、无 num_cams**；回复 `28B + term + reject + text`。新连接 = 新 episode。 |

后果：

- 按 README / 文件头写客户端，字节对不齐，表现为卡死、乱动作、偶发成功。
- `docs/tonglu0602.md` 若仍描述「num_cams 通用协议」，和当前 serve 行为冲突。
- SAM2 路径仍是 100_15 旧协议；基线路径是 `serve_tmp.py` 的 VLA 布局。两套协议靠 `use_sam2_features` 隐式切换，客户端无法从端口看出该发哪种包。

**建议**

1. **以代码为唯一真相**：把 README、`serve.py` 顶部 docstring、`docs/tonglu0602.md` 改成与 `_handle_baseline_client` / `_handle_sam2_client` 逐字节一致。
2. 启动时打印一行 `wire_protocol=baseline_vla|sam2_legacy` 和字段布局。
3. 中期：收成一个带 version 的帧（例如 `magic + proto_ver + payload`），或启动参数 `--protocol`，不要靠 ckpt 隐式分支。
4. `scripts/mock_client_sam2.py` / `smoke_infer_tonglu.py` 做成协议金标准，文档只引用脚本。

成本低、能避免整周现场联调。

---

## 4. SAM2 权重路径写死旧机器

`SAM2StreamingFeatureExtractor` 默认：

```text
/home/znyyb/hww/vla/act_robot/sam2/checkpoints/sam2.1_hiera_small.pt
```

本机、AutoDL、GitHub clone 都会在 `serve.py --use-sam2` 时直接炸。`CLAUDE.md` / 部署文档同样绑死 `anygrasp` 与 `/home/znyyb/...`。

**建议**

- `--sam2-ckpt` CLI + 环境变量 `SAM2_CKPT`，默认相对路径 `checkpoints/sam2.1_hiera_small.pt`。
- 启动时文件不存在则给出 wget 命令（README 里已有 URL）。

---

## 5. DataLoader I/O：每条样本打开 HDF5，并 pad 到 `max_episode_len`

**现状**

- `EpisodicDataset` / `SAM2EpisodicDataset` 每次 `__getitem__` `h5py.File(...)` 打开、读完关闭。
- action 先 pad 到 **整段最长 episode**，再在 policy 里切 `[:num_queries]`。磁盘和内存都在搬用不到的尾部。
- `train.py` **每个 batch** `print(f"Batch {idx}: Data loading took ...")`，训练自己也在怀疑 I/O。

**收益**

- 多 worker + 每步开文件是 ACT 训练变慢的常见原因；变慢会逼人减小 epoch / 乱调 lr，间接伤效果。
- pad 到 `max_episode_len` 让 batch 巨大（图像路径尤其明显），浪费显存，迫使 `batch-size` 降到 4～16。

**建议**

1. worker 内缓存已打开的 HDF5（`worker_init_fn` 里打开，或 `Swmr`）。
2. 只读取 `action[start_ts : start_ts+chunk_size]`，`is_pad` 长度 = `chunk_size`。
3. 删掉每 batch 的 print，改成 TensorBoard 已有的 throughput。
4. `persistent_workers=True`，并 `generator`/`worker_init_fn` 设 seed，保证可复现。

---

## 6. SAM2：训练用离线 `propagate_in_video`，推理用逐帧 streaming，F_t 不一致

代码和 README 已记录：t=0 对齐，t≥1 max abs diff 约 1.7 → 0.2。训练分布 ≠ 真机分布，是闭环「能 offline eval 但不能抓」的高嫌疑项。

**建议（按成本）**

| 方案 | 成本 | 效果 |
|------|------|------|
| 用 `scripts/eval_sam2_offline.py` 对比 streaming vs offline 的 chunk RMSE，先定量 | 低 | 决定要不要修 |
| 推理每步从 frame 0 全量 re-propagate（代码注释已写） | 延迟↑ | 分布对齐 |
| Stage-1 抽取改成与 serve 相同的逐帧 API，使训练 F_t = 线上 F_t | 中 | 根治 |
| 跟踪 SAM2 bf16 / memory encoder | 高 | 上游 |

未定量前不要把「模型不行」和「特征偏移」混为一谈。

---

## 7. 胸/顶相机 16:9 被拉成 4:3（squish）

`chest` / `top`：1280×720 → `(480, 640)`，**不保宽高比**。几何（工件长宽、抓取方位）被压扁，ResNet 看到的空间统计和真机透视不一致。`wrist_2` 640×480 倒是一致。

**建议**

- letterbox 到 640×480（黑边），训练和 serve 同一套；或中心裁 4:3。
- 改分辨率必须重转数据 + 重训，属于「一次付清」。新数据集建议现在就改，避免以后锁死 squish。

---

## 8. 复现材料不完整，`train.sh` 是另一台机器的一次性命令

- 无 `requirements.txt` / `environment.yml`（部署文档里有一串 pip，但仓库根目录没有锁版本）。
- `train.sh` 写死 `/home/ubuntu/act/...`，`lr=5e-6`，无 `--cosine-lr`；README 却说 SAM2 **必须** `--grad-clip 1.0 --cosine-lr`。
- `scan.py` 写死 `/home/ubuntu/tmp/...`。
- Python 入口假定 `anygrasp` conda，和文档「新机器建 `act` 环境」不一致。

别人（或未来的你）按 GitHub README 无法无损复现。

**建议**

- 根目录加 `requirements.txt`（torch 版本用注释标明 cu121）。
- `train.sh` 改成读环境变量 `DATA_DIR`/`CKPT_DIR`，并带上 README 的推荐开关。
- 文档只保留一套环境名。

---

## 9. `serve.py` 每步把 JPEG 和 json 写到 `logs/`

真机 20–30 Hz 时，这是延迟和磁盘抖动来源，也容易把数据盘写满。调试有用，生产有害。

**建议**：默认 `--log-every 0`；需要时 `--log-every 50` 或只记 action/state 不落图。

---

## 10. 训练/推理设备写死 CUDA

`forward_pass` 全 `.cuda()`，`policy.cuda()`。没卡或 CPU 调试直接崩。serve 虽有 `device` 字段，默认仍 cuda。

**建议**：`--device cuda|cpu`，统一 `tensor.to(device, non_blocking=...)`。AutoDL 无卡启动时至少能跑通数据管线。

---

## 11. SAM2 协议：README 比代码多 4 字节

README：

```text
refresh==1 时: 4B prompt_type + 16B bbox
```

`serve.py` 实现：`refresh==1` 只读 **16B bbox**，无 `prompt_type`。

按 README 实现的客户端会把 bbox 读偏，目标锁错。与第 3 条同类，单独列出因为只影响 SAM2 首帧。

**建议**：代码与文档二选一对齐；`prompt_type` 若暂不支持，文档删掉。

---

## 12. `wrist_2` 旋转 180° 被注释

`serve.py` 里 `img.rotate(180)` 被注释，convert 侧未见对称处理。若采集时腕部相机是倒装的，训练图像与真机朝向可能反了。这是「现场偶发、日志看不出来」的问题。

**建议**：在 `policy_config.json` 里显式 `camera_rotations: {"wrist_2": 180}`，convert 与 serve 共用。默认 0，避免静默改图。

---

## 13. `vison.py` 几乎帮不上忙

- 文件名拼写错误（vision）。
- 只画 `data['train']` 的 loss，不画 val，而选型看的是 val。
- x 轴写成 Step，实际是 epoch 列表。
- 每 batch 的 TensorBoard `Loss/train` 才是 step 级，和这个 json 不是一回事。

**建议**：同时画 train/val；标注 best epoch；或直接用 TensorBoard，脚本只做导出。

---

## 14. 训练过程噪音与次要训练动态

- 每个 batch print 数据加载耗时（见第 5 条）。
- 每个 epoch 先对**未更新（或上一轮）**权重做完整 val：epoch 0 的 val 是随机初始化，会写入 `val_history` 并可能成为 early-stop 的基线（在修第 1 条后仍要注意）。
- `ACTPolicy` 每次 forward `transforms.Normalize(...)` 新建对象，无功能性危害，可提到 `__init__`。
- `is_sim = True` 写死，不影响当前 HDF5 路径。
- `convert_episodes.py` 顶层 `a=1` 无意义，应删。

---

## 15. 仓库卫生

| 文件 | 问题 |
|------|------|
| `serve_tmp.py` | 旧 VLA 服务器，大量 TODO，与现 serve 协议纠缠，易被误启动 |
| `scan.py` | 本机探路脚本，硬编码路径，不应在主目录 |
| `scripts/scp_with_password.py` | 密码来自环境变量，可接受；不要把密码写进仓库（当前没有） |
| `vison.py` | 见上 |
| 根目录无测试 | 协议、loss mask、SE(3) compose 最值得测 |

GitHub 上 `dev`/`main` 已含这些文件。清理是低风险、低收益，但能减少「跑错入口」。

---

## 16. 上游 DETR 遗留

`detr/models/detr_vae.py` 中 `CNNMLP` 仍 `state_dim = 14`。当前主路径用 `args.state_dim`（7），CNNMLP 若被启用会 silently 错维。`latent_dim = 32 # TODO tune` 等是原版 ACT 注释，不是当前瓶颈。

有空把 `state_dim` 全部改为从 config 注入即可。

---

## 已做得好、不必再作为 level-up 重点的部分

这些已经修过或设计合理，报告里不当作待办：

- 训练统计只用 train split（`get_norm_stats`）。
- 推理只做 `/255`，ImageNet norm 在 `ACTPolicy` 内做一次（README 记录的双重归一化已修）。
- SE(3) 相对动作 + `verify.py` Euler wrap 检查。
- `detr/main.py` `parse_args([])`，不再被 CLI 污染。
- SAM2 L2/L1 的 pad mask、`--use-cvae`、`--cumulative-loss-weight`、`--action-repr delta` 是针对 chunk[0] OOD 的正确方向。
- `--resume-from` + `optimizer.pt` + cosine fast-forward。
- `.gitignore` 已排除 `.venv`、`logs`、`runs`、`*.pth`。

---

## 建议落地顺序（两周）

**第 1 天（不训模型也能完成）**

1. 修 early stopping（或直接默认关闭）。
2. 基线 ACT L1 改成与 SAM2 相同的 mask 均值。
3. 文档/协议与 `serve.py` 逐字节对齐；启动 banner 打印协议。
4. SAM2 ckpt 路径可配置；删 `a=1`。

**第 2–3 天**

5. Dataset 只读 `chunk_size`、缓存 HDF5、去掉 per-batch print。
6. serve 默认不落盘每一帧。
7. `requirements.txt` + 可移植 `train.sh`。

**有数据/真机窗口时**

8. 定量 SAM2 F_t 漂移；决定 re-extract 还是 re-propagate。
9. 新数据改 letterbox；旧 ckpt 不要混用。
10. `camera_rotations` 写进 config，现场确认腕部朝向。

---

## 怎么验证这次扫描没有漏掉「会炸」的点

优先跑这三条，成本低、能钉死 1/2/3 条：

```bash
# A. 确认 early-stop：开 20 epoch 的干跑，看是否总在 epoch 5 结束
# B. 构造 is_pad 比例不同的 batch，打印 ACTPolicy 的 l1（修前应随 pad 变小）
# C. 用 hexdump / 现有 mock client 对一次真实 serve 端口，对照 README 字节数
```

扫描未跑训练、未连真机；第 6、7、12 条需要一次 offline eval 或现场确认才能结案。
