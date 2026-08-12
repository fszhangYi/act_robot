# SAM2 Checkpoint Links

当前仓库不再本地下载权重，只保留推荐下载链接与配置映射。

## 推荐优先级

1. `sam2.1_hiera_small.pt`
   - 先跑通流程，速度和显存更友好
2. `sam2.1_hiera_base_plus.pt`
   - 更适合追求效果

## 下载链接

- `sam2.1_hiera_small.pt`
  - https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
- `sam2.1_hiera_base_plus.pt`
  - https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
- `sam2.1_hiera_large.pt`
  - https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

## 配置文件对应关系

- `sam2.1_hiera_small.pt`
  - config: `configs/sam2.1/sam2.1_hiera_s.yaml`
- `sam2.1_hiera_base_plus.pt`
  - config: `configs/sam2.1/sam2.1_hiera_b+.yaml`
- `sam2.1_hiera_large.pt`
  - config: `configs/sam2.1/sam2.1_hiera_l.yaml`

这些配置文件位于本地仓库：

- `third_party/sam2/sam2/configs/sam2.1/`
