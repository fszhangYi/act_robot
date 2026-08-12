"""Deterministic ACT head fed by frozen-SAM2 features (SAM2Grasp reproduction).

Differences from the original CVAE ACT (detr_vae.DETRVAE):
  * No image backbone — input is pre-computed F_t [B, 256, 64, 64] from SAM2.
  * No CVAE — no action-sequence encoder, no mu/logvar/latent sample.
  * Loss is plain L2 over predicted action chunks (handled by policy wrapper).
  * Single proprio token still feeds the encoder alongside the spatial F_t map.

Reuses detr.models.transformer.Transformer and detr.models.position_encoding.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer


class ACTSAM2(nn.Module):
    """ACT policy that takes a frozen-SAM2 feature map + qpos and outputs a chunk."""

    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 7,
        num_queries: int = 10,
        sam2_feat_dim: int = 256,
        hidden_dim: int = 256,
        nheads: int = 8,
        enc_layers: int = 4,
        dec_layers: int = 7,
        dim_feedforward: int = 3200,
        dropout: float = 0.1,
        pre_norm: bool = False,
        pool_size: int | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_queries = num_queries
        self.pool_size = pool_size

        # Optional spatial pooling on the SAM2 feature map before the transformer.
        # SAM2 outputs 64x64 = 4096 spatial tokens which drowns the single
        # proprio (qpos) token in encoder attention. Pooling to e.g. 16x16
        # restores a qpos:F_t ratio comparable to the original ACT (which used
        # ResNet18 → ~15x20 ≈ 300 tokens).
        self.spatial_pool: nn.Module = (
            nn.AdaptiveAvgPool2d((pool_size, pool_size)) if pool_size else nn.Identity()
        )

        self.input_proj = nn.Conv2d(sam2_feat_dim, hidden_dim, kernel_size=1)
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        self.pos_embed_2d = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # Existing Transformer.forward expects 2 prefix tokens (latent + proprio).
        # We've dropped CVAE, so the latent slot becomes a learned padding token.
        self.latent_pad = nn.Parameter(torch.zeros(1, hidden_dim))
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=True,
        )

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, sam2_feat: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        """sam2_feat: [B, 256, Hf, Wf]   qpos: [B, state_dim]
        Returns: [B, num_queries, action_dim]
        """
        bs = qpos.size(0)
        sam2_feat = self.spatial_pool(sam2_feat)    # [B, 256, pool, pool] or unchanged
        src = self.input_proj(sam2_feat)            # [B, hidden, Hf, Wf]
        pos = self.pos_embed_2d(src)                # [B, hidden, Hf, Wf]
        proprio = self.input_proj_robot_state(qpos)  # [B, hidden]
        latent_pad = self.latent_pad.expand(bs, -1)  # [B, hidden]

        # transformer.py:49 selects the 4D path (src has H,W) and prepends 2 prefix tokens.
        hs = self.transformer(
            src,
            None,
            self.query_embed.weight,
            pos,
            latent_input=latent_pad,
            proprio_input=proprio,
            additional_pos_embed=self.additional_pos_embed.weight,
        )[0]   # match existing detr_vae.py convention; shape [B, num_queries, hidden]

        return self.action_head(hs)


def build_act_sam2(config: dict) -> tuple[ACTSAM2, torch.optim.Optimizer]:
    """Construct the model + AdamW optimizer from a flat config dict.

    Recognised keys: state_dim, action_dim, num_queries, sam2_feat_dim, hidden_dim,
    nheads, enc_layers, dec_layers, dim_feedforward, dropout, pre_norm, lr,
    weight_decay.
    """
    model = ACTSAM2(
        state_dim=config.get('state_dim', 7),
        action_dim=config.get('action_dim', config.get('state_dim', 7)),
        num_queries=config['num_queries'],
        sam2_feat_dim=config.get('sam2_feat_dim', 256),
        hidden_dim=config.get('hidden_dim', 256),
        nheads=config.get('nheads', 8),
        enc_layers=config.get('enc_layers', 4),
        dec_layers=config.get('dec_layers', 7),
        dim_feedforward=config.get('dim_feedforward', 3200),
        dropout=config.get('dropout', 0.1),
        pre_norm=config.get('pre_norm', False),
        pool_size=config.get('pool_size', None),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 1e-4),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'ACTSAM2 trainable params: {n_params/1e6:.2f}M')
    return model, optimizer
