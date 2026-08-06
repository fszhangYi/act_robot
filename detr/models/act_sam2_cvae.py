"""ACT-with-CVAE head fed by frozen-SAM2 features (SAM2Grasp v4, paper-aligned).

Compared to ACTSAM2 (v1/v2/v3 deterministic):
  * Restores the CVAE encoder (cls + action seq → mu/logvar) used by the
    original ACT, providing implicit regularisation that the deterministic
    version lacks. This is the standard "ACT architecture" the SAM2Grasp paper
    inherits — Section III.B "we adopt the powerful ACT architecture".
  * Loss is L1 over masked actions + KL divergence, matching ACTPolicy.

Compared to DETRVAE (original ACT):
  * Image backbone (ResNet18) is replaced by a single 1x1 conv over the
    pre-computed SAM2 feature map.
  * Optional AdaptiveAvgPool2d on F_t (pool_size argument) to balance
    qpos:F_t token ratio.

Reuses detr.models.transformer.Transformer, detr.models.position_encoding,
and the detr_vae.reparametrize / get_sinusoid_encoding_table helpers.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .detr_vae import get_sinusoid_encoding_table, reparametrize
from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer, TransformerEncoder, TransformerEncoderLayer


def _build_cvae_encoder(hidden_dim: int, nheads: int, dim_feedforward: int,
                        enc_layers: int, dropout: float, pre_norm: bool) -> TransformerEncoder:
    """The CVAE encoder. Mirrors detr_vae.build_encoder for consistency."""
    encoder_layer = TransformerEncoderLayer(
        d_model=hidden_dim, nhead=nheads, dim_feedforward=dim_feedforward,
        dropout=dropout, activation='relu', normalize_before=pre_norm,
    )
    encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
    return TransformerEncoder(encoder_layer, enc_layers, encoder_norm)


class ACTSAM2CVAE(nn.Module):
    """ACT-with-CVAE policy fed by frozen SAM2 features.

    Inputs:
        sam2_feat: [B, sam2_feat_dim, Hf, Wf]
        qpos:      [B, state_dim]
        actions:   [B, K, action_dim]   (training only)
        is_pad:    [B, K] bool          (training only)
    Returns:
        a_hat, is_pad_hat, (mu, logvar)
    """

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
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_queries = num_queries
        self.pool_size = pool_size
        self.latent_dim = latent_dim

        # --- F_t spatial pool + projection ---
        self.spatial_pool: nn.Module = (
            nn.AdaptiveAvgPool2d((pool_size, pool_size)) if pool_size else nn.Identity()
        )
        self.input_proj = nn.Conv2d(sam2_feat_dim, hidden_dim, kernel_size=1)
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        self.pos_embed_2d = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # --- Main transformer (encoder + decoder for action chunk prediction) ---
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

        # --- CVAE encoder (action-sequence → latent) ---
        self.cvae_encoder = _build_cvae_encoder(
            hidden_dim, nheads, dim_feedforward, enc_layers, dropout, pre_norm,
        )
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2)
        self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)
        # Position embedding for CVAE encoder input: [CLS, qpos, action_0, ..., action_{K-1}]
        self.register_buffer(
            'cvae_pos_table',
            get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim),
        )

        # --- Decoder side ---
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        # Prefix position embeddings for [latent_token, proprio_token] in main encoder
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(
        self,
        sam2_feat: torch.Tensor,
        qpos: torch.Tensor,
        actions: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
    ):
        bs = qpos.size(0)
        is_training = actions is not None

        # --- CVAE encoder: predict latent z from action sequence ---
        if is_training:
            action_embed = self.encoder_action_proj(actions)           # [B, K, hidden]
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)    # [B, 1, hidden]
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # [B, 1, hidden]
            cvae_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1)  # [B, K+2, hidden]
            cvae_input = cvae_input.permute(1, 0, 2)                   # [K+2, B, hidden]
            cls_qpos_pad = torch.full((bs, 2), False, device=qpos.device)
            cvae_is_pad = torch.cat([cls_qpos_pad, is_pad], dim=1)     # [B, K+2]
            pos_embed = self.cvae_pos_table.clone().detach().permute(1, 0, 2)  # [K+2, 1, hidden]
            cvae_out = self.cvae_encoder(cvae_input, pos=pos_embed, src_key_padding_mask=cvae_is_pad)
            cvae_out = cvae_out[0]                                     # take CLS token
            latent_info = self.latent_proj(cvae_out)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32,
                                        device=qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # --- Visual + proprio for main encoder ---
        sam2_feat = self.spatial_pool(sam2_feat)
        src = self.input_proj(sam2_feat)                                # [B, hidden, Hf, Wf]
        pos = self.pos_embed_2d(src)                                    # [B, hidden, Hf, Wf]
        proprio_input = self.input_proj_robot_state(qpos)               # [B, hidden]

        # --- Main transformer: encoder gets [latent, proprio] + spatial tokens ---
        hs = self.transformer(
            src, None, self.query_embed.weight, pos,
            latent_input=latent_input,
            proprio_input=proprio_input,
            additional_pos_embed=self.additional_pos_embed.weight,
        )[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, (mu, logvar)


def build_act_sam2_cvae(config: dict) -> tuple[ACTSAM2CVAE, torch.optim.Optimizer]:
    model = ACTSAM2CVAE(
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
        latent_dim=config.get('latent_dim', 32),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 1e-4),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'ACTSAM2CVAE trainable params: {n_params/1e6:.2f}M  (latent_dim={model.latent_dim})')
    return model, optimizer
