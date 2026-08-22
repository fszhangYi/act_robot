import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
from detr.models.act_sam2 import build_act_sam2
from detr.models.act_sam2_cvae import build_act_sam2_cvae

# Keys returned by policy forward() for throughput monitoring — not losses.
THROUGHPUT_METRIC_KEYS = frozenset({
    'num_tokens',
    'num_encoder_tokens',
    'num_decoder_tokens',
    'num_cvae_encoder_tokens',
})


def count_main_transformer_tokens(
    spatial_h: int,
    spatial_w: int,
    num_queries: int,
    *,
    encoder_prefix: int = 2,
) -> tuple[int, int, int]:
    """Per-sample token counts aligned with detr.models.transformer.Transformer.

    Encoder sequence = encoder_prefix (latent + proprio) + H*W spatial tokens.
    Decoder sequence = num_queries action-chunk slots.
    Returns (encoder_tokens, decoder_tokens, total_tokens).
    """
    enc = encoder_prefix + spatial_h * spatial_w
    dec = num_queries
    return enc, dec, enc + dec


def _batch_throughput_metrics(
    batch_size: int,
    spatial_h: int,
    spatial_w: int,
    num_queries: int,
    *,
    encoder_prefix: int = 2,
    cvae_encoder_tokens: int = 0,
) -> dict[str, int]:
    enc, dec, main_total = count_main_transformer_tokens(
        spatial_h, spatial_w, num_queries, encoder_prefix=encoder_prefix,
    )
    total = main_total + cvae_encoder_tokens
    out = {
        'num_encoder_tokens': batch_size * enc,
        'num_decoder_tokens': batch_size * dec,
        'num_tokens': batch_size * total,
    }
    if cvae_encoder_tokens:
        out['num_cvae_encoder_tokens'] = batch_size * cvae_encoder_tokens
    return out


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            # 只对非 pad 位置求均值；旧 .mean() 分母含 pad，高 padding 比例的
            # episode 会人为压低 val loss，导致 policy_best.ckpt 变成早期虚假最优。
            mask = (~is_pad).unsqueeze(-1).float()
            denom = mask.sum() * actions.size(-1)
            l1 = (all_l1 * mask).sum() / denom.clamp(min=1.0)
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            # Throughput: encoder sees concat cameras along width → spatial_w *= num_cam
            B = image.shape[0]
            num_cameras = image.shape[1]
            H_feat = image.shape[-2] // 32
            W_feat = image.shape[-1] // 32
            loss_dict.update(_batch_throughput_metrics(
                B, H_feat, W_feat * num_cameras, self.model.num_queries,
            ))

            return loss_dict
        else:  # inference
            a_hat, _, (_, _) = self.model(qpos, image, env_state)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else:
            a_hat = self.model(qpos, image, env_state)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class ACTSAM2Policy(nn.Module):
    """Deterministic ACT policy fed by frozen-SAM2 features (SAM2Grasp).

    No CVAE, no ImageNet normalisation — the input is the pre-computed SAM2
    feature map `sam2_feat: [B, 256, 64, 64]` and the robot proprioception
    `qpos: [B, state_dim]`. Training loss is plain L2 over the chunk.
    """

    def __init__(self, config: dict):
        super().__init__()
        model, optimizer = build_act_sam2(config)
        self.model = model
        self.optimizer = optimizer
        self.num_queries = model.num_queries

    def __call__(self, qpos, sam2_feat, actions=None, is_pad=None):
        a_hat = self.model(sam2_feat, qpos)  # [B, num_queries, action_dim]
        if actions is None:
            return a_hat
        actions = actions[:, : self.num_queries]
        is_pad = is_pad[:, : self.num_queries]
        mask = (~is_pad).unsqueeze(-1).float()                  # [B, K, 1]
        all_l2 = (actions - a_hat) ** 2                          # [B, K, A]
        # Mean over UN-PADDED scalar predictions only. The earlier .mean()
        # divided by total elements (incl. padded), which artificially shrank
        # val loss whenever an episode had a high padding ratio and made the
        # apparent "best val" hit an early-but-misleading minimum.
        denom = mask.sum() * actions.size(-1)
        l2 = (all_l2 * mask).sum() / denom.clamp(min=1.0)
        B = sam2_feat.shape[0]
        pool = self.model.pool_size
        if pool:
            h = w = int(pool)
        else:
            h, w = int(sam2_feat.shape[-2]), int(sam2_feat.shape[-1])
        out = {'l2': l2, 'loss': l2}
        out.update(_batch_throughput_metrics(B, h, w, self.num_queries))
        return out

    def configure_optimizers(self):
        return self.optimizer


class ACTSAM2CVAEPolicy(nn.Module):
    """Paper-aligned ACT policy with CVAE, fed by frozen-SAM2 features (SAM2Grasp v4).

    Same loss formulation as the original ACTPolicy:
        loss = L1(predicted_action, gt_action; mask=~is_pad) + kl_weight * KL(mu, logvar)
    The KL term is the implicit regulariser that constrains predictions to lie
    within the training-action manifold even on novel (qpos, F_t) inputs —
    exactly the missing piece that caused v1/v2/v3's chunk[0] OOD jumps.
    """

    def __init__(self, config: dict):
        super().__init__()
        model, optimizer = build_act_sam2_cvae(config)
        self.model = model
        self.optimizer = optimizer
        self.num_queries = model.num_queries
        self.kl_weight = config.get('kl_weight', 10.0)
        # Cumulative-trajectory consistency loss (v10): for SE(3)-delta actions
        # (action_space=cartesian), the per-step L1 lets the model systematically
        # under-predict each tiny delta (mode collapse toward 0). Supervising the
        # CUMULATIVE sum of deltas makes under-prediction accumulate over the
        # chunk, so it is penalised much harder. Applies to the pose dims (0-5)
        # only; gripper (dim 6) is an absolute target, not a delta, so excluded.
        self.cumulative_loss_weight = config.get('cumulative_loss_weight', 0.0)
        print(f'ACTSAM2CVAEPolicy kl_weight={self.kl_weight}  '
              f'cumulative_loss_weight={self.cumulative_loss_weight}')

    def __call__(self, qpos, sam2_feat, actions=None, is_pad=None):
        if actions is None:
            # Inference: latent z = 0 (training-distribution centre)
            a_hat, _, _ = self.model(sam2_feat, qpos)
            return a_hat
        # Training
        actions = actions[:, : self.num_queries]
        is_pad = is_pad[:, : self.num_queries]
        a_hat, _, (mu, logvar) = self.model(sam2_feat, qpos, actions, is_pad)
        total_kld, _, _ = kl_divergence(mu, logvar)
        mask = (~is_pad).unsqueeze(-1).float()
        all_l1 = F.l1_loss(actions, a_hat, reduction='none')                   # [B, K, A]
        denom = mask.sum() * actions.size(-1)
        l1 = (all_l1 * mask).sum() / denom.clamp(min=1.0)

        loss = l1 + self.kl_weight * total_kld[0]
        out = {'l1': l1, 'kl': total_kld[0]}

        if self.cumulative_loss_weight > 0:
            # cumsum over the chunk on pose dims (0-5); zero padded steps first
            pose_hat = (a_hat[..., :6] * mask)
            pose_gt = (actions[..., :6] * mask)
            cum_hat = torch.cumsum(pose_hat, dim=1)
            cum_gt = torch.cumsum(pose_gt, dim=1)
            cum_l1_all = F.l1_loss(cum_gt, cum_hat, reduction='none')           # [B, K, 6]
            cum_denom = mask.sum() * 6
            cum_l1 = (cum_l1_all * mask).sum() / cum_denom.clamp(min=1.0)
            loss = loss + self.cumulative_loss_weight * cum_l1
            out['cum_l1'] = cum_l1

        out['loss'] = loss
        B = sam2_feat.shape[0]
        pool = self.model.pool_size
        if pool:
            h = w = int(pool)
        else:
            h, w = int(sam2_feat.shape[-2]), int(sam2_feat.shape[-1])
        # CVAE action encoder: [CLS, qpos, action_0..action_{K-1}]
        cvae_enc = 2 + self.num_queries
        out.update(_batch_throughput_metrics(
            B, h, w, self.num_queries, cvae_encoder_tokens=cvae_enc,
        ))
        return out

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld
