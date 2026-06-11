#!/usr/bin/env python3
"""
Grasp Pose Diffusion Model v2 — V6 Compatible

变更 (vs v1):
  - FEAT_DIM:  512  (V6 SA4 global_feat 实际输出)
  - 条件输入:  global_feat(512) + position(3) = 515D
  - 去噪网络:  Hidden 512, 输入维度 6+128+512+3 = 649D
  - 采样 API:  sample() 返回 (rotation_6d, approach_dir, wrist_pos)

坐标系:
  所有向量均在 object canonical frame (undo mesh_prerotation)
  TCP 位置  = position (executed_panda_hand_at_close, 去旋转后)
  手腕位置  = position - approach_dir × TCP_OFFSET
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

TCP_OFFSET = 0.105  # Franka 手腕到指尖中心距离 (m)
GRIPPER_HW = 0.030  # 夹爪半宽 (m)


# ─────────────────────────────────────────────────────────────
# 旋转工具
# ─────────────────────────────────────────────────────────────

def rotation_to_6d(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) → (...,6): 取前两列 flatten"""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rotation_from_6d(r6d: torch.Tensor) -> torch.Tensor:
    """(...,6) → (...,3,3): Gram-Schmidt 正交化"""
    c1 = F.normalize(r6d[..., :3], dim=-1)
    c2 = r6d[..., 3:6]
    c2 = F.normalize(c2 - (c2 * c1).sum(-1, keepdim=True) * c1, dim=-1)
    c3 = torch.cross(c1, c2, dim=-1)
    return torch.stack([c1, c2, c3], dim=-1)


def get_approach_dir(r6d: torch.Tensor) -> torch.Tensor:
    """从 6D rotation 提取 approach_dir（Z 轴）"""
    return rotation_from_6d(r6d)[..., :, 2]


def get_finger_dir(r6d: torch.Tensor) -> torch.Tensor:
    """从 6D rotation 提取 finger_dir（X 轴）"""
    return rotation_from_6d(r6d)[..., :, 0]


def wrist_from_pose(position: torch.Tensor, r6d: torch.Tensor) -> torch.Tensor:
    """手腕位置 = TCP - approach_dir * TCP_OFFSET"""
    return position - get_approach_dir(r6d) * TCP_OFFSET


# ─────────────────────────────────────────────────────────────
# 时间步嵌入
# ─────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ─────────────────────────────────────────────────────────────
# 去噪网络 v2 (V6-compatible)
# ─────────────────────────────────────────────────────────────

class GraspDenoiserV2(nn.Module):
    """
    条件 MLP 去噪网络 v2

    输入维度:
      rotation_6d (6) + t_emb (128) + global_feat (512) + position (3)
      = 6 + 128 + 512 + 3 = 649D

    输出: 预测噪声 ε (6D)
    """
    POSE_DIM  = 6
    T_EMB_DIM = 128
    FEAT_DIM  = 512    # V6 SA4 global feat 实际输出维度
    POS_DIM   = 3      # TCP position in canonical frame

    COND_DIM  = FEAT_DIM + POS_DIM   # 515

    def __init__(self, hidden: int = 512):
        super().__init__()
        in_dim = self.POSE_DIM + self.T_EMB_DIM + self.COND_DIM  # 649

        self.t_emb = SinusoidalEmbedding(self.T_EMB_DIM)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.SiLU(),
            nn.Linear(hidden // 4, self.POSE_DIM),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                global_feat: torch.Tensor,
                position: torch.Tensor) -> torch.Tensor:
        """
        x_t         : (B, 6)    noisy rotation_6d
        t           : (B,)      timestep
        global_feat : (B, 1024) V6 frozen feature
        position    : (B, 3)    TCP position (canonical frame)
        → (B, 6) predicted noise
        """
        t_emb = self.t_emb(t)                                        # (B,128)
        cond  = torch.cat([global_feat, position], dim=-1)           # (B,515)
        h     = torch.cat([x_t, t_emb, cond], dim=-1)               # (B,649)
        return self.net(h)


# ─────────────────────────────────────────────────────────────
# DDPM + DDIM v2
# ─────────────────────────────────────────────────────────────

class GraspDiffusionV2(nn.Module):
    """
    DDPM over 6D rotation，以 (V6_global_feat + tcp_position) 为条件。

    训练:
        loss, x0_pred = model.training_loss(rot_6d_gt, global_feat, position)

    推理:
        rot_6d = model.sample(global_feat, position, n_samples=10)
        → rotation  = rotation_from_6d(rot_6d)
        → TCP 位置  = position (外部输入，从 heatmap 采样)
        → 手腕位置  = wrist_from_pose(position, rot_6d)
    """

    def __init__(self, T: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end:   float = 0.02,
                 hidden:     int   = 512):
        super().__init__()
        self.T = T
        self.denoiser = GraspDenoiserV2(hidden=hidden)

        betas      = torch.linspace(beta_start, beta_end, T)
        alphas     = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas',      betas)
        self.register_buffer('alphas',     alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('sqrt_ab',    alphas_bar.sqrt())
        self.register_buffer('sqrt_1m_ab', (1 - alphas_bar).sqrt())

    def q_sample(self, x0, t, noise):
        sa  = self.sqrt_ab[t].unsqueeze(-1)
        s1m = self.sqrt_1m_ab[t].unsqueeze(-1)
        return sa * x0 + s1m * noise

    def training_loss(self,
                      rot_6d_gt:   torch.Tensor,
                      global_feat: torch.Tensor,
                      position:    torch.Tensor,
                      return_x0:   bool = False):
        """
        rot_6d_gt   : (B, 6)   已归一化的 GT rotation 6D
        global_feat : (B, 1024) V6 frozen feature (预缓存)
        position    : (B, 3)   TCP position (canonical)
        return_x0   : 若 True，额外返回 x0_pred 供 affordance loss 使用
        """
        B = rot_6d_gt.shape[0]
        t     = torch.randint(0, self.T, (B,), device=rot_6d_gt.device)
        noise = torch.randn_like(rot_6d_gt)
        x_t   = self.q_sample(rot_6d_gt, t, noise)
        pred  = self.denoiser(x_t, t, global_feat, position)
        loss  = F.mse_loss(pred, noise)

        x0_pred = None
        if return_x0:
            ab      = self.alphas_bar[t].unsqueeze(-1)
            x0_pred = (x_t - (1 - ab).sqrt() * pred) / (ab.sqrt() + 1e-8)

        return loss, x0_pred

    @torch.no_grad()
    def sample(self,
               global_feat: torch.Tensor,
               position:    torch.Tensor,
               n_samples:   int = 10,
               ddim_steps:  int = 50) -> torch.Tensor:
        """
        global_feat : (1, 1024) 或 (B, 1024)
        position    : (1, 3)    或 (B, 3)
        → (n_samples, 6) rotation_6d
        """
        device = global_feat.device
        if global_feat.shape[0] == 1:
            global_feat = global_feat.expand(n_samples, -1)
            position    = position.expand(n_samples, -1)

        step_ids = torch.linspace(self.T - 1, 0, ddim_steps, dtype=torch.long)
        x = torch.randn(n_samples, GraspDenoiserV2.POSE_DIM, device=device)

        for i, t_cur in enumerate(step_ids):
            t_batch    = torch.full((n_samples,), t_cur, dtype=torch.long, device=device)
            pred_noise = self.denoiser(x, t_batch, global_feat, position)

            ab     = self.alphas_bar[t_cur]
            x0_hat = (x - (1 - ab).sqrt() * pred_noise) / ab.sqrt()
            x0_hat = x0_hat.clamp(-3, 3)

            if i < ddim_steps - 1:
                t_prev  = step_ids[i + 1]
                ab_prev = self.alphas_bar[t_prev]
                x = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * pred_noise
            else:
                x = x0_hat

        return x   # (n_samples, 6)

    def save(self, path: str, epoch: int = 0, best_loss: float = 0.0,
             extra: dict = None):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            'model_state_dict': self.state_dict(),
            'epoch':      epoch,
            'best_loss':  best_loss,
            'model_version': 'v2',
            'feat_dim':   GraspDenoiserV2.FEAT_DIM,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str, device=None):
        ckpt = torch.load(path, map_location=device or 'cpu', weights_only=False)
        self.load_state_dict(ckpt['model_state_dict'])
        return ckpt.get('epoch', 0), ckpt.get('best_loss', 0.0)
