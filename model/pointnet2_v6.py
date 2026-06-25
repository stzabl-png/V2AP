#!/usr/bin/env python3
"""
PointNet++ v6 — Continuous Affordance Heatmap Regression

4-layer SA (with group_all global aggregation) + 4-layer FP + Sigmoid head.
Input:  xyz (B, N, 3) + normals (B, N, 3) = 6ch
Output: per-point affordance heatmap (B, N) in [0, 1]

Key changes from v5:
  - 4 SA layers (was 3) with group_all at SA4 for true global context
  - 4096 input points (was 1024)
  - Single-channel sigmoid regression (was 2-class softmax)
  - No force-center head (removed to focus on heatmap quality)
  - PointNeXt-style relative coordinate normalization (/radius)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PointNet++ Basic Operations
# ============================================================

def square_distance(src, dst):
    """Pairwise squared distance. src: (B,N,3), dst: (B,M,3) → (B,N,M)"""
    return torch.sum((src.unsqueeze(2) - dst.unsqueeze(1)) ** 2, dim=-1)


def _dynamo_disable(fn):
    """Skip torch.compile for ops with Python loops / randint (FPS)."""
    try:
        import torch._dynamo as dynamo
        return dynamo.disable(fn)
    except Exception:
        return fn


@_dynamo_disable
def farthest_point_sample(xyz, npoint):
    """Farthest point sampling. xyz: (B,N,3) → indices: (B,npoint)"""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def index_points(points, idx):
    """Index points by idx. points: (B,N,C), idx: (B,S,...) → (B,S,...,C)"""
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=points.device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Ball query. xyz: (B,N,3), new_xyz: (B,S,3) → group_idx: (B,S,nsample)"""
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device

    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


# ============================================================
# SA Module (with optional group_all)
# ============================================================

class PointNetSetAbstraction(nn.Module):
    """Set Abstraction module with PointNeXt-style relative pos normalization."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: (B, N, 3)
        points: (B, N, C) or None
        Returns: new_xyz (B, S, 3), new_points (B, S, D)
        """
        if self.group_all:
            # Group all points into a single group, centroid at origin
            B, N, _ = xyz.shape
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.unsqueeze(1)  # (B, 1, N, 3)
            # Normalize by bounding box diagonal instead of radius
            bbox_diag = (xyz.max(dim=1)[0] - xyz.min(dim=1)[0]).norm(dim=-1)  # (B,)
            grouped_xyz_norm = grouped_xyz / (bbox_diag.view(B, 1, 1, 1) + 1e-8)

            if points is not None:
                grouped_points = points.unsqueeze(1)  # (B, 1, N, C)
                grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B, 1, N, C+3)
            else:
                grouped_points = grouped_xyz_norm
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)  # (B, S, nsample, 3)
            # PointNeXt: relative pos normalized by radius
            grouped_xyz_norm = (grouped_xyz - new_xyz.unsqueeze(2)) / self.radius

            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz_norm

        # MLP: (B, S, nsample, C) → permute → (B, C, nsample, S)
        grouped_points = grouped_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_points = F.relu(bn(conv(grouped_points)))

        # Max pool over nsample dimension
        new_points = torch.max(grouped_points, dim=2)[0]  # (B, D, S)
        new_points = new_points.permute(0, 2, 1)  # (B, S, D)
        return new_xyz, new_points


# ============================================================
# FP Module (Feature Propagation via 3-NN interpolation)
# ============================================================

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: (B, N, 3) — target (higher-res)
        xyz2: (B, S, 3) — source (lower-res)
        points1: (B, N, C1) — skip features
        points2: (B, S, C2) — features to upsample
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # group_all case: broadcast to all N points
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)  # (B, N, S)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # 3-NN

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.unsqueeze(-1), dim=2
            )

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # (B, C, N)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)  # (B, N, C)


# ============================================================
# PointNet++ v6: 4-layer SA + group_all + Sigmoid Regression
# ============================================================

class PointNet2AffordanceV6(nn.Module):
    """PointNet++ v6 for Continuous Affordance Heatmap Prediction.

    Architecture:
        SA1: 1024pt, r=0.05, ns=32,  [3+3→64→64→128]
        SA2: 256pt,  r=0.10, ns=64,  [128+3→128→128→256]
        SA3: 64pt,   r=0.20, ns=128, [256+3→256→256→512]
        SA4: group_all                [512+3→512→512→1024]   ← global aggregation

        FP4: 1024+512 → [512, 512]
        FP3: 256+512  → [256, 256]
        FP2: 128+256  → [256, 128]
        FP1: 3+128    → [128, 128, 128]

        Head: Conv1d(128→128) + BN + ReLU + Dropout(0.3) + Conv1d(128→1) + Sigmoid
    """

    def __init__(self, in_channel=3):
        super().__init__()

        # Encoder: 4 SA layers — half-width channels to reduce overfitting
        # (2.9M → ~0.7M params for 60-sample dataset)
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.05, nsample=32,
            in_channel=in_channel + 3, mlp=[32, 32, 64]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.10, nsample=64,
            in_channel=64 + 3, mlp=[64, 64, 128]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.20, nsample=128,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=1, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 256, 512],
            group_all=True
        )

        # Decoder: 4 FP layers
        self.fp4 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 128, [128, 128])
        self.fp2 = PointNetFeaturePropagation(128 + 64, [128, 64])
        self.fp1 = PointNetFeaturePropagation(64 + in_channel, [64, 64, 64])

        # Regression head: per-point affordance [0, 1]
        self.head = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(64, 1, 1),
        )

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, C) input features (normals, etc.)
        Returns:
            heatmap: (B, N) continuous affordance in [0, 1]
        """
        # Encoder
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 1, 1024)

        # Decoder
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, features, l1_points)

        # Head: (B, N, 128) → (B, 128, N) → (B, 1, N) → (B, N)
        x = l0_points.permute(0, 2, 1)  # (B, 128, N)
        x = self.head(x)                # (B, 1, N)
        heatmap = torch.sigmoid(x.squeeze(1))  # (B, N)
        return heatmap

    def extract_global_feat(self, xyz, features):
        """Frozen inference: extract 512-d global feature from SA1-SA4."""
        with torch.no_grad():
            l1_xyz, l1_points = self.sa1(xyz, features)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
            l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
            # l4_points: (B, 1, 1024) → squeeze → (B, 1024)
            global_feat = l4_points.squeeze(1)
        return global_feat
