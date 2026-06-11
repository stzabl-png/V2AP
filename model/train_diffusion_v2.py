#!/usr/bin/env python3
"""
train_diffusion_v2.py — Grasp Pose Diffusion 训练脚本 v2

核心改动 (vs v1):
  ① 数据源:    affordance_all.h5 (点云/heatmap) + merged/*.hdf5 (Robot Posterior poses)
  ② Pose GT:   executed_panda_hand_at_close/{rotation, position}，undo mesh_prerotation
  ③ V6 适配:   global_feat 1024D (extract_global_feat, 预缓存，训练中无需 forward)
  ④ 数据量:    6582 个 Robot Posterior grasps (vs 878)
  ⑤ force_center: GT position from Robot Posterior (而非 PN2 预测)
  ⑥ Aff loss:  K=5 邻域均值 (vs K=1 单点)
  ⑦ 日志:      model/logger.py 完整记录

用法:
  cd ~/Project/Affordance2Grasp
  python3 -m model.train_diffusion_v2 \\
      --v6_ckpt     data_hub/ProcessedData/RobotPosterior/best_v6_model.pth \\
      --aff_h5      data_hub/ProcessedData/RobotPosterior/affordance_all.h5 \\
      --merged_dir  data_hub/ProcessedData/RobotPosterior/merged \\
      --split_json  data_hub/ProcessedData/RobotPosterior/min20/objects_train_val_split.json \\
      --save_dir    output/checkpoints_diffusion_v2 \\
      --epochs 500 --batch_size 64 --lr 3e-4
"""

import os, sys, glob, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import h5py
from model.grasp_diffusion_v2 import (
    GraspDiffusionV2, rotation_to_6d, rotation_from_6d,
    get_finger_dir, TCP_OFFSET, GRIPPER_HW
)
from model.logger import TrainingLogger, EvalLogger

# ─────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────
AFF_WARMUP   = 50     # 前 N epoch 不加 affordance loss
LAMBDA_AFF   = 0.3    # affordance loss 权重
AFF_K        = 5      # 指尖 KNN 邻域大小（v2: K=5，vs v1 K=1）
N_POINTS     = 4096   # 点云采样数
GRIPPER_HW   = 0.030  # 夹爪半宽 (m)
POS_MAX_ABS  = 0.5    # [P0 fix] 超过此阈值的 position 判为单位错误，跳过


# ─────────────────────────────────────────────────────────────
# 步骤1: V6 global_feat 预缓存
# ─────────────────────────────────────────────────────────────

def cache_global_feats(v6_ckpt: str, aff_h5: str,
                       device: torch.device,
                       logger: TrainingLogger) -> dict:
    """
    用冻结 V6 对所有对象点云提取 512D global_feat，缓存到 dict。
    只跑一次，训练中直接查表。

    返回: {obj_id: global_feat_cpu_tensor (1, 512)}
    """
    from model.pointnet2_v6 import PointNet2AffordanceV6

    logger.section("Caching V6 global_feat (one-time pass)")
    pn2 = PointNet2AffordanceV6().to(device)
    ckpt = torch.load(v6_ckpt, map_location=device, weights_only=False)
    pn2.load_state_dict(ckpt['model_state_dict'])
    pn2.eval()
    for p in pn2.parameters():
        p.requires_grad_(False)
    logger.info(f"  V6 loaded  epoch={ckpt.get('epoch','?')}  "
                f"val_pearson={ckpt.get('val_pearson','?')}")

    cache = {}
    with h5py.File(aff_h5, 'r') as hf:
        obj_ids = [x.decode() if isinstance(x, bytes) else x
                   for x in hf['data']['obj_ids'][:]]
        points  = hf['data']['points'][:]    # (N_obj, 4096, 3)
        normals = hf['data']['normals'][:]   # (N_obj, 4096, 3)

    for i, oid in enumerate(obj_ids):
        xyz = torch.from_numpy(points[i]).unsqueeze(0).to(device)    # (1,4096,3)
        nrm = torch.from_numpy(normals[i]).unsqueeze(0).to(device)   # (1,4096,3)
        # V6 features = concat(xyz, normals) = 6D，与 train_v6.py 一致
        features = torch.cat([xyz, nrm], dim=-1)                     # (1,4096,6)
        feat = pn2.extract_global_feat(xyz, features)                 # (1, 512)
        cache[oid] = feat.cpu()

    logger.info(f"  Cached {len(cache)} objects")
    del pn2
    torch.cuda.empty_cache()
    return cache


# ─────────────────────────────────────────────────────────────
# 步骤2: Dataset
# ─────────────────────────────────────────────────────────────

class RobotPosteriorPoseDataset(Dataset):
    """
    每个样本 = 一个成功的 Robot Posterior 抓取:
      global_feat  (512,)  : V6 预缓存
      pos_rel      (3,)    : TCP 位置（相对 PC 中心，已除以 PC 半径）← P0 fix
      rot_6d       (6,)    : rotation GT (canonical frame, undo prerotation)
      finger_dir   (3,)    : 手指方向 (canonical frame)
      aff_labels   (4096,) : per-point affordance labels
      pc_xyz       (4096,3): 点云坐标
      pc_centroid  (3,)    : 点云中心 (用于 aff loss 反归一化)
      pc_radius    (1,)    : 点云半径 (用于 aff loss 反归一化)
    """

    def __init__(self,
                 obj_ids:      list,
                 aff_h5:       str,
                 merged_dir:   str,
                 global_feats: dict,
                 logger:       TrainingLogger):
        self.samples = []   # list of dicts
        skipped_objs = []
        grasp_counts = []

        # 读 affordance_all.h5 中的点云和 affordance labels
        with h5py.File(aff_h5, 'r') as hf:
            all_ids = [x.decode() if isinstance(x, bytes) else x
                       for x in hf['data']['obj_ids'][:]]
            aff_idx = {oid: i for i, oid in enumerate(all_ids)}
            all_points  = hf['data']['points'][:]    # (N_obj, 4096, 3)
            all_normals = hf['data']['normals'][:]   # (N_obj, 4096, 3)
            all_labels  = hf['data']['labels'][:]    # (N_obj, 4096)

        for oid in obj_ids:
            # 对象需在 affordance_all.h5 中存在
            if oid not in aff_idx:
                skipped_objs.append((oid, "not in affordance_all.h5"))
                continue

            # 对象需在 merged/ 中存在
            merged_path = os.path.join(merged_dir, f"{oid}_robot_gt_merged.hdf5")
            if not os.path.exists(merged_path):
                skipped_objs.append((oid, "no merged hdf5"))
                continue

            idx = aff_idx[oid]
            pc_xyz    = all_points[idx].astype(np.float32)   # (4096, 3)
            pc_normals= all_normals[idx].astype(np.float32)  # (4096, 3) ← new
            aff_lab   = all_labels[idx].astype(np.float32)   # (4096,)
            feat      = global_feats[oid]                     # (1, 512)

            # [P0 fix] 计算 PC 中心 + 半径，用于相对位置归一化
            pc_centroid = pc_xyz.mean(0)                       # (3,)
            pc_radius   = float(np.linalg.norm(
                pc_xyz - pc_centroid, axis=1).max()) + 1e-6   # scalar

            obj_samples  = 0
            n_pos_filter = 0
            try:
                with h5py.File(merged_path, 'r') as hf:
                    sg = hf.get('successful_grasps', {})
                    for gkey in sg.keys():
                        g = sg[gkey]

                        # 读 executed_panda_hand_at_close
                        ep = g.get('executed_panda_hand_at_close')
                        if ep is None:
                            continue
                        R_exec   = ep['rotation'][:].astype(np.float32)    # (3,3)
                        pos_exec = ep['position'][:].astype(np.float32)    # (3,)
                        fd_exec  = ep.get('finger_dir')
                        if fd_exec is None:
                            fd_exec = R_exec[:, 0]
                        else:
                            fd_exec = fd_exec[:].astype(np.float32)

                        # Undo mesh_prerotation → canonical frame
                        pre = g.get('mesh_prerotation')
                        if pre is not None and 'matrix' in pre:
                            R_pre     = pre['matrix'][:].astype(np.float32)
                            R_canon   = R_pre.T @ R_exec
                            pos_canon = R_pre.T @ pos_exec
                            fd_canon  = R_pre.T @ fd_exec
                        else:
                            R_canon   = R_exec
                            pos_canon = pos_exec
                            fd_canon  = fd_exec

                        # ── [P0 fix] 单位过滤 ─────────────────────
                        # position 超过 0.5m 认为是 mm 单位错误，跳过
                        if np.abs(pos_canon).max() > POS_MAX_ABS:
                            n_pos_filter += 1
                            continue

                        # ── 相对位置归一化 ─────────────────────────
                        # pos_rel = (pos - pc_centroid) / pc_radius
                        # 物理含义：TCP 在物体半径坐标系下的位置
                        # 正常值范围约 [-3, 3]（TCP 在物体 1~2 半径处）
                        pos_rel = (pos_canon - pc_centroid) / pc_radius  # (3,)

                        rot_6d = rotation_to_6d(
                            torch.from_numpy(R_canon)).numpy()   # (6,)

                        self.samples.append({
                            'global_feat': feat,           # (1, 512)
                            'pos_rel':     pos_rel,        # (3,)  ← 相对归一化
                            'rot_6d':      rot_6d,         # (6,)
                            'finger_dir':  fd_canon,       # (3,)
                            'aff_labels':  aff_lab,        # (4096,)
                            'pc_xyz':      pc_xyz,         # (4096, 3)
                            'pc_normals':  pc_normals,     # (4096, 3) ← new
                            'pc_centroid': pc_centroid,    # (3,)
                            'pc_radius':   np.array([pc_radius], dtype=np.float32),
                            'obj_id':      oid,
                        })
                        obj_samples += 1

            except Exception as e:
                skipped_objs.append((oid, f"read error: {e}"))
                continue

            if n_pos_filter > 0:
                skipped_objs.append((oid, f"pos_unit_filter: {n_pos_filter} grasps skipped"))
            if obj_samples == 0:
                skipped_objs.append((oid, "no valid grasps after filter"))
            else:
                grasp_counts.append(obj_samples)

        # 日志
        logger.log_dataset({
            'n_objects':       len(grasp_counts),
            'n_samples':       len(self.samples),
            'skipped_objects': [f"{o}: {r}" for o, r in skipped_objs],
            'grasp_dist': {
                'min':    int(np.min(grasp_counts))    if grasp_counts else 0,
                'max':    int(np.max(grasp_counts))    if grasp_counts else 0,
                'mean':   float(np.mean(grasp_counts)) if grasp_counts else 0,
                'median': float(np.median(grasp_counts)) if grasp_counts else 0,
            } if grasp_counts else {},
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s['global_feat'].squeeze(0),                      # (512,)
            torch.from_numpy(s['pos_rel']),                   # (3,)
            torch.from_numpy(s['rot_6d']),                    # (6,)
            torch.from_numpy(s['finger_dir']),                # (3,)
            torch.from_numpy(s['aff_labels']),                # (4096,)
            torch.from_numpy(s['pc_xyz']),                    # (4096, 3)
            torch.from_numpy(s['pc_normals']),                # (4096, 3) ← new
            torch.from_numpy(s['pc_centroid']),               # (3,)
            torch.from_numpy(s['pc_radius']),                 # (1,)
        )


# ─────────────────────────────────────────────────────────────
# 步骤3: Affordance 引导损失 v3 — 几何对齐
# ─────────────────────────────────────────────────────────────
#
# 设计原则:
#   旧版 (aff_loss_k5): 惩罚指尖落在低 aff 点云区 → 失效
#     原因: 指尖在物体表面外 3cm, aff label 天然为 0
#
#   新版 (grasp_geometry_loss): 用局部法向量约束 rotation 几何
#     L1 approach: approach_dir 应反向于接触点局部法向 (从外接近)
#     L2 finger:   finger_dir 应垂直于局部法向 (沿表面张开)
#     L3 aff_cp:   接触点本身 affordance 应高 (辅助项)
# ─────────────────────────────────────────────────────────────

def grasp_geometry_loss(
        x0_pred:    torch.Tensor,   # (B, 6)    预测 rot_6d
        position:   torch.Tensor,   # (B, 3)    TCP 位置 (canonical)
        pc_xyz:     torch.Tensor,   # (B, N, 3) 点云坐标
        pc_normals: torch.Tensor,   # (B, N, 3) 点云法向 ← new input
        aff_labels: torch.Tensor,   # (B, N)    affordance labels
        k: int = AFF_K
) -> torch.Tensor:
    """
    几何对齐损失:
      L_approach: cos(approach_dir, local_normal) 应 < 0 (从外向内)
      L_finger:   |cos(finger_dir, local_normal)| 应 ≈ 0 (垂直于法向)
      L_aff_cp:   接触点处 affordance 应高 (辅助)
    """
    from model.grasp_diffusion_v2 import rotation_from_6d

    # 从 6D 恢复完整旋转矩阵
    R = rotation_from_6d(x0_pred)             # (B, 3, 3)
    approach_dir = R[..., 2]                  # Z 轴: (B, 3)
    finger_dir   = R[..., 0]                  # X 轴: (B, 3)

    # K 近邻: 接触点附近的局部法向均值
    d2       = ((pc_xyz - position.unsqueeze(1)) ** 2).sum(-1)   # (B, N)
    knn_idx  = d2.topk(k, dim=1, largest=False).indices          # (B, k)
    knn_exp  = knn_idx.unsqueeze(-1).expand(-1, -1, 3)           # (B, k, 3)

    local_nrm = pc_normals.gather(1, knn_exp).mean(1)            # (B, 3)
    local_nrm = F.normalize(local_nrm, dim=-1)

    # L1: approach_dir · local_normal 应 < 0（从外向内）
    #     cos > 0 说明夹爪从内向外，惩罚 relu(cos + margin)
    cos_approach = (approach_dir * local_nrm).sum(-1)            # (B,)
    L_approach   = F.relu(cos_approach + 0.1).mean()

    # L2: finger_dir 应垂直于法向（躺在切平面上）
    #     |cos(finger, normal)| 应接近 0
    cos_finger = (finger_dir * local_nrm).sum(-1).abs()          # (B,)
    L_finger   = cos_finger.mean()

    # L3: 接触点本身 affordance 应高
    aff_cp   = aff_labels.gather(1, knn_idx).mean(1)             # (B,)
    L_aff_cp = (1 - aff_cp).mean()

    return L_approach + 0.5 * L_finger + 0.3 * L_aff_cp



# ─────────────────────────────────────────────────────────────
# 步骤4: 训练
# ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = TrainingLogger(
        save_dir=args.save_dir,
        run_name=f"diffusion_v2_min{args.min_trusted}"
    )
    logger.section("Grasp Pose Diffusion Training v2")
    logger.info(f"  Device: {device}")
    if device.type == 'cuda':
        logger.info(f"  GPU:    {torch.cuda.get_device_name(0)}")

    # 记录配置
    cfg = vars(args).copy()
    cfg.update({
        'model':        'GraspDiffusionV2',
        'feat_dim':     1024,
        'pose_gt':      'executed_panda_hand_at_close (undo mesh_prerotation)',
        'aff_k':        AFF_K,
        'aff_warmup':   AFF_WARMUP,
        'lambda_aff':   LAMBDA_AFF,
        'n_points':     N_POINTS,
        'tcp_offset_m': TCP_OFFSET,
        'gripper_hw_m': GRIPPER_HW,
    })
    logger.log_config(cfg)

    # ── train/val split ───────────────────────────────────────
    split = json.load(open(args.split_json))
    train_ids = split['train']
    val_ids   = split['val']
    logger.info(f"  Split: {len(train_ids)} train / {len(val_ids)} val objects")

    # ── V6 global_feat 预缓存 ─────────────────────────────────
    global_feats = cache_global_feats(
        args.v6_ckpt, args.aff_h5, device, logger)

    # ── Datasets ──────────────────────────────────────────────
    logger.section("Building Datasets")
    logger.info("  Building train dataset...")
    train_set = RobotPosteriorPoseDataset(
        train_ids, args.aff_h5, args.merged_dir, global_feats, logger)
    logger.info("  Building val dataset...")
    val_set   = RobotPosteriorPoseDataset(
        val_ids,   args.aff_h5, args.merged_dir, global_feats, logger)

    if len(train_set) == 0:
        logger.error("训练集为空！检查 split_json / merged_dir / aff_h5 路径")
        return

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    logger.info(f"  Train: {len(train_set)} grasps  ({len(train_loader)} batches)")
    logger.info(f"  Val:   {len(val_set)}   grasps  ({len(val_loader)} batches)")

    # ── rotation_6d 归一化统计 ────────────────────────────────
    logger.info("\n  Computing rotation_6d statistics...")
    all_r = []
    for batch in train_loader:
        all_r.append(batch[2])   # rot_6d index=2
        if len(all_r) > 80:
            break
    all_r    = torch.cat(all_r, dim=0)
    rot_mean = all_r.mean(0)
    rot_std  = all_r.std(0).clamp(min=1e-4)
    torch.save({'rot_mean': rot_mean, 'rot_std': rot_std},
               os.path.join(args.save_dir, 'rot_stats.pt'))
    logger.info(f"  rot_mean: {rot_mean.numpy().round(3)}")
    logger.info(f"  rot_std : {rot_std.numpy().round(3)}")

    # ── pos_rel 归一化统计 (已是相对坐标，正常值 ±3) ──────────
    # [P0 fix] pos_rel = (pos_canon - pc_centroid) / pc_radius
    # 期望 pos_std ≈ 0.5~1.5，若仍很大说明有残余异常值
    all_pos = []
    for batch in train_loader:
        all_pos.append(batch[1])  # pos_rel index=1
        if len(all_pos) > 80:
            break
    all_pos  = torch.cat(all_pos, dim=0)
    pos_mean = all_pos.mean(0)
    pos_std  = all_pos.std(0).clamp(min=1e-4)
    logger.info(f"  pos_rel mean: {pos_mean.numpy().round(3)}")
    logger.info(f"  pos_rel std : {pos_std.numpy().round(3)}  (正常应 < 2.0)")
    if pos_std.max().item() > 3.0:
        logger.info("  ⚠  pos_rel std > 3, 仍有异常值！检查 POS_MAX_ABS 阈值")
    torch.save({'pos_mean': pos_mean, 'pos_std': pos_std},
               os.path.join(args.save_dir, 'pos_stats.pt'))

    # ── Model + optimizer ─────────────────────────────────────
    diffusion = GraspDiffusionV2(T=args.T, hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    rot_mean = rot_mean.to(device)
    rot_std  = rot_std.to(device)
    pos_mean = pos_mean.to(device)
    pos_std  = pos_std.to(device)

    logger.section("Training Start")
    logger.info(f"  {'Ep':>4} | {'Train':>10} | {'Val':>10} | {'AffLoss':>9} | LR")
    logger.info("  " + "-" * 55)

    best_val_loss = float('inf')
    best_epoch    = 0

    for epoch in range(1, args.epochs + 1):

        # ── Train ─────────────────────────────────────────────
        diffusion.train()
        t_losses, aff_losses = [], []

        for batch in train_loader:
            global_feat, pos_rel, rot_6d, finger_dir, aff_labels, pc_xyz, \
                pc_normals, pc_centroid, pc_radius = [x.to(device) for x in batch]

            rot_norm = (rot_6d - rot_mean) / rot_std
            pos_norm = (pos_rel - pos_mean) / pos_std

            diff_loss, x0_pred = diffusion.training_loss(
                rot_norm, global_feat, pos_norm, return_x0=True)

            # 几何对齐损失（warmup 后）
            if epoch > AFF_WARMUP and x0_pred is not None:
                x0_denorm = x0_pred.detach() * rot_std + rot_mean
                pos_canon = pos_rel * pc_radius + pc_centroid   # 反归一化
                l_aff = grasp_geometry_loss(
                    x0_denorm, pos_canon, pc_xyz, pc_normals, aff_labels)
                loss = diff_loss + LAMBDA_AFF * l_aff
                aff_losses.append(l_aff.item())
            else:
                loss = diff_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            t_losses.append(diff_loss.item())

        scheduler.step()

        # ── Val ───────────────────────────────────────────────
        diffusion.eval()
        v_losses = []
        with torch.no_grad():
            for batch in val_loader:
                global_feat, pos_rel, rot_6d, _, _, _, _, _, _ = \
                    [x.to(device) for x in batch]
                rot_norm = (rot_6d - rot_mean) / rot_std
                pos_norm = (pos_rel - pos_mean) / pos_std
                d_loss, _ = diffusion.training_loss(rot_norm, global_feat, pos_norm)
                v_losses.append(d_loss.item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        a_loss = float(np.mean(aff_losses)) if aff_losses else 0.0
        lr_now = scheduler.get_last_lr()[0]
        is_best = v_loss < best_val_loss

        if is_best:
            best_val_loss = v_loss
            best_epoch    = epoch
            diffusion.save(
                os.path.join(args.save_dir, 'best_model.pth'),
                epoch=epoch, best_loss=best_val_loss,
                extra={'rot_mean': rot_mean.cpu().tolist(),
                       'rot_std':  rot_std.cpu().tolist(),
                       'pos_mean': pos_mean.cpu().tolist(),
                       'pos_std':  pos_std.cpu().tolist()})

        if epoch % 100 == 0:
            diffusion.save(
                os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'),
                epoch=epoch, best_loss=v_loss)

        # 结构化日志
        logger.log_epoch(
            epoch, args.epochs, t_loss, v_loss, lr_now,
            extra={
                'aff_loss':      round(a_loss, 6),
                'aff_active':    epoch > AFF_WARMUP,
                'is_best':       is_best,
                'best_epoch':    best_epoch,
                'best_val_loss': round(best_val_loss, 6),
            })

    # ── 完成 ──────────────────────────────────────────────────
    diffusion.save(
        os.path.join(args.save_dir, 'final_model.pth'),
        epoch=args.epochs, best_loss=best_val_loss)
    logger.done(best_epoch, best_val_loss)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Grasp Diffusion v2 Training")

    # 数据
    p.add_argument('--v6_ckpt',    required=True,
                   help='V6 PointNet++ 权重路径 (best_v6_model.pth)')
    p.add_argument('--aff_h5',     required=True,
                   help='affordance_all.h5 路径')
    p.add_argument('--merged_dir', required=True,
                   help='merged/*.hdf5 目录 (Robot Posterior)')
    p.add_argument('--split_json', required=True,
                   help='objects_train_val_split.json 路径')
    p.add_argument('--min_trusted', type=int, default=20,
                   help='仅用 ≥N 次成功的对象（与 split_json 对应）')

    # 训练
    p.add_argument('--save_dir',   default='output/checkpoints_diffusion_v2')
    p.add_argument('--epochs',     type=int,   default=500)
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--T',          type=int,   default=1000)
    p.add_argument('--hidden',     type=int,   default=1024)

    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
