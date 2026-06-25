#!/usr/bin/env python3
"""
Grasp Pose Diffusion 训练脚本 — v2

改动:
  - 目标从 10D → 6D (只预测 rotation_6d)
  - 条件从 global_feat(512) → global_feat(512) + force_center(3) = 515D
  - force_center 用 PointNet++ 实时预测（不用 GT force_center）
    → 训练和推理条件一致，两个模型真正绑定

用法:
  cd ~/Project/V2AP
  python3 -m model.train_diffusion \\
      --pn2_ckpt  output/checkpoints/v2_sigma04/best_model.pth \\
      --gt_dirs   output/robot_gt_merged_oakink output/robot_gt_merged_dexycb \\
      --pc_dir    data_hub/training_m5 \\
      --save_dir  output/checkpoints/v2_grasp_diff \\
      --epochs    500 --batch_size 32 --lr 1e-4
"""

import os, sys, glob, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import h5py
from model.pointnet2 import PointNet2Seg
from model.grasp_diffusion import GraspDiffusion, rotation_to_6d


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class GraspDiffusionDataset(Dataset):
    """
    每个样本 = (cloud 4096×6, rotation_6d 6D, pc_xyz 4096×3, affordance 4096)

    point_cloud + normals: 从 training_m5/{obj_id}.hdf5
    rotation_6d:           从 robot_gt_merged_*/{obj_id}_robot_gt.hdf5
    affordance (robot_gt): per-point affordance label，用于 Affordance 引导损失
    """

    def __init__(self, gt_dirs: list, pc_dir: str, n_points: int = 4096):
        self.n_points = n_points
        self.samples  = []   # (gt_hdf5_path, grasp_key, pc_hdf5_path)

        for gt_dir in gt_dirs:
            for gt_path in sorted(glob.glob(os.path.join(gt_dir, '*_robot_gt.hdf5'))):
                obj_id  = os.path.basename(gt_path).replace('_robot_gt.hdf5', '')
                pc_path = os.path.join(pc_dir, f'{obj_id}.hdf5')
                if not os.path.exists(pc_path):
                    continue
                try:
                    with h5py.File(gt_path, 'r') as hf:
                        if int(hf.attrs.get('n_successful', 0)) == 0:
                            continue
                        for key in hf.get('successful_grasps', {}).keys():
                            self.samples.append((gt_path, key, pc_path))
                except Exception:
                    continue

        print(f'  Dataset: {len(self.samples)} grasp samples '
              f'from {len(set(s[0] for s in self.samples))} objects')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gt_path, grasp_key, pc_path = self.samples[idx]

        # 点云 + 法线 + affordance
        with h5py.File(pc_path, 'r') as hf:
            pc   = hf['point_cloud'][:].astype(np.float32)   # (N,3)
            nrm  = hf['normals'][:].astype(np.float32)        # (N,3)
            aff  = hf['robot_gt'][:].astype(np.float32)       # (N,) per-point affordance
        n   = pc.shape[0]
        sel = np.random.choice(n, self.n_points, replace=(n < self.n_points))
        cloud = np.concatenate([pc[sel], nrm[sel]], axis=1)   # (4096,6)
        pc_xyz = pc[sel]                                       # (4096,3)
        aff_sel = aff[sel]                                     # (4096,)

        # rotation GT → 6D
        with h5py.File(gt_path, 'r') as hf:
            g   = hf[f'successful_grasps/{grasp_key}']
            rot = g['rotation'][:].astype(np.float32)          # (3,3)

        rot_6d = rotation_to_6d(torch.from_numpy(rot))        # (6,)

        return (
            torch.from_numpy(cloud),      # (4096,6)
            rot_6d,                       # (6,)
            torch.from_numpy(pc_xyz),     # (4096,3)  用于 affordance KNN
            torch.from_numpy(aff_sel),    # (4096,)   per-point affordance
        )


# ─────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 70)
    print('Grasp Pose Diffusion Training  v2')
    print('  output: rotation_6d (6D)')
    print('  condition: global_feat(512) + force_center_pred(3) = 515D')
    print('=' * 70)
    print(f'  Device:   {device}')
    if device.type == 'cuda':
        print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  PN2 ckpt: {args.pn2_ckpt}')
    print(f'  Save dir: {args.save_dir}')
    print()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 加载冻结 PointNet++ ─────────────────────────────────
    print('Loading frozen PointNet++ ...')
    pn2 = PointNet2Seg(num_classes=2, in_channel=6,
                       predict_force_center=True).to(device)
    ckpt = torch.load(args.pn2_ckpt, map_location=device, weights_only=False)
    pn2.load_state_dict(ckpt['model_state_dict'])
    pn2.eval()
    for p in pn2.parameters():
        p.requires_grad_(False)
    print(f'  ✅ PN2 loaded (epoch={ckpt.get("epoch","?")})')

    # ── Dataset ─────────────────────────────────────────────
    print('\nBuilding dataset ...')
    dataset = GraspDiffusionDataset(gt_dirs=args.gt_dirs, pc_dir=args.pc_dir)
    if len(dataset) == 0:
        print('❌ 没有找到训练样本'); return

    n_val   = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    print(f'  Train: {n_train}  Val: {n_val}')

    # rotation_6d 的统计（force_center 归一化在推理时直接用 PN2 输出，不额外存）
    print('Computing rotation_6d statistics ...')
    all_r = []
    for cloud, rot_6d, pc_xyz, aff_labels in train_loader:
        all_r.append(rot_6d)
        if len(all_r) > 50: break
    all_r   = torch.cat(all_r, dim=0)
    rot_mean = all_r.mean(0)
    rot_std  = all_r.std(0).clamp(min=1e-4)
    torch.save({'rot_mean': rot_mean, 'rot_std': rot_std},
               os.path.join(args.save_dir, 'rot_stats.pt'))
    print(f'  rot_mean: {rot_mean.numpy().round(3)}')

    # ── Diffusion + optimizer ────────────────────────────────
    diffusion  = GraspDiffusion(T=args.T, hidden=512).to(device)
    optimizer  = torch.optim.AdamW(diffusion.parameters(),
                                   lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    rot_mean = rot_mean.to(device)
    rot_std  = rot_std.to(device)

    best_val_loss = float('inf')
    history = []

    print(f'\n  Ep | Train Loss | Val Loss  | LR')
    print('  ' + '-' * 50)

    GRIPPER_HW    = 0.030        # 夹爪半宽 3cm（用于计算指尖位置）
    LAMBDA_AFF    = 0.3          # Affordance loss 权重
    AFF_WARMUP    = 50           # 前 N epoch 不加 affordance loss（先让 diffusion 稳定）

    def affordance_loss(x0_pred, fc, pc_xyz, aff_labels, hw=GRIPPER_HW):
        """
        x0_pred:   (B,6)  — 去噪后的 rot_6d 预测
        fc:        (B,3)  — force_center
        pc_xyz:    (B,N,3)— 点云坐标
        aff_labels:(B,N)  — per-point affordance
        返回: 标量 loss
        """
        B = x0_pred.shape[0]
        # 解码 rot_6d → rotation_matrix (Gram-Schmidt)
        a1 = x0_pred[:, :3]                        # (B,3)
        a2 = x0_pred[:, 3:]                        # (B,3)
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - (a2 * b1).sum(-1, keepdim=True) * b1, dim=-1)
        finger_dir = b1                             # col0 = finger_dir

        # 指尖位置
        tip_l = fc - hw * finger_dir               # (B,3)
        tip_r = fc + hw * finger_dir               # (B,3)

        # KNN: 找每个指尖最近的点云点的 affordance（向量化，无循环）
        def query_aff(tip, pts, labels):
            # tip: (B,3)  pts: (B,N,3)  labels: (B,N)
            d2 = ((pts - tip.unsqueeze(1)) ** 2).sum(-1)  # (B,N)
            nn_idx = d2.argmin(dim=1)                      # (B,)
            aff_val = labels[torch.arange(B, device=pts.device), nn_idx]  # (B,)
            return aff_val

        aff_l = query_aff(tip_l, pc_xyz, aff_labels)   # (B,)
        aff_r = query_aff(tip_r, pc_xyz, aff_labels)   # (B,)

        # 损失: (1 - aff)^2 → 鼓励指尖落在高 affordance 区域
        return ((1 - aff_l) ** 2 + (1 - aff_r) ** 2).mean()

    for epoch in range(1, args.epochs + 1):

        # ── Train ────────────────────────────────────────────
        diffusion.train()
        train_losses = []
        for cloud, rot_6d, pc_xyz, aff_labels in train_loader:
            cloud      = cloud.to(device)       # (B,4096,6)
            rot_6d     = rot_6d.to(device)      # (B,6)
            pc_xyz     = pc_xyz.to(device)      # (B,4096,3)
            aff_labels = aff_labels.to(device)  # (B,4096)

            rot_norm = (rot_6d - rot_mean) / rot_std

            with torch.no_grad():
                xyz         = cloud[:, :, :3]
                global_feat = pn2.extract_global_feat(xyz, cloud)  # (B,512)
                _, fc_pred  = pn2(xyz, cloud)                       # (B,3)

            # Diffusion denoising loss + x0 prediction
            diff_loss, x0_pred = diffusion.training_loss(
                rot_norm, global_feat, fc_pred, return_x0=True)

            # Affordance 引导损失（warmup 后才开始加）
            if epoch > AFF_WARMUP and x0_pred is not None:
                # x0_pred 是归一化空间，先反归一化回 rot_6d
                x0_denorm = x0_pred.detach() * rot_std + rot_mean
                l_aff = affordance_loss(x0_denorm, fc_pred.detach(),
                                        pc_xyz, aff_labels)
                loss = diff_loss + LAMBDA_AFF * l_aff
            else:
                loss = diff_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            train_losses.append(diff_loss.item())

        scheduler.step()

        # ── Val ──────────────────────────────────────────────
        diffusion.eval()
        val_losses = []
        with torch.no_grad():
            for cloud, rot_6d, pc_xyz, aff_labels in val_loader:
                cloud  = cloud.to(device)
                rot_6d = rot_6d.to(device)
                rot_norm    = (rot_6d - rot_mean) / rot_std
                xyz         = cloud[:, :, :3]
                global_feat = pn2.extract_global_feat(xyz, cloud)
                _, fc_pred  = pn2(xyz, cloud)
                diff_loss, _ = diffusion.training_loss(
                    rot_norm, global_feat, fc_pred, return_x0=True)
                val_losses.append(diff_loss.item())

        t_loss = sum(train_losses) / len(train_losses)
        v_loss = sum(val_losses)   / len(val_losses)
        lr_now = scheduler.get_last_lr()[0]
        history.append({'epoch': epoch, 'train': t_loss, 'val': v_loss})

        if epoch % 10 == 0 or epoch == 1:
            star = ' ★' if v_loss < best_val_loss else ''
            print(f'  {epoch:3d} | {t_loss:.6f} | {v_loss:.6f} | {lr_now:.2e}{star}')

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            diffusion.save(os.path.join(args.save_dir, 'best_model.pth'),
                           epoch=epoch, best_loss=best_val_loss)

        if epoch % 100 == 0:
            diffusion.save(
                os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'),
                epoch=epoch, best_loss=v_loss)

    diffusion.save(os.path.join(args.save_dir, 'final_model.pth'),
                   epoch=args.epochs, best_loss=best_val_loss)

    import json
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print('\n' + '=' * 70)
    print('COMPLETE — Grasp Pose Diffusion v2')
    print(f'  Best Val Loss: {best_val_loss:.6f}')
    print(f'  Best model:    {args.save_dir}/best_model.pth')
    print('=' * 70)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pn2_ckpt',   required=True)
    parser.add_argument('--gt_dirs',    nargs='+', required=True)
    parser.add_argument('--pc_dir',     required=True)
    parser.add_argument('--save_dir',   default='output/checkpoints/v2_grasp_diff')
    parser.add_argument('--epochs',     type=int,   default=500)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--T',          type=int,   default=1000)
    args = parser.parse_args()
    train(args)
