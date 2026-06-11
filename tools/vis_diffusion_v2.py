#!/usr/bin/env python3
"""
tools/vis_diffusion_v2.py — GraspDiffusion v2 推理可视化

对验证集对象做推理并输出 PNG：
  - 点云按 Affordance heatmap 着色 (蓝→红)
  - 采样 K 个接触点 (橙色)
  - 每个接触点生成 M 个抓取候选 (箭头: approach_dir + finger_dir)
  - 叠加 GT Robot Posterior 对比 (绿色)

用法:
  # 全验证集
  python3 tools/vis_diffusion_v2.py \\
      --diff_ckpt output/checkpoints_diffusion_v2/best_model.pth \\
      --v6_ckpt   data_hub/ProcessedData/RobotPosterior/best_v6_model.pth \\
      --aff_h5    data_hub/ProcessedData/RobotPosterior/affordance_all.h5 \\
      --merged_dir data_hub/ProcessedData/RobotPosterior/merged \\
      --split_json data_hub/ProcessedData/RobotPosterior/min20/objects_train_val_split.json \\
      --save_dir  output/vis_diffusion_v2

  # 单物体
  python3 tools/vis_diffusion_v2.py --obj A01001 [...]
"""

import os, sys, json, argparse
import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.spatial import cKDTree

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from model.pointnet2_v6 import PointNet2AffordanceV6
from model.grasp_diffusion_v2 import (
    GraspDiffusionV2, rotation_from_6d, get_approach_dir, get_finger_dir,
    TCP_OFFSET, GRIPPER_HW
)

# ─────────────────────────────────────────────────────────────
# 加载模型
# ─────────────────────────────────────────────────────────────

def load_v6(ckpt_path: str, device):
    m = PointNet2AffordanceV6().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    print(f"  V6  loaded  epoch={ckpt.get('epoch','?')}  "
          f"val_pearson={ckpt.get('val_pearson','?'):.4f}")
    return m


def load_diffusion(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = GraspDiffusionV2(T=1000, hidden=512).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()

    # 归一化统计（直接从 ckpt 读，不需要单独的 rot_stats.pt）
    save_dir = os.path.dirname(ckpt_path)
    rot_stats = torch.load(os.path.join(save_dir, 'rot_stats.pt'),
                           map_location=device, weights_only=False)
    pos_stats = torch.load(os.path.join(save_dir, 'pos_stats.pt'),
                           map_location=device, weights_only=False)
    print(f"  Diff loaded epoch={ckpt.get('epoch','?')}  "
          f"best_loss={ckpt.get('best_loss','?'):.6f}")
    return m, rot_stats, pos_stats


# ─────────────────────────────────────────────────────────────
# 接触点采样 (从 heatmap)
# ─────────────────────────────────────────────────────────────

def sample_contact_points(xyz: np.ndarray,
                          heatmap: np.ndarray,
                          k: int = 3,
                          threshold: float = 0.35) -> np.ndarray:
    """
    从高 Affordance 区域采 K 个空间分散的接触点 (FPS)。
    xyz: (N,3)  heatmap: (N,)
    → contact_pts: (K, 3)
    """
    mask = heatmap >= threshold
    if mask.sum() < k:
        # threshold 太高，降级取 top-k
        idx = np.argsort(heatmap)[::-1][:k]
        return xyz[idx]

    cand_xyz = xyz[mask]
    cand_aff = heatmap[mask]

    # FPS：空间分散
    selected = [np.argmax(cand_aff)]  # 从最高 aff 点开始
    for _ in range(k - 1):
        dists = np.min(
            np.linalg.norm(cand_xyz[selected][:, None] - cand_xyz[None], axis=-1),
            axis=0)
        # 距离远 + aff 高 的综合评分
        score = dists * (cand_aff ** 0.5)
        score[selected] = -1
        selected.append(np.argmax(score))

    return cand_xyz[selected]  # (K, 3)


# ─────────────────────────────────────────────────────────────
# 推理
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_poses(v6, diffusion, rot_stats, pos_stats,
                  pc_xyz: np.ndarray, normals: np.ndarray,
                  contact_pts: np.ndarray,
                  device,
                  n_samples: int = 5,
                  ddim_steps: int = 50):
    """
    对每个 contact_pt 生成 n_samples 个抓取位姿。
    返回: List[ dict{position, approach_dir, finger_dir, rot_6d} ] × K × n_samples
    """
    # V6 → global_feat + heatmap
    xyz_t  = torch.from_numpy(pc_xyz).unsqueeze(0).to(device)
    feat_t = torch.cat([xyz_t, torch.from_numpy(normals).unsqueeze(0).to(device)], dim=-1)
    global_feat = v6.extract_global_feat(xyz_t, feat_t)  # (1, 512)

    rot_mean = rot_stats['rot_mean'].to(device)
    rot_std  = rot_stats['rot_std'].to(device)
    pos_mean = pos_stats['pos_mean'].to(device)
    pos_std  = pos_stats['pos_std'].to(device)

    all_poses = []
    for cp in contact_pts:
        pos_t = torch.from_numpy(cp).unsqueeze(0).to(device)       # (1, 3)
        pos_norm = (pos_t - pos_mean) / pos_std

        # Diffusion 采样
        rot_6d_norm = diffusion.sample(
            global_feat, pos_norm, n_samples=n_samples, ddim_steps=ddim_steps)  # (n,6)

        # 反归一化
        rot_6d = rot_6d_norm * rot_std + rot_mean

        approach = get_approach_dir(rot_6d).cpu().numpy()   # (n,3)
        finger   = get_finger_dir(rot_6d).cpu().numpy()     # (n,3)
        rot_6d_np = rot_6d.cpu().numpy()

        cp_poses = []
        for i in range(n_samples):
            cp_poses.append({
                'position':    cp,
                'approach_dir': approach[i] / (np.linalg.norm(approach[i]) + 1e-8),
                'finger_dir':   finger[i]   / (np.linalg.norm(finger[i])   + 1e-8),
                'rot_6d':       rot_6d_np[i],
            })
        all_poses.append(cp_poses)

    return all_poses  # [K][n_samples] dicts


# ─────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────

def draw_gripper_arrow(ax, position, approach_dir, finger_dir,
                       color='dodgerblue', alpha=0.8, scale=0.05):
    """在 ax 上绘制夹爪示意图：approach 箭头 + 两个指尖短线"""
    p = position

    # Approach 箭头（蓝/颜色）
    ax.quiver(*p, *(approach_dir * scale),
              color=color, alpha=alpha, linewidth=1.5, arrow_length_ratio=0.3)

    # Finger span（横线）
    tip_l = p - finger_dir * GRIPPER_HW
    tip_r = p + finger_dir * GRIPPER_HW
    ax.plot([tip_l[0], tip_r[0]], [tip_l[1], tip_r[1]], [tip_l[2], tip_r[2]],
            color=color, alpha=alpha, linewidth=2.0)

    # Wrist 小点
    wrist = p - approach_dir * TCP_OFFSET * 0.5
    ax.scatter(*wrist, c=color, marker='o', s=15, alpha=alpha, zorder=5)


def visualize_object(obj_id: str,
                     pc_xyz:   np.ndarray,
                     heatmap:  np.ndarray,
                     contact_pts: np.ndarray,
                     pred_poses,        # [K][n_samples]
                     gt_poses:  list,   # list of dicts
                     save_path: str,
                     n_samples: int = 5):
    """生成并保存可视化 PNG"""
    K = len(contact_pts)
    n_cols = max(n_samples, len(gt_poses[:n_samples]))
    n_rows = K + 1  # K 个接触点视图 + 1 个总览

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle(f'GraspDiffusion v2  —  {obj_id}', fontsize=14, y=0.98)

    # ── 子图1：总览（点云 + 所有接触点 + 所有预测）──────────────
    ax_ov = fig.add_subplot(n_rows, 1, 1, projection='3d')
    _draw_pc_heatmap(ax_ov, pc_xyz, heatmap, alpha=0.3, s=1)

    # 所有接触点
    ax_ov.scatter(*contact_pts.T, c='orange', s=80, marker='*',
                  zorder=10, label='Contact pts')

    # 第一个候选的预测
    colors_pred = ['dodgerblue', 'cyan', 'deepskyblue']
    for ki, cp_poses in enumerate(pred_poses):
        c = colors_pred[ki % len(colors_pred)]
        pose = cp_poses[0]  # 取第 1 个 sample
        draw_gripper_arrow(ax_ov, pose['position'], pose['approach_dir'],
                           pose['finger_dir'], color=c, scale=0.04)

    # GT（前几个）
    for gp in gt_poses[:3]:
        draw_gripper_arrow(ax_ov, gp['position'], gp['approach_dir'],
                           gp['finger_dir'], color='lime', alpha=0.5, scale=0.04)

    ax_ov.set_title(f'Overview  —  {K} contacts  |  GT (green)  Pred (blue)')
    _set_ax_equal(ax_ov, pc_xyz)

    # ── 子图2~K+1：每个接触点的多个候选 ─────────────────────────
    colors_samples = ['dodgerblue', 'cyan', 'royalblue', 'steelblue', 'skyblue']
    for ki, (cp, cp_poses) in enumerate(zip(contact_pts, pred_poses)):
        for si, pose in enumerate(cp_poses[:n_samples]):
            ax = fig.add_subplot(n_rows, n_cols, n_cols * (ki + 1) + si + 1,
                                 projection='3d')
            _draw_pc_heatmap(ax, pc_xyz, heatmap, alpha=0.2, s=0.5)
            ax.scatter(*cp, c='orange', s=100, marker='*', zorder=10)
            draw_gripper_arrow(ax, pose['position'], pose['approach_dir'],
                               pose['finger_dir'], color=colors_samples[si], scale=0.04)
            # GT 对比
            if si < len(gt_poses):
                gp = gt_poses[si]
                draw_gripper_arrow(ax, gp['position'], gp['approach_dir'],
                                   gp['finger_dir'], color='lime', alpha=0.4, scale=0.04)

            ax.set_title(f'Contact {ki}  Sample {si}', fontsize=7)
            ax.axis('off')
            _set_ax_equal(ax, pc_xyz)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"  ✅ saved: {save_path}")


def _draw_pc_heatmap(ax, xyz, heatmap, alpha=0.3, s=1.0):
    """点云用 heatmap 着色（冷蓝→热红）"""
    from matplotlib.cm import get_cmap
    cmap = get_cmap('plasma')
    colors = cmap(heatmap)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=colors, s=s, alpha=alpha, linewidths=0)
    ax.set_facecolor('#0d0d1a')


def _set_ax_equal(ax, xyz):
    """3D 轴等比例"""
    ranges = xyz.max(0) - xyz.min(0)
    mid    = (xyz.max(0) + xyz.min(0)) / 2
    r = ranges.max() / 2 * 1.1
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    ax.set_xlabel('X', fontsize=6)
    ax.set_ylabel('Y', fontsize=6)
    ax.set_zlabel('Z', fontsize=6)
    ax.tick_params(labelsize=5)


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载模型
    print("\nLoading models...")
    v6        = load_v6(args.v6_ckpt, device)
    diffusion, rot_stats, pos_stats = load_diffusion(args.diff_ckpt, device)

    # 对象列表
    if args.obj:
        obj_ids = [args.obj]
    else:
        split   = json.load(open(args.split_json))
        obj_ids = split['val']   # 默认可视化验证集
    print(f"\nObjects to visualize: {obj_ids}")

    # 读 affordance_all.h5
    with h5py.File(args.aff_h5, 'r') as hf:
        all_ids  = [x.decode() if isinstance(x, bytes) else x
                    for x in hf['data']['obj_ids'][:]]
        aff_idx  = {oid: i for i, oid in enumerate(all_ids)}
        all_pts  = hf['data']['points'][:]
        all_nrm  = hf['data']['normals'][:]
        all_lab  = hf['data']['labels'][:]

    os.makedirs(args.save_dir, exist_ok=True)

    for oid in obj_ids:
        print(f"\n── {oid} ──")

        if oid not in aff_idx:
            print(f"  ⚠  {oid} not in affordance_all.h5, skip")
            continue

        idx     = aff_idx[oid]
        pc_xyz  = all_pts[idx].astype(np.float32)   # (4096,3)
        normals = all_nrm[idx].astype(np.float32)   # (4096,3)
        heatmap = all_lab[idx].astype(np.float32)   # (4096,)

        # 采接触点
        contact_pts = sample_contact_points(
            pc_xyz, heatmap, k=args.n_contacts, threshold=args.aff_thresh)
        print(f"  Contact pts sampled: {contact_pts.shape[0]}")

        # 推理
        pred_poses = predict_poses(
            v6, diffusion, rot_stats, pos_stats,
            pc_xyz, normals, contact_pts, device,
            n_samples=args.n_samples, ddim_steps=args.ddim_steps)

        # 读 GT Robot Posterior
        gt_poses = []
        mp = os.path.join(args.merged_dir, f"{oid}_robot_gt_merged.hdf5")
        if os.path.exists(mp):
            with h5py.File(mp, 'r') as hf:
                sg = hf.get('successful_grasps', {})
                for gk in list(sg.keys())[:args.n_samples]:
                    g  = sg[gk]
                    ep = g.get('executed_panda_hand_at_close')
                    if ep is None:
                        continue
                    R_exec   = ep['rotation'][:].astype(np.float32)
                    pos_exec = ep['position'][:].astype(np.float32)
                    fd_exec  = ep.get('finger_dir')
                    fd_exec  = fd_exec[:].astype(np.float32) if fd_exec else R_exec[:,0]
                    ap_exec  = ep.get('approach_dir')
                    ap_exec  = ap_exec[:].astype(np.float32) if ap_exec else R_exec[:,2]

                    pre = g.get('mesh_prerotation')
                    if pre and 'matrix' in pre:
                        R_pre    = pre['matrix'][:].astype(np.float32)
                        pos_exec = R_pre.T @ pos_exec
                        fd_exec  = R_pre.T @ fd_exec
                        ap_exec  = R_pre.T @ ap_exec

                    gt_poses.append({
                        'position':    pos_exec,
                        'approach_dir': ap_exec / (np.linalg.norm(ap_exec) + 1e-8),
                        'finger_dir':   fd_exec / (np.linalg.norm(fd_exec) + 1e-8),
                    })
            print(f"  GT poses loaded:     {len(gt_poses)}")

        # 生成图
        save_path = os.path.join(args.save_dir, f"{oid}.png")
        visualize_object(oid, pc_xyz, heatmap, contact_pts,
                         pred_poses, gt_poses, save_path,
                         n_samples=args.n_samples)

    print(f"\n完成！图片保存在 {args.save_dir}/")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='GraspDiffusion v2 可视化')
    p.add_argument('--diff_ckpt', required=True, help='diffusion best_model.pth')
    p.add_argument('--v6_ckpt',   required=True, help='V6 best_v6_model.pth')
    p.add_argument('--aff_h5',    required=True, help='affordance_all.h5')
    p.add_argument('--merged_dir',required=True, help='merged/ 目录')
    p.add_argument('--split_json',default=None,  help='split json (val set)')
    p.add_argument('--save_dir',  default='output/vis_diffusion_v2')
    p.add_argument('--obj',       default=None,  help='单物体 obj_id')
    p.add_argument('--n_contacts',type=int, default=3,  help='接触点数量')
    p.add_argument('--n_samples', type=int, default=5,  help='每接触点采样数')
    p.add_argument('--ddim_steps',type=int, default=50, help='DDIM 步数')
    p.add_argument('--aff_thresh',type=float, default=0.35, help='Affordance 阈值')
    args = p.parse_args()
    run(args)
