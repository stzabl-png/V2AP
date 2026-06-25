#!/usr/bin/env python3
"""
tools/vis_diffusion_v3.py — v3 接触点对 Diffusion 可视化

风格同 vis_diffusion_pred.py (v4 style):
  - 灰色点云（物体）
  - 10 个随机接触点，每个独立预测 1 个 delta
  - 从 delta 推导 (L, R, approach) 绘制夹爪

用法:
  python3 tools/vis_diffusion_v3.py \
      --ckpt_dir output/checkpoints_diffusion_v3 \
      --split val \
      --save_dir output/vis_v3
"""

import os, sys, json, argparse
import numpy as np
import torch
import h5py
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from model.pointnet2_v6 import PointNet2AffordanceV6
from model.grasp_diffusion_v3 import GraspDiffusionV3, TCP_OFFSET

BASE = 'data_hub/ProcessedData/RobotPosterior'


# ── 模型加载 ─────────────────────────────────────────────────

def load_models(ckpt_dir, device):
    pn2 = PointNet2AffordanceV6().to(device)
    pn2.load_state_dict(
        torch.load(f'{BASE}/best_v6_model.pth',
                   map_location=device, weights_only=False)['model_state_dict'])
    pn2.eval()

    diff = GraspDiffusionV3(T=1000, hidden=512, cond_dim=517).to(device)
    diff.load_state_dict(
        torch.load(f'{ckpt_dir}/best_model.pth',
                   map_location=device, weights_only=False)['model_state_dict'])
    diff.eval()

    ds = torch.load(f'{ckpt_dir}/delta_stats.pt',
                    map_location=device, weights_only=False)
    return pn2, diff, ds['delta_mean'], ds['delta_std']


# ── 接触点采样（labels × human_prior 联合得分）──────────────

def sample_contacts(pc, lab, hp, k=10):
    score = lab * 0.6 + hp * 0.4
    thresh = np.percentile(score, 95)
    mask = score >= thresh
    cand = pc[mask]
    cand_s = score[mask]
    if len(cand) < k:
        idx = np.argsort(score)[-k:]
        return pc[idx]
    idx = np.random.choice(len(cand), size=min(k, len(cand)), replace=False)
    return cand[idx]


# ── 局部法向估计 ─────────────────────────────────────────────

def local_normal(p, pc, normals, k=5):
    d = np.linalg.norm(pc - p[None], axis=-1)
    idx = np.argsort(d)[:k]
    w = 1.0 / (d[idx] + 1e-8); w /= w.sum()
    n = (normals[idx] * w[:, None]).sum(0)
    n /= np.linalg.norm(n) + 1e-8
    # 确保法向朝外
    centroid = pc.mean(0)
    if np.dot(n, p - centroid) < 0:
        n = -n
    return n


# ── V3 推理 ──────────────────────────────────────────────────

@torch.no_grad()
def predict_v3(pn2, diff, delta_mean, delta_std,
               pc, nrm, lab, hp, cps, device, ddim_steps=50):
    """
    cps: (k, 3)  接触中点（物体坐标系，metric）
    返回 list of dict: {L, R, approach, tcp}
    """
    pc_centroid = pc.mean(0)
    pc_radius   = float(np.linalg.norm(pc - pc_centroid, axis=1).max()) + 1e-6
    pc_norm     = (pc - pc_centroid) / pc_radius

    # V6 全局特征
    xt  = torch.from_numpy(pc).unsqueeze(0).to(device)
    nt  = torch.from_numpy(nrm).unsqueeze(0).to(device)
    gf  = pn2.extract_global_feat(xt, torch.cat([xt, nt], -1))  # (1, 512)

    tree = cKDTree(pc_norm)

    results = []
    for cp in cps:
        p_norm = (cp - pc_centroid) / pc_radius

        # hp / label at p
        _, nn_i  = tree.query(p_norm)
        hp_val   = float(hp[nn_i])
        lab_val  = float(lab[nn_i])

        # 条件向量
        p_t  = torch.from_numpy(p_norm.astype(np.float32)).unsqueeze(0).to(device)
        hl_t = torch.tensor([[hp_val, lab_val]], dtype=torch.float32, device=device)
        cond = torch.cat([gf, p_t, hl_t], dim=-1)   # (1, 517)

        # 采样 delta
        delta_n = diff.sample(cond, n_samples=1, ddim_steps=ddim_steps)  # (1, 3)
        delta   = (delta_n * delta_std + delta_mean).squeeze(0).cpu().numpy() * pc_radius

        # 推导接触点对
        L = cp - delta
        R = cp + delta

        # approach = 局部法向
        approach = local_normal(cp, pc, nrm, k=5)

        # wrist (TCP) = finger_mid - approach * TCP_OFFSET
        tcp = cp - approach * TCP_OFFSET

        results.append({'L': L, 'R': R, 'approach': approach, 'tcp': tcp, 'fmid': cp})
    return results


# ── 夹爪绘制 ─────────────────────────────────────────────────

def draw_gripper_v3(ax, pose, color, alpha=0.88):
    L = pose['L']; R = pose['R']
    fmid = pose['fmid']; approach = pose['approach']
    ap_len = TCP_OFFSET * 0.5

    # 接近方向箭头（从外向内）
    wrist = fmid - approach * ap_len
    ax.quiver(*wrist, *(approach * ap_len),
              color=color, alpha=alpha, linewidth=2.5, arrow_length_ratio=0.3)

    # finger bar：L ─── R
    ax.plot([L[0], R[0]], [L[1], R[1]], [L[2], R[2]],
            color=color, alpha=alpha, linewidth=3.0, solid_capstyle='round')

    # 指尖点
    ax.scatter(*L, color=color, s=45, zorder=8, alpha=alpha,
               edgecolors='white', linewidths=0.5)
    ax.scatter(*R, color=color, s=45, zorder=8, alpha=alpha,
               edgecolors='white', linewidths=0.5)

    # 接触中点
    ax.scatter(*fmid, color='white', s=20, zorder=9, alpha=0.9)


# ── 单对象绘制 ───────────────────────────────────────────────

def visualize_one(ax, oid, pc, nrm, poses, elev=25, azim=45):
    ax.set_facecolor('#0d0d1a')

    # 灰色点云
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
               c='#aaaaaa', s=4, alpha=0.5, linewidths=0, depthshade=False)

    # 10 个夹爪，彩虹色
    colors = plt.cm.rainbow(np.linspace(0, 1, len(poses)))
    for pose, color in zip(poses, colors):
        hex_c = '#{:02x}{:02x}{:02x}'.format(
            int(color[0]*255), int(color[1]*255), int(color[2]*255))
        draw_gripper_v3(ax, pose, hex_c, alpha=0.88)

    mid = (pc.max(0) + pc.min(0)) / 2
    r   = (pc.max(0) - pc.min(0)).max() / 2 * 1.35
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(oid, color='white', fontsize=10, fontweight='bold', pad=4)
    ax.tick_params(colors='#555', labelsize=5)
    ax.set_xlabel('X', color='#555', fontsize=6)
    ax.set_ylabel('Y', color='#555', fontsize=6)
    ax.set_zlabel('Z', color='#555', fontsize=6)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('#1a1a2e')


# ── 主程序 ───────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    pn2, diff, delta_mean, delta_std = load_models(args.ckpt_dir, device)

    split = json.load(open(f'{BASE}/min20/objects_train_val_split.json'))
    obj_ids = split['val'] if args.split == 'val' else split['train']
    if args.obj:
        obj_ids = [args.obj]

    with h5py.File(f'{BASE}/affordance_all.h5') as hf:
        raw_ids = [x.decode() if isinstance(x,bytes) else x
                   for x in hf['data/obj_ids'][:]]
        aff_idx  = {o:i for i,o in enumerate(raw_ids)}
        all_pts  = hf['data/points'][:]
        all_nrm  = hf['data/normals'][:]
        all_lab  = hf['data/labels'][:]
        all_hp   = hf['data/human_priors'][:]

    os.makedirs(args.save_dir, exist_ok=True)
    VIEWS = [(30, 45), (15, 160), (70, 270)]

    for oid in obj_ids:
        if oid not in aff_idx:
            print(f'  skip {oid}'); continue

        idx = aff_idx[oid]
        pc  = all_pts[idx].astype(np.float32)
        nrm = all_nrm[idx].astype(np.float32)
        lab = all_lab[idx].astype(np.float32)
        hp  = all_hp[idx].astype(np.float32)

        # 联合得分采样接触中点
        cps   = sample_contacts(pc, lab, hp, k=args.n_contacts)
        poses = predict_v3(pn2, diff, delta_mean, delta_std,
                           pc, nrm, lab, hp, cps, device,
                           ddim_steps=args.ddim_steps)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                                 subplot_kw={'projection': '3d'},
                                 facecolor='#080810')
        fig.suptitle(f'GraspDiffusion v3 (contact-pair) -- {oid}   '
                     f'({len(cps)} random contacts, L/R predicted)',
                     color='white', fontsize=12, y=1.01)

        for ax, (elev, azim) in zip(axes, VIEWS):
            visualize_one(ax, oid, pc, nrm, poses, elev=elev, azim=azim)

        plt.tight_layout(pad=1.0)
        out = os.path.join(args.save_dir, f'{oid}.png')
        plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='#080810')
        plt.close(fig)
        print(f'  done {oid} -> {out}')

    print(f'\nDone! -> {args.save_dir}/')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir',   default='output/checkpoints_diffusion_v3')
    p.add_argument('--split',      default='val', choices=['val','train','all'])
    p.add_argument('--obj',        default=None)
    p.add_argument('--save_dir',   default='output/vis_v3')
    p.add_argument('--n_contacts', type=int, default=10)
    p.add_argument('--ddim_steps', type=int, default=50)
    main(p.parse_args())
