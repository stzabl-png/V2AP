#!/usr/bin/env python3
"""
tools/vis_diffusion_pred.py — 预测抓取位姿可视化（无 GT）

每个对象生成一张图：
  - 点云按 Affordance heatmap 着色
  - 3 个接触点（橙星）
  - 每个接触点 5 个预测夹爪（平行夹爪示意图）

用法:
  python3 tools/vis_diffusion_pred.py \
      --ckpt_dir output/checkpoints_diffusion_v2_geoloss \
      --split val \
      --save_dir output/vis_pred_geoloss
"""

import os, sys, json, argparse
import numpy as np
import torch
import h5py
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import get_cmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pointnet2_v6 import PointNet2AffordanceV6
from model.grasp_diffusion_v2 import (
    GraspDiffusionV2, rotation_from_6d, GRIPPER_HW, TCP_OFFSET
)

BASE = 'data_hub/ProcessedData/RobotPosterior'
OBJ_MESHES = 'data_hub/ProcessedData/obj_meshes'
DATASETS   = ['oakink', 'ycb', 'dexycb', 'arctic', 'egocentric', 'ho3d_v3']
CMAP = get_cmap('plasma')


def find_mesh(obj_id):
    """查找物体网格文件"""
    for ds in DATASETS:
        p = os.path.join(OBJ_MESHES, ds, obj_id, 'mesh.ply')
        if os.path.exists(p):
            return p
    return None


# ── 模型加载 ───────────────────────────────────────────────────

def load_models(ckpt_dir, device):
    pn2 = PointNet2AffordanceV6().to(device)
    pn2.load_state_dict(
        torch.load(f'{BASE}/best_v6_model.pth',
                   map_location=device, weights_only=False)['model_state_dict'])
    pn2.eval()

    diff = GraspDiffusionV2(T=1000, hidden=512).to(device)
    ckpt = torch.load(f'{ckpt_dir}/best_model.pth',
                      map_location=device, weights_only=False)
    diff.load_state_dict(ckpt['model_state_dict'])
    diff.eval()

    rs = torch.load(f'{ckpt_dir}/rot_stats.pt', map_location=device, weights_only=False)
    ps = torch.load(f'{ckpt_dir}/pos_stats.pt', map_location=device, weights_only=False)
    return pn2, diff, rs['rot_mean'], rs['rot_std'], ps['pos_mean'], ps['pos_std']


# ── 接触点采样 ─────────────────────────────────────────────────

def sample_contacts(pc, lab, k=10):
    """top-5% affordance 区域随机采 k 个接触点"""
    thresh = np.percentile(lab, 95)
    mask   = lab >= thresh
    cand   = pc[mask]
    if len(cand) < k:
        cand = pc[np.argsort(lab)[-k:]]
    idx = np.random.choice(len(cand), size=min(k, len(cand)), replace=False)
    return cand[idx]


# ── 夹爪绘制（平行夹爪）────────────────────────────────────────

def draw_gripper(ax, tcp, approach, finger, color, alpha=0.9, scale=1.0):
    """
    平行夹爪示意图:
      wrist ──→ tcp  (approach 方向箭头)
      |──── finger bar ────|  (finger_dir 方向横杆)
      ●               ●   (两个指尖点)
    """
    hw     = GRIPPER_HW * scale
    ap_len = TCP_OFFSET * 0.5 * scale

    # approach 箭头（腼部 → TCP）
    wrist = tcp - approach * ap_len
    ax.quiver(*wrist, *(approach * ap_len),
              color=color, alpha=alpha,
              linewidth=2.5, arrow_length_ratio=0.3)

    # finger 横杆
    tip_l = tcp - finger * hw
    tip_r = tcp + finger * hw
    ax.plot([tip_l[0], tip_r[0]],
            [tip_l[1], tip_r[1]],
            [tip_l[2], tip_r[2]],
            color=color, alpha=alpha, linewidth=3.0, solid_capstyle='round')

    # 指尖圆点
    ax.scatter(*tip_l, c=color, s=40, zorder=8, alpha=alpha)
    ax.scatter(*tip_r, c=color, s=40, zorder=8, alpha=alpha)
    # TCP 中心点
    ax.scatter(*tcp, c='white', s=20, zorder=9, alpha=0.9)


# ── 主可视化函数 ───────────────────────────────────────────────

def visualize_one(ax, oid, pc, nrm, lab, cps, pred_poses,
                  elev=25, azim=45):
    ax.set_facecolor('#0d0d1a')

    # ── 物体：灰色点云 ──
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
               c='#aaaaaa', s=4, alpha=0.5, linewidths=0, depthshade=False)

    # ── 10 个预测，每个一个随机颜色 ──
    colors = plt.cm.rainbow(np.linspace(0, 1, len(cps)))
    for cp, pose, color in zip(cps, pred_poses, colors):
        hex_c = '#{:02x}{:02x}{:02x}'.format(
            int(color[0]*255), int(color[1]*255), int(color[2]*255))
        # 接触点小点
        ax.scatter(*cp, color=hex_c, s=60, marker='o', zorder=10,
                   edgecolors='white', linewidths=0.5)
        draw_gripper(ax, tcp=cp, approach=pose['approach'],
                     finger=pose['finger'], color=hex_c, alpha=0.88)

    # ── 视角 + 轴 ──
    mid = (pc.max(0) + pc.min(0)) / 2
    r   = (pc.max(0) - pc.min(0)).max() / 2 * 1.35
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f'{oid}', color='white', fontsize=10,
                 fontweight='bold', pad=4)
    ax.tick_params(colors='#555', labelsize=5)
    ax.set_xlabel('X', color='#555', fontsize=6)
    ax.set_ylabel('Y', color='#555', fontsize=6)
    ax.set_zlabel('Z', color='#555', fontsize=6)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#1a1a2e')


# ── 推理函数 ───────────────────────────────────────────────────

@torch.no_grad()
def predict(pn2, diff, rot_mean, rot_std, pos_mean, pos_std,
            pc, nrm, cps, device, ddim_steps=50):
    """cps: (k,3), 每个接触点独立采样 1 个 pose"""
    pc_centroid = pc.mean(0)
    pc_radius   = float(np.linalg.norm(pc - pc_centroid, axis=1).max()) + 1e-6

    xt = torch.from_numpy(pc).unsqueeze(0).to(device)
    nt = torch.from_numpy(nrm).unsqueeze(0).to(device)
    gf = pn2.extract_global_feat(xt, torch.cat([xt, nt], -1))

    result = []
    for cp in cps:
        pos_rel  = (cp - pc_centroid) / pc_radius
        pos_norm = (torch.from_numpy(pos_rel).unsqueeze(0).to(device) - pos_mean) / pos_std
        r6d_norm = diff.sample(gf, pos_norm, n_samples=1, ddim_steps=ddim_steps)
        r6d = r6d_norm * rot_std + rot_mean
        R   = rotation_from_6d(r6d).cpu().numpy()[0]   # (3,3)
        ap  = R[:, 2]; ap /= np.linalg.norm(ap) + 1e-8
        fd  = R[:, 0]; fd /= np.linalg.norm(fd) + 1e-8
        result.append({'approach': ap, 'finger': fd})
    return result


# ── 主程序 ─────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    pn2, diff, rot_mean, rot_std, pos_mean, pos_std = load_models(args.ckpt_dir, device)

    # 对象列表
    split = json.load(open(f'{BASE}/min20/objects_train_val_split.json'))
    if args.split == 'val':
        obj_ids = split['val']
    elif args.split == 'train':
        obj_ids = split['train']
    else:
        obj_ids = split['val'] + split['train']
    if args.obj:
        obj_ids = [args.obj]

    # 点云数据
    with h5py.File(f'{BASE}/affordance_all.h5') as hf:
        raw_ids = [x.decode() if isinstance(x,bytes) else x
                   for x in hf['data/obj_ids'][:]]
        aff_idx  = {o:i for i,o in enumerate(raw_ids)}
        all_pts  = hf['data/points'][:]
        all_nrm  = hf['data/normals'][:]
        all_lab  = hf['data/labels'][:]

    os.makedirs(args.save_dir, exist_ok=True)
    VIEWS = [(30, 45), (15, 160), (70, 270)]

    for oid in obj_ids:
        if oid not in aff_idx:
            print(f'  skip {oid}'); continue

        idx = aff_idx[oid]
        pc  = all_pts[idx].astype(np.float32)
        nrm = all_nrm[idx].astype(np.float32)
        lab = all_lab[idx].astype(np.float32)

        # 10 个随机接触点，每个独立预测 1 个 pose
        cps        = sample_contacts(pc, lab, k=args.n_contacts)
        pred_poses = predict(pn2, diff, rot_mean, rot_std, pos_mean, pos_std,
                             pc, nrm, cps, device, ddim_steps=args.ddim_steps)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                                 subplot_kw={'projection': '3d'},
                                 facecolor='#080810')
        fig.suptitle(f'GraspDiffusion v2  --  {oid}   '
                     f'({len(cps)} random contact points, 1 pred each)',
                     color='white', fontsize=12, y=1.01)

        for ax, (elev, azim) in zip(axes, VIEWS):
            visualize_one(ax, oid, pc, nrm, lab, cps, pred_poses,
                          elev=elev, azim=azim)

        plt.tight_layout(pad=1.0)
        out = os.path.join(args.save_dir, f'{oid}.png')
        plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='#080810')
        plt.close(fig)
        print(f'  done {oid} -> {out}')

    print(f'\nDone! Saved to {args.save_dir}/')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir',   default='output/checkpoints_diffusion_v2_geoloss')
    p.add_argument('--split',      default='val', choices=['val','train','all'])
    p.add_argument('--obj',        default=None)
    p.add_argument('--save_dir',   default='output/vis_pred_geoloss')
    p.add_argument('--n_contacts', type=int, default=10)
    p.add_argument('--n_samples',  type=int, default=1)
    p.add_argument('--ddim_steps', type=int, default=50)
    main(p.parse_args())
