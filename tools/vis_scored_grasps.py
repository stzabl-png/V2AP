#!/usr/bin/env python3
"""
tools/vis_scored_grasps.py — V3 Diffusion 预测 + 评分 + 最佳候选可视化

流程:
  1. 加载 V3 模型
  2. 对每个物体采样 50 个接触点，预测 delta
  3. 用 GraspScorer 评分排序
  4. 可视化: 灰色淘汰候选 + 彩色 Top-5 + 绿色最佳

用法:
  python3 tools/vis_scored_grasps.py \
      --ckpt_dir output/checkpoints_diffusion_v3 \
      --split val \
      --save_dir output/vis_scored
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
from model.grasp_scorer import GraspScorer, ScoredGrasp

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


# ── 接触点采样 ───────────────────────────────────────────────

def sample_contacts(pc, lab, hp, k=50):
    score = lab * 0.6 + hp * 0.4
    thresh = np.percentile(score, 90)
    mask = score >= thresh
    cand = pc[mask]
    if len(cand) < k:
        idx = np.argsort(score)[-k:]
        return pc[idx]
    idx = np.random.choice(len(cand), size=min(k, len(cand)), replace=False)
    return cand[idx]


# ── 局部法向 ─────────────────────────────────────────────────

def local_normal(p, pc, normals, k=5):
    d = np.linalg.norm(pc - p[None], axis=-1)
    idx = np.argsort(d)[:k]
    w = 1.0 / (d[idx] + 1e-8); w /= w.sum()
    n = (normals[idx] * w[:, None]).sum(0)
    n /= np.linalg.norm(n) + 1e-8
    centroid = pc.mean(0)
    if np.dot(n, p - centroid) < 0:
        n = -n
    return n


# ── V3 推理 ──────────────────────────────────────────────────

@torch.no_grad()
def predict_all(pn2, diff, delta_mean, delta_std,
                pc, nrm, lab, hp, cps, device, ddim_steps=50):
    pc_centroid = pc.mean(0)
    pc_radius   = float(np.linalg.norm(pc - pc_centroid, axis=1).max()) + 1e-6
    pc_norm     = (pc - pc_centroid) / pc_radius

    xt  = torch.from_numpy(pc).unsqueeze(0).to(device)
    nt  = torch.from_numpy(nrm).unsqueeze(0).to(device)
    gf  = pn2.extract_global_feat(xt, torch.cat([xt, nt], -1))

    tree = cKDTree(pc_norm)

    candidates = []
    for cp in cps:
        p_norm = (cp - pc_centroid) / pc_radius
        _, nn_i  = tree.query(p_norm)
        hp_val   = float(hp[nn_i])
        lab_val  = float(lab[nn_i])

        p_t  = torch.from_numpy(p_norm.astype(np.float32)).unsqueeze(0).to(device)
        hl_t = torch.tensor([[hp_val, lab_val]], dtype=torch.float32, device=device)
        cond = torch.cat([gf, p_t, hl_t], dim=-1)

        delta_n = diff.sample(cond, n_samples=1, ddim_steps=ddim_steps)
        delta   = (delta_n * delta_std + delta_mean).squeeze(0).cpu().numpy() * pc_radius

        L = cp - delta
        R = cp + delta
        width = 2.0 * float(np.linalg.norm(delta))
        approach = local_normal(cp, pc, nrm, k=5)

        candidates.append({
            'L': L, 'R': R,
            'approach': approach,
            'finger_mid': cp.copy(),
            'width': width,
        })
    return candidates


# ── 夹爪绘制 ─────────────────────────────────────────────────

def draw_gripper(ax, cand, color, alpha=0.9, lw=3.0, s=50):
    L = cand.get('L', cand.get('finger_mid'))
    R = cand.get('R', cand.get('finger_mid'))
    fmid = cand.get('finger_mid', cand.get('fmid'))
    approach = cand['approach']

    ap_len = TCP_OFFSET * 0.5
    wrist = fmid - approach * ap_len
    ax.quiver(*wrist, *(approach * ap_len),
              color=color, alpha=alpha, linewidth=lw, arrow_length_ratio=0.3)
    ax.plot([L[0], R[0]], [L[1], R[1]], [L[2], R[2]],
            color=color, alpha=alpha, linewidth=lw, solid_capstyle='round')
    ax.scatter(*L, color=color, s=s, zorder=8, alpha=alpha,
               edgecolors='white', linewidths=0.5)
    ax.scatter(*R, color=color, s=s, zorder=8, alpha=alpha,
               edgecolors='white', linewidths=0.5)


def draw_scored_gripper(ax, sg, color, alpha, lw, s):
    """从 ScoredGrasp 绘制"""
    cand = {
        'L': sg.L if sg.L is not None else sg.finger_mid,
        'R': sg.R if sg.R is not None else sg.finger_mid,
        'finger_mid': sg.finger_mid,
        'approach': sg.approach,
    }
    draw_gripper(ax, cand, color, alpha, lw, s)


# ── 单物体可视化 ─────────────────────────────────────────────

def visualize_scored(oid, pc, nrm, ranked, save_path):
    VIEWS = [(30, 45), (15, 160), (70, 270)]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7),
                             subplot_kw={'projection': '3d'},
                             facecolor='#080810')

    # 统计
    n_valid = sum(1 for s in ranked if not s.rejected)
    n_reject = sum(1 for s in ranked if s.rejected)
    best = ranked[0] if ranked and not ranked[0].rejected else None

    title = (f'{oid}  |  {len(ranked)} candidates  |  '
             f'{n_valid} valid  {n_reject} rejected')
    if best:
        title += f'  |  Best #{best.index} = {best.total_score:.1f}'

    fig.suptitle(title, color='white', fontsize=12, y=1.02, fontweight='bold')

    for ax, (elev, azim) in zip(axes, VIEWS):
        ax.set_facecolor('#0d0d1a')

        # 灰色点云
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
                   c='#aaaaaa', s=4, alpha=0.45, linewidths=0, depthshade=False)

        # 淘汰候选: 暗红色，透明
        for sg in ranked:
            if sg.rejected:
                draw_scored_gripper(ax, sg, '#661111', alpha=0.25, lw=1.0, s=15)

        # 有效候选排名 6+: 暗灰色
        valid = [s for s in ranked if not s.rejected]
        for sg in valid[5:]:
            draw_scored_gripper(ax, sg, '#444466', alpha=0.3, lw=1.2, s=18)

        # Top 5: 彩色（蓝→紫渐变）
        top5_colors = ['#6699ff', '#8877ee', '#aa55dd', '#cc44cc', '#dd33aa']
        for i, sg in enumerate(valid[1:5]):
            draw_scored_gripper(ax, sg, top5_colors[i], alpha=0.7, lw=2.0, s=35)

        # 最佳: 亮绿色，加粗
        if best:
            draw_scored_gripper(ax, best, '#00ff88', alpha=1.0, lw=4.0, s=70)
            # 标注分数
            ax.text(best.finger_mid[0], best.finger_mid[1],
                    best.finger_mid[2] + 0.015,
                    f'{best.total_score:.0f}',
                    color='#00ff88', fontsize=8, fontweight='bold',
                    ha='center', va='bottom')

        mid = (pc.max(0) + pc.min(0)) / 2
        r   = (pc.max(0) - pc.min(0)).max() / 2 * 1.4
        ax.set_xlim(mid[0]-r, mid[0]+r)
        ax.set_ylim(mid[1]-r, mid[1]+r)
        ax.set_zlim(mid[2]-r, mid[2]+r)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(colors='#555', labelsize=5)
        ax.set_xlabel('X', color='#555', fontsize=6)
        ax.set_ylabel('Y', color='#555', fontsize=6)
        ax.set_zlabel('Z', color='#555', fontsize=6)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor('#1a1a2e')

    # 底部评分表格
    if valid:
        table_text = (f'  Top-5 Scores:  '
                      + '  |  '.join(
                          f'#{s.index}: {s.total_score:.1f} '
                          f'(dir={s.direction_score:.0f} ctr={s.centroid_score:.0f} '
                          f'hgt={s.height_score:.0f} wid={s.width_score:.0f})'
                          for s in valid[:5]))
        fig.text(0.5, -0.02, table_text, ha='center', color='#88aacc',
                 fontsize=8, fontfamily='monospace')

    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=140, bbox_inches='tight', facecolor='#080810')
    plt.close(fig)


# ── 主程序 ───────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    pn2, diff, delta_mean, delta_std = load_models(args.ckpt_dir, device)
    scorer = GraspScorer()

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

    for oid in obj_ids:
        if oid not in aff_idx:
            print(f'  skip {oid}'); continue

        idx = aff_idx[oid]
        pc  = all_pts[idx].astype(np.float32)
        nrm = all_nrm[idx].astype(np.float32)
        lab = all_lab[idx].astype(np.float32)
        hp  = all_hp[idx].astype(np.float32)

        # 采样多个接触点
        cps = sample_contacts(pc, lab, hp, k=args.n_candidates)
        candidates = predict_all(pn2, diff, delta_mean, delta_std,
                                 pc, nrm, lab, hp, cps, device,
                                 ddim_steps=args.ddim_steps)

        # 评分排序
        ranked = scorer.rank(candidates, pc)

        # 打印报告
        print(f'\n[{oid}]')
        GraspScorer.print_report(ranked, top_n=5)

        # 可视化
        out = os.path.join(args.save_dir, f'{oid}.png')
        visualize_scored(oid, pc, nrm, ranked, out)
        print(f'  → {out}')

    print(f'\nDone! → {args.save_dir}/')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir',      default='output/checkpoints_diffusion_v3')
    p.add_argument('--split',         default='val', choices=['val','train','all'])
    p.add_argument('--obj',           default=None)
    p.add_argument('--save_dir',      default='output/vis_scored')
    p.add_argument('--n_candidates',  type=int, default=50)
    p.add_argument('--ddim_steps',    type=int, default=50)
    main(p.parse_args())
