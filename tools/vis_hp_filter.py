#!/usr/bin/env python3
"""
vis_hp_filter.py
可视化 HP 过滤结果：
  - 物体灰色半透明点云
  - HP 区域渐变红色（越红 HP 越高）
  - 所有抓取的接触点（蓝色球）+ 同一 pose 两点连线
  - HP✅ 用实线，HP❌ 用虚线（颜色相同，通过 hp_match 的球颜色区分）

用法:
    python3 tools/vis_hp_filter.py                  # 批量生成全部物体
    python3 tools/vis_hp_filter.py --obj C28001     # 单个物体
    python3 tools/vis_hp_filter.py --obj C28001 --show  # 交互预览
"""
import os, sys, glob, argparse, h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HP_DIRS = {
    'oakink': os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp', 'oakink'),
    'dexycb': os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp', 'dexycb'),
}
FILTERED_DIR = os.path.join(PROJ, 'output', 'RobotPosteriorFiltered')
OUT_DIR      = os.path.join(PROJ, 'output', 'vis_hp_filter')
os.makedirs(OUT_DIR, exist_ok=True)


def load_hp(obj_id):
    ds = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
    fp = os.path.join(HP_DIRS[ds], f'{obj_id}.hdf5')
    if not os.path.exists(fp):
        return None, None
    with h5py.File(fp, 'r') as f:
        pc = f['point_cloud'][()].astype(np.float32)
        hp = f['human_prior'][()].astype(np.float32)
    return pc, hp


def load_grasps(obj_id):
    fp = os.path.join(FILTERED_DIR, f'{obj_id}_robot_gt.hdf5')
    if not os.path.exists(fp):
        return [], 0, 0
    grasps = []
    with h5py.File(fp, 'r') as f:
        n_match    = int(f.attrs.get('n_hp_match', 0))
        n_mismatch = int(f.attrs.get('n_hp_mismatch', 0))
        sg = f.get('successful_grasps', {})
        for key in sg.keys():
            g = sg[key]
            gp = g['grasp_point'][:]
            fd = g['finger_dir'][:]
            w  = float(g.attrs.get('gripper_width', 0.05))
            hp_match = bool(g.attrs.get('hp_match', False))
            hp_score = float(g.attrs.get('hp_score', 0.0))
            # 接触点
            hw = w / 2.0
            fd_n = fd / (np.linalg.norm(fd) + 1e-8)
            c_L = gp + fd_n * hw
            c_R = gp - fd_n * hw
            grasps.append({
                'c_L': c_L, 'c_R': c_R,
                'hp_match': hp_match,
                'hp_score': hp_score,
            })
    return grasps, n_match, n_mismatch


def visualize(obj_id, show=False):
    # ── 加载数据 ────────────────────────────────────────
    pc, hp_labels = load_hp(obj_id)
    if pc is None:
        print(f'  warning {obj_id}: no HP data')
        return
    grasps, n_match, n_mismatch = load_grasps(obj_id)
    n_total = n_match + n_mismatch
    match_pct = n_match / n_total * 100 if n_total else 0

    # ── 归一化坐标到 [-1, 1]（点云中心+缩放） ──────────
    center = pc.mean(axis=0)
    scale  = np.abs(pc - center).max() + 1e-8   # 最大绝对值
    pc_n   = (pc - center) / scale              # 归一化点云

    hp_norm = np.clip(hp_labels, 0, 1)

    # ── 绘图 ────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8), facecolor='white')
    ax  = fig.add_subplot(111, projection='3d', facecolor='white')

    # 1a. 灰色背景点（低 HP）
    bg = hp_norm < 0.3
    if bg.any():
        ax.scatter(pc_n[bg,0], pc_n[bg,1], pc_n[bg,2],
                   c='#b8b8b8', s=2, alpha=0.25, linewidths=0, depthshade=True)

    # 1b. 红色高 HP 区域（大点 + 深颜色）
    fg = hp_norm >= 0.3
    if fg.any():
        reds = plt.cm.Reds(0.35 + 0.65 * hp_norm[fg])
        ax.scatter(pc_n[fg,0], pc_n[fg,1], pc_n[fg,2],
                   c=reds, s=20, alpha=0.9, linewidths=0,
                   depthshade=False, zorder=4)

    # 2. 接触点（归一化后）+ 连线
    match_pts_L, match_pts_R, match_lines = [], [], []
    for g in grasps:
        if g['hp_match']:
            cL = (g['c_L'] - center) / scale
            cR = (g['c_R'] - center) / scale
            match_pts_L.append(cL)
            match_pts_R.append(cR)
            match_lines.append([cL, cR])

    if match_pts_L:
        ml = np.array(match_pts_L)
        mr = np.array(match_pts_R)
        ax.scatter(ml[:,0], ml[:,1], ml[:,2], c='royalblue', s=600,
                   zorder=15, depthshade=False, label=f'HP-match ({n_match})',
                   edgecolors='white', linewidths=2.0)
        ax.scatter(mr[:,0], mr[:,1], mr[:,2], c='royalblue', s=600,
                   zorder=15, depthshade=False,
                   edgecolors='white', linewidths=2.0)
        segs = np.array(match_lines)
        lc = Line3DCollection(segs, colors='dodgerblue', linewidths=3.0, alpha=0.9)
        ax.add_collection(lc)

    # ── 坐标轴 & 标注 ───────────────────────────────────
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_zlabel('Z', fontsize=9)
    ax.set_title(
        f'{obj_id}  |  Total:{n_total}  HP-match:{n_match}({match_pct:.0f}%)  HP-miss:{n_mismatch}',
        fontsize=12, fontweight='bold', pad=12
    )

    # 图例
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='#c0c0c0', alpha=0.6, label='Object PC'),
        Patch(facecolor='#ff2222', alpha=0.8, label='HP Region'),
        Line2D([0],[0], color='royalblue', linewidth=2.5, label=f'HP-match {n_match}'),
    ]
    ax.legend(handles=legend_elems, loc='upper left', fontsize=8,
              framealpha=0.85)

    # colorbar for HP
    sm = plt.cm.ScalarMappable(
        cmap=mcolors.LinearSegmentedColormap.from_list('grayred',['#c0c0c0','#ff2020']),
        norm=plt.Normalize(0, 1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Human Prior', fontsize=8)

    ax.view_init(elev=25, azim=45)
    plt.tight_layout()

    if show:
        matplotlib.use('TkAgg')
        plt.show()
    else:
        out_path = os.path.join(OUT_DIR, f'{obj_id}.png')
        plt.savefig(out_path, dpi=130, bbox_inches='tight',
                    facecolor='white')
        plt.close()
        print(f'  OK {obj_id}: HP-match:{n_match}({match_pct:.0f}%) HP-miss:{n_mismatch} -> {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type=str, default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    if args.obj:
        visualize(args.obj, show=args.show)
        return

    files = sorted(glob.glob(os.path.join(FILTERED_DIR, '*_robot_gt.hdf5')))
    obj_ids = [os.path.basename(f).replace('_robot_gt.hdf5','') for f in files]
    print(f'批量生成 {len(obj_ids)} 个物体可视化 → {OUT_DIR}\n')
    for obj_id in obj_ids:
        visualize(obj_id)
    print(f'\n全部完成！图片保存在: {OUT_DIR}')


if __name__ == '__main__':
    main()
