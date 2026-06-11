#!/usr/bin/env python3
"""
vis_merged_batch.py — 批量 RobotPosterior 可视化（Isaac Sim 场景风格）

坐标系:
  - grasp_point / approach_dir / finger_dir: canonical mesh 局部坐标系
  - contact_points_local: 世界坐标偏移（坐标系与 mesh 不对齐，不使用）
  - 接触点从夹爪几何估算

场景布局 (与 Sim 一致, Z朝上):
  TABLE_TOP_Z = 0.80m   桌面顶部
  物体底部贴桌面，XY 居中
"""
import os, sys, glob, h5py
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'tools'))
from mesh_utils import load_mesh_canonical, find_ply

MERGED_DIR = os.path.join(PROJ, 'output', 'grasp_collect', 'merged')
OUT_DIR    = os.path.join(PROJ, 'output', 'vis_merged')
os.makedirs(OUT_DIR, exist_ok=True)

TABLE_TOP_Z = 0.80
TABLE_HALF  = 0.45
FLOOR_Z     = 0.0
VIEW_ELEV   = 28
VIEW_AZIM   = -50
MAX_GRIPPERS = 12   # 最多显示夹爪数


# ─── 场景元素 ─────────────────────────────────────────────────────────────────
def draw_floor(ax):
    s = 0.85
    verts = [[(-s,-s,FLOOR_Z),(s,-s,FLOOR_Z),(s,s,FLOOR_Z),(-s,s,FLOOR_Z)]]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.28,
        facecolor='#3a6bc9', edgecolor='none', zorder=0))
    for v in np.arange(-s, s+0.01, 0.20):
        ax.plot([v,v],[-s,s],[FLOOR_Z,FLOOR_Z], color='white', lw=0.35, alpha=0.55)
        ax.plot([-s,s],[v,v],[FLOOR_Z,FLOOR_Z], color='white', lw=0.35, alpha=0.55)


def draw_table(ax):
    h, tz, th = TABLE_HALF, TABLE_TOP_Z, 0.05
    # 不透明白色桌面顶
    top = [[(-h,-h,tz),(h,-h,tz),(h,h,tz),(-h,h,tz)]]
    ax.add_collection3d(Poly3DCollection(top, alpha=1.0,
        facecolor='#ffffff', edgecolor='#cccccc', linewidth=0.7, zorder=3))
    # 桌面侧面
    for (ax0,ay0),(ax1,ay1) in [
        ((-h,-h),(h,-h)), ((h,-h),(h,h)),
        ((h,h),(-h,h)),   ((-h,h),(-h,-h))]:
        side = [[(ax0,ay0,tz),(ax1,ay1,tz),(ax1,ay1,tz-th),(ax0,ay0,tz-th)]]
        ax.add_collection3d(Poly3DCollection(side, alpha=1.0,
            facecolor='#e8e8e8', edgecolor='#cccccc', linewidth=0.4, zorder=3))
    # 桌腿
    for sx,sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        lx, ly = sx*(TABLE_HALF-0.05), sy*(TABLE_HALF-0.05)
        ax.plot([lx,lx],[ly,ly],[FLOOR_Z,TABLE_TOP_Z-th],
            color='#bbbbbb', lw=4.0, alpha=0.7, solid_capstyle='round')


def draw_axes(ax, origin, length=0.10):
    ox, oy, oz = origin
    for vec, col, lbl in [
        ((length,0,0),'#e03030','X'),
        ((0,length,0),'#28a745','Y'),
        ((0,0,length),'#1a7fdd','Z'),
    ]:
        ax.quiver(ox,oy,oz,*vec, color=col, linewidth=2.0,
                  arrow_length_ratio=0.30, zorder=10)
        ax.text(ox+vec[0]*1.2, oy+vec[1]*1.2, oz+vec[2]*1.2,
                lbl, color=col, fontsize=8, fontweight='bold', zorder=11)


# ─── 夹爪 ─────────────────────────────────────────────────────────────────────
def draw_gripper(ax, grasp_point, approach_dir, finger_dir, width):
    ap = np.asarray(approach_dir, float); ap /= np.linalg.norm(ap)+1e-8
    fd = np.asarray(finger_dir,   float); fd /= np.linalg.norm(fd)+1e-8
    gp = np.asarray(grasp_point,  float)

    tip  = gp + ap * 0.09
    hw   = min(width, 0.08) / 2.0
    c1, c2 = tip + fd*hw, tip - fd*hw
    t1, t2 = c1 - ap*0.022, c2 - ap*0.022
    palm   = tip + ap*0.025

    kw = dict(color='#111111', linewidth=1.5, alpha=0.85, zorder=7)
    ax.plot3D(*zip(c1,c2), **kw)
    ax.plot3D(*zip(c1,t1), **kw)
    ax.plot3D(*zip(c2,t2), **kw)
    ax.plot3D(*zip(palm,gp), **kw)

    # 绿色接近方向箭头
    ax.quiver(*gp, *(ap*0.065), color='#00bb44', linewidth=1.6,
              alpha=0.95, arrow_length_ratio=0.32, zorder=8)

    # 返回估算接触点（两指尖）
    return [t1, t2]


# ─── 主可视化 ─────────────────────────────────────────────────────────────────
def vis_object(obj_id, grasps, mesh, out_path):
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax  = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.set_facecolor('white')

    # ── 场景 ──
    draw_floor(ax)
    draw_table(ax)
    draw_axes(ax, (-TABLE_HALF+0.04, -TABLE_HALF+0.04, TABLE_TOP_Z))

    # ── 物体点云 ──
    pts, _ = trimesh.sample.sample_surface(mesh, 8000)
    pts    = np.array(pts, dtype=float)

    z_min = pts[:,2].min()
    cx    = (pts[:,0].max() + pts[:,0].min()) / 2
    cy    = (pts[:,1].max() + pts[:,1].min()) / 2
    # 平移：底部贴桌面，XY 居中
    offset = np.array([-cx, -cy, TABLE_TOP_Z - z_min])
    pts_w  = pts + offset

    # 先画物体（底层，大点高不透明）
    ax.scatter(pts_w[:,0], pts_w[:,1], pts_w[:,2],
               s=3.5, c='#4477bb', alpha=0.50,
               edgecolors='none', rasterized=True, zorder=4)

    # ── 夹爪：超过 MAX_GRIPPERS 时随机采样 ──
    vis_grasps = grasps
    if len(grasps) > MAX_GRIPPERS:
        rng  = np.random.default_rng(42)
        idxs = rng.choice(len(grasps), MAX_GRIPPERS, replace=False)
        vis_grasps = [grasps[i] for i in sorted(idxs)]

    all_contacts = []
    for g in vis_grasps:
        gp_w = np.asarray(g['grasp_point'],  float) + offset
        ad   = np.asarray(g['approach_dir'], float)
        fd   = np.asarray(g['finger_dir'],   float)
        w    = float(g.get('width', 0.05))
        contacts = draw_gripper(ax, gp_w, ad, fd, w)
        all_contacts.extend(contacts)

    # ── 接触点（从几何估算，红色，后画覆盖最顶层）──
    if all_contacts:
        cp = np.array(all_contacts)
        # 过滤在桌面以下的
        valid = cp[:,2] > TABLE_TOP_Z - 0.005
        cp = cp[valid]
        if len(cp):
            ax.scatter(cp[:,0], cp[:,1], cp[:,2],
                       s=20, c='#cc1111', alpha=0.90,
                       edgecolors='none', depthshade=False, zorder=12)

    # ── 视角 & 范围 ──
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    lim = TABLE_HALF + 0.05
    obj_h = pts[:,2].max() - z_min   # 物体高度
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(FLOOR_Z - 0.02, TABLE_TOP_Z + obj_h + 0.25)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#e0e0e0')
    ax.tick_params(colors='#aaaaaa', labelsize=5)
    ax.set_xlabel('X', color='#aaaaaa', fontsize=6)
    ax.set_ylabel('Y', color='#aaaaaa', fontsize=6)
    ax.set_zlabel('Z', color='#aaaaaa', fontsize=6)
    ax.grid(False)

    # ── 图例 ──
    n_shown = len(vis_grasps)
    n_total = len(grasps)
    shown_str = f"(showing {n_shown}/{n_total})" if n_shown < n_total else f"({n_total})"
    legend_items = [
        mpatches.Patch(facecolor='#4477bb', alpha=0.6, label='Object'),
        mpatches.Patch(facecolor='#111111', label='Gripper'),
        mpatches.Patch(facecolor='#00bb44', label='Approach dir'),
        mpatches.Patch(facecolor='#cc1111', label='Contact point'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=7.5,
              framealpha=0.9, facecolor='white', edgecolor='#cccccc')

    scores = [g.get('score',0) for g in grasps]
    sc = f"  avg={np.mean(scores):.1f}" if scores else ""
    ax.set_title(f"{obj_id}   {n_total} grasps {shown_str}{sc}",
                 fontsize=11, fontweight='bold', color='#222222', pad=6)

    fig.savefig(out_path, dpi=130, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


# ─── 批量主程序 ───────────────────────────────────────────────────────────────
def main():
    files = sorted(glob.glob(os.path.join(MERGED_DIR, '*_robot_gt_merged.hdf5')))
    print(f"找到 {len(files)} 个 merged HDF5")
    ok = 0
    for fp in files:
        obj_id   = os.path.basename(fp).replace('_robot_gt_merged.hdf5','')
        out_path = os.path.join(OUT_DIR, f'{obj_id}.png')
        grasps = []
        try:
            with h5py.File(fp,'r') as f:
                if int(f.attrs.get('n_successful',0)) == 0:
                    print(f"  ⬛ {obj_id}: 0次，跳过"); continue
                for k in f.get('successful_grasps',{}).keys():
                    g = f['successful_grasps'][k]
                    grasps.append({
                        'grasp_point':  g['grasp_point'][:],
                        'approach_dir': g['approach_dir'][:],
                        'finger_dir':   g['finger_dir'][:],
                        'width':        float(g.attrs.get('gripper_width',0.05)),
                        'score':        float(g.attrs.get('score',0)),
                    })
        except Exception as e:
            print(f"  ❌ {obj_id}: {e}"); continue
        try:
            ds   = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
            mesh = load_mesh_canonical(obj_id, ds, verbose=False)
        except Exception as e:
            print(f"  ⚠️  {obj_id}: mesh失败 {e}"); continue
        try:
            vis_object(obj_id, grasps, mesh, out_path)
            print(f"  ✅ {obj_id}: {len(grasps)} grasps")
            ok += 1
        except Exception as e:
            print(f"  ❌ {obj_id}: 可视化失败 {e}")

    print(f"\n完成! {ok} 个物体 → {OUT_DIR}")


if __name__ == '__main__':
    main()
