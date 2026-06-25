#!/usr/bin/env python3
"""
vis_coord_check.py — 坐标系一致性验证工具
==========================================
对同一个物体的三个阶段分别可视化，每个阶段都显示：
  - 物体 mesh（canonical 坐标系，与 USD/Sim 一致）
  - 物体局部坐标轴（RGB = XYZ，长度 = bbox 的 40%）
  - 抓取位姿（接触点 + approach 箭头 + 夹爪横杆）

用法:
    python3 tools/vis_coord_check.py --obj A16013 --stage candidates
    python3 tools/vis_coord_check.py --obj A16013 --stage robot_gt
    python3 tools/vis_coord_check.py --obj A16013 --stage all
"""
import os, sys, argparse, json
import numpy as np
import trimesh
import h5py
import open3d as o3d

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'tools'))
from mesh_utils import load_mesh_canonical, find_ply, get_canonical_euler

# 候选文件搜索路径（按优先级）
CANDIDATE_DIRS = [
    os.path.join(PROJ, 'output', 'grasps_candidate_r3'),
    os.path.join(PROJ, 'output', 'grasps_candidate'),
    os.path.join(PROJ, 'output', 'grasps_random'),
]
GT_DIRS = [
    os.path.join(PROJ, 'output', 'robot_gt_merged_oakink'),
    os.path.join(PROJ, 'output', 'robot_gt_merged_dexycb'),
    os.path.join(PROJ, 'output', 'robot_gt_r3'),
]

# Sim 世界坐标系中物体的位置（用于标注说明）
OBJ_WORLD_POS  = np.array([0.0, 0.55, 0.80])
ROBOT_WORLD_POS = np.array([0.2, -0.05, 0.80])


# ── Open3D 辅助函数 ──────────────────────────────────────────────────────────

def make_axes(origin, scale=0.05):
    """
    在 origin 处画粗体 XYZ 坐标轴箭头（红=X, 绿=Y, 蓝=Z）。
    每个轴 = 圆柱体身 + 圆锥箭头。
    """
    geoms = []
    colors = [[1,0,0], [0,0.85,0], [0,0.3,1]]
    dirs   = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]

    shaft_r  = scale * 0.045   # 轴身半径
    shaft_l  = scale * 0.78    # 轴身长度
    cone_r   = scale * 0.12    # 箭头半径
    cone_l   = scale * 0.22    # 箭头长度

    from scipy.spatial.transform import Rotation as _R

    for d, c in zip(dirs, colors):
        # ── 旋转: Z轴 → 目标方向 ──────────────────────────────────────────
        z = np.array([0.0, 0.0, 1.0])
        if np.allclose(d, z):
            rot = np.eye(3)
        elif np.allclose(d, -z):
            rot = _R.from_euler('x', 180, degrees=True).as_matrix()
        else:
            axis = np.cross(z, d); axis /= np.linalg.norm(axis)
            angle = np.arccos(np.clip(np.dot(z, d), -1, 1))
            rot = _R.from_rotvec(axis * angle).as_matrix()

        # ── 圆柱体（轴身）────────────────────────────────────────────────
        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=shaft_r, height=shaft_l, resolution=16)
        cyl.rotate(rot, center=[0,0,0])
        cyl.translate(origin + d * shaft_l/2)
        cyl.paint_uniform_color(c)
        cyl.compute_vertex_normals()
        geoms.append(cyl)

        # ── 圆锥（箭头）──────────────────────────────────────────────────
        cone = o3d.geometry.TriangleMesh.create_cone(
            radius=cone_r, height=cone_l, resolution=16)
        cone.rotate(rot, center=[0,0,0])
        cone.translate(origin + d * (shaft_l + cone_l/2))
        cone.paint_uniform_color(c)
        cone.compute_vertex_normals()
        geoms.append(cone)

    return geoms


def make_sphere(pos, r, color):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    s.translate(pos); s.paint_uniform_color(color); s.compute_vertex_normals()
    return s

def make_line(p0, p1, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector([p0, p1])
    ls.lines  = o3d.utility.Vector2iVector([[0,1]])
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls

def make_transparent_mesh(mesh_tri, n_surface=3000, n_wire_faces=800):
    """
    用两层模拟透明效果:
      1. 稀疏表面点云 (淡灰, 3000点) → 看得见形状，也看得透内部
      2. 极简线框 (深灰, 简化后800面) → 显示轮廓
    """
    geoms = []

    # ── 1. 稀疏表面点云（透明感）──────────────────────────────────────────
    pts, _ = trimesh.sample.sample_surface(mesh_tri, n_surface)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.75, 0.78, 0.85])   # 淡蓝灰
    geoms.append(pcd)

    # ── 2. 极简线框（轮廓）────────────────────────────────────────────────
    try:
        simple = mesh_tri.simplify_quadric_decimation(n_wire_faces)
    except Exception:
        simple = mesh_tri
    o3d_m = o3d.geometry.TriangleMesh()
    o3d_m.vertices  = o3d.utility.Vector3dVector(simple.vertices)
    o3d_m.triangles = o3d.utility.Vector3iVector(simple.faces)
    o3d_m.compute_vertex_normals()
    wf = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_m)
    wf.paint_uniform_color([0.45, 0.45, 0.50])    # 中灰轮廓
    geoms.append(wf)

    return geoms


def draw_grasp(pos, rot33, gripper_w, color, geoms):
    """画单个抓取: 接触点红蓝球 + 夹爪横杆 + approach 箭头 + 手腕橙球."""
    bb = np.ptp(np.array([0]), axis=0)  # dummy
    # approach = rot[:,2], finger = rot[:,0]
    approach = rot33[:,2]
    finger   = rot33[:,0]
    scale    = gripper_w if gripper_w > 0 else 0.04

    tl = pos - finger*scale/2
    tr = pos + finger*scale/2
    wrist = pos - approach*0.105

    geoms.append(make_sphere(pos,   0.004, [1.0,0.85,0.0]))   # 🟡 力中心
    geoms.append(make_sphere(tl,    0.003, [0.9,0.15,0.15]))  # 🔴 左接触
    geoms.append(make_sphere(tr,    0.003, [0.15,0.3,0.95]))  # 🔵 右接触
    geoms.append(make_sphere(wrist, 0.003, [0.9,0.55,0.1]))   # 🟠 手腕
    geoms.append(make_line(tl, tr, color))                    # 横杆
    geoms.append(make_line(wrist, pos, [0.15,0.9,0.3]))       # approach 箭头


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_candidates(obj_id):
    for d in CANDIDATE_DIRS:
        p = os.path.join(d, f'{obj_id}_grasp.hdf5')
        if not os.path.exists(p): continue
        grasps = []
        with h5py.File(p) as f:
            if 'candidates' not in f: continue
            meta = f.get('metadata')
            prerot = list(meta.attrs.get('mesh_prerotation_euler', [0,0,0])) if meta else [0,0,0]
            canon_applied = bool(meta.attrs.get('canonical_rotation_applied', False)) if meta else False
            cg = f['candidates']
            for k in cg.keys():
                c = cg[k]
                grasps.append({
                    'pos':    c['position'][:],
                    'rot':    c['rotation'][:],
                    'width':  float(c.attrs.get('gripper_width', 0.04)),
                    'score':  float(c.attrs.get('score', 0)),
                    'name':   c.attrs.get('name','?'),
                })
        print(f'  候选文件: {p}')
        print(f'  canonical_rotation_applied={canon_applied}  mesh_prerotation_euler={prerot}')
        return grasps, canon_applied
    return [], False


def load_robot_gt(obj_id):
    grasps = []
    for d in GT_DIRS:
        p = os.path.join(d, f'{obj_id}_robot_gt.hdf5')
        if not os.path.exists(p): continue
        with h5py.File(p) as f:
            if not f.attrs.get('success', False): continue
            for k in f['successful_grasps'].keys():
                g = f[f'successful_grasps/{k}']
                grasps.append({
                    'pos':   g['grasp_point'][:],
                    'rot':   g['rotation'][:],
                    'width': float(g.attrs.get('gripper_width', 0.04)),
                    'name':  g.attrs.get('name', k),
                })
        if grasps:
            print(f'  robot_gt: {p}  ({len(grasps)} 成功)')
            return grasps
    return grasps


# ── 主可视化函数 ──────────────────────────────────────────────────────────────

def visualize(obj_id, stage):
    dataset = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
    _ply, _ds = find_ply(obj_id, dataset)
    if _ply is None:
        print(f'❌ 找不到 PLY: {obj_id}'); return

    # canonical rotation 信息
    euler = get_canonical_euler(obj_id, _ds)
    print(f'\n[{obj_id}] canonical rotation: {[round(e,1) for e in euler]}°')

    # 加载 canonical mesh
    mesh = load_mesh_canonical(obj_id, _ds, verbose=True)
    bb_size = mesh.bounding_box.extents
    ax_scale = float(np.max(bb_size)) * 0.70  # 坐标轴长度，明显超出物体范围
    centroid = mesh.centroid

    print(f'  bbox: {bb_size[0]*100:.1f}×{bb_size[1]*100:.1f}×{bb_size[2]*100:.1f} cm')
    print(f'  centroid: ({centroid[0]:.4f},{centroid[1]:.4f},{centroid[2]:.4f})')

    geoms = []

    # ── 物体显示（半透明效果：稀疏点云 + 极简轮廓）────────────────────────
    for g in make_transparent_mesh(mesh):
        geoms.append(g)


    # ── 坐标轴 @ 原点 (物体坐标系)
    # 红=X  绿=Y  蓝=Z，从原点 [0,0,0] 出发
    origin = np.array([0.0, 0.0, 0.0])
    for ax in make_axes(origin, scale=ax_scale):
        geoms.append(ax)

    # ── 抓取可视化 ─────────────────────────────────────────────────────────
    colors = [
        [1.0,0.5,0.0],[0.0,0.85,0.4],[0.6,0.2,1.0],
        [0.9,0.9,0.1],[0.1,0.9,0.9],[1.0,0.3,0.5],
    ]

    if stage in ('candidates','all'):
        grasps, canon_applied = load_candidates(obj_id)
        print(f'  候选数: {len(grasps)}')
        if not canon_applied:
            print('  ⚠️  WARNING: canonical_rotation_applied=False')
            print('       此文件用旧坐标系生成，与 canonical mesh 不一致！')
        for i, g in enumerate(grasps):  # 显示全部
            draw_grasp(g['pos'], g['rot'], g['width'],
                       colors[i % len(colors)], geoms)

    if stage in ('robot_gt','all'):
        grasps = load_robot_gt(obj_id)
        print(f'  robot_gt 数: {len(grasps)}')
        for i, g in enumerate(grasps):
            draw_grasp(g['pos'], g['rot'], g['width'],
                       colors[i % len(colors)], geoms)

    # ── 显示 ──────────────────────────────────────────────────────────────
    stage_label = stage.upper()
    title = (f'[{stage_label}] {obj_id}  '
             f'canonical_rot={[round(e,1) for e in euler]}°  '
             f'红=X  绿=Y  蓝=Z  |  🟡力中心 🔴🔵接触点 🟠手腕 🟢approach')
    print(f'\n🖱  {title}')
    print('   左键旋转  右键平移  滚轮缩放  Q退出\n')
    o3d.visualization.draw_geometries(geoms, window_name=title)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='坐标系一致性验证')
    parser.add_argument('--obj',   required=True, help='物体 ID，如 A16013')
    parser.add_argument('--stage', default='all',
                        choices=['candidates','robot_gt','all'],
                        help='candidates=只看候选  robot_gt=只看成功抓取  all=全部')
    args = parser.parse_args()
    visualize(args.obj, args.stage)

if __name__ == '__main__':
    main()
