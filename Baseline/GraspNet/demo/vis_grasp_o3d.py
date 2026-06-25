#!/usr/bin/env python3
"""
graspnet_demo/vis_grasp_o3d.py
================================
Open3D 交互可视化: Mesh + GraspNet Top-1 抓取位姿 (+ 2.5cm shift)

复用 tools/vis_grasp_combined.py 的几何体工具函数。

用法:
    python -m graspnet_demo.vis_grasp_o3d \
        --session /media/lyh/KINGSTON/20260602_192346_chips

    # 显示 top-N (默认 1)
    python -m graspnet_demo.vis_grasp_o3d \
        --session /media/lyh/KINGSTON/20260602_192346_chips --n-top 5

鼠标操作:
    左键拖拽=旋转  右键拖拽=平移  滚轮=缩放  Q/Esc=退出
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
import open3d as o3d

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import sys as _sys
_sys.path.insert(0, str(_PROJ / "Baseline2" / "graspnet"))
from graspnet_to_hdf5 import graspgroup_to_candidates, rerank_for_reachability

SHIFT_M    = 0.025   # 2.5cm depth shift (same as sim evaluation)
TCP_OFFSET = 0.105   # panda_hand EE → fingertip midpoint


# ─────────────────────────────────────────────────────────────
# 复用 vis_grasp_combined.py 的几何体工具
# ─────────────────────────────────────────────────────────────

def _rot_z_to(direction):
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction) + 1e-8
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, direction)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(z, direction))
    if sin_a > 1e-6:
        axis /= sin_a
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            axis * np.arctan2(sin_a, cos_a))
    elif cos_a < 0:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([1.0, 0.0, 0.0]) * np.pi)
    else:
        R = np.eye(3)
    return R


def make_sphere(center, radius=0.005, color=(1, 0.5, 0)):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(np.array(center, dtype=float))
    s.paint_uniform_color(list(color))
    s.compute_vertex_normals()
    return s


def make_cylinder(p1, p2, radius=0.003, color=(0.1, 0.8, 0.9)):
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    d = p2 - p1; length = np.linalg.norm(d)
    if length < 1e-6:
        return None
    c = o3d.geometry.TriangleMesh.create_cylinder(radius=radius,
                                                   height=length, resolution=16)
    c.rotate(_rot_z_to(d), center=[0, 0, 0])
    c.translate((p1 + p2) / 2.0)
    c.paint_uniform_color(list(color))
    c.compute_vertex_normals()
    return c


def make_arrow(start, direction, length=0.05, radius=0.003, color=(0.9, 0.2, 0.2)):
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction) + 1e-8
    a = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius, cone_radius=radius * 2.5,
        cylinder_height=length * 0.72, cone_height=length * 0.28,
        resolution=16)
    a.rotate(_rot_z_to(direction), center=[0, 0, 0])
    a.translate(np.array(start, dtype=float))
    a.paint_uniform_color(list(color))
    a.compute_vertex_normals()
    return a


def build_gripper_geoms(grasp_pt, R, width,
                        color_body=(0.05, 0.95, 0.45),
                        color_tip=(0.10, 0.85, 0.95),
                        show_pregrasp=True):
    """
    构建夹爪几何体列表。
    grasp_pt: (3,) fingertip midpoint (已含 2.5cm shift)
    R:        (3,3) rotation matrix, col2=approach
    width:    gripper open width (m)
    """
    geoms = []
    approach = R[:, 2]
    f_open   = R[:, 0]
    hw = width / 2.0

    FINGER_LEN  = 0.07   # finger length along approach
    ROD_RADIUS  = width * 0.025
    TIP_RADIUS  = width * 0.042
    WRIST_RADIUS = width * 0.055

    tip_L  = grasp_pt + f_open *  hw
    tip_R  = grasp_pt - f_open *  hw
    base_L = tip_L - approach * FINGER_LEN
    base_R = tip_R - approach * FINGER_LEN
    palm   = (base_L + base_R) / 2.0

    # 指尖球 (cyan)
    geoms.append(make_sphere(tip_L, TIP_RADIUS, color_tip))
    geoms.append(make_sphere(tip_R, TIP_RADIUS, color_tip))
    # 指尖横条
    c = make_cylinder(tip_L, tip_R, ROD_RADIUS, color_tip)
    if c: geoms.append(c)
    # 左右手指杆 (green)
    c = make_cylinder(base_L, tip_L, ROD_RADIUS, color_body)
    if c: geoms.append(c)
    c = make_cylinder(base_R, tip_R, ROD_RADIUS, color_body)
    if c: geoms.append(c)
    # 背板横条
    c = make_cylinder(base_L, base_R, ROD_RADIUS, color_body)
    if c: geoms.append(c)
    # Approach 箭头 (从 grasp_pt 沿 approach 方向)
    a = make_arrow(grasp_pt, approach, length=FINGER_LEN * 0.9,
                   radius=ROD_RADIUS * 0.8, color=color_body)
    if a: geoms.append(a)
    # 手腕球 (yellow)
    wrist = grasp_pt - approach * TCP_OFFSET
    geoms.append(make_sphere(wrist, WRIST_RADIUS, (1.0, 0.9, 0.1)))
    # 手腕连杆
    c = make_cylinder(wrist, palm, ROD_RADIUS * 0.7, (0.85, 0.85, 0.2))
    if c: geoms.append(c)
    # Grasp point 球 (orange)
    geoms.append(make_sphere(grasp_pt, TIP_RADIUS * 0.8, (1.0, 0.5, 0.0)))

    # Pre-grasp 球 + 箭头 (gold)
    if show_pregrasp:
        pre = grasp_pt - approach * (TCP_OFFSET + 0.15)
        geoms.append(make_sphere(pre, TIP_RADIUS * 0.9, (1.0, 0.85, 0.0)))
        a = make_arrow(pre, approach, length=0.10,
                       radius=ROD_RADIUS * 0.7, color=(1.0, 0.80, 0.0))
        if a: geoms.append(a)

    return geoms


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def run(session_dir: Path, n_top: int = 1):
    BASE = session_dir

    # ── 1. Load mesh ──────────────────────────────────────────
    scene = trimesh.load(str(BASE / "output/mesh/object_scaled.glb"))
    mesh_tm = trimesh.util.concatenate([g for g in scene.geometry.values()])
    # Z-up: SAM3D Y → Z
    R_zup = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=float)
    T_zup = np.eye(4); T_zup[:3,:3] = R_zup
    mesh_tm.apply_transform(T_zup)
    mesh_tm.apply_translation([0, 0, -mesh_tm.vertices[:,2].min()])
    v = np.array(mesh_tm.vertices, dtype=np.float64)
    f = np.array(mesh_tm.faces)
    z_min = float(v[:,2].min())
    MARGIN = 0.05

    print(f"Mesh: {len(v):,} verts | {len(f):,} faces")
    print(f"  X=[{v[:,0].min()*100:.1f},{v[:,0].max()*100:.1f}]cm "
          f"Y=[{v[:,1].min()*100:.1f},{v[:,1].max()*100:.1f}]cm "
          f"Z=[{v[:,2].min()*100:.1f},{v[:,2].max()*100:.1f}]cm")

    # ── 2. GraspNet ───────────────────────────────────────────
    from Baseline2.graspnet.graspnet_infer import load_model, infer_grasps
    obj_pts, _ = trimesh.sample.sample_surface(mesh_tm, 20000)
    obj_pts = obj_pts.astype(np.float32)

    ckpt = str(_PROJ / "Baseline2/graspnet/checkpoints/checkpoint-rs.tar")
    net = load_model(ckpt)
    # infer_grasps: approach_dist=0.25, z_approach_max=0.3 (no below-table grasps)
    gg  = infer_grasps(net, obj_pts, n_top=max(n_top * 10, 50))

    # ── Convert to A2G convention (col2=approach) — identical to baseline ──
    # graspgroup_to_candidates applies z_approach_max=0.3 (same as batch_graspnet.py)
    # and remaps: R_a2g = [finger_dir | binormal | approach]
    # After this: R[:,0]=finger_open  R[:,1]=binormal  R[:,2]=approach
    candidates = graspgroup_to_candidates(gg, scale_factor=1.0, z_approach_max=0.3)
    candidates = rerank_for_reachability(candidates)   # prefer top-down
    candidates = candidates[:n_top]

    # Apply 2.5cm depth shift to position — same as make_shifted_graspnet_candidates.py:
    #   approach = rotation[:, 2]  (A2G convention)
    #   position += approach * SHIFT_M
    # This bakes the shift into the candidate data itself.
    for c in candidates:
        approach = np.array(c['rotation'])[:, 2]
        c['position'] = np.array(c['position']) + approach * SHIFT_M

    print(f"\nAfter A2G conversion + rerank + {SHIFT_M*100:.1f}cm shift: "
          f"{len(candidates)} candidates (top-{n_top})")
    for i, c in enumerate(candidates):
        app = c['approach']
        print(f"  rank={i} score={c['score']:.4f} "
              f"pt(+shift)=[{c['position'][0]:.3f},{c['position'][1]:.3f},{c['position'][2]:.3f}] "
              f"approach=[{app[0]:.2f},{app[1]:.2f},{app[2]:.2f}] "
              f"width={c['gripper_width']*100:.1f}cm")

    if not candidates:
        print("❌ No valid candidates after filtering.")
        return

    # ── 3. Build Open3D scene ─────────────────────────────────
    geoms = []

    # Object mesh (orange-red, Pringles color)
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(v)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(f)
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color([0.92, 0.38, 0.08])
    geoms.append(mesh_o3d)

    # Virtual table (gray point cloud)
    N_TABLE = 3000
    tx = np.random.uniform(v[:,0].min()-MARGIN, v[:,0].max()+MARGIN, N_TABLE)
    ty = np.random.uniform(v[:,1].min()-MARGIN, v[:,1].max()+MARGIN, N_TABLE)
    tz = np.full(N_TABLE, z_min)
    table_pcd = o3d.geometry.PointCloud()
    table_pcd.points = o3d.utility.Vector3dVector(
        np.column_stack([tx, ty, tz]))
    table_pcd.paint_uniform_color([0.55, 0.48, 0.38])
    geoms.append(table_pcd)

    # Coordinate axes at mesh center
    center = (v.max(0) + v.min(0)) / 2
    ax_len = (v[:,2].max() - v[:,2].min()) * 0.35
    frame  = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=ax_len, origin=center)
    geoms.append(frame)

    # Each grasp
    # Each grasp — A2G convention: R[:,0]=finger_dir, R[:,2]=approach
    colors_body = [
        (0.05, 0.95, 0.45),   # rank-0: bright green
        (0.95, 0.60, 0.05),   # rank-1: orange
        (0.55, 0.20, 0.90),   # rank-2: purple
        (0.10, 0.70, 0.95),   # rank-3: cyan
        (0.95, 0.15, 0.30),   # rank-4: red
    ]

    for rank, c in enumerate(candidates):
        pt       = np.array(c['position'], dtype=np.float64)  # already +2.5cm shifted
        R        = np.array(c['rotation'], dtype=np.float64)  # A2G: col2=approach
        width    = float(c['gripper_width'])
        approach = R[:, 2]    # A2G convention ✅

        col = colors_body[rank % len(colors_body)]
        gripper_geoms = build_gripper_geoms(
            pt, R, width,
            color_body=col,
            color_tip=(0.10, 0.85, 0.95),
            show_pregrasp=(rank == 0),
        )
        geoms.extend(gripper_geoms)

        print(f"  rank={rank} score={c['score']:.4f} "
              f"pt=[{pt[0]:.3f},{pt[1]:.3f},{pt[2]:.3f}] "
              f"approach=[{approach[0]:.2f},{approach[1]:.2f},{approach[2]:.2f}] "
              f"width={width*100:.1f}cm")

    # ── 4. Launch viewer ──────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  🔴 Orange mesh: Chips can (SAM3D, Z-up)")
    print(f"  🟢 Green gripper: Top-1 (rank=0) + 2.5cm shift")
    print(f"  🟡 Yellow sphere: Wrist (EE position)")
    print(f"  🟠 Orange sphere: Grasp point (fingertip mid)")
    print(f"  💛 Gold sphere+arrow: Pre-grasp (retreat 15cm)")
    print(f"  ─ Gray dots: Virtual table plane")
    print(f"  XYZ frame: Red=X  Green=Y  Blue=Z(up)")
    print(f"\n  🖱  Left-drag=rotate  Right-drag=pan  Scroll=zoom  Q=quit")
    print(f"{'='*55}\n")

    vis = o3d.visualization.Visualizer()
    title = f"GraspNet Top-{n_top}  |  Chips Can  |  +2.5cm shift"
    vis.create_window(window_name=title, width=1400, height=900)

    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color   = np.array([0.06, 0.06, 0.10])
    opt.point_size         = 3.0
    opt.mesh_show_back_face = True

    # Set initial viewpoint: isometric from front-right
    ctr = vis.get_view_control()
    ctr.set_zoom(0.65)
    ctr.set_front([-0.5, -0.7, 0.5])
    ctr.set_up([0, 0, 1])

    vis.run()
    vis.destroy_window()


def main():
    p = argparse.ArgumentParser(description="Open3D GraspNet visualization")
    p.add_argument("--session", type=str,
                   default="/media/lyh/KINGSTON/20260602_192346_chips")
    p.add_argument("--n-top", type=int, default=1,
                   help="Show top-N grasps (default: 1)")
    args = p.parse_args()
    run(Path(args.session), n_top=args.n_top)


if __name__ == "__main__":
    main()
