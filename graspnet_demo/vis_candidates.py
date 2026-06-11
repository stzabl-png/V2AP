#!/usr/bin/env python3
"""
graspnet_demo/vis_candidates.py
================================
从 candidates.json 读取预计算的抓取位姿，Open3D 交互可视化。
不需要重跑 GraspNet。

用法:
    python graspnet_demo/vis_candidates.py \
        --session /media/lyh/KINGSTON/20260602_192346_chips

    # 只显示 top-5
    python graspnet_demo/vis_candidates.py \
        --session /media/lyh/KINGSTON/20260602_192346_chips --n-top 5

    # 显示全部 20 个
    python graspnet_demo/vis_candidates.py \
        --session /media/lyh/KINGSTON/20260602_192346_chips --n-top 20

鼠标: 左键=旋转  右键=平移  滚轮=缩放  Q=退出
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

TCP_OFFSET = 0.105   # panda_hand EE → fingertip midpoint

BODY_COLORS = [
    (0.05, 0.95, 0.45),   # green
    (0.95, 0.60, 0.05),   # orange
    (0.55, 0.20, 0.90),   # purple
    (0.10, 0.70, 0.95),   # cyan
    (0.95, 0.15, 0.30),   # red
    (0.20, 0.90, 0.80),   # teal
    (0.90, 0.90, 0.10),   # yellow
    (0.85, 0.20, 0.85),   # magenta
    (0.50, 0.85, 0.20),   # lime
    (0.20, 0.40, 0.90),   # blue
]


# ── geometry helpers ──────────────────────────────────────────────────────

def _rot_z_to(d):
    d = np.array(d, dtype=float) / (np.linalg.norm(d) + 1e-8)
    z = np.array([0., 0., 1.])
    ax = np.cross(z, d); s = np.linalg.norm(ax); c = float(z @ d)
    if s > 1e-6:
        return o3d.geometry.get_rotation_matrix_from_axis_angle(ax / s * np.arctan2(s, c))
    return np.eye(3) if c > 0 else o3d.geometry.get_rotation_matrix_from_axis_angle(
        np.array([1., 0., 0.]) * np.pi)


def sphere(c, r=0.005, col=(1, .5, 0)):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    m.translate(c); m.paint_uniform_color(col); m.compute_vertex_normals()
    return m


def cylinder(p1, p2, r=0.003, col=(.1, .8, .9)):
    d = np.array(p2) - np.array(p1); L = np.linalg.norm(d)
    if L < 1e-6: return None
    m = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=L, resolution=16)
    m.rotate(_rot_z_to(d), [0, 0, 0])
    m.translate((np.array(p1) + np.array(p2)) / 2)
    m.paint_uniform_color(col); m.compute_vertex_normals(); return m


def arrow(start, d, length=0.05, r=0.003, col=(.9, .2, .2)):
    d = np.array(d) / (np.linalg.norm(d) + 1e-8)
    m = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=r, cone_radius=r*2.5,
        cylinder_height=length*.72, cone_height=length*.28, resolution=16)
    m.rotate(_rot_z_to(d), [0, 0, 0]); m.translate(np.array(start, float))
    m.paint_uniform_color(col); m.compute_vertex_normals(); return m


def build_gripper(pt, R, width, body_col, show_pregrasp=False):
    """pt=grasp_point(z0), R[:,2]=approach (A2G), width=open width."""
    geoms = []
    approach = R[:, 2]; f_open = R[:, 0]
    hw = width / 2; FL = 0.07; rr = width * 0.025

    tip_L = pt + f_open * hw;  tip_R = pt - f_open * hw
    bL = tip_L - approach * FL; bR = tip_R - approach * FL

    TIP = (0.10, 0.85, 0.95)
    geoms += [sphere(tip_L, width*.042, TIP), sphere(tip_R, width*.042, TIP)]
    for a, b, col in [(tip_L, tip_R, TIP), (bL, tip_L, body_col),
                      (bR, tip_R, body_col), (bL, bR, body_col)]:
        c = cylinder(a, b, rr, col)
        if c: geoms.append(c)

    a = arrow(pt, approach, length=FL*.9, r=rr*.8, col=body_col)
    if a: geoms.append(a)

    wrist = pt - approach * TCP_OFFSET
    geoms.append(sphere(wrist, width*.055, (1.0, 0.9, 0.1)))
    c = cylinder(wrist, (bL+bR)/2, rr*.7, (0.85, 0.85, 0.2))
    if c: geoms.append(c)
    geoms.append(sphere(pt, width*.033, (1.0, 0.5, 0.0)))

    if show_pregrasp:
        pre = pt - approach * (TCP_OFFSET + 0.15)
        geoms.append(sphere(pre, width*.040, (1.0, 0.85, 0.0)))
        a = arrow(pre, approach, length=0.10, r=rr*.7, col=(1.0, 0.80, 0.0))
        if a: geoms.append(a)

    return geoms


# ── main ──────────────────────────────────────────────────────────────────

def run(session_dir: Path, n_top: int = 20, candidates_path: str | None = None):
    session_dir = Path(session_dir)

    # ── 1. Load candidates.json ───────────────────────────────
    if candidates_path:
        cand_path = Path(candidates_path)
    else:
        cand_path = session_dir / "output" / "inference" / "candidates.json"
    if not cand_path.exists():
        raise FileNotFoundError(f"candidates.json not found: {cand_path}")

    with open(cand_path) as f:
        data = json.load(f)

    all_cands = data["candidates"]
    candidates = all_cands[:n_top]
    n_total    = data.get("n_candidates", len(all_cands))
    method     = data.get("inference_method", "unknown")
    T_base_mesh = np.array(data["T_base_mesh"])

    print(f"\n{'='*58}")
    print(f"  candidates.json: method={method}, total={n_total}, showing={len(candidates)}")
    print(f"  T_base_mesh t: [{T_base_mesh[0,3]:.3f},{T_base_mesh[1,3]:.3f},{T_base_mesh[2,3]:.3f}]m")

    # ── 2. Load mesh (base_aligned) ───────────────────────────
    mesh_path = session_dir / "output" / "mesh" / "object_base_aligned.glb"
    scene   = trimesh.load(str(mesh_path))
    mesh_tm = trimesh.util.concatenate([g for g in scene.geometry.values()])

    v_orig  = np.array(mesh_tm.vertices)
    frame   = data.get("mesh_frame", "base_aligned_z0")
    mesh_z_bottom = float(v_orig[:, 2].min())  # object bottom in this frame

    if frame == "base_aligned_z0":
        # z=0 already at object bottom, no shift needed
        print(f"  Frame: base_aligned_z0  (z=0 = object bottom)")
    else:
        # base_aligned: z=0 at centroid, bottom is at mesh_z_bottom
        print(f"  Frame: {frame}  (z=0 = centroid, object bottom at z={mesh_z_bottom*100:.1f}cm)")

    v = np.array(mesh_tm.vertices)
    f = np.array(mesh_tm.faces)

    print(f"  Mesh bounds:")
    print(f"    X=[{v[:,0].min()*100:.1f},{v[:,0].max()*100:.1f}]cm "
          f"Y=[{v[:,1].min()*100:.1f},{v[:,1].max()*100:.1f}]cm "
          f"Z=[{v[:,2].min()*100:.1f},{v[:,2].max()*100:.1f}]cm")

    # ── 3. Build scene ────────────────────────────────────────
    geoms = []

    # Object mesh
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices  = o3d.utility.Vector3dVector(v)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(f)
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color([0.92, 0.38, 0.08])
    geoms.append(mesh_o3d)

    # Virtual table plane (gray dots) — always at object bottom
    MARGIN = 0.06; N = 3000
    tx = np.random.uniform(v[:,0].min()-MARGIN, v[:,0].max()+MARGIN, N)
    ty = np.random.uniform(v[:,1].min()-MARGIN, v[:,1].max()+MARGIN, N)
    tz = np.full(N, mesh_z_bottom)   # ← actual object bottom, not z=0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack([tx, ty, tz]))
    pcd.paint_uniform_color([0.50, 0.44, 0.34])
    geoms.append(pcd)

    # Coordinate frame
    ctr_xyz = (v.max(0) + v.min(0)) / 2
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=(v[:,2].max() - v[:,2].min()) * 0.35, origin=ctr_xyz))

    # ── 4. Draw each candidate ────────────────────────────────
    gp_label = data.get("conventions", {}).get("grasp_point_frame", frame)
    print(f"\n  {'rank':4s} {'name':15s} {'score':7s}  {f'grasp_point ({gp_label})':32s}  {'approach':22s}")
    print(f"  {'-'*95}")

    for rank, c in enumerate(candidates):
        pt  = np.array(c["grasp_point"], dtype=float)
        R   = np.array(c["rotation"],   dtype=float)
        w   = float(c["gripper_width_m"])
        app = R[:, 2]
        col = BODY_COLORS[rank % len(BODY_COLORS)]

        print(f"  {rank:4d} {c['name']:15s} {c['score']:7.3f}  "
              f"[{pt[0]:+.3f},{pt[1]:+.3f},{pt[2]:+.3f}]  "
              f"[{app[0]:+.2f},{app[1]:+.2f},{app[2]:+.2f}]")

        geoms.extend(build_gripper(pt, R, w, col,
                                   show_pregrasp=(rank == 0)))

    # ── 5. Launch viewer ──────────────────────────────────────
    print(f"\n  Legend:")
    print(f"    🟠 Orange mesh:   Chips can (base_aligned, z0)")
    print(f"    🎨 Colored grippers: rank 0→{len(candidates)-1} (green=best)")
    print(f"    🟡 Yellow sphere:    Wrist (EE position)")
    print(f"    🟠 Orange dot:       Grasp point (fingertip mid, +2.5cm shift)")
    print(f"    💛 Gold sphere+arrow: Pre-grasp (rank-0 only, retreat 15cm)")
    print(f"    ⬜ Gray dots: Virtual table (z=0)")
    print(f"    XYZ frame: Red=X  Green=Y  Blue=Z(up)")
    print(f"\n  🖱  Left=rotate  Right=pan  Scroll=zoom  Q=quit")
    print(f"{'='*58}\n")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"GraspNet {n_top} candidates  [{method}]  session={session_dir.name}",
        width=1500, height=950)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color    = np.array([0.06, 0.06, 0.10])
    opt.point_size          = 3.0
    opt.mesh_show_back_face = True

    vc = vis.get_view_control()
    vc.set_zoom(0.58); vc.set_front([-0.45, -0.65, 0.45]); vc.set_up([0, 0, 1])

    vis.run()
    vis.destroy_window()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", default="/media/lyh/KINGSTON/20260602_192346_chips")
    p.add_argument("--n-top", type=int, default=20, help="Show top-N (default: all 20)")
    p.add_argument("--candidates", default=None,
                   help="Custom candidates JSON path (default: output/inference/candidates.json)")
    args = p.parse_args()
    run(Path(args.session), n_top=args.n_top, candidates_path=args.candidates)


if __name__ == "__main__":
    main()
