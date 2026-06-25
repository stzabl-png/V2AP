#!/usr/bin/env python3
"""
Interactive Open3D visualization: HP region + HP-matched contact points.

Left panel info (HP heatmap) + Right panel info (contact points) merged:
  - Object point cloud: gray→red gradient by HP label
  - HP-matched contact points: blue spheres
  - Connecting lines between finger pairs

Usage:
    python3 tools/vis_hp_interactive.py --obj A15027
    python3 tools/vis_hp_interactive.py --obj C28001
"""
import os, sys, glob, argparse, h5py
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

PROJ         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HP_DIRS      = {
    'oakink': os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp', 'oakink'),
    'dexycb': os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp', 'dexycb'),
}
FILTERED_DIR = os.path.join(PROJ, 'output', 'RobotPosteriorFiltered')
MESH_BASE    = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')

N_VIS = 30000


def find_mesh(obj_id):
    ds = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
    p = os.path.join(MESH_BASE, ds, obj_id, 'mesh.ply')
    return p if os.path.exists(p) else None


def load_hp(obj_id):
    ds = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
    p = os.path.join(HP_DIRS[ds], f'{obj_id}.hdf5')
    if not os.path.exists(p):
        return None, None
    with h5py.File(p, 'r') as f:
        pc = f['point_cloud'][()].astype(np.float32)
        hp = f['human_prior'][()].astype(np.float32)
    return pc, hp


def load_hp_contacts(obj_id):
    p = os.path.join(FILTERED_DIR, f'{obj_id}_robot_gt.hdf5')
    if not os.path.exists(p):
        return [], 0, 0
    pairs = []   # each pair: (c_L, c_R)
    with h5py.File(p, 'r') as f:
        n_match    = int(f.attrs.get('n_hp_match', 0))
        n_mismatch = int(f.attrs.get('n_hp_mismatch', 0))
        sg = f.get('successful_grasps', {})
        for key in sg.keys():
            g = sg[key]
            if not g.attrs.get('hp_match', False):
                continue
            gp = g['grasp_point'][:]
            fd = g['finger_dir'][:]
            w  = float(g.attrs.get('gripper_width', 0.05))
            fd_n = fd / (np.linalg.norm(fd) + 1e-8)
            c_L = gp + fd_n * w / 2
            c_R = gp - fd_n * w / 2
            pairs.append((c_L, c_R))
    return pairs, n_match, n_mismatch


def hp_to_color(hp_val):
    """Scalar HP value [0,1] → RGB color (gray→red)."""
    gray = np.array([0.75, 0.75, 0.80])
    red  = np.array([0.95, 0.10, 0.10])
    return (1 - hp_val) * gray + hp_val * red


def make_sphere(center, radius=0.005, color=(0.2, 0.4, 0.9)):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(center)
    s.paint_uniform_color(color)
    s.compute_vertex_normals()
    return s


def make_cylinder(p1, p2, radius=0.0015, color=(0.3, 0.5, 1.0)):
    """Draw a cylinder between two 3D points."""
    p1, p2 = np.array(p1), np.array(p2)
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1e-6:
        return None
    mid = (p1 + p2) / 2.0
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    # Default cylinder is along Z axis; rotate to align with d
    z = np.array([0, 0, 1.0])
    dn = d / length
    axis = np.cross(z, dn)
    ax_norm = np.linalg.norm(axis)
    if ax_norm < 1e-6:
        R = np.eye(3) if np.dot(z, dn) > 0 else -np.eye(3)
    else:
        axis /= ax_norm
        angle = np.arccos(np.clip(np.dot(z, dn), -1, 1))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    cyl.rotate(R, center=(0, 0, 0))
    cyl.translate(mid)
    cyl.paint_uniform_color(color)
    cyl.compute_vertex_normals()
    return cyl


def visualize(obj_id):
    print(f'\nLoading {obj_id}...')

    # ── Load HP data ──────────────────────────────────
    hp_pc, hp_labels = load_hp(obj_id)
    if hp_pc is None:
        print(f'  ERROR: no HP data for {obj_id}')
        return

    pairs, n_match, n_mismatch = load_hp_contacts(obj_id)
    n_total = n_match + n_mismatch
    hp_pct  = (hp_labels > 0.3).mean() * 100

    print(f'  HP region: {hp_pct:.1f}%  |  HP-match: {n_match}/{n_total} grasps')

    # ── Build dense point cloud ────────────────────────
    mesh_path = find_mesh(obj_id)
    if mesh_path:
        import trimesh
        mesh   = trimesh.load(mesh_path, force='mesh')
        vis_pts, _ = trimesh.sample.sample_surface(mesh, N_VIS)
        vis_pts = vis_pts.astype(np.float32)
        tree    = cKDTree(hp_pc)
        _, idx  = tree.query(vis_pts, k=3)
        dists   = np.linalg.norm(vis_pts[:,None,:] - hp_pc[idx], axis=2)
        wts     = 1.0 / (dists + 1e-8)
        wts    /= wts.sum(axis=1, keepdims=True)
        hp_dense = (hp_labels[idx] * wts).sum(axis=1)
    else:
        vis_pts  = hp_pc
        hp_dense = hp_labels

    # ── Point cloud with HP colors ─────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vis_pts.astype(np.float64))
    colors = np.array([hp_to_color(v) for v in hp_dense])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ── Geometries list ────────────────────────────────
    geometries = [pcd]

    # ── Contact spheres + cylinders ────────────────────
    # Estimate sphere radius as ~1% of object diagonal
    extents = vis_pts.max(0) - vis_pts.min(0)
    r_sphere = float(extents.max()) * 0.018
    r_cyl    = r_sphere * 0.25

    for c_L, c_R in pairs:
        geometries.append(make_sphere(c_L, radius=r_sphere, color=(0.15, 0.40, 0.90)))
        geometries.append(make_sphere(c_R, radius=r_sphere, color=(0.15, 0.40, 0.90)))
        cyl = make_cylinder(c_L, c_R, radius=r_cyl, color=(0.25, 0.55, 1.00))
        if cyl is not None:
            geometries.append(cyl)

    # ── Visualize ──────────────────────────────────────
    title = (f'{obj_id}  |  HP-match:{n_match}  HP-miss:{n_mismatch}'
             f'  HP-region:{hp_pct:.0f}%')
    print(f'  Opening Open3D window: "{title}"')
    print('  Controls: Left drag=rotate, Scroll=zoom, Right drag=pan, Q/Esc=quit')

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=800)
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = 2.0
    opt.light_on   = True

    # Set camera to a good initial view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)
    ctr.set_front([0.0, -0.5, -1.0])
    ctr.set_up([0.0, 1.0, 0.0])

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type=str, required=True, help='Object ID, e.g. A15027')
    args = parser.parse_args()
    visualize(args.obj)


if __name__ == '__main__':
    main()
