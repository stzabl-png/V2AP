#!/usr/bin/env python3
"""Visualize a GLB mesh with coordinate axes + bounding box for orientation debugging.

Usage:
    python scripts/vis_mesh_axes.py --mesh "/media/lyh/KINGSTON/testing data/20260604_163505_spam/output/mesh/object_base_aligned.glb"
"""
import argparse
import numpy as np
import open3d as o3d
import trimesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True, help="Path to GLB/OBJ/PLY mesh")
    args = p.parse_args()

    # Load mesh
    scene = trimesh.load(args.mesh)
    mesh_tm = trimesh.util.concatenate([g for g in scene.geometry.values()])
    v = np.array(mesh_tm.vertices)
    f = np.array(mesh_tm.faces)

    extents = v.max(0) - v.min(0)
    center = (v.max(0) + v.min(0)) / 2
    z_min = v[:, 2].min()

    print(f"\nMesh: {args.mesh}")
    print(f"  Vertices: {len(v):,}  Faces: {len(f):,}")
    print(f"  Extents (cm): X={extents[0]*100:.1f}  Y={extents[1]*100:.1f}  Z={extents[2]*100:.1f}")
    print(f"  Center (m):   [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"  Z range (m):  [{v[:,2].min():.4f}, {v[:,2].max():.4f}]")
    sorted_axes = np.argsort(extents)[::-1]
    ax_names = ["X", "Y", "Z"]
    print(f"  Longest→Shortest: {ax_names[sorted_axes[0]]}({extents[sorted_axes[0]]*100:.1f}cm) > "
          f"{ax_names[sorted_axes[1]]}({extents[sorted_axes[1]]*100:.1f}cm) > "
          f"{ax_names[sorted_axes[2]]}({extents[sorted_axes[2]]*100:.1f}cm)")

    # Build O3D mesh
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(v)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(f)
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color([0.85, 0.45, 0.15])

    geoms = [mesh_o3d]

    # Coordinate frame at center
    frame_size = max(extents) * 0.5
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size, origin=center))

    # Table plane at Z_min
    MARGIN = 0.05
    N = 3000
    tx = np.random.uniform(v[:,0].min()-MARGIN, v[:,0].max()+MARGIN, N)
    ty = np.random.uniform(v[:,1].min()-MARGIN, v[:,1].max()+MARGIN, N)
    tz = np.full(N, z_min)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack([tx, ty, tz]))
    pcd.paint_uniform_color([0.5, 0.44, 0.34])
    geoms.append(pcd)

    # Bounding box
    aabb = mesh_o3d.get_axis_aligned_bounding_box()
    aabb.color = (0.2, 0.8, 0.2)
    geoms.append(aabb)

    print(f"\n  🖱  Left=rotate  Right=pan  Scroll=zoom  Q=quit")
    print(f"  Red=+X  Green=+Y  Blue=+Z(up)")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Mesh Orientation — {args.mesh.split('/')[-3]}",
                      width=1400, height=900)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.06, 0.06, 0.10])
    opt.mesh_show_back_face = True
    vc = vis.get_view_control()
    vc.set_zoom(0.55)
    vc.set_front([-0.45, -0.65, 0.45])
    vc.set_up([0, 0, 1])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
