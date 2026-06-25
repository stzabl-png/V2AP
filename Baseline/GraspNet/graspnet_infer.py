#!/usr/bin/env python3
"""
GraspNet-Baseline Inference
============================
Mesh → Surface Point Cloud → GraspNet Forward → Collision Detection → Top-K Grasps

Input:  SAM3D rotated_mesh (from HF eval_assets) + scale.json
Output: GraspGroup (numpy array, each row 17-dim)

Usage:
    python Baseline2/graspnet/graspnet_infer.py \
        --mesh data_hub/meshes/SAM3DMesh/rotated_mesh/oakink/A01001/mesh.ply \
        --checkpoint Baseline2/graspnet/checkpoints/checkpoint-rs.tar \
        --scale-json data_hub/ProcessedData/obj_meshes/oakink/A01001/scale.json \
        --n-top 50

Standalone test:
    python Baseline2/graspnet/graspnet_infer.py \
        --mesh /path/to/mesh.ply \
        --checkpoint /path/to/checkpoint-rs.tar \
        --dump-npy /tmp/test_grasps.npy
"""

import os
import sys
import argparse
import json
import numpy as np

# ── Project paths ──────────────────────────────────────────────
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRASPNET_DIR = os.path.join(PROJ, "third_party", "graspnet-baseline")
for _sub in ("", "models", "utils", "dataset"):
    _p = os.path.join(GRASPNET_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load GraspNet pretrained model."""
    import torch
    from models.graspnet import GraspNet

    net = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval().to(device)
    print(f"  ✅ GraspNet model loaded from {checkpoint_path}")
    return net


def mesh_to_pointcloud(
    mesh_path: str,
    scale_factor: float = 1.0,
    n_points: int = 20000,
):
    """Sample surface points from mesh (metric scale)."""
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    if abs(scale_factor - 1.0) > 1e-6:
        mesh.vertices = mesh.vertices * scale_factor
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    points = points.astype(np.float32)
    print(
        f"  ✅ Point cloud: {len(points)} pts from {os.path.basename(mesh_path)}"
        f" (scale={scale_factor:.4f})"
    )
    print(
        f"     extents (cm): "
        f"X={float((points[:, 0].max() - points[:, 0].min()) * 100):.1f} "
        f"Y={float((points[:, 1].max() - points[:, 1].min()) * 100):.1f} "
        f"Z={float((points[:, 2].max() - points[:, 2].min()) * 100):.1f}"
    )
    return points, mesh


def generate_table_plane(
    object_points: np.ndarray,
    n_table_points: int = 5000,
    margin: float = 0.05,
) -> np.ndarray:
    """Generate a virtual table plane at Z_min of the object.

    Simulates the table surface that GraspNet's collision detector expects
    in a real scene point cloud.  The plane extends `margin` beyond the
    object's XY bounding box so that side-approach grasps also collide.

    Args:
        object_points: (N, 3) object surface points (Z-up).
        n_table_points: number of points to sample on the table plane.
        margin: extra extent beyond the object's XY bounding box (metres).

    Returns:
        table_points: (n_table_points, 3) points on the table plane.
    """
    z_min = float(object_points[:, 2].min())
    x_min, y_min = object_points[:, 0].min() - margin, object_points[:, 1].min() - margin
    x_max, y_max = object_points[:, 0].max() + margin, object_points[:, 1].max() + margin

    xs = np.random.uniform(x_min, x_max, n_table_points).astype(np.float32)
    ys = np.random.uniform(y_min, y_max, n_table_points).astype(np.float32)
    zs = np.full(n_table_points, z_min, dtype=np.float32)
    return np.column_stack([xs, ys, zs])


def infer_grasps(
    net,
    points: np.ndarray,
    n_top: int = 50,
    collision_thresh: float = 0.01,
    z_approach_max: float = 0.3,
):
    """
    GraspNet forward → collision detection (with table plane) → z_approach filter
    → NMS → sort → top-K.

    The network receives only object surface points (unchanged).
    Collision detection receives object + virtual table plane, matching the
    official demo's scene-level input format.

    Args:
        z_approach_max: Reject grasps whose approach Z-component > this value.
                        approach_z > 0 means "approaching from below the table",
                        which is physically impossible when the object sits on a table.
                        Default 0.3 matches graspnet_to_hdf5.py (sim evaluation).
                        Set to 1.0 to disable this filter.

    Returns:
        GraspGroup (graspnetAPI) sorted by score descending.
    """
    import torch
    from models.graspnet import pred_decode
    from utils.collision_detector import ModelFreeCollisionDetector
    from graspnetAPI import GraspGroup

    end_points = {
        "point_clouds": torch.from_numpy(
            points[np.newaxis].astype(np.float32)
        ).cuda(),
    }
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    gg = GraspGroup(grasp_preds[0].cpu().numpy())
    n_raw = len(gg)

    if len(gg) == 0:
        print("  ⚠️  No grasps predicted by network")
        return gg

    # Build scene points: object + virtual table plane
    table_pts = generate_table_plane(points)
    scene_points = np.concatenate([points, table_pts], axis=0)

    # Collision detection against scene (object + table)
    # approach_dist=0.25: check 25cm along approach (covers pre-grasp reach)
    # This ensures grasps whose arm would pass through the table are rejected.
    mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.01)
    collision_mask = mfcdetector.detect(
        gg, approach_dist=0.25, collision_thresh=collision_thresh
    )
    gg = gg[~collision_mask]
    n_after_collision = len(gg)

    if len(gg) == 0:
        print(f"  ⚠️  All {n_raw} grasps removed by collision detection (with table plane)")
        return gg

    # Filter: reject grasps approaching from below table (approach_z > z_approach_max).
    # GraspNet rotation_matrix[:, 0] = approach direction in graspnetAPI convention.
    # Our mesh_to_pointcloud sets Z-up so Z=0 is the table surface.
    # approach_z > 0 means the robot wrist must be BELOW the table → impossible.
    if z_approach_max < 1.0:
        approach_zs = np.array([g.rotation_matrix[2, 0] for g in gg])
        keep_mask   = approach_zs <= z_approach_max
        n_before_z  = len(gg)
        gg          = gg[keep_mask]
        n_filtered  = n_before_z - len(gg)
        if n_filtered > 0:
            print(f"     z_approach filter (max={z_approach_max}): "
                  f"removed {n_filtered} below-table grasps → {len(gg)} remain")
        if len(gg) == 0:
            print(f"  ⚠️  All grasps removed by z_approach filter. "
                  f"Try increasing z_approach_max or check mesh orientation.")
            return gg

    # NMS + sort + top-K (NMS requires compiled grasp_nms; skip if unavailable)
    try:
        gg = gg.nms().sort_by_score()
    except (ImportError, ModuleNotFoundError):
        gg.sort_by_score()
    gg = gg[:n_top]

    print(
        f"  ✅ Grasps: {n_raw} raw → {n_after_collision} post-collision(+table) "
        f"→ {len(gg)} after NMS+top-{n_top}"
    )
    if len(gg) > 0:
        scores = np.array([g.score for g in gg])
        print(
            f"     score range: [{scores.min():.4f}, {scores.max():.4f}] "
            f"mean={scores.mean():.4f}"
        )
    return gg


def read_scale_factor(scale_json_path: str) -> float:
    """Read scale_factor from scale.json."""
    if scale_json_path and os.path.isfile(scale_json_path):
        with open(scale_json_path) as f:
            return float(json.load(f).get("scale_factor", 1.0))
    return 1.0


def find_scale_json(obj_id: str, dataset: str | None = None) -> str | None:
    """Auto-find scale.json for an object."""
    candidates = []
    if dataset:
        candidates.append(
            os.path.join(
                PROJ, "data_hub", "ProcessedData", "obj_meshes",
                dataset, obj_id, "scale.json",
            )
        )
    for ds in ("oakink", "ycb", "dexycb", "unseen"):
        candidates.append(
            os.path.join(
                PROJ, "data_hub", "ProcessedData", "obj_meshes",
                ds, obj_id, "scale.json",
            )
        )
    return next((p for p in candidates if os.path.isfile(p)), None)


def find_rotated_mesh(obj_id: str, dataset: str | None = None) -> str | None:
    """Auto-find rotated_mesh for an object."""
    base = os.path.join(PROJ, "data_hub", "meshes", "SAM3DMesh", "rotated_mesh")
    candidates = []
    if dataset:
        candidates.append(os.path.join(base, dataset, obj_id, "mesh.ply"))
    for ds in ("oakink", "ycb", "unseen"):
        candidates.append(os.path.join(base, ds, obj_id, "mesh.ply"))
    return next((p for p in candidates if os.path.isfile(p)), None)


def main():
    parser = argparse.ArgumentParser(description="GraspNet-Baseline inference on a single mesh")
    parser.add_argument("--mesh", type=str, help="Path to mesh.ply")
    parser.add_argument("--obj-id", type=str, help="Object ID (auto-find mesh)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(PROJ, "Baseline2", "graspnet", "checkpoints", "checkpoint-rs.tar"),
        help="Path to GraspNet checkpoint",
    )
    parser.add_argument("--scale-json", type=str, default=None, help="Path to scale.json")
    parser.add_argument("--n-points", type=int, default=20000, help="Number of surface sample points")
    parser.add_argument("--n-top", type=int, default=50, help="Top-K grasps to keep")
    parser.add_argument("--collision-thresh", type=float, default=0.01, help="Collision IoU threshold")
    parser.add_argument("--dump-npy", type=str, default=None, help="Dump raw GraspGroup to .npy")
    args = parser.parse_args()

    # Resolve mesh path
    mesh_path = args.mesh
    if mesh_path is None and args.obj_id:
        mesh_path = find_rotated_mesh(args.obj_id, args.dataset)
    if mesh_path is None or not os.path.isfile(mesh_path):
        print(f"❌ Mesh not found: {mesh_path or args.obj_id}")
        sys.exit(1)

    # Resolve scale
    scale_json = args.scale_json
    if scale_json is None and args.obj_id:
        scale_json = find_scale_json(args.obj_id, args.dataset)
    scale_factor = read_scale_factor(scale_json) if scale_json else 1.0

    # Load model
    net = load_model(args.checkpoint)

    # Generate point cloud
    points, mesh = mesh_to_pointcloud(mesh_path, scale_factor, args.n_points)

    # Run inference
    gg = infer_grasps(net, points, n_top=args.n_top, collision_thresh=args.collision_thresh)

    # Dump if requested
    if args.dump_npy and len(gg) > 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_npy)), exist_ok=True)
        np.save(args.dump_npy, gg.grasp_group_array)
        print(f"  💾 Saved {len(gg)} grasps to {args.dump_npy}")

    return gg, points, scale_factor


if __name__ == "__main__":
    main()
