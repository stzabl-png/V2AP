#!/usr/bin/env python3
"""
test_foundation_pose.py — FoundationPose 管线验证脚本

用 arctic box 序列（已有 Depth Pro 深度）验证完整位姿估计流程：
  RGB + Depth + Mask + Mesh → FoundationPose → 6-DoF pose per frame

用法:
  conda activate bundlesdf
  cd $PROJ
  python tools/test_foundation_pose.py

输出: data_hub/ProcessedData/obj_poses/{dataset}/{seq_id}/
        ob_in_cam_{frame_idx}.txt   ← 4×4 pose matrix
        track_vis/                  ← pose overlay 可视化
"""

import os, sys, argparse
import numpy as np
import cv2
from glob import glob
from natsort import natsorted
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config as _cfg
FOUNDATION_POSE_ROOT = _cfg.FP_ROOT
sys.path.insert(0, FOUNDATION_POSE_ROOT)

MESH_BASE  = os.path.join(config.DATA_HUB, "ProcessedData", "obj_meshes")
DEPTH_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "third_depth")
RAW_BASE   = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")
OBJ_INPUT  = os.path.join(config.DATA_HUB, "ProcessedData", "obj_recon_input")
OUT_BASE   = os.path.join(config.DATA_HUB, "ProcessedData", "obj_poses")

# ── Test cases: (seq_id, dataset_key, obj_name, depth_dir_name) ───────────────
TEST_CASES = [
    ("s01__box_grab_01", "arctic", "box"),
    ("s01__box_use_01",  "arctic", "box"),
    ("s01__box_use_02",  "arctic", "box"),
]


def load_depth_frames(depth_dir):
    """Load depth stack from depths.npz. Returns (N, H, W) float32 in meters."""
    npz_path = os.path.join(depth_dir, "depths.npz")
    if not os.path.exists(npz_path):
        return None, None
    data = np.load(npz_path)
    depths = data["depths"].astype(np.float32)  # (N, H, W) in meters
    frame_ids = open(os.path.join(depth_dir, "frame_ids.txt")).read().strip().split("\n")
    return depths, frame_ids


def find_rgb_frames(raw_base, dataset, seq_id):
    """Find all RGB frames for a sequence."""
    if dataset == "arctic":
        subj, seq = seq_id.split("__", 1)
        cam_dir = os.path.join(raw_base, "arctic", subj, seq, "1")
        return natsorted(glob(os.path.join(cam_dir, "*.jpg")))
    return []


def prepare_scene_dir(seq_id, dataset, obj_name, scene_dir, n_frames=50):
    """
    Prepare FoundationPose scene directory.
    Only uses frames that have corresponding depth data (sparse sampling).
    """
    depth_dir_path = os.path.join(DEPTH_BASE, dataset, seq_id)
    depth_stack, frame_ids = load_depth_frames(depth_dir_path)
    if depth_stack is None:
        print(f"  ⚠️  No depth data at {depth_dir_path}"); return None
    K = np.loadtxt(os.path.join(depth_dir_path, "K.txt"))

    # Load annotated mask
    ann_mask = os.path.join(OBJ_INPUT, dataset, obj_name, "0.png")
    if not os.path.exists(ann_mask):
        print(f"  ⚠️  No annotated mask for {dataset}/{obj_name}"); return None

    # Build map: frame_basename → (rgb_path, depth_idx)
    all_rgb = find_rgb_frames(RAW_BASE, dataset, seq_id)
    rgb_by_name = {os.path.basename(p): p for p in all_rgb}

    # Select only frames that have BOTH rgb AND depth
    valid_pairs = []   # list of (rgb_path, depth_idx)
    for di, fname in enumerate(frame_ids):
        bname = os.path.basename(fname)
        if bname in rgb_by_name:
            valid_pairs.append((rgb_by_name[bname], di))

    if not valid_pairs:
        print(f"  ⚠️  No overlapping frames between RGB and depth for {seq_id}")
        return None

    # Start from beginning (annotated frame may not be in depth subset)
    pairs_to_use = valid_pairs[:n_frames]
    if not pairs_to_use:
        print(f"  ⚠️  Not enough valid pairs"); return None

    os.makedirs(os.path.join(scene_dir, "color"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "masks"), exist_ok=True)

    H, W = None, None
    for i, (rgb_path, dep_idx) in enumerate(pairs_to_use):
        out_id = f"{i:06d}"

        # Color
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        if H is None: H, W = rgb.shape[:2]
        cv2.imwrite(os.path.join(scene_dir, "color", f"{out_id}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Depth: resize to match RGB, convert m → mm uint16
        depth_m = depth_stack[dep_idx]
        depth_m = cv2.resize(depth_m, (W, H))
        depth_mm = (depth_m * 1000).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(scene_dir, "depth", f"{out_id}.png"), depth_mm)

    # Mask for frame 0 (resize to match RGB)
    mask = cv2.imread(ann_mask, 0)
    if mask is not None and H is not None:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(scene_dir, "masks", "000000.png"), mask)

    # Camera K
    np.savetxt(os.path.join(scene_dir, "cam_K.txt"), K, fmt="%.6f")

    print(f"  Scene dir: {len(pairs_to_use)} frames with depth, H={H} W={W}, "
          f"depth range={depth_stack[pairs_to_use[0][1]].max():.2f}m")
    return K


def run_foundation_pose(mesh_path, scene_dir, out_dir, est_iter=5, track_iter=2):
    """Run FoundationPose on prepared scene dir. Returns True on success."""
    import trimesh
    import torch
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, set_logging_format, set_seed, draw_posed_3d_box, draw_xyz_axis, depth2xyzmap
    import nvdiffrast.torch as dr
    import imageio

    os.makedirs(out_dir, exist_ok=True)
    set_seed(0)

    mesh = trimesh.load(mesh_path)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
        mesh=mesh, scorer=scorer, refiner=refiner,
        debug_dir=out_dir, debug=2, glctx=glctx
    )

    # Load scene
    K      = np.loadtxt(os.path.join(scene_dir, "cam_K.txt"))
    colors = natsorted(glob(os.path.join(scene_dir, "color", "*.png")))
    depths = natsorted(glob(os.path.join(scene_dir, "depth", "*.png")))
    mask0  = cv2.imread(os.path.join(scene_dir, "masks", "000000.png"), 0)

    os.makedirs(os.path.join(out_dir, "ob_in_cam"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "track_vis"),  exist_ok=True)

    pose = None
    for i, (c_path, d_path) in enumerate(zip(colors, depths)):
        color = cv2.cvtColor(cv2.imread(c_path), cv2.COLOR_BGR2RGB).astype(np.uint8)
        depth_raw = cv2.imread(d_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0  # mm→m
        depth_raw[depth_raw < 0.001] = 0

        if i == 0:
            ob_mask = (mask0 > 0).astype(bool)
            pose = est.register(K=K, rgb=color, depth=depth_raw,
                                ob_mask=ob_mask, iteration=est_iter)
            print(f"    Frame 0: pose registered")
        else:
            pose = est.track_one(rgb=color, depth=depth_raw, K=K, iteration=track_iter)

        frame_id = os.path.splitext(os.path.basename(c_path))[0]
        np.savetxt(os.path.join(out_dir, "ob_in_cam", f"{frame_id}.txt"),
                   pose.reshape(4, 4), fmt="%.6f")

        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K,
                            thickness=3, transparency=0, is_input_rgb=True)
        imageio.imwrite(os.path.join(out_dir, "track_vis", f"{frame_id}.png"), vis)

        if i % 10 == 0:
            print(f"    Frame {i}/{len(colors)} done")

    print(f"  ✅ Poses saved → {out_dir}/ob_in_cam/")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=30, help="Frames per sequence to test")
    args = parser.parse_args()

    # Prepare tmp scene dirs
    tmp_root = "/tmp/fp_test_scenes"

    success = failure = 0
    for seq_id, dataset, obj_name in TEST_CASES:
        print(f"\n{'─'*60}")
        print(f"▶ {dataset}/{seq_id}  (obj: {obj_name})")

        mesh_path = os.path.join(MESH_BASE, dataset, obj_name, "mesh.ply")
        if not os.path.exists(mesh_path):
            print(f"  ⚠️  Mesh not found: {mesh_path}"); failure += 1; continue

        scene_dir = os.path.join(tmp_root, seq_id)
        K = prepare_scene_dir(seq_id, dataset, obj_name, scene_dir, n_frames=args.n_frames)
        if K is None:
            failure += 1; continue

        out_dir = os.path.join(OUT_BASE, dataset, seq_id)
        try:
            run_foundation_pose(mesh_path, scene_dir, out_dir)
            success += 1
        except Exception as e:
            import traceback
            print(f"  ❌ {e}")
            traceback.print_exc()
            failure += 1

    print(f"\n{'='*60}")
    print(f"✅ {success} succeeded  ❌ {failure} failed")
    print(f"Results: {OUT_BASE}")


if __name__ == "__main__":
    main()
