#!/usr/bin/env python3
"""
GraspNet-Baseline inference directly from RGB-D session data.
==============================================================
Bypasses the full T2-T5 perception pipeline (SAM, SAM3D, FoundationPose)
by extracting the object point cloud directly from the depth image.

Segmentation modes (--seg):
  color   : HSV color detection (default, good for colorful objects like Pringles)
  roi     : Fixed image-space rectangle (--roi u,v,w,h in pixels)
  mask    : External mask PNG (--mask-path path/to/mask.png)
  center  : Center 160×160 px square (simplest fallback)

Usage:
    # Color segmentation (auto-detect colored object)
    python -m s2r.graspnet_from_rgbd \
        --session-dir data_hub/sessions/sessions/20260602_192346_chips

    # Manual ROI: rectangle at (280,90) size 60x140 px
    python -m s2r.graspnet_from_rgbd \
        --session-dir data_hub/sessions/sessions/20260602_192346_chips \
        --seg roi --roi 280,90,60,140

    # External mask (e.g. from SAM, manual annotation)
    python -m s2r.graspnet_from_rgbd \
        --session-dir data_hub/sessions/sessions/20260602_192346_chips \
        --seg mask --mask-path /path/to/mask.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))


# ──────────────────────────────────────────────────────────
# Depth → Point Cloud
# ──────────────────────────────────────────────────────────

def depth_to_pointcloud(
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert depth image → 3D point cloud in camera frame.

    Args:
        depth: (H, W) float32, meters. 0 = invalid.
        K: (3, 3) intrinsic matrix.
        mask: (H, W) bool, True = keep.

    Returns:
        (N, 3) float32 points in camera frame.
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    valid = depth > 0.01
    if mask is not None:
        valid = valid & mask

    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    return np.column_stack([x, y, z]).astype(np.float32)


# ──────────────────────────────────────────────────────────
# Segmentation modes
# ──────────────────────────────────────────────────────────

def segment_by_color(rgb: np.ndarray) -> np.ndarray:
    """Detect the largest saturated-color object via strict HSV thresholds.

    Uses high saturation/value minimums to avoid wood grain, beige tones, etc.
    Returns: (H, W) uint8 mask (255 = object).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

    # Strict thresholds: S≥100, V≥80 → only vivid colors
    S_MIN, V_MIN = 100, 80

    # Red (two ranges because hue wraps at 0/180)
    r1 = cv2.inRange(hsv, (0, S_MIN, V_MIN), (10, 255, 255))
    r2 = cv2.inRange(hsv, (160, S_MIN, V_MIN), (180, 255, 255))
    # Orange/Yellow
    oy = cv2.inRange(hsv, (10, S_MIN, V_MIN), (35, 255, 255))
    # Green
    gr = cv2.inRange(hsv, (35, S_MIN, V_MIN), (85, 255, 255))
    # Blue
    bl = cv2.inRange(hsv, (85, S_MIN, V_MIN), (130, 255, 255))
    # Purple
    pu = cv2.inRange(hsv, (130, S_MIN, V_MIN), (160, 255, 255))

    combined = r1 | r2 | oy | gr | bl | pu

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # Filter small contours (noise), keep only objects ≥200px area
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 200]
    if not contours:
        print("  ⚠️  Color seg: no vivid-color object found")
        return combined

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(combined)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    # Slight dilation to catch depth edges
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))

    area = cv2.contourArea(largest)
    x, y, w, h = cv2.boundingRect(largest)
    print(f"  🎨 Color seg: bbox=({x},{y},{w},{h}) area={area:.0f}px "
          f"(S≥{S_MIN}, V≥{V_MIN})")

    return mask


def segment_by_roi(H: int, W: int, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Fixed rectangle ROI. roi = (u, v, width, height) in pixels."""
    u, v, rw, rh = roi
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[max(0, v):min(H, v + rh), max(0, u):min(W, u + rw)] = 255
    print(f"  📐 ROI seg: ({u},{v},{rw},{rh})")
    return mask


def segment_by_center(H: int, W: int, size: int = 160) -> np.ndarray:
    """Center square."""
    mask = np.zeros((H, W), dtype=np.uint8)
    cy, cx = H // 2, W // 2
    r = size // 2
    mask[max(0, cy - r):min(H, cy + r), max(0, cx - r):min(W, cx + r)] = 255
    print(f"  ⬜ Center seg: {size}×{size}px at ({cx},{cy})")
    return mask


def load_mask_image(path: str | Path, H: int, W: int) -> np.ndarray:
    """Load external mask PNG (white = object)."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    n_pixels = (mask > 0).sum()
    print(f"  🖼️  Mask loaded: {path} ({n_pixels} pixels)")
    return mask


# ──────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────

def process_rgbd_session(
    session_dir: str | Path,
    *,
    device: str = "cuda",
    seg_mode: str = "color",
    roi: tuple[int, int, int, int] | None = None,
    mask_path: str | None = None,
    n_top: int = 50,
    max_candidates_json: int = 10,
    table_height_m: float | None = None,
    min_points: int = 200,
    target_points: int = 20000,
) -> Path:
    """Run GraspNet on RGB-D session with configurable segmentation.

    Returns:
        Path to candidates.json
    """
    t0 = time.time()
    session_dir = Path(session_dir)
    input_dir = session_dir / "input"
    output_dir = session_dir / "output" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  GraspNet from RGB-D (seg={seg_mode})")
    print(f"{'='*60}")
    print(f"  Session: {session_dir}")

    # ── 1. Load data ───────────────────────────────────
    depth = np.load(str(input_dir / "depth" / "depth.npy"))
    K = np.load(str(input_dir / "calib" / "K.npy"))
    rgb = cv2.imread(str(input_dir / "rgb" / "left_rgb.png"))

    with open(input_dir / "calib" / "extrinsics.json") as f:
        ext = json.load(f)
    T_base_cam = np.array(ext["T_base_cam"], dtype=np.float64)

    if table_height_m is None:
        table_path = input_dir / "scene" / "table.json"
        if table_path.exists():
            with open(table_path) as f:
                table_height_m = json.load(f).get("table_height_m", 0.98)
        else:
            table_height_m = 0.98

    H, W = depth.shape
    print(f"  Image:  {W}×{H}")
    print(f"  Depth:  [{depth[depth>0].min():.3f}, {depth[depth>0].max():.3f}]m")
    print(f"  Table:  {table_height_m:.2f}m")

    # ── 2. Segment object ──────────────────────────────
    if seg_mode == "color":
        obj_mask_img = segment_by_color(rgb)
    elif seg_mode == "roi":
        if roi is None:
            raise ValueError("--roi u,v,w,h required for roi mode")
        obj_mask_img = segment_by_roi(H, W, roi)
    elif seg_mode == "mask":
        if mask_path is None:
            raise ValueError("--mask-path required for mask mode")
        obj_mask_img = load_mask_image(mask_path, H, W)
    elif seg_mode == "center":
        obj_mask_img = segment_by_center(H, W)
    else:
        raise ValueError(f"Unknown seg mode: {seg_mode}")

    # Save mask for debugging
    seg_dir = session_dir / "output" / "segment"
    seg_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(seg_dir / "mask.png"), obj_mask_img)

    # ── 3. Extract object point cloud ──────────────────
    obj_mask_bool = (obj_mask_img > 0) & (depth > 0.01)
    obj_pts_cam = depth_to_pointcloud(depth, K, obj_mask_bool)

    if len(obj_pts_cam) < min_points:
        print(f"  ❌ Only {len(obj_pts_cam)} pts (need ≥{min_points})")
        _write_empty_candidates(output_dir, T_base_cam)
        _write_status(session_dir, success=False, error="Too few object points")
        return output_dir / "candidates.json"

    # Transform to base frame
    R_bc = T_base_cam[:3, :3]
    t_bc = T_base_cam[:3, 3]
    obj_pts_base = (R_bc @ obj_pts_cam.T).T + t_bc
    obj_pts_base = obj_pts_base.astype(np.float32)

    # Height filter: remove table-level points (z < table + 0.5cm)
    # These leak in from mask edges and contaminate the object cloud
    n_before = len(obj_pts_base)
    above_table = obj_pts_base[:, 2] > (table_height_m + 0.005)
    obj_pts_base = obj_pts_base[above_table]
    n_removed = n_before - len(obj_pts_base)
    if n_removed > 0:
        print(f"  Height filter: {n_before} → {len(obj_pts_base)} pts "
              f"(removed {n_removed} at table level)")

    if len(obj_pts_base) < min_points:
        print(f"  ❌ Only {len(obj_pts_base)} pts after height filter (need ≥{min_points})")
        _write_empty_candidates(output_dir, T_base_cam)
        _write_status(session_dir, success=False, error="Too few object points after height filter")
        return output_dir / "candidates.json"

    centroid = obj_pts_base.mean(axis=0)
    extents = obj_pts_base.max(axis=0) - obj_pts_base.min(axis=0)
    print(f"  Object: {len(obj_pts_base)} pts")
    print(f"  Centroid (base): [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
    print(f"  Extents (cm): X={extents[0]*100:.1f} Y={extents[1]*100:.1f} Z={extents[2]*100:.1f}")

    # ── 4. Prepare for GraspNet (center + resample) ────
    pts_centered = obj_pts_base - centroid

    if len(pts_centered) >= target_points:
        idx = np.random.choice(len(pts_centered), target_points, replace=False)
        pts_input = pts_centered[idx]
    else:
        # Upsample with small noise
        n_need = target_points - len(pts_centered)
        idx = np.random.choice(len(pts_centered), n_need, replace=True)
        noise = np.random.randn(n_need, 3).astype(np.float32) * 0.001
        pts_input = np.concatenate([pts_centered, pts_centered[idx] + noise])

    print(f"  GraspNet input: {len(pts_input)} pts (resampled)")

    # ── 5. Run GraspNet ────────────────────────────────
    from Baseline2.graspnet.graspnet_infer import load_model, infer_grasps

    net = load_model(
        str(PROJ / "Baseline2" / "graspnet" / "checkpoints" / "checkpoint-rs.tar"),
        device=device,
    )
    gg = infer_grasps(net, pts_input, n_top=n_top)

    if len(gg) == 0:
        print("  ⚠️  GraspNet: 0 grasps")
        _write_empty_candidates(output_dir, T_base_cam)
        _write_status(session_dir, success=True, warning="No grasps produced")
        return output_dir / "candidates.json"

    # ── 6. Convert to base frame ───────────────────────
    candidates = []
    for i in range(min(len(gg), max_candidates_json)):
        g = gg[i]
        t_base = (g.translation + centroid).tolist()   # un-center → base frame
        R = g.rotation_matrix.tolist()
        candidates.append({
            "rank": i,
            "name": f"graspnet_{i}",
            "score": round(float(g.score), 4),
            # grasp_point in base frame (T_base_mesh=I, so this IS the pinch in base)
            "grasp_point": [round(float(x), 5) for x in t_base],
            # rotation 3x3: col2 = approach direction (GraspNet convention = UCB convention)
            "rotation": [[round(float(x), 5) for x in row] for row in R],
            "gripper_width_m": round(float(g.width), 4),
            "approach_type": "graspnet_6dof",
            # position_panda_hand: not applicable for Dexmate, set None
            "position_panda_hand": None,
        })

    # ── 7. Write candidates.json (V2AP / TITAN_OUTPUT.md compatible) ──────
    # T_base_mesh = eye(4) is correct: grasp_point is already in base frame.
    # V2AP retarget.py does: T_base_pinch = T_base_mesh @ T_mesh_pinch
    # With T_base_mesh=I: T_base_pinch = T_mesh_pinch = grasp_point directly.
    object_slug = session_dir.name.split("_")[-1]
    output = {
        "schema_version": "1.1",
        # V2AP run_auto_grasp reads T_base_mesh from here
        "mesh_frame": "base_aligned",
        "base_frame": "base",
        "camera_frame": ext.get("camera_frame", "zed_left_camera"),
        "registration": {
            "method": "graspnet_rgbd",
            "T_cam_mesh": np.eye(4).tolist(),   # no mesh; placeholder
            "T_base_mesh": np.eye(4).tolist(),  # grasp_point already in base
            "T_base_cam": T_base_cam.tolist(),
            "T_base_cam_source": "input/calib/extrinsics.json",
            "object_centroid_base": centroid.tolist(),
            "notes": "GraspNet point-cloud inference; no FoundationPose mesh; T_base_mesh=I",
        },
        # Top-level T_base_mesh required by V2AP retarget.load_titan_output()
        "T_base_mesh": np.eye(4).tolist(),
        "conventions": {
            # GraspNet col2 = approach direction — matches UCB convention
            "rotation_columns": ["finger_open", "y_body", "approach"],
            "approach_column_index": 2,
            "grasp_point_frame": "base",
            "ucb_tcp_offset_m": 0.105,
            "pre_grasp_offset_m": 0.15,
            "lift_height_m": 0.15,
        },
        # mesh_span_m used by V2AP object_obstacle.py (AABB padding)
        "mesh_span_m": [round(float(x), 4) for x in extents],
        "n_candidates": len(candidates),
        # titan block read by V2AP run_auto_grasp for object_slug
        "titan": {
            "n_candidates": len(candidates),
            "object_slug": object_slug,
            "policy": "graspnet_rgbd",
            "seg_mode": seg_mode,
            "table_height_m": table_height_m,
        },
        "candidates": candidates,
    }

    out_path = output_dir / "candidates.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    tmp_path.rename(out_path)

    _write_status(session_dir, success=True, n_candidates=len(candidates))

    elapsed = time.time() - t0
    print(f"\n  ✅ {len(candidates)} candidates → {out_path}")
    print(f"  ⏱️  {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return out_path


def _write_empty_candidates(output_dir: Path, T_base_cam: np.ndarray):
    out = {"schema_version": "1.1", "T_base_mesh": np.eye(4).tolist(),
           "n_candidates": 0, "source": "graspnet_rgbd", "candidates": []}
    with open(output_dir / "candidates.json", "w") as f:
        json.dump(out, f, indent=2)


def _write_status(session_dir: Path, *, success: bool,
                  error: str = "", warning: str = "",
                  n_candidates: int = 0):
    """Write status.json compatible with V2AP run_auto_grasp.py.
    V2AP reads: status['success'], status['titan']['object_slug'].
    """
    session_dir = Path(session_dir)
    object_slug = session_dir.name.split("_")[-1]
    status = {
        "schema_version": "1.1",
        "session_id": session_dir.name,
        "success": success,
        # V2AP run_auto_grasp reads titan.object_slug
        "titan": {
            "object_slug": object_slug,
            "n_candidates": n_candidates,
            "policy": "graspnet_rgbd",
        },
        "steps": {
            "segment": "ok" if success else "skip",
            "grasp_pose": "ok" if success else "error",
        },
        "warnings": [warning] if warning else [],
        "errors": [error] if error else [],
    }
    (session_dir / "output").mkdir(parents=True, exist_ok=True)
    # Atomic write
    tmp = session_dir / "output" / "status.json.tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    tmp.rename(session_dir / "output" / "status.json")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GraspNet from RGB-D session (no mesh needed)"
    )
    parser.add_argument("--session-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seg", type=str, default="color",
                        choices=["color", "roi", "mask", "center"],
                        help="Segmentation mode")
    parser.add_argument("--roi", type=str, default=None,
                        help="ROI: u,v,w,h in pixels (for --seg roi)")
    parser.add_argument("--mask-path", type=str, default=None,
                        help="External mask PNG (for --seg mask)")
    parser.add_argument("--n-top", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--table-height", type=float, default=None)
    args = parser.parse_args()

    roi = None
    if args.roi:
        roi = tuple(int(x) for x in args.roi.split(","))
        assert len(roi) == 4, "--roi must be u,v,w,h"

    process_rgbd_session(
        session_dir=args.session_dir,
        device=args.device,
        seg_mode=args.seg,
        roi=roi,
        mask_path=args.mask_path,
        n_top=args.n_top,
        max_candidates_json=args.max_candidates,
        table_height_m=args.table_height,
    )


if __name__ == "__main__":
    main()
