#!/usr/bin/env python3
"""
Visualize RP+PDM results: Affordance Heatmap + Successful Grasp Poses

For each of the 116 evaluation objects, generates a 2-panel figure:
  Left:  RP (Robot Posterior) heatmap predicted by v6 model (jet colormap, 3panel-style)
  Right: Successful grasp poses from RP+PDM evaluation (gripper visualization)

Usage:
    python scripts/vis_rp_pdm_results.py --batch --out output/vis_rp_pdm
    python scripts/vis_rp_pdm_results.py --obj A01001
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import trimesh

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Data paths ──
MESH_BASE = PROJ / "data_hub" / "meshes" / "SAM3DMesh" / "rotated_mesh"
SCALE_BASE = PROJ / "data_hub" / "ProcessedData" / "obj_meshes"
EVAL_DIR = PROJ / "output" / "evaluation" / "rp_pdm_yaw4x10_random_xy_seed42"
EVAL_CSV = PROJ / "evaluation" / "configs" / "eval_objects_all.csv"
AFF_CKPT = PROJ / "output" / "affordance_no_rot_executed" / "min20" / "checkpoints_v6" / "best_v6_model.pth"
N_POINTS = 4096
N_VIS = 40000  # Dense point cloud for heatmap rendering


def infer_dataset(obj_id: str) -> str:
    if obj_id.startswith("ycb_dex_"):
        return "ycb"
    if obj_id.startswith("unseen_"):
        return "unseen"
    return "oakink"


def find_mesh(obj_id: str) -> Path | None:
    ds = infer_dataset(obj_id)
    p = MESH_BASE / ds / obj_id / "mesh.ply"
    return p if p.exists() else None


def find_scale(obj_id: str) -> float:
    ds = infer_dataset(obj_id)
    p = SCALE_BASE / ds / obj_id / "scale.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)["scale_factor"]
    return 1.0


def load_successful_grasps(obj_id: str) -> list[dict]:
    """Load successful grasp poses from episodes.jsonl."""
    episodes_file = EVAL_DIR / "episodes.jsonl"
    if not episodes_file.exists():
        return []
    grasps = []
    with open(episodes_file) as f:
        for line in f:
            ep = json.loads(line)
            if ep.get("obj_id") != obj_id:
                continue
            if not ep.get("success"):
                continue
            cmd = ep.get("policy_output", {}).get("command", {})
            pos = cmd.get("position")
            rot = cmd.get("rotation")
            if pos is not None and rot is not None:
                grasps.append({
                    "position": np.array(pos, dtype=float),
                    "rotation": np.array(rot, dtype=float),
                    "score": cmd.get("score", 0),
                    "gripper_width": cmd.get("gripper_width", 0.06),
                    "yaw": ep.get("yaw_deg", 0),
                    "trial": ep.get("trial", 0),
                })
    return grasps


def render_heatmap(ax, points, values, title, elev=25, azim=135):
    """Render point cloud with jet heatmap (same style as 3panel middle)."""
    cmap = plt.get_cmap("jet")
    vmin, vmax = 0.0, max(values.max(), 0.01)
    colors = cmap(np.clip(values / vmax, 0, 1))

    # Sort: low values first, high values on top
    order = np.argsort(values)
    points = points[order]
    colors = colors[order]

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, s=2.5, alpha=0.95, edgecolors="none")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect
    extents = points.max(0) - points.min(0)
    max_ext = extents.max() * 0.6
    center = (points.max(0) + points.min(0)) / 2
    for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        setter(c - max_ext, c + max_ext)
    ax.set_axis_off()


def draw_gripper_3d(ax, center, rotation, width=0.06, depth=0.08,
                    color="green", alpha=0.9):
    """Draw parallel-jaw gripper as bold lines."""
    finger_dir = rotation[:, 0]
    approach = rotation[:, 2]

    base = center - approach * depth
    left = base + finger_dir * (width / 2)
    right = base - finger_dir * (width / 2)
    left_tip = left + approach * depth * 1.3
    right_tip = right + approach * depth * 1.3

    lw = 4.0
    ax.plot(*zip(left, right), color=color, linewidth=lw, alpha=alpha, zorder=10)
    ax.plot(*zip(left, left_tip), color=color, linewidth=lw, alpha=alpha, zorder=10)
    ax.plot(*zip(right, right_tip), color=color, linewidth=lw, alpha=alpha, zorder=10)
    # Approach stem
    stem_start = center - approach * depth * 1.8
    ax.plot(*zip(stem_start, center), color=color, linewidth=2.5, linestyle="-", alpha=alpha*0.8, zorder=10)
    # Center dot
    ax.scatter(*center, color=color, s=60, zorder=11, alpha=alpha, edgecolors="black", linewidths=0.8)


def render_grasps(ax, points, grasps, title, scale_factor, elev=25, azim=135):
    """Render mesh point cloud with successful grasp poses overlaid."""
    # Point cloud in light gray
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c="steelblue", s=0.8, alpha=0.3, edgecolors="none")

    # Color palette for grasps
    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(grasps):
        color = cmap(i / max(len(grasps), 1))
        draw_gripper_3d(ax, g["position"], g["rotation"],
                        width=g["gripper_width"], color=color, alpha=0.85)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.view_init(elev=elev, azim=azim)

    extents = points.max(0) - points.min(0)
    max_ext = extents.max() * 0.6
    center = (points.max(0) + points.min(0)) / 2
    for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        setter(c - max_ext, c + max_ext)
    ax.set_axis_off()


def vis_single(obj_id: str, model, device, out_dir: Path, show: bool = False) -> bool:
    """Visualize one object: heatmap + successful grasps."""
    import torch

    mesh_path = find_mesh(obj_id)
    if mesh_path is None:
        print(f"  ⚠️ {obj_id}: mesh not found, skip")
        return False

    scale_factor = find_scale(obj_id)
    mesh = trimesh.load(str(mesh_path), force="mesh")
    verts_metric = np.array(mesh.vertices) * scale_factor

    # ── Stage 1: V6 Affordance Heatmap ──
    # Sample points for inference (N_POINTS)
    pts_infer, face_idx = trimesh.sample.sample_surface(mesh, N_POINTS, seed=42)
    pts_infer = (pts_infer * scale_factor).astype(np.float32)
    nrm_infer = mesh.face_normals[face_idx].astype(np.float32)
    nrm_infer /= np.linalg.norm(nrm_infer, axis=1, keepdims=True) + 1e-8

    # Predict heatmap
    from model.inference_v6 import predict_heatmap_batch
    pred = predict_heatmap_batch(
        model,
        pts_infer[np.newaxis],  # (1, N, 3)
        nrm_infer[np.newaxis],  # (1, N, 3)
        device,
    )[0]  # (N,)

    # Dense sampling for visualization
    pts_vis, face_idx_vis = trimesh.sample.sample_surface(mesh, N_VIS, seed=0)
    pts_vis = (pts_vis * scale_factor).astype(np.float32)

    # KNN interpolation to dense cloud
    from scipy.spatial import cKDTree
    tree = cKDTree(pts_infer)
    _, idx = tree.query(pts_vis, k=3)
    dists = np.linalg.norm(pts_vis[:, None, :] - pts_infer[idx], axis=2)
    weights = 1.0 / (dists + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    pred_dense = np.sum(pred[idx] * weights, axis=1)

    # ── Stage 2: Successful Grasps ──
    grasps = load_successful_grasps(obj_id)
    n_success = len(grasps)

    # Show up to 10 successful grasps
    grasps_show = grasps[:10]

    # ── Render: single merged view ──
    fig = plt.figure(figsize=(10, 9), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False  # Manual z-ordering: grippers on top of heatmap

    # 1. Heatmap point cloud (behind)
    cmap = plt.get_cmap("jet")
    vmax = max(pred_dense.max(), 0.01)
    colors = cmap(np.clip(pred_dense / vmax, 0, 1))
    order = np.argsort(pred_dense)
    ax.scatter(pts_vis[order, 0], pts_vis[order, 1], pts_vis[order, 2],
               c=colors[order], s=2.5, alpha=0.85, edgecolors="none", zorder=1)

    # 2. Gripper poses (in front, retreated 20cm along approach for visibility)
    RETREAT_M = 0.10
    grasp_cmap = plt.get_cmap("tab10")
    for i, g in enumerate(grasps_show):
        color = grasp_cmap(i / max(len(grasps_show), 1))
        approach = g["rotation"][:, 2]
        retreated_pos = g["position"] - approach * RETREAT_M
        draw_gripper_3d(ax, retreated_pos, g["rotation"],
                        width=g["gripper_width"], color=color, alpha=0.9)

    # Axes & limits
    extents = pts_vis.max(0) - pts_vis.min(0)
    max_ext = extents.max() * 0.65
    center = (pts_vis.max(0) + pts_vis.min(0)) / 2
    for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        setter(c - max_ext, c + max_ext)
    ax.view_init(elev=25, azim=135)
    ax.set_axis_off()

    title = (f"{obj_id}   RP Affordance + Grasps\n"
             f"heatmap max={pred_dense.max():.2f}   "
             f"success={n_success}/40   shown={len(grasps_show)}")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{obj_id}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  ✅ {obj_id}: pred_max={pred_dense.max():.2f}  success={n_success}/40  → {out_path.name}")
    return True


def main():
    import torch
    from model.inference_v6 import load_model

    p = argparse.ArgumentParser(description="Visualize RP+PDM: Heatmap + Successful Grasps")
    p.add_argument("--obj", type=str, help="Single object ID")
    p.add_argument("--batch", action="store_true", help="Process all eval objects")
    p.add_argument("--out", type=str, default=str(PROJ / "output" / "vis_rp_pdm"))
    p.add_argument("--checkpoint", type=str, default=str(AFF_CKPT))
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)

    # Load v6 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.checkpoint, device)
    print(f"✅ V6 model loaded: {args.checkpoint}")

    if args.obj:
        vis_single(args.obj, model, device, out_dir, show=args.show)
    elif args.batch:
        # Load all eval object IDs
        with open(EVAL_CSV) as f:
            obj_ids = [r["obj_id"] for r in csv.DictReader(f)]
        print(f"Processing {len(obj_ids)} objects → {out_dir}\n")
        ok = 0
        for i, oid in enumerate(obj_ids):
            print(f"[{i+1}/{len(obj_ids)}]", end="")
            if vis_single(oid, model, device, out_dir):
                ok += 1
        print(f"\n{'='*50}")
        print(f"  Done! {ok}/{len(obj_ids)} visualized → {out_dir}")
        print(f"{'='*50}")
    else:
        print("Specify --obj or --batch")


if __name__ == "__main__":
    main()
