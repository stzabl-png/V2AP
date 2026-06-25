#!/usr/bin/env python3
"""
3-Panel Visualization: Human Prior | Prediction | Overlap
==========================================================

For each eval seen object, renders:
  Left:   Human Prior heatmap (from training_fp, aligned to rotated mesh)
  Center: PointNet++ v6 Prediction heatmap
  Right:  Overlap (TP=green, FP=orange, FN=magenta, TN=gray)

Usage:
    python Baseline2/vis_pred_vs_hp.py \
        --checkpoint /home/lyh/Desktop/best_v6_model.pth \
        --out output/vis_pred_vs_hp

    # Single object
    python Baseline2/vis_pred_vs_hp.py \
        --checkpoint /home/lyh/Desktop/best_v6_model.pth \
        --obj A01001
"""

import argparse
import csv
import json
import os
import sys

import h5py
import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Paths ────────────────────────────────────────────────────────────────────
EVAL_CSV = os.path.join(PROJ, "evaluation", "configs", "eval_objects_all.csv")
MESH_DIR = os.path.join(PROJ, "data_hub", "meshes", "SAM3DMesh", "rotated_mesh")
SCALE_DIR = os.path.join(PROJ, "data_hub", "ProcessedData", "obj_meshes")
HP_DIR = os.path.join(PROJ, "data_hub", "ProcessedData", "training_fp")

N_POINTS = 4096
N_VIS = 15000
SEED = 42


def mesh_subdir(obj_id):
    return "ycb" if obj_id.startswith("ycb_dex_") else "oakink"


def hp_subdir(obj_id):
    return "dexycb" if obj_id.startswith("ycb_dex_") else "oakink"


def load_mesh_scaled(obj_id):
    sub = mesh_subdir(obj_id)
    mesh_path = os.path.join(MESH_DIR, sub, obj_id, "mesh.ply")
    scale_path = os.path.join(SCALE_DIR, sub, obj_id, "scale.json")
    mesh = trimesh.load(mesh_path, force="mesh")
    sf = 1.0
    if os.path.isfile(scale_path):
        with open(scale_path) as f:
            sf = json.load(f).get("scale_factor", 1.0)
    if abs(sf - 1.0) > 1e-8:
        mesh.vertices = mesh.vertices * sf
    return mesh, sf


def load_hp_aligned(obj_id, sf, target_pts):
    sub = hp_subdir(obj_id)
    hp_path = os.path.join(HP_DIR, sub, f"{obj_id}.hdf5")
    if not os.path.isfile(hp_path):
        for alt in ["oakink", "dexycb"]:
            alt_path = os.path.join(HP_DIR, alt, f"{obj_id}.hdf5")
            if os.path.isfile(alt_path):
                hp_path = alt_path
                break
    with h5py.File(hp_path, "r") as f:
        hp_pc = f["point_cloud"][:].astype(np.float64)
        hp_labels = f["human_prior"][:].astype(np.float32)
    # Rx(+90°)
    rot = hp_pc.copy()
    rot[:, 1] = -hp_pc[:, 2]
    rot[:, 2] = hp_pc[:, 1]
    rot *= sf
    tree = cKDTree(rot)
    _, nn = tree.query(target_pts.astype(np.float64), k=1)
    return hp_labels[nn]


def render_heatmap(ax, pts, values, title, elev=25, azim=135, cmap_name="jet"):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.clip(values, 0, 1))
    order = np.argsort(values)
    pts = pts[order]
    colors = colors[order]

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=colors, s=1.2, alpha=0.9, edgecolors="none")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#222")
    ax.view_init(elev=elev, azim=azim)

    ext = pts.max(0) - pts.min(0)
    r = ext.max() * 0.6
    c = (pts.max(0) + pts.min(0)) / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_axis_off()


def render_overlap(ax, pts, pred_bin, hp_bin, title, elev=25, azim=135):
    """TP=green, FP=orange, FN=magenta, TN=light gray."""
    colors = np.full((len(pts), 4), [0.82, 0.82, 0.85, 1.0])  # TN gray
    tp = pred_bin & hp_bin
    fp = pred_bin & ~hp_bin
    fn = ~pred_bin & hp_bin
    colors[tp] = [0.15, 0.85, 0.35, 1.0]   # green
    colors[fp] = [1.0, 0.55, 0.1, 1.0]      # orange
    colors[fn] = [0.85, 0.15, 0.75, 1.0]    # magenta

    # Draw active points last
    priority = tp.astype(int) * 3 + fn.astype(int) * 2 + fp.astype(int) * 1
    order = np.argsort(priority)
    pts = pts[order]
    colors = colors[order]

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=colors, s=1.2, alpha=0.9, edgecolors="none")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#222")
    ax.view_init(elev=elev, azim=azim)

    ext = pts.max(0) - pts.min(0)
    r = ext.max() * 0.6
    c = (pts.max(0) + pts.min(0)) / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_axis_off()


def vis_object(obj_id, model, device, thresh, out_dir, idx=0):
    from model.inference_v6 import predict_heatmap_batch

    mesh, sf = load_mesh_scaled(obj_id)

    # Sample points for model input
    pts_model, face_idx = trimesh.sample.sample_surface(mesh, N_POINTS, seed=SEED + idx)
    pts_model = pts_model.astype(np.float32)
    nrm_model = mesh.face_normals[face_idx].astype(np.float32)
    nrm_model /= np.linalg.norm(nrm_model, axis=1, keepdims=True) + 1e-8

    # Run prediction
    pred = predict_heatmap_batch(
        model, pts_model[np.newaxis], nrm_model[np.newaxis], device
    )[0]

    # Load HP
    hp = load_hp_aligned(obj_id, sf, pts_model)

    # Dense vis: sample more points, KNN interpolate
    pts_vis, fi_vis = trimesh.sample.sample_surface(mesh, N_VIS, seed=SEED + idx + 1000)
    pts_vis = pts_vis.astype(np.float32)

    tree = cKDTree(pts_model)
    dists, nn3 = tree.query(pts_vis, k=3)
    w = 1.0 / (dists + 1e-8)
    w /= w.sum(axis=1, keepdims=True)
    hp_vis = np.sum(hp[nn3] * w, axis=1)
    pred_vis = np.sum(pred[nn3] * w, axis=1)

    pred_bin = pred_vis > thresh
    hp_bin = hp_vis > thresh

    # Metrics
    tp = int(np.sum(pred_bin & hp_bin))
    fp = int(np.sum(pred_bin & ~hp_bin))
    fn = int(np.sum(~pred_bin & hp_bin))
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    # ── Plot ──
    fig = plt.figure(figsize=(20, 6.5), facecolor="white")

    ax1 = fig.add_subplot(131, projection="3d")
    render_heatmap(ax1, pts_vis, hp_vis,
                   f"Human Prior\nmax={hp_vis.max():.2f}  pos={hp_bin.mean()*100:.1f}%")

    ax2 = fig.add_subplot(132, projection="3d")
    render_heatmap(ax2, pts_vis, pred_vis,
                   f"Prediction (v6)\nmax={pred_vis.max():.2f}  pos={pred_bin.mean()*100:.1f}%")

    ax3 = fig.add_subplot(133, projection="3d")
    render_overlap(ax3, pts_vis, pred_bin, hp_bin,
                   f"Overlap  F1={f1:.3f}\n🟢TP={tp} 🟠FP={fp} 🟣FN={fn}")

    fig.suptitle(
        f"{obj_id}   |   P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}   τ={thresh}",
        fontsize=14, fontweight="bold", y=0.99, color="#333",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{obj_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [{idx+1:3d}] {obj_id:14s}  F1={f1:.3f}  → {out_path}")
    return f1


def main():
    parser = argparse.ArgumentParser(description="3-Panel: HP vs Prediction")
    parser.add_argument("--checkpoint", default="/home/lyh/Desktop/best_v6_model.pth")
    parser.add_argument("--obj", nargs="*", default=None, help="Object IDs (default: all 87 seen)")
    parser.add_argument("--thresh", type=float, default=0.3)
    parser.add_argument("--out", default=os.path.join(PROJ, "output", "vis_pred_vs_hp"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    from model.inference_v6 import load_model

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load_model(args.checkpoint, device)
    print(f"Model: epoch={ckpt.get('epoch', '?')}  device={device}  thresh={args.thresh}")

    # Object list
    if args.obj:
        objects = args.obj
    else:
        objects = []
        with open(EVAL_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                oid = row["obj_id"].strip()
                if not oid.startswith("unseen_"):
                    objects.append(oid)

    print(f"Objects: {len(objects)}")
    f1s = []
    for i, oid in enumerate(objects):
        try:
            f1 = vis_object(oid, model, device, args.thresh, args.out, idx=i)
            f1s.append(f1)
        except Exception as e:
            print(f"  [{i+1:3d}] {oid:14s}  ❌ {e}")

    if f1s:
        print(f"\n{'='*55}")
        print(f"  {len(f1s)} objects  |  Mean F1={np.mean(f1s):.4f}  Median={np.median(f1s):.4f}")
        print(f"  Output: {args.out}")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
