#!/usr/bin/env python3
"""
Compute F1 between PointNet++ v6 Prediction vs Human Prior
===========================================================

For each of the 87 seen eval objects:
  1. Load rotated_mesh + scale → sample 4096 surface points + normals
  2. Run PointNet++ v6 prediction → ŷ ∈ [0,1]
  3. Load training_fp HP, apply Rx(+90°) + scale → KNN map to sampled points → h ∈ [0,1]
  4. Binarize both → compute Precision / Recall / F1

Usage:
    python Baseline2/compute_f1_pred_vs_hp.py \
        --checkpoint /home/lyh/Desktop/best_v6_model.pth
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

# ── Paths ────────────────────────────────────────────────────────────────────
EVAL_CSV = os.path.join(PROJ, "evaluation", "configs", "eval_objects_all.csv")
MESH_DIR = os.path.join(PROJ, "data_hub", "meshes", "SAM3DMesh", "rotated_mesh")
SCALE_DIR = os.path.join(PROJ, "data_hub", "ProcessedData", "obj_meshes")
HP_DIR = os.path.join(PROJ, "data_hub", "ProcessedData", "training_fp")

N_POINTS = 4096
SEED = 42


def infer_dataset(obj_id: str) -> str:
    if obj_id.startswith("ycb_dex_"):
        return "dexycb"
    return "oakink"


def mesh_subdir(obj_id: str) -> str:
    """Mesh uses 'ycb' for YCB objects, 'oakink' for everything else."""
    if obj_id.startswith("ycb_dex_"):
        return "ycb"
    return "oakink"


def load_rotated_mesh_scaled(obj_id: str) -> tuple:
    """Load rotated mesh, apply metric scale, return (mesh, scale_factor)."""
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


def sample_surface(mesh, n_points: int, seed: int):
    """Sample surface points + face normals."""
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    pts = pts.astype(np.float32)
    normals = mesh.face_normals[face_idx].astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return pts, normals


def load_hp_aligned(obj_id: str, sf: float, target_pts: np.ndarray) -> np.ndarray:
    """
    Load HP from training_fp, apply Rx(+90°) + scale, KNN-map to target points.
    Returns hp_values (N_POINTS,) in [0,1].
    """
    ds = infer_dataset(obj_id)
    hp_path = os.path.join(HP_DIR, ds, f"{obj_id}.hdf5")
    if not os.path.isfile(hp_path):
        # Try alternative subdirs
        for alt in ["oakink", "dexycb"]:
            alt_path = os.path.join(HP_DIR, alt, f"{obj_id}.hdf5")
            if os.path.isfile(alt_path):
                hp_path = alt_path
                break

    with h5py.File(hp_path, "r") as f:
        hp_pc = f["point_cloud"][:].astype(np.float64)
        hp_labels = f["human_prior"][:].astype(np.float32)

    # Rx(+90°): y' = -z, z' = y  (unrotated → rotated frame)
    hp_rotated = hp_pc.copy()
    hp_rotated[:, 1] = -hp_pc[:, 2]
    hp_rotated[:, 2] = hp_pc[:, 1]

    # Apply metric scale
    hp_rotated *= sf

    # KNN map: for each target surface point, find nearest HP point
    tree = cKDTree(hp_rotated)
    _, nn_idx = tree.query(target_pts.astype(np.float64), k=1)
    return hp_labels[nn_idx]


def compute_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray) -> dict:
    """Compute precision, recall, F1 from binary arrays."""
    tp = np.sum(pred_binary & gt_binary)
    fp = np.sum(pred_binary & ~gt_binary)
    fn = np.sum(~pred_binary & gt_binary)
    tn = np.sum(~pred_binary & ~gt_binary)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute F1: PointNet++ Prediction vs Human Prior"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="/home/lyh/Desktop/best_v6_model.pth",
    )
    parser.add_argument("--pred-thresh", type=float, default=0.3,
                        help="Threshold to binarize prediction (default: 0.3)")
    parser.add_argument("--hp-thresh", type=float, default=0.3,
                        help="Threshold to binarize human prior (default: 0.3)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-csv", default=None,
                        help="Save per-object results to CSV")
    args = parser.parse_args()

    # ── Load model ──
    import torch
    from model.inference_v6 import load_model, predict_heatmap_batch

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, ckpt = load_model(args.checkpoint, device)
    print(f"Model: {args.checkpoint}")
    print(f"  epoch={ckpt.get('epoch', '?')}  device={device}")
    print(f"  pred_thresh={args.pred_thresh}  hp_thresh={args.hp_thresh}")

    # ── Load eval objects (seen only) ──
    seen_objects = []
    with open(EVAL_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            oid = row["obj_id"].strip()
            if not oid.startswith("unseen_"):
                seen_objects.append(oid)
    print(f"\nEval seen objects: {len(seen_objects)}")

    # ── Process each object ──
    results = []
    # Accumulators for micro-average
    total_tp = total_fp = total_fn = total_tn = 0

    for i, obj_id in enumerate(seen_objects):
        try:
            # 1. Load mesh, sample points
            mesh, sf = load_rotated_mesh_scaled(obj_id)
            pts, nrm = sample_surface(mesh, N_POINTS, SEED + i)

            # 2. Run prediction
            pts_batch = pts[np.newaxis]  # (1, N, 3)
            nrm_batch = nrm[np.newaxis]  # (1, N, 3)
            pred = predict_heatmap_batch(model, pts_batch, nrm_batch, device)[0]

            # 3. Load aligned HP
            hp = load_hp_aligned(obj_id, sf, pts)

            # 4. Binarize
            pred_bin = pred > args.pred_thresh
            hp_bin = hp > args.hp_thresh

            # 5. Compute metrics
            m = compute_metrics(pred_bin, hp_bin)
            m["obj_id"] = obj_id
            m["pred_max"] = float(pred.max())
            m["pred_mean"] = float(pred.mean())
            m["hp_max"] = float(hp.max())
            m["hp_pos_ratio"] = float(hp_bin.mean())
            m["pred_pos_ratio"] = float(pred_bin.mean())
            results.append(m)

            total_tp += m["tp"]
            total_fp += m["fp"]
            total_fn += m["fn"]
            total_tn += m["tn"]

            status = "✅" if m["f1"] > 0.3 else "⚠️" if m["f1"] > 0 else "❌"
            print(
                f"  [{i+1:3d}/{len(seen_objects)}] {obj_id:12s}  "
                f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
                f"IoU={m['iou']:.3f}  "
                f"hp+={m['hp_pos_ratio']*100:.1f}%  pred+={m['pred_pos_ratio']*100:.1f}%  "
                f"{status}"
            )

        except Exception as e:
            print(f"  [{i+1:3d}/{len(seen_objects)}] {obj_id:12s}  ❌ ERROR: {e}")
            results.append({"obj_id": obj_id, "f1": float("nan"), "error": str(e)})

    # ── Aggregate ──
    valid = [r for r in results if not np.isnan(r.get("f1", float("nan")))]
    f1s = [r["f1"] for r in valid]
    ps = [r["precision"] for r in valid]
    rs = [r["recall"] for r in valid]
    ious = [r["iou"] for r in valid]

    # Micro-average
    micro_p = total_tp / (total_tp + total_fp + 1e-8)
    micro_r = total_tp / (total_tp + total_fn + 1e-8)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-8)

    print(f"\n{'='*65}")
    print(f"  Results: {len(valid)}/{len(seen_objects)} objects computed")
    print(f"  Thresholds: pred={args.pred_thresh}  hp={args.hp_thresh}")
    print(f"{'─'*65}")
    print(f"  Macro-avg  F1={np.mean(f1s):.4f}  P={np.mean(ps):.4f}  R={np.mean(rs):.4f}  IoU={np.mean(ious):.4f}")
    print(f"  Micro-avg  F1={micro_f1:.4f}  P={micro_p:.4f}  R={micro_r:.4f}")
    print(f"  F1 std={np.std(f1s):.4f}  median={np.median(f1s):.4f}")
    print(f"  F1>0.5: {sum(1 for f in f1s if f>0.5)}/{len(f1s)}")
    print(f"  F1>0.3: {sum(1 for f in f1s if f>0.3)}/{len(f1s)}")
    print(f"  F1=0:   {sum(1 for f in f1s if f<0.01)}/{len(f1s)}")
    print(f"{'='*65}")

    # ── Save CSV ──
    if args.save_csv:
        import csv as csv_mod
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        fields = ["obj_id", "f1", "precision", "recall", "iou",
                  "tp", "fp", "fn", "tn",
                  "pred_max", "pred_mean", "hp_max",
                  "hp_pos_ratio", "pred_pos_ratio"]
        with open(args.save_csv, "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in sorted(valid, key=lambda x: -x["f1"]):
                w.writerow(r)
        print(f"\n  CSV saved: {args.save_csv}")


if __name__ == "__main__":
    main()
