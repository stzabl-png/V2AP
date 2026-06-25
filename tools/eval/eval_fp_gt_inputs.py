#!/usr/bin/env python3
"""
eval_fp_gt_inputs.py
====================
Evaluate FoundationPose object-pose quality on HOI4D under different inputs,
all compared against HOI4D objpose GT on *identical* frames.

Frame alignment (verified): extracted_images[2j] == mp4[j] == GT objpose[j].
FP runs on a 60-frame subset → GT frame index = i*5  (i = 0..59).

Regimes:
  A  MegaSAM K + MegaSAM depth + SAM3D mesh   (already in poses/Egocentric/hoi4d/)
  B  GT K      + GT depth      + SAM3D mesh   (this script, --regime B → hoi4d_gtB)
  C  GT K      + GT depth      + GT CAD       (this script, --regime C → hoi4d_gtC; rigid only)

Run (env biv2ap, from PROJ root):
  conda run -n biv2ap python tools/eval/eval_fp_gt_inputs.py --regime B
  conda run -n biv2ap python tools/eval/eval_fp_gt_inputs.py --compare
"""
import os, sys, json, argparse, time
import numpy as np
import cv2
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]          # V2AP/
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "tools" / "batch"))

import config as _cfg

# ── GT data roots ─────────────────────────────────────────────────────────────
GT_HOI   = PROJ / "data/egocentric/hoi4d"
GT_ANNO  = GT_HOI / "HOI4D_annotations"
GT_DEPTH = GT_HOI / "HOI4D_depth_video"
GT_CAD   = GT_HOI / "HOI4D_CAD_Model_for_release"
RAW_HOI  = PROJ / "data_hub/RawData/EgoRawData/hoi4d/HOI4D_release"

MESH_BASE = PROJ / "data_hub/ProcessedData/obj_meshes/hoi4d"
OBJ_INPUT = PROJ / "data_hub/ProcessedData/obj_recon_input/egocentric"
POSE_BASE = Path(_cfg.EGO_POSE_DIR)                  # poses/Egocentric
SEQS_JSON = PROJ / "output/eval/hoi4d_10seqs.json"

# HOI4D single global egocentric intrinsic (verified std=0 over 2317 files) @1920x1080
K_GT_FULL = np.array([[1376.8, 0.0, 967.7],
                      [0.0, 1376.2, 526.8],
                      [0.0, 0.0,   1.0]], dtype=np.float64)
FULL_W, FULL_H = 1920, 1080
SHORTER_SIDE = 480
N_SUB = 60
STRIDE = 5                                           # GT frame = i*STRIDE


def seq_to_relpath(seq_id):
    p = seq_id.split("_")
    return "/".join(p[:7])                            # ZY/H/C/N/S/s/T


def scaled_K(target_w, target_h):
    sx, sy = target_w / FULL_W, target_h / FULL_H
    K = K_GT_FULL.copy()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K


def decode_depth_frame(cap, idx):
    """HOI4D align_depth avi: d_mm = ch0*256 + ch1 → meters."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    if not ret:
        return None
    d_mm = f[:, :, 0].astype(np.uint16) * 256 + f[:, :, 1].astype(np.uint16)
    return d_mm.astype(np.float32) / 1000.0


def build_gt_scene(seq_id, scene_dir):
    """Build FP scene_dir (rgb/ depth/ masks/ cam_K.txt) from GT depth+K on
    GT-aligned frames [0,5,...,295]. Returns (n, H, W) or None."""
    rel = seq_to_relpath(seq_id)
    mp4  = RAW_HOI / rel / "align_rgb/image.mp4"
    davi = GT_DEPTH / rel / "align_depth/depth_video.avi"
    mask_path = OBJ_INPUT / seq_id / "0.png"
    for p in (mp4, davi, mask_path):
        if not p.exists():
            print(f"  ⚠ missing {p}")
            return None

    scale = SHORTER_SIDE / min(FULL_H, FULL_W)
    H, W = int(round(FULL_H * scale)), int(round(FULL_W * scale))
    K = scaled_K(W, H)

    for sub in ("rgb", "depth", "masks"):
        (Path(scene_dir) / sub).mkdir(parents=True, exist_ok=True)

    cap_rgb = cv2.VideoCapture(str(mp4))
    cap_d   = cv2.VideoCapture(str(davi))
    n = 0
    for i in range(N_SUB):
        gidx = i * STRIDE
        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, gidx)
        ret, bgr = cap_rgb.read()
        if not ret:
            break
        d = decode_depth_frame(cap_d, gidx)
        if d is None:
            break
        oid = f"{i:06d}"
        rgb_r = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(scene_dir, "rgb", f"{oid}.png"), rgb_r)
        d_r = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(scene_dir, "depth", f"{oid}.png"),
                    (d_r * 1000).clip(0, 65535).astype(np.uint16))
        n += 1
    cap_rgb.release(); cap_d.release()
    if n == 0:
        return None

    mask = cv2.imread(str(mask_path), 0)
    mask_r = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(scene_dir, "masks", "000000.png"), mask_r)
    np.savetxt(os.path.join(scene_dir, "cam_K.txt"), K, fmt="%.6f")
    cov = (mask_r > 0).mean() * 100
    print(f"  scene: {n} frames @ {H}×{W}  mask={cov:.1f}%")
    return n, H, W


def run_regime(regime):
    from batch_obj_pose_ego import init_fp_models, run_fp   # reuse canonical FP
    seqs = json.load(open(SEQS_JSON))
    out_tag = {"B": "hoi4d_gtB", "C": "hoi4d_gtC"}[regime]
    scene_root = Path(f"/tmp/fp_scenes_gt/{out_tag}")

    print("Init FoundationPose models…")
    scorer, refiner, glctx = init_fp_models()
    print("✅ ready\n")

    for s in seqs:
        seq_id = s["seq_id"].split("/")[-1]
        meta = s.get("gt", {})
        if regime == "C" and not meta.get("rigid", False):
            print(f"⏭  {seq_id}: articulated, skip regime C")
            continue
        mesh_path = (MESH_BASE / seq_id / "mesh.ply" if regime == "B"
                     else Path(meta["cad_path"]))
        if not Path(mesh_path).exists():
            print(f"⚠  mesh missing {mesh_path}")
            continue
        out_dir = POSE_BASE / out_tag / seq_id
        if (out_dir / "ob_in_cam").is_dir() and list((out_dir / "ob_in_cam").glob("*.txt")):
            print(f"⏭  {seq_id}: cached")
            continue
        scene_dir = scene_root / seq_id
        print(f"\n▶ [{regime}] {seq_id}  mesh={mesh_path}")
        t0 = time.time()
        r = build_gt_scene(seq_id, str(scene_dir))
        if r is None:
            print("  ⚠ scene build failed"); continue
        run_fp(str(mesh_path), str(scene_dir), str(out_dir),
               scorer=scorer, refiner=refiner, glctx=glctx)
        print(f"  ✅ {time.time()-t0:.1f}s → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["B", "C"])
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.regime:
        run_regime(args.regime)
    if args.compare:
        import compare_fp_gt  # noqa
