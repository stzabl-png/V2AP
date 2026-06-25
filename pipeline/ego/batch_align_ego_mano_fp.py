#!/usr/bin/env python3
"""
batch_align_ego_mano_fp.py — 第一人称 MANO × FoundationPose 对齐 → 接触标签 → HDF5

核心差异（vs 第三人称）:
  第三人称: MANO 已在相机坐标系，直接和 FP 的 ob_in_cam 对比
  第一人称: MANO 在世界坐标系，需先转换到相机坐标系:
              v_cam = R_w2c[t] @ v_world + t_w2c[t]
           再转换到物体坐标系:
              v_obj = inv(ob_in_cam[t]) @ v_cam  (homogeneous)

帧对齐:
  MANO:  T_rgb 帧（与提取的 RGB 帧对齐）
  FP:    N_fp 帧（MegaSAM 子采样，step = T_rgb // N_fp）
  FP 帧 i  →  MANO 帧 i * step

用法:
  conda activate bundlesdf
  python data/batch_align_ego_mano_fp.py --dataset egodex
  python data/batch_align_ego_mano_fp.py --obj slot_batteries --redo
"""

import os, sys, json, argparse
import numpy as np
import h5py
import trimesh
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# ── 路径 ──────────────────────────────────────────────────────────────────────
EGO_MANO_BASE  = os.path.join(config.DATA_HUB, "ProcessedData", "ego_mano")
FP_POSE_BASE   = config.EGO_POSE_DIR   # poses/Egocentric/{ds}/{seq} (was legacy obj_poses_ego)
MESH_BASE      = os.path.join(config.DATA_HUB, "ProcessedData", "obj_meshes", "egocentric")
DEPTH_BASE     = os.path.join(config.DATA_HUB, "ProcessedData", "egocentric_depth")
OUT_TRAIN_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "training_fp_ego")
OUT_PRIOR_BASE = os.path.join(config.DATA_HUB, "human_prior")   # shared inference interface
REGISTRY_JSON       = os.path.join(config.PROJECT_DIR, "tools", "egodex_sequence_registry.json")
TACO_REGISTRY_JSON  = os.path.join(config.PROJECT_DIR, "tools", "taco_ego_sequence_registry.json")

N_POINTS      = 4096
CONTACT_SIGMA = 0.020   # Gaussian σ for contact smoothing (m)
CONTACT_THRESH = 0.050  # hard threshold for "contact" (m), used for force_center
MAX_FP_FRAMES  = 60


# ── Registry ─────────────────────────────────────────────────────────────────

def load_registry():
    """
    Build obj_name -> [(dataset, seq_id, depth_dir)] from:
      1. tools/egodex_sequence_registry.json  (EgoDex sequences)
      2. tools/taco_ego_sequence_registry.json (TACO Ego sequences, same format)
    Registry JSON format:
      { "key": { "dataset": "egodex"|"taco_ego",
                 "seq_id":  "task/episode",
                 "obj_name": "obj_name",
                 "skipped": false } }

    depth_dir is always derived from config.DATA_HUB + seq_id so that the
    registry works on any machine regardless of what paths are stored in the JSON.
    """
    obj_map = {}  # obj_name -> list[(ds, seq_id, depth_dir)]
    ego_depth_base = os.path.join(config.DATA_HUB, "ProcessedData", "egocentric_depth")

    for reg_path in [REGISTRY_JSON, TACO_REGISTRY_JSON]:
        if not os.path.exists(reg_path):
            continue
        with open(reg_path) as f:
            jreg = json.load(f)
        for key, cfg in jreg.items():
            if cfg.get("skipped"):
                continue
            if "obj_name" not in cfg:   # skip _README / placeholder entries
                continue
            obj  = cfg["obj_name"]
            ds   = cfg["dataset"]
            sid  = cfg["seq_id"]   # e.g. "add_remove_lid/22"

            # Always derive depth_dir from config — ignore hardcoded path in JSON
            dd = os.path.join(ego_depth_base, ds, sid)

            obj_map.setdefault(obj, []).append((ds, sid, dd))
    if not obj_map:
        raise FileNotFoundError(
            f"No registry found. Expected: {REGISTRY_JSON} or {TACO_REGISTRY_JSON}")
    return obj_map


# ── Mesh utilities ────────────────────────────────────────────────────────────

def load_mesh_and_sample(obj_name, n_points=N_POINTS):
    """Load mesh, apply scale.json, sample surface points + normals."""
    mesh_path  = os.path.join(MESH_BASE, obj_name, "mesh.ply")
    scale_json = os.path.join(MESH_BASE, obj_name, "scale.json")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if os.path.exists(scale_json):
        sf = float(json.load(open(scale_json)).get("scale_factor", 1.0))
        if abs(sf - 1.0) > 0.01:
            mesh.vertices = mesh.vertices * sf
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    nrm = mesh.face_normals[face_idx]
    return pts.astype(np.float32), nrm.astype(np.float32)


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_mano_ego(dataset, seq_id):
    """
    Load HaWoR ego_mano npz.
    Returns dict with right_verts, left_verts (T,778,3) in WORLD space,
    R_w2c (T,3,3), t_w2c (T,3), pred_valid (2,T).
    """
    task, ep = seq_id.rsplit("/", 1)
    npz_path = os.path.join(EGO_MANO_BASE, dataset, task, f"{ep}.npz")
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path, allow_pickle=True)
    return {
        "right_verts": d["right_verts"].astype(np.float32),  # (T, 778, 3)
        "left_verts":  d["left_verts"].astype(np.float32),
        "R_w2c":       d["R_w2c"].astype(np.float32),        # (T, 3, 3)
        "t_w2c":       d["t_w2c"].astype(np.float32),        # (T, 3)
        "pred_valid":  d["pred_valid"].astype(bool),          # (2, T)  [0=L,1=R]
    }


def load_fp_poses(dataset, seq_id):
    """Load ob_in_cam/*.txt → (N, 4, 4)."""
    pose_dir = os.path.join(FP_POSE_BASE, dataset,
                            seq_id.replace("/", "__"), "ob_in_cam")
    files = natsorted(glob(os.path.join(pose_dir, "*.txt")))
    if not files:
        return None
    return np.stack([np.loadtxt(f).reshape(4, 4) for f in files]).astype(np.float32)


def align_sequence_2d(mano, fp_poses, K_depth, mesh_pts, mesh_nrm,
                      contact_sigma_px=15.0):
    """
    2D image-space contact accumulation (scale-invariant).

    Why 2D works for ego (same as third-person script):
      - MANO world → camera: v_cam = R_w2c @ v_world + t_w2c (SLAM scale)
      - Project to image:     u = fx*(x/z) + cx  → scale S cancels in x/z
      - FP mesh → camera:    mesh_cam = R_oc @ mesh_pts + t_oc (metric)
      - Both projected with the same K → correct image coords regardless of scale
      - Contact = Gaussian weight on 2D pixel distance

    Args:
        K_depth: (3,3) camera intrinsics at MegaSAM depth resolution
        contact_sigma_px: Gaussian σ in pixels (fixed, default 15px)

    Returns:
        contact_accum (N,): per-point Gaussian-weighted contact sum
        frame_count   (int)
    """
    T_rgb = mano["right_verts"].shape[0]
    N_fp  = fp_poses.shape[0]
    step  = max(1, T_rgb // N_fp)

    fx, fy = K_depth[0, 0], K_depth[1, 1]
    cx, cy = K_depth[0, 2], K_depth[1, 2]

    N_pts = len(mesh_pts)
    contact_accum = np.zeros(N_pts, dtype=np.float64)
    frame_count   = 0

    for i in range(N_fp):
        mano_idx = min(i * step, T_rgb - 1)
        valid_r  = bool(mano["pred_valid"][1, mano_idx])
        valid_l  = bool(mano["pred_valid"][0, mano_idx])
        if not valid_r and not valid_l:
            continue

        T_oc = fp_poses[i]
        if np.all(T_oc == 0) or np.isnan(T_oc).any():
            continue

        # Sanity: object must be in front of camera and at reasonable distance
        z_obj = float(T_oc[2, 3])
        if z_obj < 0.05 or z_obj > 5.0:
            continue

        R_oc = T_oc[:3, :3]
        t_oc = T_oc[:3,  3]

        R_w2c = mano["R_w2c"][mano_idx]   # (3,3) SLAM world→cam rotation
        t_w2c = mano["t_w2c"][mano_idx]   # (3,) SLAM world→cam (any scale)

        # ── Collect MANO image coords (scale-invariant projection) ──────────
        hand_uv_list = []
        for valid, key in [(valid_r, "right_verts"), (valid_l, "left_verts")]:
            if not valid:
                continue
            v_world = mano[key][mano_idx]              # (778, 3)
            v_cam   = (R_w2c @ v_world.T).T + t_w2c   # (778, 3) any scale OK
            z_hand  = v_cam[:, 2]
            front   = z_hand > 1e-4
            if front.sum() < 10:
                continue
            v_f = v_cam[front]
            u   = fx * v_f[:, 0] / v_f[:, 2] + cx   # (M,)
            v   = fy * v_f[:, 1] / v_f[:, 2] + cy   # (M,)
            hand_uv_list.append(np.stack([u, v], axis=1))   # (M, 2)

        if not hand_uv_list:
            continue
        hand_uv = np.concatenate(hand_uv_list, axis=0)   # (M, 2)

        # Sanity: MANO hand must be in front of camera
        z_mano = float(np.median(v_cam[:, 2]))
        if z_mano <= 0:
            continue

        # ── Project mesh via FP ob_in_cam → image coords ──────────────────
        mesh_cam = (R_oc @ mesh_pts.T).T + t_oc   # (N, 3) metric
        z_mesh   = mesh_cam[:, 2].clip(1e-4)
        u_mesh   = fx * mesh_cam[:, 0] / z_mesh + cx   # (N,)
        v_mesh   = fy * mesh_cam[:, 1] / z_mesh + cy   # (N,)
        mesh_uv  = np.stack([u_mesh, v_mesh], axis=1)  # (N, 2)

        # ── 2D pixel proximity → Gaussian weight ───────────────────────────
        hand_kd = cKDTree(hand_uv)
        dist_px, _ = hand_kd.query(mesh_uv, k=1, workers=1)   # (N,) pixels
        weight = np.exp(-0.5 * (dist_px / contact_sigma_px) ** 2)
        contact_accum += weight
        frame_count   += 1

    return contact_accum, frame_count

# Keep old function alias for reference
align_sequence = align_sequence_2d



# ── HDF5 saving ───────────────────────────────────────────────────────────────

def save_hdf5(out_path, pts, nrm, human_prior, force_center):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("point_cloud",  data=pts.astype(np.float32))
        f.create_dataset("normals",      data=nrm.astype(np.float32))
        f.create_dataset("human_prior",  data=human_prior.astype(np.float32))
        f.create_dataset("robot_gt",     data=np.zeros(len(pts), dtype=np.float32))
        f.create_dataset("force_center", data=force_center.astype(np.float32))
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="第一人称 MANO × FP 对齐 → 接触 HDF5")
    p.add_argument("--dataset",   default=None,
                   help="Filter by dataset: egodex | taco_ego")
    p.add_argument("--obj",       default=None, help="只处理此物体名子串")
    p.add_argument("--threshold", type=float, default=30.0,
                   help="接触距离阈值 mm (默认 30mm)")
    p.add_argument("--redo",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    contact_thresh_m = args.threshold / 1000.0

    obj_map = load_registry()   # obj_name → [(ds, seq_id, depth_dir)]

    if args.obj:
        obj_map = {k: v for k, v in obj_map.items() if args.obj in k}
    if args.dataset:
        obj_map = {k: [(ds, sid, dd) for ds, sid, dd in v if ds == args.dataset]
                   for k, v in obj_map.items()}
        obj_map = {k: v for k, v in obj_map.items() if v}

    print("=" * 60)
    print(f" Ego MANO × FP Alignment")
    print(f" Objects  : {len(obj_map)}")
    print(f" σ_contact: {CONTACT_SIGMA*100:.1f} cm  thresh: {args.threshold:.0f} mm")
    print("=" * 60)

    done = failed = skipped = 0

    for obj_name, seq_list in tqdm(obj_map.items(), desc="Objects"):
        tqdm.write(f"\n── {obj_name} ({len(seq_list)} seq) ──")

        out_train = os.path.join(OUT_TRAIN_BASE, "egodex", f"{obj_name}.hdf5")
        out_prior = os.path.join(OUT_PRIOR_BASE, f"{obj_name}.hdf5")

        if not args.redo and os.path.exists(out_train):
            tqdm.write(f"  ⏭  Cached: {out_train}")
            skipped += 1
            continue

        # Load mesh
        mesh_path = os.path.join(MESH_BASE, obj_name, "mesh.ply")
        if not os.path.exists(mesh_path):
            tqdm.write(f"  ❌ mesh.ply not found: {mesh_path}")
            failed += 1
            continue
        try:
            mesh_pts, mesh_nrm = load_mesh_and_sample(obj_name)
        except Exception as e:
            tqdm.write(f"  ❌ mesh load failed: {e}")
            failed += 1
            continue
        tqdm.write(f"  mesh pts: {len(mesh_pts)}")

        # Accumulate contact across sequences
        N_pts = len(mesh_pts)
        total_accum = np.zeros(N_pts, dtype=np.float64)
        total_frames = 0

        for ds, seq_id, depth_dir in seq_list:
            tqdm.write(f"    seq: {ds}/{seq_id}")

            mano = load_mano_ego(ds, seq_id)
            if mano is None:
                tqdm.write(f"    ⚠ MANO not found")
                continue

            fp_poses = load_fp_poses(ds, seq_id)
            if fp_poses is None or len(fp_poses) < 2:
                tqdm.write(f"    ⚠ FP poses not found or too few ({0 if fp_poses is None else len(fp_poses)})")
                continue

            tqdm.write(f"    MANO T={mano['right_verts'].shape[0]}  FP N={len(fp_poses)}")

            # Load K from MegaSAM depth dir
            K_path = os.path.join(depth_dir, "K.npy")
            if not os.path.exists(K_path):
                tqdm.write(f"    ⚠ K.npy not found in {depth_dir}")
                continue
            K_depth = np.load(K_path).astype(np.float32)
            tqdm.write(f"    K_depth fx={K_depth[0,0]:.1f}  σ=15px (fixed)")

            try:
                accum, n_frames = align_sequence_2d(
                    mano, fp_poses, K_depth, mesh_pts, mesh_nrm,
                    contact_sigma_px=15.0)
            except Exception as e:
                import traceback
                tqdm.write(f"    ❌ align error: {e}")
                traceback.print_exc()
                continue

            tqdm.write(f"    valid frames: {n_frames}  max_contact: {accum.max():.4f}")

            total_accum  += accum
            total_frames += n_frames

        if total_frames == 0:
            tqdm.write(f"  ❌ No valid frames for {obj_name}")
            failed += 1
            continue

        # Normalize to [0, 1]
        human_prior = total_accum / (total_frames + 1e-8)
        human_prior = (human_prior - human_prior.min()) / \
                      (human_prior.max() - human_prior.min() + 1e-8)
        human_prior = human_prior.astype(np.float32)

        # Force center: centroid of high-contact mesh points (obj space)
        high_contact = human_prior > 0.5
        if high_contact.sum() > 3:
            force_center = mesh_pts[high_contact].mean(axis=0)
        else:
            force_center = mesh_pts[np.argmax(human_prior)]

        tqdm.write(f"  contact coverage: {high_contact.sum()}/{N_pts} pts "
                   f"({high_contact.mean()*100:.1f}%)")
        tqdm.write(f"  force_center: {force_center}")

        # Save training HDF5
        save_hdf5(out_train, mesh_pts, mesh_nrm, human_prior, force_center)

        # Save human_prior_fp (inference interface, same format)
        save_hdf5(out_prior, mesh_pts, mesh_nrm, human_prior, force_center)

        done += 1

    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ⏭ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"Training HDF5 → {OUT_TRAIN_BASE}")
    print(f"Prior   HDF5  → {OUT_PRIOR_BASE}")


if __name__ == "__main__":
    main()
