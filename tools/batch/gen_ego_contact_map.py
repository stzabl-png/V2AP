#!/usr/bin/env python3
"""
gen_ego_contact_map.py — 第一人称视频接触图生成

从 HaWoR MANO 参数 + FoundationPose 物体位姿，计算物体表面接触频率，
生成与 export_arctic_prior.py / gen_m5_training_data.py 格式完全兼容的
human_prior HDF5 文件，可直接用于 PointNet++ 训练/推理。

坐标系:
  HaWoR world frame ≡ MegaSAM world frame (cam c2w 定义)
  MANO verts      → world frame (直接来自 pred_trans/rot)
  object verts    → world frame = cam_c2w[i] @ ob_in_cam[i] @ V_mesh

输入:
  world_space_res.pth   → MANO 参数 (joblib): [trans, rot, pose, betas, valid]
  cam_c2w.npy           → MegaSAM 相机位姿 (N_mega, 4, 4)
  ob_in_cam/{}.txt      → FP 物体位姿 (N_mega 帧, 4×4)
  mesh.ply              → 物体 mesh

输出:
  data_hub/human_prior/{obj_name}.hdf5
    point_cloud  (N, 3) — 物体表面采样点 (物体坐标系, 参考帧)
    normals      (N, 3)
    human_prior  (N,)   — 接触频率 [0,1]
  output/affordance_ego/{dataset}/{seq_id}/
    vert_contact_count.npy   — 原始接触计数
    contact_vis.png          — 可视化

用法:
  # hawor env 有 smplx / joblib / trimesh / h5py / scipy
  conda run -n hawor python tools/gen_ego_contact_map.py

  conda run -n hawor python tools/gen_ego_contact_map.py \\
      --obj assemble_tile --frames 0 59

注意校准物体 mesh 的 scale:
  SAM3D 输出的 mesh 可能不是米尺度. 设置 SCALE_OVERRIDE 或
  提供 mesh 旁的 scale.json {"scale_factor": x}.
  orange 真实直径 ≈ 0.08m, SAM3D mesh diameter ≈ 1.28m → scale ≈ 0.063
"""

import os, sys, argparse, json
import numpy as np
import trimesh
import joblib
import h5py
from glob import glob
from natsort import natsorted
from scipy.spatial import cKDTree
from tqdm import tqdm

# ── 路径 ─────────────────────────────────────────────────────────────────────
PROJ     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config as _cfg
HAWOR    = _cfg.HAWOR_DIR
MANO_R   = os.path.join(HAWOR, "_DATA/data/mano/MANO_RIGHT.pkl")
MANO_L   = os.path.join(HAWOR, "_DATA/data_left/mano_left/MANO_LEFT.pkl")

EGO_DEPTH_BASE = os.path.join(PROJ, "data_hub", "ProcessedData", "egocentric_depth")
POSE_BASE      = _cfg.EGO_POSE_DIR   # poses/Egocentric/{ds}/{seq} (was legacy obj_poses_ego)
MESH_BASE      = os.path.join(PROJ, "data_hub", "ProcessedData", "obj_meshes", "egocentric")
RAW_BASE       = os.path.join(PROJ, "data_hub", "RawData", "ThirdPersonRawData")
HP_DIR         = os.path.join(PROJ, "data_hub", "human_prior")
OUT_VIS_BASE   = os.path.join(PROJ, "output", "affordance_ego")
os.makedirs(HP_DIR, exist_ok=True)

N_POINTS       = 4096
CONTACT_THR    = 0.03    # 3 cm — 手顶点到物体表面 ≤ 3cm 认为接触
CONTACT_SIGMA  = 0.015   # 15 mm 高斯扩散 sigma
MAX_MEGA_FRAMES = 60

# ── 已知物体尺寸 (真实直径, m) → 用于校正 SAM3D mesh scale ─────────────────
# SAM3D mesh 直径可能严重偏大; 设置为 None 则自动用 mesh 原始尺寸
SCALE_OVERRIDE = {
    # "obj_name": target_diameter_m
    "orange":        0.08,   # 橙子约 8cm 直径
    "assemble_tile": None,   # 积木用原始尺寸（FP 深度对齐相对准确）
}

# ── 序列配置 ─────────────────────────────────────────────────────────────────
REGISTRY_JSON = os.path.join(PROJ, "tools", "egodex_sequence_registry.json")

_BUILTIN_REGISTRY = {
    # (dataset, seq_id): {obj_name, hawor_seq_dir, depth_dir, pose_dir}
    ("egodex", "assemble_disassemble_tiles/1"): {
        "obj_name":    "assemble_tile",
        "hawor_dir":   os.path.join(RAW_BASE,
                           "egodex/test/assemble_disassemble_tiles/1"),
        "depth_dir":   os.path.join(EGO_DEPTH_BASE,
                           "egodex/assemble_disassemble_tiles/1"),
        "pose_dir":    os.path.join(POSE_BASE,
                           "egodex/assemble_disassemble_tiles__1"),
    },
}


def load_sequence_registry():
    """EGO-BUG-01 修复: 从 JSON + 内置条目动态加载。"""
    reg = dict(_BUILTIN_REGISTRY)
    if os.path.exists(REGISTRY_JSON):
        with open(REGISTRY_JSON) as f:
            jreg = json.load(f)
        for key, cfg in jreg.items():
            if cfg.get("skipped"):          # 手动标记为不适合的序列
                continue
            ds  = cfg["dataset"]
            sid = cfg["seq_id"]
            reg[(ds, sid)] = {
                "obj_name":  cfg["obj_name"],
                "hawor_dir": cfg.get("hawor_dir",
                                 os.path.join(RAW_BASE,
                                     f"egodex/test/{sid}")),
                "depth_dir": cfg["depth_dir"],
                "pose_dir":  cfg.get("pose_dir",
                                 os.path.join(POSE_BASE, ds,
                                              sid.replace("/", "__"))),
            }
        print(f"[Registry] {len(jreg)} from JSON + {len(_BUILTIN_REGISTRY)} builtins"
              f" = {len(reg)} total")
    return reg


SEQUENCE_REGISTRY = None  # lazy-loaded in main()


# ── MANO forward ─────────────────────────────────────────────────────────────
_mano_cache = {}

def get_mano(is_right: bool):
    side = "right" if is_right else "left"
    if side not in _mano_cache:
        import smplx, torch
        path = MANO_R if is_right else MANO_L
        model = smplx.create(
            model_path=path,
            model_type="mano",
            is_rhand=is_right,
            use_pca=False,
            flat_hand_mean=False,
        )
        model.eval()
        _mano_cache[side] = model
    return _mano_cache[side]


def mano_forward_batch(pred_trans, pred_rot, pred_hand_pose, pred_betas, is_right: bool):
    """
    Run MANO forward pass on a batch of frames.

    Args:
        pred_trans:     (B, 3) world-frame root translation
        pred_rot:       (B, 3) world-frame root rotation (axis-angle)
        pred_hand_pose: (B, 45) hand articulation (axis-angle)
        pred_betas:     (B, 10) shape

    Returns:
        vertices: (B, 778, 3) world-frame hand vertices
    """
    import torch
    mano = get_mano(is_right)
    B = pred_trans.shape[0]
    with torch.no_grad():
        out = mano(
            global_orient=torch.from_numpy(pred_rot).float(),
            hand_pose=torch.from_numpy(pred_hand_pose).float(),
            transl=torch.from_numpy(pred_trans).float(),
            betas=torch.from_numpy(pred_betas).float(),
        )
    return out.vertices.numpy()   # (B, 778, 3)


# ── 接触计算 ─────────────────────────────────────────────────────────────────
def compute_contact_counts(mesh_verts_ref, ob_in_world_list,
                           mano_verts_list, valid_list,
                           contact_thr=CONTACT_THR):
    """
    Accumulate per-mesh-vertex contact counts across frames.

    Args:
        mesh_verts_ref:  (V, 3) object mesh vertices in a canonical (local) frame
        ob_in_world_list: list of (4,4) object-in-world transforms per frame
        mano_verts_list:  list of [left_verts(778,3), right_verts(778,3)]
                          verts in world frame; None if hand not valid
        valid_list:       list of [bool_left, bool_right]
        contact_thr:      float, threshold in meters

    Returns:
        counts: (V,) float32 — accumulated contact frequency per vertex
    """
    V = len(mesh_verts_ref)
    counts = np.zeros(V, dtype=np.float32)
    n_frames = len(ob_in_world_list)

    for fi in tqdm(range(n_frames), desc="  frames", leave=False):
        ob_in_world = ob_in_world_list[fi]   # (4, 4)
        # Transform canonical mesh verts → world
        V4 = np.hstack([mesh_verts_ref,
                        np.ones((V, 1), dtype=np.float32)])   # (V, 4)
        mesh_world = (ob_in_world @ V4.T).T[:, :3]           # (V, 3)

        tree = cKDTree(mesh_world)

        for hand_idx in range(2):         # 0=left, 1=right
            if not valid_list[fi][hand_idx]:
                continue
            hv = mano_verts_list[fi][hand_idx]     # (778, 3) world
            if hv is None:
                continue
            dists, idx = tree.query(hv, k=1)
            contact_mask = dists < contact_thr
            counts[idx[contact_mask]] += 1.0

    return counts


def diffuse_contact(mesh_verts, counts, sigma=CONTACT_SIGMA, radius=0.05):
    """
    Gaussian spatial diffusion of contact counts to smooth the map.

    Returns:
        diffused: (V,) float32 in [0, 1]
    """
    diffused = np.zeros(len(mesh_verts), dtype=np.float32)
    contact_mask  = counts > 0
    if not contact_mask.any():
        return diffused

    ctact_verts  = mesh_verts[contact_mask]
    ctact_scores = counts[contact_mask]

    tree_v = cKDTree(mesh_verts)
    for cv, cs in zip(ctact_verts, ctact_scores):
        idxs  = tree_v.query_ball_point(cv, radius)
        dists = np.linalg.norm(mesh_verts[idxs] - cv, axis=1)
        diffused[idxs] += cs * np.exp(-0.5 * (dists / sigma) ** 2)

    diffused /= (diffused.max() + 1e-8)
    return diffused


# ── 主处理函数 ────────────────────────────────────────────────────────────────
def process_sequence(dataset, seq_id, cfg, frame_start=0, frame_end=None):
    obj_name  = cfg["obj_name"]
    hawor_dir = cfg["hawor_dir"]
    depth_dir = cfg["depth_dir"]
    pose_dir  = cfg["pose_dir"]

    print(f"\n{'─'*58}")
    print(f"▶ [{dataset}] {seq_id}  obj={obj_name}")

    # ── 1. 加载 MANO 参数 ────────────────────────────────────────────────────
    mano_path = os.path.join(hawor_dir, "world_space_res.pth")
    if not os.path.exists(mano_path):
        print(f"  ❌ world_space_res.pth not found: {mano_path}"); return None
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(mano_path)
    # shapes: (2, N_total, 3/3/45/10) + (2, N_total) bool
    pred_trans = pred_trans.numpy() if hasattr(pred_trans, 'numpy') else np.array(pred_trans)
    pred_rot   = pred_rot.numpy()   if hasattr(pred_rot,   'numpy') else np.array(pred_rot)
    pred_hand_pose = pred_hand_pose.numpy() if hasattr(pred_hand_pose, 'numpy') else np.array(pred_hand_pose)
    pred_betas = pred_betas.numpy() if hasattr(pred_betas, 'numpy') else np.array(pred_betas)
    pred_valid = np.array(pred_valid, dtype=bool)
    N_total = pred_trans.shape[1]
    print(f"  MANO: N_total={N_total}  valid_R={pred_valid[1].sum()}  valid_L={pred_valid[0].sum()}")

    # ── 2. 加载 MegaSAM 相机位姿 ─────────────────────────────────────────────
    cam_c2w_path = os.path.join(depth_dir, "cam_c2w.npy")
    if not os.path.exists(cam_c2w_path):
        print(f"  ❌ cam_c2w.npy not found"); return None
    cam_c2w = np.load(cam_c2w_path).astype(np.float64)  # (N_mega, 4, 4)
    N_mega = cam_c2w.shape[0]
    step   = max(1, N_total // MAX_MEGA_FRAMES)
    print(f"  cam_c2w: {N_mega} frames  step={step}")

    # ── 3. 加载 FP 物体位姿 ─────────────────────────────────────────────────
    ob_in_cam_files = natsorted(glob(os.path.join(pose_dir, "ob_in_cam", "*.txt")))
    if not ob_in_cam_files:
        print(f"  ❌ ob_in_cam/*.txt not found in {pose_dir}"); return None
    ob_in_cam_list = [np.loadtxt(f).astype(np.float64) for f in ob_in_cam_files]
    N_fp = len(ob_in_cam_list)
    print(f"  FP poses: {N_fp} frames")
    if N_fp != N_mega:
        N_use = min(N_fp, N_mega)
        print(f"  ⚠️  FP帧数 {N_fp} ≠ MegaSAM帧数 {N_mega}，截断到 {N_use}（EGO-BUG-02修复）")
        ob_in_cam_list = ob_in_cam_list[:N_use]
        cam_c2w = cam_c2w[:N_use]
        N_mega  = N_use
        N_fp    = N_use
    else:
        N_use = N_fp

    # ── 4. 加载 & 采样物体 mesh ─────────────────────────────────────────────
    mesh_path = os.path.join(MESH_BASE, obj_name, "mesh.ply")
    if not os.path.exists(mesh_path):
        print(f"  ❌ mesh not found: {mesh_path}"); return None
    mesh = trimesh.load(mesh_path, force="mesh")

    # Scale 校正
    scale_json = os.path.join(os.path.dirname(mesh_path), "scale.json")
    scale_f = 1.0
    if os.path.exists(scale_json):
        scale_f = float(json.load(open(scale_json)).get("scale_factor", 1.0))
    elif SCALE_OVERRIDE.get(obj_name) is not None:
        target_diam = SCALE_OVERRIDE[obj_name]
        mesh_diam   = mesh.bounding_sphere.primitive.radius * 2
        scale_f     = target_diam / mesh_diam
    mesh_verts_local = (mesh.vertices * scale_f).astype(np.float32)
    mesh_diam_final  = trimesh.Trimesh(vertices=mesh_verts_local,
                                       faces=mesh.faces, process=False
                                       ).bounding_sphere.primitive.radius * 2
    print(f"  Mesh: {len(mesh_verts_local)} verts  scale_f={scale_f:.4f}  diam={mesh_diam_final:.3f}m")

    # ── 5. 收集所有有效 MANO 帧 (非 NaN, 不限于 step 倍数) ─────────────────────
    # 逐帧检查: valid 标志 AND 所有参数有限
    valid_frames = {0: [], 1: []}  # hand_idx → list of orig frame indices
    for oi in range(N_total):
        for h in range(2):
            if not pred_valid[h, oi]:
                continue
            params = [pred_trans[h, oi], pred_rot[h, oi],
                      pred_hand_pose[h, oi], pred_betas[h, oi]]
            if all(np.isfinite(p).all() for p in params):
                valid_frames[h].append(oi)

    f_start = int(frame_start)
    f_end   = int(frame_end) if frame_end is not None else N_mega - 1
    f_end   = min(f_end, N_mega - 1)
    # Clamp valid MANO frames to the time window covered by FP (f_start..f_end)
    # MegaSAM frame i covers orig frames [i*step, (i+1)*step)
    orig_start = f_start * step
    orig_end   = min(f_end * step + step - 1, N_total - 1)
    valid_frames = {h: [oi for oi in idxs if orig_start <= oi <= orig_end]
                    for h, idxs in valid_frames.items()}
    total_valid = len(valid_frames[0]) + len(valid_frames[1])
    print(f"  Processing: orig [{orig_start},{orig_end}]  "
          f"valid_L={len(valid_frames[0])}  valid_R={len(valid_frames[1])}  total={total_valid}")

    # ── 6. MANO forward: 批量处理所有有效帧 ─────────────────────────────────
    print("  Running MANO forward...")
    mano_world = {}   # (hand_idx, orig_fi) → (778, 3) world-frame verts
    for hand_idx, is_right, label in [(0, False, "LEFT"), (1, True, "RIGHT")]:
        oi_list = valid_frames[hand_idx]
        if not oi_list:
            print(f"    {label}: 0 valid frames")
            continue
        verts = mano_forward_batch(
            pred_trans    [hand_idx, oi_list],
            pred_rot      [hand_idx, oi_list],
            pred_hand_pose[hand_idx, oi_list],
            pred_betas    [hand_idx, oi_list],
            is_right=is_right
        )  # (B, 778, 3)
        nan_mask = np.isnan(verts).any(axis=(1, 2))
        good = 0
        for j, oi in enumerate(oi_list):
            if not nan_mask[j]:
                mano_world[(hand_idx, oi)] = verts[j]
                good += 1
        vrange = [float(np.nanmin(verts)), float(np.nanmax(verts))]
        print(f"    {label}: {good}/{len(oi_list)} valid frames  "
              f"range [{vrange[0]:.3f},{vrange[1]:.3f}]m")

    if not mano_world:
        print("  ❌ No valid MANO frames at all — skipping")
        return None

    # ── 7. 接触计数: 对每个有效 MANO 帧, 找最近 cam_c2w/ob_in_cam 帧 ────────
    # 这样 ph2d 这类极少有效帧的序列也能充分利用所有手部数据
    print("  Computing contact counts...")
    contact_thr = CONTACT_THR
    V  = len(mesh_verts_local)
    counts = np.zeros(V, dtype=np.float32)
    V4 = np.hstack([mesh_verts_local,
                    np.ones((V, 1), dtype=np.float32)])  # (V, 4)

    for (hand_idx, oi), hv in tqdm(mano_world.items(), desc="  contact", leave=False):
        # 最近的 MegaSAM/FP 帧
        cam_fi = int(round(oi / max(step, 1)))
        cam_fi = max(f_start, min(cam_fi, f_end))
        ob_in_world = (cam_c2w[cam_fi] @ ob_in_cam_list[cam_fi]).astype(np.float64)
        mesh_world  = (ob_in_world @ V4.T).T[:, :3].astype(np.float32)

        tree = cKDTree(mesh_world)
        dists, idx = tree.query(hv, k=1)
        contact_mask = dists < CONTACT_THR
        counts[idx[contact_mask]] += 1.0

    n_contact_verts = (counts > 0).sum()
    print(f"  Contact verts: {n_contact_verts}/{len(mesh_verts_local)} = {n_contact_verts/len(mesh_verts_local)*100:.1f}%")

    diffused = diffuse_contact(mesh_verts_local, counts)
    coverage = float((diffused > 0.1).mean())
    print(f"  Coverage (>0.1): {coverage:.1%}")

    # ── 8. 采样 N_POINTS 个表面点 ────────────────────────────────────────────
    mesh_scaled = trimesh.Trimesh(vertices=mesh_verts_local,
                                  faces=mesh.faces, process=False)
    pts, face_idx = trimesh.sample.sample_surface(mesh_scaled, N_POINTS)
    pts       = pts.astype(np.float32)
    normals   = mesh_scaled.face_normals[face_idx].astype(np.float32)

    # KNN: diffused (per-vertex) → sampled points
    tree_v = cKDTree(mesh_verts_local)
    _, knn  = tree_v.query(pts, k=1)
    hp      = diffused[knn].astype(np.float32)

    # Force center (EGO-BUG-04修复: 与第三人称 batch_align_mano_fp.py 格式一致)
    high_contact = hp > 0.1
    if high_contact.any():
        force_center = pts[high_contact].mean(0).astype(np.float32)
    else:
        any_contact = hp > 0
        force_center = (pts[any_contact].mean(0).astype(np.float32)
                        if any_contact.any() else np.zeros(3, np.float32))

    # ── 9. 保存 HDF5 (与 export_arctic_prior 格式完全一致) ─────────────────
    out_path = os.path.join(HP_DIR, f"{obj_name}.hdf5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("point_cloud",  data=pts,          compression="gzip")
        f.create_dataset("normals",      data=normals,      compression="gzip")
        f.create_dataset("human_prior",  data=hp,           compression="gzip")
        f.create_dataset("force_center", data=force_center, compression="gzip")  # BUG-04修复
        f.attrs["source"]   = "ego_hawor_fp"
        f.attrs["dataset"]  = dataset
        f.attrs["seq_id"]   = seq_id
        f.attrs["obj_name"] = obj_name
        f.attrs["n_valid_mano"] = total_valid
        f.attrs["coverage"] = coverage
    print(f"  ✅ {obj_name}: coverage={coverage:.1%}  max_hp={hp.max():.3f}  force_center={force_center.round(3)}  → {out_path}")

    # ── 10. 保存原始 contact count ──────────────────────────────────────────
    vis_dir = os.path.join(OUT_VIS_BASE, dataset, seq_id.replace("/", "__"))
    os.makedirs(vis_dir, exist_ok=True)
    np.save(os.path.join(vis_dir, "vert_contact_count.npy"), counts)
    np.save(os.path.join(vis_dir, "mesh_verts_local.npy"), mesh_verts_local)

    return hp, coverage


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global SEQUENCE_REGISTRY
    SEQUENCE_REGISTRY = load_sequence_registry()   # EGO-BUG-01 修复
    all_objs = sorted(set(v["obj_name"] for v in SEQUENCE_REGISTRY.values()))

    parser = argparse.ArgumentParser(
        description="第一人称接触图生成 — MANO + FP → human_prior HDF5")
    parser.add_argument("--obj", default=None,
                        choices=all_objs,
                        help="只处理此物体（默认全部）")
    parser.add_argument("--frames", nargs=2, type=int, default=[0, 59],
                        metavar=("START", "END"),
                        help="FP 帧范围 (default: 0 59)")
    parser.add_argument("--contact-thr", type=float, default=CONTACT_THR,
                        help=f"接触阈值 m (default {CONTACT_THR})")
    args = parser.parse_args()

    print("=" * 60)
    print(" Egocentric Contact Map Generation")
    print(f" MANO  → smplx (hawor env)")
    print(f" output→ {HP_DIR}/{{obj}}.hdf5")
    print("=" * 60)

    results = {}
    for (ds, sid), cfg in SEQUENCE_REGISTRY.items():
        if args.obj and cfg["obj_name"] != args.obj:
            continue
        hp, cov = process_sequence(ds, sid, cfg,
                                   frame_start=args.frames[0],
                                   frame_end=args.frames[1]) or (None, None)
        if hp is not None:
            results[cfg["obj_name"]] = {"coverage": cov, "max_hp": float(hp.max())}

    print(f"\n{'='*60}")
    for k, v in results.items():
        print(f"  {k:20s}  coverage={v['coverage']:.1%}  max_hp={v['max_hp']:.3f}")
    print()
    print("下一步 — 喂给 PointNet++ 模型:")
    print("  python tools/vis_m5_predict.py --obj assemble_tile  # 需提供 mesh 在 meshes/v1/")
    print("  或直接用 affordance 推理:")
    print("  python inference/predictor.py")


if __name__ == "__main__":
    main()
