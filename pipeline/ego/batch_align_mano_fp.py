#!/usr/bin/env python3
"""
batch_align_mano_fp.py — MANO × FoundationPose 对齐 → 接触标签 → 训练数据

原理
====
  每条序列有:
    FP 位姿  : obj_poses/{ds}/{seq}/ob_in_cam/XXXXXX.txt  (4×4, object→camera, metric m)
    MANO 顶点: third_mano/{ds}/{seq}.npz                 (verts_dict, camera space metric m)
    SAM3D mesh: obj_meshes/{ds}/{obj}/mesh.ply + scale.json

  对每帧 i (FP pose index):
    T = ob_in_cam[i]                      # 4×4 transform
    mesh_cam = T[:3,:3] @ M.T + T[:3,3]  # mesh vertices in camera space (metric)
    hand_verts = mano_verts[j]            # j = i × N_mano / N_fp  (proportional mapping)
    dist = min_dist(mesh_cam, hand_verts) # per-vertex distance
    contact[dist < threshold] += 1

  跨同一物体所有序列累积 → 归一化 → 点云采样 → HDF5

输出路径
========
  data_hub/ProcessedData/training_fp/{dataset}/{obj}.hdf5
    point_cloud  (N, 3)
    normals      (N, 3)
    human_prior  (N,)    — 接触概率
    robot_gt     (N,)    — 全零 (无机器人 GT)
    force_center (3,)    — 高接触区质心

  data_hub/ProcessedData/human_prior_fp/{obj}.hdf5
    (与 inference/predictor.py 相同接口)

用法
====
  conda activate bundlesdf          # trimesh / scipy 可用即可
  cd $PROJ

  # 验证: arctic/box
  python data/batch_align_mano_fp.py --dataset arctic --obj box

  # 全量
  python data/batch_align_mano_fp.py
  python data/batch_align_mano_fp.py --dataset oakink
  python data/batch_align_mano_fp.py --threshold 35   # 35mm 接触阈值
"""

import os, sys, re, json, argparse
import numpy as np
import h5py
import multiprocessing as mp
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# ── 路径 ──────────────────────────────────────────────────────────────────────
POSE_BASE      = os.path.join(config.DATA_HUB, "ProcessedData", "poses",          "ThirdPerson")
MANO_BASE      = os.path.join(config.DATA_HUB, "ProcessedData", "mano",           "ThirdPerson")
MESH_BASE      = os.path.join(config.DATA_HUB, "ProcessedData",  "obj_meshes")   # SAM3D meshes
OUT_TRAIN_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "human_prior_fp", "ThirdPerson")
OUT_PRIOR_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "human_prior_fp", "ThirdPerson")
DEPTH_BASE     = os.path.join(config.DATA_HUB, "ProcessedData", "depth",          "ThirdPerson")

N_POINTS       = 4096
CONTACT_SIGMA  = 0.020   # Gaussian smoothing radius (m)

# ── HO3D prefix → YCB object name ────────────────────────────────────────────
HO3D_OBJ = {
    'ABF': '003_cracker_box', 'BB': '011_banana',
    'GPMF': '010_potted_meat_can', 'GSF': '010_potted_meat_can',
    'MC': '003_cracker_box', 'MDF': '035_power_drill',
    'ND': '035_power_drill', 'SB': '021_bleach_cleanser',
    'ShSu': '004_sugar_box', 'SiBF': '003_cracker_box',
    'SiS': '052_extra_large_clamp', 'SM': '006_mustard_bottle',
    'SMu': '006_mustard_bottle', 'SS': '004_sugar_box',
}

# ── DexYCB: fixed per-subject object ordering ─────────────────────────────────
# Source: dex-ycb-toolkit / DexYCB paper. Each subject does 5 grasps × 20 objects
# in this fixed order, sessions sorted by timestamp.
DEXYCB_YCB_CLASSES = [
    '002_master_chef_can', '003_cracker_box', '004_sugar_box',
    '005_tomato_soup_can', '006_mustard_bottle', '007_tuna_fish_can',
    '008_pudding_box', '009_gelatin_box', '010_potted_meat_can',
    '011_banana', '019_pitcher_base', '021_bleach_cleanser',
    '024_bowl', '025_mug', '035_power_drill',
    '036_wood_block', '037_scissors', '040_large_marker',
    '051_large_clamp', '052_extra_large_clamp',
]
DEXYCB_NUM_OBJECTS  = len(DEXYCB_YCB_CLASSES)  # 20
DEXYCB_RAW_BASE     = os.path.join(config.DATA_HUB, "RawData",
                                   "ThirdPersonRawData", "dexycb")
_dexycb_session_cache: dict = {}   # subj → sorted session list

def _dexycb_sessions(subj: str):
    """Return sorted list of session timestamps for a DexYCB subject."""
    if subj not in _dexycb_session_cache:
        subj_dir = os.path.join(DEXYCB_RAW_BASE, subj)
        if os.path.isdir(subj_dir):
            _dexycb_session_cache[subj] = natsorted(
                [d for d in os.listdir(subj_dir)
                 if os.path.isdir(os.path.join(subj_dir, d))
                 and not d.startswith('.')]   # BUG-04: 过滤 .cache 等隐藏目录
            )
        else:
            _dexycb_session_cache[subj] = []
    return _dexycb_session_cache[subj]


def dexycb_seq_to_ycb(seq_id: str) -> str | None:
    """Map DexYCB seq_id (subj__session__serial) → ycb_dex_NN mesh name.

    ycb_dex_NN matches dex-ycb-toolkit index (1-based), which equals:
      ycb_dex_01 = 002_master_chef_can, ycb_dex_02 = 003_cracker_box, ...
    scale.json inside each ycb_dex_NN dir contains the actual YCB name.
    """
    parts = seq_id.split('__')
    if len(parts) != 3:
        return None
    subj, session, _ = parts
    sessions = _dexycb_sessions(subj)
    if not sessions or session not in sessions:
        return None
    idx = sessions.index(session)
    grasps_per_obj = max(1, len(sessions) // DEXYCB_NUM_OBJECTS)  # typically 5
    obj_idx = min(idx // grasps_per_obj, DEXYCB_NUM_OBJECTS - 1)  # 0-based
    return f'ycb_dex_{obj_idx + 1:02d}'  # 1-based: ycb_dex_01..ycb_dex_20



# ── seq → (mesh_dataset, obj_name) 映射 ──────────────────────────────────────
def seq_to_obj(dataset, seq_id):
    if dataset == "arctic":
        m = re.match(r'^[^_]+__(.+?)_(grab|use)_', seq_id)
        if m:
            return "arctic", m.group(1)
    elif dataset == "oakink":
        return "oakink", seq_id.split("_")[0]
    elif dataset == "ho3d_v3":
        sid = seq_id.split("__", 1)[-1]   # strip "train__" / "evaluation__"
        prefix = re.sub(r'\d+$', '', sid)
        ycb = HO3D_OBJ.get(prefix)
        if ycb:
            return "ycb", ycb
    elif dataset == "dexycb":
        ycb_name = dexycb_seq_to_ycb(seq_id)
        if ycb_name:
            return "ycb", ycb_name
    return None, None


# ── Mesh 加载 + scale 校正 ────────────────────────────────────────────────────
def load_mesh_scaled(mesh_path):
    """Load SAM3D mesh and apply scale.json correction.
    Returns trimesh.Trimesh with vertices in metric metres, or None.
    """
    import trimesh
    if not os.path.exists(mesh_path):
        return None
    mesh = trimesh.load(mesh_path, force="mesh", process=False)

    # Simplify if too dense (avoid memory issues)
    if len(mesh.faces) > 10000:
        try:
            import fast_simplification
            ratio = 1.0 - min(10000 / len(mesh.faces), 0.9999)
            vv, ff = fast_simplification.simplify(
                mesh.vertices, mesh.faces, target_reduction=ratio)
            mesh = trimesh.Trimesh(vertices=vv, faces=ff, process=False)
        except Exception:
            pass  # keep original if simplification fails

    # Apply scale factor
    scale_json = os.path.join(os.path.dirname(mesh_path), "scale.json")
    if os.path.exists(scale_json):
        with open(scale_json) as f:
            sf = float(json.load(f).get("scale_factor", 1.0))
        if sf != 1.0:
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices * sf
    else:
        d = mesh.bounding_sphere.primitive.radius * 2
        if d > 0.8:
            print(f"    ⚠️  No scale.json, mesh diameter={d:.3f}m (wrong scale?)")

    return mesh


# ── FP 位姿 加载 ──────────────────────────────────────────────────────────────
def load_fp_poses(ds, seq_id):
    """Load all ob_in_cam poses for a sequence.
    Returns list of (frame_name, T_4x4) sorted by frame name.
    """
    pose_dir = os.path.join(POSE_BASE, ds, seq_id, "ob_in_cam")
    if not os.path.isdir(pose_dir):
        return []
    files = natsorted(glob(os.path.join(pose_dir, "*.txt")))
    poses = []
    for f in files:
        try:
            T = np.loadtxt(f).reshape(4, 4)
            poses.append((os.path.basename(f).replace(".txt", ""), T))
        except Exception:
            continue
    return poses


# ── MANO 顶点 加载 ────────────────────────────────────────────────────────────
def load_mano_verts(ds, seq_id):
    """Load per-frame MANO vertices from third_mano cache.
    Returns sorted list of (frame_key, verts_778x3) or None.
    """
    npz_path = os.path.join(MANO_BASE, ds, f"{seq_id}.npz")
    if not os.path.exists(npz_path):
        return None
    try:
        data = np.load(npz_path, allow_pickle=True)
        vd = data["verts_dict"].item()
        if not isinstance(vd, dict) or len(vd) == 0:
            return None
        # Sort by frame key
        items = sorted(vd.items(), key=lambda x: x[0])
        result = []
        for k, v in items:
            v = np.array(v)
            if v is not None and v.shape == (778, 3):
                result.append((k, v))
        return result if result else None
    except Exception as e:
        print(f"    MANO load error: {e}")
        return None


# ── Depth Pro K 加载 ─────────────────────────────────────────────────────────
def load_depth_K(ds, seq_id):
    """Load Depth Pro camera intrinsics K (3x3) for a sequence.
    Returns (fx, fy, cx, cy) or None.
    """
    k_path = os.path.join(DEPTH_BASE, ds, seq_id, "K.txt")
    if not os.path.exists(k_path):
        return None
    try:
        K = np.loadtxt(k_path)
        return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    except Exception:
        return None


# ── 接触累积 (单序列) — 支持 3D 距离法 和 2D 像素投影法 ──────────────────────
def accumulate_contacts_sequence(mesh_verts, fp_poses, mano_frames, threshold,
                                 K_params=None, mode='2d', debug=False):
    """Accumulate per-vertex contact counts using 2D pixel-space contact.

    Root cause of 3D mismatch: Depth Pro overestimates focal length vs ARCTIC GT
    (fx=3347 vs fx_true≈2325 for 1000px), causing a ~4× depth scale error between
    FP (uses Depth Pro depth → object at 0.38m) and HaPTIC (uses hand-size reference
    → hand at 2.6m). Both 3D coordinate systems are inconsistent in absolute scale.

    Solution: project both mesh and hand to the 2D image space using K.txt (same K
    for both coordinates — see derivation in comments). Contact = 2D pixel proximity
    within `threshold_px` pixels. This is scale-invariant and avoids the 3D depth
    mismatch completely.

    Args:
        mesh_verts: (V, 3) mesh vertices in canonical space
        fp_poses:   list of (name, T_4x4), N_fp entries
        mano_frames: list of (key, verts_778x3), N_mano entries
        threshold:  NOT USED (kept for API compat); pixel threshold used instead
        K_params:   (fx, cx, cy) from K.txt, or None (uses default fx=3000)

    Returns:
        contact_count (V,), n_counted (int), n_diverged (int)

    Why 2D projection works:
        FP mesh in K_480 space → project to 480px → scale to 1000px:
            u_1000 = mesh_x/Z_fp × (fx_480 × 1000/480) + cx_480×(1000/480)
                   = mesh_x/Z_fp × fx_K.txt + cx_K.txt      ← same as K.txt!
        MANO in K_1000 space → project with K.txt:
            u_1000 = hand_x/Z_mano × fx_K.txt + cx_K.txt
        Both produce CORRECT image-space coordinates for the same physical point.
    """
    if K_params is not None:
        fx, fy, cx, cy = K_params
    else:
        fx, fy, cx, cy = 3000.0, 3000.0, 500.0, 500.0   # reasonable fallback

    # Pixel threshold: 30mm at ~2m true depth with fx≈2325px (ARCTIC GT)
    # = 2325 × 0.030 / 2.0 ≈ 35px. Use 50px for robustness.
    # For non-ARCTIC where Depth Pro K may be more accurate: 
    #   threshold_px = fx × threshold_m / Z_estimated
    # We use a fixed 50px which covers ±30mm at ~1.4m depth for ARCTIC K
    THRESHOLD_PX = 50.0   # pixels in 1000px image space

    V = len(mesh_verts)
    contact_count = np.zeros(V, dtype=np.float32)
    n_fp   = len(fp_poses)
    n_mano = len(mano_frames)
    n_counted  = 0
    n_diverged = 0

    for i, (fp_name, T) in enumerate(fp_poses):
        # ── Proportional mapping: FP frame i → MANO frame j ───────────────
        j = int(round(i * (n_mano - 1) / max(n_fp - 1, 1)))
        j = max(0, min(j, n_mano - 1))
        mano_key, hand_v = mano_frames[j]

        # Alignment guard: warn if temporal drift is large
        # FP frame name is zero-padded int (e.g. "000042"), MANO key same format
        try:
            fp_idx   = int(fp_name)
            mano_idx = int(mano_key)
            drift = abs(fp_idx - mano_idx)
            if drift > 30 and debug:
                print(f"      ⚠️  frame drift fp={fp_idx} mano={mano_idx} Δ={drift}")
        except ValueError:
            pass  # non-numeric keys: skip drift check

        # ── Transform mesh to FP camera space ──────────────────────────────
        R = T[:3, :3]
        t = T[:3, 3]
        dist_t = np.linalg.norm(t)
        if dist_t > 3.0 or dist_t < 0.01 or t[2] < 0.01:
            n_diverged += 1
            continue

        mesh_cam = (R @ mesh_verts.T).T + t  # (V, 3) metres, camera space

        # ── Sanity: MANO depth must be reasonable ───────────────────────────
        z_mano = float(np.median(hand_v[:, 2]))
        if z_mano < 0.05 or z_mano > 20.0:
            n_diverged += 1
            continue

        if mode == '3d':
            # ── 3D 距离法 + 深度归一化 ────────────────────────────────────────
            z_obj   = float(t[2])                          # FP 物体中心深度
            scale_d = z_obj / max(z_mano, 0.01)            # 深度比例

            # Guard: 深度比例异常 → 跳过该帧（防止错误缩放）
            # DexYCB/HO3D RealSense: 手物距离通常 0.3~1.2m, 比例应在 0.3~3.0
            if scale_d < 0.3 or scale_d > 3.0:
                n_diverged += 1
                continue

            hand_v_aligned = hand_v * scale_d
            hand_tree = cKDTree(hand_v_aligned)
            dists_m, _ = hand_tree.query(mesh_cam)
            contact_count[dists_m < threshold] += 1

        else:
            # ── 2D 像素投影法 ──────────────────────────────────────────────
            Z_mesh = mesh_cam[:, 2].clip(0.001)
            u_mesh = mesh_cam[:, 0] / Z_mesh * fx + cx
            v_mesh = mesh_cam[:, 1] / Z_mesh * fy + cy

            Z_hand = hand_v[:, 2].clip(0.001)
            u_hand = hand_v[:, 0] / Z_hand * fx + cx
            v_hand = hand_v[:, 1] / Z_hand * fy + cy

            hand_2d = np.stack([u_hand, v_hand], axis=1)
            mesh_2d = np.stack([u_mesh, v_mesh], axis=1)
            tree = cKDTree(hand_2d)
            dists_px, _ = tree.query(mesh_2d, k=1)
            contact_count[dists_px < THRESHOLD_PX] += 1

        n_counted += 1

    return contact_count, n_counted, n_diverged


# ── Gaussian 平滑 ─────────────────────────────────────────────────────────────
def smooth_contact_on_mesh(mesh, contact_freq, sigma=CONTACT_SIGMA):
    """Apply geodesic-approximate Gaussian smoothing via vertex KDTree."""
    if contact_freq.max() == 0:
        return contact_freq
    verts = np.array(mesh.vertices)
    tree = cKDTree(verts)
    radius = sigma * 3
    smoothed = np.zeros_like(contact_freq)
    for i, v in enumerate(verts):
        idxs = tree.query_ball_point(v, r=radius)
        if not idxs:
            smoothed[i] = contact_freq[i]
            continue
        dists = np.linalg.norm(verts[idxs] - v, axis=1)
        weights = np.exp(-0.5 * (dists / sigma) ** 2)
        smoothed[i] = np.dot(weights, contact_freq[idxs]) / weights.sum()
    return smoothed


# ── 点云采样 + 标签转移 ────────────────────────────────────────────────────────
def sample_pointcloud_with_labels(mesh, contact_per_vertex, n_points=N_POINTS):
    """Sample mesh surface, transfer vertex-level contact labels."""
    import trimesh
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_idx].astype(np.float32)
    labels  = np.zeros(n_points, dtype=np.float32)
    for i, fi in enumerate(face_idx):
        labels[i] = float(np.mean(contact_per_vertex[mesh.faces[fi]]))
    return points.astype(np.float32), normals, labels


# ── 主处理函数 ────────────────────────────────────────────────────────────────
def process_object(ds, obj_ds, obj_name, seqs, mesh_path, threshold,
                   out_train_base=None, out_prior_base=None, redo=False):
    """Process all sequences for one object.

    out_train_base: override output directory (default: OUT_TRAIN_BASE)
    out_prior_base: override prior directory (default: OUT_PRIOR_BASE)
    seqs: list of seq_id strings that map to this object.
    Returns True on success.
    """
    _out_train = out_train_base or OUT_TRAIN_BASE
    _out_prior = out_prior_base or OUT_PRIOR_BASE
    out_train = os.path.join(_out_train, ds, f"{obj_name}.hdf5")
    out_prior = os.path.join(_out_prior,     f"{obj_name}.hdf5")

    if not redo and os.path.exists(out_train):
        print(f"    ⏭  cached: {out_train}")
        return True

    # Load mesh once
    mesh = load_mesh_scaled(mesh_path)
    if mesh is None:
        print(f"    ❌ Mesh not found: {mesh_path}")
        return False

    mesh_verts = np.array(mesh.vertices, dtype=np.float32)
    V = len(mesh_verts)

    # Accumulate across sequences
    total_contact = np.zeros(V, dtype=np.float32)
    total_counted = 0
    total_diverged = 0
    seq_ok = 0

    for seq_id in tqdm(seqs, desc=f"  {obj_name}", leave=False):
        fp_poses = load_fp_poses(ds, seq_id)
        mano_frames = load_mano_verts(ds, seq_id)

        if not fp_poses:
            tqdm.write(f"    ⚠️  {seq_id}: no FP poses")
            continue
        if mano_frames is None:
            tqdm.write(f"    ⚠️  {seq_id}: no MANO cache")
            continue

        # Load Depth Pro K (used for 2D mode, or as reference for 3D mode logging)
        K_params = load_depth_K(ds, seq_id)
        if K_params:
            tqdm.write(f"    [K] fx={K_params[0]:.0f} fy={K_params[1]:.0f} for {seq_id}")

        # 接触检测模式：统一使用 3D + 深度缩放修正
        # ─────────────────────────────────────────────────────────────────
        # 所有第三人称数据集都用 Depth Pro 估计内参 K，存在系统性焦距误差。
        # 同时 MANO (HaPTIC/HaMeR weak-perspective) 的深度 Z 系统性高于 FP。
        # 解决方案: 用 FP 物体深度 Z_obj 对 MANO 做比例缩放，使两者在同一
        # metric 平面对比 (scale_d = Z_obj / Z_hand)。
        # 守卫条件 scale_d ∈ [0.3, 3.0] 会过滤极端异常帧。
        # 2D 投影模式仅作为兜底（当 scale_d 守卫全部失败时不会用到）。
        contact_mode = '3d'   # ← 所有第三人称数据集统一使用

        cnt, n_ok, n_div = accumulate_contacts_sequence(
            mesh_verts, fp_poses, mano_frames, threshold,
            K_params=K_params, mode=contact_mode)
        total_contact += cnt
        total_counted += n_ok
        total_diverged += n_div
        seq_ok += 1
        tqdm.write(f"    {seq_id}: {n_ok}/{len(fp_poses)} frames counted"
                   f"  diverged={n_div}")

    if total_counted == 0:
        print(f"    ❌ No valid frames for {obj_name}")
        return False

    # ── Normalize: count / max_count (per-object) ─────────────────────────
    # 旧方案: / total_counted → 序列越多, max_hp 越小（ycb_dex_03 有293条 → max=0.019）
    # 新方案: 每个顶点用接触次数 / 该物体最大接触次数 → 始终输出 [0, 1]
    max_count = total_contact.max()
    if max_count > 0:
        contact_freq = total_contact / max_count   # per-object normalization
    else:
        contact_freq = total_contact

    # Gaussian smoothing
    contact_smooth = smooth_contact_on_mesh(mesh, contact_freq)

    # Point cloud sampling + label transfer
    points, normals, human_prior = sample_pointcloud_with_labels(mesh, contact_smooth)

    # Force center (使用 top 20% 高接触区域)
    thr_fc = float(np.percentile(contact_smooth[contact_smooth > 0], 80)) \
             if (contact_smooth > 0).any() else 0.05
    high_contact = contact_smooth >= thr_fc
    if high_contact.any():
        force_center = mesh_verts[high_contact].mean(0).astype(np.float32)
    else:
        any_contact = contact_smooth > 0
        force_center = (mesh_verts[any_contact].mean(0).astype(np.float32)
                        if any_contact.any() else np.zeros(3, np.float32))

    n_contact_verts = int((contact_smooth > 0).sum())
    cov_01 = float((human_prior > 0.01).mean()) * 100
    cov_50 = float((human_prior > 0.5).mean()) * 100
    print(f"    ✅ {obj_name}: {seq_ok} seqs, {total_counted} frames, "
          f"contact_verts={n_contact_verts}/{V}, "
          f"hp_max={human_prior.max():.3f}, "
          f"cov(>0.01)={cov_01:.0f}% cov(>0.5)={cov_50:.0f}%")

    # Save training HDF5
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    with h5py.File(out_train, "w") as f:
        f.create_dataset("point_cloud",  data=points,       compression="gzip")
        f.create_dataset("normals",      data=normals,      compression="gzip")
        f.create_dataset("human_prior",  data=human_prior,  compression="gzip")
        f.create_dataset("robot_gt",     data=np.zeros(N_POINTS, np.float32), compression="gzip")
        f.create_dataset("force_center", data=force_center, compression="gzip")
        f.attrs["source"]    = "fp+mano"
        f.attrs["dataset"]   = ds
        f.attrs["object"]    = obj_name
        f.attrs["n_seqs"]    = seq_ok
        f.attrs["threshold"] = threshold

    # Save inference prior HDF5
    os.makedirs(_out_prior, exist_ok=True)
    with h5py.File(out_prior, "w") as f:
        f.create_dataset("point_cloud",  data=points,      compression="gzip")
        f.create_dataset("normals",      data=normals,     compression="gzip")
        f.create_dataset("human_prior",  data=human_prior, compression="gzip")
        f.create_dataset("force_center", data=force_center, compression="gzip")
        f.attrs["source"] = f"video_fp+mano"

    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def _align_one_object(work_item):
    """Top-level worker for multiprocessing.Pool. Processes one object end-to-end.
    Each worker writes its own HDF5 file → fully independent, no shared state."""
    (ds, mesh_ds, obj_name, seqs, mesh_path, threshold_m,
     out_train_base, out_prior_base, redo) = work_item
    try:
        ok = process_object(ds, mesh_ds, obj_name, seqs, mesh_path,
                            threshold=threshold_m,
                            out_train_base=out_train_base,
                            out_prior_base=out_prior_base,
                            redo=redo)
        return obj_name, bool(ok)
    except Exception as e:
        import traceback
        print(f"❌ [{obj_name}] worker exception: {e}")
        traceback.print_exc()
        return obj_name, False


def main():
    parser = argparse.ArgumentParser(
        description="Align MANO vertices with FoundationPose mesh poses → contact labels → HDF5")
    parser.add_argument("--dataset",   default=None,
                        choices=["arctic", "oakink", "ho3d_v3", "dexycb"])
    parser.add_argument("--obj",       default=None, help="Object name substring filter")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="Contact distance threshold in mm (default: 30mm)")
    parser.add_argument("--redo",      action="store_true",
                        help="Recompute even if output exists")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Number of parallel object-level workers (default 1 = sequential). "
                             "Each worker processes one object end-to-end; outputs are independent. "
                             "Recommended ≈ number of physical CPU cores (e.g. 16 on Ryzen 9 9950X).")
    parser.add_argument("--out_suffix", default=None,
                        help="输出目录后缀，用于和原始数据对比。\n"
                             "例如: --out_suffix v2 → training_fp_v2/ human_prior_fp_v2/\n"
                             "不指定则写入默认 training_fp/（覆盖原数据）")
    args = parser.parse_args()

    threshold_m = args.threshold / 1000.0   # mm → m

    # ── 输出路径（支持 --out_suffix 隔离对比版本）────────────────────────────
    if args.out_suffix:
        suffix = args.out_suffix.strip("_")
        out_train_base = os.path.join(config.DATA_HUB, "ProcessedData",
                                      f"training_fp_{suffix}")
        out_prior_base = os.path.join(config.DATA_HUB, "ProcessedData",
                                      f"human_prior_fp_{suffix}")
        print(f"\n⚠️  --out_suffix={suffix}: 输出到独立目录，不覆盖原数据")
        print(f"   训练数据 → {out_train_base}")
        print(f"   推理数据 → {out_prior_base}")
    else:
        out_train_base = OUT_TRAIN_BASE
        out_prior_base = OUT_PRIOR_BASE
        print(f"\n⚠️  将写入默认路径（覆盖原数据），如需隔离请用 --out_suffix")

    # ── Discover all (ds, obj_name) → [seq_ids] ───────────────────────────────
    datasets = [args.dataset] if args.dataset else ["arctic", "oakink", "ho3d_v3", "dexycb"]

    # Build: {(ds, mesh_ds, obj_name, mesh_path) → [seq_ids]}
    obj_to_seqs = {}

    for ds in datasets:
        ds_pose_dir = os.path.join(POSE_BASE, ds)
        if not os.path.isdir(ds_pose_dir):
            continue
        for seq_id in natsorted(os.listdir(ds_pose_dir)):
            ob_cam_dir = os.path.join(ds_pose_dir, seq_id, "ob_in_cam")
            if not os.path.isdir(ob_cam_dir) or not glob(os.path.join(ob_cam_dir, "*.txt")):
                continue
            mesh_ds, obj_name = seq_to_obj(ds, seq_id)
            if obj_name is None:
                continue
            if args.obj and args.obj not in obj_name:
                continue
            mesh_path = os.path.join(MESH_BASE, mesh_ds, obj_name, "mesh.ply")
            if not os.path.exists(mesh_path):
                continue
            key = (ds, mesh_ds, obj_name, mesh_path)
            obj_to_seqs.setdefault(key, []).append(seq_id)

    print(f"\n{'='*60}")
    print(f" MANO × FoundationPose Alignment → Contact Labels")
    print(f" Objects  : {len(obj_to_seqs)}")
    print(f" Threshold: {args.threshold:.0f}mm")
    print(f" Output   : {out_train_base}")
    print(f"{'='*60}\n")

    if not obj_to_seqs:
        print("⚠️  No objects found. Did you run batch_obj_pose.py first?")
        return

    done = failed = 0

    # build worklist once — each item is one (object, all its seqs)
    work = [(ds, mesh_ds, obj_name, seqs, mesh_path, threshold_m,
             out_train_base, out_prior_base, args.redo)
            for (ds, mesh_ds, obj_name, mesh_path), seqs in obj_to_seqs.items()]

    n_workers = max(1, int(args.n_workers))
    if n_workers > 1:
        n_workers = min(n_workers, len(work))
        print(f"⚙️  parallel object workers: {n_workers}  (objects to do: {len(work)})\n")
        # Each worker is independent: writes its own HDF5, no shared mutable state.
        # spawn ctx avoids inheriting heavy parent state (we re-import scipy/trimesh per worker).
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for obj_name, ok in tqdm(
                    pool.imap_unordered(_align_one_object, work),
                    total=len(work), desc="Objects"):
                if ok:
                    done += 1
                else:
                    failed += 1
    else:
        for (ds, mesh_ds, obj_name, mesh_path), seqs in tqdm(
                obj_to_seqs.items(), desc="Objects"):
            tqdm.write(f"\n──────────────────────────────────────────────────────────")
            tqdm.write(f"▶ {ds}/{obj_name}  ({len(seqs)} sequences)")
            ok = process_object(ds, mesh_ds, obj_name, seqs, mesh_path,
                                threshold=threshold_m,
                                out_train_base=out_train_base,
                                out_prior_base=out_prior_base,
                                redo=args.redo)
            if ok:
                done += 1
            else:
                failed += 1

    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ❌ Failed: {failed}")
    print(f"Training HDF5 → {out_train_base}")
    print(f"Human Prior   → {out_prior_base}")
    if args.out_suffix:
        print(f"\n对比命令:")
        print(f"  python tools/vis_contact_heatmap.py --dataset dexycb --batch ")
        print(f"    --out output/vis_contact_heatmap/compare_{args.out_suffix}")


if __name__ == "__main__":
    main()
