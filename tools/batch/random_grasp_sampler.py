#!/usr/bin/env python3
"""
M2 Random Grasp Sampler v2
==========================
内部采样 + ±XYZ 6方向 + Raycast 居中 + 评分系统 + HP引导
采样策略: 50% Human-Prior 引导 + 50% 纯随机

迭代生成: 每批20点×6方向 → 评分 → 不够20个>60分 → 再来一批
最终输出: top 20 高质量候选 (分数>60)

用法:
    # Run from project root
    python3 tools/random_grasp_sampler.py --obj A01001           # OakInk 单个物体
    python3 tools/random_grasp_sampler.py --all                  # OakInk 全部物体
    python3 tools/random_grasp_sampler.py --arctic               # ARCTIC 全部 10 个物体
    python3 tools/random_grasp_sampler.py --arctic --obj scissors # ARCTIC 单个物体
"""
import os, sys, glob, argparse, time
import numpy as np
import trimesh
import h5py
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import json
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HP_DIR        = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp')
TRAINING_FP_ROTATED_DIR = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'train_fp_rotated')
INFER_HP_DIR  = os.path.join(PROJ, 'data_hub', 'human_prior_infer')
OUTPUT_DIR    = os.path.join(PROJ, 'output', 'grasps_candidate')
INFER_OUT_DIR = os.path.join(PROJ, 'output', 'grasps_infer')

# ── 统一 Mesh 来源 ─────────────────────────────────────────────────────────────
OBJ_MESHES_DIR  = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
OBJ_MESHES_DATASETS = ['oakink', 'ycb', 'arctic', 'dexycb', 'egocentric', 'ho3d_v3']
TRAINING_FP_DIR = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp')
SAM3D_ROTATED_MESH_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'SAM3DMesh', 'rotated_mesh')
HP_FP_SUBDIRS = ('oakink', 'dexycb', 'ycb', 'arctic', 'ho3d_v3')
# Canonical rotation file (用于 SAM3D mesh 朝向修正)
CANONICAL_ROT_JSON = os.path.join(PROJ, 'sim', 'canonical_rotation.json')

# ARCTIC legacy — 先把项目根目录加入 sys.path，再 import config
import sys as _sys; _sys.path.insert(0, PROJ)
import config as _cfg
ARCTIC_ROOT = _cfg.ARCTIC_ROOT
ARCTIC_OBJS = ('box capsulemachine espressomachine ketchup microwave '
               'mixer notebook phone scissors waffleiron').split()
ARCTIC_MESH_DIR = os.path.join(ARCTIC_ROOT, 'meta', 'object_vtemplates')
MAX_GRIPPER_OPEN = 0.08

# 与 convert_arctic_to_usd.py 保持一致的规范化旋转
# Key = orig arctic obj name, Value = 3×3 R, applied as: verts = (R @ verts.T).T
ARCTIC_CANONICAL_ROT = {
    'ketchup': np.array([[ 0, 0,-1],
                          [ 0, 1, 0],
                          [ 1, 0, 0]], dtype=np.float64),  # 长轴 X→Z, 竖立
    'phone':   np.array([[ 0, 0,-1],
                          [ 0, 1, 0],
                          [ 1, 0, 0]], dtype=np.float64),  # 长轴 X→Z, 竖立
}
MIN_GRIPPER_WIDTH = 0.005
# MAX_GRIPPER_OPEN = 0.08  已在上方第45行定义 (Franka 最大开口 8cm)
N_POINTS_PER_BATCH  = 20     # 每批采样点数
N_APPROACH_PER_PT   = 6      # 每个内部点随机采样的 approach 方向数
TARGET_HIGH_QUALITY = 50     # 目标高质量候选数
SCORE_THRESHOLD     = 70.0   # 高质量门槛 (R3 提高到 70)


def max_sampler_batches(target_n: int) -> int:
    """迭代采样 batch 上限 = 2× target（至少 1 batch）。"""
    return max(int(target_n) * 2, 1)
REQUIRE_HP_CONTACT_DEFAULT = True   # 至少一个接触点在 human_prior 区域
HP_CONTACT_LABEL_THRESH = 0.3        # 与 sample_points 高 prior 阈值一致
HP_CONTACT_MAX_DIST_M = 0.015        # 接触点到 prior 点云最近邻 ≤ 15mm

# ── Structured HP contact-pair sampling (--structured-contacts) ─────────────
STRUCTURED_N_TANGENT_DIRS = 32
STRUCTURED_MIN_WIDTH_HARD = 0.02     # 硬约束最小开口 2cm（避免细颈 1cm 假抓取）
STRUCTURED_NORMAL_INWARD_COS = 0.25  # n₁·chord ≤ -τ 且 n₂·chord ≥ τ（法向朝向对侧接触）
STRUCTURED_APPROACH_Z_MAX_COS = 0.3    # 与 sample_approach_dirs 一致：禁桌底穿入
STRUCTURED_LOCAL_THICKNESS_MIN = 0.02  # c₁ 切向最大厚度 < 2cm → 跳过
SAMPLING_METHOD_RAYCAST = 'raycast_scored_v2'
SAMPLING_METHOD_EVAL_RAYCAST = 'eval_raycast_gated_v1'
SAMPLING_METHOD_STRUCTURED = 'hp_contact_pair_v1'
SAMPLING_METHOD_ANCHORED = 'anchored_contact_v2'
SAMPLING_METHOD_MIXED = 'mixed_anchored_raycast_v1'
MIXED_ANCHORED_FRACTION = 0.5


def sample_approach_dirs_uniform(n: int) -> list:
    """Uniform random unit approach directions (no tabletop z filter)."""
    dirs = []
    while len(dirs) < n:
        v = np.random.randn(3).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        dirs.append(v)
    return dirs


def sample_approach_dirs(n: int, z_max_cos: float = 0.3) -> list:
    """
    在 canonical 坐标系（Z=竖直向上）中均匀采样 approach 方向。
    
    约定: approach 向量指向夹爪进入物体的方向
      - approach.z < 0  → 从上往下 (top-down)  ✅ 允许
      - approach.z ≈ 0  → 水平侧向              ✅ 允许
      - approach.z > z_max_cos → 从下往上 (桌下穿入) ❌ 禁止
    
    z_max_cos=0.3 表示排除与+Z夹角<72°的向上方向（即wrist在物体正下方）。
    """
    dirs = []
    while len(dirs) < n:
        v = np.random.randn(3).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        if v[2] <= z_max_cos:   # 排除从桌底穿入的向上方向
            dirs.append(v)
    return dirs


def load_canonical_rotations():
    """加载 canonical_rotation.json (SAM3D mesh → 正确朝向的旋转)."""
    if os.path.exists(CANONICAL_ROT_JSON):
        with open(CANONICAL_ROT_JSON) as f:
            d = json.load(f)
        return {k: v for k, v in d.items() if not k.startswith('_')}
    return {}


def infer_obj_dataset(obj_id: str, dataset: str | None = None) -> str:
    if dataset:
        return dataset
    if obj_id.startswith('unseen_'):
        return 'unseen'
    if obj_id.startswith('ycb_dex_'):
        return 'dexycb'
    if obj_id.startswith('arctic_'):
        return 'arctic'
    return 'oakink'


def mesh_scale_json_path(obj_id: str, dataset: str | None = None) -> str | None:
    """scale.json 仍在 obj_meshes（与 convert_usd / sim 一致）。"""
    ds = infer_obj_dataset(obj_id, dataset)
    if obj_id.startswith('ycb_dex_'):
        path = os.path.join(OBJ_MESHES_DIR, 'ycb', obj_id, 'scale.json')
    else:
        path = os.path.join(OBJ_MESHES_DIR, ds, obj_id, 'scale.json')
    return path if os.path.isfile(path) else None


def read_scale_factor(obj_id: str, dataset: str | None = None) -> float:
    path = mesh_scale_json_path(obj_id, dataset)
    if path is None:
        return 1.0
    with open(path) as f:
        return float(json.load(f).get('scale_factor', 1.0))


def apply_metric_scale_to_mesh(obj_id: str, dataset: str | None = None) -> bool:
    """rotated_mesh / obj_meshes 顶点是否 × scale.json（与 convert_obj_usd / Sim 米制一致）。"""
    if obj_id.startswith('arctic_'):
        return False
    return abs(read_scale_factor(obj_id, dataset) - 1.0) > 1e-8


def apply_metric_scale_to_hp_on_load(obj_id: str, dataset: str | None = None) -> bool:
    """
    从 HDF5 读出后是否对 HP 点云 × scale_factor。
    OakInk train_fp_rotated 盘上为 unscaled；YCB dexycb 盘上通常已是 scaled。
    """
    if obj_id.startswith('ycb_dex_'):
        return False
    return apply_metric_scale_to_mesh(obj_id, dataset)


def scale_hp_to_metric(hp_pc: np.ndarray, scale_factor: float) -> np.ndarray:
    sf = float(scale_factor)
    if abs(sf - 1.0) < 1e-8:
        return hp_pc
    return (np.asarray(hp_pc, dtype=np.float64) * sf).astype(np.float32)


def find_obj_mesh(obj_id, dataset=None, *, use_legacy_assets: bool = False):
    """
    查找采样用 mesh + scale.json。

    默认: SAM3DMesh/rotated_mesh/{dataset|ycb}/{obj}/mesh.ply
    --legacy-assets: obj_meshes/.../mesh.ply

    Returns:
        mesh_path, scale_factor, dataset, apply_scale_to_mesh (bool)
    """
    ds = infer_obj_dataset(obj_id, dataset)
    scale_factor = read_scale_factor(obj_id, ds)
    apply_scale = apply_metric_scale_to_mesh(obj_id, ds)

    if use_legacy_assets:
        search_ds = [dataset] if dataset else OBJ_MESHES_DATASETS
        for sub in search_ds:
            mesh_path = os.path.join(OBJ_MESHES_DIR, sub, obj_id, 'mesh.ply')
            if os.path.isfile(mesh_path):
                return mesh_path, scale_factor, sub, apply_scale
        return None, scale_factor, None, apply_scale

    if obj_id.startswith('ycb_dex_') or ds in ('ycb', 'dexycb'):
        mesh_path = os.path.join(SAM3D_ROTATED_MESH_DIR, 'ycb', obj_id, 'mesh.ply')
        store_ds = 'dexycb'
    else:
        mesh_path = os.path.join(SAM3D_ROTATED_MESH_DIR, ds, obj_id, 'mesh.ply')
        store_ds = ds

    if not os.path.isfile(mesh_path):
        return None, scale_factor, None, apply_scale
    return mesh_path, scale_factor, store_ds, apply_scale


def list_dataset_objs(dataset, *, use_legacy_assets: bool = False):
    """列出可采样物体（rotated SAM3D + train_fp_rotated + scale.json）。"""
    if use_legacy_assets:
        ds_dir = os.path.join(OBJ_MESHES_DIR, dataset)
        if not os.path.isdir(ds_dir):
            return []
        return sorted(
            o for o in os.listdir(ds_dir)
            if os.path.isfile(os.path.join(ds_dir, o, 'mesh.ply'))
        )

    out: list[str] = []
    if dataset in ('ycb', 'dexycb'):
        hp_dir = os.path.join(TRAINING_FP_ROTATED_DIR, 'dexycb')
        sam_dir = os.path.join(SAM3D_ROTATED_MESH_DIR, 'ycb')
        if not os.path.isdir(hp_dir) or not os.path.isdir(sam_dir):
            return []
        for name in sorted(os.listdir(hp_dir)):
            if not name.endswith('.hdf5'):
                continue
            obj_id = name[:-5]
            if not obj_id.startswith('ycb_dex_'):
                continue
            if not os.path.isfile(os.path.join(sam_dir, obj_id, 'mesh.ply')):
                continue
            if mesh_scale_json_path(obj_id, 'dexycb'):
                out.append(obj_id)
        return out

    sam_dir = os.path.join(SAM3D_ROTATED_MESH_DIR, dataset)
    if not os.path.isdir(sam_dir):
        return []
    for obj_id in sorted(os.listdir(sam_dir)):
        if not os.path.isfile(os.path.join(sam_dir, obj_id, 'mesh.ply')):
            continue
        if not os.path.isfile(os.path.join(TRAINING_FP_ROTATED_DIR, dataset, f'{obj_id}.hdf5')):
            continue
        if mesh_scale_json_path(obj_id, dataset):
            out.append(obj_id)
    return out


def _hp_search_paths(obj_id: str, hp_dir: str | None, dataset: str | None, *, use_rotated_hp: bool):
    roots = []
    if use_rotated_hp:
        roots.append(TRAINING_FP_ROTATED_DIR)
    roots.append(TRAINING_FP_DIR)
    if hp_dir:
        roots.append(hp_dir)

    subdirs: list[str] = []
    if dataset:
        subdirs.append(dataset)
    if obj_id.startswith('ycb_dex_'):
        subdirs.extend(['dexycb', 'ycb'])
    subdirs.extend(['oakink', 'dexycb', 'ycb', 'arctic'])

    seen: set[str] = set()
    paths: list[str] = []
    for root in roots:
        for sub in subdirs:
            p = os.path.join(root, sub, f'{obj_id}.hdf5')
            if p not in seen:
                seen.add(p)
                paths.append(p)
    for root in ([hp_dir] if hp_dir else [HP_DIR]):
        for name in (f'{obj_id}.hdf5', f'oakink_{obj_id}.hdf5', f'arctic_{obj_id}.hdf5'):
            p = os.path.join(root, name)
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


def load_human_prior(obj_id, hp_dir=None, dataset=None, *, use_rotated_hp: bool = True):
    """
    加载 HumanPrior。默认 train_fp_rotated/；--legacy-assets 时优先 training_fp。
    """
    use_rot = use_rotated_hp and hp_dir not in (INFER_HP_DIR,)
    for path in _hp_search_paths(obj_id, hp_dir, dataset, use_rotated_hp=use_rot):
        if os.path.isfile(path):
            with h5py.File(path, 'r') as f:
                return (
                    f['point_cloud'][()].astype(np.float32),
                    f['human_prior'][()].astype(np.float32),
                    path,
                )
    return None, None, None


def _extents_cm(points: np.ndarray) -> tuple[float, float, float]:
    ext = points.max(axis=0) - points.min(axis=0)
    return float(ext[0] * 100), float(ext[1] * 100), float(ext[2] * 100)


def verify_mesh_hp_scale(
    mesh: trimesh.Trimesh,
    hp_pc: np.ndarray,
    *,
    apply_scale: bool,
    scale_factor: float,
    obj_id: str,
) -> dict:
    """检查采样 mesh 与 HP 尺度/extents 是否一致。"""
    ext_m = _extents_cm(mesh.vertices)
    ext_h = _extents_cm(hp_pc)
    n = min(1500, len(mesh.vertices), len(hp_pc))
    rng = np.random.default_rng(0)
    vm = mesh.vertices[rng.choice(len(mesh.vertices), n, replace=False)]
    vh = hp_pc[rng.choice(len(hp_pc), n, replace=False)]
    nn_cm = float(cKDTree(vh).query(vm, k=1)[0].mean() * 100)

    axis_ratio = max(
        abs(ext_m[i] - ext_h[i]) / (ext_h[i] + 1e-6) for i in range(3)
    )
    ok = nn_cm < 5.0 and axis_ratio < 0.25
    return {
        'ok': ok,
        'nn_cm': nn_cm,
        'axis_ratio': axis_ratio,
        'ext_mesh_cm': ext_m,
        'ext_hp_cm': ext_h,
        'apply_scale': apply_scale,
        'scale_factor': scale_factor,
        'obj_id': obj_id,
    }


def _points_inside(mesh, pts: np.ndarray) -> np.ndarray:
    """mesh.contains 在部分高面数/非封闭 SAM3D mesh 上会触发 trimesh ray IndexError。"""
    if len(pts) == 0:
        return np.array([], dtype=bool)
    try:
        return mesh.contains(pts)
    except (IndexError, ValueError) as e:
        print(f"  [warn] mesh.contains failed ({e}); fallback bbox-only sampling")
        return np.ones(len(pts), dtype=bool)


def contact_in_human_prior_region(
    contact,
    hp_pc: np.ndarray,
    hp_labels: np.ndarray,
    *,
    threshold: float = HP_CONTACT_LABEL_THRESH,
    max_dist: float = HP_CONTACT_MAX_DIST_M,
) -> bool:
    """接触点是否落在 human_prior 高置信区域（prior 点云 KNN）。"""
    pt = np.asarray(contact, dtype=np.float64).reshape(1, 3)
    tree = cKDTree(hp_pc)
    dist, idx = tree.query(pt, k=1)
    if float(dist[0]) > max_dist:
        return False
    return float(hp_labels[int(idx[0])]) > threshold


def passes_hp_contact_requirement(
    contact_L,
    contact_R,
    hp_pc,
    hp_labels,
    *,
    require: bool = True,
    threshold: float = HP_CONTACT_LABEL_THRESH,
    max_dist: float = HP_CONTACT_MAX_DIST_M,
) -> bool:
    """至少一个接触点在 HP 区域；无 prior 数据时不强制。"""
    if not require:
        return True
    if hp_pc is None or hp_labels is None:
        return True
    if not np.any(hp_labels > threshold):
        return True
    return (
        contact_in_human_prior_region(
            contact_L, hp_pc, hp_labels, threshold=threshold, max_dist=max_dist,
        )
        or contact_in_human_prior_region(
            contact_R, hp_pc, hp_labels, threshold=threshold, max_dist=max_dist,
        )
    )


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """表面法向 n 的切平面正交基 (u, v)。"""
    n = np.asarray(normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-8)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(n, ref)
    u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-8)
    return u.astype(np.float32), v.astype(np.float32)


def surface_point_and_normal(mesh, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将点投影到 mesh 表面，返回 (表面点, 朝外法向)。"""
    closest, _, tri = mesh.nearest.on_surface([np.asarray(point, dtype=np.float64)])
    n = mesh.face_normals[int(tri[0])].astype(np.float32)
    return closest[0].astype(np.float32), n


def sample_hp_surface_anchors(
    mesh,
    hp_pc: np.ndarray,
    hp_labels: np.ndarray,
    n_total: int,
    *,
    threshold: float = HP_CONTACT_LABEL_THRESH,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """从 HP 高 prior 点采样 c₁，并投影到 mesh 表面。"""
    high_mask = hp_labels > threshold
    if high_mask.sum() == 0:
        return []
    hp_pts = hp_pc[high_mask]
    weights = hp_labels[high_mask].astype(np.float64)
    weights = weights / (weights.sum() + 1e-8)
    n_try = max(n_total * 8, n_total)
    chosen_idx = np.random.choice(len(hp_pts), size=n_try, replace=True, p=weights)
    anchors = []
    seen = set()
    for idx in chosen_idx:
        if len(anchors) >= n_total:
            break
        c1, n1 = surface_point_and_normal(mesh, hp_pts[idx])
        key = tuple(np.round(c1, 4))
        if key in seen:
            continue
        seen.add(key)
        anchors.append((c1, n1))
    return anchors


def estimate_local_tangent_width(mesh, c1: np.ndarray, n1: np.ndarray, n_dirs: int = 8) -> float:
    """c₁ 切平面多方向射线，估计局部可夹最大厚度。"""
    u, v = _tangent_basis(n1)
    origin = c1 + n1 * 1e-4
    max_w = 0.0
    for k in range(n_dirs):
        theta = 2.0 * np.pi * k / n_dirs
        d = (np.cos(theta) * u + np.sin(theta) * v).astype(np.float32)
        for sign in (1.0, -1.0):
            try:
                hits, _, _ = mesh.ray.intersects_location([origin], [d * sign])
            except Exception:
                continue
            if len(hits) == 0:
                continue
            dists = np.linalg.norm(hits - c1, axis=1)
            valid = dists[(dists >= STRUCTURED_MIN_WIDTH_HARD) & (dists <= MAX_GRIPPER_OPEN)]
            if len(valid):
                max_w = max(max_w, float(valid.max()))
    return max_w


def _segment_grasp_valid(mesh, c1: np.ndarray, c2: np.ndarray, tol: float = 0.006) -> bool:
    """弦 c₁→c₂ 中间无额外穿模交点（空腔中段无交点视为 OK）。"""
    w = float(np.linalg.norm(c2 - c1))
    if w < 1e-6:
        return False
    d = ((c2 - c1) / w).astype(np.float32)
    try:
        hits, _, _ = mesh.ray.intersects_location([c1 + d * 1e-4], [d])
    except Exception:
        return False
    if len(hits) == 0:
        return False
    dists = np.linalg.norm(hits - c1, axis=1)
    mid = dists[(dists > tol) & (dists < w - tol)]
    return len(mid) == 0


def _antipodal_normals_face_each_other(
    n1: np.ndarray,
    n2: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    min_inward: float = STRUCTURED_NORMAL_INWARD_COS,
) -> bool:
    """对跖或柱面夹持：法向沿 chord 相向，或一侧径向（n⊥chord）另一侧朝向 c₁。"""
    chord = (c2 - c1) / (np.linalg.norm(c2 - c1) + 1e-8)
    d1 = float(np.dot(n1, chord))
    d2 = float(np.dot(n2, chord))
    # 两侧近似径向（瓶身侧面）
    if abs(d1) < 0.35 and abs(d2) < 0.35:
        return True
    # 柱面 + 对侧朝向锚点
    if abs(d1) < 0.35 and d2 >= min_inward:
        return True
    if abs(d2) < 0.35 and d1 <= -min_inward:
        return True
    # 平坦对跖面
    return d1 <= -min_inward and d2 >= min_inward


def derive_approach_from_normals(
    n1: np.ndarray,
    n2: np.ndarray,
    finger_dir: np.ndarray,
    z_max_cos: float = STRUCTURED_APPROACH_Z_MAX_COS,
) -> np.ndarray | None:
    """由两侧外法向推导 approach（指向物体内部），并满足桌面约束。"""
    f = finger_dir / (np.linalg.norm(finger_dir) + 1e-8)
    inward = -(np.asarray(n1, dtype=np.float64) + np.asarray(n2, dtype=np.float64))
    inward = inward - np.dot(inward, f) * f
    if np.linalg.norm(inward) < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        inward = np.cross(f, up)
        if np.linalg.norm(inward) < 1e-6:
            inward = np.cross(f, np.array([1.0, 0.0, 0.0]))
    approach = inward / (np.linalg.norm(inward) + 1e-8)
    if approach[2] > z_max_cos:
        approach = -approach
    if approach[2] > z_max_cos:
        return None
    return approach.astype(np.float32)


def _compute_d_near(mesh, grasp_center: np.ndarray, approach: np.ndarray) -> float:
    FINGER_ACTIVE_DEPTH = 0.040
    hits_near, _, _ = mesh.ray.intersects_location([grasp_center], [-approach])
    if len(hits_near):
        d_near = float(np.min(np.linalg.norm(hits_near - grasp_center, axis=1)))
        if d_near > FINGER_ACTIVE_DEPTH:
            return -1.0
        return d_near
    return 0.0


def _build_candidate_dict(
    mesh,
    z_min,
    z_max,
    *,
    grasp_center,
    contact_L,
    contact_R,
    width,
    approach,
    finger_dir,
    mesh_rc=None,
):
    """由接触对组装候选 dict（raycast / structured 共用）。"""
    rc = mesh_rc if mesh_rc is not None else mesh
    d_near = _compute_d_near(rc, grasp_center, approach)
    if d_near < 0:
        return None
    R = make_rotation_matrix(approach, finger_dir)
    gripper_width = float(np.clip(width + 0.005, 0.01, MAX_GRIPPER_OPEN))
    score = score_candidate(
        mesh, width, approach, finger_dir,
        grasp_center, contact_L, contact_R,
        z_min, z_max, mesh_rc=rc,
    )
    return {
        'name': '',
        'position': grasp_center,
        'grasp_point': grasp_center,
        'rotation': R,
        'gripper_width': gripper_width,
        'approach': approach.copy(),
        'finger_dir': finger_dir.copy(),
        'contact_L': contact_L.astype(np.float32),
        'contact_R': contact_R.astype(np.float32),
        'score': score,
        'cross_section_width': float(width),
        'd_near': d_near,
    }


def find_second_contacts_for_c1(
    mesh,
    mesh_rc,
    c1: np.ndarray,
    n1: np.ndarray,
    *,
    hp_pc=None,
    hp_labels=None,
    require_hp_contact: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """在 c₁ 切平面搜索满足硬约束的 c₂（每方向取最远有效穿模点 ≈ 对侧壁）。"""
    rc = mesh_rc if mesh_rc is not None else mesh
    u, v = _tangent_basis(n1)
    origin = c1 + n1 * 1e-4
    pairs: list[tuple[np.ndarray, np.ndarray, float]] = []
    seen_c2: set[tuple[float, float, float]] = set()
    for k in range(STRUCTURED_N_TANGENT_DIRS):
        theta = 2.0 * np.pi * k / STRUCTURED_N_TANGENT_DIRS
        d = (np.cos(theta) * u + np.sin(theta) * v).astype(np.float32)
        for sign in (1.0, -1.0):
            try:
                hits, _, _ = rc.ray.intersects_location([origin], [d * sign])
            except Exception:
                continue
            if len(hits) == 0:
                continue
            dists = np.linalg.norm(hits - c1, axis=1)
            valid = (dists >= STRUCTURED_MIN_WIDTH_HARD) & (dists <= MAX_GRIPPER_OPEN)
            if not np.any(valid):
                continue
            j = int(np.argmax(dists[valid]))
            hit = hits[valid][j]
            w = float(dists[valid][j])
            c2 = hit.astype(np.float32)
            key = tuple(np.round(c2, 4))
            if key in seen_c2:
                continue
            seen_c2.add(key)
            finger_dir = (c2 - c1) / (w + 1e-8)
            if w < 0.03 and not _segment_grasp_valid(rc, c1, c2, tol=0.01):
                continue
            _, n2 = surface_point_and_normal(mesh, c2)
            if not _antipodal_normals_face_each_other(n1, n2, c1, c2):
                continue
            if not passes_hp_contact_requirement(
                c1, c2, hp_pc, hp_labels, require=require_hp_contact,
            ):
                continue
            pairs.append((c2, n2, w))
    return pairs


def generate_structured_one_batch(
    mesh,
    n_anchors: int,
    z_min,
    z_max,
    mesh_rc=None,
    *,
    hp_pc=None,
    hp_labels=None,
    require_hp_contact: bool = REQUIRE_HP_CONTACT_DEFAULT,
):
    """HP 表面 c₁ + 约束搜索 c₂ → 打分。"""
    if hp_pc is None or hp_labels is None or not np.any(hp_labels > HP_CONTACT_LABEL_THRESH):
        return []

    rc = mesh_rc if mesh_rc is not None else mesh
    anchors = sample_hp_surface_anchors(
        mesh, hp_pc, hp_labels, n_anchors, threshold=HP_CONTACT_LABEL_THRESH,
    )
    candidates = []
    for c1, n1 in anchors:
        if estimate_local_tangent_width(rc, c1, n1) < STRUCTURED_LOCAL_THICKNESS_MIN:
            continue
        pairs = find_second_contacts_for_c1(
            mesh, rc, c1, n1,
            hp_pc=hp_pc, hp_labels=hp_labels,
            require_hp_contact=require_hp_contact,
        )
        if not pairs:
            continue
        best = None
        for c2, n2, w in pairs:
            finger_dir = ((c2 - c1) / (w + 1e-8)).astype(np.float32)
            approach = derive_approach_from_normals(n1, n2, finger_dir)
            if approach is None:
                continue
            grasp_center = ((c1 + c2) / 2.0).astype(np.float32)
            cand = _build_candidate_dict(
                mesh, z_min, z_max,
                grasp_center=grasp_center,
                contact_L=c1,
                contact_R=c2,
                width=w,
                approach=approach,
                finger_dir=finger_dir,
                mesh_rc=rc,
            )
            if cand is None:
                continue
            if best is None or cand['score'] > best['score']:
                best = cand
        if best is not None:
            candidates.append(best)
    return candidates


def sample_points(mesh, hp_pc, hp_labels, n_total, has_hp, mesh_contains=None):
    """50% Human-Prior-guided + 50% 纯随机采样.
    
    HP-guided: 在 human_prior > 0.3 的顶点附近 ±5mm jitter, 找 mesh 内部点.
    随机:      在 bbox 内均匀随机采样, 取 mesh 内部点.
    mesh_contains: 用于 contains 检测的 mesh（默认 mesh；建议传简化后的 mesh_rc）
    """
    m_in = mesh_contains if mesh_contains is not None else mesh
    points = []

    # ── 50% HP-guided ──────────────────────────────────────────
    n_hp = n_total // 2 if has_hp else 0
    if n_hp > 0 and hp_pc is not None and hp_labels is not None:
        high_mask = hp_labels > 0.3
        if high_mask.sum() > 0:
            hp_pts = hp_pc[high_mask]
            weights = hp_labels[high_mask]
            weights = weights / (weights.sum() + 1e-8)
            # 按 prior 概率加权采样 HP 顶点 (允许重复)
            chosen_idx = np.random.choice(len(hp_pts), size=n_hp * 10, replace=True, p=weights)
            chosen = hp_pts[chosen_idx]
            # 加 ±5mm jitter
            jitter = np.random.randn(*chosen.shape) * 0.005
            candidates = chosen + jitter
            # 只保留在 mesh 内部的点
            inside = _points_inside(m_in, candidates)
            for p in candidates[inside][:n_hp]:
                points.append(p.astype(np.float32))

    n_hp_got = len(points)

    # ── 50% 纯随机 (补足至 n_total) ────────────────────────────
    n_rand = n_total - n_hp_got
    if n_rand > 0:
        bbox_min, bbox_max = mesh.bounds[0], mesh.bounds[1]
        all_pts = np.random.uniform(bbox_min, bbox_max, size=(n_rand * 20, 3))
        inside = _points_inside(m_in, all_pts)
        for p in all_pts[inside][:n_rand]:
            points.append(p.astype(np.float32))

    return points


def sample_points_pure_random(mesh, n_total, mesh_contains=None):
    """100% uniform bbox interior samples (eval random_pure ablation)."""
    m_in = mesh_contains if mesh_contains is not None else mesh
    points = []
    if n_total <= 0:
        return points
    bbox_min, bbox_max = mesh.bounds[0], mesh.bounds[1]
    all_pts = np.random.uniform(bbox_min, bbox_max, size=(n_total * 20, 3))
    inside = _points_inside(m_in, all_pts)
    for p in all_pts[inside][:n_total]:
        points.append(p.astype(np.float32))
    return points


def generate_one_batch_eval(
    mesh,
    points,
    z_min,
    z_max,
    mesh_rc=None,
    *,
    affordance_gate=None,
    geometry_gates: bool = True,
):
    """Eval ablation raycast.

    geometry_gates=True: width 5–80mm + finger depth <= 4cm (_build_candidate_dict).
    affordance_gate: optional v6 bilateral filter on contacts.
    """
    rc = mesh_rc if mesh_rc is not None else mesh
    candidates = []
    for pt in points:
        for approach in sample_approach_dirs_uniform(N_APPROACH_PER_PT):
            finger_dir = choose_finger_dir(approach)

            hits_pos, _, _ = rc.ray.intersects_location([pt], [finger_dir])
            hits_neg, _, _ = rc.ray.intersects_location([pt], [-finger_dir])

            if len(hits_pos) == 0 or len(hits_neg) == 0:
                continue

            d_pos = np.linalg.norm(hits_pos - pt, axis=1)
            d_neg = np.linalg.norm(hits_neg - pt, axis=1)
            nearest_pos = hits_pos[np.argmin(d_pos)]
            nearest_neg = hits_neg[np.argmin(d_neg)]

            width = np.linalg.norm(nearest_pos - nearest_neg)
            if geometry_gates and (
                width > MAX_GRIPPER_OPEN or width < MIN_GRIPPER_WIDTH
            ):
                continue

            contact_l = nearest_neg.astype(np.float32)
            contact_r = nearest_pos.astype(np.float32)
            if affordance_gate is not None and not affordance_gate(contact_l, contact_r):
                continue

            grasp_center = ((nearest_pos + nearest_neg) / 2.0).astype(np.float32)
            if geometry_gates:
                cand = _build_candidate_dict(
                    mesh,
                    z_min,
                    z_max,
                    grasp_center=grasp_center,
                    contact_L=contact_l,
                    contact_R=contact_r,
                    width=float(width),
                    approach=approach,
                    finger_dir=finger_dir,
                    mesh_rc=rc,
                )
                if cand is not None:
                    candidates.append(cand)
            else:
                R = make_rotation_matrix(approach, finger_dir)
                gripper_width = float(np.clip(width + 0.005, 0.01, MAX_GRIPPER_OPEN))
                score = score_candidate(
                    mesh,
                    width,
                    approach,
                    finger_dir,
                    grasp_center,
                    contact_l,
                    contact_r,
                    z_min,
                    z_max,
                    mesh_rc=rc,
                )
                candidates.append(
                    {
                        "name": "",
                        "position": grasp_center,
                        "grasp_point": grasp_center,
                        "rotation": R,
                        "gripper_width": gripper_width,
                        "approach": approach.copy(),
                        "finger_dir": finger_dir.copy(),
                        "contact_L": contact_l,
                        "contact_R": contact_r,
                        "score": score,
                        "cross_section_width": float(width),
                        "d_near": -1.0,
                    }
                )
    return candidates


def _eval_pool_extend_candidates(
    mesh,
    *,
    mesh_rc,
    z_min: float,
    z_max: float,
    all_candidates: list[dict],
    target_n: int,
    n_batches: int,
    affordance_gate,
    geometry_gates: bool,
) -> int:
    """Run up to n_batches; return number of batches actually run."""
    rc = mesh_rc if mesh_rc is not None else mesh
    ran = 0
    for _ in range(max(0, int(n_batches))):
        if len(all_candidates) >= target_n:
            break
        ran += 1
        pts = sample_points_pure_random(mesh, N_POINTS_PER_BATCH, mesh_contains=rc)
        all_candidates.extend(
            generate_one_batch_eval(
                mesh,
                pts,
                z_min,
                z_max,
                mesh_rc=rc,
                affordance_gate=affordance_gate,
                geometry_gates=geometry_gates,
            )
        )
    return ran


def generate_candidates_eval_pool(
    mesh,
    *,
    mesh_rc=None,
    target_n: int = 50,
    affordance_gate=None,
    max_batches: int | None = None,
) -> tuple[list[dict], dict]:
    """Iterative eval pool.

    Phase 1 (max_batches): all active gates (G1–G3 + optional affordance).
    If still short of target_n: drop affordance gate and keep sampling until full.
    If still short (e.g. random_pure): drop geometry gates too and fill.
    """
    target_n = max(1, int(target_n))
    max_batches = max_sampler_batches(target_n) if max_batches is None else max(1, int(max_batches))
    rc = mesh_rc if mesh_rc is not None else mesh
    z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
    all_candidates: list[dict] = []
    n_gated_batches = _eval_pool_extend_candidates(
        mesh,
        mesh_rc=rc,
        z_min=z_min,
        z_max=z_max,
        all_candidates=all_candidates,
        target_n=target_n,
        n_batches=max_batches,
        affordance_gate=affordance_gate,
        geometry_gates=True,
    )
    n_gated = len(all_candidates)
    affordance_gate_dropped = False
    n_fill_batches = 0
    if len(all_candidates) < target_n and affordance_gate is not None:
        affordance_gate_dropped = True
        n_fill_batches = _eval_pool_extend_candidates(
            mesh,
            mesh_rc=rc,
            z_min=z_min,
            z_max=z_max,
            all_candidates=all_candidates,
            target_n=target_n,
            n_batches=max_batches * 5,
            affordance_gate=None,
            geometry_gates=True,
        )
    geometry_gates_dropped = False
    n_fill_geom_batches = 0
    if len(all_candidates) < target_n:
        geometry_gates_dropped = True
        n_fill_geom_batches = _eval_pool_extend_candidates(
            mesh,
            mesh_rc=rc,
            z_min=z_min,
            z_max=z_max,
            all_candidates=all_candidates,
            target_n=target_n,
            n_batches=max_batches * 5,
            affordance_gate=None,
            geometry_gates=False,
        )

    all_candidates.sort(key=lambda c: -float(c.get("score", 0.0)))
    selected = all_candidates[:target_n]
    for i, c in enumerate(selected):
        c["name"] = f"eval_raycast_{i}"
    stats = {
        "n_batches_gated": n_gated_batches,
        "n_candidates_after_gated": n_gated,
        "n_batches_fill_no_affordance": n_fill_batches,
        "n_batches_fill_no_geometry": n_fill_geom_batches,
        "affordance_gate_dropped": affordance_gate_dropped,
        "geometry_gates_dropped": geometry_gates_dropped,
        "n_candidates_generated": len(all_candidates),
        "n_final": len(selected),
        "n_target": target_n,
        "pool_shortfall": max(0, target_n - len(selected)),
    }
    return selected, stats


def choose_finger_dir(approach):
    up = np.array([0, 0, 1], dtype=np.float32)
    if abs(np.dot(approach, up)) > 0.9:
        return np.array([1, 0, 0], dtype=np.float32)
    else:
        finger = np.cross(approach, up)
        return (finger / (np.linalg.norm(finger) + 1e-8)).astype(np.float32)


def make_rotation_matrix(approach, finger_dir):
    z = approach / (np.linalg.norm(approach) + 1e-8)
    x = finger_dir / (np.linalg.norm(finger_dir) + 1e-8)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-8)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    R = np.column_stack([x, y, z]).astype(np.float32)
    if np.linalg.det(R) < 0:
        R = np.column_stack([-x, y, z]).astype(np.float32)
    return R


def score_candidate(mesh, width, approach, finger_dir, grasp_center,
                    contact_L, contact_R, z_min, z_max, mesh_rc=None):
    """
    物理评分 v5.1 (mesh_rc: 用简化 mesh 做法线查询，大幅加速高面数物体)
    """
    score_mesh = mesh_rc if mesh_rc is not None else mesh

    # === 1. 反力分 (Antipodal, 35%) ===
    closest_L, _, tri_L = score_mesh.nearest.on_surface([contact_L])
    closest_R, _, tri_R = score_mesh.nearest.on_surface([contact_R])
    normal_L = score_mesh.face_normals[tri_L[0]]
    normal_R = score_mesh.face_normals[tri_R[0]]
    antipodal_dot = -np.dot(normal_L, finger_dir) * np.dot(normal_R, finger_dir)
    antipodal_score = float(np.clip(antipodal_dot, 0, 1))

    # === 2. 中心轴对齐分 (Axis Alignment, 25%) ===
    # 物体竖直中轴: 过 XY 重心、方向为世界 Z 轴的直线
    # 越靠近中轴 → 抓取越对称，物体不易侧翻
    centroid_xy = mesh.centroid[:2]
    gc_xy = np.array(grasp_center[:2], dtype=np.float64)
    dist_to_axis = float(np.linalg.norm(gc_xy - centroid_xy))
    extents = mesh.bounds[1] - mesh.bounds[0]
    xy_radius = float(max(extents[0], extents[1]) / 2.0 + 1e-8)
    axis_score = float(np.clip(1.0 - dist_to_axis / xy_radius, 0, 1))

    # === 3. 宽度分 (Width, 20%) ===
    ws = float(np.clip(1.0 - abs(width - 0.035) / 0.045, 0, 1))

    # === 4. Franka 可达性分 (Reachability, 20%) — 连续评分 ===
    # +Y 正前方最可达, 从下方 (-Z) 最难
    # 用 approach 方向与理想方向的余弦相似度做连续评分
    app = np.array(approach, dtype=np.float32)
    app = app / (np.linalg.norm(app) + 1e-8)
    # 理想方向混合: 0.6×+Y + 0.4×-Z (正面偏顶部)
    ideal = np.array([0.0, 0.6, -0.4], dtype=np.float32)
    ideal /= np.linalg.norm(ideal)
    cos_sim = float(np.dot(app, ideal))         # [-1, 1]
    reach_score = float(np.clip((cos_sim + 1) / 2, 0, 1))  # → [0, 1]

    # 合计: 0.35 + 0.25 + 0.20 + 0.20 = 1.00 → × 100
    return (0.35 * antipodal_score +
            0.25 * axis_score +
            0.20 * ws +
            0.20 * reach_score) * 100


def _local_flatness(mesh, point, radius=0.01):
    """计算接触点附近的表面平整度 (法线一致性)."""
    # 找附近的面
    center = np.array(point)
    face_centers = mesh.triangles_center
    dists = np.linalg.norm(face_centers - center, axis=1)
    nearby = dists < radius
    if np.sum(nearby) < 2:
        nearby = dists < radius * 3  # 扩大搜索
    if np.sum(nearby) < 2:
        return 0.5  # 默认中等
    normals = mesh.face_normals[nearby]
    # 法线一致性: 所有法线的平均方向 vs 各法线的 cos 相似度
    mean_n = normals.mean(axis=0)
    mean_n = mean_n / (np.linalg.norm(mean_n) + 1e-8)
    cos_sims = np.dot(normals, mean_n)
    return float(np.clip(np.mean(cos_sims), 0, 1))


def check_finger_reachable(mesh, grasp_center, approach, max_finger_depth=0.04):
    """检查手指能否从 approach 方向到达抓取中心.
    
    从 grasp_center 沿 -approach (向外) 射线 → 打到物体表面
      距离 ≤ 4cm (手指长度) → 手指够得到 ✅
      距离 > 4cm → 手指伸不到 ❌
    """
    hits, _, _ = mesh.ray.intersects_location([grasp_center], [-approach])
    if len(hits) == 0:
        return True  # 没打到表面 = 抓取中心在物体外部边缘, 一定够得到
    
    dists = np.linalg.norm(hits - grasp_center, axis=1)
    nearest_dist = np.min(dists)
    
    return nearest_dist <= max_finger_depth


def generate_one_batch(
    mesh,
    points,
    z_min,
    z_max,
    mesh_rc=None,
    *,
    hp_pc=None,
    hp_labels=None,
    require_hp_contact: bool = REQUIRE_HP_CONTACT_DEFAULT,
):
    """从一批采样点生成候选并评分."""
    PALM_CLEARANCE  = 0.010   # 手掌到近端面最小间距 1cm
    FRANKA_FINGER_D = 0.040   # Franka 指深 4cm

    rc = mesh_rc if mesh_rc is not None else mesh   # 简化 mesh 用于 raycast
    candidates = []
    for pt in points:
        for approach in sample_approach_dirs(N_APPROACH_PER_PT):
            finger_dir = choose_finger_dir(approach)

            hits_pos, _, _ = rc.ray.intersects_location([pt], [finger_dir])
            hits_neg, _, _ = rc.ray.intersects_location([pt], [-finger_dir])

            if len(hits_pos) == 0 or len(hits_neg) == 0:
                continue

            d_pos = np.linalg.norm(hits_pos - pt, axis=1)
            d_neg = np.linalg.norm(hits_neg - pt, axis=1)
            nearest_pos = hits_pos[np.argmin(d_pos)]
            nearest_neg = hits_neg[np.argmin(d_neg)]

            width = np.linalg.norm(nearest_pos - nearest_neg)
            if width > MAX_GRIPPER_OPEN or width < MIN_GRIPPER_WIDTH:
                continue

            grasp_center = ((nearest_pos + nearest_neg) / 2.0).astype(np.float32)

            if not passes_hp_contact_requirement(
                nearest_neg, nearest_pos, hp_pc, hp_labels,
                require=require_hp_contact,
            ):
                continue

            cand = _build_candidate_dict(
                mesh, z_min, z_max,
                grasp_center=grasp_center,
                contact_L=nearest_neg.astype(np.float32),
                contact_R=nearest_pos.astype(np.float32),
                width=float(width),
                approach=approach,
                finger_dir=finger_dir,
                mesh_rc=rc,
            )
            if cand is not None:
                candidates.append(cand)
    return candidates


def generate_candidates_iterative(
    mesh,
    obj_id,
    hp_dir=None,
    mesh_rc=None,
    target_n=None,
    score_threshold=None,
    require_hp_contact: bool = REQUIRE_HP_CONTACT_DEFAULT,
    hp_pc=None,
    hp_labels=None,
    *,
    structured: bool = False,
):
    """迭代生成候选, 直到有 target_n 个分数 > score_threshold."""
    target_n = TARGET_HIGH_QUALITY if target_n is None else target_n
    score_threshold = SCORE_THRESHOLD if score_threshold is None else score_threshold

    if hp_pc is None or hp_labels is None:
        hp_pc, hp_labels, _ = load_human_prior(obj_id, hp_dir=hp_dir)
    has_hp = hp_pc is not None and np.any(hp_labels > 0.5)
    hp_contact_on = (
        require_hp_contact
        and hp_pc is not None
        and hp_labels is not None
        and np.any(hp_labels > HP_CONTACT_LABEL_THRESH)
    )

    # 尺寸预检（米制 mesh）: bbox 最小边 > 2×夹爪开口 → 跳过
    extents = mesh.bounding_box.extents
    min_ext = extents.min()
    if min_ext > 2 * MAX_GRIPPER_OPEN:
        print(f"  [SKIP LARGE] 最小边 {min_ext*100:.1f}cm > {2*MAX_GRIPPER_OPEN*100:.0f}cm, 跳过")
        return []

    if structured and (hp_pc is None or not np.any(hp_labels > HP_CONTACT_LABEL_THRESH)):
        print('  [structured] 无 HP 或 label>阈值 为空，无法采样 c₁')
        return []

    z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
    all_candidates = []
    
    m_contains = mesh_rc if mesh_rc is not None else mesh
    name_prefix = 'hp_pair' if structured else 'raycast'
    max_batches = max_sampler_batches(target_n)
    for batch in range(max_batches):
        if structured:
            new_cands = generate_structured_one_batch(
                mesh, N_POINTS_PER_BATCH, z_min, z_max, mesh_rc=mesh_rc,
                hp_pc=hp_pc, hp_labels=hp_labels,
                require_hp_contact=require_hp_contact,
            )
        else:
            pts = sample_points(
                mesh, hp_pc, hp_labels, N_POINTS_PER_BATCH, has_hp, mesh_contains=m_contains,
            )
            new_cands = generate_one_batch(
                mesh, pts, z_min, z_max, mesh_rc=mesh_rc,
                hp_pc=hp_pc, hp_labels=hp_labels,
                require_hp_contact=require_hp_contact,
            )
        all_candidates.extend(new_cands)

        # 统计高质量候选
        high_quality = [c for c in all_candidates if c['score'] >= score_threshold]
        if structured:
            hp_ratio = f'HP→c2 (minW={STRUCTURED_MIN_WIDTH_HARD*100:.0f}cm)'
        else:
            hp_ratio = "50%HP+50%rnd" if has_hp else "100%rnd"
        if hp_contact_on:
            hp_ratio += "+HP-contact"
        print(f"    batch {batch+1}: +{len(new_cands)} 候选, "
              f"高质量≥{score_threshold:.0f}分: {len(high_quality)}/{target_n} ({hp_ratio})")

        if len(high_quality) >= target_n:
            break

    # 按分数排序, 取 top target_n
    all_candidates.sort(key=lambda c: -c['score'])
    selected = all_candidates[:target_n]
    
    # 重命名
    for i, c in enumerate(selected):
        c['name'] = f'{name_prefix}_{i}'
    
    if selected:
        print(f"  → 最终选出 {len(selected)} 个候选 "
              f"(分数: {selected[0]['score']:.1f} ~ {selected[-1]['score']:.1f})")
    else:
        print(f"  ⚠️ 无有效候选 (物体可能太大，超出夹爪 {MAX_GRIPPER_OPEN*100:.0f}cm 张开)")
    
    return selected


def split_mixed_targets(
    target_n: int,
    anchored_fraction: float = MIXED_ANCHORED_FRACTION,
) -> tuple[int, int]:
    """Split pool target into (n_anchored, n_raycast), ~50/50 for exploration."""
    if target_n <= 0:
        return 0, 0
    n_anchor = int(round(target_n * anchored_fraction))
    n_anchor = max(0, min(target_n, n_anchor))
    if target_n >= 2:
        if n_anchor == 0:
            n_anchor = 1
        if n_anchor >= target_n:
            n_anchor = target_n - 1
    return n_anchor, target_n - n_anchor


def generate_mixed_candidates(
    mesh,
    mesh_rc,
    hp_name,
    hp_dir,
    *,
    anchor_merged_path: str,
    target_n: int,
    score_threshold: float,
    require_hp_contact: bool,
    hp_pc,
    hp_labels,
    anchor_max_rot_deg: float = 8.0,
    anchor_max_tip_jitter_mm: float = 3.0,
    anchor_max_retry_per_slot: int | None = None,
    structured: bool = False,
    anchored_fraction: float = MIXED_ANCHORED_FRACTION,
) -> tuple[list[dict], dict]:
    """
    Per-object pool: ~50% anchored (merged successes) + ~50% raycast for diversity.
    If anchored is short, raycast target is increased to fill the pool.
    """
    from anchor_grasp_gen import generate_anchored_candidates

    n_anchor_target, n_raycast_target = split_mixed_targets(target_n, anchored_fraction)
    meta: dict = {
        "sampling_method": SAMPLING_METHOD_MIXED,
        "target_n": target_n,
        "anchored_fraction": anchored_fraction,
        "n_anchored_target": n_anchor_target,
        "n_raycast_target": n_raycast_target,
        "score_threshold": score_threshold,
    }

    anchored: list[dict] = []
    anchor_meta: dict = {}
    if n_anchor_target > 0 and anchor_merged_path and os.path.isfile(anchor_merged_path):
        anchored, anchor_meta = generate_anchored_candidates(
            mesh,
            mesh_rc,
            anchor_merged_path,
            target_n=n_anchor_target,
            score_threshold=score_threshold,
            require_hp_contact=require_hp_contact,
            hp_pc=hp_pc,
            hp_labels=hp_labels,
            max_rot_deg=anchor_max_rot_deg,
            max_tip_jitter_mm=anchor_max_tip_jitter_mm,
            max_retry_per_slot=anchor_max_retry_per_slot,
        )
        meta["anchor_gen"] = anchor_meta
    elif n_anchor_target > 0:
        meta["anchor_gen"] = {"error": "missing_anchor_merged_path"}

    short_anchor = max(0, n_anchor_target - len(anchored))
    n_raycast_run = n_raycast_target + short_anchor

    raycast: list[dict] = []
    if n_raycast_run > 0:
        if structured:
            raise ValueError("mixed mode does not support structured contacts")
        raycast = generate_candidates_iterative(
            mesh,
            hp_name,
            hp_dir=hp_dir,
            mesh_rc=mesh_rc,
            target_n=n_raycast_run,
            score_threshold=score_threshold,
            require_hp_contact=require_hp_contact,
            hp_pc=hp_pc,
            hp_labels=hp_labels,
            structured=False,
        )

    for c in anchored:
        c["pool_source"] = "anchored"
        base = c.get("name", "c")
        c["name"] = f"mixed_a_{base}"
    for c in raycast:
        c["pool_source"] = "raycast"
        base = c.get("name", "c")
        c["name"] = f"mixed_r_{base}"

    selected_a = sorted(anchored, key=lambda x: x["score"], reverse=True)[:n_anchor_target]
    raycast_sorted = sorted(raycast, key=lambda x: x["score"], reverse=True)
    merged = list(selected_a)
    raycast_slots = target_n - len(merged)
    n_raycast_taken = 0
    if raycast_slots > 0:
        take_r = raycast_sorted[:raycast_slots]
        n_raycast_taken = len(take_r)
        merged.extend(take_r)
    if len(merged) < target_n:
        overflow = sorted(
            anchored[n_anchor_target:] + raycast_sorted[n_raycast_taken:],
            key=lambda x: x["score"],
            reverse=True,
        )
        merged.extend(overflow[: target_n - len(merged)])
    merged.sort(key=lambda x: x["score"], reverse=True)
    merged = merged[:target_n]

    meta.update({
        "n_anchored_selected": len(selected_a),
        "n_raycast_selected": n_raycast_taken,
        "n_anchored_in_pool": sum(1 for c in merged if c.get("pool_source") == "anchored"),
        "n_raycast_in_pool": sum(1 for c in merged if c.get("pool_source") == "raycast"),
        "n_pool": len(merged),
    })
    if merged:
        meta["score_min"] = float(min(c["score"] for c in merged))
        meta["score_max"] = float(max(c["score"] for c in merged))
    return merged, meta


def save_candidates_hdf5(
    candidates,
    obj_id,
    mesh_path,
    output_dir,
    no_rotation=False,
    dataset='oakink',
    *,
    scale_factor: float = 1.0,
    apply_scale_to_mesh: bool = False,
    hp_scale_applied: bool = False,
    hp_path: str | None = None,
    sampling_method: str = SAMPLING_METHOD_RAYCAST,
    extra_metadata: dict | None = None,
    output_hdf5: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    if output_hdf5:
        path = os.path.abspath(output_hdf5)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        path = os.path.join(output_dir, f'{obj_id}_grasp.hdf5')

    _tools_dir = os.path.dirname(os.path.abspath(__file__))
    if _tools_dir not in sys.path:
        sys.path.insert(0, _tools_dir)
    from mesh_utils import (
        applied_mesh_prerotation_record,
        infer_dataset as _infer_ds,
        write_mesh_prerotation_hdf5,
    )
    _dataset_m = _infer_ds(obj_id, dataset)
    _prerot = applied_mesh_prerotation_record(
        obj_id, _dataset_m, no_rotation=no_rotation,
    )

    with h5py.File(path, 'w') as f:
        m = f.create_group('metadata')
        m.attrs['obj_id'] = obj_id
        m.attrs['mesh_path'] = os.path.abspath(mesh_path)
        m.attrs['method'] = sampling_method
        m.attrs['sampling_method'] = sampling_method
        m.attrs['no_rotation'] = bool(no_rotation)
        m.attrs['dataset'] = _dataset_m
        m.attrs['mesh_source'] = 'SAM3DMesh/rotated_mesh'
        m.attrs['scale_factor'] = float(scale_factor)
        m.attrs['scale_applied_to_mesh'] = bool(apply_scale_to_mesh)
        m.attrs['hp_scale_applied_on_load'] = bool(hp_scale_applied)
        m.attrs['coordinate_frame'] = 'metric_scaled' if apply_scale_to_mesh else 'raw_unscaled'
        if hp_path:
            m.attrs['hp_path'] = os.path.abspath(hp_path)
        if extra_metadata:
            for key, val in extra_metadata.items():
                if isinstance(val, (str, int, float, bool)):
                    m.attrs[key] = val
                elif val is not None:
                    m.attrs[key] = str(val)
        cg = f.create_group('candidates')
        cg.attrs['n_candidates'] = len(candidates)
        for i, c in enumerate(candidates):
            ci = cg.create_group(f'candidate_{i}')
            ci.create_dataset('position', data=c['position'])
            ci.create_dataset('grasp_point', data=c['grasp_point'])
            ci.create_dataset('rotation', data=c['rotation'])
            ci.attrs['name'] = c['name']
            ci.attrs['score'] = c['score']
            ci.attrs['gripper_width'] = c['gripper_width']
            ci.attrs['cross_section_width'] = c.get('cross_section_width', 0)
            ci.attrs['d_near'] = c.get('d_near', -1.0)
            if c.get('pool_source'):
                ci.attrs['pool_source'] = c['pool_source']
            write_mesh_prerotation_hdf5(ci, c.get('mesh_prerotation', _prerot))

        if candidates:
            best = candidates[0]
            g = f.create_group('grasp')
            write_mesh_prerotation_hdf5(g, best.get('mesh_prerotation', _prerot))
            g.create_dataset('position', data=best['position'])
            g.create_dataset('grasp_point', data=best['grasp_point'])
            g.create_dataset('rotation', data=best['rotation'])
            quat_xyzw = Rotation.from_matrix(best['rotation']).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            g.create_dataset('quaternion_wxyz', data=quat_wxyz.astype(np.float32))
            g.attrs['gripper_width'] = best['gripper_width']
        
        aff = f.create_group('affordance')
        aff.attrs['n_contact'] = 0
    return path


def visualize_candidates(mesh, candidates, obj_id):
    import open3d as o3d
    
    geometries = []
    
    N_VIS = 30000
    vis_pc, _ = trimesh.sample.sample_surface(mesh, N_VIS)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vis_pc)
    pcd.paint_uniform_color([0.75, 0.75, 0.82])
    geometries.append(pcd)
    
    for i, c in enumerate(candidates):
        center = c['grasp_point']
        
        sphere_L = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sphere_L.translate(c['contact_L'])
        sphere_L.paint_uniform_color([0.9, 0.1, 0.1])
        geometries.append(sphere_L)
        
        sphere_R = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sphere_R.translate(c['contact_R'])
        sphere_R.paint_uniform_color([0.1, 0.3, 0.9])
        geometries.append(sphere_R)
        
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        center_sphere.translate(center)
        center_sphere.paint_uniform_color([1.0, 0.85, 0.0])
        geometries.append(center_sphere)
        
        pts = np.array([c['contact_L'], c['contact_R']])
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(pts)
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[0.6, 0.6, 0.6]])
        geometries.append(line)
        
        arrow_end = center - c['approach'] * 0.05
        arrow_line = o3d.geometry.LineSet()
        arrow_line.points = o3d.utility.Vector3dVector([center, arrow_end])
        arrow_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        arrow_line.colors = o3d.utility.Vector3dVector([[0.2, 0.8, 0.2]])
        geometries.append(arrow_line)
    
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geometries.append(coord)
    
    print(f"\n  🔍 Open3D: {obj_id} (top {len(candidates)} 候选)")
    print(f"     🔴 红=左接触  🔵 蓝=右接触  🟡 黄=中心  🟢 绿=approach")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Top {len(candidates)} — {obj_id}", width=1200, height=800)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = 2.0
    vis.run()
    vis.destroy_window()


def _load_sampler_mesh(mesh_path: str) -> trimesh.Trimesh:
    """Load mesh for sampling; tolerate broken PLY visuals / Scene dumps."""
    last_err: Exception | None = None
    for kwargs in (
        {"force": "mesh", "process": False, "skip_materials": True},
        {"force": "mesh", "process": False},
        {"process": False},
    ):
        try:
            loaded = trimesh.load(mesh_path, **kwargs)
            if isinstance(loaded, trimesh.Scene):
                geoms = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
                if not geoms:
                    raise ValueError("empty scene mesh")
                loaded = trimesh.util.concatenate(geoms)
            if not isinstance(loaded, trimesh.Trimesh):
                raise TypeError(f"expected Trimesh, got {type(loaded)}")
            return loaded
        except Exception as e:
            last_err = e
    raise RuntimeError(f"mesh load failed: {mesh_path}") from last_err


def _safe_mesh_repair(mesh: trimesh.Trimesh, label: str = "mesh") -> None:
    """Best-effort watertight repair; trimesh can crash on huge/broken topology."""
    try:
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
    except (IndexError, ValueError, MemoryError, RuntimeError) as e:
        print(f"  ⚠️  skip watertight repair ({label}): {e}")
        try:
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass


def process_one_object(
    obj_id,
    mesh_path,
    scale_factor,
    apply_scale_to_mesh,
    hp_name,
    hp_dir,
    output_dir,
    *,
    dataset='oakink',
    force=False,
    no_rotation=False,
    target_n=TARGET_HIGH_QUALITY,
    score_threshold=SCORE_THRESHOLD,
    arctic=False,
    require_hp_contact=REQUIRE_HP_CONTACT_DEFAULT,
    use_legacy_assets: bool = False,
    structured: bool = False,
    sampling_mode: str = SAMPLING_METHOD_RAYCAST,
    anchor_merged_path: str | None = None,
    anchor_max_rot_deg: float = 8.0,
    anchor_max_tip_jitter_mm: float = 3.0,
    anchor_max_retry_per_slot: int | None = None,
    write_anchor_meta: bool = False,
):
    """
    为单个物体生成 grasp candidates HDF5。
    返回 (out_path | None, skipped_reason | None)
    """
    skip_path = os.path.join(output_dir, f'{obj_id}.skip')
    out_path = os.path.join(output_dir, f'{obj_id}_grasp.hdf5')

    if not force:
        if os.path.exists(skip_path):
            return None, 'skip_marked'
        if os.path.exists(out_path):
            return out_path, 'exists'

    if not os.path.exists(mesh_path):
        return None, 'no_mesh'

    try:
        mesh = _load_sampler_mesh(mesh_path)
    except Exception as e:
        print(f'  ❌ mesh load failed: {e}')
        return None, 'mesh_load_failed'
    hp_scale_applied = False
    if apply_scale_to_mesh and abs(scale_factor - 1.0) > 1e-8:
        mesh.vertices = mesh.vertices * float(scale_factor)
        print(f'     [scale] mesh × {scale_factor:.6f}  (metric, same as obj_meshes / USD)')

    hp_pc, hp_labels, hp_path = load_human_prior(
        hp_name, hp_dir=hp_dir, dataset=dataset,
        use_rotated_hp=not use_legacy_assets and not arctic,
    )
    if hp_pc is not None and apply_metric_scale_to_hp_on_load(obj_id, dataset):
        hp_pc = scale_hp_to_metric(hp_pc, scale_factor)
        hp_scale_applied = True
        print(f'     [scale] HP point_cloud × {scale_factor:.6f}  (train_fp_rotated was unscaled on disk)')

    if hp_pc is None:
        print('  ⚠️  no HP found (train_fp_rotated / training_fp)')
    else:
        chk = verify_mesh_hp_scale(
            mesh, hp_pc, apply_scale=apply_scale_to_mesh,
            scale_factor=scale_factor, obj_id=obj_id,
        )
        print(f'     [align] HP {os.path.relpath(hp_path, PROJ) if hp_path else "?"}')
        print(f'     [align] ext mesh {chk["ext_mesh_cm"]}  HP {chk["ext_hp_cm"]}  '
              f'NN≈{chk["nn_cm"]:.2f}cm  axisΔ≈{chk["axis_ratio"]*100:.1f}%')
        if not chk['ok']:
            print('  ⚠️  mesh/HP scale or frame mismatch — check assets')

    # rotated SAM3D 已含 +90°X；不再叠 rotation.json
    force_no_rot = (not use_legacy_assets and not arctic) or no_rotation
    if not force_no_rot:
        import sys as _sys
        _tools_dir = os.path.dirname(os.path.abspath(__file__))
        if _tools_dir not in _sys.path:
            _sys.path.insert(0, _tools_dir)
        from mesh_utils import get_canonical_euler as _get_euler
        _dataset = (
            'dexycb' if obj_id.startswith('ycb_')
            else ('arctic' if arctic else dataset)
        )
        rot_euler = _get_euler(obj_id, _dataset)
        if any(abs(e) > 0.5 for e in rot_euler):
            R_mat = Rotation.from_euler('xyz', rot_euler, degrees=True).as_matrix()
            mesh.vertices = (R_mat @ mesh.vertices.T).T
            print(f'     [canonical rot (rotation.json): {[round(e,1) for e in rot_euler]}°]')
        else:
            print(f'     [canonical rot: identity]')
    else:
        print(f'     [no rotation: rotated SAM3D / identity frame]')

    _safe_mesh_repair(mesh, "mesh")

    ext = mesh.bounding_box.extents * 100
    print(f'  尺寸: {ext[0]:.1f}×{ext[1]:.1f}×{ext[2]:.1f} cm  ({len(mesh.faces):,} 面)')

    SIMPLIFY_TARGET = 5000
    mesh_rc = None
    if len(mesh.faces) > SIMPLIFY_TARGET * 2:
        t_s = time.time()
        mesh_rc = mesh.simplify_quadric_decimation(face_count=SIMPLIFY_TARGET)
        _safe_mesh_repair(mesh_rc, "mesh_rc")
        print(f'  → 简化为 {len(mesh_rc.faces):,} 面 (raycast用, {time.time()-t_s:.2f}s)')

    if require_hp_contact:
        print("  [scoring] require ≥1 contact in human_prior region (label>0.3)")
    if structured:
        print(
            f'  [structured] HP c1 (label>{HP_CONTACT_LABEL_THRESH}) -> c2 search '
            f'({STRUCTURED_N_TANGENT_DIRS} dirs, minW={STRUCTURED_MIN_WIDTH_HARD*100:.0f}cm, '
            f'inward-normal>={STRUCTURED_NORMAL_INWARD_COS})'
        )

    anchor_meta = None
    mix_meta = None
    if sampling_mode == SAMPLING_METHOD_MIXED:
        if not anchor_merged_path or not os.path.isfile(anchor_merged_path):
            print(f'  ❌ mixed mode requires --anchor-merged-path ({anchor_merged_path})')
            return None, 'no_anchors'
        if structured:
            print('  ❌ mixed mode does not support --structured-contacts')
            return None, 'bad_args'
        n_a, n_r = split_mixed_targets(target_n)
        print(
            f'  [mixed] merged={anchor_merged_path}  target={target_n}  '
            f'anchored={n_a} raycast={n_r}  score≥{score_threshold}',
        )
        candidates, mix_meta = generate_mixed_candidates(
            mesh,
            mesh_rc,
            hp_name,
            hp_dir,
            anchor_merged_path=anchor_merged_path,
            target_n=target_n,
            score_threshold=score_threshold,
            require_hp_contact=require_hp_contact,
            hp_pc=hp_pc,
            hp_labels=hp_labels,
            anchor_max_rot_deg=anchor_max_rot_deg,
            anchor_max_tip_jitter_mm=anchor_max_tip_jitter_mm,
            anchor_max_retry_per_slot=anchor_max_retry_per_slot,
            structured=structured,
        )
        if mix_meta:
            print(
                f'     in_pool anchored={mix_meta.get("n_anchored_in_pool", "?")}  '
                f'raycast={mix_meta.get("n_raycast_in_pool", "?")}  '
                f'score {mix_meta.get("score_max", "?")}~{mix_meta.get("score_min", "?")}',
            )
        sampling_method = SAMPLING_METHOD_MIXED
        anchor_meta = mix_meta
    elif sampling_mode == SAMPLING_METHOD_ANCHORED:
        if not anchor_merged_path or not os.path.isfile(anchor_merged_path):
            print(f'  ❌ anchored mode requires --anchor-merged-path ({anchor_merged_path})')
            return None, 'no_anchors'
        from anchor_grasp_gen import generate_anchored_candidates

        print(
            f'  [anchored] merged={anchor_merged_path}  target={target_n}  '
            f'score≥{score_threshold}'
        )
        candidates, anchor_meta = generate_anchored_candidates(
            mesh,
            mesh_rc,
            anchor_merged_path,
            target_n=target_n,
            score_threshold=score_threshold,
            require_hp_contact=require_hp_contact,
            hp_pc=hp_pc,
            hp_labels=hp_labels,
            max_rot_deg=anchor_max_rot_deg,
            max_tip_jitter_mm=anchor_max_tip_jitter_mm,
            max_retry_per_slot=anchor_max_retry_per_slot,
        )
        if anchor_meta:
            print(
                f'     anchors={anchor_meta.get("n_anchors", 0)}  '
                f'filter={anchor_meta.get("anchor_filter_used", "?")}  '
                f'retry/slot={anchor_meta.get("max_retry_per_slot", "?")}  '
                f'pool={anchor_meta.get("n_pool", "?")}  '
                f'≥{score_threshold:.0f}:{anchor_meta.get("n_high_quality", "?")}/'
                f'{anchor_meta.get("n_accepted", 0)}  '
                f'score {anchor_meta.get("score_max", "?")}~{anchor_meta.get("score_min", "?")}'
            )
        sampling_method = SAMPLING_METHOD_ANCHORED
    else:
        candidates = generate_candidates_iterative(
            mesh, hp_name, hp_dir=hp_dir, mesh_rc=mesh_rc,
            target_n=target_n, score_threshold=score_threshold,
            require_hp_contact=require_hp_contact,
            hp_pc=hp_pc, hp_labels=hp_labels,
            structured=structured,
        )
        sampling_method = (
            SAMPLING_METHOD_STRUCTURED if structured else SAMPLING_METHOD_RAYCAST
        )
    if candidates:
        extra_md = None
        if sampling_method == SAMPLING_METHOD_MIXED and mix_meta:
            extra_md = {
                "mixed_anchored_fraction": mix_meta.get("anchored_fraction", MIXED_ANCHORED_FRACTION),
                "n_anchored_in_pool": mix_meta.get("n_anchored_in_pool", 0),
                "n_raycast_in_pool": mix_meta.get("n_raycast_in_pool", 0),
            }
        path = save_candidates_hdf5(
            candidates, obj_id, mesh_path, output_dir,
            no_rotation=force_no_rot, dataset=dataset,
            scale_factor=scale_factor,
            apply_scale_to_mesh=apply_scale_to_mesh,
            hp_scale_applied=hp_scale_applied,
            hp_path=hp_path,
            sampling_method=sampling_method,
            extra_metadata=extra_md,
        )
        if write_anchor_meta and anchor_meta and sampling_method in (
            SAMPLING_METHOD_ANCHORED,
            SAMPLING_METHOD_MIXED,
        ):
            try:
                import json
                sidecar = path.replace('.hdf5', '_anchor_meta.json')
                with open(sidecar, 'w') as mf:
                    json.dump(anchor_meta, mf, indent=2, default=str)
            except Exception:
                pass
        n_hi = sum(1 for c in candidates if c["score"] >= score_threshold)
        smin = min(c["score"] for c in candidates)
        smax = max(c["score"] for c in candidates)
        print(
            f'  ✅ → {os.path.basename(path)} ({len(candidates)} 候选)  '
            f'≥{score_threshold:.0f}:{n_hi}  score {smax:.1f}~{smin:.1f}',
        )
        return path, None

    if sampling_mode == SAMPLING_METHOD_ANCHORED:
        skip_reason = 'anchored: no candidates accepted'
    elif sampling_mode == SAMPLING_METHOD_MIXED:
        skip_reason = 'mixed: no candidates accepted'
    else:
        skip_reason = (
            f'{max_sampler_batches(target_n)} sampler batches exhausted, '
            f'0 candidates >= {score_threshold}'
        )
    open(skip_path, 'w').write(f'SKIP: {skip_reason}\n')
    print(f'  ⬛ → {obj_id}.skip (难抓物体，已标记)')
    return None, 'no_candidates'


def main():
    parser = argparse.ArgumentParser(description='Grasp Sampler v2 (Scored + Iterative)')
    parser.add_argument('--obj',     help='单个物体 ID (自动在 obj_meshes/ 所有数据集中查找)')
    parser.add_argument('--all',     action='store_true', help='批量处理 (默认 oakink, 配合 --dataset 使用)')
    parser.add_argument('--dataset', default=None, help='指定数据集: oakink / ycb / arctic / dexycb / egocentric')
    parser.add_argument('--arctic',  action='store_true', help='ARCTIC 10个物体 (mm→m 自动缩放)')
    parser.add_argument('--infer',   action='store_true', help='纯推理模式: 从 human_prior_infer/ 读取, 输出到 grasps_infer/')
    parser.add_argument('--force',   action='store_true', help='强制重新生成（覆盖已有）')
    parser.add_argument('--vis',     action='store_true')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--target', type=int, default=TARGET_HIGH_QUALITY,
                        help=f'目标候选数 (默认 {TARGET_HIGH_QUALITY})')
    parser.add_argument('--score-threshold', type=float, default=SCORE_THRESHOLD,
                        help=f'分数门槛 (默认 {SCORE_THRESHOLD})')
    parser.add_argument('--no-rotation', action='store_true',
                        help='legacy: 不应用 rotation.json；默认 rotated mesh 已是 canonical')
    parser.add_argument(
        '--legacy-assets',
        action='store_true',
        help='回退 obj_meshes + training_fp（不用 rotated_mesh / train_fp_rotated）',
    )
    parser.add_argument(
        '--no-hp-contact-required',
        action='store_true',
        help='关闭硬性要求：至少一个接触点在 human_prior 区域 (默认开启)',
    )
    parser.add_argument(
        '--structured-contacts',
        action='store_true',
        help='HP 表面 c₁ + 切向搜索 c₂（硬约束 antipodal/宽度/弦）再打分；默认 raycast 内部采样',
    )
    parser.add_argument(
        '--sampling-mode',
        choices=(
            SAMPLING_METHOD_RAYCAST,
            SAMPLING_METHOD_STRUCTURED,
            SAMPLING_METHOD_ANCHORED,
            SAMPLING_METHOD_MIXED,
        ),
        default=SAMPLING_METHOD_RAYCAST,
        help=(
            'raycast_scored_v2: default iterative sampler; '
            'hp_contact_pair_v1: use with --structured-contacts; '
            'anchored_contact_v2: perturb merged successful grasps; '
            'mixed_anchored_raycast_v1: ~50%% anchored + ~50%% raycast'
        ),
    )
    parser.add_argument(
        '--anchor-merged-path',
        default=None,
        help='merged {obj}_robot_gt_merged.hdf5 (required for --sampling-mode anchored)',
    )
    parser.add_argument('--anchor-max-rot-deg', type=float, default=8.0)
    parser.add_argument('--anchor-max-tip-jitter-mm', type=float, default=3.0)
    parser.add_argument(
        '--anchor-max-retry-per-slot',
        type=int,
        default=None,
        help='每工位 retry 上限；默认 (100×target)/工位数（通常每工位 100 次）',
    )
    parser.add_argument(
        '--write-anchor-meta',
        action='store_true',
        help='写入 {obj}_grasp_anchor_meta.json（调试用；sim 不读）',
    )
    args = parser.parse_args()

    # 推理模式：切换目录
    _hp_dir  = INFER_HP_DIR if args.infer else HP_DIR
    _out_dir = args.output_dir or (INFER_OUT_DIR if args.infer else OUTPUT_DIR)
    os.makedirs(_out_dir, exist_ok=True)

    # ── 构建 obj_list：5元组 (obj_id, mesh_path, scale, hp_name, hp_dir) ──
    obj_list = []

    if args.arctic:
        objs = [args.obj] if args.obj else ARCTIC_OBJS
        for obj in objs:
            mp = os.path.join(ARCTIC_ROOT, 'meta', 'object_vtemplates', obj, 'mesh_tex.obj')
            arctic_id = f'arctic_{obj}'
            obj_list.append((arctic_id, mp, 1.0 / 1000.0, False, obj, _hp_dir))

    elif args.obj:
        mesh_path, scale_factor, ds, apply_scale = find_obj_mesh(
            args.obj, dataset=args.dataset, use_legacy_assets=args.legacy_assets,
        )
        if mesh_path is None:
            root = OBJ_MESHES_DIR if args.legacy_assets else SAM3D_ROTATED_MESH_DIR
            print(f'❌ mesh 未找到: {args.obj}')
            print(f'   搜索根目录: {root}')
            return
        print(f'   mesh: {mesh_path}')
        print(f'   scale.json factor={scale_factor:.6f}  apply_to_mesh={apply_scale}  dataset={ds}')
        obj_list = [(args.obj, mesh_path, scale_factor, apply_scale, args.obj, _hp_dir)]

    elif args.all or args.dataset:
        target_ds = [args.dataset] if args.dataset else ['oakink']
        for ds in target_ds:
            list_ds = 'dexycb' if ds == 'ycb' else ds
            for obj_id in list_dataset_objs(list_ds, use_legacy_assets=args.legacy_assets):
                mesh_path, scale_factor, _, apply_scale = find_obj_mesh(
                    obj_id, dataset=list_ds, use_legacy_assets=args.legacy_assets,
                )
                if mesh_path:
                    obj_list.append((obj_id, mesh_path, scale_factor, apply_scale, obj_id, _hp_dir))
        print(f'数据集 {target_ds}: {len(obj_list)} 个 ready 物体')

    else:
        print("用法:")
        print("  python3 tools/random_grasp_sampler.py --obj A16013          # 单个物体")
        print("  python3 tools/random_grasp_sampler.py --all                 # OakInk 全部")
        print("  python3 tools/random_grasp_sampler.py --all --dataset ycb   # YCB 全部")
        print("  python3 tools/random_grasp_sampler.py --arctic              # ARCTIC (mm→m)")
        return

    if args.arctic:
        mode = 'ARCTIC'
    elif args.legacy_assets:
        mode = f'legacy obj_meshes/{args.dataset or "oakink"}'
    else:
        mode = f'rotated_mesh + train_fp_rotated ({args.dataset or "oakink"})'
    if args.sampling_mode == SAMPLING_METHOD_ANCHORED:
        sample_mode = 'anchored contact perturb (merged successes)'
    elif args.sampling_mode == SAMPLING_METHOD_MIXED:
        sample_mode = 'mixed ~50% anchored + ~50% raycast'
    elif args.structured_contacts or args.sampling_mode == SAMPLING_METHOD_STRUCTURED:
        sample_mode = f'structured HP→c₂ (min {STRUCTURED_MIN_WIDTH_HARD*100:.0f}cm)'
    else:
        sample_mode = '50%HP + 50%rnd raycast'
    if args.structured_contacts and args.sampling_mode in (
        SAMPLING_METHOD_ANCHORED,
        SAMPLING_METHOD_MIXED,
    ):
        print('❌ --structured-contacts 与 anchored/mixed 不能同时使用')
        return
    if args.sampling_mode in (SAMPLING_METHOD_ANCHORED, SAMPLING_METHOD_MIXED):
        if not args.anchor_merged_path:
            print('❌ --sampling-mode anchored/mixed 需要 --anchor-merged-path')
            return
    eff_mode = args.sampling_mode
    if args.structured_contacts:
        eff_mode = SAMPLING_METHOD_STRUCTURED
    print('=' * 60)
    print(f'  Grasp Sampler v2 [{mode}] ({sample_mode})')
    print(f'  Target: {args.target} candidates ≥ {args.score_threshold} pts')
    if args.legacy_assets:
        print('  Assets: obj_meshes + training_fp')
    else:
        print(f'  Mesh: {SAM3D_ROTATED_MESH_DIR}  × scale.json (metric)')
        print(f'  HP:   {TRAINING_FP_ROTATED_DIR}  (OakInk × scale on load; YCB already scaled on disk)')
    if args.no_rotation or not args.legacy_assets:
        print('  Rotation: identity (+90° already in rotated_mesh; no rotation.json)')
    print(
        f'  HP contact required: {not args.no_hp_contact_required} '
        f'(label>{HP_CONTACT_LABEL_THRESH}, dist≤{HP_CONTACT_MAX_DIST_M*1000:.0f}mm)'
    )
    print('=' * 60)

    generated = 0
    for idx, entry in enumerate(obj_list):
        if len(entry) >= 6:
            obj_id, mesh_path, scale_factor, apply_scale, hp_name, hp_dir_use = entry[:6]
        elif len(entry) == 5:
            obj_id, mesh_path, scale_factor, hp_name, hp_dir_use = entry
            apply_scale = apply_metric_scale_to_mesh(obj_id, args.dataset)
        else:
            obj_id, mesh_path, scale_factor, hp_name = entry
            hp_dir_use = _hp_dir
            apply_scale = apply_metric_scale_to_mesh(obj_id, args.dataset)

        print(f'\n[{idx+1}/{len(obj_list)}] {obj_id}')

        _ds = args.dataset or ('arctic' if args.arctic else 'oakink')
        out_path, reason = process_one_object(
            obj_id, mesh_path, scale_factor, apply_scale, hp_name, hp_dir_use, _out_dir,
            dataset=_ds,
            force=args.force,
            no_rotation=args.no_rotation,
            target_n=args.target,
            score_threshold=args.score_threshold,
            arctic=args.arctic,
            require_hp_contact=not args.no_hp_contact_required,
            use_legacy_assets=args.legacy_assets,
            structured=args.structured_contacts,
            sampling_mode=eff_mode,
            anchor_merged_path=args.anchor_merged_path,
            anchor_max_rot_deg=args.anchor_max_rot_deg,
            anchor_max_tip_jitter_mm=args.anchor_max_tip_jitter_mm,
            anchor_max_retry_per_slot=args.anchor_max_retry_per_slot,
            write_anchor_meta=args.write_anchor_meta,
        )
        if reason == 'skip_marked':
            print(f' ⏭️ [SKIP标记] 已知难抓物体')
        elif reason == 'exists':
            print(' ⏭️ (已生成)')
        elif reason == 'no_mesh':
            print(f' ❌ mesh 不存在: {mesh_path}')
        elif out_path:
            generated += 1

    print(f"\n{'='*60}")
    print(f'  完成! 生成 {generated}/{len(obj_list)} 个物体的候选')
    print(f'  输出: {_out_dir}')
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
