#!/usr/bin/env python3
"""
estimate_obj_scale_ego.py — 第一人称视频物体 Scale 估计
(对应第三人称的 data/estimate_obj_scale.py)

原理与第三人称完全一致:
  1. 用 MegaSAM metric depth (depth.npz, 单位 m) + K.npy
  2. SAM2 mask (obj_recon_input/egocentric/{obj}/0.png) 筛选物体像素
  3. Back-project → 3D 点云 → 计算真实直径 d_real
  4. SAM3D mesh 直径 d_mesh
  5. scale_factor = d_real / d_mesh
  6. MANO 交叉验证 (可选)
  7. 保存 scale.json 至 obj_meshes/egocentric/{obj}/

区别于第三人称:
  - 深度来源: MegaSAM depth.npz（不需要 Depth Pro）
  - 选用 depth frame 最靠近 mask 注释帧的那一帧

用法:
    conda run -n hawor python data/estimate_obj_scale_ego.py
    conda run -n hawor python data/estimate_obj_scale_ego.py --obj assemble_tile
    conda run -n hawor python data/estimate_obj_scale_ego.py --redo
"""

import os, sys, json, json, argparse
import numpy as np
import cv2
import trimesh
from scipy.spatial import cKDTree

PROJ      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEPTH_BASE = os.path.join(PROJ, "data_hub", "ProcessedData", "egocentric_depth")
MASK_BASE  = os.path.join(PROJ, "data_hub", "ProcessedData", "obj_recon_input", "egocentric")
MESH_BASE  = os.path.join(PROJ, "data_hub", "ProcessedData", "obj_meshes", "egocentric")

# ── 序列配置：从 JSON registry 动态加载 ──────────────────────────────────────
import sys as _sys
_sys.path.insert(0, os.path.join(PROJ, "tools"))
# 读取两份 registry：batch/ 那份含 hoi4d 的 10 条（带 mesh_path），tools/ 那份含 93 条
# egodex。靠前的优先（hoi4d 的 mesh_path 不被覆盖）。
REGISTRY_JSONS = [
    os.path.join(PROJ, "tools", "batch", "egodex_sequence_registry.json"),
    os.path.join(PROJ, "tools", "egodex_sequence_registry.json"),
]

_BUILTIN_OBJ_DEPTH_MAP = {
    "assemble_tile": ("egodex",
                      "egodex/assemble_disassemble_tiles/1"),
    "orange":        ("ph2d_avp",
                      "ph2d_avp/1407-picking_orange_tj_2025-03-12_16-42-19"
                      "/processed_episode_3"),
}


def _resolve(p):
    """registry 里的路径是项目根相对（data_hub/...），统一转成绝对路径；None 原样返回。"""
    if not p:
        return None
    return p if os.path.isabs(p) else os.path.join(PROJ, p)


def load_registry_maps():
    """EGO-BUG-01 修复: 从 JSON registry 构建 obj → (dataset, depth_dir) 与
    obj → mesh_path 两个映射。跳过 skipped=True；靠前的 registry 优先。
    registry 路径转为绝对路径；builtins 仍保持 DEPTH_BASE 相对。
    """
    odm  = dict(_BUILTIN_OBJ_DEPTH_MAP)
    mesh = {}
    from_reg = set()
    for reg_path in REGISTRY_JSONS:
        if not os.path.exists(reg_path):
            continue
        with open(reg_path) as f:
            jreg = json.load(f)
        for key, cfg in jreg.items():
            if cfg.get("skipped"):
                continue
            obj_name = cfg["obj_name"]
            if obj_name in from_reg:        # 靠前的 registry（batch/hoi4d）优先
                continue
            from_reg.add(obj_name)
            odm[obj_name]  = (cfg["dataset"], _resolve(cfg["depth_dir"]))
            mesh[obj_name] = _resolve(cfg.get("mesh_path"))
    return odm, mesh


OBJ_DEPTH_MAP = None  # lazy-loaded in main()
OBJ_MESH_MAP  = None  # lazy-loaded in main()


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def depth_mask_to_pointcloud(depth_m, mask_resized, K, max_depth=3.0):
    """
    Back-project mask pixels into 3D using metric depth and K.
    Improvements v2:
      - ±20% median filter (tighter, reduces background leakage)
      - Minimum 30 valid points required
    Returns (N, 3) array or None.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    v_idx, u_idx = np.where((mask_resized > 0) &
                             (depth_m > 0.05) &
                             (depth_m < max_depth))
    if len(v_idx) < 30:   # v2: raised from 10
        return None

    z = depth_m[v_idx, u_idx].astype(np.float64)

    # v2: tighter ±20% filter (was ±40%)
    # Rationale: objects at arm's length have very consistent depth per mask region.
    # ±40% allowed background depth values to leak in, inflating diameter.
    z_med = np.median(z)
    keep  = (z > z_med * 0.80) & (z < z_med * 1.20)
    if keep.sum() < 20:   # v2: fallback only if very few survive
        # Try ±30% before giving up tight filter
        keep = (z > z_med * 0.70) & (z < z_med * 1.30)
    if keep.sum() < 10:
        keep = np.ones(len(z), dtype=bool)

    z = z[keep]
    x = (u_idx[keep] - cx) * z / fx
    y = (v_idx[keep] - cy) * z / fy

    return np.stack([x, y, z], axis=1)


def pointcloud_diameter(pts, percentile=95):
    """
    Estimate object diameter using PCA-axis spans (v2, from third-person script).
    More robust than XY-lateral: handles objects not face-on to camera.
    Returns float diameter in metres.
    """
    if pts is None or len(pts) < 10:
        return None
    c = pts.mean(axis=0)
    p = pts - c
    try:
        _, _, Vt = np.linalg.svd(p, full_matrices=False)
    except Exception:
        # Fallback: XY span
        x_span = np.percentile(pts[:, 0], percentile) - np.percentile(pts[:, 0], 100 - percentile)
        y_span = np.percentile(pts[:, 1], percentile) - np.percentile(pts[:, 1], 100 - percentile)
        return float(np.sqrt(x_span**2 + y_span**2))
    spans = []
    for v in Vt:
        proj = p @ v
        lo = np.percentile(proj, 100 - percentile)
        hi = np.percentile(proj, percentile)
        spans.append(hi - lo)
    return float(max(spans))


def mesh_diameter(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh")
    return float(mesh.bounding_sphere.primitive.radius * 2), mesh


def best_depth_frame(depths, mask_d, annotated_frame_hint=None):
    """
    Select the depth frame with the most valid masked pixels near median depth.
    If annotated_frame_hint is given (video frame index), search nearby depth frames.
    Returns frame index (int).
    """
    N = depths.shape[0]
    if N == 1:
        return 0

    # Convert video frame index to depth frame index (depths is subsampled)
    if annotated_frame_hint is not None:
        # depth frames are evenly subsampled; map proportionally
        # (annotated_frame_hint is in the extracted_images space)
        # We don't know the total video frames here, so use closest fraction
        center = int(round(annotated_frame_hint / max(annotated_frame_hint + 1, 1) * (N - 1)))
        search_range = range(max(0, center - 5), min(N, center + 6))
    else:
        search_range = range(N)

    best_fi, best_n = 0, 0
    for fi in search_range:
        dm = depths[fi]
        valid = (mask_d > 0) & (dm > 0.05) & (dm < 3.0)
        n = int(valid.sum())
        if n > best_n:
            best_n = n
            best_fi = fi
    return best_fi


def estimate_scale(obj_name, depth_dir, mask_path, mesh_path,
                   annotated_frame_hint=None, use_best_frame=True):
    """
    Core estimation function (v2).
    annotated_frame_hint: video frame index where mask was annotated
    use_best_frame: auto-select the depth frame with most valid mask pixels
    Returns result dict or None.
    """
    # 1. Load metric depth + K
    npz_path = os.path.join(depth_dir, "depth.npz")
    K_path   = os.path.join(depth_dir, "K.npy")
    if not os.path.exists(npz_path) or not os.path.exists(K_path):
        print(f"  ❌ Missing depth.npz or K.npy in {depth_dir}")
        return None

    depths = np.load(npz_path)["depths"].astype(np.float32)  # (N, H_d, W_d)
    K      = np.load(K_path)                                   # 3×3 @ depth res
    N_d, H_d, W_d = depths.shape

    # 2. Load & resize mask to depth resolution
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        print(f"  ❌ Mask not found: {mask_path}")
        return None
    mask_d = cv2.resize(mask, (W_d, H_d), interpolation=cv2.INTER_NEAREST)
    n_mask = int((mask_d > 0).sum())
    print(f"  mask: {mask.shape} → {mask_d.shape}  nonzero={n_mask}")
    if n_mask < 20:
        print("  ❌ Too few mask pixels after resize")
        return None

    # 3. Select best depth frame (v2: auto-select instead of always frame 0)
    if use_best_frame:
        fi = best_depth_frame(depths, mask_d, annotated_frame_hint)
    else:
        fi = 0
    depth_m = depths[fi]
    print(f"  depth frame {fi}/{N_d}: {H_d}×{W_d}  "
          f"range [{depth_m.min():.2f}, {depth_m.max():.2f}]m  "
          f"(annotated_hint={annotated_frame_hint})")

    # 4. Back-project → 3D PointCloud
    pts = depth_mask_to_pointcloud(depth_m, mask_d, K)
    if pts is None or len(pts) < 30:
        print(f"  ❌ Too few 3D points ({0 if pts is None else len(pts)})")
        return None
    print(f"  3D points: {len(pts)}  depth range [{pts[:,2].min():.3f},{pts[:,2].max():.3f}]m")

    # 5. Real diameter (v2: PCA-based)
    d_real = pointcloud_diameter(pts)
    if d_real is None or d_real < 0.005:
        print(f"  ❌ Degenerate diameter: {d_real}")
        return None
    print(f"  d_real = {d_real:.4f} m")

    # 6. Mesh diameter
    d_mesh, mesh = mesh_diameter(mesh_path)
    print(f"  d_mesh = {d_mesh:.4f} m")
    if d_mesh < 1e-6:
        return None

    scale_factor = d_real / d_mesh

    # 7. Plausibility check: warn if diameter is implausible for a hand-held object
    # Hand-held objects at arm's length: typically 0.03–0.50m diameter
    warn = ""
    if d_real < 0.02:
        warn = "⚠️  d_real < 2cm (very small, check mask)"
    elif d_real > 0.60:
        warn = "⚠️  d_real > 60cm (implausibly large — check frame alignment)"
    print(f"  scale_factor = {scale_factor:.6f}  {warn}")

    return {
        "scale_factor":   round(float(scale_factor), 6),
        "d_real_m":       round(float(d_real), 4),
        "d_mesh_raw":     round(float(d_mesh), 4),
        "depth_frame":    fi,
        "n_pts":          int(len(pts)),
        "method":         "megasam_depth_mask_v2",
        "obj":            obj_name,
        "plausible":      warn == "",
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Ego 物体 Scale 估计 (MegaSAM depth + SAM2 mask) v2")
    parser.add_argument("--obj",  default=None,
                        help="只处理此物体（默认全部）")
    parser.add_argument("--redo", action="store_true", help="重新计算已有 scale.json")
    parser.add_argument("--no-best-frame", action="store_true",
                        help="禁用最佳帧自动选择，始终用 depth frame 0（旧行为）")
    args = parser.parse_args()

    global OBJ_DEPTH_MAP, OBJ_MESH_MAP
    OBJ_DEPTH_MAP, OBJ_MESH_MAP = load_registry_maps()   # EGO-BUG-01 修复
    objs = [args.obj] if args.obj else list(OBJ_DEPTH_MAP.keys())

    print("=" * 60)
    print(" Ego Object Scale Estimation")
    print(" Method: MegaSAM metric depth + SAM2 mask → d_real / d_mesh")
    print("=" * 60)

    results = []
    for obj_name in objs:
        print(f"\n── {obj_name} ──")
        ds, depth_sub = OBJ_DEPTH_MAP[obj_name]
        # depth_sub may be an absolute path (from JSON) or a relative subdir (builtins)
        depth_dir = depth_sub if os.path.isabs(depth_sub) else os.path.join(DEPTH_BASE, depth_sub)
        mask_path  = os.path.join(MASK_BASE, obj_name, "0.png")
        # HOI4D meshes live under obj_meshes/hoi4d/ (registry mesh_path);
        # egodex/builtins fall back to obj_meshes/egocentric/{obj}/mesh.ply.
        mesh_path  = OBJ_MESH_MAP.get(obj_name) or os.path.join(MESH_BASE, obj_name, "mesh.ply")
        mesh_dir   = os.path.dirname(mesh_path)
        scale_json = os.path.join(mesh_dir, "scale.json")

        if not args.redo and os.path.exists(scale_json):
            with open(scale_json) as f:
                cached = json.load(f)
            sf = cached.get("scale_factor", "?")
            print(f"  ⏭  Cached: scale_factor={sf}  (use --redo to recompute)")
            results.append(cached)
            continue

        if not os.path.exists(mesh_path):
            print(f"  ❌ mesh.ply not found: {mesh_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"  ❌ mask not found: {mask_path}")
            continue

        # Read annotated_frame hint from registry if available (first match wins)
        ann_frame = None
        for reg_path in REGISTRY_JSONS:
            if not os.path.exists(reg_path):
                continue
            with open(reg_path) as f:
                jreg = json.load(f)
            hit = next((c for c in jreg.values()
                        if c.get("obj_name") == obj_name and not c.get("skipped")), None)
            if hit:
                ann_frame = hit.get("annotated_frame")
                break

        result = estimate_scale(
            obj_name, depth_dir, mask_path, mesh_path,
            annotated_frame_hint=ann_frame,
            use_best_frame=not args.no_best_frame)

        if result is None:
            print(f"  ❌ Scale estimation failed for {obj_name}")
            continue

        # Save scale.json
        os.makedirs(mesh_dir, exist_ok=True)
        with open(scale_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  ✅ Saved → {scale_json}")
        results.append(result)

    print(f"\n{'='*60}")
    print(f"{'Object':<20} {'scale_factor':>13} {'d_real_m':>10} {'d_mesh_raw':>12} {'method':>22}")
    print("-" * 80)
    for r in results:
        print(f"  {r['obj']:<18} {r['scale_factor']:>13.6f} "
              f"{r['d_real_m']:>8.4f}m {r['d_mesh_raw']:>10.4f}m "
              f"  {r.get('method','?'):>22}")

    print()
    print("下一步: 重跑接触图")
    print("  conda run -n hawor python tools/gen_ego_contact_map.py")


if __name__ == "__main__":
    main()
