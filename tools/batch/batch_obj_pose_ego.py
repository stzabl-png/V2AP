#!/usr/bin/env python3
"""
batch_obj_pose_ego.py — 第一人称（自中心）视频物体位姿估计 (FoundationPose)

区别于 batch_obj_pose.py（第三人称）：
  • 深度来自 MegaSAM (egocentric_depth/)，60 帧子采样 + cam_c2w 对齐
  • RGB 来自 HaWoR 已提帧的 extracted_images/
  • Mask  来自 SAM2 单帧标注 (obj_recon_input/egocentric/)
  • Mesh  来自 SAM3D 云端重建 (obj_meshes/egocentric/)
  • K 从 K.npy (depth 分辨率) 自动缩放到 FP 推理分辨率

处理流程（每条序列）：
  extracted_images/ + depth.npz + K.npy + Mask + Mesh
    → prepare_scene_ego()  → scene dir (rgb/ depth/ masks/ cam_K.txt)
    → run_fp()             → ob_in_cam/*.txt  track_vis/*.png

输出结构:
  data_hub/ProcessedData/poses/Egocentric/{dataset}/{seq_id}/
    ob_in_cam/000000.txt  ...
    track_vis/000000.png  ...

用法:
  conda activate bundlesdf
  cd $PROJ  # your project root

  # 验证两条序列
  python tools/batch_obj_pose_ego.py

  # 只跑 egodex
  python tools/batch_obj_pose_ego.py --dataset egodex

  # 重新跑（忽略缓存）
  python tools/batch_obj_pose_ego.py --redo

  # 前 N 条
  python tools/batch_obj_pose_ego.py --limit 2
"""

import os, sys, re, argparse, json, time
import numpy as np
import cv2
import torch
from glob import glob
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# ── FoundationPose 路径 (与第三人称共用) ──────────────────────────────────────
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import config as _cfg
FP_ROOT = _cfg.FP_ROOT
# NOTE: sys.path is injected lazily inside init_fp_models() / run_fp()
#       to avoid interfering with the top-level 'import torch'

# ── 数据路径 ──────────────────────────────────────────────────────────────────
EGO_DEPTH_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "egocentric_depth")
EGO_RGB_BASE   = os.path.join(config.DATA_HUB, "RawData",       "EgoRawData")
MESH_BASE      = os.path.join(config.DATA_HUB, "ProcessedData", "obj_meshes", "egocentric")
OBJ_INPUT      = os.path.join(config.DATA_HUB, "ProcessedData", "obj_recon_input", "egocentric")
OUT_BASE       = config.EGO_POSE_DIR   # poses/Egocentric/{ds}/{seq}; ds(hoi4d/egodex) 平级
SCENE_TMP      = "/tmp/fp_scenes_ego"     # ← 与第三人称隔离

SHORTER_SIDE   = 480   # FP 推理边长，同第三人称
MAX_MEGASAM_FRAMES = 60

# ── 序列配置：(dataset, seq_id) → (obj_name, rgb_dir, depth_dir) ─────────────
# obj_name 对应 MESH_BASE/{obj_name}/mesh.ply 以及 OBJ_INPUT/{obj_name}/0.png
_BUILTIN_REGISTRY = {
    # EgoDex (原始2条，保持向后兼容)
    ("egodex", "assemble_disassemble_tiles/1"): {
        "obj_name":   "assemble_tile",
        "rgb_dir":    os.path.join(EGO_RGB_BASE,
                          "egodex/test/assemble_disassemble_tiles/1/extracted_images"),
        "depth_dir":  os.path.join(EGO_DEPTH_BASE,
                          "egodex/assemble_disassemble_tiles/1"),
    },
    # PH2D AVP
    ("ph2d_avp", "1407-picking_orange_tj_2025-03-12_16-42-19/processed_episode_3"): {
        "obj_name":   "orange",
        "rgb_dir":    os.path.join(EGO_RGB_BASE,
                          "ph2d/1407-picking_orange_tj_2025-03-12_16-42-19"
                          "/processed_episode_3/extracted_images"),
        "depth_dir":  os.path.join(EGO_DEPTH_BASE,
                          "ph2d_avp/1407-picking_orange_tj_2025-03-12_16-42-19"
                          "/processed_episode_3"),
    },
}

REGISTRY_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "egodex_sequence_registry.json")


def load_sequence_registry():
    """Load SEQUENCE_REGISTRY from JSON + builtin hardcoded entries.
    EGO-BUG-01 修复: 不再硬编码所有序列，从标注工具生成的 JSON 读取。
    """
    reg = dict(_BUILTIN_REGISTRY)   # start with builtins
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
                "rgb_dir":   cfg["rgb_dir"],
                "depth_dir": cfg["depth_dir"],
                # Optional explicit mesh path (HOI4D meshes live under
                # obj_meshes/hoi4d/, not the egodex obj_meshes/egocentric/ base).
                "mesh_path": cfg.get("mesh_path"),
            }
        print(f"[Registry] Loaded {len(jreg)} entries from JSON  "
              f"(+{len(_BUILTIN_REGISTRY)} builtins) → {len(reg)} total")
    else:
        print(f"[Registry] JSON not found, using {len(reg)} builtin entries only")
    return reg


SEQUENCE_REGISTRY = None   # lazy-loaded in main()


# ── prepare_scene_ego ─────────────────────────────────────────────────────────
def prepare_scene_ego(dataset, seq_id, scene_dir, shorter_side=SHORTER_SIDE):
    """
    第一人称版本的 scene dir 构建:
      • depth.npz (key='depths') @ depth_res  → 缩放到 shorter_side
      • K.npy @ depth_res                     → 随分辨率缩放
      • extracted_images/*.jpg @ orig_res     → 等比缩放到 shorter_side
      • obj_recon_input/egocentric/{obj}/0.png → 随分辨率缩放

    Returns:
      (n_frames, H_fp, W_fp) on success, None on error.
    """
    key = (dataset, seq_id)
    if key not in SEQUENCE_REGISTRY:
        print(f"  ⚠  ({dataset}, {seq_id}) not in SEQUENCE_REGISTRY")
        return None

    cfg       = SEQUENCE_REGISTRY[key]
    obj_name  = cfg["obj_name"]
    rgb_dir   = cfg["rgb_dir"]
    depth_dir = cfg["depth_dir"]

    # ── 深度 & 内参 ───────────────────────────────────────────────────────────
    npz_path = os.path.join(depth_dir, "depth.npz")
    K_path   = os.path.join(depth_dir, "K.npy")
    if not os.path.exists(npz_path) or not os.path.exists(K_path):
        print(f"  ⚠  Missing depth.npz or K.npy in {depth_dir}")
        return None

    depths  = np.load(npz_path)["depths"].astype(np.float32)  # (N_mega, H_d, W_d)
    K_depth = np.load(K_path)                                  # 3×3 @ depth resolution
    N_mega, H_d, W_d = depths.shape

    # ── Annotation mask ───────────────────────────────────────────────────────
    ann_mask_path = os.path.join(OBJ_INPUT, obj_name, "0.png")
    if not os.path.exists(ann_mask_path):
        # Fallback: check /tmp/sam3d_upload location
        ann_mask_path = os.path.join("/tmp/sam3d_upload/egocentric", obj_name, "0.png")
    if not os.path.exists(ann_mask_path):
        print(f"  ⚠  Mask not found for {obj_name} (tried OBJ_INPUT and /tmp/sam3d_upload)")
        return None

    # ── RGB 帧列表 ────────────────────────────────────────────────────────────
    all_rgb = natsorted(glob(os.path.join(rgb_dir, "*.jpg")) +
                        glob(os.path.join(rgb_dir, "*.png")))
    if not all_rgb:
        print(f"  ⚠  No RGB frames in {rgb_dir}")
        return None
    N_total = len(all_rgb)

    # MegaSAM subsampling: frame i → original index i * step
    step = max(1, N_total // MAX_MEGASAM_FRAMES)
    depth_to_rgb = {}
    for i in range(N_mega):
        orig_idx = min(i * step, N_total - 1)
        depth_to_rgb[i] = all_rgb[orig_idx]

    # ── 确定 FP 推理分辨率 (from first RGB) ──────────────────────────────────
    first_rgb = np.array(Image.open(depth_to_rgb[0]).convert("RGB"))
    H0, W0    = first_rgb.shape[:2]
    # Scale to shorter_side, preserving aspect ratio
    scale_fp  = shorter_side / min(H0, W0)
    H_fp      = int(round(H0 * scale_fp))
    W_fp      = int(round(W0 * scale_fp))

    # Scale K: depth → FP resolution
    # K_depth is at (H_d, W_d); RGB is at (H0, W0); FP is at (H_fp, W_fp)
    # We take: K_fp = K_depth scaled by (H_fp/H_d, W_fp/W_d)
    K_fp = K_depth.copy()
    K_fp[0, 0] *= W_fp / W_d   # fx
    K_fp[1, 1] *= H_fp / H_d   # fy
    K_fp[0, 2] *= W_fp / W_d   # cx
    K_fp[1, 2] *= H_fp / H_d   # cy

    # ── 建 scene dir ──────────────────────────────────────────────────────────
    os.makedirs(os.path.join(scene_dir, "rgb"),    exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "depth"),  exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "masks"),  exist_ok=True)

    for i in range(N_mega):
        out_id   = f"{i:06d}"
        rgb_path = depth_to_rgb[i]

        # RGB
        rgb      = np.array(Image.open(rgb_path).convert("RGB"))
        rgb_fp   = cv2.resize(rgb, (W_fp, H_fp))
        cv2.imwrite(os.path.join(scene_dir, "rgb",   f"{out_id}.png"),
                    cv2.cvtColor(rgb_fp, cv2.COLOR_RGB2BGR))

        # Depth (m → mm uint16)
        depth_fp = cv2.resize(depths[i], (W_fp, H_fp))
        depth_mm = (depth_fp * 1000).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(scene_dir, "depth", f"{out_id}.png"), depth_mm)

    # Mask for frame 0 only (FP register)
    mask_raw = cv2.imread(ann_mask_path, 0)
    mask_fp  = cv2.resize(mask_raw, (W_fp, H_fp), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(scene_dir, "masks", "000000.png"), mask_fp)

    # Camera intrinsics
    np.savetxt(os.path.join(scene_dir, "cam_K.txt"), K_fp, fmt="%.6f")

    cov = (mask_fp > 0).mean() * 100
    print(f"  Scene: {N_mega} frames @ {H_fp}×{W_fp}  "
          f"mask={cov:.1f}%  step={step}")
    return N_mega, H_fp, W_fp


# ── FoundationPose 模型初始化 (与第三人称相同) ────────────────────────────────
def init_fp_models():
    """Initialize scorer, refiner, glctx ONCE and reuse across sequences."""
    if FP_ROOT not in sys.path:
        sys.path.insert(0, FP_ROOT)
    import nvdiffrast.torch as dr
    from estimater import ScorePredictor, PoseRefinePredictor, set_seed
    set_seed(0)
    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    return scorer, refiner, glctx


# ── run_fp (与 batch_obj_pose.py 完全相同) ───────────────────────────────────
def run_fp(mesh_path, scene_dir, out_dir, scorer, refiner, glctx,
           est_iter=5, track_iter=2, debug=1):
    if FP_ROOT not in sys.path:
        sys.path.insert(0, FP_ROOT)
    import trimesh, torch, json as _json
    import imageio
    from estimater import (FoundationPose, draw_posed_3d_box, draw_xyz_axis)
    from datareader import YcbineoatReader

    mesh = trimesh.load(mesh_path)

    # Simplify large SAM3D meshes to avoid OOM in nvdiffrast
    if len(mesh.faces) > 5000:
        import fast_simplification
        n_before = len(mesh.faces)
        ratio    = 1.0 - min(5000 / n_before, 0.9999)
        pts_out, faces_out = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_reduction=ratio)
        mesh = trimesh.Trimesh(vertices=pts_out, faces=faces_out, process=False)
        print(f"    Mesh simplified: {n_before:,} → {len(mesh.faces):,} faces")

    # Apply scale.json if present
    scale_json = os.path.join(os.path.dirname(mesh_path), "scale.json")
    if os.path.exists(scale_json):
        sf = float(_json.load(open(scale_json)).get("scale_factor", 1.0))
        if abs(sf - 1.0) > 0.01:
            mesh.vertices *= sf
            print(f"    Scale applied: ×{sf:.4f}")
    else:
        d = mesh.bounding_sphere.primitive.radius * 2
        if d > 0.8:
            print(f"    ⚠  No scale.json, mesh diameter={d:.3f}m — may need scaling")
            print(f"       Run: python data/estimate_obj_scale.py")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=scorer, refiner=refiner,
                         debug_dir=out_dir, debug=debug, glctx=glctx)

    reader = YcbineoatReader(video_dir=scene_dir, shorter_side=None, zfar=np.inf)

    os.makedirs(os.path.join(out_dir, "ob_in_cam"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "track_vis"), exist_ok=True)

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth,
                                ob_mask=mask, iteration=est_iter)
        else:
            pose = est.track_one(rgb=color, depth=depth,
                                 K=reader.K, iteration=track_iter)

        np.savetxt(os.path.join(out_dir, "ob_in_cam", f"{reader.id_strs[i]}.txt"),
                   pose.reshape(4, 4), fmt="%.6f")

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1,
                                K=reader.K, thickness=3,
                                transparency=0, is_input_rgb=True)
            imageio.imwrite(
                os.path.join(out_dir, "track_vis", f"{reader.id_strs[i]}.png"), vis)

        torch.cuda.empty_cache()

    n = len(reader.color_files)
    print(f"    {n} frames → ob_in_cam/ + track_vis/")
    return n


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FoundationPose 物体位姿估计 — 第一人称视频 (egocentric)")
    parser.add_argument("--dataset",    default=None,
                        choices=["egodex", "ph2d_avp", "hoi4d"],
                        help="只处理此数据集（默认全部）")
    parser.add_argument("--seq",        default=None, help="seq_id 子串过滤")
    parser.add_argument("--limit",      type=int, default=0)
    parser.add_argument("--redo",       action="store_true")
    parser.add_argument("--est-iter",   type=int, default=5)
    parser.add_argument("--track-iter", type=int, default=2)
    parser.add_argument("--debug",      type=int, default=1,
                        help="0=无可视化  1=track_vis")
    args = parser.parse_args()

    global SEQUENCE_REGISTRY
    SEQUENCE_REGISTRY = load_sequence_registry()   # EGO-BUG-01 修复

    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**3} GB")
    print(f"Mode: EGOCENTRIC (第一人称)  output → {OUT_BASE}\n")

    # Init FP models once
    print("Initializing FoundationPose models...")
    scorer, refiner, glctx = init_fp_models()
    print("✅ Models ready\n")

    # Build sequence list from registry
    seqs = []
    for (ds, sid), cfg in SEQUENCE_REGISTRY.items():
        if args.dataset and ds != args.dataset:
            continue
        if args.seq and args.seq not in sid:
            continue
        mesh_path = cfg.get("mesh_path") or os.path.join(MESH_BASE, cfg["obj_name"], "mesh.ply")
        if not os.path.exists(mesh_path):
            print(f"  ⚠  Mesh missing: {mesh_path}  (skipping {ds}/{sid})")
            continue
        seqs.append((ds, sid, cfg["obj_name"], mesh_path))

    if args.limit > 0:
        seqs = seqs[:args.limit]

    print(f"{'='*60}")
    print(f" FoundationPose Egocentric Pose Estimation")
    print(f" Sequences : {len(seqs)}")
    print(f" est_iter  : {args.est_iter}   track_iter: {args.track_iter}")
    print(f" Output    : {OUT_BASE}")
    print(f"{'='*60}\n")

    done = skipped = failed = 0

    for ds, seq_id, obj_name, mesh_path in tqdm(seqs, desc="EgoPose"):
        out_dir = os.path.join(OUT_BASE, ds, seq_id.replace("/", "__"))

        if not args.redo and os.path.isdir(os.path.join(out_dir, "ob_in_cam")):
            pose_files = glob(os.path.join(out_dir, "ob_in_cam", "*.txt"))
            if pose_files:
                tqdm.write(f"  ⏭  {ds}/{seq_id}: cached ({len(pose_files)} frames)")
                skipped += 1
                continue

        tqdm.write(f"\n{'─'*58}")
        tqdm.write(f"▶ [{ds}] {seq_id}")
        tqdm.write(f"  obj: {obj_name}  mesh: {mesh_path}")

        t0 = time.time()
        scene_dir = os.path.join(SCENE_TMP, ds, seq_id.replace("/", "__"))
        os.makedirs(scene_dir, exist_ok=True)

        try:
            result = prepare_scene_ego(ds, seq_id, scene_dir,
                                       shorter_side=SHORTER_SIDE)
            if result is None:
                tqdm.write("  ⚠  prepare_scene_ego failed")
                failed += 1
                continue
            n_frames, H, W = result

            n = run_fp(mesh_path, scene_dir, out_dir,
                       scorer=scorer, refiner=refiner, glctx=glctx,
                       est_iter=args.est_iter,
                       track_iter=args.track_iter,
                       debug=args.debug)
            tqdm.write(f"  ✅ {ds}/{seq_id}: {n} poses  ({time.time()-t0:.1f}s)")
            done += 1

        except Exception as e:
            import traceback
            tqdm.write(f"  ❌ {e}")
            traceback.print_exc()
            failed += 1
        finally:
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ⏭ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"Output: {OUT_BASE}")

    os.makedirs(OUT_BASE, exist_ok=True)
    with open(os.path.join(OUT_BASE, "results.json"), "w") as f:
        json.dump({"done": done, "skipped": skipped, "failed": failed}, f, indent=2)


if __name__ == "__main__":
    main()
