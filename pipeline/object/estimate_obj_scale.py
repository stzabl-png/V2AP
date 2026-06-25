#!/usr/bin/env python3
"""
estimate_obj_scale.py — 估计 SAM3D Mesh 的尺度校正因子 (通用，所有数据集)

原理
====
  每个物体在 obj_recon_input/{ds}/{obj}/ 有：
    image.png   — 标注帧原图
    0.png       — SAM2 物体 mask
    source_frame.txt — 来源序列信息

  步骤:
    1. 对 image.png 跑 Depth Pro → metric 深度 D (H×W, 单位 m) + 估计内参 K
    2. 在 mask 区域(非零像素)反投影:
         X = (u - cx) / fx * D[v, u]
         Y = (v - cy) / fy * D[v, u]
         Z = D[v, u]
       得到物体 3D 点云 (metric, 单位 m)
    3. 计算点云的实际直径 d_real (95 percentile pairwise span)
    4. 加载 SAM3D mesh, 计算其 bounding sphere 直径 d_mesh
    5. scale_factor = d_real / d_mesh
    6. MANO 手部交叉验证（可选）：读取 third_mano 缓存,
       找接触帧（MANO 手顶点中 z 最小 = 最近处），
       用 Depth Pro 估计的 K 投影到图像，
       检查手部深度与 Depth Pro 深度的一致性 → 输出警告

  输出:
    obj_meshes/{ds}/{obj}/scale.json
      {
        "scale_factor": 0.213,      # mesh × scale → metric (m)
        "d_real_m":     0.267,      # 点云直径 (m)
        "d_mesh_raw":   1.254,      # SAM3D mesh 原始直径
        "method":       "depth_mask",
        "mano_check":   "consistent" | "divergent" | "skipped"
      }

用法:
    # 需要 depth-pro 环境（有 Depth Pro 模型）
    conda activate depth-pro
    cd $PROJ

    python data/estimate_obj_scale.py              # 所有数据集所有物体
    python data/estimate_obj_scale.py --dataset arctic
    python data/estimate_obj_scale.py --obj box    # 只处理名称含 'box' 的物体
    python data/estimate_obj_scale.py --redo       # 重新计算已有 scale.json

注意:
    scale_factor 用于 batch_obj_pose.py: mesh.vertices *= scale_factor
"""

import os, sys, json, argparse, gc
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# ── Depth Pro ─────────────────────────────────────────────────────────────────
DEPTH_PRO_DIR = os.path.join(config.PROJECT_DIR, "third_party", "ml-depth-pro", "src")
sys.path.insert(0, DEPTH_PRO_DIR)

# ── 路径 ──────────────────────────────────────────────────────────────────────
OBJ_INPUT       = os.path.join(config.DATA_HUB, "ProcessedData", "obj_recon_input")
MESH_BASE       = os.path.join(config.DATA_HUB, "ProcessedData", "obj_meshes")
SAM3D_MESH_BASE = os.path.join(config.DATA_HUB, "meshes", "SAM3DMesh", "meshes")
MANO_BASE       = os.path.join(config.DATA_HUB, "ProcessedData", "third_mano")
DEPTH_BASE      = os.path.join(config.DATA_HUB, "ProcessedData", "third_depth")


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def load_depth_pro():
    import torch, depth_pro
    device = torch.device("cuda" if __import__("torch").cuda.is_available() else "cpu")
    model, transform = depth_pro.create_model_and_transforms(
        device=device, precision=__import__("torch").float16)
    model.eval()
    return model, transform, device


def run_depth_pro_on_image(model, transform, device, img_path):
    """Run Depth Pro on a single image.
    Returns: depth_m (H,W float32 metres), K (3x3), (H, W)
    """
    import depth_pro, torch
    image, _, f_px = depth_pro.load_rgb(img_path)
    image_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model.infer(image_t, f_px=f_px)
    depth_m  = pred["depth"].squeeze().cpu().float().numpy()
    focallen = float(pred["focallength_px"])
    H, W     = depth_m.shape
    cx, cy   = W / 2.0, H / 2.0
    K = np.array([[focallen, 0, cx],
                  [0, focallen, cy],
                  [0, 0, 1]], dtype=np.float32)
    return depth_m, K, H, W


def depth_mask_to_pointcloud(depth_m, mask, K):
    """Back-project masked depth pixels to 3D (metric metres).
    Returns Nx3 array.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ys, xs = np.where((mask > 0) & (depth_m > 0.05) & (depth_m < 10.0))
    if len(xs) == 0:
        return None
    z = depth_m[ys, xs]
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    return np.stack([x, y, z], axis=1)


def pointcloud_diameter(pts, percentile=95):
    """Estimate object diameter from point cloud using projected spans.
    Avoids O(N²) pairwise distance by using PCA-axis spans.
    """
    if pts is None or len(pts) < 10:
        return None
    # Center
    c = pts.mean(axis=0)
    p = pts - c
    # PCA
    try:
        _, _, Vt = np.linalg.svd(p, full_matrices=False)
    except Exception:
        return float(np.linalg.norm(pts.max(0) - pts.min(0)))
    # Project onto first 3 PCs and take max span
    spans = []
    for v in Vt:
        proj = p @ v
        lo = np.percentile(proj, 100 - percentile)
        hi = np.percentile(proj, percentile)
        spans.append(hi - lo)
    return float(max(spans))


def mesh_diameter(mesh_path):
    """Load trimesh mesh, return bounding sphere diameter."""
    import trimesh
    m = trimesh.load(mesh_path, force="mesh", process=False)
    return float(m.bounding_sphere.primitive.radius * 2), m


def mano_check(ds, seq_id, K, scale_factor, obj_mask, depth_m):
    """Optional cross-validation using MANO cache.
    Returns "consistent" | "divergent" | "skipped".
    """
    mano_path = os.path.join(MANO_BASE, ds, f"{seq_id}.npz")
    if not os.path.exists(mano_path):
        return "skipped"

    try:
        data = np.load(mano_path, allow_pickle=True)
        verts_dict = data["verts_dict"].item()
        if not verts_dict:
            return "skipped"

        # Find the frame closest to this key (middle of sequence)
        frame_keys = sorted(verts_dict.keys())
        if len(frame_keys) < 3:
            return "skipped"

        # Use the middle 30% of frames (likely contact frames)
        mid_start = len(frame_keys) // 3
        mid_end   = 2 * len(frame_keys) // 3
        mid_keys  = frame_keys[mid_start:mid_end]

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        H, W   = depth_m.shape

        depth_errors = []
        for fk in mid_keys[:10]:   # check up to 10 frames
            verts = verts_dict[fk]  # (778, 3) in camera space metres
            if verts is None or not hasattr(verts, '__len__'):
                continue
            verts = np.array(verts)
            # Project wrist joint (vertex 0) to image
            z_mano = float(verts[0, 2])
            if z_mano <= 0:
                continue
            u = int(fx * verts[0, 0] / z_mano + cx)
            v = int(fy * verts[0, 1] / z_mano + cy)
            if 0 <= u < W and 0 <= v < H:
                z_depth = float(depth_m[v, u])
                if z_depth > 0.05:
                    depth_errors.append(abs(z_mano - z_depth) / z_mano)

        if not depth_errors:
            return "skipped"

        mean_err = float(np.mean(depth_errors))
        return "consistent" if mean_err < 0.15 else "divergent"

    except Exception:
        return "skipped"


def read_source_frame(obj_dir):
    """Read source_frame.txt → (dataset, seq_id, frame_idx)"""
    p = os.path.join(obj_dir, "source_frame.txt")
    if not os.path.exists(p):
        return None, None, None
    lines = open(p).read().strip().splitlines()
    # Format written by sam2_annotate_by_object.py:
    # "dataset/seq_id  frame=N"  or just  "seq_id"
    for line in lines:
        parts = line.strip().split()
        ds_seq = parts[0]
        frame_idx = None
        for p_ in parts:
            if p_.startswith("frame="):
                frame_idx = int(p_.split("=")[1])
        if "/" in ds_seq:
            ds, seq = ds_seq.split("/", 1)
        else:
            ds, seq = None, ds_seq
        return ds, seq, frame_idx
    return None, None, None


# ── 主估算函数 ────────────────────────────────────────────────────────────────

def estimate_scale_for_object(model, transform, device, ds, obj_name, obj_dir,
                               mesh_path, redo=False):
    """
    Estimate scale_factor for one object.
    Returns dict with results, or None on failure.
    """
    # Always write scale.json to ProcessedData/obj_meshes/{ds}/{obj}/
    # (not inside SAM3DMesh, to keep outputs separate from source)
    out_dir  = os.path.join(MESH_BASE, ds, obj_name)
    out_json = os.path.join(out_dir, "scale.json")

    if not redo and os.path.exists(out_json):
        with open(out_json) as f:
            return json.load(f)

    img_path  = os.path.join(obj_dir, "image.png")
    mask_path = os.path.join(obj_dir, "0.png")

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return None

    # 1. Depth Pro on annotation image
    try:
        depth_m, K, H, W = run_depth_pro_on_image(model, transform, device, img_path)
    except Exception as e:
        print(f"    Depth Pro failed: {e}")
        return None

    # 2. Load mask, resize to depth resolution
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        return None
    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # 3. Back-project masked depth → 3D point cloud
    pts = depth_mask_to_pointcloud(depth_m, mask, K)
    if pts is None or len(pts) < 20:
        print(f"    ⚠️  Too few valid depth pixels in mask ({0 if pts is None else len(pts)})")
        return None

    d_real = pointcloud_diameter(pts)
    if d_real is None or d_real < 0.01:
        print(f"    ⚠️  Degenerate point cloud diameter: {d_real}")
        return None

    # 4. SAM3D mesh diameter
    try:
        d_mesh, _ = mesh_diameter(mesh_path)
    except Exception as e:
        print(f"    Mesh load failed: {e}")
        return None

    if d_mesh < 1e-6:
        return None

    scale_factor = d_real / d_mesh

    # 5. MANO cross-validation (use source sequence if available)
    src_ds, src_seq, _ = read_source_frame(obj_dir)
    if src_ds is None:
        src_ds = ds
    mano_result = "skipped"
    if src_seq:
        mano_result = mano_check(src_ds, src_seq, K, scale_factor, mask, depth_m)

    result = {
        "scale_factor": round(float(scale_factor), 6),
        "d_real_m":     round(float(d_real), 4),
        "d_mesh_raw":   round(float(d_mesh), 4),
        "n_pts":        int(len(pts)),
        "method":       "depth_mask",
        "mano_check":   mano_result,
        "obj":          obj_name,
        "dataset":      ds,
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Estimate SAM3D mesh scale from Depth Pro + mask")
    parser.add_argument("--dataset", default=None,
                        choices=["arctic", "oakink", "ho3d_v3", "dexycb", "ycb"])
    parser.add_argument("--obj",     default=None, help="Filter by object name substring")
    parser.add_argument("--redo",    action="store_true", help="Recompute existing scale.json")
    args = parser.parse_args()

    # Collect all (ds, obj_name, obj_dir, mesh_path)
    tasks = []
    datasets = [args.dataset] if args.dataset else \
               sorted(os.listdir(OBJ_INPUT)) if os.path.isdir(OBJ_INPUT) else []

    for ds in datasets:
        ds_dir = os.path.join(OBJ_INPUT, ds)
        if not os.path.isdir(ds_dir):
            continue
        # Map obj_recon_input/{ds} → obj_meshes/...
        # arctic → arctic, oakink → oakink, ho3d_v3 → ycb, dexycb → ycb
        if ds in ("ho3d_v3", "dexycb"):
            mesh_ds = "ycb"
        else:
            mesh_ds = ds

        for obj_name in sorted(os.listdir(ds_dir)):
            if args.obj and args.obj not in obj_name:
                continue
            obj_dir   = os.path.join(ds_dir, obj_name)
            if not os.path.isdir(obj_dir):
                continue
            # 1. SAM3DMesh (primary — SAM3D reconstructed meshes)
            mesh_path = os.path.join(SAM3D_MESH_BASE, mesh_ds, obj_name, "mesh.ply")
            if not os.path.exists(mesh_path):
                # 2. ProcessedData/obj_meshes (legacy)
                mesh_path = os.path.join(MESH_BASE, mesh_ds, obj_name, "mesh.ply")
                if not os.path.exists(mesh_path):
                    mesh_path_obj = mesh_path.replace(".ply", ".obj")
                    if os.path.exists(mesh_path_obj):
                        mesh_path = mesh_path_obj
                    else:
                        # 3. Try SAM3DMesh with original ds name (not remapped)
                        mesh_path_sam = os.path.join(SAM3D_MESH_BASE, ds, obj_name, "mesh.ply")
                        if os.path.exists(mesh_path_sam):
                            mesh_path = mesh_path_sam
                        else:
                            continue
            tasks.append((ds, obj_name, obj_dir, mesh_path))

    print(f"\n{'='*60}")
    print(f" Object Scale Estimation (Depth Pro + SAM2 Mask)")
    print(f" Objects : {len(tasks)}")
    print(f"{'='*60}\n")

    print("Loading Depth Pro model...")
    model, transform, device = load_depth_pro()
    print(f"✅ Model ready on {device}\n")

    done = skipped = failed = 0
    summary_rows = []

    for ds, obj_name, obj_dir, mesh_path in tqdm(tasks, desc="ScaleEst"):
        tqdm.write(f"  {ds}/{obj_name}")
        result = estimate_scale_for_object(
            model, transform, device,
            ds, obj_name, obj_dir, mesh_path, redo=args.redo)

        if result is None:
            tqdm.write(f"    ❌ failed")
            failed += 1
        elif result.get("scale_factor") == result.get("scale_factor"):  # not cached check
            sf   = result["scale_factor"]
            dr   = result["d_real_m"]
            dm   = result["d_mesh_raw"]
            mano = result.get("mano_check", "skipped")
            mano_icon = "✅" if mano == "consistent" else ("⚠️" if mano == "divergent" else "⏭")
            tqdm.write(f"    scale={sf:.4f}  real={dr:.3f}m  mesh={dm:.3f}  MANO:{mano_icon}")
            done += 1
            summary_rows.append(result)
        else:
            skipped += 1

        # Force GC to avoid GPU memory accumulation
        import torch
        torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ⏭ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"\n{'Dataset':<12} {'Object':<30} {'Scale':>8} {'RealDiam':>10} {'MANO':>12}")
    print("-" * 75)
    for r in sorted(summary_rows, key=lambda x: (x["dataset"], x["obj"])):
        print(f"{r['dataset']:<12} {r['obj']:<30} {r['scale_factor']:>8.4f}"
              f" {r['d_real_m']:>8.3f}m {r['mano_check']:>12}")

    # Save global summary
    out_summary = os.path.join(MESH_BASE, "scale_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSummary → {out_summary}")


if __name__ == "__main__":
    main()
