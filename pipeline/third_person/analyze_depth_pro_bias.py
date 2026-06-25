#!/usr/bin/env python3
"""
Depth Pro Systematic Bias Analysis
====================================
对4个第三人称数据集各随机取1条序列，分析 Depth Pro 估计的:
1. 内参 K (fx) — 与已知 GT K 对比
2. 深度分布 — 中位数深度、25/75百分位、逐帧变化
3. 帧间 fx 稳定性 — 看 Depth Pro 的 K 估计是否稳定

Usage:
    conda activate depth-pro
    python data/analyze_depth_pro_bias.py

    conda activate bundlesdf   # 如果只看已有 K.txt 不重跑
    python data/analyze_depth_pro_bias.py --cached-only
"""

import os, sys, argparse, json, random
import numpy as np
from glob import glob
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

DEPTH_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "third_depth")
RAW_BASE   = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")

# ── 已知 GT 内参 (1 px = 1px at native crop resolution used by each dataset) ──
# 注: 这些是文献/数据集官方公布的值，用于衡量 Depth Pro 系统性偏差
GT_K = {
    # ARCTIC cam1: raw 2000×2800 → 1000×1000 center-crop (no resize for square region)
    # misc.json intris_mat[1]: fx=4649 at 2000px → 1000px crop fx stays 4649 (crop, not resize)
    # But HaPTIC/Depth Pro use 1000×1000 images.
    # actual crop: looks like a 1000×1000 region cut from the center → fx same as full-res
    "arctic": {
        "fx_gt": 4649.0, "img_w": 2000, "img_h": 2800,
        "crop_w": 1000, "crop_h": 1000,
        "fx_gt_crop": 4649.0 * (1000 / 2000),   # if resize: fx * scale
        # NOTE: If ARCTIC provides 1000×1000 crops by CENTER CROP (no resize),
        # fx_crop = fx_raw. If downsample, fx_crop = fx_raw * (1000/2000).
        # From cx=500,cy=500 in K.txt (not 1058,1370), suggests CROP+RESIZE → use fx*0.5
        "note": "cam1, 1000×1000 crops from 2000×2800 raw (resize ×0.5 → fx_crop≈2325)",
    },
    # HO3D v3: official intrinsics from dataset paper
    # RGB camera: 640×480, K = [[617.343, 0, 312.42], [0, 617.343, 241.42], [0,0,1]]
    "ho3d_v3": {
        "fx_gt": 617.343, "img_w": 640, "img_h": 480,
        "crop_w": 640, "crop_h": 480,
        "fx_gt_crop": 617.343,
        "note": "Intel RealSense D435, 640×480",
    },
    # DexYCB: camera-specific, but one of the standard cameras: RealSense D415
    # 640×480, reported fx ≈ 605-610 px
    "dexycb": {
        "fx_gt": 607.3, "img_w": 640, "img_h": 480,
        "crop_w": 640, "crop_h": 480,
        "fx_gt_crop": 607.3,
        "note": "RealSense D415, 640×480, fx≈607",
    },
    # OakInk: custom camera rig with wide-angle lens
    # North-west camera, 848×480 typical
    "oakink": {
        "fx_gt": None,   # No single GT; varies per camera
        "note": "Unknown (multi-camera rig, no single GT published)",
    },
}


def load_per_frame_fx(seq_out_dir):
    """从 depths.npz 所在目录尝试加载逐帧 fx (如有 per_frame_fx.npy)。
    否则只返回中位数 K.txt 的值。
    """
    k_path = os.path.join(seq_out_dir, "K.txt")
    pf_path = os.path.join(seq_out_dir, "per_frame_fx.npy")
    if not os.path.exists(k_path):
        return None, None
    K = np.loadtxt(k_path)
    fx_median = float(K[0, 0])
    if os.path.exists(pf_path):
        per_frame = np.load(pf_path)
        return fx_median, per_frame
    return fx_median, None


def load_depths(seq_out_dir):
    """Load depths.npz → (N, H, W) array."""
    d_path = os.path.join(seq_out_dir, "depths.npz")
    if not os.path.exists(d_path):
        return None
    data = np.load(d_path)
    return data["depths"]   # (N, H, W)


def run_depth_pro_on_seq(seq_id, img_paths, max_frames=150, out_dir=None):
    """Run Depth Pro and save per-frame fx + depths. Returns (fx_list, depths)."""
    import torch, depth_pro
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = depth_pro.create_model_and_transforms(device=device, precision=torch.float16)
    model.eval()

    if len(img_paths) > max_frames:
        # Uniform sample
        idxs = np.linspace(0, len(img_paths) - 1, max_frames, dtype=int)
        img_paths = [img_paths[i] for i in idxs]

    fx_list, depth_list = [], []
    H_ref, W_ref = None, None

    from tqdm import tqdm
    for p in tqdm(img_paths, desc=f"  DepthPro {seq_id}", leave=False):
        image, _, f_px = depth_pro.load_rgb(p)
        image_t = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.infer(image_t, f_px=f_px)
        depth = pred["depth"].squeeze().cpu().float().numpy()
        focallen = float(pred["focallength_px"])
        fx_list.append(focallen)
        depth_list.append(depth)
        if H_ref is None:
            H_ref, W_ref = depth.shape

    depths = np.stack(depth_list).astype(np.float32)
    fx_arr = np.array(fx_list, dtype=np.float32)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "depths.npz"), depths=depths)
        np.save(os.path.join(out_dir, "per_frame_fx.npy"), fx_arr)
        fx_med = float(np.median(fx_arr))
        cx, cy = W_ref / 2.0, H_ref / 2.0
        K = np.array([[fx_med, 0, cx], [0, fx_med, cy], [0, 0, 1]], dtype=np.float32)
        np.savetxt(os.path.join(out_dir, "K.txt"), K, fmt="%.6f")
        frame_ids = [os.path.basename(p) for p in img_paths]
        with open(os.path.join(out_dir, "frame_ids.txt"), "w") as f:
            f.write("\n".join(frame_ids))

    return fx_arr, depths


def discover_seq_images(ds, seq_id):
    """Return list of image paths for a given sequence."""
    if ds == "arctic":
        subj, seq = seq_id.split("__", 1)
        cam_dir = os.path.join(RAW_BASE, "arctic", subj, seq, "1")
        imgs = natsorted(glob(os.path.join(cam_dir, "*.jpg")) +
                         glob(os.path.join(cam_dir, "*.png")))
    elif ds == "ho3d_v3":
        split, sid = seq_id.split("__", 1)
        rgb_dir = os.path.join(RAW_BASE, "ho3d_v3", split, sid, "rgb")
        imgs = natsorted(glob(os.path.join(rgb_dir, "*.jpg")))
    elif ds == "dexycb":
        parts = seq_id.split("__")
        cam_dir = os.path.join(RAW_BASE, "dexycb", *parts)
        imgs = natsorted(glob(os.path.join(cam_dir, "color_*.jpg")))
    elif ds == "oakink":
        seq_dir = os.path.join(RAW_BASE, "oakink_v1", seq_id)
        imgs = natsorted(glob(os.path.join(seq_dir, "*", "north_west_color_*.png")) +
                         glob(os.path.join(seq_dir, "north_west_color_*.png")))
    else:
        imgs = []
    return imgs


def analyze_one(ds, seq_id, cached_only=False):
    """Analyze Depth Pro output for one sequence. Returns dict of stats."""
    seq_out = os.path.join(DEPTH_BASE, ds, seq_id)
    fx_median, per_frame_fx = load_per_frame_fx(seq_out)
    depths = load_depths(seq_out)

    if fx_median is None or depths is None:
        if cached_only:
            return None
        # Run fresh
        imgs = discover_seq_images(ds, seq_id)
        if not imgs:
            print(f"  ⚠️  No images found for {ds}/{seq_id}")
            return None
        print(f"  Running Depth Pro on {ds}/{seq_id} ({len(imgs)} frames)...")
        per_frame_fx, depths = run_depth_pro_on_seq(seq_id, imgs, out_dir=seq_out)
        fx_median = float(np.median(per_frame_fx))

    # ── Depth statistics ────────────────────────────────────────────────────
    # Use center region (middle 50%) to avoid sky/background effects
    N, H, W = depths.shape
    h0, h1 = H // 4, 3 * H // 4
    w0, w1 = W // 4, 3 * W // 4
    center_depths = depths[:, h0:h1, w0:w1]   # (N, H/2, W/2)

    all_d = center_depths.flatten()
    all_d = all_d[(all_d > 0.05) & (all_d < 20.0)]   # valid range

    stats = {
        "ds": ds,
        "seq": seq_id,
        "n_frames": N,
        "img_hw": f"{H}×{W}",
        "fx_dp_median": fx_median,
        "fx_dp_min": float(np.min(per_frame_fx)) if per_frame_fx is not None else fx_median,
        "fx_dp_max": float(np.max(per_frame_fx)) if per_frame_fx is not None else fx_median,
        "fx_dp_std":  float(np.std(per_frame_fx))  if per_frame_fx is not None else 0.0,
        "depth_p10":  float(np.percentile(all_d, 10)),
        "depth_p25":  float(np.percentile(all_d, 25)),
        "depth_p50":  float(np.percentile(all_d, 50)),
        "depth_p75":  float(np.percentile(all_d, 75)),
        "depth_p90":  float(np.percentile(all_d, 90)),
    }

    # GT comparison
    gt = GT_K.get(ds, {})
    fx_gt = gt.get("fx_gt_crop")
    if fx_gt:
        stats["fx_gt"] = fx_gt
        stats["fx_ratio"] = fx_median / fx_gt      # >1: Depth Pro overestimates
        stats["depth_scale"] = fx_gt / fx_median   # corrective scale for depth
        # True depth ≈ DP depth × (fx_gt / fx_dp)  [if depth ∝ 1/fx]
        # (only valid if Depth Pro focal length is the main source of error)
    else:
        stats["fx_gt"] = None
        stats["fx_ratio"] = None
        stats["depth_scale"] = None

    return stats


def print_report(results):
    """Print formatted analysis table."""
    sep = "─" * 100
    print(f"\n{'='*100}")
    print(f"  Depth Pro Systematic Bias Analysis — {len(results)} datasets")
    print(f"{'='*100}")

    for r in results:
        if r is None:
            continue
        print(f"\n{sep}")
        print(f"  Dataset : {r['ds']}  |  Seq: {r['seq']}")
        print(f"  Frames  : {r['n_frames']}  |  Image: {r['img_hw']}px")
        print(sep)

        print(f"\n  Focal Length (fx) :")
        print(f"    Depth Pro median : {r['fx_dp_median']:>8.1f} px")
        if r["fx_gt"]:
            print(f"    Ground Truth     : {r['fx_gt']:>8.1f} px")
            ratio = r["fx_ratio"]
            direction = "⬆ over" if ratio > 1 else "⬇ under"
            print(f"    Ratio (DP/GT)    : {ratio:>8.3f}  ({direction}estimate by {abs(ratio-1)*100:.1f}%)")
            print(f"    Depth corr factor: {r['depth_scale']:>8.3f}  (multiply DP depth by this for true scale)")
        else:
            print(f"    Ground Truth     :      N/A  ({GT_K.get(r['ds'],{}).get('note','')})")
        if r["fx_dp_std"] > 0:
            print(f"    Per-frame std    : ±{r['fx_dp_std']:>6.1f} px  "
                  f"[{r['fx_dp_min']:.0f} ~ {r['fx_dp_max']:.0f}]")

        print(f"\n  Depth Distribution (center 50% of image, valid 0.05-20m):")
        print(f"    P10  P25  P50  P75  P90  (metres)")
        print(f"    {r['depth_p10']:.2f}  {r['depth_p25']:.2f}  {r['depth_p50']:.2f}  "
              f"{r['depth_p75']:.2f}  {r['depth_p90']:.2f}")
        if r["fx_gt"] and r["depth_scale"]:
            s = r["depth_scale"]
            print(f"    After GT-K correction (×{s:.2f}):")
            print(f"    {r['depth_p10']*s:.2f}  {r['depth_p25']*s:.2f}  {r['depth_p50']*s:.2f}  "
                  f"{r['depth_p75']*s:.2f}  {r['depth_p90']*s:.2f}")

        note = GT_K.get(r["ds"], {}).get("note", "")
        if note:
            print(f"\n  Note: {note}")

    print(f"\n{'='*100}")
    print("  Summary: Depth Pro systematic fx bias across datasets")
    print(f"  {'Dataset':<12}  {'DP fx':>8}  {'GT fx':>8}  {'Ratio':>7}  {'Depth P50':>10}  {'Corrected P50':>14}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*14}")
    for r in results:
        if r is None:
            continue
        fx_gt_str = f"{r['fx_gt']:.0f}" if r["fx_gt"] else "N/A"
        ratio_str = f"{r['fx_ratio']:.3f}" if r["fx_ratio"] else "N/A"
        corr_str  = (f"{r['depth_p50']*r['depth_scale']:.2f}m"
                     if r["depth_scale"] else "N/A")
        print(f"  {r['ds']:<12}  {r['fx_dp_median']:>8.0f}  {fx_gt_str:>8}  "
              f"{ratio_str:>7}  {r['depth_p50']:>9.2f}m  {corr_str:>14}")
    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-only", action="store_true",
                        help="Only use existing K.txt/depths.npz, don't run Depth Pro")
    parser.add_argument("--ds", default=None, help="Only analyze this dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for seq selection")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    datasets = ["arctic", "ho3d_v3", "dexycb", "oakink"]
    if args.ds:
        datasets = [args.ds]

    results = []
    for ds in datasets:
        ds_dir = os.path.join(DEPTH_BASE, ds)
        if not os.path.isdir(ds_dir):
            print(f"\n⚠️  No cached data for {ds}, skipping")
            continue

        seqs = natsorted([d for d in os.listdir(ds_dir)
                          if os.path.isdir(os.path.join(ds_dir, d))])
        if not seqs:
            print(f"\n⚠️  No sequences in {ds_dir}")
            continue

        # Pick random sequence (or first if only one)
        seq_id = random.choice(seqs)
        print(f"\n[{ds}] Randomly selected: {seq_id}")

        r = analyze_one(ds, seq_id, cached_only=args.cached_only)
        results.append(r)

    print_report(results)


if __name__ == "__main__":
    main()
