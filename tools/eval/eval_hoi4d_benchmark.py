#!/usr/bin/env python3
"""
eval_hoi4d_benchmark.py
========================
对 HOI4D 随机抽取 N 条序列，分别跑 MegaSAM 和 ViPE，
对比 GT 内参 + 深度，并分析误差与相机运动距离的相关性。

用法：
  # 先 MegaSAM（mega_sam 环境）
  conda activate mega_sam
  export XFORMERS_DISABLED=1
  python pipeline/ego/eval_hoi4d_benchmark.py --phase megasam --n 50

  # 再 ViPE（biv2ap 环境）
  conda activate biv2ap
  export XFORMERS_DISABLED=1
  python pipeline/ego/eval_hoi4d_benchmark.py --phase vipe --n 50

  # 最后分析出图（base 环境）
  python pipeline/ego/eval_hoi4d_benchmark.py --phase analyze

  # 单步全跑（不推荐，需要手动切环境）
  python pipeline/ego/eval_hoi4d_benchmark.py --phase all --n 50
"""

import os, sys, json, argparse, random, subprocess, shutil, time
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT      = Path(__file__).resolve().parent.parent.parent   # V2AP/
HOI4D_RGB    = PROJECT / "data/egocentric/hoi4d/HOI4D_release"
HOI4D_DEPTH  = PROJECT / "data/egocentric/hoi4d/HOI4D_depth_video"
HOI4D_INTRIN = PROJECT / "data/egocentric/hoi4d/camera_params/Egocentric_Camera_Parameters"

MEGASAM_DIR  = PROJECT / "thirdparty/megasam"
VIPE_DIR     = Path("/home/lyh/Project/Bi-V2AP/third_party/vipe")
VIPE_OUTBASE = Path("/home/lyh/Project/Bi-V2AP/Output/PipelineOutput/HOI4D")

OUT_ROOT     = PROJECT / "output/eval/hoi4d_benchmark"
RESULTS_JSON = OUT_ROOT / "results.json"
SEQS_JSON    = OUT_ROOT / "sampled_sequences.json"

SEED = 42
N_FRAMES_MEGASAM = 60   # 每条序列送给 MegaSAM 的帧数
STRIDE = 5              # 原始 300 帧 / stride = 60 帧


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def all_sequences():
    """列出 HOI4D_release 中所有有 RGB 和 depth 的序列路径"""
    seqs = []
    for rgb_mp4 in sorted(HOI4D_RGB.rglob("align_rgb/image.mp4")):
        # rgb_mp4: HOI4D_release/ZY.../H.../align_rgb/image.mp4
        rel = rgb_mp4.relative_to(HOI4D_RGB)          # ZY.../H.../align_rgb/image.mp4
        seq_rel = rel.parent.parent                    # ZY.../H.../
        depth_avi = HOI4D_DEPTH / seq_rel / "align_depth/depth_video.avi"
        if depth_avi.exists():
            seqs.append(str(seq_rel))
    return seqs


def sample_sequences(n: int):
    all_seqs = all_sequences()
    random.seed(SEED)
    chosen = random.sample(all_seqs, min(n, len(all_seqs)))
    return sorted(chosen)


def gt_intrinsic_for_seq(seq_rel: str):
    """
    HOI4D GT 内参：camera_params 按 (triplet)/(session) 组织，
    与 ZY/H/C/N/S/s/T 路径无直接映射。
    近似处理：取所有 egocentric_intrinsic.txt 的 fx 中位数作为参考。
    后续可替换为精确的 per-device 标定文件。
    """
    intrin_files = list(HOI4D_INTRIN.rglob("egocentric_intrinsic.txt"))
    Ks = [np.loadtxt(f) for f in intrin_files]
    # 取中位数 K
    fxs = np.array([K[0,0] for K in Ks])
    fys = np.array([K[1,1] for K in Ks])
    cxs = np.array([K[0,2] for K in Ks])
    cys = np.array([K[1,2] for K in Ks])
    K_med = np.eye(3)
    K_med[0,0] = np.median(fxs)
    K_med[1,1] = np.median(fys)
    K_med[0,2] = np.median(cxs)
    K_med[1,2] = np.median(cys)
    return K_med


def extract_frames(rgb_mp4: Path, out_dir: Path, stride: int = STRIDE, max_frames: int = N_FRAMES_MEGASAM):
    """从 MP4 均匀抽帧到 out_dir/*.png"""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.png"))
    if len(existing) >= max_frames:
        return len(existing)
    cap = cv2.VideoCapture(str(rgb_mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    for i in range(0, min(total, max_frames * stride), stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(out_dir / f"{saved:06d}.png"), frame)
        saved += 1
        if saved >= max_frames:
            break
    cap.release()
    return saved


def load_depth_gt(depth_avi: Path, stride: int = STRIDE, max_frames: int = N_FRAMES_MEGASAM):
    """读 GT 深度 AVI，返回 (N, H, W) float32 [m]"""
    cap = cv2.VideoCapture(str(depth_avi))
    frames = []
    for i in range(0, min(300, max_frames * stride), stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if not ret:
            break
        d_mm = f[:,:,0].astype(np.uint16)*256 + f[:,:,1].astype(np.uint16)
        frames.append(d_mm.astype(np.float32)/1000.0)
    cap.release()
    return np.array(frames)


def absrel_scale_aligned(d_pred: np.ndarray, d_gt: np.ndarray):
    """Scale-aligned AbsRel，忽略 d_gt <= 0.1m 的像素"""
    v = d_gt > 0.1
    if v.sum() < 200:
        return None
    scale = np.median(d_gt[v]) / (np.median(d_pred[v]) + 1e-8)
    return float(np.mean(np.abs(d_pred[v] * scale - d_gt[v]) / d_gt[v]))


def traj_distance(cam_c2w: np.ndarray):
    """cam_c2w: (N,4,4) → 轨迹总位移 [m]"""
    t = cam_c2w[:, :3, 3]
    return float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum())


def load_results():
    if RESULTS_JSON.exists():
        return json.loads(RESULTS_JSON.read_text())
    return {}


def save_results(results: dict):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: MegaSAM
# ══════════════════════════════════════════════════════════════════════════════

def run_megasam_on_seq(seq_rel: str, results: dict):
    seq_id = seq_rel.replace("/", "_")
    if seq_id in results and "megasam" in results[seq_id] and results[seq_id]["megasam"].get("done"):
        print(f"  [skip] {seq_id} MegaSAM already done")
        return

    rgb_mp4   = HOI4D_RGB   / seq_rel / "align_rgb/image.mp4"
    depth_avi = HOI4D_DEPTH / seq_rel / "align_depth/depth_video.avi"
    work_dir  = OUT_ROOT / "megasam_work" / seq_id

    frames_dir     = work_dir / "frames"
    mono_depth_dir = work_dir / "mono_depth" / seq_id
    unidepth_dir   = work_dir / "unidepth_out"
    recon_name     = seq_id

    try:
        # Step 1: extract frames
        n = extract_frames(rgb_mp4, frames_dir)
        print(f"  frames: {n}")

        # Step 2: DepthAnything
        mono_depth_dir.mkdir(parents=True, exist_ok=True)
        if not any(mono_depth_dir.glob("*.npy")):
            subprocess.run([
                "python", "Depth-Anything/run_videos.py",
                "--encoder", "vitl",
                "--load-from", "Depth-Anything/checkpoints/depth_anything_vitl14.pth",
                "--img-path", str(frames_dir),
                "--outdir", str(work_dir / "mono_depth"),
            ], cwd=MEGASAM_DIR, check=True, env={**os.environ, "XFORMERS_DISABLED": "1"})
            # move into scene subdir
            for f in (work_dir / "mono_depth").glob("*.npy"):
                shutil.move(str(f), str(mono_depth_dir / f.name))

        # Step 3: UniDepth
        unidepth_scene = unidepth_dir / seq_id
        unidepth_scene.mkdir(parents=True, exist_ok=True)
        if not any(unidepth_scene.glob("*.npz")):
            subprocess.run([
                "python", "UniDepth/scripts/demo_mega-sam.py",
                "--scene-name", seq_id,
                "--img-path", str(frames_dir),
                "--outdir", str(unidepth_dir),
            ], cwd=MEGASAM_DIR, check=True,
               env={**os.environ, "XFORMERS_DISABLED": "1",
                    "PYTHONPATH": str(MEGASAM_DIR / "UniDepth")})

        # Step 4: DROID-SLAM
        recon_path = MEGASAM_DIR / "reconstructions" / recon_name
        npz_path   = MEGASAM_DIR / "outputs" / f"{recon_name}_droid.npz"
        if not npz_path.exists():
            subprocess.run([
                "python", "camera_tracking_scripts/test_demo.py",
                "--datapath", str(frames_dir),
                "--scene_name", recon_name,
                "--weights", "checkpoints/megasam_final.pth",
                "--mono_depth_path", str(work_dir / "mono_depth"),
                "--metric_depth_path", str(unidepth_dir),
                "--disable_vis", "--buffer", "256",
            ], cwd=MEGASAM_DIR, check=True,
               env={**os.environ, "XFORMERS_DISABLED": "1",
                    "PYTHONPATH": str(MEGASAM_DIR / "base" / "droid_slam")})

        # Parse results
        npz = np.load(npz_path)
        K_ms  = npz["intrinsic"]
        H_sl, W_sl = npz["images"].shape[1], npz["images"].shape[2]
        fx_ms = K_ms[0,0] * (1920 / W_sl)
        fy_ms = K_ms[1,1] * (1080 / H_sl)
        cx_ms = K_ms[0,2] * (1920 / W_sl)
        cy_ms = K_ms[1,2] * (1080 / H_sl)
        cam_c2w = npz["cam_c2w"]  # (N,4,4)
        dist_ms = traj_distance(cam_c2w)

        # Depth AbsRel
        depth_gt = load_depth_gt(depth_avi)
        depths_ms = npz["depths"]
        ars = []
        for i in range(min(len(depths_ms), len(depth_gt))):
            d = cv2.resize(depths_ms[i], (1920, 1080), interpolation=cv2.INTER_LINEAR)
            ar = absrel_scale_aligned(d, depth_gt[i])
            if ar is not None:
                ars.append(ar)

        if seq_id not in results:
            results[seq_id] = {}
        results[seq_id]["megasam"] = {
            "done": True,
            "fx": fx_ms, "fy": fy_ms, "cx": cx_ms, "cy": cy_ms,
            "traj_dist_m": dist_ms,
            "absrel_depth": float(np.mean(ars)) if ars else None,
            "n_frames": int(cam_c2w.shape[0]),
        }
        save_results(results)
        print(f"  MegaSAM done: fx={fx_ms:.0f} dist={dist_ms:.3f}m absrel={np.mean(ars):.4f}")

    except Exception as e:
        print(f"  MegaSAM FAILED on {seq_id}: {e}")
        if seq_id not in results:
            results[seq_id] = {}
        results[seq_id].setdefault("megasam", {})["error"] = str(e)
        save_results(results)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: ViPE  (使用 Bi-V2AP 的 EgoPipeline / ViPEStage)
# ══════════════════════════════════════════════════════════════════════════════

BIV2AP_DIR = Path("/home/lyh/Project/Bi-V2AP")

def run_vipe_on_seq(seq_rel: str, results: dict):
    seq_id = seq_rel.replace("/", "_")
    if seq_id in results and "vipe" in results[seq_id] and results[seq_id]["vipe"].get("done"):
        print(f"  [skip] {seq_id} ViPE already done")
        return

    rgb_mp4   = HOI4D_RGB   / seq_rel / "align_rgb/image.mp4"
    depth_avi = HOI4D_DEPTH / seq_rel / "align_depth/depth_video.avi"

    # Prefer existing Bi-V2AP output; write new results to V2AP output dir
    biv2ap_out = VIPE_OUTBASE / seq_rel
    local_out  = OUT_ROOT / "vipe_out" / seq_rel

    def find_output(base):
        k = base / "K.npy"; d = base / "depth.npz"; c = base / "cam_c2w.npy"
        return (k, d, c) if (k.exists() and d.exists() and c.exists()) else None

    found = find_output(biv2ap_out) or find_output(local_out)

    try:
        if found is None:
            # Run ViPE in an isolated subprocess — avoids CUDA context reuse errors
            local_out.mkdir(parents=True, exist_ok=True)
            worker_script = Path(__file__).parent / "_vipe_worker.py"
            result = subprocess.run(
                [sys.executable, str(worker_script),
                 "--video", str(rgb_mp4),
                 "--outdir", str(local_out),
                 "--biv2ap", str(BIV2AP_DIR)],
                timeout=900,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ViPE worker exited with code {result.returncode}")
            found = find_output(local_out)

        K_npy, depth_npz, cam_npy = found

        # 读取结果
        K_vipe  = np.load(str(K_npy))
        depths  = np.load(str(depth_npz))["depths"]
        cam_c2w = np.load(str(cam_npy))

        fx_v   = float(K_vipe[0,0]); fy_v = float(K_vipe[1,1])
        cx_v   = float(K_vipe[0,2]); cy_v = float(K_vipe[1,2])
        dist_v = traj_distance(cam_c2w)

        depth_gt = load_depth_gt(depth_avi)
        ars = []
        stride_v = max(1, depths.shape[0] // N_FRAMES_MEGASAM)
        for i in range(min(N_FRAMES_MEGASAM, depths.shape[0] // stride_v)):
            d = depths[i * stride_v]
            if d.shape != (1080, 1920):
                d = cv2.resize(d, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            ar = absrel_scale_aligned(d, depth_gt[min(i, len(depth_gt)-1)])
            if ar is not None:
                ars.append(ar)

        if seq_id not in results:
            results[seq_id] = {}
        results[seq_id]["vipe"] = {
            "done": True,
            "fx": fx_v, "fy": fy_v, "cx": cx_v, "cy": cy_v,
            "traj_dist_m": dist_v,
            "absrel_depth": float(np.mean(ars)) if ars else None,
            "n_frames": int(cam_c2w.shape[0]),
        }
        save_results(results)
        print(f"  ViPE done: fx={fx_v:.0f} dist={dist_v:.3f}m absrel={np.mean(ars):.4f}")

    except Exception as e:
        import traceback
        print(f"  ViPE FAILED on {seq_id}: {e}")
        traceback.print_exc()
        if seq_id not in results:
            results[seq_id] = {}
        results[seq_id].setdefault("vipe", {})["error"] = str(e)
        save_results(results)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    K_gt = gt_intrinsic_for_seq("")
    fx_gt = K_gt[0,0]

    ms_data, vipe_data = [], []

    for seq_id, r in results.items():
        for method, data in [("megasam", r.get("megasam", {})),
                              ("vipe",    r.get("vipe",    {}))]:
            if not data.get("done"):
                continue
            fx_err_pct = abs(data["fx"] - fx_gt) / fx_gt * 100
            dist_m     = data["traj_dist_m"]
            absrel     = data.get("absrel_depth")
            entry = {"seq": seq_id, "fx_err_pct": fx_err_pct,
                     "dist_m": dist_m, "absrel": absrel}
            if method == "megasam":
                ms_data.append(entry)
            else:
                vipe_data.append(entry)

    def extract(data, key):
        return np.array([d[key] for d in data if d[key] is not None])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"MegaSAM vs ViPE — HOI4D Benchmark  (GT fx~{fx_gt:.0f}px, {len(results)} seqs)",
                 fontsize=13)

    colors = {"megasam": "#2196F3", "vipe": "#FF5722"}

    for ax_row, (method, data, color) in enumerate([
            ("MegaSAM", ms_data, colors["megasam"]),
            ("ViPE",    vipe_data, colors["vipe"])]):

        dist  = extract(data, "dist_m")
        fxerr = extract(data, "fx_err_pct")
        absr  = extract(data, "absrel")

        # Plot 1: dist vs fx_err
        ax = axes[ax_row][0]
        ax.scatter(dist, fxerr, alpha=0.7, color=color, s=40)
        if len(dist) > 2:
            r, p = stats.pearsonr(dist, fxerr)
            m, b, *_ = stats.linregress(dist, fxerr)
            xs = np.linspace(dist.min(), dist.max(), 100)
            ax.plot(xs, m*xs + b, "--", color=color, linewidth=1.5)
            ax.set_title(f"{method}: traj dist vs fx error\nPearson r={r:.3f}  p={p:.3f}",
                         fontsize=10)
        ax.set_xlabel("Camera trajectory (m)")
        ax.set_ylabel("fx relative error (%)")
        ax.grid(True, alpha=0.3)
        if len(fxerr):
            ax.axhline(np.mean(fxerr), color=color, linestyle=":", alpha=0.5,
                       label=f"mean {np.mean(fxerr):.1f}%")
            ax.legend(fontsize=8)

        # Plot 2: dist vs absrel depth
        ax = axes[ax_row][1]
        valid = [(d, a) for d, a in zip(
                    extract(data, "dist_m"), extract(data, "absrel")) if a is not None]
        if valid:
            dv, av = zip(*valid)
            dv, av = np.array(dv), np.array(av)
            ax.scatter(dv, av, alpha=0.7, color=color, s=40)
            if len(dv) > 2:
                r2, p2 = stats.pearsonr(dv, av)
                m2, b2, *_ = stats.linregress(dv, av)
                xs2 = np.linspace(dv.min(), dv.max(), 100)
                ax.plot(xs2, m2*xs2 + b2, "--", color=color, linewidth=1.5)
                ax.set_title(f"{method}: traj dist vs depth AbsRel\nPearson r={r2:.3f}  p={p2:.3f}",
                             fontsize=10)
            ax.set_xlabel("Camera trajectory (m)")
            ax.set_ylabel("Depth AbsRel (scale-aligned)")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUT_ROOT / "benchmark_correlation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")

    # 打印统计表
    print("\n" + "="*65)
    print(f"{'':20s} {'MegaSAM':>20} {'ViPE':>20}")
    print("-"*65)

    def stats_str(data, key):
        vals = extract(data, key)
        if len(vals) == 0:
            return f"{'N/A':>20}"
        return f"{np.mean(vals):>10.3f} +/- {np.std(vals):<8.3f}"

    for label, key in [("fx error (%)",   "fx_err_pct"),
                        ("depth AbsRel",   "absrel"),
                        ("traj dist (m)",  "dist_m")]:
        print(f"{label:20s} {stats_str(ms_data, key)} {stats_str(vipe_data, key)}")

    print(f"\n{'N sequences':20s} {len(ms_data):>20} {len(vipe_data):>20}")

    # 相关性摘要
    print("\nCorrelation: traj distance vs error")
    for method, data in [("MegaSAM", ms_data), ("ViPE", vipe_data)]:
        dist  = extract(data, "dist_m")
        fxerr = extract(data, "fx_err_pct")
        if len(dist) > 2:
            r, p = stats.pearsonr(dist, fxerr)
            sig = "significant" if p < 0.05 else "not significant"
            print(f"  {method}: fx error vs dist  r={r:+.3f}  p={p:.4f}  ({sig})")
        absr  = extract(data, "absrel")
        dist2 = dist[:len(absr)]
        if len(dist2) > 2:
            r2, p2 = stats.pearsonr(dist2, absr)
            sig2 = "significant" if p2 < 0.05 else "not significant"
            print(f"  {method}: depth err vs dist  r={r2:+.3f}  p={p2:.4f}  ({sig2})")

    return fig_path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["megasam", "vipe", "analyze", "all", "sample"],
                        default="sample")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seq-offset", type=int, default=0, help="从第几条序列开始（断点续跑）")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 采样（固定 seed，保证 megasam/vipe/analyze 用同一批）
    if SEQS_JSON.exists():
        seqs = json.loads(SEQS_JSON.read_text())
    else:
        print(f"采样 {args.n} 条序列...")
        seqs = sample_sequences(args.n)
        SEQS_JSON.write_text(json.dumps(seqs, indent=2))
        print(f"已保存到 {SEQS_JSON}")

    print(f"共 {len(seqs)} 条序列")

    results = load_results()

    if args.phase in ("megasam", "all"):
        print(f"\n[Phase 1] MegaSAM on {len(seqs)} sequences")
        for i, seq in enumerate(tqdm(seqs[args.seq_offset:], desc="MegaSAM")):
            print(f"\n[{i+1+args.seq_offset}/{len(seqs)}] {seq}")
            run_megasam_on_seq(seq, results)

    if args.phase in ("vipe", "all"):
        print(f"\n[Phase 2] ViPE on {len(seqs)} sequences")
        for i, seq in enumerate(tqdm(seqs[args.seq_offset:], desc="ViPE")):
            print(f"\n[{i+1+args.seq_offset}/{len(seqs)}] {seq}")
            run_vipe_on_seq(seq, results)

    if args.phase in ("analyze", "all"):
        print("\n[Phase 3] Analysis")
        if len(results) == 0:
            print("No results yet. Run --phase megasam or --phase vipe first.")
            return
        fig_path = analyze(results)
        print(f"Done! See: {fig_path}")

    if args.phase == "sample":
        print("序列列表已生成：")
        for s in seqs[:5]:
            print(f"  {s}")
        print(f"  ... ({len(seqs)} total)")
        print(f"\n运行步骤：")
        print(f"  conda activate mega_sam && export XFORMERS_DISABLED=1")
        print(f"  python {__file__} --phase megasam --n {args.n}")
        print(f"  conda activate biv2ap  && export XFORMERS_DISABLED=1")
        print(f"  python {__file__} --phase vipe --n {args.n}")
        print(f"  python {__file__} --phase analyze")


if __name__ == "__main__":
    main()
