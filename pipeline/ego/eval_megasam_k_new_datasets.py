#!/usr/bin/env python3
"""
eval_megasam_k_new_datasets.py
================================
随机抽取 HOI4D 和 TACO Ego 各 10 条序列，
用 MegaSAM 估计内参 K，与 GT 对比误差。

运行：
    cd $PROJ/mega-sam
    conda run -n mega_sam python ../data/eval_megasam_k_new_datasets.py
"""

import os, sys, json, random, subprocess, tempfile, shutil
import numpy as np
from glob import glob
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

# ── Dataset paths ──────────────────────────────────────────────────────────────
HOI4D_VIDEO_BASE  = PROJ / "data_hub/RawData/EgoRawData/hoi4d/HOI4D_release"
HOI4D_INTRIN_BASE = PROJ / "data_hub/RawData/EgoRawData/hoi4d/camera_params"
TACO_VIDEO_BASE   = PROJ / "data_hub/RawData/EgoRawData/taco/Egocentric_RGB_Videos"
TACO_INTRIN_BASE  = PROJ / "data_hub/RawData/EgoRawData/taco/Egocentric_Camera_Parameters"

# ── MegaSAM runner (must run from mega-sam/ dir) ───────────────────────────────
MEGASAM_DIR = PROJ / "mega-sam"
MEGASAM_SCRIPT = MEGASAM_DIR / "run_megasam.py"   # adjust if different

N_FRAMES = 60   # subsampled frames to feed MegaSAM
N_SAMPLE = 10   # sequences per dataset


# ── GT loaders ────────────────────────────────────────────────────────────────
def load_hoi4d_gt_K(video_path: Path) -> np.ndarray:
    """Load GT 3x3 K from camera_params/ZY*/intrin.npy."""
    # video: HOI4D_release/ZY20210800001/H*/...
    device_id = video_path.parts[video_path.parts.index("HOI4D_release") + 1]
    intrin_path = HOI4D_INTRIN_BASE / device_id / "intrin.npy"
    return np.load(intrin_path).reshape(3, 3).astype(np.float32)


def load_taco_gt_K(video_path: Path) -> np.ndarray:
    """Load GT K from Egocentric_Camera_Parameters/{triplet}/{seq}/egocentric_intrinsic.txt."""
    # video: Egocentric_RGB_Videos/{triplet}/{seq}/color.mp4
    triplet = video_path.parts[-3]
    seq     = video_path.parts[-2]
    intrin_path = TACO_INTRIN_BASE / triplet / seq / "egocentric_intrinsic.txt"
    return np.loadtxt(intrin_path).reshape(3, 3).astype(np.float32)


# ── Frame extraction ──────────────────────────────────────────────────────────
def extract_frames(video_path: Path, out_dir: Path, n_frames: int = 60):
    """Uniformly subsample n_frames from mp4 → out_dir/%05d.png"""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Get total frame count
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True, text=True
    )
    info = json.loads(r.stdout)
    total = None
    for s in info["streams"]:
        if s.get("codec_type") == "video":
            total = int(s.get("nb_frames", 0))
            if total == 0:
                dur = float(s.get("duration", 10))
                fps_str = s.get("r_frame_rate", "30/1")
                num, den = fps_str.split("/")
                total = int(dur * int(num) / int(den))
            break
    step = max(1, total // n_frames)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"select='not(mod(n\\,{step}))',setpts=N/FRAME_RATE/TB",
        "-vframes", str(n_frames),
        "-q:v", "2",
        str(out_dir / "%05d.png")
    ], capture_output=True)


# ── MegaSAM runner ────────────────────────────────────────────────────────────
def run_megasam(frame_dir: Path, out_dir: Path) -> np.ndarray | None:
    """Run MegaSAM on extracted frames, return estimated 3x3 K or None."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(MEGASAM_SCRIPT),
        "--image_dir", str(frame_dir),
        "--output_dir", str(out_dir),
        "--no_viz",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(MEGASAM_DIR))
    if r.returncode != 0:
        print(f"    MegaSAM failed: {r.stderr[-300:]}")
        return None
    # Load K from output
    k_path = out_dir / "K.npy"
    if k_path.exists():
        return np.load(k_path).reshape(3, 3).astype(np.float32)
    return None


# ── Error computation ─────────────────────────────────────────────────────────
def fx_error_pct(K_est: np.ndarray, K_gt: np.ndarray) -> float:
    """Percentage error on fx (estimated vs GT, at native resolution)."""
    # K_est may be at a different resolution - we compare fx/cx ratio (relative)
    fx_est = K_est[0, 0]
    cx_est = K_est[0, 2]
    fx_gt  = K_gt[0, 0]
    cx_gt  = K_gt[0, 2]
    # Scale K_est to GT image width
    scale  = cx_gt / cx_est if cx_est > 0 else 1.0
    fx_est_scaled = fx_est * scale
    return (fx_est_scaled - fx_gt) / fx_gt * 100.0


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate_dataset(name: str, video_list: list, gt_loader, n=10):
    print(f"\n{'='*60}")
    print(f" {name}  (randomly sampling {n} sequences)")
    print(f"{'='*60}")

    random.seed(42)
    samples = random.sample(video_list, min(n, len(video_list)))

    results = []
    for i, vid in enumerate(samples):
        vid = Path(vid)
        print(f"\n[{i+1}/{n}] {vid.parent.name}/{vid.name}")

        # GT
        try:
            K_gt = gt_loader(vid)
            print(f"  GT   fx={K_gt[0,0]:.1f}  fy={K_gt[1,1]:.1f}")
        except Exception as e:
            print(f"  ❌ GT load failed: {e}")
            continue

        # Extract frames
        with tempfile.TemporaryDirectory() as tmp:
            frame_dir = Path(tmp) / "frames"
            mega_out  = Path(tmp) / "megasam_out"
            extract_frames(vid, frame_dir, N_FRAMES)
            n_extracted = len(list(frame_dir.glob("*.png")))
            print(f"  Extracted {n_extracted} frames")
            if n_extracted < 10:
                print("  ⚠ Too few frames, skip")
                continue

            # MegaSAM
            K_est = run_megasam(frame_dir, mega_out)
            if K_est is None:
                print("  ❌ MegaSAM failed")
                continue

            err = fx_error_pct(K_est, K_gt)
            print(f"  EST  fx={K_est[0,0]:.1f} (scaled fx≈{K_est[0,0]*K_gt[0,2]/K_est[0,2]:.1f})")
            print(f"  ERR  {err:+.1f}%")
            results.append({"seq": str(vid), "fx_gt": float(K_gt[0,0]),
                            "fx_err": err, "K_est": K_est.tolist(), "K_gt": K_gt.tolist()})

    # Summary
    print(f"\n{'─'*40}")
    print(f" {name} Summary ({len(results)}/{n} succeeded)")
    if results:
        errs = [r["fx_err"] for r in results]
        print(f"  mean |err| = {np.mean(np.abs(errs)):.1f}%")
        print(f"  max  |err| = {np.max(np.abs(errs)):.1f}%")
        print(f"  ≤5%  count = {sum(abs(e)<=5 for e in errs)}/{len(errs)}")
        print(f"  ≤10% count = {sum(abs(e)<=10 for e in errs)}/{len(errs)}")
        for r in results:
            print(f"    {Path(r['seq']).parent.name:30s}  {r['fx_err']:+6.1f}%")
    return results


def main():
    # ── Collect video paths ────────────────────────────────────────────────────
    hoi4d_vids = glob(str(HOI4D_VIDEO_BASE / "**" / "image.mp4"), recursive=True)
    taco_vids  = glob(str(TACO_VIDEO_BASE  / "**" / "color.mp4"),  recursive=True)
    print(f"HOI4D videos found: {len(hoi4d_vids)}")
    print(f"TACO Ego videos found: {len(taco_vids)}")

    all_results = {}
    all_results["hoi4d"] = evaluate_dataset("HOI4D", hoi4d_vids, load_hoi4d_gt_K, N_SAMPLE)
    all_results["taco"]  = evaluate_dataset("TACO Ego", taco_vids, load_taco_gt_K, N_SAMPLE)

    # Save results
    out = PROJ / "output" / "megasam_k_eval_new_datasets.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved → {out}")


if __name__ == "__main__":
    main()
