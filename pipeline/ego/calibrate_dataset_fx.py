#!/usr/bin/env python3
"""
calibrate_dataset_fx.py
========================
One-time automatic intrinsic (fx) calibration for egocentric datasets.

Pipeline (fully automatic, no GT):
  1. Sample N sequences from dataset
  2. Run DA + UniDepth + DROID-SLAM (opt-intr) on each
  3. Filter outliers (|fx - median| > 2sigma)
  4. Apply SLAM_INTR_CORRECTION = 1.10 (systematic -10% bias)
  5. Write result to batch_megasam.py CALIB_FX automatically

Usage (from mega-sam/ directory, mega_sam conda env):
    # TACO Ego
    conda run --no-capture-output -n mega_sam \\
      python -u ../data/calibrate_dataset_fx.py --dataset taco --n 15 \\
      2>&1 | tee ../output/calib_taco.log

    # EgoDex
    conda run --no-capture-output -n mega_sam \\
      python -u ../data/calibrate_dataset_fx.py --dataset egodex --n 10 \\
      2>&1 | tee ../output/calib_egodex.log

    # HOI4D (low motion - try larger n)
    conda run --no-capture-output -n mega_sam \\
      python -u ../data/calibrate_dataset_fx.py --dataset hoi4d --n 50 \\
      2>&1 | tee ../output/calib_hoi4d.log
"""

import os, sys, json, argparse, random, re, tempfile, subprocess
import numpy as np
from pathlib import Path
from glob import glob
from natsort import natsorted

PROJ        = Path(__file__).resolve().parent.parent
MEGASAM_DIR = PROJ / "mega-sam"
sys.path.insert(0, str(MEGASAM_DIR))
sys.path.insert(0, str(MEGASAM_DIR / "UniDepth"))

import cv2
import torch

# ── Shared config (must match batch_megasam.py) ───────────────────────────────
UNIDEPTH_ID  = "lpiccinelli/unidepth-v2-vitl14"
UNIDEPTH_REV = "1d0d3c52f60b5164629d279bb9a7546458e6dcc4"
LONG_DIM     = 640
MAX_FRAMES   = 60

DA_CKPT = str(MEGASAM_DIR / "Depth-Anything" / "checkpoints" / "depth_anything_vitl14.pth")

# ── SLAM systematic bias correction ──────────────────────────────────────────
# DROID-SLAM opt-intr systematically underestimates fx by ~10%.
# Empirically validated: EgoDex (AVP 104deg FOV) -9.2%, TACO (GoPro 71deg) -9.8%.
# Apply x1.10 to recover true focal length.
SLAM_INTR_CORRECTION = 1.10


# ── Motion screening (optical flow, no GPU needed) ────────────────────────────
def motion_score(video: Path, n_frames: int = 8) -> float:
    """
    Estimate camera motion via sparse optical flow on N frames.
    Returns mean per-frame pixel displacement. Fast: <1 second per video.
    Note: detects all motion (hand + camera). Use as coarse pre-filter only.
    """
    cap   = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // n_frames)
    frames = []
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, f = cap.read()
        if ret:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < 2:
        return 0.0
    displacements = []
    for a, b in zip(frames[:-1], frames[1:]):
        pts = cv2.goodFeaturesToTrack(a, maxCorners=200, qualityLevel=0.01, minDistance=10)
        if pts is None or len(pts) < 10:
            displacements.append(0.0)
            continue
        pts2, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
        good = status.ravel() == 1
        if good.sum() < 5:
            displacements.append(0.0)
            continue
        disp = np.linalg.norm(pts2[good] - pts[good], axis=2).mean()
        displacements.append(float(disp))
    return float(np.mean(displacements)) if displacements else 0.0


# ── Dataset video lists ───────────────────────────────────────────────────────
def get_video_list(dataset: str) -> list:
    base = PROJ / "data_hub/RawData/EgoRawData"
    if dataset == "egodex":
        vids = []
        depth_base = PROJ / "data_hub/ProcessedData/egocentric_depth/egodex"
        for meta_f in sorted(depth_base.glob("*/*/meta.json")):
            parts = meta_f.parts
            task  = parts[-3]
            idx   = parts[-2]
            mp4   = base / f"egodex/test/{task}/{idx}.mp4"
            if mp4.exists():
                vids.append(mp4)
        return vids
    elif dataset == "taco":
        return [Path(p) for p in glob(
            str(base / "taco/Egocentric_RGB_Videos/**/color.mp4"), recursive=True)]
    elif dataset == "hoi4d":
        return [Path(p) for p in glob(
            str(base / "hoi4d/HOI4D_release/**/image.mp4"), recursive=True)]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ── Frame extraction ──────────────────────────────────────────────────────────
def extract_frames(video: Path, out_dir: Path, n: int) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video)],
        capture_output=True, text=True)
    info = json.loads(r.stdout)
    total = 0
    for s in info["streams"]:
        if s.get("codec_type") == "video":
            total = int(s.get("nb_frames", 0))
            if not total:
                fps_n, fps_d = s.get("r_frame_rate", "30/1").split("/")
                total = int(float(s.get("duration", 10)) * int(fps_n) / int(fps_d))
            break
    step = max(1, total // n)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"select='not(mod(n\\,{step}))',setpts=N/FRAME_RATE/TB",
        "-vframes", str(n), "-q:v", "2",
        str(out_dir / "%05d.png")
    ], capture_output=True)
    return natsorted(out_dir.glob("*.png"))


# ── Model loaders (singletons) ────────────────────────────────────────────────
_ud_model = None
def load_unidepth():
    global _ud_model
    if _ud_model is not None:
        return _ud_model
    from unidepth.models import UniDepthV2
    print("  Loading UniDepthV2 ...")
    m = UniDepthV2.from_pretrained(UNIDEPTH_ID, revision=UNIDEPTH_REV)
    m = m.to("cuda").eval()
    print("  UniDepthV2 loaded")
    _ud_model = m
    return m


_da_model = None
_da_transform = None
def load_depth_anything():
    global _da_model, _da_transform
    if _da_model is not None:
        return _da_model, _da_transform
    sys.path.insert(0, str(MEGASAM_DIR / "Depth-Anything"))
    from depth_anything.dpt import DPT_DINOv2
    from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose
    da = DPT_DINOv2(encoder="vitl", features=256,
                    out_channels=[256, 512, 1024, 1024],
                    localhub=False).cuda().eval()
    da.load_state_dict(torch.load(DA_CKPT, map_location="cpu", weights_only=False), strict=True)
    transform = Compose([
        Resize(width=768, height=768, resize_target=False,
               keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method="upper_bound",
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    print("  DepthAnything loaded")
    _da_model, _da_transform = da, transform
    return da, transform


# ── UniDepth initial fx estimate ──────────────────────────────────────────────
def unidepth_estimate_fx(frame_paths: list) -> float:
    """Estimate fx at native resolution from first 8 frames via UniDepth."""
    model = load_unidepth()
    fxs   = []
    for p in frame_paths[:8]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        rgb = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t   = torch.from_numpy(rgb).permute(2, 0, 1).cuda()
        with torch.no_grad():
            pred = model.infer(t)
        if "intrinsics" in pred:
            fxs.append(float(pred["intrinsics"][0].cpu()[0, 0]))
        elif "K" in pred:
            fxs.append(float(pred["K"][0].cpu()[0, 0]))
    return float(np.median(fxs)) if fxs else None


# ── DepthAnything on frames ───────────────────────────────────────────────────
def run_da(frame_paths: list, da_dir: Path):
    import torch.nn.functional as F
    da, transform = load_depth_anything()
    da_dir.mkdir(parents=True, exist_ok=True)
    for p in frame_paths:
        raw = cv2.imread(str(p))[..., :3]
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) / 255.0
        h, w = img.shape[:2]
        t = transform({"image": img})["image"]
        t = torch.from_numpy(t).unsqueeze(0).cuda()
        with torch.no_grad():
            disp = da(t)
        disp = F.interpolate(disp[None], (h, w), mode="bilinear",
                             align_corners=False)[0, 0]
        np.save(str(da_dir / (p.stem + ".npy")), disp.cpu().float().numpy())


# ── UniDepth metric depth ─────────────────────────────────────────────────────
def run_ud(frame_paths: list, ud_dir: Path, calib_fx: float):
    model = load_unidepth()
    ud_dir.mkdir(parents=True, exist_ok=True)
    for p in frame_paths:
        rgb = np.array(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
        t   = torch.from_numpy(rgb).permute(2, 0, 1).cuda()
        with torch.no_grad():
            pred = model.infer(t)
        depth = pred["depth"][0, 0].cpu().float().numpy()
        fov   = float(np.degrees(2 * np.arctan(depth.shape[1] / (2 * calib_fx))))
        np.savez(str(ud_dir / (p.stem + ".npz")),
                 depth=np.float32(depth), fov=fov)


# ── DROID-SLAM with opt-intr ──────────────────────────────────────────────────
def run_droid_opt_intr(frame_dir: Path, da_dir: Path, ud_dir: Path,
                       seq_id: str, init_fx: float):
    sys.path.insert(0, str(PROJ / "data"))
    from batch_megasam import run_droid
    try:
        imgs  = natsorted(frame_dir.glob("*.png"))
        img0  = cv2.imread(str(imgs[0]))
        h0, w0 = img0.shape[:2]
        scale = LONG_DIM / max(h0, w0)
        depths_m, motion, K_opt, cam_c2w = run_droid(
            str(frame_dir), str(da_dir), str(ud_dir),
            seq_id, init_fx * scale, opt_intr=True)
        if K_opt is None:
            return None
        return float(K_opt[0, 0]) / scale
    except Exception as e:
        print(f"    DROID failed: {e}")
        return None


# ── Per-sequence calibration ──────────────────────────────────────────────────
def calibrate_sequence(video: Path, seq_id: str):
    print(f"  [{seq_id}]")
    with tempfile.TemporaryDirectory() as tmp:
        tmp    = Path(tmp)
        frames = extract_frames(video, tmp / "frames", MAX_FRAMES)
        if len(frames) < 10:
            print(f"    Too few frames ({len(frames)}), skip")
            return None
        init_fx = unidepth_estimate_fx(frames)
        if init_fx is None:
            print("    UniDepth init failed")
            return None
        print(f"    UniDepth init fx = {init_fx:.1f}")
        run_da(frames, tmp / "da")
        h0, w0 = cv2.imread(str(frames[0])).shape[:2]
        scale  = LONG_DIM / max(h0, w0)
        run_ud(frames, tmp / "ud", init_fx * scale)
        fx_opt = run_droid_opt_intr(
            tmp / "frames", tmp / "da", tmp / "ud", seq_id, init_fx)
        if fx_opt is None:
            print("    SLAM opt-intr failed")
            return None
        print(f"    SLAM K_opt fx = {fx_opt:.1f}  (@{LONG_DIM}px: {fx_opt*scale:.1f})")
        return fx_opt


# ── Auto-patch CALIB_FX in batch_megasam.py ──────────────────────────────────
def update_calib_fx(dataset: str, new_fx_640: float):
    """Update or insert CALIB_FX[dataset] in batch_megasam.py."""
    target = PROJ / "data" / "batch_megasam.py"
    src    = target.read_text()
    comment = f"  # SLAM opt-intr auto-calib x{SLAM_INTR_CORRECTION}"
    entry   = f'    "{dataset}": {new_fx_640:.1f},{comment}'
    pattern = re.compile(rf'^\s*"{re.escape(dataset)}"\s*:.*$', re.MULTILINE)
    existing = pattern.search(src)
    if existing:
        src    = src[:existing.start()] + entry + src[existing.end():]
        action = "updated"
    else:
        src = re.sub(
            r'(CALIB_FX\s*=\s*\{[^}]*?)(\})',
            lambda m: m.group(1) + entry + '\n' + m.group(2),
            src, flags=re.DOTALL
        )
        action = "inserted"
    target.write_text(src)
    print(f"  batch_megasam.py CALIB_FX[\"{dataset}\"] {action}: {new_fx_640:.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Auto intrinsic calibration via SLAM opt-intr (no GT needed).\n"
                    "Samples N seqs -> SLAM -> 2sigma filter -> x1.10 -> patches CALIB_FX."
    )
    ap.add_argument("--dataset",    required=True, choices=["egodex", "taco", "hoi4d"])
    ap.add_argument("--n",          type=int, default=15,
                    help="Sequences to calibrate (default 15)")
    ap.add_argument("--min-motion", type=float, default=0.0,
                    help="Min optical-flow score px/frame (0=no filter)")
    ap.add_argument("--no-patch",   action="store_true",
                    help="Print result only, do NOT write to batch_megasam.py")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    all_videos = get_video_list(args.dataset)
    print(f"\n{'='*60}")
    print(f"  Dataset:    {args.dataset}  ({len(all_videos)} total videos)")

    # Step 1: Motion pre-screening (optional)
    if args.min_motion > 0:
        print(f"  Screening motion >= {args.min_motion} px/frame ...")
        passed = [(motion_score(v), v) for v in all_videos]
        passed = [(s, v) for s, v in passed if s >= args.min_motion]
        passed.sort(reverse=True)
        videos = [v for _, v in passed]
        print(f"  Passed: {len(videos)}/{len(all_videos)}")
        if not videos:
            print("No videos passed motion filter"); return
    else:
        videos = all_videos

    # Step 2: Random sample
    samples = random.sample(videos, min(args.n, len(videos)))
    print(f"  Calibrating on {len(samples)} sequences\n")

    # Step 3: SLAM opt-intr per sequence
    results = {}
    for vid in samples:
        seq_id = f"{vid.parts[-2]}/{vid.stem}"
        fx_opt = calibrate_sequence(vid, seq_id)
        if fx_opt is not None:
            results[seq_id] = {"fx_opt_native": float(fx_opt)}

    if not results:
        print("No sequences converged. Camera likely has insufficient ego-motion.")
        print("Try: --n 50, or use --opt-intr flag in batch_megasam.py per sequence.")
        return

    # Step 4: Get native video width
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(samples[0])],
        capture_output=True, text=True)
    native_w = next(
        (int(s["width"]) for s in json.loads(r.stdout)["streams"]
         if s.get("codec_type") == "video"), 1920)
    scale = LONG_DIM / native_w
    fxs   = np.array([v["fx_opt_native"] for v in results.values()])

    # Step 5: 2sigma outlier filter
    med_raw = float(np.median(fxs))
    std_raw = float(np.std(fxs))
    keep    = np.abs(fxs - med_raw) <= 2 * std_raw
    fxs_ok  = fxs[keep] if keep.sum() > 0 else fxs
    n_rm    = int((~keep).sum())
    med_ok  = float(np.median(fxs_ok))

    # Step 6: Apply 1.10 correction
    calib_fx_640 = round(med_ok * SLAM_INTR_CORRECTION * scale, 1)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Dataset:           {args.dataset}")
    print(f"  Converged:         {len(results)}/{len(samples)}")
    print(f"  Outliers removed:  {n_rm}  (|fx - median| > 2sigma)")
    print(f"  Clean sequences:   {len(fxs_ok)}")
    print(f"  Median fx raw:     {med_ok:.1f} px @{native_w}px")
    print(f"  x{SLAM_INTR_CORRECTION} corrected:    {med_ok*SLAM_INTR_CORRECTION:.1f} px @{native_w}px")
    print(f"  CALIB_FX:          {calib_fx_640:.1f} px @{LONG_DIM}px  <-- final")

    # Step 7: Save JSON
    seq_names = list(results.keys())
    for i, sid in enumerate(seq_names):
        results[sid]["kept"]        = bool(keep[i])
        results[sid]["fx_corr_640"] = round(
            results[sid]["fx_opt_native"] * SLAM_INTR_CORRECTION * scale, 1)
    results["__summary__"] = {
        "dataset":      args.dataset,
        "n_sampled":    len(samples),
        "n_converged":  len(results) - 1,
        "n_outliers":   n_rm,
        "median_fx_native":    med_ok,
        "correction":          SLAM_INTR_CORRECTION,
        "calib_fx_640":        calib_fx_640,
        "native_width":        native_w,
    }
    out = PROJ / "output" / f"calib_fx_{args.dataset}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON -> {out}")

    # Step 8: Patch batch_megasam.py
    if not args.no_patch:
        update_calib_fx(args.dataset, calib_fx_640)
        print(f"\nDone. Run batch_megasam.py --dataset {args.dataset} --resume")
    else:
        print(f"\n--no-patch: add manually -> CALIB_FX[\"{args.dataset}\"] = {calib_fx_640}")


if __name__ == "__main__":
    main()
