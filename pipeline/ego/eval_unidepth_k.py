#!/usr/bin/env python3
"""
eval_unidepth_k.py
==================
用 UniDepth 对 HOI4D / TACO Ego 随机 10 条序列做 per-frame 焦距估计，
与 GT 内参对比误差。

运行（在 mega_sam 环境，mega-sam/ 目录下）：
    cd $PROJ/mega-sam
    conda run -n mega_sam python ../data/eval_unidepth_k.py 2>&1 | tee ../output/unidepth_k_eval.log
"""

import os, sys, json, random, tempfile, subprocess
import numpy as np
from pathlib import Path
from glob import glob

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ         = Path(__file__).resolve().parent.parent
MEGASAM_DIR  = PROJ / "mega-sam"
sys.path.insert(0, str(MEGASAM_DIR))
sys.path.insert(0, str(MEGASAM_DIR / "UniDepth"))

HOI4D_VIDEO  = PROJ / "data_hub/RawData/EgoRawData/hoi4d/HOI4D_release"
HOI4D_INTRIN = PROJ / "data_hub/RawData/EgoRawData/hoi4d/camera_params"
TACO_VIDEO   = PROJ / "data_hub/RawData/EgoRawData/taco/Egocentric_RGB_Videos"
TACO_INTRIN  = PROJ / "data_hub/RawData/EgoRawData/taco/Egocentric_Camera_Parameters"

# Exact same revision used in batch_megasam.py
UNIDEPTH_ID  = "lpiccinelli/unidepth-v2-vitl14"
UNIDEPTH_REV = "1d0d3c52f60b5164629d279bb9a7546458e6dcc4"

N_SAMPLE  = 10
N_FRAMES  = 8   # frames per sequence (quick)

# ── GT loaders ────────────────────────────────────────────────────────────────
def gt_K_hoi4d(video_path: Path) -> np.ndarray:
    device = video_path.parts[video_path.parts.index("HOI4D_release") + 1]
    return np.load(HOI4D_INTRIN / device / "intrin.npy").reshape(3, 3).astype(np.float32)

def gt_K_taco(video_path: Path) -> np.ndarray:
    triplet = video_path.parts[-3]
    seq     = video_path.parts[-2]
    txt = TACO_INTRIN / triplet / seq / "egocentric_intrinsic.txt"
    return np.loadtxt(str(txt)).reshape(3, 3).astype(np.float32)

# ── Frame extraction ──────────────────────────────────────────────────────────
def extract_n_frames(video_path: Path, out_dir: Path, n: int) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(video_path)],
        capture_output=True, text=True
    )
    info = json.loads(r.stdout)
    total = 0
    for s in info["streams"]:
        if s.get("codec_type") == "video":
            total = int(s.get("nb_frames", 0))
            if not total:
                fps_n, fps_d = s.get("r_frame_rate","30/1").split("/")
                total = int(float(s.get("duration",10)) * int(fps_n) / int(fps_d))
            break
    step = max(1, total // n)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"select='not(mod(n\\,{step}))',setpts=N/FRAME_RATE/TB",
        "-vframes", str(n), "-q:v", "2",
        str(out_dir / "%05d.png")
    ], capture_output=True)
    return sorted(out_dir.glob("*.png"))

# ── UniDepth model (singleton) ────────────────────────────────────────────────
_ud_model = None
def load_unidepth():
    global _ud_model
    if _ud_model is not None:
        return _ud_model
    import torch
    from unidepth.models import UniDepthV2
    print("  Loading UniDepthV2 ...")
    model = UniDepthV2.from_pretrained(UNIDEPTH_ID, revision=UNIDEPTH_REV)
    model = model.to("cuda").eval()
    print("  ✅ UniDepthV2 loaded")
    _ud_model = model
    return model

# ── UniDepth per-frame fx estimation ─────────────────────────────────────────
def estimate_fx_unidepth(frame_paths: list) -> float | None:
    """
    Run UniDepth on each frame (raw uint8 RGB, as done in batch_megasam.py).
    Extract pred["intrinsics"] → median fx at original resolution.
    """
    import torch, cv2
    model  = load_unidepth()
    fxs    = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        rgb = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # UniDepth expects uint8 tensor (3, H, W) — no normalization, no resize
        t = torch.from_numpy(rgb).permute(2, 0, 1).cuda()
        with torch.no_grad():
            pred = model.infer(t)

        # Extract intrinsics from UniDepth's prediction
        if "intrinsics" in pred:
            K_pred = pred["intrinsics"][0].cpu().float().numpy()   # (3,3)
            fx_pred = float(K_pred[0, 0])
        elif "K" in pred:
            K_pred = pred["K"][0].cpu().float().numpy()
            fx_pred = float(K_pred[0, 0])
        else:
            # Reconstruct from depth shape + FOV if available
            depth = pred["depth"][0, 0].cpu().float().numpy()
            dh, dw = depth.shape
            # Fall back to FOV if we can get it from the model
            continue
        fxs.append(fx_pred)

    return float(np.median(fxs)) if fxs else None

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(name: str, videos: list, gt_fn, n=N_SAMPLE):
    print(f"\n{'='*60}")
    print(f"  {name}  (n={n}, {N_FRAMES} frames/seq)")
    print(f"{'='*60}")
    print(f"  {'Sequence':40s}  GT_fx    EST_fx   ERR%")

    random.seed(42)
    samples = random.sample(videos, min(n, len(videos)))
    results = []

    for i, vid in enumerate(samples):
        vid = Path(vid)
        label = f"{vid.parts[-3]}/{vid.parts[-2]}"[:40]
        try:
            K_gt  = gt_fn(vid)
            fx_gt = float(K_gt[0, 0])
        except Exception as e:
            print(f"  ⚠ GT fail: {label}: {e}")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            frames = extract_n_frames(vid, Path(tmp) / "frames", N_FRAMES)
            if len(frames) < 3:
                print(f"  ⚠ Too few frames: {label}")
                continue
            try:
                fx_est = estimate_fx_unidepth(frames)
            except Exception as e:
                print(f"  ❌ UniDepth error: {e}")
                continue

        if fx_est is None:
            print(f"  ❌ No intrinsics output: {label}")
            continue

        err = (fx_est - fx_gt) / fx_gt * 100.0
        flag = "✅" if abs(err) <= 10 else ("⚠" if abs(err) <= 20 else "❌")
        print(f"  {label:40s}  {fx_gt:7.1f}  {fx_est:7.1f}  {err:+6.1f}%  {flag}")
        results.append({"seq": str(vid), "fx_gt": fx_gt, "fx_est": fx_est, "err": err})

    # Summary
    print(f"\n  {'─'*50}")
    if results:
        errs  = [r["err"]      for r in results]
        abses = [abs(r["err"]) for r in results]
        print(f"  Succeeded:     {len(results)}/{n}")
        print(f"  mean  |err|:   {np.mean(abses):.1f}%")
        print(f"  median|err|:   {np.median(abses):.1f}%")
        print(f"  ≤ 5%:  {sum(a<=5  for a in abses)}/{len(abses)}")
        print(f"  ≤10%:  {sum(a<=10 for a in abses)}/{len(abses)}")
        print(f"  ≤20%:  {sum(a<=20 for a in abses)}/{len(abses)}")
    return results


def main():
    hoi4d_vids = glob(str(HOI4D_VIDEO / "**" / "image.mp4"), recursive=True)
    taco_vids  = glob(str(TACO_VIDEO  / "**" / "color.mp4"),  recursive=True)
    print(f"HOI4D: {len(hoi4d_vids)} videos | TACO Ego: {len(taco_vids)} videos")
    print("NOTE: UniDepth intrinsics estimate vs GT camera_params")

    all_results = {}
    all_results["hoi4d"] = evaluate("HOI4D",    hoi4d_vids, gt_K_hoi4d)
    all_results["taco"]  = evaluate("TACO Ego", taco_vids,  gt_K_taco)

    out = PROJ / "output" / "unidepth_k_eval.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅  Results → {out}")


if __name__ == "__main__":
    main()
