#!/usr/bin/env python3
"""
eval_unidepth_k_egodex.py
==========================
用 UniDepth 对 EgoDex 全部 10 条已处理序列估计内参 K，
与之前 CALIB_FX 手动值对比，输出新的自动估计值。

无任何 GT 或人工标定输入。

运行（mega_sam 环境，mega-sam/ 目录下）：
    cd $PROJ/mega-sam
    conda run -n mega_sam python ../data/eval_unidepth_k_egodex.py 2>&1 | tee ../output/unidepth_k_egodex.log
"""

import os, sys, json, subprocess, tempfile
import numpy as np
from pathlib import Path
from glob import glob

PROJ        = Path(__file__).resolve().parent.parent
MEGASAM_DIR = PROJ / "mega-sam"
sys.path.insert(0, str(MEGASAM_DIR))
sys.path.insert(0, str(MEGASAM_DIR / "UniDepth"))

UNIDEPTH_ID  = "lpiccinelli/unidepth-v2-vitl14"
UNIDEPTH_REV = "1d0d3c52f60b5164629d279bb9a7546458e6dcc4"

N_FRAMES = 10   # frames per video
LONG_DIM = 640  # same as batch_megasam.py

# 全部 10 条已处理的 EgoDex 序列 → 对应 mp4 路径
EGODEX_VIDEOS = {
    "basic_fold/38":                                  "egodex/test/basic_fold/38.mp4",
    "basic_pick_place/259":                           "egodex/test/basic_pick_place/259.mp4",
    "build_unstack_lego/9":                           "egodex/test/build_unstack_lego/9.mp4",
    "dry_hands/38":                                   "egodex/test/dry_hands/38.mp4",
    "flip_pages/14":                                  "egodex/test/flip_pages/14.mp4",
    "fry_bread/0":                                    "egodex/test/fry_bread/0.mp4",
    "slot_batteries/1":                               "egodex/test/slot_batteries/1.mp4",
    "stack_unstack_cups/11":                          "egodex/test/stack_unstack_cups/11.mp4",
    "throw_collect_objects/5":                        "egodex/test/throw_collect_objects/5.mp4",
    "assemble_disassemble_furniture_bench_desk/15":   "egodex/test/assemble_disassemble_furniture_bench_desk/15.mp4",
}

# 旧 CALIB_FX 手动值（供对比）
OLD_CALIB_FX = 227.8   # px at 640px wide
# AVP GT (仅参考)
AVP_GT_FX_FULL = 748.98   # px at 1920px


def extract_n_frames(video_path: Path, out_dir: Path, n: int) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True, text=True
    )
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
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"select='not(mod(n\\,{step}))',setpts=N/FRAME_RATE/TB",
        "-vframes", str(n), "-q:v", "2",
        str(out_dir / "%05d.png")
    ], capture_output=True)
    return sorted(out_dir.glob("*.png"))


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
    print("  ✅ UniDepthV2 loaded\n")
    _ud_model = model
    return model


def estimate_fx_from_video(video_path: Path) -> tuple[float | None, list]:
    """Returns (median_fx_at_native_res, [per_frame_fx])."""
    import torch, cv2
    model = load_unidepth()
    fxs = []
    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_n_frames(video_path, Path(tmp) / "frames", N_FRAMES)
        if len(frames) < 3:
            return None, []
        for p in frames:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            rgb = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            t = torch.from_numpy(rgb).permute(2, 0, 1).cuda()
            with torch.no_grad():
                pred = model.infer(t)
            if "intrinsics" in pred:
                K = pred["intrinsics"][0].cpu().float().numpy()
                fxs.append(float(K[0, 0]))
            elif "K" in pred:
                K = pred["K"][0].cpu().float().numpy()
                fxs.append(float(K[0, 0]))
    return (float(np.median(fxs)) if fxs else None), fxs


def main():
    raw_base = PROJ / "data_hub/RawData/EgoRawData"

    print(f"EgoDex UniDepth K Estimation ({N_FRAMES} frames/seq)")
    print(f"OLD CALIB_FX (manual): {OLD_CALIB_FX} px @640px wide → {OLD_CALIB_FX/640*1920:.1f} @1920px")
    print(f"AVP GT fx (reference): {AVP_GT_FX_FULL} @1920px\n")
    print(f"{'Sequence':50s}  fx@native  fx@640px  vs_old%")
    print("─" * 80)

    results = {}
    for seq_id, rel_path in EGODEX_VIDEOS.items():
        vid = raw_base / rel_path
        if not vid.exists():
            print(f"  ⚠ Missing: {vid}")
            continue

        # Get native resolution
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(vid)],
            capture_output=True, text=True
        )
        info = json.loads(r.stdout)
        native_w = 1920
        for s in info["streams"]:
            if s.get("codec_type") == "video":
                native_w = int(s.get("width", 1920))
                break

        fx_native, per_frame = estimate_fx_from_video(vid)
        if fx_native is None:
            print(f"  ❌ Failed: {seq_id}")
            continue

        # Scale to 640px wide (LONG_DIM)
        fx_640 = fx_native * (LONG_DIM / native_w)
        vs_old = (fx_640 - OLD_CALIB_FX) / OLD_CALIB_FX * 100

        print(f"  {seq_id:50s}  {fx_native:8.1f}   {fx_640:6.1f}   {vs_old:+.1f}%")
        results[seq_id] = {
            "fx_native": fx_native,
            "native_w":  native_w,
            "fx_640":    fx_640,
            "per_frame": per_frame
        }

    # Global median
    all_fx_640 = [v["fx_640"] for v in results.values()]
    if all_fx_640:
        median_fx_640 = float(np.median(all_fx_640))
        print("\n" + "─" * 80)
        print(f"  Sequences processed: {len(all_fx_640)}/10")
        print(f"  Old CALIB_FX:            {OLD_CALIB_FX:.1f} px @640")
        print(f"  New AUTO CALIB_FX:       {median_fx_640:.1f} px @640  (UniDepth median)")
        print(f"  Std dev:                 {np.std(all_fx_640):.1f} px")
        print(f"  Min / Max:               {np.min(all_fx_640):.1f} / {np.max(all_fx_640):.1f}")
        print(f"\n  → Update batch_megasam.py:")
        print(f'    CALIB_FX["egodex"] = {median_fx_640:.1f}   # UniDepth auto (was {OLD_CALIB_FX})')

    # Save
    out = PROJ / "output" / "unidepth_k_egodex.json"
    out.parent.mkdir(exist_ok=True)
    results["__summary__"] = {
        "old_calib_fx_640": OLD_CALIB_FX,
        "new_median_fx_640": float(np.median(all_fx_640)) if all_fx_640 else None,
        "avp_gt_fx_1920": AVP_GT_FX_FULL,
    }
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅  Results → {out}")


if __name__ == "__main__":
    main()
