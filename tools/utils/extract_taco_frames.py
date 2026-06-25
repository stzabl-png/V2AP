#!/usr/bin/env python3
"""
extract_taco_frames.py — TACO Allocentric MP4 → jpg 帧提取

两种模式:
  --mode annotate   每个 object 类别选 1 个代表帧（SAM2/SAM3D 标注用，9 帧）
  --mode pipeline   全量提取（DepthPro + HaPTIC 用），按 --max-frames 限制

输出结构（与 discover_taco_allocentric 期待完全一致）:
  Allocentric_RGB_Videos/(triplet)/(session)/{cam_serial}/
    000001.jpg
    000002.jpg
    ...

用法:
  # SAM2 标注：每物体选 1 帧
  conda activate base
  python tools/extract_taco_frames.py --mode annotate --cam 22139905

  # 完整 pipeline 解帧（慢，约 130GB）
  python tools/extract_taco_frames.py --mode pipeline --cam 22139905 --max-frames 150

  # 只处理某个 triplet
  python tools/extract_taco_frames.py --mode pipeline --triplet "(dust, roller, pan)"
"""

import os, sys, argparse, subprocess
from pathlib import Path
from natsort import natsorted
from glob import glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

RAW_BASE = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData",
                        "taco", "Allocentric_RGB_Videos")


def get_obj_from_triplet(triplet: str) -> str:
    """'(action, tool, object)' → 'object'"""
    inner = triplet.strip("()")
    parts = [p.strip() for p in inner.split(",")]
    return parts[-1] if parts else triplet


def extract_frames(mp4_path: str, out_dir: str, max_frames: int = 0,
                   target_frame: int = -1) -> list:
    """
    Extract frames from MP4 to out_dir as 000001.jpg, 000002.jpg, ...

    Args:
        target_frame: if >= 0, extract only this single frame index (for annotate mode)
        max_frames: if > 0, extract at most this many frames (evenly spaced)
    Returns:
        list of extracted jpg paths
    """
    os.makedirs(out_dir, exist_ok=True)

    if target_frame >= 0:
        # Single frame mode
        out_path = os.path.join(out_dir, f"{target_frame+1:06d}.jpg")
        if os.path.exists(out_path):
            return [out_path]
        cmd = [
            "ffmpeg", "-i", mp4_path,
            "-vf", f"select=eq(n\\,{target_frame})",
            "-frames:v", "1",
            "-q:v", "2",
            out_path, "-y", "-loglevel", "quiet"
        ]
        subprocess.run(cmd, check=True)
        return [out_path] if os.path.exists(out_path) else []

    # Multi-frame mode: get total frame count first
    probe = subprocess.run([
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=noprint_wrappers=1:nokey=1", mp4_path
    ], capture_output=True, text=True)
    try:
        total = int(probe.stdout.strip())
    except ValueError:
        total = 200  # fallback

    if max_frames > 0 and total > max_frames:
        # Select evenly spaced frames
        indices = [int(i * total / max_frames) for i in range(max_frames)]
        select_expr = "+".join(f"eq(n\\,{i})" for i in indices)
        vf = f"select='{select_expr}',setpts=N/FRAME_RATE/TB"
    else:
        vf = None

    cmd = ["ffmpeg", "-i", mp4_path]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-q:v", "2", os.path.join(out_dir, "%06d.jpg"),
            "-y", "-loglevel", "quiet"]
    subprocess.run(cmd, check=True)

    return natsorted(glob(os.path.join(out_dir, "*.jpg")))


def mode_annotate(cam: str, redo: bool = False):
    """
    Per-object annotation mode: for each unique object category,
    pick the first available session and extract frame 30 (roughly 1s in).
    """
    # Group triplets by object
    by_obj = defaultdict(list)
    for triplet_dir in natsorted(os.listdir(RAW_BASE)):
        full = os.path.join(RAW_BASE, triplet_dir)
        if not os.path.isdir(full):
            continue
        obj = get_obj_from_triplet(triplet_dir)
        by_obj[obj].append(triplet_dir)

    print(f"Found {len(by_obj)} unique objects: {sorted(by_obj.keys())}\n")

    for obj, triplets in sorted(by_obj.items()):
        # Pick first triplet that has an mp4 for this cam
        chosen_mp4 = None
        chosen_triplet = None
        chosen_session = None

        for triplet in triplets:
            triplet_path = os.path.join(RAW_BASE, triplet)
            for session in natsorted(os.listdir(triplet_path)):
                mp4 = os.path.join(triplet_path, session, f"{cam}.mp4")
                if os.path.exists(mp4):
                    chosen_mp4 = mp4
                    chosen_triplet = triplet
                    chosen_session = session
                    break
            if chosen_mp4:
                break

        if not chosen_mp4:
            print(f"  ⚠️  {obj}: no mp4 found for cam {cam}")
            continue

        # Output dir: alongside the mp4, in cam_serial subdirectory
        out_dir = os.path.join(RAW_BASE, chosen_triplet, chosen_session, cam)
        frame_path = os.path.join(out_dir, "000031.jpg")

        if os.path.exists(frame_path) and not redo:
            print(f"  ✅ {obj}: already extracted ({frame_path})")
            continue

        print(f"  → {obj}: {chosen_triplet}/{chosen_session}")
        frames = extract_frames(chosen_mp4, out_dir, target_frame=30)
        if frames:
            print(f"     ✅ {frames[0]}")
        else:
            print(f"     ❌ extraction failed")


def mode_pipeline(cam: str, max_frames: int, triplet_filter: str,
                  redo: bool = False):
    """
    Full pipeline extraction: convert all MP4s to jpg frames.
    """
    for triplet_dir in natsorted(os.listdir(RAW_BASE)):
        if triplet_filter and triplet_filter not in triplet_dir:
            continue
        full = os.path.join(RAW_BASE, triplet_dir)
        if not os.path.isdir(full):
            continue

        for session in natsorted(os.listdir(full)):
            mp4 = os.path.join(full, session, f"{cam}.mp4")
            if not os.path.exists(mp4):
                continue

            out_dir = os.path.join(full, session, cam)
            existing = glob(os.path.join(out_dir, "*.jpg"))
            if existing and not redo:
                print(f"  skip (already done): {triplet_dir}/{session}")
                continue

            print(f"  → {triplet_dir}/{session} ...", end=" ", flush=True)
            frames = extract_frames(mp4, out_dir, max_frames=max_frames)
            print(f"{len(frames)} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["annotate", "pipeline"],
                    default="annotate",
                    help="annotate=1 frame/object for SAM2; pipeline=all frames")
    ap.add_argument("--cam", default="22139905",
                    help="Camera serial number (filename stem of mp4)")
    ap.add_argument("--max-frames", type=int, default=150,
                    help="[pipeline mode] max frames per sequence")
    ap.add_argument("--triplet", default="",
                    help="[pipeline mode] substring filter for triplet name")
    ap.add_argument("--redo", action="store_true",
                    help="Re-extract even if output already exists")
    args = ap.parse_args()

    print(f"RAW_BASE: {RAW_BASE}")
    print(f"Mode: {args.mode}  Cam: {args.cam}\n")

    if args.mode == "annotate":
        mode_annotate(args.cam, redo=args.redo)
    else:
        mode_pipeline(args.cam, args.max_frames, args.triplet, redo=args.redo)


if __name__ == "__main__":
    main()
