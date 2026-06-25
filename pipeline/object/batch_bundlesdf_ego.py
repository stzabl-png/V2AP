#!/usr/bin/env python3
"""
batch_bundlesdf_ego.py — Batch BundleSDF object pose tracking for EgoDex
=========================================================================
For each sequence in egodex_sequence_registry.json:
  1. Prepare video_dir: rgb/ + depth/ (MegaSAM) + masks/ + cam_K.txt
  2. Run BundleSDF (bundlesdf conda env) in the BundleSDF/ directory
  3. Copy ob_in_cam/*.txt → obj_poses_ego/egodex/{task}__{ep}/ob_in_cam/

Usage:
  conda activate bundlesdf   (or run via conda run)
  cd /home/lyh/Project/V2AP
  python data/batch_bundlesdf_ego.py
  python data/batch_bundlesdf_ego.py --seq add_remove_lid/22  # single
  python data/batch_bundlesdf_ego.py --start 0 --end 10       # shard
  python data/batch_bundlesdf_ego.py --workers 2              # parallel
"""

import os, sys, json, argparse, shutil, subprocess, tempfile
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJ          = config.PROJECT_DIR
BUNDLESDF_DIR = os.path.join(PROJ, "BundleSDF")
REGISTRY_JSON = os.path.join(PROJ, "tools", "egodex_sequence_registry.json")
EGODEX_ROOT   = os.path.join(config.DATA_HUB, "RawData", "EgoRawData", "egodex", "test")
DEPTH_BASE    = os.path.join(config.DATA_HUB, "ProcessedData", "egocentric_depth", "egodex")
MASK_BASE     = os.path.join(config.DATA_HUB, "ProcessedData", "obj_recon_input", "egocentric")
OUT_POSE_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "obj_poses_ego", "egodex")
STAGING_BASE  = os.path.join(config.DATA_HUB, "ProcessedData", "bundlesdf_input", "egodex")
CONDA_ENV     = "bundlesdf"

MAX_FRAMES = 60  # BundleSDF works best with 40-80 frames


# ── Step 1: prepare video_dir ──────────────────────────────────────────────────

def extract_rgb_frames(video_path: str, rgb_dir: str, max_frames: int = MAX_FRAMES):
    """Extract evenly-sampled frames from MP4 → rgb/*.png"""
    import cv2
    os.makedirs(rgb_dir, exist_ok=True)
    if len(glob(os.path.join(rgb_dir, "*.png"))) >= 5:
        return True  # already done

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release(); return False

    step = max(1, total // max_frames)
    indices = list(range(0, total, step))[:max_frames]
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(os.path.join(rgb_dir, f"{saved:06d}.png"), frame)
        saved += 1
    cap.release()
    return saved > 0


def write_megasam_depth(seq_id: str, depth_dir: str, n_frames: int):
    """Write MegaSAM metric depth → uint16 mm PNG files."""
    from PIL import Image
    os.makedirs(depth_dir, exist_ok=True)
    if len(glob(os.path.join(depth_dir, "*.png"))) >= n_frames - 1:
        return True

    depth_npz = os.path.join(DEPTH_BASE, seq_id, "depth.npz")
    if not os.path.exists(depth_npz):
        return False

    npz = np.load(depth_npz)
    key = "depths" if "depths" in npz else "depth"
    depths = npz[key]                              # (N, H, W) float32 meters
    total = depths.shape[0]
    step = max(1, total // n_frames)
    indices = list(range(0, total, step))[:n_frames]

    for out_i, src_i in enumerate(indices):
        d_mm = (depths[src_i] * 1000).clip(0, 65535).astype(np.uint16)
        Image.fromarray(d_mm).save(os.path.join(depth_dir, f"{out_i:06d}.png"))
    return True


def write_cam_K(seq_id: str, k_path: str):
    """Write cam_K.txt from MegaSAM K.npy"""
    if os.path.exists(k_path):
        return True
    k_npy = os.path.join(DEPTH_BASE, seq_id, "K.npy")
    if not os.path.exists(k_npy):
        return False
    K = np.load(k_npy)
    if K.ndim == 1 and K.shape[0] == 9:
        K = K.reshape(3, 3)
    if K.ndim != 2 or K.shape != (3, 3):
        return False
    with open(k_path, "w") as f:
        for row in K:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
    return True


def copy_masks(task: str, masks_dir: str, n_frames: int):
    """
    Copy per-frame masks from obj_recon_input (first-frame mask → all frames).
    BundleSDF propagates tracking from frame 0 — we only need a good frame-0 mask.
    For frames 1+, use full-foreground as fallback (BundleSDF handles it).
    """
    import cv2
    os.makedirs(masks_dir, exist_ok=True)
    if len(glob(os.path.join(masks_dir, "*.png"))) >= n_frames:
        return True

    init_mask_path = os.path.join(MASK_BASE, task, "0.png")
    if not os.path.exists(init_mask_path):
        # Fallback: white mask for all frames
        init_mask = None
    else:
        init_mask = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)

    for i in range(n_frames):
        out_path = os.path.join(masks_dir, f"{i:06d}.png")
        if os.path.exists(out_path):
            continue
        if i == 0 and init_mask is not None:
            cv2.imwrite(out_path, init_mask)
        else:
            # White mask for frames 1+ — BundleSDF will refine
            if init_mask is not None:
                white = np.ones_like(init_mask) * 255
            else:
                white = np.ones((480, 640), dtype=np.uint8) * 255
            cv2.imwrite(out_path, white)
    return True


def prepare_video_dir(seq_id: str, task: str, ep: str, staging_dir: str) -> int:
    """Prepare staging dir. Returns n_frames, or 0 on failure."""
    video_path = os.path.join(EGODEX_ROOT, task, f"{ep}.mp4")
    if not os.path.exists(video_path):
        return 0

    rgb_dir   = os.path.join(staging_dir, "rgb")
    depth_dir = os.path.join(staging_dir, "depth")
    masks_dir = os.path.join(staging_dir, "masks")
    k_path    = os.path.join(staging_dir, "cam_K.txt")

    # RGB first to get n_frames
    ok = extract_rgb_frames(video_path, rgb_dir, MAX_FRAMES)
    if not ok:
        return 0

    n_frames = len(glob(os.path.join(rgb_dir, "*.png")))
    if n_frames < 4:
        return 0

    ok_d = write_megasam_depth(seq_id, depth_dir, n_frames)
    ok_k = write_cam_K(seq_id, k_path)
    copy_masks(task, masks_dir, n_frames)

    if not ok_d or not ok_k:
        return 0

    return n_frames


# ── Step 2: run BundleSDF ──────────────────────────────────────────────────────

def run_bundlesdf(staging_dir: str, cache_dir: str, timeout: int = 1800) -> tuple:
    """Run BundleSDF in subprocess under bundlesdf conda env."""
    os.makedirs(cache_dir, exist_ok=True)
    cmd = [
        "conda", "run", "--no-capture-output", "-n", CONDA_ENV,
        "python", os.path.join(BUNDLESDF_DIR, "run_custom.py"),
        "--mode", "run_video",
        "--video_dir", staging_dir,
        "--out_folder", cache_dir,
        "--use_segmenter", "0",
        "--debug_level", "0",
    ]
    try:
        result = subprocess.run(
            cmd, cwd=BUNDLESDF_DIR,
            timeout=timeout,
            capture_output=True, text=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


# ── Step 3: copy poses out ────────────────────────────────────────────────────

def collect_poses(cache_dir: str, out_pose_dir: str) -> int:
    """Copy ob_in_cam*.txt files from BundleSDF cache → final output dir."""
    # BundleSDF writes poses to cache_dir/ob_in_cam/
    src_dir = os.path.join(cache_dir, "ob_in_cam")
    if not os.path.exists(src_dir):
        # Sometimes nested under timestamp dir
        candidates = glob(os.path.join(cache_dir, "*", "ob_in_cam"))
        if candidates:
            src_dir = candidates[0]
        else:
            return 0

    os.makedirs(out_pose_dir, exist_ok=True)
    pose_files = natsorted(glob(os.path.join(src_dir, "*.txt")))
    for pf in pose_files:
        shutil.copy2(pf, os.path.join(out_pose_dir, os.path.basename(pf)))
    return len(pose_files)


# ── Main ──────────────────────────────────────────────────────────────────────

def process_one(entry_key: str, entry: dict, args) -> tuple:
    """Process a single sequence. Returns (status, key, msg)."""
    dataset = entry["dataset"]
    seq_id  = entry["seq_id"]        # "add_remove_lid/22"
    task, ep = seq_id.rsplit("/", 1) # "add_remove_lid", "22"

    out_pose_dir = os.path.join(OUT_POSE_BASE, f"{task}__{ep}", "ob_in_cam")

    # Skip if already done
    existing = glob(os.path.join(out_pose_dir, "*.txt"))
    if len(existing) >= 4 and not args.redo:
        return "skip", entry_key, f"{len(existing)} poses cached"

    staging_dir = os.path.join(STAGING_BASE, f"{task}__{ep}")
    cache_dir   = os.path.join(staging_dir, "_bundlesdf_cache")

    # Step 1: prepare
    n_frames = prepare_video_dir(seq_id, task, ep, staging_dir)
    if n_frames == 0:
        return "fail", entry_key, "prepare failed (no video or depth)"

    # Step 2: run BundleSDF
    ok = run_bundlesdf(staging_dir, cache_dir, timeout=args.timeout)
    if not ok:
        return "fail", entry_key, "BundleSDF returned non-zero"

    # Step 3: collect poses
    n_poses = collect_poses(cache_dir, out_pose_dir)
    if n_poses < 2:
        return "fail", entry_key, f"only {n_poses} pose files found"

    # Optional: clean staging to save disk
    if args.clean:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return "ok", entry_key, f"{n_poses} poses"


def main():
    p = argparse.ArgumentParser(description="Batch BundleSDF for EgoDex")
    p.add_argument("--seq",     default="", help="Filter: only sequences whose seq_id contains this")
    p.add_argument("--start",   type=int, default=0)
    p.add_argument("--end",     type=int, default=0, help="0 = all")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (be careful: GPU memory)")
    p.add_argument("--timeout", type=int, default=1800, help="Per-sequence timeout (s)")
    p.add_argument("--redo",    action="store_true", help="Re-run even if poses exist")
    p.add_argument("--clean",   action="store_true", help="Delete staging dirs after success")
    args = p.parse_args()

    with open(REGISTRY_JSON) as f:
        registry = json.load(f)

    # Filter skipped
    entries = [(k, v) for k, v in registry.items()
               if not v.get("skipped", False) and v.get("dataset") == "egodex"]

    if args.seq:
        entries = [(k, v) for k, v in entries if args.seq in v["seq_id"]]

    total = len(entries)
    end = args.end if args.end > 0 else total
    entries = entries[args.start:end]

    print(f"\n{'='*60}")
    print(f" BundleSDF EgoDex Batch")
    print(f" Registry: {total} sequences  →  processing {len(entries)}")
    print(f" Workers:  {args.workers}  Timeout: {args.timeout}s")
    print(f"{'='*60}\n")

    done = skipped = failed = 0
    pbar = tqdm(total=len(entries), desc="BundleSDF/egodex")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, k, v, args): k for k, v in entries}
        for fut in as_completed(futures):
            status, key, msg = fut.result()
            seq_id = registry[key]["seq_id"]
            if status == "skip":
                tqdm.write(f"  ⏭  {seq_id}: {msg}")
                skipped += 1
            elif status == "ok":
                tqdm.write(f"  ✅ {seq_id}  {msg}")
                done += 1
            else:
                tqdm.write(f"  ❌ {seq_id}: {msg}")
                failed += 1
            pbar.update(1)
    pbar.close()

    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ⏭ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"Output: {OUT_POSE_BASE}")


if __name__ == "__main__":
    main()
