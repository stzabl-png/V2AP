"""
Universal HaPTIC batch inference — supports all third-person datasets.

Usage:
  conda activate haptic
  python data/batch_haptic.py --dataset arctic
  python data/batch_haptic.py --dataset oakink
  python data/batch_haptic.py --dataset ho3d_v3
  python data/batch_haptic.py --dataset dexycb
  python data/batch_haptic.py --dataset selfmade
  python data/batch_haptic.py --dataset dexycb --seq 20200709-subject-01  # single subject

Output:
  data_hub/ProcessedData/third_mano/{dataset}/{seq_id}.npz
    Contains: verts_dict {frame_idx → (778,3) MANO vertices in camera space}
"""

import os, sys, argparse, numpy as np, torch
from glob import glob
from natsort import natsorted
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
sys.path.insert(0, config.HAPTIC_DIR)

# Reuse the generic HaPTIC inference function from batch_haptic_arctic
from data.batch_haptic_arctic import load_haptic_model, run_haptic_on_images
from data.megasam_utils import K_as_haptic_intrinsics

# ── Output directory ──────────────────────────────────────────────────────────
OUT_BASE   = os.path.join(config.DATA_HUB, "ProcessedData", "mano", "ThirdPerson")
RAW_BASE   = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")
DEPTH_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "depth", "ThirdPerson")
device = "cuda:0"


def load_depth_pro_K(dataset, seq_id):
    """Load Depth Pro estimated intrinsics for a sequence.
    Returns 3×3 K array or None if not available.
    """
    k_path = os.path.join(DEPTH_BASE, dataset, seq_id, "K.txt")
    if os.path.exists(k_path):
        try:
            return np.loadtxt(k_path)
        except Exception:
            pass
    return None


# ── Dataset-specific sequence discoverers ─────────────────────────────────────

def discover_arctic(input_dir):
    """ARCTIC cam1: {subj}/{seq}/1/*.jpg"""
    for subj in natsorted(os.listdir(input_dir)):
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for seq in natsorted(os.listdir(subj_dir)):
            cam_dir = os.path.join(subj_dir, seq, "1")
            if not os.path.isdir(cam_dir):
                continue
            imgs = natsorted(glob(os.path.join(cam_dir, "*.jpg")) +
                             glob(os.path.join(cam_dir, "*.png")))
            if imgs:
                yield f"{subj}__{seq}", imgs, "frames"


def discover_oakink(input_dir):
    """OakInk v1: {seq_id}/{timestamp}/north_west_color_*.png"""
    for seq_id in natsorted(os.listdir(input_dir)):
        seq_dir = os.path.join(input_dir, seq_id)
        if not os.path.isdir(seq_dir):
            continue
        imgs = natsorted(
            glob(os.path.join(seq_dir, "*", "north_west_color_*.png")) +
            glob(os.path.join(seq_dir, "north_west_color_*.png"))
        )
        if imgs:
            yield seq_id, imgs, "frames"


def discover_ho3d(input_dir):
    """HO3D v3: train/{seq}/rgb/*.jpg  +  evaluation/{seq}/rgb/*.jpg"""
    for split in ["train", "evaluation"]:
        split_dir = os.path.join(input_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for seq_id in natsorted(os.listdir(split_dir)):
            rgb_dir = os.path.join(split_dir, seq_id, "rgb")
            if not os.path.isdir(rgb_dir):
                continue
            imgs = natsorted(glob(os.path.join(rgb_dir, "*.jpg")))
            if imgs:
                yield f"{split}__{seq_id}", imgs, "frames"


def discover_dexycb(input_dir):
    """DexYCB: {subject}/{datetime}/{serial}/color_*.jpg"""
    for subj in natsorted(os.listdir(input_dir)):
        if subj.startswith('.'):   # BUG-04: 过滤 .cache 等隐藏目录
            continue
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for dt in natsorted(os.listdir(subj_dir)):
            dt_dir = os.path.join(subj_dir, dt)
            if not os.path.isdir(dt_dir):
                continue
            for serial in natsorted(os.listdir(dt_dir)):
                cam_dir = os.path.join(dt_dir, serial)
                if not os.path.isdir(cam_dir):
                    continue
                imgs = natsorted(glob(os.path.join(cam_dir, "color_*.jpg")))
                if imgs:
                    yield f"{subj}__{dt}__{serial}", imgs, "frames"


def discover_selfmade(input_dir):
    """Selfmade: {seq_id}/*.jpg|*.png  OR  {seq_id}.mp4 flat in input_dir"""
    # Case 1: flat MP4 files directly in input_dir
    mp4s = natsorted(glob(os.path.join(input_dir, "*.mp4")))
    for mp4 in mp4s:
        seq_id = os.path.splitext(os.path.basename(mp4))[0]
        yield seq_id, [mp4], "video"   # signal that it's a video

    # Case 2: subdirectories with frames
    for seq_id in natsorted(os.listdir(input_dir)):
        seq_dir = os.path.join(input_dir, seq_id)
        if not os.path.isdir(seq_dir):
            continue
        imgs = natsorted(
            glob(os.path.join(seq_dir, "*.jpg")) +
            glob(os.path.join(seq_dir, "*.png")) +
            glob(os.path.join(seq_dir, "*.jpeg"))
        )
        if imgs:
            yield seq_id, imgs, "frames"


def discover_taco_allocentric(input_dir):
    """TACO Allocentric: Allocentric_RGB_Videos/{triplet}/{session}/{cam_serial}/*.jpg
    NOTE: Company must download Allocentric_RGB_Videos from TACO dataset page.
    Camera params (K, R, T) are in Allocentric_Camera_Parameters/{triplet}/{session}/calibration.json
    """
    if not os.path.isdir(input_dir):
        return
    for triplet in natsorted(os.listdir(input_dir)):
        triplet_dir = os.path.join(input_dir, triplet)
        if not os.path.isdir(triplet_dir):
            continue
        for session in natsorted(os.listdir(triplet_dir)):
            session_dir = os.path.join(triplet_dir, session)
            if not os.path.isdir(session_dir):
                continue
            for cam_serial in natsorted(os.listdir(session_dir)):
                cam_dir = os.path.join(session_dir, cam_serial)
                if not os.path.isdir(cam_dir):
                    continue
                imgs = natsorted(glob(os.path.join(cam_dir, "*.jpg")) +
                                 glob(os.path.join(cam_dir, "*.png")))
                if imgs:
                    safe_triplet = triplet.replace("(", "").replace(")", "").replace(", ", "_")
                    yield f"{safe_triplet}__{session}__{cam_serial}", imgs, "frames"


DISCOVERERS = {
    "arctic":           (discover_arctic,           "arctic"),
    "oakink":           (discover_oakink,           "oakink_v1"),
    "ho3d_v3":          (discover_ho3d,             "ho3d_v3"),
    "dexycb":           (discover_dexycb,           "dexycb"),
    "selfmade":         (discover_selfmade,         "selfmade"),
    "taco_allocentric": (discover_taco_allocentric, "taco/Allocentric_RGB_Videos"),
}


def video_to_frames(video_path, max_frames=300):
    """Extract frames from MP4 to a temp dir, return (img_paths, tmp_dir)."""
    import tempfile, subprocess
    tmp_dir = tempfile.mkdtemp(prefix="haptic_selfmade_")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select=not(mod(n\,2))",  # take every 2nd frame
        "-vsync", "vfr",
        os.path.join(tmp_dir, "%06d.jpg")
    ], capture_output=True)
    imgs = natsorted(glob(os.path.join(tmp_dir, "*.jpg")))
    if len(imgs) > max_frames:
        step = len(imgs) // max_frames
        imgs = imgs[::step][:max_frames]
    return imgs, tmp_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Universal HaPTIC batch inference")
    parser.add_argument("--dataset",   required=True, choices=list(DISCOVERERS.keys()),
                        help="Which dataset to process")
    parser.add_argument("--input-dir", default=None,
                        help="Override input directory (default: RawData/ThirdPersonRawData/{folder})")
    parser.add_argument("--seq",       default=None,
                        help="Process only this sequence (substring match)")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Max frames per sequence to keep (default 300)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process first N sequences only (for smoke tests)")
    parser.add_argument("--start", type=int, default=0, help="Shard start index (inclusive)")
    parser.add_argument("--end",   type=int, default=0, help="Shard end index (exclusive), 0=all remaining")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip sequences that already have cached output")
    parser.add_argument("--only-with-depth-k", action="store_true", default=False,
                        help="Only process sequences that have a Depth Pro K.txt cached. "
                             "Useful to avoid fallback heuristic K for unprepared sequences.")
    args = parser.parse_args()

    discover_fn, data_folder = DISCOVERERS[args.dataset]
    input_dir  = args.input_dir or os.path.join(RAW_BASE, data_folder)
    output_dir = os.path.join(OUT_BASE, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f" Dataset   : {args.dataset}")
    print(f" Input     : {input_dir}")
    print(f" Output    : {output_dir}")
    print(f"{'='*60}")

    if not os.path.isdir(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return

    # Discover all sequences
    sequences = list(discover_fn(input_dir))
    if args.seq:
        sequences = [(sid, p, t) for sid, p, t in sequences if args.seq in sid]
    if getattr(args, 'only_with_depth_k', False):
        before = len(sequences)
        sequences = [(sid, p, t) for sid, p, t in sequences
                     if os.path.exists(os.path.join(DEPTH_BASE, args.dataset, sid, 'K.txt'))]
        print(f"  [--only-with-depth-k] {before} → {len(sequences)} sequences (have K.txt)")
    if args.limit > 0:
        sequences = sequences[:args.limit]
    if args.start > 0 or args.end > 0:
        end = args.end if args.end > 0 else len(sequences)
        sequences = sequences[args.start:end]
        print(f"[Shard] sequences [{args.start}:{end}] = {len(sequences)}")

    print(f"Found {len(sequences)} sequences\n")

    print("Loading HaPTIC model...")
    _orig_cwd = os.getcwd()
    os.chdir(config.HAPTIC_DIR)   # stay here for all HaPTIC ops
    model, model_cfg = load_haptic_model()
    print("✅ Model loaded\n")

    done, skipped, failed = 0, 0, 0

    for seq_id, paths, src_type in tqdm(sequences, desc=f"HaPTIC/{args.dataset}"):
        out_path = os.path.join(output_dir, f"{seq_id}.npz")

        if args.skip_existing and os.path.exists(out_path):
            tqdm.write(f"  ⏭  {seq_id}: cached")
            skipped += 1
            continue

        tmp_dir = None
        try:
            # Handle video files (extract frames first)
            if src_type == "video":
                img_paths, tmp_dir = video_to_frames(paths[0], args.max_frames)
                tqdm.write(f"  🎬 {seq_id}: extracted {len(img_paths)} frames from MP4")
            else:
                img_paths = paths
                # Subsample if very long sequence
                if len(img_paths) > args.max_frames:
                    step = len(img_paths) // args.max_frames
                    img_paths = img_paths[::step][:args.max_frames]

            # ── Depth Pro intrinsics → HaPTIC (ensures same metric space) ─────
            # Depth Pro estimates K per-sequence; pass to HaPTIC so MANO
            # vertices are in the same camera coordinate system as FP poses.
            dp_K = load_depth_pro_K(args.dataset, seq_id)
            if dp_K is not None:
                haptic_K = K_as_haptic_intrinsics(dp_K)
                tqdm.write(f"  [K] Using Depth Pro K: fx={dp_K[0,0]:.1f}")
            else:
                haptic_K = None  # HaPTIC uses sqrt(W²+H²) heuristic

            verts_dict = run_haptic_on_images(
                model, model_cfg, img_paths,
                scene_name=seq_id if haptic_K is None else None,
                precomputed_K=haptic_K,
            )

            if not verts_dict:
                tqdm.write(f"  ⚠️  {seq_id}: no hand detected")
                failed += 1
                continue

            np.savez_compressed(out_path, verts_dict=verts_dict)
            tqdm.write(f"  ✅ {seq_id}: {len(verts_dict)} frames → {os.path.basename(out_path)}")
            done += 1

        except Exception as e:
            tqdm.write(f"  ❌ {seq_id}: {e}")
            failed += 1

        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

        torch.cuda.empty_cache()

    os.chdir(_orig_cwd)


    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ⏭ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
