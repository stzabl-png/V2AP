"""
batch_hawor.py – Batch HaWoR hand-prior generation.

Each sequence is processed in a fresh subprocess (conda run -n hawor python run_hawor_seq.py),
so there are no shared CUDA contexts, no multiprocessing conflicts, and failures are isolated.

Usage examples:
    # All egodex sequences
    python data/batch_hawor.py --dataset egodex

    # Filter by substring
    python data/batch_hawor.py --dataset egodex --seq "slot_batteries/1"

    # Limit frames (for frame-based datasets like arctic)
    python data/batch_hawor.py --dataset arctic --max-frames 120

    # Force re-run even if .npz exists
    python data/batch_hawor.py --dataset egodex --force-rerun
"""
import os, sys, argparse, subprocess, tempfile, shutil, json
from glob import glob
from natsort import natsorted
from tqdm import tqdm

# ── project config ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

HAWOR_DIR   = os.environ.get("HAWOR_DIR",
              os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "thirdparty", "hawor"))
# Canonical source of run_hawor_seq.py is data/ (tracked in main project git).
# We auto-copy it to HAWOR_DIR before each run so imports resolve correctly.
_RUNNER_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_hawor_seq.py")
RUNNER      = os.path.join(HAWOR_DIR, "run_hawor_seq.py")
CONDA_ENV   = "hawor"


def _sync_runner():
    """Copy data/run_hawor_seq.py → third_party/hawor/ if missing or outdated."""
    import shutil
    if not os.path.exists(_RUNNER_SRC):
        raise FileNotFoundError(
            f"run_hawor_seq.py not found at {_RUNNER_SRC}. "
            "Please verify your git clone is up to date (git pull).")
    if (not os.path.exists(RUNNER) or
            os.path.getmtime(_RUNNER_SRC) > os.path.getmtime(RUNNER)):
        shutil.copy2(_RUNNER_SRC, RUNNER)
        print(f"[batch_hawor] synced run_hawor_seq.py → {RUNNER}")

_sync_runner()

OUT_BASE    = os.path.join(config.DATA_HUB, "ProcessedData", "mano", "Egocentric")

# Egocentric raw data root (also symlinked from data/egocentric/)
EGO_BASE    = os.path.join(config.DATA_HUB, "RawData", "EgoRawData")
THIRD_BASE  = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")

EGODEX_ROOT   = os.path.join(EGO_BASE, "egodex", "test")
TACO_EGO_ROOT = os.path.join(EGO_BASE, "taco",   "Egocentric_RGB_Videos")
HOI4D_ROOT    = os.path.join(EGO_BASE, "hoi4d",  "HOI4D_release")


# ── dataset discoverers ────────────────────────────────────────────────────────

def discover_egodex(input_dir):
    """EgoDex: test/{task}/{episode}.mp4  →  (seq_id, mp4_path, 'video')"""
    for task in natsorted(os.listdir(input_dir)):
        task_dir = os.path.join(input_dir, task)
        if not os.path.isdir(task_dir):
            continue
        for mp4 in natsorted(glob(os.path.join(task_dir, "*.mp4"))):
            stem = os.path.splitext(os.path.basename(mp4))[0]
            yield f"egodex/{task}/{stem}", mp4, "video"


def discover_arctic_ego(input_dir):
    """ARCTIC cam0: {subj}/{seq}/0/*.jpg  →  (seq_id, [frames], 'frames')"""
    for subj in natsorted(os.listdir(input_dir)):
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for seq in natsorted(os.listdir(subj_dir)):
            cam_dir = os.path.join(subj_dir, seq, "0")
            if not os.path.isdir(cam_dir):
                continue
            imgs = natsorted(glob(os.path.join(cam_dir, "*.jpg")) +
                             glob(os.path.join(cam_dir, "*.png")))
            if imgs:
                yield f"arctic/{subj}/{seq}", imgs, "frames"

def discover_taco_ego(input_dir):
    """TACO Ego: Egocentric_RGB_Videos/{triplet}/{seq}/color.mp4"""
    if not os.path.isdir(input_dir):
        return
    for triplet in natsorted(os.listdir(input_dir)):
        triplet_dir = os.path.join(input_dir, triplet)
        if not os.path.isdir(triplet_dir):
            continue
        for seq in natsorted(os.listdir(triplet_dir)):
            mp4 = os.path.join(triplet_dir, seq, "color.mp4")
            if os.path.exists(mp4):
                yield f"taco_ego/{triplet}/{seq}", mp4, "video"


def discover_hoi4d(input_dir):
    """HOI4D: HOI4D_release/ZY.../H.../align_rgb/image.mp4
    seq_id = hoi4d/ZY.../H1/C1/N19/S100/s02/T1  (path relative to HOI4D_release)
    """
    if not os.path.isdir(input_dir):
        return
    from glob import glob as _glob
    for mp4 in natsorted(_glob(os.path.join(input_dir, "**", "image.mp4"), recursive=True)):
        rel = os.path.relpath(mp4, input_dir)          # ZY.../H.../align_rgb/image.mp4
        seq = rel.replace("/align_rgb/image.mp4", "")  # ZY.../H.../
        yield f"hoi4d/{seq}", mp4, "video"


DISCOVERERS = {
    "egodex":   (discover_egodex,   EGODEX_ROOT),
    "taco_ego": (discover_taco_ego, TACO_EGO_ROOT),
    "arctic":   (discover_arctic_ego, os.path.join(EGO_BASE, "arctic", "images")),
    "hoi4d":    (discover_hoi4d,    HOI4D_ROOT),
}


# ── helpers ────────────────────────────────────────────────────────────────────

def frames_to_temp_video(img_paths, fps=15):
    """Encode a list of image paths into a temp mp4. Returns (mp4_path, tmp_dir)."""
    tmp_dir = tempfile.mkdtemp(prefix="hawor_tmp_")
    ext = os.path.splitext(img_paths[0])[1]
    for i, p in enumerate(img_paths):
        os.symlink(os.path.abspath(p), os.path.join(tmp_dir, f"{i:06d}{ext}"))
    out_mp4 = os.path.join(tmp_dir, "video.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", os.path.join(tmp_dir, f"%06d{ext}"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", out_mp4],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out_mp4, tmp_dir


def load_megasam_focal(seq_id, dataset):
    """Try to load focal length from MegaSAM K.npy output.

    MegaSAM stores HOI4D outputs with underscores:
      egocentric_depth/hoi4d/ZY20210800001_H1_C1_N19_S100_s02_T1/K.npy
    HaWoR seq_id uses slashes:
      hoi4d/ZY20210800001/H1/C1/N19/S100/s02/T1
    """
    try:
        import numpy as np
        parts = seq_id.split("/")        # ["hoi4d", "ZY...", "H1", ...]
        rel_slash = "/".join(parts[1:])  # "ZY.../H1/..."
        # MegaSAM uses underscores for HOI4D seq ids
        rel_under = rel_slash.replace("/", "_")

        ego_depth_base = os.path.join(config.DATA_HUB, "ProcessedData", "egocentric_depth")
        for rel_key in [rel_slash, rel_under]:
            k_path = os.path.join(ego_depth_base, dataset, rel_key, "K.npy")
            if os.path.exists(k_path):
                K = np.load(k_path)
                if K.ndim == 2:
                    return float((K[0, 0] + K[1, 1]) / 2)
        return None
    except Exception:
        return None


def run_sequence(video_path, out_npz, img_focal=None, force_rerun=False,
                 timeout=3600):
    """
    Call run_hawor_seq.py in a subprocess inside the 'hawor' conda environment.
    Returns (success: bool, message: str).
    """
    cmd = [
        "conda", "run", "--no-capture-output", "-n", CONDA_ENV,
        "python", RUNNER,
        "--video_path", video_path,
        "--out_npz",    out_npz,
        "--hawor_dir",  HAWOR_DIR,   # always explicit so script resolves paths correctly
    ]
    if img_focal is not None:
        cmd += ["--img_focal", str(img_focal)]
    if force_rerun:
        cmd += ["--force_rerun"]

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, "ok"
        else:
            # Show last 10 lines of stderr for diagnostics
            err_tail = "\n".join(result.stderr.strip().splitlines()[-10:])
            return False, err_tail
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except Exception as e:
        return False, str(e)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch HaWoR MANO prior generation")
    p.add_argument("--dataset",      required=True, choices=list(DISCOVERERS))
    p.add_argument("--seq",          default="",
                   help="Filter: only process sequences whose ID contains this string")
    p.add_argument("--seqs",         default="",
                   help="Comma-separated list of exact rel seq IDs, e.g. slot_batteries/1,fry_bread/0")
    p.add_argument("--focal",        type=float, default=None,
                   help="Fixed focal length (px) for all sequences; overrides MegaSAM per-seq focal")
    p.add_argument("--max-frames",   type=int, default=0,
                   help="Max frames to use (for frame-based datasets); 0 = all")
    p.add_argument("--fps",          type=int, default=15,
                   help="FPS for frame→video encoding (frame-based datasets)")
    p.add_argument("--force-rerun",  action="store_true",
                   help="Re-run even if output .npz already exists")
    p.add_argument("--timeout",      type=int, default=3600,
                   help="Per-sequence subprocess timeout in seconds")
    p.add_argument("--start",        type=int, default=0, help="Shard start index (inclusive)")
    p.add_argument("--end",          type=int, default=0, help="Shard end index (exclusive), 0=all remaining")
    return p.parse_args()


def main():
    args = parse_args()

    discover_fn, data_folder = DISCOVERERS[args.dataset]
    output_dir = os.path.join(OUT_BASE, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f" Dataset   : {args.dataset}")
    print(f" Input     : {data_folder}")
    print(f" Output    : {output_dir}")
    print(f" Runner    : {RUNNER}")
    print(f" Env       : {CONDA_ENV}")
    print("=" * 60)

    # Discover all sequences
    sequences = list(discover_fn(data_folder))
    if args.seqs:
        # exact rel IDs: "slot_batteries/1,fry_bread/0" → filter by dataset/rel
        wanted = set(s.strip() for s in args.seqs.split(",") if s.strip())
        sequences = [(sid, imgs, t) for sid, imgs, t in sequences
                     if "/".join(sid.split("/")[1:]) in wanted]
    elif args.seq:
        sequences = [(sid, imgs, t) for sid, imgs, t in sequences if args.seq in sid]
    if args.start > 0 or args.end > 0:
        end = args.end if args.end > 0 else len(sequences)
        sequences = sequences[args.start:end]
        print(f"[Shard] sequences [{args.start}:{end}] = {len(sequences)}")
    print(f"Found {len(sequences)} sequences\n")

    done = skipped = failed = 0

    for seq_id, img_src, src_type in tqdm(sequences, desc=f"HaWoR/{args.dataset}"):
        # Output path: OUT_BASE/dataset/{seq_id_without_dataset_prefix}.npz
        # seq_id format: "egodex/slot_batteries/1"  →  strip first component
        rel = "/".join(seq_id.split("/")[1:])  # "slot_batteries/1"
        out_npz = os.path.join(output_dir, f"{rel}.npz")

        if os.path.exists(out_npz) and not args.force_rerun:
            tqdm.write(f"  ⏭  {seq_id}: cached")
            skipped += 1
            continue

        # Resolve video path
        tmp_dir = None
        try:
            if src_type == "video":
                video_path = img_src  # already an mp4 path

                # HOI4D videos are 1920×1080 which triggers a DROID-SLAM shape bug.
                # Resize to 1280×720 via ffmpeg to avoid RuntimeError in upsample.
                if args.dataset == "hoi4d":
                    import cv2 as _cv2
                    cap = _cv2.VideoCapture(video_path)
                    vw  = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
                    cap.release()
                    if vw > 1280:
                        tmp_dir = tempfile.mkdtemp(prefix="hawor_resize_")
                        resized_mp4 = os.path.join(tmp_dir, "video_720p.mp4")
                        subprocess.run([
                            "ffmpeg", "-y", "-i", video_path,
                            "-vf", "scale=1280:720",
                            "-c:v", "libx264", "-crf", "18",
                            "-preset", "fast", resized_mp4
                        ], capture_output=True, check=True)
                        video_path = resized_mp4
                        tqdm.write(f"  🎬 {seq_id}: mp4 (resized 1920→1280)")
                    else:
                        tqdm.write(f"  🎬 {seq_id}: mp4")
                else:
                    tqdm.write(f"  🎬 {seq_id}: mp4")
            else:
                # Frame list → encode to temp video
                frames = img_src
                if args.max_frames > 0 and len(frames) > args.max_frames:
                    step   = len(frames) // args.max_frames
                    frames = frames[::step][: args.max_frames]
                tqdm.write(f"  🎬 {seq_id}: {len(frames)} frames → temp video")
                video_path, tmp_dir = frames_to_temp_video(frames, fps=args.fps)

            # Focal length: --focal flag > MegaSAM K.npy > HaWoR auto-estimate
            if args.focal is not None:
                img_focal = args.focal
                tqdm.write(f"  [focal] {img_focal:.1f}px (fixed)")
            else:
                img_focal = load_megasam_focal(seq_id, args.dataset)
                if img_focal:
                    tqdm.write(f"  [focal] {img_focal:.1f}px from MegaSAM")

            ok, msg = run_sequence(
                video_path=video_path,
                out_npz=out_npz,
                img_focal=img_focal,
                force_rerun=args.force_rerun,
                timeout=args.timeout,
            )

            if ok and os.path.exists(out_npz):
                tqdm.write(f"  ✅ {seq_id} → {out_npz}")
                done += 1
            else:
                tqdm.write(f"  ❌ {seq_id}: {msg}")
                failed += 1

        except Exception as e:
            import traceback
            tqdm.write(f"  ❌ {seq_id}: {e}")
            tqdm.write(traceback.format_exc())
            failed += 1
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ⏭ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
