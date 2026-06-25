"""
Universal Depth Pro batch inference — generates metric depth maps for all datasets.

Usage:
  conda activate depth-pro
  # Pass 1: estimate fx per sequence (fast, 30 frames each)
  python data/batch_depth_pro.py --dataset ho3d_v3 --max-frames 30

  # Compute global median fx from Pass 1 results (no GPU needed)
  python data/batch_depth_pro.py --dataset ho3d_v3 --compute-global-fx

  # Pass 2: inject calibrated fx, rerun full depth estimation
  python data/batch_depth_pro.py --dataset ho3d_v3 --fixed-fx 605.3 --redo

  # Single camera / single sequence
  python data/batch_depth_pro.py --dataset dexycb --cam 841412060263
  python data/batch_depth_pro.py --dataset ho3d_v3 --seq ABF10

Output per sequence:
  data_hub/ProcessedData/third_depth/{dataset}/{seq_id}/
    depths.npz      — depth maps, shape (N, H, W), float32, metric metres
    K.txt           — 3x3 camera intrinsic matrix (estimated or fixed fx)
    frame_ids.txt   — original frame filenames in order

Two-pass self-calibration (no GT required):
  Pass 1 estimates fx per-sequence. Global median across all sequences
  converges to ground truth (verified: HO3D -1.5%, DexYCB +0.2%).
  Pass 2 injects this median fx so Depth Pro only estimates depth scale.
"""

import os, sys, argparse, numpy as np, torch
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Depth Pro lives in third_party/ml-depth-pro
DEPTH_PRO_DIR = os.path.join(config.PROJECT_DIR, "third_party", "ml-depth-pro", "src")
sys.path.insert(0, DEPTH_PRO_DIR)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "depth", "ThirdPerson")
RAW_BASE = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")


# ── Dataset discoverers (same as batch_haptic.py) ────────────────────────────

def discover_arctic(input_dir):
    for subj in natsorted(os.listdir(input_dir)):
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir): continue
        for seq in natsorted(os.listdir(subj_dir)):
            cam_dir = os.path.join(subj_dir, seq, "1")
            if not os.path.isdir(cam_dir): continue
            imgs = natsorted(glob(os.path.join(cam_dir, "*.jpg")) +
                             glob(os.path.join(cam_dir, "*.png")))
            if imgs:
                yield f"{subj}__{seq}", imgs


def discover_oakink(input_dir):
    for seq_id in natsorted(os.listdir(input_dir)):
        seq_dir = os.path.join(input_dir, seq_id)
        if not os.path.isdir(seq_dir): continue
        imgs = natsorted(glob(os.path.join(seq_dir, "*", "north_west_color_*.png")) +
                         glob(os.path.join(seq_dir, "north_west_color_*.png")))
        if imgs:
            yield seq_id, imgs


def discover_ho3d(input_dir):
    for split in ["train", "evaluation"]:
        split_dir = os.path.join(input_dir, split)
        if not os.path.isdir(split_dir): continue
        for seq_id in natsorted(os.listdir(split_dir)):
            rgb_dir = os.path.join(split_dir, seq_id, "rgb")
            if not os.path.isdir(rgb_dir): continue
            imgs = natsorted(glob(os.path.join(rgb_dir, "*.jpg")))
            if imgs:
                yield f"{split}__{seq_id}", imgs


def discover_dexycb(input_dir, cam_filter=None):
    """Yield (seq_id, img_paths) for DexYCB.
    cam_filter: if set, only yield sequences matching this camera serial.
    """
    for subj in natsorted(os.listdir(input_dir)):
        if subj.startswith('.'):   # BUG-04: 过滤 .cache 等隐藏目录
            continue
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir): continue
        for dt in natsorted(os.listdir(subj_dir)):
            dt_dir = os.path.join(subj_dir, dt)
            if not os.path.isdir(dt_dir): continue
            for serial in natsorted(os.listdir(dt_dir)):
                if cam_filter and serial not in cam_filter:
                    continue
                cam_dir = os.path.join(dt_dir, serial)
                if not os.path.isdir(cam_dir): continue
                imgs = natsorted(glob(os.path.join(cam_dir, "color_*.jpg")))
                if imgs:
                    yield f"{subj}__{dt}__{serial}", imgs


def discover_taco_allocentric(input_dir):
    """TACO Allocentric: Allocentric_RGB_Videos/{triplet}/{session}/{cam_serial}/*.jpg
    Structure mirrors Allocentric_Camera_Parameters:
      Allocentric_Camera_Parameters/(action, tool, object)/{session}/calibration.json
    Videos expected at: Allocentric_RGB_Videos/(action, tool, object)/{session}/{cam_serial}/
    NOTE: Company must download Allocentric_RGB_Videos separately from TACO dataset.
    """
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
                    # Replace special chars in triplet name for safe seq_id
                    safe_triplet = triplet.replace("(", "").replace(")", "").replace(", ", "_")
                    yield f"{safe_triplet}__{session}__{cam_serial}", imgs


DISCOVERERS = {
    "arctic":           (discover_arctic,           "arctic"),
    "oakink":           (discover_oakink,           "oakink_v1"),
    "ho3d_v3":          (discover_ho3d,             "ho3d_v3"),
    "dexycb":           (discover_dexycb,           "dexycb"),
    "taco_allocentric": (discover_taco_allocentric, "taco/Allocentric_RGB_Videos"),
}


# ── Depth Pro inference ───────────────────────────────────────────────────────

def load_depth_pro_model():
    import depth_pro
    model, transform = depth_pro.create_model_and_transforms(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        precision=torch.float16
    )
    model.eval()
    return model, transform


def compute_global_fx(output_dir):
    """Read all existing K.txt files in output_dir and return global median fx."""
    fx_list = []
    for seq in natsorted(os.listdir(output_dir)):
        k_path = os.path.join(output_dir, seq, 'K.txt')
        if os.path.exists(k_path):
            try:
                K = np.loadtxt(k_path)
                fx_list.append(float(K[0, 0]))
            except Exception:
                pass
    if not fx_list:
        return None
    arr = np.array(fx_list)
    med = float(np.median(arr))
    print(f"  Sequences with K.txt: {len(fx_list)}")
    print(f"  fx  min={arr.min():.1f}  median={med:.1f}  max={arr.max():.1f}")
    print(f"  Recommended --fixed-fx: {med:.1f}")
    return med


def run_depth_pro(model, transform, img_paths, max_frames=150, fixed_fx=None):
    """Run Depth Pro on a list of image paths.

    Args:
        model, transform: Depth Pro model and preprocessing transform.
        img_paths:  list of image file paths.
        max_frames: subsample long sequences to at most this many frames.
        fixed_fx:   if set (float), inject this focal length (px) into every
                    infer() call — Depth Pro only estimates depth scale, not fx.
                    Use the global median from Pass 1 for self-calibration.

    Returns:
        depths:    np.ndarray (N, H, W) float32, metric depth in metres
        K:         np.ndarray (3, 3) float32, camera intrinsics
        frame_ids: list of str, original filenames
    """
    import depth_pro

    # Subsample long sequences
    if len(img_paths) > max_frames:
        step = len(img_paths) // max_frames
        img_paths = img_paths[::step][:max_frames]

    depth_list = []
    fx_list = []
    H_ref, W_ref = None, None

    device = next(model.parameters()).device
    f_px_inject = torch.tensor(fixed_fx, dtype=torch.float32).to(device) \
                  if fixed_fx is not None else None

    for p in tqdm(img_paths, desc="  frames", leave=False):
        image, _, f_px_exif = depth_pro.load_rgb(p)
        image_t = transform(image).unsqueeze(0).to(device)

        # Pass 2: use fixed fx; Pass 1: use EXIF hint (may be None)
        f_px_use = f_px_inject if f_px_inject is not None else f_px_exif

        with torch.no_grad():
            pred = model.infer(image_t, f_px=f_px_use)

        depth = pred["depth"].squeeze().cpu().float().numpy()   # (H, W)
        focallen = float(pred["focallength_px"])

        depth_list.append(depth)
        fx_list.append(focallen)
        if H_ref is None:
            H_ref, W_ref = depth.shape

    depths = np.stack(depth_list).astype(np.float32)       # (N, H, W)
    # If fixed_fx was injected, K reflects that; otherwise use frame median
    fx = fixed_fx if fixed_fx is not None else float(np.median(fx_list))
    cx, cy = W_ref / 2.0, H_ref / 2.0
    K = np.array([[fx,  0, cx],
                  [ 0, fx, cy],
                  [ 0,  0,  1]], dtype=np.float32)

    frame_ids = [os.path.basename(p) for p in img_paths]
    return depths, K, frame_ids


def run_batch(sequences, model, transform, output_dir, dataset_name,
             max_frames=150, fixed_fx=None, redo=False):
    """Run Depth Pro inference on a list of (seq_id, img_paths) tuples.

    Returns: (done, skipped, failed)
    """
    done, skipped, failed = 0, 0, 0

    for seq_id, img_paths in tqdm(sequences, desc=f"DepthPro/{dataset_name}"):
        seq_out    = os.path.join(output_dir, seq_id)
        depths_path = os.path.join(seq_out, "depths.npz")

        if os.path.exists(depths_path) and not redo:
            tqdm.write(f"  ⏭  {seq_id}: cached")
            skipped += 1
            continue

        try:
            depths, K, frame_ids = run_depth_pro(
                model, transform, img_paths,
                max_frames=max_frames, fixed_fx=fixed_fx
            )
            os.makedirs(seq_out, exist_ok=True)
            np.savez_compressed(depths_path, depths=depths)
            np.savetxt(os.path.join(seq_out, "K.txt"), K, fmt="%.6f")
            with open(os.path.join(seq_out, "frame_ids.txt"), "w") as f:
                f.write("\n".join(frame_ids))
            tqdm.write(f"  ✅ {seq_id}: depths {depths.shape}  K fx={K[0,0]:.1f}")
            done += 1

        except Exception as e:
            tqdm.write(f"  ❌ {seq_id}: {e}")
            failed += 1

        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ⏭ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"Output: {output_dir}")
    return done, skipped, failed


# ── Main ─────────────────────────────────────────────────────────────────────────────────

def main():
    import json as _json
    CAM_CFG = os.path.join(os.path.dirname(__file__), "dexycb_camera_config.json")

    parser = argparse.ArgumentParser(description="Universal Depth Pro batch inference")
    parser.add_argument("--dataset",    required=True, choices=list(DISCOVERERS.keys()))
    parser.add_argument("--input-dir",  default=None)
    parser.add_argument("--seq",        default=None, help="Substring match for sequence ID")
    parser.add_argument("--limit",      type=int, default=0, help="Process first N sequences only (for smoke tests)")
    parser.add_argument("--start",      type=int, default=0,  help="Shard start index (inclusive) for multi-GPU parallelism")
    parser.add_argument("--end",        type=int, default=0,  help="Shard end index (exclusive), 0=process all remaining")
    parser.add_argument("--max-frames", type=int, default=150,
                        help="Max frames per sequence for Pass 2 / normal mode (default 150)")
    parser.add_argument("--cam",        default=None,
                        help="DexYCB camera serial(s) to process, comma-separated.")
    parser.add_argument("--fixed-fx",   type=float, default=None,
                        help="Pass 2: inject fixed focal length (px). "
                             "Use --compute-global-fx to get this value.")
    parser.add_argument("--compute-global-fx", action="store_true",
                        help="Read all K.txt files, print global median fx, and exit (no GPU).")
    parser.add_argument("--two-pass",   action="store_true",
                        help="Auto two-pass: Pass1 (30 frames) → global median fx → "
                             "Pass2 (--max-frames, fixed fx). Works for all datasets. "
                             "Removes per-camera bias without GT intrinsics.")
    parser.add_argument("--redo",       action="store_true",
                        help="Reprocess sequences even if depths.npz already exists.")
    args = parser.parse_args()

    # Resolve DexYCB camera filter
    cam_filter = None
    if args.dataset == "dexycb":
        if args.cam:
            cam_filter = [c.strip() for c in args.cam.split(",")]
        elif os.path.exists(CAM_CFG):
            cfg = _json.load(open(CAM_CFG))
            cam_filter = cfg.get("selected", None)
        if cam_filter:
            print(f"DexYCB camera filter: {cam_filter}")

    discover_fn, data_folder = DISCOVERERS[args.dataset]
    input_dir  = args.input_dir or os.path.join(RAW_BASE, data_folder)
    output_dir = os.path.join(OUT_BASE, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # ── --compute-global-fx: read K.txt files and exit (no GPU) ──────────────
    if args.compute_global_fx:
        print(f"\n📐 Computing global fx from existing K.txt files in: {output_dir}")
        global_fx = compute_global_fx(output_dir)
        if global_fx is None:
            print("  ❌ No K.txt files found. Run Pass 1 first.")
        else:
            print(f"\n✅ Suggested command for Pass 2:")
            print(f"   python data/batch_depth_pro.py --dataset {args.dataset} "
                  f"--fixed-fx {global_fx:.1f} --redo")
        return

    # Build sequence list (DexYCB supports camera filter)
    if args.dataset == "dexycb":
        sequences = list(discover_fn(input_dir, cam_filter=cam_filter))
    else:
        sequences = list(discover_fn(input_dir))
    if args.seq:
        sequences = [(sid, imgs) for sid, imgs in sequences if args.seq in sid]
    if args.limit > 0:
        sequences = sequences[:args.limit]
    # Shard: --start/--end for multi-GPU parallelism
    if args.start > 0 or args.end > 0:
        end = args.end if args.end > 0 else len(sequences)
        sequences = sequences[args.start:end]
        print(f"[Shard] Processing sequences [{args.start}:{end}] = {len(sequences)} seqs")

    print(f"Found {len(sequences)} sequences\n")

    print("Loading Depth Pro model...")
    model, transform = load_depth_pro_model()
    print(f"✅ Model loaded (device: {next(model.parameters()).device})\n")

    if args.two_pass:
        # ── Two-Pass 自标定模式 ────────────────────────────────────────────────
        print(f"\n╔{'='*58}╗")
        print(f"║  Two-Pass 自标定模式  {args.dataset:<40}║")
        print(f"╚{'='*58}╝\n")

        # Pass 1: 快速估计 fx（30 帧，跳过已缓存）
        print("[Pass 1] 快速 fx 估计 (30 帧/序列)…")
        run_batch(sequences, model, transform, output_dir, args.dataset,
                  max_frames=30, fixed_fx=None, redo=False)

        # 计算全局中位数
        print("\n[计算] 全局 fx 中位数…")
        global_fx = compute_global_fx(output_dir)
        if global_fx is None:
            print("  ❌ Pass 1 无输出，无法进行 Pass 2")
            return

        # Pass 2: 注入固定 fx，覆盖全部序列
        print(f"\n[Pass 2] 注入 fx={global_fx:.1f}px，运行完整深度估计 ({args.max_frames} 帧/序列)…")
        run_batch(sequences, model, transform, output_dir, args.dataset,
                  max_frames=args.max_frames, fixed_fx=global_fx, redo=True)

        print(f"\n✅ Two-Pass 完成！使用 fx={global_fx:.1f}px")

    else:
        # ── 普通模式 ────────────────────────────────────────────────────────────
        if args.fixed_fx:
            print(f"⚠️  Pass 2 模式: 注入 fx={args.fixed_fx:.1f}px")
        run_batch(sequences, model, transform, output_dir, args.dataset,
                  max_frames=args.max_frames, fixed_fx=args.fixed_fx, redo=args.redo)


if __name__ == "__main__":
    main()
