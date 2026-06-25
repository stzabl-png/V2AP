"""
Prepare inputs for object 3D reconstruction (obj_recon / SAM3D pipeline).

For each dataset sequence:
1. Select the best "object visible" frame using pre-computed depth maps
2. Generate object mask via SAM2 (or use a simple bbox heuristic)
3. Compose RGBA image (mask in alpha channel)
4. Package into: data_hub/ProcessedData/obj_recon_input/{dataset}/{seq_id}/
      image.png       — RGBA image (object masked in alpha)
      rgb.png         — plain RGB
      depth.npy       — single-frame depth (float32, metres)
      K.txt           — 3x3 camera intrinsic

Usage:
  conda activate depth-pro     # has PIL, numpy, cv2
  python data/prepare_obj_recon_input.py --dataset oakink --limit 3
  python data/prepare_obj_recon_input.py --dataset arctic
  bash scripts/prepare_obj_recon_all.sh --limit 3

Output is ready to rsync to cloud and run with:
  python infer_object_mesh.py   (inside third_party/obj_recon/)
"""

import os, sys, argparse, numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

DEPTH_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "third_depth")
RAW_BASE   = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")
OUT_BASE   = os.path.join(config.DATA_HUB, "ProcessedData", "obj_recon_input")


# ── Dataset frame finders ─────────────────────────────────────────────────────
# Each returns (seq_id, all_img_paths) matching the order in depth/frame_ids.txt

def discover_arctic(input_dir):
    for subj in natsorted(os.listdir(input_dir)):
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir): continue
        for seq in natsorted(os.listdir(subj_dir)):
            cam_dir = os.path.join(subj_dir, seq, "1")
            if not os.path.isdir(cam_dir): continue
            imgs = natsorted(glob(os.path.join(cam_dir, "*.jpg")) +
                             glob(os.path.join(cam_dir, "*.png")))
            if imgs: yield f"{subj}__{seq}", imgs

def discover_oakink(input_dir):
    for seq_id in natsorted(os.listdir(input_dir)):
        seq_dir = os.path.join(input_dir, seq_id)
        if not os.path.isdir(seq_dir): continue
        imgs = natsorted(glob(os.path.join(seq_dir, "*", "north_west_color_*.png")) +
                         glob(os.path.join(seq_dir, "north_west_color_*.png")))
        if imgs: yield seq_id, imgs

def discover_ho3d(input_dir):
    for split in ["train", "evaluation"]:
        split_dir = os.path.join(input_dir, split)
        if not os.path.isdir(split_dir): continue
        for seq_id in natsorted(os.listdir(split_dir)):
            rgb_dir = os.path.join(split_dir, seq_id, "rgb")
            if not os.path.isdir(rgb_dir): continue
            imgs = natsorted(glob(os.path.join(rgb_dir, "*.jpg")))
            if imgs: yield f"{split}__{seq_id}", imgs

def discover_dexycb(input_dir):
    for subj in natsorted(os.listdir(input_dir)):
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir): continue
        for dt in natsorted(os.listdir(subj_dir)):
            dt_dir = os.path.join(subj_dir, dt)
            if not os.path.isdir(dt_dir): continue
            for serial in natsorted(os.listdir(dt_dir)):
                cam_dir = os.path.join(dt_dir, serial)
                if not os.path.isdir(cam_dir): continue
                imgs = natsorted(glob(os.path.join(cam_dir, "color_*.jpg")))
                if imgs: yield f"{subj}__{dt}__{serial}", imgs

DISCOVERERS = {
    "arctic":  (discover_arctic,  "arctic"),
    "oakink":  (discover_oakink,  "oakink_v1"),
    "ho3d_v3": (discover_ho3d,    "ho3d_v3"),
    "dexycb":  (discover_dexycb,  "dexycb"),
}


# ── Frame selection ───────────────────────────────────────────────────────────

def select_best_frame(img_paths, depth_arr, frame_ids):
    """Pick the frame where the object region is most visible.

    Heuristic: pick the frame with the SMALLEST median depth in the centre
    crop (object likely closest, clearest view). Falls back to middle frame.

    Args:
        img_paths:  list of str (all RGB paths for this sequence)
        depth_arr:  np.ndarray (N, H, W) — depths from batch_depth_pro
        frame_ids:  list of str — filenames matching depth_arr rows

    Returns:
        best_img_path: str
        best_depth:    np.ndarray (H, W)
        best_idx:      int
    """
    if depth_arr is None or len(depth_arr) == 0:
        mid = len(img_paths) // 2
        return img_paths[mid], None, mid

    N, H, W = depth_arr.shape
    # centre crop (middle 40% of image)
    r0, r1 = int(H * 0.3), int(H * 0.7)
    c0, c1 = int(W * 0.3), int(W * 0.7)
    centre_depths = depth_arr[:, r0:r1, c0:c1]
    median_centre = np.median(centre_depths.reshape(N, -1), axis=1)

    best_idx = int(np.argmin(median_centre))
    best_frame_name = frame_ids[best_idx]

    # Find matching img_path
    img_map = {os.path.basename(p): p for p in img_paths}
    best_img = img_map.get(best_frame_name, img_paths[min(best_idx, len(img_paths)-1)])

    return best_img, depth_arr[best_idx], best_idx


def simple_object_mask(img_rgba, depth, percentile=30):
    """Heuristic mask: pixels closer than `percentile`% depth threshold.

    Returns binary mask (H, W) uint8 (255 = object region).
    This is a rough mask — good enough to guide SAM2 or the reconstruction.
    For production, replace with SAM2 segmentation.
    """
    if depth is None:
        h, w = np.array(img_rgba).shape[:2]
        # fallback: centre ellipse mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cy, cx = h // 2, w // 2
        ry, rx = h // 4, w // 4
        Y, X = np.ogrid[:h, :w]
        mask[((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 <= 1] = 255
        return mask

    threshold = np.percentile(depth[depth > 0], percentile) if (depth > 0).any() else depth.max()
    # Pixels closer than threshold = candidate object
    mask = ((depth > 0) & (depth <= threshold)).astype(np.uint8) * 255

    # Morphological clean-up (requires cv2 - graceful fallback if absent)
    try:
        import cv2
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    except ImportError:
        pass

    return mask


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DISCOVERERS.keys()))
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--seq",   default=None, help="Substring filter on seq_id")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    discover_fn, data_folder = DISCOVERERS[args.dataset]
    input_dir  = args.input_dir or os.path.join(RAW_BASE, data_folder)
    depth_dir  = os.path.join(DEPTH_BASE, args.dataset)
    output_dir = os.path.join(OUT_BASE, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f" Dataset   : {args.dataset}")
    print(f" RGB input : {input_dir}")
    print(f" Depth     : {depth_dir}")
    print(f" Output    : {output_dir}")
    print(f"{'='*60}\n")

    sequences = list(discover_fn(input_dir))
    if args.seq:
        sequences = [(s, p) for s, p in sequences if args.seq in s]
    if args.limit > 0:
        sequences = sequences[:args.limit]
    print(f"Found {len(sequences)} sequences\n")

    done, skipped, failed = 0, 0, 0

    for seq_id, img_paths in tqdm(sequences, desc=f"ObjRecon/{args.dataset}"):
        out_seq = os.path.join(output_dir, seq_id)

        if os.path.exists(os.path.join(out_seq, "image.png")):
            tqdm.write(f"  ⏭  {seq_id}: cached")
            skipped += 1
            continue

        try:
            # Load depth if available
            depth_seq = os.path.join(depth_dir, seq_id)
            depth_arr, frame_ids = None, []
            if os.path.isdir(depth_seq):
                d = np.load(os.path.join(depth_seq, "depths.npz"))["depths"]
                frame_ids = open(os.path.join(depth_seq, "frame_ids.txt")).read().strip().split("\n")
                depth_arr = d
                K = np.loadtxt(os.path.join(depth_seq, "K.txt"))
            else:
                tqdm.write(f"  ⚠️  {seq_id}: no depth, using heuristic K")
                # Estimate K from image size
                img0 = Image.open(img_paths[0])
                W, H = img0.size
                fx = max(W, H)
                K = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=np.float32)

            # Select best frame
            best_img_path, best_depth, best_idx = select_best_frame(
                img_paths, depth_arr, frame_ids)

            # Load RGB
            rgb = Image.open(best_img_path).convert("RGB")
            W, H = rgb.size

            # Generate object mask
            mask = simple_object_mask(rgb, best_depth, percentile=30)

            # Compose RGBA
            rgba = Image.fromarray(np.concatenate(
                [np.array(rgb), mask[:, :, None]], axis=2).astype(np.uint8), "RGBA")

            # Save output package (matches SAM3D input format exactly)
            os.makedirs(out_seq, exist_ok=True)
            rgb.save(os.path.join(out_seq, "image.png"))           # plain RGB → SAM3D input
            Image.fromarray(mask).save(os.path.join(out_seq, "0.png"))  # binary mask → SAM3D mask
            if best_depth is not None:
                np.save(os.path.join(out_seq, "depth.npy"), best_depth.astype(np.float32))
            np.savetxt(os.path.join(out_seq, "K.txt"), K, fmt="%.6f")
            with open(os.path.join(out_seq, "source_frame.txt"), "w") as f:
                f.write(f"{best_img_path}\nframe_idx={best_idx}\n")

            tqdm.write(f"  ✅ {seq_id}: frame={best_idx} mask_area={int(mask.mean()*100)}% → image.png + 0.png")
            done += 1

        except Exception as e:
            tqdm.write(f"  ❌ {seq_id}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ⏭ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"Output: {output_dir}")
    print(f"\nNext: rsync {output_dir}/ sam3d-gpu:~/input/{args.dataset}/ && run infer_batch.py")


if __name__ == "__main__":
    main()
