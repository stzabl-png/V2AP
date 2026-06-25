"""
Prepare BundleSDF Input from ARCTIC + MegaSAM
===============================================
Converts ARCTIC egocam video data into the input format required by BundleSDF:

    video_dir/
        rgb/        ← PNG frames (extracted from video or existing frames)
        depth/      ← uint16 PNG in millimeters (from MegaSAM metric depth)
        masks/      ← binary PNG: 0=background, 255=object (from SAM2)
        cam_K.txt   ← 3x3 intrinsic matrix (from MegaSAM)

Usage:
    # Prepare data for a specific sequence
    python -m data.prepare_bundlesdf_input \\
        --subject s01 --seq box_grab_01 \\
        --output_dir output/bundlesdf_input/s01__box_grab_01

    # Then run BundleSDF inside Docker (see batch_bundlesdf_arctic.py)
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from data.megasam_utils import load_megasam_K, load_megasam_depths


def extract_rgb_frames(video_path: str, out_dir: str, max_frames: int = None):
    """Extract RGB frames from a video file into PNG images."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    saved = 0
    while cap.isOpened() and saved < total:
        ret, frame = cap.read()
        if not ret:
            break
        fname = os.path.join(out_dir, f'{saved:06d}.png')
        cv2.imwrite(fname, frame)
        saved += 1
        if saved % 50 == 0:
            print(f'  Extracted {saved}/{total} frames', flush=True)

    cap.release()
    print(f'  ✅ RGB: {saved} frames → {out_dir}')
    return saved


def write_megasam_depths(seq_name: str, out_dir: str, n_frames: int):
    """Convert MegaSAM float32 metric depth (meters) to uint16 PNG (mm)."""
    try:
        from PIL import Image
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow', '-q'])
        from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    depths = load_megasam_depths(seq_name)

    if depths is None:
        print(f'  ⚠️  No MegaSAM depth for {seq_name}. Skipping depth dir.')
        return False

    # depths: (N, H, W) in meters → convert to mm uint16
    available = depths.shape[0]
    count = min(n_frames, available)
    for i in range(count):
        depth_mm = (depths[i] * 1000).astype(np.uint16)  # meters → mm
        img = Image.fromarray(depth_mm)
        img.save(os.path.join(out_dir, f'{i:06d}.png'))

    print(f'  ✅ Depth: {count} frames (from MegaSAM, meter→mm uint16) → {out_dir}')
    return True


def write_cam_K(seq_name: str, out_path: str):
    """Write MegaSAM intrinsic matrix to cam_K.txt."""
    K = load_megasam_K(seq_name)
    if K is None:
        print(f'  ⚠️  No MegaSAM K for {seq_name}. Cannot write cam_K.txt!')
        return None

    with open(out_path, 'w') as f:
        for row in K:
            f.write(' '.join(f'{v:.6f}' for v in row) + '\n')

    print(f'  ✅ cam_K.txt written: fx={K[0,0]:.1f} fy={K[1,1]:.1f}')
    return K


def run_sam2_masks(rgb_dir: str, masks_dir: str, object_name: str = None):
    """
    Generate object masks using SAM2.

    For the first frame, we use an interactive prompt (bounding box or point).
    For subsequent frames, SAM2 propagates the mask automatically.

    Requires: conda activate sam2 (or SAM2 installed in current env).
    """
    import sys
    # Resolve SAM2 directory: env var → common locations → error
    sam2_dir = os.environ.get("SAM2_DIR", "")
    if not sam2_dir or not os.path.exists(sam2_dir):
        for _candidate in [
            os.path.expanduser("~/Project/sam2"),
            os.path.expanduser("~/sam2"),
            "/opt/sam2",
        ]:
            if os.path.exists(_candidate):
                sam2_dir = _candidate
                break
    if sam2_dir and sam2_dir not in sys.path:
        sys.path.insert(0, sam2_dir)

    os.makedirs(masks_dir, exist_ok=True)

    # Check if SAM2 already imported
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        print('  ⚠️  SAM2 not importable in this env. Skipping mask generation.')
        print('     Run with: conda activate sam2')
        return False

    # Get sorted frame paths
    frames = sorted(Path(rgb_dir).glob('*.png'))
    if not frames:
        print(f'  ⚠️  No frames in {rgb_dir}')
        return False

    import cv2, torch
    from PIL import Image

    print(f'  Running SAM2 on {len(frames)} frames...')

    # Build SAM2 Video Predictor
    sam2_checkpoint = os.path.join(sam2_dir, 'checkpoints/sam2.1_hiera_large.pt')
    model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # For the first frame: use center of image as a positive point
    # (Simple heuristic - works well when object is in center/foreground)
    first_frame = cv2.imread(str(frames[0]))
    H, W = first_frame.shape[:2]
    # Point prompt: center of image
    point = np.array([[W // 2, H // 2]], dtype=np.float32)
    label = np.array([1], dtype=np.int32)  # 1 = foreground

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=rgb_dir)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=point,
            labels=label,
        )

        # Propagate through all frames
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            mask = masks[0].squeeze().cpu().numpy()  # (H, W) bool
            out_mask = (mask > 0.5).astype(np.uint8) * 255
            out_path = os.path.join(masks_dir, f'{frame_idx:06d}.png')
            Image.fromarray(out_mask).save(out_path)

    print(f'  ✅ Masks: SAM2 propagated to {len(frames)} frames → {masks_dir}')
    return True


def create_dummy_masks(rgb_dir: str, masks_dir: str):
    """
    Create dummy masks (full foreground) as placeholder.
    Replace with SAM2 output before running BundleSDF.
    """
    import cv2
    os.makedirs(masks_dir, exist_ok=True)
    frames = sorted(Path(rgb_dir).glob('*.png'))
    for frame in frames:
        img = cv2.imread(str(frame))
        H, W = img.shape[:2]
        mask = np.ones((H, W), dtype=np.uint8) * 255  # all foreground
        cv2.imwrite(os.path.join(masks_dir, frame.name), mask)
    print(f'  ⚠️  Dummy masks (all foreground) created: {len(frames)} frames')
    print(f'     Replace with SAM2 masks before running BundleSDF!')


def prepare_sequence(subject: str, seq_name: str, output_dir: str,
                     max_frames: int = None, use_sam2: bool = False):
    """Main preparation function for a single ARCTIC sequence."""

    print(f'\n{"="*60}')
    print(f'Preparing BundleSDF input: {subject}/{seq_name}')
    print(f'Output: {output_dir}')
    print(f'{"="*60}')

    os.makedirs(output_dir, exist_ok=True)

    # 1. Find ARCTIC egocam video
    video_root = os.path.join(config.ARCTIC_ROOT, 'arctic_egocam_videos', subject)
    # Try MP4 first, then MOV
    video_path = None
    for ext in ['mp4', 'MP4', 'mov', 'MOV']:
        candidate = os.path.join(video_root, f'{seq_name}.{ext}')
        if os.path.exists(candidate):
            video_path = candidate
            break
    # Also try nested structure
    if video_path is None:
        candidate = os.path.join(video_root, seq_name, f'{seq_name}.mp4')
        if os.path.exists(candidate):
            video_path = candidate

    if video_path is None:
        # Try cropped images directory
        img_root = os.path.join(config.ARCTIC_ROOT, 'arctic_data', 'data',
                                'cropped_images', subject, seq_name, 'world')
        if os.path.exists(img_root):
            print(f'  Using pre-extracted frames from {img_root}')
            import shutil
            rgb_dir = os.path.join(output_dir, 'rgb')
            os.makedirs(rgb_dir, exist_ok=True)
            frames = sorted(Path(img_root).glob('*.jpg')) + sorted(Path(img_root).glob('*.png'))
            if max_frames:
                frames = frames[:max_frames]
            for i, src in enumerate(frames):
                import shutil as sh
                sh.copy(src, os.path.join(rgb_dir, f'{i:06d}.png'))
            n_frames = len(frames)
            print(f'  ✅ RGB: {n_frames} frames copied → {rgb_dir}')
        else:
            print(f'  ❌ No video or image dir found for {subject}/{seq_name}')
            print(f'     Tried: {video_root}')
            return False
    else:
        # 2. Extract RGB frames
        rgb_dir = os.path.join(output_dir, 'rgb')
        n_frames = extract_rgb_frames(video_path, rgb_dir, max_frames)

    # 3. Write MegaSAM depth
    depth_dir = os.path.join(output_dir, 'depth')
    has_depth = write_megasam_depths(seq_name, depth_dir, n_frames)

    # 4. Write cam_K.txt
    cam_k_path = os.path.join(output_dir, 'cam_K.txt')
    K = write_cam_K(seq_name, cam_k_path)

    # 5. Object masks (SAM2 or dummy)
    masks_dir = os.path.join(output_dir, 'masks')
    if use_sam2:
        success = run_sam2_masks(rgb_dir, masks_dir)
        if not success:
            create_dummy_masks(rgb_dir, masks_dir)
    else:
        create_dummy_masks(rgb_dir, masks_dir)

    # 6. Summary
    print(f'\n📁 BundleSDF input ready at: {output_dir}')
    print(f'   rgb/    : {len(list(Path(rgb_dir).glob("*.png")))} frames')
    depth_count = len(list(Path(depth_dir).glob("*.png"))) if os.path.exists(depth_dir) else 0
    print(f'   depth/  : {depth_count} frames {"✅" if depth_count > 0 else "❌ (missing!)"}')
    mask_count = len(list(Path(masks_dir).glob("*.png"))) if os.path.exists(masks_dir) else 0
    print(f'   masks/  : {mask_count} frames {"✅" if mask_count > 0 else "❌ (missing!)"}')
    print(f'   cam_K   : {"✅" if K is not None else "❌ (missing!)"}')

    if K is None or not has_depth:
        print('\n⚠️  Missing required inputs. BundleSDF needs both depth and K.')
        print('   Only sequences with MegaSAM outputs can be processed.')
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare BundleSDF input from ARCTIC + MegaSAM')
    parser.add_argument('--subject', type=str, required=True,
                        help='ARCTIC subject (e.g., s01)')
    parser.add_argument('--seq', type=str, required=True,
                        help='Sequence name (e.g., box_grab_01)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: output/bundlesdf_input/{subject}__{seq})')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames to process (default: all MegaSAM frames)')
    parser.add_argument('--use_sam2', action='store_true',
                        help='Use SAM2 for object segmentation (requires sam2 env)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            config.OUTPUT_DIR, 'bundlesdf_input',
            f'{args.subject}__{args.seq}'
        )

    success = prepare_sequence(
        subject=args.subject,
        seq_name=args.seq,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        use_sam2=args.use_sam2,
    )

    if success:
        print('\n✅ Success! Ready to run BundleSDF:')
        print(f'   cd BundleSDF/docker && bash run_container.sh')
        print(f'   # Inside container:')
        print(f'   python run_custom.py --mode run_video --video_dir {args.output_dir} \\')
        print(f'     --out_folder output/bundlesdf_cache/{args.subject}__{args.seq} \\')
        print(f'     --use_segmenter 0 --debug_level 2')
    else:
        print('\n❌ Preparation incomplete. Check warnings above.')


if __name__ == '__main__':
    main()
