#!/usr/bin/env python3
"""
Select Best Frames for SAM3D Object Reconstruction
====================================================
For each ARCTIC object, select the frame where the object is most visible
(least occluded by hands) for SAM3D 3D reconstruction.

Selection criteria (ranked by priority):
  1. Pre-grasp frames: before the hand reaches the object (object fully visible)
  2. Object size: the object should be large in the frame (close to camera)
  3. Sharpness: frame should not be blurry (Laplacian variance)

Output:
  - Selected frames copied to output directory
  - JSON manifest with frame metadata

Usage:
    python -m data.select_reconstruction_frames
    python -m data.select_reconstruction_frames --visualize
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# ============================================================
# ARCTIC paths
# ============================================================
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as _cfg
ARCTIC_ROOT = _cfg.ARCTIC_ROOT
MANUAL_VIDEOS = os.path.join(ARCTIC_ROOT, "arctic_manual_videos")
EGOCAM_VIDEOS = os.path.join(ARCTIC_ROOT, "arctic_egocam_videos")
OBJECT_MESHES = os.path.join(ARCTIC_ROOT, "meta", "object_vtemplates")
ONSET_JSON = os.path.join(config.OUTPUT_DIR, "arctic_grasp_onset.json")

DEFAULT_OUTPUT = os.path.join(config.OUTPUT_DIR, "sam3d_reconstruction_frames")


def compute_sharpness(image_path):
    """Compute image sharpness using Laplacian variance (higher = sharper)."""
    try:
        import cv2
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except ImportError:
        # Fallback: approximate sharpness with PIL
        img = Image.open(image_path).convert('L')
        arr = np.array(img, dtype=np.float64)
        # Simple Laplacian approximation
        lap = (arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:]
               - 4 * arr[1:-1, 1:-1])
        return float(np.var(lap))


def compute_brightness(image_path):
    """Compute average brightness (avoid too dark or overexposed frames)."""
    img = Image.open(image_path).convert('L')
    return float(np.mean(np.array(img)))


def find_sequences(subject_dir):
    """Find all sequences for a subject, grouped by object."""
    objects = {}
    for seq_dir in sorted(glob.glob(os.path.join(subject_dir, "*"))):
        if not os.path.isdir(seq_dir):
            continue
        seq_name = os.path.basename(seq_dir)
        # Parse: {object}_{action}_{trial} e.g. box_use_01
        parts = seq_name.rsplit("_", 2)
        if len(parts) >= 3:
            obj_name = parts[0]
        elif len(parts) == 2:
            obj_name = parts[0]
        else:
            obj_name = seq_name
        
        img_dir = os.path.join(seq_dir, "extracted_images")
        if not os.path.isdir(img_dir):
            continue
        
        images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + 
                       glob.glob(os.path.join(img_dir, "*.png")))
        if not images:
            continue
        
        if obj_name not in objects:
            objects[obj_name] = []
        objects[obj_name].append({
            "seq_name": seq_name,
            "seq_dir": seq_dir,
            "img_dir": img_dir,
            "images": images,
            "n_frames": len(images),
        })
    return objects


def select_best_frame(sequences, onset_data, obj_name):
    """
    Select the best frame for SAM3D reconstruction.
    
    Strategy:
      1. Look for pre-grasp frames (before onset_frame) — object is fully visible
      2. Among those, pick the sharpest frame with good brightness
      3. If no onset data, use early frames (first 30%) of the sequence
    """
    candidates = []
    
    for seq in sequences:
        seq_key_patterns = [
            f"s01__{seq['seq_name']}",
            seq['seq_name'],
        ]
        
        # Find onset frame for this sequence
        onset_frame = None
        for key, data in onset_data.items():
            if seq['seq_name'] in key:
                onset_frame = data.get('onset_frame', None)
                break
        
        # Determine candidate range
        n = seq['n_frames']
        if onset_frame is not None and onset_frame > 5:
            # Pre-grasp: frames 0 to onset_frame - 5 (hand hasn't reached object yet)
            frame_range = range(0, max(1, onset_frame - 5))
            source = "pre-grasp"
        else:
            # No onset data: use first 30% of sequence
            frame_range = range(0, max(1, n // 3))
            source = "early-sequence"
        
        # Score each candidate frame
        for i in frame_range:
            if i >= len(seq['images']):
                continue
            img_path = seq['images'][i]
            
            sharpness = compute_sharpness(img_path)
            brightness = compute_brightness(img_path)
            
            # Penalize too dark (<50) or too bright (>220) frames
            brightness_ok = 50 < brightness < 220
            
            # Combined score: sharpness is primary, brightness as filter
            score = sharpness if brightness_ok else sharpness * 0.3
            
            candidates.append({
                "image_path": img_path,
                "seq_name": seq['seq_name'],
                "frame_idx": i,
                "sharpness": round(sharpness, 1),
                "brightness": round(brightness, 1),
                "source": source,
                "video_source": seq.get('video_source', 'unknown'),
                "score": round(score, 1),
            })
    
    if not candidates:
        return None
    
    # Sort by score (highest first)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Select best frames for SAM3D reconstruction")
    parser.add_argument("--arctic_root", type=str, default=ARCTIC_ROOT,
                        help="ARCTIC dataset root directory")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output directory for selected frames")
    parser.add_argument("--visualize", action="store_true",
                        help="Show selected frames with matplotlib")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Show top K candidates per object")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load grasp onset data
    onset_data = {}
    if os.path.exists(ONSET_JSON):
        with open(ONSET_JSON, 'r') as f:
            onset_data = json.load(f)
        print(f"Loaded grasp onset data: {len(onset_data)} sequences")
    else:
        print(f"Warning: No grasp onset data at {ONSET_JSON}")
        print(f"  Will use first 30% of each sequence as candidates")
    
    # Find all subjects from both manual and egocam videos
    video_sources = [
        ("manual", MANUAL_VIDEOS),
        ("egocam", EGOCAM_VIDEOS),
    ]
    
    # Collect all subject IDs across sources
    all_subject_ids = set()
    for src_name, src_dir in video_sources:
        for d in glob.glob(os.path.join(src_dir, "s*")):
            all_subject_ids.add(os.path.basename(d))
    all_subject_ids = sorted(all_subject_ids)
    
    if not all_subject_ids:
        print(f"Error: No subjects found in {MANUAL_VIDEOS} or {EGOCAM_VIDEOS}")
        return
    
    print(f"\nScanning {len(all_subject_ids)} subjects across manual + egocam videos...")
    
    all_results = {}
    
    for subject_id in all_subject_ids:
        print(f"\n{'='*60}")
        print(f"Subject: {subject_id}")
        print(f"{'='*60}")
        
        # Merge sequences from all video sources
        objects = {}
        for src_name, src_dir in video_sources:
            subject_dir = os.path.join(src_dir, subject_id)
            if not os.path.isdir(subject_dir):
                continue
            src_objects = find_sequences(subject_dir)
            for obj_name, seqs in src_objects.items():
                # Tag each sequence with its source
                for s in seqs:
                    s['video_source'] = src_name
                if obj_name not in objects:
                    objects[obj_name] = []
                objects[obj_name].extend(seqs)
        
        for obj_name, sequences in sorted(objects.items()):
            print(f"\n  {obj_name}: {len(sequences)} sequences, "
                  f"{sum(s['n_frames'] for s in sequences)} total frames")
            
            best = select_best_frame(sequences, onset_data, obj_name)
            
            if best is None:
                print(f"    ❌ No valid frame found")
                continue
            
            print(f"    ✅ Best: {best['seq_name']} frame {best['frame_idx']}")
            print(f"       Sharpness: {best['sharpness']}, "
                  f"Brightness: {best['brightness']}, "
                  f"Source: {best['source']}")
            
            # Copy best frame to output
            ext = os.path.splitext(best['image_path'])[1]
            dst = os.path.join(args.output, f"{obj_name}{ext}")
            
            from shutil import copy2
            copy2(best['image_path'], dst)
            print(f"       Saved: {dst}")
            
            all_results[obj_name] = {
                "image_path": dst,
                "source_path": best['image_path'],
                "seq_name": best['seq_name'],
                "frame_idx": best['frame_idx'],
                "sharpness": best['sharpness'],
                "brightness": best['brightness'],
                "source": best['source'],
                "video_source": best.get('video_source', 'unknown'),
                "score": best['score'],
                "has_gt_mesh": os.path.exists(
                    os.path.join(OBJECT_MESHES, obj_name, "mesh.obj")
                ),
            }
    
    # Save manifest
    manifest_path = os.path.join(args.output, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Selected {len(all_results)} objects → {args.output}")
    print(f"Manifest: {manifest_path}")
    
    # Visualize
    if args.visualize and all_results:
        try:
            import matplotlib.pyplot as plt
            
            n = len(all_results)
            cols = min(4, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
            if n == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, (obj_name, info) in enumerate(sorted(all_results.items())):
                img = Image.open(info['image_path'])
                axes[i].imshow(img)
                axes[i].set_title(f"{obj_name}\nsharp={info['sharpness']:.0f}", fontsize=9)
                axes[i].axis('off')
            
            for j in range(i+1, len(axes)):
                axes[j].axis('off')
            
            plt.suptitle("Selected Frames for SAM3D Reconstruction", fontsize=14)
            plt.tight_layout()
            
            vis_path = os.path.join(args.output, "selected_frames.png")
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            print(f"Visualization: {vis_path}")
            plt.show()
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Object':<20} {'Sequence':<25} {'Frame':>5} {'Sharp':>8} {'GT Mesh':>8}")
    print(f"{'-'*60}")
    for name, info in sorted(all_results.items()):
        gt = "✅" if info['has_gt_mesh'] else "❌"
        print(f"{name:<20} {info['seq_name']:<25} {info['frame_idx']:>5} "
              f"{info['sharpness']:>8.0f} {gt:>8}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
