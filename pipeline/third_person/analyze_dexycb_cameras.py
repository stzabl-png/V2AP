#!/usr/bin/env python3
"""
Quick single-frame Depth Pro accuracy comparison across all 8 DexYCB cameras.
Run from depth-pro conda env.

Usage:
    conda activate depth-pro
    python data/analyze_dexycb_cameras.py
"""

import os, sys, torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from natsort import natsorted
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'ml-depth-pro', 'src'))
import config

RAW = os.path.join(config.DATA_HUB, 'RawData', 'ThirdPersonRawData', 'dexycb')
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config as _cfg
OUT = _os.path.join(_cfg.OUTPUT_DIR, 'analyze_dexycb_cameras')

# DexYCB GT focal length: all cameras are RealSense D415/D435 at 640×480
# Published in DexYCB paper: fx ≈ 607-617 depending on exact serial
# We use 607.3 as reference (commonly cited D415 value)
GT_FX = 607.3   # px at 640×480

SUBJ = '20200709-subject-01'
SESS = '20200709_141754'     # session 0 → 002_master_chef_can


def run_depth_pro_single_frame(img_path, model, transform, device):
    """Run on one image, return (depth, focal_length_px, H, W)."""
    import depth_pro
    image, _, f_px = depth_pro.load_rgb(img_path)
    image_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model.infer(image_t, f_px=f_px)
    depth = pred['depth'].squeeze().cpu().float().numpy()
    fx = float(pred['focallength_px'])
    return depth, fx, depth.shape[0], depth.shape[1]


def main():
    import depth_pro

    print("Loading Depth Pro model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, transform = depth_pro.create_model_and_transforms(device=device, precision=torch.float16)
    model.eval()
    print(f"Model ready on {device}")

    sess_dir = os.path.join(RAW, SUBJ, SESS)
    cameras  = natsorted([d for d in os.listdir(sess_dir) if os.path.isdir(os.path.join(sess_dir, d))])

    print(f"\nSession: {SUBJ}/{SESS}")
    print(f"GT fx reference: {GT_FX} px  (RealSense D415/D435 at 640×480)")
    print(f"Cameras: {len(cameras)}\n")

    results = []
    for cam in cameras:
        imgs = natsorted(glob(os.path.join(sess_dir, cam, 'color_*.jpg')))
        if not imgs:
            print(f"  {cam}: NO IMAGES")
            continue

        # Use middle frame
        img_path = imgs[len(imgs) // 2]
        print(f"  Running cam {cam}...", end=' ', flush=True)
        depth, fx, H, W = run_depth_pro_single_frame(img_path, model, transform, device)

        bias_pct = (fx / GT_FX - 1) * 100
        valid_d  = depth[(depth > 0.1) & (depth < 3.0)]
        d_med    = float(np.median(valid_d)) if len(valid_d) else 0.0

        results.append({
            'cam': cam,
            'fx_dp': fx,
            'bias_pct': bias_pct,
            'depth_med': d_med,
            'img_path': img_path,
            'depth': depth,
        })
        print(f"fx={fx:.1f}  bias={bias_pct:+.1f}%  depth_p50={d_med:.2f}m")

    # ── Ranking: closest to GT ──
    results.sort(key=lambda r: abs(r['bias_pct']))
    print(f"\n{'='*60}")
    print(f"  Ranking by fx accuracy (GT={GT_FX:.0f}px):")
    print(f"  {'Rank':<5} {'Serial':<18} {'fx_DP':>8} {'Bias%':>8} {'depth_p50':>10}")
    print(f"  {'-'*5} {'-'*18} {'-'*8} {'-'*8} {'-'*10}")
    for rank, r in enumerate(results, 1):
        star = " ← BEST" if rank == 1 else ""
        print(f"  {rank:<5} {r['cam']:<18} {r['fx_dp']:>8.1f} {r['bias_pct']:>+7.1f}% {r['depth_med']:>9.2f}m{star}")

    # ── Visual: RGB + depth grid ──
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(n * 3.5, 7))
    fig.suptitle(f'DexYCB Camera Comparison — {SUBJ}/{SESS}\nGT fx = {GT_FX:.0f}px (RealSense D415)',
                 fontsize=12, fontweight='bold')

    for col, r in enumerate(results):
        # Row 0: RGB
        ax = axes[0, col]
        ax.imshow(mpimg.imread(r['img_path']))
        color = 'green' if abs(r['bias_pct']) < 5 else ('orange' if abs(r['bias_pct']) < 10 else 'red')
        ax.set_title(f"cam {r['cam'][-6:]}\nfx={r['fx_dp']:.0f} ({r['bias_pct']:+.1f}%)",
                     fontsize=8, color=color, fontweight='bold')
        ax.axis('off')

        # Row 1: depth map
        ax = axes[1, col]
        im = ax.imshow(np.clip(r['depth'], 0.1, 2.5), cmap='plasma_r', vmin=0.1, vmax=2.5)
        ax.set_title(f"depth p50={r['depth_med']:.2f}m", fontsize=8)
        ax.axis('off')

    # Add colorbar
    plt.colorbar(im, ax=axes[1, :], label='Depth (m)', fraction=0.02, pad=0.01)
    plt.tight_layout()

    out_path = os.path.join(OUT, 'dexycb_cam_depth_comparison.png')
    plt.savefig(out_path, dpi=90, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    print(f"\n✅ Recommended camera: {results[0]['cam']}  (fx bias = {results[0]['bias_pct']:+.1f}%)")


if __name__ == '__main__':
    main()
