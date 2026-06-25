#!/usr/bin/env python3
"""
Validation run: Depth Pro on DexYCB 003_cracker_box (ycb_dex_02), camera 841412060263.
30 sequences × 72 frames ≈ 18 minutes.

Usage:
    conda activate depth-pro
    python data/run_dexycb_cracker_box.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Reuse batch_depth_pro infrastructure
os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
import subprocess

cmd = [
    'python', 'data/batch_depth_pro.py',
    '--dataset', 'dexycb',
    '--cam', '841412060263',
    '--max-frames', '72',
]

# We'll filter by seq substring - each subject's cracker_box sessions are the 5-9th sessions
# Just run the full camera 841412060263 with limit since --seq only matches ONE substring
# Instead, run batch_depth_pro.py and it will skip already-done sequences
import os, glob, numpy as np, torch
from natsort import natsorted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'ml-depth-pro', 'src'))
import config, depth_pro
from tqdm import tqdm

RAW   = os.path.join(config.DATA_HUB, 'RawData', 'ThirdPersonRawData', 'dexycb')
OUT   = os.path.join(config.DATA_HUB, 'ProcessedData', 'third_depth', 'dexycb')
CAM   = '841412060263'
OBJ_IDX       = 1   # 0-based → 003_cracker_box (2nd YCB object)
GRASPS_PER_OBJ = 5
NUM_OBJ        = 20


def get_cracker_box_seqs():
    seqs = []
    for subj in natsorted([d for d in os.listdir(RAW) if os.path.isdir(os.path.join(RAW, d))]):
        sessions = natsorted([d for d in os.listdir(os.path.join(RAW, subj))
                              if os.path.isdir(os.path.join(RAW, subj, d))])
        start = OBJ_IDX * GRASPS_PER_OBJ
        for sess in sessions[start:start + GRASPS_PER_OBJ]:
            cam_dir = os.path.join(RAW, subj, sess, CAM)
            imgs = sorted(glob.glob(os.path.join(cam_dir, 'color_*.jpg')))
            if imgs:
                seqs.append((f'{subj}__{sess}__{CAM}', imgs))
    return seqs


def run_depth_pro(model, transform, img_paths, device):
    depth_list, fx_list = [], []
    H_ref, W_ref = None, None
    for p in img_paths:
        image, _, f_px = depth_pro.load_rgb(p)
        image_t = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.infer(image_t, f_px=f_px)
        depth = pred['depth'].squeeze().cpu().float().numpy()
        fx = float(pred['focallength_px'])
        depth_list.append(depth)
        fx_list.append(fx)
        if H_ref is None:
            H_ref, W_ref = depth.shape
    depths = np.stack(depth_list).astype(np.float32)
    fx_med = float(np.median(fx_list))
    cx, cy = W_ref / 2.0, H_ref / 2.0
    K = np.array([[fx_med, 0, cx], [0, fx_med, cy], [0, 0, 1]], dtype=np.float32)
    return depths, K, [os.path.basename(p) for p in img_paths]


def main():
    seqs = get_cracker_box_seqs()
    print(f"DexYCB 003_cracker_box  |  cam {CAM}")
    print(f"Sequences: {len(seqs)}  |  ~{len(seqs)*72} frames  |  YCB obj: ycb_dex_02")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Depth Pro model on {device}...")
    model, transform = depth_pro.create_model_and_transforms(device=device, precision=torch.float16)
    model.eval()
    print("✅ Model ready\n")

    done, skipped, failed = 0, 0, 0
    for seq_id, img_paths in tqdm(seqs, desc='DepthPro/dexycb'):
        seq_out = os.path.join(OUT, seq_id)
        if os.path.exists(os.path.join(seq_out, 'K.txt')):
            tqdm.write(f'  ⏭  {seq_id}: cached')
            skipped += 1
            continue
        try:
            depths, K, fids = run_depth_pro(model, transform, img_paths, device)
            os.makedirs(seq_out, exist_ok=True)
            np.savez_compressed(os.path.join(seq_out, 'depths.npz'), depths=depths)
            np.savetxt(os.path.join(seq_out, 'K.txt'), K, fmt='%.6f')
            with open(os.path.join(seq_out, 'frame_ids.txt'), 'w') as f:
                f.write('\n'.join(fids))
            tqdm.write(f'  ✅ {seq_id}  K fx={K[0,0]:.1f}  depths {depths.shape}')
            done += 1
        except Exception as e:
            tqdm.write(f'  ❌ {seq_id}: {e}')
            failed += 1
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ⏭ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"Output: {OUT}")
    print(f"\nNext steps:")
    print(f"  conda activate haptic")
    print(f"  python data/batch_haptic.py --dataset dexycb --seq 841412060263 --obj ycb_dex_02")
    print(f"  conda activate bundlesdf")
    print(f"  python tools/batch_obj_pose.py --dataset dexycb --obj ycb_dex_02 --cam 841412060263")


if __name__ == '__main__':
    main()
