
"""
Contact Region Comparison: GT vs HaWoR predicted contact areas on object mesh.

Approach:
  - GT contact region: accumulate GT MANO vertices vs object mesh across ALL frames
  - Predicted contact region: accumulate HaWoR MANO vertices vs object mesh across ALL frames  
  - Visualize both on the same object: GT(green) vs Pred(red) vs Overlap(yellow)
  - Print original video path for manual verification

Usage:
  conda activate hawor
  python -m data.vis_contact_regions [--threshold 0.03] [--n_seqs 5]
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import trimesh
import torch

# Import project config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

sys.path.insert(0, config.HAWOR_DIR)
from hawor.utils.process import run_mano, run_mano_left

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ─── Paths (from config) ───
ARCTIC_ROOT = config.ARCTIC_ROOT
RAW_SEQS_DIR = os.path.join(ARCTIC_ROOT, "raw_seqs")
META_DIR = os.path.join(ARCTIC_ROOT, "meta")
HAWOR_CACHE = config.HAWOR_CACHE
ONSET_JSON = config.ONSET_JSON
EGOCAM_DIR = os.path.join(ARCTIC_ROOT, "arctic_egocam_videos")
VIS_DIR = config.CONTACT_VIS_DIR

OBJECTS = ['box', 'capsulemachine', 'espressomachine', 'ketchup',
           'laptop', 'microwave', 'mixer', 'notebook', 'phone',
           'scissors', 'waffleiron']


def get_object_name(seq_name):
    for obj in sorted(OBJECTS, key=len, reverse=True):
        if seq_name.startswith(obj):
            return obj
    return None


def load_object_mesh(object_name):
    """Load object mesh template (vertices in meters + faces)."""
    obj_dir = os.path.join(META_DIR, "object_vtemplates", object_name)
    if not os.path.isdir(obj_dir):
        return None, None
    for f in sorted(os.listdir(obj_dir)):
        if f.endswith(('.obj', '.ply', '.stl')):
            mesh = trimesh.load(os.path.join(obj_dir, f), force='mesh')
            return np.array(mesh.vertices) / 1000.0, np.array(mesh.faces)  # mm → m
    return None, None


def gt_mano_verts(rot, pose, trans, shape, hand='right'):
    """MANO forward pass for GT vertices."""
    N = rot.shape[0]
    trans_t = torch.from_numpy(trans).float().unsqueeze(0)
    rot_t = torch.from_numpy(rot).float().unsqueeze(0)
    pose_t = torch.from_numpy(pose).float().unsqueeze(0)
    shape_t = torch.from_numpy(np.tile(shape, (N, 1))).float().unsqueeze(0)
    if hand == 'right':
        out = run_mano(trans_t, rot_t, pose_t, betas=shape_t, use_cuda=False)
    else:
        out = run_mano_left(trans_t, rot_t, pose_t, betas=shape_t, use_cuda=False)
    return out['vertices'][0].numpy()


def visualize_comparison(obj_verts, obj_faces, gt_min_dists, pred_min_dists,
                          gt_counts, pred_counts,
                          threshold, save_path, title=""):
    """Side-by-side heatmap: GT (green gradient) | Predicted (red gradient) | Overlap."""
    gt_contact = gt_min_dists < threshold
    pred_contact = pred_min_dists < threshold
    
    # Normalize counts to [0, 1] for color intensity
    gt_max = max(gt_counts.max(), 1)
    pred_max = max(pred_counts.max(), 1)
    gt_intensity = gt_counts / gt_max    # 0=never, 1=most contacted
    pred_intensity = pred_counts / pred_max
    
    fig = plt.figure(figsize=(18, 6))
    center = obj_verts.mean(0)
    span = max(obj_verts.max(0) - obj_verts.min(0)) * 0.6
    polygons = obj_verts[obj_faces]
    
    for idx in range(3):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        face_colors = []
        
        for f in obj_faces:
            if idx == 0:  # GT heatmap (green gradient)
                intensity = gt_intensity[f].max()
                if intensity > 0:
                    # Light green → dark green as intensity increases
                    g_val = 0.4 + 0.6 * intensity
                    face_colors.append([0.1, g_val, 0.1, 0.5 + 0.5 * intensity])
                else:
                    face_colors.append([0.75, 0.75, 0.75, 0.3])
            elif idx == 1:  # Predicted heatmap (red gradient)
                intensity = pred_intensity[f].max()
                if intensity > 0:
                    r_val = 0.4 + 0.6 * intensity
                    face_colors.append([r_val, 0.1, 0.1, 0.5 + 0.5 * intensity])
                else:
                    face_colors.append([0.75, 0.75, 0.75, 0.3])
            else:  # Overlap view
                g = gt_contact[f].any()
                p = pred_contact[f].any()
                if g and p:
                    face_colors.append([1.0, 0.9, 0.0, 0.9])   # yellow
                elif g:
                    face_colors.append([0.2, 0.8, 0.2, 0.9])   # green
                elif p:
                    face_colors.append([0.9, 0.2, 0.2, 0.9])   # red
                else:
                    face_colors.append([0.75, 0.75, 0.75, 0.3])
        
        mesh_col = Poly3DCollection(polygons, facecolors=face_colors,
                                     edgecolors='none', linewidths=0.1)
        ax.add_collection3d(mesh_col)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        
        if idx == 0:
            pct = gt_contact.mean() * 100
            ax.set_title(f"GT Contact (green heatmap)\n{pct:.1f}%", fontsize=10)
        elif idx == 1:
            pct = pred_contact.mean() * 100
            ax.set_title(f"HaWoR Predicted (red heatmap)\n{pct:.1f}%", fontsize=10)
        else:
            overlap = (gt_contact & pred_contact).sum()
            union = (gt_contact | pred_contact).sum()
            region_iou = overlap / union if union > 0 else 0
            ax.set_title(f"Overlap (yellow=both)\nRegion IoU: {region_iou:.2f}", fontsize=10)
        
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    
    fig.suptitle(f"{title} @ {int(threshold*1000)}mm", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_sequence(subject, seq_name, hawor_data, onset_info, 
                      obj_verts_template, obj_faces, threshold):
    """
    Compute BOTH GT and predicted contact regions accumulated across all frames.
    Returns per-vertex minimum distances for both GT and HaWoR.
    """
    mano_path = os.path.join(RAW_SEQS_DIR, subject, f"{seq_name}.mano.npy")
    obj_path = os.path.join(RAW_SEQS_DIR, subject, f"{seq_name}.object.npy")
    ego_path = os.path.join(RAW_SEQS_DIR, subject, f"{seq_name}.egocam.dist.npy")
    
    if not all(os.path.exists(p) for p in [mano_path, obj_path, ego_path]):
        return None, "missing_gt"
    
    mano_gt = np.load(mano_path, allow_pickle=True).item()
    obj_data = np.load(obj_path, allow_pickle=True)
    egocam = np.load(ego_path, allow_pickle=True).flat[0]
    R_ego = egocam['R_k_cam_np']
    T_ego = egocam['T_k_cam_np']
    N_gt = obj_data.shape[0]
    N_hawor = hawor_data['right_verts'].shape[0]
    
    clip_start = onset_info['clip_start'] if onset_info else 0
    fi_start = clip_start
    fi_end = min(clip_start + N_hawor, N_gt)
    clip_len = fi_end - fi_start
    if clip_len < 2:
        return None, "clip_too_short"
    
    # GT MANO forward pass
    try:
        gt_right_world = gt_mano_verts(
            mano_gt['right']['rot'][fi_start:fi_end],
            mano_gt['right']['pose'][fi_start:fi_end],
            mano_gt['right']['trans'][fi_start:fi_end],
            mano_gt['right']['shape'], hand='right'
        )
        gt_left_world = gt_mano_verts(
            mano_gt['left']['rot'][fi_start:fi_end],
            mano_gt['left']['pose'][fi_start:fi_end],
            mano_gt['left']['trans'][fi_start:fi_end],
            mano_gt['left']['shape'], hand='left'
        )
    except Exception as e:
        return None, f"mano_error: {e}"
    
    n_verts = len(obj_verts_template)
    gt_min_dists = np.full(n_verts, 1e6)    # GT hand → object
    pred_min_dists = np.full(n_verts, 1e6)   # HaWoR hand → object
    gt_counts = np.zeros(n_verts, dtype=int)   # frames where GT contacts vertex
    pred_counts = np.zeros(n_verts, dtype=int) # frames where pred contacts vertex
    
    # Iterate frames (every 5th for speed)
    for i in range(0, clip_len, 5):
        fi_gt = fi_start + i
        fi_hawor = i
        if fi_hawor >= N_hawor or fi_gt >= N_gt:
            break
        
        # Object mesh in camera space
        obj_rot = Rotation.from_rotvec(obj_data[fi_gt, 1:4]).as_matrix()
        obj_trans = obj_data[fi_gt, 4:7] / 1000.0
        obj_world = (obj_rot @ obj_verts_template.T).T + obj_trans
        R_cam = R_ego[fi_gt]
        T_cam = T_ego[fi_gt].flatten()
        obj_cam = (R_cam @ obj_world.T).T + T_cam
        
        # GT hand in camera space
        gt_r_cam = (R_cam @ gt_right_world[i].T).T + T_cam
        gt_l_cam = (R_cam @ gt_left_world[i].T).T + T_cam
        gt_both_cam = np.concatenate([gt_r_cam, gt_l_cam], axis=0)
        
        # GT contact distances
        gt_dists, _ = cKDTree(gt_both_cam).query(obj_cam)
        gt_min_dists = np.minimum(gt_min_dists, gt_dists)
        gt_counts += (gt_dists < threshold).astype(int)
        
        # HaWoR hand in camera space + centroid alignment
        hawor_r = hawor_data['right_verts'][fi_hawor]
        hawor_l = hawor_data['left_verts'][fi_hawor]
        R_w2c = hawor_data['R_w2c'][fi_hawor]
        t_w2c = hawor_data['t_w2c'][fi_hawor]
        hawor_r_cam = (R_w2c @ hawor_r.T).T + t_w2c
        hawor_l_cam = (R_w2c @ hawor_l.T).T + t_w2c
        hawor_both_cam = np.concatenate([hawor_r_cam, hawor_l_cam], axis=0)
        
        # Centroid alignment
        gt_centroid = gt_both_cam.mean(0)
        hawor_centroid = hawor_both_cam.mean(0)
        offset = gt_centroid - hawor_centroid
        hawor_aligned = hawor_both_cam + offset
        
        # Predicted contact distances
        pred_dists, _ = cKDTree(hawor_aligned).query(obj_cam)
        pred_min_dists = np.minimum(pred_min_dists, pred_dists)
        pred_counts += (pred_dists < threshold).astype(int)
    
    gt_contact = gt_min_dists < threshold  # tight GT
    
    if gt_contact.sum() == 0:
        return None, "no_contact"
    
    return {
        'gt_min_dists': gt_min_dists,
        'pred_min_dists': pred_min_dists,
        'gt_counts': gt_counts,
        'pred_counts': pred_counts,
        'gt_contact': gt_contact,
        'n_verts': n_verts,
    }, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.050,
                        help='Prediction contact threshold in meters (default: 50mm)')
    parser.add_argument('--gt_threshold', type=float, default=0.015,
                        help='GT contact threshold in meters (default: 15mm, true contact)')
    parser.add_argument('--n_seqs', type=int, default=10,
                        help='Number of sequences to visualize')
    parser.add_argument('--cache_dir', type=str, default=HAWOR_CACHE)
    args = parser.parse_args()
    
    os.makedirs(VIS_DIR, exist_ok=True)
    th = args.threshold
    gt_th = args.gt_threshold
    th_mm = int(th * 1000)
    gt_th_mm = int(gt_th * 1000)
    
    # Load onset
    onset_data = {}
    if os.path.exists(ONSET_JSON):
        with open(ONSET_JSON) as f:
            onset_data = json.load(f)
    
    # Load mesh templates
    mesh_cache = {}
    for obj in OBJECTS:
        verts, faces = load_object_mesh(obj)
        if verts is not None:
            mesh_cache[obj] = {'verts': verts, 'faces': faces}
    print(f"📦 Loaded {len(mesh_cache)} object meshes")
    
    # Process cache
    cached_files = sorted([f for f in os.listdir(args.cache_dir) if f.endswith('.npz')])
    print(f"📦 Cache: {len(cached_files)} files")
    print(f"📏 GT threshold: {gt_th_mm}mm (true contact) | Pred threshold: {th_mm}mm")
    
    results_by_obj = defaultdict(list)
    skip_reasons = defaultdict(int)
    vis_count = 0
    all_region_ious = []
    
    for cf in tqdm(cached_files, desc="Processing"):
        parts = cf.replace('.npz', '').split('__')
        if len(parts) != 2:
            continue
        subj, seq = parts
        obj = get_object_name(seq)
        if obj not in mesh_cache:
            skip_reasons['no_mesh'] += 1
            continue
        
        try:
            hd = dict(np.load(os.path.join(args.cache_dir, cf)))
        except Exception:
            skip_reasons['bad_file'] += 1
            continue
        
        key = f"{subj}__{seq}"
        onset = onset_data.get(key)
        
        out, status = process_sequence(
            subj, seq, hd, onset,
            mesh_cache[obj]['verts'], mesh_cache[obj]['faces'], gt_th
        )
        
        if out is None:
            skip_reasons[status] += 1
            continue
        
        # Apply pred threshold
        pred_contact = out['pred_min_dists'] < th
        gt_contact = out['gt_contact']
        n_verts = out['n_verts']
        TP = int((gt_contact & pred_contact).sum())
        FP = int((pred_contact & ~gt_contact).sum())
        FN = int((gt_contact & ~pred_contact).sum())
        union = (gt_contact | pred_contact).sum()
        iou = float(TP / union) if union > 0 else 0.0
        prec = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
        rec = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fp_rate = float(FP / n_verts * 100)
        
        results_by_obj[obj].append({
            'subj': subj, 'seq': seq,
            'iou': iou, 'precision': prec, 'recall': rec, 'f1': f1, 'fp_rate': fp_rate,
        })
        all_region_ious.append(iou)
        
        # Visualize first N
        if vis_count < args.n_seqs:
            save_path = os.path.join(VIS_DIR, f"{subj}_{seq}_{th_mm}mm.png")
            visualize_comparison(
                mesh_cache[obj]['verts'], mesh_cache[obj]['faces'],
                out['gt_min_dists'], out['pred_min_dists'],
                out['gt_counts'], out['pred_counts'],
                gt_th, save_path, title=f"{subj}/{seq}"
            )
            # Find original video
            video_path = os.path.join(EGOCAM_DIR, subj, f"{seq}.mp4")
            if not os.path.exists(video_path):
                video_path = "(not found)"
            print(f"  💾 {save_path}")
            print(f"     IoU={iou:.2f} P={prec:.2f} R={rec:.2f} F1={f1:.2f} FP={fp_rate:.1f}%")
            print(f"     📹 Video: {video_path}")
            vis_count += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 GT@{gt_th_mm}mm vs Pred@{th_mm}mm")
    print(f"{'='*70}")
    print(f"  Evaluated: {len(all_region_ious)} seqs")
    print(f"  Skipped: {sum(skip_reasons.values())} {dict(skip_reasons)}")
    
    if all_region_ious:
        all_entries = [e for objs in results_by_obj.values() for e in objs]
        all_prec = [e['precision'] for e in all_entries]
        all_recall = [e['recall'] for e in all_entries]
        all_f1 = [e['f1'] for e in all_entries]
        all_fp = [e['fp_rate'] for e in all_entries]
        
        print(f"\n  Mean IoU:       {np.mean(all_region_ious):.3f}")
        print(f"  Mean Precision: {np.mean(all_prec):.3f}")
        print(f"  Mean Recall:    {np.mean(all_recall):.3f}")
        print(f"  Mean F1:        {np.mean(all_f1):.3f}  ⭐")
        print(f"  Mean FP rate:   {np.mean(all_fp):.1f}%")
        
        print(f"\n  Per-object:")
        for obj in sorted(results_by_obj.keys()):
            entries = results_by_obj[obj]
            ious = [e['iou'] for e in entries]
            precs = [e['precision'] for e in entries]
            recs = [e['recall'] for e in entries]
            f1s = [e['f1'] for e in entries]
            fps = [e['fp_rate'] for e in entries]
            print(f"    {obj}: F1={np.mean(f1s):.3f}  P={np.mean(precs):.3f}  R={np.mean(recs):.3f}  FP={np.mean(fps):.1f}% ({len(entries)} seqs)")
    
    print(f"\n📸 Visualizations: {VIS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
