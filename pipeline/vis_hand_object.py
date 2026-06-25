#!/usr/bin/env python3
"""
Visualize MANO hand mesh + Object mesh in aligned camera space.

Proves that video-extracted 3D hand reconstruction is spatially aligned
with the object mesh — key evidence for the pipeline's validity.

Renders a single best-contact frame showing the hand grasping the object, with:
  - Object mesh in solid grey/blue with shading
  - Right hand mesh in warm red/orange
  - Left hand mesh in green
  - Contact faces highlighted in yellow

Usage:
    conda activate hawor
    # Run from project root
    python -m data.vis_hand_object --object ketchup
    python -m data.vis_hand_object --object ketchup --n_frames 3 --source hawor
"""

import os
import sys
import pickle
import argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_mano_faces():
    """Load MANO hand topology (1538 faces, fixed for all MANO meshes)."""
    pkl_path = os.path.join(config.MANO_MODELS, "MANO_RIGHT.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return np.array(data['f'], dtype=np.int64)

# Import pipeline utilities
from data.video_to_training_data import (
    load_hawor_caches, load_haptic_caches, load_onset_annotations,
    load_arctic_object_poses, load_arctic_egocam, load_arctic_3rd_cam,
    load_gt_mano_verts, get_object_from_sequence, OBJECTS
)
from data.mesh_provider import get_provider


def load_object_mesh_arctic(object_name):
    """Load ARCTIC object mesh template (vertices in meters)."""
    meta_dir = os.path.join(config.ARCTIC_ROOT, "meta")
    obj_dir = os.path.join(meta_dir, "object_vtemplates", object_name)
    if not os.path.isdir(obj_dir):
        return None, None
    import trimesh
    for f in sorted(os.listdir(obj_dir)):
        if f.endswith(('.obj', '.ply', '.stl')):
            mesh = trimesh.load(os.path.join(obj_dir, f), force='mesh')
            return np.array(mesh.vertices) / 1000.0, np.array(mesh.faces)
    return None, None


def find_best_frames(hand_verts, obj_verts_cam_list, threshold=0.03, n_frames=6):
    """Find frames with most contact (best for visualization)."""
    contact_scores = []
    for i, obj_cam in enumerate(obj_verts_cam_list):
        if obj_cam is None:
            contact_scores.append(0)
            continue
        tree = cKDTree(hand_verts[i])
        dists, _ = tree.query(obj_cam)
        n_contact = np.sum(dists < threshold)
        contact_scores.append(n_contact)
    
    # Sort by contact count, pick top N
    ranked = np.argsort(contact_scores)[::-1]
    best = [i for i in ranked if contact_scores[i] > 0][:n_frames]
    return sorted(best)


def render_hand_object_frame(ax, obj_verts, obj_faces,
                              right_hand=None, left_hand=None,
                              mano_faces=None,
                              contact_th=0.03, title="", elev=25, azim=45):
    """Render one frame: object mesh (solid 3D) + hand mesh (solid)."""

    # ── Find contact on object ──────────────────────────────────────────
    all_hand_pts = []
    if right_hand is not None:
        all_hand_pts.append(right_hand)
    if left_hand is not None:
        all_hand_pts.append(left_hand)

    contact_mask = np.zeros(len(obj_verts), dtype=bool)
    if all_hand_pts:
        all_pts = np.concatenate(all_hand_pts, axis=0)
        tree = cKDTree(all_pts)
        dists, _ = tree.query(obj_verts)
        contact_mask = dists < contact_th

    # ── Object mesh — solid blue-grey with contact yellow ───────────────
    obj_polygons = obj_verts[obj_faces]
    # Per-face colors
    obj_fc = np.full((len(obj_faces), 4), [0.55, 0.70, 0.88, 0.85])  # steel blue, mostly opaque
    for fi, face in enumerate(obj_faces):
        if contact_mask[face].any():
            obj_fc[fi] = [1.0, 0.85, 0.15, 0.95]  # contact = bright yellow
    obj_col = Poly3DCollection(obj_polygons, facecolors=obj_fc,
                                edgecolors=[0.35, 0.35, 0.45, 0.25], linewidths=0.15)
    ax.add_collection3d(obj_col)

    # ── Hand mesh — render as solid Poly3DCollection if faces available ──
    def add_hand_mesh(verts, faces, base_color, side_label):
        if faces is None:
            # Fallback to dense scatter if no faces
            ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                       c=[base_color], s=2.0, alpha=0.85, depthshade=True)
            return
        polys = verts[faces]
        # Soft shading: slightly darken backfaces using face normal z-component
        normals = np.cross(polys[:, 1] - polys[:, 0], polys[:, 2] - polys[:, 0])
        norms_len = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        normals = normals / norms_len
        # Light from upper-left: [0.5, 0.5, 0.7]
        light = np.array([0.5, 0.5, 0.8])
        light /= np.linalg.norm(light)
        shading = np.clip(normals @ light, 0.2, 1.0)   # 0.2 ambient
        r, g, b = base_color
        fc = np.column_stack([shading * r, shading * g, shading * b,
                               np.full(len(faces), 0.90)])
        hand_col = Poly3DCollection(polys, facecolors=fc,
                                     edgecolors=[r*0.5, g*0.5, b*0.5, 0.10],
                                     linewidths=0.05)
        ax.add_collection3d(hand_col)

    if right_hand is not None:
        add_hand_mesh(right_hand, mano_faces, (0.85, 0.25, 0.15), 'Right')
    if left_hand is not None:
        add_hand_mesh(left_hand, mano_faces, (0.18, 0.72, 0.35), 'Left')

    # ── Set view bounds ─────────────────────────────────────────────────
    all_pts_list = [obj_verts]
    if right_hand is not None:
        all_pts_list.append(right_hand)
    if left_hand is not None:
        all_pts_list.append(left_hand)
    all_combined = np.concatenate(all_pts_list, axis=0)

    center = all_combined.mean(0)
    span = max(all_combined.max(0) - all_combined.min(0)) * 0.60
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    n_contact = contact_mask.sum()
    ax.set_title(f"{title}\n{n_contact} contact verts", fontsize=9)


def process_hawor_sequence(subject, seq_name, object_name, obj_template, obj_faces,
                            hawor_data, onset, n_frames=6):
    """Process one HaWoR sequence: align hand + object in camera space."""
    # Load GT data
    obj_rots, obj_trans = load_arctic_object_poses(subject, seq_name)
    if obj_rots is None:
        return None
    
    R_ego, T_ego = load_arctic_egocam(subject, seq_name)
    if R_ego is None:
        return None
    
    gt_right, gt_left = load_gt_mano_verts(subject, seq_name)
    
    # Get clip range
    key = f"{subject}/{seq_name}"
    clip_start = onset.get(key, {}).get('start', 0)
    
    N_hawor = len(hawor_data.get('right_verts', []))
    N_gt = len(obj_rots)
    
    frames = []
    for i in range(0, min(N_hawor, N_gt - clip_start), max(1, N_hawor // 30)):
        fi_gt = clip_start + i
        if fi_gt >= N_gt or fi_gt >= len(R_ego):
            break
        
        # Object in camera space
        R_obj = Rotation.from_rotvec(obj_rots[fi_gt]).as_matrix()
        t_obj = obj_trans[fi_gt]
        obj_world = (R_obj @ obj_template.T).T + t_obj
        R_cam = R_ego[fi_gt]
        T_cam = T_ego[fi_gt].flatten()
        obj_cam = (R_cam @ obj_world.T).T + T_cam
        
        # HaWoR hand in camera space
        right_cam, left_cam = None, None
        
        if 'right_verts' in hawor_data and i < len(hawor_data['right_verts']):
            rv = hawor_data['right_verts'][i]
            R_w2c = hawor_data.get('R_w2c', None)
            t_w2c = hawor_data.get('t_w2c', None)
            if R_w2c is not None and i < len(R_w2c):
                right_cam = (R_w2c[i] @ rv.T).T + t_w2c[i]
            else:
                right_cam = rv
            
            # Centroid alignment using GT MANO
            if gt_right is not None and fi_gt < len(gt_right):
                gt_r_cam = (R_cam @ gt_right[fi_gt].T).T + T_cam
                offset = gt_r_cam.mean(0) - right_cam.mean(0)
                right_cam = right_cam + offset
        
        if 'left_verts' in hawor_data and i < len(hawor_data['left_verts']):
            lv = hawor_data['left_verts'][i]
            R_w2c = hawor_data.get('R_w2c', None)
            t_w2c = hawor_data.get('t_w2c', None)
            if R_w2c is not None and i < len(R_w2c):
                left_cam = (R_w2c[i] @ lv.T).T + t_w2c[i]
            else:
                left_cam = lv
            
            if gt_left is not None and fi_gt < len(gt_left):
                gt_l_cam = (R_cam @ gt_left[fi_gt].T).T + T_cam
                offset = gt_l_cam.mean(0) - left_cam.mean(0)
                left_cam = left_cam + offset
        
        frames.append({
            'frame_idx': fi_gt,
            'obj_cam': obj_cam,
            'right_cam': right_cam,
            'left_cam': left_cam,
        })
    
    if not frames:
        return None
    
    # Score frames by contact, pick best
    contact_scores = []
    for fr in frames:
        hands = []
        if fr['right_cam'] is not None:
            hands.append(fr['right_cam'])
        if fr['left_cam'] is not None:
            hands.append(fr['left_cam'])
        if hands:
            all_h = np.concatenate(hands)
            dists, _ = cKDTree(all_h).query(fr['obj_cam'])
            contact_scores.append(np.sum(dists < 0.03))
        else:
            contact_scores.append(0)
    
    ranked = np.argsort(contact_scores)[::-1]
    best_indices = sorted(ranked[:n_frames])
    
    return [frames[i] for i in best_indices]


def process_haptic_sequence(subject, seq_name, cam_id, object_name, 
                             obj_template, obj_faces, haptic_verts, onset, n_frames=6):
    """Process one HaPTIC sequence: align hand + object in camera space."""
    obj_rots, obj_trans = load_arctic_object_poses(subject, seq_name)
    if obj_rots is None:
        return None
    
    R_cam, T_cam = load_arctic_3rd_cam(subject, cam_id)
    if R_cam is None:
        return None
    T_cam = T_cam.flatten()
    
    gt_right, gt_left = load_gt_mano_verts(subject, seq_name)
    
    key = f"{subject}/{seq_name}"
    clip_start = onset.get(key, {}).get('start', 0)
    
    N_haptic = len(haptic_verts)
    N_gt = len(obj_rots)
    
    frames = []
    for i in range(0, min(N_haptic, N_gt - clip_start), max(1, N_haptic // 30)):
        fi_gt = clip_start + i
        if fi_gt >= N_gt:
            break
        
        # Object in camera space
        R_obj = Rotation.from_rotvec(obj_rots[fi_gt]).as_matrix()
        t_obj = obj_trans[fi_gt]
        obj_world = (R_obj @ obj_template.T).T + t_obj
        obj_cam = (R_cam @ obj_world.T).T + T_cam
        
        # HaPTIC hand (already in camera space)
        hand_cam = haptic_verts[i]
        
        # Centroid alignment 
        if gt_right is not None and fi_gt < len(gt_right):
            gt_r_cam = (R_cam @ gt_right[fi_gt].T).T + T_cam
            if gt_left is not None and fi_gt < len(gt_left):
                gt_l_cam = (R_cam @ gt_left[fi_gt].T).T + T_cam
                gt_both = np.concatenate([gt_r_cam, gt_l_cam])
            else:
                gt_both = gt_r_cam
            offset = gt_both.mean(0) - hand_cam.mean(0)
            hand_cam = hand_cam + offset
        
        frames.append({
            'frame_idx': fi_gt,
            'obj_cam': obj_cam,
            'right_cam': hand_cam,  # HaPTIC doesn't distinguish L/R
            'left_cam': None,
        })
    
    if not frames:
        return None
    
    # Score and pick best
    contact_scores = []
    for fr in frames:
        dists, _ = cKDTree(fr['right_cam']).query(fr['obj_cam'])
        contact_scores.append(np.sum(dists < 0.03))
    
    ranked = np.argsort(contact_scores)[::-1]
    best_indices = sorted(ranked[:n_frames])
    return [frames[i] for i in best_indices]


def main():
    parser = argparse.ArgumentParser(description="Visualize Hand + Object alignment")
    parser.add_argument('--object', default='ketchup', help='Object name')
    parser.add_argument('--source', default='auto', choices=['hawor', 'haptic', 'auto'],
                        help='Data source')
    parser.add_argument('--out_dir', default=None, help='Output directory')
    args = parser.parse_args()

    out_dir = args.out_dir or config.CONTACT_VIS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Load MANO faces (for solid hand mesh rendering)
    mano_faces = load_mano_faces()
    if mano_faces is None:
        print("⚠️  MANO_RIGHT.pkl not found — hands will render as point clouds")
    else:
        print(f"✅ Loaded MANO faces: {mano_faces.shape}")

    # Load object mesh
    obj_template, obj_faces = load_object_mesh_arctic(args.object)
    if obj_template is None:
        print(f"❌ No object mesh for '{args.object}'")
        return
    
    onset = load_onset_annotations()
    
    # Determine source
    use_hawor = args.source in ('hawor', 'auto')
    use_haptic = args.source in ('haptic', 'auto')
    
    all_frame_data = []
    seq_labels = []
    
    # Process HaWoR sequences
    if use_hawor:
        cache_dir = config.HAWOR_CACHE
        if os.path.isdir(cache_dir):
            for fn in sorted(os.listdir(cache_dir)):
                if not fn.endswith('.npz'):
                    continue
                parts = fn.replace('.npz', '').split('__')
                if len(parts) != 2:
                    continue
                subject, seq_name = parts
                if get_object_from_sequence(seq_name) != args.object:
                    continue
                
                print(f"  📹 Processing HaWoR: {subject}/{seq_name}")
                hawor_data = dict(np.load(os.path.join(cache_dir, fn), allow_pickle=True))
                
                frames = process_hawor_sequence(
                    subject, seq_name, args.object, obj_template, obj_faces,
                    hawor_data, onset, n_frames=9999  # collect all, user will pick
                )
                if frames:
                    all_frame_data.extend(frames)
                    seq_labels.extend([f"HaWoR {subject}/{seq_name} f{fr['frame_idx']}" 
                                       for fr in frames])
                    break  # Use first valid sequence
    
    # Process HaPTIC sequences
    if use_haptic:
        cache_dir = config.HAPTIC_CACHE
        if os.path.isdir(cache_dir):
            for fn in sorted(os.listdir(cache_dir)):
                if not fn.endswith('.npz'):
                    continue
                parts = fn.replace('.npz', '').split('__')
                if len(parts) != 2:
                    continue
                rest = parts[1]
                if get_object_from_sequence(rest) != args.object:
                    continue
                subject = parts[0]
                seq_name = rest.rsplit('_cam', 1)[0]
                cam_id = 1
                if '_cam' in rest:
                    try:
                        cam_id = int(rest.rsplit('_cam', 1)[1])
                    except ValueError:
                        cam_id = 1
                
                # Load verts
                data = dict(np.load(os.path.join(cache_dir, fn), allow_pickle=True))
                verts = None
                if 'verts_dict' in data:
                    vd = data['verts_dict'].item() if data['verts_dict'].ndim == 0 else data['verts_dict']
                    if isinstance(vd, dict):
                        max_key = max(vd.keys())
                        frames_list = [vd[k] for k in range(max_key + 1) if k in vd]
                        if frames_list:
                            verts = np.stack(frames_list, axis=0)
                elif 'hand_verts' in data:
                    verts = data['hand_verts']
                elif 'right_verts' in data:
                    verts = data['right_verts']
                
                if verts is None or verts.ndim != 3:
                    continue
                
                # Apply onset clipping
                key = f"{subject}/{seq_name}"
                clip_start = onset.get(key, {}).get('start', 0)
                clip_end = onset.get(key, {}).get('end', len(verts) - 1)
                if clip_end == -1:
                    clip_end = len(verts) - 1
                verts = verts[clip_start:clip_end + 1]
                
                print(f"  📹 Processing HaPTIC: {subject}/{seq_name} cam{cam_id}")
                frames = process_haptic_sequence(
                    subject, seq_name, cam_id, args.object,
                    obj_template, obj_faces, verts, onset, n_frames=9999  # collect all
                )
                if frames:
                    all_frame_data.extend(frames)
                    seq_labels.extend([f"HaPTIC {subject}/{seq_name} cam{cam_id} f{fr['frame_idx']}" 
                                       for fr in frames])
                    break  # Use first valid sequence
    
    if not all_frame_data:
        print("❌ No valid frames found")
        return

    # ── Interactive frame selection ──────────────────────────────────────
    # Compute contact score for each candidate
    def contact_score(fr):
        hands = [h for h in [fr['right_cam'], fr['left_cam']] if h is not None]
        if not hands:
            return 0
        dists, _ = cKDTree(np.concatenate(hands)).query(fr['obj_cam'])
        return int(np.sum(dists < 0.03))

    scores = [contact_score(fr) for fr in all_frame_data]

    print(f"\n{'─'*55}")
    print(f"  #   GT-frame   contact-verts   source")
    print(f"{'─'*55}")
    for i, (fr, lbl, sc) in enumerate(zip(all_frame_data, seq_labels, scores)):
        bar = '█' * min(sc // 10, 20)
        print(f"  {i:2d}  f{fr['frame_idx']:<8d}  {sc:>5d}  {bar}")
        print(f"      {lbl}")
    print(f"{'─'*55}")

    raw = input("\n请输入要渲染的帧编号（逗号分隔，或 'all'，或直接回车选接触最多那帧）: ").strip()
    if raw.lower() == 'all':
        chosen = list(range(len(all_frame_data)))
    elif raw == '':
        chosen = [int(np.argmax(scores))]
    else:
        try:
            chosen = [int(x.strip()) for x in raw.split(',')]
            chosen = [c for c in chosen if 0 <= c < len(all_frame_data)]
        except ValueError:
            print("⚠️  输入无效，自动选接触最多的帧")
            chosen = [int(np.argmax(scores))]

    all_frame_data = [all_frame_data[i] for i in chosen]
    seq_labels     = [seq_labels[i]     for i in chosen]

    n = len(all_frame_data)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    print(f"\n🎨 Rendering {n} frame(s): {chosen}")
    
    fig = plt.figure(figsize=(7 * cols, 6 * rows + 1))
    
    azimuths = [30, 60, 120, -30, 150, 0]
    
    for i, (fr, label) in enumerate(zip(all_frame_data, seq_labels)):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        render_hand_object_frame(
            ax, fr['obj_cam'], obj_faces,
            right_hand=fr['right_cam'],
            left_hand=fr['left_cam'],
            mano_faces=mano_faces,
            contact_th=0.03,
            title=label,
            elev=25,
            azim=azimuths[i % len(azimuths)]
        )
    
    fig.suptitle(
        f"{args.object} — Hand-Object Spatial Alignment from Video\n"
        f"Object mesh (blue) + MANO hand (red/green) + Contact (yellow)",
        fontsize=14, fontweight='bold', y=0.98
    )
    
    # Legend
    fig.text(0.5, 0.01,
             "🔴 Right Hand    🟢 Left Hand    🔵 Object Mesh    🟡 Contact Region (<30mm)",
             ha='center', fontsize=11, style='italic')

    save_path = os.path.join(out_dir, f"{args.object}_hand_object_alignment.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n💾 Saved: {save_path}")


if __name__ == '__main__':
    main()
