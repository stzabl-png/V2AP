#!/usr/bin/env python3
"""
Visualize merged (HaWoR + HaPTIC) contact heatmap on object mesh.

Blue = no contact, Red = high contact frequency.
Generates multi-view PNG images.

Usage:
    conda activate hawor
    # Run from project root
    python -m data.vis_merged_heatmap --object ketchup
    python -m data.vis_merged_heatmap --object ketchup --views 4
"""

import os
import sys
import argparse
import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable


def load_mesh_and_contact(object_name):
    """Load object mesh + merged contact frequency from video_to_training_data output."""
    # Load mesh via mesh_provider
    from data.mesh_provider import get_provider
    provider = get_provider('arctic')
    mesh = provider.get_mesh(object_name)
    if mesh is None:
        print(f"  ⚠️  No mesh for '{object_name}'")
        return None, None, None

    mesh_verts = np.array(mesh.vertices) / 1000.0  # mm → m
    mesh_faces = np.array(mesh.faces)

    # Load contact from training data
    hp_path = os.path.join(config.DATA_HUB, 'training_m5', f'video_{object_name}.hdf5')
    if not os.path.exists(hp_path):
        print(f"  ⚠️  No training data at {hp_path}")
        print(f"       Run: python -m data.video_to_training_data --objects {object_name}")
        return None, None, None

    # We need per-vertex contact, not per-point-cloud contact.
    # Re-run accumulation to get vertex-level data, or use video_to_training_data internals.
    return mesh_verts, mesh_faces, object_name


def compute_vertex_contact(object_name):
    """Re-compute per-vertex contact frequency using the pipeline logic."""
    from data.video_to_training_data import (
        load_hawor_caches, load_haptic_caches, load_onset_annotations,
        load_arctic_object_poses, load_arctic_egocam, load_arctic_3rd_cam,
        load_gt_mano_verts, accumulate_contact_labels,
        get_object_from_sequence, resolve_source
    )
    from data.mesh_provider import get_provider

    provider = get_provider('arctic')
    mesh = provider.get_mesh(object_name)
    if mesh is None:
        return None, None, None

    mesh_verts_m = np.array(mesh.vertices) / 1000.0
    mesh_faces = np.array(mesh.faces)

    effective_source, label = resolve_source(object_name, 'auto')
    if effective_source == 'none':
        print(f"  ⚠️  No data for {object_name}")
        return None, None, None

    # Build hand_verts_list + metadata (same as process_object)
    hand_verts_list = []
    seq_metadata = []
    frame_offsets = []
    onset = load_onset_annotations()
    pred_threshold = config.PRED_CONTACT_TH

    if effective_source in ('hawor', 'both'):
        hawor_data = load_hawor_caches(object_name)
        cache_dir = config.HAWOR_CACHE
        for fn in sorted(os.listdir(cache_dir)):
            if not fn.endswith('.npz'):
                continue
            parts = fn.replace('.npz', '').split('__')
            if len(parts) != 2:
                continue
            subject, seq_name = parts
            obj = get_object_from_sequence(seq_name)
            if obj != object_name:
                continue
            key = f"{subject}/{seq_name}"
            clip_start = onset[key].get('start', 0) if key in onset else 0
            data = dict(np.load(os.path.join(cache_dir, fn), allow_pickle=True))
            for hand_key in ['right_verts', 'left_verts']:
                if hand_key in data and data[hand_key].ndim == 3:
                    seq_metadata.append((subject, seq_name, 'hawor'))
                    frame_offsets.append(clip_start)
        hand_verts_list.extend(hawor_data)

    n_hawor = len(hand_verts_list)

    if effective_source in ('haptic', 'both'):
        haptic_data = load_haptic_caches(object_name)
        cache_dir = config.HAPTIC_CACHE
        for fn in sorted(os.listdir(cache_dir)):
            if not fn.endswith('.npz'):
                continue
            parts = fn.replace('.npz', '').split('__')
            if len(parts) != 2:
                continue
            rest = parts[1]
            obj = get_object_from_sequence(rest)
            if obj != object_name:
                continue
            subject = parts[0]
            seq_name = rest.rsplit('_cam', 1)[0]
            key = f"{subject}/{seq_name}"
            clip_start = onset[key].get('start', 0) if key in onset else 0
            cam_id = 1
            if '_cam' in rest:
                try:
                    cam_id = int(rest.rsplit('_cam', 1)[1])
                except ValueError:
                    cam_id = 1
            seq_metadata.append((subject, seq_name, f'haptic_cam{cam_id}'))
            frame_offsets.append(clip_start)
        hand_verts_list.extend(haptic_data)

    n_haptic = len(hand_verts_list) - n_hawor

    # Load poses + extrinsics + GT MANO
    obj_poses = []
    cam_extrinsics = []
    gt_mano_data = []
    gt_mano_cache = {}
    for i, (subject, seq_name, src_type) in enumerate(seq_metadata):
        obj_rots, obj_trans = load_arctic_object_poses(subject, seq_name)
        obj_poses.append((obj_rots, obj_trans) if obj_rots is not None else None)

        R_cam_ext, T_cam_ext = None, None
        if src_type == 'hawor':
            R_cam_ext, T_cam_ext = load_arctic_egocam(subject, seq_name)
            cam_extrinsics.append((R_cam_ext, T_cam_ext) if R_cam_ext is not None else None)
        elif src_type.startswith('haptic_cam'):
            cam_id = int(src_type.split('cam')[1])
            R_cam_ext, T_cam_ext = load_arctic_3rd_cam(subject, cam_id)
            cam_extrinsics.append((R_cam_ext, T_cam_ext) if R_cam_ext is not None else None)
        else:
            cam_extrinsics.append(None)

        cache_key = (subject, seq_name)
        if cache_key not in gt_mano_cache:
            gt_mano_cache[cache_key] = load_gt_mano_verts(subject, seq_name)
        gt_r, gt_l = gt_mano_cache[cache_key]
        if gt_r is not None and R_cam_ext is not None:
            gt_mano_data.append((gt_r, gt_l, R_cam_ext, T_cam_ext))
        else:
            gt_mano_data.append(None)

    # Per-source accumulation
    if effective_source == 'both' and n_hawor > 0 and n_haptic > 0:
        contact_hawor = accumulate_contact_labels(
            mesh_verts_m, mesh_faces, hand_verts_list[:n_hawor], pred_threshold,
            obj_poses=obj_poses[:n_hawor], cam_extrinsics=cam_extrinsics[:n_hawor],
            frame_offsets=frame_offsets[:n_hawor], gt_mano_data=gt_mano_data[:n_hawor]
        )
        contact_haptic = accumulate_contact_labels(
            mesh_verts_m, mesh_faces, hand_verts_list[n_hawor:], pred_threshold,
            obj_poses=obj_poses[n_hawor:], cam_extrinsics=cam_extrinsics[n_hawor:],
            frame_offsets=frame_offsets[n_hawor:], gt_mano_data=gt_mano_data[n_hawor:]
        )
        contact_freq = np.maximum(contact_hawor, contact_haptic)
    else:
        contact_freq = accumulate_contact_labels(
            mesh_verts_m, mesh_faces, hand_verts_list, pred_threshold,
            obj_poses=obj_poses, cam_extrinsics=cam_extrinsics,
            frame_offsets=frame_offsets, gt_mano_data=gt_mano_data
        )
        contact_hawor = contact_freq if effective_source == 'hawor' else None
        contact_haptic = contact_freq if effective_source == 'haptic' else None

    return mesh_verts_m, mesh_faces, contact_freq, contact_hawor, contact_haptic


def render_heatmap(verts, faces, contact_freq, save_path, title="",
                   n_views=4, figsize_per_view=6):
    """Render mesh with blue→red heatmap from multiple viewing angles."""
    # Normalize contact_freq for coloring
    vmax = contact_freq.max() if contact_freq.max() > 0 else 1.0

    # Create blue→red colormap
    cmap = plt.cm.coolwarm  # blue → white → red
    # Alternative: jet gives blue → green → yellow → red
    # cmap = plt.cm.jet

    center = verts.mean(0)
    span = max(verts.max(0) - verts.min(0)) * 0.7
    polygons = verts[faces]

    # Pre-compute face colors: max contact of 3 vertices per face
    face_vals = np.array([contact_freq[f].max() for f in faces])
    face_vals_norm = face_vals / vmax  # [0, 1]

    # View angles: rotate around the object
    elevations = [25, 25, 25, 25, -10, 60]
    azimuths = [0, 90, 180, 270, 45, 135]

    n_views = min(n_views, len(elevations))

    fig = plt.figure(figsize=(figsize_per_view * n_views, figsize_per_view + 1))

    for vi in range(n_views):
        ax = fig.add_subplot(1, n_views, vi + 1, projection='3d')

        # Map values to colors
        face_colors = cmap(face_vals_norm)
        # Make very low values more transparent
        face_colors[:, 3] = np.clip(0.4 + 0.6 * face_vals_norm, 0.3, 1.0)

        mesh_col = Poly3DCollection(polygons, facecolors=face_colors,
                                     edgecolors='none', linewidths=0)
        ax.add_collection3d(mesh_col)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.view_init(elev=elevations[vi], azim=azimuths[vi])
        ax.set_axis_off()

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label('Contact Frequency', fontsize=12)

    stats = (f"max={contact_freq.max():.3f}  mean={contact_freq.mean():.3f}  "
             f">0.1: {(contact_freq>0.1).mean():.1%}  >0.3: {(contact_freq>0.3).mean():.1%}")
    fig.suptitle(f"{title}\n{stats}", fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  💾 Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize merged contact heatmap")
    parser.add_argument('--object', default='ketchup', help='Object name')
    parser.add_argument('--views', type=int, default=4, help='Number of views')
    parser.add_argument('--out_dir', default=None, help='Output directory')
    args = parser.parse_args()

    out_dir = args.out_dir or config.CONTACT_VIS_DIR
    os.makedirs(out_dir, exist_ok=True)

    print(f"🔥 Computing merged contact for: {args.object}")
    result = compute_vertex_contact(args.object)
    if result[0] is None:
        print("❌ Failed to compute contact")
        return

    verts, faces, contact_freq, contact_hawor, contact_haptic = result

    print(f"  Merged: max={contact_freq.max():.4f}, "
          f">0.1: {(contact_freq>0.1).mean():.1%}, "
          f">0.3: {(contact_freq>0.3).mean():.1%}")

    # Render merged heatmap
    save_path = os.path.join(out_dir, f"{args.object}_merged_heatmap.png")
    render_heatmap(verts, faces, contact_freq, save_path,
                   title=f"{args.object} — Merged Contact (HaWoR + HaPTIC)",
                   n_views=args.views)

    # Also render individual sources if available
    if contact_hawor is not None:
        save_h = os.path.join(out_dir, f"{args.object}_hawor_heatmap.png")
        render_heatmap(verts, faces, contact_hawor, save_h,
                       title=f"{args.object} — HaWoR only (egocentric)",
                       n_views=args.views)
    if contact_haptic is not None:
        save_p = os.path.join(out_dir, f"{args.object}_haptic_heatmap.png")
        render_heatmap(verts, faces, contact_haptic, save_p,
                       title=f"{args.object} — HaPTIC only (third-person)",
                       n_views=args.views)

    # Render 3-panel comparison: HaWoR | HaPTIC | Merged
    if contact_hawor is not None and contact_haptic is not None:
        save_cmp = os.path.join(out_dir, f"{args.object}_comparison_heatmap.png")
        render_comparison(verts, faces, contact_hawor, contact_haptic, contact_freq,
                          save_cmp, title=args.object)


def render_comparison(verts, faces, hawor_freq, haptic_freq, merged_freq,
                      save_path, title=""):
    """3-panel comparison: HaWoR | HaPTIC | Merged, same color scale."""
    cmap = plt.cm.coolwarm
    vmax = merged_freq.max() if merged_freq.max() > 0 else 1.0

    center = verts.mean(0)
    span = max(verts.max(0) - verts.min(0)) * 0.7
    polygons = verts[faces]

    fig = plt.figure(figsize=(18, 7))

    for idx, (freq, label) in enumerate([
        (hawor_freq, f"HaWoR (egocentric)\nmax={hawor_freq.max():.3f}"),
        (haptic_freq, f"HaPTIC (third-person)\nmax={haptic_freq.max():.3f}"),
        (merged_freq, f"Merged (max)\nmax={merged_freq.max():.3f}"),
    ]):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        face_vals = np.array([freq[f].max() for f in faces])
        face_vals_norm = face_vals / vmax
        face_colors = cmap(face_vals_norm)
        face_colors[:, 3] = np.clip(0.4 + 0.6 * face_vals_norm, 0.3, 1.0)

        mesh_col = Poly3DCollection(polygons, facecolors=face_colors,
                                     edgecolors='none', linewidths=0)
        ax.add_collection3d(mesh_col)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.view_init(elev=25, azim=45)
        ax.set_axis_off()
        ax.set_title(label, fontsize=12, fontweight='bold')

    sm = ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label('Contact Frequency', fontsize=12)

    fig.suptitle(f"{title} — Source Comparison", fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  💾 Saved comparison: {save_path}")


if __name__ == '__main__':
    main()
