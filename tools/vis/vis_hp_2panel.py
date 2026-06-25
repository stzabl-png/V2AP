#!/usr/bin/env python3
"""
2-Panel HP Filter Visualization
Left:  Human Prior heatmap on dense mesh point cloud
Right: HP-matched Robot Posterior contact points on dense mesh point cloud

Usage:
    python3 tools/vis_hp_2panel.py --obj C28001
    python3 tools/vis_hp_2panel.py --batch
"""
import os, sys, glob, argparse, h5py
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

PROJ         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HP_DIRS      = {
    'oakink': os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp', 'oakink'),
    'dexycb': os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp', 'dexycb'),
}
FILTERED_DIR = os.path.join(PROJ, 'output', 'RobotPosteriorFiltered')
MESH_BASE    = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
OUT_DIR      = os.path.join(PROJ, 'output', 'vis_hp_2panel')
os.makedirs(OUT_DIR, exist_ok=True)

N_VIS = 20000   # dense visualization points


def find_mesh(obj_id):
    ds = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
    p = os.path.join(MESH_BASE, ds, obj_id, 'mesh.ply')
    return p if os.path.exists(p) else None


def get_scale_factor(obj_id):
    """scale_factor from scale.json: mesh file coords → real meters."""
    import json
    ds = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
    p = os.path.join(MESH_BASE, ds, obj_id, 'scale.json')
    if os.path.exists(p):
        return float(json.load(open(p)).get('scale_factor', 1.0))
    return 1.0


def load_hp(obj_id):
    """Load human prior from training_fp (the correct source)."""
    ds = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
    p  = os.path.join(HP_DIRS[ds], f'{obj_id}.hdf5')
    if not os.path.exists(p):
        return None, None
    with h5py.File(p, 'r') as f:
        pc = f['point_cloud'][()].astype(np.float32)
        hp = f['human_prior'][()].astype(np.float32)
    return pc, hp


def load_hp_contacts(obj_id):
    """Load HP-matched contacts in real meters, original mesh frame.
    
    grasp_point is in object-local real meters (canonical rotation applied).
    We undo the canonical rotation to align with the original mesh frame.
    """
    import json as _json
    from scipy.spatial.transform import Rotation as _Rot
    can_rot_inv = None
    can_rot_json = os.path.join(PROJ, 'sim', 'canonical_rotation.json')
    if os.path.exists(can_rot_json):
        raw = _json.load(open(can_rot_json))
        if obj_id in raw:
            v = raw[obj_id]
            euler = v if isinstance(v, list) else v.get('euler', [0,0,0])
            if any(abs(e) > 0.1 for e in euler):
                R = _Rot.from_euler('xyz', euler, degrees=True).as_matrix()
                can_rot_inv = R.T

    p = os.path.join(FILTERED_DIR, f'{obj_id}_robot_gt.hdf5')
    if not os.path.exists(p):
        return [], 0, 0
    contacts = []
    with h5py.File(p, 'r') as f:
        n_match    = int(f.attrs.get('n_hp_match', 0))
        n_mismatch = int(f.attrs.get('n_hp_mismatch', 0))
        sg = f.get('successful_grasps', {})
        for key in sg.keys():
            g = sg[key]
            if not g.attrs.get('hp_match', False):
                continue
            gp  = g['grasp_point'][:].astype(float)
            fd  = g['finger_dir'][:].astype(float)
            w   = float(g.attrs.get('gripper_width', 0.05))
            fd_n = fd / (np.linalg.norm(fd) + 1e-8)
            c_L = gp + fd_n * w / 2
            c_R = gp - fd_n * w / 2
            # Undo canonical rotation → original mesh frame
            if can_rot_inv is not None:
                c_L = can_rot_inv @ c_L
                c_R = can_rot_inv @ c_R
            contacts.append((c_L, c_R))
    return contacts, n_match, n_mismatch


def render_pc(ax, points, values, title, cmap='jet', vmin=0, vmax=1,
              binary=False, elev=25, azim=135):
    """Render point cloud exactly like vis_3panel.py."""
    if binary:
        colors = np.zeros((len(values), 4))
        colors[values < 0.5]  = [0.75, 0.75, 0.85, 1.0]   # gray-blue
        colors[values >= 0.5] = [0.90, 0.15, 0.15, 1.0]   # red
    else:
        cmap_fn = plt.get_cmap(cmap)
        colors  = cmap_fn(np.clip(values, vmin, vmax))

    order  = np.argsort(values)
    pts_s  = points[order]
    cols_s = colors[order]

    ax.scatter(pts_s[:,0], pts_s[:,1], pts_s[:,2],
               c=cols_s, s=1.5, alpha=0.9, edgecolors='none')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.view_init(elev=elev, azim=azim)

    extents = points.max(0) - points.min(0)
    r = extents.max() * 0.6
    ctr = (points.max(0) + points.min(0)) / 2
    ax.set_xlim(ctr[0]-r, ctr[0]+r)
    ax.set_ylim(ctr[1]-r, ctr[1]+r)
    ax.set_zlim(ctr[2]-r, ctr[2]+r)
    ax.set_axis_off()


def render_contacts(ax, points, contacts, title, elev=25, azim=135):
    """Render gray object + blue contact sphere pairs on same axes."""
    ax.scatter(points[:,0], points[:,1], points[:,2],
               c='#c0c0c0', s=1.5, alpha=0.6, edgecolors='none', depthshade=True)

    if contacts:
        for c_L, c_R in contacts:
            ax.scatter(*c_L, c='royalblue', s=80, alpha=1.0,
                       edgecolors='white', linewidths=1.0,
                       depthshade=False, zorder=10)
            ax.scatter(*c_R, c='royalblue', s=80, alpha=1.0,
                       edgecolors='white', linewidths=1.0,
                       depthshade=False, zorder=10)
            ax.plot([c_L[0],c_R[0]], [c_L[1],c_R[1]], [c_L[2],c_R[2]],
                    c='dodgerblue', linewidth=1.5, alpha=0.85)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.view_init(elev=elev, azim=azim)

    extents = points.max(0) - points.min(0)
    r = extents.max() * 0.6
    ctr = (points.max(0) + points.min(0)) / 2
    ax.set_xlim(ctr[0]-r, ctr[0]+r)
    ax.set_ylim(ctr[1]-r, ctr[1]+r)
    ax.set_zlim(ctr[2]-r, ctr[2]+r)
    ax.set_axis_off()


def vis_single(obj_id, out_dir=None, show=False):
    # Load HP point cloud + labels
    hp_pc_raw, hp = load_hp(obj_id)
    if hp_pc_raw is None:
        print(f'  SKIP {obj_id}: no HP data')
        return False

    sf = get_scale_factor(obj_id)       # mesh file → real meters
    hp_pc = hp_pc_raw * sf              # HP in real meters

    contacts, n_match, n_mismatch = load_hp_contacts(obj_id)

    # Dense mesh for visualization (scale to real meters)
    mesh_path = find_mesh(obj_id)
    if mesh_path:
        mesh   = trimesh.load(mesh_path, force='mesh')
        vis_pc_raw, _ = trimesh.sample.sample_surface(mesh, N_VIS)
        vis_pc = (vis_pc_raw * sf).astype(np.float32)  # real meters

        # KNN: interpolate HP labels from hp_pc (real meters) to vis_pc (real meters)
        tree = cKDTree(hp_pc)
        _, idx = tree.query(vis_pc, k=3)
        dists   = np.linalg.norm(vis_pc[:,None,:] - hp_pc[idx], axis=2)
        weights = 1.0 / (dists + 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        hp_dense = (hp[idx] * weights).sum(axis=1)
    else:
        vis_pc   = hp_pc
        hp_dense = hp

    # Plot
    fig = plt.figure(figsize=(14, 6), facecolor='white')

    ax1 = fig.add_subplot(121, projection='3d')
    hp_pct = (hp > 0.3).mean() * 100
    render_pc(ax1, vis_pc, hp_dense,
              f'Human Prior  ({hp_pct:.0f}% region)',
              binary=True)

    ax2 = fig.add_subplot(122, projection='3d')
    # Right panel: mesh GT space background + contacts in GT space (same coord system)
    if mesh_path:
        render_contacts(ax2, vis_pc, contacts,
                        f'HP-matched Contacts  ({n_match} grasps)')
    else:
        render_contacts(ax2, hp_pc, contacts,
                        f'HP-matched Contacts  ({n_match} grasps)')

    fig.suptitle(f'{obj_id}   |   HP-match:{n_match}  HP-miss:{n_mismatch}',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f'{obj_id}.png')
        fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'  OK {obj_id}: match={n_match} miss={n_mismatch} -> {p}')
    if show:
        plt.show()
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj',   type=str)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--out',   type=str, default=OUT_DIR)
    parser.add_argument('--show',  action='store_true')
    args = parser.parse_args()

    if args.obj:
        vis_single(args.obj, out_dir=args.out, show=args.show)
    elif args.batch:
        files = sorted(glob.glob(os.path.join(FILTERED_DIR, '*_robot_gt.hdf5')))
        ok = 0
        for f in files:
            oid = os.path.basename(f).replace('_robot_gt.hdf5', '')
            if vis_single(oid, out_dir=args.out):
                ok += 1
        print(f'\nDone: {ok}/{len(files)} objects -> {args.out}')
    else:
        print('Use --obj OBJ_ID or --batch')


if __name__ == '__main__':
    main()
