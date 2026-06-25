"""
Evaluate HaPTIC under GT@15mm + Pred@30mm (same as HaWoR evaluation).
Uses cached HaPTIC results in haptic_arctic_cache/.

Usage:
  python -m data.eval_haptic_asymmetric
"""

import os, sys, json
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

ARCTIC_ROOT = config.ARCTIC_ROOT
HAPTIC_CACHE = config.HAPTIC_CACHE
ONSET_JSON = config.ONSET_JSON

OBJECTS = ['box','capsulemachine','espressomachine','ketchup','laptop',
           'microwave','mixer','notebook','phone','scissors','waffleiron']

GT_TH = config.GT_CONTACT_TH
PRED_TH = config.PRED_CONTACT_TH


def get_obj(seq):
    for obj in sorted(OBJECTS, key=len, reverse=True):
        if seq.startswith(obj):
            return obj
    return None


def main():
    # Load onset
    onset_data = {}
    if os.path.exists(ONSET_JSON):
        with open(ONSET_JSON) as f:
            onset_data = json.load(f)
    
    # Load misc for camera extrinsics
    with open(f'{ARCTIC_ROOT}/meta/misc.json') as f:
        misc = json.load(f)
    
    # Load mesh templates
    mesh_cache = {}
    for obj in OBJECTS:
        obj_dir = f'{ARCTIC_ROOT}/meta/object_vtemplates/{obj}'
        for fn in sorted(os.listdir(obj_dir)):
            if fn.endswith(('.obj', '.ply', '.stl')):
                m = trimesh.load(os.path.join(obj_dir, fn), force='mesh')
                mesh_cache[obj] = np.array(m.vertices) / 1000.0  # mm→m
                break
    print(f"📦 Loaded {len(mesh_cache)} meshes")
    
    # GT MANO model
    import smplx
    mano_r = smplx.create(config.MANO_MODELS,
                          model_type='mano', is_rhand=True, use_pca=False, flat_hand_mean=False)
    
    cached_files = sorted(os.listdir(HAPTIC_CACHE))
    print(f"📦 HaPTIC cache: {len(cached_files)} files")
    print(f"📏 GT@{int(GT_TH*1000)}mm vs Pred@{int(PRED_TH*1000)}mm")
    
    results_by_obj = defaultdict(list)
    skip_count = defaultdict(int)
    
    for cf in cached_files:
        if not cf.endswith('.npz'):
            continue
        # Parse: s01__box_grab_01_cam1.npz
        base = cf.replace('.npz', '')
        parts = base.split('__')
        if len(parts) != 2:
            continue
        subj = parts[0]
        seq_cam = parts[1]
        # Extract cam_id from end
        cam_str = seq_cam.split('_')[-1]  # "cam1"
        cam_id = int(cam_str.replace('cam', '')) - 1  # 0-indexed
        seq = '_'.join(seq_cam.split('_')[:-1])
        
        obj = get_obj(seq)
        if obj not in mesh_cache:
            skip_count['no_mesh'] += 1
            continue
        
        # Load cached HaPTIC verts
        try:
            hdata = np.load(os.path.join(HAPTIC_CACHE, cf), allow_pickle=True)
            verts_dict = hdata['verts_dict'].item()
        except Exception:
            skip_count['bad_file'] += 1
            continue
        
        # Load GT
        mano_path = f'{ARCTIC_ROOT}/raw_seqs/{subj}/{seq}.mano.npy'
        obj_path = f'{ARCTIC_ROOT}/raw_seqs/{subj}/{seq}.object.npy'
        if not os.path.exists(mano_path) or not os.path.exists(obj_path):
            skip_count['missing_gt'] += 1
            continue
        
        mg = np.load(mano_path, allow_pickle=True).item()
        od = np.load(obj_path, allow_pickle=True)
        
        # Camera extrinsics
        w2c = np.array(misc[subj]['world2cam'][cam_id])
        R_cam = w2c[:3, :3]
        T_cam = w2c[:3, 3]
        
        obj_verts = mesh_cache[obj]
        
        # Onset
        key = f"{subj}__{seq}"
        onset = onset_data.get(key)
        start = onset['clip_start'] if onset else 0
        end = start + len(verts_dict) - 1
        
        N_gt = od.shape[0]
        sl = slice(start, min(end + 1, N_gt))
        N_sl = min(end + 1, N_gt) - start
        if N_sl < 2:
            skip_count['too_short'] += 1
            continue
        
        # GT MANO forward pass (right hand only for HaPTIC)
        try:
            out_r = mano_r(
                global_orient=torch.from_numpy(mg['right']['rot'][sl]).float(),
                hand_pose=torch.from_numpy(mg['right']['pose'][sl]).float(),
                betas=torch.from_numpy(np.tile(mg['right']['shape'], (N_sl, 1))).float(),
                transl=torch.from_numpy(mg['right']['trans'][sl]).float())
            gt_r = out_r.vertices.detach().numpy()
        except Exception as e:
            skip_count['mano_error'] += 1
            continue
        
        n_verts = len(obj_verts)
        gt_min_dists = np.full(n_verts, 1e6)
        pred_min_dists = np.full(n_verts, 1e6)
        
        for i in range(0, min(len(verts_dict), N_sl), 5):
            if i not in verts_dict:
                continue
            fi = start + i
            if fi >= N_gt:
                break
            
            # Object in cam space
            oR = Rotation.from_rotvec(od[fi, 1:4]).as_matrix()
            ot = od[fi, 4:7] / 1000.0
            ow = (oR @ obj_verts.T).T + ot
            oc = (R_cam @ ow.T).T + T_cam
            
            # GT hand in cam space
            gc = (R_cam @ gt_r[i].T).T + T_cam
            gd, _ = cKDTree(gc).query(oc)
            gt_min_dists = np.minimum(gt_min_dists, gd)
            
            # HaPTIC hand + centroid align
            hr = verts_dict[i]
            off = gc.mean(0) - hr.mean(0)
            hr_a = hr + off
            hd, _ = cKDTree(hr_a).query(oc)
            pred_min_dists = np.minimum(pred_min_dists, hd)
        
        gt_contact = gt_min_dists < GT_TH
        pred_contact = pred_min_dists < PRED_TH
        
        if gt_contact.sum() == 0:
            skip_count['no_contact'] += 1
            continue
        
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
            'subj': subj, 'seq': seq, 'cam': cam_id,
            'iou': iou, 'precision': prec, 'recall': rec, 'f1': f1, 'fp_rate': fp_rate,
        })
        print(f"  {subj}/{seq} cam{cam_id+1}: F1={f1:.3f} P={prec:.3f} R={rec:.3f} FP={fp_rate:.1f}%")
    
    # Summary
    all_entries = [e for objs in results_by_obj.values() for e in objs]
    print(f"\n{'='*70}")
    print(f"📊 HaPTIC: GT@{int(GT_TH*1000)}mm vs Pred@{int(PRED_TH*1000)}mm")
    print(f"{'='*70}")
    print(f"  Evaluated: {len(all_entries)} seqs")
    print(f"  Skipped: {sum(skip_count.values())} {dict(skip_count)}")
    
    if all_entries:
        all_iou = [e['iou'] for e in all_entries]
        all_prec = [e['precision'] for e in all_entries]
        all_rec = [e['recall'] for e in all_entries]
        all_f1 = [e['f1'] for e in all_entries]
        all_fp = [e['fp_rate'] for e in all_entries]
        
        print(f"\n  Mean IoU:       {np.mean(all_iou):.3f}")
        print(f"  Mean Precision: {np.mean(all_prec):.3f}")
        print(f"  Mean Recall:    {np.mean(all_rec):.3f}")
        print(f"  Mean F1:        {np.mean(all_f1):.3f}  ⭐")
        print(f"  Mean FP rate:   {np.mean(all_fp):.1f}%")
        
        print(f"\n  Per-object:")
        for obj in sorted(results_by_obj.keys()):
            entries = results_by_obj[obj]
            f1s = [e['f1'] for e in entries]
            precs = [e['precision'] for e in entries]
            recs = [e['recall'] for e in entries]
            fps = [e['fp_rate'] for e in entries]
            print(f"    {obj}: F1={np.mean(f1s):.3f}  P={np.mean(precs):.3f}  R={np.mean(recs):.3f}  FP={np.mean(fps):.1f}% ({len(entries)} seqs)")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
