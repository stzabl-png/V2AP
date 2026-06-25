#!/usr/bin/env python3
"""
tools/eval_fingertip_aff.py
检查 Diffusion 预测的抓取点是否落在 Affordance 高响应区域内。

对比三者指尖 affordance 均值：
  - contact_pt_aff  : 从 heatmap top-5% 采样的接触点 affordance（应最高）
  - pred_tip_aff    : 预测 rotation 推导出的指尖位置 affordance
  - gt_tip_aff      : GT Robot Posterior 指尖 affordance（参考基线）

用法:
  python3 tools/eval_fingertip_aff.py \
      --ckpt_dir output/checkpoints_diffusion_v2_p0fix \
      --split    train/val/all
"""

import os, sys, json, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
from model.pointnet2_v6 import PointNet2AffordanceV6
from model.grasp_diffusion_v2 import GraspDiffusionV2, get_finger_dir, GRIPPER_HW

BASE = 'data_hub/ProcessedData/RobotPosterior'


def knn_aff(tips, pc_xyz, labels, k=5):
    """tips (M,3), pc_xyz (N,3), labels (N,) → mean aff per tip (M,)"""
    d2 = ((pc_xyz[None] - tips[:, None]) ** 2).sum(-1)   # (M, N)
    idx = np.argsort(d2, axis=1)[:, :k]
    return labels[idx].mean(axis=1)


def sample_contact_pts(pc, lab, k=3):
    """从 top-5% affordance 点 FPS 采 k 个接触点"""
    thresh = np.percentile(lab, 95)
    mask = lab >= thresh
    cand = pc[mask]
    cand_aff = lab[mask]
    if len(cand) < k:
        idx = np.argsort(lab)[-k:]
        return pc[idx]
    sel = [int(np.argmax(cand_aff))]
    for _ in range(k - 1):
        d = np.min(np.linalg.norm(
            cand[sel][:, None] - cand[None], axis=-1), axis=0)
        sel.append(int(np.argmax(d)))
    return cand[sel]   # (k, 3)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 加载 V6 ──────────────────────────────────────────────
    pn2 = PointNet2AffordanceV6().to(device)
    v6  = torch.load(f'{BASE}/best_v6_model.pth', map_location=device, weights_only=False)
    pn2.load_state_dict(v6['model_state_dict'])
    pn2.eval()

    # ── 加载 Diffusion ────────────────────────────────────────
    diff = GraspDiffusionV2(T=1000, hidden=512).to(device)
    d_ckpt = torch.load(f'{args.ckpt_dir}/best_model.pth',
                        map_location=device, weights_only=False)
    diff.load_state_dict(d_ckpt['model_state_dict'])
    diff.eval()

    rs = torch.load(f'{args.ckpt_dir}/rot_stats.pt', map_location=device, weights_only=False)
    ps = torch.load(f'{args.ckpt_dir}/pos_stats.pt', map_location=device, weights_only=False)
    rot_mean, rot_std = rs['rot_mean'], rs['rot_std']
    pos_mean, pos_std = ps['pos_mean'], ps['pos_std']

    # ── 读点云数据 ────────────────────────────────────────────
    split_json = json.load(open(f'{BASE}/min20/objects_train_val_split.json'))
    if args.split == 'train':
        obj_ids = split_json['train']
    elif args.split == 'val':
        obj_ids = split_json['val']
    else:
        obj_ids = split_json['train'] + split_json['val']

    with h5py.File(f'{BASE}/affordance_all.h5') as hf:
        raw_ids  = [x.decode() if isinstance(x, bytes) else x
                    for x in hf['data/obj_ids'][:]]
        aff_idx  = {o: i for i, o in enumerate(raw_ids)}
        all_pts  = hf['data/points'][:]
        all_nrm  = hf['data/normals'][:]
        all_lab  = hf['data/labels'][:]

    # ── 逐对象评估 ────────────────────────────────────────────
    print(f"\n{'对象':<12} {'CP-Aff':>8} {'Pred-Aff':>10} {'GT-Aff':>9} {'Delta':>8}")
    print('-' * 54)

    results = []
    for oid in obj_ids:
        if oid not in aff_idx:
            continue

        idx = aff_idx[oid]
        pc  = all_pts[idx].astype(np.float32)
        nrm = all_nrm[idx].astype(np.float32)
        lab = all_lab[idx].astype(np.float32)

        pc_centroid = pc.mean(0)
        pc_radius   = float(np.linalg.norm(pc - pc_centroid, axis=1).max()) + 1e-6

        # V6 global_feat
        with torch.no_grad():
            xt = torch.from_numpy(pc).unsqueeze(0).to(device)
            nt = torch.from_numpy(nrm).unsqueeze(0).to(device)
            ft = torch.cat([xt, nt], dim=-1)
            gf = pn2.extract_global_feat(xt, ft)   # (1,512)

        # 接触点采样
        cps = sample_contact_pts(pc, lab, k=args.n_contacts)
        cp_aff = knn_aff(cps, pc, lab, k=5).mean()

        # Diffusion 推理 → 预测指尖 aff
        pred_tip_affs = []
        with torch.no_grad():
            for cp in cps:
                pos_rel  = (cp - pc_centroid) / pc_radius
                pos_t    = torch.from_numpy(pos_rel).unsqueeze(0).to(device)
                pos_norm = (pos_t - pos_mean) / pos_std
                r6d_norm = diff.sample(gf, pos_norm,
                                       n_samples=args.n_samples,
                                       ddim_steps=50)
                r6d    = r6d_norm * rot_std + rot_mean
                finger = get_finger_dir(r6d).cpu().numpy()           # (n,3)
                finger = finger / (np.linalg.norm(finger, axis=1,
                                                  keepdims=True) + 1e-8)
                tips = np.vstack([cp - GRIPPER_HW * finger,
                                  cp + GRIPPER_HW * finger])         # (2n,3)
                pred_tip_affs.append(knn_aff(tips, pc, lab, k=5).mean())
        pred_aff = float(np.mean(pred_tip_affs))

        # GT fingertips
        mp = f'{BASE}/merged/{oid}_robot_gt_merged.hdf5'
        gt_tip_affs = []
        if os.path.exists(mp):
            with h5py.File(mp) as hf2:
                sg = hf2.get('successful_grasps', {})
                for gk in list(sg.keys())[:30]:
                    g  = sg[gk]
                    ep = g.get('executed_panda_hand_at_close')
                    if ep is None:
                        continue
                    pos_e = ep['position'][:].astype(np.float32)
                    fd_e  = ep.get('finger_dir')
                    fd_e  = (fd_e[:].astype(np.float32) if fd_e
                             else ep['rotation'][:][:, 0])
                    pre = g.get('mesh_prerotation')
                    if pre and 'matrix' in pre:
                        R = pre['matrix'][:].astype(np.float32)
                        pos_e = R.T @ pos_e
                        fd_e  = R.T @ fd_e
                    if np.abs(pos_e).max() > 0.5:
                        continue
                    fd_e /= np.linalg.norm(fd_e) + 1e-8
                    tips = np.vstack([
                        (pos_e - GRIPPER_HW * fd_e)[None],
                        (pos_e + GRIPPER_HW * fd_e)[None],
                    ])
                    gt_tip_affs.append(knn_aff(tips, pc, lab, k=5).mean())
        gt_aff = float(np.mean(gt_tip_affs)) if gt_tip_affs else float('nan')

        delta = pred_aff - gt_aff
        flag  = '✅' if pred_aff >= 0.3 else ('⚠' if pred_aff >= 0.2 else '❌')
        print(f'{oid:<12} {cp_aff:>8.4f} {pred_aff:>10.4f} {gt_aff:>9.4f} '
              f'{delta:>+8.4f} {flag}')
        results.append({'obj': oid, 'cp_aff': cp_aff,
                        'pred_aff': pred_aff, 'gt_aff': gt_aff})

    # ── 汇总 ──────────────────────────────────────────────────
    print('-' * 54)
    valid = [r for r in results if not np.isnan(r['gt_aff'])]
    if not valid:
        return
    cp_m   = np.mean([r['cp_aff']   for r in valid])
    pred_m = np.mean([r['pred_aff'] for r in valid])
    gt_m   = np.mean([r['gt_aff']   for r in valid])
    print(f'{"平均":<12} {cp_m:>8.4f} {pred_m:>10.4f} {gt_m:>9.4f} '
          f'{pred_m-gt_m:>+8.4f}')

    print(f'\n{"="*54}')
    print(f'  接触点 affordance 均值:   {cp_m:.4f}  (> 0.4 = 好)')
    print(f'  预测指尖 affordance:       {pred_m:.4f}')
    print(f'  GT 指尖 affordance:        {gt_m:.4f}')
    print(f'  差值 (pred - GT):          {pred_m-gt_m:+.4f}')
    pct_good = np.mean([r['pred_aff'] >= 0.3 for r in valid]) * 100
    print(f'  预测 ≥ 0.3 aff 的对象:    {pct_good:.0f}%')
    print(f'{"="*54}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir',   default='output/checkpoints_diffusion_v2_p0fix')
    p.add_argument('--split',      default='val', choices=['train', 'val', 'all'])
    p.add_argument('--n_contacts', type=int, default=3)
    p.add_argument('--n_samples',  type=int, default=5)
    main(p.parse_args())
