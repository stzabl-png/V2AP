#!/usr/bin/env python3
"""
compare_fp_gt.py — compare FoundationPose poses (regimes A/B/C) vs HOI4D objpose GT.

Frame map (verified): FP frame i  ↔  GT objpose frame i*5.

A  poses/Egocentric/hoi4d/      (MegaSAM K + MegaSAM depth + SAM3D mesh)
B  poses/Egocentric/hoi4d_gtB/  (GT K + GT depth + SAM3D mesh)
C  poses/Egocentric/hoi4d_gtC/  (GT K + GT depth + GT CAD)   — 3 rigid only

A/B metric : object-center error (FP pose × mesh bbox-center) vs GT center{x,y,z}.
C   metric : full 6D — ADD / ADD-S, rotation geodesic, translation — GT pose built
             from objpose(center,rotation), Euler convention auto-picked by 2dBox reproj.
"""
import os, sys, json, glob
import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation

PROJ = Path(__file__).resolve().parents[2]
GT_ANNO  = PROJ / "data/egocentric/hoi4d/HOI4D_annotations"
MESH_BASE = PROJ / "data_hub/ProcessedData/obj_meshes/hoi4d"
POSE = PROJ / "data_hub/ProcessedData/poses/Egocentric"
SEQS = json.load(open(PROJ / "output/eval/hoi4d_10seqs.json"))
STRIDE = 5
K_GT = np.array([[1376.8,0,967.7],[0,1376.2,526.8],[0,0,1.0]])  # @1920x1080


def load_fp(tag, seq_id):
    d = POSE / tag / seq_id / "ob_in_cam"
    fs = sorted(glob.glob(str(d / "*.txt")))
    return np.array([np.loadtxt(f) for f in fs]) if fs else None


def gt_record(seq_id, gidx):
    p = seq_id.split("_")
    f = GT_ANNO.joinpath(*p[:7], "objpose", f"{gidx}.json")
    d = json.load(open(f))["dataList"][0]
    c = d["center"]
    return d, np.array([c["x"], c["y"], c["z"]])


def scaled_mesh_bbox_center(seq_id):
    mp = MESH_BASE / seq_id / "mesh.ply"
    sj = MESH_BASE / seq_id / "scale.json"
    m = trimesh.load(str(mp), process=False)
    sf = float(json.load(open(sj)).get("scale_factor", 1.0)) if sj.exists() else 1.0
    v = m.vertices * sf
    return (v.min(0) + v.max(0)) / 2.0       # bbox center in (scaled) mesh frame


# ── A / B : object-center comparison ──────────────────────────────────────────
def eval_AB():
    print("\n" + "="*96)
    print("REGIME A vs B — object-center error vs GT center  (per-seq median over 60 frames, meters)")
    print("="*96)
    print(f"{'cat':>4} {'obj':<14} {'rigid':>5} | "
          f"{'A_depthErr':>10} {'A_dRatio':>8} {'A_3Derr':>8} | "
          f"{'B_depthErr':>10} {'B_dRatio':>8} {'B_3Derr':>8}")
    print("-"*96)
    rows = []
    for s in SEQS:
        seq = s["seq_id"].split("/")[-1]
        c_local = scaled_mesh_bbox_center(seq)
        c_h = np.append(c_local, 1.0)
        res = {}
        for tag in ("hoi4d", "hoi4d_gtB"):
            P = load_fp(tag, seq)
            if P is None:
                res[tag] = None; continue
            de, dr, e3 = [], [], []
            for i, T in enumerate(P):
                _, gc = gt_record(seq, i*STRIDE)
                fc = (T @ c_h)[:3]
                de.append(abs(fc[2]-gc[2])); dr.append(fc[2]/gc[2])
                e3.append(np.linalg.norm(fc-gc))
            res[tag] = (np.median(de), np.median(dr), np.median(e3))
        a, b = res["hoi4d"], res["hoi4d_gtB"]
        rows.append((s, a, b))
        fa = f"{a[0]:>10.3f} {a[1]:>8.2f} {a[2]:>8.3f}" if a else f"{'--':>10} {'--':>8} {'--':>8}"
        fb = f"{b[0]:>10.3f} {b[1]:>8.2f} {b[2]:>8.3f}" if b else f"{'--':>10} {'--':>8} {'--':>8}"
        print(f"{s['cat']:>4} {s['obj']:<14} {str(s['gt']['rigid']):>5} | {fa} | {fb}")
    # aggregate
    def med(rows, idx, key):
        v = [r[idx][key] for r in rows if r[idx]]
        return np.median(v) if v else float('nan')
    print("-"*96)
    print(f"{'ALL':>4} {'median':<14} {'':>5} | "
          f"{med(rows,1,0):>10.3f} {med(rows,1,1):>8.2f} {med(rows,1,2):>8.3f} | "
          f"{med(rows,2,0):>10.3f} {med(rows,2,1):>8.2f} {med(rows,2,2):>8.3f}")
    print("depthErr=|z_fp - z_gt|;  dRatio=z_fp/z_gt (1.0=perfect);  3Derr=||center_fp-center_gt||")
    return rows


# ── C : full 6D with GT CAD ───────────────────────────────────────────────────
def project(K, Xc):
    x = (K @ Xc.T).T
    return x[:, :2] / x[:, 2:3]


def build_gt_pose(d, R):
    """GT obj->cam for raw CAD vertices: p_cam = R @ (p_obj - c_local) + center.
    c_local handled by caller (we return R, t with t=center, and center the CAD)."""
    c = d["center"]
    return R, np.array([c["x"], c["y"], c["z"]])


def pick_euler_convention(seq_id, cad_v_centered):
    """Pick Euler order whose reprojected CAD bbox best matches the 2dBox (frame0)."""
    d, _ = gt_record(seq_id, 0)
    box = d["2dBox"]["camera1"]; box = np.array(box)
    bx0, by0 = box.min(0); bx1, by1 = box.max(0)
    rot = d["rotation"]; e = [rot["x"], rot["y"], rot["z"]]
    c = d["center"]; t = np.array([c["x"], c["y"], c["z"]])
    best, berr = None, 1e18
    for conv in ("XYZ","ZYX","xyz","zyx","YXZ","ZXY"):
        try: R = Rotation.from_euler(conv, e).as_matrix()
        except Exception: continue
        Xc = (R @ cad_v_centered.T).T + t
        if np.any(Xc[:,2] <= 0): continue
        uv = project(K_GT, Xc)
        px0,py0 = uv.min(0); px1,py1 = uv.max(0)
        err = abs(px0-bx0)+abs(py0-by0)+abs(px1-bx1)+abs(py1-by1)
        if err < berr: berr, best = err, conv
    return best, berr


def eval_C():
    print("\n" + "="*96)
    print("REGIME C — full 6D vs GT (GT CAD, 3 rigid).  per-seq median over 60 frames")
    print("="*96)
    print(f"{'cat':>4} {'obj':<10} {'eulConv':>7} {'boxErrpx':>8} | "
          f"{'ADD(m)':>8} {'ADD-S(m)':>9} {'rotErr°':>8} {'transErr':>9} {'depthErr':>9}")
    print("-"*96)
    for s in SEQS:
        if not s["gt"]["rigid"]:
            continue
        seq = s["seq_id"].split("/")[-1]
        P = load_fp("hoi4d_gtC", seq)
        if P is None:
            print(f"{s['cat']:>4} {s['obj']:<10}  (no regime-C poses)"); continue
        cad = trimesh.load(s["gt"]["cad_path"], process=False)
        V = cad.vertices
        c_local = (V.min(0)+V.max(0))/2.0
        Vc = V - c_local
        # subsample CAD pts for ADD speed
        if len(Vc) > 2000:
            idx = np.linspace(0, len(Vc)-1, 2000).astype(int); Vs = Vc[idx]
        else:
            Vs = Vc
        conv, boxerr = pick_euler_convention(seq, Vs)
        adds, addss, rots, trans, depth = [], [], [], [], []
        for i, T in enumerate(P):
            d, gc = gt_record(seq, i*STRIDE)
            rot = d["rotation"]; e = [rot["x"], rot["y"], rot["z"]]
            Rg = Rotation.from_euler(conv, e).as_matrix()
            # GT places centered CAD at center; FP places raw CAD via T (raw .obj frame)
            Pg = (Rg @ Vs.T).T + gc                       # GT cad pts in cam
            Pf = (T[:3,:3] @ (Vs + c_local).T).T + T[:3,3] # FP cad pts in cam (raw->T)
            adds.append(np.mean(np.linalg.norm(Pf - Pg, axis=1)))
            # ADD-S (closest point) — subsample for speed
            j = np.linspace(0,len(Vs)-1,300).astype(int)
            from scipy.spatial import cKDTree
            addss.append(np.mean(cKDTree(Pg).query(Pf[j])[0]))
            Rf = T[:3,:3]
            ang = np.degrees(np.arccos(np.clip((np.trace(Rf.T@Rg)-1)/2,-1,1)))
            rots.append(ang)
            fc = (T @ np.append(c_local,1))[:3]
            trans.append(np.linalg.norm(fc-gc)); depth.append(abs(fc[2]-gc[2]))
        print(f"{s['cat']:>4} {s['obj']:<10} {conv:>7} {boxerr:>8.0f} | "
              f"{np.median(adds):>8.3f} {np.median(addss):>9.3f} {np.median(rots):>8.1f} "
              f"{np.median(trans):>9.3f} {np.median(depth):>9.3f}")
    print("-"*96)
    print("rotErr is full geodesic (no symmetry handling); ADD-S accounts for shape symmetry.")
    print("boxErrpx = px mismatch of reprojected GT CAD bbox vs annotated 2dBox (GT-pose self-check).")


if __name__ == "__main__":
    eval_AB()
    eval_C()
