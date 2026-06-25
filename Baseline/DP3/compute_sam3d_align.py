#!/usr/bin/env python3
"""
Baseline1 v2 — compute the constant per-object rotation that bridges the SAM3D
mesh's canonical frame and the dataset's canonical mesh frame (the one each
dataset's GT object pose is defined against).

WHY
    Baseline1 (route A) places the SAM3D `{obj_id}/mesh.ply` into camera frame
    using the dataset's GT object pose. But that pose is defined w.r.t. the
    dataset's *own* CAD mesh, not the SAM3D reconstruction — so the rendered
    object is rotated by a constant (per-object) amount relative to where it
    physically is. `R_align` fixes that:

        object PC in camera = pose_y · R_align · sam3d_mesh_points
                              (or  obj_transf · R_align · sam3d_mesh_points for OakInk)

HOW  (ICP between two complete meshes)
    Both meshes describe the same physical object in metric metres, but with
    different canonical orientations. We sample N points from each surface,
    centre at each centroid, then run ICP from 24 cube-symmetry initial rotations
    + several random inits. The best-cost result gives R_align.

SUPPORTED DATASETS
    --dataset dexycb    (default)
        SAM3D source : data_hub/ProcessedData/obj_meshes/ycb/ycb_dex_NN/mesh.ply
        CAD target   : data_hub/RawData/.../dexycb/models/{ycb_name}/textured.obj
        Output       : Baseline1/assets/sam3d_align/ycb_dex_NN.json  (flat, 20 files)

    --dataset oakink
        SAM3D source : data_hub/ProcessedData/obj_meshes/oakink/{obj_id}/mesh.ply
        CAD target   : data_hub/RawData/.../oakink_v1/image/obj/{obj_id}.(obj|ply)
        Output       : Baseline1/assets/sam3d_align/oakink/{obj_id}.json  (~100 files)
"""
import os, sys, json, argparse, itertools, time
import numpy as np
import trimesh
from glob import glob
from natsort import natsorted
from scipy.spatial.transform import Rotation

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import config

OUT_DIR_BASE = os.path.join(PROJ, "Baseline1", "assets", "sam3d_align")


def cube_rotations():
    """24 rotational symmetries of the cube — axis-aligned coverage of SO(3)
    (any rotation in SO(3) is within ~45° of one of these). All have det = +1."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if np.linalg.det(M) > 0.5:
                out.append(M)
    return out  # 24


# ── per-dataset paths ────────────────────────────────────────────────────────
def _sam3d_path(dataset, obj_id):
    if dataset == "dexycb":
        return os.path.join(config.DATA_HUB, "ProcessedData", "obj_meshes", "ycb",
                            obj_id, "mesh.ply")
    elif dataset == "oakink":
        return os.path.join(config.DATA_HUB, "ProcessedData", "obj_meshes", "oakink",
                            obj_id, "mesh.ply")
    raise ValueError(dataset)


def _cad_path(dataset, obj_id, sam3d_dir):
    """Return the path to the dataset's canonical mesh for `obj_id`."""
    if dataset == "dexycb":
        # ycb_dex_NN's scale.json has the YCB name → look up textured.obj
        sj = os.path.join(sam3d_dir, "scale.json")
        if not os.path.exists(sj):
            return None, None
        ycb_name = json.load(open(sj)).get("ycb_name")
        if not ycb_name:
            return None, None
        path = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData",
                            "dexycb", "models", ycb_name, "textured.obj")
        return (path if os.path.exists(path) else None), ycb_name
    elif dataset == "oakink":
        # OakInk object models are flat: image/obj/{obj_id}.{obj,ply}
        base = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData",
                            "oakink_v1", "image", "obj")
        for ext in ("obj", "ply"):
            p = os.path.join(base, f"{obj_id}.{ext}")
            if os.path.exists(p):
                return p, obj_id
        return None, obj_id
    raise ValueError(dataset)


def _list_objects(dataset):
    """Return list of obj_id strings to align."""
    if dataset == "dexycb":
        root = os.path.join(config.DATA_HUB, "ProcessedData", "obj_meshes", "ycb")
        return natsorted(os.path.basename(d) for d in glob(os.path.join(root, "ycb_dex_*")))
    elif dataset == "oakink":
        root = os.path.join(config.DATA_HUB, "ProcessedData", "obj_meshes", "oakink")
        return natsorted(os.path.basename(d) for d in glob(os.path.join(root, "*"))
                         if os.path.isdir(d))
    raise ValueError(dataset)


def _out_path(dataset, obj_id):
    if dataset == "dexycb":
        return os.path.join(OUT_DIR_BASE, f"{obj_id}.json")
    elif dataset == "oakink":
        return os.path.join(OUT_DIR_BASE, "oakink", f"{obj_id}.json")
    raise ValueError(dataset)


# ── loaders ──────────────────────────────────────────────────────────────────
def load_sam3d(dataset, obj_id):
    """Returns (mesh, scale_factor) — mesh in metric metres in SAM3D canonical."""
    mesh_path = _sam3d_path(dataset, obj_id)
    if not os.path.exists(mesh_path):
        return None, None
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    sj = os.path.join(os.path.dirname(mesh_path), "scale.json")
    sf = 1.0
    if os.path.exists(sj):
        sf = float(json.load(open(sj)).get("scale_factor", 1.0))
    if sf != 1.0:
        mesh.vertices = mesh.vertices * sf
    return mesh, sf


def load_cad(dataset, obj_id, sam3d_dir):
    path, cad_name = _cad_path(dataset, obj_id, sam3d_dir)
    if path is None:
        return None, cad_name
    return trimesh.load(path, force="mesh", process=False), cad_name


# ── core alignment ───────────────────────────────────────────────────────────
def align_one(dataset, obj_id, n_samples, n_random_inits, max_iter):
    sam3d, sf = load_sam3d(dataset, obj_id)
    if sam3d is None:
        return None, "SAM3D mesh missing"
    sam3d_dir = os.path.dirname(_sam3d_path(dataset, obj_id))
    cad, cad_name = load_cad(dataset, obj_id, sam3d_dir)
    if cad is None:
        return None, f"CAD model missing ({cad_name})"

    src_full_raw, _ = trimesh.sample.sample_surface(sam3d, n_samples)
    dst_full,     _ = trimesh.sample.sample_surface(cad,   n_samples)

    # DexYCB SAM3D meshes are already scaled to real metres via scale.json (applied in
    # load_sam3d) → prescale = 1. OakInk SAM3D meshes lack scale.json (normalized recon
    # units, typically a few × the CAD size) → explicitly prescale by bbox-diagonal
    # ratio so ICP only has to find rotation. Letting ICP estimate scale itself is a
    # known failure mode: it collapses to scale ~0 (all points cluster near origin →
    # low per-point distance, near-zero "cost", but no real shape alignment).
    sam_diag = float(np.linalg.norm(sam3d.bounds[1] - sam3d.bounds[0]))
    cad_diag = float(np.linalg.norm(cad.bounds[1]   - cad.bounds[0]))
    prescale = (cad_diag / sam_diag) if (dataset == "oakink" and sam_diag > 0) else 1.0

    src_full = src_full_raw * prescale
    src_centroid_raw = src_full_raw.mean(0)
    src_centroid     = src_full.mean(0)        # = prescale * src_centroid_raw
    dst_centroid     = dst_full.mean(0)
    src = (src_full - src_centroid).astype(np.float64)
    dst = (dst_full - dst_centroid).astype(np.float64)

    inits = cube_rotations()
    rng = np.random.default_rng(0)
    inits += [Rotation.from_rotvec(rng.standard_normal(3) * np.pi).as_matrix()
              for _ in range(n_random_inits)]

    results = []
    for init_R in inits:
        T_init = np.eye(4); T_init[:3, :3] = init_R
        try:
            T_icp, _, cost = trimesh.registration.icp(
                src, dst, initial=T_init,
                threshold=1e-7, max_iterations=max_iter,
                reflection=False, scale=False)   # rigid only — prescale handles size
        except Exception:
            continue
        results.append((float(cost), T_icp))

    if not results:
        return None, "ICP failed for all inits"

    results.sort(key=lambda x: x[0])
    best_cost, best_T = results[0]
    near = sum(1 for c, _ in results if c <= best_cost * 1.5)

    # Compose R_align s.t. cad_pt ≈ R_align @ raw_sam3d_pt.
    # We had: T_icp @ ((raw_sam · prescale) − prescale·src_centroid_raw) ≈ cad_pt − dst_centroid
    # ⇒  R_lin = T_icp[:3,:3] · prescale (3×3, isotropic scale + rotation)
    #    t     = T_icp[:3,3] + dst_centroid − R_lin · src_centroid_raw
    R_lin = best_T[:3, :3] * prescale
    t     = best_T[:3, 3] + dst_centroid - R_lin @ src_centroid_raw
    R_align = np.eye(4)
    R_align[:3, :3] = R_lin
    R_align[:3, 3]  = t

    return R_align, dict(
        cad_name=cad_name, sam3d_scale_factor=sf, prescale=prescale,
        icp_cost=float(best_cost),
        near_optima=int(near), n_inits=len(results),
        likely_symmetric=bool(near > 4),
    )


def main():
    ap = argparse.ArgumentParser(description="Baseline1 v2 — ICP SAM3D↔CAD per-object alignment")
    ap.add_argument("--dataset", choices=["dexycb", "oakink"], default="dexycb")
    ap.add_argument("--objects", nargs="*", default=None,
                    help="subset of obj_ids to process (default: all)")
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--n-random-inits", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=80)
    args = ap.parse_args()

    out_dir = os.path.dirname(_out_path(args.dataset, "_"))
    os.makedirs(out_dir, exist_ok=True)

    obj_ids = _list_objects(args.dataset)
    if args.objects:
        wanted = set(args.objects)
        obj_ids = [o for o in obj_ids if o in wanted]
    print(f"dataset={args.dataset}  objects to align: {len(obj_ids)}  out_dir={out_dir}\n")
    print(f"{'obj_id':<14} {'cad_name':<26} {'cost':>12} {'near<1.5x':>10} {'sym':>5} {'sec':>6}  status")
    print("-" * 100)
    n_ok = 0
    for obj_id in obj_ids:
        t0 = time.time()
        T_align, res = align_one(obj_id=obj_id, dataset=args.dataset,
                                 n_samples=args.n_samples,
                                 n_random_inits=args.n_random_inits,
                                 max_iter=args.max_iter)
        dt = time.time() - t0
        if T_align is None:
            print(f"{obj_id:<14} {'?':<26} {'-':>12} {'-':>10} {'-':>5} {dt:>6.1f}  ⏭ {res}")
            continue
        out_file = _out_path(args.dataset, obj_id)
        with open(out_file, "w") as f:
            json.dump(dict(obj_id=obj_id, dataset=args.dataset,
                           R_align_4x4=T_align.tolist(), **res), f, indent=2)
        flag = "  ⚠️ symmetric / multi-optima" if res["likely_symmetric"] else ""
        print(f"{obj_id:<14} {res['cad_name']:<26} {res['icp_cost']:>12.6f} "
              f"{res['near_optima']:>4}/{res['n_inits']:<4} {str(res['likely_symmetric']):>5} "
              f"{dt:>6.1f}  ✅ saved{flag}")
        n_ok += 1
    print("-" * 100)
    print(f"{n_ok}/{len(obj_ids)} objects aligned → {out_dir}")


if __name__ == "__main__":
    main()
