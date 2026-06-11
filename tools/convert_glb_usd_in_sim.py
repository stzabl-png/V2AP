#!/usr/bin/env python3
"""
tools/convert_glb_usd_in_sim.py
================================
在 Isaac Sim 环境里把已导出的 OBJ 转成 USD。
pxr 必须在 SimulationApp 启动后才能 import。

必须用 Isaac Sim python 启动:
    conda deactivate
    ~/isaac-sim-5.0/python.sh tools/convert_glb_usd_in_sim.py --obj-id chips_20260602
"""
import sys, argparse, json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--obj-id", default="chips_20260602")
parser.add_argument("--force",  action="store_true")
args, _ = parser.parse_known_args()

# ── SimulationApp MUST come before any pxr import ───────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# ── pxr is now importable ────────────────────────────────────────
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt

PROJ    = Path(__file__).resolve().parent.parent
USD_DIR = PROJ / "output" / "obj_usd" / "session"
OBJ_PATH = USD_DIR / f"{args.obj_id}_simple.obj"
USD_PATH = USD_DIR / f"{args.obj_id}.usd"

print(f"\n{'='*55}")
print(f"  OBJ → USD (pxr, inside Isaac Sim)")
print(f"  OBJ: {OBJ_PATH}")
print(f"  USD: {USD_PATH}")
print(f"{'='*55}")

if not OBJ_PATH.exists():
    print(f"❌ OBJ not found: {OBJ_PATH}")
    print(f"   Run first (regular python):")
    print(f"   python V2AP-Graspnet-demo/scripts/glb_to_usd.py --session <session> --obj-id {args.obj_id}")
    simulation_app.close()
    sys.exit(1)

if USD_PATH.exists() and not args.force:
    print(f"  ✅ USD already exists (use --force to overwrite)")
else:
    # ── Parse OBJ ────────────────────────────────────────────────
    verts, faces = [], []
    with open(OBJ_PATH) as f:
        for line in f:
            tok = line.strip().split()
            if not tok: continue
            if tok[0] == 'v':
                verts.append([float(tok[1]), float(tok[2]), float(tok[3])])
            elif tok[0] == 'f':
                idx = [int(t.split('/')[0]) - 1 for t in tok[1:]]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])
    print(f"  Parsed OBJ: {len(verts)} verts, {len(faces)} faces")

    # ── Create USD stage ─────────────────────────────────────────
    stage = Usd.Stage.CreateNew(str(USD_PATH))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata('metersPerUnit', 1.0)

    xform = UsdGeom.Xform.Define(stage, "/Root")
    mesh  = UsdGeom.Mesh.Define(stage, "/Root/Mesh")

    pts = Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
    fvc = Vt.IntArray([3] * len(faces))
    fvi = Vt.IntArray([i for tri in faces for i in tri])

    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set(fvc)
    mesh.GetFaceVertexIndicesAttr().Set(fvi)
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    # ⚠️ DO NOT apply physics APIs here!
    # RigidBodyAPI, CollisionAPI, MassAPI must be applied at RUNTIME
    # by SingleRigidPrim / setup_scene, exactly like OakInk/YCB USD files.
    # Pre-baking them into the USD reference prevents physics backend registration.

    stage.SetDefaultPrim(xform.GetPrim())
    stage.Save()
    print(f"  ✅ USD saved (geometry only, no physics): {USD_PATH}")


# ── Write _meta.json (z_offset = 15.8cm, known value) ───────────
z_offset = 0.158  # |z_min| of base_aligned Pringles can mesh
meta_path = USD_DIR / f"{args.obj_id}_meta.json"
if not meta_path.exists() or args.force:
    meta = {
        "obj_id":     args.obj_id,
        "source":     "session_glb",
        "z_offset_m": round(z_offset, 4),
        "mesh_frame": "base_aligned",
        "note": "centroid at origin; z_offset lifts it so bottom rests on table"
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅ _meta.json written: z_offset={z_offset*100:.1f}cm")
else:
    print(f"  ✅ _meta.json already exists")

print(f"{'='*55}")
print(f"\n  Now run sim:")
print(f"    ~/isaac-sim-5.0/python.sh sim/run_grasp_sim.py \\")
print(f"        --hdf5 sim/output/graspnet_chips.hdf5 --headless")
print()

simulation_app.close()
