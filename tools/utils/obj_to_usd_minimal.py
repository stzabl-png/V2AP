#!/usr/bin/env python3
"""
obj_to_usd_minimal.py
========================
纯 pxr 脚本：把已有的 OBJ 转成 USD。
必须在 Isaac Sim python 环境里跑:
    conda deactivate
    ~/isaac-sim-5.0/python.sh tools/obj_to_usd_minimal.py
"""
import sys, os
from pathlib import Path

OBJ_PATH = "/home/lyh/Project/V2AP/output/obj_usd/session/chips_20260602_simple.obj"
USD_PATH = "/home/lyh/Project/V2AP/output/obj_usd/session/chips_20260602.usd"

print(f"OBJ: {OBJ_PATH}")
print(f"USD: {USD_PATH}")

try:
    from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt
    print("✅ pxr imported")
except ImportError as e:
    print(f"❌ pxr not available: {e}")
    print("   Run with: conda deactivate && ~/isaac-sim-5.0/python.sh ...")
    sys.exit(1)

# ── Parse OBJ ────────────────────────────────────────────────────
verts, faces = [], []
with open(OBJ_PATH) as f:
    for line in f:
        tok = line.strip().split()
        if not tok: continue
        if tok[0] == 'v':
            verts.append([float(tok[1]), float(tok[2]), float(tok[3])])
        elif tok[0] == 'f':
            # OBJ face indices are 1-based, may have v/vt/vn format
            idx = [int(t.split('/')[0]) - 1 for t in tok[1:]]
            if len(idx) == 3:
                faces.append(idx)
            elif len(idx) == 4:
                faces.append([idx[0], idx[1], idx[2]])
                faces.append([idx[0], idx[2], idx[3]])

print(f"  Parsed: {len(verts)} verts, {len(faces)} faces")

# ── Create USD ───────────────────────────────────────────────────
stage = Usd.Stage.CreateNew(USD_PATH)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

xform = UsdGeom.Xform.Define(stage, "/Root")
mesh  = UsdGeom.Mesh.Define(stage, "/Root/Mesh")

pts = Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts])
fvc = Vt.IntArray([3] * len(faces))
fvi = Vt.IntArray([i for tri in faces for i in tri])

mesh.GetPointsAttr().Set(pts)
mesh.GetFaceVertexCountsAttr().Set(fvc)
mesh.GetFaceVertexIndicesAttr().Set(fvi)
mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

# Physics
UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
mc = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
mc.GetApproximationAttr().Set("convexHull")
UsdPhysics.MassAPI.Apply(xform.GetPrim()).GetMassAttr().Set(0.05)

stage.SetDefaultPrim(xform.GetPrim())
stage.Save()

print(f"✅ USD saved: {USD_PATH}")
print(f"✅ Done")
