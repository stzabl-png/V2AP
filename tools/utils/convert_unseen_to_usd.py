#!/usr/bin/env python3
"""
convert_unseen_to_usd.py
=======================
Batch-convert meshes in data_hub/RawData/Unseen to a unified IsaacSim-friendly USD.

Inputs:
  - *.glb, *.obj, *.usd (USD crate/ascii)

Outputs (default):
  data_hub/RawData/Unseen/usd/{obj_id}.usd
  data_hub/RawData/Unseen/usd/{obj_id}_meta.json

Policy:
  - Normalize to a single UsdGeom.Mesh at /Root/Mesh
  - Up axis: Z
  - metersPerUnit: 1.0
  - Triangulate geometry
  - Auto-scale to a grasp-friendly size using name heuristics and/or per-object *.scale.json
  - Optional decimation for very dense meshes (best-effort)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
UNSEEN_DIR = PROJ / "data_hub" / "RawData" / "Unseen"
DEFAULT_OUT_DIR = UNSEEN_DIR / "usd"

# Objects that require a +90deg rotation about +X (Y -> Z) before scaling/centering.
BASE_ROTATE_X_POS_90: set[str] = {
    "Omnidroid2",
    "bottle_of_champagne",
    "canon_at-1_retro_camera",
    "coffee_shop_cup",
    "cosmetic_product",
    "cup",
    "free_zuk_3d_model",
    "lion_sculpture",
    "low_poly_asset_teddy_bear",
    "luigi_doll_hd",
    "nike_air_zoom_pegasus_36",
    "polaroid_camera",
    "sculpture",
    "simple_cola_can",
    "simple_glass_vase",
    "star_wars_-_c3po",
    "toy_soldier",
    "traffic_cone_scan",
    "vase",
}

# Per-object user overrides (ordered). Units for target sizes are centimeters.
# NOTE: "L/W/H" correspond to bbox extents along X/Y/Z axes.
USER_SPECS: dict[str, list[dict[str, Any]]] = {
    # 1. Jam007: L -> 6cm
    "Jam007": [{"scale_to_cm": {"axis": "x", "value": 6.0}}],
    # 2. Kettle027: rotate +90deg about Z (X->Y)
    "Kettle027": [{"rot_deg": {"axis": "z", "deg": 90.0}}],
    # 3. Kettle030: rotate -90deg about Z (Y->X)
    "Kettle030": [{"rot_deg": {"axis": "z", "deg": -90.0}}],
    # 4. Omnidroid2: W -> 8cm
    "Omnidroid2": [{"scale_to_cm": {"axis": "y", "value": 8.0}}],
    # 5. box: L -> 10cm
    "box": [{"scale_to_cm": {"axis": "x", "value": 10.0}}],
    # 6. button: L -> 7cm
    "button": [{"scale_to_cm": {"axis": "x", "value": 7.0}}],
    # 7. coffee_shop_cup: L -> 6cm
    "coffee_shop_cup": [{"scale_to_cm": {"axis": "x", "value": 6.0}}],
    # 8. free_zuk_3d_model: L -> 6cm, then rotate +90deg about Z
    "free_zuk_3d_model": [
        {"scale_to_cm": {"axis": "x", "value": 6.0}},
        {"rot_deg": {"axis": "z", "deg": 90.0}},
    ],
    # 9. lion_sculpture: rotate 180 about Z, then L -> 7cm
    "lion_sculpture": [
        {"rot_deg": {"axis": "z", "deg": 180.0}},
        {"scale_to_cm": {"axis": "x", "value": 7.0}},
    ],
    # 10. low_poly_asset_teddy_bear: L -> 16cm, then rotate 180 about Z
    "low_poly_asset_teddy_bear": [
        {"scale_to_cm": {"axis": "x", "value": 16.0}},
        {"rot_deg": {"axis": "z", "deg": 180.0}},
    ],
    # 11. luigi_doll_hd: rotate 180 about Z
    "luigi_doll_hd": [{"rot_deg": {"axis": "z", "deg": 180.0}}],
    # 12. nike_air_zoom_pegasus_36: rotate -90deg about Z (Y->X)
    "nike_air_zoom_pegasus_36": [{"rot_deg": {"axis": "z", "deg": -90.0}}],
    # 13. paper_cup_1: L -> 8cm
    "paper_cup_1": [{"scale_to_cm": {"axis": "x", "value": 8.0}}],
    # 14. polaroid_camera: L -> 8cm, then rotate -90deg about Z (Y->X)
    "polaroid_camera": [
        {"scale_to_cm": {"axis": "x", "value": 8.0}},
        {"rot_deg": {"axis": "z", "deg": -90.0}},
    ],
    # 15. simple_cola_can: L -> 7.5cm
    "simple_cola_can": [{"scale_to_cm": {"axis": "x", "value": 7.5}}],
    # 16. simple_glass_vase: L -> 12cm
    "simple_glass_vase": [{"scale_to_cm": {"axis": "x", "value": 12.0}}],
    # 17. simple_propane_tank: L -> 10cm, then rotate +90deg about X (Y->Z)
    # (User wrote "绕z从+y朝+z转90度" which is not a Z-rotation; interpret as X+90.)
    "simple_propane_tank": [
        {"scale_to_cm": {"axis": "x", "value": 10.0}},
        {"rot_deg": {"axis": "x", "deg": 90.0}},
    ],
    # 18. simple_tv_remote: rotate 180 about Y
    "simple_tv_remote": [{"rot_deg": {"axis": "y", "deg": 180.0}}],
    # 19. stapler: rotate +90 about Z (X->Y)
    "stapler": [{"rot_deg": {"axis": "z", "deg": 90.0}}],
    # 20. star_wars_-_c3po: rotate +90 about Z (X->Y)
    "star_wars_-_c3po": [{"rot_deg": {"axis": "z", "deg": 90.0}}],
    # 21. tape: L -> 6cm
    "tape": [{"scale_to_cm": {"axis": "x", "value": 6.0}}],
    # 22. toy_soldier: rotate 180 about Z
    "toy_soldier": [{"rot_deg": {"axis": "z", "deg": 180.0}}],
    # 23. vase: L -> 18cm
    "vase": [{"scale_to_cm": {"axis": "x", "value": 18.0}}],
}


def _safe_obj_id(stem: str) -> str:
    s = stem.strip().replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s if s else "object"


def _load_scale_hint(unseen_dir: Path, obj_id: str) -> dict[str, Any] | None:
    p = unseen_dir / f"{obj_id}.scale.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _name_based_target_max_dim_m(obj_id: str) -> float:
    """
    Pick a target maximum dimension (AABB max extent) in meters.
    This is intentionally conservative for grasping on a table.
    """
    s = obj_id.lower()

    # Vehicles: treat as a model car (not real car).
    if any(k in s for k in ["kia", "picanto", "xiaomi", "su7", "car", "gt-line"]):
        return 0.22

    # Wearables
    if "shoe" in s or "nike" in s:
        return 0.30

    # Containers / bottles
    if any(k in s for k in ["bottle", "wine", "champagne"]):
        return 0.30

    # Kettle-like
    if "kettle" in s:
        return 0.22

    # Cups / small tableware
    if "cup" in s:
        return 0.12

    # Small objects
    if any(k in s for k in ["button", "tape", "stapler", "cosmetic", "camera"]):
        return 0.14

    # Figurines / toys
    if any(k in s for k in ["c3po", "omnidroid", "teddy", "soldier", "toy"]):
        return 0.25

    # Vases / sculptures default a bit larger
    if any(k in s for k in ["vase", "sculpture", "lion"]):
        return 0.28

    # Generic: 20cm max extent
    return 0.20


def _mesh_bbox_extents_m(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    ext = vmax - vmin
    return vmin, vmax, ext


def _apply_rot_x_pos_90(vertices: np.ndarray) -> np.ndarray:
    """
    +90deg about +X (right-hand rule): y -> z, z -> -y
    """
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return (R @ vertices.T).T


def _apply_axis_angle_deg(vertices: np.ndarray, *, axis: str, deg: float) -> np.ndarray:
    a = axis.lower()
    th = float(deg) * math.pi / 180.0
    c = math.cos(th)
    s = math.sin(th)
    if a == "x":
        R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    elif a == "y":
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    elif a == "z":
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        raise ValueError(f"Unknown axis: {axis}")
    return (R @ vertices.T).T


def _uniform_scale_to_axis_cm(vertices_m: np.ndarray, *, axis: str, value_cm: float) -> tuple[float, str]:
    vmin, vmax, ext = _mesh_bbox_extents_m(vertices_m)
    idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    cur = float(ext[idx])
    target_m = float(value_cm) / 100.0
    if cur <= 1e-9:
        return 1.0, "degenerate_axis_extent"
    return target_m / cur, f"scale_to_{axis.lower()}_{value_cm}cm"


def _compute_scale_factor(
    *,
    obj_id: str,
    vertices_m: np.ndarray,
    scale_hint: dict[str, Any] | None,
) -> tuple[float, str]:
    vmin, vmax, ext = _mesh_bbox_extents_m(vertices_m)
    max_dim = float(ext.max())
    if not math.isfinite(max_dim) or max_dim <= 1e-9:
        return 1.0, "degenerate_bbox"

    # If we have a manual target height, honor it.
    if scale_hint and "target_height_m" in scale_hint:
        axis = str(scale_hint.get("height_axis", "z")).lower()
        axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
        height = float(ext[axis_idx])
        target_h = float(scale_hint["target_height_m"])
        if height > 1e-9 and target_h > 1e-6:
            return target_h / height, f"scale_json_target_height_{axis}"

    target_max = _name_based_target_max_dim_m(obj_id)
    return target_max / max_dim, "name_heuristic_target_max_dim"


def _load_trimesh_any(path: Path):
    import trimesh

    m = trimesh.load(str(path), force="mesh")
    if hasattr(m, "geometry") and isinstance(m.geometry, dict):
        # scene -> merge
        if len(m.geometry) == 0:
            raise ValueError(f"Empty scene: {path}")
        m = trimesh.util.concatenate(tuple(m.geometry.values()))

    if not isinstance(m, trimesh.Trimesh):
        raise TypeError(f"Not a mesh: {path}")

    if m.faces is None or len(m.faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")

    if not m.is_watertight:
        # This is fine for grasping; just keep going.
        pass

    # Ensure triangular faces
    if m.faces.shape[1] != 3:
        m = m.triangulate()
    return m


def _usd_stage_to_trimesh(path: Path):
    """
    Read a USD stage, collect UsdGeom.Mesh prims, and merge into one trimesh.
    Uses world-space points (default time).
    """
    from pxr import Usd, UsdGeom
    import trimesh

    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ValueError(f"Failed to open USD: {path}")

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)

    meshes: list[trimesh.Trimesh] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)

        points = m.GetPointsAttr().Get()
        fvi = m.GetFaceVertexIndicesAttr().Get()
        fvc = m.GetFaceVertexCountsAttr().Get()
        if not points or not fvi or not fvc:
            continue

        pts = np.array([(p[0], p[1], p[2]) for p in points], dtype=np.float64)
        # USD points are in stage units; convert to meters.
        pts = pts * meters_per_unit

        # Build faces; triangulate n-gons by fan triangulation.
        faces: list[list[int]] = []
        idx = 0
        for n in fvc:
            n = int(n)
            poly = [int(x) for x in fvi[idx : idx + n]]
            idx += n
            if n < 3:
                continue
            if n == 3:
                faces.append(poly)
            else:
                # fan triangulation: (0, i, i+1)
                for i in range(1, n - 1):
                    faces.append([poly[0], poly[i], poly[i + 1]])

        if not faces:
            continue

        tm = trimesh.Trimesh(vertices=pts, faces=np.array(faces, dtype=np.int64), process=False)
        meshes.append(tm)

    if not meshes:
        raise ValueError(f"No UsdGeom.Mesh prims found: {path}")

    merged = trimesh.util.concatenate(meshes)
    if merged.faces.shape[1] != 3:
        merged = merged.triangulate()
    return merged


def _maybe_decimate(mesh, *, obj_id: str, max_faces: int) -> tuple[Any, str]:
    if max_faces <= 0:
        return mesh, "disabled"
    try:
        n = int(len(mesh.faces))
    except Exception:
        return mesh, "unknown_faces"
    if n <= max_faces:
        return mesh, "skip"

    target = max_faces
    # Best-effort simplification (optional dependency inside trimesh)
    try:
        simplified = mesh.simplify_quadratic_decimation(target)
        if simplified is not None and len(simplified.faces) > 0:
            return simplified, f"quadratic_decimation_{n}->{len(simplified.faces)}"
    except Exception:
        pass

    return mesh, f"failed_keep_{n}"


def _write_usd_mesh(out_usd: Path, mesh, *, obj_id: str, scale_factor: float, scale_source: str, decimate_note: str):
    from pxr import Usd, UsdGeom, Vt, Gf

    verts = np.asarray(mesh.vertices, dtype=np.float64) * float(scale_factor)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Recenter so that mesh-frame origin is bbox center.
    vmin0, vmax0, _ext0 = _mesh_bbox_extents_m(verts)
    bbox_center = (vmin0 + vmax0) / 2.0
    verts = verts - bbox_center

    stage = Usd.Stage.CreateNew(str(out_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)

    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    mesh_prim = UsdGeom.Mesh.Define(stage, "/Root/Mesh")

    mesh_prim.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
    )
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * int(faces.shape[0])))
    mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.reshape(-1).astype(np.int32).tolist()))

    # Normals (optional)
    try:
        if getattr(mesh, "face_normals", None) is not None and len(mesh.face_normals) == len(mesh.faces):
            nrm = np.asarray(mesh.face_normals, dtype=np.float64)
            mesh_prim.GetNormalsAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in nrm])
            )
            mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    except Exception:
        pass

    stage.Save()

    vmin, vmax, ext = _mesh_bbox_extents_m(verts)
    z_offset = float(-vmin[2]) if float(vmin[2]) < 0 else 0.0

    meta = {
        "obj_id": obj_id,
        "source": str(mesh.metadata.get("source", "")) if hasattr(mesh, "metadata") else "",
        "scale_factor": float(scale_factor),
        "scale_source": scale_source,
        "decimate": decimate_note,
        "applied_base_pre_rotation": "x_pos_90" if obj_id in BASE_ROTATE_X_POS_90 else "none",
        "applied_user_ops": USER_SPECS.get(obj_id, []),
        "recenter_to_bbox_center": True,
        "bbox_center_before_recenter_m": [float(x) for x in bbox_center],
        "metersPerUnit": 1.0,
        "up_axis": "Z",
        "bbox_min": [float(x) for x in vmin],
        "bbox_max": [float(x) for x in vmax],
        "bbox_extent_m": [float(x) for x in ext],
        "bbox_extent_cm": [round(float(x) * 100.0, 2) for x in ext],
        "z_offset_m": round(z_offset, 6),
        "n_faces": int(faces.shape[0]),
        "n_vertices": int(verts.shape[0]),
    }
    out_meta = out_usd.with_name(out_usd.stem + "_meta.json")
    out_meta.write_text(json.dumps(meta, indent=2))
    return meta


def convert_one(src: Path, out_dir: Path, *, force: bool, max_faces: int) -> dict[str, Any]:
    obj_id = _safe_obj_id(src.stem)
    out_usd = out_dir / f"{obj_id}.usd"
    if out_usd.exists() and not force:
        return {"obj_id": obj_id, "src": str(src), "skipped": True, "out_usd": str(out_usd)}

    scale_hint = _load_scale_hint(UNSEEN_DIR, obj_id)

    # Load as trimesh in meters
    if src.suffix.lower() in [".glb", ".obj"]:
        mesh = _load_trimesh_any(src)
        mesh.metadata = {"source": str(src)}
    elif src.suffix.lower() in [".usd", ".usda", ".usdc"]:
        mesh = _usd_stage_to_trimesh(src)
        mesh.metadata = {"source": str(src)}
    else:
        raise ValueError(f"Unsupported: {src}")

    # Apply base pre-rotation (affects scale heuristics & final frame).
    if obj_id in BASE_ROTATE_X_POS_90:
        mesh.vertices = _apply_rot_x_pos_90(np.asarray(mesh.vertices, dtype=np.float64))

    # Apply user-specified ops in order.
    applied_ops: list[str] = []
    user_spec_has_scale = False
    if obj_id in USER_SPECS:
        for op in USER_SPECS[obj_id]:
            if "rot_deg" in op:
                rd = op["rot_deg"]
                mesh.vertices = _apply_axis_angle_deg(
                    np.asarray(mesh.vertices, dtype=np.float64),
                    axis=str(rd["axis"]),
                    deg=float(rd["deg"]),
                )
                applied_ops.append(f"rot_{rd['axis']}_{rd['deg']}")
            elif "scale_to_cm" in op:
                sd = op["scale_to_cm"]
                verts_m0 = np.asarray(mesh.vertices, dtype=np.float64)
                sf_u, sf_note = _uniform_scale_to_axis_cm(
                    verts_m0, axis=str(sd["axis"]), value_cm=float(sd["value"])
                )
                mesh.vertices = verts_m0 * sf_u
                applied_ops.append(sf_note)
                user_spec_has_scale = True
            else:
                raise ValueError(f"Unknown op for {obj_id}: {op}")

    verts_m = np.asarray(mesh.vertices, dtype=np.float64)
    if obj_id in USER_SPECS and user_spec_has_scale:
        # User explicitly set a target L/W/H (uniform scaling already applied).
        sf, sf_src = 1.0, "user_spec_scale_applied_in_mesh"
    else:
        # Keep grasp-friendly scaling for objects without explicit target size.
        sf, sf_src = _compute_scale_factor(obj_id=obj_id, vertices_m=verts_m, scale_hint=scale_hint)

    mesh2, dec_note = _maybe_decimate(mesh, obj_id=obj_id, max_faces=max_faces)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _write_usd_mesh(out_usd, mesh2, obj_id=obj_id, scale_factor=sf, scale_source=sf_src, decimate_note=dec_note)
    if applied_ops:
        meta["applied_ops_compact"] = applied_ops
    meta.update({"src": str(src), "out_usd": str(out_usd), "skipped": False})
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=str(UNSEEN_DIR), help="Source directory (default: Unseen)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory (default: Unseen/usd)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing USD outputs")
    ap.add_argument("--max-faces", type=int, default=200_000, help="Decimate meshes above this face count (best-effort)")
    ap.add_argument("--limit", type=int, default=0, help="Only convert first N (0 = all)")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    exts = {".glb", ".obj", ".usd", ".usda", ".usdc"}
    srcs = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if args.limit and args.limit > 0:
        srcs = srcs[: int(args.limit)]

    results = []
    ok = err = skip = 0
    for i, src in enumerate(srcs, 1):
        try:
            r = convert_one(src, out_dir, force=bool(args.force), max_faces=int(args.max_faces))
            results.append(r)
            if r.get("skipped"):
                skip += 1
            else:
                ok += 1
            ext_cm = r.get("bbox_extent_cm", None)
            if isinstance(ext_cm, list) and len(ext_cm) == 3:
                print(f"[{i}/{len(srcs)}] {r['obj_id']}: {ext_cm[0]:.1f}×{ext_cm[1]:.1f}×{ext_cm[2]:.1f} cm  (scale={r.get('scale_factor', 1.0):.4f})")
            else:
                print(f"[{i}/{len(srcs)}] {r['obj_id']}: done")
        except Exception as e:
            err += 1
            results.append({"src": str(src), "error": str(e)})
            print(f"[{i}/{len(srcs)}] ERROR {src.name}: {e}")

    manifest = out_dir / "manifest_unseen_usd.json"
    manifest.write_text(json.dumps({"ok": ok, "skipped": skip, "err": err, "results": results}, indent=2))
    print(f"\nDone. ok={ok} skipped={skip} err={err}")
    print(f"Output dir: {out_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()

