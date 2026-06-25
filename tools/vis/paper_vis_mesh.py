#!/usr/bin/env python3
"""
Render rotated SAM3D meshes (metric scale) as gray triangle meshes on transparent PNG.

Default mesh root: data_hub/meshes/SAM3DMesh/rotated_mesh

Usage::

    python tools/paper_vis_mesh.py \\
        --obj C14001 O36001 A01010 \\
        --output-dir output/paper_vis/mesh
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

# Headless GL for pyrender (must be set before importing pyrender).
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_MESH_ROOT = PROJ / "data_hub" / "meshes" / "SAM3DMesh" / "rotated_mesh"
DEFAULT_OUT = PROJ / "output" / "paper_vis" / "mesh"

sys.path.insert(0, str(PROJ / "tools"))
import random_grasp_sampler as rgs  # noqa: E402

MESH_GRAY = (0.58, 0.60, 0.64, 1.0)
WIRE_GRAY = (0.30, 0.32, 0.34, 1.0)  # soft light-black edge lines
WIRE_EMISSIVE = (0.10, 0.10, 0.11)
AMBIENT_LIGHT = (0.22, 0.22, 0.24)
DEFAULT_WIRE_FACES = 20000


def prepare_render_mesh(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """Decimate dense SAM3D meshes so triangle edges are visible at paper resolution."""
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh
    simplified = mesh.simplify_quadric_decimation(face_count=int(max_faces))
    if simplified is None or len(simplified.faces) == 0:
        return mesh
    return simplified


def add_directional_lights(scene) -> None:
    """Three soft directional lights for readable 3D shading."""
    import pyrender

    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp], dtype=np.float64)
        z /= max(float(np.linalg.norm(z)), 1e-12)
        x = np.array([-z[1], z[0], 0.0], dtype=np.float64)
        if float(np.linalg.norm(x)) < 1e-8:
            x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x /= float(np.linalg.norm(x))
        y = np.cross(z, x)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.c_[x, y, z]
        scene.add_node(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=2.2),
                matrix=matrix,
            )
        )


def load_metric_mesh(obj_id: str, mesh_root: Path) -> tuple[trimesh.Trimesh, str, str]:
    mesh_path, scale_factor, ds, apply_scale = rgs.find_obj_mesh(
        obj_id,
        dataset=None,
        use_legacy_assets=False,
    )
    if mesh_path is None:
        raise FileNotFoundError(f"mesh not found for {obj_id} under {mesh_root}")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if apply_scale and abs(float(scale_factor) - 1.0) > 1e-8:
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(scale_factor)
    return mesh, mesh_path, ds or rgs.infer_obj_dataset(obj_id)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Camera pose (4x4): camera at ``eye`` looking at ``target`` (OpenGL / pyrender)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up is None else np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    right = np.cross(forward, up)
    if float(np.linalg.norm(right)) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right /= float(np.linalg.norm(right))
    up_cam = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up_cam
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def camera_eye_from_view(
    center: np.ndarray,
    extent: float,
    *,
    elev: float,
    azim: float,
    distance_scale: float = 2.4,
) -> np.ndarray:
    """Match matplotlib ``view_init(elev, azim)`` eye position."""
    elev_r = np.radians(elev)
    azim_r = np.radians(azim)
    dist = max(float(extent), 1e-6) * distance_scale
    offset = dist * np.array([
        np.cos(elev_r) * np.cos(azim_r),
        np.cos(elev_r) * np.sin(azim_r),
        np.sin(elev_r),
    ], dtype=np.float64)
    return center + offset


def build_scene(
    mesh: trimesh.Trimesh,
    *,
    center: np.ndarray,
    extent: float,
    elev: float,
    azim: float,
    show_wireframe: bool,
) -> "pyrender.Scene":
    import pyrender

    solid_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=list(MESH_GRAY),
        emissiveFactor=[0.0, 0.0, 0.0],
        metallicFactor=0.05,
        roughnessFactor=0.72,
        alphaMode="OPAQUE",
        doubleSided=True,
    )
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=list(AMBIENT_LIGHT))
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=solid_mat, smooth=True))

    if show_wireframe:
        wire_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=list(WIRE_GRAY),
            emissiveFactor=list(WIRE_EMISSIVE),
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="OPAQUE",
            doubleSided=True,
            wireframe=True,
        )
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=wire_mat, wireframe=True, smooth=False))

    eye = camera_eye_from_view(center, extent, elev=elev, azim=azim)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    scene.add(camera, pose=look_at(eye, center))
    add_directional_lights(scene)
    return scene

def render_mesh_png(
    mesh: trimesh.Trimesh,
    out_path: Path,
    *,
    obj_id: str = "",
    elev: float = 22.0,
    azim: float = 132.0,
    resolution: int = 960,
    show_wireframe: bool = True,
    wire_faces: int = DEFAULT_WIRE_FACES,
) -> None:
    import pyrender
    from PIL import Image

    render_mesh = prepare_render_mesh(mesh, wire_faces if show_wireframe else 0)
    verts = np.asarray(render_mesh.vertices, dtype=np.float64)
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    extent = float((verts.max(axis=0) - verts.min(axis=0)).max())

    scene = build_scene(
        render_mesh,
        center=center,
        extent=extent,
        elev=elev,
        azim=azim,
        show_wireframe=show_wireframe,
    )

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    img = Image.fromarray(color, mode="RGBA")
    # Trim empty transparent border.
    alpha = np.asarray(img.split()[-1])
    ys, xs = np.where(alpha > 8)
    if len(xs) > 0:
        pad = max(4, resolution // 80)
        x0, x1 = max(0, xs.min() - pad), min(resolution - 1, xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(resolution - 1, ys.max() + pad)
        img = img.crop((x0, y0, x1 + 1, y1 + 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Gray mesh PNG (transparent bg) for paper_vis")
    p.add_argument(
        "--obj",
        nargs="+",
        default=[
            "C14001", "O36001", "A01010", "A15027", "C03001", "C22001",
            "ycb_dex_05", "Y35037", "ycb_dex_06",
        ],
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    p.add_argument("--resolution", type=int, default=960)
    p.add_argument("--wire-faces", type=int, default=DEFAULT_WIRE_FACES,
                   help="Decimate to this many faces so triangle edges are visible")
    p.add_argument("--no-wireframe", dest="show_wireframe", action="store_false",
                   help="Disable triangle edge overlay")
    p.set_defaults(show_wireframe=True)
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=132.0)
    p.add_argument("--overview", action="store_true", default=True)
    p.add_argument("--no-overview", dest="overview", action="store_false")
    args = p.parse_args()

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("paper_vis_mesh  (pyrender, shaded mesh + triangle wireframe)")
    print(f"  objects: {len(args.obj)}")
    print(f"  output:  {out_dir}")
    print("=" * 72)

    png_paths: list[str] = []
    for i, oid in enumerate(args.obj, 1):
        print(f"[{i}/{len(args.obj)}] {oid}")
        mesh, mesh_path, ds = load_metric_mesh(oid, args.mesh_root)
        print(f"  faces={len(mesh.faces):,}  verts={len(mesh.vertices):,}")
        out_path = out_dir / f"{oid}.png"
        render_mesh_png(
            mesh,
            out_path,
            obj_id=oid,
            elev=args.elev,
            azim=args.azim,
            resolution=args.resolution,
            show_wireframe=args.show_wireframe,
            wire_faces=args.wire_faces,
        )
        png_paths.append(str(out_path))
        print(f"  {mesh_path}  ({ds})  -> {out_path}")

    if args.overview and len(png_paths) > 1:
        from PIL import Image

        cells = [Image.open(p).convert("RGBA") for p in png_paths]
        cell_w = max(im.width for im in cells)
        cell_h = max(im.height for im in cells)
        cols = min(3, len(cells))
        rows = (len(cells) + cols - 1) // cols
        canvas = Image.new("RGBA", (cols * cell_w, rows * cell_h), (255, 255, 255, 255))
        for idx, im in enumerate(cells):
            r, c = divmod(idx, cols)
            x = c * cell_w + (cell_w - im.width) // 2
            y = r * cell_h + (cell_h - im.height) // 2
            canvas.paste(im, (x, y), im)
        overview = out_dir / "overview.png"
        canvas.save(overview)
        print(f"overview -> {overview}")

    print("Done.")


if __name__ == "__main__":
    main()
