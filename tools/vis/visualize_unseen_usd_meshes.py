#!/usr/bin/env python3
"""
Visualize USD meshes under data_hub/RawData/Unseen/usd.

Outputs:
  - data_hub/RawData/Unseen/vis/{obj_id}.png
  - data_hub/RawData/Unseen/vis/all_objects_grid.png   (6 per row)

Each per-object image includes:
  - the mesh (subsampled faces if needed)
  - an axis triad at the mesh frame origin (0,0,0)
  - an origin marker
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


PROJ = Path(__file__).resolve().parents[1]
UNSEEN_DIR = PROJ / "data_hub" / "RawData" / "Unseen"
USD_DIR = UNSEEN_DIR / "usd"
VIS_DIR = UNSEEN_DIR / "vis"


def load_root_mesh_usd(usd_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load /Root/Mesh from a USD file as (V,F) in meters."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise ValueError(f"Failed to open USD: {usd_path}")

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    prim = stage.GetPrimAtPath("/Root/Mesh")
    if not prim or not prim.IsValid():
        raise ValueError(f"Missing /Root/Mesh: {usd_path}")
    if not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"/Root/Mesh is not UsdGeom.Mesh: {usd_path}")

    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    fvi = mesh.GetFaceVertexIndicesAttr().Get()
    fvc = mesh.GetFaceVertexCountsAttr().Get()
    if not pts or not fvi or not fvc:
        raise ValueError(f"Empty mesh attrs: {usd_path}")

    V = np.asarray([(p[0], p[1], p[2]) for p in pts], dtype=np.float64) * meters_per_unit
    # We expect triangles (counts=3), but handle fan triangulation just in case.
    faces = []
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
            for i in range(1, n - 1):
                faces.append([poly[0], poly[i], poly[i + 1]])
    F = np.asarray(faces, dtype=np.int32)
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"Unexpected face array shape {F.shape}: {usd_path}")
    return V, F


def bbox(V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    vmin = V.min(axis=0)
    vmax = V.max(axis=0)
    ext = vmax - vmin
    diag = float(np.linalg.norm(ext))
    return vmin, vmax, ext, diag


def subsample_faces(F: np.ndarray, max_faces: int, seed: int = 0) -> np.ndarray:
    if max_faces <= 0 or len(F) <= max_faces:
        return F
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), size=max_faces, replace=False)
    return F[idx]


def set_axes_equal(ax, vmin: np.ndarray, vmax: np.ndarray):
    # Equal aspect for 3D axes: set limits to a cube.
    mid = (vmin + vmax) / 2.0
    span = float(np.max(vmax - vmin))
    if not math.isfinite(span) or span <= 0:
        span = 1.0
    r = span / 2.0
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)


def draw_triad(ax, *, length: float):
    # Axis triad at origin
    ax.quiver(0, 0, 0, length, 0, 0, color="r", linewidth=2, arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, length, 0, color="g", linewidth=2, arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, 0, length, color="b", linewidth=2, arrow_length_ratio=0.15)
    ax.scatter([0], [0], [0], color="k", s=18)


def render_single(
    obj_id: str,
    V: np.ndarray,
    F: np.ndarray,
    out_path: Path,
    *,
    max_faces_vis: int,
    elev: float,
    azim: float,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vmin, vmax, ext, diag = bbox(V)
    Fv = subsample_faces(F, max_faces_vis, seed=abs(hash(obj_id)) % (2**32))
    tri = V[Fv]
    ext_cm = ext * 100.0

    fig = plt.figure(figsize=(6.5, 6.0), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    coll = Poly3DCollection(tri, linewidths=0.05, alpha=0.95)
    coll.set_facecolor((0.75, 0.78, 0.82, 1.0))
    coll.set_edgecolor((0.15, 0.15, 0.15, 0.15))
    ax.add_collection3d(coll)

    # Triad length relative to bbox
    triad_len = float(max(ext.max(), 1e-6) * 0.25)
    draw_triad(ax, length=triad_len)

    set_axes_equal(ax, vmin, vmax)
    ax.view_init(elev=elev, azim=azim)

    ax.set_title(f"{obj_id}", fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.text2D(
        0.02,
        0.02,
        f"bbox (cm): {ext_cm[0]:.1f}×{ext_cm[1]:.1f}×{ext_cm[2]:.1f}",
        transform=ax.transAxes,
        fontsize=9,
        color="black",
    )

    # Light grid, transparent panes
    ax.grid(True, linewidth=0.3, alpha=0.4)
    try:
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
    except Exception:
        pass
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_grid(
    items: list[tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    *,
    cols: int,
    max_faces_vis: int,
    elev: float,
    azim: float,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    n = len(items)
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(cols * 3.1, rows * 3.0), dpi=160)

    for i, (obj_id, V, F) in enumerate(items):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        vmin, vmax, ext, _ = bbox(V)
        Fv = subsample_faces(F, max_faces_vis, seed=abs(hash(obj_id)) % (2**32))
        tri = V[Fv]
        ext_cm = ext * 100.0

        coll = Poly3DCollection(tri, linewidths=0.02, alpha=0.95)
        coll.set_facecolor((0.78, 0.80, 0.84, 1.0))
        coll.set_edgecolor((0.12, 0.12, 0.12, 0.12))
        ax.add_collection3d(coll)

        triad_len = float(max(ext.max(), 1e-6) * 0.25)
        draw_triad(ax, length=triad_len)
        set_axes_equal(ax, vmin, vmax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(
            f"{obj_id}\n{ext_cm[0]:.0f}×{ext_cm[1]:.0f}×{ext_cm[2]:.0f}cm",
            fontsize=7,
            pad=2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        try:
            ax.xaxis.pane.set_alpha(0.0)
            ax.yaxis.pane.set_alpha(0.0)
            ax.zaxis.pane.set_alpha(0.0)
        except Exception:
            pass
    # Hide any remaining empty subplots
    for j in range(n, rows * cols):
        ax = fig.add_subplot(rows, cols, j + 1, projection="3d")
        ax.set_axis_off()

    fig.suptitle("Unseen objects (USD) - mesh frame at origin", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd-dir", default=str(USD_DIR))
    ap.add_argument("--vis-dir", default=str(VIS_DIR))
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--max-faces-vis", type=int, default=20000, help="Max faces drawn per object (for speed)")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=45.0)
    args = ap.parse_args()

    usd_dir = Path(args.usd_dir)
    vis_dir = Path(args.vis_dir)
    usd_files = sorted(p for p in usd_dir.glob("*.usd") if p.is_file() and not p.name.endswith("_meta.usd"))
    if not usd_files:
        raise SystemExit(f"No USD files found in: {usd_dir}")

    items: list[tuple[str, np.ndarray, np.ndarray]] = []
    for p in usd_files:
        obj_id = p.stem
        V, F = load_root_mesh_usd(p)
        items.append((obj_id, V, F))

        out_png = vis_dir / f"{obj_id}.png"
        render_single(
            obj_id,
            V,
            F,
            out_png,
            max_faces_vis=int(args.max_faces_vis),
            elev=float(args.elev),
            azim=float(args.azim),
        )

    render_grid(
        items,
        vis_dir / "all_objects_grid.png",
        cols=int(args.cols),
        max_faces_vis=max(5000, int(args.max_faces_vis // 4)),
        elev=float(args.elev),
        azim=float(args.azim),
    )

    print(f"Done. Wrote {len(items)} per-object images + grid to {vis_dir}")


if __name__ == "__main__":
    main()

