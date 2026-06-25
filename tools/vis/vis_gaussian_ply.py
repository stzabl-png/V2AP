#!/usr/bin/env python3
"""Preview SAM3D Gaussian PLY files as colored point clouds.

This is a lightweight QC viewer for files like:
    data_hub/gaussians/IMG_4475.ply

It reads the Gaussian Splat PLY vertex positions and f_dc_* colors, then writes
static PNG previews. It does not render true splats; use it to quickly inspect
shape, orientation, and whether the output loaded correctly.
"""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJ = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJ / "data_hub" / "gaussians"
DEFAULT_OUT = PROJ / "output" / "gaussian_vis"


PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("f_dc_0", "<f4"),
        ("f_dc_1", "<f4"),
        ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"),
        ("scale_1", "<f4"),
        ("scale_2", "<f4"),
        ("rot_0", "<f4"),
        ("rot_1", "<f4"),
        ("rot_2", "<f4"),
        ("rot_3", "<f4"),
    ]
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def read_gaussian_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return xyz, rgb, opacity from a SAM3D/3DGS binary little-endian PLY."""
    with path.open("rb") as f:
        vertex_count = None
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} ended before PLY header completed")
            text = line.decode("utf-8", errors="replace").strip()
            header_lines.append(text)
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            if text == "end_header":
                break

        if header_lines[0] != "ply" or "format binary_little_endian 1.0" not in header_lines:
            raise ValueError(f"{path} is not a binary_little_endian PLY")
        if vertex_count is None:
            raise ValueError(f"{path} has no element vertex count")

        raw = f.read(vertex_count * PLY_DTYPE.itemsize)
        expected = vertex_count * PLY_DTYPE.itemsize
        if len(raw) != expected:
            raise ValueError(f"{path} is truncated: read {len(raw)} bytes, expected {expected}")

    data = np.frombuffer(raw, dtype=PLY_DTYPE)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    # SAM/3DGS stores SH DC values; this conversion matches common 3DGS preview.
    rgb = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)
    rgb = np.clip(rgb * 0.28209479177387814 + 0.5, 0.0, 1.0)
    opacity = sigmoid(data["opacity"].astype(np.float32))
    return xyz, rgb.astype(np.float32), opacity


def set_equal_2d(ax, a: np.ndarray, b: np.ndarray) -> None:
    lo = np.array([a.min(), b.min()])
    hi = np.array([a.max(), b.max()])
    c = (lo + hi) / 2.0
    r = float((hi - lo).max()) * 0.55 + 1e-6
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_aspect("equal")


def render_preview(
    ply_path: Path,
    out_path: Path,
    *,
    max_points: int,
    min_opacity: float,
    point_size: float,
) -> None:
    xyz, rgb, opacity = read_gaussian_ply(ply_path)
    keep = opacity >= min_opacity
    xyz, rgb, opacity = xyz[keep], rgb[keep], opacity[keep]
    if len(xyz) == 0:
        raise ValueError(f"{ply_path} has no points after opacity filtering")

    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb, opacity = xyz[idx], rgb[idx], opacity[idx]

    views = [
        ("XY", 0, 1),
        ("XZ", 0, 2),
        ("YZ", 1, 2),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="#1a1a2e")
    for ax, (title, i, j) in zip(axes, views):
        ax.set_facecolor("#1a1a2e")
        order = np.argsort(opacity)
        ax.scatter(
            xyz[order, i],
            xyz[order, j],
            c=rgb[order],
            s=point_size,
            alpha=np.clip(opacity[order], 0.15, 1.0),
            linewidths=0,
        )
        set_equal_2d(ax, xyz[:, i], xyz[:, j])
        ax.set_title(title, color="#ddd")
        ax.tick_params(colors="#888", labelsize=7)

    ext = xyz.max(axis=0) - xyz.min(axis=0)
    fig.suptitle(
        f"{ply_path.name}  points={len(xyz):,}  bbox={ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f}",
        color="#ddd",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


def iter_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.ply"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Gaussian PLY files as PNGs")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="PLY file or directory")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-points", type=int, default=120_000)
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=0.35)
    args = parser.parse_args()

    paths = iter_inputs(args.input)
    if not paths:
        raise FileNotFoundError(f"No .ply files found under {args.input}")

    for ply_path in paths:
        out_path = args.outdir / f"{ply_path.stem}.png"
        render_preview(
            ply_path,
            out_path,
            max_points=args.max_points,
            min_opacity=args.min_opacity,
            point_size=args.point_size,
        )
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
