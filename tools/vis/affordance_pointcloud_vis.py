"""Shared affordance / plain point-cloud PNG rendering (jet, transparent)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model.inference_v6 import (
    _set_equal_3d_limits,
    _style_3d_background_axes,
    affordance_rgb,
)

PLAIN_POINT_RGB = (0.44, 0.46, 0.50)


def save_pointcloud_png(
    path: Path,
    points: np.ndarray,
    *,
    values: np.ndarray | None = None,
    solid_rgb: tuple[float, float, float] | None = None,
    vmax: float = 1.0,
    elev: float = 22.0,
    azim: float = 132.0,
    dpi: int = 140,
    transparent: bool = True,
    show_decorations: bool = False,
    crop_margin: bool = True,
) -> None:
    """3D scatter PNG: heatmap (``values``) or uniform color (``solid_rgb``)."""
    pts = np.asarray(points, dtype=np.float64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if solid_rgb is not None:
        rgb = np.tile(np.asarray(solid_rgb[:3], dtype=np.float64), (len(pts), 1))
    elif values is not None:
        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        rgb = affordance_rgb(vals, vmax=vmax)
    else:
        raise ValueError("provide values for heatmap or solid_rgb for plain cloud")

    if transparent and not show_decorations:
        fig = plt.figure(figsize=(7, 7), dpi=dpi)
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor((0, 0, 0, 0))
        ax.axis("off")
    else:
        fig = plt.figure(figsize=(7, 7), facecolor="#1a1a2e", dpi=dpi)
        ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=rgb,
        s=2.5,
        alpha=0.85,
        linewidths=0,
        depthshade=False,
    )
    _set_equal_3d_limits(ax, pts)
    ax.view_init(elev=elev, azim=azim)

    if show_decorations:
        _style_3d_background_axes(ax)
    else:
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor((0, 0, 0, 0))
        ax.grid(False)

    save_kw = dict(dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    if transparent and not show_decorations:
        save_kw["transparent"] = True
        save_kw["facecolor"] = "none"
    else:
        save_kw["facecolor"] = "#1a1a2e"

    fig.savefig(path, **save_kw)
    plt.close(fig)

    if crop_margin and transparent and not show_decorations:
        from PIL import Image

        img = Image.open(path).convert("RGBA")
        alpha = np.asarray(img.split()[-1])
        ys, xs = np.where(alpha > 8)
        if len(xs) > 0:
            pad = max(6, min(img.width, img.height) // 40)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(img.width - 1, int(xs.max()) + pad)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(img.height - 1, int(ys.max()) + pad)
            img.crop((x0, y0, x1 + 1, y1 + 1)).save(path)


def save_scalar_pointcloud_png(
    path: Path,
    points: np.ndarray,
    values: np.ndarray,
    title: str | None = None,
    *,
    vmax: float = 1.0,
    elev: float = 22.0,
    azim: float = 132.0,
    dpi: int = 140,
    transparent: bool = True,
    show_decorations: bool = False,
    crop_margin: bool = True,
) -> None:
    """Heatmap point cloud (jet colormap)."""
    save_pointcloud_png(
        path,
        points,
        values=values,
        vmax=vmax,
        elev=elev,
        azim=azim,
        dpi=dpi,
        transparent=transparent,
        show_decorations=show_decorations,
        crop_margin=crop_margin,
    )
