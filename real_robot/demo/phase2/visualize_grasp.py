"""T6-style grasp preview: camera RGB + mesh overlay + PDM candidate axes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from demo.phase2.retarget import (
    TitanGraspPoses,
    TitanSessionOutput,
    pinch_pose_in_mesh_frame,
)
from demo.phase2.session_io import load_session_input


def _load_mesh_arrays(glb_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not glb_path.is_file():
        return None
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh required to overlay object mesh (pip install trimesh)")
        return None

    loaded = trimesh.load(glb_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            return None
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    else:
        mesh = loaded
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _T_cam_mesh_from_output(output: TitanSessionOutput) -> np.ndarray:
    reg = output.candidates.get("registration", {})
    if "T_cam_mesh" in reg:
        return np.asarray(reg["T_cam_mesh"], dtype=np.float64)
    path = Path(output.session_dir) / "output" / "register" / "T_cam_mesh.json"
    if path.is_file():
        import json

        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return np.asarray(data["T_cam_mesh"], dtype=np.float64)
    raise FileNotFoundError("T_cam_mesh missing from candidates.json and register/")


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    return (R @ pts.T).T + t


def _project_cam(K: np.ndarray, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    z = pts_cam[:, 2]
    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), z


def _draw_mesh_overlay(
    bgr: np.ndarray,
    K: np.ndarray,
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    *,
    face_bgr: tuple[int, int, int] = (190, 210, 90),
    alpha: float = 0.48,
) -> np.ndarray:
    import cv2

    h, w = bgr.shape[:2]
    overlay = bgr.copy()
    tris: list[tuple[float, np.ndarray]] = []

    for face in faces:
        tri3 = vertices_cam[face]
        if np.any(tri3[:, 2] <= 0.02):
            continue
        uv, z = _project_cam(K, tri3)
        if np.any(uv[:, 0] < -50) or np.any(uv[:, 0] > w + 50):
            continue
        if np.any(uv[:, 1] < -50) or np.any(uv[:, 1] > h + 50):
            continue
        tris.append((float(np.mean(z)), uv))

    tris.sort(key=lambda item: item[0], reverse=True)
    for _, uv in tris:
        pts = uv.astype(np.int32)
        cv2.fillConvexPoly(overlay, pts, face_bgr, lineType=cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0)


def _draw_axis_triad(
    img: np.ndarray,
    K: np.ndarray,
    T_cam: np.ndarray,
    *,
    axis_len_m: float,
    thickness: int,
) -> None:
    import cv2

    origin = T_cam[:3, 3]
    R = T_cam[:3, :3]
    o_uv, o_z = _project_cam(K, origin[None])
    if o_z[0] <= 0.02:
        return
    o_pt = (int(round(o_uv[0, 0])), int(round(o_uv[0, 1])))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i, col in enumerate(colors):
        p_cam = origin + R[:, i] * axis_len_m
        uv, z = _project_cam(K, p_cam[None])
        if z[0] <= 0.02:
            continue
        p_pt = (int(round(uv[0, 0])), int(round(uv[0, 1])))
        cv2.line(img, o_pt, p_pt, col, thickness, lineType=cv2.LINE_AA)


def _draw_approach_segment(
    img: np.ndarray,
    K: np.ndarray,
    T_cam_pinch: np.ndarray,
    *,
    seg_len_m: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw gripper arrival direction (-approach column) like T6 PDM vis."""
    import cv2

    origin = T_cam_pinch[:3, 3]
    approach_mesh = T_cam_pinch[:3, :3][:, 2]
    n = float(np.linalg.norm(approach_mesh))
    if n < 1e-9:
        return
    approach_cam = approach_mesh / n
    p0 = origin
    p1 = origin - approach_cam * seg_len_m
    uv0, z0 = _project_cam(K, p0[None])
    uv1, z1 = _project_cam(K, p1[None])
    if z0[0] <= 0.02 or z1[0] <= 0.02:
        return
    pt0 = (int(round(uv0[0, 0])), int(round(uv0[0, 1])))
    pt1 = (int(round(uv1[0, 0])), int(round(uv1[0, 1])))
    cv2.line(img, pt0, pt1, color, thickness, lineType=cv2.LINE_AA)
    cv2.circle(img, pt0, max(2, thickness + 1), color, -1, lineType=cv2.LINE_AA)


def _candidate_T_cam_pinch(T_cam_mesh: np.ndarray, candidate: dict) -> np.ndarray:
    T_mesh_pinch = pinch_pose_in_mesh_frame(candidate)
    return T_cam_mesh @ T_mesh_pinch


def render_t6_style_grasp_preview(
    session_dir: Path,
    output: TitanSessionOutput,
    selected: TitanGraspPoses,
) -> np.ndarray:
    """
    Build BGR image: capture RGB + mesh overlay + **selected** grasp only.
    """
    import cv2

    session_dir = Path(session_dir)
    input_dir = session_dir / "input"
    sess = load_session_input(input_dir)
    bgr = cv2.cvtColor(sess.rgb, cv2.COLOR_RGB2BGR)

    T_cam_mesh = _T_cam_mesh_from_output(output)
    glb_path = session_dir / "output" / "mesh" / "object_base_aligned.glb"
    mesh_data = _load_mesh_arrays(glb_path)
    if mesh_data is not None:
        verts_mesh, faces = mesh_data
        verts_cam = _transform_points(T_cam_mesh, verts_mesh)
        bgr = _draw_mesh_overlay(bgr, sess.K, verts_cam, faces)

    cands = list(output.candidates.get("candidates", []))
    selected_rank = int(selected.rank)

    sel_cand = next(
        (c for c in cands if int(c.get("rank", -1)) == selected_rank),
        None,
    )
    if sel_cand is not None:
        T_cam_sel = _candidate_T_cam_pinch(T_cam_mesh, sel_cand)
    else:
        T_cam_sel = np.linalg.inv(sess.T_base_cam) @ selected.T_base_pinch

    _draw_approach_segment(
        bgr,
        sess.K,
        T_cam_sel,
        seg_len_m=0.12,
        color=(0, 255, 255),
        thickness=3,
    )
    _draw_axis_triad(bgr, sess.K, T_cam_sel, axis_len_m=0.035, thickness=3)

    cv2.putText(
        bgr,
        "camera RGB + mesh overlay  |  yellow line + RGB axes = selected grasp only",
        (8, bgr.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        lineType=cv2.LINE_AA,
    )
    return bgr


TITAN_PIPELINE_VIS_REL: tuple[tuple[str, str], ...] = (
    ("T3 SAM3D", "output/vis/T3_sam3d_mesh_preview.png"),
    ("T4 Metric scale", "output/vis/T4_scale_scene_preview.png"),
    ("T5 FoundationPose", "output/vis/T5_foundationpose_overlay.png"),
    ("T6 Grasp candidates", "output/vis/T6_grasp_vis.png"),
)


def show_png_blocking(path: Path, *, title: str = "", image_title: bool = True) -> None:
    """
    Show a PNG in a GUI window and block until the user closes it.

    Prefer matplotlib (close figure → continue). Falls back to ``cv2.imshow``
    + waitKey when matplotlib is unavailable.
    """
    path = Path(path).resolve()
    if not path.is_file():
        logger.warning(f"Vis image missing, skip: {path}")
        return

    label = title or path.name
    logger.info(f"Review: {label} — close the image window to continue.")

    try:
        import matplotlib.pyplot as plt

        img = plt.imread(str(path))
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.canvas.manager.set_window_title(label)  # type: ignore[union-attr]
        ax.imshow(img)
        ax.axis("off")
        if image_title and label:
            ax.set_title(label, fontsize=11, pad=8)
        fig.tight_layout()
        plt.show(block=True)
        plt.close(fig)
        return
    except ImportError:
        pass
    except Exception as exc:
        logger.warning(f"matplotlib viewer failed ({exc}); trying OpenCV.")

    try:
        import cv2

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imread failed for {path}")
        cv2.imshow(label, bgr)
        cv2.waitKey(0)
        cv2.destroyWindow(label)
        return
    except Exception as exc:
        logger.warning(f"OpenCV viewer failed ({exc}).")

    try:
        input(f"Press Enter after reviewing {label}...")
    except EOFError:
        logger.warning("Non-interactive terminal — continuing without review.")


def show_titan_pipeline_vis(session_dir: Path) -> None:
    """Sequentially popup Titan T3–T6 PNGs; block until each window is closed."""
    session_dir = Path(session_dir)
    for step_title, rel in TITAN_PIPELINE_VIS_REL:
        show_png_blocking(session_dir / rel, title=step_title)


def _open_image_externally(path: Path) -> bool:
    """Open PNG in the default OS viewer (non-blocking, separate process)."""
    import shutil
    import subprocess
    import sys

    path = Path(path).resolve()
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        if shutil.which("xdg-open"):
            subprocess.Popen(
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
    except OSError as exc:
        logger.warning(f"Could not launch external image viewer: {exc}")
    return False


def show_candidate_grasp_preview(
    session_dir: Path,
    output: TitanSessionOutput,
    selected: TitanGraspPoses,
    *,
    title: str | None = None,
    block: bool = True,
    preview_label: str | None = None,
    save_path: Path | None = None,
) -> Path | None:
    """
    T6-style grasp preview: save PNG and optionally show in a blocking viewer.

    Does **not** use ``cv2.imshow`` during live robot control (matplotlib preferred).
    When ``block=True``, blocks until the user closes the preview window, then motion
    may start. Independent of ``run_auto_grasp --no-prompts`` (motion Enter prompts).
    """
    import cv2

    session_dir = Path(session_dir)
    if title is None:
        title = f"Grasp preview rank {selected.rank} ({selected.name})"
        if preview_label:
            title = f"{title} [{preview_label}]"

    logger.info(
        f"Grasp preview (camera RGB): rank={selected.rank} ({selected.name}), "
        f"pinch base xyz={selected.T_base_pinch[:3, 3].round(3).tolist()}"
    )
    logger.info("Rendering grasp preview (RGB + mesh overlay + candidates)...")
    try:
        bgr = render_t6_style_grasp_preview(
            session_dir,
            output,
            selected,
        )
    except Exception as exc:
        logger.error(f"Grasp preview render failed ({exc}); continuing without preview.")
        return None

    if float(np.mean(bgr)) < 5.0:
        logger.warning("Grasp preview image looks empty/dark — check input/rgb/left_rgb.png")

    if save_path is None:
        stem = f"razor_grasp_preview_rank{int(selected.rank)}"
        if preview_label:
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in preview_label)
            stem = f"{stem}_{safe}"
        save_path = (
            session_dir
            / "output"
            / "vis"
            / f"{stem}.png"
        )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(save_path), bgr):
        logger.error(f"Failed to write grasp preview PNG: {save_path}")
        return None

    logger.info(f"Grasp preview saved: {save_path}")

    if not block:
        return save_path

    show_png_blocking(save_path, title=title, image_title=False)
    return save_path
