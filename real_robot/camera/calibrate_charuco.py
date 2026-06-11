#!/usr/bin/env python3
"""
Calibrate head camera intrinsics with a ChArUco board (OpenCV).

Default board matches calib.io label:
  9x14 | Checker 20 mm | Marker 15 mm | DICT_5X5

Uses the same zed_stream RGB path as Phase 2 capture (default 640x360).

Usage:
  # zed_streamer running on dexmate-nano
  python camera/calibrate_charuco.py

  python camera/calibrate_charuco.py

  # Writes demo/phase2/calib/head_zed_left_intrinsics.json (+ K.npy)
  # capture_session.py loads that file automatically when present.

Controls (live window — click the image window first, not the terminal):
  SPACE  — add current frame to calibration set
  c      — run calibration and exit (need >= min-views frames)
  u      — undo last captured view
  q/Esc  — quit without saving

Requires: opencv-contrib-python (cv2.aruco), lz4 if using zed_stream
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase2.constants import CAMERA_FRAME, DEFAULT_HEAD_INTRINSICS_JSON  # noqa: E402

DEFAULT_SQUARES_X = 9
DEFAULT_SQUARES_Y = 14
DEFAULT_SQUARE_LENGTH_M = 0.020
DEFAULT_MARKER_LENGTH_M = 0.015
DEFAULT_ARUCO_DICT = "DICT_5X5_100"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ChArUco camera calibration (zed_stream or webcam)")
    p.add_argument("--host", default=os.environ.get("CAMERA_IP", "<your-camera-ip>"), help="zed_streamer host")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--width", type=int, default=1920, help="Detection stream width (1080p detects easier)")
    p.add_argument("--height", type=int, default=1080)
    p.add_argument(
        "--output-width",
        type=int,
        default=640,
        help="Saved intrinsics width (must match Phase 2 capture)",
    )
    p.add_argument("--output-height", type=int, default=360)
    p.add_argument("--camera", type=int, default=-1, help="If >=0, use local webcam index instead of zed_stream")
    p.add_argument("--squares-x", type=int, default=DEFAULT_SQUARES_X)
    p.add_argument("--squares-y", type=int, default=DEFAULT_SQUARES_Y)
    p.add_argument(
        "--swap-board-axes",
        action="store_true",
        help="Use 14x9 instead of 9x14 if board is not detected",
    )
    p.add_argument(
        "--auto-board-axes",
        action="store_true",
        default=True,
        help="Each frame try 9x14 and 14x9, pick best (default: on)",
    )
    p.add_argument(
        "--no-auto-board-axes",
        action="store_false",
        dest="auto_board_axes",
        help="Disable automatic 9x14 vs 14x9 retry",
    )
    p.add_argument("--square-length-mm", type=float, default=DEFAULT_SQUARE_LENGTH_M * 1000)
    p.add_argument("--marker-length-mm", type=float, default=DEFAULT_MARKER_LENGTH_M * 1000)
    p.add_argument("--dict", dest="aruco_dict", default=DEFAULT_ARUCO_DICT)
    p.add_argument("--min-views", type=int, default=15, help="Minimum captured views before calibrate")
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_HEAD_INTRINSICS_JSON,
    )
    p.add_argument("--timeout", type=float, default=15.0, help="zed_stream connect timeout")
    return p.parse_args()


def _get_aruco_dict(name: str) -> cv2.aruco.Dictionary:
    attr = getattr(cv2.aruco, name, None)
    if attr is None:
        raise SystemExit(f"Unknown ArUco dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(attr)


def _make_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    aruco_dict: cv2.aruco.Dictionary,
) -> cv2.aruco.CharucoBoard:
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            return cv2.aruco.CharucoBoard(
                (squares_x, squares_y),
                square_length_m,
                marker_length_m,
                aruco_dict,
            )
        except TypeError:
            pass
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_length_m,
            marker_length_m,
            aruco_dict,
        )
    raise SystemExit("OpenCV aruco CharucoBoard API not found — pip install opencv-contrib-python")


@dataclass
class CharucoDetectResult:
    corners: np.ndarray
    ids: np.ndarray
    vis_bgr: np.ndarray
    num_markers: int
    num_charuco: int
    board: cv2.aruco.CharucoBoard
    board_label: str


@dataclass
class DetectStatus:
    result: CharucoDetectResult | None
    num_markers: int
    num_charuco: int
    hint: str


def _aruco_detector_params() -> cv2.aruco.DetectorParameters:
    if not hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters_create()
    p = cv2.aruco.DetectorParameters()
    # More tolerant for JPEG / motion / small markers at distance.
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 43
    p.adaptiveThreshWinSizeStep = 4
    p.minMarkerPerimeterRate = 0.015
    p.maxMarkerPerimeterRate = 4.0
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


def _detect_charuco_on_board(
    gray: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    aruco_dict: cv2.aruco.Dictionary,
    det_params: cv2.aruco.DetectorParameters,
) -> tuple[int, int, np.ndarray | None, np.ndarray | None, list, np.ndarray | None]:
    """Returns marker_count, charuco_count, charuco_corners, charuco_ids, marker_corners, marker_ids."""
    if hasattr(cv2.aruco, "ArucoDetector"):
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, det_params)
        marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=det_params)

    n_markers = 0 if marker_ids is None else len(marker_ids)
    if n_markers == 0:
        return 0, 0, None, None, [], None

    if hasattr(cv2.aruco, "CharucoDetector"):
        try:
            charuco_detector = cv2.aruco.CharucoDetector(board, detectorParams=det_params)
        except TypeError:
            charuco_detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
    else:
        ok, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
        if not ok:
            charuco_corners, charuco_ids = None, None

    n_charuco = 0 if charuco_ids is None else len(charuco_ids)
    return n_markers, n_charuco, charuco_corners, charuco_ids, marker_corners, marker_ids


def _detect_charuco(
    bgr: np.ndarray,
    boards: list[tuple[str, cv2.aruco.CharucoBoard]],
    aruco_dict: cv2.aruco.Dictionary,
) -> DetectStatus:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    det_params = _aruco_detector_params()
    vis = bgr.copy()

    best: CharucoDetectResult | None = None
    best_markers = 0
    best_charuco = 0
    best_label = ""

    for label, board in boards:
        n_markers, n_charuco, ch_corners, ch_ids, marker_corners, marker_ids = _detect_charuco_on_board(
            gray, board, aruco_dict, det_params
        )
        if n_markers > best_markers:
            best_markers = n_markers
        if ch_ids is not None and n_charuco > best_charuco:
            best_charuco = n_charuco
            best_label = label
            vis = bgr.copy()
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
            cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids, (0, 255, 0))
            best = CharucoDetectResult(
                ch_corners, ch_ids, vis, n_markers, n_charuco, board, label
            )

    if best is not None and best.num_charuco >= 4:
        cv2.putText(
            vis,
            f"board {best_label}  markers={best.num_markers} charuco={best.num_charuco}",
            (8, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        return DetectStatus(best, best.num_markers, best.num_charuco, "OK")

    vis = bgr.copy()
    if best_markers > 0:
        hint = (
            f"ArUco markers={best_markers} but ChArUco corners={best_charuco} — "
            "wrong board size? try --swap-board-axes or move closer"
        )
    else:
        hint = "No ArUco markers — move closer, add light, show full board, reduce blur"
    return DetectStatus(None, best_markers, best_charuco, hint)


def _scale_intrinsics_to_size(
    camera_matrix: np.ndarray,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> np.ndarray:
    fw, fh = from_size
    tw, th = to_size
    sx, sy = tw / fw, th / fh
    K = np.asarray(camera_matrix, dtype=np.float64).copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


def _calibrate_charuco(
    all_corners: list[np.ndarray],
    all_ids: list[np.ndarray],
    board: cv2.aruco.CharucoBoard,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    w, h = image_size
    if hasattr(cv2.aruco, "calibrateCameraCharucoExtended"):
        out = cv2.aruco.calibrateCameraCharucoExtended(
            all_corners,
            all_ids,
            board,
            (w, h),
            None,
            None,
        )
        # OpenCV versions return 7 or 8+ values; first three are always rms, K, dist.
        rms, camera_matrix, dist_coeffs = out[0], out[1], out[2]
        return np.asarray(camera_matrix, dtype=np.float64), np.asarray(dist_coeffs, dtype=np.float64), float(rms)

    out = cv2.aruco.calibrateCameraCharuco(
        all_corners,
        all_ids,
        board,
        (w, h),
        None,
        None,
    )
    rms, camera_matrix, dist_coeffs = out[0], out[1], out[2]
    return np.asarray(camera_matrix, dtype=np.float64), np.asarray(dist_coeffs, dtype=np.float64), float(rms)


def _build_intrinsics_json(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    width: int,
    height: int,
    *,
    rms: float,
    num_views: int,
    board_meta: dict,
) -> dict:
    K = np.asarray(camera_matrix, dtype=np.float64)
    d = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    return {
        "camera_frame": CAMERA_FRAME,
        "width": int(width),
        "height": int(height),
        "K": K.tolist(),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "distortion_model": "plumb_bob",
        "dist_coeffs": d.tolist(),
        "source": "charuco_calibration",
        "calibration": {
            "method": "opencv_aruco_charuco",
            "reprojection_rms_px": rms,
            "num_views": num_views,
            "created_at_iso": datetime.now(timezone.utc).astimezone().isoformat(),
            "board": board_meta,
        },
    }


def _save_calibration(output: Path, intrinsics: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(intrinsics, f, indent=2)
        f.write("\n")
    K = np.asarray(intrinsics["K"], dtype=np.float64)
    np.save(output.with_name("K.npy"), K)
    print(f"Wrote {output}")
    print(f"Wrote {output.with_name('K.npy')}")


def _open_frame_source(args: argparse.Namespace):
    if args.camera >= 0:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Could not open webcam index {args.camera}")

        def read_rgb() -> np.ndarray | None:
            ok, bgr = cap.read()
            if not ok:
                return None
            bgr = cv2.resize(bgr, (args.width, args.height))
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        def close() -> None:
            cap.release()

        return read_rgb, close

    from camera.zed_stream_receiver import ZedStreamRgbdReceiver

    rx = ZedStreamRgbdReceiver(
        args.host,
        args.port,
        out_width=args.width,
        out_height=args.height,
        verbose=True,
    )
    rx.start(timeout=args.timeout)

    def read_rgb() -> np.ndarray | None:
        return rx.get_rgbd().rgb

    def close() -> None:
        rx.stop()

    return read_rgb, close


def main() -> None:
    args = _parse_args()
    if not hasattr(cv2, "aruco"):
        raise SystemExit(
            "cv2.aruco not found — install: pip install opencv-contrib-python\n"
            "(uninstall opencv-python first if both conflict)"
        )

    square_length_m = args.square_length_mm / 1000.0
    marker_length_m = args.marker_length_mm / 1000.0
    aruco_dict = _get_aruco_dict(args.aruco_dict)

    sx = args.squares_y if args.swap_board_axes else args.squares_x
    sy = args.squares_x if args.swap_board_axes else args.squares_y
    board_a = _make_charuco_board(sx, sy, square_length_m, marker_length_m, aruco_dict)
    boards: list[tuple[str, cv2.aruco.CharucoBoard]] = [(f"{sx}x{sy}", board_a)]
    if args.auto_board_axes and not args.swap_board_axes:
        bx, by = args.squares_y, args.squares_x
        board_b = _make_charuco_board(bx, by, square_length_m, marker_length_m, aruco_dict)
        if f"{bx}x{by}" != f"{sx}x{sy}":
            boards.append((f"{bx}x{by}", board_b))

    locked_board: cv2.aruco.CharucoBoard | None = None
    locked_label = ""
    board_meta_base = {
        "square_length_m": square_length_m,
        "marker_length_m": marker_length_m,
        "aruco_dict": args.aruco_dict,
        "vendor_note": "calib.io 9x14 20mm/15mm DICT_5X5",
    }

    read_rgb, close_source = _open_frame_source(args)
    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []

    detect_wh = (args.width, args.height)
    output_wh = (args.output_width, args.output_height)
    if detect_wh != output_wh:
        print(
            f"Detect at {detect_wh[0]}x{detect_wh[1]} (easier), "
            f"save intrinsics for capture {output_wh[0]}x{output_wh[1]}"
        )

    print(
        "ChArUco calibration — board flat on table, fill more of the image.\n"
        f"Trying layouts: {[b[0] for b in boards]}\n"
        "SPACE=add view | c=calibrate+exit | u=undo | q=quit (focus OpenCV window)"
    )

    def _run_calibration() -> bool:
        if locked_board is None:
            print("Capture at least one view first (SPACE)", flush=True)
            return False
        if len(all_corners) < args.min_views:
            print(
                f"Need at least {args.min_views} views, have {len(all_corners)}",
                flush=True,
            )
            return False
        print(f"Calibrating from {len(all_corners)} views...", flush=True)
        try:
            K, dist, rms = _calibrate_charuco(
                all_corners,
                all_ids,
                locked_board,
                detect_wh,
            )
        except Exception as exc:
            print(f"Calibration failed: {exc}", flush=True)
            return False
        if output_wh != detect_wh:
            K = _scale_intrinsics_to_size(K, detect_wh, output_wh)
        ow, oh = output_wh
        parts = locked_label.split("x")
        board_meta = {
            **board_meta_base,
            "squares_x": int(parts[0]),
            "squares_y": int(parts[1]),
            "detect_resolution": {"width": detect_wh[0], "height": detect_wh[1]},
        }
        intrinsics = _build_intrinsics_json(
            K,
            dist,
            ow,
            oh,
            rms=rms,
            num_views=len(all_corners),
            board_meta=board_meta,
        )
        _save_calibration(args.output, intrinsics)
        print(
            f"Calibration OK — RMS={rms:.3f} px (at {detect_wh[0]}x{detect_wh[1]})\n"
            f"  fx={intrinsics['fx']:.2f} fy={intrinsics['fy']:.2f} "
            f"cx={intrinsics['cx']:.2f} cy={intrinsics['cy']:.2f} "
            f"@ {ow}x{oh}\n"
            f"Saved to {args.output}\n"
            "capture_session.py will use this file automatically.\n"
            "  python demo/phase2/capture_session.py --object-name chips",
            flush=True,
        )
        return True

    try:
        while True:
            rgb = read_rgb()
            if rgb is None:
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            search_boards = (
                [(locked_label, locked_board)]
                if locked_board is not None
                else boards
            )
            status = _detect_charuco(bgr, search_boards, aruco_dict)
            det = status.result
            if det is not None:
                vis = det.vis_bgr
            else:
                vis = bgr.copy()
                cv2.putText(
                    vis,
                    status.hint,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                if status.num_markers > 0:
                    cv2.putText(
                        vis,
                        f"markers={status.num_markers} charuco={status.num_charuco}",
                        (8, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 165, 255),
                        1,
                        cv2.LINE_AA,
                    )

            lock_line = locked_label or "auto"
            cv2.putText(
                vis,
                f"views={len(all_corners)} (need>={args.min_views})  layout={lock_line}",
                (8, vis.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("ChArUco calibration", vis)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" ") and det is not None:
                if locked_board is None:
                    locked_board = det.board
                    locked_label = det.board_label
                    print(f"Locked board layout: {locked_label}", flush=True)
                all_corners.append(det.corners)
                all_ids.append(det.ids)
                print(
                    f"Captured view {len(all_corners)} ({len(det.ids)} charuco corners)",
                    flush=True,
                )
            elif key in (ord("u"), ord("U")) and all_corners:
                all_corners.pop()
                all_ids.pop()
                if not all_corners:
                    locked_board = None
                    locked_label = ""
                print(f"Removed last view — {len(all_corners)} remaining", flush=True)
            elif key in (ord("c"), ord("C")):
                if _run_calibration():
                    break
    finally:
        cv2.destroyAllWindows()
        close_source()


if __name__ == "__main__":
    main()
