#!/usr/bin/env python3
"""
Capture a Phase 2 input session from Dexmate head RGB-D + robot state.

Writes demo/phase2/sessions/<session_id>/input/ per demo/phase2/README.md.

Usage:
  source setup_local.sh
  python demo/phase2/capture_session.py --object-name chips
  python demo/phase2/capture_session.py --object-name chips --preview
  python demo/phase2/capture_session.py --object-name chips --dry-run
  python demo/phase2/capture_session.py --object-name chips --camera-source zenoh  # dexsensor
  python demo/phase2/capture_session.py --validate-only --session-id 20260601_143022_chips
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase1.constants import DEFAULT_TABLE_HEIGHT_M  # noqa: E402
from demo.phase1.constants import DEFAULT_START_OBJECT_NAME  # noqa: E402
from demo.phase2.capture_pose import (  # noqa: E402
    load_start_config_for_capture,
    move_to_capture_start_pose,
    return_to_start_pose_after_capture,
)
from demo.phase2.constants import DEFAULT_HEAD_INTRINSICS_JSON, SESSIONS_DIR  # noqa: E402
from demo.phase2.extrinsics import compute_T_base_cam  # noqa: E402
from demo.phase2.intrinsics import build_intrinsics_json  # noqa: E402
from demo.phase2.pack_session import (  # noqa: E402
    make_session_id,
    session_input_dir,
    validate_and_report,
    write_session_input,
)
from demo.phase2.robot_capture import (  # noqa: E402
    capture_from_robot,
    create_robot_for_capture,
    synthetic_capture_frame,
)
from demo.hardware import shutdown_hardware  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture Phase 2 Titan input session (RGB-D + calib)")
    p.add_argument(
        "--object-name",
        type=str,
        help="Object slug (e.g. chips). Used in session_id if --session-id omitted.",
    )
    p.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Override session id (default: YYYYMMDD_HHMMSS_<object>)",
    )
    p.add_argument(
        "--sessions-dir",
        type=Path,
        default=SESSIONS_DIR,
        help=f"Sessions root (default: {SESSIONS_DIR})",
    )
    p.add_argument(
        "--table-height",
        type=float,
        default=DEFAULT_TABLE_HEIGHT_M,
        help="Table top height in base frame (meters)",
    )
    p.add_argument("--notes", type=str, default="", help="Free text stored in session.json")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="No robot: write synthetic rgb/depth for pack/validate testing only",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Show RGB + depth colormap in OpenCV before saving (requires display)",
    )
    p.add_argument(
        "--no-fallback-intrinsics",
        action="store_true",
        help="Fail if camera_info lacks fx/fy (do not use nominal ZED defaults)",
    )
    p.add_argument("--fx", type=float, default=None, help="Override focal length fx (pixels)")
    p.add_argument("--fy", type=float, default=None, help="Override focal length fy (pixels)")
    p.add_argument("--cx", type=float, default=None, help="Override principal point cx (pixels)")
    p.add_argument("--cy", type=float, default=None, help="Override principal point cy (pixels)")
    p.add_argument(
        "--intrinsics-json",
        type=Path,
        default=None,
        help=(
            "Load intrinsics.json (overrides default calib file). "
            f"Default if present: {DEFAULT_HEAD_INTRINSICS_JSON}"
        ),
    )
    p.add_argument(
        "--no-calib-intrinsics",
        action="store_true",
        help=f"Do not load {DEFAULT_HEAD_INTRINSICS_JSON} even if it exists",
    )
    p.add_argument(
        "--sam-point",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Optional SAM point prompt in pixel coords (top-left origin)",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing session input/; do not capture",
    )
    p.add_argument(
        "--start-object-name",
        type=str,
        default=DEFAULT_START_OBJECT_NAME,
        help=f"Phase 1 start pose YAML name (default: {DEFAULT_START_OBJECT_NAME})",
    )
    p.add_argument(
        "--skip-move-to-start",
        action="store_true",
        help="Do not move robot; capture whatever pose the robot is in now",
    )
    p.add_argument(
        "--arm-wait-s",
        type=float,
        default=10.0,
        help="Seconds to wait per arm move at half speed (default 10; was 5 at full speed)",
    )
    p.add_argument(
        "--capture-arm-j3-deg",
        type=float,
        default=0.0,
        help="Optional outward arm_j3 spread before capture (0 = shoot at start.yaml arms)",
    )
    p.add_argument(
        "--slow-capture-move",
        action="store_true",
        help="Use slow stepped arm moves instead of normal set_joint_pos speed",
    )
    p.add_argument(
        "--skip-return-to-start",
        action="store_true",
        help="After capture, do not move arms back to start.yaml",
    )
    p.add_argument(
        "--capture-arm-step-rad",
        type=float,
        default=0.008,
        help="Max per-joint step when --slow-capture-move (default 0.008 rad)",
    )
    p.add_argument(
        "--capture-arm-max-duration-s",
        type=float,
        default=120.0,
        help="Timeout for slow capture arm move (default 120)",
    )
    p.add_argument(
        "--camera-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for head RGB-D streams (default: 30)",
    )
    p.add_argument(
        "--camera-transport",
        choices=("zenoh", "rtc"),
        default="zenoh",
        help="Head camera RGB transport when --camera-source=zenoh (depth always zenoh)",
    )
    p.add_argument(
        "--camera-source",
        choices=("zenoh", "zed_stream"),
        default="zed_stream",
        help="zed_stream=TCP from dexmate-nano (default); zenoh=dexsensor head_camera",
    )
    p.add_argument(
        "--zed-stream-host",
        type=str,
        default=os.environ.get("CAMERA_IP", "192.168.50.22")  # set via CAMERA_IP env var,
        help="zed_streamer host when --camera-source=zed_stream (default: dexmate-nano)",
    )
    p.add_argument(
        "--zed-stream-port",
        type=int,
        default=30000,
        help="zed_streamer TCP port (default: 30000)",
    )
    return p.parse_args()


def _load_intrinsics_override(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_intrinsics_json_path(args: argparse.Namespace) -> Path | None:
    if args.intrinsics_json is not None:
        return Path(args.intrinsics_json)
    if args.no_calib_intrinsics:
        return None
    default_path = DEFAULT_HEAD_INTRINSICS_JSON
    if default_path.is_file():
        return default_path
    return None


def _optional_segment_prompt(args: argparse.Namespace) -> dict | None:
    if args.sam_point is None:
        return None
    x, y = args.sam_point
    return {
        "tool": "sam2",
        "prompts": [{"type": "point", "xy": [float(x), float(y)], "label": 1}],
    }


def _preview_rgb_depth(rgb, depth) -> None:
    import cv2  # noqa: WPS433
    import numpy as np

    d = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(d) & (d > 0)
    vis = np.zeros((*d.shape, 3), dtype=np.uint8)
    if np.any(valid):
        d_min = float(np.min(d[valid]))
        d_max = float(np.max(d[valid]))
        span = max(d_max - d_min, 1e-6)
        norm = np.zeros_like(d)
        norm[valid] = (d[valid] - d_min) / span
        vis[..., 0] = (norm * 255).astype(np.uint8)
        vis[..., 1] = (255 - norm * 255).astype(np.uint8).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("phase2_capture_rgb", bgr)
    cv2.imshow("phase2_capture_depth", vis)
    logger.info("Preview windows open — press any key to save, q to abort")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if key in (ord("q"), 27):
        raise SystemExit("Capture aborted from preview")


def main() -> None:
    args = _parse_args()

    if args.validate_only:
        if not args.session_id:
            raise SystemExit("--validate-only requires --session-id")
        input_dir = session_input_dir(args.session_id, args.sessions_dir)
        result = validate_and_report(input_dir)
        for w in result.warnings:
            logger.warning(w)
        for e in result.errors:
            logger.error(e)
        if result.ok:
            logger.info(f"Validation OK: {input_dir}")
        else:
            raise SystemExit(1)
        return

    if not args.object_name and not args.session_id:
        raise SystemExit("--object-name is required (or pass --session-id with embedded slug for dry-run)")

    object_slug = args.object_name or "object"
    session_id = args.session_id or make_session_id(object_slug)
    input_dir = session_input_dir(session_id, args.sessions_dir)

    if input_dir.exists() and any(input_dir.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite non-empty input dir: {input_dir}\n"
            "Remove it or choose a new --session-id"
        )

    robot = None
    handles = None
    zed_receiver = None
    capture_pose_source = None
    start_cfg = None
    hardware_used = False
    try:
        if args.dry_run:
            logger.warning("dry-run: synthetic RGB-D — NOT for Titan inference")
            frame = synthetic_capture_frame()
            capture_pose_source = "dry_run_defaults"
        else:
            if args.camera_source == "zed_stream":
                from camera.zed_stream_receiver import ZedStreamRgbdReceiver

                logger.info(
                    f"Camera: zed_stream tcp://{args.zed_stream_host}:{args.zed_stream_port} "
                    "(start zed_streamer on dexmate-nano without --no-depth)"
                )
                zed_receiver = ZedStreamRgbdReceiver(
                    args.zed_stream_host,
                    args.zed_stream_port,
                    verbose=True,
                )
                zed_receiver.start(timeout=args.camera_timeout)

            logger.info("Connecting to Dexmate (both hands)...")
            handles = create_robot_for_capture(
                camera_source=args.camera_source,
                camera_transport=args.camera_transport,
                camera_timeout=args.camera_timeout,
            )
            robot = handles.robot

            if not args.skip_move_to_start:
                start_cfg = load_start_config_for_capture(args.start_object_name)
                move_to_capture_start_pose(
                    handles,
                    start_cfg,
                    arm_wait_s=args.arm_wait_s,
                    j3_outward_deg=args.capture_arm_j3_deg,
                    arm_step_rad=args.capture_arm_step_rad,
                    arm_max_duration_s=args.capture_arm_max_duration_s,
                    use_slow_arm_move=args.slow_capture_move,
                )
                capture_pose_source = f"phase1/configs/{start_cfg.object_name}.yaml"
            else:
                logger.warning(
                    "--skip-move-to-start: capturing current pose (head/hands may NOT match Phase 1 demo)"
                )
                capture_pose_source = "current_robot_state"

            frame = capture_from_robot(
                robot,
                left_hand=handles.left_hand,
                right_hand=handles.right_hand,
                capture_pose_source=capture_pose_source,
                camera_timeout=min(args.camera_timeout, 15.0),
                skip_camera_wait=True,
                zed_receiver=zed_receiver,
            )

        if args.preview:
            _preview_rgb_depth(frame.rgb, frame.depth)

        h, w = frame.rgb.shape[:2]

        intrinsics_path = _resolve_intrinsics_json_path(args)
        if intrinsics_path is not None:
            intrinsics = _load_intrinsics_override(intrinsics_path)
            logger.info(f"Using intrinsics from {intrinsics_path}")
        else:
            intrinsics = build_intrinsics_json(
                w,
                h,
                frame.camera_info,
                fx=args.fx,
                fy=args.fy,
                cx=args.cx,
                cy=args.cy,
                allow_fallback=not args.no_fallback_intrinsics,
            )
            if intrinsics.get("source") == "cli_or_fallback" and frame.camera_info is None:
                logger.warning(
                    "Using fallback intrinsics (camera_info unavailable). "
                    f"Run camera/calibrate_charuco.py or place calib at {DEFAULT_HEAD_INTRINSICS_JSON}"
                )

        T_base_cam = compute_T_base_cam(frame.joint_pos_dict)
        logger.info(f"T_base_cam translation (m): {T_base_cam[:3, 3].round(4).tolist()}")

        write_session_input(
            input_dir,
            session_id=session_id,
            object_slug=object_slug,
            rgb=frame.rgb,
            depth=frame.depth,
            intrinsics=intrinsics,
            T_base_cam=T_base_cam,
            joint_pos_dict=frame.joint_pos_dict,
            robot_model=frame.robot_model,
            table_height_m=args.table_height,
            notes=args.notes + (" [dry-run synthetic]" if args.dry_run else ""),
            segment_prompt=_optional_segment_prompt(args),
            left_hand_joint_pos=frame.left_hand_joint_pos,
            right_hand_joint_pos=frame.right_hand_joint_pos,
            capture_pose_source=frame.capture_pose_source or capture_pose_source,
        )

        result = validate_and_report(input_dir)
        for w in result.warnings:
            logger.warning(w)
        for e in result.errors:
            logger.error(e)

        if not result.ok:
            raise SystemExit(f"Pack written but validation failed: {input_dir}")

        if (
            not args.dry_run
            and handles is not None
            and start_cfg is not None
            and not args.skip_move_to_start
            and not args.skip_return_to_start
        ):
            return_to_start_pose_after_capture(
                handles,
                start_cfg,
                arm_wait_s=args.arm_wait_s,
                arm_step_rad=args.capture_arm_step_rad,
                arm_max_duration_s=args.capture_arm_max_duration_s,
                use_slow_arm_move=args.slow_capture_move,
            )

        logger.info(f"Session input ready: {input_dir}")
        logger.info(f"session_id={session_id}")
        logger.info(
            "Rsync to Titan:\n"
            f"  rsync -avz --progress {input_dir}/ "
            f"TITAN_HOST:$TITAN_ROOT/demo/sessions/{session_id}/input/"
        )

    finally:
        if zed_receiver is not None:
            zed_receiver.stop()
            zed_receiver = None
        if handles is not None:
            shutdown_hardware(handles)
            handles = None
            hardware_used = True
        elif robot is not None:
            try:
                robot.shutdown()
            except Exception as e:
                logger.warning(f"Robot shutdown: {e}")
            robot = None

    # dexcontrol (Zenoh) + Sharpa C++ bindings can double-free during normal Python exit.
    if hardware_used:
        os._exit(0)


if __name__ == "__main__":
    main()
