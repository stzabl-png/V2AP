"""Capture RGB-D and robot state from Dexmate head camera."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from demo.hardware import HardwareHandles, shutdown_hardware
from demo.head_camera import (
    create_robot_with_head_camera,
    robot_configs_with_head_camera,
    wait_for_head_camera,
)
from demo.phase2.constants import DEFAULT_CAPTURE_COMPONENTS


@dataclass
class CaptureFrame:
    rgb: np.ndarray  # (H,W,3) uint8 RGB
    depth: np.ndarray  # (H,W) float32 meters
    joint_pos_dict: dict[str, float]
    robot_model: str
    camera_info: dict[str, Any] | None
    left_hand_joint_pos: np.ndarray | None = None
    right_hand_joint_pos: np.ndarray | None = None
    capture_pose_source: str | None = None  # e.g. "start.yaml"


def _obs_array(obs: Any) -> np.ndarray | None:
    if obs is None:
        return None
    if isinstance(obs, dict) and "data" in obs:
        return obs["data"]
    return obs


def _align_depth_to_rgb(depth: np.ndarray, rgb_shape: tuple[int, int]) -> np.ndarray:
    if depth.shape[:2] == rgb_shape:
        return depth.astype(np.float32, copy=False)
    try:
        import cv2  # noqa: WPS433
    except ImportError as e:
        raise RuntimeError("opencv-python required to resize depth to RGB resolution") from e
    return cv2.resize(
        depth.astype(np.float32),
        (rgb_shape[1], rgb_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)


def create_robot_for_capture(
    *,
    camera_source: str = "zed_stream",
    camera_transport: str = "zenoh",
    camera_timeout: float = 30.0,
) -> HardwareHandles:
    """Robot + both Sharpa hands; optional dexsensor head camera (zenoh) or arms-only (zed_stream)."""
    from dexcontrol.robot import Robot

    from demo.hardware import connect_hands, setup_cpp_logging

    setup_cpp_logging(console_log=False)
    if camera_source == "zed_stream":
        robot = Robot(auto_shutdown=False)
        logger.info(
            f"Dexmate robot initialized: {robot.robot_model} "
            "(camera via zed_stream TCP, not dexsensor)"
        )
    else:
        configs = robot_configs_with_head_camera(camera_transport=camera_transport)
        robot = (
            Robot(configs=configs, auto_shutdown=False)
            if configs is not None
            else Robot(auto_shutdown=False)
        )
        logger.info(f"Dexmate robot initialized: {robot.robot_model}")
        # Wait for RGB-D before Sharpa hand SDK init (avoids long hand setup then camera fail).
        wait_for_head_camera(robot, timeout=camera_timeout)

    left_hand, right_hand, manager = connect_hands()
    return HardwareHandles(
        robot=robot,
        left_hand=left_hand,
        right_hand=right_hand,
        hand_manager=manager,
    )


def read_joint_positions(robot, components: tuple[str, ...] = DEFAULT_CAPTURE_COMPONENTS) -> dict[str, float]:
    return robot.get_joint_pos_dict(component=list(components))


def capture_head_rgbd(
    robot,
    *,
    max_attempts: int = 30,
    wait_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Grab one synchronized left_rgb + depth frame from head camera."""
    import time

    head_cam = robot.sensors.head_camera
    last_err = "no frames"

    for attempt in range(max_attempts):
        obs = head_cam.get_obs(obs_keys=["left_rgb", "depth"], include_timestamp=True)
        rgb = _obs_array(obs.get("left_rgb"))
        depth = _obs_array(obs.get("depth"))

        if rgb is None or depth is None:
            last_err = f"attempt {attempt + 1}: missing rgb or depth"
            time.sleep(wait_s)
            continue

        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]

        depth = np.asarray(depth, dtype=np.float32)
        depth = _align_depth_to_rgb(depth, rgb.shape[:2])

        valid = np.isfinite(depth) & (depth > 0)
        if valid.sum() < 100:
            last_err = f"attempt {attempt + 1}: too few valid depth pixels ({valid.sum()})"
            time.sleep(wait_s)
            continue

        return rgb, depth

    raise RuntimeError(f"Failed to capture RGB-D after {max_attempts} attempts: {last_err}")


def capture_from_robot(
    robot,
    *,
    left_hand=None,
    right_hand=None,
    capture_pose_source: str | None = None,
    camera_timeout: float = 10.0,
    skip_camera_wait: bool = False,
    zed_receiver=None,
) -> CaptureFrame:
    camera_info = None
    if zed_receiver is not None:
        rgbd = zed_receiver.get_rgbd()
        rgb, depth = rgbd.rgb, rgbd.depth
        logger.info(
            f"Captured RGB-D from zed_stream ({rgb.shape[1]}x{rgb.shape[0]}), "
            f"valid depth pixels: {int((np.isfinite(depth) & (depth > 0)).sum())}"
        )
    else:
        if not skip_camera_wait:
            wait_for_head_camera(robot, timeout=camera_timeout)

        rgb, depth = capture_head_rgbd(robot)
        try:
            camera_info = robot.sensors.head_camera.get_camera_info(force_refresh=True)
        except Exception as e:
            logger.warning(f"get_camera_info failed: {e}")

    joint_pos_dict = read_joint_positions(robot)
    left_hand_q = None
    right_hand_q = None
    if left_hand is not None and right_hand is not None:
        from demo.hand_close import read_hand_joint_pos

        left_hand_q = read_hand_joint_pos(left_hand)
        right_hand_q = read_hand_joint_pos(right_hand)

    model = getattr(robot, "robot_model", "vega_1")
    return CaptureFrame(
        rgb=rgb,
        depth=depth,
        joint_pos_dict=joint_pos_dict,
        robot_model=str(model),
        camera_info=camera_info,
        left_hand_joint_pos=left_hand_q,
        right_hand_joint_pos=right_hand_q,
        capture_pose_source=capture_pose_source,
    )


def synthetic_capture_frame(
    width: int = 640,
    height: int = 360,
    *,
    table_depth_m: float = 0.75,
) -> CaptureFrame:
    """Off-robot placeholder frame for testing pack/validate (not for Titan inference)."""
    yy, xx = np.mgrid[0:height, 0:width]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = (xx * 255 // max(width - 1, 1)).astype(np.uint8)
    rgb[..., 1] = (yy * 255 // max(height - 1, 1)).astype(np.uint8)
    rgb[..., 2] = 128

    depth = np.full((height, width), table_depth_m, dtype=np.float32)
    depth[: height // 3, :] = np.nan
    cx, cy = width // 2, height // 2
    Y, X = np.ogrid[:height, :width]
    obj_mask = (X - cx) ** 2 + (Y - cy) ** 2 < (min(width, height) // 6) ** 2
    depth[obj_mask] = table_depth_m - 0.08

    from demo.phase1.constants import DEFAULT_JOINT_POS
    from teleop.robot_descriptions import DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES

    from demo.phase1.config_io import load_hand_profile
    from demo.phase1.constants import DEFAULT_START_OBJECT_NAME, DEMO_HEAD_JOINT_POS
    from demo.virtual_gripper import left_hand_neutral

    joint_pos_dict: dict[str, float] = {}
    for comp in ("torso", "head", "left_arm", "right_arm"):
        names = DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES[comp]
        vals = DEFAULT_JOINT_POS[comp]
        if comp == "head":
            vals = DEMO_HEAD_JOINT_POS
        for n, v in zip(names, vals):
            joint_pos_dict[n] = float(v)

    try:
        profile = load_hand_profile()
        right_open = profile["open"]
    except Exception:
        from demo.virtual_gripper import HAND_PINCH_OPEN

        right_open = HAND_PINCH_OPEN

    return CaptureFrame(
        rgb=rgb,
        depth=depth,
        joint_pos_dict=joint_pos_dict,
        robot_model="vega_1_dry_run",
        camera_info=None,
        left_hand_joint_pos=left_hand_neutral(),
        right_hand_joint_pos=np.asarray(right_open, dtype=np.float64),
        capture_pose_source=f"dry_run_defaults (head={DEFAULT_START_OBJECT_NAME} equivalent)",
    )
