"""Dexcontrol head ZED camera connect helpers (pip + vendored dexcontrol)."""

from __future__ import annotations

import os
import time

from loguru import logger


def robot_configs_with_head_camera(*, camera_transport: str = "zenoh"):
    """Return dexcontrol Robot configs with head ZED RGB+depth enabled."""
    os.environ.setdefault("DEXCONTROL_DISABLE_HEARTBEAT", "1")

    try:
        from dexcontrol.core.config import get_robot_config

        configs = get_robot_config()
        if hasattr(configs, "enable_sensor"):
            configs.enable_sensor("head_camera")
        head_cam = configs.sensors.get("head_camera") if hasattr(configs, "sensors") else None
        if head_cam is not None:
            head_cam.enabled = True
            if hasattr(head_cam, "transport"):
                head_cam.transport = camera_transport
            if hasattr(head_cam, "use_rtc"):
                head_cam.use_rtc = camera_transport == "rtc"
            if hasattr(head_cam, "enable_rgb"):
                head_cam.enable_rgb = True
            if hasattr(head_cam, "enable_depth"):
                head_cam.enable_depth = True
        if hasattr(configs, "components") and "heartbeat" in configs.components:
            configs.components["heartbeat"].enabled = False
        return configs
    except ImportError:
        pass

    try:
        from dexcontrol.config.vega import get_vega_config

        configs = get_vega_config()
        configs.sensors.head_camera.enable = True
        configs.sensors.head_camera.use_rtc = camera_transport == "rtc"
        configs.heartbeat.enabled = False
        return configs
    except ImportError:
        pass

    logger.warning(
        "Could not build head-camera Robot config (missing dexcontrol.core.config / "
        "dexcontrol.config.vega); using Robot() defaults — head camera may stay off"
    )
    return None


def head_camera_env_hint() -> str:
    from dexcontrol.utils.constants import COMM_CFG_PATH_ENV_VAR, ROBOT_NAME_ENV_VAR

    zenoh = os.getenv(COMM_CFG_PATH_ENV_VAR, "(unset)")
    robot_name = os.getenv(ROBOT_NAME_ENV_VAR, "(unset)")
    robot_ip = os.getenv("ROBOT_IP", "(unset)")
    return (
        f"  {COMM_CFG_PATH_ENV_VAR}={zenoh}\n"
        f"  {ROBOT_NAME_ENV_VAR}={robot_name}\n"
        f"  ROBOT_IP={robot_ip}\n"
        "  Sanity: python camera/view_head_camera.py\n"
        "  Robot dexsensor head camera must be publishing on Zenoh (ping robot, port 7447)."
    )


def describe_head_camera_streams(head_cam) -> str:
    available = getattr(head_cam, "available_streams", [])
    active = getattr(head_cam, "active_streams", [])
    lines = [f"available={available}", f"active={active}"]
    streams = getattr(head_cam, "_streams", {})
    for name, stream in streams.items():
        if stream is None:
            lines.append(f"  {name}: disabled")
            continue
        transport = getattr(stream, "transport", None)
        t = transport.value if transport is not None else "?"
        topic = getattr(stream, "_subscriber", None)
        topic_str = getattr(topic, "topic", "?") if topic is not None else "?"
        lines.append(f"  {name}: transport={t} active={stream.is_active()} topic={topic_str}")
    return "\n".join(lines)


def wait_for_head_camera(
    robot,
    *,
    timeout: float = 30.0,
    required_streams: tuple[str, ...] = ("left_rgb", "depth"),
) -> None:
    """Block until streams are live; raise with diagnostics on timeout."""
    head_cam = robot.sensors.head_camera
    logger.info(
        f"Waiting up to {timeout:.0f}s for head camera streams {required_streams}..."
    )

    deadline = time.time() + timeout
    last_log = 0.0
    while time.time() < deadline:
        missing = [s for s in required_streams if not head_cam.is_stream_active(s)]
        if not missing:
            logger.info(f"Head camera ready: {list(required_streams)}")
            return

        now = time.time()
        if now - last_log >= 5.0:
            logger.warning(
                f"Still waiting for head camera streams: {missing}\n"
                f"{describe_head_camera_streams(head_cam)}"
            )
            last_log = now
        time.sleep(0.1)

    raise RuntimeError(
        "Head camera streams did not become active within "
        f"{timeout:.0f} s (missing: "
        f"{[s for s in required_streams if not head_cam.is_stream_active(s)]}).\n"
        f"{describe_head_camera_streams(head_cam)}\n"
        "Check Zenoh / robot sensor stack:\n"
        f"{head_camera_env_hint()}"
    )


def create_robot_with_head_camera(*, camera_transport: str = "zenoh"):
    """Initialize dexcontrol Robot with head ZED RGB+depth enabled (no hands)."""
    from dexcontrol.robot import Robot

    configs = robot_configs_with_head_camera(camera_transport=camera_transport)
    return Robot(configs=configs) if configs is not None else Robot()
