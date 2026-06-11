"""Robot + Sharpa hand connection helpers for V2AP demo."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

# Project root on sys.path for teleop imports.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.constants import (  # noqa: E402
    HAND_INTERPOLATE,
    LEFT_HAND_SERIAL,
    RIGHT_HAND_SERIAL,
)
from demo.virtual_gripper import validate_hand_q  # noqa: E402


@dataclass
class HardwareHandles:
    robot: object
    left_hand: object
    right_hand: object
    hand_manager: object


def connect_right_hand_only(
    right_serial: str = RIGHT_HAND_SERIAL,
    discovery_wait_s: float = 3.0,
) -> tuple[object, object]:
    """Connect right Sharpa hand only (no Dexmate robot). Returns (right_hand, manager)."""
    from sharpa import SharpaWaveManager

    manager = SharpaWaveManager.get_instance()
    time.sleep(discovery_wait_s)
    devices = manager.get_all_device_sn()
    logger.info(f"Available hand devices: {devices}")
    if not devices:
        raise RuntimeError("No hand devices found")
    if right_serial not in devices:
        raise RuntimeError(f"Right hand serial {right_serial} not found in {devices}")

    right_hand = manager.connect(right_serial)
    logger.info(f"Connected right hand: {right_serial}")
    initialize_hand(right_hand, "RIGHT")
    right_hand.start()
    logger.info("Right hand started")
    return right_hand, manager


def shutdown_right_hand(right_hand: object | None, manager: object | None = None) -> None:
    if right_hand is None:
        return
    from sharpa import SharpaWaveManager

    logger.info("Shutting down right hand...")
    try:
        right_hand.stop()
    except Exception as e:
        logger.warning(f"Hand stop error: {e}")
    try:
        if manager is not None:
            manager.disconnect_all()
        else:
            SharpaWaveManager.get_instance().disconnect_all()
    except Exception as e:
        logger.warning(f"Hand disconnect error: {e}")
    logger.info("Right hand shutdown complete")


def connect_hands(
    left_serial: str = LEFT_HAND_SERIAL,
    right_serial: str = RIGHT_HAND_SERIAL,
    discovery_wait_s: float = 3.0,
) -> tuple[object, object, object]:
    from sharpa import ControlMode, ControlSource, SharpaWave, SharpaWaveManager

    manager = SharpaWaveManager.get_instance()
    time.sleep(discovery_wait_s)
    devices = manager.get_all_device_sn()
    logger.info(f"Available hand devices: {devices}")
    if not devices:
        raise RuntimeError("No hand devices found")

    if left_serial not in devices:
        raise RuntimeError(f"Left hand serial {left_serial} not found in {devices}")
    if right_serial not in devices:
        raise RuntimeError(f"Right hand serial {right_serial} not found in {devices}")

    left_hand = manager.connect(left_serial)
    right_hand = manager.connect(right_serial)
    logger.info(f"Connected left hand: {left_serial}")
    logger.info(f"Connected right hand: {right_serial}")

    initialize_hand(left_hand, "LEFT")
    initialize_hand(right_hand, "RIGHT")
    left_hand.start()
    right_hand.start()
    logger.info("Both hands started")
    return left_hand, right_hand, manager


def initialize_hand(hand, name: str) -> None:
    from sharpa import ControlMode, ControlSource, SharpaWave

    for i in range(10):
        if hand.is_hand_ready():
            break
        logger.warning(f"[{name}] Hand not ready, attempt {i + 1}/10...")
        time.sleep(0.5)
    else:
        raise RuntimeError(f"[{name}] Hand never became ready")

    for attempt in range(5):
        error = hand.set_control_mode(ControlMode.POSITION)
        if error.code == 0:
            break
        logger.warning(f"[{name}] set_control_mode failed ({attempt + 1}/5): {error.message}")
        time.sleep(2)
    else:
        raise RuntimeError(f"[{name}] Failed to set control mode: {error.message}")

    for coeff, setter, label in [
        (0.3, hand.set_speed_coeff, "speed"),
        (0.6, hand.set_current_coeff, "current"),
    ]:
        error = setter(coeff)
        if error.code != 0:
            raise RuntimeError(f"[{name}] Failed to set {label} coeff: {error.message}")

    error = hand.set_control_source(ControlSource.SDK)
    if error.code != 0:
        raise RuntimeError(f"[{name}] Failed to set control source: {error.message}")


def setup_cpp_logging(log_path: str = "/tmp/sharpa_wave.log", console_log: bool = False) -> None:
    from sharpa import setup_cpp_logging as _setup

    _setup(log_path, console_log=console_log)


def connect_robot_and_hands(dry_run: bool = False) -> HardwareHandles | None:
    if dry_run:
        logger.info("dry_run=True: skipping robot and hand connection")
        return None

    setup_cpp_logging(console_log=False)
    from dexcontrol.robot import Robot

    robot = Robot()
    logger.info(f"Dexmate robot initialized: {robot.robot_model}")
    left_hand, right_hand, manager = connect_hands()
    return HardwareHandles(robot=robot, left_hand=left_hand, right_hand=right_hand, hand_manager=manager)


def shutdown_hardware(handles: HardwareHandles | None) -> None:
    if handles is None:
        return
    from sharpa import SharpaWaveManager

    logger.info("Shutting down hardware...")
    # Stop Sharpa hands, disconnect SDK, then tear down dexcontrol/zenoh.
    # Order matters: mixed Sharpa + dexcomm often double-frees during process exit.
    try:
        if handles.left_hand is not None:
            handles.left_hand.stop()
        if handles.right_hand is not None:
            handles.right_hand.stop()
    except Exception as e:
        logger.warning(f"Hand stop error: {e}")
    try:
        SharpaWaveManager.get_instance().disconnect_all()
    except Exception as e:
        logger.warning(f"Hand disconnect error: {e}")
    handles.left_hand = None
    handles.right_hand = None
    handles.hand_manager = None
    try:
        if handles.robot is not None:
            handles.robot.shutdown()
    except Exception as e:
        logger.warning(f"Robot shutdown error: {e}")
    handles.robot = None
    logger.info("Hardware shutdown complete")


def set_hand_joint_positions(hand, q, interpolate: bool = HAND_INTERPOLATE) -> None:
    q = validate_hand_q(q)
    error = hand.set_joint_position(q.tolist(), interpolate)
    if error.code != 0:
        raise RuntimeError(f"set_joint_position failed: {error.message}")
