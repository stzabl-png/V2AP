"""Plan arm moves ahead of operator prompts (demo-only, no teleop changes)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
from loguru import logger
from loop_rate_limiters import RateLimiter

from teleop.robot_descriptions import DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES


@dataclass
class PlannedJointMove:
    """Collision-planned left-then-right arm trajectory plus torso/head goals."""

    label: str
    torso_head_goal: dict[str, np.ndarray]
    arm_goal: dict[str, np.ndarray]
    left_arm_waypoints: list[dict[str, np.ndarray]]
    right_arm_waypoints: list[dict[str, np.ndarray]]


class BackgroundPlan:
    """Runs a planner callable on a daemon thread while the operator reads the prompt."""

    def __init__(self, label: str, plan_fn: Callable[[], PlannedJointMove]) -> None:
        self.label = label
        self._plan_fn = plan_fn
        self._thread: threading.Thread | None = None
        self._result: PlannedJointMove | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"Background plan already started: {self.label}")

        def _worker() -> None:
            try:
                self._result = self._plan_fn()
            except BaseException as exc:
                self._error = exc

        self._thread = threading.Thread(target=_worker, name=f"prefetch_{self.label}", daemon=True)
        self._thread.start()
        logger.info(f"Background planning started: {self.label}")

    def wait(self) -> PlannedJointMove:
        assert self._thread is not None
        self._thread.join()
        if self._error is not None:
            raise self._error
        assert self._result is not None
        logger.info(f"Background planning finished: {self.label}")
        return self._result


def wait_enter_with_prefetch(
    prompt: str,
    prefetch: BackgroundPlan | None,
    *,
    prefetch_already_started: bool = False,
) -> PlannedJointMove | None:
    """
    Show prompt immediately; planning may already be running.

    Blocks until the operator presses Enter and any background plan has completed.
    """
    if prefetch is not None:
        if prefetch_already_started:
            logger.info("Planning already in progress — press Enter when ready to execute.")
        else:
            logger.info("Planning in parallel — press Enter when ready to execute.")
            prefetch.start()
    input(prompt)
    if prefetch is None:
        return None
    return prefetch.wait()


def _read_arm_hand_state(hardware_lock, robot, left_hand, right_hand) -> dict[str, np.ndarray]:
    from teleop.robot_descriptions import DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES

    with hardware_lock:
        curr = robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
        left_hand_q = np.array(left_hand.get_states().angles, dtype=np.float64)
        right_hand_q = np.array(right_hand.get_states().angles, dtype=np.float64)
    return {
        "left_arm": np.array([curr[n] for n in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]]),
        "right_arm": np.array([curr[n] for n in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]]),
        "left_hand": left_hand_q,
        "right_hand": right_hand_q,
    }


def plan_joint_targets_move(
    *,
    label: str,
    planner,
    hardware_lock,
    robot,
    left_hand,
    right_hand,
    target: dict[str, np.ndarray],
    head_goal: np.ndarray,
    torso_goal: np.ndarray,
    right_arm_collision_boxes: list[dict] | None = None,
) -> PlannedJointMove:
    """OMPL plan for left arm then right arm (matches move_robot_to_position_safe)."""
    logger.info(f"Planning collision-safe move: {label}...")
    state = _read_arm_hand_state(hardware_lock, robot, left_hand, right_hand)

    # Disabled components must match between start and goal (teleop/arm_hand_control.plan).
    # Only the enabled arm moves per stage; hands stay at current until execute hold phase.
    left_waypoints = planner.plan(
        start_joint_pos=state,
        goal_joint_pos={
            "left_arm": target["left_arm"],
            "right_arm": state["right_arm"],
            "left_hand": state["left_hand"],
            "right_hand": state["right_hand"],
        },
        enabled_components={"left_arm"},
    )

    state = _read_arm_hand_state(hardware_lock, robot, left_hand, right_hand)
    if right_arm_collision_boxes:
        for box in right_arm_collision_boxes:
            planner.add_collision_box(
                name=box["name"],
                position=np.asarray(box["position"], dtype=np.float64),
                full_extents=np.asarray(box["full_extents"], dtype=np.float64),
            )
    try:
        right_waypoints = planner.plan(
            start_joint_pos=state,
            goal_joint_pos={
                "left_arm": state["left_arm"],
                "right_arm": target["right_arm"],
                "left_hand": state["left_hand"],
                "right_hand": state["right_hand"],
            },
            enabled_components={"right_arm"},
            skip_endpoint_collision_checks=bool(right_arm_collision_boxes),
        )
    finally:
        if right_arm_collision_boxes:
            for box in right_arm_collision_boxes:
                planner.remove_collision_box(box["name"])

    arm_goal = {
        "left_arm": np.asarray(target["left_arm"], dtype=np.float64).copy(),
        "right_arm": np.asarray(target["right_arm"], dtype=np.float64).copy(),
        "left_hand": np.asarray(target["left_hand"], dtype=np.float64).copy(),
        "right_hand": np.asarray(target["right_hand"], dtype=np.float64).copy(),
    }
    return PlannedJointMove(
        label=label,
        torso_head_goal={"head": head_goal.copy(), "torso": torso_goal.copy()},
        arm_goal=arm_goal,
        left_arm_waypoints=left_waypoints,
        right_arm_waypoints=right_waypoints,
    )


def execute_planned_joint_move(
    *,
    planned: PlannedJointMove,
    robot,
    left_hand,
    right_hand,
    action_buffer: dict[str, np.ndarray | None],
    action_buf_lock: threading.Lock,
    hardware_lock,
    command_hz: float,
    hold_time_s: float,
    smoothing_manager=None,
) -> None:
    """Stream pre-planned waypoints through the action buffer."""
    logger.info(f"Executing planned move: {planned.label}")
    limiter = RateLimiter(frequency=command_hz, name="planned_move_limiter", warn=True)
    hold_iters = int(hold_time_s * command_hz)

    with action_buf_lock:
        action_buffer["left_arm"] = None
        action_buffer["right_arm"] = None
        action_buffer["left_hand"] = None
        action_buffer["right_hand"] = None

    # Match teleop move_robot_to_position_safe: torso/head first, then arm waypoints.
    with hardware_lock:
        robot.set_joint_pos(planned.torso_head_goal, relative=False, wait_time=4)
    if smoothing_manager is not None:
        smoothing_manager.reset()

    for waypoint in planned.left_arm_waypoints:
        limiter.sleep()
        with action_buf_lock:
            action_buffer["left_arm"] = waypoint["left_arm"]
            action_buffer["right_arm"] = waypoint["right_arm"]
            action_buffer["left_hand"] = waypoint["left_hand"]
            action_buffer["right_hand"] = waypoint["right_hand"]

    for waypoint in planned.right_arm_waypoints:
        limiter.sleep()
        with action_buf_lock:
            action_buffer["left_arm"] = waypoint["left_arm"]
            action_buffer["right_arm"] = waypoint["right_arm"]
            action_buffer["left_hand"] = waypoint["left_hand"]
            action_buffer["right_hand"] = waypoint["right_hand"]

    last = planned.right_arm_waypoints[-1] if planned.right_arm_waypoints else planned.left_arm_waypoints[-1]
    for _ in range(hold_iters):
        limiter.sleep()
        with action_buf_lock:
            action_buffer["left_arm"] = last["left_arm"]
            action_buffer["right_arm"] = last["right_arm"]
            action_buffer["left_hand"] = last["left_hand"]
            action_buffer["right_hand"] = last["right_hand"]

    for _ in range(hold_iters):
        limiter.sleep()
        with action_buf_lock:
            action_buffer["left_arm"] = planned.arm_goal["left_arm"]
            action_buffer["right_arm"] = planned.arm_goal["right_arm"]
            action_buffer["left_hand"] = planned.arm_goal["left_hand"]
            action_buffer["right_hand"] = planned.arm_goal["right_hand"]

    with action_buf_lock:
        action_buffer["left_arm"] = None
        action_buffer["right_arm"] = None
        action_buffer["left_hand"] = None
        action_buffer["right_hand"] = None
