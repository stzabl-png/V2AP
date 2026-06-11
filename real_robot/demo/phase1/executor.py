"""Execute Phase 1 pick-and-lift motion on Dexmate + Sharpa."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from loguru import logger
from loop_rate_limiters import RateLimiter

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.constants import HAND_INTERPOLATE
from demo.hand_close import close_hand_until_stall, move_hand_to_target
from demo.phase1.config_io import GraspObjectConfig, load_hand_profile
from demo.phase1.constants import (
    ARM_ACTION_HZ,
    COMMAND_HZ,
    DEFAULT_JOINT_POS,
    DEFAULT_TABLE_HEIGHT_M,
    DEMO_HEAD_JOINT_POS,
    HEAD_PITCH_DOWN_DEG,
    PLANNED_MOTION_JOINT_SPEED_RAD_S,
    RESET_DOF_ERR_TOL,
)
from demo.phase1.planned_motion import (
    BackgroundPlan,
    execute_planned_joint_move,
    plan_joint_targets_move,
    wait_enter_with_prefetch,
)
from demo.phase1.grasp_geometry import format_pose, homogeneous_to_se3, se3_to_homogeneous
from demo.virtual_gripper import validate_hand_q
from demo.hardware import HardwareHandles, set_hand_joint_positions, shutdown_hardware
from demo.virtual_gripper import left_hand_neutral

from teleop.arm_hand_control import (  # noqa: E402
    InitializationCollisionPlanner,
    SmoothingAndSafetyManager,
    full_robot_action_loop,
)
from teleop.ik_utils import PinkLocalIK  # noqa: E402
from teleop.robot_descriptions import build_full_robot, add_env_obstacles  # noqa: E402


class GraspExecutor:
    """Runs start → pre-grasp → grasp → close → lift → hold → start (when start.yaml is used)."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.handles: HardwareHandles | None = None
        self.pink_ik: PinkLocalIK | None = None
        self.initialization_planner: InitializationCollisionPlanner | None = None
        self.smoothing_manager: SmoothingAndSafetyManager | None = None
        self.action_buffer: dict[str, np.ndarray | None] | None = None
        self.action_buf_lock: threading.Lock | None = None
        self.hardware_lock: threading.Lock | None = None
        self.action_thread: threading.Thread | None = None
        self.action_terminate_event: threading.Event | None = None
        self.pin_full_robot = None
        self.assemble_qpos = None
        self.disassemble_qpos = None
        self._planner_table_height_m: float | None = None

    def setup(self) -> None:
        from demo.hardware import connect_robot_and_hands

        default_joint = {
            k: v.copy() for k, v in DEFAULT_JOINT_POS.items()
        }
        self.pink_ik = PinkLocalIK(default_joint_by_component=default_joint)

        if self.dry_run:
            logger.info("GraspExecutor ready (dry_run)")
            return

        self.handles = connect_robot_and_hands(dry_run=False)
        assert self.handles is not None

        self.pin_full_robot, self.assemble_qpos, self.disassemble_qpos = build_full_robot(
            default_joint_by_component=default_joint
        )
        self.smoothing_manager = SmoothingAndSafetyManager(
            pin_full_robot_wrapper=self.pin_full_robot,
            assemble_qpos=self.assemble_qpos,
            disassemble_qpos=self.disassemble_qpos,
            ruckig_smoothing=False,
            action_hz=ARM_ACTION_HZ,
        )

        self.action_buf_lock = threading.Lock()
        self.hardware_lock = threading.Lock()
        self.action_buffer = {
            "left_arm": None,
            "right_arm": None,
            "left_hand": None,
            "right_hand": None,
        }
        self.action_terminate_event = threading.Event()
        self.action_thread = threading.Thread(
            target=full_robot_action_loop,
            kwargs={
                "terminate_event": self.action_terminate_event,
                "action_buf_lock": self.action_buf_lock,
                "action_buffer": self.action_buffer,
                "hardware_lock": self.hardware_lock,
                "dexmate_bimanual_robot": self.handles.robot,
                "sharpa_left_hand": self.handles.left_hand,
                "sharpa_right_hand": self.handles.right_hand,
                "smoothing_and_safety_manager": self.smoothing_manager,
                "action_hz": ARM_ACTION_HZ,
                "hand_interpolate": HAND_INTERPOLATE,
            },
            daemon=True,
        )
        self.action_thread.start()
        logger.info(f"Action thread started at {ARM_ACTION_HZ} Hz")

    def capture_config_from_robot(
        self,
        object_name: str,
        table_height: float = DEFAULT_TABLE_HEIGHT_M,
    ) -> GraspObjectConfig:
        """Build a new grasp config from the robot's current state (for pose tuning)."""
        if self.dry_run or self.handles is None or self.pink_ik is None:
            raise RuntimeError("capture_config_from_robot requires live hardware (not dry_run)")

        from teleop.robot_descriptions import DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES

        with self.hardware_lock:
            joint_d = self.handles.robot.get_joint_pos_dict(
                component=["left_arm", "right_arm", "torso", "head"]
            )
            left_hand_q = np.array(self.handles.left_hand.get_states().angles, dtype=np.float64)
            right_hand_q = np.array(self.handles.right_hand.get_states().angles, dtype=np.float64)

        left_q = np.array([joint_d[n] for n in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]])
        right_q = np.array([joint_d[n] for n in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]])
        torso_q = np.array([joint_d[n] for n in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["torso"]])
        head_q = np.array([joint_d[n] for n in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["head"]])

        fk = self.pink_ik.fk(
            frames=["R_ee"],
            joint_pos_by_component={"left_arm": left_q, "right_arm": right_q},
        )
        grasp_pose = se3_to_homogeneous(fk["R_ee"])

        hand_open, hand_closed = load_hand_profile()
        return GraspObjectConfig(
            object_name=object_name,
            table_height=table_height,
            grasp_pose=grasp_pose,
            left_arm_joint_pos=left_q.copy(),
            right_arm_joint_pos=right_q.copy(),
            torso_joint_pos=torso_q.copy(),
            head_joint_pos=head_q.copy(),
            left_hand_joint_pos=validate_hand_q(left_hand_q, "left_hand_joint_pos"),
            hand_open_joint_pos=hand_open,
            hand_closed_joint_pos=hand_closed,
            notes=f"Captured at tune session start for {object_name}.",
        )

    def shutdown(self) -> None:
        if not self.dry_run and self.action_terminate_event is not None:
            if self.action_terminate_event.is_set():
                logger.error(
                    "[GraspExecutor] Action thread already stopped (often arm tracking "
                    "mismatch or collision during execute — check logs above for "
                    "'position mismatch' or planning RuntimeError)"
                )
            with self.action_buf_lock:
                for k in self.action_buffer:
                    self.action_buffer[k] = None
            self.action_terminate_event.set()
            if self.action_thread is not None:
                self.action_thread.join(timeout=2.0)
        shutdown_hardware(self.handles)
        self.handles = None

    def _planner(self, table_height: float) -> InitializationCollisionPlanner:
        table_height = float(table_height)
        if (
            self.initialization_planner is not None
            and self._planner_table_height_m is not None
            and abs(self._planner_table_height_m - table_height) < 1e-6
        ):
            return self.initialization_planner

        default_joint = {k: v.copy() for k, v in DEFAULT_JOINT_POS.items()}
        pin_full, assemble, disassemble = build_full_robot(default_joint_by_component=default_joint)
        pin_full = add_env_obstacles(
            robot=pin_full,
            default_joint_by_component={
                "left_arm": DEFAULT_JOINT_POS["left_arm"],
                "right_arm": DEFAULT_JOINT_POS["right_arm"],
                "left_hand": left_hand_neutral(),
                "right_hand": left_hand_neutral(),
            },
            assemble_qpos=assemble,
            back_wall_distance=None,
            left_wall_distance=None,
            right_wall_distance=None,
            table_height=table_height,
        )
        logger.info(
            f"Collision planner: table obstacle enabled (top z≈{table_height:.3f} m, 2.0×4.0 m box)"
        )
        self.initialization_planner = InitializationCollisionPlanner(
            pin_full_robot_wrapper=pin_full,
            assemble_qpos=assemble,
            disassemble_qpos=disassemble,
            max_edge_joint_step=PLANNED_MOTION_JOINT_SPEED_RAD_S / COMMAND_HZ,
            plan_timeout_s=10.0,
            solve_step_s=0.1,
        )
        self._planner_table_height_m = table_height
        return self.initialization_planner

    def joint_targets_table_collision_free(
        self,
        targets: dict[str, np.ndarray],
        *,
        table_height: float,
    ) -> bool:
        """Pinocchio check: arms + hands vs table (same model as OMPL execution)."""
        planner = self._planner(table_height)
        joint_pos = {
            "left_arm": np.asarray(targets["left_arm"], dtype=np.float64),
            "right_arm": np.asarray(targets["right_arm"], dtype=np.float64),
            "left_hand": np.asarray(
                targets.get("left_hand", left_hand_neutral()),
                dtype=np.float64,
            ),
            "right_hand": np.asarray(
                targets.get("right_hand", left_hand_neutral()),
                dtype=np.float64,
            ),
        }
        return planner.is_joint_targets_collision_free(joint_pos)

    def apply_live_targets(
        self,
        config: GraspObjectConfig,
        *,
        right_arm: np.ndarray | None = None,
        right_hand: np.ndarray | None = None,
    ) -> None:
        """Send joint targets via action buffer (no collision planner). For real-time tuning."""
        if self.dry_run:
            logger.info("[dry_run] apply_live_targets")
            return
        assert self.action_buffer is not None
        assert self.action_buf_lock is not None
        targets = {
            "left_arm": config.left_arm_joint_pos.copy(),
            "right_arm": (
                config.right_arm_joint_pos.copy()
                if right_arm is None
                else np.asarray(right_arm, dtype=np.float64).copy()
            ),
            "left_hand": config.left_hand_joint_pos.copy(),
            "right_hand": (
                config.hand_open_joint_pos.copy()
                if right_hand is None
                else np.asarray(right_hand, dtype=np.float64).copy()
            ),
        }
        self._publish_action_buffer(targets)

    def _publish_action_buffer(self, targets: dict[str, np.ndarray]) -> None:
        """Keep the action thread streaming these targets (avoids hand snap-back to zero)."""
        if self.dry_run or self.action_buffer is None or self.action_buf_lock is None:
            return
        with self.action_buf_lock:
            for key in ("left_arm", "right_arm", "left_hand", "right_hand"):
                if key in targets:
                    self.action_buffer[key] = np.asarray(targets[key], dtype=np.float64).copy()

    def sync_grasp_pose_from_right_arm(self, config: GraspObjectConfig) -> None:
        """Update grasp_pose FK from config left/right arm joints (for YAML / pre-grasp math)."""
        assert self.pink_ik is not None
        fk = self.pink_ik.fk(
            frames=["R_ee"],
            joint_pos_by_component={
                "left_arm": config.left_arm_joint_pos,
                "right_arm": config.right_arm_joint_pos,
            },
        )
        config.grasp_pose = se3_to_homogeneous(fk["R_ee"])

    def _read_arm_joint_pos(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.handles is not None
        from teleop.robot_descriptions import DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES

        with self.hardware_lock:
            d = self.handles.robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
        left_q = np.array([d[n] for n in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]])
        right_q = np.array([d[n] for n in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]])
        return left_q, right_q

    def _read_right_hand_q(self) -> np.ndarray:
        assert self.handles is not None
        with self.hardware_lock:
            return np.array(self.handles.right_hand.get_states().angles, dtype=np.float64)

    @staticmethod
    def joint_targets_with_fixed_hand(
        pose_config: GraspObjectConfig,
        right_hand_q: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Arm goals from pose_config; right hand held at ``right_hand_q`` during arm move."""
        return {
            "left_arm": pose_config.left_arm_joint_pos.copy(),
            "right_arm": pose_config.right_arm_joint_pos.copy(),
            "left_hand": pose_config.left_hand_joint_pos.copy(),
            "right_hand": np.asarray(right_hand_q, dtype=np.float64).copy(),
        }

    def _move_to_start_pose_arms_then_open_hand(
        self,
        planning_config: GraspObjectConfig,
        start_config: GraspObjectConfig,
        *,
        label: str = "start_pose",
    ) -> None:
        """
        Move to demo start with arms first, then open the right hand.

        Keeps the current right-hand joints during the arm move so a hand resting on
        the table is lifted by the arm before fingers spread (safer for Sharpa HA4).

        If OMPL rejects the current configuration (hand/table collision in the model),
        falls back to a direct arm command without collision planning.
        """
        if self.dry_run:
            logger.info("[dry_run] start: arms first (hand unchanged), then open grip")
            return

        current_hand = self._read_right_hand_q()
        arm_targets = self.joint_targets_with_fixed_hand(start_config, current_hand)
        logger.info("Moving to demo start pose (arms first, right hand unchanged)...")
        try:
            self._move_to_joint_targets(
                planning_config,
                arm_targets,
                pose_config=start_config,
                label=label,
            )
        except (AssertionError, RuntimeError) as exc:
            logger.warning(
                f"Collision-safe start planning failed ({exc}). "
                "Falling back to direct arm move with hand unchanged — "
                "ensure the path is clear if the hand was on the table."
            )
            assert self.handles is not None
            assert self.hardware_lock is not None
            with self.hardware_lock:
                self.handles.robot.set_joint_pos(
                    {
                        "head": start_config.head_joint_pos,
                        "torso": start_config.torso_joint_pos,
                    },
                    relative=False,
                    wait_time=4,
                )
            self.apply_live_targets(start_config, right_hand=current_hand)
            time.sleep(6.0)
        logger.info("Opening right hand at start pose...")
        self._open_right_grip_stepped(planning_config)

    def _solve_ik_for_right_ee(
        self,
        T_right: np.ndarray,
        left_q: np.ndarray | None = None,
        right_q: np.ndarray | None = None,
        *,
        planning_config: GraspObjectConfig | None = None,
        motion_label: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.pink_ik is not None
        if left_q is None or right_q is None:
            if self.dry_run:
                left_q = DEFAULT_JOINT_POS["left_arm"].copy()
                right_q = DEFAULT_JOINT_POS["right_arm"].copy()
            else:
                left_q, right_q = self._read_arm_joint_pos()

        # Open-grip pinch IK: approach phases only. Lift uses closed hand + R_ee lift pose.
        _pinch_open_grip_labels = frozenset(
            {"pre_grasp", "grasp", "grasp_approach"},
        )
        if (
            planning_config is not None
            and planning_config.titan_T_base_pinch is not None
            and motion_label is not None
            and motion_label in _pinch_open_grip_labels
        ):
            from demo.phase2.pinch_ik import (  # noqa: WPS433
                log_pinch_ik_result,
                pinch_target_for_move,
                solve_pinch_first_ik,
            )

            T_pinch = pinch_target_for_move(
                planning_config.titan_T_base_pinch,
                label=motion_label,
                pre_grasp_offset_m=planning_config.pre_grasp_offset_m,
                lift_height_m=planning_config.lift_height_m,
            )
            open_q = planning_config.hand_open_joint_pos
            result = solve_pinch_first_ik(
                T_pinch,
                left_q,
                right_q,
                open_q,
                pin_robot=self.pin_full_robot,
                assemble=self.assemble_qpos,
                forward_offset_m=planning_config.open_pinch_forward_offset_m,
                ik_palm_soft=planning_config.open_grip_ik_palm_soft,
            )
            log_pinch_ik_result(result, label=motion_label)
            if not result.converged:
                raise RuntimeError(
                    f"Open-grip IK failed ({motion_label}): "
                    f"align_err={result.pos_err_m:.4f} m > tol"
                )
            if not result.palm_acceptable:
                raise RuntimeError(
                    f"Open-grip orientation rejected ({motion_label}): "
                    f"thumb_axis·approach={result.thumb_approach_dot:.4f} "
                    f"(open-hand thumb MC→DP vs approach must be ≤90°, dot≥0)"
                )
            return result.left_arm.copy(), result.right_arm.copy()

        fk = self.pink_ik.fk(
            frames=["L_ee"],
            joint_pos_by_component={"left_arm": left_q, "right_arm": right_q},
        )
        ik = self.pink_ik.solve_ik(
            ee_target_poses={"L_ee": fk["L_ee"], "R_ee": homogeneous_to_se3(T_right)},
            arm_initial_joint_pos={"left_arm": left_q, "right_arm": right_q},
        )
        return ik["left_arm"].copy(), ik["right_arm"].copy()

    @staticmethod
    def apply_hand_profile(config: GraspObjectConfig) -> None:
        open_q, closed_q = load_hand_profile()
        config.hand_open_joint_pos = open_q
        config.hand_closed_joint_pos = closed_q

    @staticmethod
    def apply_demo_head_pose(*configs: GraspObjectConfig | None) -> None:
        """Demo: head_j1 pitch down (see HEAD_PITCH_DOWN_DEG); j2/j3 = 0."""
        for cfg in configs:
            if cfg is not None:
                cfg.head_joint_pos = DEMO_HEAD_JOINT_POS.copy()

    def _open_right_grip_stepped(self, config: GraspObjectConfig) -> None:
        if self.dry_run:
            logger.info("[dry_run] open right grip (stepped)")
            return
        assert self.handles is not None
        assert self.hardware_lock is not None
        open_q = config.hand_open_joint_pos.copy()
        with self.hardware_lock:
            move_hand_to_target(self.handles.right_hand, open_q)
        left_q, right_q = self._read_arm_joint_pos()
        self._publish_action_buffer(
            {
                "left_arm": left_q,
                "right_arm": right_q,
                "left_hand": config.left_hand_joint_pos,
                "right_hand": open_q,
            }
        )
        logger.info("Right hand at open grip (action buffer synced)")

    @staticmethod
    def joint_targets_with_open_grip(
        pose_config: GraspObjectConfig,
        hand_config: GraspObjectConfig,
    ) -> dict[str, np.ndarray]:
        """Arm pose from pose_config; right hand always open (virtual gripper)."""
        return {
            "left_arm": pose_config.left_arm_joint_pos.copy(),
            "right_arm": pose_config.right_arm_joint_pos.copy(),
            "left_hand": pose_config.left_hand_joint_pos.copy(),
            "right_hand": hand_config.hand_open_joint_pos.copy(),
        }

    def _plan_joint_targets(
        self,
        planning_config: GraspObjectConfig,
        target: dict[str, np.ndarray],
        *,
        pose_config: GraspObjectConfig | None = None,
        table_height: float | None = None,
        label: str = "joint_move",
        right_arm_collision_boxes: list[dict] | None = None,
    ):
        assert self.handles is not None
        pose = pose_config or planning_config
        table_height = planning_config.table_height if table_height is None else table_height
        planner = self._planner(table_height)
        arm_target = {
            "left_arm": target["left_arm"],
            "right_arm": target["right_arm"],
            "left_hand": target.get("left_hand", pose.left_hand_joint_pos),
            "right_hand": target.get(
                "right_hand", planning_config.hand_open_joint_pos
            ),
        }
        return plan_joint_targets_move(
            label=label,
            planner=planner,
            hardware_lock=self.hardware_lock,
            robot=self.handles.robot,
            left_hand=self.handles.left_hand,
            right_hand=self.handles.right_hand,
            target=arm_target,
            head_goal=pose.head_joint_pos,
            torso_goal=pose.torso_joint_pos,
            right_arm_collision_boxes=right_arm_collision_boxes,
        )

    def _execute_planned_joint_targets(self, planned) -> None:
        assert self.handles is not None
        assert self.action_buffer is not None
        execute_planned_joint_move(
            planned=planned,
            robot=self.handles.robot,
            left_hand=self.handles.left_hand,
            right_hand=self.handles.right_hand,
            action_buffer=self.action_buffer,
            action_buf_lock=self.action_buf_lock,
            hardware_lock=self.hardware_lock,
            command_hz=COMMAND_HZ,
            hold_time_s=0.5,
            smoothing_manager=self.smoothing_manager,
        )

    def _move_to_joint_targets(
        self,
        planning_config: GraspObjectConfig,
        target: dict[str, np.ndarray],
        *,
        pose_config: GraspObjectConfig | None = None,
        table_height: float | None = None,
        planned=None,
        label: str = "joint_move",
        allow_direct_fallback: bool = False,
    ) -> None:
        if self.dry_run:
            logger.info(f"[dry_run] move_to_joint_targets: {list(target.keys())}")
            return

        pose = pose_config or planning_config
        try:
            if planned is None:
                planned = self._plan_joint_targets(
                    planning_config,
                    target,
                    pose_config=pose_config,
                    table_height=table_height,
                    label=label,
                )
            self._execute_planned_joint_targets(planned)
        except (AssertionError, RuntimeError) as exc:
            if not allow_direct_fallback:
                raise
            logger.warning(
                f"Collision-safe {label} failed ({exc}); direct joint move (no OMPL)."
            )
            self._publish_action_buffer(
                {
                    "left_arm": target["left_arm"],
                    "right_arm": target["right_arm"],
                    "left_hand": target.get("left_hand", pose.left_hand_joint_pos),
                    "right_hand": target.get(
                        "right_hand", planning_config.hand_open_joint_pos
                    ),
                }
            )
            time.sleep(6.0)

    def _ik_arm_seed(
        self,
        label: str,
        seed_config: GraspObjectConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Joint seed for goal IK inside OMPL planning.

        After pre-grasp / at grasp, use live arms so the goal matches the chain;
        pre_grasp from home still uses ``seed_config`` (start.yaml).
        """
        if (
            not self.dry_run
            and self.handles is not None
            and label in ("grasp_approach", "lift")
        ):
            return self._read_arm_joint_pos()
        return (
            seed_config.left_arm_joint_pos.copy(),
            seed_config.right_arm_joint_pos.copy(),
        )

    def _plan_move_to_right_ee(
        self,
        planning_config: GraspObjectConfig,
        T_right: np.ndarray,
        right_hand_q: np.ndarray,
        *,
        pose_config: GraspObjectConfig | None = None,
        ik_seed_config: GraspObjectConfig | None = None,
        label: str = "ee_move",
        right_arm_collision_boxes: list[dict] | None = None,
    ):
        pose = pose_config or planning_config
        seed = ik_seed_config or pose
        left_q, right_q = self._ik_arm_seed(label, seed)
        _, right_arm = self._solve_ik_for_right_ee(
            T_right,
            left_q,
            right_q,
            planning_config=planning_config,
            motion_label=label,
        )
        return self._plan_joint_targets(
            planning_config,
            {
                "left_arm": left_q,
                "right_arm": right_arm,
                "left_hand": pose.left_hand_joint_pos,
                "right_hand": right_hand_q,
            },
            pose_config=pose,
            label=label,
            right_arm_collision_boxes=right_arm_collision_boxes,
        )

    def _move_to_right_ee_pose(
        self,
        planning_config: GraspObjectConfig,
        T_right: np.ndarray,
        right_hand_q: np.ndarray,
        *,
        pose_config: GraspObjectConfig | None = None,
        ik_seed_config: GraspObjectConfig | None = None,
        planned=None,
        right_arm_collision_boxes: list[dict] | None = None,
        label: str = "ee_move",
        allow_direct_fallback: bool = False,
    ) -> None:
        if self.dry_run:
            logger.info(f"[dry_run] move_to_right_ee_pose ({label})")
            return
        pose = pose_config or planning_config
        seed = ik_seed_config or pose
        right_hand_q = np.asarray(right_hand_q, dtype=np.float64)
        try:
            if planned is None:
                planned = self._plan_move_to_right_ee(
                    planning_config,
                    T_right,
                    right_hand_q,
                    pose_config=pose,
                    ik_seed_config=seed,
                    label=label,
                    right_arm_collision_boxes=right_arm_collision_boxes,
                )
            self._execute_planned_joint_targets(planned)
        except (AssertionError, RuntimeError) as exc:
            if not allow_direct_fallback:
                raise
            logger.warning(
                f"Collision-safe {label} failed ({exc}); direct IK move (no OMPL)."
            )
            left_q, right_q = self._ik_arm_seed(label, seed)
            _, right_arm = self._solve_ik_for_right_ee(
                T_right,
                left_q,
                right_q,
                planning_config=planning_config,
                motion_label=label,
            )
            self.apply_live_targets(pose, right_arm=right_arm, right_hand=right_hand_q)
            time.sleep(6.0)
            return
        self._publish_action_buffer(
            {
                "left_arm": planned.arm_goal["left_arm"],
                "right_arm": planned.arm_goal["right_arm"],
                "left_hand": planned.arm_goal["left_hand"],
                "right_hand": right_hand_q,
            }
        )

    def _set_right_hand(self, config: GraspObjectConfig, q: np.ndarray, wait_s: float = 1.0) -> None:
        if self.dry_run:
            logger.info(f"[dry_run] set right hand joints (first 3): {q[:3]}")
            return
        assert self.handles is not None
        assert self.hardware_lock is not None
        with self.hardware_lock:
            set_hand_joint_positions(self.handles.right_hand, q)
        time.sleep(wait_s)

    def _close_right_hand_until_stall(self, config: GraspObjectConfig) -> np.ndarray:
        """Close pinch with stall detection; returns joint pos to hold for lift."""
        if self.dry_run:
            result = close_hand_until_stall(
                None,
                config.hand_closed_joint_pos,
                dry_run=True,
            )
            return result.q_final
        assert self.handles is not None
        assert self.hardware_lock is not None
        with self.hardware_lock:
            result = close_hand_until_stall(
                self.handles.right_hand,
                config.hand_closed_joint_pos,
            )
        logger.info(f"Grasp close finished: reason={result.reason}, steps={result.steps}")
        q_final = result.q_final.copy()
        left_q, right_q = self._read_arm_joint_pos()
        self._publish_action_buffer(
            {
                "left_arm": left_q,
                "right_arm": right_q,
                "left_hand": config.left_hand_joint_pos,
                "right_hand": q_final,
            }
        )
        logger.info("Right hand at stall-close grasp (action buffer synced)")
        return q_final

    def run_sequence(
        self,
        config: GraspObjectConfig,
        *,
        start_config: GraspObjectConfig | None = None,
        skip_home_at_end: bool = False,
    ) -> None:
        self.apply_hand_profile(config)
        if start_config is not None:
            self.apply_hand_profile(start_config)
        self.apply_demo_head_pose(config, start_config)
        logger.info(
            f"Demo head (head_j1 pitch down {HEAD_PITCH_DOWN_DEG:.0f}°): "
            f"{np.round(DEMO_HEAD_JOINT_POS, 3).tolist()} rad"
        )

        pre = config.resolved_pre_grasp_pose()
        grasp = config.grasp_pose.copy()
        lift = config.resolved_lift_pose()

        logger.info(f"=== Phase 1 grasp: {config.object_name} ===")
        if start_config is not None:
            logger.info(f"Start pose: {start_config.object_name}")
        logger.info(f"Planning table_height: {config.table_height:.3f} m (object config)")
        logger.info(format_pose(grasp, "grasp"))
        logger.info(format_pose(pre, "pre_grasp"))
        logger.info(format_pose(lift, "lift"))

        open_targets = self.joint_targets_with_open_grip(config, config)

        if self.dry_run:
            labels = ["start" if start_config else "home", "pre_grasp", "grasp", "lift"]
            for label, T in zip(labels, [None, pre, grasp, lift]):
                if T is not None:
                    _, rq = self._solve_ik_for_right_ee(T)
                    logger.info(f"[dry_run] {label}: right_arm IK = {np.round(rq, 3)}")
            logger.info("[dry_run] sequence complete (no hardware)")
            return

        if start_config is not None:
            self._move_to_start_pose_arms_then_open_hand(config, start_config)
        else:
            logger.warning(
                "No start.yaml: moving to object grasp joints as initial pose "
                "(prefer start object: python demo/phase1/run_grasp.py --object-name chips)"
            )
            self._move_to_start_pose_arms_then_open_hand(config, config, label="home_pose")

        ik_seed = start_config if start_config is not None else config
        prefetch_pre = BackgroundPlan(
            "pre_grasp",
            lambda: self._plan_move_to_right_ee(
                config,
                pre,
                config.hand_open_joint_pos,
                ik_seed_config=ik_seed,
                label="pre_grasp",
            ),
        )
        planned_pre = wait_enter_with_prefetch(
            f"Place '{config.object_name}' on the table, then press Enter to approach...",
            prefetch_pre,
        )
        self._move_to_right_ee_pose(
            config,
            pre,
            config.hand_open_joint_pos,
            planned=planned_pre,
        )

        # Grasp — tuned joints; still open until close step
        self._move_to_joint_targets(config, open_targets)

        # Close pinch (first step that closes the gripper)
        logger.info("Closing virtual gripper (stall-detect)...")
        q_grasp = self._close_right_hand_until_stall(config)

        # Lift (closed hand)
        self._move_to_right_ee_pose(config, lift, q_grasp)

        grasp_closed_targets = {
            "left_arm": config.left_arm_joint_pos,
            "right_arm": config.right_arm_joint_pos,
            "left_hand": config.left_hand_joint_pos,
            "right_hand": q_grasp,
        }
        prefetch_grasp_return = BackgroundPlan(
            "lift_to_grasp",
            lambda: self._plan_joint_targets(
                config,
                grasp_closed_targets,
                label="lift_to_grasp",
            ),
        )
        prefetch_grasp_return.start()
        logger.info(f"Holding at lift for {config.hold_time_s:.1f}s (film now; planning grasp return)...")
        time.sleep(config.hold_time_s)

        if skip_home_at_end:
            logger.info("Staying at lift (--skip-home-at-end). Grasp sequence complete.")
            return

        planned_grasp_return = wait_enter_with_prefetch(
            "Lift complete. Press Enter to move back to grasp pose...",
            prefetch_grasp_return,
            prefetch_already_started=True,
        )
        logger.info("Moving to grasp pose (hand still closed)...")
        self._move_to_joint_targets(config, grasp_closed_targets, planned=planned_grasp_return)

        logger.info("Opening gripper at grasp pose...")
        self._open_right_grip_stepped(config)

        prefetch_start = None
        if start_config is not None:
            prefetch_start = BackgroundPlan(
                "return_start",
                lambda: self._plan_joint_targets(
                    config,
                    self.joint_targets_with_open_grip(start_config, config),
                    pose_config=start_config,
                    label="return_start",
                ),
            )
        planned_start = wait_enter_with_prefetch(
            f"Remove '{config.object_name}' from the gripper, then press Enter to return to start...",
            prefetch_start,
        )
        if planned_start is not None:
            logger.info("Returning to demo start pose...")
            self._move_to_joint_targets(
                config,
                self.joint_targets_with_open_grip(start_config, config),
                pose_config=start_config,
                planned=planned_start,
            )
        else:
            logger.warning(
                "No start config; staying at grasp pose with open grip. "
                "Add start.yaml or use --start-object-name."
            )

        logger.info("Grasp sequence complete.")
