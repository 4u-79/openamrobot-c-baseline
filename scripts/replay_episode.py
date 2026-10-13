#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_episode.py
=================

OpenAMRobot 2.0 | Workstream C | Deliverable C4 — Deterministic Replay Node.

Reads a validated 19-channel LeRobot episode Parquet file (schema-v1.0-frozen)
and replays it open-loop onto the live robot (or simulation) at the canonical
30 Hz rate.  The script never invents transforms: all values are published
verbatim from the recorded action column.

Architecture
------------
* Channels 0-2  → /cmd_vel  (geometry_msgs/Twist)  published by base publisher.
* Channels 5-10 → /left_arm_controller/follow_joint_trajectory  (FollowJointTrajectory
  action); single-frame goal, 1/30 s execution time.
* Channels 12-17 → /right_arm_controller/follow_joint_trajectory  (same pattern).
* Channel 11    → /left_gripper_controller/gripper_cmd  (GripperCommand action,
  GripperCommand.goal.command.position = normalised value mapped to gripper range).
* Channel 18    → /right_gripper_controller/gripper_cmd.

Channels 3-4 (torso_pitch, torso_yaw) are fixed-mast in OpenAMRobot 2.0 and are
published as zero; no channel is silently dropped or re-mapped.

Fallback (--dry-run / no ROS 2 install)
----------------------------------------
When rclpy / geometry_msgs / control_msgs are absent, the script operates in a
pure-Python dry-run mode that:
  1. Parses and validates the Parquet file.
  2. Checks every action frame against CANONICAL_ACTION_LIMITS ±5 %.
  3. Prints a summary and exits 0 (pass) or 1 (violations found).

Usage
-----
Dry-run (no ROS 2 required):
    python scripts/replay_episode.py --parquet /data/episodes/data/chunk-000/episode_000000.parquet --dry-run

Live replay:
    ros2 run openamrobot_c_baseline replay_episode \
        --parquet /data/episodes/data/chunk-000/episode_000000.parquet

Exit codes: 0 = success / clean, 1 = violation / error, 2 = usage error.

Binding references:
  C-TEAM-PLAN-01 rev 2.0 section 3.3 (C4 Deterministic Replay)
  schema-v1.0-frozen  19-channel (P-03 Addendum rev18.2)
  openamr-platform-sw  nav2_params.yaml / velocity_smoother  → base limits
  openamr-upperbody-sw  option B (P-03 rev18.9 item 18)     → arm mount frames
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Frozen contract constants (mirrors check_dataset_integrity.py)
# ---------------------------------------------------------------------------

TARGET_FPS: float = 30.0
DT: float = 1.0 / TARGET_FPS
STATE_DIM: int = 19

# Canonical channel names (schema-v1.0-frozen ordering).
CHANNEL_NAMES: Tuple[str, ...] = (
    "base_linear_x",       #  0
    "base_linear_y",       #  1 — diff-drive constraint: always 0
    "base_angular_z",      #  2
    "torso_pitch",         #  3 — fixed-mast: always 0
    "torso_yaw",           #  4 — fixed-mast: always 0
    "left_shoulder_pan",   #  5
    "left_shoulder_lift",  #  6
    "left_elbow_flex",     #  7
    "left_wrist_flex",     #  8
    "left_wrist_roll",     #  9
    "left_wrist_yaw",      # 10
    "left_gripper",        # 11
    "right_shoulder_pan",  # 12
    "right_shoulder_lift", # 13
    "right_elbow_flex",    # 14
    "right_wrist_flex",    # 15
    "right_wrist_roll",    # 16
    "right_wrist_yaw",     # 17
    "right_gripper",       # 18
)

# Physical limits per channel [min, max] — mirrors CANONICAL_ACTION_LIMITS in
# check_dataset_integrity.py.  Keep both files in sync.
# fmt: off
CANONICAL_ACTION_LIMITS = np.array([
    [-0.20,  0.20],   #  0 base_linear_x   [m/s]
    [ 0.00,  0.00],   #  1 base_linear_y   [m/s]
    [-0.50,  0.50],   #  2 base_angular_z  [rad/s]
    [ 0.00,  0.00],   #  3 torso_pitch     [rad]
    [ 0.00,  0.00],   #  4 torso_yaw       [rad]
    [-2.618, 2.618],  #  5 left_shoulder_pan
    [-1.745, 1.745],  #  6 left_shoulder_lift
    [-2.618, 2.618],  #  7 left_elbow_flex
    [-1.833, 1.833],  #  8 left_wrist_flex
    [-3.142, 3.142],  #  9 left_wrist_roll
    [-1.833, 1.833],  # 10 left_wrist_yaw
    [ 0.00,  1.00],   # 11 left_gripper
    [-2.618, 2.618],  # 12 right_shoulder_pan
    [-1.745, 1.745],  # 13 right_shoulder_lift
    [-2.618, 2.618],  # 14 right_elbow_flex
    [-1.833, 1.833],  # 15 right_wrist_flex
    [-3.142, 3.142],  # 16 right_wrist_roll
    [-1.833, 1.833],  # 17 right_wrist_yaw
    [ 0.00,  1.00],   # 18 right_gripper
], dtype=np.float32)
# fmt: on

# Left arm joint names (ros2_control ordering for FollowJointTrajectory).
LEFT_ARM_JOINT_NAMES: List[str] = [
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_wrist_yaw",
]
# Corresponding channel indices into the 19-dim action vector.
LEFT_ARM_CHANNEL_INDICES: List[int] = [5, 6, 7, 8, 9, 10]

RIGHT_ARM_JOINT_NAMES: List[str] = [
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_wrist_yaw",
]
RIGHT_ARM_CHANNEL_INDICES: List[int] = [12, 13, 14, 15, 16, 17]

LEFT_GRIPPER_CHANNEL: int = 11
RIGHT_GRIPPER_CHANNEL: int = 18

# ---------------------------------------------------------------------------
# Parquet loading and pre-flight validation
# ---------------------------------------------------------------------------


def load_episode_parquet(parquet_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load action and timestamp arrays from a single episode Parquet file.

    Returns:
        (actions, timestamps) — shapes (T, 19) float32 and (T,) float64.

    Raises:
        RuntimeError: on any read or schema violation.
    """
    try:
        import pyarrow.parquet as pq  # lazy import; keeps dry-run lightweight
    except ImportError:
        raise RuntimeError(
            "pyarrow is required for Parquet loading.  "
            "Install with: pip install pyarrow"
        )

    try:
        table = pq.read_table(parquet_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read Parquet file: {exc}") from exc

    if "action" not in table.schema.names:
        raise RuntimeError("Parquet file is missing the required 'action' column.")

    action_col = table.column("action")
    if action_col.null_count > 0:
        raise RuntimeError("'action' column contains null values — episode is corrupt.")

    try:
        action_rows = action_col.to_pylist()
        actions = np.array(action_rows, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"'action' column is ragged or undecodable: {exc}") from exc

    if actions.ndim != 2:
        raise RuntimeError(
            f"'action' must be a 2-D matrix, got ndim={actions.ndim}."
        )
    if actions.shape[1] != STATE_DIM:
        raise RuntimeError(
            f"'action' vector length {actions.shape[1]} != frozen STATE_DIM {STATE_DIM}. "
            "Run check_dataset_integrity.py --skip-video-checks first."
        )
    if not np.all(np.isfinite(actions)):
        bad = int(np.count_nonzero(~np.isfinite(actions)))
        raise RuntimeError(
            f"'action' contains {bad} NaN/Inf values — run pre-flight gate first."
        )

    # Timestamps (optional but preferred for drift monitoring).
    if "timestamp" in table.schema.names:
        ts_col = table.column("timestamp")
        try:
            timestamps = ts_col.to_numpy(zero_copy_only=False).astype(np.float64)
            if timestamps.ndim != 1 or len(timestamps) != len(actions):
                timestamps = np.arange(len(actions), dtype=np.float64) * DT
        except Exception:
            timestamps = np.arange(len(actions), dtype=np.float64) * DT
    else:
        timestamps = np.arange(len(actions), dtype=np.float64) * DT

    return actions, timestamps


def check_bounds_dry_run(
    actions: np.ndarray,
    margin_pct: float = 0.05,
    eps: float = 1e-4,
) -> List[str]:
    """Return a list of human-readable violation strings (empty = all within limits).

    Mirrors the logic in check_dataset_integrity.check_action_bounds().
    """
    violations: List[str] = []
    for ch in range(STATE_DIM):
        lo = float(CANONICAL_ACTION_LIMITS[ch, 0])
        hi = float(CANONICAL_ACTION_LIMITS[ch, 1])
        span = hi - lo
        margin = eps if span < eps else span * margin_pct
        lower_bound = lo - margin
        upper_bound = hi + margin
        vals = actions[:, ch]
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size == 0:
            continue
        min_val = float(np.min(finite_vals))
        max_val = float(np.max(finite_vals))
        if min_val < lower_bound or max_val > upper_bound:
            violations.append(
                f"  ch {ch:2d} ({CHANNEL_NAMES[ch]:22s}): "
                f"recorded [{min_val:+.4f}, {max_val:+.4f}]  "
                f"allowed  [{lower_bound:+.4f}, {upper_bound:+.4f}]"
            )
    return violations


# ---------------------------------------------------------------------------
# Dry-run path (no ROS 2 required)
# ---------------------------------------------------------------------------


def run_dry_run(parquet_path: Path, verbose: bool = False) -> int:
    """Validate the episode without publishing any ROS 2 messages.

    Returns 0 on success, 1 on violation.
    """
    print(f"[C4 Replay] Dry-run: loading {parquet_path.name} …")
    try:
        actions, timestamps = load_episode_parquet(parquet_path)
    except RuntimeError as exc:
        print(f"[C4 Replay] FAIL — {exc}", file=sys.stderr)
        return 1

    n_frames = len(actions)
    duration = float(timestamps[-1] - timestamps[0]) if n_frames > 1 else 0.0
    print(
        f"[C4 Replay] {n_frames} frames  |  "
        f"{n_frames / TARGET_FPS:.2f} s wall-clock replay  |  "
        f"timestamp span {duration:.3f} s"
    )

    if verbose:
        for ch, name in enumerate(CHANNEL_NAMES):
            col = actions[:, ch]
            print(
                f"  ch {ch:2d} {name:22s}: "
                f"min={col.min():+.4f}  max={col.max():+.4f}  "
                f"mean={col.mean():+.4f}"
            )

    violations = check_bounds_dry_run(actions)
    if violations:
        print(
            f"[C4 Replay] FAIL — {len(violations)} channel(s) exceed ±5% physical limits:",
            file=sys.stderr,
        )
        for v in violations:
            print(v, file=sys.stderr)
        return 1

    print(
        f"[C4 Replay] PASS — all {STATE_DIM} channels within ±5% of physical limits."
    )
    return 0


# ---------------------------------------------------------------------------
# ROS 2 live-replay node
# ---------------------------------------------------------------------------


def _import_ros2():
    """Lazy-import ROS 2 packages; return None-dict on ImportError."""
    try:
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from rclpy.duration import Duration as RclpyDuration
        from geometry_msgs.msg import Twist
        from control_msgs.action import FollowJointTrajectory, GripperCommand
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration as BIDuration

        return dict(
            rclpy=rclpy,
            ActionClient=ActionClient,
            Node=Node,
            RclpyDuration=RclpyDuration,
            Twist=Twist,
            FollowJointTrajectory=FollowJointTrajectory,
            GripperCommand=GripperCommand,
            JointTrajectory=JointTrajectory,
            JointTrajectoryPoint=JointTrajectoryPoint,
            BIDuration=BIDuration,
        )
    except ImportError:
        return None


class OpenAMRobotReplayNode:
    """Stateful replay node — wraps rclpy.node.Node and drives the 30 Hz loop."""

    def __init__(self, ros2, actions: np.ndarray, timestamps: np.ndarray):
        self._ros2 = ros2
        self._actions = actions
        self._timestamps = timestamps
        self._num_frames = len(actions)
        self._idx = 0
        self._halt_sent = False

        Node = ros2["Node"]
        ActionClient = ros2["ActionClient"]
        FollowJointTrajectory = ros2["FollowJointTrajectory"]
        GripperCommand = ros2["GripperCommand"]

        self.node = Node("openamrobot_replay_node")

        # Base command publisher.
        Twist = ros2["Twist"]
        self._cmd_vel_pub = self.node.create_publisher(
            Twist, "/cmd_vel", qos_profile=10
        )

        # Arm trajectory action clients.
        self._left_arm_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            "/left_arm_controller/follow_joint_trajectory",
        )
        self._right_arm_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            "/right_arm_controller/follow_joint_trajectory",
        )

        # Gripper action clients.
        self._left_gripper_client = ActionClient(
            self.node,
            GripperCommand,
            "/left_gripper_controller/gripper_cmd",
        )
        self._right_gripper_client = ActionClient(
            self.node,
            GripperCommand,
            "/right_gripper_controller/gripper_cmd",
        )

        self.node.get_logger().info(
            f"[C4 Replay] Node ready — {self._num_frames} frames @ {TARGET_FPS} Hz"
        )

    def _make_arm_goal(self, joint_names, positions, ros2):
        """Build a single-point FollowJointTrajectory goal for one arm frame."""
        FollowJointTrajectory = ros2["FollowJointTrajectory"]
        JointTrajectory = ros2["JointTrajectory"]
        JointTrajectoryPoint = ros2["JointTrajectoryPoint"]
        BIDuration = ros2["BIDuration"]

        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = joint_names

        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        # Give the controller exactly one control period to reach this point.
        pt.time_from_start = BIDuration(sec=0, nanosec=int(DT * 1e9))
        traj.points = [pt]
        goal.trajectory = traj
        return goal

    def _publish_frame(self, frame: np.ndarray):
        """Publish one 19-dim action frame verbatim to all endpoints."""
        ros2 = self._ros2
        Twist = ros2["Twist"]

        # -- Base: channels 0..2 --
        base_cmd = Twist()
        base_cmd.linear.x = float(frame[0])
        base_cmd.linear.y = float(frame[1])   # diff-drive: expected 0.0
        base_cmd.angular.z = float(frame[2])
        self._cmd_vel_pub.publish(base_cmd)

        # -- Left arm: channels 5..10 --
        left_positions = [float(frame[i]) for i in LEFT_ARM_CHANNEL_INDICES]
        left_goal = self._make_arm_goal(LEFT_ARM_JOINT_NAMES, left_positions, ros2)
        self._left_arm_client.send_goal_async(left_goal)  # non-blocking

        # -- Right arm: channels 12..17 --
        right_positions = [float(frame[i]) for i in RIGHT_ARM_CHANNEL_INDICES]
        right_goal = self._make_arm_goal(RIGHT_ARM_JOINT_NAMES, right_positions, ros2)
        self._right_arm_client.send_goal_async(right_goal)  # non-blocking

        # -- Grippers: channels 11, 18 --
        GripperCommand = ros2["GripperCommand"]
        for client, ch in (
            (self._left_gripper_client, LEFT_GRIPPER_CHANNEL),
            (self._right_gripper_client, RIGHT_GRIPPER_CHANNEL),
        ):
            g = GripperCommand.Goal()
            g.command.position = float(frame[ch])
            g.command.max_effort = 0.0   # let the driver use its configured limit
            client.send_goal_async(g)

    def _halt_robot(self):
        """Publish a zero-velocity Twist to stop the base immediately."""
        if self._halt_sent:
            return
        self._halt_sent = True
        Twist = self._ros2["Twist"]
        zero = Twist()
        try:
            self._cmd_vel_pub.publish(zero)
            self.node.get_logger().info("[C4 Replay] Safe halt sent (/cmd_vel = 0).")
        except Exception:  # noqa: BLE001
            pass

    def run(self) -> int:
        """Spin the 30 Hz replay loop. Returns 0 on completion, 1 on error."""
        rclpy = self._ros2["rclpy"]
        logger = self.node.get_logger()

        t_start = time.monotonic()
        try:
            for self._idx in range(self._num_frames):
                # Compute deadline for this frame.
                deadline = t_start + (self._idx + 1) * DT

                self._publish_frame(self._actions[self._idx])
                rclpy.spin_once(self.node, timeout_sec=0.0)

                # Rate-limit: sleep until deadline (compensates for publish latency).
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    logger.warn(
                        f"Frame {self._idx}: {abs(remaining)*1000:.1f} ms late "
                        f"(replay falling behind {TARGET_FPS} Hz)"
                    )

            logger.info(
                f"[C4 Replay] Episode complete — {self._num_frames} frames replayed."
            )
            return 0

        except KeyboardInterrupt:
            logger.info("[C4 Replay] Interrupted by user.")
            return 0

        except Exception as exc:  # noqa: BLE001
            logger.error(f"[C4 Replay] Unexpected error: {exc}")
            return 1

        finally:
            self._halt_robot()

    def destroy(self):
        self.node.destroy_node()


def run_live(parquet_path: Path) -> int:
    """Live ROS 2 replay path."""
    ros2 = _import_ros2()
    if ros2 is None:
        print(
            "[C4 Replay] ERROR: rclpy / geometry_msgs / control_msgs not found.  "
            "Source your ROS 2 workspace or use --dry-run.",
            file=sys.stderr,
        )
        return 2

    try:
        actions, timestamps = load_episode_parquet(parquet_path)
    except RuntimeError as exc:
        print(f"[C4 Replay] FAIL — {exc}", file=sys.stderr)
        return 1

    ros2["rclpy"].init()
    replay = OpenAMRobotReplayNode(ros2, actions, timestamps)
    try:
        exit_code = replay.run()
    finally:
        replay.destroy()
        ros2["rclpy"].shutdown()

    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replay_episode.py",
        description=(
            "OpenAMRobot 2.0 | Workstream C | Deliverable C4 — Deterministic Replay.\n"
            "Reads a validated 19-channel LeRobot episode Parquet file and publishes\n"
            "open-loop trajectory goals at 30 Hz to the platform and arm controllers."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--parquet",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the episode_XXXXXX.parquet file to replay.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Parse and bound-check the episode without publishing ROS 2 messages. "
            "Exits 0 if the file passes all checks, 1 if violations are found. "
            "Does not require a ROS 2 install."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-channel statistics in dry-run mode.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.parquet.is_file():
        print(
            f"[C4 Replay] ERROR: Parquet file not found: {args.parquet}",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        return run_dry_run(args.parquet, verbose=args.verbose)
    return run_live(args.parquet)


if __name__ == "__main__":
    sys.exit(main())
