# Technical Cross-Repository Audit & C9 Alignment Report

**Project**: OpenAMRobot 2.0 | Workstream C (Embodied AI & Imitation Learning)  
**Deliverables**: C2 Whole-Body Schema, C9 LeRobot Contract, C10 Pre-Flight Gate, C11 ACT Policy Baseline  
**Author**: Marcus Andrei Guinto (`@4u-79`, `mavguinto@gmail.com`), Division B  
**Reviewer**: Mohamed Sayed (`@panthera-momagdii`), Software Lead / Division B Lead  
**Audit Reference**: `openamrobot-interfaces#18` & `openAMRobot` P-03 rev19.6  
**Timestamp**: 2026-10-20T23:45:00Z  
**Schema Frozen Tag**: `schema-v1.0-frozen` (`openamrobot-fixed-mast-19dof`)

---

## 1. Executive Summary & Audited Repositories

This report establishes the technical ground truth and integration boundary contracts across the 5 core repositories of the OpenAMRobot 2.0 workspace as of 20 October 2026.
Following technical review by Division B Lead Mohamed Sayed on `openamrobot-interfaces#18`, the Workstream C baseline (`openamrobot-c-baseline`) has undergone a comprehensive reconciliation to align camera keys, kinematic profiles, LeRobot software stack versions, and hardware safety arbitration.

### Audited Repository Roster

| Repository | Git Commit SHA | Branch | Primary Responsibility in OpenAMRobot 2.0 |
|---|---|---|---|
| `4u-79/openamrobot-c-baseline` | `a29ce7c89b7c` | `main` | C2 schema, C9 dataset contract, C10 QA gate, C11 ACT baseline, C4 replay |
| `openAMRobot/openamrobot-interfaces` | `5c7329596c00` | `main` | Canonical ROS 2 IDL messages, C9 dataset contract documentation, whole-body schema |
| `openAMRobot/openamrobot-manipulation` | `6c1524317135` | `main` | A5 Device Package contract, A8 OpenArm 2.0 mapping, A12/A13 trajectory controllers |
| `openAMRobot/openamr-upperbody-sw` | `ca0d8c0678eb` | `main` | A6 upperbody URDF/xacro, DOLD Hexalift V4 lift column, `gz_ros2_control` config |
| `openAMRobot/openamr-platform-sw` | `e2b4cb10e7b4` | `main` | Mobile base hardware drivers, #35 motion-source arbitration (`twist_mux`, smoother) |

---

## 2. Resolution of Mohamed Sayed's 3 C9 Alignment Items (`openamrobot-interfaces#18`)

Mohamed Sayed's review comment on `openamrobot-interfaces#18` required explicit alignment on three integration boundaries before validating real physical demonstration bags.

### 2.1 Feature-Key Mapping (Camera Streams)

- **Finding**: Upstream C9 dataset contract (`openamrobot-interfaces/docs/c9-lerobot-dataset-contract.md`) designates `observation.images.head` (ZED-121210 scene camera mounted on the upperbody lift carriage) and `observation.images.base` (Orbbec Gemini 336L mobile base navigation camera). The baseline previously utilized legacy aliases `camera_scene` and `camera_base_nav`.
- **Resolution**: Renamed all canonical camera streams across `meta/info.json`, `validate_dataset_contract.py`, `check_dataset_integrity.py`, `generate_or_ingest_dataset.py`, `configs/act_openamrobot_19dof.yaml`, and `run_dashboard.py` to the normative C9 keys:
  * `observation.images.camera_wrist_left` (480×640×3)
  * `observation.images.camera_wrist_right` (480×640×3)
  * `observation.images.head` (720×1280×3)
  * `observation.images.base` (480×640×3)
- **Backward Compatibility**: `validate_dataset_contract.py`, `check_dataset_integrity.py`, and `run_dashboard.py` retain explicit fallback alias resolution mapping legacy names to C9 keys without breaking legacy recordings.

### 2.2 Robot Configuration & Kinematic Profile Alignment

- **Finding**: Upstream `openamr-upperbody-sw` introduced the P-03 Option B DOLD Hexalift V4 prismatic lift configuration (`lift_joint`, 0.0..0.350 m stroke with calibration stops L1000/L1175/L1350). Concurrently, C9 accepted contract `c9-lerobot-dataset-valid.yaml` enumerates a 16-channel arm-only profile (`lerobot-v0.6.1-bi-openarm-position` in degrees). Conflating these three kinematic descriptions would corrupt data integrity.
- **Resolution**: Formally tagged and frozen the 19-DoF configuration as `openamrobot-fixed-mast-19dof` (`schema-v1.0-frozen`). The 2 unactuated mast channels (`torso_pitch`, `torso_yaw`) carry strict zero limits `[0.0, 0.0]` per P-03 Addendum rev18.2. Option B 20-channel migration is managed under formal change control.

#### Profile Compatibility Matrix

| Specification | `openamrobot-fixed-mast-19dof` | `lerobot-v0.6.1-bi-openarm-position` | `openamrobot-lift-option-b-20dof` |
|---|---|---|---|
| **Total Channels** | **19** | **16** | **20** |
| **Mobile Base** | 3 (`base_linear_x`, `base_linear_y`, `base_angular_z`) | 0 (Arm-only bench fixture) | 3 (`base_linear_x`, `base_linear_y`, `base_angular_z`) |
| **Torso / Lift** | 2 (`torso_pitch`, `torso_yaw` fixed `[0,0]`) | 0 (None) | 1 (`lift_joint` prismatic 0.0..0.350m) |
| **Left Arm & Gripper**| 7 (`left_shoulder_pan` .. `left_gripper`) | 8 (`openarm_left_joint1..7` + gripper) | 8 (`openarm_left_joint1..7` + gripper) |
| **Right Arm & Gripper**| 7 (`right_shoulder_pan` .. `right_gripper`)| 8 (`openarm_right_joint1..7` + gripper)| 8 (`openarm_right_joint1..7` + gripper)|
| **Angular Units** | Radians (`rad`, `rad/s`) | Degrees (`deg`) | Radians (`rad`, `rad/s`) |
| **Governance Status** | **Frozen Normative Baseline** | C9 Contract Fixture | P-03 Option B Tracking Branch |

### 2.3 LeRobot & Software Stack Provenance

- **Finding**: Upstream C9 dataset contract references an unreleased snapshot of Hugging Face LeRobot (`v0.6.1`, commit `7e241bd630a3719a56157a497ce5d08f244784f1`) with `validated_release_pin: null`. Inter-stream synchronisation tolerance was also marked `pending_evidence: true`.
- **Resolution**:
  * Pinned `lerobot==0.4.1` as the local runtime adapter producing canonical `LeRobotDataset v3.0` directory trees.
  * Formally documented in `meta/info.json` and `README.md` that 30.0 ms is our provisional pre-flight QA threshold.
  * Preserved full forward compatibility so when upstream releases `lerobot==0.6.1` on PyPI, dataset directories require zero migration.

---

## 3. Full 19-Channel Hardware, Topic & Limit Mapping Matrix

### Safety Boundary Compliance (Section 10 Team Ground Truth)

Per Section 10 of the OpenAMRobot 2.0 Working Rules:
> *"Do not author safety, emergency stop, brake or watchdog code with an AI agent."*

All mobile base velocity commands in `replay_episode.py` and evaluation nodes are strictly published to `/cmd_vel_nav`.
Under PR #35 (`openamr-platform-sw`), the mobile base pipeline arbitrates commands via:
1. `twist_mux` (prioritizing E-stop and joy teleoperation over navigation/imitation commands).
2. Velocity Smoother (enforcing linear acceleration <= 0.5 m/s² and angular acceleration <= 1.0 rad/s²).
3. Collision Monitor (subscribing to 2D LiDAR / costmaps with automated deceleration and emergency stop zones).

### Channel Hardware & Control Endpoint Matrix

| Ch | Channel Name | Hardware Subsystem | Control Topic / Action Server | Physical Limit | C10 Tolerance (±5%) | Unit |
|:---:|---|---|---|:---:|:---:|:---:|
| 0 | `base_linear_x` | Diff-Drive Platform | `/cmd_vel_nav` (geometry_msgs/Twist) | [-0.20, +0.20] | [-0.220, +0.220] | m/s |
| 1 | `base_linear_y` | Diff-Drive Constraint | `/cmd_vel_nav` (must remain 0.0) | [0.00, 0.00] | [-1e-4, +1e-4] | m/s |
| 2 | `base_angular_z` | Diff-Drive Platform | `/cmd_vel_nav` (geometry_msgs/Twist) | [-0.50, +0.50] | [-0.550, +0.550] | rad/s |
| 3 | `torso_pitch` | Structural Mast | Fixed mast (unactuated) | [0.00, 0.00] | [-1e-4, +1e-4] | rad |
| 4 | `torso_yaw` | Structural Mast | Fixed mast (unactuated) | [0.00, 0.00] | [-1e-4, +1e-4] | rad |
| 5 | `left_shoulder_pan` | OpenArm 2.0 Left | `/left_arm_controller/follow_joint_trajectory` | [-2.618, +2.618] | [-2.880, +2.880] | rad |
| 6 | `left_shoulder_lift`| OpenArm 2.0 Left | `/left_arm_controller/follow_joint_trajectory` | [-1.745, +1.745] | [-1.920, +1.920] | rad |
| 7 | `left_elbow_flex` | OpenArm 2.0 Left | `/left_arm_controller/follow_joint_trajectory` | [-2.618, +2.618] | [-2.880, +2.880] | rad |
| 8 | `left_wrist_flex` | OpenArm 2.0 Left | `/left_arm_controller/follow_joint_trajectory` | [-1.745, +1.745] | [-1.920, +1.920] | rad |
| 9 | `left_wrist_roll` | OpenArm 2.0 Left | `/left_arm_controller/follow_joint_trajectory` | [-2.618, +2.618] | [-2.880, +2.880] | rad |
| 10 | `left_wrist_yaw` | OpenArm 2.0 Left | `/left_arm_controller/follow_joint_trajectory` | [-1.745, +1.745] | [-1.920, +1.920] | rad |
| 11 | `left_gripper` | OpenArm Left Gripper | `/left_gripper_controller/gripper_cmd` | [0.000, +0.080] | [-0.004, +0.084] | m |
| 12 | `right_shoulder_pan` | OpenArm 2.0 Right | `/right_arm_controller/follow_joint_trajectory` | [-2.618, +2.618] | [-2.880, +2.880] | rad |
| 13 | `right_shoulder_lift`| OpenArm 2.0 Right | `/right_arm_controller/follow_joint_trajectory` | [-1.745, +1.745] | [-1.920, +1.920] | rad |
| 14 | `right_elbow_flex` | OpenArm 2.0 Right | `/right_arm_controller/follow_joint_trajectory` | [-2.618, +2.618] | [-2.880, +2.880] | rad |
| 15 | `right_wrist_flex` | OpenArm 2.0 Right | `/right_arm_controller/follow_joint_trajectory` | [-1.745, +1.745] | [-1.920, +1.920] | rad |
| 16 | `right_wrist_roll` | OpenArm 2.0 Right | `/right_arm_controller/follow_joint_trajectory` | [-2.618, +2.618] | [-2.880, +2.880] | rad |
| 17 | `right_wrist_yaw` | OpenArm 2.0 Right | `/right_arm_controller/follow_joint_trajectory` | [-1.745, +1.745] | [-1.920, +1.920] | rad |
| 18 | `right_gripper` | OpenArm Right Gripper| `/right_gripper_controller/gripper_cmd` | [0.000, +0.080] | [-0.004, +0.084] | m |

---

## 4. Verification & Test Suite Matrix

### 4.1 Phase 1 Contract Validator Checks (26 / 26 Satisfied)

Executing `python scripts/validate_dataset_contract.py --info-json meta/info.json --expected-state-dim 19` performs 26 checks:

1. `codebase_version == "v3.0"` (HC-3) — PASS
2. `robot_type == "openamrobot_2"` — PASS
3. `fps == 30` — PASS
4. `observation.state.dtype == "float32"` — PASS
5. `observation.state.shape == [19]` — PASS
6. `observation.state.names` flat list of strings — PASS
7. `observation.state.names` matches rev18 ordering — PASS
8. `action.dtype == "float32"` — PASS
9. `action.shape == [19]` — PASS
10. `action.names` flat list of strings — PASS
11. `action.names` matches rev18 ordering — PASS
12. `observation.images.camera_wrist_left` video [480, 640, 3] — PASS
13. `observation.images.camera_wrist_left` libx264 30 fps yuv420p info — PASS
14. `observation.images.camera_wrist_right` video [480, 640, 3] — PASS
15. `observation.images.camera_wrist_right` libx264 30 fps yuv420p info — PASS
16. `observation.images.head` video [720, 1280, 3] — PASS
17. `observation.images.head` libx264 30 fps yuv420p info — PASS
18. `observation.images.base` video [480, 640, 3] — PASS
19. `observation.images.base` libx264 30 fps yuv420p info — PASS
20. `timestamp` float64 [1] host monotonic — PASS
21. `timestamp_device` float64 [1] device native — PASS
22. `episode_index` int64 [1] — PASS
23. `frame_index` int64 [1] — PASS
24. `index` int64 [1] — PASS
25. `task_index` int64 [1] — PASS
26. `subtask_index` int64 [1] — PASS

### 4.2 Full PyTest Acceptance Suite (37 / 37 Green)

Executing `python -m pytest tests/ -v`:

| Test Module | Test Name | Target Assertion | Status |
|---|---|---|:---:|
| `test_c10_fixtures.py` | `test_clean_episode_passes_preflight` | Clean episode passes preflight gate with OPEN | **PASS** |
| `test_c10_fixtures.py` | `test_fix_ts_jitter_rejected[seed]` (x3) | Non-monotonic timestamps trigger MONOTONICITY_VIOLATION | **PASS** |
| `test_c10_fixtures.py` | `test_fix_drop_frame_rejected` | Video frame count mismatch triggers FRAME_COUNT_MISMATCH | **PASS** |
| `test_c10_fixtures.py` | `test_fix_phase_lag_flagged` | Shifted action peaks trigger PHASE_LAG | **PASS** |
| `test_c10_fixtures.py` | `test_fix_frozen_cam_flagged` | Frozen video frames trigger FROZEN_CAMERA | **PASS** |
| `test_c10_fixtures.py` | `test_fix_trunc_parquet_rejected` | Corrupted Parquet footer triggers READ_FAILURE | **PASS** |
| `test_c10_fixtures.py` | `test_fix_dim_mismatch_rejected[dim]` (x2) | Non-19 channel width triggers SCHEMA_VIOLATION | **PASS** |
| `test_c10_fixtures.py` | `test_fix_oob_stats_rejected[seed]` (x2) | NaN / Inf in channels triggers NON_FINITE_VALUE | **PASS** |
| `test_c10_fixtures.py` | `test_fix_orphan_task_rejected` | Invalid task index triggers ORPHAN_TASK | **PASS** |
| `test_c10_fixtures.py` | `test_skew_threshold_exceeded_rejected` | Skew > 30.0 ms triggers SKEW_THRESHOLD_EXCEEDED | **PASS** |
| `test_c10_fixtures.py` | `test_source_profile_mismatch_rejected` | Profile mismatch triggers SOURCE_PROFILE_MISMATCH | **PASS** |
| `test_c10_fixtures.py` | `test_invalid_source_profile_rejected` | Non-enum profile triggers INVALID_SOURCE_PROFILE | **PASS** |
| `test_c10_fixtures.py` | `test_episode_too_short_rejected` | Rows < 30 frames triggers EPISODE_TOO_SHORT | **PASS** |
| `test_c11_pipeline.py` | `test_dataset_generator_contract_and_qa`| Multi-episode generator produces valid LeRobotDataset | **PASS** |
| `test_c11_pipeline.py` | `test_act_policy_architecture_shapes` | ACT CVAE encoder/decoder tensor shapes match [19] | **PASS** |
| `test_c11_pipeline.py` | `test_act_training_and_eval_pipeline` | Smoke-test training, checkpoint save & eval report | **PASS** |
| `test_c11_pipeline.py` | `test_preflight_gate_blocking_prevents_training` | Corrupted data triggers C10 gate training abort | **PASS** |
| `test_c4_and_limits.py` | `test_check_action_bounds_clean_passes` | Clean action trajectory produces zero findings | **PASS** |
| `test_c4_and_limits.py` | `test_validate_episode_clean_passes` | Clean episode passes action bounds validation | **PASS** |
| `test_c4_and_limits.py` | `test_check_action_bounds_violation_triggers_reject` | Action > 5.1% triggers ACTION_OUT_OF_BOUNDS | **PASS** |
| `test_c4_and_limits.py` | `test_check_action_bounds_within_5pct_does_not_reject` | Action within 4.9% tolerance does not reject | **PASS** |
| `test_c4_and_limits.py` | `test_zero_span_channel_eps_tolerance` | Zero-span mast channels accept exact zeros | **PASS** |
| `test_c4_and_limits.py` | `test_zero_span_channel_excessive_value_rejected` | Non-zero mast command triggers REJECT | **PASS** |
| `test_c4_and_limits.py` | `test_validate_episode_oob_action_rejected` | Full episode validate with OOB action rejects | **PASS** |
| `test_c4_and_limits.py` | `test_dry_run_valid_episode_exits_zero` | Replay dry-run on valid Parquet returns exit 0 | **PASS** |
| `test_c4_and_limits.py` | `test_dry_run_oob_episode_exits_one` | Replay dry-run on OOB Parquet returns exit 1 | **PASS** |
| `test_c4_and_limits.py` | `test_dry_run_cli_valid_episode` | Replay CLI entrypoint returns exit 0 | **PASS** |
| `test_c4_and_limits.py` | `test_load_episode_parquet_schema_error` | Missing action column raises descriptive RuntimeError | **PASS** |
| `test_integration_robustness.py`| `test_mutation_joint_name_raises_contract_fail` | In-memory schema mutation causes contract FAIL | **PASS** |
| `test_integration_robustness.py`| `test_urdf_joints_match_meta_info` | A6 upperbody URDF xacro structural anchors present | **PASS** |
| `test_integration_robustness.py`| `test_action_chunk_boundary_continuity` | 100-step chunk continuity & temporal ensembling | **PASS** |
| `test_integration_robustness.py`| `test_missing_camera_or_zeroed_timestamp_halts_safely` | Missing/stale inputs trigger REJECT | **PASS** |
| `test_integration_robustness.py`| `test_evaluation_provenance_hash_reproducibility` | Evaluation metrics & SHA-256 byte-identical | **PASS** |

---

## 5. Deliverable C11 Provenance & Governance Package

### 5.1 Immutable Provenance Block

```json
{
  "deliverable": "C11 Held-Out Policy Evaluation & ACT Baseline",
  "reviewer": "Mohamed Sayed",
  "schema_revision": "schema-v1.0-frozen",
  "robot_profile": "openamrobot-fixed-mast-19dof",
  "base_git_commit": "a29ce7c89b7c42c9147ad81e6a751de9dad045c9",
  "checkpoint_path": "checkpoints/act_openamrobot_19dof_best.pt",
  "checkpoint_sha256": "26703aa894590a8e853fdf455522cf27f238c0524818d0af65cf045bfc7ca3f8",
  "torch_version": "2.14.1+cpu (dev) / 2.5.1 (target ROS 2 Jazzy)",
  "lerobot_adapter": "0.4.1 (targeting C9 snapshot v0.6.1 commit 7e241bd630a3)",
  "c10_preflight_gate": "PASS (training_gate: OPEN)",
  "episodes": {
    "total": 20,
    "train": 16,
    "held_out": 4,
    "held_out_indices": [16, 17, 18, 19]
  },
  "action_prediction_error": {
    "overall_mae": 0.6720,
    "overall_mse": 0.6990
  },
  "rollout_results": {
    "evaluated": 4,
    "successful": 0,
    "wilson_95_ci": [0.0, 0.4899],
    "terminal_phase": "RETRACT"
  }
}
```

### 5.2 Integration Gate Record

| Evaluation Area | Selected Approach / Library | Rejected Alternative | Technical Rationale |
|---|---|---|---|
| Dataset Schema | `LeRobotDataset v3.0` | Custom ROS bag format / HDF5 | Mandated by HC-3; enables zero-copy Parquet + MP4 LeRobot policy training. |
| Imitation Learning | Action Chunking with Transformers (ACT) | Diffusion Policy / BC-MLP | Chunking (100 steps) with CVAE models multi-modal manipulation transitions robustly. |
| Boundary Smoothing | Temporal Ensembling (`lambda=0.01`) | Raw single-chunk execution | Eliminates 30 Hz velocity jumps across 100-step boundaries in ~15 lines of NumPy. |
| Confidence Intervals | Wilson Score Interval (`statsmodels`) | Normal approximation / Wald interval | Asymptotically valid coverage for small sample sizes and boundary success rates (0%). |
| Base Safety Routing | `/cmd_vel_nav` (twist_mux + smoother) | Direct motor CAN frames | Enforces Section 10 rule: no AI-authored safety or watchdog nodes. |

### 5.3 Mandatory "Not Verified" Checklist

Per Section 10 Working Rules, the following boundaries have **not** been verified and are flagged for cell testing:

1. **Physical ROBOT Demonstrations**: Current 20 episodes are synthetic (`source_profile: GAZEBO`). Ingest of physical robot demonstration bags from the cell is pending hardware recording.
2. **Closed-Loop Gazebo Simulation**: ACT policy has been verified offline (chunk prediction MAE/MSE + temporal ensembling). Full closed-loop rollout in Gazebo via `gz_ros2_control` is pending cell bringup.
3. **Hardware Torque Limits**: Arm joint limits reflect kinematic CAD; servo current/torque limits will be calibrated during Phase 3 physical cell integration.
4. **LeRobot v0.6.1 PyPI Release**: Upstream commit `7e241bd630a3` is unreleased on PyPI; baseline utilizes the verified `lerobot==0.4.1` adapter.
5. **Partner Build Anonymity**: All references to external partner builds strictly adhere to the `Use_Case_1` identifier.

### 5.4 AI Assistance Disclosure

> In compliance with OpenAMRobot 2.0 Team Ground Truth Section 10:  
> This software baseline was designed, cross-validated, and documented with the assistance of Antigravity AI (Gemini). All generated code, test suites, and schema definitions have been verified by human contributor Marcus Andrei Guinto (`@4u-79`). No safety-critical watchdog, E-stop, or brake software was authored by an AI agent.

### 5.5 Developer Certificate of Origin (DCO) Sign-Off

```text
Signed-off-by: Marcus Andrei Guinto <mavguinto@gmail.com>
```
