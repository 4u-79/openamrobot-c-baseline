# OpenAMRobot 2.0 — Workstream C: Dataset Contract, QA Gate & C11 ACT Baseline

Embodied AI & Imitation Learning team repository for OpenAMRobot 2.0 (Workstream C, Division B).
This repository holds the frozen dataset contract, contract validator, C10 pre-flight QA gate (training gate),
deterministic multi-episode dataset ingest/generator, the C11 ACT imitation learning baseline pipeline,
downstream C4 replay integration, and the complete 37-test verification suite.

## Project Control & Specifications

| Field | Value |
|---|---|
| Project Plan | `C-TEAM-PLAN-01 rev 2.0` (21 Sep – 20 Nov 2026) |
| Governing Documents | `P-00 rev18` · `P-02 rev18` · `Q-01 rev18` · `A-01 rev18` |
| Baseline Configuration | OpenAMRobot 2.0 `P-03 rev18.2` (Fixed Mast) & `rev19.6` (Option B Context) |
| Frozen Schema Tag | `schema-v1.0-frozen` (`openamrobot-fixed-mast-19dof`) |
| Dataset Layout | `LeRobotDataset v3.0` (`codebase_version: "v3.0"`, HC-3 canonical, zero proprietary formats) |
| LeRobot Software Stack | `lerobot==0.4.1` (runtime adapter targeting unpinned C9 `v0.6.1` snapshot `7e241bd630a3`) |
| Robot Target | `openamrobot_2 @ 30 fps` |
| Primary Contributor | Marcus Andrei Guinto (`@4u-79`, `mavguinto@gmail.com`), Embodied AI & Data, Division B |
| Primary Reviewer | Mohamed Sayed (`@panthera-momagdii`), Software Lead / Division B Lead |

---

## Phase Status & Milestone Tracker

| Phase | Target Date | Milestone | Deliverables | Current Verified Status |
|---|---|---|---|---|
| Phase 1 | 9 Oct 2026 | Schema & Contract Freeze | C2 Whole-Body Schema, C9 LeRobot Contract | **CLOSED / FROZEN** (`schema-v1.0-frozen`, 26/26 contract checks pass) |
| Phase 2 | 16 Oct 2026 | Dataset Integrity & Pre-Flight Gate | C10 QA Pre-Flight Harness | **CLOSED / VERIFIED** (Gate OPEN, 20/20 episodes pass, 8 fixtures verified) |
| Phase 3 | 30 Oct / 6 Nov | Imitation Learning Baseline & Replay | C4 Replay Verification, C11 ACT Policy | **IMPLEMENTED / TESTED** (C4 dry-run green, C11 ACT trained & evaluated, 37/37 tests pass) |
| Phase 4 | 20 Nov 2026 | Phase Tagging & Policy Promotion | C13 Phase Tagging, C14 50-Trial Rollout Gate | **SKELETON READY** (8-phase taxonomy wired, Wilson 95% CIs calculated) |

---

## Complete Repository Layout Tree

```text
.
├── AGENTS.md                               # Section 10 Working Rules & governance requirements
├── CLAUDE.md                               # AI agent quick-reference importing AGENTS.md
├── AUDIT_AND_ALIGNMENT_REPORT.md           # Comprehensive technical cross-repository audit report
├── README.md                               # Repository documentation & usage guide
├── requirements.txt                        # Strict dependency pins (torch, pyarrow, opencv, etc.)
├── checkpoints/
│   └── act_openamrobot_19dof_best.pt       # Trained ACT policy checkpoint (SHA-256 verified)
├── configs/
│   └── act_openamrobot_19dof.yaml          # Frozen C11 ACT policy architecture & training hyperparameters
├── data/
│   └── episodes/                           # Canonical LeRobotDataset v3.0 tree (20 episodes: 16 train, 4 eval)
│       ├── meta/                           # info.json, tasks.jsonl, episodes.jsonl, subtasks.jsonl
│       ├── data/                           # chunk-000 Parquet data files
│       └── videos/                         # chunk-000 MP4 video streams for 4 C9 cameras
├── meta/
│   ├── info.json                           # Canonical C9-aligned schema metadata (single source of truth)
│   └── subtasks.jsonl                      # C13 subtask phase taxonomy (phases 0..7)
├── reports/
│   ├── qa_report.json                      # C10 pre-flight integrity gate evidence (Gate OPEN)
│   ├── c11_training_summary.json           # C11 ACT training curve & episode split metrics
│   ├── c11_eval_report.json                # C11 held-out evaluation report (per-channel MAE/MSE, Wilson CI)
│   ├── c11_evidence_summary.md             # Markdown evidence summary for Mohamed Sayed
│   └── training_loss.json                  # Per-epoch training & validation loss history
├── scripts/
│   ├── validate_dataset_contract.py        # Phase 1 gate: validates meta/info.json against rev18 schema
│   ├── check_dataset_integrity.py          # Phase 2 gate: C10 pre-flight QA gate (enforces bounds & skew)
│   ├── generate_or_ingest_dataset.py       # C11 dataset generator & ROS 2 bag ingest bridge
│   ├── train_act_baseline.py               # C11 ACT imitation learning policy training pipeline
│   ├── evaluate_act_baseline.py            # C11 held-out policy evaluation with temporal ensembling
│   ├── replay_episode.py                   # C4 30 Hz deterministic replay node (/cmd_vel_nav arbitrated)
│   └── run_dashboard.py                    # Local verification UI portal (http://127.0.0.1:8000)
└── tests/
    ├── test_c10_fixtures.py                # 8 mandatory fault-injection fixtures (17 tests)
    ├── test_c4_and_limits.py               # C4 replay dry-run and joint action bounds checks (11 tests)
    ├── test_c11_pipeline.py                # C11 ACT neural architecture, forward/backward, C10 gate (4 tests)
    └── test_integration_robustness.py      # 5 Section 10 integration-robustness invariant tests (5 tests)
```

---

## Frozen Contract Quick Reference (`schema-v1.0-frozen`)

### 1. 19-Channel State & Action Vector Ordering

Per `P-03 Addendum rev18.2`, `observation.state` and `action` share an identical, flat 1-D `float32[19]` vector structure.
Torso channels (`torso_pitch`, `torso_yaw`) are unactuated structural mast joints with physical limits strictly `[0.0, 0.0]`.

| Index | Channel Name | Type | Physical Limits | Units | Description |
|---|---|---|---|---|---|
| 0 | `base_linear_x` | Mobile Base | [-0.20, +0.20] | m/s | Base forward/reverse linear velocity |
| 1 | `base_linear_y` | Mobile Base | [0.00, 0.00] | m/s | Non-holonomic diff-drive constraint (must remain 0.0) |
| 2 | `base_angular_z` | Mobile Base | [-0.50, +0.50] | rad/s | Base rotational velocity (yaw rate) |
| 3 | `torso_pitch` | Fixed Mast | [0.00, 0.00] | rad | Fixed mast rev18.2 pitch constraint (zero-span) |
| 4 | `torso_yaw` | Fixed Mast | [0.00, 0.00] | rad | Fixed mast rev18.2 yaw constraint (zero-span) |
| 5 | `left_shoulder_pan` | Left Arm | [-2.618, +2.618] | rad | Left arm J1 shoulder pan |
| 6 | `left_shoulder_lift` | Left Arm | [-1.745, +1.745] | rad | Left arm J2 shoulder lift |
| 7 | `left_elbow_flex` | Left Arm | [-2.618, +2.618] | rad | Left arm J3 elbow flex |
| 8 | `left_wrist_flex` | Left Arm | [-1.745, +1.745] | rad | Left arm J4 wrist flex |
| 9 | `left_wrist_roll` | Left Arm | [-2.618, +2.618] | rad | Left arm J5 wrist roll |
| 10 | `left_wrist_yaw` | Left Arm | [-1.745, +1.745] | rad | Left arm J6 wrist yaw |
| 11 | `left_gripper` | Left End-Effector | [0.000, +0.080] | m / rad | Left parallel gripper stroke |
| 12 | `right_shoulder_pan` | Right Arm | [-2.618, +2.618] | rad | Right arm J1 shoulder pan |
| 13 | `right_shoulder_lift` | Right Arm | [-1.745, +1.745] | rad | Right arm J2 shoulder lift |
| 14 | `right_elbow_flex` | Right Arm | [-2.618, +2.618] | rad | Right arm J3 elbow flex |
| 15 | `right_wrist_flex` | Right Arm | [-1.745, +1.745] | rad | Right arm J4 wrist flex |
| 16 | `right_wrist_roll` | Right Arm | [-2.618, +2.618] | rad | Right arm J5 wrist roll |
| 17 | `right_wrist_yaw` | Right Arm | [-1.745, +1.745] | rad | Right arm J6 wrist yaw |
| 18 | `right_gripper` | Right End-Effector | [0.000, +0.080] | m / rad | Right parallel gripper stroke |

### 2. C9-Aligned Camera Streams

All camera streams are encoded in `libx264`, `30.0 fps`, pixel format `yuv420p`.

| Feature Key | Resolution (H×W×C) | Encoding | Physical Hardware & Role |
|---|---|---|---|
| `observation.images.camera_wrist_left` | 480 × 640 × 3 | libx264 @ 30 fps, yuv420p | Left end-effector wrist camera |
| `observation.images.camera_wrist_right` | 480 × 640 × 3 | libx264 @ 30 fps, yuv420p | Right end-effector wrist camera |
| `observation.images.head` | 720 × 1280 × 3 | libx264 @ 30 fps, yuv420p | ZED-121210 scene camera mounted on upperbody carriage (replaces legacy `camera_scene`) |
| `observation.images.base` | 480 × 640 × 3 | libx264 @ 30 fps, yuv420p | Orbbec Gemini 336L base navigation camera (replaces legacy `camera_base_nav`) |

### 3. Profile Compatibility Matrix

| Profile Identifier | Channels | Units | Lift Support | Torso Mast | Target Hardware / Subsystem | Status in Baseline |
|---|:---:|---|---|---|---|---|
| `openamrobot-fixed-mast-19dof` | **19** | rad, m/s | None | Fixed rev18.2 (`[0,0]`) | Base (3) + Mast (2) + Dual OpenArm (14) | **Normative Baseline (`schema-v1.0-frozen`)** |
| `lerobot-v0.6.1-bi-openarm-position` | **16** | degrees | None | None | Dual OpenArm position-only (C9 contract) | Upstream reference fixture |
| `openamrobot-lift-option-b-20dof` | **20** | rad, m/s | `lift_joint` (0.35m) | Prismatic Carriage | DOLD Hexalift V4 + Dual 7-DoF OpenArm | P-03 Option B migration branch |

### 4. Dual Timebase (HC-5) & Skew Gate

- `timestamp`: Host monotonic timestamp (`jetson_host_monotonic_sec`, `float64`, strictly increasing).
- `timestamp_device`: Sensor native timestamp (`device_native_sec`, `float64`, strictly increasing).
- Skew threshold: `|timestamp - timestamp_device| <= 30.0 ms` (Workstream C provisional pre-flight gate; numerical sync tolerance flagged `pending_evidence: true` per C9 contract).

### 5. Source Profiles (HC-8)

- `ROBOT`: Real physical hardware demonstration recordings.
- `GAZEBO`: Gazebo multi-robot physics simulation episodes (`source_profile: GAZEBO`).
- `ISAAC`: Isaac Sim / Isaac Lab synthetic demonstrations.
- `CALIBRATION`: Kinematic & hand-eye calibration trajectories.
- `DIAGNOSTICS`: Unit test & benchmark synthetic fixtures.

### 6. Subtask Phase Taxonomy (C13)

- `0`: IDLE
- `1`: APPROACH
- `2`: ALIGN
- `3`: GRASP
- `4`: TRANSPORT
- `5`: PLACE
- `6`: RETRACT
- `7`: FAILURE_RECOVERY

---

## Setup & End-to-End CLI Usage Guide

### 1. Environment Setup

```bash
# Clone and enter workspace
cd openamrobot_ws/openamrobot-c-baseline

# Set up Python virtual environment (Python 3.10 / 3.12 compatible)
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Generation & Ingest Bridge

```bash
# Generate 20 synthetic/Gazebo-profile episodes with 4 C9 video streams
python scripts/generate_or_ingest_dataset.py \
    --output-dir data/episodes \
    --num-episodes 20 \
    --episode-length 120 \
    --source-profile GAZEBO

# Or ingest raw ROS 2 bag recordings:
python scripts/generate_or_ingest_dataset.py \
    --bag-path /data/bags/real_demonstrations \
    --output-dir data/episodes \
    --source-profile ROBOT
```

### 3. Phase 1 Contract Validation (C2 / C9)

```bash
# Contract-only check against meta/info.json
python scripts/validate_dataset_contract.py --info-json meta/info.json --expected-state-dim 19

# Full contract + Parquet + MP4 stream data validation
python scripts/validate_dataset_contract.py \
    --dataset-path data/episodes \
    --skew-threshold-ms 30.0 \
    --expected-state-dim 19
```

### 4. Phase 2 C10 Pre-Flight QA Gate

```bash
python scripts/check_dataset_integrity.py \
    --dataset-path data/episodes \
    --skew-threshold-ms 30.0 \
    --min-frames 30 \
    --require-source-profile GAZEBO \
    --output-report reports/qa_report.json
```

**C10 Exit Code Semantics:**

| Exit Code | Meaning | Action Taken |
|---|---|---|
| `0` | **Gate OPEN** | All episodes pass; zero REJECT findings; training unblocked. |
| `1` | **Gate BLOCKED** | One or more REJECT findings detected; training immediately aborted. |
| `2` | **Usage / Env Error** | Malformed CLI arguments or missing files. |

### 5. C4 Deterministic Replay (30 Hz)

```bash
# Offline dry-run verification (checks joint limits & diff-drive constraint)
python scripts/replay_episode.py --parquet data/episodes/data/chunk-000/episode_000000.parquet --dry-run

# Live ROS 2 execution via arbitrated velocity smoother & collision monitor
python scripts/replay_episode.py \
    --parquet data/episodes/data/chunk-000/episode_000000.parquet \
    --topic /cmd_vel_nav \
    --rate 30.0
```

### 6. C11 ACT Policy Training & Held-Out Evaluation

```bash
# Fast CPU smoke-test (3 epochs, C10 pre-flight gate check, checkpoint save)
python scripts/train_act_baseline.py \
    --dataset-path data/episodes \
    --config configs/act_openamrobot_19dof.yaml \
    --output-checkpoint checkpoints/act_openamrobot_19dof_best.pt \
    --smoke-test

# Evaluate policy checkpoint on held-out episodes (evaluates MAE/MSE + temporal ensembling)
python scripts/evaluate_act_baseline.py \
    --checkpoint checkpoints/act_openamrobot_19dof_best.pt \
    --dataset-path data/episodes
```

### 7. Run Full 37-Test Suite

```bash
python -m pytest tests/ -v
```

#### Mandatory 8 Fault-Injection Fixtures (`tests/test_c10_fixtures.py`)

| # | Fixture ID | Injected Fault | Expected Severity | Asserted Diagnostic Code |
|---|---|---|---|---|
| 1 | `FIX-TS-JITTER` | Timestamp non-monotonic at random frames | `REJECT` | `MONOTONICITY_VIOLATION` |
| 2 | `FIX-DROP-FRAME` | MP4 video container truncated by 5 frames | `REJECT` | `FRAME_COUNT_MISMATCH` |
| 3 | `FIX-PHASE-LAG` | Action vector shifted +5 steps vs state | `FLAG` | `PHASE_LAG` |
| 4 | `FIX-FROZEN-CAM` | Identical video frames for >= 2.0 seconds | `FLAG` | `FROZEN_CAMERA` |
| 5 | `FIX-TRUNC-PARQUET` | Last 100 bytes of Parquet footer truncated | `REJECT` | `READ_FAILURE` |
| 6 | `FIX-DIM-MISMATCH` | Vector width 18 or 20 instead of 19 | `REJECT` | `SCHEMA_VIOLATION` |
| 7 | `FIX-OOB-STATS` | NaN / +Inf / -Inf injected into channels | `REJECT` | `NON_FINITE_VALUE` |
| 8 | `FIX-ORPHAN-TASK` | `task_index: 99` not present in tasks.jsonl | `REJECT` | `ORPHAN_TASK` |

#### Section 10 Integration-Robustness Invariants (`tests/test_integration_robustness.py`)

| # | Test Name | What It Catches | Why It Matters |
|---|---|---|---|
| 1 | `test_mutation_joint_name_raises_contract_fail` | Swaps joint name & alters camera resolution in memory; asserts `validate_contract()` fails. | Team Ground Truth Section 10 rule: every PR has a test that fails when the change is reverted. |
| 2 | `test_urdf_joints_match_meta_info` | Parses `openamr-upperbody-sw` xacro for `lift_joint` and OpenArm base mounts. | Prevents schema/URDF joint mismatches across the A6 x C9 boundary. |
| 3 | `test_action_chunk_boundary_continuity` | Validates ACT chunk prediction shape and verifies the 15-line temporal ensembling buffer (lambda=0.01). | Eliminates physical arm jerks across 100-step boundaries at 30 Hz deployment. |
| 4 | `test_missing_camera_or_zeroed_timestamp_halts_safely` | Simulates dropped camera, zeroed timestamp, or NaN actions; asserts gate emits `REJECT`. | Division B invariant: missing or stale inputs never read as healthy data. |
| 5 | `test_evaluation_provenance_hash_reproducibility` | Runs evaluation twice on held-out episodes; asserts SHA-256 and MAE/MSE metrics are byte-identical. | Proves deterministic evaluation reproducibility for tech lead review. |

### 8. Local Verification UI Portal

```bash
# Launch portal on http://127.0.0.1:8000
python scripts/run_dashboard.py --port 8000 --open
```

---

## Expected Dataset Layout (LeRobotDataset v3.0)

```text
data/episodes/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   ├── episodes.jsonl
│   ├── subtasks.jsonl
│   └── episodes_stats.jsonl
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.camera_wrist_left/
        ├── observation.images.camera_wrist_right/
        ├── observation.images.head/
        └── observation.images.base/
```

---

## Dependency Pinning Policy (HC-1, HC-11)

| Dependency | Pin | License | Policy Rationale |
|---|---|---|---|
| `lerobot` | `0.4.1` | Apache-2.0 | Runtime adapter for LeRobotDataset v3.0 targeting upstream C9 snapshot (`v0.6.1`). |
| `torch` | `2.5.1` (target) / `2.14.1+cpu` (dev) | BSD-3-Clause | PyTorch tensor compute and ACT CVAE neural model. |
| `torchvision` | `0.20.1` | BSD-3-Clause | Vision backbone and video processing transforms. |
| `pyarrow` | `>=14.0.0` | Apache-2.0 | Parquet deserialization with fast column-level vector extraction. |
| `opencv-python` | `>=4.8.0` | Apache-2.0 | Video frame entropy decoding for frozen camera detection. |
| `statsmodels` | `>=0.14.0` | BSD-3-Clause | Statistically sound Wilson score 95% confidence intervals for rollout success rates. |
| `pytest` | `>=7.0.0` | MIT | Acceptance test harness for all 37 unit and fixture tests. |

---

## Q-01 Evidence Mapping

| Q-01 Requirement | Evidence Artifact | Provided By |
|---|---|---|
| Required-field validation | Contract validator summary report | `scripts/validate_dataset_contract.py` |
| Measured sync/skew report | `reports/qa_report.json` | `scripts/check_dataset_integrity.py` |
| Dropped/stale frame statistics | Per-camera freshness in `qa_report.json` | `scripts/check_dataset_integrity.py` |
| Corrupted episode rejection | 8 fixture test results (17 assertions) | `tests/test_c10_fixtures.py` |
| Imitation learning policy | Checkpoint SHA-256 and loss curves | `scripts/train_act_baseline.py` |
| Held-out policy evaluation | `reports/c11_eval_report.json` | `scripts/evaluate_act_baseline.py` |
| Integration robustness invariants | 5 Section 10 tests green | `tests/test_integration_robustness.py` |

---

## Schema-Freeze Note (21 → 19 Resolution)

Rev18 originally declared `shape: [21]` while enumerating only 19 channel names.
This discrepancy was resolved during Phase 1: the physical robot utilizes a fixed mast (`P-03 Addendum rev18.2`) where
`torso_pitch` and `torso_yaw` are unactuated structural mast joints with physical limits strictly `[0.0, 0.0]`.
The two apparent "missing" channels were never real. The contract is permanently frozen at **19 channels**.

---

## Mandatory "Not Verified" / Known Follow-Ups

Per Section 10 Working Rules, every PR must clearly delineate what has and has not been verified:

1. **Physical ROBOT Bag Ingest**: All 20 current episodes are synthetic (`source_profile: GAZEBO`). Ingest of physical robot demonstration bags from the cell is pending bag delivery.
2. **Closed-Loop Gazebo Execution**: The ACT policy has been verified offline (trajectory MAE/MSE error + temporal ensembling). Closed-loop rollout inside `gz_ros2_control` is pending cell bringup.
3. **Hardware Arm Joint Limits**: Kinematic limits reflect `openamr-upperbody-sw` and `openamrobot-manipulation` specs; physical torque limits on real servos will be calibrated during Phase 3.
4. **LeRobot v0.6.1 Manifest Release**: The upstream snapshot `7e241bd630a3` is unreleased on PyPI; baseline utilizes the verified `lerobot==0.4.1` adapter.
5. **Partner Anonymity**: All references to external partner builds strictly adhere to the `Use_Case_1` identifier.
