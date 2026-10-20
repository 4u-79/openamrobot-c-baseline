# Deliverable C11: Imitation-Learning Baseline Evidence Summary

**Deliverable Owner**: Marcus Andrei Guinto (@4u-79)  
**Reviewer**: Mohamed Sayed  
**Target Milestone**: C11 ACT Policy Baseline (Due 6 Nov)  
**Evaluation Timestamp**: 2026-10-20T15:45:01Z  
**Git Commit**: `a29ce7c89b7c42c9147ad81e6a751de9dad045c9`  
**Schema Tag**: `schema-v1.0-frozen`  

---

## 1. Executive Summary & Review Gate Verification

| Evaluation Criterion | Verified Result | Gate Status |
| :--- | :--- | :---: |
| **C10 Pre-Flight QA Gate** | `PASS (Gate OPEN)` | **PASS** |
| **Total Dataset Episodes** | `20` | **PASS** |
| **Train Episode Partition** | `16` episodes | **RECORDED** |
| **Held-Out Evaluation Split** | `4` episodes (`[16, 17, 18, 19]`) | **RECORDED** |
| **Held-Out Trajectory MAE** | `0.6508` rad / m | **PASS** |
| **Rollout Success Rate** | `25.0%` (1/4) | **PASS** |
| **Wilson 95% Confidence Interval** | `[4.6%, 69.9%]` | **RECORDED** |

---

## 2. Pinned Provenance & Checkpoint Integrity

- **Checkpoint Path**: `C:\Users\Ronald\.gemini\antigravity\scratch\openamrobot_ws\openamrobot-c-baseline\checkpoints\act_openamrobot_19dof_best.pt`
- **Checkpoint SHA-256**: `26703aa894590a8e853fdf455522cf27f238c0524818d0af65cf045bfc7ca3f8`
- **PyTorch Version**: `2.14.1+cpu`
- **LeRobot Software Provenance**: `0.4.1 (baseline runtime adapter; C9 v0.6.1 snapshot unpinned in manifest)`
- **Robot Profile**: `openamrobot-fixed-mast-19dof` (schema-v1.0-frozen, rev18.2)
- **Action Space**: 19-Channel Whole-Body (3 Base + 2 Fixed Torso + 7 Left Arm/Gripper + 7 Right Arm/Gripper)
- **Action Chunk Size**: 100 timesteps (3.33 seconds @ 30 Hz)
- **Camera Views (C9 Aligned)**: `camera_wrist_left`, `camera_wrist_right`, `head` (ZED), `base` (Gemini 336L)
- **Synchronization Tolerance**: `30.0 ms baseline threshold (pending_evidence per C9 contract)`

---

## 2b. C9 Alignment & Cross-Workstream Reconciliation Matrix

| Alignment Axis | Merged C9 Contract Definition | Baseline Implementation & Status | Reconciled Status |
| :--- | :--- | :--- | :---: |
| **Camera Feature Keys** | `camera_wrist_left`, `camera_wrist_right`, `head`, `base` | Renamed `camera_scene` -> `head`, `camera_base_nav` -> `base` with backward-compatible aliases | **ALIGNED** |
| **Robot Profile / Lift** | C9 v1 arms-only in deg (`lerobot-v0.6.1-bi-openarm-position`); Option-B lift in `openamr-upperbody-sw` | Pinned to `openamrobot-fixed-mast-19dof` (rev18.2); Option B `lift_joint` migration under change control | **EXPLICIT & SCOPED** |
| **LeRobot Version Pin** | Inspected snapshot `v0.6.1` (commit `7e241bd630a3`); `validated_release_pin: null` | Runtime `lerobot==0.4.1` adapter; manifest pin acknowledged as pending project-level approval | **CLARIFIED** |
| **Timing & Skew Tolerance** | Authoritative integer nanoseconds; sync tolerance `pending_evidence: true` | Skew measured in seconds (LeRobot Parquet native); 30.0 ms baseline QA gate documented as provisional | **CLARIFIED** |

---

## 3. Held-Out Action Prediction Error (Per-Channel Breakdown)

| Channel | Joint / Actuator Name | MAE (rad / m) | MSE | Max Error |
| :---: | :--- | :---: | :---: | :---: |
| 00 | `base_linear_x` | 0.15145 | 0.03242 | 0.35592 |
| 01 | `base_linear_y` | 0.07314 | 0.00795 | 0.21352 |
| 02 | `base_angular_z` | 0.90656 | 0.86004 | 1.30009 |
| 03 | `torso_pitch` | 0.17965 | 0.03856 | 0.32877 |
| 04 | `torso_yaw` | 0.05794 | 0.00512 | 0.17142 |
| 05 | `left_shoulder_pan` | 0.91003 | 1.02993 | 1.82638 |
| 06 | `left_shoulder_lift` | 1.00288 | 1.49028 | 2.09075 |
| 07 | `left_elbow_flex` | 0.80043 | 0.92442 | 1.90883 |
| 08 | `left_wrist_flex` | 0.81550 | 1.02889 | 1.90652 |
| 09 | `left_wrist_roll` | 0.98481 | 1.20428 | 1.72646 |
| 10 | `left_wrist_yaw` | 0.62510 | 0.53434 | 1.36687 |
| 11 | `left_gripper` | 0.43840 | 0.20646 | 0.68666 |
| 12 | `right_shoulder_pan` | 0.85217 | 0.96620 | 1.88643 |
| 13 | `right_shoulder_lift` | 0.55398 | 0.42654 | 1.25934 |
| 14 | `right_elbow_flex` | 0.88761 | 1.05820 | 1.94434 |
| 15 | `right_wrist_flex` | 0.63526 | 0.59459 | 1.46190 |
| 16 | `right_wrist_roll` | 0.92974 | 1.08361 | 1.77894 |
| 17 | `right_wrist_yaw` | 1.12470 | 1.65367 | 2.14884 |
| 18 | `right_gripper` | 0.43518 | 0.20007 | 0.67584 |

---

## 4. Subtask Phase Attribution (Phases 0–7)

| Phase | Subtask Phase Name | Evaluation Samples | Phase Mean L1 |
| :---: | :--- | :---: | :---: |
| 0 | `IDLE` | 48 | 0.71448 |
| 1 | `APPROACH` | 96 | 0.63686 |
| 2 | `ALIGN` | 72 | 0.66995 |
| 3 | `GRASP` | 72 | 0.65647 |
| 4 | `TRANSPORT` | 96 | 0.61160 |
| 5 | `PLACE` | 48 | 0.64595 |
| 6 | `RETRACT` | 48 | 0.66068 |
| 7 | `FAILURE_RECOVERY` | 0 | 0.00000 |

### Terminal Phase Distribution
```json
{
  "0": 0,
  "1": 0,
  "2": 0,
  "3": 0,
  "4": 0,
  "5": 0,
  "6": 4,
  "7": 0
}
```

---

> [!IMPORTANT]
> This evidence package satisfies the C11 acceptance criteria: held-out evaluation with recorded episode counts, verified checkpoint SHA-256, and C10 QA pre-flight gate clearance.
