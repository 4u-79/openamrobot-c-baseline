# OpenAMRobot 2.0 — Workstream C: Dataset Contract & QA Tooling

Embodied AI & Imitation Learning team repository. This repo holds the frozen
dataset contract, the schema validator, the C10 pre-flight QA gate (training
gate), and the mandatory 8-fixture fault-injection suite.

| Field | Value |
|---|---|
| Plan | C-TEAM-PLAN-01 rev 2.0 (21 Sep – 20 Nov 2026) |
| Governing docs | P-00 rev18 · P-02 rev18 · Q-01 rev18 · A-01 rev18 |
| Plan baseline | OpenAMRobot 2.0 Plan of Record rev18 |
| Schema freeze | `schema-v1.0-frozen` — **19 channels**, fixed-mast design per P-03 Addendum rev18.2 |
| Dataset format | LeRobotDataset `codebase_version: "v3.0"` (HC-3 canonical, no proprietary formats) |
| Robot | `openamrobot_2` @ 30 fps |
| Reviewer | Mohamed Sayed (Division B Lead) |

---

## Phase status

| Phase | Milestone | Status |
|---|---|---|
| 1 — Schema & Contract Freeze (9 Oct) | C2 + C9 locked, `schema-v1.0-frozen` tagged | Done (19-channel resolution filed) |
| 2 — Dataset QA (16 Oct) | C10 harness green: 8 fixtures + pre-flight CLI | Done — awaiting the one real recorded episode pass |
| 3 — ACT Baseline + Replay (30 Oct / 6 Nov) | C4 replay, 50–100 clean episodes, C11 ACT trained | Not started |
| 4 — Phase Tagging + Evaluation (20 Nov) | C13 tagging, C14 50-trial gate + Wilson CIs | Not started |

---

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── meta/
│   ├── info.json                       # C2/C9 frozen contract (19-dim, 4 cameras, dual timebase)
│   └── subtasks.jsonl                  # C13 phase taxonomy (subtask_index 0..7)
├── scripts/
│   ├── validate_dataset_contract.py    # Phase 1 gate: schema/contract validation (C2/C9)
│   └── check_dataset_integrity.py      # Phase 2 gate: C10 pre-flight QA — blocks training
└── tests/
    └── test_c10_fixtures.py            # Mandatory 8-fixture fault-injection suite
```

---

## Frozen contract quick reference (schema-v1.0-frozen)

**State/action vectors (HC-7):** flat 1-D `float32`, length **19**, identical
ordering for `observation.state` and `action`. No nested dictionaries.

```text
 0 base_linear_x        7 left_elbow_flex       14 right_elbow_flex
 1 base_linear_y        8 left_wrist_flex       15 right_wrist_flex
 2 base_angular_z       9 left_wrist_roll       16 right_wrist_roll
 3 torso_pitch         10 left_wrist_yaw        17 right_wrist_yaw
 4 torso_yaw           11 left_gripper          18 right_gripper
 5 left_shoulder_pan   12 right_shoulder_pan
 6 left_shoulder_lift  13 right_shoulder_lift
```

**Camera keys (HC-6, fixed):**

| Key | Shape (H×W×C) | Encoding |
|---|---|---|
| `observation.images.camera_wrist_left` | 480×640×3 | libx264, 30 fps, yuv420p |
| `observation.images.camera_wrist_right` | 480×640×3 | libx264, 30 fps, yuv420p |
| `observation.images.camera_scene` | 720×1280×3 | libx264, 30 fps, yuv420p |
| `observation.images.camera_base_nav` | 480×640×3 | libx264, 30 fps, yuv420p |

**Timebase (HC-5):** `timestamp` = Jetson host monotonic clock
(`jetson_host_monotonic_sec`), `timestamp_device` = device-native
(`device_native_sec`); both `float64`, per frame, epoch-aligned by ingest so
`|timestamp − timestamp_device|` is the measured skew (default threshold
30 ms, configurable).

**Source profiles (HC-8, never silently mixed):** `ROBOT`, `GAZEBO`, `ISAAC`,
`CALIBRATION`, `DIAGNOSTICS`.

**Phase taxonomy (C13, `subtask_index`):** 0 IDLE · 1 APPROACH · 2 ALIGN ·
3 GRASP · 4 TRANSPORT · 5 PLACE · 6 RETRACT · 7 FAILURE_RECOVERY.

**Metadata files:** `meta/tasks.jsonl` (mandatory task → language instruction),
`meta/episodes.jsonl` (episode_index, task_index, length, source_profile,
robot_config hash, calibration hash, timestamps, language_instruction),
`meta/episodes_stats.jsonl` (per-episode min/max/mean/std for pre-flight).

---

## Setup

Requires **Python 3.10**. System dependency for video-level QA checks:
**ffmpeg** (`ffprobe`). OpenCV frame decoding is installed via requirements.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
sudo apt-get install -y ffmpeg        # or your platform equivalent
```

---

## Usage

### 1. Contract validation (Phase 1 — C2/C9 gate)

```bash
# Contract-only check:
python scripts/validate_dataset_contract.py --info-json meta/info.json

# Note: this validator was written on freeze day against the rev18 21-channel
# draft. Pass the frozen dimension explicitly until its constant is updated:
python scripts/validate_dataset_contract.py --info-json meta/info.json --expected-state-dim 19

# Full dataset tree (requires pandas + pyarrow):
python scripts/validate_dataset_contract.py \
    --dataset-path /data/openamrobot/episodes \
    --skew-threshold-ms 30.0 \
    --expected-state-dim 19
```

### 2. Pre-flight QA gate (Phase 2 — C10, gates all training jobs)

Canonical invocation from the plan (C-TEAM-PLAN-01 §3.2):

```bash
python scripts/check_dataset_integrity.py \
    --dataset-path /data/openamrobot/episodes/ \
    --skew-threshold-ms 30.0 \
    --min-frames 30 \
    --require-source-profile ROBOT \
    --output-report /data/reports/qa_report.json
```

Optional flags: `--frozen-threshold-sec` (default 2.0), `--max-phase-lag`
(default 3), `--skip-video-checks` (machines without ffmpeg),
`--skip-frame-decode` (skip OpenCV entropy decoding), `-v`.

**Gate semantics:** *If any episode is rejected, training must not start.*

| Exit code | Meaning |
|---|---|
| 0 | Gate OPEN — no REJECT findings (FLAGs are reported for triage but do not block) |
| 1 | Gate BLOCKED — one or more REJECT findings |
| 2 | Environment / usage error |

Severity model: `REJECT` blocks training · `FLAG` requires triage ·
`WARN` is informational. The JSON report contains per-episode stats (skew
max/mean/p95, camera freshness ratios, frozen-run analysis, phase-lag
estimates) and per-code diagnostic counts — this report doubles as the Q-01
measured synchronisation/skew evidence.

### 3. Fault-injection fixture suite (C10 acceptance evidence)

```bash
python -m pytest tests/test_c10_fixtures.py -v
```

| # | Fixture | Fault injected | Expected | Code asserted |
|---|---|---|---|---|
| 1 | FIX-TS-JITTER | `t[i] < t[i-1]` at random frames | REJECT | `MONOTONICITY_VIOLATION` |
| 2 | FIX-DROP-FRAME | MP4 truncated 5 frames vs Parquet | REJECT | `FRAME_COUNT_MISMATCH` |
| 3 | FIX-PHASE-LAG | action shifted +5 steps vs state | FLAG | `PHASE_LAG` |
| 4 | FIX-FROZEN-CAM | identical frames ≥ 2.0 s | FLAG | `FROZEN_CAMERA` |
| 5 | FIX-TRUNC-PARQUET | last 100 bytes of footer corrupted | REJECT | `READ_FAILURE` |
| 6 | FIX-DIM-MISMATCH | vector width 18 or 20 instead of 19 | REJECT | `SCHEMA_VIOLATION` |
| 7 | FIX-OOB-STATS | NaN / ±Inf in joint channels | REJECT | `NON_FINITE_VALUE` |
| 8 | FIX-ORPHAN-TASK | `task_index: 99` with only tasks 0..2 | REJECT | `ORPHAN_TASK` |

Fixture 5 requires `pyarrow` (real on-disk Parquet); all others run fully
in-memory. Deterministic seeds are used throughout for evidence
reproducibility.

---

## Expected dataset layout (LeRobotDataset v3.0)

```text
/data/openamrobot/episodes/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   ├── episodes.jsonl
│   ├── subtasks.jsonl
│   └── episodes_stats.jsonl
├── data/chunk-000/episode_XXXXXX.parquet
└── videos/chunk-000/<camera_key>/episode_XXXXXX.mp4
```

Storage tiers (HC-9): robot NVMe → checksum-verified S3 → optional HF Hub.

---

## Pinning policy (HC-1, HC-11)

| Component | Pinned value | Recorded in |
|---|---|---|
| LeRobot | `lerobot==0.4.1` (v3.0 dataset format; upstream, never forked) | `requirements.txt` |
| Torch stack | `torch==2.5.1`, `torchvision==0.20.1` | `requirements.txt` |
| Schema tag | `schema-v1.0-frozen` (19 channels, P-03 Addendum rev18.2) | git tag + `meta/info.json` |
| Robot / format | `openamrobot_2`, `codebase_version: "v3.0"`, 30 fps | `meta/info.json` |

The Phase 3/4 `RELEASE_MANIFEST.md` will extend this record with the upstream
LeRobot commit hash corresponding to the pinned release, checkpoint SHA-256,
training-config hash, dataset revision, calibration hashes, and evaluation
evidence (HC-10 policy manifest). Do **not** track upstream `main` (risk R5).

---

## Q-01 evidence mapping (team deliverables)

| Q-01 requirement | Artifact | Provided by |
|---|---|---|
| Required-field validation | contract validator CI output | `scripts/validate_dataset_contract.py` |
| Measured synchronisation/skew report | `qa_report.json` | `scripts/check_dataset_integrity.py` |
| Dropped/stale-frame statistics | per-episode frame integrity stats in `qa_report.json` | same |
| Corrupted-episode rejection evidence | 8 fixture test results + 1 real episode pass | `tests/test_c10_fixtures.py` + CLI run |
| Pinned LeRobot release | `requirements.txt` + release manifest entry | this repo |
| Per-stream timestamp-source metadata | `timestamp_source` fields in `meta/episodes.jsonl` | ingest layer |

---

## Schema-freeze note (21 → 19 resolution)

Rev18 declared `shape: [21]` while enumerating 19 channel names. This was
resolved at freeze: the fixed-mast design carries **19 DoF** (P-03 Addendum
rev18.2), and the frozen contract uses the enumerated 19-channel ordering
above. All Phase 2 tooling (`check_dataset_integrity.py`,
`test_c10_fixtures.py`) hard-codes `STATE_DIM = 19`. The Phase 1 contract
validator defaults to the rev18 draft value; run it with
`--expected-state-dim 19` (or update `EXPECTED_STATE_DIM` in a single commit
referencing the addendum) before wiring it into CI branch protection on
`meta/info.json`, vector ordering, and camera keys.

---

## Known follow-ups

- **Action-range QA check** ("within URDF joint limits ± 5%", §3.2) is wired
  in once Tushar's A6 canonical URDF limits file lands; no guessed limits are
  hard-coded into the gate.
- **Skew epoch alignment** is performed by the ingest layer (C9): device-native
  timestamps are stored epoch-aligned to the Jetson host monotonic domain so
  `|timestamp − timestamp_device|` measures transfer skew, not epoch offset.
- **Gazebo episodes (A7)** may be used for pipeline validation only, tagged
  `source_profile: GAZEBO` (HC-8); they cannot satisfy physical-validation
  evidence (HC-14). Gate A slips are escalated in weekly P-01, never worked
  around silently.
- **One real recorded episode** must pass the C10 gate end-to-end to close the
  16 Oct evidence filing.

Post-freeze schema changes require Project Lead sign-off and an escalation
record (HC-12). Readiness is evidence, not optimism (HC-13).