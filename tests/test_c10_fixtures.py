"""
test_c10_fixtures.py
====================

OpenAMRobot 2.0 | Workstream C | Deliverable C10 — Dataset QA
Mandatory fault-injection fixture suite (all 8 fixtures, C-TEAM-PLAN-01
rev2.0 section 3.2). Frozen schema: schema-v1.0-frozen, 19 channels
(fixed-mast design per P-03 Addendum rev18.2).

Fixture matrix:

    #  FIX-TS-JITTER      t[i] < t[i-1] at random frames              -> REJECT  monotonicity violation
    #  FIX-DROP-FRAME     video truncated 5 frames vs parquet          -> REJECT  frame count mismatch
    #  FIX-PHASE-LAG      action shifted +5 steps vs observation.state -> FLAG    cross-correlation anomaly
    #  FIX-FROZEN-CAM     identical frames for 2.0+ seconds            -> FLAG    entropy below threshold
    #  FIX-TRUNC-PARQUET  last 100 bytes of parquet footer corrupted   -> REJECT  read/deserialization failure
    #  FIX-DIM-MISMATCH   state/action width 18 or 20 instead of 19    -> REJECT  schema violation
    #  FIX-OOB-STATS      NaN / Inf injected into joint channels       -> REJECT  non-finite numerical values
    #  FIX-ORPHAN-TASK    task_index 99 with only tasks 0..2 present   -> REJECT  referential integrity failure

Dependencies: numpy + pytest. Only FIX-TRUNC-PARQUET requires pyarrow
(it exercises a real parquet file on disk); it skips cleanly when pyarrow
is not installed. All other fixtures run fully in-memory via the shared
EpisodeRecord abstraction imported from scripts/check_dataset_integrity.py,
so the tests exercise the exact gate logic that blocks training.

Run:
    python -m pytest tests/test_c10_fixtures.py -v
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the QA gate under test (scripts/ is not a package; bootstrap path).
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_dataset_integrity as cdi  # noqa: E402

# ---------------------------------------------------------------------------
# Test constants (mirror the frozen contract)
# ---------------------------------------------------------------------------

STATE_DIM = 19
FPS = 30
NUM_FRAMES = 64                 # comfortably above --min-frames default of 30
LEFT_GRIPPER_IDX = 11           # per frozen 19-channel ordering
RIGHT_GRIPPER_IDX = 18          # per frozen 19-channel ordering

CONFIG = cdi.QAConfig()         # defaults: skew 30 ms, min 30 frames, ROBOT, 30 fps
VALID_TASKS = frozenset({0, 1, 2})


def validate(record: cdi.EpisodeRecord) -> cdi.EpisodeResult:
    return cdi.validate_episode(record, CONFIG, VALID_TASKS)


def codes_of(result: cdi.EpisodeResult, severity: str = None) -> List[str]:
    return [f.code for f in result.findings if severity is None or f.severity == severity]


# ---------------------------------------------------------------------------
# Synthetic clean episode generator (in-memory, deterministic)
# ---------------------------------------------------------------------------

def make_clean_record(
    num_frames: int = NUM_FRAMES,
    fps: int = FPS,
    seed: int = 20261016,
    episode_index: int = 0,
) -> cdi.EpisodeRecord:
    """One contract-compliant synthetic episode (ROBOT profile surrogate)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / float(fps)

    # HC-5 dual timebase: device-native stream is epoch-aligned with the host
    # monotonic clock by the ingest layer (C9 skew definition), leaving only
    # bounded <= 2 ms per-frame transfer jitter.
    host = 1000.0 + np.arange(num_frames, dtype=np.float64) * dt
    device = host + rng.uniform(-0.002, 0.002, num_frames)

    # Smooth bounded trajectories across all 19 channels (2 periods per episode
    # gives a sharp cross-correlation peak for the phase-lag detector).
    # Amplitudes are scaled to 80% of each channel's physical half-range so the
    # clean-record always passes the CANONICAL_ACTION_LIMITS bounds check that
    # was wired into validate_episode() as step 10.
    #
    # Per-channel amplitude map (80% of |limit| half-range):
    #   ch 0  base_linear_x  : 0.80 * 0.20 = 0.160 m/s
    #   ch 1  base_linear_y  : 0.0  (differential drive — always 0)
    #   ch 2  base_angular_z : 0.80 * 0.50 = 0.400 rad/s
    #   ch 3  torso_pitch     : 0.0  (fixed mast — not actuated)
    #   ch 4  torso_yaw       : 0.0  (fixed mast — not actuated)
    #   ch 5-11 left arm + gripper  : 0.80 * π ≈ 2.513 rad  (OpenArm ±π limits)
    #   ch 12-18 right arm + gripper: 0.80 * π ≈ 2.513 rad
    _AMP = np.array([
        0.160, 0.000, 0.400,          # base (x, y, z)
        0.000, 0.000,                  # torso pitch, yaw (fixed mast)
        2.513, 2.513, 2.513, 2.513, 2.513, 2.513,  # left arm joints
        0.800,                         # left gripper (0-1 range, 80%)
        2.513, 2.513, 2.513, 2.513, 2.513, 2.513,  # right arm joints
        0.800,                         # right gripper (0-1 range, 80%)
    ], dtype=np.float32)
    assert len(_AMP) == STATE_DIM, f"Amplitude map length mismatch: {len(_AMP)} != {STATE_DIM}"

    wave = np.linspace(0.0, 4.0 * np.pi, num_frames)
    state = np.zeros((num_frames, STATE_DIM), dtype=np.float32)
    for j in range(STATE_DIM):
        state[:, j] = _AMP[j] * np.sin(wave + 0.35 * j)
    state = state + rng.normal(0.0, 0.001, state.shape).astype(np.float32)

    # Clamp to ensure noise doesn't push any channel over limits.
    _LIMITS = cdi.CANONICAL_ACTION_LIMITS  # shape (19, 2) [min, max]
    state = np.clip(state, _LIMITS[:, 0], _LIMITS[:, 1]).astype(np.float32)

    # Gripper channels: remap to [0, 1] range regardless of noise.
    state[:, LEFT_GRIPPER_IDX] = np.clip(0.5 + 0.5 * np.sin(wave), 0.0, 1.0).astype(np.float32)
    state[:, RIGHT_GRIPPER_IDX] = np.clip(0.5 + 0.5 * np.cos(wave), 0.0, 1.0).astype(np.float32)

    # Action leads the achieved state by one frame (realistic command ordering).
    action = np.ascontiguousarray(np.roll(state, -1, axis=0))

    # C13 phase tagging across phases 0..6.
    subtask = np.zeros(num_frames, dtype=np.int64)
    segment = max(1, num_frames // 7)
    for phase in range(7):
        lo = phase * segment
        hi = num_frames if phase == 6 else (phase + 1) * segment
        subtask[lo:hi] = phase

    return cdi.EpisodeRecord(
        episode_index=episode_index,
        parquet_path=f"data/chunk-000/episode_{episode_index:06d}.parquet",
        parquet_rows=num_frames,
        timestamp=host,
        timestamp_device=device,
        observation_state=state,
        action=action,
        subtask_index=subtask,
        task_index=1,
        source_profile="ROBOT",
        declared_length=None,
        video_frame_counts={cam: num_frames for cam in cdi.CANONICAL_CAMERAS},
        video_durations={cam: num_frames / float(fps) for cam in cdi.CANONICAL_CAMERAS},
        camera_frame_digests={cam: list(range(num_frames)) for cam in cdi.CANONICAL_CAMERAS},
    )


# ---------------------------------------------------------------------------
# Fault injectors (one per mandatory fixture)
# ---------------------------------------------------------------------------

def fix_ts_jitter(base: cdi.EpisodeRecord, seed: int = 7531) -> cdi.EpisodeRecord:
    """FIX-TS-JITTER: force t[i] < t[i-1] at random frames (host clock)."""
    rec = copy.deepcopy(base)
    rng = np.random.default_rng(seed)
    candidates = np.arange(2, rec.timestamp.size - 1)
    for i in rng.choice(candidates, size=3, replace=False):
        idx = int(i)
        rec.timestamp[idx] = rec.timestamp[idx - 1] - rng.uniform(0.005, 0.020)
    return rec


def fix_drop_frame(base: cdi.EpisodeRecord, dropped_frames: int = 5) -> cdi.EpisodeRecord:
    """FIX-DROP-FRAME: simulate MP4 truncation; parquet side left untouched."""
    rec = copy.deepcopy(base)
    rec.video_frame_counts = {
        cam: n - dropped_frames for cam, n in rec.video_frame_counts.items()
    }
    return rec


def fix_phase_lag(base: cdi.EpisodeRecord, shift: int = 5) -> cdi.EpisodeRecord:
    """FIX-PHASE-LAG: shift action by +5 steps relative to observation.state."""
    rec = copy.deepcopy(base)
    rec.action = np.ascontiguousarray(np.roll(rec.action, shift, axis=0))
    return rec


def fix_frozen_cam(
    base: cdi.EpisodeRecord,
    cam: str = "observation.images.head",
    frozen_frames: int = 61,
) -> cdi.EpisodeRecord:
    """FIX-FROZEN-CAM: identical consecutive frame digests for 2.0+ seconds.

    61 frames at 30 fps = 2.033 s, safely over the 2.0 s threshold.
    """
    rec = copy.deepcopy(base)
    digests = list(rec.camera_frame_digests[cam])
    for i in range(frozen_frames):
        digests[i] = 0xC0FFEE
    rec.camera_frame_digests[cam] = digests
    return rec


def fix_dim_mismatch(
    base: cdi.EpisodeRecord, field_name: str, width: int
) -> cdi.EpisodeRecord:
    """FIX-DIM-MISMATCH: truncate to 18 or pad to 20 channels."""
    rec = copy.deepcopy(base)
    arr = getattr(rec, field_name)
    if width < arr.shape[1]:
        new = arr[:, :width]
    else:
        pad = np.zeros((arr.shape[0], width - arr.shape[1]), dtype=np.float32)
        new = np.concatenate([arr, pad], axis=1)
    setattr(rec, field_name, np.ascontiguousarray(new))
    return rec


def fix_oob_stats(base: cdi.EpisodeRecord, seed: int = 90210) -> cdi.EpisodeRecord:
    """FIX-OOB-STATS: inject NaN, +Inf and -Inf into joint channels."""
    rec = copy.deepcopy(base)
    rng = np.random.default_rng(seed)
    rows = rng.choice(rec.observation_state.shape[0], size=3, replace=False)
    cols = rng.integers(0, STATE_DIM, size=3)
    rec.observation_state[int(rows[0]), int(cols[0])] = np.nan
    rec.observation_state[int(rows[1]), int(cols[1])] = np.inf
    rec.observation_state[int(rows[2]), int(cols[2])] = -np.inf
    return rec


def fix_orphan_task(base: cdi.EpisodeRecord, task_index: int = 99) -> cdi.EpisodeRecord:
    """FIX-ORPHAN-TASK: point the episode at a non-existent task."""
    rec = copy.deepcopy(base)
    rec.task_index = task_index
    return rec


# ---------------------------------------------------------------------------
# Parquet round-trip helper (FIX-TRUNC-PARQUET only)
# ---------------------------------------------------------------------------

def _build_clean_parquet(record: cdi.EpisodeRecord, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    T = record.parquet_rows

    def list_col(mat: np.ndarray):
        return pa.array(
            [[float(v) for v in row] for row in mat], type=pa.list_(pa.float32())
        )

    table = pa.table(
        {
            "observation.state": list_col(record.observation_state),
            "action": list_col(record.action),
            "timestamp": pa.array(record.timestamp.astype(np.float64), type=pa.float64()),
            "timestamp_device": pa.array(
                record.timestamp_device.astype(np.float64), type=pa.float64()
            ),
            "episode_index": pa.array([record.episode_index] * T, type=pa.int64()),
            "frame_index": pa.array(list(range(T)), type=pa.int64()),
            "index": pa.array(list(range(T)), type=pa.int64()),
            "task_index": pa.array([record.task_index] * T, type=pa.int64()),
            "subtask_index": pa.array(record.subtask_index.astype(np.int64), type=pa.int64()),
        }
    )
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Sanity gate: the harness must not reject a compliant episode
# ---------------------------------------------------------------------------

def test_clean_episode_passes_preflight() -> None:
    result = validate(make_clean_record())
    assert result.status == "PASS", result.findings
    assert result.findings == []
    assert abs(int(result.stats["phase_lag_frames"])) <= CONFIG.max_phase_lag
    assert result.stats["skew_ms"]["max"] < CONFIG.skew_threshold_ms


# ---------------------------------------------------------------------------
# Fixture 1: FIX-TS-JITTER -> REJECT (monotonicity violation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [7531, 11009, 424242])
def test_fix_ts_jitter_rejected(seed: int) -> None:
    result = validate(fix_ts_jitter(make_clean_record(), seed=seed))
    assert result.status == "REJECT", "non-monotonic host timestamps must be rejected"
    assert cdi.CODE_MONOTONICITY_VIOLATION in codes_of(result, cdi.SEV_REJECT)


# ---------------------------------------------------------------------------
# Fixture 2: FIX-DROP-FRAME -> REJECT (frame count mismatch)
# ---------------------------------------------------------------------------

def test_fix_drop_frame_rejected() -> None:
    result = validate(fix_drop_frame(make_clean_record(), dropped_frames=5))
    assert result.status == "REJECT", "video/parquet frame-count mismatch must be rejected"
    mismatches = [f for f in result.findings if f.code == cdi.CODE_FRAME_COUNT_MISMATCH]
    assert len(mismatches) == len(cdi.CANONICAL_CAMERAS), result.findings


# ---------------------------------------------------------------------------
# Fixture 3: FIX-PHASE-LAG -> FLAG (cross-correlation anomaly)
# ---------------------------------------------------------------------------

def test_fix_phase_lag_flagged() -> None:
    result = validate(fix_phase_lag(make_clean_record(), shift=5))
    assert codes_of(result, cdi.SEV_REJECT) == [], result.findings
    assert result.status == "FLAG", "phase lag must be flagged, not rejected"
    assert cdi.CODE_PHASE_LAG in codes_of(result, cdi.SEV_FLAG)
    assert abs(int(result.stats["phase_lag_frames"])) > CONFIG.max_phase_lag


# ---------------------------------------------------------------------------
# Fixture 4: FIX-FROZEN-CAM -> FLAG (entropy below threshold)
# ---------------------------------------------------------------------------

def test_fix_frozen_cam_flagged() -> None:
    cam = "observation.images.head"
    result = validate(fix_frozen_cam(make_clean_record(), cam=cam, frozen_frames=61))
    assert codes_of(result, cdi.SEV_REJECT) == [], result.findings
    assert result.status == "FLAG", "frozen camera must be flagged, not rejected"
    assert cdi.CODE_FROZEN_CAMERA in codes_of(result, cdi.SEV_FLAG)
    analysis = result.stats["frozen_camera_analysis"][cam]
    assert analysis["max_identical_run_sec"] >= CONFIG.frozen_threshold_sec
    assert analysis["run_start_frame"] == 0
    # The other cameras stay clean.
    for other in cdi.CANONICAL_CAMERAS:
        if other != cam:
            assert result.stats["frozen_camera_analysis"][other]["max_identical_run_frames"] == 1


# ---------------------------------------------------------------------------
# Fixture 5: FIX-TRUNC-PARQUET -> REJECT (read/deserialization failure)
# ---------------------------------------------------------------------------

def test_fix_trunc_parquet_rejected(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")  # real parquet IO required for this fixture

    clean = make_clean_record()
    clean_path = tmp_path / "episode_000000.parquet"
    _build_clean_parquet(clean, clean_path)

    # Control: the intact file loads and passes.
    ok_record = cdi.load_parquet_episode(clean_path)
    assert ok_record.read_error is None
    assert ok_record.parquet_rows == clean.parquet_rows

    # Corrupt the last 100 bytes (parquet footer region).
    raw = clean_path.read_bytes()
    assert len(raw) > 100
    trunc_path = tmp_path / "episode_000001.parquet"
    trunc_path.write_bytes(raw[:-100])

    rec = cdi.load_parquet_episode(trunc_path)
    assert rec.read_error is not None, "truncated parquet must fail to deserialize"
    result = cdi.validate_episode(rec, CONFIG, VALID_TASKS)
    assert result.status == "REJECT"
    assert cdi.CODE_READ_FAILURE in codes_of(result, cdi.SEV_REJECT)


# ---------------------------------------------------------------------------
# Fixture 6: FIX-DIM-MISMATCH -> REJECT (schema violation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field_name,width", [("observation_state", 18), ("action", 20)]
)
def test_fix_dim_mismatch_rejected(field_name: str, width: int) -> None:
    result = validate(fix_dim_mismatch(make_clean_record(), field_name, width))
    assert result.status == "REJECT", f"{field_name} width {width} must be rejected"
    assert cdi.CODE_SCHEMA_VIOLATION in codes_of(result, cdi.SEV_REJECT)


# ---------------------------------------------------------------------------
# Fixture 7: FIX-OOB-STATS -> REJECT (non-finite numerical values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [90210, 42])
def test_fix_oob_stats_rejected(seed: int) -> None:
    result = validate(fix_oob_stats(make_clean_record(), seed=seed))
    assert result.status == "REJECT", "NaN/Inf joint samples must be rejected"
    assert cdi.CODE_NON_FINITE_VALUE in codes_of(result, cdi.SEV_REJECT)


# ---------------------------------------------------------------------------
# Fixture 8: FIX-ORPHAN-TASK -> REJECT (referential integrity failure)
# ---------------------------------------------------------------------------

def test_fix_orphan_task_rejected() -> None:
    result = validate(fix_orphan_task(make_clean_record(), task_index=99))
    assert result.status == "REJECT", "task_index 99 with tasks 0..2 must be rejected"
    assert cdi.CODE_ORPHAN_TASK in codes_of(result, cdi.SEV_REJECT)


# ---------------------------------------------------------------------------
# Supplementary acceptance-rule tests (CLI spec checks without dedicated
# fixtures in the mandatory eight)
# ---------------------------------------------------------------------------

def test_skew_threshold_exceeded_rejected() -> None:
    rec = make_clean_record()
    rec.timestamp_device = rec.timestamp_device + 0.050  # 50 ms constant offset
    result = validate(rec)
    assert result.status == "REJECT"
    assert cdi.CODE_SKEW_THRESHOLD_EXCEEDED in codes_of(result, cdi.SEV_REJECT)


def test_source_profile_mismatch_rejected() -> None:
    rec = make_clean_record()
    rec.source_profile = "GAZEBO"  # valid enum member, but not the required ROBOT
    result = validate(rec)
    assert result.status == "REJECT"
    assert cdi.CODE_SOURCE_PROFILE_MISMATCH in codes_of(result, cdi.SEV_REJECT)


def test_invalid_source_profile_rejected() -> None:
    rec = make_clean_record()
    rec.source_profile = "WEBCAM"  # outside the HC-8 enum
    result = validate(rec)
    assert result.status == "REJECT"
    assert cdi.CODE_INVALID_SOURCE_PROFILE in codes_of(result, cdi.SEV_REJECT)


def test_episode_too_short_rejected() -> None:
    result = validate(make_clean_record(num_frames=20))  # < min_frames default 30
    assert result.status == "REJECT"
    assert cdi.CODE_EPISODE_TOO_SHORT in codes_of(result, cdi.SEV_REJECT)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))