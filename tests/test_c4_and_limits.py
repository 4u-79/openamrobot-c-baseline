"""
test_c4_and_limits.py
=====================

OpenAMRobot 2.0 | Workstream C | Tests for Deliverable C4 + action-bound gate.

Test matrix
-----------
    Test 1 — check_action_bounds: clean episode passes (no findings)
    Test 2 — check_action_bounds: action exceeds limit+5.1% → ACTION_OUT_OF_BOUNDS REJECT
    Test 3 — replay_episode.py --dry-run: valid 19-channel Parquet exits 0
    Test 4 — replay_episode.py --dry-run: OOB Parquet exits 1

Dependencies: numpy, pytest, pyarrow (test 3/4 only; auto-skipped when absent).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap import paths (scripts/ is not a package)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check_dataset_integrity as cdi  # noqa: E402

# ---------------------------------------------------------------------------
# Constants mirrored from the contract
# ---------------------------------------------------------------------------

STATE_DIM = 19
FPS = 30
NUM_FRAMES = 64


# ---------------------------------------------------------------------------
# Synthetic episode helpers
# ---------------------------------------------------------------------------


def _make_clean_actions(seed: int = 42) -> np.ndarray:
    """Generate a (NUM_FRAMES, 19) action matrix that is strictly within limits."""
    rng = np.random.default_rng(seed)
    actions = np.zeros((NUM_FRAMES, STATE_DIM), dtype=np.float32)
    lims = cdi.CANONICAL_ACTION_LIMITS

    for ch in range(STATE_DIM):
        lo = float(lims[ch, 0])
        hi = float(lims[ch, 1])
        span = hi - lo
        if span < 1e-6:
            # Zero-span channel (torso, base_linear_y): stay exactly at 0.
            actions[:, ch] = 0.0
        else:
            # Fill with small-amplitude sine within 80% of the range.
            mid = (lo + hi) / 2.0
            half = span * 0.40
            wave = np.sin(np.linspace(0.0, 4 * np.pi, NUM_FRAMES))
            actions[:, ch] = (mid + half * wave).astype(np.float32)

    return actions


def _make_clean_record(actions: np.ndarray) -> cdi.EpisodeRecord:
    """Wrap actions in a minimal compliant EpisodeRecord."""
    dt = 1.0 / FPS
    host = 1000.0 + np.arange(NUM_FRAMES, dtype=np.float64) * dt
    state = actions.copy()  # use action itself as state for simplicity

    rec = cdi.EpisodeRecord(
        episode_index=0,
        parquet_path="synthetic",
        parquet_rows=NUM_FRAMES,
        timestamp=host,
        timestamp_device=host + 0.001,
        observation_state=state,
        action=actions,
        task_index=0,
        source_profile="ROBOT",
        declared_length=NUM_FRAMES,
    )
    # Simulate all cameras present to avoid video-missing REJECTs.
    for cam in cdi.CANONICAL_CAMERAS:
        rec.video_frame_counts[cam] = NUM_FRAMES
        rec.video_durations[cam] = NUM_FRAMES / FPS
    return rec


CONFIG = cdi.QAConfig(
    skew_threshold_ms=30.0,
    min_frames=30,
    require_source_profile="ROBOT",
    video_checks_enabled=False,   # no ffprobe in test environment
)
VALID_TASKS = frozenset({0, 1, 2})


def codes_of(findings, severity=None) -> List[str]:
    return [f.code for f in findings if severity is None or f.severity == severity]


# ===========================================================================
# Test 1 — check_action_bounds: clean episode produces no findings
# ===========================================================================


def test_check_action_bounds_clean_passes():
    """Action strictly within physical limits must produce zero findings."""
    actions = _make_clean_actions()
    findings = cdi.check_action_bounds(actions)
    assert findings == [], (
        f"Expected zero findings, got: {[(f.code, f.message) for f in findings]}"
    )


def test_validate_episode_clean_passes():
    """Full validate_episode on a clean synthetic record must PASS (no REJECT)."""
    actions = _make_clean_actions()
    rec = _make_clean_record(actions)
    result = cdi.validate_episode(rec, CONFIG, VALID_TASKS)
    reject_codes = codes_of(result.findings, cdi.SEV_REJECT)
    assert cdi.CODE_ACTION_OUT_OF_BOUNDS not in reject_codes, (
        f"Unexpected ACTION_OUT_OF_BOUNDS on clean episode: {result.findings}"
    )


# ===========================================================================
# Test 2 — check_action_bounds: value exceeding limit+5.1% triggers REJECT
# ===========================================================================


def test_check_action_bounds_violation_triggers_reject():
    """Injecting a value 5.1% beyond the physical maximum must trigger REJECT."""
    actions = _make_clean_actions()

    # Choose a non-zero-span arm channel: ch 5 = left_shoulder_pan, limit [-2.618, 2.618].
    TEST_CH = 5
    lo = float(cdi.CANONICAL_ACTION_LIMITS[TEST_CH, 0])
    hi = float(cdi.CANONICAL_ACTION_LIMITS[TEST_CH, 1])
    span = hi - lo
    # Inject a value just beyond the ±5% window (5.1% over hi).
    overflow_value = hi + span * 0.051
    actions[0, TEST_CH] = np.float32(overflow_value)

    findings = cdi.check_action_bounds(actions)
    codes = codes_of(findings, cdi.SEV_REJECT)

    assert cdi.CODE_ACTION_OUT_OF_BOUNDS in codes, (
        f"Expected ACTION_OUT_OF_BOUNDS REJECT for ch {TEST_CH} value={overflow_value:.4f} "
        f"(limit max={hi:.4f}, allowed up to {hi + span * 0.05:.4f}). "
        f"Got findings: {[(f.code, f.message) for f in findings]}"
    )


def test_check_action_bounds_within_5pct_does_not_reject():
    """A value exactly at the 5% boundary must NOT trigger a REJECT."""
    actions = _make_clean_actions()

    # ch 5 = left_shoulder_pan; inject value right at the 4.9% margin (below threshold).
    TEST_CH = 5
    hi = float(cdi.CANONICAL_ACTION_LIMITS[TEST_CH, 1])
    span = hi - float(cdi.CANONICAL_ACTION_LIMITS[TEST_CH, 0])
    safe_value = hi + span * 0.049   # 4.9% — safely within the 5% window
    actions[0, TEST_CH] = np.float32(safe_value)

    findings = cdi.check_action_bounds(actions)
    codes = codes_of(findings, cdi.SEV_REJECT)

    assert cdi.CODE_ACTION_OUT_OF_BOUNDS not in codes, (
        f"False REJECT triggered at 4.9% over limit for ch {TEST_CH}: "
        f"{[(f.code, f.message) for f in findings]}"
    )


def test_zero_span_channel_eps_tolerance():
    """Zero-span channels (torso, base_linear_y) must accept exact zeros without REJECT."""
    actions = _make_clean_actions()   # already zero for zero-span channels
    findings = cdi.check_action_bounds(actions)
    codes = codes_of(findings, cdi.SEV_REJECT)
    assert cdi.CODE_ACTION_OUT_OF_BOUNDS not in codes, (
        f"Zero-span channels incorrectly rejected: {findings}"
    )


def test_zero_span_channel_excessive_value_rejected():
    """Zero-span channel with value > eps must trigger REJECT."""
    actions = _make_clean_actions()
    # ch 3 = torso_pitch [0, 0], eps = 1e-4; inject 0.01 to force rejection.
    actions[:, 3] = np.float32(0.01)
    findings = cdi.check_action_bounds(actions)
    codes = codes_of(findings, cdi.SEV_REJECT)
    assert cdi.CODE_ACTION_OUT_OF_BOUNDS in codes, (
        f"Expected REJECT for torso_pitch=0.01 on zero-span channel. Got: {findings}"
    )


def test_validate_episode_oob_action_rejected():
    """validate_episode must emit ACTION_OUT_OF_BOUNDS when action exceeds limits."""
    actions = _make_clean_actions()
    TEST_CH = 5
    hi = float(cdi.CANONICAL_ACTION_LIMITS[TEST_CH, 1])
    span = hi - float(cdi.CANONICAL_ACTION_LIMITS[TEST_CH, 0])
    actions[0, TEST_CH] = np.float32(hi + span * 0.051)

    rec = _make_clean_record(actions)
    result = cdi.validate_episode(rec, CONFIG, VALID_TASKS)

    assert result.status == cdi.SEV_REJECT, (
        f"Expected REJECT status, got {result.status}"
    )
    reject_codes = codes_of(result.findings, cdi.SEV_REJECT)
    assert cdi.CODE_ACTION_OUT_OF_BOUNDS in reject_codes, (
        f"ACTION_OUT_OF_BOUNDS not in REJECT findings: {reject_codes}"
    )


# ===========================================================================
# Test 3 — replay_episode.py --dry-run on valid Parquet (requires pyarrow)
# ===========================================================================


pyarrow = pytest.importorskip("pyarrow", reason="pyarrow not installed — skipping Parquet tests")


def _write_synthetic_parquet(path: Path, actions: np.ndarray) -> None:
    """Write a minimal episode Parquet consumable by replay_episode.py."""
    import pyarrow as pa
    import pyarrow.parquet as pq_w

    n = len(actions)
    dt = 1.0 / FPS
    host = 1000.0 + np.arange(n, dtype=np.float64) * dt

    table = pa.table({
        "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
        "observation.state": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
        "timestamp": pa.array(host, type=pa.float64()),
        "timestamp_device": pa.array(host + 0.001, type=pa.float64()),
        "episode_index": pa.array([0] * n, type=pa.int64()),
        "frame_index": pa.array(list(range(n)), type=pa.int64()),
        "index": pa.array(list(range(n)), type=pa.int64()),
        "task_index": pa.array([0] * n, type=pa.int64()),
        "subtask_index": pa.array([0] * n, type=pa.int64()),
    })
    pq_w.write_table(table, path)


def test_dry_run_valid_episode_exits_zero(tmp_path):
    """replay_episode.py --dry-run on a valid 19-channel Parquet must exit 0."""
    import replay_episode as re_mod

    actions = _make_clean_actions()
    parquet_path = tmp_path / "episode_000000.parquet"
    _write_synthetic_parquet(parquet_path, actions)

    exit_code = re_mod.run_dry_run(parquet_path, verbose=False)
    assert exit_code == 0, (
        f"Expected exit 0 for valid episode, got {exit_code}"
    )


def test_dry_run_oob_episode_exits_one(tmp_path):
    """replay_episode.py --dry-run on an OOB Parquet must exit 1."""
    import replay_episode as re_mod

    actions = _make_clean_actions()
    # Inject violation: ch 5 = left_shoulder_pan, 6% beyond max.
    TEST_CH = 5
    hi = float(cdi.CANONICAL_ACTION_LIMITS[TEST_CH, 1])
    span = hi - float(cdi.CANONICAL_ACTION_LIMITS[TEST_CH, 0])
    actions[0, TEST_CH] = np.float32(hi + span * 0.06)

    parquet_path = tmp_path / "episode_000001.parquet"
    _write_synthetic_parquet(parquet_path, actions)

    exit_code = re_mod.run_dry_run(parquet_path, verbose=False)
    assert exit_code == 1, (
        f"Expected exit 1 for OOB episode, got {exit_code}"
    )


def test_dry_run_cli_valid_episode(tmp_path):
    """CLI entrypoint main(['--parquet', ..., '--dry-run']) must return 0."""
    import replay_episode as re_mod

    actions = _make_clean_actions()
    parquet_path = tmp_path / "episode_000002.parquet"
    _write_synthetic_parquet(parquet_path, actions)

    exit_code = re_mod.main(["--parquet", str(parquet_path), "--dry-run", "--verbose"])
    assert exit_code == 0, f"CLI dry-run returned {exit_code} instead of 0"


def test_load_episode_parquet_schema_error(tmp_path):
    """load_episode_parquet must raise RuntimeError if 'action' column is absent."""
    import pyarrow as pa
    import pyarrow.parquet as pq_w
    import replay_episode as re_mod

    # Write a Parquet file without the 'action' column.
    table = pa.table({"timestamp": pa.array([0.0, 1.0], type=pa.float64())})
    bad_path = tmp_path / "bad.parquet"
    pq_w.write_table(table, bad_path)

    with pytest.raises(RuntimeError, match="action"):
        re_mod.load_episode_parquet(bad_path)
