#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_dataset_integrity.py
==========================

OpenAMRobot 2.0 | Workstream C | Deliverable C10 — Dataset QA pre-flight gate.

This tool is the training gate mandated by C-TEAM-PLAN-01 rev2.0 section 3.2:
"This script gates all training jobs. If any episode is rejected, training
must not start."

Schema reference
----------------
Frozen schema: schema-v1.0-frozen — 19-channel whole-body state/action
(fixed-mast design per P-03 Addendum rev18.2), LeRobotDataset v3.0 layout:

    dataset/
    |-- meta/
    |   |-- info.json
    |   |-- tasks.jsonl        (mandatory: task_index -> language instruction)
    |   +-- episodes.jsonl     (mandatory: episode_index, task_index, length,
    |                           source_profile, robot_config hash, ...)
    |-- data/chunk-000/episode_XXXXXX.parquet
    +-- videos/chunk-000/<camera_key>/episode_XXXXXX.mp4

Checks and diagnostic codes
---------------------------
 REJECT  MONOTONICITY_VIOLATION         timestamp not strictly increasing (FIX-TS-JITTER class)
 REJECT  DEVICE_MONOTONICITY_VIOLATION  timestamp_device not strictly increasing
 REJECT  SKEW_THRESHOLD_EXCEEDED        max |timestamp - timestamp_device| > threshold
 REJECT  FRAME_COUNT_MISMATCH           video container frames != parquet rows, or
                                        episodes.jsonl length != parquet rows (FIX-DROP-FRAME class)
 REJECT  CAMERA_FPS_LOW                 frame_count / duration < fps * 0.95
 REJECT  VIDEO_STREAM_MISSING           no container metadata for a canonical camera
 FLAG    FROZEN_CAMERA                  identical frames >= frozen threshold (FIX-FROZEN-CAM class)
 FLAG    PHASE_LAG                      action/state cross-correlation peak lagged (FIX-PHASE-LAG class)
 REJECT  SCHEMA_VIOLATION               vector length != 19, wrong dtype, ragged/missing column
                                        (FIX-DIM-MISMATCH class; HC-7)
 REJECT  NON_FINITE_VALUE               NaN / Inf / null in state or action (FIX-OOB-STATS class)
 REJECT  TIMEBASE_INVALID               timestamp columns missing / null / wrong length
 REJECT  READ_FAILURE                   parquet deserialization failure (FIX-TRUNC-PARQUET class)
 REJECT  EPISODE_TOO_SHORT              rows < --min-frames
 REJECT  ORPHAN_TASK                    task_index not present in meta/tasks.jsonl (FIX-ORPHAN-TASK class)
 REJECT  MISSING_EPISODE_METADATA       episode absent from meta/episodes.jsonl
 REJECT  INVALID_SOURCE_PROFILE         source_profile outside HC-8 enum
 REJECT  SOURCE_PROFILE_MISMATCH        source_profile != --require-source-profile
 WARN    SUBTASK_OUT_OF_RANGE           subtask_index outside C13 taxonomy 0..7

Severity semantics: REJECT blocks training (exit 1); FLAG is reported and must
be triaged but does not block; WARN is informational.

Usage (matches plan section 3.2):
    python scripts/check_dataset_integrity.py \
        --dataset-path /data/openamrobot/episodes/ \
        --skew-threshold-ms 30.0 \
        --min-frames 30 \
        --require-source-profile ROBOT \
        --output-report /data/reports/qa_report.json

Dependencies:
    hard    : numpy, pyarrow (parquet IO)
    video   : ffprobe (ffmpeg) for container frame counts / durations;
              OpenCV (cv2) for frozen-frame (entropy) decoding.
              Use --skip-video-checks on machines without ffmpeg.

Exit codes: 0 = gate OPEN (no REJECT), 1 = gate BLOCKED (>= 1 REJECT),
            2 = environment / usage error.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import zlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("c10-qa")

# Populated here once numpy is available (referenced in the constants block above).
# fmt: off
CANONICAL_ACTION_LIMITS = np.array([
    [-0.20,  0.20],   #  0 base_linear_x         [m/s]
    [ 0.00,  0.00],   #  1 base_linear_y          [m/s]  diff-drive constraint
    [-0.50,  0.50],   #  2 base_angular_z         [rad/s]
    [ 0.00,  0.00],   #  3 torso_pitch            [rad]   fixed-mast
    [ 0.00,  0.00],   #  4 torso_yaw              [rad]   fixed-mast
    [-2.618, 2.618],  #  5 left_shoulder_pan      [rad]
    [-1.745, 1.745],  #  6 left_shoulder_lift     [rad]
    [-2.618, 2.618],  #  7 left_elbow_flex        [rad]
    [-1.833, 1.833],  #  8 left_wrist_flex        [rad]
    [-3.142, 3.142],  #  9 left_wrist_roll        [rad]
    [-1.833, 1.833],  # 10 left_wrist_yaw         [rad]
    [ 0.00,  1.00],   # 11 left_gripper           [norm 0=open]
    [-2.618, 2.618],  # 12 right_shoulder_pan     [rad]
    [-1.745, 1.745],  # 13 right_shoulder_lift    [rad]
    [-2.618, 2.618],  # 14 right_elbow_flex       [rad]
    [-1.833, 1.833],  # 15 right_wrist_flex       [rad]
    [-3.142, 3.142],  # 16 right_wrist_roll       [rad]
    [-1.833, 1.833],  # 17 right_wrist_yaw        [rad]
    [ 0.00,  1.00],   # 18 right_gripper          [norm 0=open]
], dtype=np.float32)
# fmt: on

# ---------------------------------------------------------------------------
# Frozen contract constants (schema-v1.0-frozen, P-03 Addendum rev18.2)
# ---------------------------------------------------------------------------

STATE_DIM = 19
CANONICAL_FPS = 30.0
SCHEMA_REVISION = "schema-v1.0-frozen (19-channel fixed-mast, P-03 Addendum rev18.2)"

# HC-6: fixed camera keys.
CANONICAL_CAMERAS: Tuple[str, ...] = (
    "observation.images.camera_wrist_left",
    "observation.images.camera_wrist_right",
    "observation.images.camera_scene",
    "observation.images.camera_base_nav",
)

# HC-8: data classes are never silently mixed.
ALLOWED_SOURCE_PROFILES = frozenset(
    {"ROBOT", "GAZEBO", "ISAAC", "CALIBRATION", "DIAGNOSTICS"}
)

# C13 phase taxonomy bounds.
SUBTASK_INDEX_MIN = 0
SUBTASK_INDEX_MAX = 7

# ---------------------------------------------------------------------------
# C4 / Action-bound constants (cross-repo hardware audit 2026-10-13)
# ---------------------------------------------------------------------------

# Frozen 19-channel names (schema-v1.0-frozen ordering).
CANONICAL_CHANNEL_NAMES: Tuple[str, ...] = (
    "base_linear_x",       #  0
    "base_linear_y",       #  1 — diff-drive: always 0.0
    "base_angular_z",      #  2
    "torso_pitch",         #  3 — fixed-mast: always 0.0
    "torso_yaw",           #  4 — fixed-mast: always 0.0
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


# ---------------------------------------------------------------------------
# Finding codes and severities
# ---------------------------------------------------------------------------

SEV_REJECT = "REJECT"
SEV_FLAG = "FLAG"
SEV_WARN = "WARN"
STATUS_PASS = "PASS"

CODE_MONOTONICITY_VIOLATION = "MONOTONICITY_VIOLATION"
CODE_DEVICE_MONOTONICITY_VIOLATION = "DEVICE_MONOTONICITY_VIOLATION"
CODE_SKEW_THRESHOLD_EXCEEDED = "SKEW_THRESHOLD_EXCEEDED"
CODE_FRAME_COUNT_MISMATCH = "FRAME_COUNT_MISMATCH"
CODE_CAMERA_FPS_LOW = "CAMERA_FPS_LOW"
CODE_VIDEO_STREAM_MISSING = "VIDEO_STREAM_MISSING"
CODE_FROZEN_CAMERA = "FROZEN_CAMERA"
CODE_PHASE_LAG = "PHASE_LAG"
CODE_SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
CODE_NON_FINITE_VALUE = "NON_FINITE_VALUE"
CODE_TIMEBASE_INVALID = "TIMEBASE_INVALID"
CODE_READ_FAILURE = "READ_FAILURE"
CODE_EPISODE_TOO_SHORT = "EPISODE_TOO_SHORT"
CODE_ORPHAN_TASK = "ORPHAN_TASK"
CODE_MISSING_EPISODE_METADATA = "MISSING_EPISODE_METADATA"
CODE_INVALID_SOURCE_PROFILE = "INVALID_SOURCE_PROFILE"
CODE_SOURCE_PROFILE_MISMATCH = "SOURCE_PROFILE_MISMATCH"
CODE_MISSING_META_FILE = "MISSING_META_FILE"
CODE_NO_EPISODES_FOUND = "NO_EPISODES_FOUND"
CODE_SUBTASK_OUT_OF_RANGE = "SUBTASK_OUT_OF_RANGE"
CODE_ACTION_OUT_OF_BOUNDS = "ACTION_OUT_OF_BOUNDS"


@dataclass
class QAConfig:
    """Tunable thresholds for the pre-flight gate."""

    skew_threshold_ms: float = 30.0
    min_frames: int = 30
    require_source_profile: str = "ROBOT"
    fps: float = CANONICAL_FPS
    frozen_threshold_sec: float = 2.0
    max_phase_lag: int = 3
    state_dim: int = STATE_DIM
    video_checks_enabled: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "skew_threshold_ms": self.skew_threshold_ms,
            "min_frames": self.min_frames,
            "require_source_profile": self.require_source_profile,
            "fps": self.fps,
            "frozen_threshold_sec": self.frozen_threshold_sec,
            "max_phase_lag": self.max_phase_lag,
            "state_dim": self.state_dim,
            "video_checks_enabled": self.video_checks_enabled,
        }


@dataclass
class Finding:
    """One diagnostic emitted by the gate."""

    code: str
    severity: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


@dataclass
class EpisodeRecord:
    """In-memory representation of one episode (parquet side + video metadata).

    The CLI hydrates this from disk; the C10 fixture suite builds it
    synthetically. The validation engine only ever sees this type.
    """

    episode_index: int = -1
    parquet_path: str = ""
    parquet_rows: int = 0
    timestamp: Optional[np.ndarray] = None          # (T,) float64 host monotonic [s]
    timestamp_device: Optional[np.ndarray] = None   # (T,) float64 device-native [s]
    observation_state: Optional[np.ndarray] = None  # (T, 19) float32
    action: Optional[np.ndarray] = None             # (T, 19) float32
    subtask_index: Optional[np.ndarray] = None      # (T,) int64
    task_index: Optional[int] = None
    source_profile: Optional[str] = None
    declared_length: Optional[int] = None           # meta/episodes.jsonl "length"
    video_frame_counts: Dict[str, int] = field(default_factory=dict)
    video_durations: Dict[str, float] = field(default_factory=dict)
    camera_frame_digests: Dict[str, List[int]] = field(default_factory=dict)
    load_notes: Dict[str, str] = field(default_factory=dict)
    read_error: Optional[str] = None


@dataclass
class EpisodeResult:
    episode_index: int
    status: str                       # PASS | FLAG | REJECT
    findings: List[Finding]
    stats: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "episode_index": self.episode_index,
            "status": self.status,
            "findings": [f.to_dict() for f in self.findings],
            "stats": self.stats,
        }


# ---------------------------------------------------------------------------
# Analysis primitives
# ---------------------------------------------------------------------------

def longest_identical_run(digests: Sequence[int]) -> Tuple[int, int]:
    """Return (run_length, start_index) of the longest run of equal digests."""
    if not digests:
        return 0, 0
    best_len, best_start = 1, 0
    cur_len, cur_start = 1, 0
    for i in range(1, len(digests)):
        if digests[i] == digests[i - 1]:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
            cur_len, cur_start = 1, i
    if cur_len > best_len:
        best_len, best_start = cur_len, cur_start
    return best_len, best_start


def estimate_phase_lag(
    state: np.ndarray, action: np.ndarray, max_lag: int = 15
) -> int:
    """Lag k maximizing mean per-channel Pearson correlation of
    action[t] vs state[t + k]. Positive k means action leads state."""
    T = state.shape[0]
    limit = min(max_lag, max(0, T - 8))
    best_k, best_score = 0, -2.0
    for k in range(-limit, limit + 1):
        if k >= 0:
            a, s = action[: T - k], state[k:]
        else:
            a, s = action[-k:], state[: T + k]
        if a.shape[0] < 8:
            continue
        corrs: List[float] = []
        for j in range(a.shape[1]):
            x = a[:, j].astype(np.float64)
            y = s[:, j].astype(np.float64)
            x -= x.mean()
            y -= y.mean()
            denom = float(np.linalg.norm(x) * np.linalg.norm(y))
            if denom < 1e-12:
                continue
            corrs.append(float(np.dot(x, y) / denom))
        if not corrs:
            continue
        score = sum(corrs) / len(corrs)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


# ---------------------------------------------------------------------------
# Action-bound check (C4 / ±5% tolerance gate, wired into validate_episode)
# ---------------------------------------------------------------------------

def check_action_bounds(
    action: Optional[np.ndarray],
    channel_names: Sequence[str] = CANONICAL_CHANNEL_NAMES,
    limits: Optional[np.ndarray] = None,
    margin_pct: float = 0.05,
    eps: float = 1e-4,
) -> List[Finding]:
    """Return REJECT findings for any action channel exceeding physical limits ±5%.

    For channels whose physical span is zero (torso_pitch, torso_yaw,
    base_linear_y in the fixed-mast / diff-drive configuration) the ±eps absolute
    tolerance is applied instead of a percentage so that valid zero-valued frames
    never trigger a false rejection.

    Args:
        action:        (T, 19) float32 action matrix from the parquet episode.
        channel_names: per-channel label sequence (used in finding messages).
        limits:        (19, 2) float32 array of [min, max] physical bounds.
                       Defaults to CANONICAL_ACTION_LIMITS.
        margin_pct:    fractional tolerance applied to the span (default 5 %).
        eps:           absolute tolerance for zero-span channels (default 1e-4).

    Returns:
        List of Finding objects; empty when all channels are within bounds.
    """
    findings: List[Finding] = []
    if action is None or action.ndim != 2 or action.shape[0] == 0:
        return findings
    if limits is None:
        limits = CANONICAL_ACTION_LIMITS
    dim = min(action.shape[1], limits.shape[0])
    for ch in range(dim):
        ch_name = channel_names[ch] if ch < len(channel_names) else f"ch_{ch}"
        lo = float(limits[ch, 0])
        hi = float(limits[ch, 1])
        span = hi - lo
        # Zero-span channel: use eps absolute window (e.g. torso_pitch = 0.0).
        if span < eps:
            margin = eps
        else:
            margin = span * margin_pct
        lower_bound = lo - margin
        upper_bound = hi + margin
        vals = action[:, ch]
        valid_mask = np.isfinite(vals)
        if not np.any(valid_mask):
            continue
        finite_vals = vals[valid_mask]
        min_val = float(np.min(finite_vals))
        max_val = float(np.max(finite_vals))
        if min_val < lower_bound or max_val > upper_bound:
            findings.append(
                Finding(
                    code=CODE_ACTION_OUT_OF_BOUNDS,
                    severity=SEV_REJECT,
                    message=(
                        f"channel {ch} ({ch_name}) action range "
                        f"[{min_val:.5f}, {max_val:.5f}] exceeds physical limits "
                        f"[{lo:.5f}, {hi:.5f}] beyond ±{margin_pct * 100:.0f}% tolerance "
                        f"(allowed window [{lower_bound:.5f}, {upper_bound:.5f}])"
                    ),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Validation engine (shared by the CLI and the C10 fixture suite)
# ---------------------------------------------------------------------------

def validate_episode(

    record: EpisodeRecord, config: QAConfig, valid_task_indices: frozenset
) -> EpisodeResult:
    findings: List[Finding] = []
    stats: Dict[str, object] = {
        "parquet_path": record.parquet_path,
        "parquet_rows": record.parquet_rows,
    }

    # 0. Deserialization failure (FIX-TRUNC-PARQUET class): reject immediately.
    if record.read_error is not None:
        findings.append(
            Finding(
                CODE_READ_FAILURE,
                SEV_REJECT,
                f"deserialization failure: {record.read_error}",
            )
        )
        return EpisodeResult(record.episode_index, SEV_REJECT, findings, stats)

    rows = record.parquet_rows

    # 1. Source profile: HC-8 enum membership + required profile match.
    if record.source_profile is None:
        findings.append(
            Finding(
                CODE_MISSING_EPISODE_METADATA,
                SEV_REJECT,
                "episode has no entry in meta/episodes.jsonl (source_profile unknown)",
            )
        )
    elif record.source_profile not in ALLOWED_SOURCE_PROFILES:
        findings.append(
            Finding(
                CODE_INVALID_SOURCE_PROFILE,
                SEV_REJECT,
                f"source_profile '{record.source_profile}' not in {sorted(ALLOWED_SOURCE_PROFILES)}",
            )
        )
    elif record.source_profile != config.require_source_profile:
        findings.append(
            Finding(
                CODE_SOURCE_PROFILE_MISMATCH,
                SEV_REJECT,
                f"source_profile '{record.source_profile}' != required "
                f"'{config.require_source_profile}'",
            )
        )

    # 2. Referential integrity: task_index must exist in meta/tasks.jsonl.
    note_task = record.load_notes.get("task_index")
    if record.task_index is None:
        detail = f" ({note_task})" if note_task else ""
        findings.append(
            Finding(
                CODE_ORPHAN_TASK,
                SEV_REJECT,
                f"episode has no usable task_index{detail}",
            )
        )
    elif record.task_index not in valid_task_indices:
        findings.append(
            Finding(
                CODE_ORPHAN_TASK,
                SEV_REJECT,
                f"task_index {record.task_index} not present in meta/tasks.jsonl "
                f"(known: {sorted(valid_task_indices)})",
            )
        )

    # 3. Episode completeness.
    if rows < config.min_frames:
        findings.append(
            Finding(
                CODE_EPISODE_TOO_SHORT,
                SEV_REJECT,
                f"{rows} frames < minimum {config.min_frames}",
            )
        )
    if record.declared_length is not None and record.declared_length != rows:
        findings.append(
            Finding(
                CODE_FRAME_COUNT_MISMATCH,
                SEV_REJECT,
                f"meta/episodes.jsonl declares length {record.declared_length} "
                f"but parquet has {rows} rows",
            )
        )

    # 4. Dimension and value integrity: flat 1-D float32 vectors of state_dim,
    #    no NaN / Inf / null (HC-7).
    for name, arr in (
        ("observation.state", record.observation_state),
        ("action", record.action),
    ):
        note = record.load_notes.get(name)
        if arr is None:
            if note == "null":
                findings.append(
                    Finding(CODE_NON_FINITE_VALUE, SEV_REJECT, f"'{name}' contains null values")
                )
            elif note == "ragged":
                findings.append(
                    Finding(
                        CODE_SCHEMA_VIOLATION,
                        SEV_REJECT,
                        f"'{name}' rows are ragged / not uniform 1-D vectors (HC-7)",
                    )
                )
            else:
                findings.append(
                    Finding(
                        CODE_SCHEMA_VIOLATION,
                        SEV_REJECT,
                        f"'{name}' column missing or undecodable",
                    )
                )
            continue
        if arr.ndim != 2:
            findings.append(
                Finding(
                    CODE_SCHEMA_VIOLATION,
                    SEV_REJECT,
                    f"'{name}' must be a stack of flat 1-D vectors, got ndim={arr.ndim} (HC-7)",
                )
            )
            continue
        if arr.shape[1] != config.state_dim:
            findings.append(
                Finding(
                    CODE_SCHEMA_VIOLATION,
                    SEV_REJECT,
                    f"'{name}' vector length {arr.shape[1]} != frozen state dim "
                    f"{config.state_dim} (FIX-DIM-MISMATCH class)",
                )
            )
            continue
        if arr.dtype != np.float32:
            findings.append(
                Finding(
                    CODE_SCHEMA_VIOLATION,
                    SEV_REJECT,
                    f"'{name}' dtype {arr.dtype} != float32 (HC-7)",
                )
            )
        if not np.all(np.isfinite(arr)):
            bad = int(np.count_nonzero(~np.isfinite(arr)))
            findings.append(
                Finding(
                    CODE_NON_FINITE_VALUE,
                    SEV_REJECT,
                    f"'{name}' contains {bad} NaN/Inf sample(s) (FIX-OOB-STATS class)",
                )
            )

    # 5. Timebase: strict monotonicity + inter-stream skew (HC-5).
    host = record.timestamp
    dev = record.timestamp_device
    if host is None or dev is None:
        missing = []
        if host is None:
            missing.append(f"timestamp ({record.load_notes.get('timestamp', 'missing')})")
        if dev is None:
            missing.append(
                f"timestamp_device ({record.load_notes.get('timestamp_device', 'missing')})"
            )
        findings.append(
            Finding(CODE_TIMEBASE_INVALID, SEV_REJECT, "timebase unavailable: " + ", ".join(missing))
        )
    elif host.ndim != 1 or dev.ndim != 1 or host.shape[0] != rows or dev.shape[0] != rows:
        findings.append(
            Finding(
                CODE_TIMEBASE_INVALID,
                SEV_REJECT,
                f"timebase lengths (host={host.shape}, device={dev.shape}) != parquet rows {rows}",
            )
        )
    else:
        stats["duration_sec"] = float(host[-1] - host[0]) if host.size else 0.0
        if host.size >= 2:
            d_host = np.diff(host)
            if np.any(d_host <= 0.0):
                i = int(np.argmax(d_host <= 0.0)) + 1
                findings.append(
                    Finding(
                        CODE_MONOTONICITY_VIOLATION,
                        SEV_REJECT,
                        f"timestamp not strictly increasing at frame {i}: "
                        f"{host[i]:.9f} <= {host[i - 1]:.9f} (FIX-TS-JITTER class)",
                    )
                )
            d_dev = np.diff(dev)
            if np.any(d_dev <= 0.0):
                i = int(np.argmax(d_dev <= 0.0)) + 1
                findings.append(
                    Finding(
                        CODE_DEVICE_MONOTONICITY_VIOLATION,
                        SEV_REJECT,
                        f"timestamp_device not strictly increasing at frame {i}: "
                        f"{dev[i]:.9f} <= {dev[i - 1]:.9f}",
                    )
                )
        skew_ms = np.abs(host - dev) * 1000.0
        stats["skew_ms"] = {
            "max": float(skew_ms.max()) if skew_ms.size else 0.0,
            "mean": float(skew_ms.mean()) if skew_ms.size else 0.0,
            "p95": float(np.percentile(skew_ms, 95)) if skew_ms.size else 0.0,
        }
        if skew_ms.size and float(skew_ms.max()) > config.skew_threshold_ms:
            findings.append(
                Finding(
                    CODE_SKEW_THRESHOLD_EXCEEDED,
                    SEV_REJECT,
                    f"max inter-stream skew {skew_ms.max():.2f} ms > threshold "
                    f"{config.skew_threshold_ms:.1f} ms",
                )
            )

    # 6. Camera freshness + container/parquet frame-count agreement.
    freshness: Dict[str, object] = {}
    for cam in CANONICAL_CAMERAS:
        count = record.video_frame_counts.get(cam)
        if count is None:
            if config.video_checks_enabled:
                findings.append(
                    Finding(
                        CODE_VIDEO_STREAM_MISSING,
                        SEV_REJECT,
                        f"no container frame count available for '{cam}'",
                    )
                )
            continue
        if count != rows:
            findings.append(
                Finding(
                    CODE_FRAME_COUNT_MISMATCH,
                    SEV_REJECT,
                    f"'{cam}' container frames {count} != parquet rows {rows} "
                    "(FIX-DROP-FRAME class)",
                )
            )
        dur = record.video_durations.get(cam)
        required_fps = config.fps * 0.95
        if dur is not None and dur > 0:
            ratio = count / dur
            ok = ratio >= required_fps
            freshness[cam] = {
                "frame_count": int(count),
                "duration_sec": float(dur),
                "measured_fps": float(ratio),
                "required_fps": float(required_fps),
                "ok": bool(ok),
            }
            if not ok:
                findings.append(
                    Finding(
                        CODE_CAMERA_FPS_LOW,
                        SEV_REJECT,
                        f"'{cam}' freshness {ratio:.2f} fps < required {required_fps:.2f} fps",
                    )
                )
        else:
            freshness[cam] = {
                "frame_count": int(count),
                "duration_sec": None,
                "measured_fps": None,
                "required_fps": float(required_fps),
                "ok": None,
            }
    stats["camera_freshness"] = freshness

    # 7. Entropy / frozen-frame detection (FIX-FROZEN-CAM class).
    frozen_report: Dict[str, object] = {}
    for cam, digests in record.camera_frame_digests.items():
        if not digests:
            continue
        run_len, run_start = longest_identical_run(digests)
        run_sec = run_len / config.fps
        frozen_report[cam] = {
            "max_identical_run_frames": int(run_len),
            "max_identical_run_sec": float(run_sec),
            "run_start_frame": int(run_start),
        }
        if run_sec >= config.frozen_threshold_sec:
            findings.append(
                Finding(
                    CODE_FROZEN_CAMERA,
                    SEV_FLAG,
                    f"'{cam}' frozen for {run_sec:.2f} s starting at frame {run_start} "
                    f"(>= {config.frozen_threshold_sec:.1f} s threshold; entropy below minimum)",
                )
            )
    stats["frozen_camera_analysis"] = frozen_report

    # 8. Phase lag: action vs observation.state cross-correlation
    #    (FIX-PHASE-LAG class, FLAG).
    if (
        isinstance(record.observation_state, np.ndarray)
        and isinstance(record.action, np.ndarray)
        and record.observation_state.ndim == 2
        and record.action.ndim == 2
        and record.observation_state.shape[1] == config.state_dim
        and record.action.shape[1] == config.state_dim
        and record.observation_state.shape[0] == record.action.shape[0]
        and record.observation_state.shape[0] >= 16
    ):
        lag = estimate_phase_lag(record.observation_state, record.action)
        stats["phase_lag_frames"] = int(lag)
        if abs(lag) > config.max_phase_lag:
            findings.append(
                Finding(
                    CODE_PHASE_LAG,
                    SEV_FLAG,
                    f"action/state cross-correlation peaks at lag {lag} frames "
                    f"(tolerance +/-{config.max_phase_lag}); cross-correlation anomaly",
                )
            )

    # 9. C13 subtask histogram + taxonomy sanity (informational).
    if record.subtask_index is not None and record.subtask_index.size:
        vals, counts = np.unique(record.subtask_index, return_counts=True)
        stats["subtask_histogram"] = {str(int(v)): int(c) for v, c in zip(vals, counts)}
        out_of_range = [int(v) for v in vals if v < SUBTASK_INDEX_MIN or v > SUBTASK_INDEX_MAX]
        if out_of_range:
            findings.append(
                Finding(
                    CODE_SUBTASK_OUT_OF_RANGE,
                    SEV_WARN,
                    f"subtask_index values {out_of_range} outside taxonomy "
                    f"{SUBTASK_INDEX_MIN}..{SUBTASK_INDEX_MAX}",
                )
            )

    # 10. Action out-of-bounds gate (±5% of physical limits, C4 audit 2026-10-13).
    findings.extend(check_action_bounds(record.action))

    if any(f.severity == SEV_REJECT for f in findings):
        status = SEV_REJECT
    elif any(f.severity == SEV_FLAG for f in findings):
        status = SEV_FLAG
    else:
        status = STATUS_PASS
    return EpisodeResult(record.episode_index, status, findings, stats)


# ---------------------------------------------------------------------------
# Parquet loading (lazy pyarrow import keeps fixture tests dependency-light)
# ---------------------------------------------------------------------------

_EPISODE_NAME_RE = re.compile(r"episode_(\d+)\.parquet$")


def _episode_index_from_path(path: Path) -> int:
    m = _EPISODE_NAME_RE.search(path.name)
    return int(m.group(1)) if m else -1


def _extract_matrix(table, name: str, notes: Dict[str, str]) -> Optional[np.ndarray]:
    if name not in table.schema.names:
        notes[name] = "missing"
        return None
    col = table.column(name)
    if col.null_count > 0:
        notes[name] = "null"
        return None
    try:
        rows = col.to_pylist()
    except Exception as exc:
        notes[name] = f"undecodable: {type(exc).__name__}: {exc}"
        return None
    if any(r is None for r in rows):
        notes[name] = "null"
        return None
    try:
        arr = np.asarray(rows, dtype=np.float32)
    except (TypeError, ValueError):
        notes[name] = "ragged"
        return None
    if arr.ndim != 2:
        notes[name] = "ragged"
        return None
    return arr


def _extract_timestamps(table, name: str, notes: Dict[str, str]) -> Optional[np.ndarray]:
    if name not in table.schema.names:
        notes[name] = "missing"
        return None
    col = table.column(name)
    if col.null_count > 0:
        notes[name] = "null"
        return None
    try:
        arr = col.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    except Exception:
        notes[name] = "undecodable"
        return None
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        notes[name] = "invalid"
        return None
    return arr


def load_parquet_episode(path: Path) -> EpisodeRecord:
    """Load one episode parquet into an EpisodeRecord.

    Any deserialization failure (truncated footer, corrupt pages, missing
    magic bytes) is captured in read_error and later rejected as READ_FAILURE.
    """
    rec = EpisodeRecord(episode_index=_episode_index_from_path(path), parquet_path=str(path))
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
    except Exception as exc:
        rec.read_error = f"{type(exc).__name__}: {exc}"
        return rec

    rec.parquet_rows = int(table.num_rows)
    rec.observation_state = _extract_matrix(table, "observation.state", rec.load_notes)
    rec.action = _extract_matrix(table, "action", rec.load_notes)
    rec.timestamp = _extract_timestamps(table, "timestamp", rec.load_notes)
    rec.timestamp_device = _extract_timestamps(table, "timestamp_device", rec.load_notes)

    if "task_index" in table.schema.names:
        vals = [v for v in table.column("task_index").to_pylist() if v is not None]
        uniq = {int(v) for v in vals}
        if len(uniq) == 1:
            rec.task_index = int(next(iter(uniq)))
        elif len(uniq) > 1:
            rec.load_notes["task_index"] = "inconsistent within episode"
    else:
        rec.load_notes["task_index"] = "missing"

    if "subtask_index" in table.schema.names:
        col = table.column("subtask_index")
        if col.null_count == 0:
            try:
                rec.subtask_index = col.to_numpy(zero_copy_only=False).astype(np.int64)
            except Exception:
                pass
    return rec


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no}: invalid JSON ({exc})") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path.name}:{line_no}: expected JSON object")
            rows.append(obj)
    return rows


def load_task_registry(path: Path) -> Dict[int, str]:
    """task_index -> language instruction (mandatory per C9)."""
    tasks: Dict[int, str] = {}
    for row in load_jsonl(path):
        ti = row.get("task_index")
        if not isinstance(ti, int):
            continue
        instruction = row.get("task")
        if instruction is None:
            instruction = row.get("language_instruction")
        tasks[ti] = str(instruction) if instruction is not None else ""
    return tasks


def load_episode_registry(path: Path) -> Dict[int, dict]:
    """episode_index -> meta/episodes.jsonl entry."""
    episodes: Dict[int, dict] = {}
    for row in load_jsonl(path):
        ei = row.get("episode_index")
        if isinstance(ei, int):
            episodes[ei] = row
    return episodes


def read_dataset_fps(meta_dir: Path, default: float) -> float:
    info_path = meta_dir / "info.json"
    if not info_path.is_file():
        return default
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        fps = info.get("fps")
        if isinstance(fps, (int, float)) and not isinstance(fps, bool) and fps > 0:
            return float(fps)
    except (OSError, json.JSONDecodeError):
        LOG.warning("meta/info.json unreadable; assuming fps=%s", default)
    return default


def discover_parquet_files(dataset_path: Path) -> List[Path]:
    return sorted((dataset_path / "data").glob("chunk-*/episode_*.parquet"))


# ---------------------------------------------------------------------------
# Video tooling (ffprobe + optional OpenCV)
# ---------------------------------------------------------------------------

def probe_video_metadata(path: Path) -> Optional[Dict[str, object]]:
    """Return {'nb_frames': int|None, 'duration': float|None}; None on failure."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        quick = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,duration", "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=120, check=True,
        )
        payload = json.loads(quick.stdout)
        streams = payload.get("streams") or []
        nb_frames: Optional[int] = None
        duration: Optional[float] = None
        if streams:
            raw_nb = streams[0].get("nb_frames")
            if raw_nb not in (None, "N/A"):
                nb_frames = int(raw_nb)
            raw_dur = streams[0].get("duration")
            if raw_dur not in (None, "N/A"):
                duration = float(raw_dur)
        if nb_frames is None:
            counted = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                    "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
                ],
                capture_output=True, text=True, timeout=600, check=True,
            )
            text = counted.stdout.strip()
            if text.isdigit():
                nb_frames = int(text)
        if duration is None:
            fmt = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", str(path),
                ],
                capture_output=True, text=True, timeout=120, check=True,
            )
            try:
                duration = float(fmt.stdout.strip())
            except ValueError:
                duration = None
        return {"nb_frames": nb_frames, "duration": duration}
    except (
        subprocess.SubprocessError, OSError, ValueError, KeyError, IndexError,
        json.JSONDecodeError,
    ) as exc:
        LOG.warning("ffprobe failed on %s: %s", path, exc)
        return None


def decode_frame_digests(path: Path, max_frames: Optional[int] = None) -> Optional[List[int]]:
    """Per-frame CRC32 digests of a downscaled grayscale render.

    Identical source frames produce identical digests, which powers the
    frozen-camera (entropy) check. Returns None when OpenCV is unavailable.
    """
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        LOG.warning("cv2 could not open %s", path)
        return None
    digests: List[int] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
            digests.append(zlib.crc32(small.tobytes()) & 0xFFFFFFFF)
            if max_frames is not None and len(digests) >= max_frames:
                break
    finally:
        cap.release()
    return digests


def _cv2_available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_qa(
    dataset_path: Path,
    config: QAConfig,
    skip_video_checks: bool = False,
    decode_frames: bool = True,
) -> Dict[str, object]:
    dataset_path = Path(dataset_path)
    dataset_findings: List[Finding] = []
    meta_dir = dataset_path / "meta"

    # Mandatory metadata files.
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks: Dict[int, str] = {}
    episodes_meta: Dict[int, dict] = {}
    tasks_ok = tasks_path.is_file()
    episodes_ok = episodes_path.is_file()
    if not tasks_ok:
        dataset_findings.append(
            Finding(CODE_MISSING_META_FILE, SEV_REJECT, "mandatory meta/tasks.jsonl is missing")
        )
    else:
        try:
            tasks = load_task_registry(tasks_path)
            if not tasks:
                dataset_findings.append(
                    Finding(
                        CODE_MISSING_META_FILE,
                        SEV_REJECT,
                        "meta/tasks.jsonl contains no valid task entries",
                    )
                )
        except (OSError, ValueError) as exc:
            tasks_ok = False
            dataset_findings.append(
                Finding(CODE_MISSING_META_FILE, SEV_REJECT, f"meta/tasks.jsonl unreadable: {exc}")
            )
    if not episodes_ok:
        dataset_findings.append(
            Finding(
                CODE_MISSING_META_FILE, SEV_REJECT, "mandatory meta/episodes.jsonl is missing"
            )
        )
    else:
        try:
            episodes_meta = load_episode_registry(episodes_path)
        except (OSError, ValueError) as exc:
            episodes_ok = False
            dataset_findings.append(
                Finding(
                    CODE_MISSING_META_FILE,
                    SEV_REJECT,
                    f"meta/episodes.jsonl unreadable: {exc}",
                )
            )

    # FPS source of truth: meta/info.json when present.
    config = replace(config, fps=read_dataset_fps(meta_dir, config.fps))

    # Environment guard: ffprobe is required unless video checks are skipped.
    probe_available = shutil.which("ffprobe") is not None
    if not skip_video_checks and not probe_available:
        raise EnvironmentError(
            "ffprobe not found: install ffmpeg or re-run with --skip-video-checks"
        )

    files = discover_parquet_files(dataset_path)
    if not files:
        dataset_findings.append(
            Finding(
                CODE_NO_EPISODES_FOUND,
                SEV_REJECT,
                f"no episode parquet files under {dataset_path / 'data'}",
            )
        )

    valid_tasks = frozenset(tasks.keys())
    results: List[EpisodeResult] = []

    for pf in files:
        rec = load_parquet_episode(pf)

        # Attach episodes.jsonl metadata (source_profile, declared length).
        meta_entry = episodes_meta.get(rec.episode_index)
        if meta_entry is None:
            rec.load_notes["episodes.jsonl"] = "missing entry"
            rec.source_profile = None
        else:
            sp = meta_entry.get("source_profile")
            rec.source_profile = sp if isinstance(sp, str) else None
            dl = meta_entry.get("length")
            if isinstance(dl, int):
                rec.declared_length = dl

        # Attach video container metadata + frame digests.
        if not skip_video_checks and rec.read_error is None:
            chunk_dir = pf.parent.name
            for cam in CANONICAL_CAMERAS:
                mp4 = dataset_path / "videos" / chunk_dir / cam / (pf.stem + ".mp4")
                if not mp4.is_file():
                    LOG.debug("video file missing for %s: %s", pf.name, mp4)
                    continue
                meta = probe_video_metadata(mp4)
                if meta is not None:
                    if meta["nb_frames"] is not None:
                        rec.video_frame_counts[cam] = int(meta["nb_frames"])
                    if meta["duration"] is not None:
                        rec.video_durations[cam] = float(meta["duration"])
                if decode_frames:
                    digests = decode_frame_digests(mp4)
                    if digests is not None:
                        rec.camera_frame_digests[cam] = digests

        result = validate_episode(rec, config, valid_tasks)
        results.append(result)
        if result.status == SEV_REJECT:
            LOG.warning(
                "episode %s REJECTED (%d finding(s))", rec.episode_index, len(result.findings)
            )
        elif result.status == SEV_FLAG:
            LOG.info(
                "episode %s FLAGGED (%d finding(s))", rec.episode_index, len(result.findings)
            )
        else:
            LOG.info("episode %s PASS", rec.episode_index)

    # Aggregate report.
    rejected = sum(1 for r in results if r.status == SEV_REJECT)
    flagged = sum(1 for r in results if r.status == SEV_FLAG)
    passed = sum(1 for r in results if r.status == STATUS_PASS)
    code_counts: Dict[str, int] = {}
    for r in results:
        for f in r.findings:
            code_counts[f.code] = code_counts.get(f.code, 0) + 1
    for f in dataset_findings:
        code_counts[f.code] = code_counts.get(f.code, 0) + 1
    dataset_blocked = any(f.severity == SEV_REJECT for f in dataset_findings)
    gate = "BLOCKED" if (rejected > 0 or dataset_blocked) else "OPEN"

    report: Dict[str, object] = {
        "qa_report_schema": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "deliverable": "C10 Dataset QA (C-TEAM-PLAN-01 rev2.0 section 3.2)",
        "schema_revision": SCHEMA_REVISION,
        "dataset_path": str(dataset_path),
        "config": config.to_dict(),
        "dataset_checks": {
            "meta_tasks_jsonl": {"present": tasks_ok, "task_count": len(tasks)},
            "meta_episodes_jsonl": {"present": episodes_ok, "episode_count": len(episodes_meta)},
            "ffprobe_available": probe_available,
            "video_checks_enabled": not skip_video_checks,
            "frame_decoding_enabled": bool(decode_frames and _cv2_available()),
            "camera_keys": list(CANONICAL_CAMERAS),
            "episodes_discovered": len(files),
            "findings": [f.to_dict() for f in dataset_findings],
        },
        "summary": {
            "episodes_total": len(results),
            "passed": passed,
            "flagged": flagged,
            "rejected": rejected,
            "finding_code_counts": code_counts,
            "training_gate": gate,
            "overall_status": "FAIL" if gate == "BLOCKED" else "PASS",
        },
        "episodes": [r.to_dict() for r in results],
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EXAMPLE = """
examples:
  # Plan section 3.2 invocation (gates training jobs):
  python scripts/check_dataset_integrity.py \\
      --dataset-path /data/openamrobot/episodes/ \\
      --skew-threshold-ms 30.0 \\
      --min-frames 30 \\
      --require-source-profile ROBOT \\
      --output-report /data/reports/qa_report.json

  # Degraded mode on machines without ffmpeg:
  python scripts/check_dataset_integrity.py --dataset-path ./episodes --skip-video-checks

exit codes:
  0  training gate OPEN (no REJECT findings; FLAGs are reported for triage)
  1  training gate BLOCKED (one or more REJECT findings)
  2  environment or usage error
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_dataset_integrity.py",
        description="OpenAMRobot 2.0 Workstream C - C10 dataset QA pre-flight gate",
        epilog=_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-path", required=True, type=Path,
                        help="Directory containing episode files and metadata")
    parser.add_argument("--skew-threshold-ms", type=float, default=30.0,
                        help="Maximum allowed |timestamp - timestamp_device| in ms (default: 30.0)")
    parser.add_argument("--min-frames", type=int, default=30,
                        help="Minimum frames required per episode (default: 30)")
    parser.add_argument("--require-source-profile", type=str, default="ROBOT",
                        help="Required source profile (default: ROBOT)")
    parser.add_argument("--output-report", type=Path, default=Path("qa_report.json"),
                        help="Path to write the JSON QA report (default: qa_report.json)")
    parser.add_argument("--frozen-threshold-sec", type=float, default=2.0,
                        help="Identical-frame run length that triggers FROZEN_CAMERA (default: 2.0)")
    parser.add_argument("--max-phase-lag", type=int, default=3,
                        help="Tolerated action/state cross-correlation lag in frames (default: 3)")
    parser.add_argument("--skip-video-checks", action="store_true",
                        help="Disable container frame-count / freshness / decode checks")
    parser.add_argument("--skip-frame-decode", action="store_true",
                        help="Disable OpenCV frame decoding (frozen-camera check)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )

    profile = args.require_source_profile.upper()
    if profile not in ALLOWED_SOURCE_PROFILES:
        LOG.error(
            "--require-source-profile must be one of %s", sorted(ALLOWED_SOURCE_PROFILES)
        )
        return 2
    if not args.dataset_path.is_dir():
        LOG.error("dataset path does not exist: %s", args.dataset_path)
        return 2

    config = QAConfig(
        skew_threshold_ms=args.skew_threshold_ms,
        min_frames=args.min_frames,
        require_source_profile=profile,
        frozen_threshold_sec=args.frozen_threshold_sec,
        max_phase_lag=args.max_phase_lag,
        video_checks_enabled=not args.skip_video_checks,
    )

    try:
        report = run_qa(
            args.dataset_path,
            config,
            skip_video_checks=args.skip_video_checks,
            decode_frames=not args.skip_frame_decode,
        )
    except EnvironmentError as exc:
        LOG.error("%s", exc)
        return 2

    if args.output_report is not None:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(
            json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        LOG.info("QA report written to %s", args.output_report)

    s = report["summary"]
    print("=" * 78)
    print(
        f"C10 PRE-FLIGHT QA | episodes: {s['episodes_total']} total | "
        f"{s['passed']} passed | {s['flagged']} flagged | {s['rejected']} rejected"
    )
    if s["finding_code_counts"]:
        for code, n in sorted(s["finding_code_counts"].items()):
            print(f"  {code}: {n}")
    print(f"TRAINING GATE: {s['training_gate']} (overall {s['overall_status']})")
    print("=" * 78)
    return 0 if s["training_gate"] == "OPEN" else 1


if __name__ == "__main__":
    sys.exit(main())