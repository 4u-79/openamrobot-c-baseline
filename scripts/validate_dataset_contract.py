#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_dataset_contract.py
============================

OpenAMRobot 2.0 | Workstream C | Deliverables C2 + C9
Schema & LeRobotDataset contract validator (Phase 1 — Schema Freeze Gate).

Binding references
------------------
- C-TEAM-PLAN-01 rev 2.0 (section 3.1, Phase 1 schedule)
- P-02 rev18 C2 / C9 / A6, P-00 rev18 sections 20-21, Q-01 rev18
- Hard constraints: HC-3 (LeRobotDataset canonical), HC-5 (timebase),
  HC-6 (camera keys), HC-7 (flat 1-D float32 vectors), HC-12 (freeze rules)

Checks performed
----------------
Contract level (meta/info.json):
  1. codebase_version == "v3.0", robot_type == "openamrobot_2", fps == 30
  2. observation.state / action: dtype float32, flat 1-D shape [21],
     names a flat list of strings (nested feature dicts rejected, HC-7),
     channel ordering consistent with the rev18 enumeration
  3. All 4 canonical camera keys present with declared shapes + video info
  4. Dual timebase: timestamp (jetson_host_monotonic_sec) and
     timestamp_device (device_native_sec), float64, shape [1] (HC-5)
  5. Index/metadata columns: episode_index, frame_index, index,
     task_index, subtask_index (int64, shape [1])

Data level (--dataset-path; requires pandas + pyarrow):
  6. Per-episode parquet: vector dimension/dtype, non-finite guard,
     timestamp / timestamp_device monotonicity, frame_index continuity
  7. Optional inter-stream skew drift against --skew-threshold-ms
  8. Per-camera video file presence for every declared camera key

Usage
-----
Contract only (Phase 1 gate):
    python scripts/validate_dataset_contract.py --info-json meta/info.json

Contract + recorded episode data:
    python scripts/validate_dataset_contract.py \
        --dataset-path /data/openamrobot/episodes --skew-threshold-ms 30.0

Exit codes: 0 = PASS, 1 = FAIL (contract violation), 2 = usage/environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger("contract-validator")

# ---------------------------------------------------------------------------
# Frozen contract constants (C-TEAM-PLAN-01 rev2.0 section 3.1)
# ---------------------------------------------------------------------------

EXPECTED_CODEBASE_VERSION = "v3.0"
EXPECTED_ROBOT_TYPE = "openamrobot_2"
EXPECTED_FPS = 30
EXPECTED_STATE_DIM = 21  # HC-7: flat 1-D torch.float32 / float32 vectors

# Channel names enumerated by rev18 (P-02 C2 schema definition).
# FREEZE BLOCKER 2026-10-09: rev18 declares shape [21] but enumerates only 19
# channel names. The two missing channels must be confirmed against the A6
# canonical URDF (owner: Tushar) BEFORE tagging schema-v1.0-frozen (HC-12).
# Ordering here is normative: the final schema must preserve this relative
# order of the enumerated channels.
CANONICAL_VECTOR_NAMES_REV18: Tuple[str, ...] = (
    "base_linear_x",
    "base_linear_y",
    "base_angular_z",
    "torso_pitch",
    "torso_yaw",
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_wrist_yaw",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_wrist_yaw",
    "right_gripper",
)

# HC-6: fixed camera keys; shape order is (height, width, channels).
CANONICAL_CAMERA_FEATURES: Dict[str, Tuple[int, int, int]] = {
    "observation.images.camera_wrist_left": (480, 640, 3),
    "observation.images.camera_wrist_right": (480, 640, 3),
    "observation.images.camera_scene": (720, 1280, 3),
    "observation.images.camera_base_nav": (480, 640, 3),
}

EXPECTED_VIDEO_INFO: Dict[str, object] = {
    "codec": "libx264",
    "fps": 30.0,
    "pix_fmt": "yuv420p",
}

# HC-5: canonical timebase is the Jetson host monotonic clock; device-native
# timestamps are retained alongside.
TIMESTAMP_FEATURES: Dict[str, str] = {
    "timestamp": "jetson_host_monotonic_sec",
    "timestamp_device": "device_native_sec",
}

INDEX_FEATURES: Tuple[str, ...] = (
    "episode_index",
    "frame_index",
    "index",
    "task_index",
    "subtask_index",
)

REQUIRED_DATA_COLUMNS: Tuple[str, ...] = (
    "observation.state",
    "action",
    "timestamp",
    "timestamp_device",
    "episode_index",
    "frame_index",
    "index",
    "task_index",
    "subtask_index",
)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class Report:
    """Collects pass/fail results for every contract check."""

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    def record(self, name: str, passed: bool, detail: str) -> None:
        self.results.append((name, passed, detail))
        if passed:
            LOG.info("[PASS] %-42s %s", name, detail)
        else:
            LOG.error("[FAIL] %-42s %s", name, detail)

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.results)

    def summary(self) -> None:
        failures = [(n, d) for n, ok, d in self.results if not ok]
        total = len(self.results)
        print()
        print("=" * 78)
        if self.passed:
            print(f"RESULT: PASS - {total}/{total} contract checks satisfied")
        else:
            print(f"RESULT: FAIL - {len(failures)}/{total} contract check(s) violated")
            for name, detail in failures:
                print(f"  - {name}: {detail}")
        print("=" * 78)


# ---------------------------------------------------------------------------
# Contract-level checks (meta/info.json)
# ---------------------------------------------------------------------------

def _check_name_ordering(key: str, names: List[str], report: Report) -> None:
    """The rev18-enumerated channels must appear, in their relative order."""
    positions: Dict[str, int] = {}
    for i, n in enumerate(names):
        positions.setdefault(n, i)
    last = -1
    for expected in CANONICAL_VECTOR_NAMES_REV18:
        pos = positions.get(expected)
        if pos is None:
            report.record(
                f"{key}.names.order",
                False,
                f"canonical rev18 channel '{expected}' not present in schema",
            )
            return
        if pos < last:
            report.record(
                f"{key}.names.order",
                False,
                f"canonical channel '{expected}' out of rev18 order",
            )
            return
        last = pos
    report.record(f"{key}.names.order", True, "rev18 channel ordering preserved")


def _check_flat_vector(
    features: dict, key: str, report: Report, expected_dim: int
) -> None:
    feat = features.get(key)
    if not isinstance(feat, dict):
        report.record(key, False, "feature missing or not an object")
        return

    # HC-7: reject nested feature containers outright.
    if "features" in feat:
        report.record(
            key,
            False,
            "nested feature container detected - HC-7 requires flat 1-D vectors",
        )
        return

    dtype = feat.get("dtype")
    if dtype != "float32":
        report.record(f"{key}.dtype", False, f"expected 'float32', found {dtype!r}")
    else:
        report.record(f"{key}.dtype", True, "float32")

    shape = feat.get("shape")
    flat = (
        isinstance(shape, list)
        and len(shape) == 1
        and isinstance(shape[0], int)
        and not isinstance(shape[0], bool)
    )
    if not flat:
        report.record(
            f"{key}.shape",
            False,
            f"shape must be flat 1-D [N], found {shape!r} (HC-7)",
        )
        return
    dim = shape[0]
    if dim != expected_dim:
        report.record(
            f"{key}.shape", False, f"expected [{expected_dim}], found [{dim}]"
        )
        return
    report.record(f"{key}.shape", True, f"[{dim}] flat 1-D")

    names = feat.get("names")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        report.record(
            f"{key}.names",
            False,
            "names must be a flat list of strings - nested dicts rejected (HC-7)",
        )
        return
    if len(names) != dim:
        report.record(
            f"{key}.names",
            False,
            f"shape declares {dim} channels but names enumerate {len(names)} - "
            "resolve against A6 canonical URDF before schema-v1.0-frozen (HC-12)",
        )
    else:
        report.record(f"{key}.names", True, f"{len(names)} channel names")
    if len(set(names)) != len(names):
        report.record(f"{key}.names.unique", False, "duplicate channel names detected")

    _check_name_ordering(key, names, report)


def _check_cameras(features: dict, report: Report) -> None:
    for cam_key, expected_shape in CANONICAL_CAMERA_FEATURES.items():
        feat = features.get(cam_key)
        if not isinstance(feat, dict):
            report.record(cam_key, False, "canonical camera key missing (HC-6)")
            continue

        shape_ok = feat.get("shape") == list(expected_shape)
        dtype_ok = feat.get("dtype") == "video"
        if shape_ok and dtype_ok:
            report.record(cam_key, True, f"video {list(expected_shape)}")
        else:
            report.record(
                cam_key,
                False,
                f"expected video {list(expected_shape)}, found "
                f"dtype={feat.get('dtype')!r} shape={feat.get('shape')!r} (HC-6)",
            )

        info = feat.get("info")
        if not isinstance(info, dict):
            report.record(f"{cam_key}.info", False, "video encoding info block missing")
            continue
        mismatches = [k for k, v in EXPECTED_VIDEO_INFO.items() if info.get(k) != v]
        if mismatches:
            report.record(
                f"{cam_key}.info",
                False,
                f"encoding mismatch on {mismatches} (expected {EXPECTED_VIDEO_INFO})",
            )
        else:
            report.record(f"{cam_key}.info", True, "libx264 @ 30.0 fps, yuv420p")


def _check_timestamps(features: dict, report: Report) -> None:
    for key, canonical_name in TIMESTAMP_FEATURES.items():
        feat = features.get(key)
        if not isinstance(feat, dict):
            report.record(key, False, "timebase feature missing (HC-5)")
            continue
        names = feat.get("names")
        ok = (
            feat.get("dtype") == "float64"
            and feat.get("shape") == [1]
            and isinstance(names, list)
            and names == [canonical_name]
        )
        if ok:
            report.record(key, True, f"float64 [1] '{canonical_name}'")
        else:
            report.record(
                key,
                False,
                f"expected float64 [1] names=['{canonical_name}'], found "
                f"dtype={feat.get('dtype')!r} shape={feat.get('shape')!r} "
                f"names={names!r} (HC-5)",
            )


def _check_index_fields(features: dict, report: Report) -> None:
    for key in INDEX_FEATURES:
        feat = features.get(key)
        if not isinstance(feat, dict):
            report.record(key, False, "index/metadata feature missing")
            continue
        ok = feat.get("dtype") == "int64" and feat.get("shape") == [1]
        report.record(
            key,
            ok,
            "int64 [1]"
            if ok
            else f"expected int64 [1], found dtype={feat.get('dtype')!r} shape={feat.get('shape')!r}",
        )


def validate_contract(info: dict, report: Report, expected_state_dim: int) -> None:
    report.record(
        "codebase_version",
        info.get("codebase_version") == EXPECTED_CODEBASE_VERSION,
        f"expected {EXPECTED_CODEBASE_VERSION!r}, found {info.get('codebase_version')!r}",
    )
    report.record(
        "robot_type",
        info.get("robot_type") == EXPECTED_ROBOT_TYPE,
        f"expected {EXPECTED_ROBOT_TYPE!r}, found {info.get('robot_type')!r}",
    )
    report.record(
        "fps",
        info.get("fps") == EXPECTED_FPS,
        f"expected {EXPECTED_FPS}, found {info.get('fps')!r}",
    )

    features = info.get("features")
    if not isinstance(features, dict):
        report.record("features", False, "top-level 'features' object missing")
        return

    _check_flat_vector(features, "observation.state", report, expected_state_dim)
    _check_flat_vector(features, "action", report, expected_state_dim)
    _check_cameras(features, report)
    _check_timestamps(features, report)
    _check_index_fields(features, report)


# ---------------------------------------------------------------------------
# Data-level checks (parquet episodes + video presence)
# ---------------------------------------------------------------------------

def _validate_episode_dataframe(
    pf: Path,
    df,
    report: Report,
    videos_root: Path,
    dataset_path: Path,
    skew_threshold_ms: Optional[float],
    expected_state_dim: int,
) -> None:
    import numpy as np

    ep_name = pf.stem
    missing = [c for c in REQUIRED_DATA_COLUMNS if c not in df.columns]
    if missing:
        report.record(f"{ep_name}.columns", False, f"missing columns: {missing}")
        return
    report.record(
        f"{ep_name}.columns", True, f"{len(REQUIRED_DATA_COLUMNS)} required columns present"
    )

    n_rows = len(df)

    # Flat vector features: (T, expected_state_dim), float32, finite.
    for col in ("observation.state", "action"):
        cells = [np.asarray(v) for v in df[col]]
        cell_dims = {c.ndim for c in cells}
        cell_lens = {int(c.shape[0]) for c in cells if c.ndim == 1}
        if cell_dims != {1}:
            report.record(
                f"{ep_name}.{col}",
                False,
                f"nested/ragged vector cells detected (ndims={sorted(cell_dims)}) - HC-7",
            )
            continue
        if cell_lens != {expected_state_dim}:
            report.record(
                f"{ep_name}.{col}",
                False,
                f"vector length {sorted(cell_lens)} != {expected_state_dim}",
            )
            continue
        mat = np.stack(cells)
        if mat.dtype != np.float32:
            report.record(f"{ep_name}.{col}", False, f"dtype {mat.dtype} != float32")
            continue
        if not np.all(np.isfinite(mat)):
            bad = int(np.count_nonzero(~np.isfinite(mat)))
            report.record(
                f"{ep_name}.{col}",
                False,
                f"{bad} non-finite value(s) (NaN/Inf) - FIX-OOB-STATS class",
            )
            continue
        report.record(f"{ep_name}.{col}", True, f"({n_rows}, {expected_state_dim}) float32 finite")

    # Timebase monotonicity (HC-5; FIX-TS-JITTER rejection class).
    for col in ("timestamp", "timestamp_device"):
        ts = df[col].to_numpy(dtype=np.float64).reshape(-1)
        if ts.size < 2:
            report.record(
                f"{ep_name}.{col}.monotonic",
                False,
                "fewer than 2 frames - cannot verify monotonicity",
            )
            continue
        diffs = np.diff(ts)
        if np.any(diffs < 0.0):
            i = int(np.argmax(diffs < 0.0)) + 1
            report.record(
                f"{ep_name}.{col}.monotonic",
                False,
                f"monotonicity violation at frame {i}: {ts[i]:.9f} < {ts[i - 1]:.9f} "
                "(FIX-TS-JITTER class)",
            )
        else:
            dup = int(np.count_nonzero(diffs == 0.0))
            detail = f"{ts.size} samples non-decreasing"
            if dup:
                detail += f"; WARNING {dup} duplicate timestamp(s) - possible stale frames"
            report.record(f"{ep_name}.{col}.monotonic", True, detail)

    # Optional inter-stream skew: drift of (host - device) offset about its
    # median. Host and device clocks have unrelated epochs, so only the
    # offset stability is meaningful.
    if skew_threshold_ms is not None:
        host = df["timestamp"].to_numpy(dtype=np.float64).reshape(-1)
        dev = df["timestamp_device"].to_numpy(dtype=np.float64).reshape(-1)
        offset = host - dev
        drift_ms = float(np.max(np.abs(offset - np.median(offset)))) * 1000.0
        report.record(
            f"{ep_name}.skew",
            drift_ms <= skew_threshold_ms,
            f"max host/device drift {drift_ms:.2f} ms vs threshold {skew_threshold_ms:.1f} ms",
        )

    # frame_index continuity.
    fi = df["frame_index"].to_numpy().reshape(-1)
    if fi.size == n_rows and bool((fi == np.arange(n_rows)).all()):
        report.record(f"{ep_name}.frame_index", True, f"contiguous 0..{n_rows - 1}")
    else:
        report.record(f"{ep_name}.frame_index", False, "frame_index not contiguous from 0")

    # Per-camera video file presence (deep frame-count decode is C10 scope).
    if videos_root.is_dir():
        chunk = pf.parent.name
        for cam_key in CANONICAL_CAMERA_FEATURES:
            short = cam_key.split(".")[-1]
            video_file = videos_root / chunk / cam_key / f"{ep_name}.mp4"
            report.record(
                f"{ep_name}.video.{short}",
                video_file.is_file(),
                str(video_file.relative_to(dataset_path))
                + ("" if video_file.is_file() else " - MISSING"),
            )


def validate_dataset_data(
    dataset_path: Path,
    report: Report,
    skew_threshold_ms: Optional[float],
    expected_state_dim: int,
) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        report.record(
            "data-level checks",
            False,
            f"missing dependency ({exc}); install pandas + pyarrow for data mode",
        )
        return

    parquet_files = sorted((dataset_path / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        report.record(
            "data-level checks",
            False,
            f"no episode parquet files under {dataset_path / 'data'}",
        )
        return
    report.record("episodes discovered", True, f"{len(parquet_files)} parquet file(s)")

    videos_root = dataset_path / "videos"
    if videos_root.is_dir():
        report.record("videos/ tree", True, str(videos_root))
    else:
        report.record(
            "videos/ tree",
            False,
            "videos directory missing - camera features are dtype 'video'",
        )

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            _validate_episode_dataframe(
                pf, df, report, videos_root, dataset_path,
                skew_threshold_ms, expected_state_dim,
            )
        except Exception as exc:  # any parse failure is a rejection
            report.record(pf.stem, False, f"unreadable episode data: {exc!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="OpenAMRobot 2.0 Workstream C - C2/C9 dataset contract validator",
    )
    parser.add_argument(
        "--info-json", type=Path, default=None,
        help="Path to meta/info.json (defaults to <dataset-path>/meta/info.json)",
    )
    parser.add_argument(
        "--dataset-path", type=Path, default=None,
        help="Root of a LeRobotDataset tree; enables parquet/video/timebase data checks",
    )
    parser.add_argument(
        "--skew-threshold-ms", type=float, default=None,
        help="Reject episodes whose host/device clock drift exceeds this value "
             "(plan default for C10 pre-flight: 30.0)",
    )
    parser.add_argument(
        "--expected-state-dim", type=int, default=EXPECTED_STATE_DIM,
        help="Contract state/action dimension. Default 21 per rev18. Any other "
             "value requires Project Lead sign-off and an HC-12 escalation record.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    info_path = args.info_json
    if info_path is None:
        if args.dataset_path is None:
            LOG.error("provide --info-json and/or --dataset-path")
            return 2
        info_path = args.dataset_path / "meta" / "info.json"

    if not info_path.is_file():
        LOG.error("info.json not found at %s", info_path)
        return 2
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOG.error("info.json is not valid JSON: %s", exc)
        return 2

    report = Report()
    LOG.info("Validating contract: %s", info_path)
    validate_contract(info, report, args.expected_state_dim)

    if args.dataset_path is not None:
        LOG.info("Validating episode data: %s", args.dataset_path)
        validate_dataset_data(
            args.dataset_path, report, args.skew_threshold_ms, args.expected_state_dim
        )

    report.summary()
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())