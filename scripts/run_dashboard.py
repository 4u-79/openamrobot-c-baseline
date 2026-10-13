#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_dashboard.py
================

OpenAMRobot 2.0 | Workstream C | Local Verification Portal
Single-file, zero-dependency dashboard (Python stdlib only) for inspecting
pipeline outputs before Step 2 (Google Antigravity) runs.

Sections
--------
1. Pre-Flight QA Report Viewer (C10) — upload / load-path / auto-discover
   qa_report.json produced by scripts/check_dataset_integrity.py.
2. Fixture & Test Suite Status (C10 / Step 2) — runs the real pytest suite
   tests/test_c10_fixtures.py and maps results onto the 8 mandatory fixtures.
3. Downstream milestone placeholders (C4 replay, C11 ACT, C14 promotion).

Endpoints
---------
  GET  /                      single-page UI
  GET  /api/health            repo metadata (git tag/branch when available)
  GET  /api/qa-report         resolved QA report (disk or built-in sample)
  POST /api/qa-report/upload  body: report JSON -> validated + saved to qa_report.json
  POST /api/qa-report/load    body: {"path": "..."} -> load report from disk path
  GET  /api/tests             cached/initial fixture status
  POST /api/tests/run         executes pytest suite, caches result
  GET  /api/downstream        downstream milestone stubs (C4 / C11 / C14)

Usage
-----
    python scripts/run_dashboard.py                 # http://127.0.0.1:8300
    python scripts/run_dashboard.py --port 9000 --open
    python scripts/run_dashboard.py --report reports/qa_report.json

Security note: binds to 127.0.0.1 by default. This is a local inspection
tool, not an authenticated service; do not expose the port publicly.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Contract constants (schema-v1.0-frozen, 19-channel fixed-mast)
# ---------------------------------------------------------------------------

SCHEMA_TAG = "schema-v1.0-frozen"
SYSTEM_LABEL = "19-Channel Whole-Body"
TARGET_LABEL = "OpenAMRobot 2.0"
ROBOT_TYPE = "openamrobot_2"
DATASET_FORMAT = "LeRobotDataset v3.0"

CAMERA_KEYS: Tuple[str, ...] = (
    "observation.images.camera_wrist_left",
    "observation.images.camera_wrist_right",
    "observation.images.camera_scene",
    "observation.images.camera_base_nav",
)

TESTS_REL_PATH = "tests/test_c10_fixtures.py"
CACHE_FILE_NAME = ".dashboard_test_cache.json"
REPORT_SAVE_NAME = "qa_report.json"
MAX_BODY_BYTES = 25 * 1024 * 1024
PYTEST_TIMEOUT_SEC = 600

# Mandatory fault-injection fixtures (C-TEAM-PLAN-01 rev2.0 section 3.2).
FIXTURE_DEFINITIONS: List[Dict[str, str]] = [
    {
        "id": "FIX-TS-JITTER",
        "test": "test_fix_ts_jitter_rejected",
        "fault": "t[i] < t[i-1] at random frames",
        "expected": "REJECT",
        "reason": "monotonicity violation",
    },
    {
        "id": "FIX-DROP-FRAME",
        "test": "test_fix_drop_frame_rejected",
        "fault": "MP4 truncated 5 frames vs Parquet",
        "expected": "REJECT",
        "reason": "frame count mismatch",
    },
    {
        "id": "FIX-PHASE-LAG",
        "test": "test_fix_phase_lag_flagged",
        "fault": "action shifted +5 steps vs observation.state",
        "expected": "FLAG",
        "reason": "cross-correlation anomaly",
    },
    {
        "id": "FIX-FROZEN-CAM",
        "test": "test_fix_frozen_cam_flagged",
        "fault": "identical frames for 2.0+ seconds",
        "expected": "FLAG",
        "reason": "entropy below threshold",
    },
    {
        "id": "FIX-TRUNC-PARQUET",
        "test": "test_fix_trunc_parquet_rejected",
        "fault": "last 100 bytes of Parquet footer corrupted",
        "expected": "REJECT",
        "reason": "read/deserialization failure",
    },
    {
        "id": "FIX-DIM-MISMATCH",
        "test": "test_fix_dim_mismatch_rejected",
        "fault": "vector width 18 or 20 instead of 19",
        "expected": "REJECT",
        "reason": "schema violation",
    },
    {
        "id": "FIX-OOB-STATS",
        "test": "test_fix_oob_stats_rejected",
        "fault": "NaN / +Inf / -Inf in joint channels",
        "expected": "REJECT",
        "reason": "non-finite numerical values",
    },
    {
        "id": "FIX-ORPHAN-TASK",
        "test": "test_fix_orphan_task_rejected",
        "fault": "task_index 99 with only tasks 0..2",
        "expected": "REJECT",
        "reason": "referential integrity failure",
    },
]

DOWNSTREAM_SECTIONS: List[Dict[str, object]] = [
    {
        "key": "C4",
        "title": "C4 Deterministic Replay Verification",
        "due": "30 Oct",
        "status": "AWAITING_DATA",
        "metrics": [
            "trajectory tracking error (per-episode RMSE)",
            "dynamic violation flags (base/arm interlock)",
            "result / duration / intervention records",
        ],
        "artifact": ">= 5 replay result records persisted to S3",
    },
    {
        "key": "C11",
        "title": "C11 ACT Policy Training & Loss Curves",
        "due": "6 Nov",
        "status": "AWAITING_DATA",
        "metrics": [
            "epoch loss (L1 action + KL)",
            "action chunk error (chunk_size=100)",
            "held-out evaluation results",
        ],
        "artifact": "checkpoint SHA-256 + immutable training config hash",
    },
    {
        "key": "C14",
        "title": "C14 Policy Promotion Evaluation",
        "due": "20 Nov",
        "status": "AWAITING_DATA",
        "metrics": [
            "success rate with 95% Wilson score CIs",
            "subtask phase completion (phases 0-7)",
            "failure taxonomy TYPE-A..TYPE-E",
        ],
        "artifact": "policy manifest binding dataset / checkpoint / calibration",
    },
]

# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class ServerState:
    def __init__(self, repo_root: Path, report_path: Optional[Path]) -> None:
        self.repo_root = repo_root
        self.report_path = report_path
        self.sample_report = build_sample_report()
        self.test_run_lock = threading.Lock()
        self.test_running = False
        self.cache_path = repo_root / CACHE_FILE_NAME

STATE: Optional[ServerState] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Built-in sample qa_report.json (used when no report exists on disk)
# ---------------------------------------------------------------------------

def _sample_episode(
    idx: int,
    rows: int,
    skew_max: float,
    skew_mean: float,
    findings: Optional[List[dict]] = None,
    phase_lag: int = 1,
    frozen_cam: Optional[str] = None,
    video_rows: Optional[int] = None,
) -> dict:
    duration = rows / 30.0
    vr = rows if video_rows is None else video_rows
    freshness = {}
    for i, cam in enumerate(CAMERA_KEYS):
        measured = round(29.80 + 0.05 * ((idx + i) % 4), 2)
        freshness[cam] = {
            "frame_count": int(vr),
            "duration_sec": round(duration, 4),
            "measured_fps": measured,
            "required_fps": 28.5,
            "ok": bool(measured >= 28.5),
        }
    frozen = {}
    for cam in CAMERA_KEYS:
        if cam == frozen_cam:
            frozen[cam] = {
                "max_identical_run_frames": 61,
                "max_identical_run_sec": 2.033,
                "run_start_frame": 34,
            }
        else:
            frozen[cam] = {
                "max_identical_run_frames": 1,
                "max_identical_run_sec": 0.033,
                "run_start_frame": 0,
            }
    findings = findings or []
    if any(f.get("severity") == "REJECT" for f in findings):
        status = "REJECT"
    elif any(f.get("severity") == "FLAG" for f in findings):
        status = "FLAG"
    else:
        status = "PASS"
    return {
        "episode_index": idx,
        "status": status,
        "findings": findings,
        "stats": {
            "parquet_path": f"data/chunk-000/episode_{idx:06d}.parquet",
            "parquet_rows": rows,
            "duration_sec": round(duration, 3),
            "skew_ms": {
                "max": skew_max,
                "mean": skew_mean,
                "p95": round(skew_max * 0.8, 2),
            },
            "camera_freshness": freshness,
            "frozen_camera_analysis": frozen,
            "phase_lag_frames": phase_lag,
            "subtask_histogram": {"0": rows // 8, "1": rows // 6, "2": rows // 8,
                                  "3": rows // 10, "4": rows // 4, "5": rows // 8,
                                  "6": rows // 10},
        },
    }


def build_sample_report() -> dict:
    """Deterministic sample matching check_dataset_integrity.py report schema."""
    episodes = []
    rows_cycle = [212, 188, 240, 176, 204, 228, 196, 168, 220]
    skew_cycle = [(6.4, 2.1), (4.9, 1.8), (9.7, 3.3), (3.8, 1.4), (7.2, 2.6),
                  (11.3, 3.9), (5.5, 2.0), (4.2, 1.6), (8.8, 2.9)]
    for i in range(9):
        episodes.append(_sample_episode(i, rows_cycle[i], skew_cycle[i][0], skew_cycle[i][1]))
    episodes.append(_sample_episode(
        9, 208, 5.9, 2.2, phase_lag=6,
        findings=[{
            "code": "PHASE_LAG", "severity": "FLAG",
            "message": "action/state cross-correlation peaks at lag 6 frames "
                       "(tolerance +/-3); cross-correlation anomaly",
        }],
    ))
    episodes.append(_sample_episode(
        10, 232, 7.6, 2.7,
        frozen_cam="observation.images.camera_scene",
        findings=[{
            "code": "FROZEN_CAMERA", "severity": "FLAG",
            "message": "'observation.images.camera_scene' frozen for 2.03 s starting "
                       "at frame 34 (>= 2.0 s threshold; entropy below minimum)",
        }],
    ))
    episodes.append(_sample_episode(
        11, 190, 12.4, 4.1, video_rows=185,
        findings=[
            {"code": "MONOTONICITY_VIOLATION", "severity": "REJECT",
             "message": "timestamp not strictly increasing at frame 87: "
                        "1002.866666667 <= 1002.900000000 (FIX-TS-JITTER class)"},
            {"code": "FRAME_COUNT_MISMATCH", "severity": "REJECT",
             "message": "'observation.images.camera_wrist_left' container frames 185 != parquet rows 190 (FIX-DROP-FRAME class)"},
            {"code": "FRAME_COUNT_MISMATCH", "severity": "REJECT",
             "message": "'observation.images.camera_wrist_right' container frames 185 != parquet rows 190 (FIX-DROP-FRAME class)"},
            {"code": "FRAME_COUNT_MISMATCH", "severity": "REJECT",
             "message": "'observation.images.camera_scene' container frames 185 != parquet rows 190 (FIX-DROP-FRAME class)"},
            {"code": "FRAME_COUNT_MISMATCH", "severity": "REJECT",
             "message": "'observation.images.camera_base_nav' container frames 185 != parquet rows 190 (FIX-DROP-FRAME class)"},
        ],
    ))
    return {
        "qa_report_schema": "1.0",
        "generated_utc": "2026-10-15T18:42:11+00:00",
        "deliverable": "C10 Dataset QA (C-TEAM-PLAN-01 rev2.0 section 3.2)",
        "schema_revision": "schema-v1.0-frozen (19-channel fixed-mast, P-03 Addendum rev18.2)",
        "dataset_path": "/data/openamrobot/episodes",
        "config": {
            "skew_threshold_ms": 30.0,
            "min_frames": 30,
            "require_source_profile": "ROBOT",
            "fps": 30.0,
            "frozen_threshold_sec": 2.0,
            "max_phase_lag": 3,
            "state_dim": 19,
            "video_checks_enabled": True,
        },
        "dataset_checks": {
            "meta_tasks_jsonl": {"present": True, "task_count": 3},
            "meta_episodes_jsonl": {"present": True, "episode_count": 12},
            "ffprobe_available": True,
            "video_checks_enabled": True,
            "frame_decoding_enabled": True,
            "camera_keys": list(CAMERA_KEYS),
            "episodes_discovered": 12,
            "findings": [],
        },
        "summary": {
            "episodes_total": 12,
            "passed": 9,
            "flagged": 2,
            "rejected": 1,
            "finding_code_counts": {
                "MONOTONICITY_VIOLATION": 1,
                "FRAME_COUNT_MISMATCH": 4,
                "PHASE_LAG": 1,
                "FROZEN_CAMERA": 1,
            },
            "training_gate": "BLOCKED",
            "overall_status": "FAIL",
        },
        "episodes": episodes,
    }


# ---------------------------------------------------------------------------
# Report resolution
# ---------------------------------------------------------------------------

def resolve_report(state: ServerState) -> Tuple[Optional[dict], Optional[str]]:
    candidates: List[Path] = []
    if state.report_path is not None:
        candidates.append(Path(state.report_path).expanduser())
    candidates += [
        state.repo_root / REPORT_SAVE_NAME,
        state.repo_root / "reports" / REPORT_SAVE_NAME,
        Path.cwd() / REPORT_SAVE_NAME,
    ]
    seen = set()
    warning = None
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "summary" in data:
                return {"source": f"disk:{p}", "path": str(p), "report": data}, None
            warning = f"{p}: missing 'summary' block"
        except Exception as exc:
            warning = f"{p}: {type(exc).__name__}: {exc}"
    return None, warning


def validate_report_shape(data: object) -> Optional[str]:
    if not isinstance(data, dict):
        return "report root must be a JSON object"
    if "summary" not in data:
        return "report missing 'summary' block (is this a check_dataset_integrity.py report?)"
    return None


# ---------------------------------------------------------------------------
# Git metadata (best effort)
# ---------------------------------------------------------------------------

def detect_git_ref(repo_root: Path) -> Optional[Dict[str, str]]:
    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=2,
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=2,
        )
        if tag.returncode == 0 and tag.stdout.strip():
            return {
                "tag": tag.stdout.strip(),
                "branch": branch.stdout.strip() if branch.returncode == 0 else None,
            }
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Pytest execution + parsing
# ---------------------------------------------------------------------------

RESULT_LINE_RE = re.compile(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)")
SUMMARY_LINE_RE = re.compile(r"^=+ .* =+$")


def parse_pytest_output(output: str) -> Tuple[Dict[str, str], List[dict], Optional[str]]:
    """Map pytest -v output onto fixture statuses."""
    node_status: List[Tuple[str, str]] = []
    summary_line = None
    for line in output.splitlines():
        m = RESULT_LINE_RE.match(line.strip())
        if m:
            node_status.append((m.group(1), m.group(2)))
            continue
        if SUMMARY_LINE_RE.match(line.strip()):
            summary_line = line.strip()
    test_to_nodes: Dict[str, List[str]] = {}
    for node, status in node_status:
        base = node.split("::", 1)[1].split("[", 1)[0]
        test_to_nodes.setdefault(base, []).append(status)
    fixtures: Dict[str, str] = {}
    for fx in FIXTURE_DEFINITIONS:
        statuses = test_to_nodes.get(fx["test"], [])
        if not statuses:
            fixtures[fx["id"]] = "NOT_RUN"
        elif any(s in ("FAILED", "ERROR") for s in statuses):
            fixtures[fx["id"]] = "FAIL"
        elif all(s == "SKIPPED" for s in statuses):
            fixtures[fx["id"]] = "SKIPPED"
        else:
            fixtures[fx["id"]] = "PASS"
    fixture_tests = {fx["test"] for fx in FIXTURE_DEFINITIONS}
    auxiliary = [
        {"name": node.split("::", 1)[1], "status": status}
        for node, status in node_status
        if node.split("::", 1)[1].split("[", 1)[0] not in fixture_tests
    ]
    return fixtures, auxiliary, summary_line


def build_test_payload(cached: bool, **kwargs) -> dict:
    payload = {
        "cached": cached,
        "finished_utc": None,
        "duration_sec": None,
        "exit_code": None,
        "summary": None,
        "output": "",
        "error": None,
        "definitions": FIXTURE_DEFINITIONS,
        "statuses": {fx["id"]: "NOT_RUN" for fx in FIXTURE_DEFINITIONS},
        "auxiliary": [],
    }
    payload.update(kwargs)
    return payload


def load_cached_tests(state: ServerState) -> Optional[dict]:
    if not state.cache_path.is_file():
        return None
    try:
        data = json.loads(state.cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "statuses" in data:
            return data
    except Exception:
        pass
    return None


def run_pytest_suite(state: ServerState) -> dict:
    test_file = state.repo_root / TESTS_REL_PATH
    if not test_file.is_file():
        return build_test_payload(
            False,
            error=f"test file not found: {test_file}",
        )
    cmd = [
        sys.executable, "-m", "pytest", str(test_file),
        "-v", "--tb=short", "-p", "no:cacheprovider",
    ]
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=str(state.repo_root),
            capture_output=True, text=True, timeout=PYTEST_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return build_test_payload(
            False, error=f"pytest timed out after {PYTEST_TIMEOUT_SEC} s",
            duration_sec=round(time.monotonic() - start, 2),
        )
    except FileNotFoundError as exc:
        return build_test_payload(False, error=f"could not launch pytest: {exc}")
    duration = round(time.monotonic() - start, 2)
    output = (proc.stdout or "")
    if proc.stderr:
        output += "\n" + proc.stderr
    error = None
    if "No module named pytest" in output:
        error = "pytest is not installed in this interpreter (pip install pytest)"
    fixtures, auxiliary, summary_line = parse_pytest_output(output)
    payload = build_test_payload(
        False,
        finished_utc=utc_now_iso(),
        duration_sec=duration,
        exit_code=proc.returncode,
        summary=summary_line,
        output=output,
        error=error,
        statuses=fixtures,
        auxiliary=auxiliary,
    )
    try:
        cache = dict(payload)
        cache.pop("definitions", None)
        state.cache_path.write_text(
            json.dumps(cache, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        pass
    return payload


# ---------------------------------------------------------------------------
# HTML/CSS/JS single-page frontend (embedded, no CDN required)
# ---------------------------------------------------------------------------

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenAMRobot 2.0 · Workstream C Verification Portal</title>
<style>
  :root {
    --bg: #0b0f14;
    --panel: #121821;
    --panel-2: #0f141c;
    --border: #1f2937;
    --border-soft: #182130;
    --text: #d7dee8;
    --muted: #8b98a9;
    --dim: #5c6878;
    --accent: #22d3ee;
    --green: #34d399;
    --amber: #fbbf24;
    --red: #f87171;
    --violet: #a78bfa;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { color-scheme: dark; }
  body {
    background: radial-gradient(1200px 600px at 80% -10%, #10202e 0%, var(--bg) 55%);
    color: var(--text);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    min-height: 100vh;
  }
  .wrap { max-width: 1240px; margin: 0 auto; padding: 0 24px; }
  a { color: var(--accent); text-decoration: none; }

  header {
    position: sticky; top: 0; z-index: 50;
    background: rgba(11, 15, 20, 0.82);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border-soft);
  }
  .header-inner { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 24px; flex-wrap: wrap; }
  .brand { display: flex; align-items: center; gap: 14px; }
  .logo {
    width: 44px; height: 44px; border-radius: 12px; flex: none;
    background: linear-gradient(135deg, #164e63, #0e7490);
    border: 1px solid #155e75;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 15px; letter-spacing: 0.5px; color: #cffafe;
  }
  h1 { font-size: 17px; font-weight: 650; }
  h1 .dim { color: var(--muted); font-weight: 450; }
  .brand .sub { font-size: 12.5px; color: var(--muted); }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .chip {
    font: 12px var(--mono);
    padding: 5px 11px; border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.02);
    color: var(--muted);
    white-space: nowrap;
  }
  .chip-accent { color: var(--accent); border-color: #155e75; background: rgba(34,211,238,0.06); }
  .chip-live { display: inline-flex; align-items: center; gap: 7px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dim); display: inline-block; }
  .dot.ok { background: var(--green); box-shadow: 0 0 8px rgba(52,211,153,0.7); animation: pulse 2.4s infinite; }
  .dot.bad { background: var(--red); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.45; } }

  main { padding: 28px 0 60px; }
  section { margin-bottom: 44px; }
  .section-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
  .overline { font-size: 11px; letter-spacing: 1.6px; text-transform: uppercase; color: var(--accent); font-weight: 650; margin-bottom: 3px; }
  h2 { font-size: 19px; font-weight: 650; }
  .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }

  .btn {
    font: 500 13.5px inherit; color: var(--text);
    background: #1a2332; border: 1px solid var(--border);
    padding: 8px 14px; border-radius: 9px; cursor: pointer;
    transition: background .15s, border-color .15s, transform .05s;
  }
  .btn:hover { background: #223048; border-color: #2c3b52; }
  .btn:active { transform: translateY(1px); }
  .btn:disabled { opacity: 0.55; cursor: wait; }
  .btn-primary { background: #0e7490; border-color: #155e75; color: #ecfeff; }
  .btn-primary:hover { background: #0d91b4; }
  .input {
    font: 13px var(--mono); color: var(--text);
    background: var(--panel-2); border: 1px solid var(--border);
    padding: 8px 12px; border-radius: 9px; min-width: 300px;
  }
  .input:focus { outline: none; border-color: var(--accent); }
  .src-chip { font: 12px var(--mono); color: var(--muted); max-width: 480px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .dim { color: var(--muted); }
  .mono { font-family: var(--mono); font-size: 12.5px; }
  .config-line { font: 12px var(--mono); color: var(--dim); margin-top: 10px; }

  .grid { display: grid; gap: 14px; }
  .cards { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
  .card {
    background: linear-gradient(180deg, var(--panel), var(--panel-2));
    border: 1px solid var(--border);
    border-radius: 13px; padding: 16px 18px;
  }
  .card-label { font-size: 11px; letter-spacing: 1.2px; text-transform: uppercase; color: var(--muted); font-weight: 650; margin-bottom: 8px; }
  .card-value { font-size: 26px; font-weight: 700; letter-spacing: -0.5px; }
  .card-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .v-green { color: var(--green); } .v-amber { color: var(--amber); }
  .v-red { color: var(--red); } .v-accent { color: var(--accent); }

  .gate { display: flex; flex-direction: column; justify-content: center; align-items: flex-start; }
  .gate-badge {
    font: 700 15px var(--mono); letter-spacing: 1.5px;
    padding: 7px 16px; border-radius: 10px; display: inline-block;
  }
  .gate-open .gate-badge { color: #022c22; background: var(--green); box-shadow: 0 0 22px rgba(52,211,153,0.35); }
  .gate-blocked .gate-badge { color: #2b0707; background: var(--red); box-shadow: 0 0 22px rgba(248,113,113,0.35); }
  .gate-unknown .gate-badge { color: var(--muted); background: #1f2937; }

  .metrics-grid { display: grid; grid-template-columns: 1fr 1.4fr; gap: 14px; margin-top: 14px; }
  @media (max-width: 900px) { .metrics-grid { grid-template-columns: 1fr; } }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    text-align: left; font-size: 11px; letter-spacing: 1px; text-transform: uppercase;
    color: var(--muted); font-weight: 650; padding: 8px 10px; border-bottom: 1px solid var(--border);
  }
  td { padding: 8px 10px; border-bottom: 1px solid var(--border-soft); vertical-align: middle; }
  tr:last-child td { border-bottom: none; }

  .badge {
    font: 650 10.5px var(--mono); letter-spacing: 0.8px;
    padding: 3px 9px; border-radius: 999px; display: inline-block; white-space: nowrap;
  }
  .b-pass { color: var(--green); background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.35); }
  .b-flag { color: var(--amber); background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.35); }
  .b-reject { color: var(--red); background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.4); }
  .b-warn { color: var(--violet); background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.35); }
  .b-muted { color: var(--muted); background: rgba(139,152,169,0.08); border: 1px solid rgba(139,152,169,0.3); }
  .b-info { color: var(--accent); background: rgba(34,211,238,0.07); border: 1px solid rgba(34,211,238,0.3); }

  details.ep { background: var(--panel); border: 1px solid var(--border); border-radius: 11px; margin-bottom: 8px; overflow: hidden; }
  details.ep summary {
    display: flex; gap: 14px; align-items: center; padding: 11px 16px; cursor: pointer; list-style: none;
    font-size: 13px;
  }
  details.ep summary::-webkit-details-marker { display: none; }
  details.ep summary:hover { background: rgba(255,255,255,0.02); }
  .ep-idx { font: 650 13px var(--mono); color: var(--accent); min-width: 96px; }
  .ep-body { padding: 4px 18px 14px 18px; border-top: 1px solid var(--border-soft); }
  .finding { display: flex; gap: 10px; align-items: baseline; padding: 6px 0; font-size: 12.5px; border-bottom: 1px dashed var(--border-soft); }
  .finding:last-child { border-bottom: none; }
  .finding .msg { color: var(--muted); font-family: var(--mono); font-size: 12px; }
  .kv { display: flex; gap: 18px; flex-wrap: wrap; padding: 8px 0 4px; font-size: 12px; color: var(--muted); }
  .kv b { color: var(--text); font-weight: 600; }

  .fixtures { grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); }
  .fx { background: linear-gradient(180deg, var(--panel), var(--panel-2)); border: 1px solid var(--border); border-radius: 13px; padding: 15px 16px; display: flex; flex-direction: column; gap: 8px; }
  .fx-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  .fx-id { font: 700 13px var(--mono); color: var(--text); letter-spacing: 0.4px; }
  .fx-fault { font-size: 12.5px; color: var(--muted); min-height: 36px; }
  .fx-bottom { display: flex; justify-content: space-between; align-items: center; margin-top: 2px; }
  .fx-expected { font: 11px var(--mono); color: var(--dim); }
  .st-pass { color: var(--green); background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.4); }
  .st-fail { color: var(--red); background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.45); }
  .st-skip { color: var(--violet); background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.35); }
  .st-idle { color: var(--dim); background: rgba(92,104,120,0.08); border: 1px solid rgba(92,104,120,0.3); }
  .suite-summary { font: 12.5px var(--mono); color: var(--muted); margin-top: 12px; }
  pre {
    background: #0a0e13; border: 1px solid var(--border); border-radius: 11px;
    padding: 14px 16px; font: 12px/1.55 var(--mono); color: #a8b3c2;
    overflow-x: auto; margin-top: 12px; white-space: pre-wrap; word-break: break-word;
  }
  details.raw summary { cursor: pointer; color: var(--muted); font-size: 12.5px; margin-top: 12px; }

  .cards3 { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
  .ds { border-style: dashed; position: relative; }
  .ds h3 { font-size: 14.5px; font-weight: 650; margin-bottom: 4px; }
  .ds .due { font: 11px var(--mono); color: var(--dim); }
  .ds ul { margin: 10px 0 10px 18px; color: var(--muted); font-size: 12.5px; }
  .ds li { margin-bottom: 3px; }
  .ds .artifact { font: 11.5px var(--mono); color: var(--dim); border-top: 1px dashed var(--border); padding-top: 9px; margin-top: 6px; }

  footer { border-top: 1px solid var(--border-soft); padding: 18px 24px 34px; color: var(--dim); font-size: 12px; }
  footer .wrap { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
</style>
</head>
<body>
<header>
  <div class="header-inner wrap">
    <div class="brand">
      <div class="logo">OR</div>
      <div>
        <h1>OpenAMRobot 2.0 <span class="dim">· Workstream C Verification Portal</span></h1>
        <div class="sub">Embodied AI &amp; Imitation Learning — local pipeline inspection (pre-Step 2)</div>
      </div>
    </div>
    <div class="chips">
      <span class="chip chip-accent" id="chip-ref">schema-v1.0-frozen</span>
      <span class="chip" id="chip-system">19-Channel Whole-Body</span>
      <span class="chip" id="chip-target">openamrobot_2 · LeRobotDataset v3.0</span>
      <span class="chip chip-live"><span class="dot" id="server-dot"></span><span id="server-state">connecting…</span></span>
    </div>
  </div>
</header>

<main class="wrap">

  <!-- ======================= SECTION 1: QA REPORT ======================= -->
  <section id="qa-section">
    <div class="section-head">
      <div>
        <div class="overline">Section 1 · Deliverable C10</div>
        <h2>Pre-Flight QA Report Viewer</h2>
      </div>
      <div class="controls">
        <input type="file" id="qa-file" accept=".json,application/json" hidden>
        <button class="btn" id="qa-upload-btn">Upload qa_report.json</button>
        <input type="text" class="input" id="qa-path" placeholder="…or absolute path to qa_report.json">
        <button class="btn" id="qa-load-btn">Load Path</button>
        <button class="btn" id="qa-refresh-btn">Refresh</button>
      </div>
    </div>
    <div class="src-chip" id="qa-source"></div>
    <div class="config-line" id="qa-config"></div>

    <div class="grid cards" style="margin-top:14px" id="qa-cards"></div>

    <div class="metrics-grid">
      <div class="card">
        <div class="card-label">Inter-Stream Skew — |timestamp − timestamp_device|</div>
        <div id="skew-body"></div>
      </div>
      <div class="card">
        <div class="card-label">Camera Freshness (frame_count ÷ duration ≥ fps × 0.95)</div>
        <table id="freshness-table">
          <thead><tr><th>Camera stream</th><th>Episodes</th><th>Avg fps</th><th>Min fps</th><th>Status</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <div class="metrics-grid">
      <div class="card">
        <div class="card-label">Diagnostic Code Breakdown</div>
        <table id="codes-table">
          <thead><tr><th>Code</th><th>Severity</th><th>Count</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="card">
        <div class="card-label">Episodes (click to expand findings)</div>
        <div id="episode-list"></div>
      </div>
    </div>
  </section>

  <!-- ======================= SECTION 2: TEST SUITE ======================= -->
  <section id="tests-section">
    <div class="section-head">
      <div>
        <div class="overline">Section 2 · Deliverable C10 — Step 2 Verification</div>
        <h2>Fixture &amp; Test Suite Status</h2>
      </div>
      <div class="controls">
        <span class="src-chip" id="tests-meta"></span>
        <button class="btn btn-primary" id="run-tests">Run Test Suite</button>
      </div>
    </div>
    <div class="grid fixtures" id="fixture-grid"></div>
    <div class="suite-summary" id="suite-summary"></div>
    <details class="raw" id="pytest-output-wrap" hidden>
      <summary>pytest raw output</summary>
      <pre id="pytest-output"></pre>
    </details>
  </section>

  <!-- ======================= SECTION 3: DOWNSTREAM ======================= -->
  <section id="downstream-section">
    <div class="section-head">
      <div>
        <div class="overline">Section 3 · Downstream Milestones</div>
        <h2>Replay · Training · Promotion</h2>
      </div>
    </div>
    <div class="grid cards3" id="downstream-grid"></div>
  </section>

</main>

<footer>
  <div class="wrap">
    <span>OpenAMRobot 2.0 · C-TEAM-PLAN-01 rev2.0 · schema-v1.0-frozen (19-channel, P-03 Addendum rev18.2)</span>
    <span>Local verification tool — bound to 127.0.0.1 · HC-13: readiness is evidence, not optimism</span>
  </div>
</footer>

<script>
"use strict";

const CAMERAS = [
  "observation.images.camera_wrist_left",
  "observation.images.camera_wrist_right",
  "observation.images.camera_scene",
  "observation.images.camera_base_nav",
];
const CAM_SHORT = {
  "observation.images.camera_wrist_left": "camera_wrist_left",
  "observation.images.camera_wrist_right": "camera_wrist_right",
  "observation.images.camera_scene": "camera_scene",
  "observation.images.camera_base_nav": "camera_base_nav",
};
const SEVERITY_BY_CODE = {
  MONOTONICITY_VIOLATION: "REJECT", DEVICE_MONOTONICITY_VIOLATION: "REJECT",
  SKEW_THRESHOLD_EXCEEDED: "REJECT", FRAME_COUNT_MISMATCH: "REJECT",
  CAMERA_FPS_LOW: "REJECT", VIDEO_STREAM_MISSING: "REJECT",
  SCHEMA_VIOLATION: "REJECT", NON_FINITE_VALUE: "REJECT",
  TIMEBASE_INVALID: "REJECT", READ_FAILURE: "REJECT",
  EPISODE_TOO_SHORT: "REJECT", ORPHAN_TASK: "REJECT",
  MISSING_EPISODE_METADATA: "REJECT", INVALID_SOURCE_PROFILE: "REJECT",
  SOURCE_PROFILE_MISMATCH: "REJECT", MISSING_META_FILE: "REJECT",
  NO_EPISODES_FOUND: "REJECT",
  FROZEN_CAMERA: "FLAG", PHASE_LAG: "FLAG",
  SUBTASK_OUT_OF_RANGE: "WARN",
};
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[c]));
const f1 = (x) => (x == null || isNaN(x)) ? "—" : Number(x).toFixed(1);
const f2 = (x) => (x == null || isNaN(x)) ? "—" : Number(x).toFixed(2);

function setServerState(ok, label) {
  const dot = $("#server-dot");
  dot.className = "dot " + (ok ? "ok" : "bad");
  $("#server-state").textContent = label;
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " -> HTTP " + r.status);
  return r.json();
}

/* ---------------------------- Header ---------------------------- */
async function loadHealth() {
  try {
    const h = await getJSON("/api/health");
    const ref = h.git && h.git.tag ? h.git.tag : h.schema_tag;
    $("#chip-ref").textContent = ref + (h.git && h.git.branch ? " · " + h.git.branch : "");
    $("#chip-system").textContent = h.system;
    $("#chip-target").textContent = h.robot_type + " · " + h.dataset_format;
    setServerState(true, "online · " + h.server_time.slice(11, 19) + "Z");
  } catch (e) {
    setServerState(false, "offline");
  }
}

/* ---------------------------- QA report ---------------------------- */
function badgeForStatus(status) {
  const map = { PASS: "b-pass", FLAG: "b-flag", REJECT: "b-reject", WARN: "b-warn" };
  return '<span class="badge ' + (map[status] || "b-muted") + '">' + esc(status) + "</span>";
}
function badgeForSeverity(sev) {
  const map = { REJECT: "b-reject", FLAG: "b-flag", WARN: "b-warn" };
  return '<span class="badge ' + (map[sev] || "b-muted") + '">' + esc(sev) + "</span>";
}

function renderQA(data) {
  const src = $("#qa-source");
  if (data.source === "sample") {
    src.innerHTML = '<span class="badge b-flag">SAMPLE DATA</span> <span class="dim">' +
      esc(data.warning || "no qa_report.json found on disk — showing built-in sample") + "</span>";
  } else {
    src.innerHTML = '<span class="badge b-info">DISK</span> <span class="mono">' + esc(data.path) + "</span>";
    if (data.warning) src.innerHTML += ' <span class="dim">· ' + esc(data.warning) + "</span>";
  }

  const r = data.report || {};
  const s = r.summary || {};
  const cfg = r.config || {};
  $("#qa-config").textContent = "config: skew ≤ " + f1(cfg.skew_threshold_ms) + " ms · ≥ " +
    (cfg.min_frames ?? "—") + " frames · profile " + (cfg.require_source_profile ?? "—") +
    " · " + f1(cfg.fps) + " fps · state_dim " + (cfg.state_dim ?? "—") +
    " · generated " + esc(r.generated_utc || "—");

  const gate = s.training_gate || "UNKNOWN";
  const gateClass = gate === "OPEN" ? "gate-open" : (gate === "BLOCKED" ? "gate-blocked" : "gate-unknown");
  $("#qa-cards").innerHTML =
    '<div class="card gate ' + gateClass + '">' +
      '<div class="card-label">Training Gate</div>' +
      '<span class="gate-badge">' + esc(gate) + "</span>" +
      '<div class="card-sub">overall ' + esc(s.overall_status || "—") + "</div></div>" +
    card("Total Episodes", s.episodes_total, "v-accent", "discovered in dataset") +
    card("Passed", s.passed, "v-green", "no findings") +
    card("Flagged", s.flagged, "v-amber", "triage required, non-blocking") +
    card("Rejected", s.rejected, "v-red", "blocks training (HC gate)");

  renderSkew(r);
  renderFreshness(r.episodes || []);
  renderCodes(s.finding_code_counts || {});
  renderEpisodes(r.episodes || [], cfg.skew_threshold_ms);
}
function card(label, value, cls, sub) {
  return '<div class="card"><div class="card-label">' + esc(label) + '</div>' +
    '<div class="card-value ' + cls + '">' + esc(value ?? "—") + "</div>" +
    '<div class="card-sub">' + esc(sub) + "</div></div>";
}

function renderSkew(r) {
  const eps = r.episodes || [];
  let maxSkew = null, meanSum = 0, meanN = 0, p95 = null, worst = null;
  for (const ep of eps) {
    const sk = ep.stats && ep.stats.skew_ms;
    if (!sk) continue;
    if (sk.max != null && (maxSkew == null || sk.max > maxSkew)) { maxSkew = sk.max; worst = ep.episode_index; }
    if (sk.mean != null) { meanSum += sk.mean; meanN++; }
    if (sk.p95 != null && (p95 == null || sk.p95 > p95)) p95 = sk.p95;
  }
  const thr = (r.config || {}).skew_threshold_ms;
  const ok = maxSkew != null && thr != null && maxSkew <= thr;
  $("#skew-body").innerHTML =
    '<div class="grid cards" style="grid-template-columns:repeat(3,1fr);margin-top:6px">' +
      '<div><div class="dim" style="font-size:11px">MAX</div><div class="card-value ' + (ok ? "v-green" : "v-red") + '" style="font-size:22px">' + f2(maxSkew) + ' <span style="font-size:12px">ms</span></div></div>' +
      '<div><div class="dim" style="font-size:11px">MEAN (of episodes)</div><div class="card-value" style="font-size:22px">' + f2(meanN ? meanSum / meanN : null) + ' <span style="font-size:12px">ms</span></div></div>' +
      '<div><div class="dim" style="font-size:11px">P95 (worst)</div><div class="card-value" style="font-size:22px">' + f2(p95) + ' <span style="font-size:12px">ms</span></div></div>' +
    "</div>" +
    '<div class="card-sub" style="margin-top:10px">threshold ' + f1(thr) + " ms" +
      (worst != null ? " · worst episode: <span class='mono'>" + String(worst).padStart(6, "0") + "</span>" : "") +
      "</div>";
}

function renderFreshness(episodes) {
  const tbody = $("#freshness-table tbody");
  tbody.innerHTML = "";
  for (const cam of CAMERAS) {
    let n = 0, sum = 0, min = null, allOk = true, any = false;
    for (const ep of episodes) {
      const fr = ep.stats && ep.stats.camera_freshness && ep.stats.camera_freshness[cam];
      if (!fr) continue;
      any = true; n++;
      if (fr.measured_fps != null) { sum += fr.measured_fps; if (min == null || fr.measured_fps < min) min = fr.measured_fps; }
      if (fr.ok === false) allOk = false;
    }
    const status = !any ? '<span class="badge b-muted">NO DATA</span>'
      : allOk ? '<span class="badge b-pass">OK</span>'
      : '<span class="badge b-reject">LOW FPS</span>';
    tbody.insertAdjacentHTML("beforeend",
      "<tr><td class='mono'>" + esc(CAM_SHORT[cam]) + "</td><td>" + n + "</td><td>" +
      f2(n ? sum / n : null) + "</td><td>" + f2(min) + "</td><td>" + status + "</td></tr>");
  }
}

function renderCodes(counts) {
  const tbody = $("#codes-table tbody");
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    tbody.innerHTML = "<tr><td colspan='3' class='dim'>no findings recorded</td></tr>";
    return;
  }
  tbody.innerHTML = entries.map(([code, n]) =>
    "<tr><td class='mono'>" + esc(code) + "</td><td>" +
    badgeForSeverity(SEVERITY_BY_CODE[code] || "WARN") + "</td><td>" + esc(n) + "</td></tr>"
  ).join("");
}

function renderEpisodes(episodes, skewThreshold) {
  const box = $("#episode-list");
  if (!episodes.length) { box.innerHTML = "<div class='dim'>no episodes in report</div>"; return; }
  box.innerHTML = "";
  const sorted = [...episodes].sort((a, b) => (a.episode_index ?? 0) - (b.episode_index ?? 0));
  for (const ep of sorted) {
    const st = ep.stats || {};
    const sk = st.skew_ms || {};
    const skOk = sk.max == null || skewThreshold == null || sk.max <= skewThreshold;
    const codes = (ep.findings || []).map((f) =>
      '<span class="badge ' + (f.severity === "REJECT" ? "b-reject" : f.severity === "FLAG" ? "b-flag" : "b-warn") + '">' +
      esc(f.code) + "</span>").join(" ");
    const findings = (ep.findings || []).map((f) =>
      '<div class="finding">' + badgeForSeverity(f.severity) +
      ' <span class="mono">' + esc(f.code) + '</span> <span class="msg">' + esc(f.message) + "</span></div>"
    ).join("") || "<div class='dim' style='padding:6px 0'>no findings</div>";
    box.insertAdjacentHTML("beforeend",
      "<details class='ep'><summary>" +
        '<span class="ep-idx">ep ' + String(ep.episode_index ?? "?").padStart(6, "0") + "</span>" +
        badgeForStatus(ep.status) +
        '<span class="dim mono">' + (st.parquet_rows ?? "—") + " frames · " + f2(st.duration_sec) + " s</span>" +
        '<span class="dim mono" style="color:' + (skOk ? "var(--muted)" : "var(--red)") + '">skew ' + f2(sk.max) + " ms</span>" +
        '<span style="display:flex;gap:5px;flex-wrap:wrap">' + codes + "</span>" +
      "</summary><div class='ep-body'>" +
        "<div class='kv'>" +
          "<span>phase lag: <b>" + (st.phase_lag_frames ?? "—") + " frames</b></span>" +
          "<span>skew mean: <b>" + f2(sk.mean) + " ms</b></span>" +
          "<span>skew p95: <b>" + f2(sk.p95) + " ms</b></span>" +
          "<span class='mono dim'>" + esc(st.parquet_path || "") + "</span>" +
        "</div>" + findings +
      "</div></details>");
  }
}

async function loadQA() {
  try {
    renderQA(await getJSON("/api/qa-report"));
  } catch (e) {
    $("#qa-source").textContent = "failed to load QA report: " + e.message;
  }
}

/* ---------------------------- Test suite ---------------------------- */
const FX_STATUS_CLASS = { PASS: "st-pass", FAIL: "st-fail", SKIPPED: "st-skip", NOT_RUN: "st-idle", RUNNING: "st-idle" };

function renderTests(data) {
  const meta = $("#tests-meta");
  if (data.error) meta.innerHTML = '<span class="badge b-reject">ERROR</span> <span class="dim">' + esc(data.error) + "</span>";
  else if (data.finished_utc) meta.innerHTML = (data.cached ? '<span class="badge b-muted">CACHED</span> ' : '<span class="badge b-info">LIVE</span> ') +
      '<span class="dim mono">finished ' + esc(data.finished_utc) + " · " + f1(data.duration_sec) + " s · exit " + esc(data.exit_code) + "</span>";
  else meta.textContent = "no test run yet — click Run Test Suite";

  const grid = $("#fixture-grid");
  grid.innerHTML = "";
  for (const fx of data.definitions || []) {
    const status = (data.statuses || {})[fx.id] || "NOT_RUN";
    grid.insertAdjacentHTML("beforeend",
      "<div class='fx'>" +
        '<div class="fx-top"><span class="fx-id">' + esc(fx.id) + '</span>' +
          '<span class="badge ' + FX_STATUS_CLASS[status] + '">' + esc(status.replace("_", " ")) + "</span></div>" +
        '<div class="fx-fault">' + esc(fx.fault) + "</div>" +
        '<div class="fx-bottom"><span class="fx-expected">expected: ' + esc(fx.expected) + " — " + esc(fx.reason) + "</span></div>" +
      "</div>");
  }

  const counts = { PASS: 0, FAIL: 0, SKIPPED: 0, NOT_RUN: 0 };
  for (const v of Object.values(data.statuses || {})) counts[v] = (counts[v] || 0) + 1;
  $("#suite-summary").textContent = data.summary ||
    ("fixtures: " + counts.PASS + " pass · " + counts.FAIL + " fail · " + counts.SKIPPED + " skipped · " + counts.NOT_RUN + " not run");

  const wrap = $("#pytest-output-wrap");
  if (data.output) { wrap.hidden = false; $("#pytest-output").textContent = data.output; }
  else wrap.hidden = true;
}

async function loadTests() {
  try { renderTests(await getJSON("/api/tests")); }
  catch (e) { $("#tests-meta").textContent = "failed to load test status: " + e.message; }
}

async function runTests() {
  const btn = $("#run-tests");
  btn.disabled = true; btn.textContent = "Running suite…";
  $("#tests-meta").textContent = "executing pytest — this can take a minute…";
  try {
    const r = await fetch("/api/tests/run", { method: "POST" });
    const data = await r.json();
    if (!r.ok && !data.definitions) throw new Error(data.error || ("HTTP " + r.status));
    renderTests(data);
  } catch (e) {
    $("#tests-meta").innerHTML = '<span class="badge b-reject">ERROR</span> <span class="dim">' + esc(e.message) + "</span>";
  } finally {
    btn.disabled = false; btn.textContent = "Run Test Suite";
  }
}

/* ---------------------------- Downstream ---------------------------- */
async function loadDownstream() {
  try {
    const d = await getJSON("/api/downstream");
    const grid = $("#downstream-grid");
    grid.innerHTML = "";
    for (const sec of d.sections) {
      grid.insertAdjacentHTML("beforeend",
        "<div class='card ds'>" +
          '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">' +
            '<span class="badge b-muted">' + esc(sec.status) + '</span><span class="due">due ' + esc(sec.due) + "</span></div>" +
          "<h3>" + esc(sec.title) + "</h3>" +
          "<ul>" + sec.metrics.map((m) => "<li>" + esc(m) + "</li>").join("") + "</ul>" +
          '<div class="artifact">artifact: ' + esc(sec.artifact) + "</div>" +
        "</div>");
    }
  } catch (e) { /* non-critical */ }
}

/* ---------------------------- Wiring ---------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  loadHealth();
  loadQA();
  loadTests();
  loadDownstream();

  $("#qa-refresh-btn").addEventListener("click", loadQA);
  $("#run-tests").addEventListener("click", runTests);

  $("#qa-upload-btn").addEventListener("click", () => $("#qa-file").click());
  $("#qa-file").addEventListener("change", async (e) => {
    const f = e.target.files && e.target.files[0];
    e.target.value = "";
    if (!f) return;
    let parsed;
    try { parsed = JSON.parse(await f.text()); }
    catch { alert("Selected file is not valid JSON."); return; }
    try {
      const r = await fetch("/api/qa-report/upload", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed),
      });
      const data = await r.json();
      if (!r.ok) { alert(data.error || "upload rejected"); return; }
      renderQA(data.payload);
    } catch (err) { alert("Upload failed: " + err.message); }
  });

  $("#qa-load-btn").addEventListener("click", async () => {
    const p = $("#qa-path").value.trim();
    if (!p) return;
    try {
      const r = await fetch("/api/qa-report/load", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: p }),
      });
      const data = await r.json();
      if (!r.ok) { alert(data.error || "load failed"); return; }
      renderQA(data.payload);
    } catch (err) { alert("Load failed: " + err.message); }
  });
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "OpenAMRobotCPortal/1.0"

    # ---- plumbing ----
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: object) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json_body(self) -> Tuple[Optional[object], Optional[str]]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None, "invalid Content-Length"
        if length <= 0:
            return None, "empty request body"
        if length > MAX_BODY_BYTES:
            return None, f"body too large (>{MAX_BODY_BYTES} bytes)"
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")), None
        except Exception as exc:
            return None, f"invalid JSON body: {exc}"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (datetime.now().strftime("%H:%M:%S"), fmt % args))

    # ---- GET routes ----
    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/health":
                self._json(200, {
                    "schema_tag": SCHEMA_TAG,
                    "system": SYSTEM_LABEL,
                    "target": TARGET_LABEL,
                    "robot_type": ROBOT_TYPE,
                    "dataset_format": DATASET_FORMAT,
                    "repo_root": str(STATE.repo_root),
                    "git": detect_git_ref(STATE.repo_root),
                    "python": sys.version.split()[0],
                    "server_time": utc_now_iso(),
                })
            elif path == "/api/qa-report":
                payload, warning = resolve_report(STATE)
                if payload is None:
                    payload = {
                        "source": "sample",
                        "path": None,
                        "report": STATE.sample_report,
                        "warning": warning or "no qa_report.json found — showing built-in sample",
                    }
                elif warning:
                    payload["warning"] = warning
                self._json(200, payload)
            elif path == "/api/tests":
                cached = load_cached_tests(STATE)
                if cached is not None:
                    payload = build_test_payload(True)
                    payload.update({k: v for k, v in cached.items() if k in payload})
                    self._json(200, payload)
                else:
                    self._json(200, build_test_payload(False))
            elif path == "/api/downstream":
                self._json(200, {"sections": DOWNSTREAM_SECTIONS})
            else:
                self._json(404, {"error": f"unknown route: {path}"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    # ---- POST routes ----
    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/tests/run":
                with STATE.test_run_lock:
                    if STATE.test_running:
                        self._json(409, {"error": "test run already in progress"})
                        return
                    STATE.test_running = True
                try:
                    self._json(200, run_pytest_suite(STATE))
                finally:
                    STATE.test_running = False
            elif path == "/api/qa-report/upload":
                body, err = self._read_json_body()
                if err:
                    self._json(400, {"error": err})
                    return
                shape_err = validate_report_shape(body)
                if shape_err:
                    self._json(400, {"error": shape_err})
                    return
                save_to = STATE.repo_root / REPORT_SAVE_NAME
                save_to.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
                self._json(200, {
                    "saved_to": str(save_to),
                    "payload": {"source": f"disk:{save_to}", "path": str(save_to), "report": body},
                })
            elif path == "/api/qa-report/load":
                body, err = self._read_json_body()
                if err or not isinstance(body, dict) or not body.get("path"):
                    self._json(400, {"error": err or "expected JSON object with 'path'"})
                    return
                p = Path(str(body["path"])).expanduser().resolve()
                if not p.is_file():
                    self._json(400, {"error": f"file not found: {p}"})
                    return
                if p.suffix.lower() != ".json":
                    self._json(400, {"error": "path must point to a .json file"})
                    return
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception as exc:
                    self._json(400, {"error": f"unreadable JSON: {exc}"})
                    return
                shape_err = validate_report_shape(data)
                if shape_err:
                    self._json(400, {"error": shape_err})
                    return
                self._json(200, {"payload": {"source": f"disk:{p}", "path": str(p), "report": data}})
            else:
                self._json(404, {"error": f"unknown route: {path}"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    global STATE
    parser = argparse.ArgumentParser(
        prog="run_dashboard.py",
        description="OpenAMRobot 2.0 Workstream C — local verification portal (stdlib only)",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default: 127.0.0.1; keep local-only)")
    parser.add_argument("--port", type=int, default=8300, help="port (default: 8300)")
    parser.add_argument("--report", type=Path, default=None,
                        help="explicit path to qa_report.json (otherwise auto-discovered)")
    parser.add_argument("--open", action="store_true", help="open the browser on startup")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    STATE = ServerState(repo_root=repo_root, report_path=args.report)

    httpd = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print("=" * 74)
    print("OpenAMRobot 2.0 · Workstream C Verification Portal")
    print(f"  URL        : {url}")
    print(f"  Repo root  : {repo_root}")
    print(f"  Schema     : {SCHEMA_TAG} ({SYSTEM_LABEL})")
    print(f"  QA report  : {args.report if args.report else 'auto-discover (fallback: built-in sample)'}")
    print(f"  Test suite : {repo_root / TESTS_REL_PATH}")
    print("  Ctrl-C to stop.")
    print("=" * 74)

    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down portal.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
