"""
test_c11_pipeline.py
====================

OpenAMRobot 2.0 | Workstream C | Tests for Deliverable C11 (ACT Policy Pipeline).

Test matrix:
  1. test_dataset_generator_contract_and_qa:
     - Generates synthetic LeRobotDataset v3.0 into tmp_path.
     - Verifies contract validation passes all checks.
     - Verifies C10 pre-flight gate returns training_gate == 'OPEN'.
  2. test_act_policy_architecture_shapes:
     - Validates ACT CVAE encoder, decoder, positional encodings.
     - Verifies forward pass, backward pass, and loss computation shapes.
  3. test_act_training_and_eval_pipeline:
     - Runs train_act_model in smoke-test mode.
     - Verifies checkpoint creation, SHA-256 hash, and training curves.
     - Runs evaluate_checkpoint on the held-out split.
     - Verifies c11_eval_report.json and c11_evidence_summary.md generation.
     - Checks Wilson 95% confidence intervals and subtask attribution.
  4. test_preflight_gate_blocking_prevents_training:
     - Injects a fault into the dataset (NaN / OOB values).
     - Asserts training aborts immediately with pre-flight gate error.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest  # openAMRobot/openamrobot-c-baseline#16

pyarrow = pytest.importorskip("pyarrow")  # openAMRobot/openamrobot-c-baseline#16
torch = pytest.importorskip("torch")  # openAMRobot/openamrobot-c-baseline#16

# ---------------------------------------------------------------------------
# Bootstrap import paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIGS_DIR = REPO_ROOT / "configs"

for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check_dataset_integrity as cdi  # noqa: E402
import validate_dataset_contract as vdc  # noqa: E402
from generate_or_ingest_dataset import generate_synthetic_dataset  # noqa: E402
from train_act_baseline import ACTPolicy, train_act_model, STATE_DIM, ACTION_DIM  # noqa: E402
from evaluate_act_baseline import evaluate_checkpoint  # noqa: E402


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_dataset_generator_contract_and_qa(tmp_path: Path):
    """Verify that generate_synthetic_dataset produces a schema-valid dataset."""
    ds_dir = tmp_path / "dataset"
    generate_synthetic_dataset(
        output_dir=ds_dir,
        num_episodes=2,
        episode_length=35,
        source_profile="GAZEBO",
        write_videos=False,  # keep test fast
    )

    # 1. Verify contract schema
    info_json = ds_dir / "meta" / "info.json"
    assert info_json.is_file(), "meta/info.json must exist"

    info_data = json.loads(info_json.read_text(encoding="utf-8"))
    report = vdc.Report()
    vdc.validate_contract(info_data, report=report, expected_state_dim=STATE_DIM)
    assert report.passed, "Generated info.json must pass contract validation"

    # 2. Verify C10 pre-flight gate
    config = cdi.QAConfig(
        skew_threshold_ms=30.0,
        min_frames=30,
        require_source_profile="GAZEBO",
        fps=30.0,
        video_checks_enabled=False,
    )
    qa_report = cdi.run_qa(dataset_path=ds_dir, config=config, skip_video_checks=True)
    summary = qa_report.get("summary", {})
    assert summary.get("training_gate") == "OPEN"
    assert summary.get("overall_status") == "PASS"
    assert summary.get("passed") == 2


def test_act_policy_architecture_shapes():
    """Verify forward and backward tensor dimensions through ACT policy."""
    batch_size = 2
    chunk_size = 100
    model = ACTPolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        chunk_size=chunk_size,
        d_model=64,
        n_heads=4,
        enc_layers=2,
        dec_layers=2,
        latent_dim=16,
    )
    model.train()

    # Training forward pass
    dummy_state = torch.randn(batch_size, STATE_DIM)
    dummy_actions = torch.randn(batch_size, chunk_size, ACTION_DIM)

    pred, mu, log_var = model(dummy_state, dummy_actions)
    assert pred.shape == (batch_size, chunk_size, ACTION_DIM)
    assert mu.shape == (batch_size, 16)
    assert log_var.shape == (batch_size, 16)

    # Loss computation & backprop
    total_loss, l1_loss, kl_loss = model.compute_loss(pred, dummy_actions, mu, log_var)
    assert total_loss.item() > 0
    assert l1_loss.item() > 0

    total_loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient must exist for {name}"

    # Inference forward pass
    model.eval()
    with torch.no_grad():
        inf_pred, inf_mu, inf_log = model(dummy_state)
        assert inf_pred.shape == (batch_size, chunk_size, ACTION_DIM)
        assert inf_mu is None
        assert inf_log is None


def test_act_training_and_eval_pipeline(tmp_path: Path):
    """End-to-end integration: dataset generation -> ACT training -> held-out evaluation."""
    ds_dir = tmp_path / "episodes"
    reports_dir = tmp_path / "reports"
    checkpoints_dir = tmp_path / "checkpoints"
    chk_file = checkpoints_dir / "test_act.pt"
    eval_json = reports_dir / "c11_eval_report.json"
    eval_md = reports_dir / "c11_evidence_summary.md"

    # Generate 4 episodes
    generate_synthetic_dataset(
        output_dir=ds_dir,
        num_episodes=4,
        episode_length=35,
        source_profile="GAZEBO",
        write_videos=False,
    )

    # Train in smoke-test mode (deterministic 1 held-out episode)
    summary = train_act_model(
        dataset_path=ds_dir,
        config_path=CONFIGS_DIR / "act_openamrobot_19dof.yaml",
        output_checkpoint=chk_file,
        epochs=2,
        batch_size=2,
        smoke_test=True,
        eval_split=0.25,
        source_profile="GAZEBO",
        output_summary=tmp_path / "c11_training_summary.json",
    )
    assert chk_file.is_file(), "Checkpoint must be saved"
    assert len(summary["checkpoint_sha256"]) == 64
    assert summary["episodes"]["total"] == 4
    assert summary["episodes"]["held_out_count"] == 1

    # Run evaluation
    eval_payload = evaluate_checkpoint(
        checkpoint_path=chk_file,
        dataset_path=ds_dir,
        output_json=eval_json,
        output_md=eval_md,
        device="cpu",
    )

    assert eval_json.is_file(), "Evaluation JSON must exist"
    assert eval_md.is_file(), "Evaluation Markdown must exist"

    # Check evidence metrics
    assert "action_prediction_error" in eval_payload
    assert len(eval_payload["action_prediction_error"]["per_channel"]) == STATE_DIM
    assert "subtask_phase_attribution" in eval_payload
    assert len(eval_payload["subtask_phase_attribution"]["phases"]) == 8

    # Wilson 95% Confidence Interval check
    ci = eval_payload["rollout_results"]["wilson_95_ci"]
    assert len(ci) == 2
    assert 0.0 <= ci[0] <= ci[1] <= 1.0


def test_preflight_gate_blocking_prevents_training(tmp_path: Path):
    """Verify that corrupt dataset blocks training immediately."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds_dir = tmp_path / "bad_dataset"
    generate_synthetic_dataset(
        output_dir=ds_dir,
        num_episodes=2,
        episode_length=35,
        source_profile="GAZEBO",
        write_videos=False,
    )

    # Corrupt episode 0 by injecting NaN values into action column
    ep0_path = ds_dir / "data" / "chunk-000" / "episode_000000.parquet"
    table = pq.read_table(ep0_path)
    bad_dict = table.to_pydict()
    bad_dict["action"][5][2] = float("nan")  # inject NaN into angular_z
    pq.write_table(pa.Table.from_pydict(bad_dict), ep0_path)

    chk_file = tmp_path / "should_not_exist.pt"

    # Assert train_act_model raises RuntimeError due to pre-flight gate failure
    with pytest.raises(RuntimeError, match="Pre-flight QA Gate BLOCKED"):
        train_act_model(
            dataset_path=ds_dir,
            config_path=CONFIGS_DIR / "act_openamrobot_19dof.yaml",
            output_checkpoint=chk_file,
            smoke_test=True,
            source_profile="GAZEBO",
        )

    assert not chk_file.exists(), "Model checkpoint must NOT be saved when QA gate fails"
