#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_act_baseline.py
=====================

OpenAMRobot 2.0 | Workstream C | Deliverable C11 (Due 6 Nov).
ACT (Action Chunking with Transformers) Baseline Training Pipeline.

Implements the imitation-learning policy baseline for the 19-channel
whole-body robot specification:
  - Enforces mandatory C10 pre-flight integrity gate before training starts.
  - Deterministically partitions episodes into Train and Held-Out Evaluation splits.
  - Trains or smoke-tests the CVAE-based ACT policy.
  - Saves the policy checkpoint, SHA-256 hash, and loss curves.

Usage:
  # Fast CPU smoke-test (default 3 epochs, verifies forward/backward passes)
  python scripts/train_act_baseline.py --dataset-path data/episodes --smoke-test

  # Full training run
  python scripts/train_act_baseline.py --dataset-path data/episodes --epochs 50 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

# ---------------------------------------------------------------------------
# Setup logging and path resolution
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("act-trainer")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIGS_DIR = REPO_ROOT / "configs"

for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check_dataset_integrity as cdi  # noqa: E402

# ---------------------------------------------------------------------------
# Contract Constants
# ---------------------------------------------------------------------------

STATE_DIM = 19
ACTION_DIM = 19
DEFAULT_CHUNK_SIZE = 100
SCHEMA_REVISION = "schema-v1.0-frozen"


# ---------------------------------------------------------------------------
# ACT Neural Architecture (PyTorch implementation)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ACTEncoder(nn.Module):
    """CVAE Encoder: maps (state, action_chunk) -> latent distribution (mu, log_var)."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 4,
        latent_dim: int = 32,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.latent_proj = nn.Linear(d_model, latent_dim * 2)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # state: (B, state_dim), actions: (B, chunk_size, action_dim)
        b = state.size(0)
        s_tok = self.state_proj(state).unsqueeze(1)  # (B, 1, d_model)
        a_tok = self.action_proj(actions)            # (B, chunk_size, d_model)
        cls_tok = self.cls_token.expand(b, -1, -1)   # (B, 1, d_model)

        tokens = torch.cat([cls_tok, s_tok, a_tok], dim=1)  # (B, 2 + chunk_size, d_model)
        tokens = self.pos_enc(tokens)
        encoded = self.transformer_encoder(tokens)
        cls_out = encoded[:, 0, :]  # extract cls token representation

        latent_params = self.latent_proj(cls_out)
        mu, log_var = torch.chunk(latent_params, 2, dim=-1)
        return mu, log_var


class ACTDecoder(nn.Module):
    """ACT Policy Decoder: maps (state, latent_z) -> predicted action chunk (B, chunk_size, 19)."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 4,
        latent_dim: int = 32,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.state_proj = nn.Linear(state_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # Learned target query tokens for each timestep in the action chunk
        self.query_tokens = nn.Parameter(torch.randn(1, chunk_size, d_model))
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # state: (B, state_dim), z: (B, latent_dim)
        b = state.size(0)
        s_emb = self.state_proj(state).unsqueeze(1)  # (B, 1, d_model)
        z_emb = self.latent_proj(z).unsqueeze(1)    # (B, 1, d_model)
        memory = torch.cat([s_emb, z_emb], dim=1)   # conditioning context (B, 2, d_model)

        queries = self.query_tokens.expand(b, -1, -1)
        queries = self.pos_enc(queries)
        decoded = self.transformer_decoder(tgt=queries, memory=memory)
        predicted_actions = self.action_head(decoded)  # (B, chunk_size, action_dim)
        return predicted_actions


class ACTPolicy(nn.Module):
    """Complete Action Chunking with Transformers (ACT) Policy."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        d_model: int = 256,
        n_heads: int = 8,
        enc_layers: int = 4,
        dec_layers: int = 4,
        latent_dim: int = 32,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        kl_weight: float = 10.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.encoder = ACTEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=enc_layers,
            latent_dim=latent_dim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = ACTDecoder(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=dec_layers,
            latent_dim=latent_dim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if actions is not None:
            # Training: sample from encoder posterior
            mu, log_var = self.encoder(state, actions)
            z = self.reparameterize(mu, log_var)
            pred_actions = self.decoder(state, z)
            return pred_actions, mu, log_var
        else:
            # Inference: sample from prior N(0, I)
            b = state.size(0)
            z = torch.zeros(b, self.latent_dim, device=state.device)
            pred_actions = self.decoder(state, z)
            return pred_actions, None, None

    def compute_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l1_loss = F.l1_loss(pred_actions, target_actions)
        # KL( q(z|a,s) || N(0, I) )
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
        total_loss = l1_loss + (self.kl_weight * kl_loss)
        return total_loss, l1_loss, kl_loss


# ---------------------------------------------------------------------------
# Dataset and Dataloading
# ---------------------------------------------------------------------------

class ACTDataset(Dataset):
    """Loads 19-dim trajectories and extracts (state, action_chunk) pairs."""

    def __init__(
        self,
        episode_paths: Sequence[Path],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        import pyarrow.parquet as pq

        self.samples: List[Tuple[np.ndarray, np.ndarray, int]] = []
        self.chunk_size = chunk_size

        for ep_path in episode_paths:
            table = pq.read_table(ep_path)
            state_raw = np.array(table.column("observation.state").to_pylist(), dtype=np.float32)
            act_raw = np.array(table.column("action").to_pylist(), dtype=np.float32)

            num_rows = len(state_raw)
            if num_rows < 1:
                continue

            for t in range(num_rows):
                # Target action chunk of length chunk_size (padded with last frame if near end)
                if t + chunk_size <= num_rows:
                    chunk = act_raw[t : t + chunk_size]
                else:
                    pad_len = (t + chunk_size) - num_rows
                    chunk = np.concatenate([act_raw[t:], np.repeat(act_raw[-1:], pad_len, axis=0)], axis=0)

                self.samples.append((state_raw[t], chunk, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        state, chunk, _ = self.samples[idx]
        return torch.from_numpy(state), torch.from_numpy(chunk)


# ---------------------------------------------------------------------------
# Pre-Flight Verification Gate
# ---------------------------------------------------------------------------

def enforce_preflight_gate(
    dataset_path: Path,
    source_profile: str = "GAZEBO",
    reports_dir: Optional[Path] = None,
) -> bool:
    """Run mandatory pre-flight integrity check; return True if OPEN."""
    LOG.info("=== Running C10 Pre-Flight Gate on %s ===", dataset_path)
    qa_report_file = (reports_dir or (REPO_ROOT / "reports")) / "qa_report.json"
    qa_report_file.parent.mkdir(parents=True, exist_ok=True)

    ffprobe_ok = shutil.which("ffprobe") is not None
    config = cdi.QAConfig(
        skew_threshold_ms=30.0,
        min_frames=30,
        require_source_profile=source_profile,
        fps=30.0,
        video_checks_enabled=ffprobe_ok,
    )
    report = cdi.run_qa(dataset_path=dataset_path, config=config, skip_video_checks=not ffprobe_ok)
    qa_report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary = report.get("summary", {})
    gate = summary.get("training_gate", "BLOCKED")
    overall = summary.get("overall_status", "FAIL")

    is_open = (gate == "OPEN" or overall == "PASS")
    if not is_open:
        LOG.error("MANDATORY GATE VIOLATION: training_gate is %s (overall %s)", gate, overall)
        return False

    LOG.info("Pre-flight gate PASSED: training_gate is OPEN (%d episodes verified)", summary.get("passed", 0))
    return True


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train_act_model(
    dataset_path: Path,
    config_path: Path,
    output_checkpoint: Path,
    epochs: int = 5,
    batch_size: int = 8,
    smoke_test: bool = False,
    device: str = "cpu",
    eval_split: float = 0.2,
    source_profile: str = "GAZEBO",
    output_summary: Optional[Path] = None,
) -> Dict[str, object]:
    """Execute training pipeline and return training summary dict."""
    dataset_path = Path(dataset_path).resolve()
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mandatory Pre-Flight Gate
    if not enforce_preflight_gate(dataset_path, source_profile=source_profile, reports_dir=reports_dir):
        raise RuntimeError("Pre-flight QA Gate BLOCKED: Cannot start C11 ACT training.")

    # 2. Discover episodes and partition deterministically
    parquet_files = sorted((dataset_path / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode Parquet files found under {dataset_path}/data")

    total_episodes = len(parquet_files)
    num_held_out = max(1, int(round(total_episodes * eval_split))) if total_episodes > 1 else 0
    num_train = total_episodes - num_held_out

    # Deterministic split: last num_held_out episodes are held out for validation
    train_files = parquet_files[:num_train]
    held_out_files = parquet_files[num_train:]

    train_indices = [int(p.stem.split("_")[-1]) for p in train_files]
    held_out_indices = [int(p.stem.split("_")[-1]) for p in held_out_files]

    LOG.info(
        "Partitioned %d episodes: Train=%d (%s), Held-out=%d (%s)",
        total_episodes,
        len(train_files),
        train_indices,
        len(held_out_files),
        held_out_indices,
    )

    # Load configuration
    cfg_data = {}
    if config_path.is_file():
        cfg_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    chunk_size = cfg_data.get("model", {}).get("chunk_size", DEFAULT_CHUNK_SIZE)
    latent_dim = cfg_data.get("model", {}).get("latent_dim", 32)
    d_model = cfg_data.get("model", {}).get("d_model", 256)
    n_heads = cfg_data.get("model", {}).get("n_heads", 8)
    enc_layers = cfg_data.get("model", {}).get("enc_layers", 4)
    dec_layers = cfg_data.get("model", {}).get("dec_layers", 4)
    kl_weight = cfg_data.get("training", {}).get("kl_weight", 10.0)

    # Fast parameters for smoke test
    if smoke_test:
        epochs = min(epochs, 3)
        d_model = 128
        n_heads = 4
        enc_layers = 2
        dec_layers = 2
        LOG.info("Running in fast --smoke-test mode: %d epochs, d_model=%d", epochs, d_model)

    train_ds = ACTDataset(train_files, chunk_size=chunk_size)
    eval_ds = ACTDataset(held_out_files, chunk_size=chunk_size) if held_out_files else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False) if eval_ds else None

    # Instantiate model
    model = ACTPolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        chunk_size=chunk_size,
        d_model=d_model,
        n_heads=n_heads,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        latent_dim=latent_dim,
        kl_weight=kl_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Training loop
    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_l1_acc = 0.0
        train_kl_acc = 0.0
        train_total_acc = 0.0
        batches = 0

        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            pred, mu, log_var = model(states, actions)
            total_loss, l1_loss, kl_loss = model.compute_loss(pred, actions, mu, log_var)
            total_loss.backward()
            optimizer.step()

            train_total_acc += total_loss.item()
            train_l1_acc += l1_loss.item()
            train_kl_acc += kl_loss.item()
            batches += 1

            if smoke_test and batches >= 5:
                break  # smoke test: keep epoch very fast

        avg_train_total = train_total_acc / max(1, batches)
        avg_train_l1 = train_l1_acc / max(1, batches)
        avg_train_kl = train_kl_acc / max(1, batches)

        # Eval step
        avg_eval_l1 = 0.0
        if eval_loader:
            model.eval()
            eval_l1_acc = 0.0
            eval_batches = 0
            with torch.no_grad():
                for states, actions in eval_loader:
                    states = states.to(device)
                    actions = actions.to(device)
                    pred, _, _ = model(states)  # inference mode
                    eval_l1_acc += F.l1_loss(pred, actions).item()
                    eval_batches += 1
                    if smoke_test and eval_batches >= 3:
                        break
            avg_eval_l1 = eval_l1_acc / max(1, eval_batches)

        LOG.info(
            "Epoch %02d/%02d | Train Loss: %.4f (L1: %.4f, KL: %.4f) | Held-Out L1: %.4f",
            epoch,
            epochs,
            avg_train_total,
            avg_train_l1,
            avg_train_kl,
            avg_eval_l1,
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_total, 5),
            "train_l1": round(avg_train_l1, 5),
            "train_kl": round(avg_train_kl, 5),
            "held_out_l1": round(avg_eval_l1, 5),
        })

    duration_sec = time.time() - start_time

    # Save best checkpoint
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg_data,
        "model_architecture": {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "chunk_size": chunk_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "latent_dim": latent_dim,
        },
        "train_episodes": train_indices,
        "held_out_episodes": held_out_indices,
        "final_epoch": epochs,
        "schema_revision": SCHEMA_REVISION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    torch.save(checkpoint_payload, output_checkpoint)

    # Compute SHA-256 hash of the checkpoint file
    chk_bytes = output_checkpoint.read_bytes()
    checkpoint_sha256 = hashlib.sha256(chk_bytes).hexdigest()

    # Save training loss history
    loss_file = reports_dir / "training_loss.json"
    loss_file.write_text(json.dumps({"history": history}, indent=2) + "\n", encoding="utf-8")

    # Save C11 training summary
    summary = {
        "deliverable": "C11 ACT Baseline Training",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema_revision": SCHEMA_REVISION,
        "checkpoint_path": str(output_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": len(chk_bytes),
        "epochs_completed": epochs,
        "duration_sec": round(duration_sec, 2),
        "episodes": {
            "total": total_episodes,
            "train_count": len(train_indices),
            "train_indices": train_indices,
            "held_out_count": len(held_out_indices),
            "held_out_indices": held_out_indices,
        },
        "final_metrics": history[-1] if history else {},
        "smoke_test": smoke_test,
    }
    summary_file = Path(output_summary) if output_summary is not None else (reports_dir / "c11_training_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    LOG.info("Checkpoint saved: %s (SHA-256: %s)", output_checkpoint, checkpoint_sha256[:16])
    LOG.info("Saved loss curve to %s and summary to %s", loss_file, summary_file)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OpenAMRobot 2.0 ACT Training Pipeline")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=REPO_ROOT / "data" / "episodes",
        help="Path to LeRobotDataset directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "act_openamrobot_19dof.yaml",
        help="ACT model configuration file",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "act_openamrobot_19dof_best.pt",
        help="Path to save trained policy checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast verification smoke test on CPU",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.2,
        help="Fraction of episodes to hold out for evaluation (default: 0.2)",
    )
    parser.add_argument(
        "--source-profile",
        type=str,
        default="GAZEBO",
        help="Required source profile for C10 gate (default: GAZEBO)",
    )

    args = parser.parse_args(argv)

    try:
        train_act_model(
            dataset_path=args.dataset_path,
            config_path=args.config,
            output_checkpoint=args.output_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            smoke_test=args.smoke_test,
            device=args.device,
            eval_split=args.eval_split,
            source_profile=args.source_profile,
        )
        return 0
    except Exception as exc:
        LOG.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
