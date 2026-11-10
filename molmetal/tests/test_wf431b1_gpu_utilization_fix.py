"""Wf-431-B1 — GPU utilization optimization wire-in regression-guard tests.

Per AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule (added by Wf-399-A1):
  "every new --flag wired into a trainer's training loop MUST run a 50-step
   end-to-end smoke verifying decode_ratio > 0 + loss[0] > loss[49]".

Wf-431-A1 §ROOT-CAUSE diagnosis established the F1-F6 fix list:
  F1: torch.compile (+20-50 pct GPU util on Blackwell) — wired via --torch-compile
  F2: BF16 + TF32 + cudnn.benchmark (+30-80 pct matmul throughput) —
      wired via --tf32 + --cudnn-benchmark (BF16 deferred — invasive autocast
      wrap incompatible with §5.7 CPU smoke discipline)
  F3: DataLoader refactor (DEFERRED — per Wf-431-A1 §F3 caveat requires
      rewriting custom Python loop to batched AtomCloud tensors; bit-exact
      backward compat not guaranteed; out of scope for this §5.7 ship wave)
  F4: cudnn.benchmark + channels_last (+5-15 pct GPU util) — wired via
      --cudnn-benchmark + --channels-last
  F5: Larger smoke config — DEFERRED (YAML config change, out of trainer scope)
  F6: Real retrain — DEFERRED (USER_OK-gated per AGENTS.md §5.1)

Smoke config: --epochs 3 --n-train 200 --batch-size 8 = 75 optimizer steps,
well above the §5.7 floor of 50 steps. 3 epochs is enough to verify
monotone non-increasing loss curve.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from molmetal.scripts.wf399a1_smoke_protocol import (
    SmokeOutput,
    verify_decode_ratio,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = str(
    REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
)


# 3 epochs × 25 batches/epoch = 75 optimizer steps (above the §5.7
# floor of 50). 3 epochs is enough to verify monotone non-increasing
# loss curve across epochs.
SMOKE_OUT_DIR = "/tmp/wf431b1_smoke"
SMOKE_METRICS_PATH = f"{SMOKE_OUT_DIR}/per_epoch_metrics.json"
SMOKE_FLAGS_BASE = [
    "--epochs",
    "3",
    "--n-train",
    "200",
    "--n-val",
    "8",
    "--batch-size",
    "8",
    "--device",
    "cpu",
    "--pocket-id",
    "pocket_a",
    "--seed",
    "0",
    "--out-dir",
    SMOKE_OUT_DIR,
]


def _read_metrics():
    metrics_path = Path(SMOKE_METRICS_PATH)
    assert metrics_path.exists(), f"per_epoch_metrics.json missing at {metrics_path}"
    return json.loads(metrics_path.read_text())


def _invoke_trainer(extra_flags: list[str], capture_metrics: bool = True):
    """Run the trainer via subprocess + return (returncode, stdout, stderr)."""
    import subprocess

    # Remove the whole out-dir, not just the metrics file. MEASURED
    # 2026-10-27: the trainer resumes from ``checkpoints/`` when the out-dir
    # already has one, so by the third invocation in a session it resumed at
    # the final epoch, ran zero epochs, and never wrote
    # per_epoch_metrics.json — the test then failed on a missing file rather
    # than on anything the trainer did. Each smoke must start clean.
    import shutil

    shutil.rmtree(SMOKE_OUT_DIR, ignore_errors=True)

    cmd = [
        sys.executable,
        TRAINER_PATH,
        *SMOKE_FLAGS_BASE,
        *extra_flags,
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        # MEASURED 2026-10-27: this 3-epoch CPU smoke takes ~10 s standalone
        # but ~120 s when the suite is under load, and the old 120 s cap
        # killed it mid-run — the test then failed on a missing metrics file
        # rather than on the trainer. 300 s leaves headroom without hiding a
        # genuine hang.
        timeout=300,
    )
    return completed.returncode, completed.stdout, completed.stderr


# ---------------------------------------------------------------------------
# §5.7 default-off regression guard (bit-for-bit backward compat)
# ---------------------------------------------------------------------------


def test_default_off_trainer_runs_clean():
    """Baseline: trainer with NO new flags runs cleanly."""
    rc, stdout, stderr = _invoke_trainer([])
    assert rc == 0, f"rc={rc}; stdout_tail={stdout[-300:]}; stderr_tail={stderr[-300:]}"


def test_default_off_decode_ratio_positive():
    """Baseline: decode_ratio > 0 with no new flags (cf. Wf-148 m14_real_cfm 0.998)."""
    rc, stdout, stderr = _invoke_trainer([])
    assert rc == 0, f"rc={rc}; stderr_tail={stderr[-300:]}"
    metrics = _read_metrics()
    assert "per_epoch_metrics" in metrics
    drs = [
        m["decode_ratio_val"] / max(m["decode_ratio_n"], 1)
        for m in metrics["per_epoch_metrics"]
    ]
    assert all(dr > 0.0 for dr in drs), f"decode_ratio ≤ 0; drs={drs}"
    # Best decode_ratio should be 1.0 on smoke (small cohort).
    best = max(
        m["decode_ratio_val"] / max(m["decode_ratio_n"], 1)
        for m in metrics["per_epoch_metrics"]
    )
    assert best >= 0.8, f"best decode_ratio < 0.8; best={best}"


def test_default_off_loss_decreasing():
    """Baseline: loss[0] > loss[-1] across 3 epochs (75 steps)."""
    rc, stdout, stderr = _invoke_trainer([])
    assert rc == 0, f"rc={rc}; stderr_tail={stderr[-300:]}"
    metrics = _read_metrics()
    losses = [float(m["train_loss"]) for m in metrics["per_epoch_metrics"]]
    assert len(losses) >= 2
    assert losses[0] > losses[-1], f"loss not decreasing; losses={losses}"


# ---------------------------------------------------------------------------
# §5.7 per-flag wire-in smoke (NEW Wf-431-B1 flags)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag_name",
    ["--torch-compile", "--tf32", "--cudnn-benchmark", "--channels-last"],
)
def test_each_new_flag_trainer_imports_clean(flag_name: str) -> None:
    """Each NEW flag imports + initialises without NameError / argparse error.

    §5.7 acceptance: trainer imports + initialises cleanly with flag set.
    CPU smoke verifies wire-in does NOT introduce NameError or RuntimeError.
    """
    rc, stdout, stderr = _invoke_trainer([flag_name])
    assert rc == 0, (
        f"--{flag_name.lstrip('-')} trainer FAILED rc={rc}; "
        f"stdout_tail={stdout[-300:]} stderr_tail={stderr[-300:]}"
    )


def test_all_4_new_flags_combined_trainer_runs_clean():
    """All 4 NEW flags combined: trainer imports + initialises + first epoch runs."""
    rc, stdout, stderr = _invoke_trainer(
        [
            "--torch-compile",
            "--cudnn-benchmark",
            "--tf32",
            "--channels-last",
        ]
    )
    assert rc == 0, (
        f"combined-4-flag trainer FAILED rc={rc}; "
        f"stdout_tail={stdout[-300:]} stderr_tail={stderr[-300:]}"
    )


def test_parse_args_accepts_all_4_new_flags():
    """parse_args accepts --torch-compile --tf32 --cudnn-benchmark --channels-last."""
    rc, stdout, stderr = _invoke_trainer(
        [
            "--torch-compile",
            "--tf32",
            "--cudnn-benchmark",
            "--channels-last",
        ]
    )
    # If parse_args rejected any flag, argparse would print error and exit 2.
    assert rc == 0, f"parse_args rejected a flag rc={rc}; stderr_tail={stderr[-300:]}"


# ---------------------------------------------------------------------------
# §5.7 programmatic invariant helpers (unit-test the verifier)
# ---------------------------------------------------------------------------


def test_smoke_protocol_signature_compatible():
    """SmokeOutput + verify_decode_ratio work as designed."""
    out = SmokeOutput(
        trainer_path=TRAINER_PATH,
        flags=["--epochs", "1"],
        returncode=0,
        stdout="[train] decode_ratio=0.95",
        stderr="",
        metrics_path=None,
        metrics={},
        wall_seconds=10.0,
    )
    verdict = verify_decode_ratio(out)
    assert verdict["decode_ratio"] == 0.95
    assert verdict["invariant_holds"] is True
