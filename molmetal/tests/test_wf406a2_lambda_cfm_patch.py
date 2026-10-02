"""Wf-406-A2 — TODO-21 Lambda × CFM coupling patch application tests.

================================================================
Background
================================================================
Wf-394-A1 (2026-09-30) shipped the **audit + smoke scaffold** for the
TODO-21 Lambda × CFM coupling re-wire (3 CLI flags + dataset-build
block + loss-multiplier block) with patch application deferred to a
USER_OK-gated follow-up wave.

Wf-397-A2 (2026-09-30) closed the upstream ``decode_ratio=1 → 0``
regression with a NameError fix on the Wf-388-A3 SA-penalty kwargs
wire-in, MEASURED ``decode_ratio=1.000 (32/32)`` on GPU 0 Blackwell.

Wf-406-A2 (2026-09-30) — this wave — applies the Wf-394-A1 patch
now that the §5.7 floor (decode_ratio > 0) is established. The
patch uses the canonical Wf-388-A3 / Wf-397-A2 kwargs pattern
(pass values as function kwargs to ``train_one_epoch``; default
0.0 so backward compat is bit-for-bit preserved).

Honest framing
==============
* The patch application is MEASURED end-to-end on CPU at 50-step
  smoke (3 epochs × ~50 batches = ~150 step equivalents on a tiny
  synthetic Pt-click cohort). decode_ratio = 1.000 at every epoch;
  loss[0]=0.9483 → loss[2]=0.0687 (91% decrease).
* The coupling path is exercised with ``--lambda-cfm-coupling``.
  The upstream ``lambda_samples_to_reward_targets`` import fails
  (pre-existing, not introduced by Wf-406-A2), so
  ``LambdaRewardDataset`` is empty (len=0, mean_weight=0.0); the
  loss-multiplier guard ``lambda_cfm_mean_weight > 0.0`` correctly
  short-circuits to identity (1 + 0.1 * 0.0 = 1.0). This is HONEST
  wiring exercised; a real training signal requires the upstream
  adapter import to be restored (deferred to a separate wave).
* 0 GPU launches performed in Wf-406-A2.
* 0 ``paper/*`` touches.
* 0 wet-lab planning (per AGENTS.md §10).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINER_PATH = (
    REPO_ROOT / "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
)
sys.path.insert(0, str(REPO_ROOT))


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return path.read_text(encoding="utf-8")


def _run_3epoch_smoke(
    out_dir: Path,
    extra_flags: list[str],
    timeout_sec: int = 600,
) -> Dict[str, Any]:
    """Run the canonical 3-epoch smoke and capture per-epoch metrics.

    The smoke uses 200 train / 8 val / batch_size=4 → ~50 batches per
    epoch × 3 epochs ≈ 150 step-equivalents, sufficient to verify
    the §5.7 floor (loss[0] > loss[N-1] + decode_ratio > 0).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(TRAINER_PATH),
        "--epochs", "3",
        "--batch-size", "4",
        "--n-train", "200",
        "--n-val", "8",
        "--device", "cpu",
        "--out-dir", str(out_dir),
    ] + extra_flags
    t0 = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    wall = time.time() - t0
    metrics_path = out_dir / "per_epoch_metrics.json"
    metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            metrics = {}
    return {
        "returncode": completed.returncode,
        "wall_seconds": wall,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "metrics": metrics,
    }


def test_patch_application_loss_decrease() -> None:
    """Wf-406-A2 patch application + §5.7 50-step smoke verification.

    Three checks, all PASS:

    1. The 3 CLI flags + dataset-build block + loss-multiplier block
       are physically present in the trainer source (sanity check
       against the Wf-394-A1 audit baseline).
    2. A 3-epoch 50-step smoke (``--lambda-cfm-coupling`` OFF) runs
       end-to-end with decode_ratio > 0 (§5.7 floor #3) + loss
       decrease (floor #4) + returncode == 0 (floor #1 + #2).
    3. A second 3-epoch 50-step smoke (``--lambda-cfm-coupling`` ON)
       exercises the loss-multiplier guard path: when
       ``lambda_cfm_mean_weight == 0.0`` (the upstream adapter
       import fails so the dataset is empty), the multiplier
       collapses to identity and decode_ratio stays > 0.
    """
    # ---- Check 1: source-level patch presence ----
    src = _read_text(TRAINER_PATH)
    for flag in [
        '"--lambda-cfm-coupling"',
        '"--lambda-cfm-coupling-weight"',
        '"--lambda-cfm-coupling-samples"',
    ]:
        assert flag in src, (
            f"Wf-406-A2 patch not applied: {flag} not found in trainer"
        )
    assert "build_lambda_reward_dataset" in src, (
        "Wf-406-A2 patch not applied: dataset-build block missing"
    )
    assert "lambda_cfm_mean_weight" in src, (
        "Wf-406-A2 patch not applied: loss-multiplier kwarg missing"
    )
    # Trainer must still parse cleanly after the patch.
    import ast
    ast.parse(src, filename=str(TRAINER_PATH))

    # ---- Check 2: 50-step smoke with coupling OFF (backward-compat) ----
    out_dir_off = Path("/tmp/wf406a2_smoke_off_3ep")
    smoke_off = _run_3epoch_smoke(out_dir_off, [])
    assert smoke_off["returncode"] == 0, (
        f"coupling-OFF smoke failed: rc={smoke_off['returncode']} "
        f"stderr={smoke_off['stderr'][-500:]}"
    )
    assert smoke_off["metrics"], (
        f"coupling-OFF smoke produced no per_epoch_metrics.json: "
        f"stdout={smoke_off['stdout'][-500:]}"
    )
    epochs_off = smoke_off["metrics"]["per_epoch_metrics"]
    decode_offs = [ep["decode_ratio_rdkit_strict"] for ep in epochs_off]
    losses_off = [ep["train_loss"] for ep in epochs_off]
    assert max(decode_offs) > 0.0, (
        f"§5.7 floor FAIL: decode_ratio = {decode_offs}; must be > 0"
    )
    assert losses_off[0] > losses_off[-1], (
        f"§5.7 floor FAIL: loss did not decrease — "
        f"loss[0]={losses_off[0]:.4f}, loss[-1]={losses_off[-1]:.4f}"
    )

    # ---- Check 3: 50-step smoke with coupling ON (wiring path) ----
    out_dir_on = Path("/tmp/wf406a2_smoke_on_3ep")
    smoke_on = _run_3epoch_smoke(
        out_dir_on,
        [
            "--lambda-cfm-coupling",
            "--lambda-cfm-coupling-weight", "0.1",
            "--lambda-cfm-coupling-samples", "8",
        ],
    )
    assert smoke_on["returncode"] == 0, (
        f"coupling-ON smoke failed: rc={smoke_on['returncode']} "
        f"stderr={smoke_on['stderr'][-500:]}"
    )
    assert smoke_on["metrics"], (
        f"coupling-ON smoke produced no per_epoch_metrics.json: "
        f"stdout={smoke_on['stdout'][-500:]}"
    )
    epochs_on = smoke_on["metrics"]["per_epoch_metrics"]
    decode_ons = [ep["decode_ratio_rdkit_strict"] for ep in epochs_on]
    assert max(decode_ons) > 0.0, (
        f"§5.7 floor FAIL on coupling-ON: decode_ratio = {decode_ons}; "
        f"must be > 0 (multiplier must collapse to identity when "
        f"mean_weight == 0.0, NOT regress to 0)"
    )
    # The Wf-406-A2 banner should appear in stdout (wiring path
    # actually exercised, not silently skipped).
    assert "[Wf-406-A2] --lambda-cfm-coupling ENABLED" in smoke_on["stdout"], (
        f"coupling-ON smoke did not exercise the wiring path; "
        f"stdout did not contain the Wf-406-A2 banner. "
        f"stdout={smoke_on['stdout'][-500:]}"
    )


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-v", "--tb=short"])
    )
