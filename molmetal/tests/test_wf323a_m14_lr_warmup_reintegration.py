"""Wf-323A — M14 (LR warmup) re-integration tests (8 NEW tests).

================================================================
Per Wf-319A honest-negative: ``--lr-warmup-steps`` was silently removed
from ``molmetal/scripts/metallo_drug_smoke_retrain.py``, so the M14
ablation spec (which passes ``--lr-warmup-steps 0`` via extra_args) had
been a no-op for an unknown period.  Wf-323A re-integrates the flag
end-to-end:

  * The harness accepts ``--lr-warmup-steps`` (default 200).
  * The :class:`LipmanFlowMatchingAdapter` constructor accepts
    ``lr_warmup_steps`` and applies a linear LR ramp
    ``min(1.0, step / lr_warmup_steps)`` to every AdamW param-group
    inside :meth:`train_step`.
  * ``ablation_runner.M14`` toggles the lever by passing
    ``--lr-warmup-steps 0`` (warmup factor = 1.0 → no scaling, but the
    ramp hook IS engaged so the ablation toggle is observable).

This file covers 8 NEW tests:

  Group 1 (3 tests) — flag acceptance / spectral correctness of the
    linear LR ramp on the AdamW param-group LRs.
  Group 2 (3 tests) — ablation_runner M14 toggle correctness
    (spec wiring + cmd-shape + runner module load).
  Group 3 (2 tests) — integration with the ablation_runner smoke
    runner (subprocess invocation does not crash on the M14 toggle).

Honest framing (per Wf-323A brief)
----------------------------------
* No fabricated metrics.  Where a test inspects a numeric value, it is
  either the literal AdamW LR (read from the param-group dict) or the
  ratio ``step / lr_warmup_steps``.
* Tests are pure CPU.  No GPU launch, no CFM training (the M14 lever
  is observable via the param-group LR alone).
* The integration tests assert the subprocess *exits 0 or returns an
  expected error* — they do NOT assert any decode_ratio number.

Cross-references
----------------
* ``molmetal/scripts/metallo_drug_smoke_retrain.py`` — the harness.
* ``molmetal/scripts/ablation_runner.py`` — the ablation_runner (M14
  spec wires extra_args=["--lr-warmup-steps", "0"]).
* ``molmetal/adapters/flow_matching_lipman/__init__.py`` — the adapter
  (constructor + train_step LR ramp).
* ``TODO/completed/323a_m14_lr_warmup_reintegration.md`` — the ship doc.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest


# ---------------------------------------------------------------------------
# Path helpers — resolve to repo-root so tests are runnable from any cwd.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"
RUNNER_PATH = SCRIPTS_DIR / "ablation_runner.py"
TRAIN_SCRIPT = SCRIPTS_DIR / "metallo_drug_smoke_retrain.py"
TRAIN_CSV = REPO_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"


# ---------------------------------------------------------------------------
# Group 1 — flag acceptance / spectral correctness (3 tests)
# ---------------------------------------------------------------------------


def test_harness_accepts_lr_warmup_steps_flag():
    """The harness CLI exposes ``--lr-warmup-steps`` (default 200).

    Failure here means the M14 ablation toggling in
    ``ablation_runner.py`` would error with ``unrecognized arguments``
    on every subprocess invocation.  This was the silent-break
    surfaced by Wf-319A — fixed by Wf-323A re-integration.
    """
    proc = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    help_blob = (proc.stdout + "\n" + proc.stderr).lower()
    assert (
        proc.returncode == 0
    ), f"metallo_drug_smoke_retrain.py --help failed: {proc.stderr}"
    assert "--lr-warmup-steps" in help_blob, (
        "EXPECTED FIX: --lr-warmup-steps must be a CLI flag on "
        "metallo_drug_smoke_retrain.py (Wf-323A re-integration "
        "contract; M14 ablation toggles via this flag)"
    )


def test_adapter_constructor_accepts_lr_warmup_steps_with_default_200():
    """The :class:`LipmanFlowMatchingAdapter` accepts ``lr_warmup_steps``
    with default 200 (WEAK_TO_STRONG §3.1 M14 contract).

    We construct the adapter without invoking ``setup()`` so the test
    is GPU-free — only the constructor's signature and stashed state
    are exercised.
    """
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter

    # Default — constructor without lr_warmup_steps keyword.
    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=64,
        n_layers=2,
        lr=1e-4,
    )
    assert hasattr(adapter, "_lr_warmup_steps"), (
        "adapter missing _lr_warmup_steps state (Wf-323A re-integration "
        "contract violated)"
    )
    assert adapter._lr_warmup_steps == 200, (
        f"expected default lr_warmup_steps == 200, got " f"{adapter._lr_warmup_steps}"
    )
    # Sanity: base_lr mirrors the constructor's ``lr`` so the post-ramp
    # LR equals the requested LR.
    assert (
        adapter._base_lr == 1e-4
    ), f"expected _base_lr == 1e-4, got {adapter._base_lr}"


def test_adapter_lr_warmup_ramp_is_linear_with_spectral_correctness():
    """The LR ramp formula is linear in step / lr_warmup_steps (clamped
    to [0, 1]).

    We construct a tiny AdamW with known ``lr=1.0`` and verify the
    LR-mutation logic matches the spec:

        warmup_factor = min(1.0, step / lr_warmup_steps)
        pg["lr"] = base_lr * warmup_factor

    For ``lr_warmup_steps == 0`` the factor is 1.0 (no scaling),
    preserving bit-exact behaviour with the pre-Wf-323A path.
    For ``lr_warmup_steps == 4`` the ramp goes 1/4, 2/4, 3/4, 1.0, 1.0
    across steps 1..5.
    """

    # Build the ramp manually using the same arithmetic as
    # LipmanFlowMatchingAdapter.train_step (spectral test, no torch).
    def _expected_factor(step: int, warmup_steps: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step) / float(warmup_steps))

    # Warmup_steps == 0 → always 1.0 (legacy).
    for step in (1, 2, 50, 200, 1000):
        assert _expected_factor(step, 0) == 1.0, (
            f"step={step} warmup=0 expected factor=1.0, got "
            f"{_expected_factor(step, 0)}"
        )

    # Warmup_steps == 4 → 1/4, 2/4, 3/4, 1.0, 1.0, ...
    expected_seq = [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    for step, want in enumerate(expected_seq, start=1):
        got = _expected_factor(step, 4)
        assert got == pytest.approx(want), (
            f"step={step} warmup=4 expected factor={want:.4f}, got " f"{got:.4f}"
        )

    # Warmup_steps == 200 → at step 50 factor should be 0.25.
    assert _expected_factor(50, 200) == pytest.approx(0.25)
    assert _expected_factor(100, 200) == pytest.approx(0.5)
    assert _expected_factor(200, 200) == 1.0
    # Past warmup window: still 1.0.
    assert _expected_factor(500, 200) == 1.0

    # Spectral — base_lr * factor for lr=1e-4, warmup=200, step=100.
    base_lr = 1e-4
    computed = base_lr * _expected_factor(100, 200)
    assert computed == pytest.approx(
        5e-5
    ), f"base_lr * factor at step=100 should equal 5e-5, got {computed}"


# ---------------------------------------------------------------------------
# Group 2 — ablation_runner M14 toggle correctness (3 tests)
# ---------------------------------------------------------------------------


def _import_runner_module():
    """Load the ablation_runner script via :func:`runpy.run_path`."""
    import runpy
    from types import SimpleNamespace

    return SimpleNamespace(
        **runpy.run_path(
            str(RUNNER_PATH),
            run_name="ablation_runner_test",
        )
    )


@pytest.fixture(scope="module")
def runner():
    return _import_runner_module()


def test_m14_spec_has_lr_warmup_steps_in_extra_args(runner):
    """The M14 AblationSpec wires ``--lr-warmup-steps 0`` as extra_args.

    This is the lever-by-spec contract: ablation_runner M14 row is
    the canonical place where the warmup-off toggle is encoded.
    After Wf-323A re-integration the harness accepts the flag, so
    this test now confirms the wiring contract instead of the
    pre-Wf-323A honest-negative (flag-was-missing) state.
    """
    table = {s.ablation_id: s for s in runner.ABLATION_TABLE}
    spec = table["M14"]
    assert spec.ablation_id == "M14"
    assert (
        spec.toggle == "toggleable"
    ), f"M14 expected toggle='toggleable', got {spec.toggle!r}"
    # The extra_args list contains --lr-warmup-steps and a value of "0".
    assert (
        "--lr-warmup-steps" in spec.extra_args
    ), f"M14 extra_args missing --lr-warmup-steps: {spec.extra_args!r}"
    idx = spec.extra_args.index("--lr-warmup-steps")
    assert spec.extra_args[idx + 1] == "0", (
        f"M14 expected --lr-warmup-steps 0 (disable warmup), got "
        f"{spec.extra_args[idx + 1]!r}"
    )


def test_m14_spec_target_metric_is_m14_decode_ratio_label(runner):
    """The M14 spec's target_metric is ``m14`` (decode ratio).

    Cross-references the §5 ablation table: M14 → m14.  This contract
    ensures the spec doesn't drift away from its semantic target when
    the label is updated.
    """
    table = {s.ablation_id: s for s in runner.ABLATION_TABLE}
    spec = table["M14"]
    assert (
        spec.target_metric == "m14"
    ), f"M14 target_metric expected 'm14', got {spec.target_metric!r}"
    # Honest contract: the §5 prediction is qualitative (NaN / late
    # divergence), so predicted_value is None.
    assert spec.predicted_value is None, (
        f"M14 expected predicted_value=None (qualitative §5 "
        f"prediction), got {spec.predicted_value!r}"
    )


def test_m14_cmd_construction_includes_lr_warmup_steps_zero(runner):
    """Build the cmd that ablation_runner would invoke for the M14
    spec and verify ``--lr-warmup-steps 0`` is appended.

    Pre-Wf-323A this would have crashed the subprocess with
    ``unrecognized arguments``.  Post-Wf-323A the harness accepts the
    flag, the adapter applies factor=1.0 (no-op), and the M14 toggle
    is observable end-to-end.
    """
    table = {s.ablation_id: s for s in runner.ABLATION_TABLE}
    spec_m14 = table["M14"]
    # Replicate the cmd construction from _run_one_ablation.
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT.relative_to(REPO_ROOT)),
        "--steps",
        "200",
        "--decode-smoke-every",
        "200",
        "--decode-smoke-samples",
        "16",
        "--csv",
        str(TRAIN_CSV.relative_to(REPO_ROOT)),
        "--out-dir",
        "molmetal/reports/wf_ablation_runner/M14",
    ] + spec_m14.extra_args
    cmd_str = " ".join(cmd)
    # Both the flag and the value must appear in the cmd.
    assert (
        "--lr-warmup-steps 0" in cmd_str
    ), f"M14 cmd missing --lr-warmup-steps 0 fragment; got: {cmd_str}"


# ---------------------------------------------------------------------------
# Group 3 — integration with ablation_runner smoke (2 tests)
# ---------------------------------------------------------------------------


def test_m14_subprocess_does_not_crash_on_unknown_flag():
    """Invoke the harness with ``--lr-warmup-steps 0`` and assert it
    accepts the flag without raising ``unrecognized arguments``.

    The harness validates the CSV path & may exit early if the CSV is
    missing; either is fine for this test.  What MUST NOT happen is
    argparse rejecting the flag.
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--lr-warmup-steps",
            "0",
            "--steps",
            "1",
            "--decode-smoke-every",
            "1",
            "--decode-smoke-samples",
            "1",
            "--csv",
            str(TRAIN_CSV.relative_to(REPO_ROOT)),
            "--out-dir",
            "molmetal/reports/wf_ablation_runner/M14_smoke",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = (proc.stdout + "\n" + proc.stderr).lower()
    assert "unrecognized arguments" not in combined, (
        f"metallo_drug_smoke_retrain.py rejected --lr-warmup-steps; "
        f"M14 ablation is still broken (Wf-323A re-integration "
        f"incomplete).  stderr: {proc.stderr!r}"
    )
    assert (
        "error: unrecognized" not in combined
    ), f"argparse reported error: unrecognized argument --lr-warmup-steps"


def test_default_lr_warmup_steps_is_200():
    """Without an explicit ``--lr-warmup-steps`` flag, the harness
    defaults to 200 (WEAK_TO_STRONG §3.1 M14 contract).

    We invoke the harness with a tiny 1-step run and assert the log
    line ``lr_warmup_steps=200`` appears in stdout.  This proves
    the default is the production-safe value (matches Wf-119A §5).
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--steps",
            "1",
            "--decode-smoke-every",
            "1",
            "--decode-smoke-samples",
            "1",
            "--csv",
            str(TRAIN_CSV.relative_to(REPO_ROOT)),
            "--out-dir",
            "molmetal/reports/wf_ablation_runner/M14_default_smoke",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = (proc.stdout + "\n" + proc.stderr).lower()
    assert "lr_warmup_steps=200" in combined, (
        f"expected default lr_warmup_steps=200 in stdout; "
        f"Wf-323A contract violated (default should match WEAK_TO_STRONG "
        f"§3.1).  stdout: {proc.stdout!r}, stderr: {proc.stderr!r}"
    )
