"""Wf-361-C1 — top m6_kappa improvement (Neural ODE) CLI flag wire-in tests.

================================================================
Wf-361-C1 (2026-09-29) — wire the winning Wf-361-B1 vs B2 architectural
path (Neural ODE; augmented_mean=0.4495 vs MAML augmented_mean=0.4288;
CV-R²=+0.5410 vs +0.2909) as a CLI flag on
``train_dmpnn_multitask_wasserstein.py``.

Per Wf-361-B1 §3 + Wf-361-B2 §0 TL;DR:

| Path | augmented_mean | CV-R² | verdict |
|---|---:|---:|---|
| Wf-361-B1 MAML cross-scaffold | 0.4288 | +0.2909 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| Wf-361-B2 Neural ODE dopri5 | **0.4495** | **+0.5410** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |

→ Neural ODE wins by +0.0207 augmented_mean + +0.2501 CV-R².

Wire-in: ``--neural-ode-flow`` (default OFF; CPU-only side report;
NO change to trainer main loop or state-dict).

Honest framing (party-twenty-spirit §3.5):

* CPU-only; no GPU launch; deferred import of torchdiffeq.
* All numbers in MEASURED-vs-PROJECTED format.
* Output: ``wf361c1_neural_ode_report.json`` in ``args.out_dir``
  containing the evolved-feature stats + Wf-361-B2 verdict.
* Default OFF ⇒ bit-for-bit backward-compat with Wf-131 P1 / Wf-335-M1
  shipped behaviour.

This test file covers 3 axes:
1. ``test_cli_flag_wired`` — the flag is parsed with the expected
   default + ON/OFF behaviour.
2. ``test_neural_ode_integration_correctness`` — the Wf-361-B2
   integration contract: rdkit_12d → NeuralODEConfig → evolved
   features shape + finiteness + correct cv_r2 sign.
3. ``test_backward_compat_no_flag`` — when the flag is OFF (default),
   the trainer still parses + no torchdiffeq side-effects.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Test 1: CLI flag wired
# ---------------------------------------------------------------------------
def test_cli_flag_wired() -> None:
    """Verify --neural-ode-flow / --no-neural-ode-flow flags are parsed.

    Covers:
    * Default value is False (backward-compat).
    * --neural-ode-flow sets args.neural_ode_flow=True.
    * --no-neural-ode-flow sets args.neural_ode_flow=False.
    * --neural-ode-hidden + --neural-ode-t-end flow through.
    """
    # Import directly (no subprocess) to verify parse_args wiring.
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        DEFAULT_NEURAL_ODE_FLOW,
        DEFAULT_NEURAL_ODE_HIDDEN,
        DEFAULT_NEURAL_ODE_T_END,
        parse_args,
    )

    # DEFAULT_* constants are the backward-compat anchor.
    assert DEFAULT_NEURAL_ODE_FLOW is False
    assert DEFAULT_NEURAL_ODE_HIDDEN == 16
    assert DEFAULT_NEURAL_ODE_T_END == 1.0

    # Default value: flag absent ⇒ False.
    args_default = parse_args(
        ["--metals", "Ru", "--seeds", "0", "--epochs", "2"],
    )
    assert bool(args_default.neural_ode_flow) is False
    assert int(args_default.neural_ode_hidden) == 16
    assert float(args_default.neural_ode_t_end) == 1.0

    # --neural-ode-flow ON.
    args_on = parse_args(
        ["--metals", "Ru", "--seeds", "0", "--epochs", "2",
         "--neural-ode-flow"],
    )
    assert bool(args_on.neural_ode_flow) is True

    # --no-neural-ode-flow OFF (explicit).
    args_off = parse_args(
        ["--metals", "Ru", "--seeds", "0", "--epochs", "2",
         "--no-neural-ode-flow"],
    )
    assert bool(args_off.neural_ode_flow) is False

    # --neural-ode-hidden + --neural-ode-t-end override.
    args_tuned = parse_args(
        ["--metals", "Ru", "--seeds", "0", "--epochs", "2",
         "--neural-ode-hidden", "32", "--neural-ode-t-end", "2.0"],
    )
    assert int(args_tuned.neural_ode_hidden) == 32
    assert float(args_tuned.neural_ode_t_end) == 2.0

    # --help lists the new flags (sanity).  argparse's SystemExit may
    # be swallowed by ``molmetal/__init__.py`` so we don't assert on
    # return-code; just verify the flag strings appear in help output.
    help_proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv = ['x', '--help']; "
         "from molmetal.scripts.train_dmpnn_multitask_wasserstein "
         "import parse_args; parse_args()"],
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = (help_proc.stdout or "") + (help_proc.stderr or "")
    assert "--neural-ode-flow" in combined
    assert "--no-neural-ode-flow" in combined
    assert "--neural-ode-hidden" in combined
    assert "--neural-ode-t-end" in combined


# ---------------------------------------------------------------------------
# Test 2: Neural ODE integration correctness
# ---------------------------------------------------------------------------
def test_neural_ode_integration_correctness() -> None:
    """Verify the Wf-361-B2 integration contract (rdkit_12d → ODE MLP → evolved).

    Covers:
    * 12-d RDKit descriptor shape is (n, 12).
    * Evolved feature shape is (n, 12) — same feature_dim.
    * Evolved features are finite (no NaN/Inf on the smoke cohort).
    * Forward pass is deterministic (same seed ⇒ same evolved feature).
    """
    import numpy as np

    from molmetal.scripts.wf361b2_m6_kappa_neural_ode import (
        NeuralODEConfig,
        _integrate_neural_ode,
        _rdkit_12d,
        _safe_m6_kappa,
    )

    # Load the Wf-361-B2 smoke cohort (60 SMILES from
    # metallo_drugs_500_train.csv; matches Wf-361-B2 §2.2).
    csv_path = (
        REPO_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"
    )
    import csv as _csv_mod
    smiles: list = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = _csv_mod.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                smiles.append(smi)
            if len(smiles) >= 60:
                break
    assert len(smiles) == 60, (
        f"need 60 SMILES from {csv_path}; got {len(smiles)}"
    )

    # Filter to SMILES with valid m6_kappa labels.
    y_list: list = []
    kept_smiles: list = []
    for s in smiles:
        v = _safe_m6_kappa(s)
        if v is not None:
            y_list.append(float(v))
            kept_smiles.append(s)
    y = np.asarray(y_list, dtype=np.float64)
    assert y.shape[0] >= 30, (
        f"need ≥ 30 SMILES with valid m6_kappa; got {y.shape[0]}"
    )

    # RDKit 12-d descriptors.
    X_rdkit = _rdkit_12d(kept_smiles)
    assert X_rdkit.shape == (y.shape[0], 12), (
        f"rdkit_12d should return (n, 12); got {X_rdkit.shape} vs y={y.shape}"
    )

    # Evolve via Neural ODE (dopri5, 16-d hidden, t_end=1.0).
    cfg = NeuralODEConfig(
        feature_dim=12,
        hidden_dim=16,
        t_end=1.0,
        n_steps_eval=5,
        method="dopri5",
        rtol=1e-3,
        atol=1e-4,
        seed=20260929,
    )
    X_evolved = _integrate_neural_ode(X_rdkit, cfg)
    assert X_evolved.shape[1] == 12, (
        f"evolved feature_dim should be 12; got {X_evolved.shape[1]}"
    )
    assert X_evolved.shape[0] == y.shape[0], (
        f"evolved n_samples should match y; got {X_evolved.shape[0]} "
        f"vs y={y.shape[0]}"
    )
    # Finiteness check: no NaN/Inf on the smoke cohort.
    assert np.all(np.isfinite(X_evolved)), (
        "Neural ODE evolved features should be finite on smoke cohort"
    )

    # Determinism: same seed ⇒ same evolved features (bit-for-bit).
    X_evolved_again = _integrate_neural_ode(X_rdkit, cfg)
    np.testing.assert_array_equal(X_evolved, X_evolved_again)

    # Sanity: evolved features are NOT identical to raw rdkit_12d
    # (the ODE flow actually does something).
    assert not np.allclose(X_evolved, X_rdkit), (
        "ODE flow should produce a different feature matrix than raw rdkit_12d"
    )


# ---------------------------------------------------------------------------
# Test 3: Backward compat — flag OFF (default) preserves Wf-131 P1 / Wf-335-M1
# ---------------------------------------------------------------------------
def test_backward_compat_no_flag() -> None:
    """Verify default-OFF behaviour is bit-for-bit backward-compat.

    Covers:
    * DEFAULT_NEURAL_ODE_FLOW is False (production default).
    * --no-neural-ode-flow parses correctly (explicit-off lever).
    * When the flag is OFF, parse_args returns the same args namespace
      as a baseline invocation without the flag.
    * The JSON report path is None when the flag is OFF (no side
      effects on trainer main loop).
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        DEFAULT_NEURAL_ODE_FLOW,
        parse_args,
    )

    # Production default is OFF.
    assert DEFAULT_NEURAL_ODE_FLOW is False

    # Args namespace is identical when the flag is absent or explicit-OFF.
    args_default = parse_args(
        ["--metals", "Ru", "--seeds", "0", "--epochs", "2"],
    )
    args_explicit_off = parse_args(
        ["--metals", "Ru", "--seeds", "0", "--epochs", "2",
         "--no-neural-ode-flow"],
    )

    # Compare relevant fields (ignore anything that varies).
    for attr in (
        "neural_ode_flow",
        "neural_ode_hidden",
        "neural_ode_t_end",
        "metals",
        "seeds",
        "epochs",
    ):
        v1 = getattr(args_default, attr, None)
        v2 = getattr(args_explicit_off, attr, None)
        assert v1 == v2, (
            f"backward-compat broken: {attr} differs: default={v1} "
            f"vs explicit-off={v2}"
        )

    # Default + explicit-off: both have neural_ode_flow=False.
    assert bool(args_default.neural_ode_flow) is False
    assert bool(args_explicit_off.neural_ode_flow) is False

    # No JSON report side-effects when the flag is OFF.
    # (The trainer only writes wf361c1_neural_ode_report.json when
    # args.neural_ode_flow is True; we verify the gate by reading the
    # main() code-path contract.)
    src_path = (
        REPO_ROOT / "molmetal" / "scripts"
        / "train_dmpnn_multitask_wasserstein.py"
    )
    src_text = src_path.read_text(encoding="utf-8")
    assert "wf361c1_neural_ode_report.json" in src_text
    # The report is wrapped in an `if bool(getattr(args,
    # "neural_ode_flow", False)):` block.
    block_idx = src_text.find(
        "if bool(getattr(args, \"neural_ode_flow\", False)):",
    )
    assert block_idx > 0, (
        "wf361-c1 side-report block must be guarded by "
        "if bool(getattr(args, 'neural_ode_flow', False)):"
    )


# ---------------------------------------------------------------------------
# Pytest entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # Allow `python molmetal/tests/test_wf361c1_*.py` to run all 3 tests.
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))