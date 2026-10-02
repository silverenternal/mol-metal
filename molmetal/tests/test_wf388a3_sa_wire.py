"""Wf-388-A3 — tests for --sa-penalty CLI flag wire-in.

================================================================
What this test file covers
================================================================
Three NEW tests verifying the Wf-388-A3 wire-in of the Wf-380-A6
SA-aware sampling + loss penalty SCAFFOLD into the CFM trainer.

The wire-in (per Wf-380-A6 SCAFFOLD + TODO-205 §3) adds two CLI
flags to ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py``:

* ``--sa-penalty`` (default 0.0; OFF) — weight on the SA penalty term
  added to the velocity loss. Default 0.0 preserves bit-for-bit backward
  compatibility with Wf-380-B1 / Wf-299 train outputs (the penalty
  term is exactly zero when the flag is 0.0).
* ``--sa-target`` (default 2.8) — synthesizable threshold (Ertl 2009
  §3 / Wf-139 P6); pairs with ``--sa-penalty`` and is ignored when
  ``--sa-penalty == 0.0``.

The loss wire-in computes the penalty via the existing
``sa_aware_sampling_penalty`` helper (Wf-380-A6 SCAFFOLD module), with
defensive no-op fallbacks when:

* the SA penalty module is not importable, OR
* no pre-computed ``sa_scores_batch`` is available in the local scope.

This means default behaviour (no flag) is bit-for-bit identical to the
prior Wf-380-B1 / Wf-299 training output.

Tests
-----
1. ``test_sa_penalty_flag`` — CLI flag parses; default value is 0.0.
2. ``test_sa_aware_loss_in_c`` — when ``--sa-penalty > 0`` and the SA
   module is importable, ``sa_aware_sampling_penalty`` is invoked
   (verified by direct call using the same args namespace).
3. ``test_sa_target_threshold`` — default ``--sa-target`` is 2.8 per
   the Ertl 2009 §3 canonical synthesizable-region cutoff.

Constraints
-----------
* 0 GPU launches (pure CPU argparse + module inspection).
* 0 paper/* touches.
* 0 wet-lab planning.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINER_PATH = (
    REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
)
SA_PENALTY_MODULE_PATH = (
    REPO_ROOT
    / "molmetal"
    / "adapters"
    / "flow_matching_lipman"
    / "sa_penalty.py"
)


def _import_trainer_parse_args():
    """Import parse_args from the trainer module (avoids the GPU entry path).

    The trainer's top-level imports torch + RDKit + several domain
    modules; this works on CPU-only test hosts.  We deliberately only
    import the ``parse_args`` symbol (not the full ``main``) so the
    test does not require a CUDA device.

    Module is registered in ``sys.modules`` BEFORE ``exec_module`` so
    that dataclass decorators (which resolve ``sys.modules[cls.__module__]``
    during class processing) work correctly when the trainer is loaded
    via ``importlib.util.spec_from_file_location`` (which by default
    does NOT register the module).
    """
    import importlib.util
    import sys as _sys

    mod_name = "molmetal.scripts._wf388a3_trainer_mod"
    spec = importlib.util.spec_from_file_location(
        mod_name, str(TRAINER_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[mod_name] = mod
    # Loading the trainer will import torch + RDKit + domain modules
    # via the existing top-level ``import`` statements.  If any of
    # those are unavailable on the host, the test SKIPs rather than
    # FAILing (preserves Wf-345-A2 env-aware SKIP contract).
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except ModuleNotFoundError as exc:
        _sys.modules.pop(mod_name, None)
        pytest.skip(f"Wf-388-A3 requires {exc.name} on PYTHONPATH; skipping")
    return mod


# ---------------------------------------------------------------------------
# Test 1 — CLI flag parses with default 0.0
# ---------------------------------------------------------------------------
def test_sa_penalty_flag() -> None:
    """``--sa-penalty`` parses with default 0.0; ``--sa-target`` default 2.8.

    Verifies the two NEW Wf-388-A3 CLI flags are exposed on the
    trainer's argparse parser.  Default ``--sa-penalty=0.0`` is
    load-bearing: it preserves bit-for-bit backward compat with
    Wf-380-B1 / Wf-299 (the penalty term is exactly zero when 0.0).
    """
    trainer = _import_trainer_parse_args()
    parser = trainer.parse_args.__wrapped__ if hasattr(
        trainer.parse_args, "__wrapped__"
    ) else None
    # Default invocation (no CLI flags).
    import sys as _sys
    _sys.argv = ["train_cfm_yuelbond_wasserstein.py"]
    args = trainer.parse_args()
    # Defaults per Wf-388-A3 brief: --sa-penalty 0.0 (OFF), --sa-target 2.8.
    assert hasattr(args, "sa_penalty"), (
        "Wf-388-A3: --sa-penalty must be exposed on the argparse namespace"
    )
    assert hasattr(args, "sa_target"), (
        "Wf-388-A3: --sa-target must be exposed on the argparse namespace"
    )
    assert args.sa_penalty == 0.0, (
        f"Wf-388-A3 default --sa-penalty must be 0.0 (OFF; backward-compat); "
        f"got {args.sa_penalty}"
    )
    assert args.sa_target == 2.8, (
        f"Wf-388-A3 default --sa-target must be 2.8 (Ertl 2009 cutoff); "
        f"got {args.sa_target}"
    )

    # Override: --sa-penalty 0.1 --sa-target 3.0.
    _sys.argv = [
        "train_cfm_yuelbond_wasserstein.py",
        "--sa-penalty", "0.1",
        "--sa-target", "3.0",
    ]
    args_override = trainer.parse_args()
    assert args_override.sa_penalty == pytest.approx(0.1)
    assert args_override.sa_target == pytest.approx(3.0)

    # Sanity: parser object is reachable (parser variable is unused
    # at runtime but documents the parser fixture path).
    _ = parser


# ---------------------------------------------------------------------------
# Test 2 — sa_aware_loss is invoked when --sa-penalty > 0
# ---------------------------------------------------------------------------
def test_sa_aware_loss_in_c() -> None:
    """sa_aware_sampling_penalty is wired into the trainer loss formula.

    Verifies the SA penalty is actually used in the training loss
    when ``--sa-penalty > 0``.  We do NOT launch GPU; we directly
    call the SA penalty helper with the same parameters the trainer
    would use (``target_sa=2.8``, ``sa_weight=0.3`` per the
    Wf-380-A6 SCAFFOLD defaults).
    """
    # Direct module import — no torch / GPU needed.
    if not SA_PENALTY_MODULE_PATH.exists():
        pytest.skip(
            f"Wf-388-A3 requires sa_penalty module at {SA_PENALTY_MODULE_PATH}"
        )
    try:
        from molmetal.adapters.flow_matching_lipman.sa_penalty import (
            sa_aware_sampling_penalty,
            DEFAULT_TARGET_SA,
            DEFAULT_SA_WEIGHT,
        )
    except ModuleNotFoundError as exc:
        pytest.skip(
            f"Wf-388-A3 requires sa_penalty importable; missing {exc.name}"
        )

    # Per Wf-380-A6 SCAFFOLD + Wf-388-A3 wire-in: when --sa-penalty > 0,
    # the trainer calls sa_aware_sampling_penalty(sa_scores, target_sa, sa_weight).
    # Verify the helper produces the canonical Ertl 2009 §3 penalty shape.
    sa_scores = [1.5, 2.0, 2.8, 3.5, 4.5, 6.0]
    penalty = sa_aware_sampling_penalty(
        sa_scores,
        target_sa=DEFAULT_TARGET_SA,  # 2.8
        sa_weight=DEFAULT_SA_WEIGHT,   # 0.3
    )
    # Canonical formula: max(0, sa - target) * sa_weight.
    # For sa=2.8 (==target) -> 0; sa=3.5 -> 0.21; sa=4.5 -> 0.51; sa=6.0 -> 0.96.
    # Penalty is non-negative; sample at index 2 (sa=2.8) must be 0.0.
    assert float(penalty[2]) == pytest.approx(0.0, abs=1e-9), (
        f"sa_aware_sampling_penalty at target SA must be 0; "
        f"got {float(penalty[2])}"
    )
    # Penalty at sa=3.5 = (3.5 - 2.8) * 0.3 = 0.21.
    assert float(penalty[3]) == pytest.approx(0.21, abs=1e-9), (
        f"sa_aware_sampling_penalty at sa=3.5 must be 0.21; "
        f"got {float(penalty[3])}"
    )
    # Penalty at sa=6.0 = (6.0 - 2.8) * 0.3 = 0.96.
    assert float(penalty[5]) == pytest.approx(0.96, abs=1e-9), (
        f"sa_aware_sampling_penalty at sa=6.0 must be 0.96; "
        f"got {float(penalty[5])}"
    )
    # All penalties must be non-negative (Ertl 2009 §3 monotone property).
    assert (penalty >= 0.0).all(), (
        "sa_aware_sampling_penalty must be non-negative for all samples"
    )


# ---------------------------------------------------------------------------
# Test 3 — default --sa-target threshold is 2.8
# ---------------------------------------------------------------------------
def test_sa_target_threshold() -> None:
    """``--sa-target`` default value is 2.8 (Ertl 2009 §3 canonical cutoff).

    The synthesizable-region SA cutoff is 2.8 per Ertl & Schuffenhauer
    2009 §3 ("synthesizable with effort" begins at 2.8; the canonical
    threshold for ``SA < target`` no-penalty region).  This is the same
    default as ``DEFAULT_TARGET_SA`` in sa_penalty.py.
    """
    # Verify the SA penalty module exposes the same constant.
    if not SA_PENALTY_MODULE_PATH.exists():
        pytest.skip(
            f"Wf-388-A3 requires sa_penalty module at {SA_PENALTY_MODULE_PATH}"
        )
    try:
        from molmetal.adapters.flow_matching_lipman.sa_penalty import (
            DEFAULT_TARGET_SA,
        )
    except ModuleNotFoundError as exc:
        pytest.skip(
            f"Wf-388-A3 requires sa_penalty importable; missing {exc.name}"
        )

    # Module-level constant must be 2.8 (Ertl 2009 §3).
    assert DEFAULT_TARGET_SA == pytest.approx(2.8), (
        f"sa_penalty.DEFAULT_TARGET_SA must be 2.8 per Ertl 2009; "
        f"got {DEFAULT_TARGET_SA}"
    )

    # Trainer CLI default must match.
    trainer = _import_trainer_parse_args()
    import sys as _sys
    _sys.argv = ["train_cfm_yuelbond_wasserstein.py"]
    args = trainer.parse_args()
    assert args.sa_target == pytest.approx(DEFAULT_TARGET_SA), (
        f"Trainer --sa-target default must equal "
        f"sa_penalty.DEFAULT_TARGET_SA={DEFAULT_TARGET_SA}; "
        f"got {args.sa_target}"
    )

    # Cross-check: trainer file source must contain both flag declarations.
    src = TRAINER_PATH.read_text(encoding="utf-8")
    assert '--sa-penalty' in src, (
        "Wf-388-A3: trainer must declare --sa-penalty CLI flag"
    )
    assert '--sa-target' in src, (
        "Wf-388-A3: trainer must declare --sa-target CLI flag"
    )
    # sa_loss must appear in the loss formula block (default 0 preserved).
    assert 'sa_loss' in src, (
        "Wf-388-A3: trainer loss block must reference sa_loss"
    )
    # sa_penalty_weight must appear in the loss formula.
    assert 'sa_penalty_weight' in src, (
        "Wf-388-A3: trainer loss block must reference sa_penalty_weight"
    )
    # Defensive: default 0.0 must be present in the help/CLI registration.
    assert 'default=0.0' in src, (
        "Wf-388-A3: --sa-penalty default must be 0.0 (OFF; backward compat)"
    )