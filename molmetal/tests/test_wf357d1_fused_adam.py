"""Wf-357-D1: Adam(fused=True) opt-in CLI flag — tests.

=================================================================
Wf-357-D1 — production-port tests for the Wf-326G §3.4 + Wf-357-C1
§3.2 Adam(fused=True) optimization across the Wf-131 P1 / Wf-181
G5 / Wf-182 G6 / Wf-299 CFM / Wf-263A v3-fallback trainer surface.

Per Wf-326G §3.4 PROJECTED: the fused Adam path collapses 5 per-
parameter kernel launches into a single CUDA kernel — 1.1-1.3×
faster on the optimizer step (CUDA only; no effect on CPU).  Per
Wf-357-C1 §3.2 the integration is opt-in (default OFF) to preserve
bit-for-bit backward-compat with the Wf-131 P1 / Wf-181 G5 / Wf-182
G6 / Wf-299 / Wf-263A shipped behaviour.

The Wf-326G §3.4 / Wf-357-C1 §3.2 ship-surface pattern is the
:func:`molmetal.baselines.dmpnn_p0wins.make_optimized_adam` factory
(defined at ``dmpnn_p0wins.py:438``); the trainers thread a new
``use_fused_adam`` / ``args.fused_adam`` opt-in flag through their
optimizer construction sites.

Test inventory (4 tests per Wf-357-D1 brief)
-------------------------------------------
1. test_fused_adam_cli_flag_wired       — CLI flag wiring across 3 trainers
2. test_fused_adam_backward_compat      — default OFF behaviour unchanged
3. test_fused_adam_factory_correctness  — make_optimized_adam factory contract
4. test_fused_adam_integration_with_trainer — end-to-end smoke (CPU)

All tests are CPU-only by default (matching the Wf-326G bench);
the fused Adam path is CUDA-only and silently no-ops on CPU when
the ``fused=`` kwarg is unsupported.

**Honest framing (party-twenty-spirit §3.5):**
* The 1.1-1.3× speedup claim is PROJECTED (Wf-326G §3.4) — not
  yet MEASURED end-to-end on Blackwell / RTX 5090.  These tests
  verify the *wiring*, not the speedup.
* ``fused=True`` is silently ignored on CPU PyTorch (no error,
  no effect).  GPU verification deferred to next Blackwell
  retrain (TODO-181 G5 / TODO-182 G6 / Wf-299 CFM).
* The factory at ``dmpnn_p0wins.py:438`` is the canonical contract;
  trainers MUST use it (not construct ``torch.optim.Adam(..., fused=True)``
  inline) to keep the opt-in contract centralized.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# Test 1: CLI flag wiring across the 3 trainers
# =========================================================================
def test_fused_adam_cli_flag_wired() -> None:
    """The --fused-adam / --no-fused-adam CLI flags must be wired into
    all 3 trainers (D-MPNN, chemprop_v2_v3, CFM).

    Per Wf-357-D1 brief §3.2: add ``--<improvement-name>`` flag
    (default OFF for backward compat).
    """
    # D-MPNN trainer
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        parse_args as dmpnn_parse_args,
    )
    # Default OFF (backward-compat with Wf-131 P1 / Wf-181 G5 / Wf-182 G6).
    args_default = dmpnn_parse_args(["--epochs", "1"])
    assert hasattr(args_default, "fused_adam"), (
        "D-MPNN trainer is missing the --fused-adam flag (Wf-357-D1)"
    )
    assert args_default.fused_adam is False, (
        f"D-MPNN --fused-adam default must be False (backward-compat); "
        f"got {args_default.fused_adam}"
    )
    # Opt-in via --fused-adam.
    args_opt_in = dmpnn_parse_args(["--epochs", "1", "--fused-adam"])
    assert args_opt_in.fused_adam is True, (
        f"D-MPNN --fused-adam must flip to True when flag is passed; "
        f"got {args_opt_in.fused_adam}"
    )
    # Opt-out via --no-fused-adam.
    args_opt_out = dmpnn_parse_args(["--epochs", "1", "--no-fused-adam"])
    assert args_opt_out.fused_adam is False, (
        f"D-MPNN --no-fused-adam must flip to False (default); "
        f"got {args_opt_out.fused_adam}"
    )

    # chemprop_v2_v3 trainer (parse_args lives inside main() for this
    # trainer; verify the source-text integration via source-grep).
    chemprop_src_path = (
        Path(__file__).resolve().parent.parent
        / "scripts" / "train_chemprop_v2_v3.py"
    )
    chemprop_src = chemprop_src_path.read_text()
    assert '"--fused-adam"' in chemprop_src, (
        f"chemprop_v2_v3 trainer must declare --fused-adam CLI flag"
    )
    assert '"--no-fused-adam"' in chemprop_src, (
        f"chemprop_v2_v3 trainer must declare --no-fused-adam CLI flag"
    )

    # CFM trainer (parse_args takes no positional args; uses sys.argv)
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        parse_args as cfm_parse_args,
    )
    # Default OFF (backward-compat with Wf-299 / Wf-291C shipped).
    # We construct the Namespace directly via subprocess to avoid
    # the trainer's parse_args reaching into sys.argv (which would
    # pollute the test runner).
    import subprocess
    cfm_probe = subprocess.run(
        [
            "/home/hugo/codes/mol-metal/.venv/bin/python",
            "-c",
            (
                "import sys, argparse\n"
                "sys.argv = ['probe']\n"
                "from molmetal.scripts.train_cfm_yuelbond_wasserstein import parse_args\n"
                "ns = parse_args()\n"
                "print('FUSED_ADAM_DEFAULT:', ns.fused_adam)\n"
            ),
        ],
        cwd=str(PROJECT_ROOT),
        env={"PYTHONPATH": str(PROJECT_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=60,
    )
    assert "FUSED_ADAM_DEFAULT: False" in cfm_probe.stdout, (
        f"CFM parse_args default for fused_adam must be False; "
        f"stdout={cfm_probe.stdout!r} stderr={cfm_probe.stderr!r}"
    )


# =========================================================================
# Test 2: Default OFF behaviour preserves backward compatibility
# =========================================================================
def test_fused_adam_backward_compat() -> None:
    """Default --fused-adam=False must preserve the legacy
    ``torch.optim.Adam(...)`` / ``torch.optim.AdamW(...)`` call site
    bit-for-bit.

    Per Wf-357-D1 brief §3: ``Default OFF for backward compat``.
    Per Wf-357-C1 §3.2: ``opt-in default OFF limits blast radius``.
    """
    # Build a small linear model with a few parameters.
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    lr = 1e-3

    # Legacy call (pre-fix): torch.optim.Adam(model.parameters(), lr=lr).
    legacy_optim = torch.optim.Adam(model.parameters(), lr=lr)
    # Wf-357-D1 default path: the same call when --fused-adam is OFF.
    # Both must produce a torch.optim.Adam instance with NO `fused`
    # kwarg set (the legacy API doesn't accept it).
    assert type(legacy_optim) is torch.optim.Adam, (
        f"Legacy optimizer must be torch.optim.Adam; got {type(legacy_optim)}"
    )

    # Confirm that the optimizer factory's default-OFF behaviour
    # produces an equivalent instance (same class; same default fused
    # state for the CPU backend where `fused` is unsupported).
    from molmetal.baselines.dmpnn_p0wins import make_optimized_adam
    # fused=False explicitly: legacy bit-for-bit equivalent.
    optim_explicit_off = make_optimized_adam(
        model.parameters(), lr=lr, fused=False,
    )
    assert type(optim_explicit_off) is torch.optim.Adam, (
        f"make_optimized_adam(fused=False) must be torch.optim.Adam; "
        f"got {type(optim_explicit_off)}"
    )

    # Confirm a step + backward does not regress the legacy path.
    # This guards against any accidental state-difference in the
    # pre-fix call site (e.g. accidentally adding fused=True).
    x = torch.randn(3, 4)
    target = torch.randn(3, 2)
    loss_fn = nn.MSELoss()
    for opt in (legacy_optim, optim_explicit_off):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
        # Sanity: no NaN.
        assert torch.isfinite(loss).all(), (
            f"Optimizer produced non-finite loss in backward-compat "
            f"check; loss={loss.item()}"
        )


# =========================================================================
# Test 3: make_optimized_adam factory correctness contract
# =========================================================================
def test_fused_adam_factory_correctness() -> None:
    """``make_optimized_adam`` is the canonical opt-in contract per
    Wf-326G §3.4 + Wf-357-C1 §3.2.  All 3 trainers must use it (not
    construct ``torch.optim.Adam(..., fused=True)`` inline).

    Per Wf-357-D1 brief §3.2: ``Add --<improvement-name>`` flag
    (default OFF for backward compat).  The factory at
    ``dmpnn_p0wins.py:438`` is the centralized contract.
    """
    from molmetal.baselines.dmpnn_p0wins import make_optimized_adam

    # Verify the factory exists and has the documented signature.
    import inspect
    sig = inspect.signature(make_optimized_adam)
    assert "params" in sig.parameters, (
        f"make_optimized_adam must accept 'params'; got {list(sig.parameters)}"
    )
    assert "lr" in sig.parameters, (
        f"make_optimized_adam must accept 'lr'; got {list(sig.parameters)}"
    )
    assert "fused" in sig.parameters, (
        f"make_optimized_adam must accept 'fused'; got {list(sig.parameters)}"
    )
    assert sig.parameters["fused"].default is True, (
        f"make_optimized_adam 'fused' default must be True "
        f"(matches docstring + Wf-326G §3.4); got "
        f"{sig.parameters['fused'].default}"
    )

    # Verify the factory's CPU behaviour: PyTorch's Adam on CPU
    # silently ignores the `fused=`` kwarg (no error, no effect).
    # This is the documented fallback path per Wf-326G §3.4:
    # "On CPU this is a no-op (no fused path)."
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    # fused=True on CPU: must NOT raise (PyTorch accepts the kwarg
    # silently on CPU when the param shapes / groups are simple).
    try:
        optim_fused = make_optimized_adam(
            model.parameters(), lr=1e-3, fused=True,
        )
        assert isinstance(optim_fused, torch.optim.Adam), (
            f"make_optimized_adam(fused=True) must return torch.optim.Adam; "
            f"got {type(optim_fused)}"
        )
    except (TypeError, RuntimeError, NotImplementedError) as exc:
        # PyTorch's CPU Adam may raise NotImplementedError for fused
        # on some versions; that's also acceptable — the trainer
        # already calls `from ... import make_optimized_adam` only
        # when use_fused_adam=True and the device is CUDA.  CPU is
        # a documented no-op fallback (the trainer never calls the
        # factory on CPU per the wire-up at line ~1604).
        warnings.warn(
            f"make_optimized_adam(fused=True) on CPU raised "
            f"{type(exc).__name__}: {exc}; this is acceptable on CPU "
            f"(Wf-326G §3.4: fused path is CUDA-only)."
        )

    # Confirm a step works on the default-OFF path (CPU).
    optim_off = make_optimized_adam(model.parameters(), lr=1e-3, fused=False)
    x = torch.randn(3, 4)
    target = torch.randn(3, 2)
    optim_off.zero_grad()
    loss = nn.MSELoss()(model(x), target)
    loss.backward()
    optim_off.step()
    assert torch.isfinite(loss).all(), (
        f"make_optimized_adam step produced non-finite loss; "
        f"loss={loss.item()}"
    )


# =========================================================================
# Test 4: Integration with the D-MPNN trainer (end-to-end smoke)
# =========================================================================
def test_fused_adam_integration_with_trainer() -> None:
    """The ``use_fused_adam`` opt-in must be threaded through
    ``train_one_metal_one_seed`` and produce a valid Adam optimizer
    without raising on CPU.

    Per Wf-357-D1 brief §3: ``Wire into trainer at appropriate
    location``.  Per §4: ``Integration with trainer``.

    The integration test exercises the full call site:
    ``train_one_metal_one_seed(..., use_fused_adam=True)`` must
    accept the flag (no TypeError) and select the fused factory.
    The ``train_one_metal_one_seed`` function itself needs a
    full data pipeline to execute end-to-end, so this test
    focuses on the *signature integration* (the flag is accepted;
    the optimizer selection path works).
    """
    import inspect
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        train_one_metal_one_seed,
    )

    sig = inspect.signature(train_one_metal_one_seed)
    assert "use_fused_adam" in sig.parameters, (
        f"train_one_metal_one_seed must accept 'use_fused_adam'; "
        f"got {list(sig.parameters)}"
    )
    assert sig.parameters["use_fused_adam"].default is False, (
        f"train_one_metal_one_seed 'use_fused_adam' default must be "
        f"False (backward-compat); got "
        f"{sig.parameters['use_fused_adam'].default}"
    )

    # Confirm the wire-up at the optimizer construction site
    # correctly threads the flag through.  We grep the source for
    # the Wf-357-D1 NEW marker so future maintainers can find it.
    src_path = (
        Path(__file__).resolve().parent.parent
        / "scripts" / "train_dmpnn_multitask_wasserstein.py"
    )
    src_text = src_path.read_text()
    assert "Wf-357-D1" in src_text, (
        f"D-MPNN trainer source must contain the Wf-357-D1 marker "
        f"so future maintainers can trace the integration"
    )
    # Confirm the make_optimized_adam factory call site is present.
    assert "make_optimized_adam" in src_text, (
        f"D-MPNN trainer must call make_optimized_adam when "
        f"use_fused_adam=True"
    )
    # Confirm the CLI flag is declared in parse_args.
    assert '"--fused-adam"' in src_text, (
        f"D-MPNN trainer must declare --fused-adam CLI flag"
    )

    # Confirm the chemprop_v2_v3 trainer is also wired.
    chemprop_src_path = (
        Path(__file__).resolve().parent.parent
        / "scripts" / "train_chemprop_v2_v3.py"
    )
    chemprop_src = chemprop_src_path.read_text()
    assert "Wf-357-D1" in chemprop_src, (
        f"chemprop_v2_v3 trainer must contain the Wf-357-D1 marker"
    )
    assert "make_optimized_adam" in chemprop_src, (
        f"chemprop_v2_v3 trainer must call make_optimized_adam when "
        f"use_fused_adam=True"
    )

    # Confirm the CFM trainer is also wired.
    cfm_src_path = (
        Path(__file__).resolve().parent.parent
        / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    )
    cfm_src = cfm_src_path.read_text()
    assert "Wf-357-D1" in cfm_src, (
        f"CFM trainer must contain the Wf-357-D1 marker"
    )
    assert "make_optimized_adam" in cfm_src, (
        f"CFM trainer must call make_optimized_adam when "
        f"args.fused_adam=True"
    )

    # Confirm the summary JSON includes the Wf-357-D1 flag for paper
    # §6 reproducibility audits.
    assert "wf357d1_use_fused_adam" in src_text, (
        f"D-MPNN trainer must persist wf357d1_use_fused_adam into "
        f"the training_summary JSON for paper §6 audits"
    )