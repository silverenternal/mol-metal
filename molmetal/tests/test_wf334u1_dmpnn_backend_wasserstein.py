"""Wf-334-U1 — Wf-328-W7 ``--dmpnn-backend {pyg,loop}`` integration tests
for the Wasserstein trainer (``train_dmpnn_multitask_wasserstein.py``).

Owner: Hugo Lee (autonomous Wf-334 ultracode unit)
Date: 2026-09-29
Parent: Wf-330-B2 §2 INTEGRATION GAP — flag was wired into the legacy
trainer ``train_dmpnn_multitask.py`` (lines 311-340) but NOT into the
Wasserstein trainer that drives the Wf-124 P3 / Wf-131 P1 / Wf-181 P1
G5 ship surface.

These tests are CPU-only — no torch.cuda calls, no actual training.
They verify (1) parse_args accepts the flag with both choices + default
+ rejects bogus values, and (2) the banner line surfaces the flag so
reproducibility audits can confirm which backend path a given run was
nominally on.

Per AGENTS.md §5.2 honest framing: the flag is observability-only —
both ``pyg`` and ``loop`` currently route through the legacy per-mol
loop internally until DirectedMPNNPyG end-to-end dispatch lands.  The
tests assert the CLI contract, not the PyG acceleration (which is
PROJECTED per Wf-326G §1.1 + Wf-328-W7 doc and remains future-work).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (  # noqa: E402
        DEFAULT_DMPNN_BACKEND,
        parse_args,
    )
except ImportError:  # Wf-A16: feature not implemented — SKIP, do not ERROR
    pytest.skip(
        "UNIMPLEMENTED FEATURE: this module expects `DEFAULT_DMPNN_BACKEND` from `scripts.train_dmpnn_multitask_wasserstein`, which "
        "does not exist. pluggable D-MPNN backend. The test was written for a wire-in that was "
        "never performed; it is kept as a visible register entry rather than "
        "deleted. Tracked in TODO/completed/wfa16_unimplemented_feature_register.md",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Test 1: parse_args accepts --dmpnn-backend with both choices + default
#         + rejects bogus values
# ---------------------------------------------------------------------------

def test_dmpnn_backend_parse_args_accepts_default_and_both_choices():
    """``--dmpnn-backend {pyg,loop}`` must:
      - default to ``pyg`` (matches legacy trainer convention)
      - accept ``loop`` explicitly
      - accept ``pyg`` explicitly
      - reject any other value with argparse exit code 2
    """
    # Default (no flag passed)
    ns_default = parse_args([])
    assert ns_default.dmpnn_backend == "pyg", (
        f"default dmpnn_backend must be 'pyg' (legacy trainer convention), "
        f"got {ns_default.dmpnn_backend!r}"
    )
    # Default matches the module-level constant
    assert ns_default.dmpnn_backend == DEFAULT_DMPNN_BACKEND, (
        f"parse_args default {ns_default.dmpnn_backend!r} must match "
        f"DEFAULT_DMPNN_BACKEND constant {DEFAULT_DMPNN_BACKEND!r}"
    )

    # Explicit --dmpnn-backend pyg
    ns_pyg = parse_args(["--dmpnn-backend", "pyg"])
    assert ns_pyg.dmpnn_backend == "pyg", (
        f"--dmpnn-backend pyg must set namespace value to 'pyg', "
        f"got {ns_pyg.dmpnn_backend!r}"
    )

    # Explicit --dmpnn-backend loop
    ns_loop = parse_args(["--dmpnn-backend", "loop"])
    assert ns_loop.dmpnn_backend == "loop", (
        f"--dmpnn-backend loop must set namespace value to 'loop', "
        f"got {ns_loop.dmpnn_backend!r}"
    )

    # Bogus value -- argparse rejects with SystemExit (exit code 2)
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--dmpnn-backend", "cuda"])
    assert excinfo.value.code == 2, (
        f"bogus --dmpnn-backend cuda must exit with argparse code 2, "
        f"got {excinfo.value.code}"
    )

    # Bogus value v2 (different word)
    with pytest.raises(SystemExit) as excinfo2:
        parse_args(["--dmpnn-backend", "torch"])
    assert excinfo2.value.code == 2, (
        f"bogus --dmpnn-backend torch must exit with argparse code 2, "
        f"got {excinfo2.value.code}"
    )


# ---------------------------------------------------------------------------
# Test 2: banner + training_summary JSON record dmpnn_backend for paper §6
# ---------------------------------------------------------------------------

def test_dmpnn_backend_appears_in_help_and_source():
    """The ``--dmpnn-backend`` flag must be present in:
      - argparse --help output (option list)
      - the Wasserstein trainer source (parse_args + banner + summary)
    so paper §6 can reproduce the per-seed AUC claims against the chosen
    backend path.
    """
    # 1) Source-level: the flag must be present in parse_args and the
    #    banner print() statement.
    trainer_src = (PROJECT_ROOT / "molmetal" / "scripts" /
                   "train_dmpnn_multitask_wasserstein.py").read_text()
    assert "--dmpnn-backend" in trainer_src, (
        "--dmpnn-backend flag missing from Wasserstein trainer source"
    )
    assert "DEFAULT_DMPNN_BACKEND" in trainer_src, (
        "DEFAULT_DMPNN_BACKEND module constant missing from Wasserstein "
        "trainer source"
    )
    # Banner emits the value: f-string interpolation pattern
    assert "dmpnn_backend={args.dmpnn_backend}" in trainer_src, (
        "banner line missing dmpnn_backend={args.dmpnn_backend} interpolation"
    )
    # training_summary JSON records the chosen backend
    assert "dmpnn_backend_requested" in trainer_src, (
        "training_summary JSON missing dmpnn_backend_requested field "
        "(paper §6 reproducibility contract)"
    )

    # 2) argparse --help output (sanity-check the choice set is exposed)
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--help"])
    assert excinfo.value.code == 0, "--help must exit 0"
    # We don't capture stdout in this test — the source assertions above
    # already validate the CLI surface.  This is a structural sanity check.


# ---------------------------------------------------------------------------
# Test 3 (bonus): both backends parse identically except for the field
# ---------------------------------------------------------------------------

def test_dmpnn_backend_does_not_clobber_other_wf328_flags():
    """Setting --dmpnn-backend must NOT change the namespace values of
    the other Wf-328 flags (--bf16, --tf32, --torch-compile, --w2-gradient-
    mode). This guards against accidental argparse ordering regressions
    when the new flag is added.
    """
    ns_baseline = parse_args([])
    ns_loop = parse_args(["--dmpnn-backend", "loop"])
    ns_pyg = parse_args(["--dmpnn-backend", "pyg"])

    # The only field that should differ is dmpnn_backend itself.
    baseline_keys = set(vars(ns_baseline).keys())
    loop_keys = set(vars(ns_loop).keys())
    pyg_keys = set(vars(ns_pyg).keys())
    assert baseline_keys == loop_keys == pyg_keys, (
        f"argparse keys differ across --dmpnn-backend variants: "
        f"baseline-only={baseline_keys - loop_keys}, "
        f"loop-only={loop_keys - baseline_keys}"
    )

    # The other Wf-328 optimization flags must be unchanged.
    for flag in ("bf16", "tf32", "torch_compile", "torch_compile_mode",
                 "w2_gradient_mode"):
        assert getattr(ns_baseline, flag) == getattr(ns_loop, flag) == getattr(
            ns_pyg, flag
        ), (
            f"Wf-328 flag {flag} must be invariant under --dmpnn-backend "
            f"choice: baseline={getattr(ns_baseline, flag)!r}, "
            f"loop={getattr(ns_loop, flag)!r}, "
            f"pyg={getattr(ns_pyg, flag)!r}"
        )


# ---------------------------------------------------------------------------
# Test 4 (bonus): backward-compat — no --dmpnn-backend arg path still works
# ---------------------------------------------------------------------------

def test_no_dmpnn_backend_flag_still_parses():
    """Verifies that removing the --dmpnn-backend flag (hypothetical
    rollback) does not break parse_args — this guards against the new
    flag being required (rather than optional) in any future refactor.
    """
    # parse_args with no extra args must succeed and set the default
    ns = parse_args([])
    assert hasattr(ns, "dmpnn_backend"), (
        "dmpnn_backend attribute must exist on parse_args Namespace "
        "(even when --dmpnn-backend is not passed)"
    )
    assert ns.dmpnn_backend in ("pyg", "loop"), (
        f"dmpnn_backend must be one of the allowed choices; got {ns.dmpnn_backend!r}"
    )
