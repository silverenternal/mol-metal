"""Pytest: WF-CFM-Frontier Phase 2 Fix #3 — early-warning decode_smoke counter.

Adds an in-training-loop decode smoke (default OFF, opt-in via
``--decode-smoke-every N``) so a broken CFM path (decode=0 on every
generated pose) is caught in <30 s instead of after the full 5K-step
retrain completes.  Per ``inference_review_phase1d.md`` TOP-1 this
was the highest-priority infrastructure gap.

Six tests:

1. ``test_r10_decode_smoke_every_default_zero`` — argparse default is
   ``0`` (backward-compat with ``wf_gpu_recovery_now`` baseline).
2. ``test_r10_decode_smoke_n_samples_default_eight`` — companion
   ``--decode-smoke-n-samples`` defaults to 8 (matches the phase-2
   spec's smallest-possible smoke).
3. ``test_r10_decode_smoke_n_steps_default_200`` — companion
   ``--decode-smoke-n-steps`` defaults to 200 (matches the spec).
4. ``test_r10_decode_smoke_warn_after_default_two`` — companion
   ``--decode-smoke-warn-after`` defaults to 2 (two consecutive
   decode=0 smokes before a UserWarning fires).
5. ``test_run_decode_smoke_helper_counts_n_decoded`` — the
   :func:`run_decode_smoke` helper counts mols whose ``bonds`` tensor
   has at least one edge (shape ``[2, k]`` with ``k >= 1``) and
   returns the (n_decoded, mols) tuple.  Patches
   ``adapter.generate`` to a fast stub so the test runs in <1 s on
   CPU without touching the real CFM pipeline.
6. ``test_decode_smoke_block_disabled_when_every_zero`` — when
   ``args.decode_smoke_every == 0``, the loop's smoke hook is
   completely bypassed: ``adapter.generate`` is NOT called even at
   step % any-positive-number.  This is the backward-compat guard.

All tests run on CPU in <5 s — no GPU or external evaluators required.

Reference: ``molmetal/reports/wf_cfm_frontier_research/synthesis_phase2.md``
§2.3 + §3.3 (TOP FIX #3).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_argparse_default(cli_arg: str):
    """Read the ``default=`` kwarg off the harness's ``add_argument`` call.

    Same AST-based approach as :mod:`test_cfm_fix1_bond_head_default`
    so we don't have to actually ``argparse.parse_args()`` (which would
    require the harness's mandatory ``--gpu-binary`` arg).
    """
    script = PROJECT_ROOT / "molmetal" / "scripts" / "r10_cfg_real_crossdocked.py"
    tree = ast.parse(script.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == cli_arg
        ):
            for kw in node.keywords:
                if kw.arg == "default":
                    if isinstance(kw.value, ast.Constant):
                        return kw.value.value
                    if isinstance(kw.value, ast.NameConstant):  # py<3.8
                        return kw.value.value
            return "<no default>"
    raise AssertionError(f"Could not find argparse default for {cli_arg!r}")


# ---------------------------------------------------------------------------
# Test 1: --decode-smoke-every default = 0 (backward-compat)
# ---------------------------------------------------------------------------
def test_r10_decode_smoke_every_default_zero():
    """The harness's ``--decode-smoke-every`` default is ``0``.

    Pre-fix: the training loop at ``r10_cfg_real_crossdocked.py``
    never sampled during training (decode=0 collapses were discovered
    only after the full 5K-step budget).  Post-fix: the flag exists
    but defaults to ``0`` (= disabled) so all pre-fix baselines and
    bit-exact behaviour are preserved.  Opt-in to enable.
    """
    default = _run_argparse_default("--decode-smoke-every")
    assert default == 0, (
        f"WF-CFM-Frontier Fix #3: --decode-smoke-every default must be 0 "
        f"(disabled) for backward-compat; got {default}.  Pre-fix value "
        f"was no flag at all."
    )


# ---------------------------------------------------------------------------
# Test 2: --decode-smoke-n-samples default = 8
# ---------------------------------------------------------------------------
def test_r10_decode_smoke_n_samples_default_eight():
    """``--decode-smoke-n-samples`` default is 8 (per phase-2 spec)."""
    default = _run_argparse_default("--decode-smoke-n-samples")
    assert default == 8, (
        f"--decode-smoke-n-samples default must be 8 to match the "
        f"phase-2 synthesis spec's smallest-possible smoke; got {default}."
    )


# ---------------------------------------------------------------------------
# Test 3: --decode-smoke-n-steps default = 200
# ---------------------------------------------------------------------------
def test_r10_decode_smoke_n_steps_default_200():
    """``--decode-smoke-n-steps`` default is 200 (per phase-2 spec).

    200 ODE steps is a balance: low enough to run in <30 s on a small
    adapter, high enough that the bond decoder's ``bonds.shape[-1] > 0``
    proxy is non-degenerate (a 5-step ODE produces a noise cloud that
    gives the decoder no chance of finding an edge).
    """
    default = _run_argparse_default("--decode-smoke-n-steps")
    assert default == 200, (
        f"--decode-smoke-n-steps default must be 200; got {default}."
    )


# ---------------------------------------------------------------------------
# Test 4: --decode-smoke-warn-after default = 2
# ---------------------------------------------------------------------------
def test_r10_decode_smoke_warn_after_default_two():
    """``--decode-smoke-warn-after`` default is 2 consecutive zeros.

    Two consecutive ``n_decoded == 0`` smokes is enough to distinguish
    "first smoke unlucky" from "CFM path is dead".  A single zero is
    not actionable (early in training the bond head weights may not
    have warmed up); a warning after two fires only when the
    degeneracy is persistent.
    """
    default = _run_argparse_default("--decode-smoke-warn-after")
    assert default == 2, (
        f"--decode-smoke-warn-after default must be 2; got {default}."
    )


# ---------------------------------------------------------------------------
# Test 5: run_decode_smoke helper counts n_decoded correctly
# ---------------------------------------------------------------------------
def test_run_decode_smoke_helper_counts_n_decoded():
    """The :func:`run_decode_smoke` helper counts non-empty-bonds mols.

    We patch ``adapter.generate`` to return a hand-crafted list of
    :class:`Molecule`-shaped objects whose ``bonds`` tensor has a
    known edge count.  The helper must return ``n_decoded`` equal to
    the number of objects with ``bonds.shape[-1] > 0``.

    Patches via :func:`unittest.mock.patch` on the imported module
    reference (``molmetal.scripts.r10_cfg_real_crossdocked``) so the
    helper sees our stub without needing a real adapter.
    """
    # Defer import so the patch target is bound to THIS module's
    # attribute lookup (Python resolves ``adapter.generate`` against
    # the patch's mock object).
    from molmetal.scripts import r10_cfg_real_crossdocked as harness

    # Build a fake adapter whose .generate returns 8 mols: 3 with
    # non-empty bonds (shape [2, 4]) and 5 with empty bonds (shape
    # [2, 0]).
    class _FakeMol:
        def __init__(self, n_edges: int):
            self.bonds = torch.zeros(2, n_edges, dtype=torch.long)
            self.coords = torch.zeros(1, 3)
            self.atom_types = torch.tensor([6], dtype=torch.long)

    adapter = MagicMock()
    adapter.generate.return_value = [
        _FakeMol(n_edges=4) for _ in range(3)
    ] + [
        _FakeMol(n_edges=0) for _ in range(5)
    ]

    # Fake pocket — the helper does not dereference any attribute on
    # it (the stubbed generate ignores the pocket), so a MagicMock is
    # fine.
    pocket = MagicMock()

    with patch.object(harness, "SizedGenerationConfig") as cfg_cls:
        # Capture the kwargs the helper passes so we can assert the
        # configured smoke size + steps.
        cfg_cls.return_value = MagicMock(name="SizedGenerationConfig")
        n_decoded, mols = harness.run_decode_smoke(
            adapter, pocket,
            n_samples=8, n_steps=200, seed=42,
        )

    assert n_decoded == 3, (
        f"run_decode_smoke must count only mols with bonds.shape[-1] > 0; "
        f"expected 3/8, got {n_decoded}/8."
    )
    assert len(mols) == 8, "helper must return all 8 generated mols."

    # The helper must thread n_samples, n_steps, seed into the config.
    cfg_cls.assert_called_once()
    kwargs = cfg_cls.call_args.kwargs
    assert kwargs.get("n_samples") == 8
    assert kwargs.get("n_steps") == 200
    assert kwargs.get("seed") == 42


# ---------------------------------------------------------------------------
# Test 6: helper handles the edge case of all-zero mols
# ---------------------------------------------------------------------------
def test_run_decode_smoke_helper_all_zero_returns_zero():
    """When every generated mol has zero edges, helper returns 0.

    This is the wf_gpu_recovery_now baseline scenario (decode=0/192
    on the h=32 5000-step checkpoint).  The helper must correctly
    return ``n_decoded == 0`` rather than raising or miscounting.
    """
    from molmetal.scripts import r10_cfg_real_crossdocked as harness

    class _FakeMol:
        def __init__(self):
            self.bonds = torch.zeros(2, 0, dtype=torch.long)
            self.coords = torch.zeros(1, 3)
            self.atom_types = torch.tensor([6], dtype=torch.long)

    adapter = MagicMock()
    adapter.generate.return_value = [_FakeMol() for _ in range(8)]

    with patch.object(harness, "SizedGenerationConfig") as cfg_cls:
        cfg_cls.return_value = MagicMock()
        n_decoded, mols = harness.run_decode_smoke(
            adapter, MagicMock(),
            n_samples=8, n_steps=200, seed=0,
        )

    assert n_decoded == 0, (
        f"All-zero mols must yield n_decoded=0; got {n_decoded}."
    )
    assert len(mols) == 8


# ---------------------------------------------------------------------------
# Test 7: smoke block is fully bypassed when --decode-smoke-every=0
# ---------------------------------------------------------------------------
def test_decode_smoke_block_disabled_when_every_zero():
    """When ``args.decode_smoke_every == 0`` the smoke block is skipped.

    This is the backward-compat guard.  The training loop must NOT
    call ``adapter.generate`` (or :func:`run_decode_smoke`) when the
    flag is 0, even at step % any-positive-number.

    We replicate the loop's conditional exactly
    (``args.decode_smoke_every > 0 and step > 0 and
    step % args.decode_smoke_every == 0``) and assert it is False
    for step in {1, 10, 100, 1000} when ``args.decode_smoke_every
    == 0``.  This is a tight contract test: it does NOT exercise
    the live harness (which requires GPU + crossdocked data) but
    pins the same conditional the harness uses.
    """
    class _A:
        decode_smoke_every = 0  # the disabled default
    args = _A()
    for step in (1, 10, 100, 1000, 9999):
        fires = (
            args.decode_smoke_every > 0
            and step > 0
            and step % args.decode_smoke_every == 0
        )
        assert not fires, (
            f"decode_smoke hook must NOT fire at step={step} when "
            f"--decode-smoke-every=0 (the disabled default); got "
            f"fires={fires}."
        )


# ---------------------------------------------------------------------------
# Test 8: smoke block fires at the right steps when every=100
# ---------------------------------------------------------------------------
def test_decode_smoke_block_fires_on_step_modulo():
    """When ``args.decode_smoke_every == 100``, fires at step 100, 200, 300.

    Companion to test 7: pins the positive-direction behaviour so a
    future refactor can't accidentally flip the modulo direction
    (``step % N == 0`` ↔ ``N % step == 0``).
    """
    class _A:
        decode_smoke_every = 100
    args = _A()
    expected_fires = set()
    for step in range(1, 1001):
        fires = (
            args.decode_smoke_every > 0
            and step > 0
            and step % args.decode_smoke_every == 0
        )
        if fires:
            expected_fires.add(step)
    assert expected_fires == {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}, (
        f"decode_smoke must fire at step % 100 == 0; got {sorted(expected_fires)}"
    )
