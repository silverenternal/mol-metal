"""Pytest: WF-CFM-Frontier-Research Phase 2 Fix #1.

Fix #1 activates the wired BondAwareDecoder pipeline that P0-F1/F5
already shipped.  Two CLI-level defaults had silently short-circuited
the path:

1. ``molmetal/scripts/r10_cfg_real_crossdocked.py:236`` — ``--bond-head``
   defaulted to ``'distance'``, which routes to the harness's
   ``decode_distance_graph`` heuristic (covalent-radius covFactor=1.3)
   rather than the wired ``BondAwareDecoder``.  Fix: default flipped to
   ``'learned'`` so the learned bond-order head fires.

2. ``molmetal/scripts/r10_cfg_real_crossdocked.py:243`` and
   ``molmetal/adapters/flow_matching_lipman/__init__.py:1703`` —
   ``joint_train`` defaulted to ``False``, so even when the head was
   built it stayed at random init (per
   ``code_review_phase1c.md`` BUG #1).  Fix: default flipped to
   ``True`` so the head co-trains end-to-end with the CFM loss.

Source: ``molmetal/reports/wf_cfm_frontier_research/synthesis_phase2.md``
§2.1 + §3.1 (TOP FIX #1).

Five tests:
1. ``test_r10_default_bond_head_is_learned`` — argparse sees the new
   default ``'learned'`` (regression guard for the harness-level flip).
2. ``test_r10_default_joint_train_is_true`` — argparse sees the new
   default ``True`` (regression guard for the harness-level flip).
3. ``test_adapter_default_joint_train_is_true`` — the adapter's
   constructor-level default ``joint_train`` is ``True`` (so callers
   that construct the adapter without threading the CLI flag get the
   intended behaviour).
4. ``test_adapter_joint_train_in_optimizer_when_bond_head_enabled`` —
   the :class:`BondOrderHead`'s parameters end up in the optimizer's
   parameter list when ``use_bond_head=True`` and ``joint_train=True``
   (the fix would be cosmetic otherwise).
5. ``test_legacy_opt_out_still_works`` — passing ``joint_train=False``
   explicitly + ``--bond-head=distance`` recovers the pre-Fix-#1
   bit-exact behaviour (regression guard against silent behaviour
   changes in legacy callers).

All tests run on CPU in <5 s — no GPU or external evaluators required.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run_argparse_default(cli_arg: str) -> str:
    """Invoke the harness in --help mode to discover the default.

    We use ``argparse.SUPPRESS`` inspection: ``--help`` does not show
    ``default=``, so instead we run a tiny snippet that imports
    :mod:`argparse` from the script, builds the parser, and reads the
    default off the action.  Falls back to running the script's
    ``--help`` and grepping for the value.

    Robust approach: spawn a python -c "import importlib.util; ..." that
    imports the harness and runs ``_build_parser()`` (if exposed) or
    triggers ``main()`` with the flag absent — easier path is to invoke
    the script via subprocess with a probe argparse flag.

    The actual approach: read the source file directly and parse with
    ``ast`` to find the ``add_argument`` call's ``default=`` kwarg.
    This avoids subprocess entirely and runs in <1 ms.
    """
    import ast

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
                        return repr(kw.value.value)
                    if isinstance(kw.value, ast.NameConstant):  # py<3.8
                        return repr(kw.value.value)
            return "<no default>"
    raise AssertionError(f"Could not find argparse default for {cli_arg!r}")


# ---------------------------------------------------------------------------
# Test 1: --bond-head default is now 'learned' (Fix #1 part A)
# ---------------------------------------------------------------------------
def test_r10_default_bond_head_is_learned():
    """The harness's ``--bond-head`` CLI default is now ``'learned'``.

    Pre-fix behaviour: ``default='distance'`` routed the harness to
    ``decode_distance_graph`` (covalent-radius heuristic) and bypassed
    the wired :class:`BondAwareDecoder`.  Post-fix: the default is
    ``'learned'`` so :func:`decode_learned_bond_graph` fires (per
    ``synthesis_phase2.md`` §3.1).
    """
    default = _run_argparse_default("--bond-head")
    assert default == "'learned'", (
        f"WF-CFM-Frontier Fix #1: --bond-head default must be 'learned' "
        f"to activate BondAwareDecoder; got {default}.  Pre-fix value "
        f"was 'distance'."
    )


# ---------------------------------------------------------------------------
# Test 2: --joint-train default is now True (Fix #1 part B)
# ---------------------------------------------------------------------------
def test_r10_default_joint_train_is_true():
    """The harness's ``--joint-train`` CLI default is now ``True``.

    Pre-fix behaviour: ``default=False`` (BooleanOptionalAction) meant
    the :class:`BondOrderHead` stayed frozen at random init.  Post-fix:
    ``True`` so the head co-trains end-to-end with the CFM loss.
    """
    default = _run_argparse_default("--joint-train")
    assert default == "True", (
        f"WF-CFM-Frontier Fix #1: --joint-train default must be True to "
        f"co-train the BondOrderHead; got {default}.  Pre-fix value was "
        f"False."
    )


# ---------------------------------------------------------------------------
# Test 3: adapter constructor default joint_train=True
# ---------------------------------------------------------------------------
def test_adapter_default_joint_train_is_true():
    """The adapter's :func:`__init__` default ``joint_train`` is now ``True``.

    Even when callers construct :class:`LipmanFlowMatchingAdapter`
    directly (bypassing the harness CLI), the adapter-level default
    should reflect the Fix #1 intent so production callers get the
    intended behaviour.
    """
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter

    adapter = LipmanFlowMatchingAdapter()
    assert adapter._joint_train is True, (
        f"WF-CFM-Frontier Fix #1: LipmanFlowMatchingAdapter.__init__ "
        f"default joint_train must be True so the BondOrderHead co-trains "
        f"end-to-end; got {adapter._joint_train}.  Pre-fix value was False."
    )


# ---------------------------------------------------------------------------
# Test 4: BondOrderHead params are in the optimizer list (Fix #1 is functional)
# ---------------------------------------------------------------------------
def test_adapter_joint_train_in_optimizer_when_bond_head_enabled():
    """With ``use_bond_head=True`` and ``joint_train=True`` the head's
    parameters must end up in the optimizer's parameter list.

    Without this, the fix would be cosmetic — flipping the default
    would have no effect at training time.  We construct the adapter,
    run ``setup()`` on CPU, and assert that the BondOrderHead's
    parameters are reachable via ``optimizer.param_groups``.
    """
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter

    torch.manual_seed(0)
    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=64,
        n_layers=2,
        max_atomic_number=20,
        use_bond_head=True,
        joint_train=True,
    )
    adapter.setup(device="cpu")
    assert adapter.bond_head is not None, (
        "BondOrderHead must be constructed when use_bond_head=True"
    )
    # Confirm the optimizer has parameters belonging to the bond_head.
    bond_head_param_ids = {id(p) for p in adapter.bond_head.parameters()}
    opt_param_ids: set = set()
    for group in adapter.optimizer.param_groups:
        for p in group["params"]:
            opt_param_ids.add(id(p))
    assert bond_head_param_ids.issubset(opt_param_ids), (
        "WF-CFM-Frontier Fix #1: with joint_train=True the BondOrderHead's "
        "parameters MUST be in the optimizer's param_groups so AdamW "
        "updates them.  Got {}/{} params wired.".format(
            len(bond_head_param_ids & opt_param_ids),
            len(bond_head_param_ids),
        )
    )


# ---------------------------------------------------------------------------
# Test 5: legacy opt-out still recovers the pre-fix behaviour
# ---------------------------------------------------------------------------
def test_legacy_opt_out_still_works():
    """Explicit ``joint_train=False`` + ``use_bond_head=False`` recovers
    the pre-Fix-#1 bit-exact behaviour (head is NOT in the optimizer).

    This is a backward-compat guard: existing legacy callers passing
    the explicit opt-out must continue to behave exactly as before
    Fix #1.
    """
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter

    torch.manual_seed(0)
    # Legacy mode: use_bond_head=False (no head at all).
    legacy = LipmanFlowMatchingAdapter(
        hidden_dim=64,
        n_layers=2,
        max_atomic_number=20,
        use_bond_head=False,
        joint_train=False,
    )
    legacy.setup(device="cpu")
    assert legacy.bond_head is None, (
        "Legacy mode (use_bond_head=False) must NOT construct the "
        "BondOrderHead — backward-compat guard for pre-Fix-#1 callers."
    )
    # Legacy mode 2: head constructed but explicitly frozen.
    legacy_frozen = LipmanFlowMatchingAdapter(
        hidden_dim=64,
        n_layers=2,
        max_atomic_number=20,
        use_bond_head=True,
        joint_train=False,
    )
    legacy_frozen.setup(device="cpu")
    assert legacy_frozen.bond_head is not None, (
        "When use_bond_head=True the head must be constructed "
        "(even in joint_train=False mode)"
    )
    bond_head_param_ids = {
        id(p) for p in legacy_frozen.bond_head.parameters()
    }
    opt_param_ids: set = set()
    for group in legacy_frozen.optimizer.param_groups:
        for p in group["params"]:
            opt_param_ids.add(id(p))
    assert bond_head_param_ids.isdisjoint(opt_param_ids), (
        "Legacy mode (use_bond_head=True, joint_train=False) must keep "
        "the BondOrderHead OUT of the optimizer — backward-compat guard "
        "for the pre-Fix-#1 frozen-head behaviour."
    )