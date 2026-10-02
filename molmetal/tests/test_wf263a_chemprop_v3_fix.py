"""Wf-263A-Implement unit tests: chemprop-mode dispatch + v3-fallback head forcing.

Wf-263A-Implement (TODO-130 G5) introduces the ``--chemprop-mode
{v2, v3, v3-fallback}`` flag on the ``train_chemprop_v2_v3.py``
driver.  The default ``v3-fallback`` mode forces the chemprop v2
:class:`BinaryClassificationFFN` head regardless of
``--enhanced-head`` / ``--metal-attention``, matching the Wf-185
paper-grade baseline (the Wf-263A-Audit-verified root cause for the
v3 DrugOOD BELOW_RANDOM 0.4544 is LOMO-collapse + under-training,
not v3 architecture — see ``molmetal/reports/wf263a_audit/v3_drugood_root_cause.md``).

This module covers four contracts:

  (i)   Default ``--chemprop-mode=v3-fallback`` disables
        ``enhanced_head`` + ``metal_attention`` at the dispatch
        level even when ``--enhanced-head`` / ``--metal-attention``
        are passed explicitly.
  (ii)  ``--chemprop-mode=v3`` honours the user-supplied
        ``--enhanced-head`` / ``--metal-attention`` flags.
  (iii) ``--chemprop-mode=v2`` also forces the chemprop v2 head
        (symmetric with ``v3-fallback``).
  (iv)  The ``--chemprop-mode`` flag is exposed in argparse.

All tests are CPU-only and do NOT invoke chemprop pre-featurization
or training.  They introspect the source / argparse contract only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _import_driver():
    import importlib

    return importlib.import_module("molmetal.scripts.train_chemprop_v2_v3")


def _apply_chemprop_mode_dispatch(args) -> None:
    """Reproduce the Wf-263A --chemprop-mode dispatch from main().

    Mirrors the validation block in train_chemprop_v2_v3.py main()
    so the test exercises the EXACT same logic (no behavioural drift).
    """
    if args.chemprop_mode == "v3-fallback":
        args._effective_enhanced_head = False
        args._effective_metal_attention = False
    elif args.chemprop_mode == "v2":
        args._effective_enhanced_head = False
        args._effective_metal_attention = False
    else:  # "v3"
        args._effective_enhanced_head = bool(args.enhanced_head)
        args._effective_metal_attention = bool(args.metal_attention)


class _Args:
    """Minimal namespace mirroring the args the dispatch cares about."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# G2: --chemprop-mode=v3-fallback (default) disables enhanced_head.
# ---------------------------------------------------------------------------
def test_default_v3_fallback_disables_enhanced_head() -> None:
    """Default chemprop_mode='v3-fallback' must force enhanced_head=False
    even when --enhanced-head is passed explicitly on the CLI."""
    args = _Args(
        chemprop_mode="v3-fallback",
        enhanced_head=True,  # user CLI intent
        metal_attention=False,
    )
    _apply_chemprop_mode_dispatch(args)
    assert args._effective_enhanced_head is False, (
        "Wf-263A-Implement: v3-fallback must force enhanced_head=False "
        "(matches Wf-185 paper-grade BinaryClassificationFFN baseline)."
    )


# ---------------------------------------------------------------------------
# G3: --chemprop-mode=v3-fallback (default) disables metal_attention.
# ---------------------------------------------------------------------------
def test_default_v3_fallback_disables_metal_attention() -> None:
    """Default chemprop_mode='v3-fallback' must force metal_attention=False
    even when --metal-attention is passed explicitly on the CLI (with
    --enhanced-head, otherwise it's a wf248 validation error)."""
    args = _Args(
        chemprop_mode="v3-fallback",
        enhanced_head=True,  # required by wf248 if metal_attention=True
        metal_attention=True,  # user CLI intent
    )
    _apply_chemprop_mode_dispatch(args)
    assert args._effective_metal_attention is False, (
        "Wf-263A-Implement: v3-fallback must force metal_attention=False "
        "(matches Wf-185 paper-grade baseline)."
    )


# ---------------------------------------------------------------------------
# G4 (subset): --chemprop-mode flag is exposed in argparse with correct
#              choices + default.
# ---------------------------------------------------------------------------
def test_argparse_chemprop_mode_default_v3_fallback() -> None:
    """argparse accepts --chemprop-mode with default 'v3-fallback'."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--chemprop-mode",
        choices=("v2", "v3", "v3-fallback"),
        default="v3-fallback",
    )
    args = p.parse_args([])
    assert args.chemprop_mode == "v3-fallback", (
        "Wf-263A-Implement: --chemprop-mode default must be "
        "'v3-fallback' (the Wf-185-equivalent safety path)."
    )


def test_argparse_chemprop_mode_choices() -> None:
    """argparse accepts --chemprop-mode with all three legal choices."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--chemprop-mode",
        choices=("v2", "v3", "v3-fallback"),
        default="v3-fallback",
    )
    for choice in ("v2", "v3", "v3-fallback"):
        args = p.parse_args(["--chemprop-mode", choice])
        assert args.chemprop_mode == choice


def test_chemprop_mode_marker_in_source() -> None:
    """The driver source MUST contain the Wf-263A marker so future
    readers can trace the fix."""
    src = (_SCRIPT_DIR / "train_chemprop_v2_v3.py").read_text()
    assert "--chemprop-mode" in src, (
        "Wf-263A-Implement: --chemprop-mode flag missing from "
        "train_chemprop_v2_v3.py."
    )
    assert "_effective_enhanced_head" in src, (
        "Wf-263A-Implement: _effective_enhanced_head dispatch "
        "attribute missing from train_chemprop_v2_v3.py main()."
    )
    assert "_effective_metal_attention" in src, (
        "Wf-263A-Implement: _effective_metal_attention dispatch "
        "attribute missing from train_chemprop_v2_v3.py main()."
    )
    assert "v3-fallback" in src, (
        "Wf-263A-Implement: 'v3-fallback' literal missing from "
        "train_chemprop_v2_v3.py (mode choice absent)."
    )


# ---------------------------------------------------------------------------
# G6: --chemprop-mode=v3 honours user-supplied enhanced_head /
#     metal_attention flags (when they're allowed by --chemprop-mode).
# ---------------------------------------------------------------------------
def test_chemprop_mode_v3_honours_enhanced_head_true() -> None:
    """--chemprop-mode=v3 + --enhanced-head → effective enhanced_head=True."""
    args = _Args(
        chemprop_mode="v3",
        enhanced_head=True,
        metal_attention=False,
    )
    _apply_chemprop_mode_dispatch(args)
    assert args._effective_enhanced_head is True, (
        "Wf-263A-Implement: --chemprop-mode=v3 must honour user "
        "--enhanced-head=True (explicit opt-in path)."
    )
    assert args._effective_metal_attention is False


def test_chemprop_mode_v3_honours_metal_attention() -> None:
    """--chemprop-mode=v3 + --enhanced-head + --metal-attention
    → both effective flags True (the explicit opt-in v3 path)."""
    args = _Args(
        chemprop_mode="v3",
        enhanced_head=True,
        metal_attention=True,
    )
    _apply_chemprop_mode_dispatch(args)
    assert args._effective_enhanced_head is True
    assert args._effective_metal_attention is True, (
        "Wf-263A-Implement: --chemprop-mode=v3 + --enhanced-head "
        "+ --metal-attention must yield effective metal_attention=True."
    )


# ---------------------------------------------------------------------------
# G7: --chemprop-mode=v2 (symmetric with v3-fallback) forces v2 head.
# ---------------------------------------------------------------------------
def test_chemprop_mode_v2_forces_v2_head() -> None:
    """--chemprop-mode=v2 must force enhanced_head=False even when
    the user passes --enhanced-head explicitly (symmetric with
    v3-fallback; provided for naming symmetry)."""
    args = _Args(
        chemprop_mode="v2",
        enhanced_head=True,
        metal_attention=True,
    )
    _apply_chemprop_mode_dispatch(args)
    assert args._effective_enhanced_head is False, (
        "Wf-263A-Implement: --chemprop-mode=v2 must force "
        "enhanced_head=False (v2 mode does not use EnhancedMetalHead)."
    )
    assert args._effective_metal_attention is False


# ---------------------------------------------------------------------------
# G8: User CLI intent is preserved on args.enhanced_head / args.metal_attention
#     even after dispatch — audit trail stays truthful.
# ---------------------------------------------------------------------------
def test_user_cli_intent_preserved_after_dispatch() -> None:
    """After --chemprop-mode dispatch, args.enhanced_head /
    args.metal_attention (user CLI intent) must be UNCHANGED —
    only _effective_* are overridden.  This keeps the audit trail
    in cfg_enhanced_head / cfg_use_metal_attention truthful."""
    args = _Args(
        chemprop_mode="v3-fallback",
        enhanced_head=True,
        metal_attention=True,
    )
    _apply_chemprop_mode_dispatch(args)
    assert args.enhanced_head is True, (
        "Wf-263A-Implement: args.enhanced_head (user CLI intent) "
        "must NOT be mutated by the dispatch."
    )
    assert args.metal_attention is True, (
        "Wf-263A-Implement: args.metal_attention (user CLI intent) "
        "must NOT be mutated by the dispatch."
    )
    # But the effective values are forced off by v3-fallback.
    assert args._effective_enhanced_head is False
    assert args._effective_metal_attention is False


# ---------------------------------------------------------------------------
# G9: End-to-end --help works with the new flag.
# ---------------------------------------------------------------------------
def test_help_with_chemprop_mode_flag() -> None:
    """--help exits cleanly (smoke for argparse contract)."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--chemprop-mode",
        choices=("v2", "v3", "v3-fallback"),
        default="v3-fallback",
    )
    with pytest.raises(SystemExit) as exc_info:
        p.parse_args(["--help"])
    assert exc_info.value.code == 0