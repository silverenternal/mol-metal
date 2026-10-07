"""Wf-R32 regression test: YAML-vs-CLI precedence for LIST-valued argparse dests.

Context
-------
``wfA2_config_loader.apply_yaml_overrides`` lets YAML override argparse
defaults, but an explicitly-passed CLI flag must always win.  For scalar
dests this falls out of the ``cur == dft`` guard.  For LIST-valued dests the
same guard needs an explicit ``isinstance`` branch, because a list default
(e.g. the built-in metals cohort) is rarely the caller's intent, so YAML is
allowed to replace it -- but ONLY while it is still pristine.

The bug this guards against (fixed in Wf-R27 FIX 4, 2026-10-07): the list
branch used to ``setattr`` unconditionally, which made ``cur == dft``
unreachable for lists.  A launcher that enumerates one (metal, seed) pair
per invocation -- e.g. ``scripts/launch_wf124_p3_g6.sh``, which passes
``--metals Pt --seeds 0`` for each of 60 tuples -- would have been silently
clobbered into running the full YAML cohort on every tuple.

These tests pin the current, correct behaviour so a future refactor cannot
silently regress it back.

Run::

    .venv/bin/python -m pytest molmetal/tests/test_wfA2_list_precedence.py -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides  # noqa: E402

DEFAULT_COHORT = ["Ru", "Ir", "Rh", "Os", "Re"]


def _parser() -> argparse.ArgumentParser:
    """A parser shaped like the D-MPNN trainer's list-valued args."""
    p = argparse.ArgumentParser()
    p.add_argument("--metals", dest="metals", nargs="+", default=list(DEFAULT_COHORT))
    p.add_argument("--seeds", dest="seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", dest="epochs", type=int, default=10)
    return p


def _apply(cli: list[str], yaml_text: str) -> argparse.Namespace:
    """Parse ``cli``, then apply ``yaml_text`` as the config layer.

    ``apply_yaml_overrides`` calls ``parser.parse_args([])`` internally to
    recover the defaults, so the ambient ``sys.argv`` is neutralised first.
    """
    import tempfile

    parser = _parser()
    saved_argv = sys.argv
    sys.argv = ["prog"]
    try:
        args = parser.parse_args(cli)
    finally:
        sys.argv = saved_argv

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(yaml_text)
        path = fh.name
    try:
        saved_argv = sys.argv
        sys.argv = ["prog"]
        try:
            return apply_yaml_overrides(parser, args, path)
        finally:
            sys.argv = saved_argv
    finally:
        Path(path).unlink(missing_ok=True)


def test_explicit_cli_list_beats_yaml_list():
    """The Wf-R27 FIX 4 regression: an explicit --metals must survive YAML.

    Before the fix this returned the YAML cohort for every tuple, so all 60
    G6 (and all 15 G5) pairs would have run the identical full sweep.
    """
    args = _apply(
        ["--metals", "Pt", "--seeds", "0"],
        "metals: [Ru, Ir]\nseeds: [0, 1, 2]\n",
    )
    assert args.metals == ["Pt"], f"YAML clobbered explicit --metals: {args.metals}"
    assert args.seeds == [0], f"YAML clobbered explicit --seeds: {args.seeds}"


def test_untouched_cli_list_default_is_replaced_by_yaml():
    """The intended half of the list special-case: a PRISTINE default yields to YAML.

    The built-in 5-metal cohort is almost never what a caller wants, so when
    the user did not enumerate one, YAML is the more specific intent.
    """
    args = _apply([], "metals: [Pt, Au]\n")
    assert args.metals == ["Pt", "Au"], f"YAML failed to fill pristine list default: {args.metals}"


def test_partial_cli_presence_is_per_dest():
    """Precedence is decided per-dest, not per-invocation.

    Passing only --metals must not make --seeds immune to a YAML cohort.
    """
    args = _apply(["--metals", "Pt"], "metals: [Ru]\nseeds: [7]\n")
    assert args.metals == ["Pt"], f"explicit --metals was clobbered: {args.metals}"
    assert args.seeds == [7], f"pristine --seeds did not take YAML value: {args.seeds}"


def test_scalar_precedence_unaffected():
    """Scalars keep the long-standing rule: explicit CLI > YAML > default."""
    args = _apply(["--epochs", "3"], "epochs: 50\n")
    assert args.epochs == 3, f"explicit scalar --epochs was clobbered: {args.epochs}"


def test_shipped_config_does_not_regress_a_launcher_style_invocation():
    """Regression against the real configs/dmpnn.yaml, not just a fixture.

    Mirrors how scripts/launch_wf124_p3_g6.sh invokes the trainer per tuple.
    Skipped if the config is absent so the test is hermetic in isolation.
    """
    cfg = _REPO_ROOT / "configs" / "dmpnn.yaml"
    if not cfg.is_file():
        pytest.skip("configs/dmpnn.yaml not present")

    parser = _parser()
    saved_argv = sys.argv
    sys.argv = ["prog"]
    try:
        args = parser.parse_args(["--metals", "Pt", "--seeds", "0"])
        apply_yaml_overrides(parser, args, str(cfg))
    finally:
        sys.argv = saved_argv

    assert args.metals == ["Pt"], f"configs/dmpnn.yaml clobbered --metals: {args.metals}"
    assert args.seeds == [0], f"configs/dmpnn.yaml clobbered --seeds: {args.seeds}"
