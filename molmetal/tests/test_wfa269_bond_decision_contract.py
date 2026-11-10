"""Wf-A269 wave-20 — the ``--bond-decision`` ``choices`` contract.

THE DEFECT CLASS (a new one, in the §6.269.17 DEAD-GATE family — a check that
cannot fail because the check does not exist)
----------------------------------------------------------------------
``argparse``'s ``choices=`` is a **DECLARED contract**: it enumerates the
values the trainer claims to accept.  Nothing in the repo checked that
declaration against the two things it has to agree with:

1. **the rule set the DECODER actually implements**, and
2. **the values the SHIPPED CONFIGS actually select**.

MEASURED 2026-11-04 (read-only, AST + a CPU probe):

* ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py`` declared
  ``choices=("argmax", "threshold")``;
* ``molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py::bond_decision``
  implements a **THIRD** rule, ``"per_class"`` (the Wf-A80 per-class
  calibrated thresholds);
* **TWO** shipped configs select it —
  ``configs/cfm_5metal_wfa80_per_class_cpu_smoke.yaml`` and
  ``configs/cfm_5metal_wfa80_per_class_decision.yaml`` (``flow.bond_decision:
  per_class``);
* and the YAML path **BYPASSES argparse's ``choices`` entirely** — MEASURED
  with a CPU probe of ``wfA2_config_loader.apply_yaml_overrides``:
  ``args.bond_decision`` becomes ``"per_class"`` although the parser forbids
  it.  So the stale declaration was invisible to EVERY gate in the repo, and
  the CLI could not express what the YAML could.

⇒ the class is not "the value is wrong"; it is **"a declared contract that no
gate checks, guarding a value the implementation accepts and the configs
select"**.

WHAT THIS TEST DOES (it is the MECHANISM, not a restatement)
-------------------------------------------------------------
AST-based on purpose: **no torch, no model, no GPU, no GPU smoke needed** —
the contract lives in the source text, and reading source text is exactly what
a CPU-only read-only review can do.

* **T1** the parser's ``choices`` for ``--bond-decision`` ⊇ the rule set the
  decoder implements (both derived from source, neither hard-coded here);
* **T2** the parser's ``choices`` ⊇ every value a shipped config selects;
* **T3** the parser's ``choices`` ⊇ ``{"argmax", "threshold"}`` (the historical
  pair, kept so the test cannot be satisfied by emptying the tuple).

MUTATION-PROVABLE: reverting the trainer's ``choices`` to
``("argmax", "threshold")`` makes **T1 and T2 FAIL**; emptying the tuple makes
**T1, T2 and T3 FAIL**.  A test that cannot fail is the defect class this
repo is hunting — see ``docs/HONEST_NEGATIVE_CATALOG.md`` §6.269.17.

APPEND-ONLY (§5.5 rule #6): the pre-repair rendering
``choices=("argmax", "threshold")`` is preserved VERBATIM in the trainer's
dated comment, and this file's docstring records the superseded declaration.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]

TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
DECODER = REPO / "molmetal" / "molmetal_lam" / "lam_chem" / "yuelbond_decoder.py"

#: The historical pair.  MEASURED 2026-11-04 — this is what the trainer
#: declared BEFORE the Wf-A269 wave-20 repair, and what the pre-repair
#: rendering in the trainer's dated comment still shows.
HISTORICAL_CHOICES = ("argmax", "threshold")

#: Where a shipped config may put the key.  The loader resolves
#: ``flow.bond_decision`` to the argparse dest ``bond_decision``.
CONFIG_ROOTS = (REPO / "configs", REPO / "molmetal" / "configs")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _bond_decision_choices() -> tuple[str, ...]:
    """The ``choices`` the trainer declares for ``--bond-decision``.

    Read off the AST, so the test fails if the declaration is changed —
    including to an EMPTY tuple, which is the other way to make a declared
    contract meaningless.
    """
    tree = _parse(TRAINER)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and first.value == "--bond-decision"):
            continue
        for kw in node.keywords:
            if kw.arg != "choices":
                continue
            if isinstance(kw.value, (ast.Tuple, ast.List)):
                return tuple(
                    e.value for e in kw.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                )
            raise AssertionError(
                "--bond-decision 'choices' is not a literal tuple/list — "
                "this test can no longer read the declared contract"
            )
    raise AssertionError("--bond-decision add_argument not found in the trainer")


def _decoder_rule_set() -> set[str]:
    """The decision rules ``yuelbond_decoder.bond_decision`` implements.

    Derived from the decoder's own source: every string the function compares
    ``decision`` against.  Not hard-coded — so adding a fourth rule to the
    decoder without widening the parser's ``choices`` breaks T1.
    """
    tree = _parse(DECODER)
    fn = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "bond_decision"):
            fn = node
            break
    assert fn is not None, "bond_decision() not found in yuelbond_decoder.py"

    rules: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Name) and left.id == "decision"):
            continue
        for op, comp in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                continue
            if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                rules.add(comp.value)
    return rules


def _configs_selecting_bond_decision() -> dict[str, str]:
    """``{config path (repo-relative): selected value}`` for every config.

    Only the files whose TEXT mentions ``bond_decision`` are parsed, so the
    test stays fast on a repo with hundreds of configs.
    """
    found: dict[str, str] = {}
    for root in CONFIG_ROOTS:
        if not root.is_dir():
            continue
        for cfg in sorted(root.rglob("*.yaml")):
            try:
                text = cfg.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "bond_decision" not in text:
                continue
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError:
                continue
            if not isinstance(data, dict):
                continue
            flow = data.get("flow")
            if not isinstance(flow, dict):
                continue
            value = flow.get("bond_decision")
            if isinstance(value, str):
                found[str(cfg.relative_to(REPO))] = value
    return found


# --------------------------------------------------------------------------
# T1 — the parser's declared contract vs the decoder's implemented rules
# --------------------------------------------------------------------------


def test_t1_parser_choices_cover_every_decoded_rule() -> None:
    declared = set(_bond_decision_choices())
    implemented = _decoder_rule_set()

    assert implemented, ("no decision rule found in the decoder's source — "
                         "the extraction is broken, which would make this "
                         "test vacuous")
    missing = implemented - declared
    assert not missing, (
        "the trainer's --bond-decision choices do not cover the decision "
        f"rules the decoder implements: missing {sorted(missing)} "
        f"(declared={sorted(declared)}, implemented={sorted(implemented)}). "
        "A YAML value outside argparse's 'choices' still reaches the dest "
        "(MEASURED 2026-11-04), so this is a DECLARED-vs-IMPLEMENTED "
        "divergence, not a crash — but it makes the CLI unable to express "
        "what the configs select."
    )


# --------------------------------------------------------------------------
# T2 — the parser's declared contract vs the shipped configs
# --------------------------------------------------------------------------


def test_t2_parser_choices_cover_every_shipped_config_value() -> None:
    declared = set(_bond_decision_choices())
    selected = _configs_selecting_bond_decision()

    # The two per-class configs are the ones that made this a defect; if a
    # future edit removes them the test must still be able to FAIL on the
    # trainer's side alone, so do not assert on their presence here.
    assert selected, ("no shipped config selects flow.bond_decision — the "
                       "extraction is broken, which would make this test "
                       "vacuous")
    offenders = {p: v for p, v in selected.items() if v not in declared}
    assert not offenders, (
        "shipped configs select bond_decision values the trainer's parser "
        f"forbids: {offenders} (declared={sorted(declared)}). The YAML path "
        "bypasses argparse's 'choices', so the run still works — the "
        "DECLARATION is what is stale."
    )


# --------------------------------------------------------------------------
# T3 — the historical pair survives (anti-vacuity)
# --------------------------------------------------------------------------


def test_t3_historical_choices_are_still_declared() -> None:
    declared = set(_bond_decision_choices())
    missing = set(HISTORICAL_CHOICES) - declared
    assert not missing, (
        f"--bond-decision no longer declares {sorted(missing)} — the "
        "historical pair was dropped. AGENTS.md §5.5 rule #6 forbids a "
        "silent removal of a published contract."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
