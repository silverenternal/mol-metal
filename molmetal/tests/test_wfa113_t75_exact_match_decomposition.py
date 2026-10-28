"""Wf-A113 (T75, 2026-10-28) — the bond "exact" pair must be a decomposition,
not one number printed twice.

Before this fix ``evaluate_bond_accuracy`` incremented ``exact`` and ``exact_type`` in
the same branch, under the identical condition::

    if set(pred_set) == set(truth) and all(pred_set[k] == truth[k] ...):
        exact += 1
    if set(pred_set) == set(truth) and all(pred_set[k] == truth[k] ...):
        exact_type += 1

so ``bond_exact_match`` and ``bond_exact_type_rate`` were equal in every
archived run — MEASURED: 903 rows across 103 ``per_epoch_metrics.json`` files,
903 identical, 0 differing.  A run could not say whether it had failed on
connectivity or on bond order.

The bench (``bench_tmqm_bond_perception.score_bonds``) always reported both
levels — ``exact`` = set only, ``exact_type`` = set + order — so the trainer
was the odd one out.  These tests pin the three properties that make the
decomposition real:

1. the two counters are guarded by DIFFERENT conditions (one mentions the
   per-class ``all(...)`` check, the other does not);
2. the set-only quantity reaches the artefact (``EpochMetrics`` field +
   ``evaluate_bond_accuracy`` output key);
3. the strict key keeps its literature meaning: ``bond_exact_match`` is the
   alias of ``bond_exact_type_rate``, not of the set-only number.
"""

from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path

import pytest

_TRAINER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_cfm_yuelbond_wasserstein.py"
)


def _module_ast() -> ast.Module:
    return ast.parse(_TRAINER.read_text())


def _bond_accuracy_fn(tree: ast.Module) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_bond_accuracy":
            return node
    pytest.fail("_evaluate_bond_accuracy not found — was it renamed?")


def _if_guarding(fn: ast.FunctionDef, counter: str) -> ast.If:
    """Return the ``if`` whose body increments ``counter`` by one."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.AugAssign):
                continue
            if not (isinstance(stmt.target, ast.Name) and stmt.target.id == counter):
                continue
            if isinstance(stmt.value, ast.Constant) and stmt.value.value == 1:
                return node
    pytest.fail(f"no `{counter} += 1` branch found in _bond_accuracy")


def _mentions_per_class_check(node: ast.AST) -> bool:
    """True if the guard contains an ``all(...)`` over per-class equality."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            if sub.func.id == "all":
                return True
    return False


def test_set_only_and_type_counters_use_different_guards() -> None:
    fn = _bond_accuracy_fn(_module_ast())
    set_guard = _if_guarding(fn, "exact_set")
    type_guard = _if_guarding(fn, "exact_type")
    assert not _mentions_per_class_check(set_guard), (
        "the set-only counter is guarded by a per-class `all(...)` check, so it "
        "is identical to the type counter by construction — the T75 defect"
    )
    assert _mentions_per_class_check(type_guard), (
        "the strict counter no longer requires every bond class to match"
    )


def test_bond_accuracy_emits_set_only_rate() -> None:
    fn = _bond_accuracy_fn(_module_ast())
    emitted = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "out"
                    and isinstance(tgt.slice, ast.Constant)
                ):
                    emitted.add(tgt.slice.value)
                # `out = {"bond_exact_match": ..., ...}` literal form
                if isinstance(tgt, ast.Name) and tgt.id == "out":
                    if isinstance(node.value, ast.Dict):
                        for key in node.value.keys:
                            if isinstance(key, ast.Constant):
                                emitted.add(key.value)
    assert "bond_set_exact_rate" in emitted
    assert "bond_exact_type_rate" in emitted
    assert "bond_exact_match" in emitted


def test_epoch_metrics_carries_set_only_rate() -> None:
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import EpochMetrics

    row = EpochMetrics(
        epoch=0,
        wall_time_s=0.0,
        train_loss=0.0,
        bond_ce_loss=0.0,
        wasserstein_loss=0.0,
        catherine_audit_pass=False,
        decode_ratio_rdkit_strict=0.0,
        decode_ratio_val=0,
        decode_ratio_n=0,
        decoding_attempted=False,
        lr=1e-4,
        grad_norm=0.0,
    )
    assert row.bond_set_exact_rate == 0.0
    payload = asdict(row)
    assert "bond_set_exact_rate" in payload
    assert "bond_exact_type_rate" in payload


def test_docstring_states_the_strict_meaning() -> None:
    """`bond_exact_match` must keep the strict (set + order) meaning."""
    fn = _bond_accuracy_fn(_module_ast())
    doc = ast.get_docstring(fn) or ""
    assert "bond_set_exact_rate" in doc
    assert "alias" in doc.lower(), (
        "the docstring must say `bond_exact_match` is an alias of "
        "`bond_exact_type_rate`, or a reader will compare it to the bench's "
        "set-only `exact`"
    )
