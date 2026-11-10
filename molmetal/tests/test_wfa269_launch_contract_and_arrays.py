# Licensed under the Apache License, Version 2.0.
"""Wf-A269 wave-19: the launch contract, and the bash-array blind spot.

Two regressions are locked here, and both are regressions of a GATE THAT
COULD PASS WITHOUT CHECKING (§6.269.19):

1. **The bash-array blind spot.**  ``"${COMMON_ARGS[@]}"`` puts the trainer's
   flags in an array body, not in the invocation line, so a line-oriented scan
   sees none of them.  MEASURED 2026-11-03: 13 live wrappers hid 130 flags
   this way, and ``scripts/launch_wf390a1_m2_r2_path_a.sh`` — 16 flags — was
   PASSED by the gate (exit 0).

2. **The fixed allowlist.**  §5.11's allowlist is the same four strings for
   every trainer, but the legal set is defined by the TRAINER'S OWN PARSER:
   MEASURED 2026-11-03, ``--ckpt-dir`` is declared by 8 of 38 trainers, and
   the repo's own run log records a launch that aborted on it —
   ``molmetal/reports/wf125_p1_controlled_pair/run_synthetic/run.log:18``.

Every test below is a MUTATION test: it plants a defect and requires the
verdict to FLIP.  A guard that has never been seen to refuse is not a guard.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def yaml_gate():
    return _load(REPO_ROOT / "scripts" / "_yaml_only_gate.py", "_wfa269_yaml_gate")


@pytest.fixture(scope="module")
def contract_gate():
    return _load(
        REPO_ROOT / "scripts" / "_launch_contract_gate.py", "_wfa269_contract_gate"
    )


# ---------------------------------------------------------------------------
# 1. The bash-array blind spot
# ---------------------------------------------------------------------------
ARRAY_WRAPPER = """#!/usr/bin/env bash
set -euo pipefail
PY="${REPO_ROOT}/.venv/bin/python"
TRAINER="${REPO_ROOT}/molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
COMMON_ARGS=(
  --epochs 30
  --hidden 128
  --lr 0.001
)
"${PY}" "${TRAINER}" \\
  --config "${CFG}" \\
"${COMMON_ARGS[@]}" 2>&1 | tee "${LOG}"
"""


def test_array_hidden_flags_are_reported(yaml_gate, tmp_path):
    """A flag inside a bash array must be reported at the trainer call site."""
    w = tmp_path / "array_wrapper.sh"
    w.write_text(ARRAY_WRAPPER, encoding="utf-8")
    found, _ = yaml_gate._audit_wrapper(w)
    flags = {v["detail"].split()[0] for v in found if v["kind"] == "trainer_flag"}
    assert {"--epochs", "--hidden", "--lr"} <= flags, (
        "the array-hidden flags were not reported — the gate has gone blind to "
        "bash arrays again"
    )


def test_the_array_flags_are_genuinely_absent_from_the_line_text(yaml_gate, tmp_path):
    """Lock WHY the repair is needed: the raw invocation line carries no flag.

    Without this, the test above could pass for the wrong reason (e.g. if a
    future refactor started inlining the array into the command).
    """
    w = tmp_path / "array_wrapper.sh"
    w.write_text(ARRAY_WRAPPER, encoding="utf-8")
    text = w.read_text(encoding="utf-8")
    env = yaml_gate._collect_assignments(text)
    assert isinstance(env.get("COMMON_ARGS"), list), "the array was not captured"
    for _lineno, line in yaml_gate._logical_lines(text):
        via = yaml_gate._trainer_call(line, env)
        if via is None:
            continue
        raw = yaml_gate._FLAG.findall(yaml_gate._strip_detach_head(line))
        assert "--epochs" not in raw, (
            "the fixture no longer exercises the blind spot: --epochs is now "
            "literal in the invocation line"
        )


def test_unresolvable_array_is_fail_closed(yaml_gate, tmp_path):
    """An array the gate cannot read must be a VIOLATION, not a clean line."""
    w = tmp_path / "opaque_wrapper.sh"
    w.write_text(
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        'TRAINER="${REPO_ROOT}/molmetal/scripts/train_dmpnn_multitask_wasserstein.py"\n'
        '"${PY}" "${TRAINER}" --config "${CFG}" "${OPAQUE_ARGS[@]}"\n',
        encoding="utf-8",
    )
    found, _ = yaml_gate._audit_wrapper(w)
    kinds = {v["kind"] for v in found}
    assert "unresolved_expansion" in kinds, (
        "an unreadable array expansion was treated as clean — the gate must "
        "fail closed on what it cannot see"
    )


def test_a_readable_array_is_not_fail_closed(yaml_gate, tmp_path):
    """The fail-closed branch must not fire on an array the gate CAN read."""
    w = tmp_path / "array_wrapper.sh"
    w.write_text(ARRAY_WRAPPER, encoding="utf-8")
    found, _ = yaml_gate._audit_wrapper(w)
    assert not [v for v in found if v["kind"] == "unresolved_expansion"]


# ---------------------------------------------------------------------------
# 2. The launch contract
# ---------------------------------------------------------------------------
def test_undeclared_flag_is_a_violation(contract_gate, tmp_path):
    """`--ckpt-dir` is in §5.11's allowlist but NOT declared by the lomo trainer."""
    w = tmp_path / "ckpt_wrapper.sh"
    w.write_text(
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        'TRAINER="${REPO_ROOT}/molmetal/scripts/train_metal_lomo_wasserstein.py"\n'
        '"${PY}" "${TRAINER}" --config "${CFG}" --ckpt-dir "${CK}"\n',
        encoding="utf-8",
    )
    report = contract_gate.audit([w])
    assert report["verdict"] == "FAIL", (
        "a flag the trainer does not declare was accepted — the launch would "
        'abort with "unrecognized arguments"'
    )
    assert any("--ckpt-dir" in v["detail"] for v in report["violations"])


def test_declared_flag_is_accepted(contract_gate, tmp_path):
    """`--out-dir` IS declared by the lomo trainer ⇒ no violation."""
    w = tmp_path / "ok_wrapper.sh"
    w.write_text(
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        'TRAINER="${REPO_ROOT}/molmetal/scripts/train_metal_lomo_wasserstein.py"\n'
        '"${PY}" "${TRAINER}" --config "${CFG}" --out-dir "${OUT}"\n',
        encoding="utf-8",
    )
    report = contract_gate.audit([w])
    assert report["verdict"] == "PASS", report["violations"]


def test_the_mutation_flips_the_verdict(contract_gate, tmp_path):
    """THE MUTATION TEST: plant one defect, require the verdict to flip.

    This is the §6.269.17 rule — a gate is only a gate if it has been seen to
    refuse.  The fixture is the SHIPPED wrapper with one flag added.
    """
    shipped = REPO_ROOT / "scripts" / "launch_wf125_p1_controlled_pair.sh"
    assert shipped.is_file()
    text = shipped.read_text(encoding="utf-8")

    clean = tmp_path / "clean.sh"
    clean.write_text(text, encoding="utf-8")
    before = contract_gate.audit([clean])
    assert before["verdict"] == "PASS", (
        f"the shipped wrapper is already violating the launch contract: "
        f"{before['violations'][:3]}"
    )

    # Plant the defect the repo's own run log records.
    mutated_text = text.replace(
        '--config "${run_cfg}" \\',
        '--config "${run_cfg}" \\\n    --ckpt-dir "${OUT_BASE}/run_${run_mode}/ckpts" \\',
        1,
    )
    assert mutated_text != text, "the mutation did not apply — the fixture moved"
    mutated = tmp_path / "mutated.sh"
    mutated.write_text(mutated_text, encoding="utf-8")
    after = contract_gate.audit([mutated])
    assert after["verdict"] == "FAIL", (
        "the planted defect did not flip the verdict — the gate cannot fail"
    )
    assert any("--ckpt-dir" in v["detail"] for v in after["violations"])


# ---------------------------------------------------------------------------
# 3. The shipped tree
# ---------------------------------------------------------------------------
def test_the_shipped_tree_satisfies_the_launch_contract(contract_gate):
    """Every live wrapper must pass only flags its trainer declares.

    This is the test that keeps the §5.11 migration honest: a migration that
    replaces sixteen hyperparameters with one flag the trainer rejects is not
    a repair, and this test is what says so.
    """
    report = contract_gate.audit()
    offenders = [
        f"{v['file']}:{v['line']} {v['detail'].split()[0]} -> {v['trainer']}"
        for v in report["violations"]
    ]
    assert not offenders, "launch-contract violations in the shipped tree:\n" + "\n".join(
        offenders
    )


def test_the_shipped_tree_satisfies_the_yaml_only_rule(yaml_gate):
    """No live wrapper may pass a hyperparameter on a trainer invocation line."""
    report = yaml_gate.audit()
    offenders = [
        f"{v['file']}:{v['line']} [{v['kind']}] {v['detail'][:120]}"
        for v in report["violations"]
    ]
    assert not offenders, "yaml-only violations in the shipped tree:\n" + "\n".join(
        offenders
    )


# ---------------------------------------------------------------------------
# 3. The gate's own capture bug — and the fail-OPEN branch it fed
# ---------------------------------------------------------------------------
# MEASURED 2026-11-03: `scripts/_launch_contract_gate.py` called
# `mod.main([])` UNCONDITIONALLY.  Most trainers declare `def main():` with
# NO parameters, so the call raised `TypeError`, the `except BaseException`
# swallowed it, and the trainer was reported as "could not capture a parser
# namespace".  Over the corpus `molmetal/scripts/train_*.py` (43 files) that
# made 5 trainers UNREADABLE: train_hybrid.py, train_metal_hybrid_v4.py,
# train_metal_hybrid_v4_mb1.py, train_property_predictor.py,
# train_symbolic_regression.py.
#
# And the branch that consumed the unreadable parser was FAIL-OPEN: the gate
# SKIPPED the wrapper's flags and still reported PASS — a check that can PASS
# WITHOUT CHECKING, which is the §6.269.19.5 defect class.
#
# BOTH are locked below.  The first test is the census that would have caught
# the capture bug; the second is the MUTATION that proves the fail-open branch
# has been turned into a fail-closed one.

ZERO_ARG_MAIN_TRAINERS = (
    "train_hybrid.py",
    "train_metal_hybrid_v4.py",
    "train_metal_hybrid_v4_mb1.py",
    "train_property_predictor.py",
    "train_symbolic_regression.py",
)


def test_zero_arg_main_trainers_are_captured(contract_gate):
    """A `def main():` trainer must have its parser read, not skipped.

    Before the repair these five reported "could not capture a parser
    namespace" and their flags were invisible to the gate.
    """
    for name in ZERO_ARG_MAIN_TRAINERS:
        path = REPO_ROOT / "molmetal" / "scripts" / name
        assert path.is_file(), path
        flags, err = contract_gate._declared_flags(path)
        assert flags is not None, (
            f"{name} has a zero-argument main() and the gate still cannot "
            f"read its parser ({err}) — the gate calls mod.main([]) first "
            f"and swallows the TypeError"
        )
        assert "--help" in flags, f"{name} captured an implausible flag set: {flags}"


def test_every_trainer_in_the_corpus_is_readable(contract_gate):
    """No trainer may be silently unreadable — the census that catches it.

    MEASURED 2026-11-03: 43 of 43 captured after the repair; 38 of 43 before.
    """
    corpus = sorted((REPO_ROOT / "molmetal" / "scripts").glob("train_*.py"))
    assert len(corpus) >= 43, f"corpus shrank: {len(corpus)}"
    unreadable = []
    for path in corpus:
        flags, err = contract_gate._declared_flags(path)
        if flags is None:
            unreadable.append(f"{path.name}: {err}")
    assert not unreadable, (
        "trainers whose parser the gate cannot read (their flags are "
        "invisible to the launch-contract check):\n  " + "\n  ".join(unreadable)
    )


def test_a_coverage_gap_is_fail_closed(contract_gate, tmp_path, monkeypatch):
    """THE MUTATION: a trainer the gate cannot read must make the verdict FAIL.

    The superseded behaviour was `coverage_gaps[rel] = err; continue` — the
    gate skipped the wrapper's flags and still said PASS.
    """
    w = tmp_path / "gap_wrapper.sh"
    w.write_text(
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        'TRAINER="${REPO_ROOT}/molmetal/scripts/train_metal_lomo_wasserstein.py"\n'
        '"${PY}" "${TRAINER}" --config "${CFG}" --out-dir "${OUT}"\n',
        encoding="utf-8",
    )
    before = contract_gate.audit([w])
    assert before["verdict"] == "PASS", before["violations"]

    real = contract_gate._declared_flags

    def broken(_path):
        return None, "could not capture a parser namespace"

    monkeypatch.setattr(contract_gate, "_declared_flags", broken)
    after = contract_gate.audit([w])
    monkeypatch.setattr(contract_gate, "_declared_flags", real)

    assert after["verdict"] == "FAIL", (
        "an unreadable parser did not fail the gate — the gate can still "
        "PASS without checking"
    )
    assert after["n_coverage_gaps"] == 1, after
    kinds = {v["kind"] for v in after["violations"]}
    assert "coverage_gap" in kinds, after["violations"]
    assert after["n_flags_checked"] == 0, (
        "the gap branch counted flags it never checked"
    )


def test_the_shipped_tree_has_no_coverage_gap(contract_gate):
    """The live tree must not depend on the fail-closed branch to pass."""
    report = contract_gate.audit()
    assert report["n_coverage_gaps"] == 0, report["coverage_gaps"]
