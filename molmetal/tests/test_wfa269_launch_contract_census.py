# Licensed under the Apache License, Version 2.0.
"""Wf-A269 census repair: the launch-contract gate's counters and its fail-open branch.

Two defects, both in ``scripts/_launch_contract_gate.py`` (the AGENTS.md §5.11
launch-contract gate), are locked here.

1. **THE CENSUS WAS INFLATED.**  The shared shell-parsing helper
   ``_trainer_call`` returns non-None for ANY line whose TEXT mentions a
   ``train_*.py``: a shell assignment (``TRAINER="${SCRIPTS_DIR}/train_x.py"``),
   a JSON receipt line inside a heredoc (``"trainer": "...train_x.py",``), a
   python assignment inside a heredoc (``legacy_script = '...'``), a
   comparison (``Path(w).name == '...'``).  The gate counted every one of them
   as a "trainer call", so ``n_trainer_calls`` claimed a coverage the gate
   does not have.

   ⇒ **A CENSUS DEFECT, NOT A VERDICT DEFECT.**  A mention contributes no
   argv and therefore no flag and no violation: the VERDICTS were right and
   the COUNTER was wrong.  Both facts are asserted below — the counter is
   repaired, and the verdict set is proved unchanged.

2. **A FAIL-OPEN BRANCH.**  ``audit()`` resolved each invocation's trainer
   and then did ``if trainer is None: continue`` — so a trainer invocation
   whose trainer file could not be resolved was SKIPPED and the gate still
   reported PASS, having checked nothing for it.  That is the §6.269.19.5
   defect class (a check that can PASS WITHOUT CHECKING) reproduced inside
   the mechanism built to detect it.

   ⇒ **NOW FAIL-CLOSED:** such an invocation is a VIOLATION
   (``kind="unresolved_trainer"``).

Every test below is a MUTATION test where it can be: it plants a defect and
requires the verdict to FLIP.  A guard that has never been seen to refuse is
not a guard (§6.269.17).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: A trainer that DOES exist, and one that does NOT.  The second is the
#: fail-closed case: the wrapper names a trainer the gate cannot find.
REAL_TRAINER = "${REPO_ROOT}/molmetal/scripts/train_metal_lomo_wasserstein.py"
MISSING_TRAINER = "${REPO_ROOT}/molmetal/scripts/train_does_not_exist_wfa269.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def contract_gate():
    return _load(
        REPO_ROOT / "scripts" / "_launch_contract_gate.py",
        "_wfa269_census_contract_gate",
    )


# ---------------------------------------------------------------------------
# 1. A mention is not an invocation
# ---------------------------------------------------------------------------
#: Four lines that MENTION a trainer and invoke nothing:
#:   - a shell assignment,
#:   - a JSON receipt line inside a heredoc,
#:   - a python assignment inside a heredoc,
#:   - a python comparison inside a heredoc.
MENTIONS_ONLY = """#!/usr/bin/env bash
set -euo pipefail
TRAINER="${REPO_ROOT}/molmetal/scripts/train_metal_lomo_wasserstein.py"
cat <<'JSON' > "${RECEIPT}"
{
  "trainer": "molmetal/scripts/train_metal_lomo_wasserstein.py",
  "note": "not an invocation"
}
JSON
cat <<'PY' > /dev/null
legacy_script = 'molmetal/scripts/train_metal_lomo_wasserstein.py'
if Path(w).name == 'train_metal_lomo_wasserstein.py':
    pass
PY
"""


def test_a_mention_does_not_inflate_the_invocation_counter(contract_gate, tmp_path):
    """THE CENSUS TEST: mentions are counted separately from invocations."""
    w = tmp_path / "mentions.sh"
    w.write_text(MENTIONS_ONLY, encoding="utf-8")
    report = contract_gate.audit([w])

    assert report["n_trainer_lines_seen"] == 4, (
        "the fixture no longer mentions a trainer four times — the test has "
        f"stopped exercising the defect: {report}"
    )
    assert report["n_trainer_invocations"] == 0, (
        "a MENTION was counted as a trainer invocation — the census is "
        "inflated again"
    )
    assert report["n_mentions"] == 4, report
    assert report["n_trainer_calls"] == report["n_trainer_invocations"], (
        "the historical `n_trainer_calls` key must mean the same thing as "
        "`n_trainer_invocations`"
    )
    assert report["n_flags_checked"] == 0, (
        "a mention contributed flags — a mention has no argv, so it cannot"
    )


def test_every_mention_is_classified(contract_gate, tmp_path):
    """A mention must say WHICH kind of mention it is, not just that it is one."""
    w = tmp_path / "mentions.sh"
    w.write_text(MENTIONS_ONLY, encoding="utf-8")
    report = contract_gate.audit([w])

    kinds = {m["kind"] for m in report["mentions"]}
    assert kinds <= {
        "mention_no_script_in_command_position",
        "mention_command_is_not_a_trainer",
    }, kinds
    assert kinds, "the mentions carry no classification at all"
    for m in report["mentions"]:
        assert m["file"].endswith("mentions.sh")
        assert isinstance(m["line"], int) and m["line"] > 0


def test_a_non_trainer_python_file_is_not_a_false_positive(contract_gate, tmp_path):
    """A wrapper that runs a HELPER (not a trainer) must not be flagged."""
    w = tmp_path / "helper.sh"
    w.write_text(
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        # a mention of a trainer, on a line that runs nothing:
        'echo "the CFM trainer is molmetal/scripts/train_cfm_yuelbond_wasserstein.py"\n'
        # a python invocation of a file that is NOT a trainer:
        '"${PY}" "${REPO_ROOT}/scripts/helper_check.py" --config "${CFG}"\n',
        encoding="utf-8",
    )
    report = contract_gate.audit([w])
    assert report["verdict"] == "PASS", report["violations"]
    assert report["n_violations"] == 0, report["violations"]
    assert report["n_trainer_invocations"] == 0, (
        "a helper script was counted as a trainer invocation"
    )


def test_a_mention_never_produces_a_violation(contract_gate, tmp_path):
    """THE INVARIANT: a mention contributes no violation, on any wrapper."""
    w = tmp_path / "mentions.sh"
    w.write_text(MENTIONS_ONLY, encoding="utf-8")
    report = contract_gate.audit([w])
    mention_sites = {(m["file"], m["line"]) for m in report["mentions"]}
    for v in report["violations"]:
        assert (v["file"], v["line"]) not in mention_sites, (
            "a violation was raised on a MENTION line — the repair changed "
            "the verdicts, which it must not"
        )


# ---------------------------------------------------------------------------
# 2. The fail-open branch, now fail-closed
# ---------------------------------------------------------------------------
UNRESOLVED = """#!/usr/bin/env bash
set -euo pipefail
PY="${REPO_ROOT}/.venv/bin/python"
TRAINER="${REPO_ROOT}/molmetal/scripts/train_does_not_exist_wfa269.py"
"${PY}" "${TRAINER}" --config "${CFG}"
"""


def test_an_unresolvable_trainer_invocation_is_fail_closed(contract_gate, tmp_path):
    """THE MUTATION: an invocation whose trainer cannot be found must FAIL.

    The superseded behaviour was ``if trainer is None: continue`` — the gate
    SKIPPED the invocation and still reported PASS, having checked nothing
    for it.  That is the §6.269.19.5 defect class: a check that can PASS
    WITHOUT CHECKING.
    """
    w = tmp_path / "unresolved.sh"
    w.write_text(UNRESOLVED, encoding="utf-8")
    report = contract_gate.audit([w])

    assert report["verdict"] == "FAIL", (
        "an unresolvable trainer invocation was accepted — the gate can "
        "still PASS without checking"
    )
    assert report["n_unresolved_trainers"] == 1, report
    kinds = {v["kind"] for v in report["violations"]}
    assert "unresolved_trainer" in kinds, report["violations"]
    offender = [v for v in report["violations"] if v["kind"] == "unresolved_trainer"]
    assert offender[0]["trainer"] == "train_does_not_exist_wfa269.py", offender


def test_the_unresolvable_invocation_is_not_a_silent_skip(contract_gate, tmp_path):
    """The gate must neither certify it nor merely count it in a census.

    A census entry is not enough: the verdict must FAIL, because a gate that
    reports a gap and then says PASS is lying about what it checked.
    """
    w = tmp_path / "unresolved.sh"
    w.write_text(UNRESOLVED, encoding="utf-8")
    report = contract_gate.audit([w])

    assert report["n_flags_checked"] == 0, (
        "the gate counted flags it never checked against a parser"
    )
    assert report["n_violations"] >= 1, report
    assert report["verdict"] == "FAIL", report


def test_the_resolvable_twin_still_passes(contract_gate, tmp_path):
    """The fail-closed branch must fire ONLY when the trainer is missing.

    This is the other half of the mutation: if the SAME wrapper named a
    trainer that exists, the verdict must be PASS.  A fail-closed branch
    that fires on everything is not a gate either.
    """
    w = tmp_path / "resolved.sh"
    w.write_text(
        UNRESOLVED.replace(
            "train_does_not_exist_wfa269.py", "train_metal_lomo_wasserstein.py"
        ),
        encoding="utf-8",
    )
    report = contract_gate.audit([w])

    assert report["verdict"] == "PASS", report["violations"]
    assert report["n_unresolved_trainers"] == 0, report
    assert report["n_trainer_invocations"] == 1, report


# ---------------------------------------------------------------------------
# 3. The verdicts did not change
# ---------------------------------------------------------------------------
def test_a_real_defect_still_fails(contract_gate, tmp_path):
    """The repair must not have blunted the gate's actual teeth."""
    w = tmp_path / "defect.sh"
    w.write_text(
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        f'TRAINER="{REAL_TRAINER}"\n'
        '"${PY}" "${TRAINER}" --config "${CFG}" --ckpt-dir "${CK}"\n',
        encoding="utf-8",
    )
    report = contract_gate.audit([w])
    assert report["verdict"] == "FAIL", report["violations"]
    kinds = {v["kind"] for v in report["violations"]}
    assert "undeclared_flag" in kinds, report["violations"]
    assert report["n_trainer_invocations"] == 1, report


def test_a_mention_next_to_a_real_defect_does_not_change_the_verdict(
    contract_gate, tmp_path
):
    """A mention on one line and a defect on another: ONE violation, not two."""
    w = tmp_path / "mixed.sh"
    w.write_text(
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        f'TRAINER="{REAL_TRAINER}"\n'
        'echo "trainer: ${TRAINER}"\n'
        '"${PY}" "${TRAINER}" --config "${CFG}" --ckpt-dir "${CK}"\n',
        encoding="utf-8",
    )
    report = contract_gate.audit([w])
    assert report["verdict"] == "FAIL", report["violations"]
    assert report["n_violations"] == 1, report["violations"]
    assert report["n_trainer_lines_seen"] == 2, report
    assert report["n_trainer_invocations"] == 1, report
    assert report["n_mentions"] == 1, report


# ---------------------------------------------------------------------------
# 4. The shipped tree
# ---------------------------------------------------------------------------
def test_the_shipped_tree_census_is_internally_consistent(contract_gate):
    """The census must add up: seen == invocations + mentions."""
    report = contract_gate.audit()
    assert report["n_trainer_calls"] == report["n_trainer_invocations"]
    assert (
        report["n_trainer_lines_seen"]
        == report["n_trainer_invocations"] + report["n_mentions"]
    ), report


def test_the_shipped_tree_has_no_unresolved_trainer(contract_gate):
    """The fail-closed branch must not fire on the live tree.

    MEASURED 2026-11-04 on this tree: 87 lines mention a trainer, 31 are
    invocations, 56 are mentions, and every invocation's trainer resolves —
    so the fail-closed branch flags **0** lines here.  It is armed for a
    renamed or typo'd trainer, not for the shipped surface.
    """
    report = contract_gate.audit()
    assert report["n_unresolved_trainers"] == 0, report["violations"]
    assert report["n_coverage_gaps"] == 0, report["coverage_gaps"]


def test_the_shipped_tree_still_passes(contract_gate):
    """The repair changed the census, not the verdict."""
    report = contract_gate.audit()
    assert report["n_violations"] == 0, report["violations"]
    assert report["verdict"] == "PASS", report["violations"]
