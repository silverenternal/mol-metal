# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Wf-A269: the §5.11 script-contract gate — the NON-``train_*`` blind spot.

THE DEFECT
==========
``scripts/_launch_contract_gate.py`` asks "does the flag reach a parser that
DECLARES it?" — and it finds the script to ask by matching
``^train_[a-z0-9_]+\\.py$`` against the wrapper's invocation line.
``scripts/_yaml_only_gate.py`` rule 2 is anchored the same way.

So for every repo-local script that is NOT named ``train_*`` — the runners,
the drivers, the smoke helpers, the pre-flight — "does the flag reach a
parser that declares it?" was NEVER CHECKED.  A wrapper could pass a flag
the script does not declare, argparse would exit 2 with "unrecognized
arguments", and the launch would abort before the run started.

THE MEASURED EVIDENCE (2026-11-04, read-only census over the 101 in-contract
wrappers) found exactly TWO live instances of that class:

1. ``molmetal/scripts/smoke/smoke_lambda.sh`` passed ``--out-dir`` to
   ``molmetal/scripts/r4_lambda_only_run.py``, whose parser declares
   ``--output-dir``.  The smoke had NEVER passed.
2. ``scripts/launch_wf306b_wf219_ru_gpu.sh`` passed ``--out`` to
   ``molmetal/scripts/wf219_krasnov_aligned.py``, whose parser declares
   ``--json-dir``.  The wrapper could never have run.

Both were repaired by the main thread, so the live tree no longer carries
them — which is exactly why the gate has to be exercised against FIXTURES
that reproduce both.  A gate that passes on the live tree AND on a fixture
is the whole point: a gate that only ever sees a clean tree has never been
seen to refuse.

THE REPAIR
==========
``scripts/_script_contract_gate.py`` — a NEW §5.11 gate that

* takes its POPULATION from the owning module
  (``scripts/_yaml_only_gate.py::_population()``), so the denominator and
  the excluded set have exactly one source of truth;
* reads every invoked script's declared options by AST — never by import,
  because importing a repo script can WRITE FILES
  (``molmetal/scripts/wf394a3_wf299_pre_launch_verify.py`` runs its whole
  check at import time and rewrites
  ``molmetal/reports/wf394a3_wf299_pre_launch/result.json``);
* reports a flag the invoked script does not declare as
  ``kind="undeclared_flag"``;
* reports a script whose declared options CANNOT be read as
  ``kind="coverage_gap"`` — fail-closed, because a gate that skipped what
  it could not read and still reported PASS is a check that can PASS
  WITHOUT CHECKING (AGENTS.md §6.269.17);
* reports an UNDECIDABLE command position as ``kind="undecidable_script"``
  — also fail-closed, never as "clean".

THE LOOKAHEAD (the blind spot the gate must beat)
===================================================
``scripts/_yaml_only_gate.py::_script_index`` returns the token at the
COMMAND position.  When the command is an interpreter held in an
UNRESOLVED variable (the repo's wrappers assign it in three branches), the
value is unresolvable, so ``_script_index`` returns the interpreter token
itself and the gate never sees ``foo.py``.  This gate looks PAST an
unresolved interpreter to the next non-option token and uses THAT as the
script; if that token is itself a variable, the invocation is UNDECIDABLE
and is reported as such.

CPU-only, no GPU, no torch, no network.  Depends on no other test file.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

# REPO-RELATIVE root (never an absolute path literal: an absolute path in a
# test is a test that is red on every other machine).
REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "scripts" / "_script_contract_gate.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate():
    return _load(GATE_PATH, "_wfa269_script_contract_gate")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
#: The interpreter is an UNRESOLVED variable (``${REPO_ROOT}`` is never
#: assigned in the fixture), so reaching the script at all requires the
#: LOOKAHEAD — the gate must look past the interpreter to the next
#: non-option token.
HISTORICAL_FIXTURE = """#!/usr/bin/env bash
set -euo pipefail
PY="${REPO_ROOT}/.venv/bin/python"

# Historical defect 1 (molmetal/scripts/smoke/smoke_lambda.sh): the smoke
# passed --out-dir to a runner that declares --output-dir.  argparse would
# have exited 2 with "unrecognized arguments: --out-dir".
"${PY}" molmetal/scripts/r4_lambda_only_run.py \\
    --config configs/lambda_smoke.yaml \\
    --out-dir "${OUT_NAME}"

# Historical defect 2 (scripts/launch_wf306b_wf219_ru_gpu.sh): the wrapper
# passed --out to a driver that declares --json-dir.
"${PY}" molmetal/scripts/wf219_krasnov_aligned.py \\
    --out "${REPORT_DIR}" \\
    --quick
"""

#: The same two invocations with the DECLARED spellings.
CLEAN_FIXTURE = """#!/usr/bin/env bash
set -euo pipefail
PY="${REPO_ROOT}/.venv/bin/python"
"${PY}" molmetal/scripts/r4_lambda_only_run.py \\
    --config configs/lambda_smoke.yaml \\
    --output-dir "${OUT_NAME}"
"${PY}" molmetal/scripts/wf219_krasnov_aligned.py \\
    --json-dir "${REPORT_DIR}" \\
    --quick
"""

#: A script whose option vocabulary is empty AND KNOWN to be empty:
#: ``molmetal/scripts/eval_all_evaluators.py`` declares no parser, no
#: ``add_argument`` and does not read ``sys.argv`` (MEASURED 2026-11-04).
#: A FLAGLESS invocation of it is legal — there is no option to violate.
EMPTY_VOCABULARY_FIXTURE = """#!/usr/bin/env bash
set -euo pipefail
PY="${REPO_ROOT}/.venv/bin/python"
"${PY}" molmetal/scripts/eval_all_evaluators.py
"""

#: The same script with a FLAG on the line.  The vocabulary is empty, so the
#: flag is undeclared — and SILENTLY IGNORED, because the script has no
#: parser to abort with "unrecognized arguments".
EMPTY_VOCABULARY_WITH_FLAG_FIXTURE = """#!/usr/bin/env bash
set -euo pipefail
PY="${REPO_ROOT}/.venv/bin/python"
"${PY}" molmetal/scripts/eval_all_evaluators.py \\
    --config configs/lambda_smoke.yaml
"""

#: A script whose option vocabulary the gate CANNOT enumerate: it reads
#: ``sys.argv`` on its own, so it may consume options the gate cannot see
#: (MEASURED 2026-11-04: 12 repo scripts are in this state).
UNREADABLE_VOCABULARY_FIXTURE = """#!/usr/bin/env bash
set -euo pipefail
PY="${REPO_ROOT}/.venv/bin/python"
"${PY}" molmetal/scripts/registry.py \\
    --config configs/lambda_smoke.yaml
"""

#: A bash array the gate cannot read: the flags inside it are invisible, so
#: the invocation cannot be certified.
OPAQUE_ARRAY_FIXTURE = """#!/usr/bin/env bash
set -euo pipefail
PY="${REPO_ROOT}/.venv/bin/python"
"${PY}" molmetal/scripts/r4_lambda_only_run.py \\
    --config configs/lambda_smoke.yaml \\
    "${OPAQUE_ARGS[@]}"
"""

#: The command position is an unresolved variable and the only token after
#: it is another unresolved variable: UNDECIDABLE, fail-closed.
UNDECIDABLE_FIXTURE = """#!/usr/bin/env bash
set -euo pipefail
"${CMD}" --out "${REPORT_DIR}"
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    wrapper = tmp_path / name
    wrapper.write_text(text, encoding="utf-8")
    return wrapper


def _undeclared(report: dict) -> dict[str, str]:
    return {
        v["flag"]: v["script"]
        for v in report["violations"]
        if v["kind"] == "undeclared_flag"
    }


def _kinds(report: dict) -> set[str]:
    return {v["kind"] for v in report["violations"]}


def _without_comments(text: str) -> str:
    """The shell code of a wrapper, with the COMMENT lines removed.

    AGENTS.md §5.5 rule #6 keeps a superseded rendering VERBATIM inside a
    comment, so a naive text search would read the preserved history as a
    live defect.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


# ---------------------------------------------------------------------------
# 0. The population is not empty
# ---------------------------------------------------------------------------
def test_the_population_is_not_empty(gate):
    """A glob that matches nothing makes every other assertion vacuous.

    The gate's population is the OWNING module's
    (``scripts/_yaml_only_gate.py::_population()``): the tracked ``*.sh``
    files minus the excluded ones.  If that set were empty, "the gate found
    no violation" would mean "the gate looked at nothing".
    """
    paths, coverage = gate.GATE._population()
    assert paths, "the wrapper population is empty — the gate would certify "
    "nothing while reporting PASS"
    assert coverage["n_tracked_sh_total"] > 0
    assert coverage["n_scanned"] > 0
    assert coverage["n_scanned"] + coverage["n_excluded"] == coverage["n_tracked_sh_total"]


def test_the_population_and_the_exclusions_come_from_the_owner_module(gate):
    """The gate must NOT re-declare the excluded set — it would drift.

    ``scripts/_yaml_only_gate.py`` owns the population AND the exclusions
    (``EXCLUDED_WRAPPER_DIRS`` / ``EXCLUDED_WRAPPER_FILES``).  A second copy
    of that list is a second source of truth that WILL drift (AGENTS.md
    §6.269.22: a census is a COVERAGE census, and its denominator must have
    one owner).
    """
    assert "EXCLUDED_WRAPPER_DIRS" not in gate.__dict__
    assert "EXCLUDED_WRAPPER_FILES" not in gate.__dict__
    _paths, coverage = gate.GATE._population()
    assert coverage["population_source"] == "git ls-files '*.sh'"
    assert coverage["excluded"], "the owner module excluded nothing — the "
    "exclusion set would be silently empty"


# ---------------------------------------------------------------------------
# 1. The two HISTORICAL defects must be found
# ---------------------------------------------------------------------------
def test_the_lambda_smoke_out_dir_defect_is_found(gate, tmp_path):
    """Historical defect 1: ``--out-dir`` on a script that declares
    ``--output-dir``."""
    wrapper = _write(tmp_path, "historical_lambda.sh", HISTORICAL_FIXTURE)
    report = gate.audit([wrapper])
    undeclared = _undeclared(report)
    assert "--out-dir" in undeclared, (
        "the --out-dir defect was not reported — the gate has gone blind to "
        "a non-train_* script again"
    )
    assert undeclared["--out-dir"] == "molmetal/scripts/r4_lambda_only_run.py"
    assert report["verdict"] == "FAIL"


def test_the_krasnov_out_defect_is_found(gate, tmp_path):
    """Historical defect 2: ``--out`` on a script that declares
    ``--json-dir``."""
    wrapper = _write(tmp_path, "historical_krasnov.sh", HISTORICAL_FIXTURE)
    report = gate.audit([wrapper])
    undeclared = _undeclared(report)
    assert "--out" in undeclared, (
        "the --out defect was not reported — the gate has gone blind to a "
        "non-train_* script again"
    )
    assert undeclared["--out"] == "molmetal/scripts/wf219_krasnov_aligned.py"
    assert report["verdict"] == "FAIL"


def test_both_historical_defects_are_found_in_one_fixture(gate, tmp_path):
    """Both instances, from ONE wrapper — the shape the census found."""
    wrapper = _write(tmp_path, "historical_both.sh", HISTORICAL_FIXTURE)
    report = gate.audit([wrapper])
    undeclared = _undeclared(report)
    assert set(undeclared) == {"--out-dir", "--out"}, (
        f"expected exactly the two historical flags, got {sorted(undeclared)}"
    )
    assert undeclared["--out-dir"] == "molmetal/scripts/r4_lambda_only_run.py"
    assert undeclared["--out"] == "molmetal/scripts/wf219_krasnov_aligned.py"
    assert report["n_invocations"] == 2
    assert report["n_flags_checked"] >= 4


def test_the_clean_fixture_passes(gate, tmp_path):
    """The SAME two invocations with the declared spellings must PASS.

    Without this, the gate could be a check that always refuses, which is
    not a gate.
    """
    wrapper = _write(tmp_path, "clean.sh", CLEAN_FIXTURE)
    report = gate.audit([wrapper])
    assert report["violations"] == [], (
        f"the clean fixture was reported as broken: {report['violations']}"
    )
    assert report["verdict"] == "PASS"
    assert report["n_invocations"] == 2
    assert report["n_flags_checked"] >= 4


# ---------------------------------------------------------------------------
# 2. Fail-closed: the coverage gap is REPORTED, not swallowed
#
# Wf-A269 wave-19 (2026-11-04) — THE VOCABULARY-STATE CORRECTION.
#
# The superseded test asserted that ANY script whose options the gate
# cannot read is a coverage gap.  Its fixture was
# ``molmetal/scripts/eval_all_evaluators.py``, and MEASURED 2026-11-04:
# that script has NO ``argparse``, NO ``ArgumentParser``, NO
# ``add_argument`` and NO read of ``sys.argv``.  Its option vocabulary is
# therefore empty AND KNOWN to be empty — it was never an unreadable
# vocabulary, and the gate was reporting a gap it could have DECIDED
# (the same false-attribution class as §6.269.28.5).
#
# MEASURED 2026-11-04 over ``molmetal/scripts/*.py``: 134 scripts declare
# no options and do not read argv (an EMPTY vocabulary), 12 declare no
# options but DO read argv (an UNREADABLE one), and 1 has a parser whose
# option strings are not literals (also UNREADABLE).  The three states
# have three different verdicts, and the gate now tells them apart.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#
#   def test_an_unreadable_parser_is_a_coverage_gap(gate, tmp_path):
#       """A script whose options cannot be read is a VIOLATION, never "clean"."""
#       wrapper = _write(tmp_path, "gap.sh", COVERAGE_GAP_FIXTURE)
#       report = gate.audit([wrapper])
#       assert "coverage_gap" in _kinds(report), (
#           "a script with no readable option vocabulary was treated as clean — "
#           "a gate that skips what it cannot read can PASS WITHOUT CHECKING"
#       )
#       assert report["verdict"] == "FAIL"
# ---------------------------------------------------------------------------
def test_an_unreadable_option_vocabulary_is_a_coverage_gap(gate, tmp_path):
    """A script whose options CANNOT be enumerated is a VIOLATION, never "clean".

    ``molmetal/scripts/registry.py`` reads ``sys.argv`` on its own, so the
    gate cannot know what it accepts — and a check that cannot check must
    not certify the launch.
    """
    wrapper = _write(tmp_path, "gap.sh", UNREADABLE_VOCABULARY_FIXTURE)
    report = gate.audit([wrapper])
    assert "coverage_gap" in _kinds(report), (
        "a script whose option vocabulary could not be enumerated was "
        "treated as clean — a gate that skips what it cannot read can PASS "
        "WITHOUT CHECKING"
    )
    assert report["verdict"] == "FAIL"


def test_an_empty_option_vocabulary_with_no_flag_is_legal(gate, tmp_path):
    """An EMPTY vocabulary + a flagless invocation violates nothing.

    MEASURED 2026-11-04: ``eval_all_evaluators.py`` declares no options and
    does not read argv, so the line passes no flag and there is nothing the
    gate could certify — reporting a violation here would be the gate
    inventing a defect.
    """
    wrapper = _write(tmp_path, "empty.sh", EMPTY_VOCABULARY_FIXTURE)
    report = gate.audit([wrapper])
    assert report["violations"] == [], (
        f"an empty vocabulary with no flag was reported as broken: "
        f"{report['violations']}"
    )
    assert report["verdict"] == "PASS"


def test_an_empty_option_vocabulary_with_a_flag_is_a_violation(gate, tmp_path):
    """An EMPTY vocabulary + a flag ⇒ the flag is undeclared and IGNORED.

    The launch does not abort (there is no parser to abort with
    "unrecognized arguments") — the flag simply does nothing, which is the
    quieter and therefore worse failure.
    """
    wrapper = _write(tmp_path, "emptyflag.sh", EMPTY_VOCABULARY_WITH_FLAG_FIXTURE)
    report = gate.audit([wrapper])
    undeclared = _undeclared(report)
    assert "--config" in undeclared, (
        f"a flag passed to a script with NO option vocabulary was not "
        f"reported: {report['violations']}"
    )
    assert report["verdict"] == "FAIL"


def test_an_unreadable_bash_array_is_a_coverage_gap(gate, tmp_path):
    """The flags inside an unreadable array are invisible, so the invocation
    cannot be certified."""
    wrapper = _write(tmp_path, "opaque.sh", OPAQUE_ARRAY_FIXTURE)
    report = gate.audit([wrapper])
    gaps = [v for v in report["violations"] if v["kind"] == "coverage_gap"]
    assert gaps, "an unreadable bash array was treated as clean"
    assert any("OPAQUE_ARGS" in v["detail"] for v in gaps), (
        f"the coverage gap does not name the array it could not read: {gaps}"
    )
    assert report["verdict"] == "FAIL"


def test_an_undecidable_command_is_fail_closed(gate, tmp_path):
    """A command position the gate cannot resolve is reported, never clean."""
    wrapper = _write(tmp_path, "undecidable.sh", UNDECIDABLE_FIXTURE)
    report = gate.audit([wrapper])
    assert "undecidable_script" in _kinds(report), (
        "an undecidable command position was treated as clean — the gate "
        "must never certify a line it cannot read"
    )
    assert report["verdict"] == "FAIL"


def test_an_undecidable_command_position_is_undecidable(gate):
    """Unit level: no candidate token after the unresolved command."""
    resolution = gate._resolve_segment(['"${CMD}"'], {})
    assert resolution.status == "undecidable"


def test_the_gate_looks_past_an_unresolved_interpreter(gate):
    """THE LOOKAHEAD.

    The command position holds an interpreter this gate cannot resolve, so
    the script is the NEXT non-option token.  This is the blind spot of
    ``scripts/_yaml_only_gate.py::_script_index``, which returns the
    interpreter token itself and never sees the script.
    """
    resolution = gate._resolve_segment(
        ['"${PY}"', "molmetal/scripts/wf219_krasnov_aligned.py", "--out", "x"],
        {},
    )
    assert resolution.status == "literal", (
        "the gate did not look past the unresolved interpreter"
    )
    assert resolution.token == "molmetal/scripts/wf219_krasnov_aligned.py"


# ---------------------------------------------------------------------------
# 3. The AGENTS.md §4 stdout + exit-code contract
# ---------------------------------------------------------------------------
def test_the_exit_codes_follow_agents_md_section_4(gate, tmp_path, capsys):
    """0=PASS, 1=FAIL, and the stdout carries the literal verdict word."""
    clean = _write(tmp_path, "clean.sh", CLEAN_FIXTURE)
    rc = gate.main(["--wrapper", str(clean)])
    out = capsys.readouterr().out
    assert rc == 0, f"a clean wrapper must exit 0, got {rc}"
    assert "PASS" in out

    broken = _write(tmp_path, "broken.sh", HISTORICAL_FIXTURE)
    rc = gate.main(["--wrapper", str(broken)])
    out = capsys.readouterr().out
    assert rc == 1, f"a broken wrapper must exit 1, got {rc}"
    assert "FAIL" in out


def test_an_empty_population_is_a_skip(gate):
    """Nothing scanned => SKIP, never PASS (a gate that certifies nothing
    must not claim it certified everything)."""
    report = gate.audit([])
    assert report["verdict"] == "SKIP"
    assert report["n_violations"] == 0


def test_the_coverage_line_is_printed_in_the_owner_style(gate, tmp_path, capsys):
    """The denominator line must be the owner's, so the two §5.11 gates can
    never disagree about what they audited."""
    clean = _write(tmp_path, "clean.sh", CLEAN_FIXTURE)
    gate.main(["--wrapper", str(clean)])
    out = capsys.readouterr().out
    assert "[script-contract] COVERAGE:" in out
    assert "tracked *.sh in the repo" in out
    assert "scanned" in out and "excluded" in out
    assert "source:" in out


def test_a_missing_wrapper_is_reported_not_ignored(gate, tmp_path):
    """A wrapper the gate cannot read is a violation, not a skip."""
    report = gate.audit([tmp_path / "does_not_exist.sh"])
    assert "missing_wrapper" in _kinds(report)
    assert report["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# 4. The HAZARD: the gate must not IMPORT the script it audits
# ---------------------------------------------------------------------------
def test_the_gate_never_imports_the_script_it_audits(gate, tmp_path):
    """Importing a repo script can WRITE FILES.

    ``molmetal/scripts/wf394a3_wf299_pre_launch_verify.py`` runs its whole
    check at import time and rewrites
    ``molmetal/reports/wf394a3_wf299_pre_launch/result.json``.  The gate
    therefore reads the declared options by AST and never imports.
    """
    target = "wf394a3_wf299_pre_launch_verify"
    report_file = (
        REPO_ROOT / "molmetal" / "reports" / "wf394a3_wf299_pre_launch" / "result.json"
    )
    before = report_file.stat().st_mtime_ns if report_file.exists() else None

    wrapper = _write(
        tmp_path,
        "hazard.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        f'"${{PY}}" molmetal/scripts/{target}.py\n',
    )
    gate.audit([wrapper])

    assert not [name for name in sys.modules if target in name], (
        "the gate IMPORTED the script it audited — importing a repo script "
        "can rewrite the repo's report files"
    )
    if before is not None:
        assert report_file.stat().st_mtime_ns == before, (
            "the gate's audit changed a repo report file — that is what "
            "importing the script would have done"
        )


def test_the_gate_does_not_import_torch(gate):
    """The gate must stay CPU-only and import-light.

    The check is on the GATE's own source, not on ``sys.modules``: the
    surrounding suite may legitimately have loaded torch, and a check that
    can be satisfied by somebody else's import proves nothing.
    """
    source = GATE_PATH.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch" not in gate.__dict__, "the gate module holds a torch reference"


# ---------------------------------------------------------------------------
# 5. The live tree: the two historical defects are REPAIRED
# ---------------------------------------------------------------------------
def test_the_two_historical_defects_are_repaired_on_the_live_tree(gate):
    """MEASURED 2026-11-04: the main thread repaired both instances.

    This test locks the REPAIR, not the gate: it asserts that the live
    wrappers no longer carry the flags that argparse would have refused.
    The comment lines are stripped first — AGENTS.md §5.5 rule #6 requires
    the superseded rendering to be PRESERVED VERBATIM inside a comment, so
    the raw file text still carries ``--out``.
    """
    smoke = REPO_ROOT / "molmetal" / "scripts" / "smoke" / "smoke_lambda.sh"
    smoke_live = _without_comments(smoke.read_text(encoding="utf-8"))
    assert not re.search(r"(?<![\w-])--out-dir(?![\w-])", smoke_live), (
        "molmetal/scripts/smoke/smoke_lambda.sh still passes --out-dir to a "
        "script that declares --output-dir"
    )
    assert re.search(r"(?<![\w-])--output-dir(?![\w-])", smoke_live)

    launcher = REPO_ROOT / "scripts" / "launch_wf306b_wf219_ru_gpu.sh"
    launcher_live = _without_comments(
        launcher.read_text(encoding="utf-8")
    )
    assert not re.search(r"(?<![\w-])--out(?![\w-])", launcher_live), (
        "scripts/launch_wf306b_wf219_ru_gpu.sh still passes --out"
    )
    assert re.search(r"(?<![\w-])--json-dir(?![\w-])", launcher_live)


def test_the_repaired_live_wrappers_pass_the_gate(gate):
    """The two repaired wrappers must now be clean under the gate."""
    wrappers = [
        REPO_ROOT / "molmetal" / "scripts" / "smoke" / "smoke_lambda.sh",
        REPO_ROOT / "scripts" / "launch_wf306b_wf219_ru_gpu.sh",
    ]
    for wrapper in wrappers:
        if not wrapper.is_file():
            pytest.skip(f"{wrapper} is not in the tree")
    report = gate.audit(wrappers)
    undeclared = _undeclared(report)
    assert not undeclared, (
        f"a repaired wrapper still carries an undeclared flag: {undeclared}"
    )


# ---------------------------------------------------------------------------
# 6. The gate must not be fooled by the shell
# ---------------------------------------------------------------------------
def test_a_flag_in_a_quoted_value_is_not_a_flag(gate, tmp_path):
    """``--flags "--config --out-dir"`` passes ONE token that happens to
    contain two flag-shaped strings.  Counting those would report a legal
    launch as broken."""
    wrapper = _write(
        tmp_path,
        "value.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        '"${PY}" molmetal/scripts/wf_argv_capability_check.py '
        '--flags "--config --out-dir"\n',
    )
    report = gate.audit([wrapper])
    assert "--config" not in _undeclared(report)
    assert "--out-dir" not in _undeclared(report)


def test_a_heredoc_body_is_data_not_commands(gate, tmp_path):
    """A heredoc body holds DATA.  Reading it as shell commands is how a
    gate ends up reporting a python program as an undecidable invocation."""
    wrapper = _write(
        tmp_path,
        "heredoc.sh",
        "#!/usr/bin/env bash\n"
        "cat <<'PYEOF'\n"
        "import sys\n"
        "print('--not-a-flag')\n"
        "PYEOF\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        '"${PY}" molmetal/scripts/wf219_krasnov_aligned.py --json-dir x\n',
    )
    report = gate.audit([wrapper])
    assert report["verdict"] == "PASS", (
        f"the heredoc body was read as commands: {report['violations']}"
    )


def test_a_command_substitution_body_is_scanned(gate, tmp_path):
    """``pf_verdict="$( python .../foo.py ... )"`` is CODE, not a string.

    The owner's tokenizer keeps the whole assignment as ONE token, so the
    body is invisible unless it is extracted explicitly.
    """
    wrapper = _write(
        tmp_path,
        "substitution.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        'verdict="$("${PY}" molmetal/scripts/wf219_krasnov_aligned.py '
        '--out "${REPORT_DIR}")"\n',
    )
    report = gate.audit([wrapper])
    undeclared = _undeclared(report)
    assert "--out" in undeclared, (
        "a script invoked inside a command substitution was not audited — "
        "the gate skipped it"
    )


# ---------------------------------------------------------------------------
# 9. ARGPARSE'S OWN ARITY RULES (Wf-A269 wave-19, 2026-11-04)
# ---------------------------------------------------------------------------
def test_a_value_containing_a_space_is_a_positional_for_argparse(gate, tmp_path):
    """THE SPACE RULE.

    argparse's ``_parse_optional`` returns None — "this is a POSITIONAL" —
    for a token that CONTAINS A SPACE, so such a token is ACCEPTED as the
    value of a value-taking option.  MEASURED 2026-11-04 on CPython 3:

        p.add_argument("--flags", action="append")
        p.parse_args(["--flags", "--config --out-dir"])
        -> Namespace(flags=['--config --out-dir'])

    This is the repo's OWN legal form:
    ``molmetal/scripts/wf_argv_capability_check.py`` declares ``--flags`` as
    a whitespace-separated STRING payload and its help text says so.

    PRE-REPAIR the gate reported this as "the option was passed without a
    value", which is a false attribution on a launch that works — the
    §6.269.28.5 class.
    """
    wrapper = _write(
        tmp_path,
        "space_value.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        'FLAGS="--config --out-dir"\n'
        '"${PY}" molmetal/scripts/wf_argv_capability_check.py \\\n'
        '    --trainer molmetal/scripts/train_cfm_yuelbond_wasserstein.py \\\n'
        '    --flags "${FLAGS}"\n',
    )
    report = gate.audit([wrapper])
    missing = [v for v in report["violations"] if v["kind"] == "missing_value"]
    assert not missing, (
        "a value that contains a space is a POSITIONAL for argparse, so it "
        f"IS the value of --flags; reporting it as a missing value is a "
        f"false attribution: {missing}"
    )


def test_a_quoted_flag_blob_is_disclosed_not_fatal(gate, tmp_path):
    """The same blob is DISCLOSED, not fatal.

    A quoted expansion stays ONE argv word, so argparse reads it as a
    positional and the flags inside it never reach the script as options.
    Whether that is a bug or a deliberate string payload is NOT decidable
    from the argv alone (the repo has both), so the gate counts it and
    prints it on every run (no silent caps) instead of failing the launch.
    """
    wrapper = _write(
        tmp_path,
        "blob.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        'FLAGS="--config --out-dir"\n'
        '"${PY}" molmetal/scripts/wf_argv_capability_check.py \\\n'
        '    --trainer molmetal/scripts/train_cfm_yuelbond_wasserstein.py \\\n'
        '    --flags "${FLAGS}"\n',
    )
    report = gate.audit([wrapper])
    disclosed = report.get("disclosed", [])
    assert disclosed, (
        "a quoted flag blob was silently dropped — the gate must disclose "
        "it on every run (no silent caps)"
    )
    assert any(item["kind"] == "quoted_flag_blob" for item in disclosed), disclosed
    assert report["verdict"] == "PASS", (
        "a quoted flag blob is a stray POSITIONAL, not an undeclared FLAG; "
        f"failing the launch on it is a false attribution: {report['violations']}"
    )


def test_an_option_looking_value_is_a_missing_value(gate, tmp_path):
    """The NEGATIVE case of the space rule: a value-taking option followed
    by an option-looking word WITHOUT a space.

    argparse refuses an option-looking word there ("argument --x: expected
    one argument"), so the launch dies before the run starts — MEASURED
    2026-11-04 on CPython 3.
    """
    wrapper = _write(
        tmp_path,
        "missing_value.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        '"${PY}" molmetal/scripts/wf_argv_capability_check.py \\\n'
        "    --trainer --flags x\n",
    )
    report = gate.audit([wrapper])
    missing = [v for v in report["violations"] if v["kind"] == "missing_value"]
    assert missing, (
        "an option-looking word in a value position was not reported: "
        f"{report['violations']}"
    )
    assert missing[0]["flag"] == "--trainer", missing
    assert report["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# 10. THE REVIEWER ROUND (Wf-A269 wave-21, 2026-11-04)
#
# Two reviewers attacked the gate: one with a FALSE-NEGATIVE lens, one with
# a FALSE-POSITIVE lens.  Every finding below was RE-PRODUCED against the
# live gate before it was accepted, and every REJECTED finding says why.
# The fixtures live in this file (a pytest ``tmp_path``), so the test does
# not depend on any other test file and does not modify anything.
# ---------------------------------------------------------------------------
def _run(gate, tmp_path, name: str, text: str) -> dict:
    return gate.audit([_write(tmp_path, name, text)])


# --- Reviewer 1, finding M-1 / reviewer 2, finding FN-1: RUNTIME TRANSPORT ---
def test_a_bash_c_payload_is_audited_not_skipped(gate, tmp_path):
    """``bash -c "$TRAINER_CMD"`` HIDES the whole invocation.

    MEASURED 2026-11-04: the gate classified a shell interpreter as a
    non-fatal counter, so the command string the wrapper actually launches
    at RUN TIME was never parsed — the exact transport §6.269.23 documents
    (the CFM trainer had no way to switch the ``events.jsonl`` sink off).

    The transport is live at ``scripts/start.sh:1053`` and
    ``scripts/_wf_r28_gpu_master.sh:183``.
    """
    report = _run(
        gate,
        tmp_path,
        "runtime.sh",
        "#!/usr/bin/env bash\n"
        'TRAINER_CMD="/x/.venv/bin/python -u '
        "molmetal/scripts/r4_lambda_only_run.py --bogus-runtime-flag\"\n"
        'nohup bash -c "$TRAINER_CMD" &\n',
    )
    assert "--bogus-runtime-flag" in _undeclared(report), (
        "a command assembled at RUN TIME and launched through bash -c was "
        f"not audited: {report['violations']}"
    )
    assert report["n_runtime_payloads"] >= 1
    assert report["verdict"] == "FAIL"


# --- Reviewer 1, finding M-3: eval is a RUNTIME TRANSPORT, not DATA ---
def test_an_eval_payload_is_audited_not_skipped(gate, tmp_path):
    """``eval "$CMD"`` evaluates a command STRING at RUN TIME.

    ``eval`` is in the owner module's ``_DATA_COMMANDS``, so the segment
    resolved to "none" and the whole invocation was out of contract.
    """
    report = _run(
        gate,
        tmp_path,
        "eval.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        'eval "$PY -u molmetal/scripts/r4_lambda_only_run.py --out-dir /tmp/out"\n',
    )
    assert "--out-dir" in _undeclared(report), (
        f"an eval'd command string was not audited: {report['violations']}"
    )


# --- Reviewer 1, finding M-4: a script path relative to a cd ---
def test_a_script_path_relative_to_a_cd_is_in_contract(gate, tmp_path):
    """``cd molmetal/scripts`` then ``python r4_lambda_only_run.py``.

    The gate resolved relative paths against the repo root only, so this
    invocation was out of contract.
    """
    report = _run(
        gate,
        tmp_path,
        "cd_relative.sh",
        "#!/usr/bin/env bash\n"
        "cd molmetal/scripts\n"
        "python r4_lambda_only_run.py --out-dir /tmp/out\n",
    )
    assert report["n_invocations"] == 1, (
        "a script named relative to the wrapper's working directory was not "
        f"counted as an invocation: {report['violations']}"
    )
    assert "--out-dir" in _undeclared(report)


# --- Reviewer 1, finding M-5 / reviewer 2, finding FN-3: python -m ---
def test_the_module_form_is_in_contract(gate, tmp_path):
    """``python -m molmetal.scripts.X`` names the same repo-local script."""
    report = _run(
        gate,
        tmp_path,
        "dashm.sh",
        "#!/usr/bin/env bash\n"
        "python -m molmetal.scripts.r4_lambda_only_run --out-dir /tmp/out\n",
    )
    assert report["n_invocations"] == 1
    assert "--out-dir" in _undeclared(report)


# --- Reviewer 1, finding M-6: a WHOLE COMMAND in a variable ---
def test_a_whole_command_in_a_variable_is_in_contract(gate, tmp_path):
    """``CMD="python ... --out-dir /tmp/out"; $CMD`` — the variable holds a
    command, not a path."""
    report = _run(
        gate,
        tmp_path,
        "cmdvar.sh",
        "#!/usr/bin/env bash\n"
        'CMD="/x/.venv/bin/python -u '
        "molmetal/scripts/r4_lambda_only_run.py --out-dir /tmp/out\"\n"
        "$CMD\n",
    )
    assert "--out-dir" in _undeclared(report), (
        f"a command held in a variable was not audited: {report['violations']}"
    )


# --- Reviewer 1, finding M-7: a bash array WITHOUT the [@] subscript ---
def test_an_array_without_the_subscript_is_in_contract(gate, tmp_path):
    """``$ARGS`` expands to the array's elements joined by spaces, so the
    flags inside it ARE separate argv words."""
    report = _run(
        gate,
        tmp_path,
        "array_no_subscript.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        "ARGS=(--out-dir /tmp/out)\n"
        '"${PY}" -u molmetal/scripts/r4_lambda_only_run.py $ARGS\n',
    )
    assert "--out-dir" in _undeclared(report), (
        f"an array referenced without [@] was invisible: {report['violations']}"
    )


# --- Reviewer 1, finding M-8: the branch union of a SCALAR ---
def test_a_scalar_branch_union_is_audited(gate, tmp_path):
    """A scalar assigned in two branches must be audited in BOTH.

    MEASURED 2026-11-04: only the LAST assignment was checked, so the
    defective branch was invisible.
    """
    report = _run(
        gate,
        tmp_path,
        "branch_scalar.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        'if [[ -n "${WHICH:-}" ]]; then\n'
        '  SCRIPT="molmetal/scripts/r4_lambda_only_run.py"\n'
        "else\n"
        '  SCRIPT="molmetal/scripts/wf219_krasnov_aligned.py"\n'
        "fi\n"
        '"${PY}" -u "$SCRIPT" --json-dir /tmp/out\n',
    )
    assert report["n_invocations"] == 2, (
        f"the branch union of a scalar was not audited: {report['violations']}"
    )
    assert "--json-dir" in _undeclared(report), (
        "the branch that names a script NOT declaring --json-dir was "
        f"invisible: {report['violations']}"
    )


# --- Reviewer 1, finding M-9: a single-dash option ---
def test_a_single_dash_option_is_collected(gate, tmp_path):
    """``-Z`` is an option for argparse (``-Z`` ⇒ SystemExit(2))."""
    report = _run(
        gate,
        tmp_path,
        "shortflag.sh",
        "#!/usr/bin/env bash\n"
        "python molmetal/scripts/r4_lambda_only_run.py -Z /tmp/out\n",
    )
    assert "-Z" in _undeclared(report), (
        f"a single-dash option was not collected: {report['violations']}"
    )


# --- Reviewer 1, finding M-2 (the scalar case that was STILL open) ---
def test_an_unresolved_flag_variable_is_a_coverage_gap(gate, tmp_path):
    """``$EXTRA_FLAGS`` standing where an option could stand.

    MEASURED 2026-11-04 (reviewer finding M-2): the gate was fail-closed
    on an unresolved SCRIPT token but not on an unresolved FLAG token, so
    ``"$PY" -u molmetal/scripts/foo.py $EXTRA_FLAGS`` was reported as PASS
    with ZERO flags checked — the flag was invisible AND uncounted.

    THE REPAIR, and the trap in it: ``GATE._expand`` leaves an unresolvable
    reference IN PLACE, so ``expanded == raw`` and the "the expansion
    changed something" guard never fired for exactly the case the reviewer
    named.  The gate now tests the EXPANDED text for a surviving
    reference, and a reference standing where an option could stand is a
    fail-closed coverage gap.
    """
    report = _run(
        gate,
        tmp_path,
        "flagvar.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        '"${PY}" -u molmetal/scripts/r4_lambda_only_run.py $EXTRA_FLAGS\n',
    )
    gaps = [v for v in report["violations"] if v["kind"] == "coverage_gap"]
    assert gaps, (
        "an unresolved shell reference where an option could stand was "
        f"treated as clean: {report['violations']}"
    )
    assert any("EXTRA_FLAGS" in v["detail"] for v in gaps), gaps
    assert report["verdict"] == "FAIL"


def test_an_unresolved_value_position_is_harmless(gate, tmp_path):
    """The NEGATIVE case of the rule above: a reference standing in a VALUE
    position is harmless — the option takes whatever the value turns out to
    be, so there is nothing to certify and nothing to report."""
    report = _run(
        gate,
        tmp_path,
        "valuevar.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        '"${PY}" -u molmetal/scripts/r4_lambda_only_run.py --config $CFG\n',
    )
    assert report["violations"] == [], (
        "an unresolved reference in a VALUE position was reported: "
        f"{report['violations']}"
    )
    assert report["verdict"] == "PASS"


# --- Reviewer 1, finding B-1: requirement #4 is UNCONDITIONAL ---
def test_an_unreadable_vocabulary_is_a_coverage_gap_even_without_flags(gate, tmp_path):
    """Requirement #4: "a script whose declared flags cannot be read =>
    VIOLATION (kind coverage_gap), fail-closed".

    MEASURED 2026-11-04 (reviewer finding B-1): the gate made the gap
    NON-FATAL when the invocation carried no flags, which contradicts
    requirement #4 and re-opens the sibling gate's "a check that can PASS
    WITHOUT CHECKING" shape.

    The distinction that DOES hold is the one the gate already draws: an
    EMPTY vocabulary (provably no parser, no ``add_argument``, no read of
    ``sys.argv``) is DECIDABLE, and a flagless invocation of it is legal —
    see :func:`test_an_empty_option_vocabulary_with_no_flag_is_legal`.
    ``molmetal/scripts/registry.py`` is NOT that case: it reads ``sys.argv``
    on its own, so its vocabulary is UNREADABLE.
    """
    report = _run(
        gate,
        tmp_path,
        "gap_noflags.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        '"${PY}" molmetal/scripts/registry.py\n',
    )
    assert "coverage_gap" in _kinds(report), (
        "an unreadable option vocabulary was treated as clean even with no "
        f"flags on the line: {report['violations']}"
    )
    assert report["verdict"] == "FAIL"


# --- Reviewer 1, finding B-2: the counter must reach the REPORT ---
def test_the_vocabulary_state_counters_reach_the_report(gate, tmp_path):
    """The "DISCLOSED BUT NOT ENFORCED" class (§6.269.22): a counter that
    never reaches the report makes the in-code claim "counted, never hidden"
    FALSE."""
    report = _run(
        gate,
        tmp_path,
        "gap_noflags.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        '"${PY}" molmetal/scripts/registry.py\n',
    )
    for key in (
        "n_vocab_empty",
        "n_vocab_unreadable",
        "n_vocab_broken",
        "n_vocab_partial",
        "n_vocab_no_flags",
    ):
        assert key in report, f"{key} never reaches the report"
        assert isinstance(report[key], int)
    assert report["n_vocab_unreadable"] == 1
    assert report["n_vocab_no_flags"] == 1


# --- Reviewer 2, finding FP-1: an option string held in a VARIABLE ---
def test_an_option_string_held_in_a_variable_is_read(gate, tmp_path):
    """``molmetal/scripts/train_dmpnn_multitask_wasserstein.py`` declares
    four real options through ``for _flag, _dest in (...)``.

    MEASURED 2026-11-04 (reviewer finding FP-1): the AST scan read only
    string LITERAL option strings, so the four REAL options were invisible
    and the gate reported them as undeclared — a false positive on a script
    that does declare them.
    """
    report = _run(
        gate,
        tmp_path,
        "variable_addarg.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        '"${PY}" molmetal/scripts/train_dmpnn_multitask_wasserstein.py '
        "--use-triton-bond-head\n",
    )
    assert "--use-triton-bond-head" not in _undeclared(report), (
        "an option declared through a variable was reported as undeclared: "
        f"{report['violations']}"
    )
    assert report["verdict"] == "PASS"


# --- Reviewer 2, finding FP-2: the -- end-of-options marker ---
def test_the_double_dash_marker_ends_option_parsing(gate, tmp_path):
    """argparse stops parsing options at a bare ``--``: every token after it
    is a POSITIONAL."""
    report = _run(
        gate,
        tmp_path,
        "ddash.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        '"${PY}" molmetal/scripts/r4_lambda_only_run.py '
        "--output-dir /tmp/x -- --not-a-flag\n",
    )
    assert "--not-a-flag" not in _undeclared(report), (
        "a positional after -- was reported as an undeclared option: "
        f"{report['violations']}"
    )
    assert report["verdict"] == "PASS"


# --- Reviewer 2, finding FP-3: a VALUE that starts with -- ---
def test_an_option_looking_value_is_not_an_undeclared_flag(gate, tmp_path):
    """The diagnosis must be the missing VALUE, not an unknown option.

    argparse's real error is "argument --output-dir: expected one
    argument" — the launch dies, but for the missing value.
    """
    report = _run(
        gate,
        tmp_path,
        "value_dashdash.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        '"${PY}" molmetal/scripts/r4_lambda_only_run.py --output-dir --weird-dir\n',
    )
    assert "--weird-dir" not in _undeclared(report), (
        "the VALUE was reported as an undeclared option: "
        f"{report['violations']}"
    )
    missing = [v for v in report["violations"] if v["kind"] == "missing_value"]
    assert missing and missing[0]["flag"] == "--output-dir", report["violations"]


# --- Reviewer 2, finding FP-5: a script that does NOT EXIST ---
def test_a_non_existent_script_is_its_own_kind(gate, tmp_path):
    """The defect is "No such file or directory", NOT "the gate could not
    read the declared options" — the second implies the file is there."""
    report = _run(
        gate,
        tmp_path,
        "missing_script.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        '"${PY}" molmetal/scripts/_tui_dashboard.py --foo\n',
    )
    kinds = _kinds(report)
    assert "missing_script" in kinds, kinds
    assert "coverage_gap" not in kinds, kinds
    assert report["verdict"] == "FAIL"


# --- Reviewer 2, finding FP-6: dead code (REJECTED as a defect) ---
def test_unreachable_code_is_still_reported(gate, tmp_path):
    """REJECTED as a defect, PINNED as a behaviour.

    The reviewer called the report of dead code "static conservatism;
    defensible but worth disclosing".  It is NOT repaired: a gate that
    skipped an ``if false`` branch would need a constant evaluator, and a
    skip is exactly the DEAD-GATE shape (a check that PASSES WITHOUT
    CHECKING).  The conservative report is fail-closed, and the finding is
    disclosed here instead of being silently dropped.
    """
    report = _run(
        gate,
        tmp_path,
        "deadcode.sh",
        "#!/usr/bin/env bash\n"
        'PY="${REPO_ROOT}/.venv/bin/python"\n'
        "if false; then\n"
        '  "${PY}" molmetal/scripts/r4_lambda_only_run.py --out-dir /tmp/out\n'
        "fi\n",
    )
    assert "--out-dir" in _undeclared(report), (
        "an unreachable invocation was silently skipped: "
        f"{report['violations']}"
    )


# --- Reviewer 2, finding FP-7: the --wrapper coverage percentage ---
def test_the_wrapper_coverage_line_carries_a_percentage(gate, tmp_path, capsys):
    """``None% scanned`` told a reader nothing about what the run covered."""
    clean = _write(tmp_path, "clean.sh", CLEAN_FIXTURE)
    gate.main(["--wrapper", str(clean)])
    out = capsys.readouterr().out
    assert "None%" not in out, out
    assert "% scanned" in out, out


# --- Reviewer 2, finding FN-2: the ZERO-INVOCATION disclosure ---
def test_the_zero_invocation_wrappers_are_disclosed_and_named(gate, tmp_path, capsys):
    """A glob that matches but audits nothing is the "silent caps" shape.

    MEASURED 2026-11-04: 31 of the 101 in-contract wrappers contributed no
    invocation this gate could see — including ``scripts/start.sh`` — and
    the closing line claimed a UNIVERSAL truth the data did not support.
    """
    wrapper = _write(
        tmp_path,
        "empty_wrapper.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\necho 'nothing to audit here'\n",
    )
    report = gate.audit([wrapper])
    assert report["n_zero_invocation_wrappers"] == 1
    assert report["zero_invocation_wrappers"], report["zero_invocation_wrappers"]

    gate.main(["--wrapper", str(wrapper)])
    out = capsys.readouterr().out
    assert "ZERO-INVOCATION WRAPPERS" in out, out
    assert "empty_wrapper.sh" in out, out


def test_the_closing_claim_is_scoped_to_what_the_gate_read(gate, tmp_path, capsys):
    """The closing line must not claim more than the run measured."""
    wrapper = _write(
        tmp_path,
        "empty_wrapper.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\necho 'nothing to audit here'\n",
    )
    gate.main(["--wrapper", str(wrapper)])
    out = capsys.readouterr().out
    assert "THAT THIS GATE COULD RESOLVE" in out, out


# --- the accounting nit: the census must add up ---
def test_the_census_accounting_adds_up(gate):
    """``n_out_of_contract`` must be a SUBSET of ``n_segments``.

    MEASURED 2026-11-04: the census printed "17325 out of contract" out of
    "16298 segment(s)" — an impossible line a reader could not interpret.
    """
    report = gate.audit()
    assert report["n_out_of_contract_segments"] <= report["n_segments"], (
        "the census does not add up: "
        f"{report['n_out_of_contract_segments']} out of contract out of "
        f"{report['n_segments']} segments"
    )


# --- the gate's own honest limit, pinned ---
def test_the_gate_states_its_own_coverage_limit(gate):
    """The gate must not overclaim: it audits the wrappers it can RESOLVE,
    and it discloses the rest (zero-invocation wrappers, runtime residual).

    This is the §6.269.28.5 false-attribution class: a gate that guesses
    where it could decide is worse than one that says what it cannot.
    """
    report = gate.audit()
    assert "n_zero_invocation_wrappers" in report
    assert "n_runtime_residual" in report
    assert "n_runtime_payloads" in report
    assert report["n_undecidable"] == 0, report["undecidable"]


# ---------------------------------------------------------------------------
# 11. THE LIVE TREE (Wf-A269 wave-21, 2026-11-04)
# ---------------------------------------------------------------------------
def test_the_live_tree_has_no_undeclared_flag(gate):
    """The live tree must be CLEAN of the defect class this gate closes.

    MEASURED 2026-11-04: the two historical instances were repaired by the
    main thread, and this gate reads 218 invocations / 560 flags across the
    101 in-contract wrappers with ZERO decidable violations.  A new
    ``undeclared_flag`` here is a real defect — argparse would exit 2 on it.
    """
    report = gate.audit()
    undeclared = [v for v in report["violations"] if v["kind"] == "undeclared_flag"]
    assert not undeclared, (
        "the live tree carries a flag that a repo-local script does not "
        f"declare: {undeclared}"
    )
    assert report["n_invocations"] > 0
    assert report["n_flags_checked"] > 0
    assert report["n_scripts"] > 0, (
        "the gate audited no repo-local script at all — a clean tree that "
        "the gate never looked at"
    )


def test_the_live_run_prints_the_denominator_and_the_census_rows(gate, capsys):
    """The brief: the gate must print its denominator AND the census rows.

    A reader who cannot see the coverage numbers cannot tell what the
    verdict covers (the §6.269.22 disclosure duty).
    """
    gate.main([])
    out = capsys.readouterr().out
    for needle in (
        "[script-contract] COVERAGE:",
        "tracked *.sh in the repo",
        "scanned",
        "excluded",
        "% scanned",
        "source:",
        "[script-contract] CENSUS:",
        "RUNTIME TRANSPORT:",
        "ZERO-INVOCATION WRAPPERS:",
        "[script-contract] PASS",
    ):
        assert needle in out, f"{needle!r} is missing from the live run:\n{out}"


# ---------------------------------------------------------------------------
# 12. THE ``--no-`` TWIN — a BooleanOptionalAction option DECLARES its twin,
#     and a ``store_true`` option does NOT (Wf-A269 wave-19, 2026-11-04)
# ---------------------------------------------------------------------------
#: ``molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py`` declares
#: ``--smoke`` with ``action=argparse.BooleanOptionalAction``, so argparse
#: registers BOTH ``--smoke`` AND ``--no-smoke`` from ONE ``add_argument``
#: call.  MEASURED on the real script, 2026-11-04, 0 GPU, no work done:
#: its own usage line reads ``[--smoke | --no-smoke]``, and invoking it
#: with ``--no-smoke --metals NOT_A_METAL`` fails on ``--metals``
#: ("invalid choice"), NOT on ``--no-smoke``.
BOOLEAN_OPTIONAL_FIXTURE = """#!/usr/bin/env bash
set -euo pipefail
PY="${REPO_ROOT}/.venv/bin/python"
"${PY}" molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py \\
    --metal-coord-term \\
    --metals Ru \\
    --seeds 0 \\
    --epochs 5 \\
    --no-smoke \\
    --out-dir "${OUT_DIR}"
"""

#: ``molmetal/scripts/analyze_feature_target_correlation.py`` declares
#: ``--quiet`` with ``action="store_true"``.  ``store_true`` registers NO
#: ``--no-`` twin, so ``--no-quiet`` is NOT declared — and argparse would
#: abort with "unrecognized arguments: --no-quiet".  This is the OTHER
#: direction of the same rule, and it is the direction that makes the first
#: one meaningful: a gate that made every ``--no-X`` legal would be blind
#: to a real defect, and a gate that made every ``--no-X`` illegal would
#: report a defect that does not exist.
STORE_TRUE_FIXTURE = """#!/usr/bin/env bash
set -euo pipefail
PY="${REPO_ROOT}/.venv/bin/python"
"${PY}" molmetal/scripts/analyze_feature_target_correlation.py \\
    --no-quiet \\
    --out-dir "${OUT_DIR}"
"""


def test_a_boolean_optional_action_option_declares_its_no_twin(gate, tmp_path):
    """``--no-smoke`` IS declared by the scaffold — the launch does NOT die.

    This is the rule a wave-19 REVIEW report got WRONG.  That report
    asserted that ``--no-smoke`` was "genuinely undeclared (no declared
    option is a prefix of it)" and named it a live launch-killer; the
    claim was then published in
    ``molmetal/scripts/pre_flight_check.py``'s gate docstring, where it
    stood as a tracked-file statement about a defect that does not exist
    (corrected there on 2026-11-04, additively, per AGENTS.md §5.5
    rule #6).

    The gate's CODE was always right about this — the vocabulary
    extractor models ``BooleanOptionalAction`` — so the false claim was
    confined to the prose.  This test pins the CODE, so the prose cannot
    be re-derived wrongly a second time.
    """
    declared, error, state = gate.declared_flags(
        REPO_ROOT / "molmetal" / "scripts" / "wf406a4_metal_coord_retrain_scaffold.py"
    )
    assert state == "declared", f"the scaffold's vocabulary is {state}: {error}"
    assert "--smoke" in declared
    assert "--no-smoke" in declared, (
        "the BooleanOptionalAction twin is missing from the vocabulary — "
        "the gate would report a legal launch as broken"
    )

    wrapper = _write(tmp_path, "boolean_optional.sh", BOOLEAN_OPTIONAL_FIXTURE)
    report = gate.audit([wrapper])
    undeclared = _undeclared(report)
    assert "--no-smoke" not in undeclared, (
        "the gate reported a DECLARED flag as undeclared: " f"{undeclared}"
    )
    assert report["verdict"] == "PASS", report["violations"]


def test_a_store_true_option_does_not_declare_a_no_twin(gate, tmp_path):
    """The OTHER direction: ``store_true`` registers NO ``--no-`` twin.

    Without this test, the rule above could be satisfied by a gate that
    simply treated every ``--no-X`` as legal — which would be a gate that
    cannot fail on the whole ``--no-`` family.
    """
    declared, error, state = gate.declared_flags(
        REPO_ROOT
        / "molmetal"
        / "scripts"
        / "analyze_feature_target_correlation.py"
    )
    assert state == "declared", f"the vocabulary is {state}: {error}"
    assert "--quiet" in declared
    assert "--no-quiet" not in declared, (
        "a store_true option grew a --no- twin it does not have"
    )

    wrapper = _write(tmp_path, "store_true.sh", STORE_TRUE_FIXTURE)
    report = gate.audit([wrapper])
    undeclared = _undeclared(report)
    assert "--no-quiet" in undeclared, (
        "the gate did not report an UNDECLARED flag: " f"{undeclared}"
    )
    assert undeclared["--no-quiet"] == (
        "molmetal/scripts/analyze_feature_target_correlation.py"
    )
    assert report["verdict"] == "FAIL"
