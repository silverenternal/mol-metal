# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2026 Mol-Metal authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Wf-A269 wave-28 — guards for the PUBLISHED-NUMBER ↔ THE-FILE-IT-DESCRIBES class.

Run ONLY this file, and always with the GPU hidden::

    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
        -m pytest molmetal/tests/test_wfa269_wave28_published_number_guards.py -v

WHY THIS FILE EXISTS (the §6.269.38 class).

The wave-26/27 census found three places where a PUBLISHED number no longer
described the file it names.  Two were producer defects, one was an artefact
lag.  The producers are repaired; **the repair alone does not stop the class
from recurring** — a repaired producer with nothing watching it drifts back,
and that is the §6.269.29 "a guard that no runner executes" shape.  So this
file is the MECHANISM: it pins each repaired invariant to a value that is
RE-MEASURED at test time, so the number cannot go stale away from the file
again.

TEN GROUPS (A-J), each pinning one invariant.

  (Wf-A269 wave-29, 2026-11-07 — the previous headline of this list read
  "EIGHT GROUPS".  That was ALREADY wrong before this wave: groups A-I are
  NINE.  The superseded rendering is PRESERVED VERBATIM per AGENTS.md §5.5
  rule #6 and the correction is ADDITIVE + DATED: the count is TEN (A-J),
  group J being this wave's addition.)

  A  R1  the Wf-299 wrapper's byte count is MEASURED AT RUN TIME, not a
          literal, and the live rendering interpolates the measurement.
  B  R2a the canonical evaluator script's summary line names its POPULATION
          (PROBES), not the wrong label (evaluators).
  C  R2b the Wf-395-A1 audit MEASURES the canonical probe count from the
          source (cross-checked against an INDEPENDENT AST oracle, §6.269.35.4)
          and NAMES every population; the legacy token keeps its NAME + VALUE.
  D  R3  the Wf-A167 command-prefix guard's verdict axis follows the SELECTED
          flag, so the CLI can exit 0 on the LIVE axis (a gate that cannot
          PASS is as dead as a gate that cannot FAIL).
  E  THIS FILE is named by a CI job, and that job is in ``ci-summary``'s
          ``needs:`` AND in its summary loop.
  F  EVERY test path named in ``.github/workflows/ci.yml`` exists on disk —
          the general mechanism behind the dangling ``test_rocm_lipman.py``.
  G  every reason bucket the trainer's ``_rdkit_strict_reason`` docstring
          documents is a string the function can actually return.
  H  the audit's synthetic probe input satisfies the contract the CALLEE
          declares (a non-empty ``cohort.pockets`` / ``cohort.seeds``).
  I  the gate-inventory ARTEFACT carries its PROVENANCE — a production
          timestamp plus a pin on BOTH the producer and the gate SURFACE it
          describes — so a stale artefact can no longer read as a current one.
  J  the artefact's DERIVED ``source_line`` pointers are COVERED — REPORTED
          as informational drift, never ASSERTED (the CLAIM / RESOLUTION
          split; see the group header).

CPU-ONLY BY CONSTRUCTION: this file imports NOTHING but the standard library
(``ast``, ``hashlib``, ``importlib.util``, ``json``, ``re``, ``sys``,
``pathlib``, ``warnings``) plus ``pytest`` itself (for the one documented SKIP
in group I).
The producer modules it loads are themselves stdlib-only at import time, and
the guard's own tests assert that (``test_wfa167_command_prefix_guard.py``).
MEASURED 2026-11-06: ``molmetal.scripts.pre_flight_check`` imports cleanly
with ``torch`` / ``numpy`` / ``pandas`` / ``rdkit`` / ``scipy`` / ``sklearn``
/ ``yaml`` / ``matplotlib`` ALL BLOCKED ⇒ group I's live-surface comparison
runs in the pytest-only CI venv, not only on a full developer box.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

REPO = Path(__file__).resolve().parents[2]

CI = ".github/workflows/ci.yml"
THIS_FILE = "molmetal/tests/test_wfa269_wave28_published_number_guards.py"

LAUNCHER = "molmetal/scripts/wf426a3_post_fix_retrain_launcher.sh"
EVAL_SCRIPT = "molmetal/scripts/eval_all_evaluators.py"
AUDIT = "molmetal/scripts/wf395a1_evaluator_audit.py"
CALLEE = "molmetal/scripts/wf380a4_vina_param_tuning.py"
GUARD = "molmetal/scripts/wfa167_command_prefix_guard.py"
TRAINER = "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"

#: The function whose docstring advertises the failure taxonomy.
TAXONOMY_FUNC = "_rdkit_strict_reason"

#: A stale number may only appear in a clause that ANNOUNCES it as
#: superseded/corrected — AGENTS.md §5.5 rule #6 keeps the old rendering
#: VERBATIM, but it must be labelled as superseded, not presented as live.
SUPERSEDED_MARKERS = (
    "CORRECTION",
    "SUPERSEDED",
    "hardcoded",
    "STALE",
    "stale",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    assert spec is not None and spec.loader is not None, f"cannot load {rel}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ci_job_ids(text: str) -> List[str]:
    """The job ids declared under ``jobs:`` (2-space indent, bare key)."""
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if re.match(r"^jobs:\s*$", ln))
    except StopIteration:  # pragma: no cover - the file always has `jobs:`
        return []
    out: List[str] = []
    for ln in lines[start + 1 :]:
        m = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", ln)
        if m:
            out.append(m.group(1))
    return out


def _ci_job_body(text: str, job_id: str) -> str:
    lines = text.splitlines()
    try:
        start = next(
            i
            for i, ln in enumerate(lines)
            if re.match(rf"^  {re.escape(job_id)}:\s*$", ln)
        )
    except StopIteration:  # pragma: no cover
        return ""
    body: List[str] = []
    for ln in lines[start + 1 :]:
        if re.match(r"^  [A-Za-z0-9_.-]+:\s*$", ln):
            break
        body.append(ln)
    return "\n".join(body)


def _ci_pytest_argvs(text: str) -> List[List[str]]:
    """The pytest ARGVs declared in a CI workflow, gathered across the
    shell's line continuations.

    PRECISION (Wf-A269 wave-28, 2026-11-06) — **A MENTION IS NOT AN
    INVOCATION.**  The first version of this guard scanned the workflow's
    RAW TEXT, so a path quoted inside a COMMENT was counted as a path the
    CI RUNS: this wave's own ci.yml note says
    ``molmetal/tests/test_rocm_lipman.py`` was deleted in 09c2202a, and
    the raw-text scan then demanded that the deleted file exist on disk
    (group F) and that the blackwell job be the owner of this guard file
    (group E).  Both were FALSE POSITIVES created by the guard, not
    defects in the workflow.  The repository already learned this lesson
    for the §5.9 single-venv gate ("it cannot tell an interpreter
    INVOCATION from a MENTION").

    So the scan is over the ARGV of each ``python -m pytest`` invocation.
    A line that is a comment (``#``) is skipped outright, and a command is
    gathered from its first line through the lines that end with a
    backslash.
    """
    argvs: List[List[str]] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        is_pytest = "-m pytest" in stripped or stripped.endswith(" pytest")
        if not is_pytest:
            i += 1
            continue
        block = [stripped]
        j = i + 1
        while j < n and lines[j].rstrip().endswith("\\"):
            cont = lines[j].strip()
            if cont.startswith("#"):
                j += 1
                continue
            block.append(cont)
            j += 1
        tokens: List[str] = []
        for tok in " ".join(block).replace("\\", " ").split():
            tok = tok.strip("'\"")
            if tok:
                tokens.append(tok)
        argvs.append(tokens)
        i = j if j > i else i + 1
    return argvs


def _ci_jobs_running(text: str, rel_path: str) -> List[str]:
    """The CI jobs whose pytest ARGV names ``rel_path`` (comments excluded)."""
    out: List[str] = []
    for job in _ci_job_ids(text):
        body = _ci_job_body(text, job)
        if any(rel_path in argv for argv in _ci_pytest_argvs(body)):
            out.append(job)
    return out


def _ci_summary_needs(text: str) -> List[str]:
    m = re.search(r"^\s*needs:\s*\[([^\]]*)\]", text, re.MULTILINE)
    if not m:
        return []
    return [x.strip() for x in m.group(1).split(",") if x.strip()]


def _ci_summary_loop(text: str) -> List[str]:
    m = re.search(r"for job in ([^\n;]+); do", text)
    if not m:
        return []
    return m.group(1).split()


def _yaml_list_items(text: str, key: str) -> List[str]:
    """The scalar items of a block sequence under ``key:`` (stdlib mini-YAML)."""
    out: List[str] = []
    in_block = False
    for ln in text.splitlines():
        if re.match(rf"^\s*{re.escape(key)}\s*:\s*$", ln):
            in_block = True
            continue
        if not in_block:
            continue
        m = re.match(r"^\s*-\s*(\S+)\s*$", ln)
        if m:
            out.append(m.group(1))
            continue
        if re.match(r"^\s*\S+:", ln) or (ln.strip() and not ln.startswith((" ", "\t"))):
            break
    return out


# ===========================================================================
# GROUP A — R1: the wrapper's byte count is measured at RUN TIME
# ===========================================================================
def test_the_wrapper_byte_count_is_measured_not_literalised():
    text = _read(LAUNCHER)
    assert 'WF299_BYTES="$(stat -c %s' in text, (
        "the wrapper must MEASURE the Wf-299 script's size at run time — a "
        "literal is what went stale in the first place"
    )
    assert 'WF299_MODE="$(stat -c %A' in text


def test_the_live_notes_clause_interpolates_the_measurement():
    text = _read(LAUNCHER)
    occurrences = [m.start() for m in re.finditer(r"filesystem-truth \(", text)]
    assert occurrences, "the notes clause no longer carries a filesystem-truth size"
    first = occurrences[0]
    assert text[first:].startswith("filesystem-truth (${WF299_BYTES} B"), (
        "the FIRST (live) size rendering must interpolate the measurement; "
        "the superseded ones may follow, labelled"
    )


def test_a_stale_size_only_appears_in_a_superseded_clause():
    text = _read(LAUNCHER)
    offenders = [
        i + 1
        for i, ln in enumerate(text.splitlines())
        if "35072" in ln and not any(m in ln for m in SUPERSEDED_MARKERS)
    ]
    assert not offenders, (
        f"the stale byte count 35072 appears in an UNLABELLED live clause at "
        f"line(s) {offenders} — AGENTS.md §5.5 rule #6 keeps a superseded "
        f"rendering VERBATIM but it must be labelled as superseded"
    )


# ===========================================================================
# GROUP B — R2a: the summary line names its population
# ===========================================================================
def test_the_canonical_script_names_the_probe_population():
    text = _read(EVAL_SCRIPT)
    assert "PROBES PASS" in text
    assert "evaluators PASS" not in text, (
        "the summary line still calls probes 'evaluators' — the number is "
        "right and the LABEL is wrong (§6.269.38)"
    )


def test_the_failure_branch_names_the_same_population():
    text = _read(EVAL_SCRIPT)
    assert "evaluators need attention" not in text, (
        "the FAILURE branch must name the same population as the PASS branch"
    )


# ===========================================================================
# GROUP C — R2b: the audit MEASURES the probe count and names every population
# ===========================================================================
def _canonical_probe_count_by_ast() -> int:
    """An INDEPENDENT ORACLE for the canonical probe count (§6.269.35.4).

    The producer counts a STRING (``n_total += 1``); this counts the NODES.
    Agreement between the two is therefore not vacuous: a comment carrying
    the text would move one and not the other.
    """
    tree = ast.parse(_read(EVAL_SCRIPT))
    n = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "n_total"
            and isinstance(node.value, ast.Constant)
            and node.value.value == 1
        ):
            n += 1
    return n


def test_the_probe_count_is_measured_from_the_source():
    audit = _load(AUDIT, "wfa269_wave28_audit")
    out = audit.measure_canonical_probe_count()
    assert set(out) == {
        "canonical_probe_total",
        "canonical_probe_source",
        "canonical_probe_rule",
    }, f"the measured probe count must carry its three keys, got {sorted(out)}"
    assert out["canonical_probe_source"] == EVAL_SCRIPT
    assert out["canonical_probe_total"] == _canonical_probe_count_by_ast(), (
        "the producer's STRING count and the independent AST oracle disagree"
    )


def test_the_legacy_token_keeps_its_name_and_value():
    text = _read(AUDIT)
    m = re.search(r'"r17_canonical_pass"\s*:\s*14\b', text)
    assert m, (
        "the legacy token `r17_canonical_pass` must keep its NAME and its "
        "VALUE (14) verbatim — AGENTS.md §5.5 rule #6"
    )


def test_every_population_is_named_in_the_audit_json():
    text = _read(AUDIT)
    for key in (
        "r17_canonical_pass_population",
        "r17_canonical_probe_total",
        "r17_canonical_probe_total_source",
        "r17_canonical_probe_total_rule",
        "r17_tier_evaluator_total",
        "r17_tier_evaluator_passed",
    ):
        assert f'"{key}"' in text, f"the audit JSON must publish `{key}`"


def _audit_json_key(key: str) -> Optional[str]:
    """The VALUE the audit's JSON dict binds to ``key``.

    Read through the AST (not a regex): the value is an implicit string
    CONCATENATION, so a single-line regex sees only the first fragment.
    """
    tree = ast.parse(_read(AUDIT))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == key:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    return v.value
                if isinstance(v, ast.JoinedStr):
                    parts = [
                        p.value
                        for p in v.values
                        if isinstance(p, ast.Constant) and isinstance(p.value, str)
                    ]
                    return "".join(parts)
    return None


def test_the_legacy_population_string_names_probes_not_evaluators():
    body = _audit_json_key("r17_canonical_pass_population")
    assert body is not None, "the legacy token's population string is missing"
    assert "PROBES" in body
    assert "NOT evaluators" in body


# ===========================================================================
# GROUP D — R3: the guard's verdict axis follows the SELECTED flag
# ===========================================================================
def test_the_verdict_axis_is_selected_by_the_flag():
    text = _read(GUARD)
    assert (
        'axis = "real_defeaters" if args.include_archived else "real_defeaters_live"'
        in text
    )
    assert 'return 1 if report["counts"][axis] else 0' in text


def test_the_two_axes_partition_the_defeaters():
    guard = _load(GUARD, "wfa269_wave28_guard")
    counts = guard.scan_repository(str(REPO))["counts"]
    assert (
        counts["real_defeaters"]
        == counts["real_defeaters_live"] + counts["real_defeaters_archived"]
    ), "the LIVE and ARCHIVED axes must partition the defeaters exactly"


def test_the_cli_can_pass_on_the_live_axis():
    """A gate that can NEVER pass is as dead as one that can never fail."""
    guard = _load(GUARD, "wfa269_wave28_guard")
    counts = guard.scan_repository(str(REPO))["counts"]
    rc_live = guard.main(["--repo-root", str(REPO), "--quiet"])
    assert rc_live == (1 if counts["real_defeaters_live"] else 0)
    rc_strict = guard.main(
        ["--repo-root", str(REPO), "--quiet", "--include-archived"]
    )
    assert rc_strict == (1 if counts["real_defeaters"] else 0)


def test_advisory_always_exits_zero():
    guard = _load(GUARD, "wfa269_wave28_guard")
    assert guard.main(["--repo-root", str(REPO), "--quiet", "--advisory"]) == 0


# ===========================================================================
# GROUP E — THIS FILE is wired into CI
# ===========================================================================
def test_ci_names_this_guard_file():
    text = _read(CI)
    owners = _ci_jobs_running(text, THIS_FILE)
    assert owners, (
        "no CI job RUNS this file — an unwatched guard is the §6.269.29 "
        "class.  (A COMMENT that names the file does not count: a mention "
        "is not an invocation.)"
    )


def test_the_job_that_runs_this_file_is_in_the_summary():
    text = _read(CI)
    owners = _ci_jobs_running(text, THIS_FILE)
    assert owners, "this file is run by no CI job"
    needs = _ci_summary_needs(text)
    loop = _ci_summary_loop(text)
    for job in owners:
        assert job in needs, (
            f"job `{job}` runs this file but is not in ci-summary's `needs:` "
            f"— it is invisible to the summary"
        )
        assert job in loop, (
            f"job `{job}` runs this file but is not in ci-summary's summary "
            f"loop — its result is never surfaced"
        )


# ===========================================================================
# GROUP F — every test path CI names exists on disk
# ===========================================================================
def test_every_test_path_named_in_ci_exists_on_disk():
    text = _read(CI)
    paths = sorted(
        {
            tok
            for argv in _ci_pytest_argvs(text)
            for tok in argv
            if tok.startswith("molmetal/tests/") and tok.endswith(".py")
        }
    )
    assert paths, "no test paths found in the CI workflow — the scan is wrong"
    missing = [p for p in paths if not (REPO / p).exists()]
    assert not missing, (
        f"CI names test files that do NOT exist on disk: {missing} — pytest "
        f"exits 4 (usage error) on a missing path, so the step can never pass"
    )


# ===========================================================================
# GROUP G — the documented failure taxonomy is the emitted one
# ===========================================================================
def _taxonomy_function(tree: ast.AST) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == TAXONOMY_FUNC
        ):
            return node
    raise AssertionError(f"{TAXONOMY_FUNC} not found in {TRAINER}")


def _documented_buckets() -> List[str]:
    """The bucket table in the docstring — read from the SOURCE LINES.

    Read from the source (not from ``__doc__``) because the table's shape is
    "four spaces, a double-backticked token, then the aligned description
    column".  A token mentioned mid-sentence (``strict_ok``) is prose; a token
    at the head of the description column is a bucket.
    """
    src = _read(TRAINER)
    tree = ast.parse(src)
    fn = _taxonomy_function(tree)
    doc = fn.body[0]
    assert isinstance(doc, ast.Expr), "the function has no docstring"
    lines = src.splitlines()
    out: List[str] = []
    for ln in lines[doc.lineno - 1 : doc.end_lineno]:
        m = re.match(r"^\s{4}``([a-z][a-z0-9_]*)``\s{2,}", ln)
        if m:
            out.append(m.group(1))
    return out


def _emitted_literals() -> set:
    """The reason strings the function can actually RETURN.

    Only the ``return`` statements count.  A literal that merely appears
    somewhere in the body is not a bucket — e.g. the ``or "unknown"`` fallback
    inside ``_cls = _err.split(":", 1)[0].strip() or "unknown"`` is a component
    of ``sanitize_failed:<class>``, not a reason of its own.
    """
    src = _read(TRAINER)
    tree = ast.parse(src)
    fn = _taxonomy_function(tree)
    out = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            out.add(value.value)
        elif isinstance(value, ast.JoinedStr):
            for part in value.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    out.add(part.value)
    return out


def test_the_taxonomy_table_was_found():
    buckets = _documented_buckets()
    assert buckets, (
        "the bucket table was not found — the docstring's shape changed, so "
        "this guard is now watching nothing (§6.269.17's 'skipped' costume)"
    )


def test_every_documented_bucket_is_actually_emitted():
    buckets = _documented_buckets()
    literals = _emitted_literals()
    missing = [
        b
        for b in buckets
        if not any(b == lit or lit.startswith(b) for lit in literals)
    ]
    assert not missing, (
        f"the docstring documents reason bucket(s) {missing} that "
        f"{TAXONOMY_FUNC} NEVER returns — the documented taxonomy and the "
        f"emitted taxonomy are different"
    )


def test_the_emitted_buckets_are_all_documented():
    """The reverse direction: a bucket the code emits but the docstring does
    not name is a taxonomy the reader cannot act on."""
    buckets = set(_documented_buckets())
    literals = _emitted_literals()
    emitted = {
        lit
        for lit in literals
        if re.fullmatch(r"[a-z][a-z0-9_]*(:[A-Za-z0-9_]+)?", lit or "")
    }
    undocumented = {
        lit for lit in emitted if not any(lit == b or lit.startswith(b) for b in buckets)
    }
    assert not undocumented, (
        f"{TAXONOMY_FUNC} emits reason(s) {sorted(undocumented)} that the "
        f"docstring's bucket table does not name"
    )


# ===========================================================================
# GROUP H — the probe's synthetic input satisfies the callee's contract
# ===========================================================================
def _synthetic_yaml() -> Optional[str]:
    """The synthetic config the audit hands to a ``Path``-annotated parameter."""
    tree = ast.parse(_read(AUDIT))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "write"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and "levels:" in node.args[0].value
        ):
            return node.args[0].value
    return None


def test_the_synthetic_probe_input_carries_a_non_empty_cohort():
    y = _synthetic_yaml()
    assert y is not None, "the audit's synthetic config literal was not found"
    assert "cohort:" in y, (
        "the synthetic config must carry a `cohort:` block — the callee "
        "REFUSES an empty or non-list cohort.pockets / cohort.seeds"
    )
    for key in ("pockets", "seeds"):
        items = _yaml_list_items(y, key)
        assert items, f"the synthetic config's cohort.{key} is empty"


def test_the_callee_declares_the_contract_the_probe_must_satisfy():
    """Pin the contract from the OTHER side: the callee must still enforce it.

    Without this, the repair could be undone by relaxing the callee, and the
    probe would go back to failing for a reason that has nothing to do with
    the evaluator it probes.
    """
    text = _read(CALLEE)
    assert 'for key in ("pockets", "seeds"):' in text, (
        "the callee no longer declares the cohort contract the probe obeys"
    )
    assert "empty or non-list" in text


# ===========================================================================
# GROUP I — the gate-inventory ARTEFACT carries its PROVENANCE
# ===========================================================================
#: The producer that writes the inventory artefact.
INVENTORY_PRODUCER = "molmetal/scripts/gate_inventory.py"
#: The module that owns ``build_gates`` — the SURFACE the inventory describes.
GATE_SURFACE = "molmetal/scripts/pre_flight_check.py"
#: The artefact itself.  GITIGNORED — ``.gitignore:158`` matches
#: ``molmetal/reports/**/*.json`` — so a fresh clone has NO such file.  The
#: artefact-freshness half of this group is therefore LOCAL-ONLY and SKIPs in
#: CI; the producer half below is CI-ENFORCEABLE, which is why it exists.
INVENTORY_ARTEFACT = "molmetal/reports/gate_inventory.json"


def _sha256_file(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def _live_gate_counts() -> Dict[str, int]:
    """The LIVE registration surface — MEASURED, with NO gate executed."""
    from molmetal.scripts.pre_flight_check import build_gates

    return {name: len(build_gates(name)) for name in ("all", "launch")}


def _produce_inventory(tmp_path, pipeline: str = "all") -> Dict[str, Any]:
    """Run the producer into a tmp file and return the payload it wrote.

    The CLI's exit code is deliberately NOT asserted: it is non-zero when an
    UNDECLARED opt-in gate exists, and that is a verdict about the TREE, not
    about the producer's ability to write an artefact.
    """
    module = _load(INVENTORY_PRODUCER, "wfa269_wave28_gate_inventory")
    out = tmp_path / f"gate_inventory_{pipeline}.json"
    module.main(["--pipeline", pipeline, "--json-out", str(out)])
    assert out.is_file(), f"the producer wrote no artefact for pipeline {pipeline!r}"
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_producer_emits_the_provenance_fields(tmp_path):
    """Without a timestamp and a producer pin, a stale artefact reads as a
    current one (the §6.269.38 class)."""
    payload = _produce_inventory(tmp_path)
    stamp = payload.get("generated_at")
    assert stamp, "the artefact carries no production timestamp"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stamp), (
        f"generated_at is not an ISO-8601 UTC instant: {stamp!r}"
    )
    assert payload.get("producer_sha256") == _sha256_file(INVENTORY_PRODUCER), (
        "the producer did not pin its OWN version into the artefact"
    )
    assert payload.get("surface_sha256") == _sha256_file(GATE_SURFACE), (
        "the producer did not pin the gate SURFACE it describes"
    )


def test_the_artefact_gate_count_matches_the_live_surface(tmp_path):
    """The artefact's gate count must equal the LIVE registration surface.

    The stale artefact said 129 while the live surface registers 133 — the
    artefact lagged its producer by four gates and, having no timestamp, read
    as current.
    """
    payload = _produce_inventory(tmp_path)
    counts = _live_gate_counts()
    assert payload["n_gates"] == counts["all"], (
        f"the artefact claims {payload['n_gates']} gates but the live surface "
        f"registers {counts['all']}"
    )
    assert payload["n_gates"] == len(payload["inventory"])
    assert payload["n_zombie_gates"] == len(payload["zombie_gates"])
    assert (
        payload["n_declared_zombie_gates"] + payload["n_undisclosed_zombie_gates"]
        == payload["n_zombie_gates"]
    ), "the declared / undisclosed zombie bookkeeping does not add up"


def test_the_pipeline_flag_reaches_the_artefact_label(tmp_path):
    """A flag that reaches the producer but not its ARTEFACT is the
    §6.269.29 class: ``--pipeline launch`` used to write the LAUNCH
    inventory (135 gates) under a hard-coded ``"all"`` label."""
    payload = _produce_inventory(tmp_path, pipeline="launch")
    counts = _live_gate_counts()
    assert payload["pipeline"] == "launch", (
        "the artefact's pipeline label does not follow the --pipeline flag"
    )
    assert payload["n_gates"] == counts["launch"], (
        f"the artefact claims {payload['n_gates']} gates but the live "
        f"'launch' surface registers {counts['launch']}"
    )


def test_the_repo_local_artefact_is_not_stale():
    """LOCAL-ONLY half of this group.

    ``molmetal/reports/gate_inventory.json`` is GITIGNORED
    (``.gitignore:158``), so a fresh clone has no such file and this check
    SKIPs there — stated, not hidden.  The CI-enforceable half is
    ``test_the_producer_emits_the_provenance_fields`` above.

    When the artefact IS present it must describe the CURRENT producer and the
    CURRENT gate surface; otherwise it is the §6.269.38 class — an artefact
    that lags its repaired producer and, having carried no timestamp, read as
    a current one.
    """
    path = REPO / INVENTORY_ARTEFACT
    if not path.is_file():
        pytest.skip(
            f"{INVENTORY_ARTEFACT} is gitignored (.gitignore:158 matches "
            "molmetal/reports/**/*.json) => a fresh clone has no such file "
            "=> this check is LOCAL-ONLY"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.get("producer_sha256") == _sha256_file(INVENTORY_PRODUCER), (
        "the artefact on disk was produced by an OLDER producer - re-run "
        "`python -m molmetal.scripts.gate_inventory`"
    )
    assert payload.get("surface_sha256") == _sha256_file(GATE_SURFACE), (
        "the artefact on disk describes an OLDER gate surface - re-run "
        "`python -m molmetal.scripts.gate_inventory`"
    )
    counts = _live_gate_counts()
    assert payload["n_gates"] == counts["all"], (
        f"the artefact on disk claims {payload['n_gates']} gates but the live "
        f"surface registers {counts['all']} - re-run "
        "`python -m molmetal.scripts.gate_inventory`"
    )
    assert payload.get("generated_at"), (
        "the artefact on disk carries no production timestamp"
    )


# ===========================================================================
# GROUP J — the artefact's DERIVED ``source_line`` pointers (REPORT, not ASSERT)
# ===========================================================================
#: The row fields the artefact stores per gate, and whether each is a DURABLE
#: claim or a DERIVED one.
#:
#: ``source_line`` is DERIVED: it is a LINE NUMBER computed from the surface
#: at production time, and it rots on the next edit made above the gate.
#: MEASURED 2026-11-06 (before this group existed): the artefact carried
#: **129/129 stale ``source_line`` pointers (0 of 129 resolved)** and the
#: wave-28 guard was **GREEN** — it compared ``n_gates`` and
#: ``n_zombie_gates`` but NOT the pointers the SAME artefact stores, i.e. it
#: covered a SUBSET of what the artefact holds (the §6.269.35.3 class).
#:
#: THE FRONTIER DESIGN RULE THIS GROUP ENCODES: **a guard may FAIL only on a
#: DURABLE claim.**  A line number is DERIVED, so its drift is REPORTED as
#: informational drift and is NEVER asserted; refreshing it is a SEPARATE,
#: NON-BLOCKING regeneration step (re-run the producer).
#:
#: This is the CLAIM / RESOLUTION split the receipt-locator guard
#: (``molmetal/tests/test_wfa269_receipt_locator_guard.py``) is adopting in
#: this wave: the CLAIM (does the gate still exist, is the gate count still
#: right) is what a gate may FAIL on; the RESOLUTION (which exact line it
#: sits on) is reported, and the reader is told how to refresh it.
ROW_FIELDS: Dict[str, str] = {
    "name": "durable",
    "callable_name": "durable",
    "priority": "durable",
    "opt_in_env": "durable",
    "opt_in_env_source": "durable",
    "source_line": "derived",
}

#: The DERIVED fields — the ones whose drift is reported, never asserted.
DERIVED_FIELDS = frozenset(f for f, kind in ROW_FIELDS.items() if kind == "derived")

#: The NON-BLOCKING regeneration step a reader runs to refresh the pointers.
#: It rewrites the artefact and its exit code says nothing about pointers.
REFRESH_COMMAND = (
    'CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python '
    "-m molmetal.scripts.gate_inventory"
)


class SourceLineDrift(UserWarning):
    """Informational drift in the artefact's DERIVED ``source_line`` pointers.

    A dedicated category so a reader (and a test) can tell the REPORT
    channel apart from an ordinary warning — and so the rule "the drift is
    REPORTED, never ASSERTED" is executable.
    """


def _live_inventory_rows(pipeline: str = "all") -> List[Dict[str, Any]]:
    """The LIVE inventory rows — the PRODUCER's own resolution.

    The producer's ``_gate_source_line`` resolves a pointer with ``inspect``,
    so re-using it here makes the comparison exact BY CONSTRUCTION: the
    ARTEFACT's pointer and the LIVE pointer come from the SAME code path, so
    a difference is real drift and never a difference of method.  (Re-deriving
    the line with a text scan — e.g. "the line must start with
    ``def <callable_name>(``" — would be a SECOND method, and the repo's own
    lesson is that two methods can disagree about the same question.)
    """
    module = _load(INVENTORY_PRODUCER, "wfa269_wave28_gate_inventory_live")
    return module.build_inventory(pipeline)


def _field_drift(
    payload: Dict[str, Any], live_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare an artefact's stored rows with the LIVE rows, field by field.

    REPORT-ONLY: the return value is a MEASUREMENT, never a verdict.  The
    caller decides what to do with it (here: warn).
    """
    live_by_name = {row.get("name"): row for row in live_rows}
    entries = payload.get("inventory") or []
    mismatches: List[Dict[str, Any]] = []
    for entry in entries:
        name = entry.get("name")
        live = live_by_name.get(name)
        if live is None:
            mismatches.append(
                {"name": name, "field": "*", "stored": "present", "live": "absent"}
            )
            continue
        for field in ROW_FIELDS:
            stored = entry.get(field, "<absent>")
            if stored != live.get(field):
                mismatches.append(
                    {
                        "name": name,
                        "field": field,
                        "stored": stored,
                        "live": live.get(field),
                    }
                )
    return {
        "n_checked": len(entries),
        "n_mismatched_fields": len(mismatches),
        "n_derived_mismatches": sum(
            1 for m in mismatches if m["field"] in DERIVED_FIELDS
        ),
        "mismatches": mismatches,
    }


def _report_field_drift(label: str, drift: Dict[str, Any]) -> None:
    """SURFACE the drift — a WARNING.  Never an assertion, never a raise."""
    lines = [
        f"[source-line-drift] {label}: "
        f"{drift['n_derived_mismatches']}/{drift['n_checked']} DERIVED "
        f"source_line pointer(s) no longer resolve against the live surface "
        f"({drift['n_mismatched_fields']} field mismatch(es) in total) — "
        f"REPORTED, never asserted",
        "  refresh (NON-BLOCKING regeneration; its exit code says nothing "
        f"about pointers): {REFRESH_COMMAND}",
    ]
    for m in drift["mismatches"][:20]:
        lines.append(
            f"    {m['name']}.{m['field']}: stored={m['stored']!r} "
            f"live={m['live']!r}"
        )
    msg = "\n".join(lines)
    warnings.warn(msg, SourceLineDrift, stacklevel=2)
    print(msg)


def test_the_source_line_drift_detector_sees_a_corrupted_pointer():
    """The REPORT channel must be ABLE to report — a detector that cannot
    detect is a dead guard (§6.269.17's class).

    The payload is built FROM the live rows, so the pristine comparison is 0
    by construction and the result does not depend on whether another agent
    edited the surface while the test ran (a tree that moves under the
    comparison would otherwise make this test flaky for a reason that has
    nothing to do with the detector).
    """
    live_rows = _live_inventory_rows("all")
    assert live_rows, "the live inventory is empty — the producer is broken"
    payload = {
        "n_gates": len(live_rows),
        "inventory": [dict(row) for row in live_rows],
    }
    clean = _field_drift(payload, live_rows)
    assert clean["n_checked"] == len(live_rows)
    assert clean["n_mismatched_fields"] == 0, (
        "a payload built FROM the live rows cannot mismatch — the comparison "
        "itself is inconsistent"
    )
    victim = next(
        (r for r in payload["inventory"] if r.get("source_line") is not None),
        None,
    )
    assert victim is not None, (
        "the producer emits no source_line pointer at all — there is nothing "
        "for the drift measurement to compare"
    )
    victim["source_line"] = int(victim["source_line"]) + 1
    dirty = _field_drift(payload, live_rows)
    assert dirty["n_derived_mismatches"] >= 1, (
        "corrupting one source_line was NOT reported — the REPORT channel is "
        "DEAD (a guard that cannot see the drift it was built to see)"
    )
    assert any(
        m["name"] == victim["name"] and m["field"] == "source_line"
        for m in dirty["mismatches"]
    ), (
        f"the corrupted pointer of {victim['name']} was not NAMED in the "
        f"report — the drift must be actionable, not just counted"
    )


def test_the_drift_report_is_informational_and_never_fails():
    """The FRONTIER RULE, made EXECUTABLE: a DERIVED pointer's drift is
    REPORTED, never ASSERTED.

    If a later change turns the report into an assertion (or into a raise),
    this test goes RED — the guard would then be failing on a DERIVED claim,
    which the rule forbids.  This is the same CLAIM / RESOLUTION split the
    receipt-locator guard is adopting: the CLAIM may fail the guard, the
    RESOLUTION may only be reported.
    """
    live_rows = _live_inventory_rows("all")
    payload = {
        "n_gates": len(live_rows),
        "inventory": [dict(row) for row in live_rows],
    }
    victim = next(
        (r for r in payload["inventory"] if r.get("source_line") is not None),
        None,
    )
    assert victim is not None, "the producer emits no source_line pointer"
    victim["source_line"] = int(victim["source_line"]) + 1
    drift = _field_drift(payload, live_rows)
    assert drift["n_derived_mismatches"] >= 1
    with pytest.warns(SourceLineDrift):
        _report_field_drift("mutation-proof", drift)


def test_the_local_artefact_pointer_drift_is_reported():
    """The GITIGNORED artefact half — and it NEVER SKIPS.

    ``molmetal/reports/gate_inventory.json`` is gitignored
    (``.gitignore:158``), so a fresh clone has no such file.  A SKIP here
    would HIDE the drift in CI, so this test REPORTS instead: with no local
    artefact the measurement is over an empty population, and the report says
    so out loud.  The drift is therefore never silently skipped.
    """
    path = REPO / INVENTORY_ARTEFACT
    if not path.is_file():
        _report_field_drift(
            f"{INVENTORY_ARTEFACT} (ABSENT — gitignored, .gitignore:158)",
            {
                "n_checked": 0,
                "n_mismatched_fields": 0,
                "n_derived_mismatches": 0,
                "mismatches": [],
            },
        )
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    pipeline = payload.get("pipeline") or "all"
    live_rows = _live_inventory_rows(pipeline)
    _report_field_drift(INVENTORY_ARTEFACT, _field_drift(payload, live_rows))


def test_the_artefact_row_schema_is_fully_covered():
    """The §6.269.35.3 SUBSET class, closed by CONSTRUCTION.

    The artefact stores one row per gate; the drift measurement compares a
    DECLARED set of row fields.  If the producer ever grows a field, the
    declared set would be a SUBSET again — so the two sets are compared here
    and this test goes RED until the measurement is extended.  (Enumerating
    which fields to check is itself a hand-written subset — §6.269.35.3's
    own lesson — so the enumeration is DECLARED and ENFORCED to be complete
    rather than left implicit.)
    """
    module = _load(INVENTORY_PRODUCER, "wfa269_wave28_gate_inventory_schema")
    live_rows = module.build_inventory("all")
    assert live_rows, "the live inventory is empty — the producer is broken"
    produced = set(live_rows[0].keys())
    declared = set(ROW_FIELDS.keys())
    missing = produced - declared
    extra = declared - produced
    assert not missing, (
        f"the producer stores row field(s) {sorted(missing)} that the drift "
        f"measurement does NOT compare — the guard would cover a SUBSET of "
        f"what the artefact stores (§6.269.35.3)"
    )
    assert not extra, (
        f"the drift measurement compares row field(s) {sorted(extra)} that "
        f"the producer does not store"
    )
    assert "source_line" in ROW_FIELDS, (
        "the source_line pointer must stay in the compared set — dropping it "
        "is the §6.269.35.3 SUBSET defect this group repairs"
    )
    assert DERIVED_FIELDS == frozenset({"source_line"}), (
        "source_line is the only DERIVED row field; a second one would need "
        "the same REPORT-not-ASSERT treatment, and a DERIVED field that is "
        "not declared as such would be a silent subset"
    )
