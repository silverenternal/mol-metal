"""Wf-A127 — the Wf-A125 residue-closure regression guards (CPU-only).

Wf-A125 closed eleven guard/gate gaps (catalog §6.224..§6.234) and then
recorded four OPEN residues of its own (§6.234) plus a blocking defect its
own critic found: the repo-root CPU lock it shipped was **git-ignored**, so
it could not ship at all.  Wf-A127 closed that residue.

Every test here is **fast** (no pre-flight subprocess — the pre-flight costs
~180 s) and **CPU-only**.  The tests that need the pre-flight's actual
counts assert them by *introspection* (the code constant) rather than by
running the whole gate suite, and the pre-flight's own behaviour is covered
by the existing suites
(``test_wf415a3_preflight_fail_audit.py``,
``test_wf327f11_gate_count_consistency.py``, ``test_pre_flight_check.py``).

MEASURED 2026-10-31, CPU-only.

Catalog: §6.235 (git-ignored lock) · §6.236 (the count lives in 4 places) ·
§6.237 (the WARN contract) · §6.238 (the census docstring) · §6.239 (G7
durability) · §6.240 (the registry header stamp).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

AUDIT_DOC = REPO_ROOT / "TODO" / "completed" / "321d_preflight_gate_audit.md"
CENSUS_SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wf_cfm_run_census.py"
CENSUS_ARTIFACT = REPO_ROOT / "metrics" / "by_metric" / "cfm_run_census.json"
PB_BENCH = REPO_ROOT / "molmetal" / "scripts" / "bench_pb_valid.py"
REGISTRY_SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "registry.py"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# §6.235 — the repo-root CPU lock must be SHIPPABLE
# ---------------------------------------------------------------------------
def test_the_root_cpu_lock_is_not_git_ignored() -> None:
    """(§6.235) ``git check-ignore`` must NOT ignore the root ``conftest.py``.

    While ``.gitignore`` carried ``/conftest.py`` the Wf-A125 CPU lock
    existed on disk only: a fresh clone ran the top-level ``tests/`` tree
    UNLOCKED and silently took VRAM, which is exactly what the standing
    禁止使用gpu directive forbids.
    """
    proc = _git("check-ignore", "-v", "conftest.py")
    assert proc.returncode == 1, (
        "the repo-root conftest.py is git-ignored again "
        f"(check-ignore rc={proc.returncode}): {proc.stdout.strip()}"
    )


def test_the_root_cpu_lock_exists_and_blanks_cuda() -> None:
    """(§6.235) The lock must exist AND actually blank CUDA_VISIBLE_DEVICES."""
    conftest = REPO_ROOT / "conftest.py"
    assert conftest.is_file(), conftest
    text = conftest.read_text(encoding="utf-8")
    assert 'CUDA_VISIBLE_DEVICES' in text, "the root conftest no longer blanks CUDA"
    assert "MOLMETAL_TEST_ALLOW_GPU" in text, (
        "the root conftest must keep an explicit opt-in env var so a "
        "deliberate GPU session is still possible"
    )


def test_the_package_conftests_are_still_tracked() -> None:
    """(§6.235) Dropping the root ignore must NOT un-track the package shims.

    The historical harm the ignore rule was anchored against was the
    UNANCHORED pattern excluding ``molmetal/conftest.py`` and
    ``molmetal/tests/conftest.py`` from version control.
    """
    proc = _git("ls-files")
    assert proc.returncode == 0, proc.stderr
    tracked = {line.strip() for line in proc.stdout.splitlines()}
    for rel in (
        "molmetal/conftest.py",
        "molmetal/tests/conftest.py",
    ):
        assert rel in tracked, (rel, sorted(t for t in tracked if "conftest" in t))


# ---------------------------------------------------------------------------
# §6.236 — the pre-flight gate count lives in FOUR places
# ---------------------------------------------------------------------------
def test_the_audit_doc_header_count_matches_the_code_constant() -> None:
    """(§6.236) §2.1 of the SSOT doc must name the current ``all`` count."""
    from molmetal.scripts.pre_flight_check import EXPECTED_GATE_COUNTS

    text = AUDIT_DOC.read_text(encoding="utf-8")
    match = re.search(
        r"### §2\.1\s+`all`\s+pipeline\s+—\s+(\d+)\s+gates", text
    )
    assert match is not None, "§2.1 header not found"
    assert int(match.group(1)) == EXPECTED_GATE_COUNTS["all"], (
        f"doc={match.group(1)} vs code={EXPECTED_GATE_COUNTS['all']}"
    )


def test_the_audit_doc_summary_table_matches_the_code_constants() -> None:
    """(§6.236) The §6 summary table must match every pipeline's constant."""
    from molmetal.scripts.pre_flight_check import EXPECTED_GATE_COUNTS

    text = AUDIT_DOC.read_text(encoding="utf-8")
    match = re.search(
        r"(\d+)\s+\(`all`\)\s+/\s+(\d+)\s+\(`frontier`\)"
        r"\s+/\s+(\d+)\s+\(`chemprop_v2_v3`\)\s+/\s+(\d+)\s+\(`cfm`\)",
        text,
    )
    assert match is not None, "§6 summary table row not found"
    for doc_val, pipeline in zip(match.groups(), ("all", "frontier", "chemprop_v2_v3", "cfm")):
        assert int(doc_val) == EXPECTED_GATE_COUNTS[pipeline], (
            pipeline,
            doc_val,
            EXPECTED_GATE_COUNTS[pipeline],
        )


def test_the_preflight_total_count_pin_is_the_current_count() -> None:
    """(§6.236) ``test_wf415a3`` must pin the CURRENT total, not a stale one."""
    from molmetal.scripts.pre_flight_check import EXPECTED_GATE_COUNTS

    path = REPO_ROOT / "molmetal" / "tests" / "test_wf415a3_preflight_fail_audit.py"
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'assert preflight_summary\["total"\]\s*==\s*(\d+)', text
    )
    assert match is not None, "the total-count assertion was not found"
    assert int(match.group(1)) == EXPECTED_GATE_COUNTS["all"], (
        f"the pin says {match.group(1)} but the code constant says "
        f"{EXPECTED_GATE_COUNTS['all']}"
    )


def test_the_gate_counts_are_what_this_wave_measured() -> None:
    """(§6.236) The live counts this wave recorded (MEASURED 2026-10-31).

    Wf-A172 (2026-11-01) REPAIR — the assertion target moved off the
    hard-coded 127 / 129 literals onto the SSOT-vs-runtime invariant.

    SUPERSEDED (2026-11-01, Wf-A127 rendering — PRESERVED VERBATIM per
    AGENTS.md §5.5 rule #6):
        assert len(build_gates("all")) == 127, len(build_gates("all"))
        assert len(build_gates("launch")) == 129, len(build_gates("launch"))
        assert EXPECTED_GATE_COUNTS["all"] == 127
        assert EXPECTED_GATE_COUNTS["launch"] == 129
    with the superseded counts all=127 and launch=129 — the counts this
    wave measured on 2026-10-31.

    WHY: the counts moved on.  MEASURED 2026-11-01 by live import:
    all=129, launch=131 in this working tree (126 / 128 at HEAD), and
    the SSOT constant carries the live value on every tree.  The
    invariant §6.236 is about — "the gate count lives in FOUR places and
    they must agree" — is the SSOT-vs-runtime comparison, which is green
    on both trees and still FAILs if the constant is bumped without
    registering the gate.  The four PLACES themselves are checked by the
    two doc-derived tests above (the §2.1 header and the §6 summary
    table) and by ``test_the_preflight_total_count_pin_is_the_current_count``.
    """
    from molmetal.scripts.pre_flight_check import EXPECTED_GATE_COUNTS, build_gates

    assert len(build_gates("all")) == EXPECTED_GATE_COUNTS["all"], (
        f"the 'all' pipeline holds {len(build_gates('all'))} gates but "
        f"EXPECTED_GATE_COUNTS['all'] says "
        f"{EXPECTED_GATE_COUNTS['all']}"
    )
    assert len(build_gates("launch")) == EXPECTED_GATE_COUNTS["launch"], (
        f"the 'launch' pipeline holds {len(build_gates('launch'))} gates "
        f"but EXPECTED_GATE_COUNTS['launch'] says "
        f"{EXPECTED_GATE_COUNTS['launch']}"
    )


# ---------------------------------------------------------------------------
# §6.237 — WARN is a first-class status in the TEST contract too
# ---------------------------------------------------------------------------
def test_the_schema_contract_allows_warn() -> None:
    """(§6.237) ``ALLOWED_STATUS`` must contain ``WARN``."""
    path = REPO_ROOT / "molmetal" / "tests" / "test_pre_flight_check.py"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"ALLOWED_STATUS\s*=\s*\{([^}]*)\}", text)
    assert match is not None, "ALLOWED_STATUS not found"
    assert '"WARN"' in match.group(1), match.group(1)


def test_the_schema_contract_identity_counts_warn() -> None:
    """(§6.237) The identity must include ``warn`` on both sides."""
    path = REPO_ROOT / "molmetal" / "tests" / "test_pre_flight_check.py"
    text = path.read_text(encoding="utf-8")
    assert re.search(
        r's\["pass"\]\s*\+\s*s\["fail"\]\s*\+\s*s\["skip"\]\s*\+\s*s\["warn"\]',
        text,
    ), "the summary identity does not include s['warn']"
    assert "warn" in re.search(
        r"REQUIRED_SUMMARY_KEYS\s*=\s*\{([^}]*)\}", text
    ).group(1), "REQUIRED_SUMMARY_KEYS does not carry 'warn'"


def test_the_code_summary_actually_carries_a_warn_counter() -> None:
    """(§6.237) The producer must emit the key the contract now requires."""
    src = (REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py").read_text(
        encoding="utf-8"
    )
    assert '"warn"' in src, "pre_flight_check.py no longer emits a warn counter"
    assert "n_warn" in src, "pre_flight_check.py no longer computes n_warn"


# ---------------------------------------------------------------------------
# §6.238 — the census docstring must match the committed artifact
# ---------------------------------------------------------------------------
def test_the_census_artifact_split_is_eleven_and_eighteen() -> None:
    """(§6.238) MEASURED against the committed artifact, not from prose.

    Counted over the STRUCTURED run rows (``runs[].fragmentation_gate.basis``),
    not over substring occurrences: the artifact's ``changelog`` section also
    names the two reasons in prose, so a substring count would read 12.

    Wf-A150 (2026-10-31) — the split MOVED: the Wf-A127 republication
    published the empty-denominator chemistry rate as ``null`` instead of the
    trainer's 0.0 placeholder, and the two runs whose LAST epoch is in that
    state (cfm_fm_control_2026_10_09, cfm_bond_head_30ep_onecycle_2026_10_09)
    moved from FAIL/end_to_end to SKIP/denominator_empty.  The split is
    therefore 11 / 18 over **29** SKIP rows, not 11 / 16 over 27.
    """
    payload = json.loads(CENSUS_ARTIFACT.read_text(encoding="utf-8"))
    counts = {}
    for run in payload["runs"]:
        gate = run.get("fragmentation_gate") or {}
        key = (gate.get("status"), gate.get("basis"))
        counts[key] = counts.get(key, 0) + 1
    assert counts.get(("SKIP", "field_absent")) == 11, counts
    assert counts.get(("SKIP", "denominator_empty")) == 18, counts
    assert counts.get(("SKIP", "field_absent")) + counts.get(
        ("SKIP", "denominator_empty")
    ) == 29, counts


def test_the_census_docstring_states_the_measured_split() -> None:
    """(§6.238) The docstring must name 11 / 18, not the refuted 9 / 18.

    Wf-A150: the docstring itself carried the SUPERSEDED 11 / 16 over 27
    split after the Wf-A127 republication moved two runs into the
    denominator_empty bucket; it is now the MEASURED 11 / 18 over 29.
    """
    text = CENSUS_SCRIPT.read_text(encoding="utf-8")
    window = text[text.index("TWO DISTINCT SKIP CAUSES") :][:2000]
    assert "**11** of the 29 SKIP runs" in window, window[:600]
    assert "**18** of the 29" in window, window[:600]
    assert "9 of the 27 SKIP runs" not in window, "the refuted 9/18 split is still live"


def test_field_present_is_documented_as_an_input_flag_not_a_basis() -> None:
    """(§6.238) ``field_present`` is the INPUT, not a third ``basis`` value."""
    text = CENSUS_SCRIPT.read_text(encoding="utf-8")
    window = text[text.index("TWO DISTINCT SKIP CAUSES") :][:2000]
    assert "INPUT FLAG" in window, window[-800:]


# ---------------------------------------------------------------------------
# §6.239 — the PB-valid artifact's honesty must be DURABLE
# ---------------------------------------------------------------------------
def test_the_pb_bench_computes_its_own_honesty_blocks() -> None:
    """(§6.239) The generator must emit the blocks, not a hand-edit."""
    from molmetal.scripts.bench_pb_valid import (
        build_honesty_blocks,
        load_spec_target_of_record,
    )

    spec = load_spec_target_of_record()
    assert spec is not None, "the declaration of record is missing"
    assert spec["n_cells"] == 100, spec

    blocks = build_honesty_blocks(
        n_evaluations=2,
        n_target_evaluations=2,
        per_evaluation=[
            {"status": "skipped", "note": "posebusters not installed"},
            {"status": "skipped", "note": "posebusters not installed"},
        ],
        spec_target=spec,
    )
    assert blocks["spec_target"] == spec, blocks["spec_target"]
    deferral = blocks["wf432a1_pb_validity_deferred_block"]
    assert deferral["n_measured"] == 0, deferral
    assert deferral["n_skipped"] == 2, deferral
    assert deferral["n_spec_cells"] == 100, deferral
    framing = deferral["honest_framing"]
    for needle in ("NOT MEASURED", "NEVER RUN", "do not cite this artifact as a PoseBusters validity result"):
        assert needle in framing, (needle, framing)


def test_the_pb_bench_reports_a_partial_run_as_partial() -> None:
    """(§6.239) A run that DID measure must not claim it measured nothing."""
    from molmetal.scripts.bench_pb_valid import build_honesty_blocks

    spec = {
        "n_pockets": 10,
        "n_candidates": 5,
        "n_modes": 2,
        "n_cells": 100,
    }
    blocks = build_honesty_blocks(
        n_evaluations=2,
        n_target_evaluations=100,
        per_evaluation=[
            {"status": "passed", "note": ""},
            {"status": "skipped", "note": "posebusters not installed"},
        ],
        spec_target=spec,
    )
    deferral = blocks["wf432a1_pb_validity_deferred_block"]
    assert deferral["n_measured"] == 1, deferral
    assert deferral["n_skipped"] == 1, deferral
    framing = deferral["honest_framing"]
    assert "PARTIAL" in framing, framing
    assert "98 cell(s) of the declared sweep were NOT run" in framing, framing


def test_the_pb_bench_writes_atomically() -> None:
    """(§6.239) §5.6 — the artifact must not be written by a bare write_text."""
    text = PB_BENCH.read_text(encoding="utf-8")
    assert "args.metrics_path.write_text" not in text, "non-atomic write is back"
    assert "args.summary_path.write_text" not in text, "non-atomic write is back"
    assert "atomic_write_file" in text, "the atomic-write helper is gone"


def test_the_committed_pb_artifact_still_carries_its_honesty() -> None:
    """(§6.239) The artifact on disk must still be self-describing."""
    artifact = json.loads(
        (REPO_ROOT / "metrics" / "by_metric" / "wf32_pb_valid.json").read_text(
            encoding="utf-8"
        )
    )
    assert "spec_target" in artifact, sorted(artifact)
    assert artifact["spec_target"]["n_cells"] == 100, artifact["spec_target"]
    framing = artifact["wf432a1_pb_validity_deferred_block"]["honest_framing"]
    assert "NOT MEASURED" in framing, framing
    assert "NEVER RUN" in framing, framing


# ---------------------------------------------------------------------------
# §6.240 — the registry header must not date itself before its own rows
# ---------------------------------------------------------------------------
def test_the_registry_repoint_bumps_the_file_level_stamp() -> None:
    """(§6.240) A re-point must move ``generated_at:`` with the row."""
    text = REGISTRY_SCRIPT.read_text(encoding="utf-8")
    assert "generated_at:" in text, "the generated_at bump is gone"
    window = text[text.index("FILE-level `generated_at:` stamp") :][:900]
    assert "generated_at" in window, window[:400]


def test_the_registry_header_is_not_older_than_its_rows() -> None:
    """(§6.240) The file header must not predate the newest row stamp."""
    import datetime

    lines = (REPO_ROOT / "registry" / "metrics.yaml").read_text(
        encoding="utf-8"
    ).splitlines()
    header = next(line for line in lines if line.startswith("generated_at:"))
    header_stamp = datetime.datetime.fromisoformat(
        header.split(":", 1)[1].strip().strip("'")
    )
    stamps = [
        datetime.datetime.fromisoformat(line.split(":", 1)[1].strip().strip("'"))
        for line in lines
        if line.strip().startswith("last_refreshed:")
    ]
    assert stamps, "no last_refreshed rows found"
    newest = max(stamps)
    assert header_stamp >= newest, (header_stamp, newest)


# §6.235 — the CLASS guard: no SOURCE file may be git-ignored
# ---------------------------------------------------------------------------
# A file git cannot see is a file that did not ship (catalog §6.235).  The
# Wf-A125 CPU lock was exactly that: present on disk, invisible to git, so a
# fresh clone ran `tests/` unlocked and took VRAM.  This asserts the
# repo-wide invariant, so the class cannot recur at a path nobody thought to
# look at.


def test_no_python_source_file_is_git_ignored() -> None:
    """(§6.235) ZERO git-ignored ``.py`` files anywhere in the repo.

    MEASURED 2026-10-31, before the repair: ``.gitignore`` carried a
    ``/conftest.py`` rule, so the repo-root CPU lock was invisible to git.
    """
    proc = _git("status", "--porcelain", "--ignored")
    assert proc.returncode == 0, proc.stderr
    ignored_py = [
        line
        for line in proc.stdout.splitlines()
        if line.startswith("!!") and line.rstrip().endswith(".py")
    ]
    assert not ignored_py, (
        "git-ignored Python source files cannot ship: "
        f"{ignored_py}. A repair that lives in an ignored file is not a "
        "repair (catalog §6.235)."
    )
