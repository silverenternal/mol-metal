# SPDX-License-Identifier: Apache-2.0
"""Wf-A169 — tests for the gate-count pinner (``wfa169_gate_count_pins``).

Why this file exists
--------------------
The pre-flight gate count is pinned in several places and on 2026-11-01 the
Wf-A169 census MEASURED six test files whose pins were stale — RED on a
clean tree for a reason that had nothing to do with the code they guard.
Wf-A154 re-synced one pin file and missed the others.

This file ships the tool that stops that from recurring:
``molmetal/scripts/wfa169_gate_count_pins.py`` enumerates every executable
gate-count pin and says which are stale.

The four contracts this file enforces
-------------------------------------
1.  **NEGATIVE control** — a synthetic stale pin IS detected.  Without this
    the scanner could be a no-op that always reports "OK".
2.  **POSITIVE control** — the repo's CURRENT pins are reported as OK, and
    the known-stale ones are reported as stale.
3.  **VACUITY guard** — the reported counts equal the number of listed
    findings.  A census that drops rows silently must not be able to
    report "0 problems".
4.  **CPU-ONLY guard** — the module's import graph contains no ``torch``,
    statically (AST) and at runtime (isolated subprocess).

CPU-ONLY: this file imports no torch, allocates no GPU memory, and its one
subprocess runs with ``CUDA_VISIBLE_DEVICES=""`` set explicitly.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import wfa169_gate_count_pins as pins  # noqa: E402

MODULE_PATH = REPO_ROOT / "molmetal" / "scripts" / "wfa169_gate_count_pins.py"
PREFLIGHT_PATH = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"

#: The six test files that were RED on 2026-11-01 (MEASURED by a pytest run
#: over exactly these six files: 36 failed, 72 passed).
WF_A169_RED_FILES = (
    "molmetal/tests/test_wfa127_residue_closure.py",
    "molmetal/tests/test_wfa128_preflight_classification.py",
    "molmetal/tests/test_wf433a1_gate_v2_wire.py",
    "molmetal/tests/test_wf350c1_gpu_env_probe.py",
    "molmetal/tests/test_wf323f_m4_sa_qed_subcriterion.py",
    "molmetal/tests/test_wfa159_timeout_verdict_census.py",
)

#: The files that hold an EXECUTABLE gate-count pin, MEASURED 2026-11-01 by
#: the census itself (Wf-A172, after the ten stale pins were repaired).
#:
#: This is a FILE-level coverage requirement, and it is deliberately not a
#: row count: a file's pins are repaired file by file, so a row-count bound
#: decays every time a stale pin is fixed — which is exactly what happened
#: to the Wf-A169 ``>= 10`` bound (see the dated note in
#: ``test_the_repo_scan_reports_pins_at_all``).
#:
#: MEASURED 2026-11-01: 6 files, 8 executable pins (2 doc-derived in
#: ``test_wfa127_residue_closure.py``, 2 in
#: ``test_wfa159_timeout_verdict_census.py``, 1 each in the other four).
KNOWN_PIN_FILES = (
    "molmetal/tests/test_preflight_ssot_guard_wiring.py",
    "molmetal/tests/test_wf350a2_compute_manifest.py",
    "molmetal/tests/test_wf350c1_gpu_env_probe.py",
    "molmetal/tests/test_wf415a3_preflight_fail_audit.py",
    "molmetal/tests/test_wfa127_residue_closure.py",
    "molmetal/tests/test_wfa159_timeout_verdict_census.py",
)

#: The files whose stale pins the Wf-A172 repair retired.  A census that
#: still reports a stale pin in any of them means the repair silently kept
#: a literal.
WFA172_REPAIRED_FILES = (
    "molmetal/tests/test_wf323f_m4_sa_qed_subcriterion.py",
    "molmetal/tests/test_wfa127_residue_closure.py",
    "molmetal/tests/test_wfa128_preflight_classification.py",
    "molmetal/tests/test_wf433a1_gate_v2_wire.py",
)


@pytest.fixture(scope="session")
def census() -> dict:
    """One census per session — the scan touches ~3000 python files."""
    return pins.pin_census(REPO_ROOT)


@pytest.fixture(scope="session")
def repo_pins() -> list:
    return pins.scan_gate_count_pins(REPO_ROOT)


# ---------------------------------------------------------------------------
# 1 — the live counts and the invariant
# ---------------------------------------------------------------------------
def test_live_gate_counts_agree_with_the_expected_constant() -> None:
    """MEASURED by live import: live == EXPECTED_GATE_COUNTS, per pipeline."""
    live = pins.live_gate_counts()
    expected = pins.expected_gate_counts()
    assert live == expected, (
        f"gate-count drift: live={live} expected={expected}. "
        f"Update EXPECTED_GATE_COUNTS in pre_flight_check.py."
    )
    assert live["all"] == len(pins._load_preflight().build_gates("all"))


def test_assert_gate_counts_match_expected_passes_on_the_live_tree() -> None:
    """The invariant helper must not raise on the current tree."""
    report = pins.assert_gate_counts_match_expected()
    assert report["all"]["status"] == "ok", report
    assert report["all"]["live"] == report["all"]["expected"]


def test_assert_gate_counts_match_expected_reports_every_drifted_pipeline() -> None:
    """NEGATIVE control at the API level: a drifted count must RAISE.

    The report must name *every* drifted pipeline, not just the first —
    the Wf-A169 census found six stale pins at once, and a one-at-a-time
    report hides that.
    """
    pfc = pins._load_preflight()
    original = dict(pfc.EXPECTED_GATE_COUNTS)
    try:
        pfc.EXPECTED_GATE_COUNTS["all"] = original["all"] + 1
        pfc.EXPECTED_GATE_COUNTS["cfm"] = original["cfm"] + 2
        with pytest.raises(AssertionError) as excinfo:
            pins.assert_gate_counts_match_expected()
        message = str(excinfo.value)
        assert "all" in message and "cfm" in message, message
    finally:
        pfc.EXPECTED_GATE_COUNTS.update(original)


# ---------------------------------------------------------------------------
# 2 — NEGATIVE / POSITIVE controls on the scanner itself
# ---------------------------------------------------------------------------
SYNTHETIC_STALE = '''\
from molmetal.scripts.pre_flight_check import build_gates


def test_synthetic_stale_pin():
    assert len(build_gates("all")) == 999
'''

SYNTHETIC_CURRENT = '''\
from molmetal.scripts.pre_flight_check import build_gates


def test_synthetic_current_pin():
    assert len(build_gates("all")) == {live_all}
'''

SYNTHETIC_CONST_STALE = '''\
from molmetal.scripts.pre_flight_check import EXPECTED_GATE_COUNTS


def test_synthetic_const_stale_pin():
    assert EXPECTED_GATE_COUNTS["all"] == 999
'''

SYNTHETIC_PROSE = '''\
"""A module that merely TALKS about the gate count.

EXPECTED_GATE_COUNTS["all"] is 999 in this prose and must NOT be reported
as a pin: prose does not fail, it only misleads.
"""


def test_prose_only():
    assert True
'''


def _write(tmp_path: Path, name: str, body: str) -> Path:
    target = tmp_path / name
    target.write_text(body, encoding="utf-8")
    return target


def test_a_synthetic_stale_pin_is_detected(tmp_path: Path) -> None:
    """NEGATIVE control — a synthetic stale pin is reported as stale."""
    path = _write(tmp_path, "synthetic_stale.py", SYNTHETIC_STALE)
    found = pins.scan_file_for_pins(path, root=tmp_path)
    exact = [p for p in found if p.status in ("ok", "stale", "unknown-pipeline")]
    assert len(exact) == 1, [p.as_dict() for p in found]
    pin = exact[0]
    assert pin.status == "stale", pin.as_dict()
    assert pin.pipeline == "all", pin.as_dict()
    assert pin.pinned == 999, pin.as_dict()
    assert pin.live == pins.live_gate_counts()["all"], pin.as_dict()


def test_a_synthetic_current_pin_is_reported_ok(tmp_path: Path) -> None:
    """POSITIVE control — a synthetic CURRENT pin is reported as ok."""
    live_all = pins.live_gate_counts()["all"]
    path = _write(
        tmp_path, "synthetic_current.py", SYNTHETIC_CURRENT.format(live_all=live_all)
    )
    found = pins.scan_file_for_pins(path, root=tmp_path)
    exact = [p for p in found if p.status in ("ok", "stale", "unknown-pipeline")]
    assert len(exact) == 1, [p.as_dict() for p in found]
    assert exact[0].status == "ok", exact[0].as_dict()
    assert exact[0].pinned == live_all


def test_a_synthetic_constant_form_stale_pin_is_detected(tmp_path: Path) -> None:
    """The ``EXPECTED_GATE_COUNTS[...]`` form is detected too."""
    path = _write(tmp_path, "synthetic_const_stale.py", SYNTHETIC_CONST_STALE)
    found = pins.scan_file_for_pins(path, root=tmp_path)
    exact = [p for p in found if p.status in ("ok", "stale", "unknown-pipeline")]
    assert len(exact) == 1, [p.as_dict() for p in found]
    assert exact[0].status == "stale", exact[0].as_dict()
    assert exact[0].pipeline == "all"


def test_prose_about_a_gate_count_is_not_a_pin(tmp_path: Path) -> None:
    """PROSE control — narrative about a count is not an executable pin.

    AGENTS.md §5.5 rule 6: the superseded rendering of prose is preserved
    verbatim, so prose must never be reported as something to edit.
    """
    path = _write(tmp_path, "synthetic_prose.py", SYNTHETIC_PROSE)
    found = pins.scan_file_for_pins(path, root=tmp_path)
    assert found == [], [p.as_dict() for p in found]


# ---------------------------------------------------------------------------
# 3 — the real repo: POSITIVE control + VACUITY guard
# ---------------------------------------------------------------------------
def test_the_repo_scan_reports_pins_at_all(repo_pins: list) -> None:
    """POSITIVE control on the real tree — the scanner finds the known pins.

    Wf-A172 (2026-11-01) REPAIR — the bound moved off the hard-coded
    ``>= 10`` onto the number of FILES the census must cover, and the
    "known" tuple moved off a fixed six-file list onto the files that
    actually hold an executable pin.

    SUPERSEDED (2026-11-01, Wf-A169 rendering — PRESERVED VERBATIM per
    AGENTS.md §5.5 rule #6):
        assert len(exact) >= 10, (
            f"expected at least 10 executable gate-count pins in the repo, "
            f"found {len(exact)} — the scanner has gone blind"
        )
        paths = {p.path for p in exact}
        for known in (
            "molmetal/tests/test_wfa127_residue_closure.py",
            "molmetal/tests/test_wfa128_preflight_classification.py",
            "molmetal/tests/test_wf433a1_gate_v2_wire.py",
            "molmetal/tests/test_wf350c1_gpu_env_probe.py",
            "molmetal/tests/test_wf323f_m4_sa_qed_subcriterion.py",
            "molmetal/tests/test_wf415a3_preflight_fail_audit.py",
        ):
            assert known in paths, f"{known} is missing from the pin census"
    with the superseded bound of 10 rows and the superseded six-file tuple.

    WHY: the bound of 10 was itself a stale pin — it counted the TEN
    stale rows that the Wf-A172 repair retires, so a repo whose stale
    pins had all been fixed would have FAILED this guard.  The bound is
    now the number of files the census must cover: a scanner that has
    gone blind reports nothing and fails both the count and the coverage
    loop, while a single repaired pin does not move the bound.
    """
    exact = [p for p in repo_pins if p.status in ("ok", "stale", "unknown-pipeline")]
    paths = {p.path for p in exact}
    for known in KNOWN_PIN_FILES:
        assert known in paths, f"{known} is missing from the pin census"
    assert len(exact) >= len(KNOWN_PIN_FILES), (
        f"expected at least {len(KNOWN_PIN_FILES)} executable gate-count "
        f"pins in the repo (one per covered file), found {len(exact)} — "
        f"the scanner has gone blind"
    )


def test_the_wfa172_repaired_files_hold_no_stale_pin(repo_pins: list) -> None:
    """POSITIVE control for the Wf-A172 repair — no stale pin survives.

    Wf-A172 (2026-11-01) moved ten stale count pins across three files
    onto the SSOT constant, and moved the four stale baseline assertions
    in ``test_wf433a1_gate_v2_wire.py`` onto the SSOT and the runtime.
    A census that still reported a stale pin in any of those files would
    mean the repair silently kept a literal.

    The four files must still be COVERED by the census — a file that
    vanished from the census is not "clean", it is unseen.
    """
    for target in WFA172_REPAIRED_FILES:
        rows = [p for p in repo_pins if p.path == target]
        assert rows, (
            f"{target} is not covered by the pin census at all — "
            f"'no stale pin' must mean 'seen and clean', not 'not seen'"
        )
        stale = [p for p in rows if p.status == "stale"]
        assert not stale, [p.as_dict() for p in stale]


def test_the_known_current_pins_are_reported_ok(repo_pins: list) -> None:
    """POSITIVE control — the pins Wf-A154 re-synced must be OK.

    2026-11-04 (Wf-A269 wave-5/6) — the site is now looked up by PATH + KIND
    instead of a hard-coded LINE.  A dated re-sync comment inserted above
    the pin MOVES it, and a hard-coded line number then reports a MISSING
    site for a pin that is perfectly fine — the locator itself was a stale
    pin of exactly the kind AGENTS.md §5.5 warns about (one that decays the
    moment the file is edited).  SUPERSEDED RENDERING PRESERVED VERBATIM
    per AGENTS.md section 5.5 rule #6: the three lines

        site = ("molmetal/tests/test_wf415a3_preflight_fail_audit.py", 150)
        assert site in by_site, sorted(by_site)
        assert by_site[site].status == "ok", by_site[site].as_dict()

    were the whole lookup.
    """
    path = "molmetal/tests/test_wf415a3_preflight_fail_audit.py"
    rows = [p for p in repo_pins if p.path == path and p.kind == "summary-total"]
    assert rows, f"no summary-total pin reported for {path}"
    for row in rows:
        assert row.status == "ok", row.as_dict()
        assert row.pinned == pins.live_gate_counts()["all"], row.as_dict()


def test_the_census_is_not_vacuous(census: dict) -> None:
    """VACUITY guard — the reported counts equal the number of listed rows.

    A census that silently drops rows must not be able to report "0
    problems".  Every reported counter is re-derived from the listed rows.
    """
    listed = census["pins"]
    assert len(listed) == census["executable_pin_count"] + census[
        "floor_guard_count"
    ] + census["dynamic_invariant_count"] + len(
        [p for p in listed if p["status"] == "derived-unresolved"]
    ), "the census counters do not add up to the listed findings"
    stale = [p for p in listed if p["status"] == "stale"]
    assert len(stale) == census["stale_pin_count"]
    assert len({p["path"] for p in stale}) == census["stale_pin_file_count"]
    assert len([p for p in listed if p["status"] == "ok"]) == census["ok_pin_count"]
    assert census["files_scanned_python"] > 0
    assert census["files_scanned_prose"] > 0
    assert census["prose_mention_count"] == len(census["prose_mentions"])


def test_the_census_covers_every_wfa169_red_file(repo_pins: list) -> None:
    """Every one of the six RED files must be covered by the pin census.

    Five of the six were RED because of a stale count pin; the sixth
    (``test_wfa159_timeout_verdict_census.py``) was RED because its census
    *artefact* was produced from an older ``pre_flight_check.py`` (a
    sha256 mismatch), NOT because of a count pin — its two count pins are
    already current and must stay ``ok``.
    """
    paths = {p.path for p in repo_pins}
    for known in WF_A169_RED_FILES:
        assert known in paths, (
            f"{known} is not covered by the pin census — the census must "
            f"see every file that holds a gate-count pin"
        )
    wfa159 = [p for p in repo_pins if p.path.endswith("test_wfa159_timeout_verdict_census.py")]
    exact = [p for p in wfa159 if p.status in ("ok", "stale", "unknown-pipeline")]
    assert len(exact) == 2, [p.as_dict() for p in wfa159]
    assert all(p.status == "ok" for p in exact), [p.as_dict() for p in exact]


def test_floor_guards_are_not_reported_as_stale_pins(repo_pins: list) -> None:
    """A monotonic floor (``>= 71``) is never a stale pin.

    The repo's own convention (Wf-345A2) is that a hard equality fires on
    every legitimate gate addition while a floor still catches shrinkage.
    """
    floors = [p for p in repo_pins if p.kind.endswith("-floor")]
    assert floors, "no floor guards found — the scanner has gone blind"
    assert all(p.status == "floor-satisfied" for p in floors), [
        p.as_dict() for p in floors
    ]


def test_a_filtered_subset_count_is_not_a_gate_pin(tmp_path: Path) -> None:
    """A filtered comprehension's length is a SUBSET size, not a pipeline size."""
    body = (
        "from molmetal.scripts.pre_flight_check import build_gates\n"
        "\n"
        "\n"
        "def test_subset():\n"
        "    names = [n for n, _ in build_gates('all') if n.startswith('p3_')]\n"
        "    assert len(names) == 5\n"
    )
    path = _write(tmp_path, "synthetic_subset.py", body)
    found = pins.scan_file_for_pins(path, root=tmp_path)
    assert found == [], [p.as_dict() for p in found]


# ---------------------------------------------------------------------------
# 4 — CPU-ONLY guard (no torch in the import graph)
# ---------------------------------------------------------------------------
def _top_level_imports(path: Path) -> set[str]:
    """Module names imported at any level of ``path`` (AST, no import)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_the_pinner_module_has_no_torch_in_its_import_graph() -> None:
    """CPU-ONLY guard — AST check: no torch anywhere in the import graph."""
    imported = _top_level_imports(MODULE_PATH)
    assert "torch" not in imported, sorted(imported)
    assert "cupy" not in imported, sorted(imported)


def test_the_preflight_module_has_no_top_level_torch_import() -> None:
    """The only module the pinner imports must not pull torch in either."""
    tree = ast.parse(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    top_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert "torch" not in top_level, sorted(top_level)


def test_importing_the_pinner_pulls_no_torch_at_runtime() -> None:
    """CPU-ONLY guard — runtime check in an isolated subprocess.

    The static check cannot see a transitive import, so the module is
    imported in a clean interpreter and ``sys.modules`` is inspected.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, json;"
            "import molmetal.scripts.wfa169_gate_count_pins as m;"
            "print(json.dumps({'torch': 'torch' in sys.modules,"
            " 'all': m.live_gate_counts()['all']}))",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["torch"] is False, payload
    assert payload["all"] == pins.live_gate_counts()["all"], payload


def test_the_module_declares_no_gpu_side_effect() -> None:
    """CPU-ONLY guard — the module must not set a device or spawn a process."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("cuda", "torch.cuda", "device_count", "nvidia-smi"):
        # 'cuda' appears only inside string literals that are *described*,
        # never as an attribute access on a module.
        assert f".{forbidden}" not in source, forbidden
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert not (
                isinstance(node.func, ast.Attribute) and node.func.attr == "run"
            ), "the pinner must not shell out"
