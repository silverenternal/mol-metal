# SPDX-License-Identifier: Apache-2.0
"""Wf-A164 -- the bond-threshold measurement tool's narrative must be COMPUTED
from the pool it actually read.

THE DEFECT (class, established by Wf-A155 -> Wf-A162 -> Wf-A163)
----------------------------------------------------------------
``molmetal/scripts/wfa156_bond_threshold_frontier.py`` embedded POOL-SPECIFIC
CLAIMS as string literals inside a measurement artefact.  Every one of them
was TRUE for the tool's DEFAULT pool
(``molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv``) and
FALSE for any other pool:

* ``population_cross_check.note`` carried four counts and a denominator as
  literals ("529/2112 rows with one fragment, 2082/2112 metal atoms sharing a
  fragment with a non-metal, 30/2112 metal atoms alone, 0/2112 rows containing
  '->'") while the very machine fields two lines above carried the SAME
  quantities -- a transcription that could not disagree with them, because
  nobody ever checked;
* the comparability caveat carried "2,112-row", the default pool's PATH and
  "every one of our pool rows carries a metal (metal_ratio 1.0)" as literals;
* the report quoted two lines of the DEFAULT run's ``master.log`` as literals.

Measured on the c7 pool (12669 rows, 12668 of them carrying a metal), the old
tool's caveat would have claimed "metal_ratio 1.0" and "every one of our pool
rows carries a metal" -- both FALSE for that pool.

THE REPAIR
----------
1. Every number describing OUR population is COMPUTED from the run.
2. The narrative is checked against the artefact's OWN machine fields: an
   unbacked number FAILS the run (``status: FAIL``, exit 1).
3. The artefact may name exactly ONE pool path -- the one the run read.

THE NEGATIVE CONTROL
--------------------
A guard that is never shown a failing case is not known to work.  The two
tests below re-inject the OLD literals into a REAL artefact and assert that
the guards reject them.  Under the pre-repair behaviour those two assertions
FAIL (the literals are simply rendered into the artefact).

CPU-only.  No GPU is touched, queried or trained on.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest

from molmetal.scripts import wfa156_bond_threshold_frontier as tool

#: The repository root -- the parent of the ``molmetal`` package directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wfa156_bond_threshold_frontier.py"

#: The tool's DEFAULT pool -- the one whose literals must never leak into an
#: artefact produced against a different pool.
DEFAULT_POOL = REPO_ROOT / "molmetal" / "reports" / "cfm_c8_fullgen_2026_10_24" / "generated_smiles.csv"

#: The default pool's literals.  Every one of them is TRUE for the default
#: pool and FALSE for any other.
STALE_LITERALS = ("2112", "529", "2082", "cfm_c8_fullgen_2026_10_24")

#: The numeric ones.  These are checked as STANDALONE numbers: "529" occurs
#: as a SUBSTRING of unrelated floats (e.g. 0.8802698145025295), so a naive
#: substring test on the whole JSON would be flaky and wrong.
STALE_NUMBERS = ("2112", "529", "2082")

#: The default pool's DIRECTORY name -- a path fragment, checked as a plain
#: substring anywhere in the artefact.
STALE_POOL_DIR = "cfm_c8_fullgen_2026_10_24"


def _narrative_text(report: Dict[str, Any]) -> str:
    """The artefact's prose -- where a pool-specific claim can live."""
    return (
        report["comparability_caveat"]
        + "\n"
        + report["pool"]["population_cross_check"]["note"]
    )

#: The OLD ``population_cross_check.note``, verbatim from the pre-repair
#: artefact.  Used only by the negative controls.
OLD_NOTE = (
    "Computed by THIS tool from the same RDKit parse, so a reader can "
    "confirm the population matches the branch brief: 529/2112 rows "
    "with one fragment, 2082/2112 metal atoms sharing a fragment with "
    "a non-metal, 30/2112 metal atoms alone, 0/2112 rows containing "
    "'->'."
)

#: The OLD comparability caveat -- a MODULE CONSTANT, so it rendered the
#: default pool's row count, path and metal ratio for EVERY pool the tool was
#: ever pointed at.  The three fragments below are the pool-specific ones;
#: they are spliced into the repaired caveat to reproduce the old behaviour
#: exactly.
OLD_CAVEAT_DENOMINATOR = (
    "(1) DENOMINATOR.  Their rates are computed over 10,000 generated molecules per\n"
    "    method (App. G.3: \"We sample 10,000 molecules\"; Table 5 caption: \"All\n"
    "    rates are computed over 10,000 generated molecules\").  Ours are over the\n"
    "    2,112-row raw decode pool\n"
    "    molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv -- ONE\n"
    "    decode of ONE checkpoint.\n"
)
OLD_CAVEAT_METAL = (
    "that: every one of our pool rows carries a metal (metal_ratio 1.0) and every\n"
    "    paper cohort is all-organic.\n"
)

#: A small synthetic pool.  Six rows, all with a metal, one of them with an
#: arrow, so every count in the artefact differs from the default pool's.
SYNTHETIC_ROWS: List[tuple] = [
    ("0", "CCC(C)C(CC)C(C)C(CC(C)C)SC(C)N(C)[CH2][Ru].Cl", 2, 0.9545),
    ("0", "C.C.COC(C)C(NC(C)CCCOCO)[CH](C)[Pt]([CH3])([CH3])[CH3]", 3, 0.9091),
    ("0", "CCC(C)[CH]1[CH](C)[Re]12([OH])([N]1CC1C)[CH](C(C)OO)[CH]2C1NC1CO", 1, 1.0),
    ("0", "CC[CH2][Os]([CH3])([CH3])([CH3])[O]C(CC)C(C)CCC(CC(CN[AsH2])OC)C(O)C", 2, 0.9744),
    ("0", "C[CH2][Rh]([CH3])([CH3])[CH3].Cl", 2, 0.8),
    ("0", "CCO->CC", 1, 1.0),
]


def _write_pool(directory: Path, rows: List[tuple]) -> Path:
    """Write a pool CSV (the tool's own header) under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "generated_smiles.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["epoch", "smiles", "n_frags_raw", "largest_frag_frac"])
        for row in rows:
            writer.writerow(list(row))
    return path


def _run(pool_path: Path, out_dir: Path) -> tuple:
    """Run the tool in-process and return the artefacts it wrote.

    Returns ``(report.json as a dict, report.md as text)`` -- the JSON is
    the machine payload, the markdown is the TRACKED, published form.  The
    defect class lives in published prose, so both are checked.
    """
    rc = tool.main(
        [
            "--pool",
            str(pool_path),
            "--skip-baseline",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0, f"the tool exited {rc}"
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    markdown = (out_dir / "report.md").read_text(encoding="utf-8")
    return report, markdown


@pytest.fixture(scope="module")
def synthetic_artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """A SMALL synthetic pool, in a directory that is NOT the default one."""
    root = tmp_path_factory.mktemp("wfa164")
    pool_dir = root / "synthetic_pool_2026_11_01"
    pool_path = _write_pool(pool_dir, SYNTHETIC_ROWS)
    return _run(pool_path, root / "out_synthetic")


@pytest.fixture(scope="module")
def copied_artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """A COPY of the default pool, under a DIFFERENT directory name.

    The copy is truncated to its first 40 rows so that every count differs
    from the default pool's -- a same-content copy would legitimately carry
    the default pool's own row count, and asserting its absence would then
    be asserting the wrong thing.
    """
    root = tmp_path_factory.mktemp("wfa164_copy")
    pool_dir = root / "cfm_c9_relabelled_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    copied = pool_dir / "generated_smiles.csv"
    with DEFAULT_POOL.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [next(reader) for _ in range(40)]
    with copied.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return _run(copied, root / "out_copy")


# ---------------------------------------------------------------------------
# 1. The tool must not emit the default pool's literals for another pool
# ---------------------------------------------------------------------------


def test_synthetic_pool_artifact_has_no_stale_default_pool_literal(
    synthetic_artifact: tuple,
) -> None:
    """No default-pool literal survives in a NON-default pool's artefact."""
    report, markdown = synthetic_artifact
    narrative = _narrative_text(report)
    for literal in STALE_NUMBERS:
        assert not tool._standalone_in_any(literal, [narrative]), (
            f"the narrative still carries the default pool's count {literal!r}"
        )
    blob = json.dumps(report)
    assert STALE_POOL_DIR not in blob, (
        "the artefact still names the default pool's directory"
    )
    # the PUBLISHED markdown is the tracked artefact -- it must be clean too
    assert STALE_POOL_DIR not in markdown, (
        "the published markdown still names the default pool's directory"
    )
    for literal in STALE_NUMBERS:
        assert not tool._standalone_in_any(literal, [markdown]), (
            f"the published markdown still carries the default pool's count {literal!r}"
        )


def test_synthetic_pool_artifact_carries_its_own_numbers(
    synthetic_artifact: tuple,
) -> None:
    """The narrative carries THIS pool's own counts, not the default's."""
    report, markdown = synthetic_artifact
    pcc = report["pool"]["population_cross_check"]
    n_rows = report["pool_n_rows"]
    assert n_rows == len(SYNTHETIC_ROWS)
    note = pcc["note"]
    assert f"{pcc['rows_with_one_fragment']}/{n_rows}" in note
    assert (
        f"{pcc['metal_atoms_sharing_a_fragment_with_a_nonmetal']}/"
        f"{pcc['metal_atoms_total']}" in note
    )
    assert f"{pcc['metal_atoms_alone_in_their_fragment']}/{pcc['metal_atoms_total']}" in note
    assert f"{pcc['rows_containing_arrow']}/{n_rows}" in note
    # the caveat names THIS pool
    assert report["pool_path_display"] in report["comparability_caveat"]
    assert str(n_rows) in report["comparability_caveat"]
    # and the published markdown reports THIS pool's own counts
    assert f"rows read: **{n_rows}**" in markdown
    assert (
        f"rows carrying a metal: **{pcc['rows_with_metal']}/{n_rows}**" in markdown
    )


def test_copied_pool_under_a_different_directory_name(
    copied_artifact: tuple,
) -> None:
    """A COPY of the pool, under a DIFFERENT directory name, is self-consistent."""
    report, markdown = copied_artifact
    blob = json.dumps(report)
    narrative = _narrative_text(report)
    for literal in STALE_NUMBERS:
        assert not tool._standalone_in_any(literal, [narrative]), (
            f"the narrative still carries the default pool's count {literal!r}"
        )
    assert STALE_POOL_DIR not in blob, (
        "the artefact still names the default pool's directory"
    )
    assert STALE_POOL_DIR not in markdown
    for literal in STALE_NUMBERS:
        assert not tool._standalone_in_any(literal, [markdown]), (
            f"the published markdown still carries the default pool's count {literal!r}"
        )
    assert report["pool_n_rows"] == 40
    assert "cfm_c9_relabelled_pool" in report["comparability_caveat"]
    assert report["narrative_self_consistency"]["pass"]
    assert report["pool_path_scan"]["pass"]


def test_guards_pass_on_a_non_default_pool(synthetic_artifact: tuple) -> None:
    """Both guards PASS on a pool that is not the default one."""
    report, _ = synthetic_artifact
    assert report["narrative_self_consistency"]["pass"]
    assert report["narrative_self_consistency"]["unbacked_numbers"] == {}
    assert report["pool_path_scan"]["pass"]
    assert report["pool_path_scan"]["n_foreign_pool_paths"] == 0


# ---------------------------------------------------------------------------
# 2. NEGATIVE CONTROLS -- the guards must reject the OLD behaviour
# ---------------------------------------------------------------------------


def test_the_guard_catches_the_old_hard_coded_note(
    synthetic_artifact: tuple,
) -> None:
    """Re-injecting the OLD note makes the guard FAIL.  Under the old
    behaviour the note was simply rendered into the artefact."""
    report, _ = synthetic_artifact
    mutated = json.loads(json.dumps(report))
    mutated["pool"]["population_cross_check"]["note"] = OLD_NOTE
    guard = tool._narrative_self_consistency(mutated)
    assert guard["pass"] is False
    unbacked = set(guard["unbacked_numbers"])
    assert unbacked & {"2112", "529", "2082"}, (
        f"the guard did not flag the old pool's counts: {sorted(unbacked)}"
    )


def test_the_path_scan_catches_the_old_hard_coded_pool_path(
    synthetic_artifact: tuple,
) -> None:
    """Re-injecting the OLD pool path makes the path scan FAIL."""
    report, _ = synthetic_artifact
    mutated = json.loads(json.dumps(report))
    mutated["comparability_caveat"] = mutated["comparability_caveat"].replace(
        mutated["pool_path_display"],
        "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv",
    )
    scan = tool._pool_path_scan(mutated, Path(mutated["pool_path"]))
    assert scan["pass"] is False
    assert "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv" in (
        scan["foreign_pool_paths"]
    )


def test_the_old_behaviour_would_have_failed_this_file(
    synthetic_artifact: tuple,
) -> None:
    """THE NEGATIVE CONTROL, in full: the OLD tool's artefact for a NON-default
    pool carries the default pool's literals, and both guards reject it.

    Under the pre-repair behaviour the two assertions made by
    ``test_synthetic_pool_artifact_has_no_stale_default_pool_literal`` FAIL:
    the narrative literally contains "529/2112" and
    "cfm_c8_fullgen_2026_10_24" for a pool that is neither the default one
    nor 2112 rows long.
    """
    report, _ = synthetic_artifact
    # -- reproduce the OLD artefact for THIS (non-default) pool ------------
    old = json.loads(json.dumps(report))
    old["pool"]["population_cross_check"]["note"] = OLD_NOTE
    caveat = old["comparability_caveat"]
    head, _, rest = caveat.partition("(1) DENOMINATOR")
    _, _, tail = rest.partition("(2) METRIC")
    caveat = head + OLD_CAVEAT_DENOMINATOR + "(2) METRIC" + tail
    caveat = re.sub(
        r"that: .*?all-organic\.",
        OLD_CAVEAT_METAL.strip(),
        caveat,
        flags=re.S,
    )
    old["comparability_caveat"] = caveat

    # -- the old artefact DOES carry the default pool's literals ----------
    pcc = old["pool"]["population_cross_check"]
    # the synthetic pool deliberately has a row WITHOUT a metal, so the old
    # "every one of our pool rows carries a metal (metal_ratio 1.0)" is not
    # merely stale but FALSE for this pool.
    assert pcc["rows_with_metal"] < old["pool_n_rows"]
    assert pcc["rows_with_metal_frac"] != 1.0
    old_narrative = _narrative_text(old)
    for literal in STALE_NUMBERS:
        assert tool._standalone_in_any(literal, [old_narrative]), (
            f"the reproduced OLD artefact does not carry {literal!r}"
        )
    assert STALE_POOL_DIR in old_narrative
    assert "2,112-row raw decode pool" in old_narrative
    assert "metal_ratio 1.0" in old_narrative

    # -- and both guards reject it ----------------------------------------
    guard = tool._narrative_self_consistency(old)
    assert guard["pass"] is False
    assert set(guard["unbacked_numbers"]) & {"2112", "529", "2082"}
    scan = tool._pool_path_scan(old, Path(old["pool_path"]))
    assert scan["pass"] is False
    assert "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv" in (
        scan["foreign_pool_paths"]
    )


def test_the_tool_fails_the_run_when_a_narrative_number_is_unbacked(
    tmp_path: Path,
) -> None:
    """An unbacked number FAILS the run: ``status: FAIL`` on disk, exit 1."""
    pool_dir = tmp_path / "synthetic_pool"
    pool_path = _write_pool(pool_dir, SYNTHETIC_ROWS)
    out_dir = tmp_path / "out"

    original = tool.build_report

    def broken_build_report(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        report = original(*args, **kwargs)
        # Re-inject the OLD hard-coded note -- the exact defect Wf-A164 fixed.
        report["pool"]["population_cross_check"]["note"] = OLD_NOTE
        report["narrative_self_consistency"] = tool._narrative_self_consistency(report)
        report["pool_path_scan"] = tool._pool_path_scan(report, Path(report["pool_path"]))
        return report

    tool.build_report = broken_build_report  # type: ignore[assignment]
    try:
        rc = tool.main(
            ["--pool", str(pool_path), "--skip-baseline", "--out-dir", str(out_dir)]
        )
    finally:
        tool.build_report = original  # type: ignore[assignment]

    assert rc == 1
    written = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert written["status"] == "FAIL"
    assert "unbacked_numbers" in written["failure"]
    assert set(written["narrative_self_consistency"]["unbacked_numbers"]) & {
        "2112",
        "529",
        "2082",
    }


# ---------------------------------------------------------------------------
# 3. The default pool still measures the same thing
# ---------------------------------------------------------------------------


def test_default_pool_note_is_unchanged_in_substance() -> None:
    """The default pool's note still carries the default pool's own counts.

    The repair changed HOW the note is produced, not WHAT it says about the
    default pool: the four counts and the denominator are identical.
    """
    assert DEFAULT_POOL.is_file()
    pool = tool.analyse_pool(tool.read_pool(DEFAULT_POOL), {})
    report = tool.build_report(
        pool_path=DEFAULT_POOL,
        pool=pool,
        baseline=None,
        tmqm_root=tool.DEFAULT_TMQM_ROOT,
        lam_list=tool.LAMBDAS,
        conventions=tool.CONVENTIONS,
        baseline_codes=None,
    )
    pcc = report["pool"]["population_cross_check"]
    assert report["pool_n_rows"] == 2112
    assert pcc["rows_with_one_fragment"] == 529
    assert pcc["metal_atoms_sharing_a_fragment_with_a_nonmetal"] == 2082
    assert pcc["metal_atoms_alone_in_their_fragment"] == 30
    assert pcc["metal_atoms_total"] == 2112
    assert pcc["rows_containing_arrow"] == 0
    assert pcc["rows_with_metal"] == 2112
    assert "529/2112 rows with one fragment" in pcc["note"]
    assert "2082/2112 metal atoms sharing a fragment" in pcc["note"]
    assert "30/2112 metal atoms alone" in pcc["note"]
    assert "0/2112 rows containing '->'" in pcc["note"]
    assert report["narrative_self_consistency"]["pass"]
    assert report["pool_path_scan"]["pass"]


def test_a_pool_without_a_master_log_is_reported_as_such(tmp_path: Path) -> None:
    """The run-log excerpt is READ from the pool's own run directory.

    With no ``master.log`` next to the pool the tool says so instead of
    quoting another run's log.
    """
    pool_dir = tmp_path / "no_log_pool"
    pool_path = _write_pool(pool_dir, SYNTHETIC_ROWS)
    report, markdown = _run(pool_path, tmp_path / "out")
    assert report["decoder_run_log"]["present"] is False
    assert "master.log" in report["decoder_run_log"]["source"]
    assert report["narrative_self_consistency"]["pass"]
    # the markdown must NOT claim this run's master.log was measured
    assert "MEASURED from this run's `master.log`" not in markdown
    assert "NO `master.log` next to this pool" in markdown
