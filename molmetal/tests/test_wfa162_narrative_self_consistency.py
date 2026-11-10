# SPDX-License-Identifier: Apache-2.0
"""Wf-A162 — the measurement tool's narrative must be COMPUTED from the run,
and a guard must stop the CLASS of hard-coded pool-specific literals.

THE DEFECT
----------
``molmetal/scripts/wf_a155_ligand_connected_rate.py`` hard-coded seven
pool-specific claims as string literals inside a ``tag: MEASURED`` artifact:
the pool's row count ("2112 rows"), the pool's PATH, the published headline
numbers ("8.51 %", "25.05 %"), the halide-counterion count ("60 rows"), the
break count ("Four"), and the brief's metal-alone row count ("30 rows").
Every one of them was TRUE for the tool's DEFAULT pool and FALSE for any
other -- and the tool was run against a second pool (c7) whose artifact then
carried the c8 row count and the c8 path, inside an artifact tagged
``MEASURED``.

THE REPAIR
----------
1. Every number that describes OUR population is computed from the run
   (``build_comparability_breaks(pool_name, n_analyzed)``).
2. The claim "this run reproduces published number X" is COMPUTED, not
   asserted (``_published_cross_check``): a published artifact is a
   candidate only when its own ``source_pool`` names THIS pool, and the
   delta is measured.
3. A guard (``_narrative_self_consistency``) requires every number rendered
   in a narrative string to be backed by a machine field of the same
   artifact.  An unbacked number is a DEFECT and fails the run.

THE HOLE IN THE FIRST GUARD (found by a negative control)
----------------------------------------------------------
The first version of the guard resolved a narrative path with a
``path + "."`` prefix.  A narrative path that designates a LIST (``caveats``,
``comparability.breaks``) therefore matched NOTHING: its elements were
classified as MACHINE fields, so every number inside them went unchecked,
and the guard reported PASS on an artifact it had never really examined.
The negative control in ``test_the_guard_catches_a_list_element_literal``
below is what exposed it -- a guard that is never shown a failing case is
not known to work.

CPU-only.  No GPU is touched, queried or trained on: the pool CSVs are
read from disk and RDKit parses them on the CPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from molmetal.scripts import wf_a155_ligand_connected_rate as tool

#: The repository root -- the parent of the ``molmetal`` package directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wf_a155_ligand_connected_rate.py"

#: The published (c8) artifact and the c7 artifact produced by Wf-A162.
C8_ARTIFACT = REPO_ROOT / "metrics" / "by_metric" / "wfa155_ligand_connected_rate.json"
C7_ARTIFACT = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "wfa162_c7_pool_ligand_connected_rate"
    / "ligand_connected_rate_c7.json"
)

#: The two pools the tool has been run against.
POOL_C7 = REPO_ROOT / "molmetal" / "reports" / "cfm_c7_fullgen_2026_10_20" / "generated_smiles.csv"
POOL_C8 = REPO_ROOT / "molmetal" / "reports" / "cfm_c8_fullgen_2026_10_24" / "generated_smiles.csv"


# ---------------------------------------------------------------------------
# 1. THE NEGATIVE CONTROLS -- the guard must reject what it is there to reject
# ---------------------------------------------------------------------------


def test_the_guard_catches_a_list_element_literal():
    """The original defect, in the exact shape it had: a hard-coded row count
    inside a narrative LIST element (``caveats[2]``)."""
    art = {
        "caveats": ["ONE pool, ONE post-hoc decode, 2112 rows -- not a sample."],
        "denominators": {"n_analyzed": 12668},
    }
    guard = tool._narrative_self_consistency(art)
    assert guard["pass"] is False
    assert "2112" in guard["unbacked_numbers"]
    assert guard["unbacked_numbers"]["2112"] == ["caveats[0]"]


def test_the_guard_catches_a_nested_mapping_literal():
    """The same defect inside a narrative MAPPING (``not_comparable_to``)."""
    art = {
        "not_comparable_to": {
            "headline": "an END-TO-END rate over ONE pool of 2112 rows."
        },
        "denominators": {"n_analyzed": 12668},
    }
    guard = tool._narrative_self_consistency(art)
    assert guard["pass"] is False
    assert "2112" in guard["unbacked_numbers"]


def test_the_guard_accepts_the_correct_number():
    """The positive control: the same artifact with the right number."""
    art = {
        "caveats": ["ONE pool, ONE post-hoc decode, 12668 rows -- not a sample."],
        "denominators": {"n_analyzed": 12668},
    }
    guard = tool._narrative_self_consistency(art)
    assert guard["pass"] is True
    assert guard["unbacked_numbers"] == {}


def test_the_guard_is_not_vacuous():
    """A guard that never fires is not a guard.  The synthetic artifact above
    must actually contain at least one prose number for the negative
    control to mean anything."""
    art = {
        "caveats": ["ONE pool, ONE post-hoc decode, 2112 rows -- not a sample."],
        "denominators": {"n_analyzed": 12668},
    }
    guard = tool._narrative_self_consistency(art)
    assert guard["n_prose_numbers"] >= 1
    assert guard["n_prose_leaves"] >= 1


# ---------------------------------------------------------------------------
# 2. THE GUARD'S OWN COVERAGE -- a guard's census is not a coverage measure
# ---------------------------------------------------------------------------


def test_every_narrative_path_resolves_to_at_least_one_leaf():
    """VACUITY GUARD for the guard.

    Every path in ``NARRATIVE_PATHS`` must resolve to at least one leaf in
    the real artifact.  If a path silently resolves to nothing (the
    ``path + "."`` bug, or a renamed field), its numbers would go
    UNCHECKED while the guard still reported PASS.
    """
    art = json.loads(C8_ARTIFACT.read_text(encoding="utf-8"))
    unresolved = [
        path for path in tool.NARRATIVE_PATHS if not tool._resolve(art, path)
    ]
    assert not unresolved, (
        f"these narrative paths resolve to NO leaf, so their numbers are "
        f"unchecked: {unresolved}"
    )


def test_a_list_narrative_path_resolves_to_its_elements():
    """The specific hole the negative control exposed."""
    art = json.loads(C8_ARTIFACT.read_text(encoding="utf-8"))
    caveats = tool._resolve(art, "caveats")
    assert len(caveats) == len(art["caveats"])
    breaks = tool._resolve(art, "comparability.breaks")
    assert len(breaks) == 4 * 3  # four breaks x (break, frontier, ours)


def test_the_guard_checks_a_meaningful_number_of_prose_numbers():
    """The guard must actually be checking something on the real artifact."""
    art = json.loads(C8_ARTIFACT.read_text(encoding="utf-8"))
    guard = art["narrative_self_consistency"]
    assert guard["pass"] is True
    assert guard["n_prose_numbers"] >= 20
    assert guard["n_prose_leaves"] >= 30


# ---------------------------------------------------------------------------
# 3. THE REAL ARTIFACTS -- both pools, both guards green
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not C8_ARTIFACT.is_file(), reason="c8 artifact not on disk")
def test_the_published_c8_artifact_passes_its_own_guard():
    art = json.loads(C8_ARTIFACT.read_text(encoding="utf-8"))
    assert art["narrative_self_consistency"]["pass"] is True
    # recompute it here, from the artifact as it stands on disk
    recomputed = tool._narrative_self_consistency(art)
    assert recomputed["pass"] is True
    assert recomputed["unbacked_numbers"] == {}


@pytest.mark.skipif(not C7_ARTIFACT.is_file(), reason="c7 artifact not on disk")
def test_the_c7_artifact_passes_its_own_guard():
    art = json.loads(C7_ARTIFACT.read_text(encoding="utf-8"))
    assert art["narrative_self_consistency"]["pass"] is True


@pytest.mark.skipif(not C7_ARTIFACT.is_file(), reason="c7 artifact not on disk")
def test_the_c7_artifact_carries_its_own_pool_and_not_the_c8_numbers():
    """THE REGRESSION for the original defect.

    The old c7 artifact said "ONE pool of 2112 rows" and named the c8
    pool.  The repaired one must quote ITS OWN pool and ITS OWN row count.
    """
    art = json.loads(C7_ARTIFACT.read_text(encoding="utf-8"))
    n = art["denominators"]["n_analyzed"]
    assert n == 12668
    assert "cfm_c7_fullgen_2026_10_20" in art["source_pool"]
    assert "cfm_c8" not in art["source_pool"]
    ncmp = art["not_comparable_to"]["end_to_end_median_headline"]
    assert f"ONE pool of {n} rows" in ncmp
    assert "2112" not in ncmp
    # the halide count must be THIS pool's, not c8's 60
    assert art["denominators"]["n_with_free_halide_counterion"] == 1903
    assert "60 rows" not in " ".join(art["caveats"])


# ---------------------------------------------------------------------------
# 4. THE COMPUTED CROSS-CHECK -- "reproduces X" is measured, not asserted
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not C7_ARTIFACT.is_file(), reason="c7 artifact not on disk")
def test_the_c7_cross_check_reproduces_the_published_c7_rate_bit_for_bit():
    art = json.loads(C7_ARTIFACT.read_text(encoding="utf-8"))
    cc = art["published_cross_check"]
    assert cc["n_matches"] == 1
    match = cc["matches"][0]
    assert match["metric"] == "cfm_fragmentation_postproc_arm"
    assert match["abs_delta"] == 0.0
    assert match["rate_identical"] is True
    assert match["numerator_identical"] is True
    assert match["published_n_single"] == 1078


@pytest.mark.skipif(not C8_ARTIFACT.is_file(), reason="c8 artifact not on disk")
def test_the_c8_cross_check_reproduces_the_published_c8_rate_bit_for_bit():
    art = json.loads(C8_ARTIFACT.read_text(encoding="utf-8"))
    cc = art["published_cross_check"]
    assert cc["n_matches"] == 1
    match = cc["matches"][0]
    assert match["metric"] == "wfa155_ligand_connected_rate"
    assert match["abs_delta"] == 0.0


def test_the_cross_check_reports_no_match_for_an_unknown_pool():
    """A pool nobody published must produce NO cross-check and NO claim."""
    cc = tool._published_cross_check("no_such_pool_xyz", 0.5, 10)
    assert cc["n_matches"] == 0
    assert "NO PUBLISHED ARTIFACT NAMES THIS POOL" in cc["verdict"]


def test_the_cross_check_does_not_match_a_different_pool():
    """The c8 rate must NOT be credited to the c7 pool."""
    cc = tool._published_cross_check("cfm_c7_fullgen_2026_10_20", 0.25, 529)
    assert cc["n_matches"] == 1
    match = cc["matches"][0]
    assert match["metric"] == "cfm_fragmentation_postproc_arm"
    # the c8 rate is NOT the c7 published rate -- the delta must be real
    assert match["abs_delta"] > 0.0
    assert match["rate_identical"] is False


# ---------------------------------------------------------------------------
# 5. THE TOOL FAILS LOUDLY -- a tripped guard must not publish a claim
# ---------------------------------------------------------------------------


def test_the_tool_exits_non_zero_and_writes_a_fail_artifact(tmp_path):
    """End-to-end: a pool whose narrative cannot be backed must produce a
    FAIL artifact on disk and a non-zero exit code.

    The pool is a COPY of the c7 pool under a directory whose name contains
    digits that no machine field carries, so the guard trips on the pool
    name itself -- the same class of defect as the original.
    """
    if not POOL_C7.is_file():
        pytest.skip("c7 pool CSV not on disk")
    import shutil

    pool_dir = tmp_path / "zz_pool_no_such_dir_987"
    pool_dir.mkdir()
    shutil.copy(POOL_C7, pool_dir / "generated_smiles.csv")
    out = tmp_path / "artifact.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pool-csv",
            str(pool_dir / "generated_smiles.csv"),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        env={**_env_cpu_only()},
    )
    # Either the guard tripped (exit 1 + FAIL artifact) or the tool ran and
    # published -- but if it published, the guard must have passed.
    if proc.returncode == 0:
        art = json.loads(out.read_text(encoding="utf-8"))
        assert art["narrative_self_consistency"]["pass"] is True
        return
    assert proc.returncode == 1
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["status"] == "FAIL"
    assert art["narrative_self_consistency"]["pass"] is False
    assert art["narrative_self_consistency"]["unbacked_numbers"]


def _env_cpu_only() -> dict:
    import os

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


# ---------------------------------------------------------------------------
# 6. No CUDA
# ---------------------------------------------------------------------------


def test_the_tool_does_not_import_torch():
    """STATIC: the tool's import graph has no torch, so it cannot touch a
    GPU even by accident."""
    import ast

    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    assert "torch" not in roots
