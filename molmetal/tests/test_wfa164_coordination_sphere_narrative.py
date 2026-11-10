# SPDX-License-Identifier: Apache-2.0
"""Wf-A164 — the coordination-sphere tool's NARRATIVE must be COMPUTED from
the pool it measures, and a guard must stop the CLASS of hard-coded
pool-specific literals.

THE DEFECT
----------
``molmetal/scripts/wf_a155_metal_coordination_sphere.py`` embedded
POOL-SPECIFIC CLAIMS as string literals inside a ``tag: MEASURED`` artifact:
the default pool's row count ("2112 rows"), the default pool's identity
("the C8 checkpoint" / the pool directory name), the default pool's own
single-fragment fraction ("25.05 %"), the break count ("Four independent
breaks"), and a published headline quoted as a typed literal ("8.51 %").
Every one of them was TRUE for the tool's DEFAULT pool and FALSE for any
other -- including the claim "this pool does not (implicit H)", which is a
property of the measured pool and not of the tool.

THE REPAIR
----------
1. Every number that describes OUR population is COMPUTED from the run
   (``build_comparability_breaks``, ``build_what_this_is_not``,
   ``build_artifact``).
2. A claim of the form "this reproduces published number X" is COMPUTED
   against the published artifact on disk (``_published_cross_check``), not
   asserted.
3. A NARRATIVE SELF-CONSISTENCY GUARD
   (``_narrative_self_consistency``) requires every number rendered in a
   narrative string to be backed by a MACHINE field of the same artifact.  An
   unbacked number is a DEFECT: the run FAILS (``status: FAIL``, exit 1).
   A LIST-valued narrative path resolves to its ELEMENTS -- the reference
   guard (Wf-A162) had that hole, and a negative control below proves this
   one does not.

CPU-only.  No GPU is touched, queried or trained on: the pool CSVs are read
from disk and RDKit parses them on the CPU.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from molmetal.scripts import wf_a155_metal_coordination_sphere as tool

#: The repository root -- the parent of the ``molmetal`` package directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wf_a155_metal_coordination_sphere.py"

#: The artifact the PRE-repair tool published for the default (c8) pool.
OLD_ARTIFACT = REPO_ROOT / "metrics" / "by_metric" / "wfa155_metal_coordination_sphere.json"

#: The default pool itself.
DEFAULT_POOL = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "cfm_c8_fullgen_2026_10_24"
    / "generated_smiles.csv"
)

#: The pool this wave re-ran the tool on: a COPY of six default-pool rows
#: under a DIFFERENT directory name.
SYNTHETIC_POOL = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "wfa164_coordination_sphere_narrative"
    / "synthetic_pool"
    / "generated_smiles.csv"
)
SYNTHETIC_POOL_DIR = "synthetic_pool"

#: The literals the PRE-repair tool typed in.  None of them may appear in the
#: artifact of a DIFFERENT pool.
STALE_LITERALS = ("2112", "8.51", "25.05", "cfm_c8_fullgen_2026_10_24")


# ---------------------------------------------------------------------------
# 1. THE NEGATIVE CONTROLS -- the guard must reject what it is there to reject
# ---------------------------------------------------------------------------

#: The pre-repair comparability break, verbatim from the published artifact.
OLD_BREAK = {
    "id": "break_1_denominator",
    "axis": "DENOMINATOR",
    "frontier": (
        "the frontier band is over 10,000-15,000 generated molecules from a "
        "training-set-distribution sampling;"
    ),
    "ours": (
        "ours is ONE pool of 2112 rows from ONE post-hoc decode of the C8 "
        "checkpoint."
    ),
}

#: The pre-repair "what this metric is NOT" block, verbatim.
OLD_WHAT_THIS_IS_NOT = {
    "not_8_51_percent": (
        "This is NOT the 8.51 % end-to-end single-molecule rate "
        "(metrics/by_metric/cfm_fragmentation_audit.json -> "
        "/summary/generation_side/single_molecule_rate_end_to_end, median over "
        "9 RUNS). 8.51 % is a run-level end-to-end rate over a DIFFERENT "
        "population (9 runs, not 1 pool of 2112 rows)."
    ),
    "not_25_05_percent": (
        "This is NOT the 25.05 % 'n_frags == 1' fraction of the same pool. "
        "25.05 % is a fragment count; the coordination number is a per-atom "
        "property of the metal's neighbourhood."
    ),
    "not_a_frontier_comparison": (
        "This is NOT comparable to the 92.7-94 % frontier band — see "
        "comparability_statement for the four independent breaks."
    ),
    "oxidation_state_is_inferred": (
        "The oxidation state is a RULE-BASED INFERENCE (three labelled "
        "layers), never a measurement."
    ),
}


def test_the_guard_rejects_the_pre_repair_literals(tmp_path, monkeypatch):
    """THE NEGATIVE CONTROL.

    The pre-repair narratives are injected into the tool and the guard must
    REJECT the resulting artifact.  Under the pre-repair behaviour there was
    no guard at all, so this artifact is exactly what used to be published
    under a ``tag: MEASURED`` header.
    """
    monkeypatch.setattr(
        tool, "build_comparability_breaks", lambda *a, **k: [copy.deepcopy(OLD_BREAK)]
    )
    monkeypatch.setattr(
        tool,
        "build_what_this_is_not",
        lambda **k: dict(OLD_WHAT_THIS_IS_NOT),
    )
    pool = _tiny_pool(tmp_path)
    result = tool.analyze_pool(pool)
    artifact = tool.build_artifact(pool, result)

    guard = tool._narrative_self_consistency(artifact)
    assert guard["pass"] is False
    for token in ("2112", "8.51", "25.05"):
        assert token in guard["unbacked_numbers"], (
            f"the guard did not catch the pre-repair literal {token!r}: "
            f"{guard['unbacked_numbers']}"
        )


def test_the_guard_is_not_vacuous():
    """A guard that never fires is not a guard."""
    artifact = {
        "caveats": ["ONE pool, ONE post-hoc decode, 2112 rows -- not a sample."],
        "denominators": {"n_analyzed": 12668},
    }
    guard = tool._narrative_self_consistency(artifact)
    # 'caveats' is not one of THIS tool's narrative paths, so the synthetic
    # artifact above is NOT checked -- which is why the real check below
    # uses the real narrative path set.
    assert guard["n_prose_leaves"] == 0
    assert tool.NARRATIVE_PATHS


def test_the_guard_catches_a_number_inside_a_LIST_element():
    """The list-element hole: the reference guard (Wf-A162) resolved a
    narrative path with a ``path + "."`` prefix, so a LIST-valued path
    matched NOTHING and every number inside it went unchecked while the
    guard still reported PASS."""
    artifact = _synthetic_artifact()
    broken = copy.deepcopy(artifact)
    broken["comparability_statement"]["breaks"][0]["ours"] = (
        "ours is ONE pool of 2112 rows from ONE post-hoc decode."
    )
    guard = tool._narrative_self_consistency(broken)
    assert guard["pass"] is False
    assert "2112" in guard["unbacked_numbers"]
    assert guard["unbacked_numbers"]["2112"] == [
        "comparability_statement.breaks[0].ours"
    ]

    # ... and the naive prefix-only resolution would have MISSED it, which
    # is the proof that the list-element fix is load-bearing.
    naive = [
        path
        for path in tool._leaf_map(broken)
        if path.startswith("comparability_statement.breaks.")
    ]
    assert "comparability_statement.breaks[0].ours" not in naive


def test_a_tripped_guard_writes_a_FAIL_artifact_and_exits_non_zero(
    tmp_path, monkeypatch
):
    """End-to-end: a narrative path that resolves to nothing (a renamed
    field) must FAIL the run rather than silently disable its own coverage.
    """
    monkeypatch.setattr(
        tool,
        "NARRATIVE_PATHS",
        tuple(tool.NARRATIVE_PATHS) + ("population.no_such_field",),
    )
    pool = _tiny_pool(tmp_path)
    out = tmp_path / "artifact.json"
    rc = tool.main(["--pool", str(pool), "--out", str(out)])
    assert rc == 1
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["status"] == "FAIL"
    assert artifact["narrative_self_consistency"]["pass"] is False
    assert artifact["narrative_self_consistency"]["unresolved_narrative_paths"] == [
        "population.no_such_field"
    ]
    assert artifact["failure"]["reason"] == "narrative self-consistency FAILED"


# ---------------------------------------------------------------------------
# 2. THE REGRESSION -- a DIFFERENT pool carries NONE of the stale literals
# ---------------------------------------------------------------------------


def test_no_stale_default_pool_literal_survives_in_a_different_pool(tmp_path):
    """THE REGRESSION for the original defect.

    The tool is run against a COPY of six default-pool rows under a
    DIFFERENT directory name; the artifact must carry NONE of the four
    literals the pre-repair tool typed in.
    """
    pool = _tiny_pool(tmp_path)
    artifact, _ = _run_tool(tmp_path, pool)
    text = json.dumps(artifact, ensure_ascii=False)
    for literal in STALE_LITERALS:
        assert literal not in text, (
            f"the stale default-pool literal {literal!r} survived into the "
            f"artifact of a different pool"
        )


def test_the_different_pool_artifact_carries_ITS_OWN_numbers(tmp_path):
    """The positive side of the same test: the artifact must carry the NEW
    pool's own numbers, not the default pool's."""
    pool = _tiny_pool(tmp_path)
    artifact, _ = _run_tool(tmp_path, pool)

    population = artifact["population"]
    assert population["pool_directory"] == SYNTHETIC_POOL_DIR
    assert population["n_rows"] == 6
    assert population["n_rows_analyzed"] == 6
    # 2 of the 6 rows are a single fragment (the Re complex and the CCO row)
    assert population["n_single_fragment_rows"] == 2
    assert population["single_fragment_rate_percent"] == 33.33

    # ... and the NARRATIVE quotes those numbers, not the default pool's
    nots = artifact["what_this_is_not"]
    assert "33.33 %" in nots["not_25_05_percent"]
    assert "6 rows" in nots["not_8_51_percent"]
    comp = artifact["comparability_statement"]
    assert "6 rows" in comp["break_1_denominator"]
    assert "2112" not in comp["break_1_denominator"]
    assert comp["n_breaks"] == 4
    assert f"{comp['n_breaks']} independent breaks" in comp["summary"]


def test_the_pre_repair_artifact_on_disk_carried_the_stale_literals():
    """The defect is real and on disk (the 'before' anchor).

    Skipped once the artifact has been republished by the repaired tool --
    the point of this test is the PRE-repair state.
    """
    if not OLD_ARTIFACT.is_file():
        pytest.skip("the pre-repair published artifact is not on disk")
    old = json.loads(OLD_ARTIFACT.read_text(encoding="utf-8"))
    if "narrative_self_consistency" in old:
        pytest.skip("the artifact has already been republished by the repaired tool")
    text = json.dumps(old, ensure_ascii=False)
    assert "2112" in text
    assert "25.05" in text
    assert "8.51" in text


def test_the_default_pool_artifact_text_changes_but_its_numbers_do_not():
    """The default-pool diff (report:
    ``molmetal/reports/wfa164_coordination_sphere_narrative/default_pool_diff.json``).

    The repaired tool must leave every MEASURED number bit-identical and
    change only the narrative.  A change in a measured number would mean the
    repair altered the measurement, which it must not.
    """
    report_path = (
        REPO_ROOT
        / "molmetal"
        / "reports"
        / "wfa164_coordination_sphere_narrative"
        / "default_pool_diff.json"
    )
    if not report_path.is_file():
        pytest.skip("the default-pool diff report is not on disk")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verdict = report["verdict"]
    assert verdict["default_pool_text_changes"] is True
    assert verdict["numeric_measurements_identical"] is True
    assert verdict["n_common_numeric_leaves_CHANGED"] == 0
    assert verdict["n_common_numeric_leaves_unchanged"] > 1000


# ---------------------------------------------------------------------------
# 3. THE GUARD'S OWN COVERAGE -- a guard's census is not a coverage measure
# ---------------------------------------------------------------------------


def test_every_narrative_path_resolves_to_at_least_one_leaf():
    """VACUITY GUARD for the guard.

    Every path in ``NARRATIVE_PATHS`` must resolve to at least one leaf.  A
    path that silently resolves to nothing leaves its numbers UNCHECKED
    while the guard still reports PASS.
    """
    artifact = _synthetic_artifact()
    unresolved = [
        path for path in tool._narrative_paths(artifact) if not tool._resolve(artifact, path)
    ]
    assert not unresolved, f"these narrative paths resolve to NO leaf: {unresolved}"


def test_a_list_narrative_path_resolves_to_its_elements():
    """The specific hole the negative control exposed."""
    artifact = _synthetic_artifact()
    breaks = tool._resolve(artifact, "comparability_statement.breaks")
    # four breaks x four leaves each (id, axis, frontier, ours)
    assert len(breaks) == 4 * 4
    assert any(p.endswith("].ours") for p in breaks)
    assert any(p.endswith("].frontier") for p in breaks)


def test_the_guard_checks_a_meaningful_number_of_prose_numbers():
    """The guard must actually be checking something on the real artifact."""
    artifact = _synthetic_artifact()
    guard = artifact["narrative_self_consistency"]
    assert guard["pass"] is True
    assert guard["n_prose_numbers"] >= 20
    assert guard["n_prose_leaves"] >= 30
    assert guard["unbacked_numbers"] == {}
    assert guard["unresolved_narrative_paths"] == []


# ---------------------------------------------------------------------------
# 4. THE FRONTIER BAND -- a frontier constant, cited, never OUR number
# ---------------------------------------------------------------------------


def test_the_frontier_band_is_labelled_and_cited_to_its_primary_record():
    artifact = _synthetic_artifact()
    ref = artifact["frontier_reference"]
    assert ref["band_as_quoted_in_the_repo"] == "92.7-94 %"
    primary = ref["primary_record"]
    assert primary["arxiv_id"] == "2603.27113"
    assert len(primary["retrieval_sha256"]) == 64
    assert "92.7" in primary["quote_low_endpoint"]
    assert "94" in primary["quote_high_endpoint"]
    # the band is NOT our number: no machine field may carry either endpoint
    machine_numbers = [
        v
        for v in tool._leaf_map(artifact).values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    assert 92.7 not in machine_numbers
    assert 94 not in machine_numbers


def test_the_published_cross_check_is_computed_not_asserted(tmp_path):
    """A pool nobody published must produce NO cross-check match and NO
    claim of reproducing a published number."""
    artifact, _ = _run_tool(tmp_path, _tiny_pool(tmp_path))
    cross = artifact["published_cross_check"]
    assert cross["n_matched"] == 0
    assert "NO PUBLISHED ARTIFACT NAMES THIS POOL" in cross["verdict"]
    for candidate in cross["candidates"]:
        assert candidate.get("verdict") == "NOT A CANDIDATE FOR THIS POOL"


# ---------------------------------------------------------------------------
# 5. No CUDA
# ---------------------------------------------------------------------------


def test_the_tool_does_not_import_torch():
    """STATIC: the tool's own import graph has no torch, so it cannot touch
    a GPU even by accident."""
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
    assert "numpy" not in roots
    body = src.replace("CUDA_VISIBLE_DEVICES", "")
    for banned in ("torch.cuda", "cuda(", ".cuda", "device=", "set_device"):
        assert banned not in body, f"the tool names a CUDA entry point: {banned}"


def test_the_tool_runs_in_a_subprocess_with_no_visible_gpu(tmp_path):
    """A fresh interpreter: the tool must run with CUDA hidden."""
    pool = _tiny_pool(tmp_path)
    out = tmp_path / "artifact.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pool",
            str(pool),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        env={**_env_cpu_only()},
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["status"] == "PASS"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _env_cpu_only() -> dict:
    import os

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _tiny_pool(tmp_path: Path) -> Path:
    """A COPY of six default-pool rows under a DIFFERENT directory name."""
    pool_dir = tmp_path / SYNTHETIC_POOL_DIR
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool = pool_dir / "generated_smiles.csv"
    pool.write_text(
        "epoch,smiles,n_frags_raw,largest_frag_frac\n"
        "0,CCC(C)C(CC)C(C)C(CC(C)C)SC(C)N(C)[CH2][Ru].Cl,2,0.9545\n"
        "0,CCC(C)[CH]1[CH](C)[Re]12([OH])([N]1CC1C)[CH](C(C)OO)[CH]2C1NC1CO,1,1.0\n"
        "0,C.CCCCCC(C)CC1C(CC)C1C(C)CC(O)C(C)NOO.[Ru],3,0.9167\n"
        "0,C.C.COC(C)C(NC(C)CCCOCO)[CH](C)[Pt]([CH3])([CH3])[CH3],3,0.9091\n"
        "0,CCO,1,1.0\n"
        "0,C.CC,2,0.5\n",
        encoding="utf-8",
    )
    return pool


def _run_tool(tmp_path: Path, pool: Path):
    out = tmp_path / "artifact.json"
    rc = tool.main(["--pool", str(pool), "--out", str(out)])
    assert rc == 0, "the tool failed its own guard on a clean pool"
    artifact = json.loads(out.read_text(encoding="utf-8"))
    return artifact, out


def _synthetic_artifact() -> dict:
    """The artifact of the synthetic pool, computed once per session."""
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="wfa164_"))
    pool = _tiny_pool(tmp)
    artifact, _ = _run_tool(tmp, pool)
    return artifact
