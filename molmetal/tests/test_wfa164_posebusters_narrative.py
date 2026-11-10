"""Wf-A164 regression tests for the PoseBusters-validity narrative guard.

Defect class (Wf-A155 -> Wf-A162 -> Wf-A163 -> Wf-A164)
------------------------------------------------------
A measurement tool embedded POOL-SPECIFIC CLAIMS as string literals inside a
``tag: MEASURED`` artifact.  For the tool's DEFAULT pool the literals were
TRUE; for ANY OTHER pool the tool emitted FALSE statements -- wrong row
count, wrong epoch count, wrong pool path, wrong headline percentages.

The tests below:

* run the REPAIRED tool against a COPY of the pool placed under a DIFFERENT
  directory name and assert the artifact carries NO stale default-pool
  literal and DOES carry the copy pool's own numbers;
* a NEGATIVE CONTROL that restores the OLD hard-coded comparability text and
  asserts the guard REJECTS the run (and writes a ``status: FAIL`` artifact
  and exits non-zero);
* the two holes the reference guard (Wf-A162) still had: a LIST-valued
  narrative path resolving to nothing, and an unaudited new string leaf.

CPU-only by construction: the tool's PoseBusters path is CPU, and the tests
never touch CUDA.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wf_a155_posebusters_validity.py"

sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import wf_a155_posebusters_validity as tool  # noqa: E402


# ---------------------------------------------------------------------------
# Constants: the stale default-pool literals that must never come back
# ---------------------------------------------------------------------------

#: Literals that were TRUE for the default pool and FALSE for any other.
STALE_LITERALS = (
    "2112",
    "704 rows",
    "8.51",
    "25.05",
    "cfm_c8_fullgen_2026_10_24",
)

#: The default pool's directory name -- the copy must NOT be under it.
DEFAULT_POOL_DIRNAME = "cfm_c8_fullgen_2026_10_24"

#: The copy pool lives under a DIFFERENT directory name.
COPY_POOL_DIRNAME = "wfa164_pool_copy"

#: The OLD hard-coded comparability break, verbatim from the pre-Wf-A164
#: script (``COMPARABILITY_BREAKS[0]``).  Restoring it must make the guard
#: REJECT the run -- this is the negative control.
OLD_HARDCODED_BREAK = {
    "id": 1,
    "axis": "DENOMINATOR",
    "frontier": (
        "over 10,000-15,000 generated molecules drawn from a "
        "training-set-distribution sampling"
    ),
    "ours": (
        "ONE pool of 2112 rows from THREE epoch-wise decodes of ONE "
        "checkpoint (704 rows each, 6 seeds per epoch; "
        "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv)"
    ),
    "consequence": "different populations; the two rates are not exchangeable",
}


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

#: Five small, real, RDKit-parseable SMILES with the pool's own columns.
COPY_POOL_ROWS = (
    ("0", "CCO"),
    ("1", "c1ccccc1"),
    ("2", "CC(=O)O"),
    ("0", "c1ccncc1"),
    ("1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
)


def _write_copy_pool(directory: Path) -> Path:
    """Write a small, REAL pool CSV under a DIFFERENT directory name."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "generated_smiles.csv"
    lines = ["epoch,smiles,n_frags_raw,largest_frag_frac"]
    for epoch, smiles in COPY_POOL_ROWS:
        lines.append(f"{epoch},{smiles},1,1.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _cpu_env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def _fake_records(smiles, frag, epoch):
    """Synthetic per-row records -- no PoseBusters, for guard-level tests."""
    return [
        {
            "row_index": idx,
            "smiles": smi,
            "status": tool.STATUS_PB_INVALID,
            "pb_valid": False,
            "n_conformers": 5,
            "n_pb_rows": 5,
            "checks": {},
            "failures": [],
            "missing_checks": [],
            "elapsed_s": 1.0,
            "error": None,
            "n_frags_raw": frag[idx],
            "epoch": epoch[idx],
        }
        for idx, smi in enumerate(smiles)
    ]


def _fake_protocol():
    return {
        "n_conformers": 5,
        "embed_method": "ETKDGv3",
        "optimize_method": "MMFF94",
        "posebusters_config": "mol",
        "embed_random_seed": 42,
        "embed_random_seed_source": "molmetal/validation/posebusters_runner.py::_embed_conformers",
        "wall_seconds": 1.0,
        "n_workers": 2,
        "row_timeout_s": 45.0,
        "device": "CPU",
    }


def _fake_scope(n):
    return {
        "kind": "full_pool",
        "n_selected": n,
        "n_pool": n,
        "sampling": {
            "method": "every row of the pool was taken (no sampling)",
            "seed": None,
            "population": "all pool rows",
        },
        "is_subset": False,
        "label": "FULL POOL -- every row of the pool was checked",
    }


def _build_for_pool(pool_path: Path):
    """Call ``build_artifact`` on a pool with synthetic records."""
    smiles, columns = tool.read_pool(pool_path)
    frag = tool.read_pool_fragment_counts(pool_path)
    epoch = tool.read_pool_epoch_labels(pool_path)
    composition = tool.pool_composition(smiles)
    return tool.build_artifact(
        pool_path=pool_path,
        pool_sha256="deadbeef",
        pool_columns=columns,
        n_data_rows=len(smiles),
        records=_fake_records(smiles, frag, epoch),
        protocol=_fake_protocol(),
        scope=_fake_scope(len(smiles)),
        timestamp="2026-11-01T00:00:00+00:00",
        conformer_probe=None,
        timeout_recheck=None,
        row_timeout_s=45.0,
        composition=composition,
        epoch_labels=epoch,
        fragment_counts=frag,
    )


@pytest.fixture(scope="module")
def copy_pool(tmp_path_factory):
    """A COPY of the pool under a DIFFERENT directory name."""
    base = tmp_path_factory.mktemp("wfa164")
    return _write_copy_pool(base / COPY_POOL_DIRNAME)


@pytest.fixture(scope="module")
def copy_pool_artifact(tmp_path_factory, copy_pool):
    """The REAL artifact produced by the repaired tool on the copy pool."""
    out = tmp_path_factory.mktemp("wfa164_out") / "copy_pool.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pool",
            str(copy_pool),
            "--out",
            str(out),
            "--sample-size",
            "5",
            "--probe-size",
            "0",
            "--workers",
            "4",
            "--row-timeout-s",
            "60",
        ],
        cwd=str(REPO_ROOT),
        env=_cpu_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"tool failed:\n{proc.stdout}\n{proc.stderr}"
    assert out.exists()
    return json.loads(out.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Test 1 -- the copy pool produces the COPY pool's numbers, not the default's
# ---------------------------------------------------------------------------

def test_copy_pool_artifact_has_no_stale_default_pool_literal(copy_pool_artifact):
    """No stale default-pool literal may appear anywhere in the artifact."""
    text = json.dumps(copy_pool_artifact, ensure_ascii=False)
    for literal in STALE_LITERALS:
        assert literal not in text, (
            f"stale default-pool literal {literal!r} leaked into the artifact "
            f"produced for the copy pool"
        )


def test_copy_pool_artifact_reports_the_copy_pools_own_numbers(
    copy_pool_artifact, copy_pool
):
    """The artifact must carry the COPY pool's own row count and path."""
    profile = copy_pool_artifact["source_pool"]["measured_profile"]
    n_rows = len(COPY_POOL_ROWS)

    assert profile["n_data_rows"] == n_rows
    assert profile["path"].endswith(f"{COPY_POOL_DIRNAME}/generated_smiles.csv")
    assert DEFAULT_POOL_DIRNAME not in profile["path"]
    assert sum(
        entry["n_rows"] for entry in profile["rows_per_epoch"]
    ) == n_rows

    # The narrative that used to say "2112 rows ... 704 rows each ... 6 seeds
    # per epoch" must now say THIS pool's numbers.
    note = copy_pool_artifact["source_pool"]["pool_structure_note"]
    assert f"{n_rows} rows" in note

    # ... and the comparability break must name THIS pool's path, not the
    # default pool's.
    ours = copy_pool_artifact["comparability_vs_frontier_band"]["breaks"][0]["ours"]
    assert COPY_POOL_DIRNAME in ours
    assert DEFAULT_POOL_DIRNAME not in ours


def test_copy_pool_artifact_guard_passes(copy_pool_artifact):
    guard = copy_pool_artifact["narrative_self_consistency"]
    assert guard["pass"] is True, json.dumps(guard, indent=2)[:4000]
    assert guard["unbacked_numbers"] == {}
    assert guard["stale_paths"] == {}
    assert guard["unclassified_leaves"] == {}
    assert guard["unresolved_narrative_paths"] == []


# ---------------------------------------------------------------------------
# Test 2 -- NEGATIVE CONTROL: the old hard-coded text must be REJECTED
# ---------------------------------------------------------------------------

def test_negative_control_old_hardcoded_break_fails_the_guard(
    monkeypatch, copy_pool
):
    """Restoring the OLD literal comparability text must FAIL the run."""
    monkeypatch.setattr(
        tool, "build_comparability_breaks", lambda profile: (OLD_HARDCODED_BREAK,)
    )
    with pytest.raises(tool.NarrativeSelfConsistencyError) as excinfo:
        _build_for_pool(copy_pool)

    guard = excinfo.value.guard
    assert guard["pass"] is False
    # The stale row count and the stale pool path are both caught.
    assert "2112" in guard["unbacked_numbers"]
    assert "704" in guard["unbacked_numbers"]
    stale = "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv"
    assert stale in guard["stale_paths"]


def test_negative_control_writes_fail_artifact_and_exits_nonzero(
    monkeypatch, tmp_path, copy_pool
):
    """A guard failure must land ON DISK as ``status: FAIL`` and exit 1."""
    monkeypatch.setattr(
        tool, "build_comparability_breaks", lambda profile: (OLD_HARDCODED_BREAK,)
    )
    out = tmp_path / "guarded_out.json"
    rc = tool.main(
        [
            "--pool",
            str(copy_pool),
            "--out",
            str(out),
            "--sample-size",
            "5",
            "--probe-size",
            "0",
            "--workers",
            "4",
            "--row-timeout-s",
            "60",
        ]
    )
    assert rc == 1
    assert out.exists(), "the FAIL artifact must be written to disk"
    failed = json.loads(out.read_text(encoding="utf-8"))
    assert failed["status"] == "FAIL"
    assert "narrative self-consistency" in failed["failure"]
    assert "2112" in failed["failure"]


# ---------------------------------------------------------------------------
# Test 3 -- the LIST-valued narrative path hole (found in the Wf-A162 guard)
# ---------------------------------------------------------------------------

def test_list_valued_narrative_path_resolves_to_its_elements(monkeypatch):
    """A narrative path designating a LIST must resolve to the ELEMENTS.

    With a naive ``path + "."`` prefix the elements would never match, they
    would be silently classified as MACHINE fields, and every number inside
    them would go unchecked.
    """
    artifact = {
        "narrative": {"lim": ["the count is 4242 rows", "second"]},
        "machine": {"note": "identifier", "n": 7},
    }
    resolved = tool._resolve(artifact, "narrative.lim")
    assert resolved == ["narrative.lim[0]", "narrative.lim[1]"]

    monkeypatch.setattr(tool, "NARRATIVE_PATHS", ("narrative.lim",))
    monkeypatch.setattr(tool, "IDENTIFIER_PATHS", ("machine.note",))
    guard = tool._narrative_self_consistency(artifact)
    assert guard["pass"] is False
    assert "4242" in guard["unbacked_numbers"]
    assert guard["unbacked_numbers"]["4242"] == ["narrative.lim[0]"]


# ---------------------------------------------------------------------------
# Test 4 -- a new, unaudited string leaf must FAIL the guard
# ---------------------------------------------------------------------------

def test_unclassified_string_leaf_fails_the_guard(monkeypatch):
    """Adding prose without classifying it must fail the run."""
    artifact = {
        "narrative": {"note": "hello"},
        "machine": {"n": 1, "extra": "a new string nobody classified"},
    }
    monkeypatch.setattr(tool, "NARRATIVE_PATHS", ("narrative.note",))
    monkeypatch.setattr(tool, "IDENTIFIER_PATHS", ())
    guard = tool._narrative_self_consistency(artifact)
    assert guard["pass"] is False
    assert "machine.extra" in guard["unclassified_leaves"]


# ---------------------------------------------------------------------------
# Test 5 -- a renamed narrative path must not silently disable the check
# ---------------------------------------------------------------------------

def test_unresolved_narrative_path_fails_the_guard(monkeypatch):
    """A narrative path that resolves to nothing must FAIL the guard."""
    artifact = {"narrative": {"note": "hello"}, "machine": {"n": 1}}
    monkeypatch.setattr(tool, "NARRATIVE_PATHS", ("narrative.note", "renamed.key"))
    monkeypatch.setattr(tool, "IDENTIFIER_PATHS", ())
    guard = tool._narrative_self_consistency(artifact)
    assert guard["pass"] is False
    assert guard["unresolved_narrative_paths"] == ["renamed.key"]


# ---------------------------------------------------------------------------
# Test 6 -- the copy pool's provenance is NOT inherited from the default pool
# ---------------------------------------------------------------------------

def test_copy_pool_does_not_inherit_default_pool_provenance(copy_pool_artifact):
    """A pool with no sibling run dir must NOT claim the default's seeds."""
    profile = copy_pool_artifact["source_pool"]["measured_profile"]
    assert profile["run_provenance_available"] is False
    assert profile["n_seeds"] is None
    assert profile["n_checkpoints"] is None
    text = json.dumps(copy_pool_artifact, ensure_ascii=False)
    assert "6 seeds per epoch" not in text
    assert "cfm_yuelbond_wasserstein_seed0" not in text
