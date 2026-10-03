"""Wf-44 Phase 3 — tests for the data leakage diagnostic.

================================================================
Validates that ``molmetal/scripts/check_data_leakage.py`` correctly:

1. **Artefact existence** — ``leakage_check.json`` is written to the
   expected location after a CLI run (1 test).
2. **Schema correctness** — top-level keys + per-dataset + per-split
   keys, JSON round-trip (3 tests).
3. **Diagnostic helpers** — ``canonicalise_smiles``, ``compute_scaffolds``,
   ``random_split_indices``, ``scaffold_split_indices``,
   ``compute_overlap`` (5 tests).
4. **End-to-end drivers** — ``run_dataset`` returns the expected
   ``DatasetLeakageReport`` shape, scaffold split is strict (1 test).
5. **Honest-framing property** — random split on a heavily-duplicated
   dataset shows non-zero scaffold overlap (1 test).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Import helpers — load the script as a module
# ---------------------------------------------------------------------------
def _import_script():
    """Import ``check_data_leakage`` from its file path."""
    import importlib.util
    import sys as _sys

    mod_name = "check_data_leakage"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(_REPO_ROOT / "molmetal" / "scripts" / "check_data_leakage.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_cli_once() -> None:
    """Run the script once so the artefact exists."""
    script = _REPO_ROOT / "molmetal" / "scripts" / "check_data_leakage.py"
    out = __import__("subprocess").run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=180,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"check_data_leakage CLI failed: {out.stderr or out.stdout}"
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPORT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf44_r2_debug"
LEAKAGE_PATH = REPORT_DIR / "leakage_check.json"

PLATINAI_CSV = _REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"
METALCYTOTOX_CSV = (
    _REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)


# ---------------------------------------------------------------------------
# 1. CLI / artefact existence
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not PLATINAI_CSV.exists() and not METALCYTOTOX_CSV.exists(),
    reason="Neither platinai_5k.csv nor metalcytotox_5metal_multitask_v4.csv is available",
)
def test_leakage_check_json_exists() -> None:
    """The leakage_check.json artefact exists after a CLI run."""
    _run_cli_once()
    assert LEAKAGE_PATH.exists(), f"missing: {LEAKAGE_PATH}"


def test_cli_exits_zero() -> None:
    """``check_data_leakage.py --help`` exits 0; the script parses cleanly."""
    import subprocess
    script = _REPO_ROOT / "molmetal" / "scripts" / "check_data_leakage.py"
    out = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert out.returncode == 0


# ---------------------------------------------------------------------------
# 2. Schema correctness
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def leakage() -> dict:
    """Load ``leakage_check.json`` from the canonical location."""
    _run_cli_once()
    with open(LEAKAGE_PATH) as fh:
        return json.load(fh)


@pytest.mark.skipif(
    not PLATINAI_CSV.exists() and not METALCYTOTOX_CSV.exists(),
    reason="No data CSV available",
)
def test_leakage_top_level_keys(leakage: dict) -> None:
    """Top-level keys include workflow, title, n_datasets, datasets."""
    expected = {"workflow", "title", "description", "n_datasets", "datasets"}
    missing = expected - set(leakage.keys())
    assert not missing, f"missing top-level keys: {missing}"


@pytest.mark.skipif(
    not PLATINAI_CSV.exists() and not METALCYTOTOX_CSV.exists(),
    reason="No data CSV available",
)
def test_leakage_dataset_keys(leakage: dict) -> None:
    """Each dataset entry carries the Wf-44 contract keys."""
    expected_dataset = {
        "dataset_name", "csv_path", "n_total",
        "n_unique_smiles", "n_unique_scaffolds",
        "n_parse_failures", "splits", "overlaps",
        "strict_pass", "summary",
    }
    expected_overlap_pair = {
        "n_intersect_smiles", "n_intersect_scaffolds",
        "n_unique_train_smiles", "n_unique_test_smiles",
        "n_unique_train_scaffolds", "n_unique_test_scaffolds",
        "frac_test_smiles_seen_in_train",
        "frac_test_scaffolds_seen_in_train",
    }
    for d in leakage["datasets"]:
        missing = expected_dataset - set(d.keys())
        assert not missing, f"missing dataset keys: {missing}"
        for split_name, overlap_by_pair in d["overlaps"].items():
            for pair_name, overlap in overlap_by_pair.items():
                missing_pair = expected_overlap_pair - set(overlap.keys())
                assert not missing_pair, (
                    f"missing overlap keys for {split_name}/{pair_name}: "
                    f"{missing_pair}"
                )


@pytest.mark.skipif(
    not PLATINAI_CSV.exists() and not METALCYTOTOX_CSV.exists(),
    reason="No data CSV available",
)
def test_leakage_dataset_counts(leakage: dict) -> None:
    """``n_datasets`` matches the number of dataset entries."""
    assert leakage["n_datasets"] == len(leakage["datasets"])
    assert leakage["n_datasets"] >= 1


# ---------------------------------------------------------------------------
# 3. Diagnostic helpers
# ---------------------------------------------------------------------------
def test_canonicalise_smiles_handles_failures() -> None:
    """``canonicalise_smiles`` returns the right ``n_failures`` count."""
    mod = _import_script()
    pytest.importorskip("rdkit")
    out, n_fail = mod.canonicalise_smiles(
        ["CCO", "c1ccccc1", "totally-not-a-smiles"],
    )
    assert out[0] == "CCO"
    assert out[1] == "c1ccccc1"
    # The third entry fails to parse — gets sentinel "" + a failure.
    assert out[2] == ""
    assert n_fail >= 1


def test_compute_scaffolds_returns_per_row() -> None:
    """``compute_scaffolds`` returns a per-row list of scaffolds."""
    mod = _import_script()
    pytest.importorskip("rdkit")
    canon, _ = mod.canonicalise_smiles(["c1ccccc1", "Cc1ccccc1"])
    scafs, n_fail = mod.compute_scaffolds(canon)
    # Both benzene-derivatives share the benzene Bemis-Murcko scaffold.
    assert scafs[0] is not None
    assert scafs[1] is not None
    assert scafs[0] == scafs[1]
    assert n_fail == 0


def test_random_split_indices_is_deterministic() -> None:
    """``random_split_indices`` with the same seed yields identical splits."""
    mod = _import_script()
    a = mod.random_split_indices(100, seed=7)
    b = mod.random_split_indices(100, seed=7)
    assert a.train_idx == b.train_idx
    assert a.val_idx == b.val_idx
    assert a.test_idx == b.test_idx


def test_scaffold_split_indices_is_strict() -> None:
    """``scaffold_split_indices`` never assigns the same scaffold twice."""
    mod = _import_script()
    pytest.importorskip("rdkit")
    canon, _ = mod.canonicalise_smiles(
        ["c1ccccc1", "Cc1ccccc1", "c1ccccc1F", "CCCCO", "CCCCCO"],
    )
    scafs, _ = mod.compute_scaffolds(canon)
    split = mod.scaffold_split_indices(scafs, seed=0)
    # Each scaffold group must land in exactly one split.
    assigned = {}
    for i in split.train_idx:
        assigned.setdefault(scafs[i], "train")
        assert assigned[scafs[i]] == "train"
    for i in split.val_idx:
        assigned.setdefault(scafs[i], "val")
        assert assigned[scafs[i]] == "val"
    for i in split.test_idx:
        assigned.setdefault(scafs[i], "test")
        assert assigned[scafs[i]] == "test"


def test_compute_overlap_zero_for_disjoint_indices() -> None:
    """``compute_overlap`` returns 0 overlap for index-disjoint splits."""
    mod = _import_script()
    canon = ["CCO", "c1ccccc1", "CCCO", "CCCCO"]
    scafs = ["scafA", "scafB", "scafC", "scafD"]
    split = mod.SplitStats(
        train_idx=[0, 1], val_idx=[], test_idx=[2, 3],
        n_train=2, n_val=0, n_test=2,
    )
    overlaps = mod.compute_overlap(canon, scafs, split)
    assert overlaps["train_vs_test"].n_intersect_smiles == 0
    assert overlaps["train_vs_test"].n_intersect_scaffolds == 0


# ---------------------------------------------------------------------------
# 4. End-to-end driver (platinai_5k)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not PLATINAI_CSV.exists(),
    reason="platinai_5k.csv not available",
)
def test_run_dataset_platinai_has_expected_shape() -> None:
    """``run_dataset`` on platinai_5k returns the Wf-44 contract shape."""
    mod = _import_script()
    rep = mod.run_dataset("platinai_5k_smoke", PLATINAI_CSV, seed=0)
    assert rep.dataset_name == "platinai_5k_smoke"
    assert rep.n_total == 5000
    assert rep.n_unique_smiles > 0
    assert rep.n_unique_scaffolds > 0
    # Both random and scaffold splits must be present.
    assert "random" in rep.splits
    assert "scaffold" in rep.splits
    # Scaffold split must be strict (zero scaffold overlap).
    assert rep.strict_pass["scaffold"] is True


@pytest.mark.skipif(
    not PLATINAI_CSV.exists(),
    reason="platinai_5k.csv not available",
)
def test_run_dataset_platinai_scaffold_split_zero_scaffold_overlap() -> None:
    """On the scaffold split, ``n_intersect_scaffolds`` between train and
    test is *exactly zero* — the whole point of the strict split."""
    mod = _import_script()
    rep = mod.run_dataset("platinai_5k_smoke", PLATINAI_CSV, seed=0)
    tvt = rep.overlaps["scaffold"]["train_vs_test"]
    assert tvt.n_intersect_scaffolds == 0
    assert tvt.frac_test_scaffolds_seen_in_train == 0.0


# ---------------------------------------------------------------------------
# 5. Honest-framing property — random split shows non-zero scaffold overlap
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not PLATINAI_CSV.exists(),
    reason="platinai_5k.csv not available",
)
def test_random_split_shows_some_scaffold_overlap_on_platinai() -> None:
    """On the platinai_5k dataset, the *random* split (default
    ``train_test_split``) shows non-zero scaffold overlap — the
    diagnosis that motivates the scaffold-split remediation.

    This is the **honest-framing** property: we expect the random
    split to fail the scaffold-axis strictness check, because random
    splitting does not honour the Bemis-Murcko axis.  The script
    must surface that finding rather than hide it.
    """
    mod = _import_script()
    rep = mod.run_dataset("platinai_5k_smoke", PLATINAI_CSV, seed=0)
    tvt = rep.overlaps["random"]["train_vs_test"]
    assert tvt.n_intersect_scaffolds >= 1, (
        "random split should leak scaffolds by construction — if this "
        "fails the synthetic dataset may not be exercising the "
        "diagnostic."
    )


@pytest.mark.skipif(
    not METALCYTOTOX_CSV.exists(),
    reason="metalcytotox_5metal_multitask_v4.csv not available",
)
def test_metalcytotox_scaffold_split_strict() -> None:
    """The scaffold split on the metalcytotox 5-metal dataset is
    strict at the scaffold axis (zero overlap)."""
    mod = _import_script()
    rep = mod.run_dataset("metalcytotox_5metal_smoke", METALCYTOTOX_CSV, seed=0)
    tvt = rep.overlaps["scaffold"]["train_vs_test"]
    assert tvt.n_intersect_scaffolds == 0


@pytest.mark.skipif(
    not METALCYTOTOX_CSV.exists(),
    reason="metalcytotox_5metal_multitask_v4.csv not available",
)
def test_metalcytotox_random_split_shows_heavy_overlap() -> None:
    """The metalcytotox 5-metal dataset has heavy SMILES duplication
    (cf. leakage_diagnosis.md) — a random split shows large SMILES
    overlap (>1000)."""
    mod = _import_script()
    rep = mod.run_dataset("metalcytotox_5metal_smoke", METALCYTOTOX_CSV, seed=0)
    tvt = rep.overlaps["random"]["train_vs_test"]
    assert tvt.n_intersect_smiles >= 1000, (
        f"expected heavy SMILES duplication on metalcytotox 5-metal; "
        f"got n_intersect_smiles={tvt.n_intersect_smiles}"
    )


# ---------------------------------------------------------------------------
# 6. CLI exit code
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not PLATINAI_CSV.exists() and not METALCYTOTOX_CSV.exists(),
    reason="No data CSV available",
)
def test_cli_exits_zero_with_real_data(tmp_path: Path) -> None:
    """CLI exits 0 when both default CSVs are available."""
    import subprocess
    script = _REPO_ROOT / "molmetal" / "scripts" / "check_data_leakage.py"
    out = subprocess.run(
        [sys.executable, str(script), "--seed", "1"],
        capture_output=True, text=True, timeout=180,
    )
    assert out.returncode == 0, (
        f"CLI exited with {out.returncode}: {out.stderr or out.stdout}"
    )
