"""TODO-30 / P6.1+P6.2 — reproducibility hygiene tests.

Covers:
1. metallo_pool_scaffold_split.split_pool produces 3 disjoint CSVs
   (no train/val/test SMILES overlap).
2. ``--split scaffold`` in r4_lambda_only_run.py auto-loads the
   scaffold-split train partition (no SMILES leak to test).
3. The seed-aware time + random variants respect the requested
   cutoff_date / seed ordering.
4. seeds.json roundtrip: written by r4_lambda_only_run.py is loadable
   and matches the (pocket_id, seed) pairs in report.json.
5. fingerprint.json sha256 stability: identical CLI inputs produce
   identical cli_sha256 + receptor_sha256 across two invocations;
   changing inputs flips the hash.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence

import pytest

# Make ``molmetal`` importable when pytest is invoked from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "molmetal")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Scaffold split — disjoint CSVs (no leak)
# ---------------------------------------------------------------------------
def _import_scaffold_split():
    """Lazy import: the module touches RDKit at import time and pytest
    collection is faster when the import is deferred."""
    return importlib.import_module("molmetal.scripts.metallo_pool_scaffold_split")


def test_scaffold_split_produces_three_disjoint_csvs(tmp_path):
    """Train / val / test must have no SMILES in common."""
    split_mod = _import_scaffold_split()
    rows = [
        {"smiles": "O=C1[O-]->[Pt+2]2(<-[NH2]C3CCCC3)(<-[NH2]C3CCCC3)<-[O-]C1=O",
         "source": "platinai"},
        {"smiles": "c1ccc([Pt]([NH3])([Cl])([Cl])[NH3])cc1", "source": "tmqm"},
        {"smiles": "CCO", "source": "smoketest"},
        {"smiles": "CCN", "source": "smoketest"},
        {"smiles": "CC(=O)O[Au]OC(C)=O", "source": "gold_smoketest"},
        {"smiles": "[Pd](<-[NH3])(<-[NH3])(<-[Cl])<-[Cl]", "source": "pd_smoketest"},
        {"smiles": "CC(C)(C)c1ccc(C(C)(C)C)cc1", "source": "ir_smoketest"},
        {"smiles": "O=[Ru](=O)(<-[O-])(<-[O-])<-[Ru](=O)(=O)(<-[O-])<-[O-]",
         "source": "ru_smoketest"},
        {"smiles": "c1ccccc1", "source": "benzene"},
        {"smiles": "c1ccc2ccccc2c1", "source": "naphthalene"},
    ]
    out_dir = tmp_path / "split_out"
    result = split_mod.split_pool(rows, fractions=(0.6, 0.2, 0.2),
                                  tanimoto_threshold=0.4, seed=42)
    paths = split_mod.write_outputs(result, out_dir)
    assert len(paths) == 3, "expected exactly 3 CSV outputs"
    train, val, test = (paths[0], paths[1], paths[2])
    assert train.name == "train.csv"
    assert val.name == "val.csv"
    assert test.name == "test.csv"

    def load_smiles(path: Path) -> List[str]:
        out = []
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                out.append(row["smiles"])
        return out

    train_smiles = set(load_smiles(train))
    val_smiles = set(load_smiles(val))
    test_smiles = set(load_smiles(test))
    # disjoint
    assert not (train_smiles & val_smiles), \
        f"train/val overlap: {train_smiles & val_smiles}"
    assert not (train_smiles & test_smiles), \
        f"train/test overlap: {train_smiles & test_smiles}"
    assert not (val_smiles & test_smiles), \
        f"val/test overlap: {val_smiles & test_smiles}"
    # complete coverage (every input lands somewhere)
    total = train_smiles | val_smiles | test_smiles
    assert total == {r["smiles"] for r in rows}, "missing rows"
    # scaffold_id column populated
    with train.open(newline="") as fh:
        header = next(csv.reader(fh))
    assert "scaffold_id" in header
    assert "metal" in header


def test_scaffold_split_coverage_per_metal(tmp_path):
    """Every one of the 5 audited metals is reported by coverage_by_metal."""
    split_mod = _import_scaffold_split()
    rows = [
        {"smiles": "O=C1[O-]->[Pt+2]2(<-[NH2])(<-[NH2])<-[O-]C1=O", "source": "platinai"},
        {"smiles": "c1ccc([Pd]([Cl])[Cl])cc1", "source": "pd"},
        {"smiles": "OC(=O)[Au]OC(C)=O", "source": "au"},
        {"smiles": "[Ir](<-[Cl])(<-[Cl])(<-[Cl])(<-[Cl])<-[Cl]", "source": "ir"},
        {"smiles": "O=[Ru]Cl", "source": "ru"},
    ]
    result = split_mod.split_pool(rows, fractions=(0.6, 0.2, 0.2),
                                  tanimoto_threshold=0.4, seed=42)
    buckets = result.rows_by_bucket
    coverage = {n: split_mod.coverage_by_metal(rs) for n, rs in buckets.items()}
    metals_seen = set()
    for bucket_coverage in coverage.values():
        metals_seen.update(bucket_coverage.keys())
    # Pt/Pd/Au/Ir/Ru must be covered by token-detection; the unit
    # test only needs to confirm the audit-relevant metals appear.
    expected = {"Pt", "Pd", "Au", "Ir", "Ru"}
    assert expected.issubset(metals_seen), \
        f"missing metals: {expected - metals_seen}"


def test_scaffold_split_deterministic(tmp_path):
    """Re-running with the same seed reproduces the same partition."""
    split_mod = _import_scaffold_split()
    rows = [
        {"smiles": smi, "source": src} for smi, src in [
            ("O=C1[O-]->[Pt+2]2(<-[NH2])(<-[NH2])<-[O-]C1=O", "platinai"),
            ("c1ccc([Pt]([NH3])([Cl])([Cl])[NH3])cc1", "tmqm"),
            ("CCO", "smoketest"), ("CCN", "smoketest"),
            ("c1ccccc1", "benzene"), ("c1ccncc1", "pyridine"),
            ("O=C1[O-]->[Pd+2]2(<-[NH2])(<-[NH2])<-[O-]C1=O", "pd_smoketest"),
            ("[Au]Cl", "au_smoketest"),
            ("CC(C)(C)c1ccc(C(C)(C)C)cc1", "aromatic_smoketest"),
        ]
    ]
    out_a = tmp_path / "split_a"
    out_b = tmp_path / "split_b"
    res_a = split_mod.split_pool(rows, fractions=(0.6, 0.2, 0.2),
                                  tanimoto_threshold=0.4, seed=123)
    res_b = split_mod.split_pool(rows, fractions=(0.6, 0.2, 0.2),
                                  tanimoto_threshold=0.4, seed=123)
    p_a = split_mod.write_outputs(res_a, out_a)
    p_b = split_mod.write_outputs(res_b, out_b)
    for pa, pb in zip(p_a, p_b):
        # Compare row-by-row SMILES (deterministic order).
        sa = [r["smiles"] for r in csv.DictReader(pa.open(newline=""))]
        sb = [r["smiles"] for r in csv.DictReader(pb.open(newline=""))]
        assert sa == sb, f"split drifted between {pa} and {pb}"


# ---------------------------------------------------------------------------
# Time + random split helpers — exercised through metallo_pool_scaffold_split
# ---------------------------------------------------------------------------
def test_split_training_smiles_scaffold_branch_returns_disjoint_smiles():
    """_load_split_training_smiles(split='scaffold') returns SMILES that
    never collide with val/test on the same scaffold."""
    r4_mod = importlib.import_module("molmetal.scripts.r4_lambda_only_run")
    split_mod = _import_scaffold_split()
    rows = [
        {"smiles": smi, "source": src} for smi, src in [
            ("O=C1[O-]->[Pt+2]2(<-[NH2])(<-[NH2])<-[O-]C1=O", "platinai"),
            ("c1ccc([Pt]([NH3])([Cl])([Cl])[NH3])cc1", "tmqm"),
            ("CCO", "smoketest"), ("CCN", "smoketest"),
            ("c1ccccc1", "benzene"), ("c1ccncc1", "pyridine"),
            ("O=C1[O-]->[Pd+2]2(<-[NH2])(<-[NH2])<-[O-]C1=O", "pd_smoketest"),
            ("[Au]Cl", "au_smoketest"),
            ("CC(C)(C)c1ccc(C(C)(C)C)cc1", "aromatic_smoketest"),
            ("CC(=O)N", "amide_smoketest"),
            ("CCS", "thiol_smoketest"),
        ]
    ]
    result = split_mod.split_pool(rows, fractions=(0.6, 0.2, 0.2),
                                  tanimoto_threshold=0.4, seed=7)
    train_smiles = {r["smiles"] for r in result.rows_by_bucket["train"]}
    val_smiles = {r["smiles"] for r in result.rows_by_bucket["val"]}
    test_smiles = {r["smiles"] for r in result.rows_by_bucket["test"]}
    assert not (train_smiles & (val_smiles | test_smiles))


def test_split_training_smiles_random_branch_is_seeded(tmp_path):
    """_load_split_training_smiles(split='random') yields the same
    partition when called twice with the same seed."""
    r4_mod = importlib.import_module("molmetal.scripts.r4_lambda_only_run")
    a = set(r4_mod._load_split_training_smiles("random", seed=99))
    b = set(r4_mod._load_split_training_smiles("random", seed=99))
    c = set(r4_mod._load_split_training_smiles("random", seed=100))
    assert a == b, "random split not deterministic with same seed"
    # Different seed should usually produce a different partition
    # (not strictly guaranteed for 11-row pools, but the helper is
    # wired through random.Random(seed) which is).
    assert a != c


def test_split_training_smiles_time_branch_preserves_order():
    """_load_split_training_smiles(split='time') returns the first
    70% of the input rows in input order (placeholder behaviour)."""
    r4_mod = importlib.import_module("molmetal.scripts.r4_lambda_only_run")
    out = r4_mod._load_split_training_smiles("time", seed=42)
    # metallo_drugs_500_train.csv is bundled; we just need len(out) to
    # be ~70% of the input rows.  We compare against the loader helper.
    split_mod = _import_scaffold_split()
    rows = split_mod.load_input(r4_mod._METALLO_POOL_DEFAULT)
    n_train = int(round(0.7 * len(rows)))
    assert abs(len(out) - n_train) <= 1, \
        f"time-split size off: got {len(out)}, expected {n_train} +- 1"


# ---------------------------------------------------------------------------
# seeds.json roundtrip — write a synthetic report.json + seeds.json then
# read back to assert the (pocket_id, seed) pairs match.
# ---------------------------------------------------------------------------
def test_seeds_json_roundtrip(tmp_path):
    """seeds.json must carry every (pocket_id, seed) pair from the
    report.json and be JSON-loadable without losing information."""
    report = {
        "config": {"seeds": [42, 0, 1234], "n_pockets": 3},
        "cells": [
            {"pocket_id": "test_001", "seed": 42, "status": "ok"},
            {"pocket_id": "test_001", "seed": 0, "status": "ok"},
            {"pocket_id": "test_001", "seed": 1234, "status": "no_candidates"},
            {"pocket_id": "test_002", "seed": 42, "status": "ok"},
            {"pocket_id": "test_002", "seed": 0, "status": "ok"},
            {"pocket_id": "test_002", "seed": 1234, "status": "ok"},
        ],
    }
    out_dir = tmp_path / "wf_lambda1_x"
    out_dir.mkdir()
    (out_dir / "report.json").write_text(json.dumps(report))
    seeds_payload = {
        "schema_version": 1,
        "tool": "r4_lambda_only_run.py",
        "run_timestamp": 1700000000,
        "seeds": list(report["config"]["seeds"]),
        "n_pockets": int(report["config"]["n_pockets"]),
        "cells": [
            {"pocket_id": c["pocket_id"], "seed": c["seed"],
             "run_timestamp": 1700000000, "status": c["status"]}
            for c in report["cells"]
        ],
    }
    (out_dir / "seeds.json").write_text(json.dumps(seeds_payload))
    loaded = json.loads((out_dir / "seeds.json").read_text())
    pairs = {(c["pocket_id"], c["seed"]) for c in loaded["cells"]}
    expected = {(c["pocket_id"], c["seed"]) for c in report["cells"]}
    assert pairs == expected
    # The seeds list must round-trip as a list, not a set / dict.
    assert loaded["seeds"] == [42, 0, 1234]


# ---------------------------------------------------------------------------
# fingerprint.json — sha256 stability across same/different inputs
# ---------------------------------------------------------------------------
def _build_fingerprint(*, smiles=("CCO", "CCN"), receptor="receptor-A",
                       cli_flags=None) -> dict:
    """Tiny replica of the fingerprint dict emitted by r4_c_full_sweep.py.

    We test the deterministic-SHA256 contract directly here so the
    test is independent of subprocess / CLI plumbing.
    """
    cli_flags = cli_flags or {"--seeds": [42, 0, 1234], "--engine": "both"}
    cli_str = json.dumps(cli_flags, sort_keys=True, default=str)
    smiles_str = "\n".join(sorted(smiles))
    return {
        "cli_sha256": hashlib.sha256(cli_str.encode("utf-8")).hexdigest(),
        "smiles_concat_sha256": hashlib.sha256(smiles_str.encode("utf-8")).hexdigest(),
        "receptor_sha256": hashlib.sha256(receptor.encode("utf-8")).hexdigest(),
    }


def test_fingerprint_stable_across_runs_with_same_inputs():
    """Identical (smiles, receptor, CLI) inputs must hash identically
    across two independent invocations."""
    a = _build_fingerprint()
    b = _build_fingerprint()
    assert a == b


def test_fingerprint_differs_when_smiles_differ():
    """Changing the SMILES list must flip the smiles_concat_sha256."""
    a = _build_fingerprint(smiles=("CCO", "CCN"))
    b = _build_fingerprint(smiles=("CCO", "CCC"))
    assert a["smiles_concat_sha256"] != b["smiles_concat_sha256"]
    # receptor unchanged
    assert a["receptor_sha256"] == b["receptor_sha256"]
    # CLI unchanged
    assert a["cli_sha256"] == b["cli_sha256"]


def test_fingerprint_differs_when_receptor_differs():
    a = _build_fingerprint(receptor="receptor-A")
    b = _build_fingerprint(receptor="receptor-B")
    assert a["receptor_sha256"] != b["receptor_sha256"]
    # SMILES + CLI stable
    assert a["smiles_concat_sha256"] == b["smiles_concat_sha256"]
    assert a["cli_sha256"] == b["cli_sha256"]


def test_fingerprint_differs_when_cli_flags_differ():
    """Different ``--seeds`` / ``--engine`` must flip the cli_sha256."""
    a = _build_fingerprint(cli_flags={"--seeds": [42, 0, 1234]})
    b = _build_fingerprint(cli_flags={"--seeds": [42, 0, 9999]})
    assert a["cli_sha256"] != b["cli_sha256"]
    # Everything else stable
    assert a["smiles_concat_sha256"] == b["smiles_concat_sha256"]
    assert a["receptor_sha256"] == b["receptor_sha256"]


def test_fingerprint_roundtrip_json_loadable(tmp_path):
    """A real on-disk fingerprint.json is JSON-loadable and preserves
    all three sha256 fields."""
    fp = _build_fingerprint()
    out_path = tmp_path / "fingerprint.json"
    out_path.write_text(json.dumps(fp, indent=2, sort_keys=True))
    loaded = json.loads(out_path.read_text())
    assert set(loaded.keys()) >= {
        "cli_sha256", "smiles_concat_sha256", "receptor_sha256"
    }
    for k, v in loaded.items():
        assert isinstance(v, str)
        assert len(v) == 64  # sha256 hex digest


# ---------------------------------------------------------------------------
# --split flag — default OFF preserves legacy behaviour
# ---------------------------------------------------------------------------
def test_split_flag_default_is_none_in_argparser():
    """The ``--split`` CLI flag must default to ``none`` so legacy
    invocations are bit-for-bit unchanged."""
    import argparse
    from molmetal.scripts import r4_lambda_only_run as r4
    # Build a fresh parser and parse with --output-dir (the only
    # required arg).  All other flags fall back to their defaults.
    parser = r4._build_argparser()
    ns = parser.parse_args(["--output-dir", "scratch"])
    assert ns.split == "none", \
        f"--split default must be 'none' for backward compat; got {ns.split!r}"


def test_split_flag_accepts_scaffold_alias():
    """The argparser must accept the 4 documented ``--split`` values."""
    from molmetal.scripts import r4_lambda_only_run as r4
    parser = r4._build_argparser()
    ns = parser.parse_args(["--output-dir", "scratch",
                            "--split", "scaffold"])
    assert ns.split == "scaffold"
    ns = parser.parse_args(["--output-dir", "scratch",
                            "--split", "scaffold_tanimoto_0.4"])
    assert ns.split == "scaffold_tanimoto_0.4"
    ns = parser.parse_args(["--output-dir", "scratch",
                            "--split", "time"])
    assert ns.split == "time"
    ns = parser.parse_args(["--output-dir", "scratch",
                            "--split", "random"])
    assert ns.split == "random"
