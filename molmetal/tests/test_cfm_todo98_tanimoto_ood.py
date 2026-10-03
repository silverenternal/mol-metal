"""Tests for the Wf-257-Apply TODO-98 CFM Tanimoto<0.4 OOD filter.

Covers ``molmetal.molmetal_lam.lam_chem.cfm_tanimoto_ood_filter``:

  1. :func:`apply_tanimoto_ood_filter` — boolean mask per query.
  2. :func:`cfm_ood_filter_smiles_list` — partition payloads into
     (in_domain, ood) buckets.
  3. :func:`summarise_tanimoto_ood` — full DIAGNOSTIC summary with
     per-query max-train-Tanimoto.

The filter is **CPU-only** (pure RDKit Morgan-ECFP4) and **DIAGNOSTIC**
(does not alter the CFM loss surface — per pre-retrain checklist
§1.2.5).  The CLI integration with ``train_cfm_yuelbond_wasserstein.py``
is exercised by ``test_cfm_yuelbond_wasserstein.py::test_target_decode_ratio_set_via_args``
(``--use-tanimoto-ood-filter`` arg present and round-trips).

Honest framing (party-twenty-spirit §3.5):
* The threshold default (0.4) is the Pocket2Mol / TargetDiff convention,
  not a Mol-Metal invention.
* Synthetic Pt-click clouds have no SMILES fingerprint, so the
  filter on a purely-synthetic cohort returns the train pool
  itself (backward-compat).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


@pytest.fixture(scope="module")
def ood_module():
    from molmetal.molmetal_lam.lam_chem.cfm_tanimoto_ood_filter import (
        TanimotoOODSummary,
        apply_tanimoto_ood_filter,
        cfm_ood_filter_smiles_list,
        summarise_tanimoto_ood,
        write_tanimoto_ood_summary_json,
    )

    return {
        "TanimotoOODSummary": TanimotoOODSummary,
        "apply_tanimoto_ood_filter": apply_tanimoto_ood_filter,
        "cfm_ood_filter_smiles_list": cfm_ood_filter_smiles_list,
        "summarise_tanimoto_ood": summarise_tanimoto_ood,
        "write_tanimoto_ood_summary_json": write_tanimoto_ood_summary_json,
    }


# ---------------------------------------------------------------------------
# 1. apply_tanimoto_ood_filter
# ---------------------------------------------------------------------------
class TestApplyTanimotoOODFilter:
    def test_returns_bool_per_query(self, ood_module):
        apply = ood_module["apply_tanimoto_ood_filter"]
        flags = apply(
            ["N.N.[Cl][Pt][Cl]", "c1ccccc1O"],
            ["N.N.[Cl][Pt][Cl]"],
        )
        assert len(flags) == 2
        assert all(isinstance(f, bool) for f in flags)

    def test_self_match_is_in_domain(self, ood_module):
        """A query that matches a train entry should NOT be OOD."""
        apply = ood_module["apply_tanimoto_ood_filter"]
        flags = apply(["c1ccccc1O"], ["c1ccccc1O"])
        assert flags == [False]

    def test_dissimilar_query_is_ood(self, ood_module):
        """Phenol vs cisplatin-Pt complex is OOD (< 0.4 Tanimoto)."""
        apply = ood_module["apply_tanimoto_ood_filter"]
        flags = apply(["c1ccccc1O"], ["N.N.[Cl][Pt][Cl]"])
        assert flags == [True]

    def test_empty_train_pool_marks_all_ood(self, ood_module):
        """Mirrors TanimotoOODProtocol.is_ood empty-train semantics."""
        apply = ood_module["apply_tanimoto_ood_filter"]
        flags = apply(["c1ccccc1O", "CCN"], [])
        assert flags == [True, True]

    def test_invalid_smiles_conservatively_ood(self, ood_module):
        """A SMILES that fails RDKit parse is flagged OOD (fail-closed)."""
        apply = ood_module["apply_tanimoto_ood_filter"]
        flags = apply(["not_a_real_smiles"], ["c1ccccc1O"])
        assert flags == [True]

    def test_threshold_boundary(self, ood_module):
        """OOD iff max-train-Tanimoto < threshold (strict less-than)."""
        apply = ood_module["apply_tanimoto_ood_filter"]
        # Same input → Tanimoto = 1.0, NOT OOD at any threshold.
        flags = apply(["c1ccccc1O"], ["c1ccccc1O"], threshold=0.99)
        assert flags == [False]
        # Same input → Tanimoto = 1.0 > 0.4 → NOT OOD.
        flags_low = apply(["c1ccccc1O"], ["c1ccccc1O"], threshold=0.4)
        assert flags_low == [False]

    def test_default_threshold_is_0_4(self, ood_module):
        """Default threshold must be 0.4 (Pocket2Mol / TargetDiff)."""
        from molmetal.molmetal_lam.lam_chem.ood_protocol import (
            DEFAULT_TANIMOTO_THRESHOLD,
        )

        assert DEFAULT_TANIMOTO_THRESHOLD == 0.4


# ---------------------------------------------------------------------------
# 2. cfm_ood_filter_smiles_list
# ---------------------------------------------------------------------------
class TestCFMOODFilterSmilesList:
    def test_partition_lengths(self, ood_module):
        """Sum of (in_domain, ood) must equal len(query_smiles)."""
        fn = ood_module["cfm_ood_filter_smiles_list"]
        in_d, ood = fn(
            ["c1ccccc1O", "N.N.[Cl][Pt][Cl]", "c1ccc2ccccc2c1"],
            ["phenol", "cisplatin", "naphthalene"],
            train_smiles=["N.N.[Cl][Pt][Cl]"],
        )
        assert len(in_d) + len(ood) == 3

    def test_payloads_round_trip(self, ood_module):
        """Per-query payload is kept or dropped together with its SMILES."""
        fn = ood_module["cfm_ood_filter_smiles_list"]
        payloads = [
            {"smi": s, "idx": i}
            for i, s in enumerate(["c1ccccc1O", "N.N.[Cl][Pt][Cl]"])
        ]
        in_d, ood = fn(
            ["c1ccccc1O", "N.N.[Cl][Pt][Cl]"],
            payloads,
            train_smiles=["N.N.[Cl][Pt][Cl]"],
        )
        # cisplatin payload is in-domain; phenol payload is OOD.
        assert len(in_d) == 1
        assert len(ood) == 1
        assert in_d[0]["idx"] == 1
        assert ood[0]["idx"] == 0

    def test_length_mismatch_raises(self, ood_module):
        fn = ood_module["cfm_ood_filter_smiles_list"]
        with pytest.raises(ValueError):
            fn(["c1ccccc1O"], ["a", "b"], train_smiles=[])

    def test_handles_atom_cloud_payloads(self, ood_module):
        """Payloads can be AtomCloud objects (no type coupling)."""
        fn = ood_module["cfm_ood_filter_smiles_list"]
        import torch
        from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
            AtomCloud,
        )

        cloud1 = AtomCloud(
            positions=torch.zeros(5, 3),
            atomic_numbers=torch.tensor([78, 17, 17, 7, 7]),
        )
        cloud2 = AtomCloud(
            positions=torch.zeros(5, 3),
            atomic_numbers=torch.tensor([78, 17, 17, 7, 7]),
        )
        in_d, ood = fn(
            ["N.N.[Cl][Pt][Cl]", "c1ccccc1O"],
            [cloud1, cloud2],
            train_smiles=["N.N.[Cl][Pt][Cl]"],
        )
        assert len(in_d) + len(ood) == 2
        # First payload is in-domain (cisplatin-like), second is OOD.
        assert in_d[0] is cloud1
        assert ood[0] is cloud2


# ---------------------------------------------------------------------------
# 3. summarise_tanimoto_ood
# ---------------------------------------------------------------------------
class TestSummariseTanimotoOOD:
    def test_summary_counts(self, ood_module):
        fn = ood_module["summarise_tanimoto_ood"]
        summary = fn(
            ["c1ccccc1O", "N.N.[Cl][Pt][Cl]"],
            ["N.N.[Cl][Pt][Cl]"],
        )
        assert summary.n_total == 2
        assert summary.n_in_domain == 1
        assert summary.n_ood == 1
        assert 0.0 < summary.ood_fraction < 1.0

    def test_max_train_tanimoto_per_query(self, ood_module):
        fn = ood_module["summarise_tanimoto_ood"]
        summary = fn(
            ["c1ccccc1O", "N.N.[Cl][Pt][Cl]"],
            ["N.N.[Cl][Pt][Cl]"],
        )
        assert len(summary.max_train_tanimoto) == 2
        # Phenol should have low Tanimoto to cisplatin; cisplatin
        # itself should have Tanimoto ≈ 1.0 (self-match).
        assert summary.max_train_tanimoto[1] > summary.max_train_tanimoto[0]

    def test_empty_query_zero_fraction(self, ood_module):
        fn = ood_module["summarise_tanimoto_ood"]
        summary = fn([], ["c1ccccc1O"])
        assert summary.n_total == 0
        assert summary.ood_fraction == 0.0

    def test_threshold_round_trips(self, ood_module):
        fn = ood_module["summarise_tanimoto_ood"]
        summary = fn(["c1ccccc1O"], ["N.N.[Cl][Pt][Cl]"], threshold=0.99)
        assert summary.threshold == 0.99

    def test_to_dict_is_json_serializable(self, ood_module):
        fn = ood_module["summarise_tanimoto_ood"]
        summary = fn(
            ["c1ccccc1O", "N.N.[Cl][Pt][Cl]"],
            ["N.N.[Cl][Pt][Cl]"],
        )
        # Must be JSON-serializable for the
        # ``metrics/by_metric/tanimoto_ood.json`` reporting surface.
        json.dumps(summary.to_dict())


# ---------------------------------------------------------------------------
# 4. JSON IO helper
# ---------------------------------------------------------------------------
class TestWriteTanimotoOODSummaryJSON:
    def test_writes_to_disk(self, ood_module, tmp_path: Path):
        write_fn = ood_module["write_tanimoto_ood_summary_json"]
        sum_fn = ood_module["summarise_tanimoto_ood"]
        summary = sum_fn(["c1ccccc1O"], ["N.N.[Cl][Pt][Cl]"])
        path = tmp_path / "ood_summary.json"
        write_fn(summary, str(path))
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["n_total"] == 1
        assert "max_train_tanimoto" in loaded


# ---------------------------------------------------------------------------
# 5. Backward-compat: existing CFM tests must still pass
# ---------------------------------------------------------------------------
def test_existing_cfm_imports_unaffected(ood_module):
    """Adding the OOD filter module must not break CFM imports."""
    from molmetal.scripts import train_cfm_yuelbond_wasserstein as tcfm

    assert hasattr(tcfm, "evaluate_decode_ratio")
    assert hasattr(tcfm, "build_training_cohort")


def test_ood_filter_cli_flag_exists_in_train_script(ood_module):
    """``--use-tanimoto-ood-filter`` must appear in ``--help`` output."""
    import subprocess

    proc = subprocess.run(
        [
            "/home/hugo/codes/mol-metal/.venv/bin/python",
            str(
                Path(_REPO)
                / "molmetal"
                / "scripts"
                / "train_cfm_yuelbond_wasserstein.py"
            ),
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "--use-tanimoto-ood-filter" in proc.stdout
    assert "--ood-filter-threshold" in proc.stdout
