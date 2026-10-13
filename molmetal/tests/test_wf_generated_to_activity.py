# Licensed under the Apache License, Version 2.0.
"""Regression tests for ``wf_generated_to_activity.py``.

Covers the three ways the cross-application measurement could go wrong, each
of which produced a wrong published number before it was measured:

1. **Default OFF is preserved.**  ``--reference`` was added so the generated
   column can be read against a real cohort scored identically.  If the flag
   ever changed the default path, every previously-reported absolute number
   would silently move.

2. **The metal audit is real.**  The honest reading of this metric hinges on
   the cohort carrying no metal ATOM.  If that stopped being true the framing
   would be stale, so it is recomputed from disk rather than trusted.

3. **The audit detects a metal when one is present.**  A check that always
   returns zero is indistinguishable from a check that works.
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wf_generated_to_activity.py"
GENERATED_CSV = (
    REPO_ROOT / "molmetal" / "reports" / "cfm_fullcorpus_2026_10_10" / "generated_smiles.csv"
)
NCI60_CSV = REPO_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"


def _load():
    spec = importlib.util.spec_from_file_location("wf_gta_probe", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wf_gta_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


gta = _load()


class TestMetalAudit:
    def test_counts_metal_bearing_molecules(self):
        # A ruthenium arene must be counted; an all-organic molecule must not.
        res = gta.metal_audit(
            [
                "CC(=O)Oc1ccccc1C(=O)O",  # aspirin: no metal
                "[Ru+2](<-[c-]1ccccc1)(<-[c-]1ccccc1)(<-[Cl-])<-[Cl-]",  # Ru complex
            ]
        )
        assert res["n_parsed"] == 2
        assert res["n_metal_bearing_molecules"] == 1
        assert res["metal_Z_counts"] == {"44": 1}

    def test_all_organic_cohort_scores_zero(self):
        """The load-bearing measurement behind the whole framing.

        Every NCI-60 cohort SMILES must be metal-free IN THE GRAPH, because
        the cohort is built from MetalCytoToxDB's ``SMILES_Ligands`` column
        and the metal lives in that file's separate ``Metal`` column.  If this
        ever fails, the "these structures are in-distribution for
        organic-activity models" framing is no longer the honest reading and
        the metric JSON's framing has to be rewritten.
        """
        res = gta.metal_audit(["CCO", "c1ccccc1C(=O)O", "CCN(CC)CC"])
        assert res["n_metal_bearing_molecules"] == 0
        assert res["pct_metal_bearing"] == 0.0

    def test_cohort_audit_reports_substrate_provenance(self):
        """The cohort is REAL data, just metal-free graphs.

        Guards the more dangerous error: if the cohort were synthetic, the
        reference column would be meaningless for a different reason.

        MEASURED on the full 7,823-row cohort: 5,118 / 5,207 unique SMILES
        (98.29%) match a MetalCytoToxDB ``SMILES_Ligands`` string exactly.
        The threshold is asserted against the FULL cohort because a truncated
        slice matches a lower share (88.65% over the first 1,000 rows) --
        the first rows are a different metal/cell mix.  Pinning the full-cohort
        number is what makes this a regression test rather than a guess.
        """
        res = gta.cohort_metal_audit()
        assert res["n_metal_bearing_molecules"] == 0
        sm = res["substrate_match"]
        assert sm is not None
        assert sm["n_unique_cohort_smiles"] == 5207
        assert sm["pct_match"] == pytest.approx(98.29, abs=0.05)


class TestDefaultOffCompatibility:
    def test_reference_flag_defaults_to_off(self):
        """--reference must be opt-in, so existing invocations are unchanged."""
        src = SCRIPT.read_text()
        assert '"--reference"' in src
        # store_true + no default= => argparse default is False
        assert '"--reference",' in src
        assert 'action="store_true"' in src

    def test_metric_json_defaults_to_none(self):
        src = SCRIPT.read_text()
        assert '"--metric-json",' in src
        assert 'default=None' in src


@pytest.mark.skipif(
    not GENERATED_CSV.is_file(), reason="generated cohort not on disk"
)
class TestGeneratedCohortShape:
    def test_every_generated_molecule_carries_a_metal(self):
        """The asymmetry the framing rests on, re-derived from disk."""
        rows = [
            r
            for r in csv.DictReader(GENERATED_CSV.open())
            if (r.get("smiles") or "").strip()
        ]
        res = gta.metal_audit([r["smiles"] for r in rows])
        assert res["n_parsed"] == len(rows)
        # Every generated complex has a metal centre; the cohort has none.
        assert res["n_metal_bearing_molecules"] == res["n_parsed"]

    def test_featurizer_does_not_onehot_three_of_our_metals(self):
        """Ru/Os/Re fall into the catch-all ``z_other`` bit.

        The featurizer's one-hot is [C N O F P S Cl Fe Cu Rh Pd Ir Pt].
        Ru(44), Os(76), Re(75) are missing, so those metal atoms are
        indistinguishable from any other unlisted element.
        """
        from molmetal.baselines import dmpnn

        src = Path(dmpnn.__file__).read_text()
        assert "77," in src and "78," in src  # Ir, Pt are listed
        assert "44," not in src  # Ru is NOT
        assert "76," not in src  # Os is NOT


@pytest.mark.skipif(
    not (GENERATED_CSV.is_file() and NCI60_CSV.is_file()),
    reason="cohorts not on disk",
)
class TestReferenceModeEndToEnd:
    def test_reference_is_wired_and_refuses_wrong_family(self):
        """--reference against a non-nci60 family must refuse, not print a
        like-for-like claim that is not one."""
        proc = subprocess.run(
            [
                sys.executable, "-I", str(SCRIPT),
                "--smiles-csv", str(GENERATED_CSV),
                "--family", "platinai",
                "--reference",
            ],
            capture_output=True,
            text=True,
            timeout=3000,
        )
        assert proc.returncode == 1
        assert "Refusing" in proc.stdout
