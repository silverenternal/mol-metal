# Licensed under the Apache License, Version 2.0.
"""Wf-A114 — the honest-fragmentation gate and the census verdict.

``gen_validity`` cannot tell a molecule from a pile of loose atoms: it counts
a sample as valid when RDKit can sanitise the string, and ``C.C.C.C``
sanitises.  MEASURED on ``molmetal/reports/cfm_c8_fullgen_2026_10_24``:
``gen_validity`` read **1.000** at all three epochs it completed while
``gen_single_fragment_ratio`` read **0.203 / 0.200 / 0.348**.

These tests pin the three things that could silently regress:

1. the gate's fragment definition IS the trainer's — it imports
   ``decode_chemistry`` from the trainer rather than re-deriving it, so the
   metric the gate reports cannot drift from the metric the run records;
2. the verdict and exit-code convention (PASS/FAIL/SKIP -> 0/1/2) still hold,
   including the blind-validity signature that names the failure mode;
3. the SSOT (``cfm_run_census.json``) still carries the fragmentation columns
   and still renders a verdict per run.

CPU-only, RDKit-only, no GPU.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.cfm_fragmentation_gate import (  # noqa: E402
    DEFAULT_MAX_FRAGMENTED_RATIO,
    DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
    VALIDITY_SATURATION,
    evaluate,
    main,
    read_generated_smiles,
    score_run_dir,
    score_smiles_pool,
)
from molmetal.scripts import train_cfm_yuelbond_wasserstein as _trainer_mod  # noqa: E402

#: One connected molecule, two fragmented, one metal complex.
POOL = ["CCO", "C.C.C", "[Pt](Cl)(Cl)(N)N", "c1ccccc1C(=O)[O-].[Na+]", "CC(C)(C)"]


class TestDefinitionCannotDrift:
    def test_gate_uses_the_trainers_own_decode_chemistry(self):
        """The gate must score with the TRAINER's function, not a copy.

        The lookup is against the LIVE module attribute, not a name bound at
        import time: ``molmetal/tests/test_wf350c2_detect_anomaly.py`` calls
        ``importlib.reload`` on the trainer, which replaces the module object
        and therefore replaces ``decode_chemistry`` with a NEW function
        object.  A test that compared against the pre-reload binding failed
        depending on the order pytest collected the two files -- the check
        was about "the gate does not define its own copy", and that holds
        either way.
        """
        from molmetal.scripts.cfm_fragmentation_gate import _trainer_decode_chemistry

        assert _trainer_decode_chemistry() is _trainer_mod.decode_chemistry

    def test_a_connected_pool_scores_one_fragment(self):
        pool = score_smiles_pool(["CCO", "CC(C)(C)"])
        assert pool["n_scored"] == 2
        assert pool["single_fragment_ratio"] == 1.0
        assert pool["fragmented_ratio"] == 0.0
        assert pool["largest_fragment_frac_mean"] == 1.0

    def test_loose_atoms_are_three_fragments(self):
        pool = score_smiles_pool(["C.C.C"])
        assert pool["n_scored"] == 1
        assert pool["single_fragment_ratio"] == 0.0
        assert pool["n_fragments_mean"] == 3.0
        assert pool["largest_fragment_frac_mean"] == pytest.approx(1.0 / 3.0)

    def test_metal_complex_is_one_fragment_with_a_metal(self):
        pool = score_smiles_pool(["[Pt](Cl)(Cl)(N)N"])
        assert pool["single_fragment_ratio"] == 1.0
        assert pool["metal_ratio"] == 1.0


class TestPoolScoring:
    def test_mixed_pool_ratio(self):
        pool = score_smiles_pool(POOL)
        # CCO, [Pt](Cl)(Cl)(N)N, CC(C)(C) are single fragments -> 3 of 5
        assert pool["n_scored"] == 5
        assert pool["single_fragment_ratio"] == pytest.approx(3.0 / 5.0)
        assert pool["fragmented_ratio"] == pytest.approx(2.0 / 5.0)

    def test_unparseable_strings_are_counted_not_dropped(self):
        pool = score_smiles_pool(["CCO", "not_a_smiles", ""])
        assert pool["n_scored"] == 1
        assert pool["n_unparseable"] == 1
        assert pool["n_empty"] == 1
        assert pool["n_input"] == 3

    def test_the_end_to_end_rate_counts_unparseable_against_the_pool(self):
        # A string RDKit refuses is not a molecule, so it must not leave the
        # denominator just because it failed to parse.  The CONDITIONAL rate
        # (over the parsed rows) stays 1.0; the end-to-end rate is 1/3.
        pool = score_smiles_pool(["CCO", "not_a_smiles", "CC(C)(C)"])
        assert pool["single_fragment_ratio"] == 1.0
        assert pool["single_molecule_rate"] == pytest.approx(2.0 / 3.0)
        assert pool["n_single"] == 2

    def test_empty_pool_returns_nulls_not_zeroes(self):
        pool = score_smiles_pool([])
        assert pool["n_scored"] == 0
        assert pool["single_fragment_ratio"] is None
        assert pool["single_molecule_rate"] is None
        assert pool["fragmented_ratio"] is None


class TestVerdict:
    def _scored(self, pool, **met):
        return {
            "run_dir": "<test>",
            "epoch": None,
            "pool": score_smiles_pool(pool),
            "sources": {"per_epoch_metrics_csv": met or None},
        }

    def test_connected_pool_passes(self):
        v = evaluate(self._scored(["CCO", "CC(C)(C)"]))
        assert v["status"] == "PASS"
        assert v["single_fragment_ratio"] == 1.0
        assert v["blind_validity_signature"] is False

    def test_fragmented_pool_fails(self):
        v = evaluate(self._scored(["C.C.C"]))
        assert v["status"] == "FAIL"
        assert v["fragmented_ratio"] == 1.0
        assert any("one connected molecule" in r for r in v["reasons"])

    def test_no_number_is_skip_not_pass(self):
        v = evaluate({"run_dir": "<test>", "pool": None,
                      "sources": {"per_epoch_metrics_csv": None}})
        assert v["status"] == "SKIP"
        assert v["single_fragment_ratio"] is None

    def test_the_verdict_says_which_denominator_it_used(self):
        # A conditional number that reads 0.79 and an end-to-end number that
        # reads 0.07 are the same pool seen through two denominators, and
        # only one of them is what the model produced.
        v = evaluate(self._scored(["C.C.C"]))
        assert v["basis"] == "pool_end_to_end"
        assert v["single_molecule_rate"] == 0.0
        assert any("basis:" in r for r in v["reasons"])

    def test_the_trainer_columns_give_a_derived_end_to_end_rate(self):
        # No pool on disk: gen_validity x gen_single_fragment_ratio is the
        # end-to-end rate, and the verdict says it was derived, not measured.
        met = {"gen_validity": 0.0859, "gen_single_fragment_ratio": 0.7879}
        v = evaluate({
            "run_dir": "<test>", "pool": None,
            "sources": {"per_epoch_metrics_csv": met},
        })
        assert v["basis"] == "trainer_columns_derived"
        assert v["single_molecule_rate"] == pytest.approx(0.0859 * 0.7879)
        assert v["status"] == "FAIL"

    def test_a_conditional_only_run_is_labelled_as_such(self):
        met = {"gen_single_fragment_ratio": 0.9}
        v = evaluate({
            "run_dir": "<test>", "pool": None,
            "sources": {"per_epoch_metrics_csv": met},
        })
        assert v["basis"] == "trainer_conditional_column"
        assert v["single_molecule_rate"] is None
        assert any("CONDITIONAL" in r or "conditional" in r for r in v["reasons"])

    def test_blind_validity_signature_is_reported(self):
        # gen_validity saturated while the pool is fragmented: the failure
        # mode that survived 41 runs.
        met = {"gen_validity_val": 192, "gen_validity_n": 192}
        v = evaluate(self._scored(["C.C.C"], **met))
        assert v["status"] == "FAIL"
        assert v["blind_validity_signature"] is True
        assert any("BLIND-METRIC SIGNATURE" in r for r in v["reasons"])

    def test_blind_signature_needs_saturation(self):
        met = {"gen_validity_val": 100, "gen_validity_n": 192}
        v = evaluate(self._scored(["C.C.C"], **met))
        assert v["blind_validity_signature"] is False

    def test_decoder_reference_is_surfaced(self):
        met = {"decode_single_fragment_ratio": 0.9375}
        v = evaluate(self._scored(["CCO"], **met))
        assert v["decoder_reference_single_fragment_ratio"] == 0.9375

    def test_thresholds_are_policy_not_discovered_optima(self):
        # A pool that fails the default bar can pass a looser declared bar.
        v = evaluate(
            self._scored(["C.C.C"]),
            max_fragmented_ratio=1.0,
            min_largest_fragment_frac=0.0,
        )
        assert v["status"] == "PASS"
        assert DEFAULT_MAX_FRAGMENTED_RATIO == 0.50
        assert DEFAULT_MIN_LARGEST_FRAGMENT_FRAC == 0.90
        assert VALIDITY_SATURATION == 0.99


class TestRunDirScoring:
    def _write_run(self, tmp_path: Path, rows):
        d = tmp_path / "run"
        d.mkdir()
        with (d / "generated_smiles.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["epoch", "smiles", "n_frags_raw",
                                               "largest_frag_frac"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return d

    def test_csv_columns_are_cross_checked(self, tmp_path):
        d = self._write_run(
            tmp_path,
            [
                {"epoch": 0, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
                {"epoch": 0, "smiles": "C.C.C", "n_frags_raw": 3, "largest_frag_frac": 0.33},
            ],
        )
        scored = score_run_dir(d)
        cc = scored["sources"]["generated_smiles_csv"]["csv_column_cross_check"]
        assert cc["n_checked"] == 2
        assert cc["n_mismatch"] == 0

    def test_a_wrong_csv_column_is_detected(self, tmp_path):
        d = self._write_run(
            tmp_path,
            [{"epoch": 0, "smiles": "C.C.C", "n_frags_raw": 1, "largest_frag_frac": 1.0}],
        )
        scored = score_run_dir(d)
        cc = scored["sources"]["generated_smiles_csv"]["csv_column_cross_check"]
        assert cc["n_checked"] == 1
        assert cc["n_mismatch"] == 1

    def test_an_unparseable_row_does_not_misalign_the_cross_check(self, tmp_path):
        # Regression (Wf-A114, 2026-10-30): the pool skips empty and
        # unparseable strings, so lining the scored rows up with the source
        # rows POSITIONALLY misaligns every row after the first skip.  On the
        # c7 pool (one unparseable row at index 10370) the naive zip reported
        # 2299 mismatches against a re-derivation that agrees on all 12668
        # scored rows.
        d = self._write_run(
            tmp_path,
            [
                {"epoch": 0, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
                {"epoch": 0, "smiles": "not_a_smiles", "n_frags_raw": 0,
                 "largest_frag_frac": 0.0},
                {"epoch": 0, "smiles": "C.C.C", "n_frags_raw": 3, "largest_frag_frac": 0.33},
                {"epoch": 0, "smiles": "CC(C)(C)", "n_frags_raw": 1,
                 "largest_frag_frac": 1.0},
            ],
        )
        scored = score_run_dir(d)
        cc = scored["sources"]["generated_smiles_csv"]["csv_column_cross_check"]
        assert cc["n_checked"] == 3          # the unparseable row is skipped
        assert cc["n_mismatch"] == 0

    def test_a_real_mismatch_after_an_unparseable_row_is_still_caught(self, tmp_path):
        d = self._write_run(
            tmp_path,
            [
                {"epoch": 0, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
                {"epoch": 0, "smiles": "not_a_smiles", "n_frags_raw": 0,
                 "largest_frag_frac": 0.0},
                {"epoch": 0, "smiles": "C.C.C", "n_frags_raw": 1, "largest_frag_frac": 1.0},
            ],
        )
        scored = score_run_dir(d)
        cc = scored["sources"]["generated_smiles_csv"]["csv_column_cross_check"]
        assert cc["n_checked"] == 2
        assert cc["n_mismatch"] == 1         # the C.C.C row really is wrong

    def test_epoch_filter(self, tmp_path):
        d = self._write_run(
            tmp_path,
            [
                {"epoch": 0, "smiles": "C.C.C", "n_frags_raw": 3, "largest_frag_frac": 0.33},
                {"epoch": 1, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
            ],
        )
        rows = read_generated_smiles(d / "generated_smiles.csv", epoch=1)
        assert [r["smiles"] for r in rows] == ["CCO"]


class TestExitCodes:
    def _run_dir(self, tmp_path: Path, smiles):
        d = tmp_path / "run"
        d.mkdir()
        with (d / "generated_smiles.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["epoch", "smiles"])
            w.writeheader()
            for s in smiles:
                w.writerow({"epoch": 0, "smiles": s})
        return d

    def test_pass_exits_0(self, tmp_path):
        d = self._run_dir(tmp_path, ["CCO", "CC(C)(C)"])
        assert main(["--run-dir", str(d)]) == 0

    def test_fail_exits_1(self, tmp_path):
        d = self._run_dir(tmp_path, ["C.C.C"])
        assert main(["--run-dir", str(d)]) == 1

    def test_skip_exits_2(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert main(["--run-dir", str(d)]) == 2

    def test_strict_turns_skip_into_1(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert main(["--run-dir", str(d), "--strict"]) == 1

    def test_out_artifact_is_written(self, tmp_path):
        d = self._run_dir(tmp_path, ["CCO"])
        out = tmp_path / "verdict.json"
        assert main(["--run-dir", str(d), "--out", str(out)]) == 0
        art = json.loads(out.read_text(encoding="utf-8"))
        assert art["metric"] == "cfm_fragmentation_gate"
        assert art["tag"] == "MEASURED"
        assert art["verdict"]["status"] == "PASS"
        assert art["thresholds"]["max_fragmented_ratio"] == DEFAULT_MAX_FRAGMENTED_RATIO


class TestCensusCarriesFragmentation:
    def _verdict(self, single, largest):
        from molmetal.scripts.wf_cfm_run_census import _fragmentation_verdict

        return _fragmentation_verdict(single, largest)

    def test_no_number_is_skip(self):
        v = self._verdict(None, None)
        assert v["status"] == "SKIP"

    def test_connected_run_passes(self):
        v = self._verdict(0.94, 0.99)
        assert v["status"] == "PASS"

    def test_fragmented_run_fails(self):
        v = self._verdict(0.1222, 0.90)
        assert v["status"] == "FAIL"
        assert v["fragmented_ratio"] == pytest.approx(0.8778, abs=1e-6)

    def test_the_shipped_census_has_the_columns(self):
        p = REPO_ROOT / "metrics" / "by_metric" / "cfm_run_census.json"
        if not p.is_file():
            pytest.skip("census artifact not on disk")
        art = json.loads(p.read_text(encoding="utf-8"))
        runs = art.get("runs") or []
        assert runs, "census carries no runs"
        assert "fragmentation_gate" in runs[0]
        assert "gen_single_fragment_ratio_end" in runs[0]
        assert "gen_single_molecule_rate_end" in runs[0]
        assert "fragmentation_summary" in art
        summary = art["fragmentation_summary"]
        assert summary["n_runs_with_verdict"] + summary.get("n_skip", 0) > 0

    def test_the_census_says_which_denominator_the_verdict_used(self):
        p = REPO_ROOT / "metrics" / "by_metric" / "cfm_run_census.json"
        if not p.is_file():
            pytest.skip("census artifact not on disk")
        art = json.loads(p.read_text(encoding="utf-8"))
        for r in art["runs"]:
            gate = r.get("fragmentation_gate") or {}
            if gate.get("status") in ("PASS", "FAIL"):
                assert gate.get("basis") in (
                    "end_to_end",
                    "conditional",
                ), f"{r['run']} has no verdict basis"
                break
        else:
            pytest.fail("no run carries a fragmentation verdict")

    def test_the_census_end_to_end_rate_is_the_product_of_the_two_columns(self):
        p = REPO_ROOT / "metrics" / "by_metric" / "cfm_run_census.json"
        if not p.is_file():
            pytest.skip("census artifact not on disk")
        art = json.loads(p.read_text(encoding="utf-8"))
        checked = 0
        for r in art["runs"]:
            rate = r.get("gen_single_molecule_rate_end")
            v = r.get("gen_validity_end")
            s = r.get("gen_single_fragment_ratio_end")
            if rate is None or v is None or s is None:
                continue
            assert rate == pytest.approx(float(v) * float(s), abs=1e-9)
            checked += 1
        assert checked > 0, "no run carries both columns"


class TestAuditKeepsTheTwoPoolKindsApart:
    """A reconstruction is not a generation — the audit must never mix them."""

    def _artifact(self):
        p = REPO_ROOT / "metrics" / "by_metric" / "cfm_fragmentation_audit.json"
        if not p.is_file():
            pytest.skip("audit artifact not on disk")
        return json.loads(p.read_text(encoding="utf-8"))

    def test_the_audit_splits_the_pools_by_pipeline(self):
        art = self._artifact()
        s = art["summary"]
        assert s["n_generation_pools"] > 0
        assert s["n_reconstruction_pools"] > 0
        assert (
            s["n_generation_pools"] + s["n_reconstruction_pools"]
            + s["n_unknown_pools"]
            == s["n_runs_with_a_pool"]
        )

    def test_every_pool_row_carries_its_measurement_meaning(self):
        art = self._artifact()
        for r in art["runs"]:
            assert r["pipeline"] in ("generation", "reconstruction", "unknown")
            assert r["measurement_meaning"]
            assert r["applies_to"]

    def test_a_reconstruction_rate_is_not_in_the_generator_statistics(self):
        # The two kinds are separate SETS of pools: no pool is on both sides,
        # and the pool of record is on the GENERATION side.
        #
        # Wf-A114D: cfm_fullcorpus_2026_10_10 is FREE GENERATION (the
        # 2026-10-12+ ODE path of the decode-dump script), not a
        # reconstruction: every one of its 66 rows carries sanitized == 0 and
        # an empty metal_z.  The two era-A full-corpus pools of 2026-10-11
        # ARE reconstructions (sanitized == 1, the cloud's metal).
        art = self._artifact()
        gen_runs = {r["run"] for r in art["runs"] if r["pipeline"] == "generation"}
        rec_runs = {r["run"] for r in art["runs"] if r["pipeline"] == "reconstruction"}
        assert gen_runs and rec_runs
        assert not (gen_runs & rec_runs)
        assert "cfm_fullcorpus_2026_10_10" in gen_runs
        assert {
            "cfm_fullcorpus_b64_2026_10_11",
            "cfm_objective_terms_2026_10_11",
        } <= rec_runs

    def test_the_pool_of_record_is_scored_over_its_own_attempts(self):
        # The end-to-end denominator of the pool of record is the decode-dump
        # script's OWN attempt count (192), NOT the trainer's 768 epoch-level
        # draws from events.jsonl: the pool holds 66 survivors of those 192
        # attempts, so the rate is 64/192 = 0.3333 and the pool FAILS the
        # fragmentation gate.  A rate over the pool's own 66 rows (0.9697)
        # would be a rate over the survivors and would hide the failure.
        art = self._artifact()
        row = next(
            r for r in art["runs"] if r["run"] == "cfm_fullcorpus_2026_10_10"
        )
        assert row["pipeline"] == "generation"
        assert row["end_to_end_denominator"] == 192
        assert (
            "cfm_free_generation_panel_seed0.json"
            in row["end_to_end_denominator_source"]
        )
        assert row["single_molecule_rate_end_to_end"] == pytest.approx(
            64 / 192, abs=1e-9
        )
        assert row["verdict"]["status"] == "FAIL"
        # ... and the generation-side gate counts it.
        generation = art["summary"]["generation_side"]
        assert generation["n_pools"] == 10
        assert generation["gate_pass"] == 0
        assert generation["gate_fail"] == 10

    def test_the_audit_uses_the_canonical_pipeline_markers(self):
        art = self._artifact()
        pc = art["pipeline_classification"]
        assert pc["markers"]["generation"] == "n_frags_raw"
        assert pc["markers"]["reconstruction"] == "metal_z"
        # Wf-A114 CLUSTER C: the rule has ONE owner --
        # molmetal/scripts/cfm_pool_kind.py.  The shipped artifact still names
        # the previous owner until the integrator regenerates it, so the
        # artifact-side assertion accepts the migration; the OWNERSHIP itself
        # is pinned against the source tree, which is what must not drift.
        assert pc["canonical_source"] in (
            "molmetal/scripts/cfm_pool_kind.py",
            "molmetal/scripts/c7_scale_matched_pools.py",
        )
        owner = REPO_ROOT / "molmetal" / "scripts" / "cfm_pool_kind.py"
        assert owner.is_file()
        owner_text = owner.read_text(encoding="utf-8")
        assert 'RECONSTRUCTION_MARKER = "metal_z"' in owner_text
        assert 'GENERATION_MARKER = "n_frags_raw"' in owner_text
        # The audit must take the rule from that one owner, not re-derive it.
        audit_src = (
            REPO_ROOT / "molmetal" / "scripts" / "cfm_fragmentation_audit.py"
        ).read_text(encoding="utf-8")
        assert "from molmetal.scripts.cfm_pool_kind import" in audit_src

    def test_the_project_baseline_is_end_to_end_not_conditional(self):
        art = self._artifact()
        base = art["project_generation_baseline"]
        assert base["n_runs_with_both_columns"] > 0
        e2e = base["gen_single_molecule_rate_end"]
        cond = base["gen_single_fragment_ratio_end"]
        assert e2e["n"] <= cond["n"], (
            "the end-to-end rate must be computed on the runs that actually "
            "carry a validity column, which is a subset"
        )
        assert e2e["median"] is not None


class TestTheTrainerColumnDoesNotClaimAMoleculeItCannotSee:
    """The `n_frags_raw` column must not assert connectivity for a string
    RDKit cannot parse.

    Wf-A114 (2026-10-30): the trainer's generation-CSV writer defaulted to
    ``n_frags_raw = 1`` / ``largest_frag_frac = 1.0`` for any SMILES it could
    not parse — i.e. it claimed the row was one connected molecule holding all
    its atoms.  MEASURED on the pools on disk: 1 row of 21,679
    (`molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`).
    The sentinel is -1 = UNKNOWN.
    """

    @staticmethod
    def _source():
        from pathlib import Path

        return (
            Path(__file__).resolve().parents[2]
            / "molmetal"
            / "scripts"
            / "train_cfm_yuelbond_wasserstein.py"
        ).read_text(encoding="utf-8")

    def test_the_writer_no_longer_defaults_to_one_fragment(self):
        src = self._source()
        assert "_s_out, _nfrag, _lfrac = _s, 1, 1.0" not in src, (
            "the unparseable-SMILES default is back to claiming one fragment"
        )
        assert "_nfrag, _lfrac = -1, -1.0" in src, (
            "the -1 = UNKNOWN sentinel is gone from the writer"
        )

    def test_the_writer_uses_the_canonical_decode_chemistry(self):
        # The duplicated inline RDKit block is where the bug lived.  The
        # counts now come from the same function the audit scores with.
        src = self._source()
        assert "_chem = decode_chemistry(_s)" in src

    def test_decode_chemistry_returns_none_for_an_unparseable_string(self):
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            decode_chemistry,
        )

        assert decode_chemistry("not-a-smiles") is None
        assert decode_chemistry("") is None

    def test_decode_chemistry_agrees_with_the_inline_counts(self):
        # MEASURED bit-exact over every parseable row in the generation pools
        # on disk: 0 differences on n_fragments and on largest_fragment_frac.
        # Pin it on the cases that matter.
        from rdkit import Chem

        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            decode_chemistry,
        )

        for smi in ("C", "CCO", "c1ccccc1", "C.C", "CC(=O)[O-].[Pt+2]"):
            mol = Chem.MolFromSmiles(smi)
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            heavy = mol.GetNumHeavyAtoms()
            inline_n = len(frags)
            inline_l = (
                max(f.GetNumHeavyAtoms() for f in frags) / heavy if heavy else 0.0
            )
            chem = decode_chemistry(smi)
            assert chem is not None, smi
            assert int(chem["n_fragments"]) == inline_n, smi
            assert abs(float(chem["largest_fragment_frac"]) - inline_l) < 1e-9, smi
