# Licensed under the Apache License, Version 2.0.
"""Wf-A128 — the standalone fragmentation pre-flight gate + the panel's cohort.

Two defects, two surfaces:

1. **The fragmentation gate was a REPORT COLUMN, not a launch gate.**  The
   verdict is computed by
   ``molmetal/scripts/wf_cfm_run_census.py::_fragmentation_verdict`` and
   recorded in ``metrics/by_metric/cfm_run_census.json``, but
   ``molmetal/scripts/pre_flight_check.py`` has NO fragmentation check, so
   every FAIL verdict consumed nothing.  This file pins the new standalone
   gate: it uses the census builder's OWN verdict function (so the basis
   vocabulary cannot drift), the same thresholds, and it **SKIPs instead of
   FAILing** whenever the input is absent or the denominator is empty — a
   gate that FAILs on missing data is how the repo got a pile of meaningless
   FAILs.

   **NO TALLY IS QUOTED HERE.**  The counts depend on WHICH artifact the gate
   is pointed at: the working-tree artifact carries a verdict block per run,
   the artifact at ``HEAD`` carries fewer runs and no verdict block at all,
   and a REGENERATED artifact (the census builder re-run) applies the
   Wf-A127 nulling rule and publishes ``fragmentation_gate_last_measured``
   beside the last epoch's verdict.  So the tests here assert the SHAPE OF
   THE RULE — a rate with an empty denominator is a SKIP under ANY artifact,
   a measured verdict is never hidden, and the disagreement list is empty
   exactly when the gate and the artifact agree — and they assert the
   artifact-specific facts only where they hold for every shape.

   Point the file at another artifact with ``WFA128_CENSUS=/path/to.json``.

2. **The property panel's decode cohort was invisible.**  The panel decodes
   with ``euler`` / 10 steps / noise sigma 1.0 while the trainer's own run
   configs declare ``midpoint`` / 50 steps / noise sigma 2.485, so a panel
   number was not comparable with the trainer's number for the same
   checkpoint and the panel's output artifact did not say so.  The panel now
   DISCLOSES its own cohort (``decode_cohort`` in the output artifact); its
   decode parameters are NOT changed.

CPU-only, RDKit-only, no GPU.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import pre_flight_fragmentation_check as frag  # noqa: E402
from molmetal.scripts.cfm_fragmentation_gate import (  # noqa: E402
    DEFAULT_MAX_FRAGMENTED_RATIO,
    DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
)

#: The census artifact this file is pointed at.  The working-tree artifact by
#: default; ``WFA128_CENSUS`` points it at another shape (the artifact at
#: HEAD, or a regeneration) so the same assertions can be run against both.
CENSUS = Path(
    os.environ.get("WFA128_CENSUS")
    or (REPO_ROOT / "metrics" / "by_metric" / "cfm_run_census.json")
)

#: The c7 pool.  Untracked on disk (the reports tree is not git-tracked), so
#: the test that reads it SKIPs when it is absent.
C7_POOL = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "cfm_c7_fullgen_2026_10_20"
    / "generated_smiles.csv"
)

#: The two runs whose LAST epoch scored none of the samples it drew, so the
#: last epoch's rate is 0/0 — undefined, not 0.0.
PLACEHOLDER_FAIL_RUNS = (
    "cfm_bond_head_30ep_onecycle_2026_10_09",
    "cfm_fm_control_2026_10_09",
)

#: The two run directories whose last epoch is unmeasured but which DID
#: measure a connectivity rate at an earlier epoch.
REAL_UNMEASURED_LAST_EPOCH_RUNS = {
    "cfm_bond_head_30ep_onecycle_2026_10_09": 22,
    "cfm_fm_control_2026_10_09": 20,
}


def _write_pool(path: Path, rows) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "smiles"])
        for i, smi in enumerate(rows):
            writer.writerow([i, smi])
    return path


def _write_epoch_csv(path: Path, row: dict) -> Path:
    fields = list(row.keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    return path


def _write_epoch_rows(path: Path, rows) -> Path:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_census(path: Path, runs) -> Path:
    path.write_text(json.dumps({"runs": list(runs)}), encoding="utf-8")
    return path


def _census_row(doc, name):
    for run in doc.get("runs") or []:
        if str(run.get("run")) == name:
            return run
    return None


@pytest.fixture(scope="module")
def census():
    if not CENSUS.is_file():
        pytest.skip(f"{CENSUS} is not on disk")
    return json.loads(CENSUS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def panel_mod():
    import importlib.util

    path = REPO_ROOT / "molmetal" / "scripts" / "cfm_property_panel.py"
    spec = importlib.util.spec_from_file_location(
        "wfa128_cfm_property_panel", str(path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTheGateContract:
    def test_script_exists_and_is_importable(self):
        assert (
            REPO_ROOT / "molmetal" / "scripts" / "pre_flight_fragmentation_check.py"
        ).is_file()

    def test_basis_vocabulary_is_the_census_builders(self):
        """The four basis values are the ARTIFACT's values, not a restatement."""
        assert frag.BASIS_VOCABULARY == (
            "field_absent",
            "denominator_empty",
            "end_to_end",
            "conditional",
        )

    def test_the_verdict_comes_from_the_census_builder_itself(self):
        """The gate must not re-implement the verdict — it must CALL it.

        A copy would drift.  So the function the gate uses IS the census
        builder's function object.
        """
        from molmetal.scripts.wf_cfm_run_census import _fragmentation_verdict

        assert frag._verdict_builder() is _fragmentation_verdict

    def test_thresholds_are_the_gate_modules_constants(self):
        out = frag.check_pool_csv(REPO_ROOT / "definitely" / "not_there.csv")
        assert out["thresholds"]["max_fragmented_ratio"] == DEFAULT_MAX_FRAGMENTED_RATIO
        assert (
            out["thresholds"]["min_largest_fragment_frac"]
            == DEFAULT_MIN_LARGEST_FRAGMENT_FRAC
        )
        assert DEFAULT_MAX_FRAGMENTED_RATIO == 0.50
        assert DEFAULT_MIN_LARGEST_FRAGMENT_FRAC == 0.90

    def test_exit_codes_follow_the_project_convention(self):
        assert frag._exit_code("PASS") == 0
        assert frag._exit_code("FAIL") == 1
        assert frag._exit_code("SKIP") == 2

    def test_the_module_states_no_hard_coded_tally(self):
        """The counts belong to an artifact, not to the module.

        A docstring that says "59 runs: SKIP 27 / FAIL 27 / PASS 5" is a
        claim about ONE artifact's shape, and it goes stale the moment the
        artifact is regenerated or the checkout is clean.  The tally is
        recomputed at RUN TIME instead, so the module must not quote one.
        """
        doc = frag.__doc__ or ""
        for forbidden in (
            "committed",
            "SKIP 27",
            "FAIL 27",
            "PASS 5",
            "59 runs",
            "2 disagreements",
        ):
            assert forbidden not in doc, forbidden

    def test_the_default_census_is_the_repositorys_artifact(self):
        """``DEFAULT_CENSUS`` must be the LIVE default, not a dead constant."""
        assert frag.DEFAULT_CENSUS == (
            REPO_ROOT / "metrics" / "by_metric" / "cfm_run_census.json"
        )

    def test_the_default_census_is_used_when_no_mode_is_named(self, monkeypatch):
        seen = []
        original = frag.check_census

        def spy(path):
            seen.append(Path(path))
            return original(path)

        monkeypatch.setattr(frag, "check_census", spy)
        frag.main([])
        assert seen == [frag.DEFAULT_CENSUS]


class TestSkipNeverFails:
    """A gate that FAILs on missing data is worse than no gate."""

    def test_absent_pool_is_skip_field_absent(self, tmp_path):
        out = frag.check_pool_csv(tmp_path / "nope.csv")
        assert out["status"] == "SKIP"
        assert out["basis"] == "field_absent"
        assert out["measured_rate"] is None

    def test_empty_pool_is_skip_not_fail(self, tmp_path):
        out = frag.check_pool_csv(_write_pool(tmp_path / "empty.csv", []))
        assert out["status"] == "SKIP"
        assert out["basis"] == "field_absent"

    def test_a_pool_nothing_parses_is_skip_denominator_empty(self, tmp_path):
        out = frag.check_pool_csv(
            _write_pool(tmp_path / "bad.csv", ["not_a_smiles", "%%%"])
        )
        assert out["status"] == "SKIP"
        assert out["basis"] == "denominator_empty"
        assert out["measured_rate"] is None

    def test_absent_census_is_skip(self, tmp_path):
        out = frag.check_census(tmp_path / "nope.json")
        assert out["status"] == "SKIP"
        assert out["basis"] == "field_absent"

    def test_absent_run_dir_is_skip(self, tmp_path):
        out = frag.check_run_dir(tmp_path / "no_such_run")
        assert out["status"] == "SKIP"
        assert out["basis"] == "field_absent"

    def test_epoch_columns_with_an_empty_scored_denominator_skip(self, tmp_path):
        """gen_validity == 0.0 means the eval scored NONE of what it drew."""
        out = frag.check_epoch_columns(
            _write_epoch_csv(
                tmp_path / "pem.csv",
                {
                    "epoch": 29,
                    "gen_validity": 0.0,
                    "gen_validity_val": 0,
                    "gen_validity_n": 48,
                    "gen_single_fragment_ratio": 0.0,
                    "gen_largest_fragment_frac": 0.0,
                },
            )
        )
        assert out["status"] == "SKIP"
        assert out["basis"] == "denominator_empty"

    def test_epoch_columns_without_the_ratio_column_skip_field_absent(self, tmp_path):
        out = frag.check_epoch_columns(
            _write_epoch_csv(
                tmp_path / "pem.csv",
                {"epoch": 3, "gen_validity": 1.0, "gen_validity_n": 10},
            )
        )
        assert out["status"] == "SKIP"
        assert out["basis"] == "field_absent"

    def test_an_unknown_basis_is_reported_not_raised(self):
        """A basis outside the vocabulary must degrade, never assert-crash."""
        out = frag._skip({}, "a_basis_from_the_future", "because")
        assert out["status"] == "SKIP"
        assert out["basis"] == "a_basis_from_the_future"
        assert out["basis_in_vocabulary"] is False
        assert any("a_basis_from_the_future" in r for r in out["reasons"]), out[
            "reasons"
        ]

    def test_an_unknown_basis_in_an_artifact_does_not_crash_the_gate(self, tmp_path):
        artifact = _write_census(
            tmp_path / "census.json",
            [
                {
                    "run": "a_run_with_a_new_basis",
                    "gen_single_fragment_ratio_end": None,
                    "gen_largest_fragment_frac_end": None,
                    "gen_validity_end": None,
                    "gen_eval_ran_on_last_epoch": False,
                    "fragmentation_gate": {
                        "status": "SKIP",
                        "basis": "a_basis_from_the_future",
                        "reason": "the census recorded a SKIP",
                    },
                }
            ],
        )
        out = frag.check_census(artifact)
        assert out["status"] == "SKIP"
        assert out["n_runs"] == 1
        assert out["runs"][0]["basis"] == "a_basis_from_the_future"
        assert out["runs"][0]["basis_in_vocabulary"] is False


class TestPoolScoring:
    def test_a_connected_pool_passes(self, tmp_path):
        out = frag.check_pool_csv(
            _write_pool(tmp_path / "good.csv", ["CCO", "c1ccccc1", "CCN"])
        )
        assert out["status"] == "PASS"
        assert out["basis"] == "end_to_end"
        assert out["pool"]["n_input"] == 3
        assert out["pool"]["n_scored"] == 3
        assert out["pool"]["n_single"] == 3
        assert out["measured_rate"] == pytest.approx(1.0)

    def test_a_fragmented_pool_fails(self, tmp_path):
        out = frag.check_pool_csv(
            _write_pool(tmp_path / "frag.csv", ["C.C.C.C", "C.C"])
        )
        assert out["status"] == "FAIL"
        assert out["basis"] == "end_to_end"
        assert out["fragmented_ratio"] == pytest.approx(1.0)
        assert out["pool"]["n_single"] == 0

    def test_the_verdict_is_on_the_pool_own_population(self, tmp_path):
        """The end-to-end denominator must be the pool's own rows here."""
        out = frag.check_pool_csv(
            _write_pool(tmp_path / "good.csv", ["CCO", "c1ccccc1"])
        )
        assert out["end_to_end_denominator"] == 2
        assert "pool file's own rows" in out["end_to_end_denominator_source"]

    def test_a_foreign_denominator_falls_back_to_the_conditional_basis(self, tmp_path):
        """A denominator from ANOTHER run is not this pool's population."""
        out = frag.check_pool_csv(
            _write_pool(tmp_path / "good.csv", ["CCO", "c1ccccc1"])
        )
        # Force the foreign-denominator branch by patching the pool scorer.
        from molmetal.scripts import cfm_fragmentation_gate as gate_mod

        original = gate_mod.pool_attempts

        def fake_attempts(_path):
            return {
                "n_attempted": 192,
                "source": "some_other_run.json",
                "source_path": "/nowhere",
            }

        gate_mod.pool_attempts = fake_attempts
        try:
            out = frag.check_pool_csv(
                _write_pool(tmp_path / "good2.csv", ["CCO", "c1ccccc1"])
            )
        finally:
            gate_mod.pool_attempts = original
        assert out["basis"] == "conditional"
        assert out["status"] == "PASS"
        assert any("ANOTHER run's artifact" in r for r in out["reasons"]), out[
            "reasons"
        ]


class TestTheRuleIsArtifactIndependent:
    """The RULE, pinned against every shape the census artifact can take.

    A test that hard-codes one artifact's counts is a test that fails the
    moment the artifact is regenerated.  These assert what must hold for
    EVERY shape instead.
    """

    # -- shape A: the PRE-normalisation artifact -------------------------
    # The artifact publishes the trainer's 0.0 placeholder as a measured
    # FAIL, and carries no `fragmentation_gate_last_measured` block.
    def _pre_normalisation_run(self):
        return {
            "run": "a_run_with_a_placeholder_zero",
            "gen_single_fragment_ratio_end": 0.0,
            "gen_largest_fragment_frac_end": 0.0,
            "gen_validity_end": 0.0,
            "gen_eval_ran_on_last_epoch": True,
            "fragmentation_gate": {
                "status": "FAIL",
                "basis": "end_to_end",
                "measured_rate": 0.0,
                "fragmented_ratio": 1.0,
                "single_fragment_ratio_conditional": 0.0,
                "reasons": ["fragmented_ratio 1.0000 > 0.5000"],
                "thresholds": {
                    "max_fragmented_ratio": 0.50,
                    "min_largest_fragment_frac": 0.90,
                },
                "epoch": 29,
            },
        }

    # -- shape B: the POST-normalisation artifact ------------------------
    # The rate is nulled, so the gate is SKIP / denominator_empty.
    def _post_normalisation_run(self):
        return {
            "run": "a_run_with_a_placeholder_zero",
            "gen_single_fragment_ratio_end": None,
            "gen_largest_fragment_frac_end": None,
            "gen_validity_end": 0.0,
            "gen_eval_ran_on_last_epoch": True,
            "fragmentation_gate": {
                "status": "SKIP",
                "basis": "denominator_empty",
                "reason": "the chemistry rates have an empty denominator",
                "epoch": 29,
            },
        }

    # -- shape C: post-normalisation PLUS the last-MEASURED verdict ------
    def _post_normalisation_with_last_measured_run(self):
        run = self._post_normalisation_run()
        run["fragmentation_gate_last_measured"] = {
            "status": "FAIL",
            "basis": "end_to_end",
            "measured_rate": 0.0,
            "fragmented_ratio": 1.0,
            "single_fragment_ratio_conditional": 0.0,
            "reasons": ["fragmented_ratio 1.0000 > 0.5000"],
            "thresholds": {
                "max_fragmented_ratio": 0.50,
                "min_largest_fragment_frac": 0.90,
            },
            "epoch": 22,
        }
        return run

    def test_an_empty_denominator_is_a_skip_under_the_pre_normalisation_shape(
        self, tmp_path
    ):
        artifact = _write_census(tmp_path / "a.json", [self._pre_normalisation_run()])
        out = frag.check_census(artifact)
        run = out["runs"][0]
        assert run["status"] == "SKIP", run
        assert run["basis"] == "denominator_empty", run
        assert run["measured_rate"] is None, run

    def test_an_empty_denominator_is_a_skip_under_the_post_normalisation_shape(
        self, tmp_path
    ):
        artifact = _write_census(tmp_path / "b.json", [self._post_normalisation_run()])
        out = frag.check_census(artifact)
        run = out["runs"][0]
        assert run["status"] == "SKIP", run
        assert run["basis"] == "denominator_empty", run

    def test_the_last_measured_verdict_is_not_hidden(self, tmp_path):
        """A measured FAIL must not be hidden behind an unmeasured last epoch.

        The census builder publishes ``fragmentation_gate_last_measured`` for
        exactly this case, so the gate reads it: the verdict is made at the
        last MEASURED epoch, not at the last epoch.
        """
        artifact = _write_census(
            tmp_path / "c.json", [self._post_normalisation_with_last_measured_run()]
        )
        out = frag.check_census(artifact)
        run = out["runs"][0]
        assert run["status"] == "FAIL", run
        assert run["basis"] == "end_to_end", run
        assert run["measured_rate"] == pytest.approx(0.0), run
        assert run["epoch"] == 22, run
        assert "MEASURED" in str(run.get("resourced_from")), run

    def test_the_disagreement_list_is_empty_when_the_gate_agrees(self, tmp_path):
        """A disagreement means the two verdicts are comparable and differ."""
        artifact = _write_census(
            tmp_path / "d.json",
            [
                {
                    "run": "a_run_that_agrees",
                    "gen_single_fragment_ratio_end": 0.95,
                    "gen_largest_fragment_frac_end": 0.99,
                    "gen_validity_end": 1.0,
                    "gen_eval_ran_on_last_epoch": True,
                    "fragmentation_gate": {
                        "status": "PASS",
                        "basis": "end_to_end",
                        "measured_rate": 0.95,
                        "fragmented_ratio": 0.05,
                        "single_fragment_ratio_conditional": 0.95,
                        "reasons": [],
                        "thresholds": {
                            "max_fragmented_ratio": 0.50,
                            "min_largest_fragment_frac": 0.90,
                        },
                        "epoch": 7,
                    },
                }
            ],
        )
        out = frag.check_census(artifact)
        assert out["runs"][0]["status"] == "PASS", out["runs"][0]
        assert out["n_disagreements_with_artifact"] == 0, out[
            "disagreements_with_artifact"
        ]

    def test_a_census_without_any_verdict_block_is_all_field_absent(self, tmp_path):
        """The shape at HEAD: fewer runs, no verdict block at all."""
        artifact = _write_census(
            tmp_path / "e.json",
            [
                {
                    "run": f"run_{i}",
                    "gen_single_fragment_ratio_end": None,
                    "gen_largest_fragment_frac_end": None,
                    "gen_validity_end": None,
                    "gen_eval_ran_on_last_epoch": False,
                }
                for i in range(3)
            ],
        )
        out = frag.check_census(artifact)
        assert out["status"] == "SKIP", out
        assert out["n_runs"] == 3
        assert out["n_pass"] == 0 and out["n_fail"] == 0 and out["n_skip"] == 3
        assert out["basis_counts"] == {"field_absent": 3}, out["basis_counts"]
        assert out["n_disagreements_with_artifact"] == 0

    def test_the_only_reason_the_gate_may_disagree_is_the_empty_denominator(
        self, tmp_path
    ):
        """Shape A disagrees (the artifact FAILs a placeholder 0.0).

        Shape B does not — the artifact already says SKIP / denominator_empty,
        so the gate and the artifact agree and the list is empty.
        """
        artifact = _write_census(tmp_path / "f.json", [self._pre_normalisation_run()])
        out = frag.check_census(artifact)
        assert out["n_disagreements_with_artifact"] == 1, out
        d = out["disagreements_with_artifact"][0]
        assert d["recorded_status"] == "FAIL", d
        assert d["rescored_status"] == "SKIP", d
        assert d["rescored_basis"] == "denominator_empty", d

        artifact = _write_census(tmp_path / "g.json", [self._post_normalisation_run()])
        out = frag.check_census(artifact)
        assert out["n_disagreements_with_artifact"] == 0, out[
            "disagreements_with_artifact"
        ]


class TestCensusMode:
    def test_every_run_gets_a_verdict_in_the_shared_vocabulary(self, census):
        out = frag.check_census(CENSUS)
        assert out["status"] in ("PASS", "FAIL", "SKIP")
        assert out["n_runs"] == len(census["runs"])
        for run in out["runs"]:
            assert run["status"] in ("PASS", "FAIL", "SKIP")
            if run["status"] == "SKIP":
                assert run["basis"] in frag.BASIS_VOCABULARY
            else:
                assert run["basis"] in ("end_to_end", "conditional")

    def test_no_run_fails_on_an_empty_denominator(self, census):
        """The gate's whole point: an undefined rate is a SKIP, not a FAIL."""
        out = frag.check_census(CENSUS)
        for run in out["runs"]:
            if run.get("recorded_status") != "FAIL":
                continue
            assert not (run["status"] == "FAIL" and run["basis"] == "denominator_empty")
        for run in out["runs"]:
            if run.get("status") == "FAIL":
                assert run.get("measured_rate") is not None

    def test_the_two_placeholder_runs_have_an_empty_last_epoch_denominator(
        self, census
    ):
        """Their LAST epoch drew samples and scored none of them.

        That is true of every artifact shape: it is a property of the run,
        not of the artifact.
        """
        for name in PLACEHOLDER_FAIL_RUNS:
            row = _census_row(census, name)
            if row is None:
                pytest.skip(f"{name} is not in {CENSUS}")
            assert row.get("gen_validity_end") == pytest.approx(0.0), (name, row)

    def test_the_two_placeholder_runs_are_never_failed_on_an_empty_denominator(
        self, census
    ):
        """Whatever the artifact, they are never FAILed on an empty rate.

        * artifact with a verdict block but no last-MEASURED verdict
          -> SKIP / denominator_empty;
        * artifact with a last-MEASURED verdict -> that verdict, made on a
          MEASURED rate;
        * artifact with NO verdict block at all (the shape at HEAD) -> the
          gate cannot tell the two SKIP causes apart, and says
          field_absent.  It is still a SKIP, never a FAIL.
        """
        out = frag.check_census(CENSUS)
        by_run = {r["run"]: r for r in out["runs"]}
        rows = {r.get("run"): r for r in census["runs"]}
        for name in PLACEHOLDER_FAIL_RUNS:
            if name not in by_run:
                pytest.skip(f"{name} is not in {CENSUS}")
            block = by_run[name]
            if not (rows.get(name) or {}).get("fragmentation_gate"):
                assert block["status"] == "SKIP", block
                assert block["basis"] == "field_absent", block
                continue
            if block["status"] == "SKIP":
                assert block["basis"] == "denominator_empty", block
                assert block["measured_rate"] is None, block
            else:
                assert block["status"] == "FAIL", block
                assert block["basis"] in ("end_to_end", "conditional"), block
                assert block["basis"] != "denominator_empty", block
                assert block.get("measured_rate") is not None, block
                assert "MEASURED" in str(block.get("resourced_from")), block

    def test_every_disagreement_is_the_empty_denominator_rule(self, census):
        """Only the empty-denominator rule may make the gate differ.

        NO COUNT IS ASSERTED: the number of disagreements is a property of
        the artifact (a stale one still FAILs a placeholder 0.0, a
        regenerated one does not), and a test that pins it breaks whenever
        the artifact changes.
        """
        out = frag.check_census(CENSUS)
        assert out["n_disagreements_with_artifact"] == len(
            out["disagreements_with_artifact"]
        )
        for d in out["disagreements_with_artifact"]:
            assert d["rescored_status"] == "SKIP", d
            assert d["rescored_basis"] == "denominator_empty", d

    def test_the_tally_is_recomputed_from_the_artifact(self, census):
        """The counts are DERIVED, so they hold for every artifact shape."""
        out = frag.check_census(CENSUS)
        assert out["n_runs"] == len(census["runs"])
        assert out["n_pass"] + out["n_fail"] + out["n_skip"] == out["n_runs"]
        expected = "FAIL" if out["n_fail"] else ("SKIP" if out["n_skip"] else "PASS")
        assert out["status"] == expected, out
        assert sum(out["basis_counts"].values()) == out["n_runs"], out["basis_counts"]

    def test_a_verdict_is_made_at_the_last_measured_epoch_when_one_is_published(
        self, census
    ):
        """A run whose last epoch is unmeasured keeps its measured verdict.

        The census builder publishes ``fragmentation_gate_last_measured`` for
        exactly this case, so the gate must not drop it.
        """
        out = frag.check_census(CENSUS)
        by_run = {r["run"]: r for r in out["runs"]}
        n_checked = 0
        for name, row in ((r.get("run"), r) for r in census["runs"]):
            lm = (row or {}).get("fragmentation_gate_last_measured")
            if not lm or lm.get("status") not in ("PASS", "FAIL"):
                continue
            n_checked += 1
            block = by_run.get(name)
            assert block is not None, name
            assert block["status"] == lm["status"], (name, block)
            assert block["basis"] == lm["basis"], (name, block)
            assert block["epoch"] == lm.get("epoch"), (name, block)
        if n_checked == 0:
            pytest.skip("the artifact publishes no last-MEASURED verdict")


class TestTheEpochModeFallback:
    """``--run-dir`` epoch mode: the last MEASURED epoch, not the last one."""

    def test_the_last_row_is_used_when_it_carries_a_measurement(self, tmp_path):
        out = frag.check_epoch_columns(
            _write_epoch_rows(
                tmp_path / "pem.csv",
                [
                    {
                        "epoch": 0,
                        "gen_validity": 1.0,
                        "gen_validity_n": 10,
                        "gen_validity_val": 10,
                        "gen_single_fragment_ratio": 0.9,
                        "gen_largest_fragment_frac": 0.95,
                    },
                    {
                        "epoch": 1,
                        "gen_validity": 1.0,
                        "gen_validity_n": 10,
                        "gen_validity_val": 10,
                        "gen_single_fragment_ratio": 0.9,
                        "gen_largest_fragment_frac": 0.95,
                    },
                ],
            )
        )
        assert out["status"] == "PASS", out
        assert out["row"]["epoch"] == 1.0, out["row"]
        assert "last row" in out["epoch_source"], out["epoch_source"]

    def test_the_last_measured_epoch_is_used_when_the_last_row_is_unmeasured(
        self, tmp_path
    ):
        """A measured FAIL must not be hidden behind an unmeasured last epoch.

        The last row's denominator is empty (the eval scored none of what it
        drew), so the rate there is 0/0 — undefined, not 0.0.  The gate falls
        back to the last epoch that WAS measured.
        """
        out = frag.check_epoch_columns(
            _write_epoch_rows(
                tmp_path / "pem.csv",
                [
                    {
                        "epoch": 5,
                        "gen_validity": 0.0208,
                        "gen_validity_n": 48,
                        "gen_validity_val": 1,
                        "gen_single_fragment_ratio": 0.0,
                        "gen_largest_fragment_frac": 0.913,
                    },
                    {
                        "epoch": 6,
                        "gen_validity": 0.0,
                        "gen_validity_n": 48,
                        "gen_validity_val": 0,
                        "gen_single_fragment_ratio": 0.0,
                        "gen_largest_fragment_frac": 0.0,
                    },
                ],
            )
        )
        assert out["status"] == "FAIL", out
        assert out["basis"] == "end_to_end", out
        assert out["row"]["epoch"] == 5.0, out["row"]
        assert out["last_row"]["epoch"] == 6.0, out["last_row"]
        assert out["measured_rate"] == pytest.approx(0.0), out

    def test_no_measured_epoch_at_all_is_a_skip(self, tmp_path):
        out = frag.check_epoch_columns(
            _write_epoch_rows(
                tmp_path / "pem.csv",
                [
                    {
                        "epoch": 0,
                        "gen_validity": 0.0,
                        "gen_validity_n": 48,
                        "gen_validity_val": 0,
                        "gen_single_fragment_ratio": 0.0,
                        "gen_largest_fragment_frac": 0.0,
                    },
                    {
                        "epoch": 1,
                        "gen_validity": 0.0,
                        "gen_validity_n": 48,
                        "gen_validity_val": 0,
                        "gen_single_fragment_ratio": 0.0,
                        "gen_largest_fragment_frac": 0.0,
                    },
                ],
            )
        )
        assert out["status"] == "SKIP", out
        assert out["basis"] == "denominator_empty", out
        assert out["measured_rate"] is None, out

    @pytest.mark.parametrize(
        "name,epoch", sorted(REAL_UNMEASURED_LAST_EPOCH_RUNS.items())
    )
    def test_the_real_run_dirs_on_disk(self, name, epoch):
        """The two real runs whose last epoch is unmeasured.

        Their last row carries the trainer's 0.0 placeholder, and an EARLIER
        epoch carries a measured rate, so the gate must judge that epoch.
        """
        directory = REPO_ROOT / "molmetal" / "reports" / name
        if not (directory / "per_epoch_metrics.csv").is_file():
            pytest.skip(f"{name} is not on disk")
        out = frag.check_run_dir(directory)
        assert out["status"] == "FAIL", out
        assert out["basis"] == "end_to_end", out
        assert out["row"]["epoch"] == float(epoch), out["row"]
        assert out["measured_rate"] is not None, out


class TestTheAdvisoryOnThePoolColumns:
    def test_the_fabricated_single_fragment_claim_is_counted(self):
        """REPORT-ONLY finding, measured by the new gate.

        The c7 pool's single unparseable row carries ``n_frags_raw = 1`` and
        ``largest_frag_frac = 1.0`` — a fabricated "perfect single fragment"
        on a string that is not a molecule.  The trainer now writes the
        sentinel -1 / -1.0 for such rows.
        """
        if not C7_POOL.is_file():
            pytest.skip(f"{C7_POOL} is not on disk")
        out = frag.check_pool_csv(C7_POOL)
        claims = out["csv_connectivity_claims"]
        assert claims["has_n_frags_raw_column"] is True
        assert claims["n_claimed_single_fragment_on_unparseable"] == 1
        assert claims["examples"][0]["n_frags_raw"] == 1.0
        assert claims["examples"][0]["largest_frag_frac"] == 1.0

    def test_no_pool_on_disk_carries_an_n_attempted_column(self):
        """REPORT-ONLY finding: the gate's denominator never comes from the pool.

        ``cfm_fragmentation_gate.pool_attempts`` looks for the pool's own
        ``n_attempted`` column first.  No pool on disk has one, so the
        fallback is the free-generation panel — a DIFFERENT run's artifact.
        """
        pools = sorted(
            (REPO_ROOT / "molmetal" / "reports").glob("*/generated_smiles.csv")
        )
        if not pools:
            pytest.skip("no generation pools on disk")
        for pool in pools:
            with pool.open(newline="", encoding="utf-8") as fh:
                header = csv.DictReader(fh).fieldnames or []
            assert "n_attempted" not in header, pool


class TestPanelCohortDisclosure:
    def test_the_panel_decode_params_are_unchanged(self, panel_mod):
        """Wf-A128 discloses the cohort; it does NOT change it.

        Changing them would silently move every published panel number.
        """
        assert panel_mod.PANEL_DECODE_METHOD == "euler"
        assert panel_mod.PANEL_DECODE_N_STEPS == 10
        assert panel_mod.PANEL_DECODE_NOISE_SIGMA == 1.0

    def test_the_sampler_call_uses_the_named_constants(self, panel_mod):
        src = (REPO_ROOT / "molmetal" / "scripts" / "cfm_property_panel.py").read_text(
            encoding="utf-8"
        )
        assert "method=PANEL_DECODE_METHOD" in src
        assert "n_steps=PANEL_DECODE_N_STEPS" in src
        assert "noise_sigma=PANEL_DECODE_NOISE_SIGMA" in src

    def test_the_output_artifact_carries_the_disclosure(self, panel_mod, tmp_path):
        pool = _write_pool(tmp_path / "in.csv", ["CCO", "c1ccccc1"])
        out_json = tmp_path / "panel.json"
        rc = panel_mod.main(
            [
                "--smiles-csv",
                str(pool),
                "--out-json",
                str(out_json),
            ]
        )
        assert rc == 0
        artifact = json.loads(out_json.read_text(encoding="utf-8"))
        cohort = artifact["decode_cohort"]
        assert cohort["decode_source"] == "smiles_csv"
        assert cohort["is_trainer_decode_cohort"] is False
        assert cohort["panel_decode_params"] is None
        assert cohort["trainer_decode_cohort"] is None
        assert "notes"

    def test_the_disclosure_names_the_differences(self, panel_mod):
        cohort = panel_mod._decode_cohort_disclosure(
            decode_source="checkpoint",
            ckpt_path=None,
            seed=0,
            n_samples=64,
            trainer_args={"n_steps": 50, "method": "midpoint", "fm_noise_sigma": 2.485},
        )
        assert cohort["panel_decode_params"] == {
            "method": "euler",
            "n_steps": 10,
            "noise_sigma": 1.0,
        }
        assert cohort["trainer_decode_cohort"] == {
            "n_steps": 50,
            "method": "midpoint",
            "fm_noise_sigma": 2.485,
        }
        assert cohort["is_trainer_decode_cohort"] is False
        diffs = cohort["panel_vs_trainer_differences"]
        assert any("method: panel=euler vs trainer=midpoint" in d for d in diffs)
        assert any("n_steps: panel=10 vs trainer=50" in d for d in diffs)
        assert any("noise_sigma: panel=1.0 vs trainer=2.485" in d for d in diffs)

    def test_a_checkpoint_mtime_is_recorded(self, panel_mod, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        ckpt.write_bytes(b"not a real checkpoint")
        cohort = panel_mod._decode_cohort_disclosure(
            decode_source="checkpoint",
            ckpt_path=ckpt,
            seed=0,
            n_samples=4,
            trainer_args=None,
        )
        assert cohort["checkpoint_path"] == str(ckpt)
        assert cohort["checkpoint_mtime_epoch"] == pytest.approx(ckpt.stat().st_mtime)
        assert cohort["checkpoint_mtime_iso"]
        assert cohort["trainer_decode_cohort"] is None
        assert cohort["trainer_decode_cohort_source"] is None

    def test_the_trainer_configs_declare_the_cited_cohort(self):
        """The disclosure's cited config values must be true on disk."""
        for name, lines in (
            ("c7_fullgen_2026_10_20.yaml", (579, 580, 271)),
            ("c8_fullgen_b64_2026_10_24.yaml", (632, 633, 307)),
        ):
            path = REPO_ROOT / "configs" / name
            assert path.is_file(), path
            text = path.read_text(encoding="utf-8").splitlines()
            n_steps_line = text[lines[0] - 1]
            method_line = text[lines[1] - 1]
            sigma_line = text[lines[2] - 1]
            assert n_steps_line.strip() == "n_steps: 50", n_steps_line
            assert method_line.strip() == "method: midpoint", method_line
            assert sigma_line.strip() == "fm_noise_sigma: 2.485", sigma_line


class TestNotWiredIntoBuildGates:
    """The gate count is quoted in 7 documents — Wf-A128 does NOT move it.

    The main thread owns ``build_gates``; this file only asserts that the
    new check is NOT registered there, so a later wiring lands deliberately.
    """

    def test_the_fragmentation_check_is_not_a_pre_flight_gate_yet(self):
        from molmetal.scripts.pre_flight_check import (
            EXPECTED_GATE_COUNTS,
            build_gates,
        )

        names = [name for name, _runner in build_gates("all")]
        assert not any("fragmentation" in n for n in names), names
        assert "p3_cfm_fragmentation_preflight" not in names
        assert EXPECTED_GATE_COUNTS["all"] == len(build_gates("all"))
        assert EXPECTED_GATE_COUNTS["frontier"] == len(build_gates("frontier"))
