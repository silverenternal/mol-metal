"""Tests for Wf-125 Phase 2 — LOMO Wasserstein + Rh oversample trainer.

Covers:

* CLI parses without error
* ``load_rh_synthetic_smiles`` filters to Catherine-6/6 PASS compounds
* ``append_synthetic_rh_to_dataset`` extends the dataset and reports the
  correct number of synthetic rows
* ``extract_per_metal_embeddings`` returns a per-metal dict with non-empty
  tensors of the right shape
* ``WassersteinLOMORunner.run`` smoke-completes on a tiny subset (≤ 200
  rows, 1 seed, 1 epoch, Rh held-out) and produces a result with all
  required fields + a Wasserstein loss history
* The Wasserstein loss history is non-empty and finite
* The ``--quick`` flag works end-to-end on the smoke path
* The end-to-end ``main()`` entry point produces the expected JSON
  artefacts (smoke mode, 1 metal × 1 seed)

Test count: 16 unit tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from molmetal.scripts.train_metal_lomo_wasserstein import (
    WassersteinLOMORunner,
    WassersteinLOMOSplitter,
    _aggregate,
    _parse_args,
    append_synthetic_rh_to_dataset,
    extract_per_metal_embeddings,
    load_rh_synthetic_smiles,
    main as trainer_main,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rh_cohort_path() -> Path:
    p = (
        PROJECT_ROOT
        / "molmetal"
        / "reports"
        / "wf125_p1_rh_oversample"
        / "synthetic_rh_cohort.json"
    )
    if not p.exists():
        pytest.skip(f"Wf-125 P1 Rh cohort not found at {p}; cannot test.")
    return p


@pytest.fixture(scope="module")
def tiny_lomo_csv(tmp_path_factory) -> Path:
    """Build a minimal 5-metal LOMO CSV (~ 200 rows) for smoke tests."""
    import pandas as pd

    src_csv = PROJECT_ROOT / "molmetal" / "data" / "metal_lomo.csv"
    if not src_csv.exists():
        pytest.skip(f"metal_lomo.csv not found at {src_csv}")
    df = pd.read_csv(src_csv, low_memory=False)
    rng = np.random.default_rng(0)
    per_metal = 40
    keep_idx = []
    for metal in ["Ru", "Ir", "Os", "Rh", "Re"]:
        m_idx = df.index[df["Metal"] == metal].to_numpy()
        if len(m_idx) >= per_metal:
            sel = rng.choice(m_idx, size=per_metal, replace=False)
            keep_idx.extend(sel.tolist())
    df_small = df.iloc[sorted(keep_idx)].reset_index(drop=True)
    out = PROJECT_ROOT / "tests" / "_tmp_metal_lomo_tiny.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_small.to_csv(out, index=False)
    return out


@pytest.fixture(scope="module")
def tiny_dataset(tiny_lomo_csv, tmp_path_factory):
    """Build a MetalCytotoxDataset directly via the constructor to avoid
    the ``from_csv`` default cache_dir path (``/mnt/storage/...``), which
    is not writable on this host."""
    import pandas as pd
    from molmetal.data.cytotox import _add_active, _add_pic50, _apply_filters

    cache_dir = tmp_path_factory.mktemp("cytotox_cache")
    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=None,
        compute_pic50=True,
        compute_active=True,
    )
    df = pd.read_csv(tiny_lomo_csv, low_memory=False)
    df = _apply_filters(df, flt)
    df = _add_pic50(df)
    df = _add_active(df)
    ds = MetalCytotoxDataset(df=df, cache_dir=cache_dir)
    return ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        ns = _parse_args([])
        assert ns.metals == ["Ru", "Ir", "Os", "Rh", "Re"]
        assert ns.seeds == [0, 1, 2]
        assert ns.epochs == 30
        # Wf-E1 (2026-10-05): the W2 term is OFF by default. It was 0.05
        # until Wf-C3 measured it as a detached no-op costing 2.4x
        # wall-time; the real gradient now requires --wasserstein-real.
        assert ns.wasserstein_weight == 0.0
        assert ns.wasserstein_real is False
        assert ns.rh_oversample == 500
        assert ns.n_mc == 32
        # Wf-127 P2 + P3 guards default off (backward-compat with Wf-125 P2).
        assert ns.use_l2_guard is False
        assert ns.use_astr_scale_guard is False
        assert ns.l2_guard_max_norm == 5.0
        assert ns.astr_scale_clip_min == -10.0
        assert ns.astr_scale_clip_max == 10.0

    def test_quick_flag(self):
        ns = _parse_args(["--quick"])
        assert ns.quick is True

    def test_explicit_metals(self):
        ns = _parse_args(["--metals", "Ru", "Rh", "--seeds", "0"])
        assert ns.metals == ["Ru", "Rh"]
        assert ns.seeds == [0]

    def test_l2_guard_flag_engaged(self):
        ns = _parse_args(["--use-l2-guard"])
        assert ns.use_l2_guard is True

    def test_astr_scale_guard_flag_engaged(self):
        ns = _parse_args(["--use-astr-scale-guard"])
        assert ns.use_astr_scale_guard is True

    def test_guard_clip_values_overridable(self):
        ns = _parse_args(
            [
                "--use-l2-guard",
                "--l2-guard-max-norm",
                "2.0",
                "--use-astr-scale-guard",
                "--astr-scale-clip-min",
                "-3.0",
                "--astr-scale-clip-max",
                "12.0",
            ]
        )
        assert ns.use_l2_guard is True
        assert ns.l2_guard_max_norm == 2.0
        assert ns.use_astr_scale_guard is True
        assert ns.astr_scale_clip_min == -3.0
        assert ns.astr_scale_clip_max == 12.0


# ---------------------------------------------------------------------------
# Rh synthetic loader
# ---------------------------------------------------------------------------


class TestRHSyntheticLoader:
    def test_filters_to_pass_compounds(self, rh_cohort_path):
        smis = load_rh_synthetic_smiles(rh_cohort_path)
        assert len(smis) >= 1
        assert all(isinstance(s, str) and s for s in smis)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_rh_synthetic_smiles(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# Dataset appender
# ---------------------------------------------------------------------------


class TestAppendSyntheticRh:
    def test_appends_to_dataset(self, tiny_dataset, rh_cohort_path):
        ds_new, n_appended = append_synthetic_rh_to_dataset(
            tiny_dataset, rh_cohort_path, n_target=50
        )
        assert n_appended == 50
        assert len(ds_new) == len(tiny_dataset) + 50
        # All appended rows are Rh.
        rh_rows = int((ds_new.metals == "Rh").sum())
        assert rh_rows >= int((tiny_dataset.metals == "Rh").sum()) + 50

    def test_full_cohort_appended(self, tiny_dataset, rh_cohort_path):
        ds_new, n_appended = append_synthetic_rh_to_dataset(
            tiny_dataset, rh_cohort_path, n_target=500
        )
        # 500 - capped by n_target if n_target < cohort size.
        assert n_appended <= 500
        assert len(ds_new) == len(tiny_dataset) + n_appended


# ---------------------------------------------------------------------------
# Per-metal embedding extraction
# ---------------------------------------------------------------------------


class TestExtractPerMetalEmbeddings:
    def test_returns_dict_with_per_metal_tensors(self, tiny_dataset):
        from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel

        device = torch.device("cpu")
        model = DMPNNMultiTaskModel(hidden=64, depth=2, dropout=0.1).to(device)
        smiles = tiny_dataset.smiles
        metals = tiny_dataset.metals
        embeds = extract_per_metal_embeddings(
            model, smiles, metals, device, max_per_metal=20
        )
        assert isinstance(embeds, dict)
        assert len(embeds) >= 2
        for m, t in embeds.items():
            assert isinstance(t, torch.Tensor)
            assert t.dim() == 2
            assert t.shape[1] == 2 * 64
            assert t.shape[0] >= 1


# ---------------------------------------------------------------------------
# End-to-end smoke run
# ---------------------------------------------------------------------------


class TestWassersteinLOMORunnerSmoke:
    def test_smoke_completes_with_default_weight(self, tiny_dataset, rh_cohort_path):
        # Append a few synthetic Rh rows so the oversample path runs.
        ds, _ = append_synthetic_rh_to_dataset(
            tiny_dataset, rh_cohort_path, n_target=30
        )
        runner = WassersteinLOMORunner(
            ds=ds,
            held_out_metal="Rh",
            wasserstein_weight=0.05,
            hidden=32,
            depth=1,
            dropout=0.0,
            lr=1e-3,
            epochs=1,
            batch_size=8,
            alpha=0.5,
            seed=0,
            device=torch.device("cpu"),
            synthetic_rh_count=30,
        )
        res = runner.run(verbose=False)
        # All required fields are populated.
        assert res.fold == "Rh"
        assert res.seed == 0
        assert res.n_train >= 1
        assert res.n_test >= 1
        assert 0.0 <= res.test_auc <= 1.0
        assert isinstance(res.wasserstein_loss_history, list)
        assert len(res.wasserstein_loss_history) >= 1
        assert all(np.isfinite(v) for v in res.wasserstein_loss_history)
        assert isinstance(res.final_per_metal_w2, dict)

    def test_smoke_completes_with_zero_weight(self, tiny_dataset):
        runner = WassersteinLOMORunner(
            ds=tiny_dataset,
            held_out_metal="Ru",
            wasserstein_weight=0.0,
            hidden=32,
            depth=1,
            dropout=0.0,
            lr=1e-3,
            epochs=1,
            batch_size=8,
            alpha=0.5,
            seed=0,
            device=torch.device("cpu"),
            synthetic_rh_count=0,
        )
        res = runner.run(verbose=False)
        assert res.fold == "Ru"
        assert 0.0 <= res.test_auc <= 1.0

    def test_w2_off_by_default_reports_zero_effective_weight(self, tiny_dataset):
        """Wf-E1 — a requested weight with no ``--wasserstein-real`` must be
        recorded as NOT applied, so the emitted JSON cannot imply the model
        was W₂-regularised."""
        runner = WassersteinLOMORunner(
            ds=tiny_dataset,
            held_out_metal="Ru",
            wasserstein_weight=0.05,
            hidden=32,
            depth=1,
            dropout=0.0,
            lr=1e-3,
            epochs=1,
            batch_size=8,
            alpha=0.5,
            seed=0,
            device=torch.device("cpu"),
            synthetic_rh_count=0,
            wasserstein_real=False,
        )
        assert runner.w2_enabled is False
        res = runner.run(verbose=False)
        assert res.wasserstein_weight == 0.05  # what was asked for
        assert res.w2_real is False  # what actually happened
        assert res.w2_effective_weight == 0.0
        # No W2 work means no W2 diagnostics (and none of their wall-time).
        assert res.final_per_metal_w2 == {}

    def test_smoke_runs_with_real_w2(self, tiny_dataset):
        """Wf-E1 — ``--wasserstein-real`` must survive a full training loop.

        The W2 graph is built once per epoch and back-propagated once, so the
        failure mode this guards is ``RuntimeError: Trying to backward
        through the graph a second time`` on the second batch of an injection
        epoch. It is not reachable from a single-batch smoke, hence the
        2-epoch run with a small batch size (many batches per epoch).
        """
        runner = WassersteinLOMORunner(
            ds=tiny_dataset,
            held_out_metal="Ru",
            wasserstein_weight=0.05,
            hidden=32,
            depth=1,
            dropout=0.0,
            lr=1e-3,
            epochs=2,
            batch_size=8,
            alpha=0.5,
            seed=0,
            device=torch.device("cpu"),
            synthetic_rh_count=0,
            wasserstein_real=True,
        )
        assert runner.w2_enabled is True
        res = runner.run(verbose=False)
        assert res.w2_real is True
        assert res.w2_effective_weight == 0.05
        assert 0.0 <= res.test_auc <= 1.0
        # The W2 diagnostics only exist when W2 is actually on.
        assert isinstance(res.wasserstein_loss_history, list)
        assert len(res.wasserstein_loss_history) == 2
        assert isinstance(res.final_per_metal_w2, dict)

    def test_smoke_completes_with_l2_and_astr_guards(self, tiny_dataset):
        # Wf-127 P2 + P3 guards engaged together: grad-norm clip + pIC50
        # value-space clip.  Confirms the flags round-trip into the
        # WassersteinLOMOResult dataclass without breaking the smoke path.
        runner = WassersteinLOMORunner(
            ds=tiny_dataset,
            held_out_metal="Ru",
            wasserstein_weight=0.05,
            hidden=32,
            depth=1,
            dropout=0.0,
            lr=1e-3,
            epochs=1,
            batch_size=8,
            alpha=0.5,
            seed=0,
            device=torch.device("cpu"),
            synthetic_rh_count=0,
            use_l2_guard=True,
            use_astr_scale_guard=True,
            l2_guard_max_norm=5.0,
            astr_scale_clip_min=-10.0,
            astr_scale_clip_max=10.0,
        )
        res = runner.run(verbose=False)
        assert res.use_l2_guard is True
        assert res.use_astr_scale_guard is True
        assert res.l2_guard_max_norm == 5.0
        assert res.astr_scale_clip_min == -10.0
        assert res.astr_scale_clip_max == 10.0
        assert 0.0 <= res.test_auc <= 1.0


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_aggregates_per_fold(self):
        per_run = [
            {"fold": "Ru", "seed": 0, "test_auc": 0.8},
            {"fold": "Ru", "seed": 1, "test_auc": 0.7},
            {"fold": "Rh", "seed": 0, "test_auc": 0.6},
        ]
        agg = _aggregate(per_run)
        assert agg["n_runs"] == 3
        assert agg["n_folds"] == 2
        assert abs(agg["test_auc_mean"] - 0.7) < 1e-6
        assert abs(agg["per_fold_auc_mean"]["Ru"] - 0.75) < 1e-6
        assert abs(agg["per_fold_auc_mean"]["Rh"] - 0.6) < 1e-6

    def test_handles_errors(self):
        per_run = [
            {"fold": "Ru", "seed": 0, "test_auc": 0.7},
            {"fold": "Ru", "seed": 1, "error": "boom"},
        ]
        agg = _aggregate(per_run)
        assert agg["n_runs"] == 1
        assert "Ru" in agg["per_fold_auc_mean"]


# ---------------------------------------------------------------------------
# End-to-end CLI smoke (1 metal × 1 seed)
# ---------------------------------------------------------------------------


class TestMainSmoke:
    def test_end_to_end_smoke(self, tmp_path, tiny_lomo_csv, rh_cohort_path):
        out_dir = tmp_path / "wf125_p2_smoke"
        metric_path = tmp_path / "metal_lomo_auc_wasserstein_smoke.json"
        rc = trainer_main(
            [
                "--metals",
                "Ru",
                "Rh",
                "--seeds",
                "0",
                "--epochs",
                "1",
                "--wasserstein-weight",
                "0.05",
                "--rh-oversample",
                "20",
                "--csv",
                str(tiny_lomo_csv),
                "--rh-cohort-json",
                str(rh_cohort_path),
                "--out-dir",
                str(out_dir),
                "--out-metric",
                str(metric_path),
                "--quick",
                "--hidden",
                "32",
                "--depth",
                "1",
                "--batch-size",
                "8",
            ]
        )
        # The script returns 0 (PASS) or 2 (BELOW_GATE) — both are valid
        # smoke outcomes. What we care about is that the artefacts exist.
        assert rc in (0, 2)
        assert out_dir.exists()
        assert (out_dir / "synthesis.md").exists()
        assert (out_dir / "aggregate.json").exists()
        agg = json.loads((out_dir / "aggregate.json").read_text())
        assert agg["n_runs"] >= 1
        assert metric_path.exists()
        payload = json.loads(metric_path.read_text())
        assert payload["metric"] == "metal_lomo_auc_wasserstein"
        # The payload must carry the Rh target + overall target hit flags
        # so the synthesis report can render them honestly.
        assert "rh_target_hit" in payload
        assert "overall_target_hit" in payload


# ---------------------------------------------------------------------------
# Wf-A3 — synthetic Rh routing + two-cohort class balance
#
# Regression cover for the defect found in the 2026-10-05 audit: the
# synthetic Rh rows carry ``Metal == "Rh"`` by design, so a plain
# ``metals == held_out`` test mask swept all 500 of them into the Rh TEST
# fold (82 % of a 608-row test set, every one labelled inactive, none of them
# ever trained on) and the reported Rh test AUC was measured mostly on
# fabricated rows.
# ---------------------------------------------------------------------------


class TestSyntheticRhRouting:
    def test_synthetic_rows_never_reach_the_held_out_test_fold(
        self, tiny_dataset, rh_cohort_path
    ):
        ds, n = append_synthetic_rh_to_dataset(
            tiny_dataset, rh_cohort_path, n_target=60
        )
        assert n == 60
        synth = (ds.df["Cell_line"].astype(str) == "SYNTHETIC_RH").to_numpy()
        assert int(synth.sum()) == 60

        for metal in ("Rh", "Ru", "Ir"):
            split = WassersteinLOMOSplitter(held_out_metal=metal, seed=0)(ds)
            assert int(synth[split.test_idx].sum()) == 0, (
                f"synthetic rows leaked into the {metal} test fold"
            )
            # They must still be trained on, not silently dropped.
            assert int(synth[split.train_idx].sum()) == 60

    def test_synthetic_rows_never_reach_val(self, tiny_dataset, rh_cohort_path):
        ds, _ = append_synthetic_rh_to_dataset(
            tiny_dataset, rh_cohort_path, n_target=60
        )
        synth = (ds.df["Cell_line"].astype(str) == "SYNTHETIC_RH").to_numpy()
        split = WassersteinLOMOSplitter(held_out_metal="Rh", seed=0)(ds)
        assert int(synth[split.val_idx].sum()) == 0

    def test_test_fold_is_real_rows_only(self, tiny_dataset, rh_cohort_path):
        """With the fix the Rh test fold is the real Rh rows, unchanged."""
        ds, _ = append_synthetic_rh_to_dataset(
            tiny_dataset, rh_cohort_path, n_target=60
        )
        split = WassersteinLOMOSplitter(held_out_metal="Rh", seed=0)(ds)
        real_rh = int(
            (
                (ds.df["Metal"].astype(str) == "Rh")
                & (ds.df["Cell_line"].astype(str) != "SYNTHETIC_RH")
            ).sum()
        )
        assert len(split.test_idx) == real_rh

    def test_splitter_unchanged_without_synthetic_rows(self, tiny_dataset):
        """No synthetic rows -> the split is the plain by-metal split."""
        split = WassersteinLOMOSplitter(held_out_metal="Rh", seed=0)(tiny_dataset)
        assert len(split.test_idx) == int((tiny_dataset.metals == "Rh").sum())


class TestTwoCohortBalance:
    def _write_active(self, tmp_path, n=40):
        import json

        recs = [
            {
                "compound_id": f"rh_syn_act_{i:04d}",
                "smiles": f"CC[{i}]",
                "is_active": 1,
                "catherine_6of6_pass": True,
            }
            for i in range(n)
        ]
        p = tmp_path / "active.json"
        p.write_text(json.dumps({"per_compound": recs}))
        return p

    def test_active_cohort_adds_positives(self, tiny_dataset, rh_cohort_path, tmp_path):
        act = self._write_active(tmp_path)
        ds, n = append_synthetic_rh_to_dataset(
            tiny_dataset, rh_cohort_path, n_target=60, active_cohort_json=act
        )
        assert n == 60
        synth = (ds.df["Cell_line"].astype(str) == "SYNTHETIC_RH").to_numpy()
        lab = ds.active.astype(int)
        assert int(synth.sum()) == 60
        # 50/50 split: both classes present among the synthetic block.
        assert int(lab[synth].sum()) > 0
        assert int((lab[synth] == 0).sum()) > 0

    def test_backward_compat_single_cohort_all_inactive(
        self, tiny_dataset, rh_cohort_path
    ):
        """No active cohort -> the original all-inactive behaviour."""
        ds, n = append_synthetic_rh_to_dataset(
            tiny_dataset, rh_cohort_path, n_target=60
        )
        synth = (ds.df["Cell_line"].astype(str) == "SYNTHETIC_RH").to_numpy()
        assert n == 60
        assert int(ds.active.astype(int)[synth].sum()) == 0

    def test_active_and_inactive_labels_agree_with_pic50(
        self, tiny_dataset, rh_cohort_path, tmp_path
    ):
        """pIC50 must not contradict the classification label."""
        act = self._write_active(tmp_path)
        ds, _ = append_synthetic_rh_to_dataset(
            tiny_dataset, rh_cohort_path, n_target=40, active_cohort_json=act
        )
        synth = (ds.df["Cell_line"].astype(str) == "SYNTHETIC_RH").to_numpy()
        lab = ds.active.astype(int)[synth]
        pic = ds.pic50.astype(float)[synth]
        assert (pic[lab == 1] > pic[lab == 0]).all()

    def test_missing_active_file_raises(
        self, tiny_dataset, rh_cohort_path, tmp_path
    ):
        with pytest.raises(FileNotFoundError):
            append_synthetic_rh_to_dataset(
                tiny_dataset,
                rh_cohort_path,
                n_target=40,
                active_cohort_json=tmp_path / "nope.json",
            )


class TestActiveCohortFile:
    def test_active_cohort_json_on_disk_is_all_active(self):
        p = (
            PROJECT_ROOT
            / "molmetal"
            / "reports"
            / "wf125_p1_rh_oversample"
            / "synthetic_rh_cohort_active.json"
        )
        if not p.exists():
            pytest.skip(f"Wf-A3 active cohort not found at {p}; cannot test.")
        import json

        d = json.loads(p.read_text())
        recs = d["per_compound"]
        assert len(recs) >= 500
        assert all(int(r.get("is_active", r.get("active", 0))) == 1 for r in recs)
        # Must be disjoint from the inactive cohort it is meant to balance.
        inactive = PROJECT_ROOT / "molmetal" / "reports" / "wf125_p1_rh_oversample" / "synthetic_rh_cohort.json"
        if inactive.exists():
            old = {e["smiles"] for e in json.loads(inactive.read_text())["per_compound"]}
            assert not (old & {r["smiles"] for r in recs})
