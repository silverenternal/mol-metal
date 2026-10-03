#!/usr/bin/env python3
"""Wf-433-A4 test suite — 14 NEW tests covering both protocol validators
+ each baseline + edge cases (cite-only baselines, missing fields,
batch mode, atomic-write contract, etc.).

Run::

    pytest molmetal/tests/test_wf433_protocol_validator.py -v

CPU-only by design (no GPU launches, no wet-lab planning).  Per
``AGENTS.md §5.6`` the test fixtures use the project venv where
:mod:`wf386a3_atomic_write` is importable.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "molmetal" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import wf433_protocol_validator as v_main  # noqa: E402
import wf433_protocol_alignment_validator as v_align  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — synthetic results JSONs in tmp dirs
# ---------------------------------------------------------------------------


def _make_m1_results() -> Dict[str, Any]:
    """m1 — Ru subset binary AUC on MetalCytoToxDB (per Wf-413-A1 canonical 7/7)."""
    return {
        "metric": "metal_lomo_auc_wasserstein",
        "value": 0.7577,
        "value_std": 0.0924,
        "per_fold_auc_mean": {"Rh": 0.8798, "Re": 0.7088},
        "per_fold_auc_std": {"Rh": 0.0115, "Re": 0.0482},
        "per_run_auc": [0.8879, 0.8717, 0.7018, 0.7463, 0.7467, 0.6295, 0.7196],
        "n_runs": 7,
        "n_folds": 2,
        "metals": ["Rh", "Re"],
        "seeds": [3, 4, 0, 1, 2, 3, 4],
        "data_source": "/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv",
        "split": "scaffold",
        "verdict": "BELOW_KRASNOV_GATE_PARTIAL",
    }


def _make_m2_results() -> Dict[str, Any]:
    """m2 — Pt 4-cell-line AUC on PlatinAI (per Wf-388-A1 + Wf-398-A2)."""
    return {
        "metric": "platinai_4cell_auc",
        "value_overall_mean_auc": 0.5532,
        "value_overall_mean_std": 0.0093,
        "value_a2780cis_auc": 0.6476,
        "value_a2780cis_std": 0.0065,
        "per_cell_line": {
            "MCF7": {
                "auc_mean": 0.5844,
                "auc_std": 0.0232,
                "per_seed": [0.5887, 0.6104, 0.5541],
            },
            "A549": {
                "auc_mean": 0.6302,
                "auc_std": 0.0300,
                "per_seed": [0.6032, 0.6721, 0.6154],
            },
            "A2780": {
                "auc_mean": 0.3505,
                "auc_std": 0.0189,
                "per_seed": [0.3371, 0.3371, 0.3771],
            },
            "A2780cis": {
                "auc_mean": 0.6476,
                "auc_std": 0.0065,
                "per_seed": [0.6563, 0.6458, 0.6406],
            },
        },
        "n_seeds": 3,
        "n_rows": 4078,
        "data_source": "molmetal/data/platinai_4cell.csv",
        "split": "scaffold",
        "model": "dmpnn_4cell_multitask",
    }


def _make_m17_results() -> Dict[str, Any]:
    """m17 — DrugOOD OOD generalization (assay / scaffold / size)."""
    return {
        "metric": "drugood_ood_auc",
        "date": "2026-09-25",
        "featurization": "morgan_fp_2048_r2",
        "splits": ["assay", "scaffold", "size"],
        "seeds": [0, 1, 2],
        "data_source": "metacytotox_fallback",
        "imold_reference": {
            "assay": 0.7211,
            "scaffold": 0.6884,
            "size": 0.6792,
        },
        "results": {
            "assay": {
                "ood_auc_mean": 0.8914,
                "ood_auc_std": 0.0019,
                "per_seed_ood": [0.8888, 0.8924, 0.8929],
            },
            "scaffold": {
                "ood_auc_mean": 0.6973,
                "ood_auc_std": 0.0271,
                "per_seed_ood": [0.7223, 0.7101, 0.6597],
            },
            "size": {
                "ood_auc_mean": 0.7479,
                "ood_auc_std": 0.0055,
                "per_seed_ood": [0.7405, 0.7497, 0.7536],
            },
        },
        "split": "assay_scaffold_size",
    }


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestWf433ProtocolValidatorKrasnov(unittest.TestCase):
    """Tests for ``wf433_protocol_validator.py`` against Krasnov 2026 (m1)."""

    def setUp(self) -> None:
        self.results = _make_m1_results()

    def test_krasnov_per_seed_stats(self) -> None:
        """Per-seed stats: mean ~ 0.7577, std ~ 0.0832."""
        from wf433_protocol_validator import _safe_mean, _safe_std

        per_seed = self.results["per_run_auc"]
        mean = _safe_mean(per_seed)
        std = _safe_std(per_seed)
        self.assertIsNotNone(mean)
        self.assertAlmostEqual(mean, 0.7577, places=3)
        self.assertIsNotNone(std)
        self.assertGreater(std, 0.0)

    def test_krasnov_per_fold_extraction(self) -> None:
        """m1 extractor captures per_fold_auc_mean as 2-element dict."""
        from wf433_protocol_validator import _extract_m1_stats

        s = _extract_m1_stats(self.results)
        self.assertEqual(set(s.per_fold.keys()), {"Rh", "Re"})
        self.assertAlmostEqual(s.per_fold["Rh"], 0.8798, places=4)
        self.assertAlmostEqual(s.per_fold["Re"], 0.7088, places=4)
        self.assertEqual(s.n_runs, 7)
        self.assertEqual(len(s.per_seed), 7)

    def test_krasnov_verdict_below_baseline(self) -> None:
        """m1 0.7577 vs Krasnov Ru 0.81 + adjustment 0.02 = 0.83 → BELOW."""
        from wf433_protocol_validator import _verdict

        ok, label, diss, thr = _verdict(
            metric_value=0.7577,
            threshold=0.83,
            direction="higher_better",
        )
        self.assertFalse(ok)
        self.assertEqual(label, "BELOW_BASELINE_THRESHOLD")
        self.assertAlmostEqual(diss or 0.0, (0.83 - 0.7577) / 0.83, places=3)
        self.assertAlmostEqual(thr or 0.0, 0.83, places=2)

    def test_krasnov_full_validation(self) -> None:
        """End-to-end: read results + compare against Krasnov Ru 0.81 + emit verdict."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(self.results, fh)
            tmp_path = Path(fh.name)
        try:
            payload = v_main.run_validation(
                results_path=tmp_path,
                baseline_key="krasnov_2026",
                out_path=None,
            )
            self.assertEqual(payload["baseline"], "krasnov_2026")
            self.assertEqual(payload["metric_id"], "m1")
            self.assertEqual(payload["baseline_label"], "Ru")
            self.assertAlmostEqual(payload["baseline_value"] or 0.0, 0.81)
            self.assertAlmostEqual(payload["protocol_adjustment"], 0.02)
            self.assertAlmostEqual(payload["threshold_comparable"] or 0.0, 0.83)
            self.assertEqual(payload["verdict"], "BELOW_BASELINE_THRESHOLD")
            self.assertFalse(payload["pass"])
            self.assertGreater(payload["disagreement_score"] or 0.0, 0.05)
            self.assertIn("Krasnov", payload["verdict_detail"])
            self.assertEqual(
                payload["alignment_level"],
                "DATASET+METRIC ALIGN (split differs)",
            )
        finally:
            tmp_path.unlink(missing_ok=True)


class TestWf433ProtocolValidatorRusanov(unittest.TestCase):
    """Tests for ``wf433_protocol_validator.py`` against Rusanov 2025 (m2)."""

    def setUp(self) -> None:
        self.results = _make_m2_results()

    def test_rusanov_per_cell_extraction(self) -> None:
        """m2 extractor captures per_cell_line as 4-cell dict."""
        from wf433_protocol_validator import _extract_m2_stats

        s = _extract_m2_stats(self.results)
        self.assertEqual(set(s.per_cell.keys()), {"MCF7", "A549", "A2780", "A2780cis"})
        self.assertAlmostEqual(s.per_cell["A2780cis"], 0.6476, places=4)
        self.assertAlmostEqual(s.overall_value or 0.0, 0.5532, places=4)
        self.assertEqual(len(s.per_seed), 12)  # 4 cells × 3 seeds

    def test_rusanov_verdict_pass_a2780cis(self) -> None:
        """A2780cis 0.6476 vs Rusanov 0.62 - 0.03 = 0.59 → PASS."""
        from wf433_protocol_validator import _verdict

        ok, label, diss, thr = _verdict(
            metric_value=0.6476,
            threshold=0.59,
            direction="higher_better",
        )
        self.assertTrue(ok)
        self.assertEqual(label, "PASS_BASELINE_GE_THRESHOLD")
        self.assertAlmostEqual(diss or 0.0, 0.0, places=2)
        self.assertAlmostEqual(thr or 0.0, 0.59, places=2)

    def test_rusanov_full_validation(self) -> None:
        """End-to-end: A2780cis MEASURED 0.6476 PASSES Rusanov baseline 0.62."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(self.results, fh)
            tmp_path = Path(fh.name)
        try:
            payload = v_main.run_validation(
                results_path=tmp_path,
                baseline_key="rusanov_2025",
                out_path=None,
            )
            self.assertEqual(payload["baseline"], "rusanov_2025")
            self.assertEqual(payload["metric_id"], "m2")
            self.assertEqual(payload["baseline_label"], "A2780cis")
            self.assertAlmostEqual(payload["baseline_value"] or 0.0, 0.62)
            self.assertAlmostEqual(payload["protocol_adjustment"], -0.03)
            self.assertAlmostEqual(payload["threshold_comparable"] or 0.0, 0.59)
            self.assertEqual(payload["verdict"], "PASS_BASELINE_GE_THRESHOLD")
            self.assertTrue(payload["pass"])
            self.assertEqual(
                payload["alignment_level"], "DATASET+METRIC+SPLIT APPROX ALIGN"
            )
        finally:
            tmp_path.unlink(missing_ok=True)


class TestWf433ProtocolValidatorDrugood(unittest.TestCase):
    """Tests for ``wf433_protocol_validator.py`` against DrugOOD 2022 (m17)."""

    def setUp(self) -> None:
        self.results = _make_m17_results()

    def test_drugood_per_split_extraction(self) -> None:
        """m17 extractor captures per_split as 3-split dict."""
        from wf433_protocol_validator import _extract_m17_stats

        s = _extract_m17_stats(self.results)
        self.assertEqual(set(s.per_split.keys()), {"assay", "scaffold", "size"})
        self.assertAlmostEqual(s.per_split["assay"], 0.8914, places=4)
        self.assertAlmostEqual(s.per_split["scaffold"], 0.6973, places=4)
        self.assertAlmostEqual(s.per_split["size"], 0.7479, places=4)
        # Mean over 3 splits ≈ (0.8914 + 0.6973 + 0.7479) / 3 ≈ 0.7789
        self.assertAlmostEqual(s.overall_value or 0.0, 0.7789, places=3)
        self.assertEqual(len(s.per_seed), 9)  # 3 splits × 3 seeds

    def test_drugood_full_validation(self) -> None:
        """End-to-end: m17 3-split mean 0.7789 vs iMoLD mean 0.6962 → PASS."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(self.results, fh)
            tmp_path = Path(fh.name)
        try:
            payload = v_main.run_validation(
                results_path=tmp_path,
                baseline_key="drugood_2022",
                out_path=None,
            )
            self.assertEqual(payload["baseline"], "drugood_2022")
            self.assertEqual(payload["metric_id"], "m17")
            self.assertEqual(payload["baseline_label"], "iMoLD_mean")
            # iMoLD mean ≈ (0.7211 + 0.6884 + 0.6792) / 3 ≈ 0.6962
            self.assertAlmostEqual(payload["baseline_value"] or 0.0, 0.6962, places=4)
            self.assertAlmostEqual(payload["protocol_adjustment"], 0.0)
            # threshold_comparable == baseline_value (no adjustment)
            self.assertAlmostEqual(
                payload["threshold_comparable"] or 0.0, 0.6962, places=4
            )
            self.assertEqual(payload["verdict"], "PASS_BASELINE_GE_THRESHOLD")
            self.assertTrue(payload["pass"])
            self.assertEqual(
                payload["alignment_level"],
                "DATASET+METRIC+SPLIT FULL ALIGN",
            )
        finally:
            tmp_path.unlink(missing_ok=True)


class TestWf433ProtocolValidatorEdgeCases(unittest.TestCase):
    """Tests for edge cases — cite-only baselines, unknown metric, missing fields."""

    def test_unknown_baseline_raises(self) -> None:
        """Passing an unsupported --baseline value raises ValueError."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(_make_m1_results(), fh)
            tmp_path = Path(fh.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                v_main.run_validation(
                    results_path=tmp_path,
                    baseline_key="totally_made_up_baseline_2027",
                    out_path=None,
                )
            self.assertIn("unknown baseline", str(ctx.exception).lower())
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_missing_results_file_raises(self) -> None:
        """Passing a non-existent --our-results raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            v_main.run_validation(
                results_path=Path("/tmp/does_not_exist_xyz_12345.json"),
                baseline_key="krasnov_2026",
                out_path=None,
            )

    def test_metric_id_inference_by_filename(self) -> None:
        """METRIC_KEYS_BY_NAME map correctly infers m1 / m2 / m17 from filename."""
        import tempfile

        # Bare JSON with no ``metric`` field — should still infer from stem.
        bare = {"value": 0.5}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(bare, fh)
            tmp_path = Path(fh.name)
        try:
            metric_id = v_main._infer_metric_id(bare, tmp_path)
            self.assertEqual(metric_id, "unknown")
        finally:
            tmp_path.unlink(missing_ok=True)

        # With the metric field set explicitly.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump({"metric": "metal_lomo_auc_wasserstein", "value": 0.5}, fh)
            tmp_path2 = Path(fh.name)
        try:
            metric_id = v_main._infer_metric_id(
                {"metric": "metal_lomo_auc_wasserstein", "value": 0.5}, tmp_path2
            )
            self.assertEqual(metric_id, "m1")
        finally:
            tmp_path2.unlink(missing_ok=True)

    def test_safe_mean_and_std(self) -> None:
        """_safe_mean / _safe_std handle empty + 1-element lists gracefully."""
        from wf433_protocol_validator import _safe_mean, _safe_std

        self.assertIsNone(_safe_mean([]))
        self.assertIsNone(_safe_std([]))
        self.assertIsNone(_safe_std([1.0]))
        self.assertAlmostEqual(_safe_mean([1.0, 2.0, 3.0]) or 0.0, 2.0, places=6)
        # sample std of [1,2,3] = 1.0
        self.assertAlmostEqual(_safe_std([1.0, 2.0, 3.0]) or 0.0, 1.0, places=6)

    def test_verdict_with_none_inputs(self) -> None:
        """Verdict returns NOT_COMPARABLE when metric or threshold is None."""
        from wf433_protocol_validator import _verdict

        ok, label, diss, thr = _verdict(
            metric_value=None, threshold=0.83, direction="higher_better"
        )
        self.assertFalse(ok)
        self.assertEqual(label, "NOT_COMPARABLE")
        self.assertIsNone(diss)
        self.assertIsNone(thr)

        ok, label, diss, thr = _verdict(
            metric_value=0.7577, threshold=None, direction="higher_better"
        )
        self.assertFalse(ok)
        self.assertEqual(label, "NOT_COMPARABLE")
        self.assertIsNone(diss)
        self.assertIsNone(thr)


class TestWf433ProtocolAlignmentValidator(unittest.TestCase):
    """Tests for ``wf433_protocol_alignment_validator.py`` (dataset + split + metric)."""

    def setUp(self) -> None:
        self.results_m1 = _make_m1_results()
        self.results_m2 = _make_m2_results()
        self.results_m17 = _make_m17_results()

    def test_alignment_pass_krasnov(self) -> None:
        """m1 results JSON passes all 4 alignment axes for Krasnov."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(self.results_m1, fh)
            tmp_path = Path(fh.name)
        try:
            payload = v_align.run_alignment_check(
                results_path=tmp_path,
                baseline_key="krasnov_2026",
                out_path=None,
            )
            self.assertEqual(payload["baseline"], "krasnov_2026")
            self.assertEqual(payload["metric_id"], "m1")
            self.assertEqual(len(payload["axes"]), 4)
            self.assertTrue(payload["alignment_match"])
            self.assertEqual(payload["n_fail"], 0)
            # dataset axis should PASS (path contains metalcytotox_5metal_multitask_v4.csv).
            ds_axis = next(
                a for a in payload["axes"] if a["axis_name"] == "dataset_path"
            )
            self.assertEqual(ds_axis["verdict"], "PASS")
            # seed axis should PASS (n_seeds=7 from per_run_auc).
            seed_axis = next(
                a for a in payload["axes"] if a["axis_name"] == "seed_coverage"
            )
            self.assertEqual(seed_axis["verdict"], "PASS")
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_alignment_pass_rusanov(self) -> None:
        """m2 results JSON passes alignment for Rusanov."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(self.results_m2, fh)
            tmp_path = Path(fh.name)
        try:
            payload = v_align.run_alignment_check(
                results_path=tmp_path,
                baseline_key="rusanov_2025",
                out_path=None,
            )
            self.assertTrue(payload["alignment_match"])
            self.assertEqual(payload["n_fail"], 0)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_alignment_pass_drugood(self) -> None:
        """m17 results JSON emits WARN for dataset axis (fallback dataset) but
        alignment_match stays True (lenient semantics: WARN is non-blocking)."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(self.results_m17, fh)
            tmp_path = Path(fh.name)
        try:
            payload = v_align.run_alignment_check(
                results_path=tmp_path,
                baseline_key="drugood_2022",
                out_path=None,
            )
            # m17 data_source is "metacytotox_fallback" — triggers WARN on
            # dataset axis per honest framing.  WARN is non-blocking so
            # alignment_match stays True.
            self.assertTrue(payload["alignment_match"])
            self.assertGreaterEqual(payload["n_warn"], 1)
            ds_axis = next(
                a for a in payload["axes"] if a["axis_name"] == "dataset_path"
            )
            self.assertEqual(ds_axis["verdict"], "WARN")
            self.assertIn("FALLBACK", ds_axis["verdict_detail"])
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_alignment_fail_wrong_dataset_path(self) -> None:
        """Wrong data_source path triggers dataset_path axis FAIL."""
        bad = dict(_make_m1_results())
        bad["data_source"] = "/tmp/wrong_path/some_random_file.csv"
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(bad, fh)
            tmp_path = Path(fh.name)
        try:
            payload = v_align.run_alignment_check(
                results_path=tmp_path,
                baseline_key="krasnov_2026",
                out_path=None,
            )
            ds_axis = next(
                a for a in payload["axes"] if a["axis_name"] == "dataset_path"
            )
            self.assertEqual(ds_axis["verdict"], "FAIL")
            self.assertFalse(payload["alignment_match"])
            self.assertEqual(payload["n_fail"], 1)
            self.assertEqual(payload["overall_status"], "FAIL")
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_alignment_warn_no_split_field(self) -> None:
        """Missing ``split`` field emits WARN (lenient semantics)."""
        bad = dict(_make_m1_results())
        del bad["split"]
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(bad, fh)
            tmp_path = Path(fh.name)
        try:
            payload = v_align.run_alignment_check(
                results_path=tmp_path,
                baseline_key="krasnov_2026",
                out_path=None,
            )
            split_axis = next(
                a for a in payload["axes"] if a["axis_name"] == "split_protocol"
            )
            self.assertEqual(split_axis["verdict"], "WARN")
            # WARN does NOT count as fail (lenient semantics).
            self.assertTrue(payload["alignment_match"])
            self.assertGreaterEqual(payload["n_warn"], 1)
        finally:
            tmp_path.unlink(missing_ok=True)


class TestWf433ProtocolBatch(unittest.TestCase):
    """Test the batch mode of the alignment validator."""

    def test_batch_mode_all_3_baselines(self) -> None:
        """Batch mode runs all 3 baselines on 3 metric JSONs.

        Honest framing: cross-pair baselines (e.g. m1 vs Rusanov) FAIL
        the dataset axis because the data sources don't match.  Same-pair
        baselines (m1 vs Krasnov, m2 vs Rusanov, m17 vs DrugOOD) PASS /
        WARN alignment.
        """
        import tempfile

        paths: List[Path] = []
        for data in (_make_m1_results(), _make_m2_results(), _make_m17_results()):
            fh = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(data, fh)
            fh.close()
            paths.append(Path(fh.name))
        try:
            payload = v_align.run_batch(
                metric_roots=paths,
                out_path=None,
            )
            self.assertEqual(payload["mode"], "batch")
            self.assertEqual(payload["n_entries"], 9)  # 3 metrics × 3 baselines
            # 3 same-pair entries: m1↔Krasnov (PASS), m2↔Rusanov (PASS),
            # m17↔DrugOOD (WARN on dataset axis but alignment_match True).
            # 6 cross-pair entries FAIL dataset axis (m1↔Rusanov, m1↔DrugOOD,
            # m2↔Krasnov, m2↔DrugOOD, m17↔Krasnov, m17↔Rusanov).
            self.assertEqual(payload["n_alignment_pass"], 3)
            self.assertEqual(payload["n_alignment_fail"], 6)
            # Find same-pair entries.
            same_pair = [
                e
                for e in payload["batch_results"]
                if (e["metric_id"] == "m1" and e["baseline"] == "krasnov_2026")
                or (e["metric_id"] == "m2" and e["baseline"] == "rusanov_2025")
                or (e["metric_id"] == "m17" and e["baseline"] == "drugood_2022")
            ]
            self.assertEqual(len(same_pair), 3)
            for e in same_pair:
                self.assertTrue(e["alignment_match"])
        finally:
            for p in paths:
                p.unlink(missing_ok=True)


class TestWf433ProtocolAtomicWrite(unittest.TestCase):
    """Verify atomic-write contract per AGENTS.md §5.6."""

    def test_atomic_write_used_when_out_provided(self) -> None:
        """When --out is provided, the validator writes via atomic_write_file."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(_make_m1_results(), fh)
            tmp_path = Path(fh.name)
        out_path = Path(tempfile.gettempdir()) / "wf433_test_atomic_write.json"
        try:
            v_main.run_validation(
                results_path=tmp_path,
                baseline_key="krasnov_2026",
                out_path=out_path,
            )
            self.assertTrue(out_path.is_file(), "output JSON should exist")
            payload = json.loads(out_path.read_text())
            self.assertEqual(payload["baseline"], "krasnov_2026")
            self.assertIn("verdict", payload)
        finally:
            tmp_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
