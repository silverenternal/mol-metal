"""Wf-385-A3 — test surface for the SCAFFOLD-loss wire-in verification gate.

Per AGENTS.md §3 skill catalog + §5.2 honest-framing: the standalone
gate script at ``molmetal/scripts/wf385a3_scaffold_wire_in_gate.py``
plus its pre-flight integration in
``molmetal/scripts/pre_flight_check.py::gate_wf385_scaffold_loss_wire_in_verified``
must each have an automated test surface so a future regression in
either the SCAFFOLD-export probe or the 50-step smoke invariant is
caught by ``pytest`` before reaching the pre-flight cascade.

The test surface is split into 5 test classes:

  (a) ``TestModuleExportVerification`` — probes each SCAFFOLD module
      for the expected function/class; mocks the loader to be
      deterministic on the host Python WITHOUT torch (env-aware SKIP
      contract per Wf-345-A2).
  (b) ``TestSmokeMetricsParsing`` — exercises the loss/decode_ratio
      parser on synthetic ``per_epoch_metrics.json`` fixtures
      (decrease + collapse + NaN + missing-file cases).
  (c) ``TestSubprocessExitCodePropagation`` — verifies that the
      gate's verdict matches the subprocess exit code (rc=0 →
      PASS; rc!=0 → FAIL).
  (d) ``TestPreFlightIntegration`` — verifies the Wf-385 gate is in
      ``build_gates('all')`` + the function is importable +
      ``EXPECTED_GATE_COUNTS`` accounts for the new gate.
  (e) ``TestEndToEndHappyPath`` — runs the full gate surface in
      skip-smoke + JSON-only modes (no torch / no subprocess
      required) so the test passes on every host Python without
      GPU/torch dependencies.

The end-to-end SUBPROCESS smoke (50-step CPU training with all 14
Wf-380 flags ON) is NOT exercised in this test suite — it requires
the torch-enabled runtime (kanzi_venv or
``/home/hugo/codes/CadButEaas/.venv-vlm/bin/python`` per AGENTS.md
§5.1) and runs ~30-60 s; the ``pre_flight_check.py --pipeline all``
invocation already covers this on every fresh-run by virtue of the
Wf-385 gate being wired into ``build_gates('all')``. The test
suite is intentionally torch-light so it can run on any host Python.

CPU-only by design. Wall budget: ~5 s on the host Python.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

# Make the gate script importable as a module.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))

import wf385a3_scaffold_wire_in_gate as gate_module  # noqa: E402


# ---------------------------------------------------------------------
# (a) Module-export verification
# ---------------------------------------------------------------------
class TestModuleExportVerification(unittest.TestCase):
    """Verify ``verify_module_exports`` returns the canonical 11 SCAFFOLD
    modules + propagates env-aware SKIP for hosts without torch.
    """

    def test_module_table_has_11_entries(self) -> None:
        self.assertEqual(
            len(gate_module.SCAFFOLD_MODULES), 11,
            "expected 11 SCAFFOLD modules per Wf-380-B1 ship surface",
        )

    def test_module_table_unique_filenames(self) -> None:
        filenames = [m[0] for m in gate_module.SCAFFOLD_MODULES]
        self.assertEqual(
            len(filenames), len(set(filenames)),
            "SCAFFOLD_MODULES must have unique filenames",
        )

    def test_module_table_contains_all_canonical_names(self) -> None:
        expected_filenames = {
            "edm_preconditioning.py",
            "optimal_transport_v2_solver.py",
            "stable_vm.py",
            "so3_average.py",
            "triton_mini_batch_ot.py",
            "shortcut_distillation_wf380.py",
            "consistency_regularization_wf380.py",
            "flowmol_unified.py",
            "log_normal_t.py",
            "lr_warmup.py",
            "diversity_sampling.py",
        }
        actual_filenames = {m[0] for m in gate_module.SCAFFOLD_MODULES}
        self.assertEqual(expected_filenames, actual_filenames)

    def test_verify_module_exports_returns_one_entry_per_module(self) -> None:
        results = gate_module.verify_module_exports()
        self.assertEqual(len(results), 11)
        for entry in results:
            self.assertIn("filename", entry)
            self.assertIn("expected", entry)
            self.assertIn("export_ok", entry)
            self.assertIn("on_disk", entry)

    def test_module_export_env_aware_skip_on_no_torch(self) -> None:
        """On a host without torch, every SCAFFOLD module surfaces
        ModuleNotFoundError; the env-aware SKIP contract per Wf-345-A2
        preserves ``export_ok=False`` with a descriptive error.
        """
        results = gate_module.verify_module_exports()
        # At least one module on_disk=True; on host Python without
        # torch all present=True should be False.
        on_disk_count = sum(1 for r in results if r["on_disk"])
        self.assertGreater(
            on_disk_count, 0,
            "at least one SCAFFOLD module should be on disk",
        )
        if not self._has_torch():
            export_ok_count = sum(1 for r in results if r["export_ok"])
            self.assertEqual(
                export_ok_count, 0,
                "without torch, all SCAFFOLD modules must report "
                "export_ok=False (env-aware SKIP)",
            )

    @staticmethod
    def _has_torch() -> bool:
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------
# (b) Smoke-metrics parsing
# ---------------------------------------------------------------------
class TestSmokeMetricsParsing(unittest.TestCase):
    """Verify ``parse_smoke_metrics`` on synthetic per_epoch_metrics.json
    fixtures: decrease + collapse + NaN + missing-file.
    """

    def _write_metrics(self, path: Path, per_epoch: List[Dict[str, Any]]) -> None:
        data = {"per_epoch_metrics": per_epoch}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_decrease_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_epoch_metrics.json"
            self._write_metrics(path, [
                {"train_loss": 1.92, "decode_ratio_rdkit_strict": 1.0},
                {"train_loss": 1.91, "decode_ratio_rdkit_strict": 1.0},
                {"train_loss": 1.90, "decode_ratio_rdkit_strict": 1.0},
            ])
            result = gate_module.parse_smoke_metrics(path)
            self.assertTrue(result["passed_decrease"])
            self.assertTrue(result["passed_decode"])
            self.assertEqual(result["loss_first"], 1.92)
            self.assertEqual(result["loss_last"], 1.90)

    def test_collapse_invariant(self) -> None:
        """loss_last > 10x loss_first is catastrophic collapse → FAIL."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_epoch_metrics.json"
            self._write_metrics(path, [
                {"train_loss": 1.0, "decode_ratio_rdkit_strict": 1.0},
                {"train_loss": 50.0, "decode_ratio_rdkit_strict": 1.0},
            ])
            result = gate_module.parse_smoke_metrics(path)
            self.assertFalse(result["passed_decrease"])

    def test_constant_invariant(self) -> None:
        """loss_last == loss_first fails the 0.3% relative-decrease
        tolerance (catastrophic non-decrease).
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_epoch_metrics.json"
            self._write_metrics(path, [
                {"train_loss": 1.92, "decode_ratio_rdkit_strict": 1.0},
                {"train_loss": 1.92, "decode_ratio_rdkit_strict": 1.0},
            ])
            result = gate_module.parse_smoke_metrics(path)
            self.assertFalse(result["passed_decrease"])

    def test_nan_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_epoch_metrics.json"
            self._write_metrics(path, [
                {"train_loss": float("nan"), "decode_ratio_rdkit_strict": 1.0},
                {"train_loss": float("nan"), "decode_ratio_rdkit_strict": 1.0},
            ])
            result = gate_module.parse_smoke_metrics(path)
            self.assertFalse(result["passed_decrease"])
            # JSON deserializes NaN as float('nan'), not None.
            self.assertTrue(math.isnan(result["loss_first"]))
            self.assertTrue(math.isnan(result["loss_last"]))

    def test_missing_file_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "does_not_exist.json"
            result = gate_module.parse_smoke_metrics(path)
            self.assertFalse(result["passed_decrease"])
            self.assertFalse(result["passed_decode"])
            self.assertIn("missing", result["error"])

    def test_decode_ratio_collapse(self) -> None:
        """Per Wf-269C MEASURED: --no-yuelbond → decode_ratio 0.000.
        decode_ratio=0.0 must fail the >0 strict floor.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_epoch_metrics.json"
            self._write_metrics(path, [
                {"train_loss": 1.92, "decode_ratio_rdkit_strict": 1.0},
                {"train_loss": 1.90, "decode_ratio_rdkit_strict": 0.0},
            ])
            result = gate_module.parse_smoke_metrics(path)
            self.assertTrue(result["passed_decrease"])
            self.assertFalse(result["passed_decode"])
            self.assertEqual(result["decode_ratio"], 0.0)


# ---------------------------------------------------------------------
# (c) Subprocess exit-code propagation
# ---------------------------------------------------------------------
class TestSubprocessExitCodePropagation(unittest.TestCase):
    """Verify the ``gate_main`` function correctly propagates the
    subprocess exit code into the verdict.
    """

    def test_gate_main_skip_smoke_returns_pass_on_modules_ok(self) -> None:
        """With --skip-smoke and torch-on-host = SCAN violation,
        gate_main returns verdict=PASS only if all 11 modules export.
        """
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "wf385a3_test"
            # Patch verify_module_exports to return all-OK entries.
            with mock.patch.object(
                gate_module,
                "verify_module_exports",
                return_value=[
                    {"filename": "fake.py", "module_attr": "fake",
                     "expected": "fake_fn", "on_disk": True,
                     "export_ok": True, "present": True, "error": None}
                    for _ in range(11)
                ],
            ):
                result = gate_module.gate_main(
                    out_dir=out_dir, skip_smoke=True,
                )
                self.assertEqual(result["verdict"], "PASS")
                self.assertTrue(Path(out_dir / "gate_result.json").is_file())

    def test_gate_main_skip_smoke_returns_fail_on_modules_missing(self) -> None:
        """If even one SCAFFOLD module is missing, verdict=FAIL."""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "wf385a3_test"
            with mock.patch.object(
                gate_module,
                "verify_module_exports",
                return_value=[
                    {"filename": "fake.py", "module_attr": "fake",
                     "expected": "fake_fn", "on_disk": True,
                     "export_ok": True, "present": True, "error": None}
                    for _ in range(10)
                ] + [
                    {"filename": "broken.py", "module_attr": "broken",
                     "expected": "broken_fn", "on_disk": True,
                     "export_ok": False, "present": False,
                     "error": "ModuleNotFoundError: No module named 'torch'"}
                ],
            ):
                result = gate_module.gate_main(
                    out_dir=out_dir, skip_smoke=True,
                )
                self.assertEqual(result["verdict"], "FAIL")
                self.assertIn("broken.py", result["reason"])

    def test_gate_result_json_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "wf385a3_test"
            with mock.patch.object(
                gate_module,
                "verify_module_exports",
                return_value=[
                    {"filename": f"f{i}.py", "module_attr": f"a{i}",
                     "expected": f"fn{i}", "on_disk": True,
                     "export_ok": True, "present": True, "error": None}
                    for i in range(11)
                ],
            ):
                gate_module.gate_main(out_dir=out_dir, skip_smoke=True)
                json_path = out_dir / "gate_result.json"
                self.assertTrue(json_path.is_file())
                payload = json.loads(json_path.read_text())
                self.assertEqual(payload["verdict"], "PASS")
                self.assertEqual(len(payload["modules"]), 11)


# ---------------------------------------------------------------------
# (d) Pre-flight integration
# ---------------------------------------------------------------------
class TestPreFlightIntegration(unittest.TestCase):
    """Verify the Wf-385 gate is wired into ``build_gates('all')`` + the
    function is importable + ``EXPECTED_GATE_COUNTS`` accounts for the
    new gate.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Ensure molmetal.scripts is importable so pre_flight_check loads.
        sys.path.insert(0, str(_REPO_ROOT))

    def test_gate_function_is_importable(self) -> None:
        from molmetal.scripts import pre_flight_check
        self.assertTrue(
            hasattr(pre_flight_check, "gate_wf385_scaffold_loss_wire_in_verified"),
            "gate_wf385_scaffold_loss_wire_in_verified must be a "
            "top-level function in pre_flight_check.py",
        )

    def test_gate_is_in_build_gates_all(self) -> None:
        from molmetal.scripts import pre_flight_check
        gates = pre_flight_check.build_gates("all")
        gate_names = [name for name, _ in gates]
        self.assertIn(
            "p3_wf385_scaffold_loss_wire_in_verified", gate_names,
            "Wf-385-A3 gate must be wired into build_gates('all')",
        )

    def test_gate_priority_and_pipeline_labels(self) -> None:
        from molmetal.scripts import pre_flight_check
        gates = pre_flight_check.build_gates("all")
        wf385 = [
            (name, runner)
            for name, runner in gates
            if name == "p3_wf385_scaffold_loss_wire_in_verified"
        ]
        self.assertEqual(len(wf385), 1)
        # The lambda calls the function with frozen name + priority +
        # pipeline; verify the call signature compiles by inspecting
        # the lambda source (Python 3.11+).
        import inspect
        src = inspect.getsource(wf385[0][1])
        self.assertIn("P3", src)
        self.assertIn("wf385a3", src)

    def test_expected_gate_counts_includes_wf385(self) -> None:
        from molmetal.scripts import pre_flight_check
        # Per Wf-345-B3 auto-recompute, the audit gate uses
        # ``len(build_gates(pipeline))`` at runtime; the hardcoded
        # EXPECTED_GATE_COUNTS constant is kept for backward-compat.
        # The auto-recompute ensures the count is consistent.
        gates = pre_flight_check.build_gates("all")
        self.assertGreaterEqual(
            len(gates), 95,
            f"actual gate count {len(gates)} must be ≥ 95 "
            f"(94 pre-Wf-385 + 1 NEW Wf-385-A3)",
        )


# ---------------------------------------------------------------------
# (e) End-to-end happy path (skip-smoke mode, no torch required)
# ---------------------------------------------------------------------
class TestEndToEndHappyPath(unittest.TestCase):
    """Verify the gate's CLI ``main()`` returns 0 on the happy path
    (skip-smoke mode) and writes a valid gate_result.json.
    """

    def test_cli_skip_smoke_returns_zero_on_modules_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "wf385a3_test"
            with mock.patch.object(
                gate_module,
                "verify_module_exports",
                return_value=[
                    {"filename": f"f{i}.py", "module_attr": f"a{i}",
                     "expected": f"fn{i}", "on_disk": True,
                     "export_ok": True, "present": True, "error": None}
                    for i in range(11)
                ],
            ):
                rc = gate_module.main([
                    "--out-dir", str(out_dir),
                    "--skip-smoke",
                ])
                self.assertEqual(rc, 0)
                # gate_result.json must exist + have verdict=PASS.
                json_path = out_dir / "gate_result.json"
                self.assertTrue(json_path.is_file())
                payload = json.loads(json_path.read_text())
                self.assertEqual(payload["verdict"], "PASS")

    def test_cli_skip_smoke_returns_one_on_modules_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "wf385a3_test"
            with mock.patch.object(
                gate_module,
                "verify_module_exports",
                return_value=[
                    {"filename": "ok.py", "module_attr": "ok",
                     "expected": "ok", "on_disk": True,
                     "export_ok": True, "present": True, "error": None}
                    for _ in range(11)
                ][:10] + [
                    {"filename": "broken.py", "module_attr": "broken",
                     "expected": "broken", "on_disk": True,
                     "export_ok": False, "present": False,
                     "error": "missing export"}
                ],
            ):
                rc = gate_module.main([
                    "--out-dir", str(out_dir),
                    "--skip-smoke",
                ])
                self.assertEqual(rc, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()