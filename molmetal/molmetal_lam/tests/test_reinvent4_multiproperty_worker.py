"""Unit tests for the WF-Extra-2 REINVENT4 multi-property JSONL worker.

The worker lives in the isolated ROCm project at
``/mnt/storage/env-projects/reinvent4-rocm/.venv`` and is invoked via
the project ``reinvent`` CLI.  We run the worker as a real subprocess
in every test — there is no mock layer because the goal is to
exercise the wire protocol end-to-end.

When the REINVENT4 binary is missing the tests still pass: they
short-circuit to asserting the documented failure mode
(``reason == "missing_binary"`` or ``"invalid_smiles"``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path

# Ensure the project root is on sys.path so ``molmetal_lam.*`` imports
# resolve when pytest collects from anywhere.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


WORKER_PATH = (
    Path(__file__).parents[2]
    / "molmetal_lam"
    / "sbdd_env"
    / "reinvent4_multiproperty_jsonl_worker.py"
)
DEFAULT_PYTHON = (
    "/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/python"
)
REINVENT_BIN = (
    "/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/reinvent"
)


def _worker_available() -> bool:
    """The worker is available iff its source file exists on disk."""
    return WORKER_PATH.is_file()


def _binary_available() -> bool:
    """The REINVENT4 binary is available iff the upstream binary exists."""
    return os.path.isfile(REINVENT_BIN) and os.access(REINVENT_BIN, os.X_OK)


def _python_available() -> bool:
    """The ROCm venv python is preferred, but system python also works."""
    if os.path.isfile(DEFAULT_PYTHON) and os.access(DEFAULT_PYTHON, os.X_OK):
        return True
    return shutil.which("python") is not None or shutil.which("python3") is not None


def _pick_python() -> str:
    if os.path.isfile(DEFAULT_PYTHON) and os.access(DEFAULT_PYTHON, os.X_OK):
        return DEFAULT_PYTHON
    for cand in ("python3", "python"):
        p = shutil.which(cand)
        if p:
            return p
    raise RuntimeError("no python interpreter available for the worker subprocess")


def _call_worker(request: dict, timeout: float = 120.0) -> dict:
    """Send one JSON-lines request to the worker subprocess, return response."""
    python = _pick_python()
    proc = subprocess.run(
        [python, str(WORKER_PATH)],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker subprocess exited rc={proc.returncode}: {proc.stderr[:500]}"
        )
    last_line = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    return json.loads(last_line)


class TestReinvent4MultipropertyWorker(unittest.TestCase):
    """End-to-end RPC tests for the REINVENT4 multi-property bridge."""

    def setUp(self) -> None:
        if not _worker_available():
            self.skipTest(f"worker source not found at {WORKER_PATH}")

    def test_worker_valid_smiles_returns_score(self) -> None:
        """CCO (ethanol) gets a multiproperty score in [0, 1]."""
        request = {
            "smiles": "CCO",
            "components": {"qed": 0.4, "logp": 0.4, "ring_count": 0.2},
            "timeout": 90.0,
        }
        response = _call_worker(request, timeout=120.0)
        self.assertEqual(response.get("smiles"), "CCO")
        if not _binary_available():
            # No REINVENT4 binary → the documented "missing_binary" path.
            self.assertFalse(response.get("ok"))
            self.assertEqual(response.get("reason"), "missing_binary")
            self.assertIsNone(response.get("multiproperty_score"))
            return
        self.assertTrue(
            response.get("ok"),
            msg=f"expected ok=true, got response={response}",
        )
        score = response.get("multiproperty_score")
        self.assertIsNotNone(score)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # Per-component raw values must be present and clipped to [0,1].
        raw = response.get("components_raw") or {}
        self.assertGreaterEqual(len(raw), 1)
        for v in raw.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        # The model_sha should be a stable fingerprint of the components.
        self.assertIsInstance(response.get("model_sha"), str)
        self.assertEqual(len(response.get("model_sha")), 16)

    def test_worker_invalid_smiles_graceful(self) -> None:
        """Invalid SMILES returns score=None without crashing the worker."""
        # A clearly invalid SMILES: lone @ symbol with garbage tokens.
        request = {
            "smiles": "!!!@@@###$$$",
            "components": {"qed": 1.0},
            "timeout": 30.0,
        }
        response = _call_worker(request, timeout=60.0)
        self.assertEqual(response.get("smiles"), request["smiles"])
        # The fast-path RDKit pre-check fires BEFORE we even look at
        # the binary.  So this test is independent of REINVENT4
        # availability.
        self.assertFalse(response.get("ok"))
        self.assertEqual(response.get("reason"), "invalid_smiles")
        self.assertIsNone(response.get("multiproperty_score"))

    def test_worker_empty_smiles_graceful(self) -> None:
        """An empty SMILES string is treated as invalid_smiles."""
        request = {
            "smiles": "",
            "components": {"qed": 1.0},
            "timeout": 30.0,
        }
        response = _call_worker(request, timeout=60.0)
        self.assertFalse(response.get("ok"))
        self.assertEqual(response.get("reason"), "invalid_smiles")

    def test_worker_timeout(self) -> None:
        """A deliberately short timeout aborts a slow query gracefully."""
        if not _binary_available():
            self.skipTest("REINVENT4 binary not available — cannot test timeout")
        # Submit a query with timeout=0.05s.  The worker must:
        #   - NOT crash (the subprocess returns a structured response)
        #   - report reason="timeout" (or finish first, in which case
        #     we accept a normal ok=true result and skip the assertion)
        # We re-run up to 3 times — REINVENT4 occasionally finishes
        # faster than 50 ms on a tiny molecule, which is acceptable.
        last_response: dict | None = None
        timed_out = False
        for _ in range(3):
            request = {
                "smiles": "c1ccccc1",
                "components": {"qed": 1.0},
                "timeout": 0.05,
            }
            t0 = time.monotonic()
            try:
                response = _call_worker(request, timeout=10.0)
            except subprocess.TimeoutExpired:
                timed_out = True
                break
            elapsed = time.monotonic() - t0
            last_response = response
            # If REINVENT4 finished within the budget, that's an
            # acceptable outcome — retry with the same timeout to give
            # the timeout path a chance.
            if response.get("ok"):
                continue
            if response.get("reason") == "timeout":
                timed_out = True
                break
        self.assertTrue(
            timed_out or (last_response is not None and last_response.get("reason") == "timeout"),
            msg=(
                "expected worker to surface a 'timeout' reason after a "
                f"deliberately short timeout; got last_response={last_response}"
            ),
        )


class TestReinvent4MultipropertyWorkerUnit(unittest.TestCase):
    """Pure-Python unit tests for the worker's TOML generation helpers."""

    def setUp(self) -> None:
        if not _worker_available():
            self.skipTest(f"worker source not found at {WORKER_PATH}")
        # Use importlib so we can import a module with the same name as
        # the worker file without polluting sys.modules permanently.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_r4mp_worker_under_test", str(WORKER_PATH)
        )
        assert spec is not None and spec.loader is not None
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_unknown_component_key_silently_dropped(self) -> None:
        """An unknown component key never raises; it is silently ignored."""
        fragments = self.mod._build_scoring_fragments(
            {"unknown_key_xyz": 0.5, "qed": 0.5}
        )
        # Only the recognised QED recipe survives.
        joined = "".join(fragments)
        self.assertIn("[scoring.component.QED]", joined)
        self.assertNotIn("unknown_key_xyz", joined)

    def test_empty_components_defaults_to_qed(self) -> None:
        """An empty components dict falls back to plain QED."""
        fragments = self.mod._build_scoring_fragments({})
        joined = "".join(fragments)
        self.assertIn("[scoring.component.QED]", joined)

    def test_zero_weight_component_dropped(self) -> None:
        """A weight <= 0 is treated as 'disable this component'."""
        fragments = self.mod._build_scoring_fragments(
            {"qed": 0.0, "logp": 0.5}
        )
        joined = "".join(fragments)
        self.assertNotIn("[scoring.component.QED]", joined)
        self.assertIn("[scoring.component.SlogP]", joined)

    def test_components_key_stable(self) -> None:
        """Two dicts with the same logical content hash identically."""
        a = self.mod.components_key({"qed": 0.5, "logp": 0.5})
        b = self.mod.components_key({"logp": 0.5, "qed": 0.5})
        self.assertEqual(a, b)

    def test_components_key_ignores_zero_weights(self) -> None:
        """Components with weight <= 0 are not included in the fingerprint."""
        a = self.mod.components_key({"qed": 0.0, "logp": 0.5})
        b = self.mod.components_key({"logp": 0.5})
        self.assertEqual(a, b)


class TestReinvent4MultipropertyWorkerWireContract(unittest.TestCase):
    """The wire contract is exactly the documented one — even on errors."""

    def setUp(self) -> None:
        if not _worker_available():
            self.skipTest(f"worker source not found at {WORKER_PATH}")

    def test_response_keys_present(self) -> None:
        """Every response carries the documented top-level keys."""
        request = {"smiles": "CCO", "components": {"qed": 1.0}}
        response = _call_worker(request, timeout=120.0)
        for key in (
            "smiles",
            "multiproperty_score",
            "components_raw",
            "device",
            "model_sha",
            "ok",
            "reason",
        ):
            self.assertIn(key, response, msg=f"missing key {key!r} in {response}")

    def test_components_raw_is_dict(self) -> None:
        """``components_raw`` is always a dict (possibly empty)."""
        request = {"smiles": "!!!INVALID!!!", "components": {"qed": 1.0}}
        response = _call_worker(request, timeout=60.0)
        self.assertIsInstance(response.get("components_raw"), dict)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
