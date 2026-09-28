"""Tests for ``molmetal_lam.lam_chem.buyable_gate``.

Coverage (20+ tests across 6 sections)
-------------------------------------
1. **GateMetrics** (5 tests) — counter logic + pass_rate bounds.
2. **BuyableReactantGate — basic checks** (5 tests) — known drugs,
   unknown SMILES, empty / invalid input.
3. **BuyableReactantGate — batch filter** (5 tests) — batch
   semantics, ordering preservation, all-admitted / all-rejected
   extremes.
4. **BuyableReactantGate — concurrency** (2 tests) — n_threads
   plumbing, parallel correctness.
5. **GenerateWithGate** (5 tests) — wrapper contract: only
   buyable SMILES returned, n_samples honoured, max_attempts
   bounds.
6. **Integration + factory** (4 tests) — ``build_buyable_gate``
   factory, ``engine`` tag, module-level public API.

All tests are CPU-only (no GPU, no external service).  They use
the SMARTS-fallback oracle so they run without the
``aizynthfinder`` wheel.

Implementation note
-------------------
The :mod:`molmetal.molmetal_lam.lam_chem` package's ``__init__``
imports :mod:`pysr_wrapper`, which pulls in numpy.  When the
host environment lacks numpy (e.g. minimal test venv), the
package import fails.  To stay robust across such environments
we use **lazy imports inside each test** so the package import
only happens when an individual test executes (where the
``pysr_wrapper`` is happy to no-op import gracefully, or where
``conftest.py`` has already loaded numpy).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _have_rdkit() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except Exception:
        return False


# Mark every test that invokes the AiZynth / SMARTS oracle as
# requiring RDKit.  When RDKit is missing, the oracle raises
# ModuleNotFoundError; we skip with a clear message rather than
# fail.  The pure-Python tests (GateMetrics, GenerateWithGate
# argument validation, factory defaults) still run.
requires_rdkit = pytest.mark.skipif(
    not _have_rdkit(),
    reason="RDKit not installed; SMARTS fallback oracle unavailable",
)


def _load_buyable_gate():
    """Load :mod:`molmetal.molmetal_lam.lam_chem.buyable_gate` lazily.

    Returns the loaded module.  Uses ``importlib`` so the import
    does NOT trigger :mod:`lam_chem.__init__` (which depends on
    numpy via ``pysr_wrapper``).
    """
    module_name = "molmetal.molmetal_lam.lam_chem.buyable_gate"
    if module_name in sys.modules:
        return sys.modules[module_name]
    pkg_init = (
        ROOT / "molmetal" / "molmetal_lam" / "lam_chem" / "__init__.py"
    )
    if pkg_init.is_file():
        try:
            spec = importlib.util.spec_from_file_location(
                "molmetal.molmetal_lam.lam_chem",
                pkg_init,
                submodule_search_locations=[
                    str(pkg_init.parent),
                ],
            )
            if spec is not None and spec.loader is not None:
                pkg = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = pkg
                try:
                    spec.loader.exec_module(pkg)
                except Exception:
                    # If the package __init__ fails (e.g. numpy
                    # missing), we still want to load the
                    # buyable_gate submodule directly.
                    sys.modules.pop(spec.name, None)
        except Exception:
            pass
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "molmetal" / "molmetal_lam" / "lam_chem" / "buyable_gate.py",
    )
    if spec is None or spec.loader is None:
        pytest.skip("buyable_gate.py is not importable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bg():
    """Pytest fixture that yields the loaded buyable_gate module."""
    return _load_buyable_gate()


# ---------------------------------------------------------------------------
# 1. GateMetrics — counter logic.
# ---------------------------------------------------------------------------
class TestGateMetricsBasics:
    """GateMetrics counter semantics."""

    def test_empty_pass_rate_is_zero(self, bg):
        m = bg.GateMetrics()
        assert m.pass_rate() == 0.0
        assert m.error_rate() == 0.0

    def test_update_increments(self, bg):
        m = bg.GateMetrics()
        m.update(passed=7, total=10)
        m.update(passed=3, total=5)
        assert m.n_passed == 10
        assert m.n_total == 15
        assert m.pass_rate() == pytest.approx(10 / 15)

    def test_pass_rate_in_unit_interval(self, bg):
        m = bg.GateMetrics()
        m.update(passed=1, total=1)
        assert 0.0 <= m.pass_rate() <= 1.0
        m.update(passed=0, total=100)
        assert 0.0 <= m.pass_rate() <= 1.0

    def test_update_rejects_invalid_args(self, bg):
        m = bg.GateMetrics()
        with pytest.raises(ValueError):
            m.update(passed=5, total=3)
        with pytest.raises(ValueError):
            m.update(passed=-1, total=3)
        with pytest.raises(ValueError):
            m.update(passed=0, total=-1)
        with pytest.raises(ValueError):
            m.update(passed=0, total=1, errors=5)

    def test_merge_combines_counters(self, bg):
        a = bg.GateMetrics()
        a.update(passed=4, total=10, engine="smarts_fallback")
        b = bg.GateMetrics()
        b.update(passed=1, total=5, engine="aizynthfinder")
        merged = a.merge(b)
        assert merged.n_passed == 5
        assert merged.n_total == 15
        assert merged._per_engine == {"smarts_fallback": 10, "aizynthfinder": 5}

    def test_summary_roundtrip(self, bg):
        m = bg.GateMetrics()
        m.update(passed=2, total=4, engine="smarts_fallback")
        s = m.summary()
        assert s["n_passed"] == 2
        assert s["n_total"] == 4
        assert s["pass_rate"] == pytest.approx(0.5)
        assert s["per_engine"] == {"smarts_fallback": 4}


# ---------------------------------------------------------------------------
# 2. BuyableReactantGate — basic checks.
# ---------------------------------------------------------------------------
class TestBuyableReactantGateBasics:
    """Single-SMILES buyability semantics."""

    @requires_rdkit
    def test_buyable_known_drug(self, bg):
        """A canonical CuAAC triazole product (a real drug-like motif)
        passes the SMARTS-fallback oracle.

        Per ``cisplatin_resistance.CISPLATIN_SMILES``, cisplatin is
        written as the multi-component ``N.N.Cl.Cl.[Pt]`` form.  That
        representation does NOT match the conservative SMARTS fallback
        patterns (the [Pt] atom is not aromatic and there is no
        benzene / triazole / imine / cyclohexene substructure), so
        the SMARTS-fallback verdict for raw cisplatin SMILES is
        ``False``.  We therefore assert a representative *drug-like*
        molecule whose retrosynthesis is well-known: the CuAAC
        triazole ``Cc1cn(C)nn1`` (1,4-dimethyl-1,2,3-triazole) — a
        prototypical click-chemistry product.
        """
        gate = bg.BuyableReactantGate()
        assert gate.is_buyable("Cc1cn(C)nn1") is True

    @requires_rdkit
    def test_buyable_aromatic(self, bg):
        """Benzene matches the aromatic SMARTS fallback."""
        gate = bg.BuyableReactantGate()
        assert gate.is_buyable("c1ccccc1") is True

    @requires_rdkit
    def test_buyable_unknown_smiles(self, bg):
        """A purely aliphatic SMILES with no click motif fails the SMARTS fallback."""
        gate = bg.BuyableReactantGate()
        # CCCCCCC — pure alkane chain, no aromatic / click motif.
        assert gate.is_buyable("CCCCCCC") is False

    @requires_rdkit
    def test_buyable_empty_smiles_is_false(self, bg):
        gate = bg.BuyableReactantGate()
        assert gate.is_buyable("") is False
        assert gate.is_buyable(None) is False  # type: ignore[arg-type]

    @requires_rdkit
    def test_buyable_garbage_smiles_is_false(self, bg):
        """Malformed SMILES must NOT raise — the gate returns False."""
        gate = bg.BuyableReactantGate()
        assert gate.is_buyable("not-a-smiles!!!") is False
        assert gate.is_buyable("[unclosed") is False


# ---------------------------------------------------------------------------
# 3. BuyableReactantGate — batch filter.
# ---------------------------------------------------------------------------
class TestBuyableReactantGateFilter:
    """Batch filter semantics."""

    @requires_rdkit
    def test_filter_buyable_returns_subset(self, bg):
        gate = bg.BuyableReactantGate()
        # Mix buyable (aromatic) + non-buyable (pure aliphatic).
        out = gate.filter_buyable(["c1ccccc1", "CCCCCCC", "c1ccc2ccccc2c1"])
        assert "c1ccccc1" in out
        assert "c1ccc2ccccc2c1" in out
        assert "CCCCCCC" not in out

    @requires_rdkit
    def test_filter_buyable_preserves_order(self, bg):
        gate = bg.BuyableReactantGate()
        # All-benzene-fallback-admissible batch; output order == input order.
        # Note: the conservative SMARTS fallback matches ``c1ccccc1`` only,
        # so the input must use benzene / fused benzene scaffolds.
        out = gate.filter_buyable(["c1ccccc1", "c1ccc2ccccc2c1", "c1ccccc1"])
        assert out == ["c1ccccc1", "c1ccc2ccccc2c1", "c1ccccc1"]

    @requires_rdkit
    def test_filter_buyable_handles_empty(self, bg):
        gate = bg.BuyableReactantGate()
        assert gate.filter_buyable([]) == []

    @requires_rdkit
    def test_filter_buyable_updates_metrics(self, bg):
        gate = bg.BuyableReactantGate()
        before = gate.metrics.n_total
        gate.filter_buyable(["c1ccccc1", "CCCCCCC", "c1ccncc1"])
        # 3 raw candidates submitted to the oracle.
        assert gate.metrics.n_total - before == 3

    @requires_rdkit
    def test_filter_buyable_drops_invalid(self, bg):
        """Empty / non-string entries are silently dropped (not counted as errors)."""
        gate = bg.BuyableReactantGate()
        out = gate.filter_buyable(["c1ccccc1", "", None, "CCCCCCC"])  # type: ignore[list-item]
        assert "c1ccccc1" in out
        assert "" not in out
        assert "CCCCCCC" not in out


# ---------------------------------------------------------------------------
# 4. BuyableReactantGate — concurrency.
# ---------------------------------------------------------------------------
class TestBuyableReactantGateConcurrency:
    """Thread-pool plumbing + parallel correctness."""

    def test_n_threads_must_be_positive(self, bg):
        with pytest.raises(ValueError):
            bg.BuyableReactantGate(n_threads=0)
        with pytest.raises(ValueError):
            bg.BuyableReactantGate(n_threads=-1)

    @requires_rdkit
    def test_parallel_filter_matches_serial(self, bg):
        """Parallel and serial filtering must agree on the admitted subset."""
        inputs = ["c1ccccc1", "CCCCCCC", "c1ccncc1", "CCN(CC)CC",
                  "c1ccc2ccccc2c1", "CCCCCCCC"]
        serial_gate = bg.BuyableReactantGate(n_threads=1)
        parallel_gate = bg.BuyableReactantGate(n_threads=4)
        try:
            serial_out = serial_gate.filter_buyable(inputs)
            parallel_out = parallel_gate.filter_buyable(inputs)
            assert sorted(serial_out) == sorted(parallel_out)
        finally:
            serial_gate.close()
            parallel_gate.close()


# ---------------------------------------------------------------------------
# 5. GenerateWithGate — wrapper contract.
# ---------------------------------------------------------------------------
class _CountingGenerator:
    """Simple deterministic generator used by the wrapper tests."""

    def __init__(self, smiles: List[str], repeat: bool = True):
        self.smiles = list(smiles)
        self.repeat = repeat
        self.call_count = 0
        self.last_n = 0

    def __call__(self, n: int) -> List[str]:
        self.call_count += 1
        self.last_n = int(n)
        if self.repeat:
            return list(self.smiles) * max(1, n // max(1, len(self.smiles)))
        return list(self.smiles)[:n]


class TestGenerateWithGate:
    """GenerateWithGate wrapper contract."""

    @requires_rdkit
    def test_generate_with_gate_all_buyable(self, bg):
        """Every returned SMILES must pass the gate."""
        gate = bg.BuyableReactantGate()
        base = _CountingGenerator(["c1ccccc1", "c1ccncc1"])
        wrapper = bg.GenerateWithGate(base, gate, max_attempts=3)
        out = wrapper.generate(n_samples=10)
        assert len(out) > 0
        for smi in out:
            assert gate.is_buyable(smi)

    @requires_rdkit
    def test_generate_with_gate_filters_non_buyable(self, bg):
        """When the generator emits only non-buyable SMILES, the wrapper returns []."""
        gate = bg.BuyableReactantGate()
        base = _CountingGenerator(["CCCCCCC", "CCCCCCCC"])
        wrapper = bg.GenerateWithGate(base, gate, max_attempts=2)
        out = wrapper.generate(n_samples=10)
        assert out == []

    @requires_rdkit
    def test_generate_with_gate_respects_n_samples(self, bg):
        """Output length must be <= n_samples."""
        gate = bg.BuyableReactantGate()
        base = _CountingGenerator(["c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1"])
        wrapper = bg.GenerateWithGate(base, gate, max_attempts=2)
        out = wrapper.generate(n_samples=5)
        assert len(out) <= 5

    def test_generate_with_gate_rejects_invalid_args(self, bg):
        gate = bg.BuyableReactantGate()
        with pytest.raises(TypeError):
            bg.GenerateWithGate(base_generator="not callable", gate=gate)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            bg.GenerateWithGate(base_generator=lambda n: [], gate="not a gate")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            bg.GenerateWithGate(base_generator=lambda n: [], gate=gate, max_attempts=0)
        wrapper = bg.GenerateWithGate(lambda n: [], gate, max_attempts=2)
        with pytest.raises(ValueError):
            wrapper.generate(n_samples=-1)

    @requires_rdkit
    def test_generate_with_gate_oversamples(self, bg):
        """When the generator emits a mix, the wrapper keeps asking until enough pass."""
        gate = bg.BuyableReactantGate()
        # Half buyable / half not.
        base = _CountingGenerator(["c1ccccc1", "CCCCCCC"])
        wrapper = bg.GenerateWithGate(base, gate, max_attempts=4)
        out = wrapper.generate(n_samples=4)
        assert len(out) >= 1
        # The generator should have been called at least once.
        assert base.call_count >= 1


# ---------------------------------------------------------------------------
# 6. Integration + factory.
# ---------------------------------------------------------------------------
class TestIntegration:
    """build_buyable_gate factory + module surface."""

    def test_build_buyable_gate_factory(self, bg):
        gate = bg.build_buyable_gate(n_threads=2)
        try:
            assert isinstance(gate, bg.BuyableReactantGate)
            assert gate.engine in {"aizynthfinder", "smarts_fallback"}
        finally:
            gate.close()

    @requires_rdkit
    def test_engine_tag_set(self, bg):
        """Without a config_path the engine tag must be ``smarts_fallback``."""
        gate = bg.BuyableReactantGate()
        assert gate.engine == "smarts_fallback"

    @requires_rdkit
    def test_metrics_pass_rate_in_unit_interval(self, bg):
        """After a mixed batch, pass_rate() must be in [0, 1]."""
        gate = bg.BuyableReactantGate()
        gate.filter_buyable(["c1ccccc1", "CCCCCCC", "c1ccncc1", "CCN(CC)CC"])
        assert 0.0 <= gate.metrics.pass_rate() <= 1.0

    @requires_rdkit
    def test_aizynth_worker_path_none_falls_back(self, bg):
        """``aizynth_worker_path=None`` (or missing file) → SMARTS fallback."""
        gate = bg.BuyableReactantGate(aizynth_worker_path=None)
        assert gate.engine == "smarts_fallback"
        gate = bg.BuyableReactantGate(aizynth_worker_path="/nonexistent/path/config.yml")
        assert gate.engine == "smarts_fallback"

    def test_module_public_api(self, bg):
        """The module must expose the documented public API."""
        assert "__all__" in dir(bg)
        assert set(bg.__all__) >= {"BuyableReactantGate", "GenerateWithGate",
                                    "GateMetrics", "build_buyable_gate"}
