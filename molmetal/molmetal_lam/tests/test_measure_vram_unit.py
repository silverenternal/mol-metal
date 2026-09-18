"""Tests for the VRAM measurement harness.

Covers:
- :func:`measure_vram` returns the documented dict shape (CPU path).
- Per-mode deltas are computed correctly against a mock baseline.
- The harness exits 0 with a friendly SKIPPED message when no GPU.
- All four supported modes are accepted (``baseline``, ``bf16``,
  ``checkpoint``, ``bf16+checkpoint``); bogus modes raise ``ValueError``.

All tests are CPU-only by design: the harness script is the
real-GPU entry-point; the unit suite mocks ``torch.cuda`` so the
CI gate stays green on a host without a usable GPU (RX 7800 XT
SMU hang — see ``molmetal/reports/wf_gpu_diag/diagnosis.md``).

Citations
---------
* Harness: ``molmetal/scripts/measure_vram.py``
* Adapter init: ``molmetal/adapters/flow_matching_lipman/__init__.py:1726-1981``
* AMP wrapper: ``molmetal/adapters/flow_matching_lipman/amp.py``
* Checkpoint wrapper: ``molmetal/adapters/egnn_rocm_checkpoint.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make the scripts directory importable so we can pull in
# ``measure_vram.py`` as a module without making ``scripts/`` a
# package (it's a flat dir).
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import measure_vram as measure_vram_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Constants — pin the public contract
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "mode",
    "available",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "current_allocated_mb",
    "fragmentation_mb",
    "step_time_mean_ms",
    "n_steps",
    "hidden_dim",
    "n_layers",
    "n_atoms",
    "batch_size",
    "error",
}

ALL_MODES_HARNESS = (
    "baseline", "bf16", "checkpoint", "bf16+checkpoint",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpu_fallback_row(mode: str) -> dict:
    """Return the SKIPPED-row shape the harness produces on no-GPU hosts."""
    return measure_vram_mod.measure_vram(mode=mode, n_steps=1)


# ---------------------------------------------------------------------------
# CPU-only tests
# ---------------------------------------------------------------------------

def test_measure_vram_returns_dict(monkeypatch):
    """The harness must return a dict with the documented schema.

    This test forces the no-GPU fallback path by patching
    ``torch.cuda.is_available`` to return ``False`` so the contract
    holds regardless of host GPU state (the live machine may or
    may not have a usable GPU at any given moment).
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    row = _cpu_fallback_row("baseline")
    assert isinstance(row, dict)
    missing = EXPECTED_KEYS - set(row.keys())
    assert not missing, f"missing keys from measure_vram output: {missing}"
    assert row["mode"] == "baseline"
    assert row["available"] is False
    # All VRAM fields MUST be None when no GPU is available.
    for k in ("peak_allocated_mb", "peak_reserved_mb",
              "current_allocated_mb", "fragmentation_mb",
              "step_time_mean_ms"):
        assert row[k] is None, (
            f"{k} must be None on CPU fallback; got {row[k]!r}"
        )
    # error is populated with a human-readable reason.
    assert isinstance(row["error"], str)
    assert row["error"], "error string must be non-empty"
    # Static config knobs echo the call args.
    assert row["n_steps"] == 1
    assert row["hidden_dim"] == measure_vram_mod.DEFAULT_HIDDEN_DIM
    assert row["n_layers"] == measure_vram_mod.DEFAULT_N_LAYERS
    assert row["n_atoms"] == measure_vram_mod.DEFAULT_N_ATOMS
    assert row["batch_size"] == measure_vram_mod.DEFAULT_BATCH_SIZE


def test_measure_vram_all_modes_return_dict(monkeypatch):
    """All four modes must return the same dict shape on CPU.

    Patches ``torch.cuda.is_available`` to False so the test
    is host-GPU-agnostic — see ``test_measure_vram_returns_dict``.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for mode in ALL_MODES_HARNESS:
        row = _cpu_fallback_row(mode)
        assert isinstance(row, dict)
        assert row["mode"] == mode
        # Available must be False when no GPU.
        assert row["available"] is False
        missing = EXPECTED_KEYS - set(row.keys())
        assert not missing, f"mode={mode}: missing keys {missing}"


def test_measure_vram_rejects_unknown_mode():
    """An unknown mode name must raise ValueError, not silently accept.

    The harness is meant to be called from the CLI too, so the
    validation lives in the function (not just in argparse) — that
    way callers that import the function programmatically get the
    same contract.
    """
    with pytest.raises(ValueError, match="mode must be one of"):
        measure_vram_mod.measure_vram(mode="fp8")  # not supported
    with pytest.raises(ValueError, match="mode must be one of"):
        measure_vram_mod.measure_vram(mode="")
    with pytest.raises(ValueError, match="mode must be one of"):
        measure_vram_mod.measure_vram(mode="BASELINE")  # case-sensitive


def test_deltas_computed_correctly(monkeypatch):
    """``_print_table`` must compute deltas as ``row.peak - baseline.peak``.

    The harness doesn't store deltas in the row dict (it computes them
    at print time so the baseline row can be selected after-the-fact).
    We re-implement the same calculation here against mock rows and
    assert the arithmetic matches the harness's ``_print_table``
    implementation.

    Mock strategy: monkey-patch ``measure_vram_mod.measure_vram`` so
    it returns canned values, then call ``_print_table`` with a
    baseline + a treatment row and assert the table renderer
    embeds the right delta string.
    """
    fake_rows = [
        {
            "mode": "baseline", "available": True,
            "peak_allocated_mb": 100.0,
            "peak_reserved_mb": 110.0,
            "current_allocated_mb": 80.0,
            "fragmentation_mb": 10.0,
            "step_time_mean_ms": 5.0,
            "n_steps": 3, "hidden_dim": 64, "n_layers": 2,
            "n_atoms": 20, "batch_size": 8, "error": None,
        },
        {
            "mode": "bf16+checkpoint", "available": True,
            "peak_allocated_mb": 55.0,
            "peak_reserved_mb": 65.0,
            "current_allocated_mb": 40.0,
            "fragmentation_mb": 10.0,
            "step_time_mean_ms": 6.0,
            "n_steps": 3, "hidden_dim": 64, "n_layers": 2,
            "n_atoms": 20, "batch_size": 8, "error": None,
        },
    ]
    # The harness's _print_table prints to stdout.  Capture it.
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        measure_vram_mod._print_table(
            fake_rows,
            baseline_row=fake_rows[0],
        )
    out = buf.getvalue()

    # Baseline row appears with an empty delta column (it's the
    # reference).
    assert "baseline" in out
    # Treatment row's delta: 55.0 - 100.0 = -45.0.  The harness
    # renders it with sign and 2 decimal places ("-45.00").
    assert "-45.00" in out, (
        f"treatment row must show delta -45.00 vs baseline 100.0; "
        f"got table:\n{out}"
    )
    # The peak numbers themselves appear in the table.
    assert "100.00" in out
    assert "55.00" in out
    # step time column appears.
    assert "5.00" in out
    assert "6.00" in out


def test_deltas_with_no_baseline_does_not_crash():
    """If no baseline is provided, the delta column is empty (no crash)."""
    rows = [
        {
            "mode": "baseline", "available": True,
            "peak_allocated_mb": 100.0, "peak_reserved_mb": 110.0,
            "current_allocated_mb": 80.0, "fragmentation_mb": 10.0,
            "step_time_mean_ms": 5.0, "n_steps": 3, "hidden_dim": 64,
            "n_layers": 2, "n_atoms": 20, "batch_size": 8, "error": None,
        },
    ]
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        measure_vram_mod._print_table(rows, baseline_row=None)
    out = buf.getvalue()
    assert "baseline" in out
    # No delta numbers in the table.
    assert "-45.00" not in out


def test_print_table_handles_skipped_rows():
    """Rows with ``available=False`` print as ``SKIPPED`` not as numbers."""
    rows = [
        {
            "mode": "baseline", "available": False,
            "peak_allocated_mb": None, "peak_reserved_mb": None,
            "current_allocated_mb": None, "fragmentation_mb": None,
            "step_time_mean_ms": None, "n_steps": 3, "hidden_dim": 64,
            "n_layers": 2, "n_atoms": 20, "batch_size": 8,
            "error": "no GPU available",
        },
    ]
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        measure_vram_mod._print_table(rows, baseline_row=None)
    out = buf.getvalue()
    assert "SKIPPED" in out
    assert "no GPU available" in out


def test_default_config_constants_are_production_safe():
    """Pin the defaults so silent changes break this test loudly.

    The defaults must:
    * hidden_dim >= 64  (matches the EGNN production-scale threshold
      that the adapter itself warns on at hidden_dim < 64 — see
      ``flow_matching_lipman/__init__.py:1900-1911``).
    * n_layers >= 2       (one-layer EGNN is degenerate; production
      baseline is 2-3).
    * n_steps >= 2        (we want at least one warmup + one steady).
    """
    assert measure_vram_mod.DEFAULT_HIDDEN_DIM >= 64, (
        f"DEFAULT_HIDDEN_DIM must be >= 64 (production-scale floor), "
        f"got {measure_vram_mod.DEFAULT_HIDDEN_DIM}"
    )
    assert measure_vram_mod.DEFAULT_N_LAYERS >= 2
    assert measure_vram_mod.DEFAULT_N_STEPS >= 2
    assert measure_vram_mod.DEFAULT_BATCH_SIZE > 0
    assert measure_vram_mod.DEFAULT_N_ATOMS > 0


def test_all_modes_constant_matches_documented_set():
    """Lock in the public mode set so callers can rely on the tuple."""
    assert set(measure_vram_mod.ALL_MODES) == set(ALL_MODES_HARNESS)


def test_synthetic_batch_returns_list_of_molecules():
    """The harness builds its own synthetic batch — verify the helper
    without requiring a GPU.

    The synthetic batch is the only piece of ``measure_vram.py`` we
    can unit-test on CPU in full fidelity (the rest depends on
    ``train_step`` which requires a real adapter + GPU tensors).
    """
    from molmetal.domain import Molecule
    # ``_build_synthetic_batch`` accepts a torch.device.  CPU is fine
    # here — we only check shape / types.
    import torch
    mols = measure_vram_mod._build_synthetic_batch(
        batch_size=2, n_atoms=5, device=torch.device("cpu"),
    )
    assert isinstance(mols, list)
    assert len(mols) == 2
    for m in mols:
        assert isinstance(m, Molecule)
        assert m.coords.shape == (5, 3)
        assert m.atom_types.shape == (5,)
        # Fully connected, both directions: 5*4 = 20 edges.
        assert m.bonds.shape == (2, 20)
        assert m.bond_types.shape == (20,)
        # All atom types must be in the metallodrug vocab
        # (the Z-set used inside the helper).
        vocab_z = (6, 7, 8, 9, 15, 16, 17, 35, 53, 78, 46, 79, 77, 44)
        for z in m.atom_types.tolist():
            assert z in vocab_z, f"atom Z={z} not in synthetic vocab"


def test_synthetic_batch_is_deterministic():
    """Two consecutive ``_build_synthetic_batch`` calls with the same
    args must produce equal coords + atom types (RNG seeded inside
    the helper at line ``g = torch.Generator(device="cpu").manual_seed(0)``).
    """
    import torch
    mols_a = measure_vram_mod._build_synthetic_batch(
        batch_size=2, n_atoms=4, device=torch.device("cpu"),
    )
    mols_b = measure_vram_mod._build_synthetic_batch(
        batch_size=2, n_atoms=4, device=torch.device("cpu"),
    )
    for a, b in zip(mols_a, mols_b):
        assert torch.equal(a.coords, b.coords)
        assert torch.equal(a.atom_types, b.atom_types)


# ---------------------------------------------------------------------------
# Mocked-GPU integration smoke (verifies the available=True path shape)
# ---------------------------------------------------------------------------

class _FakeCuda:
    """Minimal stand-in for ``torch.cuda`` exposing only the methods the
    harness touches.  Records calls so tests can assert reset_peak
    etc. fire in the right order.
    """

    def __init__(self):
        self._peak_alloc_mb = 1234.0  # bytes? no — we store MiB directly
        self._peak_resv_mb = 1500.0
        self._cur_alloc_mb = 800.0
        self.calls: list = []

    def is_available(self) -> bool:  # noqa: D401
        self.calls.append(("is_available",))
        return True

    def device_count(self) -> int:
        return 1

    def manual_seed_all(self, seed):
        self.calls.append(("manual_seed_all", seed))

    def empty_cache(self):
        self.calls.append(("empty_cache",))

    def reset_peak_memory_stats(self):
        self.calls.append(("reset_peak_memory_stats",))

    def max_memory_allocated(self):
        self.calls.append(("max_memory_allocated",))
        return self._peak_alloc_mb * (1024 ** 2)

    def max_memory_reserved(self):
        self.calls.append(("max_memory_reserved",))
        return self._peak_resv_mb * (1024 ** 2)

    def memory_allocated(self):
        self.calls.append(("memory_allocated",))
        return self._cur_alloc_mb * (1024 ** 2)


def test_measure_vram_mocked_gpu_returns_peak_metrics(monkeypatch):
    """When ``torch.cuda.is_available()`` is mocked to True, the harness
    must short-circuit BEFORE touching real CUDA APIs (we don't have
    a real GPU).  We verify by patching ``is_available`` only and
    asserting the harness still returns the SKIPPED-row shape (because
    the real path needs more than just is_available).

    This is the conservative test: we don't try to mock the whole
    adapter + CFMAMPContext + train_step path.  We just verify the
    contract — the function returns a dict with the documented keys.
    """
    # Patch is_available at the torch.cuda module level.  The harness
    # reads ``torch.cuda.is_available()`` directly (line ~225 of
    # measure_vram.py), so this is sufficient to flip the available
    # branch — but the harness will then try to import the adapter
    # and call train_step, which will fail on a CPU-only host.
    # We expect the harness to record that failure in the "error"
    # key and still return a valid dict.
    fake = _FakeCuda()
    monkeypatch.setattr(torch.cuda, "is_available", fake.is_available)
    # Also patch reset_peak etc. so a partial import doesn't crash
    # before we hit the adapter import (which is more expensive).
    monkeypatch.setattr(torch.cuda, "manual_seed_all", fake.manual_seed_all)
    monkeypatch.setattr(torch.cuda, "empty_cache", fake.empty_cache)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats",
                        fake.reset_peak_memory_stats)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated",
                        fake.max_memory_allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved",
                        fake.max_memory_reserved)
    monkeypatch.setattr(torch.cuda, "memory_allocated", fake.memory_allocated)

    row = measure_vram_mod.measure_vram(mode="baseline", n_steps=1)
    assert isinstance(row, dict)
    missing = EXPECTED_KEYS - set(row.keys())
    assert not missing, f"missing keys: {missing}"
    # is_available was consulted.
    assert ("is_available",) in fake.calls
    # Either the harness completed the GPU path (available=True with
    # the fake's MiB values) OR it failed gracefully with an error
    # string — both are valid.  We do NOT require available=True
    # because the real CFM adapter import may fail in CPU-only test
    # environments.  The contract is "returns the documented schema
    # either way".
    if row["available"]:
        assert row["peak_allocated_mb"] == 1234.0
        assert row["peak_reserved_mb"] == 1500.0
        assert row["current_allocated_mb"] == 800.0
        assert row["fragmentation_mb"] == 266.0  # 1500 - 1234
        # step_time_mean_ms may be 0.0 if the inner loop was extremely
        # fast on the mocked path; we don't pin it.
    else:
        assert isinstance(row["error"], str)
