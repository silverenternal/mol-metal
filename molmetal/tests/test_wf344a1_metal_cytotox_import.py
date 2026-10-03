"""Wf-344A1 — regression tests for the p1_chemprop_smoke gate import error.

==============================================================================
Wf-344A1 — pin metal_cytotox loader import path + canonical 4-column contract
==============================================================================

Wf-338-A3 finding 1: ``p1_chemprop_smoke`` pre-flight gate FAIL with
``ValueError: .../MetalCytoToxDB.csv is missing active / activity;
present columns: ['smiles', 'Counterion', ...]`` raised at
``molmetal/data/metal_cytotox.py:217``.

Root cause (MEASURED at Wf-338-A3): the canonical MetalCytoToxDB.csv
ships with the v3 schema (24 uppercase provenance columns including
``IC50_Dark_value`` but no ``active`` / ``activity`` lowercase column).
The loader's synonym map already accepts ``IC50_Dark_value`` as a
fallback for ``active`` and the derive path produces bool from
``IC50_Dark_value < 10 µM`` — so the loader now resolves correctly.
Wf-323B (SHIPPED 2026-09-29) shipped the synonym-aware + derive
fallback path that Wf-344A1 pins.

This module adds **2 NEW tests** (CPU-only, < 5 s total):

  1. ``TestMetalCytotoxImportPath::test_module_import_is_error_free`` —
     Importing :mod:`molmetal.data.metal_cytotox` and accessing
     :func:`load_metal_cytotox` MUST NOT raise.  This catches any
     future top-level ImportError / NameError / AttributeError that
     would cascade into ``train_chemprop_v2_v3 --quick`` and FAIL
     ``p1_chemprop_smoke`` (TODO-207 §4.2 P1).

  2. ``TestMetalCytotoxCanonicalContract::test_load_metal_cytotox_csv_round_trip``
     Loading the canonical :file:`MetalCytoToxDB.csv` (the file Wf-338-A3
     flagged) MUST return the 4-column contract ``[smiles, metal, pIC50,
     active]`` with non-zero rows + bool ``active`` dtype + numeric
     ``pIC50`` dtype — pinning the Wf-338-A3 regression at the loader
     boundary so it cannot silently re-FAIL the gate.

Both tests are pure CPU and self-contained (no GPU, no network, no
external state, no chemprop dependency).  Total runtime < 5 s.

Honest framing (party-twenty-spirit §3.5): the Wf-338-A3 finding is
MEASURED-fixed at the loader layer (verified by direct
``load_metal_cytotox()`` invocation returning 26801 rows today).  The
gate itself (``p1_chemprop_smoke``) is currently FAIL on a separate
``ModuleNotFoundError: No module named 'chemprop'`` (env-side, not
metal_cytotox-side) — outside Wf-344A1 scope.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _ensure_repo_on_path() -> None:
    """Pin ``PROJECT_ROOT`` on ``sys.path`` for direct pytest invocation.

    The harness path used elsewhere in the project is
    ``/home/hugo/codes/mol-metal/.venv/bin/python``; this test
    is venv-agnostic so it runs against any python with molmetal
    importable.  We insert PROJECT_ROOT first so a checkout-level
    install beats any stale system copy.
    """
    import sys

    repo = str(PROJECT_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


class TestMetalCytotoxImportPath:
    """Pin the metal_cytotox module import path is error-free.

    Regression target: Wf-338-A3 finding 1 — pre-flight
    ``p1_chemprop_smoke`` gate FAIL because the chemprop trainer's
    top-level ``from molmetal.data.metal_cytotox import load_metal_cytotox``
    raised at module import time.  The fix is to keep the module
    importable without raising.
    """

    def test_module_import_is_error_free(self) -> None:
        """Importing molmetal.data.metal_cytotox must not raise.

        This is the canonical regression target for Wf-338-A3 finding
        1: if this test FAILs, the loader has regressed and every
        downstream trainer (chemprop v2/v3, dmpnn multitask, etc.)
        that imports ``load_metal_cytotox`` will also FAIL on import.
        """
        _ensure_repo_on_path()
        # Import the module — this exercises top-level imports
        # (numpy, pandas, pathlib, typing) and any module-level
        # assignments (constants like _CANDIDATES, _COLUMN_SYNONYMS).
        module = importlib.import_module("molmetal.data.metal_cytotox")
        assert module is not None, (
            "molmetal.data.metal_cytotox imported as None — "
            "module-level code path is broken"
        )

        # Access the public API — this exercises the __all__ binding
        # and any lazy attribute resolution.
        load_fn = getattr(module, "load_metal_cytotox", None)
        assert callable(load_fn), (
            "molmetal.data.metal_cytotox.load_metal_cytotox is not "
            "callable — __all__ / public API regression"
        )

        # Re-import to confirm idempotence (no module-level state
        # mutation that would break a second import).
        module_again = importlib.import_module("molmetal.data.metal_cytotox")
        assert module_again is module, (
            "molmetal.data.metal_cytotox re-import returned a different "
            "module object — module-level state corruption"
        )


class TestMetalCytotoxCanonicalContract:
    """Pin the canonical 4-column contract on the real MetalCytoToxDB.csv.

    Regression target: Wf-338-A3 finding 1 — the loader raised
    ``ValueError: ... is missing active / activity`` when the CSV
    carries only ``IC50_Dark_value`` (no lowercase ``active`` / no
    ``activity``).  The synonym map's ``IC50_Dark_value`` fallback +
    ``_derive_active`` derive path MUST resolve to a successful load
    with the canonical 4-column contract.
    """

    def test_load_metal_cytotox_csv_round_trip(self) -> None:
        """Loading the canonical CSV must yield 4-column contract.

        This pins the Wf-338-A3 regression: the loader must accept the
        v3-style CSV (no lowercase ``active`` / ``activity`` columns,
        only ``IC50_Dark_value``) and still return the canonical
        ``[smiles, metal, pIC50, active]`` DataFrame.
        """
        from molmetal.data.metal_cytotox import load_metal_cytotox

        csv_path = PROJECT_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv"
        if not csv_path.exists():
            pytest.skip(
                f"Canonical CSV not found at {csv_path}; "
                "Wf-344A1 requires the Wf-323B canonical dataset on disk."
            )

        df = load_metal_cytotox(path=csv_path)

        # Column contract — exactly 4 canonical columns in this order.
        assert list(df.columns) == ["smiles", "metal", "pIC50", "active"], (
            f"Loader returned unexpected columns: {list(df.columns)}; "
            "canonical contract is [smiles, metal, pIC50, active]."
        )

        # Non-empty load — the canonical CSV carries 26801 rows.
        assert (
            len(df) > 0
        ), "Loader returned 0 rows; IC50_Dark_value fallback regressed."

        # dtypes per Wf-323B contract: smiles/str, metal/str, pIC50/float, active/bool.
        assert pd.api.types.is_string_dtype(
            df["smiles"]
        ), f"smiles dtype regressed: {df['smiles'].dtype}"
        assert pd.api.types.is_string_dtype(
            df["metal"]
        ), f"metal dtype regressed: {df['metal'].dtype}"
        assert pd.api.types.is_numeric_dtype(
            df["pIC50"]
        ), f"pIC50 dtype regressed: {df['pIC50'].dtype}"
        assert (
            df["active"].dtype == bool
        ), f"active dtype regressed: {df['active'].dtype} (must be bool)"

        # No NaN pIC50 — derive path drops NaN inputs already; this
        # guards against a future regression where a non-derive path
        # leaks NaN into the canonical column.
        assert (
            df["pIC50"].notna().all()
        ), f"pIC50 contains NaN values: {df['pIC50'].isna().sum()} rows"

        # The active column MUST have BOTH True and False values on
        # the canonical CSV — this proves _derive_active (10 µM
        # threshold) is actually firing.  A regressed load that
        # returns all-True / all-False indicates the threshold is
        # wrong or the derive path was bypassed.
        active_counts = df["active"].value_counts().to_dict()
        assert True in active_counts and False in active_counts, (
            f"active column is single-valued: {active_counts}; "
            "_derive_active 10µM threshold regressed."
        )
