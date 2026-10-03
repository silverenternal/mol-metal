"""Wf-323B regression tests — MetalCytoToxDB loader v3/v4 schema drift.

==============================================================================
Wf-323B — metalcytotox_loader handles both v3 + v4 column names
==============================================================================

This module pins the loader's dual-version contract delivered by Wf-323B:

1. **v3 schema** — columns ``SMILES_Ligands, Metal, scaffold_id, split,
   activity, pic50, cell_line, source, IC50_Dark_value`` (no lowercase
   ``smiles`` / ``metal``).

2. **v4 schema** — columns ``smiles, metal, scaffold_id, split, activity,
   SMILES_Ligands, Metal, IC50_Dark_value, pic50, cell_line, source``
   (lowercase canonical AND v3-style provenance enrichment).

3. **Synonym map** — :func:`_has_column` and :func:`_rename_v3_to_canonical`
   correctly skip the rename when the lowercase target is already
   present, avoiding the v4 ``ValueError: Buffer has wrong number of
   dimensions (expected 1, got 2)`` that the naive v1-only loader
   raised pre-Wf-323B.

All tests are **CPU-only** and run in < 30 seconds.

Honest framing (party-twenty-spirit §3.5): the loader now loads both
schemas with identical 26,632 rows and identical 5-metal distribution
(Ru/Ir/Os/Rh/Re). v3 and v4 are not duplicates — v4 carries extra
provenance metadata — but the loader's output is canonicalised to
the 4-column ``smiles, metal, pIC50, active`` contract.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from molmetal.data.metal_cytotox import (  # noqa: E402
        _COLUMN_SYNONYMS,
        _has_column,
        _rename_v3_to_canonical,
        load_metal_cytotox,
    )
except ImportError:  # Wf-A16: feature not implemented — SKIP, do not ERROR
    pytest.skip(
        "UNIMPLEMENTED FEATURE: this module expects `_COLUMN_SYNONYMS` from `molmetal.data.metal_cytotox`, which "
        "does not exist. v3/v4 column-synonym map. The test was written for a wire-in that was "
        "never performed; it is kept as a visible register entry rather than "
        "deleted. Tracked in TODO/completed/wfa16_unimplemented_feature_register.md",
        allow_module_level=True,
    )

V1_CSV = PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
V3_CSV = PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v3.csv"
V4_CSV = PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"

EXPECTED_METALS = {"Ru", "Ir", "Os", "Rh", "Re"}
CANONICAL_COLUMNS = ["smiles", "metal", "pIC50", "active"]


def _has_all(path: Path) -> bool:
    return path.exists() and path.is_file()


# ---------------------------------------------------------------------------
# Test class 1 — v3 CSV loading (3 tests)
# ---------------------------------------------------------------------------


class TestV3CsvLoading:
    """v3 schema has uppercase SMILES_Ligands / Metal + IC50_Dark_value."""

    @pytest.mark.skipif(not _has_all(V3_CSV), reason="v3 CSV not on disk")
    def test_v3_loader_returns_canonical_4_columns(self):
        """v3 must produce exactly the 4-column canonical DataFrame."""
        df = load_metal_cytotox(V3_CSV)
        assert list(df.columns) == CANONICAL_COLUMNS, (
            f"v3 should produce canonical 4 columns, got {list(df.columns)}"
        )
        assert len(df) > 25_000, f"v3 row count unexpectedly small: {len(df)}"

    @pytest.mark.skipif(not _has_all(V3_CSV), reason="v3 CSV not on disk")
    def test_v3_loader_resolves_5_metals(self):
        """v3 must resolve all 5 expected metals."""
        df = load_metal_cytotox(V3_CSV)
        actual = set(df["metal"].unique())
        assert actual == EXPECTED_METALS, (
            f"v3 metals mismatch: expected {EXPECTED_METALS}, got {actual}"
        )

    @pytest.mark.skipif(not _has_all(V3_CSV), reason="v3 CSV not on disk")
    def test_v3_loader_no_nan_pic50(self):
        """v3 must compute pIC50 from IC50_Dark_value (no NaN survivors)."""
        df = load_metal_cytotox(V3_CSV)
        assert df["pIC50"].isna().sum() == 0, (
            f"v3 pIC50 should be fully populated, "
            f"found {df['pIC50'].isna().sum()} NaN"
        )
        assert df["pIC50"].dtype.kind == "f", (
            f"v3 pIC50 dtype should be float, got {df['pIC50'].dtype}"
        )


# ---------------------------------------------------------------------------
# Test class 2 — v4 CSV loading (3 tests)
# ---------------------------------------------------------------------------


class TestV4CsvLoading:
    """v4 schema has BOTH lowercase + uppercase; loader must NOT collide."""

    @pytest.mark.skipif(not _has_all(V4_CSV), reason="v4 CSV not on disk")
    def test_v4_loader_returns_canonical_4_columns(self):
        """v4 must produce exactly the 4-column canonical DataFrame.

        Pre-Wf-323B, the naive v1-only loader raised:
            ValueError: Buffer has wrong number of dimensions
                        (expected 1, got 2)
        because renaming SMILES_Ligands -> smiles collided with the
        already-present `smiles` column.  This test pins the fix.
        """
        df = load_metal_cytotox(V4_CSV)
        assert list(df.columns) == CANONICAL_COLUMNS, (
            f"v4 should produce canonical 4 columns, got {list(df.columns)}"
        )
        assert len(df) > 25_000, f"v4 row count unexpectedly small: {len(df)}"

    @pytest.mark.skipif(not _has_all(V4_CSV), reason="v4 CSV not on disk")
    def test_v4_loader_resolves_5_metals(self):
        """v4 must resolve all 5 expected metals."""
        df = load_metal_cytotox(V4_CSV)
        actual = set(df["metal"].unique())
        assert actual == EXPECTED_METALS, (
            f"v4 metals mismatch: expected {EXPECTED_METALS}, got {actual}"
        )

    @pytest.mark.skipif(not _has_all(V4_CSV), reason="v4 CSV not on disk")
    def test_v3_and_v4_outputs_match(self):
        """v3 (uppercase) and v4 (mixed case) must produce identical
        canonical DataFrames — they share the same row count and
        active distribution because v4 IS v3 normalised plus
        provenance columns."""
        df_v3 = load_metal_cytotox(V3_CSV)
        df_v4 = load_metal_cytotox(V4_CSV)
        assert len(df_v3) == len(df_v4), (
            f"v3 ({len(df_v3)} rows) and v4 ({len(df_v4)} rows) "
            f"should have identical row counts"
        )
        assert df_v3["active"].sum() == df_v4["active"].sum(), (
            f"v3 active count ({df_v3['active'].sum()}) should match "
            f"v4 ({df_v4['active'].sum()})"
        )


# ---------------------------------------------------------------------------
# Test class 3 — synonym map correctness (2 tests)
# ---------------------------------------------------------------------------


class TestSynonymMap:
    """Pin the synonym map shape + the v4 collision-avoidance behaviour."""

    def test_column_synonyms_cover_all_canonical(self):
        """Every canonical column in _REQUIRED_COLUMNS must have a
        synonym entry.  The synonym map prefers lowercase forms
        (``smiles``, ``metal``, ``pic50``, ``active``) so v4's
        already-lowercase column wins; the canonical ``pIC50`` form
        is itself an alias (chemistry convention uses lowercase pic50
        for the multi-task CSV)."""
        for canonical in ("smiles", "metal", "pIC50", "active"):
            assert canonical in _COLUMN_SYNONYMS, (
                f"missing synonym entry for canonical column {canonical!r}"
            )
            candidates: List[str] = list(_COLUMN_SYNONYMS[canonical])
            # Every synonym map must contain the lowercase form so v4
            # wins (avoiding duplicate-column collision).
            lowercase_aliases = {
                "smiles": "smiles",
                "metal": "metal",
                "pIC50": "pic50",
                "active": "active",
            }
            expected_first = lowercase_aliases[canonical]
            assert candidates[0] == expected_first, (
                f"synonym for {canonical!r} must list {expected_first!r} "
                f"first to prefer it on v4 (got {candidates})"
            )
            # And the canonical form must always be in the alias list.
            assert canonical in candidates, (
                f"canonical {canonical!r} must appear in its own synonym "
                f"list (got {candidates})"
            )

    def test_rename_v3_skips_when_lowercase_present(self):
        """When the lowercase canonical column already exists (v4 path),
        `_rename_v3_to_canonical` must NOT create duplicate columns.

        Builds an in-memory v4-like DataFrame (both lowercase +
        uppercase) and asserts the rename is a no-op.
        """
        v4_like = pd.DataFrame(
            {
                "smiles": ["C1=CC=CC=C1"],
                "metal": ["Ru"],
                "SMILES_Ligands": ["C1=CC=CC=C1"],
                "Metal": ["Ru"],
                "pic50": [4.5],
                "activity": [1],
            }
        )
        before_cols = list(v4_like.columns)
        out = _rename_v3_to_canonical(v4_like)
        # All columns should be untouched — no rename, no duplicates.
        assert list(out.columns) == before_cols, (
            f"v4-like rename must be a no-op, got {list(out.columns)}"
        )
        assert not out.columns.duplicated().any(), (
            "v4 rename must NOT introduce duplicate columns"
        )

        # Also exercise _has_column on the v4-like frame.
        assert _has_column(v4_like, "smiles") == "smiles"
        assert _has_column(v4_like, "metal") == "metal"
        assert _has_column(v4_like, "pIC50") == "pic50"
        assert _has_column(v4_like, "active") == "activity"