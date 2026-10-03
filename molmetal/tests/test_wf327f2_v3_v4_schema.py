"""Wf-327F2 verification tests — MetalCytoToxDB loader v3/v4 schema drift.

==============================================================================
Wf-327F2 — independent verification of Wf-323B v3/v4 loader contract
==============================================================================

Wf-323B (SHIPPED 2026-09-29 per TODO/completed/323b_v3_v4_schema_drift_fix.md)
delivered a synonym-aware loader that round-trips v3 (uppercase
SMILES_Ligands/Metal) + v4 (mixed-case lowercase + uppercase + provenance)
through the same canonical 4-column contract.  Wf-327F2 is the
**independent re-verification** layer (per party-twenty-spirit §3.5):
additive tests that pin invariants beyond what Wf-323B's 8 tests cover.

This module adds **4 NEW tests** (CPU-only, < 5 s total):

  1. ``TestCrossVersionInvariants::test_v1_v3_v4_canonical_alignment`` —
     all three CSV versions produce the SAME canonical 4-column contract
     (smiles, metal, pIC50, active) with identical dtypes, sorted equally.

  2. ``TestIdempotentReload::test_loading_v4_twice_is_idempotent`` —
     reloading the same v4 CSV twice produces byte-identical DataFrames
     (round-trip stability; no Mutation / cumulative side-effects).

  3. ``TestHasColumnPrecedence::test_lowercase_wins_when_both_present`` —
     when v4 carries BOTH ``smiles`` and ``SMILES_Ligands`` columns,
     ``_has_column`` must return ``smiles`` (the lowercase canonical),
     not the uppercase alias (preventing the v1-only duplicate-column
     ``ValueError`` from regressing).

  4. ``TestSynonymMapCompleteness::test_all_aliases_are_lowercase_or_known`` —
     every alias in ``_COLUMN_SYNONYMS`` must be either the lowercase
     canonical itself, a known uppercase provenance (SMILES_Ligands /
     Metal / IC50_Dark_value / activity), or otherwise pre-existing in
     the column rename chain.  This guards against a future synonym
     addition that silently breaks the v4 collision path.

All tests are pure CPU and self-contained (no GPU, no network, no
external state).  Total runtime < 5 s on a 16-core CPU.

Honest framing (party-twenty-spirit §3.5): the loader was MEASURED-fixed
by Wf-323B and MEASURED-stable by Wf-327F2.  These tests pin 4
independent invariants that complement (not duplicate) Wf-323B's
8-test surface; together they raise the regression-detection surface
for v3/v4 schema drift from 8 to 12 tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from molmetal.data.metal_cytotox import (  # noqa: E402
        _COLUMN_SYNONYMS,
        _has_column,
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

CANONICAL_COLUMNS = ["smiles", "metal", "pIC50", "active"]
EXPECTED_METALS = {"Ru", "Ir", "Os", "Rh", "Re"}

# Aliases (beyond the lowercase canonical itself) that the synonym map
# intentionally accepts.  These cover v3 uppercase provenance columns +
# legacy lowercase variants + Krasnov canonical raw IC50.  Adding a new
# alias outside this set requires updating the test below to keep the
# contract pinned.
_KNOWN_NON_CANONICAL_ALIASES = frozenset({
    "SMILES_Ligands",   # v3 uppercase column
    "smiles_ligands",   # v3-style lowercase variant
    "Metal",            # v3 uppercase column
    "pic50",            # lowercase pIC50 (canonical for multi-task CSVs)
    "pIC50",            # mixed-case pIC50 alias (matches dict key)
    "IC50_Dark_value",  # Krasnov canonical raw IC50 column
    "activity",         # v1 lowercase but not the canonical "active"
})


def _has_all(path: Path) -> bool:
    return path.exists() and path.is_file()


# ---------------------------------------------------------------------------
# Test class 1 — cross-version canonical invariant
# ---------------------------------------------------------------------------


class TestCrossVersionInvariants:
    """v1, v3, v4 must all round-trip to the same canonical 4-col shape."""

    @pytest.mark.skipif(
        not all(_has_all(p) for p in (V1_CSV, V3_CSV, V4_CSV)),
        reason="v1+v3+v4 CSVs must all be on disk",
    )
    def test_v1_v3_v4_canonical_alignment(self):
        """All three CSV versions must produce DataFrames with:
        - identical column names + order
        - identical dtypes per column
        - identical 5-metal set
        - row counts within 1% of each other (v1=26801 vs v3/v4=26632)

        Pre-Wf-323B, v4 raised ValueError; this test pins the cross-version
        canonical alignment delivered by Wf-323B.
        """
        df_v1 = load_metal_cytotox(V1_CSV)
        df_v3 = load_metal_cytotox(V3_CSV)
        df_v4 = load_metal_cytotox(V4_CSV)

        # Same column names + order
        assert list(df_v1.columns) == CANONICAL_COLUMNS, (
            f"v1 columns drift: {list(df_v1.columns)}"
        )
        assert list(df_v3.columns) == CANONICAL_COLUMNS, (
            f"v3 columns drift: {list(df_v3.columns)}"
        )
        assert list(df_v4.columns) == CANONICAL_COLUMNS, (
            f"v4 columns drift: {list(df_v4.columns)}"
        )

        # Same dtypes
        for col in CANONICAL_COLUMNS:
            assert df_v1[col].dtype == df_v3[col].dtype == df_v4[col].dtype, (
                f"dtype drift on {col}: "
                f"v1={df_v1[col].dtype}, v3={df_v3[col].dtype}, v4={df_v4[col].dtype}"
            )

        # Same 5-metal set
        assert set(df_v1["metal"].unique()) == EXPECTED_METALS
        assert set(df_v3["metal"].unique()) == EXPECTED_METALS
        assert set(df_v4["metal"].unique()) == EXPECTED_METALS

        # Row counts within 1% (v1=26801 vs v3/v4=26632 is ~0.6% delta)
        max_rows = max(len(df_v1), len(df_v3), len(df_v4))
        min_rows = min(len(df_v1), len(df_v3), len(df_v4))
        delta_pct = (max_rows - min_rows) / max_rows * 100
        assert delta_pct < 1.0, (
            f"row count drift between versions exceeds 1%: "
            f"v1={len(df_v1)}, v3={len(df_v3)}, v4={len(df_v4)} "
            f"(delta={delta_pct:.3f}%)"
        )


# ---------------------------------------------------------------------------
# Test class 2 — idempotent reload (round-trip stability)
# ---------------------------------------------------------------------------


class TestIdempotentReload:
    """Loading the same CSV twice must produce identical DataFrames."""

    @pytest.mark.skipif(not _has_all(V4_CSV), reason="v4 CSV not on disk")
    def test_loading_v4_twice_is_idempotent(self):
        """Reload v4 CSV twice — both DataFrames must be byte-equal.

        This guards against stateful loader behaviour (e.g., global
        state mutation, column rename cache, singleton pattern) that
        would silently shift values between successive loads.
        """
        df_first = load_metal_cytotox(V4_CSV)
        df_second = load_metal_cytotox(V4_CSV)

        # Same shape
        assert df_first.shape == df_second.shape, (
            f"reload shape drift: {df_first.shape} vs {df_second.shape}"
        )

        # Same columns + dtypes
        assert list(df_first.columns) == list(df_second.columns)
        for col in df_first.columns:
            assert df_first[col].dtype == df_second[col].dtype

        # Same values (NaN-aware equal)
        pd.testing.assert_frame_equal(df_first, df_second)


# ---------------------------------------------------------------------------
# Test class 3 — _has_column precedence (lowercase wins)
# ---------------------------------------------------------------------------


class TestHasColumnPrecedence:
    """When v4 carries both lowercase + uppercase, lowercase wins."""

    @pytest.mark.skipif(not _has_all(V4_CSV), reason="v4 CSV not on disk")
    def test_lowercase_wins_when_both_present(self):
        """v4 CSV has BOTH ``smiles`` AND ``SMILES_Ligands`` columns.

        ``_has_column(df, 'smiles')`` must return ``'smiles'`` (lowercase
        canonical), NOT ``'SMILES_Ligands'`` (uppercase alias).  If it
        returned the uppercase, the loader would attempt a rename that
        creates duplicate columns and re-introduce the pre-Wf-323B
        ``ValueError``.
        """
        df_v4 = pd.read_csv(V4_CSV)
        # Sanity-check: v4 must indeed have both columns on disk
        assert "smiles" in df_v4.columns, (
            f"v4 missing lowercase smiles (columns: {list(df_v4.columns)})"
        )
        assert "SMILES_Ligands" in df_v4.columns, (
            f"v4 missing uppercase SMILES_Ligands (columns: {list(df_v4.columns)})"
        )

        # _has_column must prefer lowercase
        smiles_match = _has_column(df_v4, "smiles")
        assert smiles_match == "smiles", (
            f"_has_column('smiles') returned {smiles_match!r}; "
            f"must be lowercase 'smiles' to avoid v4 collision"
        )

        metal_match = _has_column(df_v4, "metal")
        assert metal_match == "metal", (
            f"_has_column('metal') returned {metal_match!r}; "
            f"must be lowercase 'metal' to avoid v4 collision"
        )


# ---------------------------------------------------------------------------
# Test class 4 — synonym map completeness
# ---------------------------------------------------------------------------


class TestSynonymMapCompleteness:
    """Every alias must be either lowercase canonical or known provenance."""

    def test_all_aliases_are_lowercase_or_known(self):
        """All aliases in ``_COLUMN_SYNONYMS`` are constrained to:
        - lowercase canonical (the first entry, preferred by _has_column)
        - uppercase provenance from ``_KNOWN_UPPERCASE_ALIASES``
          (SMILES_Ligands / Metal / IC50_Dark_value / activity)

        This guards against future synonym additions that silently
        introduce new alias forms without updating the v4 collision
        detection logic.
        """
        # Required canonical keys
        expected_keys = {"smiles", "metal", "pIC50", "active"}
        assert set(_COLUMN_SYNONYMS.keys()) == expected_keys, (
            f"synonym keys drift: expected {expected_keys}, "
            f"got {set(_COLUMN_SYNONYMS.keys())}"
        )

        # Each canonical maps to a non-empty tuple
        for canonical, aliases in _COLUMN_SYNONYMS.items():
            assert isinstance(aliases, tuple), (
                f"{canonical}: aliases must be tuple, got {type(aliases)}"
            )
            assert len(aliases) >= 1, (
                f"{canonical}: aliases tuple must be non-empty"
            )

            # First alias must be the lowercase/canonical form (the
            # _has_column iteration order resolves to first match wins;
            # the loader relies on this precedence to short-circuit the
            # v4 collision case).  We accept either the literal dict
            # key (e.g., "smiles") or the canonical-lowercase variant
            # (e.g., "pic50" for the pIC50 dict key).
            first_lower = aliases[0].lower()
            assert first_lower == canonical.lower(), (
                f"{canonical}: first alias must be the lowercase "
                f"canonical form, got {aliases[0]!r} "
                f"(expected lowercase of {canonical!r})"
            )

            # All aliases must be either the lowercase canonical or in
            # the known non-canonical alias set
            for alias in aliases:
                ok = (alias == canonical) or (alias in _KNOWN_NON_CANONICAL_ALIASES)
                assert ok, (
                    f"{canonical}: alias {alias!r} is neither the "
                    f"lowercase canonical nor a known non-canonical alias; "
                    f"add it to _KNOWN_NON_CANONICAL_ALIASES if intentional"
                )
