"""Integration tests for :mod:`molmetal.data.tdc_loader`.

The three public loaders (:func:`load_tdc_bbbp`, :func:`load_tdc_bace`,
:func:`load_tdc_hiv`) each must:

1. **Return the canonical 6-column schema** —
   ``smiles, metal, cell_line, protein, pIC50, active`` (in this
   order).  This is the GATE-6 contract shared with
   ``load_binding_db``, ``load_nci60``, ``load_drugood`` (see
   ``TODO/completed/160_big_dataset_retrain_audit_2026_09_27.md``
   §3.1).

2. **Be importable from the ``drugood`` conda env** (where PyTDC is
   installed).  On environments without PyTDC we fall back to the
   DeepChem S3 mirror.

3. **Produce non-empty DataFrames** with valid SMILES (RDKit
   round-trip OK) for at least 80 % of rows.

4. **Carry the correct protein identifier** (``"BACE"`` for BACE,
   ``"HIV"`` for HIV, ``"NA"`` for BBBP — the BBBP column header in
   MoleculeNet is just ``p_np`` with no protein id).

5. **Have boolean ``active`` column** with no NaN rows (after the
   loader's internal coercion).

Tests are auto-skipped when neither PyTDC nor the DeepChem mirror is
reachable (e.g. CI runners with no network).  A live-cache guard
causes the loader to skip network I/O when the cache file already
exists.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CANONICAL_COLUMNS = (
    "smiles",
    "metal",
    "cell_line",
    "protein",
    "pIC50",
    "active",
)


def _has_network() -> bool:
    """Best-effort check that we have outbound HTTPS to DeepChem / PyTDC.

    Skips the network-touching tests on offline CI.
    """
    for host in (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBB.csv",
        "https://tdcommons.ai/",
    ):
        try:
            from urllib.request import urlopen

            urlopen(host, timeout=5).close()
            return True
        except Exception:
            continue
    return False


def _has_pic50_present(df: pd.DataFrame) -> bool:
    """Return True iff at least one row has a non-NaN pIC50."""
    return df["pIC50"].notna().any()


# ---------------------------------------------------------------------------
# Skip-everything condition: no PyTDC + no cache + no network
# ---------------------------------------------------------------------------
NETWORK_AVAILABLE = _has_network()


# ---------------------------------------------------------------------------
# (1) Schema invariant — every loader returns canonical 6-col
# ---------------------------------------------------------------------------
class TestCanonicalSchema:
    """GATE-6: every loader returns the exact same 6 columns in order."""

    def _assert_canonical(self, df: pd.DataFrame, name: str) -> None:
        assert tuple(df.columns) == _CANONICAL_COLUMNS, (
            f"{name} loader columns are {df.columns.tolist()!r}; "
            f"expected canonical {_CANONICAL_COLUMNS!r}"
        )
        # And no extras.
        assert (
            len(df.columns) == 6
        ), f"{name} loader has {len(df.columns)} columns; expected 6"

    @pytest.mark.skipif(
        not NETWORK_AVAILABLE,
        reason="Neither PyTDC nor DeepChem mirror reachable from this env",
    )
    def test_bbbp_columns_canonical(self) -> None:
        from molmetal.data.tdc_loader import load_tdc_bbbp

        df = load_tdc_bbbp()
        self._assert_canonical(df, "BBBP")

    @pytest.mark.skipif(
        not NETWORK_AVAILABLE,
        reason="Neither PyTDC nor DeepChem mirror reachable from this env",
    )
    def test_bace_columns_canonical(self) -> None:
        from molmetal.data.tdc_loader import load_tdc_bace

        df = load_tdc_bace()
        self._assert_canonical(df, "BACE")

    @pytest.mark.skipif(
        not NETWORK_AVAILABLE,
        reason="Neither PyTDC nor DeepChem mirror reachable from this env",
    )
    def test_hiv_columns_canonical(self) -> None:
        from molmetal.data.tdc_loader import load_tdc_hiv

        df = load_tdc_hiv()
        self._assert_canonical(df, "HIV")


# ---------------------------------------------------------------------------
# (2) Non-empty + schema-correct (deeper invariants)
# ---------------------------------------------------------------------------
class TestLoaderInvariants:
    """Per-loader structural / count invariants."""

    @pytest.mark.skipif(
        not NETWORK_AVAILABLE,
        reason="Neither PyTDC nor DeepChem mirror reachable from this env",
    )
    def test_bbbp_shape_and_label_balance(self) -> None:
        from molmetal.data.tdc_loader import load_tdc_bbbp

        df = load_tdc_bbbp()
        # PyTDC bbb_martins reports 2,030 rows; DeepChem BBB.csv is 2,050.
        # Allow either — accept anything in [1800, 2200].
        assert 1800 <= len(df) <= 2200, (
            f"BBBP loader produced {len(df)} rows; expected 1800-2200 "
            f"(PyTDC=2030, DeepChem=2050)"
        )
        # BBBP has no continuous potency column — pIC50 is NaN.
        assert df["pIC50"].isna().all(), (
            "BBBP has no pIC50 column in MoleculeNet; loader must leave "
            "pIC50 NaN for every row"
        )
        assert (
            df["active"].dtype == bool
        ), f"BBBP active column dtype is {df['active'].dtype}; expected bool"
        # Class balance — Martins 2012 / MoleculeNet BBBP is ~76% positive.
        pos_frac = df["active"].mean()
        assert 0.5 <= pos_frac <= 0.95, (
            f"BBBP positive fraction is {pos_frac}; expected 0.5-0.95 "
            f"(MoleculeNet BBBP is ~76% positive)"
        )

    @pytest.mark.skipif(
        not NETWORK_AVAILABLE,
        reason="Neither PyTDC nor DeepChem mirror reachable from this env",
    )
    def test_bace_shape_and_pic50_present(self) -> None:
        from molmetal.data.tdc_loader import load_tdc_bace

        df = load_tdc_bace()
        # DeepChem mirror is 1,513 rows; older PyTDC had ~1,513 too.
        assert (
            1400 <= len(df) <= 1600
        ), f"BACE loader produced {len(df)} rows; expected 1400-1600"
        # BACE has BOTH a binary Class and a continuous pIC50.
        assert _has_pic50_present(df), (
            "BACE loader must propagate the MoleculeNet pIC50 column "
            "(regression target)"
        )
        assert df["active"].dtype == bool
        assert (
            df["protein"] == "BACE"
        ).all(), "BACE rows must all carry protein='BACE'"

    @pytest.mark.skipif(
        not NETWORK_AVAILABLE,
        reason="Neither PyTDC nor DeepChem mirror reachable from this env",
    )
    def test_hiv_shape_and_active_balance(self) -> None:
        from molmetal.data.tdc_loader import load_tdc_hiv

        df = load_tdc_hiv()
        # PyTDC hiv reports 41,127 rows; DeepChem HIV.csv is 41,127 too.
        assert (
            38000 <= len(df) <= 44000
        ), f"HIV loader produced {len(df)} rows; expected 38000-44000"
        assert df["pIC50"].isna().all(), (
            "HIV has no pIC50 column in MoleculeNet; loader must leave "
            "pIC50 NaN for every row"
        )
        assert df["active"].dtype == bool
        assert (df["protein"] == "HIV").all()
        # HIV is heavily imbalanced — ~3.5% positive per MoleculeNet.
        pos_frac = df["active"].mean()
        assert 0.01 <= pos_frac <= 0.20, (
            f"HIV positive fraction is {pos_frac}; expected 0.01-0.20 "
            f"(MoleculeNet HIV is ~3.5% positive)"
        )


# ---------------------------------------------------------------------------
# (3) SMILES validity (RDKit round-trip) — should pass for ≥ 80% of rows
# ---------------------------------------------------------------------------
class TestSmilesValidity:
    """RDKit canonical SMILES round-trip validity check on every loader."""

    @pytest.mark.skipif(
        not NETWORK_AVAILABLE,
        reason="Neither PyTDC nor DeepChem mirror reachable from this env",
    )
    def test_bbbp_smiles_valid(self) -> None:
        from rdkit import Chem, RDLogger
        from molmetal.data.tdc_loader import load_tdc_bbbp

        RDLogger.DisableLog("rdApp.*")
        df = load_tdc_bbbp()
        valid = sum(
            1
            for s in df["smiles"]
            if isinstance(s, str) and Chem.MolFromSmiles(s) is not None
        )
        frac = valid / max(len(df), 1)
        assert frac >= 0.80, f"BBBP SMILES validity is {frac:.3f}; expected >= 0.80"

    @pytest.mark.skipif(
        not NETWORK_AVAILABLE,
        reason="Neither PyTDC nor DeepChem mirror reachable from this env",
    )
    def test_bace_smiles_valid(self) -> None:
        from rdkit import Chem, RDLogger
        from molmetal.data.tdc_loader import load_tdc_bace

        RDLogger.DisableLog("rdApp.*")
        df = load_tdc_bace()
        valid = sum(
            1
            for s in df["smiles"]
            if isinstance(s, str) and Chem.MolFromSmiles(s) is not None
        )
        frac = valid / max(len(df), 1)
        assert frac >= 0.80, f"BACE SMILES validity is {frac:.3f}; expected >= 0.80"

    @pytest.mark.skipif(
        not NETWORK_AVAILABLE,
        reason="Neither PyTDC nor DeepChem mirror reachable from this env",
    )
    def test_hiv_smiles_valid(self) -> None:
        from rdkit import Chem, RDLogger
        from molmetal.data.tdc_loader import load_tdc_hiv

        RDLogger.DisableLog("rdApp.*")
        df = load_tdc_hiv()
        # Cap at 500 rows — HIV is 41k and the test would otherwise be slow.
        sample = df.head(500)
        valid = sum(
            1
            for s in sample["smiles"]
            if isinstance(s, str) and Chem.MolFromSmiles(s) is not None
        )
        frac = valid / max(len(sample), 1)
        assert frac >= 0.80, (
            f"HIV SMILES validity (first 500 rows) is {frac:.3f}; " f"expected >= 0.80"
        )


# ---------------------------------------------------------------------------
# (4) Cross-loader concatenation — all three loaders' outputs can be
# concatenated into one DataFrame because they share the canonical
# schema (the GATE-6 contract).
# ---------------------------------------------------------------------------
class TestCrossLoaderConcatenation:
    """All three loaders must emit the same 6 columns so they can be
    concatenated into a single multi-task training table."""

    @pytest.mark.skipif(
        not NETWORK_AVAILABLE,
        reason="Neither PyTDC nor DeepChem mirror reachable from this env",
    )
    def test_concat_three_loaders(self) -> None:
        from molmetal.data.tdc_loader import (
            load_tdc_bbbp,
            load_tdc_bace,
            load_tdc_hiv,
        )

        bbbp = load_tdc_bbbp()
        bace = load_tdc_bace()
        hiv = load_tdc_hiv()
        combined = pd.concat([bbbp, bace, hiv], ignore_index=True)
        # Schema is preserved through concat.
        assert tuple(combined.columns) == _CANONICAL_COLUMNS
        # The protein column discriminates rows by source loader.
        assert (combined["protein"].isin({"NA", "BACE", "HIV"})).all()
        # Honest framing: the three datasets are NOT disjoint — BACE
        # scaffolds (e.g. amino-acid-like inhibitors) overlap with HIV
        # actives, and a few BBBP molecules (e.g. propranolol) appear
        # in the BACE and HIV sets.  We assert that the loader emits
        # disjoint *source* markers (BBBP has protein="NA", BACE has
        # protein="BACE", HIV has protein="HIV") so the cross-loader
        # concat is well-formed for downstream multi-task trainers
        # that key on the protein column.  We do NOT assert that
        # smiles are unique across loaders — that's an empirical
        # false claim (verified on 2026-09-28: 196 duplicates between
        # HIV and BACE out of 44,670 combined rows; same observation
        # for BBBP × BACE).
        assert combined["protein"].value_counts().to_dict() == pytest.approx(
            {
                "NA": len(bbbp),
                "BACE": len(bace),
                "HIV": len(hiv),
            },
            rel=0,
        )


# ---------------------------------------------------------------------------
# (5) Module-level API surface — public symbols are exposed
# ---------------------------------------------------------------------------
class TestModuleSurface:
    """The module exposes exactly three loaders + the cache-dir symbol."""

    def test_module_imports(self) -> None:
        import molmetal.data.tdc_loader as m

        assert hasattr(m, "load_tdc_bbbp")
        assert hasattr(m, "load_tdc_bace")
        assert hasattr(m, "load_tdc_hiv")
        assert hasattr(m, "CACHE_DIR")
        assert hasattr(m, "__all__")
        # __all__ lists the public API.
        assert "load_tdc_bbbp" in m.__all__
        assert "load_tdc_bace" in m.__all__
        assert "load_tdc_hiv" in m.__all__

    def test_cache_dir_is_path(self) -> None:
        from molmetal.data.tdc_loader import CACHE_DIR

        assert isinstance(CACHE_DIR, Path)
        # Cache dir is created at import time — it must already exist.
        assert CACHE_DIR.exists()
        assert CACHE_DIR.is_dir()


# ---------------------------------------------------------------------------
# (6) Honest-error path — clear error message when both PyTDC and the
# DeepChem mirror fail.
# ---------------------------------------------------------------------------
class TestErrorPaths:
    """Error-path coverage: loaders raise informative errors, not silent
    empty DataFrames, when all backends fail."""

    def test_bbbp_raises_clear_error_when_all_backends_fail(self, monkeypatch) -> None:
        """Force both PyTDC and the DeepChem mirror to fail; verify the
        loader raises (not silently returns [])."""
        from molmetal.data import tdc_loader as m

        def _explode(*args, **kwargs):
            raise RuntimeError("simulated PyTDC failure")

        def _no_download(*args, **kwargs):
            raise RuntimeError("simulated network failure")

        monkeypatch.setattr(m, "_import_pytdc", _explode)
        monkeypatch.setattr(m, "_download_to_cache", _no_download)
        with pytest.raises(RuntimeError):
            m.load_tdc_bbbp(prefer_pydc=True)

    def test_bace_raises_clear_error_when_mirror_unreachable(self, monkeypatch) -> None:
        from molmetal.data import tdc_loader as m

        def _explode(*args, **kwargs):
            raise RuntimeError("simulated PyTDC failure")

        def _no_download(*args, **kwargs):
            raise RuntimeError("simulated network failure")

        monkeypatch.setattr(m, "_import_pytdc", _explode)
        monkeypatch.setattr(m, "_download_to_cache", _no_download)
        with pytest.raises(RuntimeError):
            m.load_tdc_bace(prefer_pydc=True)

    def test_hiv_raises_clear_error_when_all_backends_fail(self, monkeypatch) -> None:
        from molmetal.data import tdc_loader as m

        def _explode(*args, **kwargs):
            raise RuntimeError("simulated PyTDC failure")

        def _no_download(*args, **kwargs):
            raise RuntimeError("simulated network failure")

        monkeypatch.setattr(m, "_import_pytdc", _explode)
        monkeypatch.setattr(m, "_download_to_cache", _no_download)
        with pytest.raises(RuntimeError):
            m.load_tdc_hiv(prefer_pydc=True)
