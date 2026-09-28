"""Tests for the real BindingDB IC50 domain (domain 3) in FourDomainDataset.

Wired 2026-09-27 (Wf-145).  Verifies that the symlinked TDC
``bindingdb_ic50.csv`` is readable, has the right columns, and ≥ 100 K
rows of usable IC50 data.

The BindingDB CSV lives at:

    .data/molmetal/bindingdb_ic50.csv
        → symlink → /home/hugo/codes/pocket/data/tdc/bindingdb_ic50.csv
            (838 MB, 1,072,931 rows)

If the symlink is missing on a developer's host, the loader transparently
falls back to ``molmetal/data/protein_metal_binding.csv`` (5,258 rows).
The fallback path is also tested here so we don't regress the
honest-negatives story.

License
-------
BindingDB data is published under
`Creative Commons Attribution 3.0 (CC BY 3.0) <https://creativecommons.org/licenses/by/3.0/>`_.
This repo does NOT redistribute the BindingDB CSV — it is referenced via a
project-local symlink.  Users without the symlink get the local 5K fallback.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Wf-50 pre-condition: honour MOLMETAL_DATA_DIR for the cytotox loader.
_DEFAULT_DATA = PROJECT_ROOT / ".data" / "molmetal"
if not os.environ.get("MOLMETAL_DATA_DIR") and _DEFAULT_DATA.exists():
    os.environ["MOLMETAL_DATA_DIR"] = str(_DEFAULT_DATA)

from molmetal.data.four_domain_combined import (
    DOMAIN_BINDINGDB,
    FourDomainDataset,
    DEFAULT_BINDINGDB_CSV,
    DEFAULT_BINDINGDB_FALLBACK_CSV,
    _BindingDBRows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_real_bindingdb_symlink() -> bool:
    """True iff the real BindingDB IC50 symlink exists on this host."""
    return DEFAULT_BINDINGDB_CSV.exists()


# Skip marker for tests that REQUIRE the symlink.
requires_real_bindingdb = pytest.mark.skipif(
    not _has_real_bindingdb_symlink(),
    reason="real BindingDB IC50 symlink missing — only fallback path tested",
)


# ---------------------------------------------------------------------------
# 1. Loader-level smoke
# ---------------------------------------------------------------------------
def test_bindingdb_loader_constructs_either_real_or_fallback() -> None:
    """``_BindingDBRows`` must construct even if the real CSV is missing."""
    loader = _BindingDBRows()
    # Exactly one of these must hold:
    assert loader.using_fallback != _has_real_bindingdb_symlink()


def test_bindingdb_loader_exposes_row_count() -> None:
    """``len(loader)`` returns a non-negative integer (line-count based)."""
    loader = _BindingDBRows()
    n = len(loader)
    assert isinstance(n, int)
    assert n >= 0


# ---------------------------------------------------------------------------
# 2. Real BindingDB CSV — column sanity (only if symlink is present)
# ---------------------------------------------------------------------------
@requires_real_bindingdb
def test_bindingdb_csv_header_has_id1_x1_id2_x2_y() -> None:
    """Real BindingDB IC50 CSV has columns ``ID1, X1, ID2, X2, Y``."""
    with DEFAULT_BINDINGDB_CSV.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert header[:5] == ["ID1", "X1", "ID2", "X2", "Y"], (
        f"unexpected header: {header[:5]}"
    )


@requires_real_bindingdb
def test_bindingdb_csv_has_at_least_100k_rows() -> None:
    """Real BindingDB IC50 CSV has ≥ 100 K rows of usable data."""
    n = 0
    with DEFAULT_BINDINGDB_CSV.open("rb") as fh:
        for _ in fh:
            n += 1
    # Subtract header line
    n -= 1
    assert n >= 100_000, f"BindingDB IC50 has only {n} rows, expected ≥ 100K"


@requires_real_bindingdb
def test_bindingdb_loader_loads_at_least_100k_rows() -> None:
    """``_BindingDBRows`` exposes ≥ 100 K rows after loading real CSV."""
    loader = _BindingDBRows()
    assert loader.using_fallback is False
    n = len(loader)
    assert n >= 100_000, f"_BindingDBRows loaded only {n} rows, expected ≥ 100K"


@requires_real_bindingdb
def test_bindingdb_loader_item_has_required_keys() -> None:
    """First item of real BindingDB loader has ``smiles``, ``pIC50``, ``active``,
    ``target_id``, ``target_seq``."""
    loader = _BindingDBRows()
    # Lazily load by accessing one item
    item = loader[0]
    assert isinstance(item, dict)
    for k in ("smiles", "pIC50", "active", "target_id", "target_seq"):
        assert k in item, f"missing key {k!r} in BindingDB item"
    assert isinstance(item["smiles"], str) and item["smiles"], "empty SMILES"
    assert isinstance(item["pIC50"], float), f"pIC50 not float: {item['pIC50']!r}"
    assert -2.0 <= item["pIC50"] <= 14.0, (
        f"pIC50 {item['pIC50']} out of expected [-2, 14] range"
    )
    assert isinstance(item["active"], bool)
    assert isinstance(item["target_id"], str)


@requires_real_bindingdb
def test_bindingdb_pic50_in_active_range() -> None:
    """For real BindingDB rows, ``pIC50 ≥ 5`` ↔ ``active = True``."""
    loader = _BindingDBRows()
    # Spot-check 100 rows for consistency
    n = min(len(loader), 100)
    for i in range(0, n, 7):  # sparse sample
        item = loader[i]
        if item["active"]:
            assert item["pIC50"] >= 5.0, (
                f"row {i}: active=True but pIC50={item['pIC50']}"
            )
        else:
            assert item["pIC50"] < 5.0, (
                f"row {i}: active=False but pIC50={item['pIC50']}"
            )


@requires_real_bindingdb
def test_bindingdb_conversion_y_nm_to_pic50_correct() -> None:
    """``pIC50 = 9 - log10(Y_nM)`` for a known row.

    First row of the CSV has ``Y=50000.0`` (nM); pIC50 should be
    ``9 - log10(50000) = 9 - 4.69897 = 4.30103``.
    """
    loader = _BindingDBRows()
    item = loader[0]
    expected_pIC50 = 9.0 - math.log10(50_000.0)
    assert math.isclose(item["pIC50"], expected_pIC50, abs_tol=1e-6), (
        f"row 0 pIC50={item['pIC50']}, expected {expected_pIC50}"
    )


# ---------------------------------------------------------------------------
# 3. FourDomainDataset integration
# ---------------------------------------------------------------------------
def test_four_domain_dataset_constructs_with_or_without_symlink() -> None:
    """``FourDomainDataset`` constructs regardless of symlink presence."""
    ds = FourDomainDataset(max_rows_per_domain={DOMAIN_BINDINGDB: 1_000})
    assert ds._domain_3 is not None  # type: ignore[attr-defined]


def test_four_domain_dataset_bindingdb_length_reflects_symlink() -> None:
    """Domain-3 length is ≥ 100K with symlink, ~5K without (fallback)."""
    ds = FourDomainDataset()
    n3 = ds._lens[DOMAIN_BINDINGDB]  # type: ignore[attr-defined]
    if _has_real_bindingdb_symlink():
        assert n3 >= 100_000, (
            f"BindingDB domain-3 has {n3} rows; expected ≥ 100K with symlink"
        )
    else:
        assert 5_000 < n3 < 6_000, (
            f"BindingDB fallback has {n3} rows; expected ~5,258"
        )


@requires_real_bindingdb
def test_four_domain_dataset_bindingdb_item_carries_target_id() -> None:
    """Domain-3 item carries ``target_id`` (UniProt accession) when real."""
    ds = FourDomainDataset(max_rows_per_domain={DOMAIN_BINDINGDB: 1_000})
    # The first item of domain 3 is at offset = sum(other domains)
    item = ds[ds._offsets[DOMAIN_BINDINGDB]]  # type: ignore[attr-defined]
    assert item.domain_id == DOMAIN_BINDINGDB
    assert "target_id" in item.aux, (
        f"target_id missing from domain-3 aux: {item.aux}"
    )


# ---------------------------------------------------------------------------
# 4. License metadata
# ---------------------------------------------------------------------------
def test_bindingdb_loader_docstring_cites_cc_by_3() -> None:
    """The ``_BindingDBRows`` docstring cites CC BY 3.0 (BindingDB license)."""
    doc = _BindingDBRows.__doc__ or ""
    assert "CC BY 3.0" in doc, (
        "_BindingDBRows docstring must cite BindingDB's CC BY 3.0 license"
    )


def test_four_domain_combined_docstring_cites_cc_by_3() -> None:
    """The :file:`four_domain_combined.py` module docstring cites CC BY 3.0."""
    from molmetal.data import four_domain_combined

    doc = four_domain_combined.__doc__ or ""
    assert "CC BY 3.0" in doc, (
        "four_domain_combined module docstring must cite BindingDB's CC BY 3.0 license"
    )


# ---------------------------------------------------------------------------
# 5. Symlink integrity (when present)
# ---------------------------------------------------------------------------
@requires_real_bindingdb
def test_bindingdb_symlink_points_to_pocket_data_tdc() -> None:
    """The symlink resolves to a real file under ``pocket/data/tdc/``."""
    target = os.readlink(DEFAULT_BINDINGDB_CSV)
    assert "pocket/data/tdc/bindingdb_ic50.csv" in target, (
        f"unexpected symlink target: {target}"
    )
    resolved = (DEFAULT_BINDINGDB_CSV.parent / target).resolve()
    assert resolved.exists(), f"symlink target missing: {resolved}"
    # 838 MB file — allow some slack for housekeeping copies
    assert resolved.stat().st_size > 100_000_000, (
        f"BindingDB IC50 file is only {resolved.stat().st_size} bytes; expected > 100MB"
    )