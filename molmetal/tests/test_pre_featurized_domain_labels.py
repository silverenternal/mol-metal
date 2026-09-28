"""Regression test: pre-featurized .pt files use CANONICAL domain labels.

Background
----------
The D-MPNN four-domain pretrain pipeline writes pre-featurized
``.pt`` files to ``molmetal/data/pre_featurized/``.  Each row carries
a ``domain`` int64 tensor identifying which of the four cross-domain
sources that row came from.

The **CANONICAL** mapping (mandated by
``molmetal/data/four_domain_combined.py``,
``molmetal/scripts/pre_featurize_dmpnn.py``, and
``train_dmpnn_multitask.py --domain-id``) is::

    0 = MetalCytoToxDB  (file: metal_cytotox.pt)
    1 = DrugOOD         (file: drugood.pt)
    2 = NCI-60          (file: nci60.pt)
    3 = BindingDB       (file: binding_db.pt)

A previous stub mapping
(``0=metal_cytotox 1=nci60 2=binding_db 3=drugood``) was used by the
2026-09-26 pre-featurized batch; that batch was regenerated on
2026-09-27 with the canonical labels (Wf-124 P0 #2).  This test pins
the on-disk labels so that any future drift — e.g. someone re-running
``pre_featurize_dmpnn.py`` against a stale ``DOMAIN_ID`` dict, or a
new contributor who copies an old mapping — fails CI immediately.

HONEST FRAMING (Wf-124, 2026-09-27)
-----------------------------------
This test is a pure-read regression probe: it loads each ``.pt`` file
under :mod:`torch.load(weights_only=False)`, inspects the ``domain``
tensor, and asserts label consistency.  No mutable state is touched
and no training is triggered.  These tests pass on the post-2026-09-27
canonical rebuild; they will start failing on any future drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# CANONICAL domain-id mapping — single source of truth for this test.
# Must agree with molmetal/data/four_domain_combined.py and
# molmetal/scripts/pre_featurize_dmpnn.py.
# ---------------------------------------------------------------------------
CANONICAL_DOMAIN_ID = {
    "metal_cytotox": 0,   # MetalCytoToxDB
    "drugood":       1,   # DrugOOD
    "nci60":         2,   # NCI-60
    "binding_db":    3,   # BindingDB
}
ALL_DOMAIN_IDS = {0, 1, 2, 3}

PRE_FEATURIZED_DIR = PROJECT_ROOT / "molmetal" / "data" / "pre_featurized"

# Mapping from filename -> canonical domain_id expected for every row.
FILE_TO_CANONICAL_ID = {
    "metal_cytotox.pt": CANONICAL_DOMAIN_ID["metal_cytotox"],   # 0
    "drugood.pt":       CANONICAL_DOMAIN_ID["drugood"],         # 1
    "nci60.pt":         CANONICAL_DOMAIN_ID["nci60"],           # 2
    "binding_db.pt":    CANONICAL_DOMAIN_ID["binding_db"],      # 3
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_pt(filename: str) -> dict:
    """Load a pre-featurized .pt file with weights_only=False."""
    path = PRE_FEATURIZED_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not present — pre_featurize_dmpnn never ran")
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# 1) Per-domain files use the canonical domain_id for every row
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filename,expected_id",
    [
        ("metal_cytotox.pt", CANONICAL_DOMAIN_ID["metal_cytotox"]),
        ("drugood.pt",       CANONICAL_DOMAIN_ID["drugood"]),
        ("nci60.pt",         CANONICAL_DOMAIN_ID["nci60"]),
        ("binding_db.pt",    CANONICAL_DOMAIN_ID["binding_db"]),
    ],
)
def test_per_domain_files_use_canonical_labels(filename, expected_id):
    """Every row in ``<domain>.pt`` must carry the canonical domain_id."""
    pack = _load_pt(filename)
    assert "domain" in pack, f"{filename}: missing 'domain' tensor"
    dom = pack["domain"]
    assert torch.is_tensor(dom), f"{filename}: 'domain' must be a Tensor"
    assert dom.dtype == torch.int64, (
        f"{filename}: 'domain' dtype must be int64, got {dom.dtype}"
    )
    uniq = dom.unique().tolist()
    assert uniq == [expected_id], (
        f"{filename}: expected only canonical domain_id={expected_id} "
        f"(from CANONICAL_DOMAIN_ID[{next(k for k, v in CANONICAL_DOMAIN_ID.items() if v == expected_id)}]) "
        f"but got unique domain ids {uniq}.  Pre-featurized .pt files "
        f"must be regenerated with the canonical mapping — see "
        f"molmetal/scripts/pre_featurize_dmpnn.py:DOMAIN_ID and "
        f"molmetal/data/four_domain_combined.py for the single source of "
        f"truth."
    )


# ---------------------------------------------------------------------------
# 2) all_pretrain.pt concatenates all four canonical domain_ids
# ---------------------------------------------------------------------------
def test_all_pretrain_includes_all_domains():
    """The aggregate ``all_pretrain.pt`` must contain rows from all four
    canonical domain_ids (no domain missing, no extra domain present)."""
    pack = _load_pt("all_pretrain.pt")
    assert "domain" in pack, "all_pretrain.pt: missing 'domain' tensor"
    uniq = set(pack["domain"].unique().tolist())
    assert uniq == ALL_DOMAIN_IDS, (
        f"all_pretrain.pt: expected canonical domain_ids {sorted(ALL_DOMAIN_IDS)} "
        f"but got {sorted(uniq)}.  The aggregate file must contain every "
        f"domain exactly once.  Drift here indicates the pre-featurize "
        f"pipeline is no longer consistent with CANONICAL_DOMAIN_ID."
    )


# ---------------------------------------------------------------------------
# 3) No stale domain_id per per-domain file
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filename,expected_id",
    [
        ("metal_cytotox.pt", CANONICAL_DOMAIN_ID["metal_cytotox"]),
        ("drugood.pt",       CANONICAL_DOMAIN_ID["drugood"]),
        ("nci60.pt",         CANONICAL_DOMAIN_ID["nci60"]),
        ("binding_db.pt",    CANONICAL_DOMAIN_ID["binding_db"]),
    ],
)
def test_no_stale_domain_ids(filename, expected_id):
    """No row in ``<domain>.pt`` may carry a domain_id that does not match
    the canonical id.  This is the strongest pin: we check that 100% of
    rows (not just unique() rows) agree with the canonical mapping.

    The pre-fix world (2026-09-26) had ``nci60.pt`` labelled with id=1
    and ``drugood.pt`` labelled with id=3 — this test fails loudly on
    that world so a reviewer can immediately spot the regen need.
    """
    pack = _load_pt(filename)
    assert "domain" in pack, f"{filename}: missing 'domain' tensor"
    dom = pack["domain"]
    n_stale = int((dom != expected_id).sum().item())
    n_total = int(dom.numel())
    assert n_stale == 0, (
        f"{filename}: {n_stale} / {n_total} rows carry a STALE domain_id "
        f"(expected canonical id={expected_id}).  Pre-featurized .pt "
        f"files must be regenerated — see "
        f"molmetal/scripts/pre_featurize_dmpnn.py for the regen path.  "
        f"This message indicates the pre-featurize pipeline has drifted "
        f"back to the 2026-09-26 stub mapping and downstream D-MPNN "
        f"training would silently consume the wrong domain labels."
    )
