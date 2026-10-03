"""Pytest: canonical-metal filter on ``build_pretrain_dataloader``.

TODO-160 §4.2 — 5-line patch to
:func:`molmetal.scripts.train_dmpnn_multitask.build_pretrain_dataloader`
that drops ``metal_idx == -1`` sentinel rows before the optional
``max_rows`` slice.  Without the filter, the pretrain path CRASHES at
step 1 inside ``V4MultiTaskDMPNN.forward_cached`` because ``metal_idx
== -1`` is used as a tensor index and triggers an out-of-bounds.

Three properties are verified here:

  (i)   Rows with ``metal_idx == -1`` (and any out-of-{0..4} value)
        are removed from the loader.  The per-row ``atom_features`` /
        ``edge_index`` / ``edge_attr`` / ``atom_mask`` / ``y`` /
        ``active`` / ``domain`` tensors stay aligned (same number of
        rows post-filter as canonical rows in the source pack).
  (ii)  Surviving ``metal_idx`` values lie in the canonical range
        {0..4} (i.e. 5 precious metals: Pt/Ru/Ir/Au/Pd).
  (iii) The number of dropped rows is logged via stdout in the
        canonical ``[dmpnn_multitask] pretrain: filtered N ...``
        message — the count matches ``n_source - n_canonical``.

All tests are CPU-only and use a 24-row synthetic pack written to
``tmp_path`` with monkeypatched ``PRE_FEATURIZED_ALL``.  The shape of
the synthetic pack matches the production
``molmetal/data/pre_featurized/all_pretrain.pt`` keys.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts import train_dmpnn_multitask as td


# ---------------------------------------------------------------------------
# Synthetic pre-featurized pack
# ---------------------------------------------------------------------------
def _make_synthetic_pack_with_negatives(
    n_rows: int = 24,
    n_atoms: int = 5,
    n_edges: int = 8,
    atom_feat_dim: int = 39,
    edge_feat_dim: int = 6,
) -> tuple[dict, int]:
    """Build a synthetic pack with a deliberate mix of sentinel + canonical.

    Layout (24 rows):

      * 4 sentinel rows with ``metal_idx == -1`` (rows 0..3)
      * 1 row with out-of-range value 7 (row 4)
      * 19 canonical rows with ``metal_idx`` cycling {0,1,2,3,4}
        (rows 5..23)

    Returns
    -------
    (pack, n_canonical)
        ``n_canonical`` is 19 — the count of rows with
        ``metal_idx`` in {0..4}.
    """
    torch.manual_seed(0)
    af = torch.randn(n_rows, n_atoms, atom_feat_dim, dtype=torch.float32)
    ei = torch.zeros(n_rows, 2, n_edges, dtype=torch.int64)
    ea = torch.randn(n_rows, n_edges, edge_feat_dim, dtype=torch.float32)
    mask = torch.ones(n_rows, n_atoms, dtype=torch.bool)

    metal_idx_list: list[int] = []
    # Rows 0..3: sentinel -1 (non-canonical)
    metal_idx_list.extend([-1, -1, -1, -1])
    # Row 4: out-of-range 7 (non-canonical — not in {0..4})
    metal_idx_list.append(7)
    # Rows 5..23: canonical {0..4}
    for i in range(5, n_rows):
        metal_idx_list.append((i - 5) % 5)
    metal_idx = torch.tensor(metal_idx_list, dtype=torch.int64)

    y = torch.randn(n_rows, dtype=torch.float32)
    active = torch.tensor([1.0, 0.0] * (n_rows // 2 + 1), dtype=torch.float32)[:n_rows]
    domain = torch.tensor([0, 1, 2, 3] * (n_rows // 4 + 1), dtype=torch.int64)[:n_rows]
    pack = {
        "atom_features": af,
        "edge_index": ei,
        "edge_attr": ea,
        "atom_mask": mask,
        "metal_idx": metal_idx,
        "y": y,
        "active": active,
        "domain": domain,
        "smiles": ["CCO"] * n_rows,
    }
    n_canonical = sum(1 for m in metal_idx_list if 0 <= m <= 4)
    return pack, n_canonical


@pytest.fixture()
def synthetic_pretrain_pt(tmp_path: Path) -> Path:
    """Write a small all_pretrain.pt with sentinel + out-of-range rows."""
    pack, _ = _make_synthetic_pack_with_negatives(n_rows=24)
    p = tmp_path / "all_pretrain.pt"
    torch.save(pack, p)
    return p


@pytest.fixture()
def patched_loader_path(
    synthetic_pretrain_pt: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Redirect PRE_FEATURIZED_ALL to the synthetic file."""
    monkeypatch.setattr(td, "PRE_FEATURIZED_ALL", synthetic_pretrain_pt)
    return synthetic_pretrain_pt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_filter_removes_metal_idx_minus_one(
    patched_loader_path: Path,
) -> None:
    """Rows with ``metal_idx == -1`` (and any non-canonical value) are dropped.

    We pull every batch from the loader and verify the per-row
    ``atom_features`` count matches the canonical-only count (19),
    i.e. the 4 sentinel rows and the 1 out-of-range row (5 total) are
    filtered out.
    """
    loader = td.build_pretrain_dataloader(
        batch_size=8, shuffle=False, seed=0, path=patched_loader_path
    )
    # Dataset tensors live in train_dmpnn_multitask.td — pull them out
    # via the underlying TensorDataset.
    ds = loader.dataset
    assert isinstance(ds, torch.utils.data.TensorDataset)
    tensors = ds.tensors
    # 5 keys_tensors + 1 optional 'smiles' key (smiles stays on the
    # pack but is NOT in keys_tensors, so it is not part of the
    # dataset).  All shape[0] values must agree.
    n_after = tensors[0].size(0)
    # Source had 24 rows; 4 sentinel + 1 out-of-range = 5 dropped;
    # 19 should remain.
    assert n_after == 19, (
        f"Expected 19 canonical rows after filter, got {n_after}; "
        f"source had 24 rows with 5 non-canonical entries."
    )

    # Sanity-check via the loader iteration as well — total samples
    # seen across all batches must equal 19.
    seen = 0
    for batch in loader:
        seen += batch[0].size(0)
    assert seen == 19


def test_filter_normalizes_metal_idx_range(
    patched_loader_path: Path,
) -> None:
    """Surviving ``metal_idx`` values lie in the canonical {0..4} range.

    After the filter, every remaining row must satisfy
    ``0 <= metal_idx <= 4``.  This guards against stray out-of-range
    values that may sneak into future regenerations of the pretrain
    corpus.
    """
    loader = td.build_pretrain_dataloader(
        batch_size=8, shuffle=False, seed=0, path=patched_loader_path
    )
    ds = loader.dataset
    assert isinstance(ds, torch.utils.data.TensorDataset)
    tensors = ds.tensors
    # keys_tensors index 4 = metal_idx
    metal_idx = tensors[4]
    assert torch.all(metal_idx >= 0), (
        f"Negative metal_idx survived filter: " f"{metal_idx[metal_idx < 0].tolist()}"
    )
    assert torch.all(metal_idx <= 4), (
        f"Out-of-range metal_idx survived filter: "
        f"{metal_idx[metal_idx > 4].tolist()}"
    )
    # All five canonical values should be represented in the surviving
    # rows (cycle 0..4 across 19 rows).
    unique_vals = sorted(set(metal_idx.tolist()))
    assert unique_vals == [
        0,
        1,
        2,
        3,
        4,
    ], f"Expected canonical {{0..4}}, got {unique_vals}"


def test_filter_logs_filtered_count(
    patched_loader_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The loader prints the number of filtered non-canonical rows.

    Captures stdout and asserts the canonical
    ``[dmpnn_multitask] pretrain: filtered N non-canonical rows ...``
    message is emitted with the right count.
    """
    loader = td.build_pretrain_dataloader(
        batch_size=8, shuffle=False, seed=0, path=patched_loader_path
    )
    # Force the loader to materialise so the print() actually flushes.
    out = capsys.readouterr().out
    # Trigger dataset access (some print buffering paths).
    _ = loader.dataset[0]

    # Search the captured stdout for the canonical log line.
    expected_fragment = "[dmpnn_multitask] pretrain: filtered 5 non-canonical rows"
    assert (
        expected_fragment in out
    ), f"Expected log line '{expected_fragment}' not found in stdout:\n{out}"
    # Also assert the surviving row count is reported.
    assert "remaining=19" in out, f"Expected 'remaining=19' in stdout:\n{out}"


# ---------------------------------------------------------------------------
# Backward-compat smoke test — when the source has only canonical rows
# the filter is a no-op (no drop) and downstream behaviour is unchanged.
# ---------------------------------------------------------------------------
def test_filter_is_noop_when_all_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the source has no sentinel rows, the filter drops 0 rows.

    Regression guard against accidentally breaking the
    canonical-only pretrain slice (e.g. ``chemprop_v2_mt_smoke_100``).
    """
    torch.manual_seed(1)
    n_rows, n_atoms, n_edges = 12, 4, 6
    pack = {
        "atom_features": torch.randn(n_rows, n_atoms, 39, dtype=torch.float32),
        "edge_index": torch.zeros(n_rows, 2, n_edges, dtype=torch.int64),
        "edge_attr": torch.randn(n_rows, n_edges, 6, dtype=torch.float32),
        "atom_mask": torch.ones(n_rows, n_atoms, dtype=torch.bool),
        "metal_idx": torch.tensor([i % 5 for i in range(n_rows)], dtype=torch.int64),
        "y": torch.randn(n_rows, dtype=torch.float32),
        "active": torch.tensor([1.0, 0.0] * (n_rows // 2 + 1), dtype=torch.float32)[
            :n_rows
        ],
        "domain": torch.tensor([i % 4 for i in range(n_rows)], dtype=torch.int64),
        "smiles": ["CCO"] * n_rows,
    }
    p = tmp_path / "all_pretrain.pt"
    torch.save(pack, p)
    monkeypatch.setattr(td, "PRE_FEATURIZED_ALL", p)

    loader = td.build_pretrain_dataloader(batch_size=4, shuffle=False, seed=0, path=p)
    ds = loader.dataset
    assert isinstance(ds, torch.utils.data.TensorDataset)
    assert ds.tensors[0].size(0) == n_rows  # no drop
