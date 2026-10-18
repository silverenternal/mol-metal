"""Tests for FIX-M1's D-MPNN surface wiring (``msrs_dmpnn_surface``).

The script itself is a runner; what must be right is the *data contract* around
the model:

* the three arms differ only in the metal block (the control is a real control),
* graphs are deduplicated on the same key the cohort uses,
* the conditioning vector replaces — never appends to — the graph's metal vector,
* a bad config fails loudly instead of training with a default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data.metal_graph import METALS5, METAL_FEAT_DIM  # noqa: E402
from molmetal.scripts.msrs_dmpnn_surface import (  # noqa: E402
    attach_conditioning,
    build_graph_cache,
    cell_one_hot,
    conditioning_matrix,
    load_config,
    split_groups,
)


def _frame(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ligand_canonical": ["CCO"] * n,
            "ligand_main_fragment": ["CCO", "CCO", "CCN", "CCN"][:n],
            "metal": ["Ru", "Ru", "Ir", "Ir"][:n],
            "oxidation_state": [2, 2, 3, 3][:n],
            "charge_complex": [0, 0, 1, 1][:n],
            "cell_line_key": ["A549", "HeLa", "A549", "other"][:n],
        }
    )


def _vocab() -> list:
    return ["A549", "HeLa"]


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------
def test_cell_one_hot_puts_unknown_in_other():
    oh = cell_one_hot(_frame(), _vocab())
    assert oh.shape == (4, 3)
    assert oh[0, 0] == 1.0  # A549
    assert oh[1, 1] == 1.0  # HeLa
    assert oh[3, 2] == 1.0  # other
    assert np.all(oh.sum(axis=1) == 1.0)


def test_blind_arm_zeroes_the_whole_metal_block():
    frame = _frame()
    oh = cell_one_hot(frame, _vocab())
    cond = conditioning_matrix(frame, "blind", oh)
    assert cond.shape == (4, METAL_FEAT_DIM + 3)
    assert np.all(cond[:, :METAL_FEAT_DIM] == 0.0)
    assert np.array_equal(cond[:, METAL_FEAT_DIM:], oh)


def test_identity_arm_keeps_only_the_identity_one_hot():
    frame = _frame()
    oh = cell_one_hot(frame, _vocab())
    cond = conditioning_matrix(frame, "identity", oh)
    metal = cond[:, :METAL_FEAT_DIM]
    assert metal[0, METALS5.index("Ru")] == 1.0
    assert metal[2, METALS5.index("Ir")] == 1.0
    # Everything after the identity block (state / geometry) is zero.
    assert np.all(metal[:, len(METALS5) :] == 0.0)
    assert np.array_equal(cond[:, METAL_FEAT_DIM:], oh)


def test_state_arm_differs_from_identity_only_after_the_identity_block():
    frame = _frame()
    oh = cell_one_hot(frame, _vocab())
    ident = conditioning_matrix(frame, "identity", oh)
    state = conditioning_matrix(frame, "state", oh)
    assert np.array_equal(ident[:, : len(METALS5)], state[:, : len(METALS5)])
    assert not np.array_equal(ident[:, len(METALS5) : METAL_FEAT_DIM], state[:, len(METALS5) : METAL_FEAT_DIM])
    # Cell block is arm-independent.
    assert np.array_equal(ident[:, METAL_FEAT_DIM:], state[:, METAL_FEAT_DIM:])


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
def test_graph_cache_deduplicates_on_complex_metal_state_charge():
    frame = _frame()
    cache = build_graph_cache(frame, use_metal_node=True)
    # 4 rows, but only (CCO,Ru,2,0) and (CCO,Ir,3,1) are distinct keys.
    assert len(cache) == 2
    assert all(v is not None for v in cache.values())


def test_attach_conditioning_replaces_the_metal_vector():
    frame = _frame()
    cache = build_graph_cache(frame, use_metal_node=True)
    key = next(iter(cache))
    graph = cache[key]
    cond = np.arange(METAL_FEAT_DIM + 3, dtype=np.float32)
    out = attach_conditioning(graph, cond)
    assert out is not None
    af, bf, es, ed, mf = out
    assert np.array_equal(mf, cond)
    assert af.shape == graph[0].shape
    assert bf.shape == graph[1].shape
    assert np.array_equal(es, graph[2])
    assert np.array_equal(ed, graph[3])


def test_attach_conditioning_passes_none_through():
    assert attach_conditioning(None, np.zeros(3, dtype=np.float32)) is None


# ---------------------------------------------------------------------------
# Splits / config
# ---------------------------------------------------------------------------
def test_split_groups_uses_the_declared_column():
    frame = _frame()
    assert list(split_groups(frame, "fragment_grouped")) == ["CCO", "CCO", "CCN", "CCN"]
    assert list(split_groups(frame, "canonical_grouped")) == ["CCO"] * 4
    with pytest.raises(ValueError):
        split_groups(frame, "doi_grouped")


def _write_config(tmp_path: Path, **overrides) -> Path:
    import yaml

    cfg = {
        "data": {"long": "x.parquet", "endpoint": "potency", "top_cell_lines": 40, "limit": 0},
        "model": {"hidden": 128, "depth": 3, "dropout": 0.1},
        "train": {
            "epochs": 1,
            "batch_size": 8,
            "lr": 0.001,
            "weight_decay": 0.0,
            "patience": 1,
            "inner_val_fraction": 0.1,
            "seed": 1,
        },
        "compute": {"device": "cpu", "num_threads": 1, "deterministic": True},
        "eval": {"splits": ["fragment_grouped", "canonical_grouped"], "baseline_arm": "complex"},
        "arms": {
            "ligand_cell": {
                "metal_mode": "blind",
                "use_metal_node": False,
                "metal_injection": "readout",
            }
        },
        "output": {"path": "metrics/x.json"},
    }
    cfg.update(overrides)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_load_config_accepts_the_declared_schema(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    assert cfg["compute"]["device"] == "cpu"


def test_load_config_rejects_unknown_metal_mode(tmp_path):
    bad = {
        "arms": {
            "a": {"metal_mode": "quantum", "use_metal_node": True, "metal_injection": "message"}
        }
    }
    with pytest.raises(ValueError, match="metal_mode"):
        load_config(_write_config(tmp_path, **bad))


def test_load_config_rejects_missing_section(tmp_path):
    import yaml

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"data": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required section"):
        load_config(path)
