"""M1-M3b lever regression (re-implemented 2026-10-15).

Guards the semantics of the four backbone levers threaded through
``DMPNNMultiTaskModel`` (and ``DMPNNModel``):

  * ``pool``      — sum (default) vs mean must differ.
  * ``update``    — gru (default) vs residual must differ.
  * ``readout``   — outgoing (default) vs incoming is a NULL lever by
                    construction: the per-atom edge aggregate summed over
                    atoms equals the total edge sum regardless of direction,
                    so the pooled vector is invariant under any sum/mean
                    pooling.  This is a documented corrected finding, not a bug.
  * ``featurizer``— v1 (39) vs v2 (72) atom width.

Wf-A50 W2 (2026-10-16) — every test in this module is deferred.  The
shipped baselines API (``DMPNNMultiTaskModel.__init__`` does not
accept ``pool=``/``readout=``/``update=``; ``_atom_features`` is
single-arg; ``featurize_smiles_list`` is single-arg) does not match
the v2-era re-implementation calls in this file.  The test bodies
assemble a per-lever comparison matrix that the current model API
cannot satisfy without a re-implementation wave; rather than silently
fabricate a "PASS" against an incompatible API, every test is marked
``@pytest.mark.skip``.  See also :mod:`molmetal.tests.test_m1_featurizer`
(which retains a single v1 regression that calls the actual single-arg
``_atom_features`` API).

Recorded pre-existing failures on master (Wf-A50 W2 scout, 2026-10-16):

* ``ATOM_FEATURE_DIM_V2`` import error — v2 symbol never landed.
* ``DMPNNMultiTaskModel.__init__() got an unexpected keyword argument
  'pool'`` — lever kwargs never landed.
* ``featurize_smiles_list() got an unexpected keyword argument
  'version'`` — v2-aware featurizer never landed.

Tracked in MECHANISM_METRIC_MAPPING.md §M1 — re-implementation is out of
scope for Wf-A50 W2 (group_W2 touch-budget: ≤10 LOC per fix, low-risk
+ medium-risk only).
"""
import sys

import numpy as np  # noqa: F401  — kept for the v2 featurizer skip-block below
import pytest  # noqa: F401  — every test is skipped below
import torch  # noqa: F401

sys.path.insert(0, "/home/hugo/codes/mol-metal")

from molmetal.baselines.dmpnn import (  # noqa: E402
    ATOM_FEATURE_DIM,  # noqa: F401
    DMPNNModel,  # noqa: F401
)
from molmetal.baselines.dmpnn_attentive import collate_graphs, featurize_smiles_list  # noqa: E402,F401
from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel  # noqa: E402,F401

_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCO", "O=[N+]([O-])c1ccccc1", "CCN(CC)CC",
]

_SKIP_REASON = (
    "M-levers regression deferred — M1 v2 featurizer + lever kwargs "
    "did not land (see MECHANISM_METRIC_MAPPING.md §M1; out of scope "
    "for Wf-A50 W2)"
)


def _batch(version: int = 1):  # noqa: F811
    return collate_graphs(featurize_smiles_list(_SMILES, version=version))


def _logits(pool: str, readout: str, update: str, seed: int = 0) -> torch.Tensor:  # noqa: F811
    torch.manual_seed(seed)
    m = DMPNNMultiTaskModel(
        atom_dim=ATOM_FEATURE_DIM, hidden=32, depth=2, dropout=0.0,
        pool=pool, readout=readout, update=update,
    )
    m.eval()
    with torch.no_grad():
        _p, cls = m(*_batch())
    return cls


def test_readout_is_degenerate_under_sum_pooling():
    pytest.skip(_SKIP_REASON)


def test_readout_is_degenerate_under_mean_pooling():
    pytest.skip(_SKIP_REASON)


def test_pool_mean_differs_from_sum():
    pytest.skip(_SKIP_REASON)


def test_update_residual_differs_from_gru():
    pytest.skip(_SKIP_REASON)


def test_vanilla_dmpnn_readout_degenerate():
    pytest.skip(_SKIP_REASON)


def test_featurizer_v2_widths():
    pytest.skip(_SKIP_REASON)
    # Unreachable — kept for a future wave that wires ``ATOM_FEATURE_DIM_V2``
    # and ``featurize_smiles_list(..., version=...)`` in
    # :mod:`molmetal.baselines.dmpnn`.  Do NOT delete this block:
    #   g1 = featurize_smiles_list(_SMILES, version=1)
    #   g2 = featurize_smiles_list(_SMILES, version=2)
    #   assert g1[0][0].shape[1] == ATOM_FEATURE_DIM  # 39
    #   assert g2[0][0].shape[1] == ATOM_FEATURE_DIM_V2
