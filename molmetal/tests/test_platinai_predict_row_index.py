# SPDX-License-Identifier: Apache-2.0
"""Every row must get written, not just the first batch.

WHY THIS EXISTS
---------------
``train_platinai_4cell.py::_predict`` walked its results back with the
BATCH-local index instead of the row index::

    for vi, k in enumerate(valid_rows):
        out_active[k, line_idx] = ...          # k, not start + k

Every batch therefore overwrote rows ``0..batch_size-1`` and left the
rest at the ``-1`` sentinel, so the caller's ``>= 0`` mask kept exactly
``batch_size`` rows. With ``MT2_BATCH_SIZE = 32`` and ``n_test = 815``,
**every AUC in ``metrics/by_metric/platinai_4cell_auc.json`` was measured
on 32 molecules.** The recorded values are the tell: they are exact
rationals whose denominators are divisors of 32 (0.65625 = 21/32,
0.33714285714285713 = 59/175 = 59/(25*7)), which a 815-row AUC would not
produce.

The test below fails against the old indexing, which is the only thing
that makes it worth having: a coverage assertion on a helper that no
other test touches.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_platinai_4cell_idx_probe",
        REPO / "molmetal" / "scripts" / "train_platinai_4cell.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic_graph(n_atoms: int = 4):
    """A minimal graph tuple the module's collate + model stub can consume."""
    atom_f = np.zeros((n_atoms, 39), dtype=np.float32)
    bond_f = np.zeros((n_atoms - 1, 10), dtype=np.float32)
    e_src = np.arange(n_atoms - 1, dtype=np.int64)
    e_dst = e_src + 1
    return atom_f, bond_f, e_src, e_dst


def test_predict_writes_every_row_not_just_the_first_batch():
    """Rows past the first batch must survive the >= 0 mask."""
    T = _load_trainer()
    import torch

    n = 100  # 3 full batches + remainder at batch_size=32 -> 32/32/32/4
    bs = 32
    graphs = [_synthetic_graph() for _ in range(n)]

    # Row r is active iff r >= n - 50, so a correct run leaves exactly 50
    # rows with a non -1 label. The old indexing left exactly `bs` of them.
    active = np.full((n, T.N_CELL_LINES), 0, dtype=np.int32)
    active[n - 50 :, :] = 1

    class _Stub(torch.nn.Module):
        """Returns a fixed score so every row is 'valid' and predictable."""

        def forward(self, atom_f, bond_f, e_src, e_dst):
            b = atom_f.shape[0]
            # (b, N_CELL_LINES, 2): _predict takes softmax(...)[..., 1], so
            # the logit tensor needs a class axis last.
            return None, torch.zeros(
                (b, T.N_CELL_LINES, 2), dtype=torch.float32
            )

    out_active, out_prob = T._predict(
        _Stub(),
        graphs,
        active,
        device=torch.device("cpu"),
        batch_size=bs,
    )

    kept = (out_active >= 0).sum(axis=1)
    n_kept = int((kept > 0).sum())
    assert n_kept == n, (
        f"_predict kept only {n_kept}/{n} rows; batch_size={bs}. "
        "This is the batch-local-index bug: out_active is indexed by row, "
        "so the write must use start + k, not k."
    )
    # And the kept rows must be the RIGHT ones, not just the right count.
    assert (out_active[: n - 50, 0] == 0).all(), "row indices are misaligned"
    assert (out_active[n - 50 :, 0] == 1).all(), "row indices are misaligned"


def test_predict_row_labels_round_trip_to_the_input():
    """Whatever _predict keeps must carry back the input labels verbatim."""
    T = _load_trainer()
    import torch

    n, bs = 70, 32
    graphs = [_synthetic_graph() for _ in range(n)]
    rng = np.random.default_rng(0)
    active = rng.integers(0, 2, size=(n, T.N_CELL_LINES)).astype(np.int32)

    class _Stub(torch.nn.Module):
        def forward(self, atom_f, bond_f, e_src, e_dst):
            b = atom_f.shape[0]
            # (b, N_CELL_LINES, 2): _predict takes softmax(...)[..., 1], so
            # the logit tensor needs a class axis last.
            return None, torch.zeros(
                (b, T.N_CELL_LINES, 2), dtype=torch.float32
            )

    out_active, _ = T._predict(
        _Stub(), graphs, active, device=torch.device("cpu"), batch_size=bs
    )
    mask = out_active >= 0
    assert mask.any(), "nothing survived; the fixture is wrong, not the code"
    np.testing.assert_array_equal(out_active[mask], active[mask])


def test_predict_single_batch_smaller_than_batch_size():
    """n < batch_size must still keep every row (the degenerate old case)."""
    T = _load_trainer()
    import torch

    n = 5
    graphs = [_synthetic_graph() for _ in range(n)]
    active = np.ones((n, T.N_CELL_LINES), dtype=np.int32)

    class _Stub(torch.nn.Module):
        def forward(self, atom_f, bond_f, e_src, e_dst):
            b = atom_f.shape[0]
            # (b, N_CELL_LINES, 2): _predict takes softmax(...)[..., 1], so
            # the logit tensor needs a class axis last.
            return None, torch.zeros(
                (b, T.N_CELL_LINES, 2), dtype=torch.float32
            )

    out_active, _ = T._predict(
        _Stub(), graphs, active, device=torch.device("cpu"), batch_size=32
    )
    assert int((out_active[:, 0] >= 0).sum()) == n