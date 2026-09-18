"""Guard the real-data ablation against mixed molecular metadata and fallback."""

import pytest
import torch
from types import SimpleNamespace

from data._base import MoleculeSample
from molmetal.scripts.r10_ot_qm9_3seed import compatible_batch, sample_hash, select_groups, write_tables, forward_loss


def sample(atoms, offset):
    coords = torch.arange(len(atoms) * 3, dtype=torch.float32).reshape(-1, 3) / 10 + offset
    return MoleculeSample(coords=coords, atom_types=torch.tensor(atoms), label=torch.zeros(12))


def test_selection_is_deterministic_and_uses_separate_input_splits():
    train = [sample([6, 1, 1, 1], i) for i in range(10)]
    val = [sample([6, 1, 1, 1], i + 100) for i in range(5)]
    first = select_groups(train, val, n_groups=1)
    assert first == select_groups(train, val, n_groups=1)
    train_indices, val_indices = first
    assert train_indices == [list(range(8))]
    assert val_indices == [list(range(4))]
    assert not ({sample_hash(train[i]) for i in train_indices[0]} &
                {sample_hash(val[i]) for i in val_indices[0]})


def test_compatible_graph_metadata_is_invariant_to_coordinate_repairing():
    a = sample([6, 1, 1, 1], 0)
    b = sample([6, 1, 1, 1], 3)
    # Real molecules have distinct geometries, not only different centers.
    b.coords[1, 0] += 2
    batch = compatible_batch([a, b], torch.device("cpu"), 1.0)
    swap = torch.tensor([1, 0])
    for key in ("atomic_numbers", "node_mask", "edge_mask", "edge_index"):
        assert torch.equal(batch[key], batch[key][swap])
    assert batch["edge_mask"].sum(1).tolist() == [12, 12]
    assert not torch.equal(batch["positions"][0], batch["positions"][1])
    torch.testing.assert_close(batch["positions"].sum(1), torch.zeros(2, 3), atol=1e-5, rtol=0)


def test_mixed_atom_signatures_fail_instead_of_corrupting_training():
    with pytest.raises(ValueError, match="identical ordered atom types"):
        compatible_batch([sample([6, 1, 1, 1], 0), sample([7, 1, 1, 1], 0)],
                         torch.device("cpu"), 1.0)


def test_insufficient_compatible_data_does_not_generate_synthetic_rows():
    with pytest.raises(ValueError, match="Insufficient compatible real-QM9"):
        select_groups([sample([6, 1, 1, 1], 0)], [sample([6, 1, 1, 1], 1)], n_groups=1)


def test_failure_report_does_not_assert_unverified_data_provenance(tmp_path):
    prefix = tmp_path / "failed"
    write_tables({"status": "failed", "command": "example", "total_wall_seconds": 0,
                  "runs": [], "failures": [{"type": "FileNotFoundError", "message": "Missing real data"}]},
                 prefix)
    text = prefix.with_suffix(".md").read_text()
    assert "Missing real data" in text
    assert "provenance was not fully validated" in text
    assert "tensor was matched byte-for-byte" not in text


def test_unconverged_soft_plan_is_rejected_before_loss_evaluation():
    pytest.importorskip("ot")
    torch.manual_seed(5)
    items = [MoleculeSample(coords=torch.randn(4, 3), atom_types=torch.tensor([6, 1, 1, 1]))
             for _ in range(8)]
    batch = compatible_batch(items, torch.device("cpu"), 1.0)

    def must_not_execute(*args, **kwargs):
        pytest.fail("Rejected soft plan must not reach the training objective")

    diagnostics = []
    with pytest.raises(RuntimeError, match="Coupling rejected before optimizer update"):
        forward_loss(must_not_execute, batch, 10000, torch.device("cpu"), diagnostics,
                     {"reg": 0.05, "max_iter": 1}, max_marginal_error=1e-3)
    assert diagnostics[0]["marginal_max_abs_error"] > 1e-3


def test_converged_plan_passes_the_explicit_training_gate():
    pytest.importorskip("ot")
    torch.manual_seed(5)
    items = [MoleculeSample(coords=torch.randn(4, 3), atom_types=torch.tensor([6, 1, 1, 1]))
             for _ in range(8)]
    batch = compatible_batch(items, torch.device("cpu"), 1.0)
    diagnostics = []

    def objective(x0, x1, **kwargs):
        return SimpleNamespace(loss=(x1 - x0).square().mean())

    loss = forward_loss(objective, batch, 10000, torch.device("cpu"), diagnostics,
                        {"reg": 1.0, "max_iter": 2000}, max_marginal_error=1e-3)
    assert torch.isfinite(loss)
    assert diagnostics[0]["effective_backend"] == "pot_sinkhorn_log"
    assert diagnostics[0]["marginal_max_abs_error"] <= 1e-3
