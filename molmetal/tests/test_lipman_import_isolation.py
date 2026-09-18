"""Official Lipman and local MolFlow must coexist throughout one process."""
from __future__ import annotations

import builtins
import importlib
from pathlib import Path
import pickle
import subprocess
import sys

import pytest
import torch

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter, _import_fm_lib


REFERENCE = Path(__file__).resolve().parents[1] / "references" / "flow_matching"


@pytest.fixture(autouse=True)
def preserve_torch_rng():
    # Adapter construction and real loss sampling must not change the random
    # starting point of unrelated training tests in the same pytest process.
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(123)
        yield


def test_repeated_adapter_setup_preserves_local_loss_and_official_execution():
    local = importlib.import_module("flow_matching")
    local_loss = importlib.import_module("flow_matching.loss")
    original_modules = {name: module for name, module in sys.modules.items()
                        if name == "flow_matching" or name.startswith("flow_matching.")}
    original_path = list(sys.path)
    original_import = builtins.__import__
    from models.velocity_net import VelocityNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VelocityNet(hidden_dim=8, n_layers=1).to(device)
    objective = local_loss.ConditionalFlowMatchingLoss(net)
    adapter = LipmanFlowMatchingAdapter(hidden_dim=8, n_layers=1, ref_repo_path=str(REFERENCE))
    identities = None
    for _ in range(3):
        adapter.setup(str(device))
        classes = _import_fm_lib(REFERENCE)
        if identities is None:
            identities = classes
        assert all(a is b for a, b in zip(classes, identities))
        assert type(adapter.path) is classes[0]
        assert type(adapter.scheduler) is classes[1]
        assert adapter._ODESolver is classes[2]
        assert importlib.import_module("flow_matching") is local
        from flow_matching.loss import ConditionalFlowMatchingLoss
        assert ConditionalFlowMatchingLoss is local_loss.ConditionalFlowMatchingLoss
        assert all(sys.modules.get(name) is value for name, value in original_modules.items())
        assert sys.path == original_path
        assert builtins.__import__ is original_import

        # Execute the real local loss after official initialization, including
        # gradients, so retaining a stale class name alone cannot pass.
        x0, x1 = torch.zeros(2, 2, 3, device=device), torch.ones(2, 2, 3, device=device)
        edges = torch.tensor([[[0, 1], [1, 0]], [[0, 1], [1, 0]]], device=device)
        result = objective(x0, x1, h_node=torch.ones(2, 2, 8, device=device), edge_index=edges)
        assert isinstance(result, local_loss.LossOutput)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        net.zero_grad()

        sample = adapter.path.sample(x_0=x0, x_1=x1, t=torch.tensor([.25, .25], device=device))
        torch.testing.assert_close(sample.x_t, x1 * .25)
        torch.testing.assert_close(sample.dx_t, x1)
        solver = classes[2](velocity_model=lambda x, t: torch.ones_like(x))
        endpoint = solver.sample(x_init=x0, step_size=.25, method="euler")
        torch.testing.assert_close(endpoint, x1)
        restored = pickle.loads(pickle.dumps(adapter.path))
        assert type(restored) is classes[0] and type(restored.scheduler) is classes[1]


@pytest.mark.parametrize("first", ["local", "reference"])
def test_both_import_orders_in_fresh_process(first):
    # Fresh interpreters cover reference-first use despite pytest collecting
    # other local-loss tests first. No sys.modules cleanup conceals pollution.
    code = f'''
import importlib, sys
from molmetal.adapters.flow_matching_lipman import _import_fm_lib
path_before = list(sys.path)
if {first!r} == "local":
    local = importlib.import_module("flow_matching.loss")
classes = _import_fm_lib({str(REFERENCE)!r})
from flow_matching.loss import ConditionalFlowMatchingLoss
assert ConditionalFlowMatchingLoss.__module__ == "flow_matching.loss"
assert all(c.__module__.startswith("molmetal.adapters.flow_matching_lipman._reference.") for c in classes)
assert all(a is b for a,b in zip(classes, _import_fm_lib({str(REFERENCE)!r})))
assert sys.path == path_before
'''
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_reference_pickle_loads_in_fresh_process_without_adapter_setup(tmp_path):
    affine, scheduler, _, _ = _import_fm_lib(REFERENCE)
    payload = tmp_path / "official_path.pkl"
    payload.write_bytes(pickle.dumps(affine(scheduler())))
    code = f'''
import pickle, torch
from pathlib import Path
path = pickle.loads(Path({str(payload)!r}).read_bytes())
from molmetal.adapters.flow_matching_lipman import _import_fm_lib
affine, scheduler, _, _ = _import_fm_lib({str(REFERENCE)!r})
assert type(path) is affine and type(path.scheduler) is scheduler
sample = path.sample(x_0=torch.zeros(1, 2), x_1=torch.ones(1, 2), t=torch.tensor([.5]))
torch.testing.assert_close(sample.x_t, torch.full((1,2), .5))
from flow_matching.loss import ConditionalFlowMatchingLoss
assert ConditionalFlowMatchingLoss.__module__ == "flow_matching.loss"
'''
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_reference_location_change_rejected_without_changing_public_imports(tmp_path):
    classes = _import_fm_lib(REFERENCE)
    local = importlib.import_module("flow_matching.loss")
    (tmp_path / "flow_matching").mkdir()
    (tmp_path / "flow_matching" / "__init__.py").write_text("")
    with pytest.raises(RuntimeError, match="cannot switch"):
        _import_fm_lib(tmp_path)
    assert importlib.import_module("flow_matching.loss") is local
    assert all(a is b for a, b in zip(classes, _import_fm_lib(REFERENCE)))
