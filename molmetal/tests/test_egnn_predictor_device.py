"""The predictor must allocate and load weights on its resolved device."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from molmetal.adapters import egnn_predictor


@pytest.mark.parametrize("resolved", [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires ROCm/CUDA",
    )),
])
@pytest.mark.parametrize("requested", [None, "cpu"])
def test_setup_allocates_and_loads_on_resolved_device(
    monkeypatch, tmp_path, resolved, requested,
):
    monkeypatch.setattr(egnn_predictor, "get_device", lambda: torch.device(resolved))
    monkeypatch.setattr(egnn_predictor, "_build_egnn_stub", lambda *_args: nn.Linear(3, 2))
    source = nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(0.75)
        source.bias.fill_(-0.5)
    checkpoint = tmp_path / "predictor.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    predictor = egnn_predictor.EGNNPropertyPredictor(
        egnn_predictor.EGNNConfig(checkpoint_path=str(checkpoint)),
    )
    expected = torch.device(resolved if requested is None else requested)

    with patch.object(torch, "load", wraps=torch.load) as load:
        predictor.setup(device=requested)

    assert predictor._model is not None
    assert predictor._checkpoint_loaded
    assert all(parameter.device == expected for parameter in predictor._model.parameters())
    assert torch.device(predictor.get_metadata()["device"]) == expected
    assert torch.device(load.call_args.kwargs["map_location"]) == expected
    torch.testing.assert_close(predictor._model.weight.cpu(), source.weight)
    torch.testing.assert_close(predictor._model.bias.cpu(), source.bias)
