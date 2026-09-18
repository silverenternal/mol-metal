"""Assay predictions must not depend on unrelated padded molecules."""
import pytest
import torch
from molmetal.baselines.dmpnn_attentive import AttentiveDMPNNModel, collate_graphs, featurize_smiles_list


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_prediction_and_parameter_gradients_independent_of_batch_padding(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('GPU not available')
    with torch.random.fork_rng():
        torch.manual_seed(41)
        model=AttentiveDMPNNModel(hidden=16,depth=1,dropout=0).to(device).eval()
    graphs=featurize_smiles_list(['CCO','c1ccc(CCOCCOCCOCCO)cc1'])
    batch=[t.to(device) for t in collate_graphs(graphs)]
    actual=model(*batch)
    separate=torch.cat([model(*[t.to(device) for t in collate_graphs([g])]) for g in graphs])
    torch.testing.assert_close(actual,separate,atol=1e-6,rtol=1e-5)
    actual.sum().backward()
    gradients={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    separate.sum().backward()
    for name,param in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(param.grad,gradients[name],atol=2e-6,rtol=2e-5)


def test_real_zero_feature_edges_are_not_discarded_and_single_atoms_are_predicted():
    with torch.random.fork_rng():
        torch.manual_seed(41)
        model=AttentiveDMPNNModel(hidden=16,depth=1,dropout=0).eval()
    bonded,atom=featurize_smiles_list(['CC','[Na+]'])
    bonded=list(bonded)
    bonded[1]=bonded[1]*0
    data=collate_graphs([tuple(bonded),atom])
    actual=model(*data)
    reference=model._forward_one_mol(*[torch.as_tensor(x) for x in bonded])
    torch.testing.assert_close(actual[0],reference)
    assert torch.isfinite(actual).all()
    model.zero_grad(set_to_none=True)
    actual[1].backward()
    assert model.atom_emb.weight.grad is not None
    assert model.atom_emb.weight.grad.abs().sum()>0
