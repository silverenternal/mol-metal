"""Real sampler contracts: explicit topology, all graphs, no silent donor inference."""
import pytest
import torch

from molmetal.ports import GenerationConfig
from molmetal.molmetal_lam.priors.metal_geometry import MetalGeometryPrior, square_planar_penalty


def complex_data(device='cpu'):
    coords = torch.tensor([[0.,0.,0.],[2.,0.,0.],[-2.,0.,0.],[0.,2.,0.],[0.,-2.,0.],[.8,.8,.8]], device=device)
    atoms = torch.tensor([78,7,7,17,17,6], device=device)
    edges = torch.tensor([[1,2,3,4],[0,0,0,0]], device=device)
    return coords, atoms, edges


def test_explicit_donors_ignore_nearby_non_donor_and_edge_order():
    x, atoms, edges = complex_data()
    prior = MetalGeometryPrior()
    x.requires_grad_()
    types = torch.full((4,),2)
    baseline = prior.prior_loss(x,atoms,types,edge_index=edges)
    assert baseline < .001
    assert square_planar_penalty(x,edges,types.bool(),atoms) < .001
    changed = x.detach().clone(); changed[5] = torch.tensor([1.2,.2,1.])
    torch.testing.assert_close(prior.prior_loss(changed,atoms,types,edge_index=edges.flip(1)),baseline)
    gradient = torch.autograd.grad(baseline,x)[0]
    assert gradient[5].count_nonzero() == 0
    with pytest.raises(ValueError,match='Explicit edge_index'):
        prior.prior_loss(x,atoms,types)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_batched_prior_loss_has_independent_graph_gradients(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('ROCm GPU unavailable')
    from molmetal.adapters.flow_matching_lipman import EGNNVelocityField
    x, atoms, edges = complex_data(device)
    second = x.clone(); second[1,2] = 1.
    batch = torch.stack([x,second]).requires_grad_()
    field = EGNNVelocityField(hidden_dim=16,n_layers=1).to(device)
    types = torch.full((2,4),2,device=device)
    loss = field.apply_metal_geometry_step(batch,atoms.expand(2,-1),types,step=0,weight=1.,edge_index=edges)
    expected = torch.stack([MetalGeometryPrior().prior_loss(v,atoms,types[0],edge_index=edges) for v in batch]).mean()
    torch.testing.assert_close(loss,expected)
    grad = torch.autograd.grad(loss,batch)[0]
    assert grad[1].norm() > 0
    assert grad[:,5].count_nonzero() == 0


def test_actual_rocm_sampler_prior_active_on_every_graph_and_reduces_energy():
    if not torch.cuda.is_available():
        pytest.skip('ROCm GPU unavailable')
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
    adapter = LipmanFlowMatchingAdapter(hidden_dim=16,n_layers=1,metal_prior_weight=.2,metal_prior_k_every=1)
    adapter.setup('cuda:0')
    _, atoms, edges = complex_data()
    config = GenerationConfig(n_samples=2,n_steps=20,seed=42,conditioning={
        'fixed_atom_types':atoms.tolist(),'dative_edge_index':edges.tolist()})
    adapter._metal_prior_weight=0.
    off = adapter.generate(None,config)
    adapter._metal_prior_weight=.2
    on = adapter.generate(None,config)
    diagnostics = adapter.get_metadata()['metal_prior_last_generation']
    assert diagnostics['status']=='active'
    assert diagnostics['gradient_evaluations']==20
    assert diagnostics['nonzero_gradient_graph_updates']==40
    prior = MetalGeometryPrior()
    for before,after in zip(off,on):
        assert torch.equal(after.atom_types,atoms)
        types = torch.full((4,),2)
        assert prior.prior_loss(after.coords,atoms,types,edge_index=edges) < prior.prior_loss(before.coords,atoms,types,edge_index=edges)
    adapter.generate(None,GenerationConfig(n_samples=1,n_steps=2))
    assert adapter.last_metal_prior_diagnostics['status']=='inactive'
    with pytest.raises(ValueError,match='requires fixed_atom_types'):
        adapter.generate(None,GenerationConfig(n_samples=1,n_steps=2,conditioning={'dative_edge_index':edges.tolist()}))


def test_pocket_vocabulary_fails_before_gpu_embedding():
    from molmetal.adapters.flow_matching_lipman import PocketEncoder
    encoder=PocketEncoder(hidden_dim=16,max_atomic_number=10)
    with pytest.raises(ValueError,match='encoder vocabulary'):
        encoder(torch.zeros(1,2,3),torch.tensor([[6,16]]),torch.ones(1,2,dtype=torch.bool))
