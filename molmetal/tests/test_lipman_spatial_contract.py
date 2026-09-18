"""Nonzero velocity and actual sampling obey rigid spatial transformations."""
import pytest
import torch
from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.domain import Molecule
from molmetal.ports import GenerationConfig


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_nonzero_velocity_equivariance_and_first_update_gradients(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('ROCm unavailable')
    torch.manual_seed(42)
    adapter=LipmanFlowMatchingAdapter(hidden_dim=16,n_layers=2,metal_prior_weight=0.)
    adapter.setup(device);field=adapter.velocity_field
    x=torch.randn(2,6,3,device=device);atoms=torch.zeros(2,6,dtype=torch.long,device=device)
    edges=adapter._make_dummy_edge_index(2,6,device);t=torch.tensor([.2,.7],device=device)
    initial=field(x,atoms,edges,t)['vel']
    assert initial.count_nonzero()==0
    target=torch.randn_like(initial)
    (initial-target).square().sum().backward()
    assert field.vel_head.weight.grad.norm()>0
    assert field.layers[-1].msg_vector_head.weight.grad.norm()>0
    # Activate both heads so zero outputs cannot trivially pass equivariance.
    with torch.no_grad():
        field.vel_head.weight.normal_(std=.03)
        field.layers[-1].msg_vector_head.weight.normal_(std=.03)
    field.eval()
    rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],device=device)
    shift=torch.tensor([3.,-2.,1.],device=device)
    before=field(x,atoms,edges,t)
    after=field(x@rotation+shift,atoms,edges,t)
    assert before['vel'].norm()>0
    torch.testing.assert_close(after['vel'],before['vel']@rotation,rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(after['atom_logits'],before['atom_logits'],rtol=2e-5,atol=2e-5)
    # The architecture bounds spatial growth using learned scalar gates,
    # independently of their magnitude; it never clips output coordinates.
    large=x*100
    velocity=field(large,atoms,edges,t)['vel']
    centroid_radius=(large-large.mean(1,keepdim=True)).norm(dim=-1)
    mean_pair_distance=torch.cdist(large,large).sum(-1)/5
    assert torch.isfinite(velocity).all()
    assert (velocity.norm(dim=-1)<=centroid_radius+mean_pair_distance+1e-3).all()


def test_actual_sampler_transforms_with_initial_noise(monkeypatch):
    if not torch.cuda.is_available():pytest.skip('ROCm unavailable')
    torch.manual_seed(42)
    adapter=LipmanFlowMatchingAdapter(hidden_dim=16,n_layers=1,metal_prior_weight=0.)
    adapter.setup('cuda:0')
    mol=Molecule(torch.randn(6,3),torch.tensor([6,6,7,8,6,9]),torch.zeros(2,0,dtype=torch.long),
                 torch.zeros(0,dtype=torch.long),torch.zeros(6,dtype=torch.long))
    for _ in range(3):adapter.train_step(None,[mol])
    base=torch.randn(2,8,3,device=adapter.device)
    rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],device=adapter.device)
    shift=torch.tensor([3.,-2.,1.],device=adapter.device)
    supplied=[base]
    def noise(*shape,**kwargs):
        assert shape==(2,8,3)
        return supplied[0].clone()
    monkeypatch.setattr(torch,'randn',noise)
    config=GenerationConfig(n_samples=2,n_steps=16,seed=123)
    before=adapter.generate(None,config)
    supplied[0]=base@rotation+shift
    after=adapter.generate(None,config)
    for a,b in zip(before,after):
        torch.testing.assert_close(b.coords,a.coords@rotation.cpu()+shift.cpu(),rtol=2e-5,atol=2e-5)
        assert torch.equal(a.atom_types,b.atom_types)
