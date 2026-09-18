"""Train the prediction head without leaking its labels into its input."""
import pytest
import torch
import torch.nn.functional as F

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.domain import Molecule
from molmetal.scripts.r10_cfg_real_crossdocked import decode_distance_graph
from molmetal.ports import GenerationConfig


def test_atom_targets_are_supervised_but_not_supplied_as_features():
    torch.manual_seed(42)
    model=LipmanFlowMatchingAdapter(hidden_dim=16,n_layers=1,atom_loss_weight=1.)
    model.setup()
    atoms=torch.tensor([6,7,8,9])
    mol=Molecule(torch.randn(4,3),atoms,torch.zeros(2,0,dtype=torch.long),
                 torch.zeros(0,dtype=torch.long),torch.zeros(4,dtype=torch.long))
    captured={}
    def observe(module,args,output):
        captured['input_atoms']=args[1].detach().clone()
        captured['logits']=output['atom_logits'].detach().clone()
    hook=model.velocity_field.register_forward_hook(observe)
    before=model.velocity_field.atom_head.weight.detach().clone()
    model.train_step(None,[mol])
    hook.remove()
    assert captured['input_atoms'].count_nonzero()==0
    expected=F.cross_entropy(captured['logits'].flatten(0,1),atoms.to(model.device))
    assert model.last_losses['atom']==pytest.approx(float(expected))
    assert not torch.equal(before,model.velocity_field.atom_head.weight)


def test_decoder_rejects_unsupported_or_disconnected_output_without_repair():
    xyz=[[0.,0.,0.],[20.,0.,0.]]
    assert decode_distance_graph([6,78],xyz)==(None,'atom_outside_training_vocabulary')
    assert decode_distance_graph([6,6],xyz)==(None,'disconnected_distance_graph')


def test_train_sample_use_same_graph_and_gaussian_forward_time_contract():
    model=LipmanFlowMatchingAdapter(hidden_dim=16,n_layers=1)
    model.setup()
    mol=Molecule(torch.randn(8,3),torch.full((8,),6),torch.tensor([[0,2],[1,3]]),
                 torch.tensor([1,2]),torch.zeros(8,dtype=torch.long))
    observed=[]
    original_forward=model.velocity_field.forward
    def capture(*args,**kwargs):
        observed.append(tuple(x.detach().clone() for x in args[:4]))
        return original_forward(*args,**kwargs)
    model.velocity_field.forward=capture
    model.train_step(None,[mol])
    train=observed[0];observed.clear()
    model.generate(None,GenerationConfig(n_samples=1,n_steps=4,seed=123))
    model.velocity_field.forward=original_forward
    sample=observed[0]
    torch.testing.assert_close(sample[2],train[2])
    assert sample[2].shape[-1]==8*7  # same complete graph, no true bonds supplied
    generator=torch.Generator(device=model.device).manual_seed(123)
    torch.testing.assert_close(sample[0],torch.randn(1,8,3,device=model.device,generator=generator))
    assert float(sample[3])==0.
    x0=torch.randn(2,8,3,device=model.device);x1=torch.randn_like(x0)
    result=model.path.sample(x_0=x0,x_1=x1,t=torch.tensor([0.,1.],device=model.device))
    torch.testing.assert_close(result.x_t[0],x0[0])
    torch.testing.assert_close(result.x_t[1],x1[1])
    torch.testing.assert_close(result.dx_t,x1-x0)
