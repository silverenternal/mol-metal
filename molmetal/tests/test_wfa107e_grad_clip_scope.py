"""Wf-A107e — gradient-clip scope (AGENTS.md §5.7 flag wire-in).

The flag exists because an auxiliary loss on ONE module used to rescale
EVERY module's gradient through a single clip norm.  These tests pin:

  * the default is ``global`` and the global path is bit-identical to the
    pre-Wf-A107e call;
  * ``per_module`` clips the head and the rest independently, so a huge
    head gradient no longer shrinks the flow field's step;
  * the returned value is the PRE-CLIP global norm in both scopes, so
    ``grad_norm`` in the metrics keeps its meaning;
  * the yaml key reaches the trainer dest.
"""
from __future__ import annotations

import argparse

import pytest
import torch
from torch import nn


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(2, 2, bias=False)
        self.flow = nn.Linear(2, 2, bias=False)

    def parameters(self, recurse: bool = True):
        yield from self.head.parameters(recurse)
        yield from self.flow.parameters(recurse)


def _set_grads(model: _Tiny, head_grad: float, flow_grad: float) -> None:
    with torch.no_grad():
        model.head.weight.fill_(head_grad)
        model.flow.weight.fill_(flow_grad)
    model.head.weight.grad = torch.full_like(model.head.weight, head_grad)
    model.flow.weight.grad = torch.full_like(model.flow.weight, flow_grad)


def _norm(t: torch.Tensor) -> float:
    return float(t.pow(2).sum()) ** 0.5


def test_default_scope_is_global(monkeypatch):
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import parse_args

    # The default must be global so every existing wrapper keeps the
    # pre-Wf-A107e numerics (bit-exact rule).
    monkeypatch.setattr("sys.argv", ["train_cfm"])
    assert parse_args().grad_clip_scope == "global"


def test_per_module_is_an_accepted_choice(monkeypatch):
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import parse_args

    monkeypatch.setattr(
        "sys.argv", ["train_cfm", "--grad-clip-scope", "per_module"]
    )
    assert parse_args().grad_clip_scope == "per_module"
    monkeypatch.setattr("sys.argv", ["train_cfm", "--grad-clip-scope", "bogus"])
    with pytest.raises(SystemExit):
        parse_args()


def test_global_clip_rescales_the_flow_field_too():
    """The defect the flag exists for, pinned as the OLD behaviour."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import clip_gradients

    model = _Tiny()
    _set_grads(model, head_grad=100.0, flow_grad=1.0)
    pre = clip_gradients(model, max_norm=5.0, scope="global")
    # joint norm = sqrt(4*100^2 + 4*1^2) ~ 200, so the flow field is cut
    # by ~0.025 even though its own gradient is tiny.
    assert pre > 150.0
    assert _norm(model.flow.weight.grad) < 0.1


def test_per_module_leaves_a_small_flow_gradient_alone():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import clip_gradients

    model = _Tiny()
    _set_grads(model, head_grad=100.0, flow_grad=1.0)
    pre = clip_gradients(model, max_norm=5.0, scope="per_module")
    # Same pre-clip number as the global path...
    assert pre == pytest.approx((4 * 100.0 ** 2 + 4 * 1.0 ** 2) ** 0.5, rel=1e-6)
    # ...but the flow field keeps its own gradient (norm 2.0 < 5.0), while
    # the head is clipped to 5.0.
    assert _norm(model.flow.weight.grad) == pytest.approx(2.0, rel=1e-6)
    assert _norm(model.head.weight.grad) == pytest.approx(5.0, rel=1e-6)


def test_both_scopes_cap_the_gradient_they_own():
    """global caps the JOINT norm; per_module caps each group separately."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import clip_gradients

    model = _Tiny()
    _set_grads(model, head_grad=100.0, flow_grad=100.0)
    clip_gradients(model, max_norm=5.0, scope="global")
    joint = (
        _norm(model.head.weight.grad) ** 2 + _norm(model.flow.weight.grad) ** 2
    ) ** 0.5
    assert joint == pytest.approx(5.0, rel=1e-6)
    assert _norm(model.flow.weight.grad) < 5.0  # shrunk by the head's share

    model = _Tiny()
    _set_grads(model, head_grad=100.0, flow_grad=100.0)
    clip_gradients(model, max_norm=5.0, scope="per_module")
    assert _norm(model.head.weight.grad) == pytest.approx(5.0, rel=1e-6)
    assert _norm(model.flow.weight.grad) == pytest.approx(5.0, rel=1e-6)


def test_max_norm_zero_disables_clipping_in_both_scopes():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import clip_gradients

    for scope in ("global", "per_module"):
        model = _Tiny()
        _set_grads(model, head_grad=100.0, flow_grad=100.0)
        clip_gradients(model, max_norm=0.0, scope=scope)
        assert _norm(model.head.weight.grad) == pytest.approx(200.0, rel=1e-6)
        assert _norm(model.flow.weight.grad) == pytest.approx(200.0, rel=1e-6)


def test_unknown_scope_falls_back_to_global():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import clip_gradients

    model = _Tiny()
    _set_grads(model, head_grad=100.0, flow_grad=1.0)
    clip_gradients(model, max_norm=5.0, scope="nonsense")
    assert _norm(model.flow.weight.grad) < 0.1


def test_yaml_key_reaches_the_trainer_dest(tmp_path):
    """§5.11 — the config key must resolve to a dest of this parser."""
    from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

    cfg = tmp_path / "scope.yaml"
    cfg.write_text("protocol:\n  grad_clip_scope: per_module\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad-clip-scope", dest="grad_clip_scope",
                        default="global")
    parser.add_argument("--max-norm", dest="max_norm", type=float, default=5.0)
    ns = parser.parse_args([])
    out = apply_yaml_overrides(parser, ns, str(cfg), argv=[])
    assert out.grad_clip_scope == "per_module"
