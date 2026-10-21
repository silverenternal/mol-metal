"""Wf-A107 — YAML -> dest -> head wiring for the four A/B arms.

The failure this guards against is the one AGENTS.md §5.11 describes: a
knob that parses into `args` and then never reaches the object that
should consume it.  For every arm config the test asserts the parsed
value AND the constructed head's actual attributes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from molmetal.scripts import train_cfm_yuelbond_wasserstein as T

_CFG = Path(__file__).resolve().parents[2] / "configs"

_EXPECTED = {
    "control": dict(weight=0.0, formulation="count", rbf=False, chem=False),
    "count": dict(weight=0.01, formulation="count", rbf=False, chem=False),
    "per_atom": dict(weight=0.01, formulation="per_atom", rbf=False, chem=False),
    "features": dict(weight=0.0, formulation="count", rbf=True, chem=True),
    "edge_ring_w1": dict(weight=1.0, formulation="edge_ring", rbf=False, chem=False),
    "edge_ring_w03": dict(weight=0.3, formulation="edge_ring", rbf=False, chem=False),
}


def _args(arm: str):
    path = _CFG / f"wfa107_{arm}.yaml"
    if not path.exists():
        pytest.skip(f"{path} not present")
    old = sys.argv
    try:
        sys.argv = ["train_cfm_yuelbond_wasserstein.py", "--config", str(path)]
        return T.parse_args()
    finally:
        sys.argv = old


@pytest.mark.parametrize("arm", sorted(_EXPECTED))
def test_config_round_trip(arm: str):
    a = _args(arm)
    e = _EXPECTED[arm]
    assert a.ring_template_weight == pytest.approx(e["weight"])
    assert a.ring_template_formulation == e["formulation"]
    assert a.bond_rbf_features is e["rbf"]
    assert a.bond_pair_chemistry is e["chem"]
    assert a.n_rbf == 8


@pytest.mark.parametrize("arm", sorted(_EXPECTED))
def test_flags_reach_the_head(arm: str):
    a = _args(arm)
    model = T.build_joint_model(a, device="cpu")
    head = model.head
    assert bool(head.use_rbf_distance) is _EXPECTED[arm]["rbf"]
    assert bool(head.use_pair_chemistry) is _EXPECTED[arm]["chem"]
    assert int(head.n_rbf) == 8
    if _EXPECTED[arm]["rbf"]:
        # The feature block must actually widen the first Linear.
        assert head.edge_init.in_features == head.hidden_dim * 2 + 4 + 3 + 8 + 5
    else:
        assert head.edge_init.in_features == head.hidden_dim * 2 + 4 + 3


@pytest.mark.parametrize("arm", ["count", "per_atom", "edge_ring_w1", "edge_ring_w03"])
def test_formulation_reaches_the_prior_config(arm: str):
    """The YAML string must be one RingTemplatePriorConfig accepts.

    Wf-A107b: `--ring-template-formulation` gained a third choice.  A typo
    in either the argparse `choices` list or the config's own validation
    would otherwise only show up as a crash mid-epoch on the GPU.
    """
    from molmetal.loss.ring_template import RingTemplatePriorConfig

    a = _args(arm)
    cfg = RingTemplatePriorConfig(formulation=a.ring_template_formulation)
    assert cfg.formulation == _EXPECTED[arm]["formulation"]
