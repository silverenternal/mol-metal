"""Wf-A71 (2026-10-18) — BOND_NO_BOND bias sign flip regression tests.

Background
----------

Every bond-head class historically initialised the BOND_NO_BOND logit bias
to **-0.5**::

    self.pair_head[-1].bias[BOND_NO_BOND] = -0.5

The accompanying comment said this was to make the head "not default to
bonding every pair" — but the math runs the OTHER way.  With bias=-0.5
and the other 4 class biases at 0, ``softmax[NO_BOND] = 0.132`` at init
and ``p_bonded = 1 - 0.132 = 0.868``.  The threshold rule
(``p_bonded >= 0.5``) then predicts a bond for ~100% of candidate
pairs at init.

MEASURED on the Wf-A70 5-epoch smoke
(``configs/cfm_5metal_wfa70_w3_smoke.yaml``, ep 4):
``bond_pr_auc = 0.9934`` (great ranking) but ``bond_precision = 0.0474``
(broken calibration) — the head has the right ranking but the wrong
default.

Wf-A71 flips the bias to **+2.0** at all 4 init sites:
``softmax[NO_BOND] = 0.649`` and ``p_bonded = 0.351`` at init, so the
head defaults to NO_BOND (matches the 95% NO_BOND empirical prior on
5-metal complexes).  After training, the MLP output overrides the prior
for actual bonds.

Five tests, all CPU, <1 s total.  Each one constructs a fresh head
module (no checkpoint load, no training) and checks the bias direction
on a 5-atom cisplatin-like cloud.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECODER = (
    PROJECT_ROOT
    / "molmetal"
    / "molmetal_lam"
    / "lam_chem"
    / "yuelbond_decoder.py"
)
TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
BOND_HEAD = PROJECT_ROOT / "molmetal" / "models" / "bond_head.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cisplatin_like_cloud() -> "AtomCloud":
    """5-atom Pt(II) cloud (matches test_cfm_rescue_yuelbond.py:43)."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Pt
            [2.05, 0.0, 0.0],  # Cl
            [-2.05, 0.0, 0.0],  # Cl
            [0.0, 2.05, 0.0],  # N
            [0.0, -2.05, 0.0],  # N
        ],
        dtype=torch.float32,
    )
    atomic_numbers = torch.tensor([78, 17, 17, 7, 7], dtype=torch.long)
    AtomCloud = _load(BOND_HEAD, "_bh_for_wfa71").AtomCloud
    return AtomCloud(positions=positions, atomic_numbers=atomic_numbers)


def _logits_of(head) -> torch.Tensor:
    """Forward a fresh YuelBond head on the cisplatin cloud; return (E, 5) logits.

    Only the YuelBond heads (YuelBondDecoderHead, YuelBondEdgeMPNNHead) take
    an :class:`AtomCloud` directly.  ``BondOrderHead`` takes a flat
    ``(E, in_dim)`` pair_features tensor — we only check its bias for
    those tests, no forward pass.
    """
    torch.manual_seed(0)
    head.eval()
    cloud = _cisplatin_like_cloud()
    out = head(cloud)
    if isinstance(out, tuple):
        return out[0].detach()
    return out.detach()


# ---------------------------------------------------------------------------
# Test 1 — YuelBondDecoderHead (legacy atom-MPNN) defaults to NO_BOND
# ---------------------------------------------------------------------------
def test_yuelbond_decoder_head_defaults_to_no_bond():
    """``YuelBondDecoderHead`` at init has softmax[:, 0].mean() > 0.5.

    Pre-Wf-A71: bias=-0.5 → softmax[:, 0].mean() ≈ 0.132 (every pair
    defaults to BOND_SINGLE, the head over-predicts bonds).
    Post-Wf-A71: bias=+2.0 → softmax[:, 0].mean() > 0.5 (NO_BOND is
    the default class on the majority of candidate pairs at init).
    """
    dec = _load(DECODER, "_dec_for_wfa71_t1")
    head = dec.YuelBondDecoderHead(hidden_dim=32, n_layers=2)
    # Sanity: the bias itself is +2.0
    assert float(head.pair_head[-1].bias[dec.BOND_NO_BOND].item()) == pytest.approx(
        +2.0, abs=1e-6
    ), (
        f"WF-A71: YuelBondDecoderHead.pair_head bias[BOND_NO_BOND] must be "
        f"+2.0 to make NO_BOND the default class; got "
        f"{float(head.pair_head[-1].bias[dec.BOND_NO_BOND].item()):.3f}. "
        f"Pre-Wf-A71 value was -0.5 (a sign error)."
    )
    logits = _logits_of(head)
    sm = torch.softmax(logits.float(), dim=-1)
    mean_no_bond = float(sm[:, dec.BOND_NO_BOND].mean())
    assert mean_no_bond > 0.5, (
        f"WF-A71: YuelBondDecoderHead should default to NO_BOND at init "
        f"(softmax[:, 0].mean() > 0.5); got {mean_no_bond:.3f}. "
        f"With the legacy -0.5 bias this reads 0.132 — the head was "
        f"predicting BOND_SINGLE for ~100% of pairs."
    )


# ---------------------------------------------------------------------------
# Test 2 — YuelBondEdgeMPNNHead (Wf-A70 Fix 6 head) defaults to NO_BOND
# ---------------------------------------------------------------------------
def test_yuelbond_edge_mpnn_head_defaults_to_no_bond():
    """``YuelBondEdgeMPNNHead`` at init has softmax[:, 0].mean() > 0.5.

    Same structural bug as the legacy atom-MPNN head; the smoke config
    uses this head, so the fix matters for the headline W3 metric.
    """
    dec = _load(DECODER, "_dec_for_wfa71_t2")
    head = dec.YuelBondEdgeMPNNHead(hidden_dim=32, n_edge_layers=2)
    assert float(head.pair_head[-1].bias[dec.BOND_NO_BOND].item()) == pytest.approx(
        +2.0, abs=1e-6
    ), (
        f"WF-A71: YuelBondEdgeMPNNHead.pair_head bias[BOND_NO_BOND] must be "
        f"+2.0; got {float(head.pair_head[-1].bias[dec.BOND_NO_BOND].item()):.3f}."
    )
    logits = _logits_of(head)
    sm = torch.softmax(logits.float(), dim=-1)
    mean_no_bond = float(sm[:, dec.BOND_NO_BOND].mean())
    assert mean_no_bond > 0.5, (
        f"WF-A71: YuelBondEdgeMPNNHead should default to NO_BOND at init; "
        f"got softmax[:, 0].mean() = {mean_no_bond:.3f}."
    )


# ---------------------------------------------------------------------------
# Test 3 — BondOrderHead in molmetal.models.bond_head defaults to NO_BOND
# ---------------------------------------------------------------------------
def test_bond_order_head_in_models_defaults_to_no_bond():
    """``molmetal.models.bond_head.BondOrderHead`` at init defaults to NO_BOND.

    The canonical pre-Wf-A67 head.  Same bias-direction bug.  This test
    checks the bias value directly because BondOrderHead takes a flat
    ``(E, in_dim)`` pair_features tensor (not an AtomCloud), so the
    "softmax over the bias alone" calculation suffices.
    """
    bh = _load(BOND_HEAD, "_bh_for_wfa71_t3")
    head = bh.BondOrderHead(hidden_dim=32, n_layers=2)
    bias_val = float(head.out.bias[bh.BOND_NO_BOND].item())
    assert bias_val == pytest.approx(+2.0, abs=1e-6), (
        f"WF-A71: BondOrderHead.out bias[BOND_NO_BOND] must be +2.0; "
        f"got {bias_val:.3f}.  Pre-Wf-A71 was -0.5 (sign error)."
    )
    # Mathematical sanity: with bias=+2.0 and 4 other classes at 0,
    # softmax[NO_BOND] = 0.649 → the head defaults to NO_BOND.
    bias_vec = head.out.bias.detach().float()
    sm = torch.softmax(bias_vec, dim=0)
    assert float(sm[bh.BOND_NO_BOND].item()) > 0.5, (
        f"WF-A71: BondOrderHead bias alone should give softmax[NO_BOND] > 0.5; "
        f"got {float(sm[bh.BOND_NO_BOND].item()):.3f}."
    )


# ---------------------------------------------------------------------------
# Test 4 — BondOrderHead in trainer defaults to NO_BOND
# ---------------------------------------------------------------------------
def test_bond_order_head_in_trainer_defaults_to_no_bond():
    """The in-trainer BondOrderHead (D-MPNN path, ``bond_predictor=dmpnn``)
    at init also defaults to NO_BOND.  Same fix applied to keep the
    three heads consistent — if a future caller switches from
    ``bond_predictor=yuelbond_edge_mpnn`` to ``dmpnn`` the calibration
    should not regress.
    """
    # Loading the trainer is heavy (it imports torch_geometric, the EGNN
    # adapter, etc.).  We only need the BondOrderHead class — read the
    # source and exec the class body in isolation is brittle, so instead
    # we assert via the module's own `grep` of the constant.
    src = TRAINER.read_text()
    # Find every assignment of pair_head bias to a numeric value
    import re

    pattern = re.compile(
        r"self\.pair_head\[-1\]\.bias\[BOND_NO_BOND\]\s*=\s*([+\-]?\d+(?:\.\d+)?)"
    )
    matches = pattern.findall(src)
    assert matches, (
        "WF-A71: could not find any pair_head bias init in the trainer "
        "source — the fix may have been applied to the wrong file."
    )
    for v in matches:
        assert float(v) > 0.0, (
            f"WF-A71: trainer pair_head bias[BOND_NO_BOND] = {v} (must be "
            f"a positive value, post-Wf-A71 +2.0).  Pre-Wf-A71 was -0.5."
        )


# ---------------------------------------------------------------------------
# Test 5 — p_bonded < 0.5 across the YuelBond heads (threshold rule stays sane)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "head_kind",
    ["yuelbond", "yuelbond_edge_mpnn"],
)
def test_p_bonded_threshold_satisfied(head_kind: str) -> None:
    """``p_bonded.mean() < 0.5`` so the threshold rule (default 0.5)
    predicts NO_BOND for the majority of pairs at init.

    This is the "is the calibration actually fixed" check — the
    head's RANKING is great (PR-AUC=0.99 in the Wf-A70 smoke), so
    a high p_bonded would mean the head still defaults to bonding
    at init.  Pre-Wf-A71 ``p_bonded`` was 0.868 (defaults to
    bonding); post-Wf-A71 it should be 0.351 (defaults to NO_BOND).

    Only the YuelBond heads are tested here because they take an
    :class:`AtomCloud` directly.  ``BondOrderHead`` is bias-only
    checked in test #3 (it takes a flat ``(E, in_dim)`` pair_features
    tensor and a different forward contract).
    """
    dec = _load(DECODER, f"_dec_for_wfa71_t5_{head_kind}")
    if head_kind == "yuelbond":
        head = dec.YuelBondDecoderHead(hidden_dim=32, n_layers=2)
    else:
        head = dec.YuelBondEdgeMPNNHead(hidden_dim=32, n_edge_layers=2)
    no_bond_idx = dec.BOND_NO_BOND

    logits = _logits_of(head)
    prob = torch.softmax(logits.float(), dim=-1)
    p_bonded = 1.0 - prob[:, no_bond_idx]
    mean_p_bonded = float(p_bonded.mean())
    assert mean_p_bonded < 0.5, (
        f"WF-A71: {head_kind} head at init must have p_bonded.mean() < 0.5 "
        f"so the threshold rule (default 0.5) predicts NO_BOND for the "
        f"majority.  Got {mean_p_bonded:.3f} (pre-Wf-A71 = 0.868, "
        f"the head defaulted to bonding every pair)."
    )
