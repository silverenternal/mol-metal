"""I1 + I3 — metal-aware graph + model tests, incl. the METAL_GATE (U4).

Pins:
1. ``build_metal_graph`` appends a CONNECTED metal node (degree == cap), fixing
   the G2 disconnection (metal_disconnected_rate=1.0).
2. ``metal_feature_vector`` has the documented shape.
3. All three ``metal_injection`` modes run and produce finite logits.
4. METAL_GATE — the metal signal actually ENGAGES: for ``message`` and
   ``readout``, changing ``metal_feats`` changes the logits; for ``none`` it
   does not (the metal is provably ignored).
"""

from __future__ import annotations

import numpy as np
import torch

from molmetal.data.metal_graph import (
    METAL_FEAT_DIM,
    build_metal_graph,
    coordination_capacity,
    metal_feature_vector,
)
from molmetal.models.metal_aware_dmpnn import MetalAwareDMPNN


def _sample_row():
    # ethylenediamine-like bidentate ligand; Ru(II)
    return "NCCN", "Ru", 2, 2


def test_metal_node_is_connected():
    lig, metal, os_, chg = _sample_row()
    out = build_metal_graph(lig, metal, os_, chg, use_metal_node=True)
    assert out is not None
    af, bf, es, ed, mf = out
    cap = coordination_capacity(metal, os_)
    metal_idx = af.shape[0] - 1
    deg = int((ed == metal_idx).sum())  # incoming edges to metal (each coord bond is directed both ways)
    n_donors = sum(1 for c in lig if c in "NOSP")  # NCCN -> 2
    assert deg == min(cap, n_donors), (deg, cap, n_donors)
    assert deg > 0, "metal must be CONNECTED (fixes G2 disconnection)"
    assert af[metal_idx, 30] == 1.0  # is_metal marker slot


def test_no_metal_node_when_disabled():
    lig, metal, os_, chg = _sample_row()
    out = build_metal_graph(lig, metal, os_, chg, use_metal_node=False)
    af, bf, es, ed, mf = out
    assert af[0].shape[0] == af.shape[1]
    # ligand only: no is_metal marker anywhere
    assert af[:, 30].sum() == 0.0


def test_metal_feature_vector_shape():
    v = metal_feature_vector("Ru", 2, 2, 6)
    assert v.shape == (METAL_FEAT_DIM,)
    assert v[0] == 1.0  # Ru is first in METALS5
    v2 = metal_feature_vector("Re", 2, 0, 6)
    assert v2[4] == 1.0  # Re is last


def _graphs(n=4):
    ligs = ["NCCN", "c1ccccc1", "CC(=O)O", "c1ccncc1"]
    out = []
    for i in range(n):
        out.append(build_metal_graph(ligs[i % len(ligs)], "Ru", 2, 2, use_metal_node=True))
    return out


def test_all_injection_modes_run():
    graphs = _graphs()
    for inj in ("none", "readout", "message", "film"):
        torch.manual_seed(0)
        m = MetalAwareDMPNN(metal_injection=inj)
        m.eval()
        with torch.no_grad():
            logits = m.forward_graphs(graphs)
        assert logits.shape == (len(graphs),)
        assert torch.isfinite(logits).all()


def test_metal_gate_signal_engages():
    """message/readout MUST respond to metal_feats; none MUST ignore it."""
    graphs = _graphs()
    for inj, should_engage in (("none", False), ("readout", True), ("message", True)):
        torch.manual_seed(0)
        m = MetalAwareDMPNN(metal_injection=inj)
        m.eval()
        g_a = [g[:4] + (np.zeros(METAL_FEAT_DIM, np.float32),) for g in graphs]
        g_b = [g[:4] + (np.ones(METAL_FEAT_DIM, np.float32),) for g in graphs]
        with torch.no_grad():
            la = m.forward_graphs(g_a)
            lb = m.forward_graphs(g_b)
        changed = not torch.allclose(la, lb, atol=1e-6)
        assert changed == should_engage, (inj, changed, should_engage)


def test_w6_film_identity_at_init_then_engages():
    """W6 (#99): the per-layer FiLM starts as identity (== metal-blind) and, once
    its params move, the output responds to the metal — the gate is closed at init
    and opens after adaptation."""
    graphs = _graphs()
    g_a = [g[:4] + (np.zeros(METAL_FEAT_DIM, np.float32),) for g in graphs]
    g_b = [g[:4] + (np.ones(METAL_FEAT_DIM, np.float32),) for g in graphs]

    torch.manual_seed(0)
    m = MetalAwareDMPNN(metal_injection="film").eval()
    with torch.no_grad():
        la, lb = m.forward_graphs(g_a), m.forward_graphs(g_b)
    assert torch.allclose(la, lb, atol=1e-6), "film must be identity at init"

    with torch.no_grad():
        for layer in m.film:
            layer.to_gb.weight.normal_(0, 0.1)
            layer.to_gb.bias.normal_(0, 0.1)
        la, lb = m.forward_graphs(g_a), m.forward_graphs(g_b)
    assert not torch.allclose(la, lb, atol=1e-6), "film must respond to metal once moved"

    m2 = MetalAwareDMPNN(metal_injection="film")
    m2.forward_graphs(graphs).sum().backward()
    gnorm = sum(p.grad.abs().sum().item() for p in m2.film.parameters() if p.grad is not None)
    assert gnorm > 0.0, "film params must receive gradient"


def test_message_mode_backprop_touches_metal_proj():
    torch.manual_seed(0)
    m = MetalAwareDMPNN(metal_injection="message")
    graphs = _graphs()
    logits = m.forward_graphs(graphs)
    logits.sum().backward()
    g = m.metal_proj.weight.grad
    assert g is not None and g.abs().sum().item() > 0.0
