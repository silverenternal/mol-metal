# -*- coding: utf-8 -*-
"""Tests for ConfidenceHead (mirror OmegaFold omegafold/confidence.py)."""
import torch
import pytest

from molmetal.adapters.flow_matching_lipman.confidence_head import (
    ConfidenceHead,
    ConfidenceHeadOutput,
    _compute_confidence,
    get_all_confidence,
    make_confidence_head,
)


class TestConfidenceHead:
    """4+ tests covering output shape, range, integration, mirroring."""

    def test_compute_confidence_output_shape_and_range(self):
        """pLDDT output shape matches input leading dims, values in [0, 1]."""
        logits = torch.randn(8, 50)  # [num_res=8, num_bins=50]
        plddt = _compute_confidence(logits)
        assert plddt.shape == (8,)
        assert plddt.min().item() >= 0.0
        assert plddt.max().item() <= 1.0
        assert torch.all(plddt == plddt)  # finite check
        print(f"test_compute_confidence_output_shape_and_range PASS: plddt in [{plddt.min():.4f}, {plddt.max():.4f}]")

    def test_confidence_head_forward_shape(self):
        """ConfidenceHead.forward([num_res, node_dim]) → [num_res]."""
        head = ConfidenceHead(node_dim=128, hidden_dim=64, num_bins=50)
        node_repr = torch.randn(12, 128)
        plddt = head(node_repr)
        assert plddt.shape == (12,)
        assert plddt.min().item() >= 0.0
        assert plddt.max().item() <= 1.0
        print(f"test_confidence_head_forward_shape PASS: shape={plddt.shape}")

    def test_confidence_head_get_all_confidence_aggregation(self):
        """get_all_confidence aggregates per-residue into [0, 1] overall."""
        head = ConfidenceHead(node_dim=64, hidden_dim=32, num_bins=10)
        node_repr = torch.randn(5, 64)
        out = head.get_all_confidence(node_repr)
        assert isinstance(out, ConfidenceHeadOutput)
        assert out.plddt_per_residue.shape == (5,)
        assert 0.0 <= out.overall_confidence <= 1.0
        d = out.to_dict()
        assert "plddt_mean" in d and "overall_confidence" in d
        print(f"test_confidence_head_get_all_confidence_aggregation PASS: overall={out.overall_confidence:.3f}")

    def test_make_confidence_head_factory(self):
        """make_confidence_head from argparse cfg returns ConfidenceHead."""
        import argparse
        cfg = argparse.Namespace(node_dim=64, hidden_dim=32, num_bins=20)
        head = make_confidence_head(cfg)
        assert isinstance(head, ConfidenceHead)
        assert head.node_dim == 64 and head.hidden_dim == 32 and head.num_bins == 20
        print("test_make_confidence_head_factory PASS")

    def test_confidence_head_integration_with_random_node_repr(self):
        """Integration test: 100-residue random node_repr → pLDDT per residue."""
        torch.manual_seed(42)
        head = ConfidenceHead(node_dim=256, hidden_dim=128, num_bins=50)
        node_repr = torch.randn(100, 256)
        plddt = head(node_repr)
        assert plddt.shape == (100,)
        assert 0.0 <= plddt.min().item() and plddt.max().item() <= 1.0
        # Mean should be near 0.5 for uniform random
        mean_plddt = plddt.mean().item()
        assert 0.2 < mean_plddt < 0.8  # not pathological
        print(f"test_confidence_head_integration PASS: mean_plddt={mean_plddt:.3f}")
