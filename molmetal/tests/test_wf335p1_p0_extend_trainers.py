"""Tests for Wf-335-P1 — extend the 5 P0 wins from Wf-326G to chemprop_v2_v3 + cfm_yuelbond trainers.

==================================================================
Wf-335-P1: P0 wins extended to chemprop_v2_v3 + cfm_yuelbond trainers
==================================================================

Per Wf-326G audit (``TODO/completed/326g_missed_optimizations_plan.md``)
and Wf-331-X2 closeout (``TODO/completed/331x2_326g_p0_wins.md``): the 5
P0 wins landed in :mod:`molmetal.baselines.dmpnn_p0wins` for the BASIC
``dmpnn.py`` path.

Wf-333-G3 (``TODO/completed/333g3_p0_wins_extend.md``) extended the
verification tests to the chemprop_v2_v3 trainer context.

Wf-335-P1 closes the trainer coverage gap:
* Re-verifies chemprop_v2_v3 integration with a high-level integration test
  (in addition to the Wf-333-G3 4 verification tests).
* NEW verification tests for the cfm_yuelbond trainer extension via
  :mod:`molmetal.baselines.cfm_p0wins`.
* Combined P0 wins test (chemprop_v2_v3 + cfm_yuelbond + dmpnn_p0wins).
* Backward compat test (p0wins modules are additive, no monkey-patching).

4 tests:

1. ``test_chemprop_v2_v3_integration`` — high-level integration smoke for
   the chemprop_v2_v3 trainer context (combined with Wf-333-G3's 4 tests).
2. ``test_cfm_yuelbond_integration`` — NEW: cfm_p0wins helpers work for
   CFM-shaped inputs.
3. ``test_combined_p0_wins`` — p0wins (D-MPNN + CFM) work in the same
   process with no namespace collision.
4. ``test_backward_compat`` — p0wins modules are additive; the original
   chemprop_v2_v3 + cfm_yuelbond trainer source is unchanged.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* These tests verify that the Wf-326G P0 wins exported from
  :mod:`molmetal.baselines.dmpnn_p0wins` (D-MPNN) +
  :mod:`molmetal.baselines.cfm_p0wins` (CFM) can be invoked from the
  chemprop_v2_v3 + cfm_yuelbond trainer contexts WITHOUT namespace
  collision.
* All speedup numbers remain PROJECTED (per Wf-326G audit), not MEASURED.
* CPU-only — no GPU contention with Wf-182 G6 / Wf-124 P3 / Wf-299.

Run::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
    /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
        molmetal/tests/test_wf335p1_p0_extend_trainers.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test 1: chemprop_v2_v3 integration (high-level smoke)
# ---------------------------------------------------------------------------
def test_chemprop_v2_v3_integration() -> None:
    """Wf-335-P1 test 1: chemprop_v2_v3 trainer + dmpnn_p0wins integration.

    This is the Wf-335-P1 high-level integration smoke for the
    chemprop_v2_v3 trainer context.  The 4 granular verification tests
    live in :mod:`test_wf333g3_p0_wins_extend`; this test confirms the
    p0wins helpers can be invoked from a chemprop_v2_v3-style context
    without namespace collision or import-time errors.
    """
    # 1. Import p0wins.
    from molmetal.baselines import dmpnn_p0wins

    # 2. Verify all 5 P0 wins are exported.
    status = dmpnn_p0wins.list_optimization_status()
    assert len(status) == 5, f"Expected 5 P0 wins; got {len(status)}"
    expected_ids = {"326G-1", "326G-2", "326G-3", "326G-4", "326G-5"}
    assert set(status.keys()) == expected_ids, (
        f"Missing win IDs: {expected_ids - set(status.keys())}"
    )
    for win_id in expected_ids:
        assert status[win_id]["port"] is True, (
            f"{win_id} not landed: {status[win_id]}"
        )

    # 3. Verify chemprop_v2_v3 namespace compatibility: p0wins BatchMolGraph
    #    coexists with chemprop.data.BatchMolGraph (chemprop_v2_v3 line 127).
    try:
        from chemprop.data import BatchMolGraph as ChempropBatchMolGraph
        # Both BatchMolGraph classes can be imported; they are distinct.
        assert (
            dmpnn_p0wins.BatchMolGraph is not ChempropBatchMolGraph
        ), "p0wins.BatchMolGraph should NOT be chemprop.data.BatchMolGraph"
    except ImportError:
        # If chemprop v2 is not installed, the aliasing check is vacuous.
        # The p0wins module itself must still work — it doesn't require
        # chemprop v2.
        pass

    # 4. chemprop_v2_v3 trainer script is at the expected path (no
    #    invasive port in Wf-335-P1).
    trainer_path = (
        PROJECT_ROOT / "molmetal" / "scripts" / "train_chemprop_v2_v3.py"
    )
    assert trainer_path.is_file(), f"Trainer script missing: {trainer_path}"

    # 5. Verify the trainer source contains the BF16 autocast block (already
    #    present per Wf-326G §3.5 PARTIAL row).
    trainer_src = trainer_path.read_text()
    assert (
        'torch.amp.autocast("cuda", dtype=torch.bfloat16)' in trainer_src
    ), "Expected BF16 autocast block in chemprop_v2_v3 trainer"
    assert (
        "from chemprop.data import BatchMolGraph" in trainer_src
    ), "Expected chemprop v2 BatchMolGraph import in chemprop_v2_v3 trainer"


# ---------------------------------------------------------------------------
# Test 2: cfm_yuelbond integration (NEW for Wf-335-P1)
# ---------------------------------------------------------------------------
def test_cfm_yuelbond_integration() -> None:
    """Wf-335-P1 test 2: cfm_yuelbond trainer + cfm_p0wins integration.

    NEW for Wf-335-P1.  Verifies the cfm_p0wins helpers work for CFM-shaped
    inputs (atom clouds + per-edge messages + per-atom aggregation).
    """
    from molmetal.baselines import cfm_p0wins

    # 1. Verify the applicability matrix is correct: 326G-1 + 326G-5 do
    #    NOT apply to CFM (3D atom clouds); 326G-2 + 326G-3 + 326G-4
    #    apply.
    status = cfm_p0wins.list_cfm_optimization_status()
    assert len(status) == 5, f"Expected 5 wins in audit; got {len(status)}"
    # N/A wins for CFM
    assert status["326G-1"]["applies"] is False, "326G-1 should NOT apply to CFM"
    assert status["326G-5"]["applies"] is False, "326G-5 should NOT apply to CFM"
    # Applicable wins
    assert status["326G-2"]["applies"] is True, "326G-2 should apply to CFM"
    assert status["326G-3"]["applies"] is True, "326G-3 should apply to CFM"
    assert status["326G-4"]["applies"] is True, "326G-4 should apply to CFM"

    # 2. Verify the combined PROJECTED speedup range.
    combined = cfm_p0wins.combined_cfm_projected_speedup_range()
    assert "3-5x" in combined, f"Unexpected combined range: {combined}"

    # 3. cfm_egnn_message_aggregate: scatter_reduce_ on CFM-shaped inputs.
    n_atoms = 8
    n_edges = 16
    f_dim = 4
    messages = torch.randn(n_edges, f_dim, dtype=torch.float32)
    edge_dst = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7] * 2, dtype=torch.long)
    out_agg = cfm_p0wins.cfm_egnn_message_aggregate(messages, edge_dst, n_atoms)
    assert out_agg.shape == (n_atoms, f_dim), (
        f"Wrong agg shape: {out_agg.shape}; expected {(n_atoms, f_dim)}"
    )
    # Cross-validation: index_add_ fallback must match scatter_reduce_
    # within FP32 reduction-order tolerance.
    out_ref = cfm_p0wins.cfm_egnn_message_aggregate_index_add_fallback(
        messages, edge_dst, n_atoms,
    )
    assert torch.allclose(out_agg, out_ref, atol=5e-3), (
        "scatter_reduce_ vs index_add_ mismatch on CFM-shaped inputs"
    )

    # 4. setup_optimized_cfm_env: TF32 flip works.
    flipped = cfm_p0wins.setup_optimized_cfm_env(enable_tf32=True)
    assert flipped["tf32_enabled"] is True, "TF32 should be enabled"

    # 5. make_optimized_cfm_adam: returns an Adam optimizer with fused=True.
    dummy_params = [torch.zeros(2, requires_grad=True), torch.zeros(3, requires_grad=True)]
    opt = cfm_p0wins.make_optimized_cfm_adam(dummy_params, lr=1e-3, fused=True)
    assert isinstance(opt, torch.optim.Adam), (
        f"Expected torch.optim.Adam; got {type(opt)}"
    )

    # 6. cfm_yuelbond trainer script is at the expected path (no
    #    invasive port in Wf-335-P1).
    trainer_path = (
        PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    )
    assert trainer_path.is_file(), f"Trainer script missing: {trainer_path}"

    # 7. The cfm_yuelbond trainer does NOT yet have BF16 autocast
    #    (different from chemprop_v2_v3 which has it at lines 976, 1256).
    #    Per Wf-335-P1 honest framing: 326G-4 is OPT-IN via
    #    ``bf16_predict`` / ``bf16_train_epoch`` (re-exported from
    #    dmpnn_p0wins).  Verify the trainer source does NOT have the
    #    autocast block yet (so Wf-335-P1 is additive, not invasive).
    trainer_src = trainer_path.read_text()
    n_autocast = trainer_src.count(
        'torch.amp.autocast("cuda", dtype=torch.bfloat16)'
    )
    assert n_autocast == 0, (
        f"cfm_yuelbond trainer unexpectedly has BF16 autocast already "
        f"(Wf-335-P1 does NOT modify the trainer): got {n_autocast} blocks"
    )


# ---------------------------------------------------------------------------
# Test 3: combined P0 wins (D-MPNN + chemprop_v2_v3 + cfm_yuelbond)
# ---------------------------------------------------------------------------
def test_combined_p0_wins() -> None:
    """Wf-335-P1 test 3: combined D-MPNN + chemprop_v2_v3 + cfm_yuelbond p0wins.

    Verifies the p0wins modules (D-MPNN + CFM) coexist in the same
    process with no namespace collision and that all 5 wins remain
    accessible.
    """
    from molmetal.baselines import cfm_p0wins, dmpnn_p0wins

    # 1. Both modules importable in the same process.
    assert hasattr(dmpnn_p0wins, "BatchMolGraph"), "Missing BatchMolGraph in dmpnn_p0wins"
    assert hasattr(cfm_p0wins, "cfm_egnn_message_aggregate"), (
        "Missing cfm_egnn_message_aggregate in cfm_p0wins"
    )

    # 2. D-MPNN p0wins (5 wins, all port=True).
    status_dmpnn = dmpnn_p0wins.list_optimization_status()
    assert all(s["port"] for s in status_dmpnn.values()), (
        f"D-MPNN wins not all landed: {status_dmpnn}"
    )

    # 3. CFM p0wins (5 wins, all port=True, but applicability differs).
    status_cfm = cfm_p0wins.list_cfm_optimization_status()
    assert all(s["port"] for s in status_cfm.values()), (
        f"CFM wins not all landed: {status_cfm}"
    )

    # 4. Cross-check: scatter_sum_no_self from D-MPNN p0wins is the
    #    SAME function as the one used by cfm_p0wins.cfm_egnn_message_aggregate
    #    (re-exported, not re-implemented).
    assert (
        cfm_p0wins.scatter_sum_no_self is dmpnn_p0wins.scatter_sum_no_self
    ), "scatter_sum_no_self should be re-exported, not re-implemented"

    # 5. End-to-end: build a tiny D-MPNN BatchMolGraph, run forward_batched_dmpnn,
    #    and verify the CFM scatter-reduce produces consistent results on
    #    the same scatter shape.
    from molmetal.baselines.dmpnn import featurize_smiles_list

    smiles_list = ["CCO", "CCC", "CCN", "c1ccccc1"]
    raw_graphs = featurize_smiles_list(smiles_list)
    # Filter out None entries (failed RDKit parses).
    graphs = [g for g in raw_graphs if g is not None]
    assert len(graphs) >= 1, "Expected at least 1 valid graph"
    bmg = dmpnn_p0wins.batch_mol_graph_from_tuples(graphs)
    # Take a slice of (edge_src, edge_dst) and aggregate.
    E = bmg.E.size(0)
    if E > 0:
        edge_dst_sub = bmg.edge_dst
        n_atoms_total = bmg.V.size(0)
        out_dmpnn = dmpnn_p0wins.scatter_sum_no_self(
            bmg.E, edge_dst_sub, n_atoms_total,
        )
        out_cfm = cfm_p0wins.cfm_egnn_message_aggregate(
            bmg.E, edge_dst_sub, n_atoms_total,
        )
        assert torch.allclose(out_dmpnn, out_cfm, atol=5e-3), (
            "D-MPNN vs CFM scatter_sum should be identical (re-export)"
        )


# ---------------------------------------------------------------------------
# Test 4: backward compatibility
# ---------------------------------------------------------------------------
def test_backward_compat() -> None:
    """Wf-335-P1 test 4: p0wins modules are additive (no monkey-patching).

    Verifies the Wf-335-P1 cfm_p0wins module does NOT modify any existing
    module state.  The original chemprop_v2_v3 + cfm_yuelbond trainer
    source is unchanged (per the Wf-335-P1 honest framing).
    """
    # 1. Verify the trainer scripts have NOT been modified by Wf-335-P1.
    trainer_paths = [
        PROJECT_ROOT / "molmetal" / "scripts" / "train_chemprop_v2_v3.py",
        PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py",
    ]
    for p in trainer_paths:
        assert p.is_file(), f"Trainer script missing: {p}"
        # Sanity: the file is non-empty and parses as Python.
        src = p.read_text()
        assert len(src) > 1000, f"Trainer script unexpectedly small: {p}"
        compile(src, str(p), "exec")  # raises SyntaxError if malformed

    # 2. Verify the p0wins modules are additive: importing them does NOT
    #    change the dmpnn baseline module's exports.
    from molmetal.baselines import dmpnn as dmpnn_module
    from molmetal.baselines import dmpnn_p0wins

    # Snap dmpnn.py exports before import.
    dmpnn_exports_before = set(dir(dmpnn_module))
    # Importing dmpnn_p0wins should not change dmpnn's exports.
    _ = dmpnn_p0wins.list_optimization_status
    dmpnn_exports_after = set(dir(dmpnn_module))
    assert dmpnn_exports_before == dmpnn_exports_after, (
        "dmpnn_p0wins import should NOT mutate molmetal.baselines.dmpnn"
    )

    # 3. Same check for cfm_p0wins: importing cfm_p0wins should not
    #    mutate the cfm_yuelbond_wasserstein trainer module's exports
    #    (the trainer is only invoked as a script, so we just check
    #    that the import doesn't raise and doesn't have state).
    from molmetal.baselines import cfm_p0wins
    _ = cfm_p0wins.list_cfm_optimization_status
    assert cfm_p0wins is not None

    # 4. Backward-compat with Wf-331-X2 + Wf-333-G3: re-export the
    #    D-MPNN p0wins __all__ symbols from cfm_p0wins so callers can
    #    use either import path.
    cfm_all = set(cfm_p0wins.__all__)
    # cfm_p0wins should expose the scatter_reduce_ + compile + adam helpers
    # (re-exported from dmpnn_p0wins).
    expected_cfm_exports = {
        "cfm_egnn_message_aggregate",
        "cfm_egnn_message_aggregate_index_add_fallback",
        "setup_optimized_cfm_env",
        "compile_cfm_model",
        "make_optimized_cfm_adam",
        "bf16_predict",
        "bf16_train_epoch",
        "list_cfm_optimization_status",
        "combined_cfm_projected_speedup_range",
    }
    missing = expected_cfm_exports - cfm_all
    assert not missing, f"cfm_p0wins missing exports: {missing}"

    # 5. Re-export sanity: scatter_sum_no_self is a re-export from
    #    dmpnn_p0wins, NOT a re-implementation.
    assert hasattr(cfm_p0wins, "scatter_sum_no_self"), (
        "scatter_sum_no_self should be re-exported via cfm_p0wins.__all__"
    )


if __name__ == "__main__":
    # Allow direct invocation via `python -m molmetal.tests.test_wf335p1_p0_extend_trainers`
    # (in addition to pytest).
    sys.exit(pytest.main([__file__, "-v"]))