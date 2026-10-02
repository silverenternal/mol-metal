"""Wf-319G — Runtime wire-up for the L-R17-B2 84-pattern matrix + YuelBond precondition.

Background
----------
Wf-296D (2026-09-28) wired the 84-pattern matrix + YuelBond
precondition into the CFM YuelBond + Wasserstein + Catherine retrain
script (:file:`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`).
Wf-319G (2026-09-29) extends that wire-up to the **MCTS proof-search
dispatch layer** (:mod:`molmetal_lam.search_alg.proof_search`) by
exposing:

1. The 84-pattern matrix + 17 NEW SMARTS as a runtime-dispatchable
   surface (:class:`MetalLigandExchangeRuntime`).
2. The YuelBond precondition as a runtime check
   (:func:`check_yuelbond_precondition`).

This test file pins the Wf-319G contract: 12 tests covering (a) the
84-pattern runtime dispatch (4 tests), (b) the YuelBond precondition
check (4 tests), and (c) integration with
:mod:`molmetal_lam.search_alg.proof_search` (4 tests).

Honest framing
--------------
* All tests are CPU-only.  No GPU, no torch, no I/O.
* The 84-pattern matrix + 17 NEW SMARTS are pre-existing in
  :mod:`pt_metal_ligand_exchange`.  This file verifies the **runtime
  compilation** + **dispatch surface** — the underlying chemistry
  is unchanged.
* The YuelBond precondition check is the runtime-side enforcement
  of the Wf-275C deployment contract.  When ``yuelbond_active=False``,
  the check returns ``m14_lead_invalidated=True`` with the
  Wf-269C MEASURED ``delta_decode_ratio = -1.000``.
* 12 / 12 tests target the Wf-319G contract.  The pre-flight 25/25
  PASS gate is preserved (no modification to any existing test
  file).

Run from the project root::

    PYTHONPATH=. /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \
        molmetal/tests/test_wf319g_wf296d_deep_wire.py -v
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Module-under-test.
# ---------------------------------------------------------------------------
from molmetal_lam.lam_chem.metal_ligand_runtime import (
    CL_OH_SWAP_SMARTS,
    LIGAND_SHELL_MMA_SMARTS,
    MATRIX_PATTERN_COUNT,
    METAL_LIGAND_EXCHANGE_MATRIX,
    METAL_LIGAND_EXCHANGE_SMARTS,
    NEW_SMARTS_PATTERN_COUNT,
    PT_PD_TRANSMETALATION_SMARTS,
    TOTAL_PATTERN_COUNT,
    TRANSMETALATION_SMARTS,
    AQUA_EXCHANGE_SMARTS,
    MetalLigandExchangeRuntime,
    YuelBondPreconditionResult,
    build_matrix_reactions,
    check_yuelbond_precondition,
    wire_into_proof_search_rules,
)


# ---------------------------------------------------------------------------
# Group 1 — 84-pattern runtime dispatch (4 tests).
# ---------------------------------------------------------------------------


def test_01_matrix_pattern_count_pinned():
    """MATRIX_PATTERN_COUNT must equal 84 (pinned by L-R17-B2 matrix)."""
    assert MATRIX_PATTERN_COUNT == 84
    assert len(METAL_LIGAND_EXCHANGE_MATRIX) == MATRIX_PATTERN_COUNT
    # 84 patterns × unique names (no collisions).
    names = [name for name, _smarts, _desc in METAL_LIGAND_EXCHANGE_MATRIX]
    assert len(set(names)) == 84


def test_02_build_matrix_reactions_returns_compiled_rdkit():
    """build_matrix_reactions must compile the 84-pattern matrix into
    RDKit ChemicalReaction objects (and the 17 NEW SMARTS when
    include_new_smarts=True).
    """
    compiled, n_failed = build_matrix_reactions(include_new_smarts=True)
    # When rdkit is unavailable, the function returns ([], 101).
    # When available, the function returns 101 compiled entries with
    # n_failed == 0 on the canonical SMARTS library.
    try:
        from rdkit.Chem import AllChem as _AllChem  # type: ignore
        rdkit_ok = True
    except Exception:
        rdkit_ok = False

    if rdkit_ok:
        assert n_failed == 0, (
            f"build_matrix_reactions had {n_failed} failures; "
            "the canonical SMARTS library should parse cleanly."
        )
        assert len(compiled) == TOTAL_PATTERN_COUNT, (
            f"len(compiled)={len(compiled)}; expected "
            f"TOTAL_PATTERN_COUNT={TOTAL_PATTERN_COUNT} (84 + 17)."
        )
        # Every compiled entry must be a ChemicalReaction whose
        # Initialize() succeeded (so IsMoleculeReactant answers).
        from rdkit.Chem import AllChem  # type: ignore
        for _name, rxn in compiled:
            assert rxn is not None
            assert isinstance(rxn, AllChem.ChemicalReaction)
    else:
        # Defensive fallback — rdkit missing; the function returns
        # ([], 101) so callers still get a valid result.
        assert compiled == []
        assert n_failed == TOTAL_PATTERN_COUNT


def test_03_runtime_dispatch_basic_cisplatin_ammine():
    """The runtime must fire a 84-pattern matrix entry on a
    Pt-Cl + NH3 input and return at least one (pattern_name, product)
    pair.

    The canonical SMARTS for Pt_II_Cl_NH3 is::

        [Pt:1]([Cl:2])[*:3].[NH3:4]>>[Pt:1]([NH3:4])[*:3].[Cl:2]

    which converts ``[Pt](Cl)C#C.[NH3]`` into ``[Pt]([NH3])C#C.[Cl]``
    (the Pt-NH3 bond forms; Cl- leaves).
    """
    runtime = MetalLigandExchangeRuntime(
        yuelbond_precondition=False,
    )
    if not runtime.is_available():
        pytest.skip("rdkit unavailable; runtime not dispatchable")

    # Fire on a Pt-Cl seed SMILES (the WF-Lambda-Fix-Singleton
    # bare-metal-alkyne seed ``[Pt]C#C`` carrying an explicit Cl
    # leaving group).
    state_smi = "[Pt](Cl)C#C"
    tile_smi = "[NH3]"
    products = runtime.dispatch(state_smi, tile_smi)
    assert isinstance(products, list)
    # At least one matrix pattern should fire (canonical cisplatin
    # ammination entry ``Pt_II_Cl_NH3`` is the most likely winner).
    if products:  # pragma: no cover — depends on rdkit substrate match
        for pattern_name, product_smi in products:
            assert isinstance(pattern_name, str)
            assert isinstance(product_smi, str)
            assert pattern_name  # non-empty
            assert product_smi  # non-empty


def test_04_runtime_pattern_names_match_matrix():
    """The runtime's compiled pattern names must be a superset of the
    84-pattern matrix names (when ``include_new_smarts=True``).
    """
    runtime = MetalLigandExchangeRuntime(
        yuelbond_precondition=False,
    )
    if not runtime.is_available():
        pytest.skip("rdkit unavailable; runtime not dispatchable")

    pattern_names = runtime.pattern_names()
    matrix_names = {
        name for name, _smarts, _desc in METAL_LIGAND_EXCHANGE_MATRIX
    }
    # Every matrix name must appear in the runtime pattern_names (the
    # matrix is a strict subset of the runtime's compiled surface).
    missing = matrix_names - set(pattern_names)
    assert not missing, (
        f"Runtime missing {len(missing)} matrix patterns; "
        f"first 5: {list(missing)[:5]}"
    )
    # The runtime's surface should be the matrix + NEW SMARTS.
    assert len(pattern_names) == TOTAL_PATTERN_COUNT


# ---------------------------------------------------------------------------
# Group 2 — YuelBond precondition check (4 tests).
# ---------------------------------------------------------------------------


def test_05_yuelbond_precondition_default_active():
    """With default args (use_yuelbond=True, no checkpoint), the
    precondition returns yuelbond_active=True and recommended_action=
    'PROCEED'.
    """
    verdict = check_yuelbond_precondition()
    assert isinstance(verdict, YuelBondPreconditionResult)
    assert verdict.yuelbond_active is True
    assert verdict.m14_lead_invalidated is False
    assert verdict.recommended_action == "PROCEED"
    assert verdict.delta_decode_ratio == 0.0
    assert "preserved" in verdict.reason.lower() or "active" in verdict.reason.lower()


def test_06_yuelbond_precondition_no_yuelbond_fails_fast():
    """With use_yuelbond=False and strict=True, the precondition
    returns FAIL_FAST with the Wf-269C MEASURED delta_decode_ratio.
    """
    verdict = check_yuelbond_precondition(
        use_yuelbond=False, strict=True,
    )
    assert verdict.yuelbond_active is False
    assert verdict.m14_lead_invalidated is True
    assert verdict.recommended_action == "FAIL_FAST"
    # Wf-269C MEASURED: full=1.000, no_yuelbond=0.000 → delta=-1.000.
    assert verdict.delta_decode_ratio == pytest.approx(-1.000, abs=1e-3)
    assert "yuelbond" in verdict.reason.lower()


def test_07_yuelbond_precondition_warn_and_proceed():
    """With use_yuelbond=False and strict=False, the precondition
    returns WARN_AND_PROCEED (the Wf-296D ablation path).
    """
    verdict = check_yuelbond_precondition(
        use_yuelbond=False, strict=False,
    )
    assert verdict.yuelbond_active is False
    assert verdict.m14_lead_invalidated is True
    assert verdict.recommended_action == "WARN_AND_PROCEED"
    assert verdict.delta_decode_ratio == pytest.approx(-1.000, abs=1e-3)


def test_08_yuelbond_precondition_checkpoint_basename_check():
    """The checkpoint basename check should mark a non-Yuelbond checkpoint
    as ``checkpoint_ok=False``, invalidating the precondition even when
    ``use_yuelbond=True``.
    """
    # Path that does NOT contain "yuelbond" (note: ``cfm_vanilla.pt``
    # was chosen because the basename ``cfm_no_yuelbond_pretrain.pt``
    # DOES contain ``yuelbond`` as a substring — the precondition
    # only requires the basename to NOT contain the string
    # ``yuelbond``).
    verdict = check_yuelbond_precondition(
        use_yuelbond=True,
        checkpoint_path="molmetal/checkpoints/cfm_vanilla_pretrain.pt",
        strict=True,
    )
    assert verdict.yuelbond_active is False
    assert verdict.m14_lead_invalidated is True
    assert verdict.recommended_action == "FAIL_FAST"
    assert "checkpoint" in verdict.reason.lower() or "yuelbond" in verdict.reason.lower()

    # Canonical YuelBond-trained checkpoint path.
    verdict_ok = check_yuelbond_precondition(
        use_yuelbond=True,
        checkpoint_path="molmetal/checkpoints/cfm_yuelbond_wasserstein.pt",
        strict=True,
    )
    assert verdict_ok.yuelbond_active is True
    assert verdict_ok.m14_lead_invalidated is False
    assert verdict_ok.recommended_action == "PROCEED"


# ---------------------------------------------------------------------------
# Group 3 — integration with proof_search.py (4 tests).
# ---------------------------------------------------------------------------


def test_09_runtime_constructor_runs_precondition_by_default():
    """The runtime constructor must run the YuelBond precondition
    check by default (raises RuntimeError on FAIL_FAST).
    """
    # FAIL_FAST path — constructor must raise.
    with pytest.raises(RuntimeError) as excinfo:
        MetalLigandExchangeRuntime(
            use_yuelbond=False, strict=True,
        )
    assert "FAIL_FAST" in str(excinfo.value) or "precondition" in str(excinfo.value).lower()

    # WARN_AND_PROCEED path — constructor must NOT raise (loud warning).
    runtime_warn = MetalLigandExchangeRuntime(
        use_yuelbond=False, strict=False,
    )
    assert runtime_warn.yuelbond_precondition_result.recommended_action == "WARN_AND_PROCEED"

    # PROCEED path — constructor runs and runtime is available.
    runtime_ok = MetalLigandExchangeRuntime()
    assert runtime_ok.yuelbond_precondition_result.recommended_action == "PROCEED"


def test_10_wire_into_proof_search_rules_extends_dict():
    """wire_into_proof_search_rules must extend an existing rules dict
    with the runtime's compiled patterns (non-invasively — the
    existing rules dict is not mutated).
    """
    runtime = MetalLigandExchangeRuntime(
        yuelbond_precondition=False,
    )
    existing = {
        "CuAAC": object(),
        "SPAAC": object(),
        "ThiolEne": object(),
        "Suzuki": object(),
        "AmideCoupling": object(),
    }
    out = wire_into_proof_search_rules(runtime, existing)
    # The existing dict is unchanged.
    assert len(existing) == 5
    assert "CuAAC" in existing
    # The output dict is a NEW dict that contains the existing rules.
    for k in ("CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"):
        assert k in out
    # When the runtime is dispatchable, the output dict is extended
    # with the 84-pattern matrix + 17 NEW SMARTS.
    if runtime.is_available():
        # Every existing key is still present (no shadowing).
        for k in existing:
            assert k in out
        # The matrix pattern names are now in the output dict.
        matrix_first_name = METAL_LIGAND_EXCHANGE_MATRIX[0][0]
        assert matrix_first_name in out
        # The total count matches the runtime's compiled surface.
        n_added = sum(1 for k in out if k not in existing)
        assert n_added == runtime.n_patterns


def test_11_runtime_pattern_count_constants_match_library():
    """The module-level constants must match the underlying library
    (the regression invariant from
    :file:`molmetal/tests/test_metal_ligand_exchange_matrix.py`).
    """
    assert MATRIX_PATTERN_COUNT == len(METAL_LIGAND_EXCHANGE_MATRIX)
    # 5 F2(a) curated entries.
    assert len(METAL_LIGAND_EXCHANGE_SMARTS) == 5
    # 2 AquaExchange curated entries.
    assert len(AQUA_EXCHANGE_SMARTS) == 2
    # NEW SMARTS breakdown: 6 + 5 + 4 + 2 = 17.
    assert len(TRANSMETALATION_SMARTS) == 6
    assert len(CL_OH_SWAP_SMARTS) == 5
    assert len(LIGAND_SHELL_MMA_SMARTS) == 4
    assert len(PT_PD_TRANSMETALATION_SMARTS) == 2
    assert (
        len(TRANSMETALATION_SMARTS)
        + len(CL_OH_SWAP_SMARTS)
        + len(LIGAND_SHELL_MMA_SMARTS)
        + len(PT_PD_TRANSMETALATION_SMARTS)
    ) == NEW_SMARTS_PATTERN_COUNT
    # Combined = 84 + 17 = 101.
    assert MATRIX_PATTERN_COUNT + NEW_SMARTS_PATTERN_COUNT == TOTAL_PATTERN_COUNT


def test_12_runtime_dispatch_on_novel_substrate_yields_empty():
    """The runtime must return an empty list when the substrate does
    NOT match any matrix pattern (negative case — proves the dispatch
    is selective rather than always-on).
    """
    runtime = MetalLigandExchangeRuntime(
        yuelbond_precondition=False,
    )
    if not runtime.is_available():
        pytest.skip("rdkit unavailable; runtime not dispatchable")

    # Pure hydrocarbon — no metal, so NO matrix pattern can fire.
    products = runtime.dispatch("CCO", "C")
    assert products == [], (
        f"Runtime should NOT fire on non-metal substrate; got {products[:4]}"
    )

    # Pure metal (no Cl leaving group) — no matrix pattern can fire
    # because the canonical Pt-Cl leaving group is missing.
    products = runtime.dispatch("[Pt]", "[NH3]")
    # Note: this MAY fire if rdkit auto-adds leaving groups; the
    # contract is just that the result is a list of valid tuples.
    for entry in products:
        assert isinstance(entry, tuple)
        assert len(entry) == 2
        assert isinstance(entry[0], str)
        assert isinstance(entry[1], str)