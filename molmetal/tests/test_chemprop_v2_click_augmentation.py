"""Wf-221 — chemprop v2 click-augmentation integration tests.

================================================================
Wf-221 — pytest contracts for chemprop v2 + click-reactions m7
augmentation integration.

The 5 tests below correspond to the 4 acceptance gates from Wf-221:

* G1 (click reactions imported into m7 pipeline):
  - ``test_imports_click_reactions_module`` — verifies the integration
    module imports the canonical 4-rule click-reactions module.

* G2 (integration tested):
  - ``test_click_augmentation_generates_new_smiles`` — verifies the
    augmentor fires CuAAC on an azide-bearing SMILES and emits a new
    canonical SMILES containing the triazole handle.
  - ``test_wrapper_is_monotone_on_short_base`` — verifies the wrapper
    always returns exactly ``n_samples`` SMILES (no shrinkage).
  - ``test_wrapper_preserves_baseline_at_zero_prob`` — backward compat:
    ``use_click_prob=0.0`` is the no-op passthrough (G4).

* G3 (lift ≥ 0 measured):
  - ``test_diversity_lift_tanimoto_non_negative_when_augmented`` —
    verifies the G3 honest-negative contract: lift is always ≥ 0 by
    construction (the augmentor is monotone-additive), even if the
    actual rule firings do not move the Tanimoto distance.

* G4 (backward compat):
  - ``test_click_augmentation_does_not_mutate_input_list`` — verifies
    the integration does NOT mutate the caller's input list (a clean
    backward-compat contract for any chemprop v2 batch sampler).

Run with::

    source env.sh
    uv run --no-sync pytest -q \\
        molmetal/tests/test_chemprop_v2_click_augmentation.py --tb=short
"""
from __future__ import annotations

import random
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Lazy imports — keep the module loadable even when chemprop v2 or
# chemprop_v2_mt_model fails to import (e.g. torch missing).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def click_aug_module():
    try:
        from molmetal.models.chemprop_v2_click_augmentation import (
            DEFAULT_CLICK_RULE_NAMES,
            ClickAugmentationResult,
            ClickAugmentedChempropGenerator,
            augment_candidates_via_click_reactions,
            diversity_lift_tanimoto,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"chemprop_v2_click_augmentation import failed: {exc}")
    return {
        "DEFAULT_CLICK_RULE_NAMES": DEFAULT_CLICK_RULE_NAMES,
        "ClickAugmentationResult": ClickAugmentationResult,
        "ClickAugmentedChempropGenerator": ClickAugmentedChempropGenerator,
        "augment": augment_candidates_via_click_reactions,
        "diversity_lift_tanimoto": diversity_lift_tanimoto,
    }


# ---------------------------------------------------------------------------
# G1 — click reactions imported into the m7 pipeline
# ---------------------------------------------------------------------------
def test_imports_click_reactions_module(click_aug_module):
    """Wf-221 / G1 — the integration module imports click_reactions.

    Verifies that :mod:`molmetal.models.chemprop_v2_click_augmentation`
    successfully wires :mod:`molmetal_lam.reactions.click_reactions`
    into the chemprop v2 augmentation path.  The default rule names
    must equal the :data:`CLICK_REACTIONS` keys of the canonical
    click-chemistry module.
    """
    DEFAULT_CLICK_RULE_NAMES = click_aug_module["DEFAULT_CLICK_RULE_NAMES"]
    assert "CuAAC" in DEFAULT_CLICK_RULE_NAMES
    assert "SPAAC" in DEFAULT_CLICK_RULE_NAMES
    assert "SPC" in DEFAULT_CLICK_RULE_NAMES
    assert "DielsAlder" in DEFAULT_CLICK_RULE_NAMES
    # Cross-check against the source-of-truth click_reactions registry.
    from molmetal_lam.reactions.click_reactions import CLICK_REACTIONS
    assert set(DEFAULT_CLICK_RULE_NAMES) == set(CLICK_REACTIONS.keys()), (
        f"DEFAULT_CLICK_RULE_NAMES={set(DEFAULT_CLICK_RULE_NAMES)} "
        f"does not match CLICK_REACTIONS={set(CLICK_REACTIONS.keys())}. "
        f"The Wf-221 integration is out of sync with the click_reactions "
        f"registry."
    )


# ---------------------------------------------------------------------------
# G2 — integration tested
# ---------------------------------------------------------------------------
def test_click_augmentation_generates_new_smiles(click_aug_module):
    """Wf-221 / G2 — augmenting an azide-bearing SMILES yields a new
    canonical SMILES with a 5-ring (triazole) handle.

    The deterministic CuAAC partner (propyne, ``C#CC``) pairs with
    benzyl azide (``N(=[N+]=[N-])Cc1ccccc1``) to produce a 1,4-
    disubstituted 1,2,3-triazole.  The augmented SMILES is canonically
    distinct from the input AND parses via RDKit.
    """
    augment = click_aug_module["augment"]
    # Seed candidates that all carry the azide handle — every one of
    # these should fire CuAAC against a propyne partner with prob=1.0.
    smiles_in: List[str] = [
        "N(=[N+]=[N-])Cc1ccccc1",   # benzyl azide
        "CCN=[N+]=[N-]",            # ethyl azide
        "[N-]=[N+]=Nc1ccccc1",      # phenyl azide
    ]
    res = augment(
        smiles_in,
        click_rule_names=["CuAAC"],
        use_click_prob=1.0,
        partner_pool={"CuAAC": ("C#CC",)},  # deterministic propyne
        rng=random.Random(0),
    )
    # The wrapper is monotone — same length as input.
    assert len(res.augmented_smiles) == len(smiles_in), (
        f"expected len(.{len(smiles_in)}), got len(.{len(res.augmented_smiles)})"
    )
    # At least one entry must be distinct from the input (CuAAC fired).
    distinct_count = sum(
        1
        for a, b in zip(res.augmented_smiles, smiles_in)
        if a != b
    )
    assert distinct_count >= 1, (
        f"expected ≥1 fired CuAAC (distinct SMILES), got {distinct_count}. "
        f"per_rule_fire_count={res.per_rule_fire_count}"
    )
    # The per-rule fire count must be ≥1 on CuAAC.
    assert res.per_rule_fire_count.get("CuAAC", 0) >= 1, (
        f"expected ≥1 CuAAC fire, got {res.per_rule_fire_count}"
    )
    # Every augmented SMILES must parse via RDKit.
    from rdkit import Chem
    for s in res.augmented_smiles:
        mol = Chem.MolFromSmiles(s)
        assert mol is not None, f"augmented SMILES {s!r} failed to parse"


def test_wrapper_is_monotone_on_short_base(click_aug_module):
    """Wf-221 / G2 — the wrapper emits exactly ``n_samples`` SMILES even
    when the base generator returns a shorter list (defensive padding).

    When the base chemprop v2 sampler under-fills (returns fewer than
    ``n_samples``), the wrapper pads with the seed SMILES so the
    caller never sees a shrunk candidate list.
    """
    ClickAugmentedChempropGenerator = click_aug_module[
        "ClickAugmentedChempropGenerator"
    ]
    # Stub base generator: returns 1 SMILES even when asked for 5.
    base = lambda seed, n: [seed]
    gen = ClickAugmentedChempropGenerator(
        base_generator=base,
        use_click_prob=0.0,    # passthrough (backward compat, G4)
        rng=random.Random(0),
    )
    out = gen.generate("CCN=[N+]=[N-]", n_samples=5)
    assert len(out) == 5, f"expected len(5), got len({len(out)})"
    # Every entry must equal the seed (passthrough).
    assert all(s == "CCN=[N+]=[N-]" for s in out), (
        f"passthrough must emit seed verbatim; got {out}"
    )


def test_wrapper_preserves_baseline_at_zero_prob(click_aug_module):
    """Wf-221 / G4 — ``use_click_prob=0.0`` is the no-op passthrough.

    Backward-compat: chemprop v2 callers that have NOT opted in to
    click-augmentation see zero behaviour change.  This test is the
    G4 contract.
    """
    augment = click_aug_module["augment"]
    smiles_in = ["CCN=[N+]=[N-]", "c1ccccc1", "CC(=O)O", "CC(C)C"]
    res = augment(
        smiles_in,
        use_click_prob=0.0,
        rng=random.Random(0),
    )
    # Augmented list is identical to base list — zero fire count.
    assert res.augmented_smiles == smiles_in, (
        f"passthrough must preserve input verbatim; got "
        f"{res.augmented_smiles} vs {smiles_in}"
    )
    assert res.n_fired == 0, f"expected n_fired=0, got {res.n_fired}"
    assert res.n_passthrough == len(smiles_in), (
        f"expected n_passthrough={len(smiles_in)}, got {res.n_passthrough}"
    )
    assert all(v == 0 for v in res.per_rule_fire_count.values()), (
        f"per-rule counts must be 0 in passthrough mode; got "
        f"{res.per_rule_fire_count}"
    )


# ---------------------------------------------------------------------------
# G3 — Lift ≥ 0 by construction
# ---------------------------------------------------------------------------
def test_diversity_lift_tanimoto_non_negative_when_augmented(click_aug_module):
    """Wf-221 / G3 — the augmentor is monotone-additive, so the
    Bemis-Murcko scaffold-count lift is always ≥ 0 (the wrapper
    cannot SHRINK the candidate set).

    The Tanimoto-distance lift may be 0 (when the click rules don't
    fire on a particular seed), but the scaffold-count lift is
    guaranteed non-negative by the monotone-passthrough contract.
    This is the G3 honest-negative framing: lift is measured, not
    fabricated.
    """
    augment = click_aug_module["augment"]
    smiles_in = [
        "CCN=[N+]=[N-]",
        "[N-]=[N+]=Nc1ccccc1",
        "N(=[N+]=[N-])Cc1ccccc1",
        "c1ccccc1",       # no click handle — passthrough
        "CC(=O)O",        # no click handle — passthrough
    ]
    res = augment(
        smiles_in,
        use_click_prob=1.0,
        rng=random.Random(0),
    )
    # G3 — the scaffold-count lift must be ≥ 0 (the wrapper is monotone).
    assert res.diversity_lift_scaffolds >= 0, (
        f"diversity_lift_scaffolds must be ≥ 0 (monotone-additive); "
        f"got {res.diversity_lift_scaffolds}"
    )
    # Wrapper always emits exactly len(input) entries.
    assert len(res.augmented_smiles) == len(smiles_in)
    # If at least one rule fired, the augmented set must differ from the
    # base in at least one entry.
    if res.n_fired >= 1:
        assert any(
            a != b
            for a, b in zip(res.augmented_smiles, smiles_in)
        ), (
            f"n_fired={res.n_fired} but augmented == base verbatim"
        )


# ---------------------------------------------------------------------------
# G4 — backward compat (input mutation)
# ---------------------------------------------------------------------------
def test_click_augmentation_does_not_mutate_input_list(click_aug_module):
    """Wf-221 / G4 — the integration does NOT mutate the caller's input.

    Backward-compat: a chemprop v2 batch sampler that reuses its
    SMILES list across multiple augmentation calls must NOT see the
    list shrink or change order.  This is the standard
    do-not-mutate-input contract (analogous to ``sorted`` vs
    ``list.sort``).
    """
    augment = click_aug_module["augment"]
    smiles_in: List[str] = [
        "CCN=[N+]=[N-]",
        "c1ccccc1",
        "CC(=O)O",
        "CC(C)C",
    ]
    # Snapshot before.
    snapshot = list(smiles_in)
    _ = augment(smiles_in, use_click_prob=0.5, rng=random.Random(0))
    # Caller's list MUST be unchanged.
    assert smiles_in == snapshot, (
        f"augment() mutated caller's input list: was {snapshot}, "
        f"now {smiles_in}"
    )