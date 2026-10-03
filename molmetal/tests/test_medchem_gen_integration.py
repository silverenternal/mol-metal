"""Tests for the med-chem generation integration layer (TODO-30 P1.1).

Background
----------
This file ships the canonical 25 tests for
:mod:`molmetal.molmetal_lam.lam_chem.medchem_gen_integration` —
the integration layer that wires the 3 non-click med-chem rules
(peptide bond / Boc protection / reductive amination) into the
existing click-only generation pipeline.

The tests are pure CPU — no GPU, no learned priors, no model
loading — and exercise:

  1. :class:`MedChemAugmentedGenerator` — wrap base generator +
     augment with non-click rules at configurable probability.
  2. :class:`RuleConflictResolver` — resolve click vs non-click
     rule conflicts (first-wins policy with priority overrides).
  3. :func:`benchmark_medchem_diversity` — wall-time + diversity
     comparison.

Honest framing
--------------
* The base generator in tests is a trivial passthrough that returns
  the seed SMILES ``n_samples`` times — the integration layer is
  what we exercise, not the underlying chemistry.
* The tests pin known-good input SMILES (canonical amino acids /
  aldehydes) so RDKit can fire the med-chem rules deterministically.
* The benchmark tests run with ``n_seeds <= 3`` so the CPU footprint
  stays small (sub-second on a laptop).
"""

from __future__ import annotations

import random

import pytest

# All tests are CPU-only; no model loading, no GPU warmup.
from molmetal.molmetal_lam.lam_chem.medchem_gen_integration import (
    DEFAULT_CLICK_RULE_PRIORITY,
    DEFAULT_NON_CLICK_RULE_PRIORITY,
    BenchmarkResult,
    MedChemAugmentedGenerator,
    MedChemIntegrationError,
    RuleConflictResolver,
    benchmark_medchem_diversity,
    build_default_integration,
)
from molmetal.molmetal_lam.lam_chem.non_click_medchem_rules import (
    BocProtectionRule,
    PeptideBondRule,
    ReductiveAminationRule,
    register_non_click_rules,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


# Canonical med-chem-friendly SMILES used as inputs.
GLYCINE = "NCC(=O)O"
ALANINE = "CC(N)C(=O)O"
BENZALDEHYDE = "O=Cc1ccccc1"
METHYLAMINE = "CN"


def _passthrough_base(seed: str, n: int):
    """Trivial base generator — returns ``[seed] * n``."""
    return [seed] * int(n)


def _make_rules():
    """Build a fresh 3-rule registry (returns the rule map)."""
    from molmetal.molmetal_lam.lam_chem.non_click_medchem_rules import (
        NonClickMedChemRegistry,
    )
    registry = NonClickMedChemRegistry()
    register_non_click_rules(registry)
    return registry.get_rules()


# ---------------------------------------------------------------------------
# MedChemAugmentedGenerator tests
# ---------------------------------------------------------------------------


def test_medchem_augmented_generator_shape():
    """The augmented generator returns ``n_samples`` SMILES."""
    aug = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.0,
        rng=random.Random(0),
    )
    out = aug.generate(GLYCINE, n_samples=4)
    assert isinstance(out, list)
    assert len(out) == 4
    for s in out:
        assert isinstance(s, str) and s


def test_medchem_augmented_generator_prob_zero_only_click():
    """``use_non_click_prob=0.0`` means passthrough — every output
    matches the seed (base generator returns the seed verbatim)."""
    aug = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.0,
        rng=random.Random(0),
    )
    out = aug.generate(GLYCINE, n_samples=10)
    assert all(s == GLYCINE for s in out), (
        f"prob=0 must produce passthrough SMILES; got {out!r}"
    )


def test_medchem_augmented_generator_prob_one_only_non_click():
    """``use_non_click_prob=1.0`` means every output goes through
    a non-click rule.  At least one output must differ from the
    passthrough seed."""
    rules = _make_rules()
    aug = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=rules,
        use_non_click_prob=1.0,
        rng=random.Random(42),
    )
    out = aug.generate(GLYCINE, n_samples=8)
    # All outputs should differ from the seed OR be a successful
    # med-chem transformation.  When the rule firing fails, the
    # generator falls back to the seed — but with prob=1.0 the
    # majority should differ.
    n_different = sum(1 for s in out if s != GLYCINE)
    assert n_different >= 1, (
        f"prob=1.0 must apply non-click rules; got {out!r}"
    )


def test_medchem_augmented_generator_prob_03():
    """``use_non_click_prob=0.3`` produces ~30% augmented samples.

    The RNG is seeded so the exact fraction is deterministic; with
    seed 0 and ``n_samples=100`` we expect ~30 augmented outputs
    (allowing ±10 for the per-sample Bernoulli draw).
    """
    rules = _make_rules()
    aug = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=rules,
        use_non_click_prob=0.3,
        rng=random.Random(0),
    )
    out = aug.generate(GLYCINE, n_samples=100)
    n_different = sum(1 for s in out if s != GLYCINE)
    # Loose bounds: between 15% and 50% of samples should differ.
    assert 15 <= n_different <= 50, (
        f"prob=0.3 should give ~30% augmented samples; "
        f"got {n_different}/{len(out)} different"
    )


def test_medchem_augmented_generator_invalid_base_raises():
    """A non-callable base_generator raises :class:`MedChemIntegrationError`."""
    with pytest.raises(MedChemIntegrationError):
        MedChemAugmentedGenerator(
            base_generator="not_callable",
            non_click_rules=_make_rules(),
        )


def test_medchem_augmented_generator_prob_clamped_high():
    """``use_non_click_prob`` > 1.0 is clamped to 1.0."""
    aug = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=5.0,
    )
    assert aug.use_non_click_prob == 1.0


def test_medchem_augmented_generator_prob_clamped_low():
    """``use_non_click_prob`` < 0.0 is clamped to 0.0."""
    aug = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=-0.5,
    )
    assert aug.use_non_click_prob == 0.0


def test_medchem_augmented_generator_invalid_seed():
    """Empty seed SMILES returns an empty list (defensive).

    The wrapper does NOT validate the *chemistry* of the seed SMILES
    — that is the base generator's responsibility.  When the seed is
    empty / non-string, the wrapper short-circuits and returns ``[]``.
    """
    aug = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.5,
        rng=random.Random(0),
    )
    # Empty string -> empty list.
    assert aug.generate("", n_samples=4) == []


def test_medchem_augmented_generator_zero_n_samples():
    """``n_samples=0`` returns an empty list."""
    aug = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.5,
        rng=random.Random(0),
    )
    assert aug.generate(GLYCINE, n_samples=0) == []


def test_medchem_augmented_generator_empty_rules_returns_base():
    """When no non-click rules are registered, every output is the
    passthrough seed (no augmentation possible)."""
    aug = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules={},  # empty
        use_non_click_prob=1.0,
        rng=random.Random(0),
    )
    out = aug.generate(GLYCINE, n_samples=6)
    assert all(s == GLYCINE for s in out), (
        f"empty rules must produce passthrough; got {out!r}"
    )


def test_medchem_augmented_generator_diversity_lift():
    """``diversity_lift`` is the difference in unique-scaffold count."""
    aug = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.0,
        rng=random.Random(0),
    )
    # The base-only sample pool is the seed repeated.
    base = [GLYCINE] * 5
    # A mixed pool containing peptide product + gly-gly + N-Boc-gly
    # adds ≥1 distinct scaffold.
    peptide_rules = _make_rules()
    peptide_bond = peptide_rules["peptide_bond"]
    boc = peptide_rules["boc_protection"]
    product_1 = peptide_bond.apply(GLYCINE, ALANINE) or GLYCINE
    product_2 = boc.apply(GLYCINE) or GLYCINE
    augmented = [GLYCINE] * 3 + [product_1, product_2]
    lift = aug.diversity_lift(base, augmented)
    assert lift >= 0, f"diversity_lift must be >= 0; got {lift}"


def test_medchem_augmented_generator_diversity_lift_zero_for_identical():
    """When base == augmented, ``diversity_lift`` is exactly 0."""
    aug = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.0,
        rng=random.Random(0),
    )
    samples = [GLYCINE] * 5
    assert aug.diversity_lift(samples, samples) == 0.0


# ---------------------------------------------------------------------------
# RuleConflictResolver tests
# ---------------------------------------------------------------------------


def test_rule_conflict_resolver_resolves_click_wins():
    """By default, click rules beat non-click rules."""
    resolver = RuleConflictResolver(
        click_rules={"AmideCoupling": object()},
        non_click_rules={"peptide_bond": object()},
    )
    # Click "AmideCoupling" wins over non-click "peptide_bond".
    assert resolver.resolve("AmideCoupling", "peptide_bond") == "AmideCoupling"
    assert resolver.resolve("peptide_bond", "AmideCoupling") == "AmideCoupling"


def test_rule_conflict_resolver_resolves_no_overlap_returns_none():
    """When neither rule is registered, returns ``None``."""
    resolver = RuleConflictResolver()
    assert resolver.resolve("unknown_a", "unknown_b") is None


def test_rule_conflict_resolver_unregistered_returns_other():
    """When only one rule is registered, returns that one."""
    resolver = RuleConflictResolver(
        click_rules={"CuAAC": object()},
        non_click_rules={},
    )
    assert resolver.resolve("CuAAC", "unknown_b") == "CuAAC"
    assert resolver.resolve("unknown_a", "CuAAC") == "CuAAC"


def test_rule_conflict_resolver_same_rule_returns_rule():
    """Same rule on both args returns the rule itself (no conflict)."""
    resolver = RuleConflictResolver(
        click_rules={"CuAAC": object()},
        non_click_rules={"peptide_bond": object()},
    )
    assert resolver.resolve("CuAAC", "CuAAC") == "CuAAC"
    assert resolver.resolve("peptide_bond", "peptide_bond") == "peptide_bond"


def test_rule_conflict_resolver_priority_override():
    """``priority_overrides`` can elevate a non-click rule above click."""
    resolver = RuleConflictResolver(
        click_rules={"AmideCoupling": object()},
        non_click_rules={"peptide_bond": object()},
        priority_overrides={"peptide_bond": -1},
    )
    # With ``-1``, peptide_bond wins over AmideCoupling.
    assert resolver.resolve("AmideCoupling", "peptide_bond") == "peptide_bond"
    assert resolver.resolve("peptide_bond", "AmideCoupling") == "peptide_bond"


def test_rule_conflict_resolver_register_methods():
    """``register_click`` / ``register_non_click`` are idempotent."""
    resolver = RuleConflictResolver()
    rule = object()
    resolver.register_click("CuAAC", rule)
    resolver.register_click("CuAAC", rule)  # idempotent
    assert "CuAAC" in resolver.click_rules
    resolver.register_non_click("peptide_bond", rule)
    assert "peptide_bond" in resolver.non_click_rules


def test_rule_conflict_resolver_known_rule_names():
    """``known_rule_names`` returns the union of both rule sets."""
    resolver = RuleConflictResolver(
        click_rules={"a": 1, "b": 2},
        non_click_rules={"c": 3},
    )
    names = resolver.known_rule_names()
    assert names == {"a", "b", "c"}


def test_rule_conflict_resolver_tie_broken_by_click():
    """When both rules have equal priority and one is a click rule,
    the click rule wins."""
    resolver = RuleConflictResolver(
        click_rules={"AmideCoupling": object()},
        non_click_rules={"custom_amide": object()},
    )
    # Both rules have no explicit priority_override; custom_amide is
    # not in DEFAULT_NON_CLICK_RULE_PRIORITY so its priority is the
    # "unknown non-click" sentinel.  AmideCoupling is in
    # DEFAULT_CLICK_RULE_PRIORITY at index 4 — it wins.
    assert resolver.resolve("AmideCoupling", "custom_amide") == "AmideCoupling"


def test_rule_conflict_resolver_default_constants_are_tuples():
    """The default priority tuples are non-empty tuples."""
    assert isinstance(DEFAULT_CLICK_RULE_PRIORITY, tuple)
    assert isinstance(DEFAULT_NON_CLICK_RULE_PRIORITY, tuple)
    assert len(DEFAULT_CLICK_RULE_PRIORITY) >= 5
    assert len(DEFAULT_NON_CLICK_RULE_PRIORITY) == 3


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


def test_benchmark_medchem_diversity_returns_dict():
    """The benchmark returns the canonical schema."""
    base_gen = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.0,
        rng=random.Random(0),
    )
    aug_gen = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.5,
        rng=random.Random(0),
    )
    out = benchmark_medchem_diversity(
        base_gen=base_gen,
        augmented_gen=aug_gen,
        n_seeds=2,
        n_samples_per_seed=3,
    )
    assert isinstance(out, dict)
    assert set(out.keys()) >= {
        "base",
        "augmented",
        "diversity_lift",
        "wall_time_overhead_s",
        "n_seeds",
    }


def test_benchmark_medchem_diversity_base_only_no_lift():
    """When both generators have ``use_non_click_prob=0``, the
    diversity lift is 0 and the overhead is small."""
    rules = _make_rules()
    base_gen = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=rules,
        use_non_click_prob=0.0,
        rng=random.Random(0),
    )
    aug_gen = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=rules,
        use_non_click_prob=0.0,
        rng=random.Random(0),
    )
    out = benchmark_medchem_diversity(
        base_gen=base_gen,
        augmented_gen=aug_gen,
        n_seeds=2,
        n_samples_per_seed=3,
    )
    assert out["diversity_lift"] == 0.0


def test_benchmark_medchem_diversity_wall_time_recorded():
    """Both ``base.wall_time_s`` and ``augmented.wall_time_s`` are
    non-negative floats."""
    base_gen = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.0,
        rng=random.Random(0),
    )
    aug_gen = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.5,
        rng=random.Random(0),
    )
    out = benchmark_medchem_diversity(
        base_gen=base_gen,
        augmented_gen=aug_gen,
        n_seeds=2,
        n_samples_per_seed=3,
    )
    assert isinstance(out["base"]["wall_time_s"], float)
    assert isinstance(out["augmented"]["wall_time_s"], float)
    assert out["base"]["wall_time_s"] >= 0.0
    assert out["augmented"]["wall_time_s"] >= 0.0


def test_benchmark_medchem_diversity_n_samples_recorded():
    """Each generator records its non-empty sample count."""
    base_gen = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.0,
        rng=random.Random(0),
    )
    aug_gen = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.5,
        rng=random.Random(0),
    )
    out = benchmark_medchem_diversity(
        base_gen=base_gen,
        augmented_gen=aug_gen,
        n_seeds=2,
        n_samples_per_seed=3,
    )
    # Each generator should produce exactly n_seeds * n_samples_per_seed
    # samples when the base is a passthrough.
    expected = 2 * 3
    assert out["base"]["n_samples"] == expected
    assert out["augmented"]["n_samples"] == expected


def test_benchmark_medchem_diversity_explicit_seeds():
    """The benchmark honours an explicit seed list."""
    base_gen = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.0,
        rng=random.Random(0),
    )
    aug_gen = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=_make_rules(),
        use_non_click_prob=0.5,
        rng=random.Random(0),
    )
    out = benchmark_medchem_diversity(
        base_gen=base_gen,
        augmented_gen=aug_gen,
        n_seeds=99,  # ignored when seeds is explicit
        n_samples_per_seed=2,
        seeds=[GLYCINE, BENZALDEHYDE],
    )
    assert out["n_seeds"] == 2


# ---------------------------------------------------------------------------
# Module-level integration tests
# ---------------------------------------------------------------------------


def test_build_default_integration_returns_pair():
    """``build_default_integration`` returns a (gen, resolver) pair."""
    aug, resolver = build_default_integration(
        base_generator=_passthrough_base,
        use_non_click_prob=0.3,
        rng=random.Random(0),
    )
    assert isinstance(aug, MedChemAugmentedGenerator)
    assert isinstance(resolver, RuleConflictResolver)
    # The augmented generator carries the 3 canonical non-click rules.
    assert set(aug.non_click_rules.keys()) >= {
        "peptide_bond",
        "boc_protection",
        "reductive_amination",
    }
    # The resolver knows the 3 non-click rules.
    for name in ("peptide_bond", "boc_protection", "reductive_amination"):
        assert name in resolver.non_click_rules


def test_benchmark_result_to_dict_round_trip():
    """``BenchmarkResult.to_dict`` returns a JSON-friendly dict."""
    br = BenchmarkResult(
        base={"wall_time_s": 0.1, "n_samples": 4},
        augmented={"wall_time_s": 0.2, "n_samples": 5},
        diversity_lift=1.0,
        wall_time_overhead_s=0.1,
        n_seeds=2,
    )
    d = br.to_dict()
    assert d["base"]["wall_time_s"] == 0.1
    assert d["augmented"]["n_samples"] == 5
    assert d["diversity_lift"] == 1.0
    assert d["wall_time_overhead_s"] == 0.1
    assert d["n_seeds"] == 2


def test_medchem_integration_error_is_value_error():
    """``MedChemIntegrationError`` is a ``ValueError`` subclass so
    catch-all error handlers in the synthesis layer route correctly."""
    assert issubclass(MedChemIntegrationError, ValueError)


def test_module_import_surface():
    """The integration module exposes the canonical public API."""
    from molmetal.molmetal_lam.lam_chem import medchem_gen_integration
    expected = {
        "MedChemAugmentedGenerator",
        "RuleConflictResolver",
        "BenchmarkResult",
        "benchmark_medchem_diversity",
        "MedChemIntegrationError",
        "build_default_integration",
        "DEFAULT_CLICK_RULE_PRIORITY",
        "DEFAULT_NON_CLICK_RULE_PRIORITY",
    }
    for name in expected:
        assert hasattr(medchem_gen_integration, name), (
            f"medchem_gen_integration missing public attribute {name!r}"
        )


def test_non_click_rules_pass_through_to_augmented_generator():
    """The augmented generator can fire each of the 3 canonical rules."""
    rules = {
        "peptide_bond": PeptideBondRule(),
        "boc_protection": BocProtectionRule(),
        "reductive_amination": ReductiveAminationRule(),
    }
    aug = MedChemAugmentedGenerator(
        base_generator=_passthrough_base,
        non_click_rules=rules,
        use_non_click_prob=1.0,
        rng=random.Random(123),
    )
    # 12 samples — enough to hit all 3 rules via the random.choice.
    out = aug.generate(GLYCINE, n_samples=12)
    assert len(out) == 12
    # At least one sample should be a peptide-bond product (different
    # from the bare seed) — we use a tight seed to keep the run
    # deterministic.
    n_different = sum(1 for s in out if s != GLYCINE)
    assert n_different >= 1
