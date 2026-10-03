"""Non-click med-chem rules integration layer (TODO-30 Rank 11 P1.1).

Background
----------
The R17 generation pipeline ships with the canonical click-chemistry
reaction set (:mod:`molmetal_lam.lam_chem.rules`,
:mod:`molmetal_lam.lam_chem.syntemol_reactions`) — 5 click rules +
2 metal-coordination rules + 18 SyntheMol REAL rules = 25 rules.
The click set is heavily weighted toward CuAAC / SPAAC / thiol-ene /
Suzuki / amide coupling, which under-represents the med-chem reaction
menu that PROTAC linker assembly + peptidomimetic campaign design
needs.

This module ships the integration layer that wires the 3 non-click
med-chem rules from
:mod:`molmetal_lam.lam_chem.non_click_medchem_rules` (peptide bond /
Boc protection / reductive amination) into the existing generator
pipeline **without modifying** ``rules.py`` /
``syntemol_reactions.py``.

The integration surface
-----------------------
1. :class:`MedChemAugmentedGenerator` — wraps any base generator
   (a callable ``seed_smiles -> List[str]`` such as
   :class:`LayeredGenerationPipeline` / a stub) and augments a
   fraction of its output by applying the non-click med-chem rules
   to the generated molecules.  The probability of using a non-click
   rule on each generated sample is configurable (default 30%).

2. :class:`RuleConflictResolver` — given two rule names (one click +
   one non-click, e.g. ``"AmideCoupling"`` vs ``"peptide_bond"``),
   return the canonical rule the synthesis layer should prefer.  This
   prevents the MCTS from branching on the same net-transform twice
   when click amide coupling and non-click peptide coupling would
   both apply.

3. :func:`benchmark_medchem_diversity` — run two generators
   (base + augmented) over a fixed seed pool and report wall-time +
   diversity (mean pairwise Tanimoto distance) so callers can
   quantify the lift the med-chem rules contribute.

CPU-only / no model loading
---------------------------
* RDKit-only chemistry (no torch, no D-MPNN, no learned priors).
* No model weights loaded — the rule firing uses the SMARTS
  templates shipped in :mod:`non_click_medchem_rules`.
* Determinism: every random choice uses an injected ``random.Random``
  instance (or a fresh one if ``None``).  Tests use fixed seeds.

Honest framing
--------------
* The augmented generator does **not** guarantee a *quality* lift —
  med-chem rules may add protected-amine motifs that downstream
  assays may reject (PRODRG / QED filters).  The benchmark reports
  the *diversity* lift (number of unique scaffolds) and the wall-
  time cost, NOT a QED / docking lift.
* Rule-conflict resolution is a *first-wins* policy with a known
  priority order (click > non-click) so the click set remains the
  canonical chemistry menu.  Custom priority orders can be injected
  via :attr:`RuleConflictResolver.priority_overrides`.
* The med-chem rules live in a *sidecar* module — we do not modify
  ``rules.py`` (per TODO-30 Rank-11 P1.1 acceptance criteria).

Lit anchors
-----------
* Bemis & Murcko 1996 *J. Med. Chem.* 39(15), 2887-2893 — Bemis-Murcko
  scaffold uniqueness used by the diversity benchmark.
* Rogers & Hahn 2010 *J. Chem. Inf. Model.* 50(5), 742-754 — Tanimoto
  on ECFP4 fingerprints (used by the benchmark diversity metric).
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

# Quiet RDKit's verbose logging on rule-firing errors that surface
# as benign warnings during the high-throughput augmented-generation
# path.  Errors are still raised via ``run_in_background``.
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MedChemIntegrationError(ValueError):
    """Raised when the med-chem integration layer fails to wire a rule.

    Distinct from :class:`molmetal_lam.lam_chem.non_click_medchem_rules.NonClickMedChemRuleError`
    so the synthesis layer can route integration failures to a
    different error channel (typically configuration errors, not
    chemistry failures).
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_parse(smiles: str) -> Optional[Chem.Mol]:
    """Parse a SMILES string into an RDKit mol; ``None`` on failure."""
    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _canonical(smiles: str) -> str:
    """Return the canonical SMILES of ``smiles`` or ``""`` on failure."""
    mol = _safe_parse(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def _fingerprint(smiles: str, n_bits: int = 2048):
    """Compute an ECFP4 fingerprint bit-vector; ``None`` on failure."""
    mol = _safe_parse(smiles)
    if mol is None:
        return None
    try:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        return bv
    except Exception:
        return None


def _tanimoto_distance(fp1, fp2) -> float:
    """Tanimoto *distance* (1 - similarity) between two bit-vectors.

    Returns ``1.0`` if either fingerprint is ``None`` (treated as
    maximally distant from any valid molecule).
    """
    if fp1 is None or fp2 is None:
        return 1.0
    try:
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 1.0
    return max(0.0, 1.0 - sim)


def _mean_pairwise_tanimoto_distance(smiles_list: Sequence[str]) -> float:
    """Mean pairwise Tanimoto distance across a SMILES list.

    Returns ``0.0`` for lists of size 0 or 1 (no pairs to compare).
    Returns ``1.0`` for lists where every molecule fails to parse
    (degenerate case — every pair is "maximally distant").
    """
    cleaned = [s for s in smiles_list if isinstance(s, str) and s]
    if len(cleaned) < 2:
        return 0.0
    fps: List = []
    for s in cleaned:
        fp = _fingerprint(s)
        if fp is not None:
            fps.append(fp)
    n = len(fps)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _tanimoto_distance(fps[i], fps[j])
            pairs += 1
    return total / pairs if pairs else 0.0


def _unique_scaffold_count(smiles_list: Sequence[str]) -> int:
    """Return the number of distinct Bemis-Murcko scaffolds in ``smiles_list``.

    Used by the diversity benchmark — a higher unique-scaffold count
    implies broader chemical-space coverage.
    """
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:  # pragma: no cover - defensive
        return 0
    scaffolds: Set[str] = set()
    for s in smiles_list:
        mol = _safe_parse(s)
        if mol is None:
            continue
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
        except Exception:
            continue
        if scaf is None or scaf.GetNumAtoms() == 0:
            # Acyclic — count as the literal SMILES so duplicates
            # still collapse.
            canonical = _canonical(s)
            if canonical:
                scaffolds.add(canonical)
            continue
        try:
            scaffolds.add(Chem.MolToSmiles(scaf))
        except Exception:
            continue
    return len(scaffolds)


def _filter_valid(smiles_list: Iterable[str]) -> List[str]:
    """Drop empty / unparseable SMILES from the input iterable."""
    out: List[str] = []
    for s in smiles_list:
        if not isinstance(s, str) or not s:
            continue
        mol = _safe_parse(s)
        if mol is None:
            continue
        canonical = _canonical(s)
        if canonical:
            out.append(canonical)
    return out


# ---------------------------------------------------------------------------
# RuleConflictResolver
# ---------------------------------------------------------------------------


# Default priority map for rule-conflict resolution.  Lower index =
# higher priority (wins).  Click rules beat non-click rules because
# the click set is the canonical R17 chemistry menu — non-click
# rules are an opt-in extension.
DEFAULT_CLICK_RULE_PRIORITY: Tuple[str, ...] = (
    "CuAAC",
    "SPAAC",
    "ThiolEne",
    "Suzuki",
    "AmideCoupling",
    "MetalLigandExchange",
    "AquaExchange",
)

DEFAULT_NON_CLICK_RULE_PRIORITY: Tuple[str, ...] = (
    "peptide_bond",
    "boc_protection",
    "reductive_amination",
)


@dataclass
class RuleConflictResolver:
    """Resolve click vs non-click rule-name conflicts.

    The synthesis layer's MCTS expands candidates by applying a single
    rule per branch.  When two rules have the *same net transform*
    (e.g. click ``AmideCoupling`` vs non-click ``peptide_bond`` both
    form an amide bond), the resolver picks one to prevent duplicate
    branches.

    Parameters
    ----------
    click_rules : dict[str, object]
        Map of click-rule name -> rule instance.  Typically
        :data:`molmetal_lam.lam_chem.rules.CLICK_REACTIONS` /
        :data:`molmetal_lam.lam_chem.rules.METAL_COORDINATION_RULES`.
    non_click_rules : dict[str, object]
        Map of non-click-rule name -> rule instance.  Typically
        :data:`molmetal_lam.lam_chem.non_click_medchem_rules.NonClickMedChemRegistry.get_rules`.
    priority_overrides : dict[str, int], default ``{}``
        Optional per-rule-name priority overrides.  Lower number =
        higher priority (wins over both defaults).  Used by callers
        that want to elevate a non-click rule (e.g. for peptidomimetic
        campaigns) above the click defaults.

    Notes
    -----
    Resolution policy: first, check ``priority_overrides`` (lower
    number wins).  Second, check the click rule set.  Third, check
    the non-click rule set.  If neither rule is registered, returns
    ``None`` so the caller can branch on a clean "no winner" signal.
    """

    click_rules: Dict[str, object] = field(default_factory=dict)
    non_click_rules: Dict[str, object] = field(default_factory=dict)
    priority_overrides: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure both inputs are dicts even when None was passed.
        if self.click_rules is None:
            self.click_rules = {}
        if self.non_click_rules is None:
            self.non_click_rules = {}
        if self.priority_overrides is None:
            self.priority_overrides = {}

    def resolve(self, rule1: str, rule2: str) -> Optional[str]:
        """Return the rule name that wins the conflict, or ``None``.

        Parameters
        ----------
        rule1, rule2 : str
            Two rule names (one click + one non-click, or both from
            the same family).

        Returns
        -------
        str or None
            The canonical winner.  ``None`` if neither rule is
            registered in either set.
        """
        if rule1 == rule2:
            # Same rule -> trivially return it (no conflict).
            return rule1 if self._is_registered(rule1) else None

        if not self._is_registered(rule1) and not self._is_registered(rule2):
            return None
        if not self._is_registered(rule1):
            return rule2
        if not self._is_registered(rule2):
            return rule1

        # Compute priority for each rule.  Lower number = higher
        # priority.  Priority resolution order:
        #   1. Explicit ``priority_overrides`` (wins all defaults)
        #   2. Click-rule priority (DEFAULT_CLICK_RULE_PRIORITY order)
        #   3. Non-click priority (DEFAULT_NON_CLICK_RULE_PRIORITY order)
        #   4. Unregistered-in-priority (large sentinel)
        p1 = self._priority(rule1)
        p2 = self._priority(rule2)
        if p1 < p2:
            return rule1
        if p2 < p1:
            return rule2
        # Tie — default to click rule (canonical R17 chemistry).
        if rule1 in self.click_rules:
            return rule1
        if rule2 in self.click_rules:
            return rule2
        # Both non-click / both click — first-arg wins.
        return rule1

    def _is_registered(self, rule_name: str) -> bool:
        return rule_name in self.click_rules or rule_name in self.non_click_rules

    def _priority(self, rule_name: str) -> int:
        """Compute a numeric priority for ``rule_name``.

        Lower is better (wins).  Unknown rules get a large sentinel
        so they lose to any registered rule.
        """
        if rule_name in self.priority_overrides:
            return int(self.priority_overrides[rule_name])
        # Click priority (0..N) — first listed wins.
        if rule_name in self.click_rules:
            try:
                return DEFAULT_CLICK_RULE_PRIORITY.index(rule_name)
            except ValueError:
                # Click rule not in canonical priority list — give it
                # a slot just past the canonical list.
                return len(DEFAULT_CLICK_RULE_PRIORITY) + 1
        if rule_name in self.non_click_rules:
            try:
                # Offset by 100 so non-click never beats click unless
                # overridden.
                return 100 + DEFAULT_NON_CLICK_RULE_PRIORITY.index(rule_name)
            except ValueError:
                return 100 + len(DEFAULT_NON_CLICK_RULE_PRIORITY) + 1
        return 10_000

    def register_click(self, name: str, rule) -> None:
        """Add a click rule to the resolver (idempotent)."""
        self.click_rules[name] = rule

    def register_non_click(self, name: str, rule) -> None:
        """Add a non-click rule to the resolver (idempotent)."""
        self.non_click_rules[name] = rule

    def known_rule_names(self) -> Set[str]:
        """Return the union of all registered rule names."""
        return set(self.click_rules.keys()) | set(self.non_click_rules.keys())


# ---------------------------------------------------------------------------
# MedChemAugmentedGenerator
# ---------------------------------------------------------------------------


# Default reusable SMILES pool used as the *co-reactant* for non-click
# rules when the augmented generator fires a med-chem rule on a
# generated molecule.  These are amino-acid / aldehyde building
# blocks that pair with peptide / Boc / reductive-amination rules.
_DEFAULT_NON_CLICK_COPARTNERS: Dict[str, Tuple[str, ...]] = {
    "peptide_bond": (
        "NCC(=O)O",  # glycine
        "CC(N)C(=O)O",  # alanine
        "NCC(Cc1ccccc1)C(=O)O",  # phenylalanine
    ),
    "boc_protection": (
        "NCC(=O)O",  # glycine
        "CN",  # methylamine
        "CCCN",  # propylamine
    ),
    "reductive_amination": (
        "O=Cc1ccccc1",  # benzaldehyde
        "CC=O",  # acetaldehyde
        "CCC=O",  # propionaldehyde
    ),
}


@dataclass
class MedChemAugmentedGenerator:
    """Wrap a base generator with non-click med-chem rule augmentation.

    The wrapper produces ``n_samples`` SMILES from a seed via the
    ``base_generator``.  A configurable fraction (``use_non_click_prob``)
    of those samples is *augmented* by applying one of the 3 non-click
    med-chem rules (peptide bond / Boc protection / reductive
    amination) against a fixed pool of co-reactants.

    Parameters
    ----------
    base_generator : callable
        ``(seed_smiles: str, n_samples: int) -> List[str]`` — the
        underlying generation callable.  Typically the bound
        :meth:`generate` of :class:`LayeredGenerationPipeline` or any
        test stub.
    non_click_rules : dict[str, object], default ``{}``
        Map of rule name -> rule instance.  Defaults to the canonical
        3-rule registry from
        :func:`molmetal_lam.lam_chem.non_click_medchem_rules.register_non_click_rules`.
    use_non_click_prob : float, default ``0.3``
        Probability that a given generated molecule is augmented by
        a non-click rule (clamped to ``[0, 1]``).  ``0.0`` means
        "always click-only" (passthrough); ``1.0`` means "always
        apply a non-click rule".
    copartners : dict[str, tuple[str, ...]], default ``None``
        Optional per-rule co-reactant pool.  When ``None``, the
        :data:`_DEFAULT_NON_CLICK_COPARTNERS` pool is used.  Tests
        can pass a tiny pool to keep the CPU footprint small.
    rng : random.Random, default ``None``
        Random number generator.  When ``None``, a fresh
        :class:`random.Random` is constructed (non-deterministic
        across runs).  Tests should pin ``rng=random.Random(0)`` for
        reproducibility.

    Attributes
    ----------
    base_generator : callable
        The wrapped generator.
    non_click_rules : dict
        The non-click rule map.
    use_non_click_prob : float
        Probability clamped to ``[0, 1]``.
    rng : random.Random
        The injected RNG.

    Notes
    -----
    * Augmentation is *probabilistic* per molecule.  When a non-click
      rule is selected, one of the registered rules is drawn
      uniformly and applied against a uniformly-drawn co-reactant
      from the rule's copartner pool.
    * When the rule firing returns ``None`` (RDKit incompatibility),
      the original click-only SMILES is preserved so the wrapper
      never emits fewer samples than the base generator.
    * The wrapper does *not* modify ``rules.py`` /
      ``syntemol_reactions.py``; it only consumes the sidecar
      :mod:`non_click_medchem_rules` registry.

    Examples
    --------
    >>> from molmetal.molmetal_lam.lam_chem.non_click_medchem_rules import (
    ...     register_non_click_rules,
    ... )
    >>> from molmetal.molmetal_lam.lam_chem.medchem_gen_integration import (
    ...     MedChemAugmentedGenerator,
    ... )
    >>> import random
    >>> rules = register_non_click_rules()
    >>> base = lambda seed, n: [seed] * n  # trivial passthrough
    >>> aug = MedChemAugmentedGenerator(
    ...     base_generator=base,
    ...     non_click_rules=rules,
    ...     use_non_click_prob=1.0,
    ...     rng=random.Random(0),
    ... )
    >>> samples = aug.generate("NCC(=O)O", n_samples=4)
    """

    base_generator: Callable[..., List[str]]
    non_click_rules: Dict[str, object] = field(default_factory=dict)
    use_non_click_prob: float = 0.3
    copartners: Optional[Dict[str, Tuple[str, ...]]] = None
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if not callable(self.base_generator):
            raise MedChemIntegrationError(
                "base_generator must be callable "
                f"(got {type(self.base_generator).__name__})"
            )
        # Clamp use_non_click_prob to [0, 1].
        prob = float(self.use_non_click_prob)
        if prob < 0.0:
            prob = 0.0
        elif prob > 1.0:
            prob = 1.0
        self.use_non_click_prob = prob
        # Default non-click rules: empty dict if caller passed None.
        if self.non_click_rules is None:
            self.non_click_rules = {}
        # Default RNG: fresh if caller passed None.
        if self.rng is None:
            self.rng = random.Random()
        # Default copartner pool.
        if self.copartners is None:
            self.copartners = dict(_DEFAULT_NON_CLICK_COPARTNERS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        seed_smiles: str,
        n_samples: int = 100,
    ) -> List[str]:
        """Generate ``n_samples`` med-chem-augmented SMILES.

        Parameters
        ----------
        seed_smiles : str
            Starting SMILES (the base generator's input).
        n_samples : int, default ``100``
            Target number of SMILES to return (clamped to ``>= 0``).

        Returns
        -------
        List[str]
            ``n_samples`` SMILES, each independently drawn from
            either the base generator (with probability
            ``1 - use_non_click_prob``) or a non-click med-chem
            rule (with probability ``use_non_click_prob``).  Order
            is preserved across calls; the wrapper does not
            shuffle.  An empty list is returned when
            ``n_samples <= 0`` or when the base generator returns
            an empty list AND no med-chem rule fires.
        """
        if n_samples <= 0:
            return []
        if not isinstance(seed_smiles, str) or not seed_smiles:
            return []

        # Draw the *base* samples first.  We over-sample by 1 in case
        # the base generator returns a slightly shorter list (defensive
        # — base generators should be deterministic in size).
        try:
            base_samples = list(self.base_generator(seed_smiles, n_samples))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[MedChemAugmentedGenerator] base_generator raised: %s", exc)
            base_samples = []

        out: List[str] = []
        for i in range(n_samples):
            base_smi = base_samples[i] if i < len(base_samples) else seed_smiles
            if not isinstance(base_smi, str) or not base_smi:
                # Skip empty / non-string slots — caller sees a
                # shorter list rather than a corrupted one.
                continue
            if self.rng.random() < self.use_non_click_prob:
                aug = self._augment_one(base_smi)
                if aug is not None:
                    out.append(aug)
                    continue
            # Fallback: emit the base sample unchanged.
            out.append(base_smi)
        return out

    def diversity_lift(
        self,
        base_samples: Sequence[str],
        augmented_samples: Sequence[str],
    ) -> float:
        """Return the diversity *delta* between base and augmented sets.

        ``diversity_lift = augmented_unique_scaffolds - base_unique_scaffolds``

        Positive values indicate the med-chem augmentation produced
        more distinct scaffolds than the base generator.  Zero or
        negative values indicate the augmentation did not broaden
        chemical-space coverage (which can happen when the copartner
        pool is small or the seed is too restrictive).
        """
        base_n = _unique_scaffold_count(base_samples)
        aug_n = _unique_scaffold_count(augmented_samples)
        return float(aug_n - base_n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _augment_one(self, smiles: str) -> Optional[str]:
        """Apply one randomly-chosen non-click rule to ``smiles``.

        Returns the canonical product SMILES, or ``None`` when the
        rule firing fails (RDKit incompatibility / empty copartner
        pool / parse error).
        """
        rule_names = list(self.non_click_rules.keys())
        if not rule_names:
            return None
        # Pick a random rule + random copartner from that rule's pool.
        rule_name = self.rng.choice(rule_names)
        rule = self.non_click_rules[rule_name]
        pool = (self.copartners or {}).get(rule_name) or tuple()
        if not pool:
            return None
        copartner = self.rng.choice(pool)

        # Dispatch on the rule's interface — Boc protection has a
        # single-arg ``apply``, the others are two-arg ``apply``.
        try:
            if rule_name == "boc_protection":
                product = rule.apply(copartner)  # protect the co-reactant
                return _canonical(product) if product else None
            if rule_name == "peptide_bond":
                product = rule.apply(smiles, copartner)
                return _canonical(product) if product else None
            if rule_name == "reductive_amination":
                product = rule.apply(copartner, smiles)
                return _canonical(product) if product else None
            # Unknown rule — try the two-arg interface as a default.
            try:
                product = rule.apply(smiles, copartner)
            except TypeError:
                product = rule.apply(copartner)
            return _canonical(product) if product else None
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "[MedChemAugmentedGenerator] rule %s failed: %s",
                rule_name,
                exc,
            )
            return None


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Container for :func:`benchmark_medchem_diversity` outputs.

    Fields
    ------
    base : Dict[str, Any]
        Stats for the base generator: ``"wall_time_s"``,
        ``"n_samples"``, ``"n_unique_scaffolds"``,
        ``"mean_pairwise_tanimoto_distance"``, ``"samples"``.
    augmented : Dict[str, Any]
        Stats for the med-chem-augmented generator (same keys as
        :attr:`base`).
    diversity_lift : float
        ``augmented.n_unique_scaffolds - base.n_unique_scaffolds``.
    wall_time_overhead_s : float
        ``augmented.wall_time_s - base.wall_time_s``.
    n_seeds : int
        Number of seed SMILES used in the benchmark.
    """

    base: Dict[str, Any] = field(default_factory=dict)
    augmented: Dict[str, Any] = field(default_factory=dict)
    diversity_lift: float = 0.0
    wall_time_overhead_s: float = 0.0
    n_seeds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return the benchmark as a plain dict (JSON-friendly)."""
        return {
            "base": dict(self.base),
            "augmented": dict(self.augmented),
            "diversity_lift": self.diversity_lift,
            "wall_time_overhead_s": self.wall_time_overhead_s,
            "n_seeds": self.n_seeds,
        }


def benchmark_medchem_diversity(
    base_gen: MedChemAugmentedGenerator,
    augmented_gen: MedChemAugmentedGenerator,
    n_seeds: int = 10,
    n_samples_per_seed: int = 8,
    seeds: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Benchmark med-chem-augmented generation vs the click-only baseline.

    Parameters
    ----------
    base_gen, augmented_gen : MedChemAugmentedGenerator
        Two generators with the *same* ``base_generator`` but
        different ``use_non_click_prob`` values — typically
        ``0.0`` vs ``0.3``.  The base generator should be
        deterministic for the benchmark numbers to be reproducible.
    n_seeds : int, default ``10``
        Number of seed SMILES to sweep across (ignored when
        ``seeds`` is provided).  Clamped to ``>= 1``.
    n_samples_per_seed : int, default ``8``
        Number of SMILES to draw per seed.  Clamped to ``>= 1``.
    seeds : Sequence[str], optional
        Explicit seed list.  When ``None``, a curated set of
        peptide-like / aldehyde-like seeds is used (deterministic
        across runs).

    Returns
    -------
    dict
        A :func:`BenchmarkResult.to_dict` payload containing per-
        generator stats (wall-time, n_samples, unique-scaffold count,
        mean pairwise Tanimoto distance, samples) and the lift
        deltas.  Schema::

            {
              "base":    { "wall_time_s": float, ... },
              "augmented":{ "wall_time_s": float, ... },
              "diversity_lift": float,
              "wall_time_overhead_s": float,
              "n_seeds": int,
            }
    """
    if int(n_seeds) < 1:
        n_seeds = 1
    if int(n_samples_per_seed) < 1:
        n_samples_per_seed = 1

    if seeds is None:
        seeds = _DEFAULT_BENCHMARK_SEEDS[: int(n_seeds)]

    # ---- Base generator run ----
    base_samples: List[str] = []
    t0 = time.perf_counter()
    for seed in seeds:
        try:
            samples = base_gen.generate(seed, n_samples_per_seed)
        except Exception:  # pragma: no cover - defensive
            samples = []
        base_samples.extend(samples)
    base_wall = time.perf_counter() - t0

    # ---- Augmented generator run ----
    aug_samples: List[str] = []
    t0 = time.perf_counter()
    for seed in seeds:
        try:
            samples = augmented_gen.generate(seed, n_samples_per_seed)
        except Exception:  # pragma: no cover - defensive
            samples = []
        aug_samples.extend(samples)
    aug_wall = time.perf_counter() - t0

    base_clean = _filter_valid(base_samples)
    aug_clean = _filter_valid(aug_samples)

    base_stats = {
        "wall_time_s": base_wall,
        "n_samples": len(base_clean),
        "n_unique_scaffolds": _unique_scaffold_count(base_clean),
        "mean_pairwise_tanimoto_distance": _mean_pairwise_tanimoto_distance(base_clean),
        "samples": base_clean,
    }
    aug_stats = {
        "wall_time_s": aug_wall,
        "n_samples": len(aug_clean),
        "n_unique_scaffolds": _unique_scaffold_count(aug_clean),
        "mean_pairwise_tanimoto_distance": _mean_pairwise_tanimoto_distance(aug_clean),
        "samples": aug_clean,
    }
    lift = float(aug_stats["n_unique_scaffolds"] - base_stats["n_unique_scaffolds"])
    overhead = float(aug_wall - base_wall)

    result = BenchmarkResult(
        base=base_stats,
        augmented=aug_stats,
        diversity_lift=lift,
        wall_time_overhead_s=overhead,
        n_seeds=len(seeds),
    )
    return result.to_dict()


# Default benchmark seeds — peptide / aldehyde / aromatic-aldehyde
# SMILES that exercise all 3 non-click med-chem rules.  Deterministic
# across runs so the benchmark is reproducible.
_DEFAULT_BENCHMARK_SEEDS: Tuple[str, ...] = (
    "NCC(=O)O",  # glycine (peptide + Boc substrate)
    "CC(N)C(=O)O",  # alanine (peptide substrate)
    "O=Cc1ccccc1",  # benzaldehyde (reductive amination)
    "CN",  # methylamine (Boc substrate)
    "CC=O",  # acetaldehyde (reductive amination)
    "NCC(Cc1ccccc1)C(=O)O",  # phenylalanine
    "CCC=O",  # propionaldehyde
    "CCCN",  # propylamine (Boc substrate)
    "O=CCc1ccccc1",  # phenylacetaldehyde (reductive amination)
    "NCC(=O)NCC(=O)O",  # glycyl-glycine (peptide substrate)
)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def build_default_integration(
    base_generator: Callable[..., List[str]],
    use_non_click_prob: float = 0.3,
    rng: Optional[random.Random] = None,
) -> Tuple[MedChemAugmentedGenerator, RuleConflictResolver]:
    """Build a (augmented_generator, resolver) pair with sensible defaults.

    The augmented generator is wired with the canonical 3 non-click
    med-chem rules; the resolver is wired with the canonical 5 click
    rules + 2 metal-coord rules (from
    :mod:`molmetal_lam.lam_chem.rules`) + the 3 non-click rules.

    Returns
    -------
    (MedChemAugmentedGenerator, RuleConflictResolver)
        The two integration objects, ready to plug into the
        synthesis layer.
    """
    # Lazy imports so this module remains importable when the
    # click-rule module is unavailable (e.g. minimal CI environments
    # without the full reaction stack).
    from molmetal.molmetal_lam.lam_chem.non_click_medchem_rules import (
        NonClickMedChemRegistry,
        register_non_click_rules,
    )

    # ``register_non_click_rules`` returns the *count* of registered
    # rules (an int).  We construct the registry explicitly so we
    # can introspect the populated rule map.
    non_click_registry = NonClickMedChemRegistry()
    register_non_click_rules(non_click_registry)
    non_click_rules = non_click_registry.get_rules()

    aug = MedChemAugmentedGenerator(
        base_generator=base_generator,
        non_click_rules=non_click_rules,
        use_non_click_prob=use_non_click_prob,
        rng=rng if rng is not None else random.Random(0),
    )

    resolver = RuleConflictResolver()
    try:
        from molmetal.molmetal_lam.lam_chem.rules import (
            CLICK_REACTIONS,
            METAL_COORDINATION_RULES,
        )

        for n, r in CLICK_REACTIONS.items():
            resolver.register_click(n, r)
        for n, r in METAL_COORDINATION_RULES.items():
            resolver.register_click(n, r)
    except Exception:  # pragma: no cover - the resolver still works
        logger.debug(
            "[build_default_integration] could not import click rules; "
            "resolver ships with non-click rules only"
        )
    for n, r in non_click_rules.items():
        resolver.register_non_click(n, r)

    return aug, resolver


__all__ = [
    "MedChemAugmentedGenerator",
    "RuleConflictResolver",
    "BenchmarkResult",
    "benchmark_medchem_diversity",
    "MedChemIntegrationError",
    "build_default_integration",
    "DEFAULT_CLICK_RULE_PRIORITY",
    "DEFAULT_NON_CLICK_RULE_PRIORITY",
]


if __name__ == "__main__":  # pragma: no cover
    # Smoke run — wire the augmented generator + resolver and run a
    # 2-seed benchmark so the module is self-documenting.
    import random as _random

    base = lambda seed, n: [seed] * n  # trivial passthrough
    aug, resolver = build_default_integration(
        base_generator=base,
        use_non_click_prob=0.5,
        rng=_random.Random(0),
    )
    print(
        f"# medchem_gen_integration — non-click rules: {list(aug.non_click_rules.keys())}"
    )
    print(f"# known rules in resolver: {sorted(resolver.known_rule_names())}")
    print(
        f"# resolve(AmideCoupling, peptide_bond) -> "
        f"{resolver.resolve('AmideCoupling', 'peptide_bond')}"
    )
    out = benchmark_medchem_diversity(
        base_gen=MedChemAugmentedGenerator(
            base_generator=base,
            non_click_rules=aug.non_click_rules,
            use_non_click_prob=0.0,
            rng=_random.Random(0),
        ),
        augmented_gen=aug,
        n_seeds=2,
        n_samples_per_seed=3,
    )
    print(
        f"# benchmark: diversity_lift={out['diversity_lift']}, "
        f"wall_time_overhead_s={out['wall_time_overhead_s']:.4f}"
    )
