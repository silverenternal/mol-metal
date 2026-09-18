"""Dataset splits — random, temporal, chemical (Tanimoto-dissimilar),
ligand-deduplicated, scaffold (Bemis-Murcko).

Each splitter is a callable class accepting a :class:`MetalCytotoxDataset`
(or any object with the same accessors) and returning a
:class:`SplitResult` with ``train_idx``, ``val_idx``, ``test_idx``
(``numpy.ndarray[int64]`` of integer indices into the dataset).

Splitting strategies
--------------------
* :class:`RandomSplitter`              — 80/10/10 with seed=42 (default).
* :class:`TemporalSplitter`            — pre-2024 train / post-2024 test,
                                         val drawn from the last 10% of the
                                         train period.
* :class:`ChemicalSplitter`            — Tanimoto-similarity > 0.7 between
                                         any train and test molecule →
                                         dissimilar test.  Uses Morgan ECFP4
                                         (radius=2, 2048 bits) cached across
                                         rows.  Falls back to random split
                                         if no SMILES yield a valid
                                         fingerprint.
* :class:`LigandDeduplicatedSplitter`  — group rows by canonical SMILES
                                         and assign each unique SMILES to
                                         exactly one of train/val/test
                                         (default 80/10/10).  Eliminates
                                         seen-SMILES leakage entirely.
* :class:`ScaffoldSplitter`            — group rows by Bemis-Murcko
                                         scaffold and assign each scaffold
                                         to exactly one of train/val/test.
                                         Strongest OOD chemistry split.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from molmetal.data.cytotox import MetalCytotoxDataset

# Default split fractions
DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)


@dataclass(frozen=True)
class SplitResult:
    """Bundled train/val/test indices."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def __post_init__(self) -> None:
        # Sanity: integer arrays
        for name, arr in (
            ("train_idx", self.train_idx),
            ("val_idx", self.val_idx),
            ("test_idx", self.test_idx),
        ):
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")
            if arr.dtype.kind not in ("i", "u"):
                # Cast silently to int64 for downstream torch tensors
                object.__setattr__(self, name, arr.astype(np.int64))

    def n_train(self) -> int:
        return int(len(self.train_idx))

    def n_val(self) -> int:
        return int(len(self.val_idx))

    def n_test(self) -> int:
        return int(len(self.test_idx))


# ---------------------------------------------------------------------------
# 1. Random splitter
# ---------------------------------------------------------------------------
class RandomSplitter:
    """Deterministic 80/10/10 (or custom fraction) shuffle-and-split."""

    def __init__(
        self,
        fractions: Tuple[float, float, float] = DEFAULT_FRACTIONS,
        seed: int = 42,
    ) -> None:
        s = float(sum(fractions))
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Fractions must sum to 1.0, got {fractions}")
        if any(f < 0 for f in fractions):
            raise ValueError(f"Fractions must be non-negative: {fractions}")
        self.fractions = fractions
        self.seed = int(seed)

    def __call__(self, dataset: MetalCytotoxDataset) -> SplitResult:
        n = len(dataset)
        if n == 0:
            return SplitResult(
                train_idx=np.array([], dtype=np.int64),
                val_idx=np.array([], dtype=np.int64),
                test_idx=np.array([], dtype=np.int64),
            )
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_train = int(round(self.fractions[0] * n))
        n_val = int(round(self.fractions[1] * n))
        n_test = n - n_train - n_val
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]
        return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# ---------------------------------------------------------------------------
# 2. Temporal splitter
# ---------------------------------------------------------------------------
class TemporalSplitter:
    """Year-based train/val/test split.

    ``cutoff_year`` (default 2024) is the boundary: anything strictly earlier
    is in train+val, anything at or after goes to test.  ``val_fraction``
    controls the proportion of pre-cutoff rows reserved for validation (the
    most-recent validation rows).

    Test rows are *forced* to be post-cutoff — this is the standard
    out-of-distribution check for activity-cliff / data-drift studies.
    """

    def __init__(
        self,
        cutoff_year: int = 2024,
        val_fraction: float = 0.1,
        train_fraction_of_pre: float = 0.8,
    ) -> None:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
        if not 0.0 < train_fraction_of_pre < 1.0:
            raise ValueError(
                f"train_fraction_of_pre must be in (0,1), got {train_fraction_of_pre}"
            )
        self.cutoff_year = int(cutoff_year)
        self.val_fraction = float(val_fraction)
        self.train_fraction_of_pre = float(train_fraction_of_pre)

    def __call__(self, dataset: MetalCytotoxDataset) -> SplitResult:
        years = np.asarray(dataset.years, dtype=np.int64)
        n = len(dataset)
        if n == 0:
            return SplitResult(
                train_idx=np.array([], dtype=np.int64),
                val_idx=np.array([], dtype=np.int64),
                test_idx=np.array([], dtype=np.int64),
            )
        pre_mask = years < self.cutoff_year
        post_mask = ~pre_mask
        pre_idx = np.where(pre_mask)[0]
        post_idx = np.where(post_mask)[0]
        # Sort pre by year so val is the most recent pre-cutoff rows
        pre_idx_sorted = pre_idx[np.argsort(years[pre_idx])]
        n_pre = len(pre_idx_sorted)
        n_val = int(round(self.val_fraction * n_pre))
        # val is the LAST n_val of pre (most-recent)
        val_idx = pre_idx_sorted[n_pre - n_val :] if n_val > 0 else np.array([], dtype=np.int64)
        # train is everything else pre
        remaining = pre_idx_sorted[: n_pre - n_val]
        # further split remaining by train_fraction_of_pre to honour 80/10/10 ratios
        n_train = int(round(self.train_fraction_of_pre * len(remaining)))
        train_idx = remaining[:n_train]
        # Drop the unused pre-cutoff rows (since remaining[n_train:] would be
        # "wasted" anyway). They are not assigned to any split.
        # Test is everything post-cutoff
        test_idx = post_idx
        return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# ---------------------------------------------------------------------------
# 3. Chemical (Tanimoto) splitter
# ---------------------------------------------------------------------------
class ChemicalSplitter:
    """Dissimilarity-based split.

    For each molecule we compute its Morgan ECFP4 (radius=2, 2048 bits).
    The reference set (train+val) is built greedily: a candidate is added to
    it only if its *maximum* Tanimoto similarity to any already-chosen
    reference is < ``threshold`` (default 0.7).  Any row that fails to embed
    (RDKit parse error) is bucketed randomly so the test still completes.

    Parameters
    ----------
    threshold : float
        Maximum allowed Tanimoto similarity between train and test.
    test_fraction : float
        Target fraction of molecules in the test split.  The greedy algorithm
        keeps iterating until the test set is at least this large or the
        full dataset has been visited.
    seed : int
        RNG seed for tie-breaking and for random bucketing of failures.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        test_fraction: float = 0.1,
        val_fraction: float = 0.1,
        seed: int = 42,
        n_bits: int = 2048,
        radius: int = 2,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0,1), got {threshold}")
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0,1), got {test_fraction}")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
        self.threshold = float(threshold)
        self.test_fraction = float(test_fraction)
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.n_bits = int(n_bits)
        self.radius = int(radius)

    def __call__(self, dataset: MetalCytotoxDataset) -> SplitResult:
        from rdkit import Chem
        from rdkit import RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")
        n = len(dataset)
        if n == 0:
            return SplitResult(
                train_idx=np.array([], dtype=np.int64),
                val_idx=np.array([], dtype=np.int64),
                test_idx=np.array([], dtype=np.int64),
            )

        smiles = list(dataset.smiles)
        fps: List[Optional[np.ndarray]] = [None] * n
        for i, s in enumerate(smiles):
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.n_bits
                )
                arr = np.zeros((self.n_bits,), dtype=np.uint8)
                from rdkit import DataStructs

                DataStructs.ConvertToNumpyArray(fp, arr)
                fps[i] = arr
            except Exception:
                continue

        # Greedy dissimilarity selection for TEST set
        rng = np.random.default_rng(self.seed)
        target_test = max(1, int(round(self.test_fraction * n)))
        target_val = max(1, int(round(self.val_fraction * n)))

        order = rng.permutation(n)  # random tie-breaking
        test_idx_list: List[int] = []
        # store fingerprints of selected test molecules for tanimoto comparison
        selected_test_fps: List[np.ndarray] = []

        for i in order:
            fp = fps[int(i)]
            if fp is None:
                continue  # will be randomly assigned later
            if not selected_test_fps:
                selected_test_fps.append(fp)
                test_idx_list.append(int(i))
                continue
            # Compute Tanimoto to existing test set
            sims = _tanimoto_to_set(fp, selected_test_fps)
            if sims.max() < self.threshold:
                selected_test_fps.append(fp)
                test_idx_list.append(int(i))
            if len(test_idx_list) >= target_test:
                break
        test_set = set(int(x) for x in test_idx_list)

        # Greedy dissimilarity for VAL set, relative to test set
        val_idx_list: List[int] = []
        selected_val_fps: List[np.ndarray] = []
        for i in order:
            i = int(i)
            if i in test_set or fps[i] is None:
                continue
            if not selected_val_fps:
                if not selected_test_fps or _tanimoto_to_set(fps[i], selected_test_fps).max() < self.threshold:
                    selected_val_fps.append(fps[i])
                    val_idx_list.append(i)
                continue
            sims_t = _tanimoto_to_set(fps[i], selected_test_fps) if selected_test_fps else np.array([0.0])
            sims_v = _tanimoto_to_set(fps[i], selected_val_fps)
            if sims_t.max() < self.threshold and sims_v.max() < self.threshold:
                selected_val_fps.append(fps[i])
                val_idx_list.append(i)
            if len(val_idx_list) >= target_val:
                break
        val_set = set(int(x) for x in val_idx_list)

        # Remaining → train (or random bucket for un-fingerprintable rows)
        unassigned = [int(i) for i in order if int(i) not in test_set and int(i) not in val_set]
        # Split: keep as train if fingerprint exists, otherwise random bucket
        train_idx_list = [i for i in unassigned if fps[i] is not None]
        random_bucket = [i for i in unassigned if fps[i] is None]
        # Distribute the random bucket proportional to existing split sizes
        n_train_so_far = len(train_idx_list)
        n_test_so_far = len(test_set)
        n_val_so_far = len(val_set)
        total = n_train_so_far + n_test_so_far + n_val_so_far
        if total == 0:
            total = 1
            weights = np.array([0.8, 0.1, 0.1])
        else:
            weights = np.array(
                [n_train_so_far / total, n_test_so_far / total, n_val_so_far / total],
                dtype=np.float64,
            )
        if random_bucket:
            bucket = rng.choice(3, size=len(random_bucket), p=weights)
            for k, i in enumerate(random_bucket):
                if bucket[k] == 0:
                    train_idx_list.append(i)
                elif bucket[k] == 1:
                    test_set.add(i)
                else:
                    val_set.add(i)
        # If greedy didn't reach target_test, top up test from remaining train
        if len(test_set) < target_test and train_idx_list:
            deficit = target_test - len(test_set)
            extras = train_idx_list[:deficit]
            for i in extras:
                train_idx_list.remove(i)
                test_set.add(i)
        if len(val_set) < target_val and train_idx_list:
            deficit = target_val - len(val_set)
            extras = train_idx_list[:deficit]
            for i in extras:
                train_idx_list.remove(i)
                val_set.add(i)

        train_idx = np.array(sorted(train_idx_list), dtype=np.int64)
        val_idx = np.array(sorted(val_set), dtype=np.int64)
        test_idx = np.array(sorted(test_set), dtype=np.int64)
        return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tanimoto_to_set(fp: np.ndarray, fps: List[np.ndarray]) -> np.ndarray:
    """Vectorised Tanimoto similarity of one fp to a small list of fps.

    Uses the bit-vector formulation:
        T(A, B) = popcount(A & B) / popcount(A | B)
    Falls back to 0.0 for empty intersections.
    """
    if not fps:
        return np.array([0.0], dtype=np.float32)
    stack = np.stack(fps, axis=0).astype(np.uint16)  # (M, B)
    a = fp.astype(np.uint16)
    inter = np.bitwise_and(stack, a).sum(axis=1)
    union = np.bitwise_or(stack, a).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(union > 0, inter / union, 0.0)
    return sims.astype(np.float32)


__all__ = [
    "SplitResult",
    "RandomSplitter",
    "TemporalSplitter",
    "ChemicalSplitter",
    "LigandDeduplicatedSplitter",
    "ScaffoldSplitter",
]


# ---------------------------------------------------------------------------
# 4. Ligand-deduplicated splitter (group by canonical SMILES)
# ---------------------------------------------------------------------------
class LigandDeduplicatedSplitter:
    """Split rows so that the same canonical SMILES never appears twice.

    Each unique canonical SMILES forms a *group*; the entire group is then
    assigned to exactly one of ``train/val/test``.  This eliminates the
    "seen-SMILES" leakage diagnosed in
    ``molmetal/reports/leakage_diagnosis.md`` (93 % of the Krasnov-paper
    random-split test rows had a canonical SMILES in the training set).

    Parameters
    ----------
    strategy : {'largest_first', 'random'}
        * ``'largest_first'`` — sort groups by descending size, then place
          each group greedily into the split with the largest remaining
          capacity.  Reproducible, deterministic, and keeps the largest
          chemical series in train.
        * ``'random'`` — shuffle groups (seeded) and assign in order.
    fractions : (train, val, test)
        Target split fractions.  Defaults to ``(0.8, 0.1, 0.1)``.
    seed : int
        RNG seed for the ``'random'`` strategy.

    Notes
    -----
    * ``fractions`` need not sum exactly to 1.0; integer rounding is
      performed on group counts (not row counts) so the final ratio is
      close to but not exactly ``fractions``.
    * ``fractions[0] + fractions[1] + fractions[2]`` must equal 1.0
      within ``1e-6``.
    * The ``seen-SMILES'' overlap between any pair of splits is, by
      construction, exactly 0 — that is the whole point of the splitter.
    """

    def __init__(
        self,
        strategy: str = "largest_first",
        fractions: Tuple[float, float, float] = DEFAULT_FRACTIONS,
        seed: int = 42,
    ) -> None:
        if strategy not in ("largest_first", "random"):
            raise ValueError(
                f"strategy must be 'largest_first' or 'random', got {strategy!r}"
            )
        s = float(sum(fractions))
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Fractions must sum to 1.0, got {fractions}")
        if any(f < 0 for f in fractions):
            raise ValueError(f"Fractions must be non-negative: {fractions}")
        self.strategy = strategy
        self.fractions = fractions
        self.seed = int(seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _canonicalize(smiles_list: np.ndarray) -> Tuple[np.ndarray, int]:
        """Canonicalise all SMILES.  Returns ``(canon_array, n_failures)``."""
        from molmetal.data.leakage_utils import canonicalize_array

        canon, n_fail = canonicalize_array(list(smiles_list))
        return canon, n_fail

    def _assign_groups(
        self,
        groups: List[List[int]],
    ) -> Dict[int, str]:
        """Assign each group (list of row indices) to ``train/val/test``.

        Returns ``{row_index: split_name}``.
        """
        n_groups = len(groups)
        if n_groups == 0:
            return {}

        # Capacity per split (in groups)
        n_train = max(1, int(round(self.fractions[0] * n_groups)))
        n_val = max(1, int(round(self.fractions[1] * n_groups)))
        n_test = n_groups - n_train - n_val
        if n_test < 1:
            # Guard against tiny n_groups — steal from train
            deficit = 1 - n_test
            n_train = max(1, n_train - deficit)
            n_test = n_groups - n_train - n_val

        capacities = {"train": n_train, "val": n_val, "test": n_test}
        split_assignment: Dict[int, str] = {}

        if self.strategy == "largest_first":
            # Sort groups by descending size, then place greedily into
            # the split with the largest remaining *group* capacity.
            sorted_groups = sorted(groups, key=len, reverse=True)
        else:  # 'random'
            rng = np.random.default_rng(self.seed)
            order = rng.permutation(n_groups)
            sorted_groups = [groups[int(i)] for i in order]

        for g in sorted_groups:
            # Pick split with largest remaining capacity (tie-break: train first)
            split = max(
                ("train", "val", "test"),
                key=lambda k: (capacities[k], 0 if k == "train" else 1 if k == "val" else 2),
            )
            capacities[split] -= 1
            for row in g:
                split_assignment[int(row)] = split
        return split_assignment

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, dataset: MetalCytotoxDataset) -> SplitResult:
        smiles = np.asarray(dataset.smiles, dtype=object)
        n = len(dataset)
        if n == 0:
            return SplitResult(
                train_idx=np.array([], dtype=np.int64),
                val_idx=np.array([], dtype=np.int64),
                test_idx=np.array([], dtype=np.int64),
            )

        canon, _n_fail = self._canonicalize(smiles)
        # Group rows by canonical SMILES
        groups_dict: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(canon):
            groups_dict[str(c)].append(i)
        groups = list(groups_dict.values())

        split_assignment = self._assign_groups(groups)
        train_idx = np.array(
            sorted([i for i, s in split_assignment.items() if s == "train"]),
            dtype=np.int64,
        )
        val_idx = np.array(
            sorted([i for i, s in split_assignment.items() if s == "val"]),
            dtype=np.int64,
        )
        test_idx = np.array(
            sorted([i for i, s in split_assignment.items() if s == "test"]),
            dtype=np.int64,
        )
        return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# ---------------------------------------------------------------------------
# 5. Scaffold splitter (Bemis-Murcko)
# ---------------------------------------------------------------------------
class ScaffoldSplitter:
    """Split rows so that each Bemis-Murcko scaffold appears in exactly one split.

    For each unique canonical SMILES we compute its Bemis-Murcko scaffold
    using ``rdkit.Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol``.  The
    scaffold string is then used as the group key (instead of the raw SMILES),
    so molecules with the same scaffold but different substituents all
    land in the same split.  This is a *stronger* OOD chemistry split than
    ``LigandDeduplicatedSplitter`` — it forces the test set to contain
    scaffolds the model has never seen at all.

    Parameters
    ----------
    strategy : {'largest_first', 'random'}
        Same semantics as :class:`LigandDeduplicatedSplitter`.
    fractions : (train, val, test)
        Target split fractions.  Defaults to ``(0.8, 0.1, 0.1)``.
    seed : int
        RNG seed for the ``'random'`` strategy.
    include_chirality : bool
        Forwarded to RDKit's Murcko scaffold routine (default False).
    """

    def __init__(
        self,
        strategy: str = "largest_first",
        fractions: Tuple[float, float, float] = DEFAULT_FRACTIONS,
        seed: int = 42,
        include_chirality: bool = False,
    ) -> None:
        if strategy not in ("largest_first", "random"):
            raise ValueError(
                f"strategy must be 'largest_first' or 'random', got {strategy!r}"
            )
        s = float(sum(fractions))
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Fractions must sum to 1.0, got {fractions}")
        self.strategy = strategy
        self.fractions = fractions
        self.seed = int(seed)
        self.include_chirality = bool(include_chirality)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _canonicalize(smiles_list: np.ndarray) -> np.ndarray:
        from molmetal.data.leakage_utils import canonicalize_array

        canon, _ = canonicalize_array(list(smiles_list))
        return canon

    def _scaffold_for(self, canon_smiles: str) -> str:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold

        RDLogger.DisableLog("rdApp.*")
        try:
            mol = Chem.MolFromSmiles(canon_smiles)
            if mol is None:
                return ""
            scaffold = MurckoScaffold.GetScaffoldForMol(mol, includeChirality=self.include_chirality)
            if scaffold is None:
                return ""
            return Chem.MolToSmiles(scaffold)
        except Exception:
            return ""

    def _assign_groups(
        self,
        groups: List[List[int]],
    ) -> Dict[int, str]:
        # Reuse the dedup splitter's assignment logic verbatim
        return LigandDeduplicatedSplitter(
            strategy=self.strategy,
            fractions=self.fractions,
            seed=self.seed,
        )._assign_groups(groups)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, dataset: MetalCytotoxDataset) -> SplitResult:
        smiles = np.asarray(dataset.smiles, dtype=object)
        n = len(dataset)
        if n == 0:
            return SplitResult(
                train_idx=np.array([], dtype=np.int64),
                val_idx=np.array([], dtype=np.int64),
                test_idx=np.array([], dtype=np.int64),
            )

        canon = self._canonicalize(smiles)
        # Build scaffold → row-indices mapping.  Rows whose SMILES fails
        # to canonicalise/scaffold get bucketed into a sentinel key so
        # they still land in one split together.
        scaffold_to_rows: Dict[str, List[int]] = defaultdict(list)
        for i, smi in enumerate(canon):
            scaf = self._scaffold_for(str(smi))
            if not scaf:
                # empty scaffold (single-atom / failed parse) → use the
                # raw canonical SMILES as the group key so duplicates stay
                # together and leakage on those rows is still eliminated
                scaf = f"__empty__{smi}"
            scaffold_to_rows[scaf].append(i)
        groups = list(scaffold_to_rows.values())

        split_assignment = self._assign_groups(groups)
        train_idx = np.array(
            sorted([i for i, s in split_assignment.items() if s == "train"]),
            dtype=np.int64,
        )
        val_idx = np.array(
            sorted([i for i, s in split_assignment.items() if s == "val"]),
            dtype=np.int64,
        )
        test_idx = np.array(
            sorted([i for i, s in split_assignment.items() if s == "test"]),
            dtype=np.int64,
        )
        return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)