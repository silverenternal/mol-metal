"""tmQM dataloader for transition-metal bond patterns.

============================================================
Background — why a tmQM-specific dataloader?
============================================================
tmQM (Balcells & Skjelstad, *J. Chem. Inf. Model.* 2020,
60, 6135-6146) is the **108k-row transition-metal organometallic
corpus** this project uses for:

* Per-metal-element bond-pattern statistics (Pt-Cl, Pt-N, Pd-N, Au-N,
  etc.) — the empirical prior that should *replace* our chemistry-
  knowledge heuristic in :mod:`molmetal_lam.lam_chem.pt_click_compat`
  once enough data is gathered.
* Warm-start fragment / sub-pocket initialisation for the Lambda-MCTS
  generator (the data here drives
  :mod:`molmetal_lam.lam_chem.cisplatin_builder` + the
  ``metal-seed`` CLI flag).
* Validation of the F2(a) metal-coordination rules + the 5x5
  ``pt_click_compat`` matrix: ``for (metal, lig) in
  get_metal_bond_statistics('Pt'): assert ('Pt', lig) in matrix``.

The parsed CSV at ``/mnt/storage/data/molmetal/tmQM/tmqm_parsed.csv``
(108,543 records × 19 columns) provides the *SMILES column* we need to
extract M-L bond patterns via RDKit.  The geometry / bond-length
columns are populated for a small subset (those with resolved
crystallographic coordinates) so we only use them as auxiliary labels.

Module surface
==============
* :class:`TmQMDataset`
    Lazy loader mirroring the :class:`PlatinAIDataset` contract.
    Returns ``(smiles, metal, oxidation_state, bond_dict)`` tuples
    where ``bond_dict`` is ``{(metal, lig_symbol): count}`` of M-L
    bonds in the molecule (e.g. ``{('Pt', 'Cl'): 2, ('Pt', 'N'): 2}``
    for cisplatin).

* :class:`TmQMBondPattern`
    Aggregate corpus-level statistics + diversity-subset selection
    (Morgan-ECFP4 MaxMin).

This is the data foundation for the metallodrug de novo vertical;
the ``WF-Metallodrug-Vertical Phase 2.5`` task closes the gap between
the PlatinAI loader (Phase 2) and the tmQM warm-start task
(WF-TmQM-Pretrained-Init).

Honest framing
==============
This is **read-only inference + filtering** — it does NOT train a
model, NOT call a docking engine, and NOT modify the tmQM file.  It
is a CPU-only pure-RDKit loader; we tested it with
``pytest -q molmetal/molmetal_lam/tests/test_tmqm_dataset.py``.
"""

from __future__ import annotations

import csv
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None
    RDLogger = None
    _HAS_RDKIT = False

try:
    from rdkit.Chem import AllChem
    _HAS_ALCHEM = True
except Exception:  # pragma: no cover
    AllChem = None
    _HAS_ALCHEM = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TMQM_DEFAULT_PATH = Path("/mnt/storage/data/molmetal/tmQM/tmqm_parsed.csv")

#: Heavy-atom window — mirrors the PlatinAI loader defaults.
HEAVY_ATOM_MIN = 4
HEAVY_ATOM_MAX = 60

#: 30 d-block transition metals (subset the tmQM corpus covers).
TRANSITION_METALS: Tuple[str, ...] = (
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
)
_TM_SET: Set[str] = frozenset(TRANSITION_METALS)

#: Default target metal set for the metallodrug vertical — the elements
#: covered by PlatinAI + MetalCytoTox + tmQM.
DEFAULT_TARGET_METALS: Tuple[str, ...] = ("Pt", "Pd", "Au", "Ir", "Ru")

#: Ligand-donor atoms of interest (used for M-L bond pattern extraction).
#: Includes halides, chalcogens, pnictogens, and carbon (alkyl / aryl).
DEFAULT_DONOR_ATOMS: Tuple[str, ...] = (
    "F", "Cl", "Br", "I",
    "O", "S", "Se", "Te",
    "N", "P", "As",
    "C",
)

#: Map from atomic number to element symbol (used for bond-pattern
#: extraction from RDKit atom objects).
_ATNUM_TO_ELEMENT: Dict[int, str] = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
    17: "Cl", 34: "Se", 35: "Br", 53: "I",
    # Add more if needed — kept short for speed.
}

#: SMILES bracket regex used to identify metal centres (re-uses the
#: pattern established in :mod:`molmetal_lam.lam_chem.platinai_dataset`).
_BRACKET_RE = re.compile(r"\[([A-Z][a-z]?)")
#: Oxidation state extractor from a bracket atom like ``[Pt+2]``.
_CHRG_RE = re.compile(r"\[([A-Z][a-z]?)([+\-]\d*)")

#: CSV column layout (from ``tmqm_parsed.csv`` header inspection).
#: The exact column count is 19; ``smiles`` is column index 18 (0-based).
CSV_COL_CSD = 0
CSV_COL_STOICH = 1
CSV_COL_METAL = 2
CSV_COL_CHARGE = 3
CSV_COL_SPIN = 4
CSV_COL_COORD = 5
CSV_COL_BO_TOTAL = 6
CSV_COL_BO_SUM = 7
CSV_COL_BO_NEIGHBORS = 8
CSV_COL_ELEC_E = 9
CSV_COL_DISP_E = 10
CSV_COL_DIPOLE = 11
CSV_COL_METAL_Q = 12
CSV_COL_HL_GAP = 13
CSV_COL_HOMO = 14
CSV_COL_LUMO = 15
CSV_COL_POLAR = 16
CSV_COL_CSD_YEARS = 17
CSV_COL_SMILES = 18


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bracket_elements(smiles: str) -> List[str]:
    """Return bracket-element list (preserves order, dedups)."""
    if not smiles:
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for tok in _BRACKET_RE.findall(smiles):
        if tok not in seen:
            out.append(tok)
            seen.add(tok)
    return out


def _infer_metal_from_csv(metal_str: str, smiles: str) -> Optional[str]:
    """Pick the *primary* transition metal — prefer CSV column, fall
    back to SMILES bracket-scan."""
    if metal_str and metal_str in _TM_SET:
        return metal_str
    bracket = _bracket_elements(smiles)
    for tok in bracket:
        if tok in _TM_SET:
            return tok
    return None


def _infer_oxidation_state(mol: "Chem.Mol", metal_atom_idx: int) -> Optional[int]:
    """Infer formal oxidation state from RDKit atom formal charge.

    Falls back to ``None`` if charge cannot be parsed.  Note this is
    the *formal* charge, not the DFT-derived oxidation state in the
    CSV column.
    """
    if mol is None or metal_atom_idx is None:
        return None
    try:
        atom = mol.GetAtomWithIdx(metal_atom_idx)
        charge = atom.GetFormalCharge()
        return int(charge) if charge is not None else None
    except Exception:
        return None


def _extract_metal_bond_dict(
    mol: "Chem.Mol",
    metal_atom_idx: int,
    donor_atoms: Sequence[str] = DEFAULT_DONOR_ATOMS,
) -> Dict[Tuple[str, str], int]:
    """Return ``{(metal_sym, donor_sym): count}`` of metal-ligand bonds.

    Walks all bonds incident on the metal atom; each non-metal
    neighbour's element symbol is the donor.  Counts duplicates.
    """
    if mol is None or metal_atom_idx is None:
        return {}
    donor_set = frozenset(donor_atoms)
    metal_atom = mol.GetAtomWithIdx(metal_atom_idx)
    metal_sym = metal_atom.GetSymbol()
    out: Dict[Tuple[str, str], int] = {}
    for bond in metal_atom.GetBonds():
        other = bond.GetOtherAtom(metal_atom)
        other_sym = other.GetSymbol()
        if other_sym not in donor_set:
            continue
        key = (metal_sym, other_sym)
        out[key] = out.get(key, 0) + 1
    return out


def _canonicalise(smiles: str) -> Optional[str]:
    if not _HAS_RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _heavy_atom_count(smiles: str) -> Optional[int]:
    if not _HAS_RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None
    if mol is None:
        return None
    return mol.GetNumHeavyAtoms()


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------
@dataclass
class TmQMDataset:
    """Lazy tmQM CSV loader for transition-metal organometallics.

    Parameters
    ----------
    path
        Path to ``tmqm_parsed.csv`` (default
        ``/mnt/storage/data/molmetal/tmQM/tmqm_parsed.csv``).
    metal_filter
        Single metal symbol or iterable — keep only rows whose SMILES
        contains that metal.  ``None`` (default) keeps all transition-
        metal rows.
    heavy_atom_min / heavy_atom_max
        Heavy-atom window (defaults 4–60 — wider than PlatinAI because
        tmQM includes many small coordination complexes with 4–8 atoms).
    dedup
        Deduplicate by canonical SMILES (default True).
    n_max
        Hard cap on returned rows (after filtering); ``None`` keeps all.
    require_smiles
        Drop rows with empty SMILES column (default True).
    """

    path: Optional[Path] = None
    metal_filter: Optional[Union[str, Sequence[str]]] = None
    heavy_atom_min: int = HEAVY_ATOM_MIN
    heavy_atom_max: int = HEAVY_ATOM_MAX
    dedup: bool = True
    n_max: Optional[int] = None
    require_smiles: bool = True

    def __post_init__(self) -> None:
        if self.path is None:
            self.path = TMQM_DEFAULT_PATH
        else:
            self.path = Path(self.path)
        if isinstance(self.metal_filter, str):
            self.metal_filter = [self.metal_filter]
        if self.metal_filter is not None:
            for m in self.metal_filter:
                if m not in _TM_SET:
                    raise ValueError(
                        f"metal_filter must be a known TM, got {m!r}"
                    )
        self._records: List[Tuple[str, str, Optional[int], Dict[Tuple[str, str], int]]] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        if not _HAS_RDKIT:
            raise ImportError("rdkit is required for TmQMDataset")
        if not self.path.is_file():
            raise FileNotFoundError(f"tmQM CSV missing: {self.path}")

        seen: Set[str] = set()
        records: List[
            Tuple[str, str, Optional[int], Dict[Tuple[str, str], int]]
        ] = []
        n_rows = 0
        n_with_smiles = 0
        n_parsed = 0
        n_kept = 0
        n_filtered = 0
        n_dedup = 0
        with self.path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            for row in reader:
                n_rows += 1
                if len(row) <= CSV_COL_SMILES:
                    continue
                raw_smiles = row[CSV_COL_SMILES].strip()
                if not raw_smiles:
                    if self.require_smiles:
                        continue
                    raw_smiles = ""
                n_with_smiles += int(bool(raw_smiles))
                csv_metal = row[CSV_COL_METAL].strip() if CSV_COL_METAL < len(row) else ""
                # Prefer csv metal column for the metal label even when
                # the SMILES column is empty; fall back to bracket scan.
                metal = _infer_metal_from_csv(csv_metal, raw_smiles)
                if metal is None:
                    # No transition metal in the molecule at all — skip.
                    if csv_metal and csv_metal not in _TM_SET:
                        # Non-TM entry — skip quietly.
                        continue
                    if raw_smiles:
                        # Skip non-parseable / non-TM entries.
                        continue
                    n_filtered += 1
                    continue
                if self.metal_filter is not None and metal not in self.metal_filter:
                    continue
                # Try to parse with RDKit; if parse fails, retain the
                # SMILES anyway (with empty bond_dict + ox_state=None)
                # unless require_smiles forces a parse.
                canon = _canonicalise(raw_smiles)
                if canon is None:
                    n_filtered += 1
                    continue
                n_parsed += 1
                mol = Chem.MolFromSmiles(canon)
                if mol is None:
                    n_filtered += 1
                    continue
                n_heavy = mol.GetNumHeavyAtoms()
                if n_heavy < self.heavy_atom_min or n_heavy > self.heavy_atom_max:
                    continue
                if self.dedup:
                    if canon in seen:
                        n_dedup += 1
                        continue
                    seen.add(canon)
                # Identify the *primary* metal atom — first atom with
                # symbol == metal.
                metal_idx: Optional[int] = None
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == metal:
                        metal_idx = atom.GetIdx()
                        break
                if metal_idx is None:
                    n_filtered += 1
                    continue
                bond_dict = _extract_metal_bond_dict(mol, metal_idx)
                ox_state = _infer_oxidation_state(mol, metal_idx)
                records.append((canon, metal, ox_state, bond_dict))
                n_kept += 1
        self._records = records
        self._loaded = True
        # Stats — kept as instance attribute for diagnostics.
        self.load_stats: Dict[str, int] = {
            "n_rows": n_rows,
            "n_with_smiles": n_with_smiles,
            "n_parsed": n_parsed,
            "n_kept": n_kept,
            "n_filtered": n_filtered,
            "n_dedup": n_dedup,
        }
        logger.info(
            "tmQM load: rows=%d with_smiles=%d parsed=%d kept=%d "
            "filtered=%d dedup=%d",
            n_rows, n_with_smiles, n_parsed, n_kept,
            n_filtered, n_dedup,
        )
        if self.n_max is not None:
            self._records = self._records[: int(self.n_max)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        self._load()
        return len(self._records)

    def __getitem__(
        self, i: int
    ) -> Tuple[str, str, Optional[int], Dict[Tuple[str, str], int]]:
        self._load()
        return self._records[i]

    def __iter__(self) -> Iterator[Tuple[str, str, Optional[int], Dict[Tuple[str, str], int]]]:
        self._load()
        return iter(self._records)

    def get_smiles_list(self) -> List[str]:
        """Return just the canonical SMILES."""
        self._load()
        return [r[0] for r in self._records]

    def get_metal_list(self) -> List[str]:
        """Return the metal-symbol list."""
        self._load()
        return [r[1] for r in self._records]

    def get_records(
        self,
    ) -> List[Tuple[str, str, Optional[int], Dict[Tuple[str, str], int]]]:
        """Return all records as ``(smiles, metal, ox_state, bond_dict)``."""
        self._load()
        return list(self._records)

    def filter_by_metal(self, metal: Union[str, Sequence[str]]) -> "TmQMDataset":
        """Return a NEW dataset with the metal filter *replaced* by ``metal``.

        Convenience wrapper — does not mutate the original dataset.
        Use this to tighten the filter (e.g. multi-metal → Pt-only).
        """
        if isinstance(metal, str):
            metal = [metal]
        # Validate the new filter against the TM set
        for m in metal:
            if m not in _TM_SET:
                raise ValueError(
                    f"filter_by_metal: not a known TM, got {m!r}"
                )
        return TmQMDataset(
            path=self.path,
            metal_filter=list(metal),
            heavy_atom_min=self.heavy_atom_min,
            heavy_atom_max=self.heavy_atom_max,
            dedup=self.dedup,
            n_max=self.n_max,
            require_smiles=self.require_smiles,
        )


# ---------------------------------------------------------------------------
# Aggregate bond-pattern class
# ---------------------------------------------------------------------------
@dataclass
class TmQMBondPattern:
    """Aggregate bond-pattern statistics across the tmQM corpus.

    Parameters
    ----------
    dataset
        A loaded :class:`TmQMDataset` (will call ``get_records()``).
    metals
        Subset of metals to include (default :data:`DEFAULT_TARGET_METALS`).
    donor_atoms
        Subset of donor-atom symbols to include in the bond-pattern
        dictionary (default :data:`DEFAULT_DONOR_ATOMS`).

    Attributes
    ----------
    bond_pattern_count : Dict[Tuple[str, str], int]
        Counts of ``(metal, donor)`` bond occurrences across the
        filtered corpus.
    metal_count : Dict[str, int]
        Counts of records per metal.
    oxidation_state_count : Dict[str, Dict[int, int]]
        Per-metal histogram of formal oxidation states.
    """

    dataset: Optional[TmQMDataset] = None
    metals: Tuple[str, ...] = DEFAULT_TARGET_METALS
    donor_atoms: Tuple[str, ...] = DEFAULT_DONOR_ATOMS
    bond_pattern_count: Dict[Tuple[str, str], int] = field(default_factory=dict)
    metal_count: Dict[str, int] = field(default_factory=dict)
    oxidation_state_count: Dict[str, Dict[int, int]] = field(default_factory=dict)
    _computed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.bond_pattern_count:
            self.bond_pattern_count = {}
        if not self.metal_count:
            self.metal_count = {}
        if not self.oxidation_state_count:
            self.oxidation_state_count = {}
        if self.dataset is not None:
            self._compute()

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    def _compute(self) -> None:
        if self._computed:
            return
        if self.dataset is None:
            raise ValueError("TmQMBondPattern needs a TmQMDataset to compute")
        records = self.dataset.get_records()
        metal_set = frozenset(self.metals)
        for _smi, metal, ox_state, bond_dict in records:
            if metal not in metal_set:
                continue
            self.metal_count[metal] = self.metal_count.get(metal, 0) + 1
            for (m_sym, d_sym), count in bond_dict.items():
                if d_sym not in frozenset(self.donor_atoms):
                    continue
                key = (m_sym, d_sym)
                self.bond_pattern_count[key] = (
                    self.bond_pattern_count.get(key, 0) + count
                )
            if ox_state is not None:
                hist = self.oxidation_state_count.setdefault(metal, {})
                hist[ox_state] = hist.get(ox_state, 0) + 1
        self._computed = True

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    def get_metal_bond_statistics(
        self, metal: str = "Pt"
    ) -> Dict[Tuple[str, str], int]:
        """Return ``{(metal, donor): count}`` for a single metal.

        This is the primary API consumed by the ``pt_click_compat``
        matrix validator — for a Pt seed you want to confirm
        ``('Pt', 'Cl')`` + ``('Pt', 'N')`` both have non-zero counts.
        """
        self._compute()
        out: Dict[Tuple[str, str], int] = {}
        for (m_sym, d_sym), count in self.bond_pattern_count.items():
            if m_sym == metal:
                out[(m_sym, d_sym)] = count
        return out

    def get_diverse_subset(
        self, n: int = 500, seed: int = 42
    ) -> List[Tuple[str, str, Optional[int], Dict[Tuple[str, str], int]]]:
        """Return a *Morgan-ECFP4 MaxMin* diverse subset of size ``n``.

        Algorithm
        ---------
        1. Pick a random pivot from the filtered records (seed-deterministic).
        2. Iteratively add the record whose Morgan-ECFP4 fingerprint is
           *farthest* (Tanimoto) from the closest already-selected
           record (MaxMin heuristic, Ashton 2002 *J. Chem. Inf. Model.*).

        Falls back to a deterministic uniform random sample if RDKit
        fingerprints cannot be generated for any record.
        """
        if self.dataset is None:
            return []
        records = self.dataset.get_records()
        if not records:
            return []
        if n >= len(records):
            return list(records)

        if not _HAS_RDKIT:
            # Fallback to deterministic uniform random subset.
            rng = random.Random(seed)
            return rng.sample(records, n)

        # Compute ECFP4 fingerprints (cached by SMILES key).
        fps: Dict[str, Any] = {}
        from rdkit import DataStructs
        for smi, metal, ox_state, bond_dict in records:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            except Exception:
                continue
            fps[smi] = fp

        # Records that successfully fingerprinted
        fittable = [
            r for r in records
            if r[0] in fps
        ]
        if not fittable:
            rng = random.Random(seed)
            return rng.sample(records, n)

        rng = random.Random(seed)
        # 1. Initial pivot
        selected: List[Tuple[str, str, Optional[int], Dict[Tuple[str, str], int]]] = []
        selected_set: Set[str] = set()
        first_idx = rng.randrange(len(fittable))
        first = fittable[first_idx]
        selected.append(first)
        selected_set.add(first[0])

        # Track min-distance from each record to the selected set
        min_dist: Dict[str, float] = {}
        for r in fittable:
            if r[0] in selected_set:
                min_dist[r[0]] = 0.0
                continue
            sim = DataStructs.TanimotoSimilarity(fps[first[0]], fps[r[0]])
            min_dist[r[0]] = 1.0 - sim

        # 2. Iteratively add the max-min record
        for _ in range(1, n):
            if len(selected) >= len(fittable):
                break
            # Find record with max min-distance, skipping already-selected
            best_key: Optional[str] = None
            best_score: float = -1.0
            for k, d in min_dist.items():
                if k in selected_set:
                    continue
                if d > best_score:
                    best_score = d
                    best_key = k
            if best_key is None:
                break
            candidate = next(r for r in fittable if r[0] == best_key)
            selected.append(candidate)
            selected_set.add(best_key)
            # Update min distances for remaining records
            for r in fittable:
                if r[0] in selected_set:
                    continue
                sim = DataStructs.TanimotoSimilarity(
                    fps[best_key], fps[r[0]]
                )
                d = 1.0 - sim
                if d < min_dist[r[0]]:
                    min_dist[r[0]] = d
        return selected

    # ------------------------------------------------------------------
    # pt_click_compat coverage validation
    # ------------------------------------------------------------------
    def validate_pt_click_compat_coverage(
        self,
        required_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Confirm the empirical tmQM statistics cover the canonical
        Pt-bond patterns consumed by :mod:`pt_click_compat`.

        Returns
        -------
        report : dict
            ``{"required": [...], "covered": [...], "missing": [...],
            "stats": {...}}`` — convenient for logging + paper
            reproducibility.
        """
        self._compute()
        if required_pairs is None:
            # Default Pt-bond patterns from cisplatin + carboplatin +
            # oxaliplatin + nedaplatin + satraplatin (the 5 canonical
            # Pt drugs in this project).
            required_pairs = [
                ("Pt", "Cl"),
                ("Pt", "N"),
                ("Pt", "O"),
                ("Pt", "C"),
                ("Pt", "S"),
                ("Pt", "P"),
                ("Pt", "Br"),
            ]
        covered = [
            pair for pair in required_pairs
            if self.bond_pattern_count.get(pair, 0) > 0
        ]
        missing = [
            pair for pair in required_pairs
            if self.bond_pattern_count.get(pair, 0) == 0
        ]
        return {
            "required": list(required_pairs),
            "covered": covered,
            "missing": missing,
            "stats": {
                "n_records": sum(self.metal_count.values()),
                "n_bond_patterns": len(self.bond_pattern_count),
                "metal_count": dict(self.metal_count),
            },
        }


__all__ = [
    "TMQM_DEFAULT_PATH",
    "HEAVY_ATOM_MIN",
    "HEAVY_ATOM_MAX",
    "TRANSITION_METALS",
    "DEFAULT_TARGET_METALS",
    "DEFAULT_DONOR_ATOMS",
    "TmQMDataset",
    "TmQMBondPattern",
    "_extract_metal_bond_dict",
    "_infer_metal_from_csv",
    "_infer_oxidation_state",
]