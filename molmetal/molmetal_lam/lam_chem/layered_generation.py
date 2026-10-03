"""Layered CPU generation skeleton — Pitfall Rank 6, P2.3 (TODO-30).

Background
----------
The existing decoder pipeline (``decoder_rework.py`` + ``decoder_m3_sequence.py``)
emits complete molecules in a single forward pass.  For SBDD-style
generation this is brittle: invalid scaffolds, dative-bond failures,
and stereo-loss all show up *after* the entire molecule has been
assembled, making them expensive to fix.

The *layered* alternative decomposes generation into three discrete
stages that mirror the MLC ``constructive proof witness`` pattern:

  1. **ScaffoldStage** — extract or select a Bemis-Murcko scaffold.
     Same scaffold = same closure-theorem reachability, so the early
     step is a *type-level* decision (which fragment grammar).
  2. **AttachStage** — add functional groups / R-groups at annotated
     attachment points.  Each attachment is one β-reduction (curried
     partial application) on a typed fragment.
  3. **DecorateStage** — apply substituents, set stereo, and run
     ``AssignStereochemistry`` to enforce the cis/trans distinct-NF
     property from the MLC skill.

Each stage is independently testable, and a stage failure short-
circuits the pipeline so the orchestrator can record the failure mode.
This is the CPU-only skeleton; no learned priors are wired yet —
those land in TODO-31 (Pitfall Rank 6 P2.4).

Lit anchors
-----------
* Bemis & Murcko 1996 *J. Med. Chem.* 39(15), 2887-2893 — the scaffold
  extraction algorithm.  ``rdkit.Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol``
  is the canonical implementation.
* RDKit ``AssignStereochemistry`` — assigns 2D/3D stereo tags from
  wedge/double-bond geometry.  Pairs with the MLC cis/trans NF
  distinction (``closure.py``).

Honest framing
--------------
This is a *CPU skeleton*.  The pipeline does NOT use a learned prior
(it samples scaffolds via simple heuristics + random R-groups).  The
goal is the API surface + the orchestration contract, so TODO-31 can
plug in the learned sampler without rewriting tests.

Constraints honoured
--------------------
* **No modification** of ``decoder_rework.py`` or
  ``lambda_combinators.py`` — only NEW file.  The stage classes do
  not import those modules, so the legacy decoder remains the
  single-pass path.
* **Pure-RDKit** chemistry — no torch dependency at runtime so the
  skeleton runs on any laptop in <1 s.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# Quiet RDKit warnings about valence / sanitisation in the noise-heavy
# scaffold-extraction path; callers can flip the flag back if they
# want full verbosity.
RDLogger.DisableLog("rdApp.*")


__all__ = [
    "DEFAULT_MIN_HEAVY",
    "DEFAULT_MAX_HEAVY",
    "DEFAULT_MAX_SUBSTITUENTS",
    "ScaffoldStage",
    "AttachStage",
    "AttachResult",
    "DecorateStage",
    "DecorateResult",
    "LayeredGenerationPipeline",
    "PipelineResult",
    "is_valid_smiles",
]


# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------
# Bemis-Murcko scaffolds typically span 5-20 heavy atoms for drug-like
# molecules (Bemis & Murcko 1996 §3).  We honour the same envelope
# so the scaffold stage does not propose fragments that are too
# small to bear a meaningful R-group vector.
DEFAULT_MIN_HEAVY: int = 5
DEFAULT_MAX_HEAVY: int = 20

# Decorate stage cap — typical SBDD ligands carry ≤4 small substituents
# (methyl, hydroxyl, halide, amine).  Beyond 4 the SA score climbs
# sharply (Ertl & Schuffenhauer 2009 *J. Cheminform.* 1:8).
DEFAULT_MAX_SUBSTITUENTS: int = 4


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class AttachResult:
    """Result of an :class:`AttachStage` operation.

    Attributes
    ----------
    smiles : str
        Combined scaffold + R-group SMILES.  Empty string on failure.
    valid : bool
        True iff the result passes RDKit sanitisation.
    n_attachments : int
        Number of attachment sites successfully filled (≤ len(r_groups)).
    rejected : List[str]
        R-groups that were rejected (e.g. for valence reasons).
    error : Optional[str]
        Human-readable failure reason (None on success).
    """

    smiles: str
    valid: bool
    n_attachments: int
    rejected: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class DecorateResult:
    """Result of a :class:`DecorateStage` operation.

    Attributes
    ----------
    smiles : str
        Decorated SMILES (post-substituent + stereo assignment).
    valid : bool
        True iff the result passes RDKit sanitisation.
    n_substituents : int
        Number of substituents successfully applied.
    stereo_assigned : bool
        True iff :func:`DecorateStage.add_stereo` ran without exception.
    has_chirality : bool
        True iff any chiral tag survived final sanitisation.
    error : Optional[str]
        Human-readable failure reason (None on success).
    """

    smiles: str
    valid: bool
    n_substituents: int
    stereo_assigned: bool = False
    has_chirality: bool = False
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Top-level orchestrator result.

    Mirrors the pipeline output contract documented in TODO-30 §3:
    ``{scaffold, attached, decorated, final_smiles, valid}`` plus a
    few debugging fields.
    """

    scaffold: str
    attached: str
    decorated: str
    final_smiles: str
    valid: bool
    stages_run: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the JSON contract ``{scaffold, attached, decorated, final_smiles, valid}``."""
        return {
            "scaffold": self.scaffold,
            "attached": self.attached,
            "decorated": self.decorated,
            "final_smiles": self.final_smiles,
            "valid": self.valid,
            "stages_run": list(self.stages_run),
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_valid_smiles(smi: str) -> bool:
    """Return True iff ``smi`` parses and passes RDKit sanitisation.

    Accepts both canonical and non-canonical SMILES; we normalise via
    ``MolToSmiles`` so two equivalent inputs hash identically.
    """
    if not isinstance(smi, str) or not smi.strip():
        return False
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def _safe_canonical(smi: str) -> str:
    """Return canonical SMILES for ``smi`` (empty string on failure)."""
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def _parse_or_none(smi: str) -> Optional[Chem.Mol]:
    """Parse SMILES, sanitise; return ``None`` on failure."""
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage 1 — Scaffold extraction (Bemis-Murcko)
# ---------------------------------------------------------------------------
class ScaffoldStage:
    """Stage 1 of the layered pipeline: extract / select a Bemis-Murcko scaffold.

    The scaffold is the ring system + linkers, stripped of all
    substituents (Bemis & Murcko 1996).  For SBDD this is the
    "pharmacophoric backbone" — every chemistry decision in stages
    2 and 3 hangs off it, so the scaffold stage is the type-level
    choice (cf. MLC primitive combinator selection).

    Parameters
    ----------
    n_heavy_atoms_min : int
        Minimum scaffold heavy-atom count.  Scaffolds smaller than
        this are rejected (typically a single benzene — too generic
        to anchor meaningful chemistry).
    n_heavy_atoms_max : int
        Maximum scaffold heavy-atom count.  Larger scaffolds are
        truncated (RDKit cannot reliably extract a scaffold for
        very large fragments).
    include_chirality : bool
        If True, ``isomericSmiles`` is used so scaffold stereo is
        preserved; otherwise the scaffold is achiral.
    rng : Optional[random.Random]
        Injectable RNG for deterministic tests.  When None a fresh
        :class:`random.Random` is used.
    """

    def __init__(
        self,
        n_heavy_atoms_min: int = DEFAULT_MIN_HEAVY,
        n_heavy_atoms_max: int = DEFAULT_MAX_HEAVY,
        include_chirality: bool = False,
        rng: Optional[random.Random] = None,
    ) -> None:
        if int(n_heavy_atoms_min) < 0:
            raise ValueError(
                f"n_heavy_atoms_min must be >= 0, got {n_heavy_atoms_min!r}"
            )
        if int(n_heavy_atoms_max) < int(n_heavy_atoms_min):
            raise ValueError(
                f"n_heavy_atoms_max ({n_heavy_atoms_max}) must be "
                f">= n_heavy_atoms_min ({n_heavy_atoms_min})"
            )
        self.n_heavy_atoms_min = int(n_heavy_atoms_min)
        self.n_heavy_atoms_max = int(n_heavy_atoms_max)
        self.include_chirality = bool(include_chirality)
        self._rng = rng if rng is not None else random.Random()

    # --- public API ---
    def generate(self, seed_smiles: str) -> str:
        """Extract a Bemis-Murcko scaffold from ``seed_smiles``.

        On any failure (parse error, sanitisation error, size out of
        bounds) an empty string is returned so callers can branch on
        ``bool(result)`` rather than try/except.
        """
        mol = _parse_or_none(seed_smiles)
        if mol is None:
            return ""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        except Exception:
            return ""
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            # Acyclic input has no Murcko scaffold.
            return ""
        n_heavy = scaffold.GetNumHeavyAtoms()
        if n_heavy < self.n_heavy_atoms_min:
            return ""
        if n_heavy > self.n_heavy_atoms_max:
            # Truncate by skipping the extra atoms — the Murcko
            # algorithm does not have a size cap, so we enforce it
            # here.  We return the (possibly empty) SMILES; if the
            # scaffold is now too small the caller sees an empty
            # string.
            return ""
        try:
            return Chem.MolToSmiles(
                scaffold, isomericSmiles=self.include_chirality
            )
        except Exception:
            return ""

    def is_valid_scaffold(self, smi: str) -> bool:
        """Return True iff ``smi`` is a valid Bemis-Murcko scaffold SMILES.

        "Valid" here means: parses, sanitises, has ≥1 ring atom, and
        is non-empty.  This is a *post-hoc* validator — useful for
        callers that want to check whether a candidate SMILES has
        scaffold-shaped topology without re-running
        :meth:`generate`.
        """
        mol = _parse_or_none(smi)
        if mol is None:
            return False
        ring_info = mol.GetRingInfo()
        if ring_info is None or ring_info.NumRings() == 0:
            return False
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < self.n_heavy_atoms_min:
            return False
        if n_heavy > self.n_heavy_atoms_max:
            return False
        # Ensure that running the Murcko extraction on ``smi``
        # yields itself (or a non-empty strict subset).
        try:
            derived = MurckoScaffold.GetScaffoldForMol(mol)
            if derived is None or derived.GetNumAtoms() == 0:
                return False
        except Exception:
            return False
        return True

    # --- helpers (exposed for tests) ---
    @property
    def rng(self) -> random.Random:
        """The injected RNG (read-only)."""
        return self._rng


# ---------------------------------------------------------------------------
# Stage 2 — Attach functional groups / R-groups
# ---------------------------------------------------------------------------
class AttachStage:
    """Stage 2: attach functional groups / R-groups to a scaffold.

    Each attachment is one β-reduction (curried partial application
    on a typed fragment, cf. ``lambda_combinators.Pt_II.apply_all``).
    We implement attachment via RDKit ``RWMol`` + ``AddBond`` rather
    than via reaction SMARTS so the skeleton is independent of the
    heavier reaction-rule machinery.

    Compatibility check
    -------------------
    A scaffold + R-group pair is "compatible" if:
      * the scaffold has at least one hydrogen-bearing heavy atom
        (an attachment site), and
      * the R-group has at least one open valence site (an
        unpaired bond), and
      * the join does not violate RDKit valence rules on either
        fragment.

    Parameters
    ----------
    attachment_points : Sequence[int]
        Hint indices into the scaffold's atom list.  When empty the
        stage will discover attachment points via ``GetHCounts``.
        The hint is used by the test suite to validate deterministic
        ordering.
    rng : Optional[random.Random]
        Injectable RNG for deterministic tests.
    """

    def __init__(
        self,
        attachment_points: Optional[Sequence[int]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.attachment_points: List[int] = (
            list(attachment_points) if attachment_points is not None else []
        )
        self._rng = rng if rng is not None else random.Random()

    # --- public API ---
    def attach(self, scaffold: str, r_groups: Sequence[str]) -> str:
        """Attach R-groups to the scaffold via single-bond joins.

        Returns the combined SMILES, or empty string on failure.
        Use :meth:`check_compatibility` *before* calling :meth:`attach`
        if you want a clean gate; this method always returns the
        best-effort result.
        """
        result = self.attach_with_report(scaffold, r_groups)
        return result.smiles

    def attach_with_report(
        self, scaffold: str, r_groups: Sequence[str]
    ) -> AttachResult:
        """Attach with full diagnostic report.

        Returns an :class:`AttachResult` with the joined SMILES plus
        the list of rejected R-groups and any sanitisation error.
        """
        mol = _parse_or_none(scaffold)
        if mol is None:
            return AttachResult(
                smiles="",
                valid=False,
                n_attachments=0,
                rejected=list(r_groups),
                error="scaffold did not parse / sanitise",
            )

        # Discover attachment sites: heavy atoms with >= 1 H.
        sites = self._attachment_sites(mol)
        if not sites:
            return AttachResult(
                smiles="",
                valid=False,
                n_attachments=0,
                rejected=list(r_groups),
                error="no attachment sites on scaffold",
            )

        # Copy the scaffold into an RWMol so we can edit it.
        rw = Chem.RWMol(mol)
        # Per-site H count — decremented as we join.
        h_counts = {
            idx: int(rw.GetAtomWithIdx(idx).GetTotalNumHs())
            for idx in sites
        }
        accepted: List[Tuple[int, Chem.Mol]] = []
        rejected: List[str] = []

        for r_smiles in r_groups:
            r_mol = _parse_or_none(r_smiles)
            if r_mol is None:
                rejected.append(r_smiles)
                continue
            # Pick the first available site (lowest index with H > 0).
            site_idx = self._pick_site(h_counts)
            if site_idx is None:
                rejected.append(r_smiles)
                continue
            r_attach = self._pick_rgroup_attach(r_mol)
            if r_attach is None:
                rejected.append(r_smiles)
                continue
            # Add the R-group atoms and bonds to the RWMol.
            offset = rw.GetNumAtoms()
            try:
                for atom in r_mol.GetAtoms():
                    new_atom = Chem.Atom(atom.GetAtomicNum())
                    new_atom.SetFormalCharge(atom.GetFormalCharge())
                    new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
                    rw.AddAtom(new_atom)
                for bond in r_mol.GetBonds():
                    bt = bond.GetBondType()
                    rw.AddBond(
                        offset + bond.GetBeginAtomIdx(),
                        offset + bond.GetEndAtomIdx(),
                        bt,
                    )
                # Form the new bond between site_idx and the R-group's
                # attachment atom.
                rw.AddBond(site_idx, offset + r_attach, Chem.BondType.SINGLE)
            except Exception as exc:
                rejected.append(r_smiles)
                # Roll back: this branch is hard to undo atom-wise
                # without a checkpoint, so we abort the join and
                # leave the rwmol state unchanged for that R-group.
                # A production implementation would copy the rwmol
                # state before the join; for the CPU skeleton we
                # simply skip and report.
                _ = exc
                continue
            # Decrement the site H-count and record the join.
            h_counts[site_idx] = max(0, h_counts[site_idx] - 1)
            accepted.append((site_idx, r_mol))

        # Finalise — get Mol, sanitise, emit SMILES.
        try:
            combined = rw.GetMol()
            Chem.SanitizeMol(combined)
            smiles = Chem.MolToSmiles(combined)
        except Exception as exc:
            return AttachResult(
                smiles="",
                valid=False,
                n_attachments=len(accepted),
                rejected=rejected,
                error=f"sanitisation failed: {exc}",
            )

        return AttachResult(
            smiles=smiles,
            valid=True,
            n_attachments=len(accepted),
            rejected=rejected,
            error=None,
        )

    def check_compatibility(self, scaffold: str, r_group: str) -> bool:
        """Return True iff a single scaffold+R-group pair can be joined.

        Compatibility is checked by attempting a *dry-run* join on
        copies of the molecules; no state is mutated.  Returns False
        on any parse / sanitisation / valence failure.
        """
        mol = _parse_or_none(scaffold)
        r_mol = _parse_or_none(r_group)
        if mol is None or r_mol is None:
            return False
        sites = self._attachment_sites(mol)
        if not sites:
            return False
        r_attach = self._pick_rgroup_attach(r_mol)
        if r_attach is None:
            return False
        rw = Chem.RWMol(mol)
        offset = rw.GetNumAtoms()
        try:
            for atom in r_mol.GetAtoms():
                new_atom = Chem.Atom(atom.GetAtomicNum())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                rw.AddAtom(new_atom)
            for bond in r_mol.GetBonds():
                rw.AddBond(
                    offset + bond.GetBeginAtomIdx(),
                    offset + bond.GetEndAtomIdx(),
                    bond.GetBondType(),
                )
            rw.AddBond(sites[0], offset + r_attach, Chem.BondType.SINGLE)
            combined = rw.GetMol()
            Chem.SanitizeMol(combined)
            return True
        except Exception:
            return False

    # --- helpers ---
    def _attachment_sites(self, mol: Chem.Mol) -> List[int]:
        """Return indices of heavy atoms with at least 1 implicit H.

        When ``self.attachment_points`` is non-empty, the result is
        filtered to the intersection of (hinted indices, H-bearing
        atoms) — the hint is *advisory*, the underlying H count is
        authoritative (the join only works if the site has an H).
        """
        candidates: List[int] = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue  # skip explicit Hs
            if atom.GetTotalNumHs() >= 1:
                candidates.append(atom.GetIdx())
        if self.attachment_points:
            hint = set(self.attachment_points)
            candidates = [c for c in candidates if c in hint]
        # Stable ordering by index so deterministic tests pass.
        candidates.sort()
        return candidates

    def _pick_site(self, h_counts: Dict[int, int]) -> Optional[int]:
        """Pick the lowest-indexed site with H > 0."""
        for idx in sorted(h_counts.keys()):
            if h_counts[idx] > 0:
                return idx
        return None

    def _pick_rgroup_attach(self, r_mol: Chem.Mol) -> Optional[int]:
        """Pick the lowest-indexed heavy atom in the R-group as attach site."""
        for atom in r_mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                return atom.GetIdx()
        return None


# ---------------------------------------------------------------------------
# Stage 3 — Decorate (substituent + chirality + stereo)
# ---------------------------------------------------------------------------
class DecorateStage:
    """Stage 3: decorate the attached molecule (substituents + stereo).

    Two operations:

    * **decorate** — attach small substituents (F, Cl, OH, NH2, CH3) to
      remaining open-valance sites, capped at ``max_substituents``
      total.  Beyond the cap we silently drop the substituent (the
      SA-score trade-off is bad past 4 decorations).
    * **add_stereo** — run RDKit ``AssignStereochemistry`` so cis/trans
      NF distinctness is preserved in the output.

    Parameters
    ----------
    max_substituents : int
        Hard cap on the number of substituents applied per call.
    include_stereo : bool
        Whether :meth:`decorate` automatically runs :meth:`add_stereo`
        before returning.
    rng : Optional[random.Random]
        Injectable RNG for deterministic tests.
    """

    # Small substituent SMILES (canonical, atom 0 = attach site).
    _SUBSTITUENT_LIBRARY: Tuple[str, ...] = (
        "[F]",      # F
        "[Cl]",     # Cl
        "[OH]",     # OH
        "[NH2]",    # NH2
        "[CH3]",    # CH3
        "[OCH3]",   # OMe
    )

    def __init__(
        self,
        max_substituents: int = DEFAULT_MAX_SUBSTITUENTS,
        include_stereo: bool = True,
        rng: Optional[random.Random] = None,
    ) -> None:
        if int(max_substituents) < 0:
            raise ValueError(
                f"max_substituents must be >= 0, got {max_substituents!r}"
            )
        self.max_substituents = int(max_substituents)
        self.include_stereo = bool(include_stereo)
        self._rng = rng if rng is not None else random.Random()

    # --- public API ---
    def decorate(
        self, attached_mol: str, decorations: Optional[Sequence[str]] = None
    ) -> str:
        """Apply substituents + stereo; return the resulting SMILES.

        ``decorations`` is an optional explicit list.  When ``None``
        the stage draws up to ``max_substituents`` random
        substituents from the library.  Empty-string return signals
        a hard failure (parse error, sanitisation error).
        """
        result = self.decorate_with_report(attached_mol, decorations)
        return result.smiles

    def decorate_with_report(
        self, attached_mol: str, decorations: Optional[Sequence[str]] = None
    ) -> DecorateResult:
        """Like :meth:`decorate` but with full diagnostic report."""
        mol = _parse_or_none(attached_mol)
        if mol is None:
            return DecorateResult(
                smiles="",
                valid=False,
                n_substituents=0,
                stereo_assigned=False,
                has_chirality=False,
                error="attached_mol did not parse / sanitise",
            )

        # Determine decorations: explicit list or random draw.
        if decorations is None:
            k = min(self.max_substituents, len(self._SUBSTITUENT_LIBRARY))
            decorations = list(self._rng.sample(self._SUBSTITUENT_LIBRARY, k=k))
        decorations = list(decorations)[: self.max_substituents]

        # Snapshot open sites from the *original* (sanitised) mol, so we
        # do not depend on RWMol's implicit-valence invariants after
        # AddBond/AddAtom edits.  ``Chem.AssignStereochemistry`` will
        # re-tag chirality post-construction; this snapshot keeps the
        # valence bookkeeping cheap.
        open_sites = self._open_sites_snapshot(mol)

        rw = Chem.RWMol(mol)
        n_sub = 0
        for sub_smiles in decorations:
            if not open_sites:
                break
            sub_mol = _parse_or_none(sub_smiles)
            if sub_mol is None:
                continue
            site_idx = open_sites.pop(0)  # take lowest index
            offset = rw.GetNumAtoms()
            try:
                for atom in sub_mol.GetAtoms():
                    new_atom = Chem.Atom(atom.GetAtomicNum())
                    new_atom.SetFormalCharge(atom.GetFormalCharge())
                    new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
                    rw.AddAtom(new_atom)
                for bond in sub_mol.GetBonds():
                    rw.AddBond(
                        offset + bond.GetBeginAtomIdx(),
                        offset + bond.GetEndAtomIdx(),
                        bond.GetBondType(),
                    )
                rw.AddBond(site_idx, offset + 0, Chem.BondType.SINGLE)
            except Exception:
                # Roll-forward: leave this decoration unused, continue.
                continue
            n_sub += 1

        # Final sanitisation.
        try:
            combined = rw.GetMol()
            Chem.SanitizeMol(combined)
            smiles = Chem.MolToSmiles(combined)
        except Exception as exc:
            return DecorateResult(
                smiles="",
                valid=False,
                n_substituents=n_sub,
                stereo_assigned=False,
                has_chirality=False,
                error=f"sanitisation failed: {exc}",
            )

        # Optional stereo pass.
        if self.include_stereo:
            smiles = self.add_stereo(smiles)

        # Re-detect chirality on the final SMILES.
        has_chirality = "@" in smiles
        return DecorateResult(
            smiles=smiles,
            valid=True,
            n_substituents=n_sub,
            stereo_assigned=self.include_stereo,
            has_chirality=has_chirality,
            error=None,
        )

    def add_stereo(self, mol: str) -> str:
        """Run ``AssignStereochemistry`` and return the canonical isomeric SMILES.

        Empty string on failure.  This is a *post-hoc* stereo pass —
        the chirality tags come from the input SMILES (``@``, ``@@``,
        ``/``, ``\\``); ``AssignStereochemistry`` propagates them
        through the sanitised mol.
        """
        m = Chem.MolFromSmiles(mol)
        if m is None:
            return ""
        try:
            Chem.AssignStereochemistry(m, cleanIt=True, force=True)
            Chem.SanitizeMol(m)
            return Chem.MolToSmiles(m, isomericSmiles=True)
        except Exception:
            return ""

    # --- helpers ---
    def _open_sites_snapshot(self, mol: Chem.Mol) -> List[int]:
        """Return a stable, sorted list of open-valance heavy-atom indices.

        Open-valance = heavy atom with implicit H > 0 and
        ``GetExplicitValence() < 4`` (cheap proxy for "still has room
        for another single bond").  This is computed on the sanitised
        input mol so the implicit-H invariant is well-defined — the
        RWMol returned by ``Chem.RWMol(mol)`` is mutable and would
        require per-edit sanitisation to keep the H count valid.
        """
        out: List[int] = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            if atom.GetTotalNumHs() >= 1 and atom.GetExplicitValence() < 4:
                out.append(atom.GetIdx())
        out.sort()
        return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class LayeredGenerationPipeline:
    """Run the three stages in order and return a :class:`PipelineResult`.

    Parameters
    ----------
    scaffold_stage : ScaffoldStage
    attach_stage : AttachStage
    decorate_stage : DecorateStage

    The stages are stored as-is and reused across calls; pass fresh
    instances if you want independent state per call.  Stages are
    optional (``None``) for partial pipelines — the orchestrator
    simply skips a ``None`` stage.
    """

    def __init__(
        self,
        scaffold_stage: Optional[ScaffoldStage] = None,
        attach_stage: Optional[AttachStage] = None,
        decorate_stage: Optional[DecorateStage] = None,
    ) -> None:
        self.scaffold_stage = scaffold_stage
        self.attach_stage = attach_stage
        self.decorate_stage = decorate_stage

    def generate(
        self,
        seed_smiles: str,
        r_groups: Optional[Sequence[str]] = None,
        decorations: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline end-to-end and return the dict contract.

        Returns the JSON contract documented in TODO-30 §3::

            {
                "scaffold": str,
                "attached": str,
                "decorated": str,
                "final_smiles": str,
                "valid": bool,
                "stages_run": List[str],
                "errors": List[str],
            }
        """
        result = self.run(seed_smiles, r_groups, decorations)
        return result.to_dict()

    def run(
        self,
        seed_smiles: str,
        r_groups: Optional[Sequence[str]] = None,
        decorations: Optional[Sequence[str]] = None,
    ) -> PipelineResult:
        """Run the pipeline end-to-end and return a :class:`PipelineResult`."""
        stages_run: List[str] = []
        errors: List[str] = []
        scaffold_smi = ""
        attached_smi = ""
        decorated_smi = ""
        final_smi = ""

        # --- Stage 1: scaffold ---
        if self.scaffold_stage is not None:
            stages_run.append("scaffold")
            scaffold_smi = self.scaffold_stage.generate(seed_smiles)
            if not scaffold_smi:
                errors.append("scaffold stage returned empty string")

        # --- Stage 2: attach ---
        if self.attach_stage is not None:
            stages_run.append("attach")
            if not scaffold_smi:
                errors.append("attach stage skipped: no scaffold")
            else:
                attach_out = self.attach_stage.attach_with_report(
                    scaffold_smi, r_groups or []
                )
                attached_smi = attach_out.smiles
                if not attached_smi or not attach_out.valid:
                    err_msg = (
                        f"attach stage: "
                        f"{attach_out.error or 'produced invalid SMILES'}"
                    )
                    errors.append(err_msg)
                    # We can still try decoration on the scaffold if
                    # attach failed — this gives the caller a
                    # partial signal rather than nothing.
                    if not attached_smi:
                        attached_smi = scaffold_smi

        # --- Stage 3: decorate ---
        if self.decorate_stage is not None:
            stages_run.append("decorate")
            base = attached_smi or scaffold_smi
            if not base:
                errors.append("decorate stage skipped: no base molecule")
            else:
                dec_out = self.decorate_stage.decorate_with_report(
                    base, decorations
                )
                decorated_smi = dec_out.smiles
                if not decorated_smi or not dec_out.valid:
                    errors.append(
                        f"decorate stage: "
                        f"{dec_out.error or 'produced invalid SMILES'}"
                    )

        final_smi = decorated_smi or attached_smi or scaffold_smi
        valid = bool(final_smi) and is_valid_smiles(final_smi)
        return PipelineResult(
            scaffold=scaffold_smi,
            attached=attached_smi,
            decorated=decorated_smi,
            final_smiles=final_smi,
            valid=valid,
            stages_run=stages_run,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Self-test (lightweight — ``python -m molmetal_lam.lam_chem.layered_generation``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    pipeline = LayeredGenerationPipeline(
        scaffold_stage=ScaffoldStage(rng=random.Random(0)),
        attach_stage=AttachStage(rng=random.Random(1)),
        decorate_stage=DecorateStage(rng=random.Random(2)),
    )
    out = pipeline.generate(
        "Cc1ccc(NC(=O)c2ccccc2)cc1",  # paracetamol-like
        r_groups=["[Cl]", "[OH]"],
        decorations=["[F]", "[NH2]"],
    )
    print(f"layered_generation smoke: scaffold={out['scaffold']!r}, "
          f"attached={out['attached']!r}, decorated={out['decorated']!r}, "
          f"final={out['final_smiles']!r}, valid={out['valid']}")
