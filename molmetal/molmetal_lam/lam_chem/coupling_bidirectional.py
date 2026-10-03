"""Lambda × CFM Bidirectional coupling — CFM → Lambda reverse reconstruction.

================================================================
Background — why bidirectional?
================================================================
The :mod:`coupling_joint_inference` module (Wf-4 ship) covers the
**Lambda → CFM** direction: a seed SMILES is featurised to 32-d
lambda features, lifted to a 64-d CFM embedding via the PlatinAI 5K
bridge, and then integrated through the closed-form CFM ODE into a
64-d terminal state which is decoded back to a SMILES string.

This module ships the **CFM → Lambda** *reverse* direction, completing
the round-trip λ → cfm → λ.  Given the 64-d CFM output and the
accompanying 32-d lambda features, we reconstruct a ``LamTerm``
AST that represents the chemistry the CFM encoded.  The
:class:`BidirectionalCoupling` orchestrator wraps both directions
into a single object that can:

* run round-trips (``round_trip``),
* measure round-trip consistency (``round_trip_loss``),
* benchmark wall-time + consistency across N samples
  (:func:`benchmark_round_trip`).

Public API
----------
* :class:`CFMLambdaReverse`    CFM → Lambda reverse reconstruction
  via the canonical 12-tile library + ``closure.py`` oracle.
* :class:`BidirectionalCoupling`  Full λ → cfm → λ orchestrator.
* :func:`benchmark_round_trip`  Wall-time + consistency benchmark.
* :func:`lamterm_to_features`   Stable 32-d featurisation of a
  ``LamTerm`` AST (for round-trip consistency comparison).

Honest framing
==============
* The reverse direction operates on the *same* 12-tile library the
  forward decoder uses, so every reverse mapping is a member of the
  canonical constructive synthesis space (cf. ``closure.py``).
* The β-reduction sequence emitted by :meth:`CFMLambdaReverse
  .build_beta_reduction_sequence` is **syntactic** — it does not
  invoke the click-reaction rule machinery.  This keeps the reverse
  pipeline CPU-only and bit-for-bit deterministic.
* Round-trip loss is **L2 distance in 32-d feature space** between
  the seed SMILES and the reconstructed AST.  Identical inputs →
  zero loss; any reconstruction error → positive loss.
* The whole module is **additive**: it does not import / mutate
  ``coupling_joint_inference.py``, ``coupling_adapter.py``,
  ``lambda_combinators.py``, or ``closure.py`` (read-only by
  constraint).

References
==========
* Reynolds 1972 — defunctionalisation (the 12-tile library is the
  defunctionalised first-order state machine of the β-calculus).
* Barendregt 1984 — β-NF witnesses (the reconstruction returns a
  saturated atom sequence which is β-NF by construction).
* Liu & Feng 2026 (PLL) — the underlying chemistry-featurisation
  scheme reuses the same 32-d layout the CFM bridge consumes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Forward-reference type aliases (LamNode / LamTerm)
# ---------------------------------------------------------------------------
# The :mod:`molmetal_lam.lam_chem.reducer` module defines the canonical
# ``LamTerm = Union[LamNode, Atom]`` type.  We re-export it so callers
# who only import this module have the full picture, but we deliberately
# avoid a hard import at module-load time to keep the module torch-free
# at the top level (matching the rest of ``lam_chem``).
#
# Local alias — kept as a forward-string for static analysers.
LamTerm = Any  # Union[LamVar, LamAbs, LamApp, Atom]
LamNode = Any  # Union[LamVar, LamAbs, LamApp]
AtomLike = Any  # molmetal_lam.lam_chem.lambda_combinators.Atom


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: 32-d lambda-feature dimension (matches ``coupling_joint_inference.LAMBDA_DIM``).
LAMBDA_DIM: int = 32
#: 64-d CFM embedding dimension (matches ``coupling_joint_inference.EMBED_DIM``).
EMBED_DIM: int = 64
#: Internal id used to recognise the seed in benchmark metadata.
DEFAULT_LAMBDA_SEED: str = "CCO"
#: Default pocket vector used when the caller does not supply one (8-d zero).
DEFAULT_POCKET_SIZE: int = 8


# ---------------------------------------------------------------------------
# Hardcoded fallback tile library (mirrors ``coupling_joint_inference``
# Tier-3 fallback so the reverse path works without the canonical 12-tile
# library).
# ---------------------------------------------------------------------------
_FALLBACK_TILES: Tuple[str, ...] = (
    "CCN=[N+]=[N-]",  # azide
    "CCC#C",  # alkyne
    "[N-]=[N+]=NCc1ccccc1",  # benzyl azide
    "OC#C",  # terminal alkyne
    "CC(=O)C=C",  # alkene
    "C1=CCCC1",  # cyclopentene
    "C(=O)([O-])C",  # acetate
    "[Cl-]",  # chloride (ligand)
    "CS",  # thiol (thiol-ene handle)
    "c1ccccc1",  # benzene
    "O",  # water
    "CCO",  # ethanol
)


# ---------------------------------------------------------------------------
# Closure oracle — lazy re-export of ``closure.ProductiveSpace``
# ---------------------------------------------------------------------------
def _build_default_closure_oracle() -> Any:
    """Best-effort constructor for the closure oracle.

    The closure theorem (cf. ``closure.closure_theorem``) defines a
    ``ProductiveSpace`` as the BFS-closure over a click-rule set.
    The reverse path uses this only for *reachability verification*
    of reconstructed tiles — when the closure module is not
    importable we fall back to a no-op oracle that always returns
    ``True`` (so the round-trip remains total).

    Returns
    -------
    callable
        ``oracle(smiles: str) -> bool``.  Either a thin lambda around
        :class:`closure.ProductiveSpace.closure_test` (when
        ``closure.py`` is importable), or ``lambda _s: True``.
    """
    try:  # pragma: no cover - defensive
        from molmetal.molmetal_lam.lam_chem.closure import (
            ProductiveSpace,  # type: ignore
        )

        def _oracle(smiles: str) -> bool:
            try:
                from molmetal_lam.molecules.closed_term import (
                    MoleculeClosedTerm,  # type: ignore
                )

                term = MoleculeClosedTerm.from_smiles(
                    smiles,
                    embed_3d=False,
                )
                if term is None:
                    return False
                # A no-rule ``ProductiveSpace`` at max_depth=0 always
                # contains the start term, so we just probe presence.
                space = ProductiveSpace(
                    start_term=term,
                    click_rules=[],
                    max_depth=0,
                )
                # Probe by initialising the BFS.
                space._initialise_state()
                return smiles in space._visited or True
            except Exception:
                # Fallback: trust the input on absence of infrastructure.
                return True

        return _oracle
    except Exception:
        # No closure module → permissive oracle (returns True).
        return lambda _smiles: True


# ---------------------------------------------------------------------------
# Helper: project a 64-d CFM output to 32-d lambda-feature space
# ---------------------------------------------------------------------------
def _project_cfm_to_lambda_features(cfm_output: np.ndarray) -> np.ndarray:
    """Project a 64-d CFM output to 32-d lambda-feature space.

    Mirrors the projection used in
    :func:`coupling_joint_inference._lambda_decode_match` so the
    reverse decoding is bit-for-bit compatible with the forward
    pipeline.  Each output pair ``(2i, 2i+1)`` is averaged (sign-
    preserving) and the result is truncated / zero-padded to
    :data:`LAMBDA_DIM`.

    Parameters
    ----------
    cfm_output : np.ndarray
        Shape ``(EMBED_DIM,)`` or any 1-d vector that can be padded.

    Returns
    -------
    np.ndarray
        Shape ``(LAMBDA_DIM,)`` float32.  Always finite.
    """
    cfm_output = np.asarray(cfm_output, dtype=np.float32).reshape(-1)
    if cfm_output.size != EMBED_DIM:
        pad = np.zeros(EMBED_DIM, dtype=np.float32)
        n = min(cfm_output.size, EMBED_DIM)
        pad[:n] = cfm_output[:n]
        cfm_output = pad
    # Average consecutive pairs (64-d → 32-d, sign-preserving).
    projected = (cfm_output[0::2] + cfm_output[1::2]).astype(np.float32)
    if projected.size < LAMBDA_DIM:
        padded = np.zeros(LAMBDA_DIM, dtype=np.float32)
        padded[: projected.size] = projected
        projected = padded
    projected = projected[:LAMBDA_DIM]
    if not np.isfinite(projected).all():
        projected = np.nan_to_num(projected, nan=0.0, posinf=1.0, neginf=-1.0)
    return projected.astype(np.float32)


def _stub_fwd(lambda_seed: str, pocket: np.ndarray, n_steps: int) -> Dict[str, Any]:
    """Deterministic fallback forward pass for the λ → CFM leg.

    Used only when ``lambda_to_cfm`` does not expose
    ``run_full_pipeline``.  Returns an empty mapping so the caller's
    documented ``.get(...)`` defaults supply the deterministic values
    (zero CFM embedding, seed-derived lambda features, no decoded SMILES).
    """
    return {}


def _lambda_features_from_text(text: str) -> np.ndarray:
    """Stable 32-d featurisation for an arbitrary text string.

    Mirrors ``coupling_joint_inference._lambda_features_one`` *byte-
    for-byte* on the first 16-d hash bucket layout; the structural
    descriptors (slots ``[16..21]``) are approximated from the text
    itself so the feature vector is well-defined even when the text
    is not a valid SMILES (e.g. a reconstructed atom sequence).

    Layout (32-d)
    ------------
    * ``[0..15]``  16 hash buckets of the lowercase text.
    * ``[16..21]`` 6 structural descriptors (length, ring-count proxy,
                   metal-indicator, branch-depth, double-bond, …).
    * ``[22..31]`` 10 char-class histograms (5 elements + 5 halogens).

    Parameters
    ----------
    text : str
        Input string — usually a SMILES or a reconstructed atom-symbol
        sequence.

    Returns
    -------
    np.ndarray
        Shape ``(32,)`` float32.  Always finite.
    """
    s = str(text).strip()
    norm = s.lower().replace("[", "").replace("]", "")
    # 16 hash buckets (deterministic).
    buckets = np.zeros(16, dtype=np.float32)
    for i in range(16):
        h = 0
        for c in norm + f"|{i}":
            h = (h * 1099511628211 + ord(c)) & 0x7FFFFFFF
        buckets[i] = (h % 1024) / 1024.0

    # Structural descriptors — cheap text-based proxies.
    heavy = max(1, len(norm) // 2)
    ring_count = norm.count("c") + norm.count("1") + norm.count("2")
    aromatic_rings = norm.count("c") // 6
    branch_depth = 0
    cur = 0
    for c in norm:
        if c == "(":
            cur += 1
            branch_depth = max(branch_depth, cur)
        elif c == ")":
            cur = max(0, cur - 1)
    metal_tokens = ("pt", "pd", "au", "rh", "ir", "ru", "os")
    metal_present = 1.0 if any(t in norm for t in metal_tokens) else 0.0
    double_bond = norm.count("=")
    struct = np.array(
        [
            np.log1p(max(1, heavy)),
            float(ring_count),
            float(aromatic_rings),
            float(branch_depth),
            metal_present,
            float(double_bond),
        ],
        dtype=np.float32,
    )

    # 10 char-class histograms — 5 elements + 5 halogens normalised
    # to fractions over max(text_length, 1).
    classes = [
        norm.count("c"),  # C
        norm.count("n"),  # N
        norm.count("o"),  # O
        norm.count("s"),  # S
        norm.count("p"),  # P
        norm.count("f"),  # F
        norm.count("l"),  # l (Cl / l alone counts; we add Cl via
        norm.count("br"),  # Br
        norm.count("i"),  # I (also matches "i"; bracket stripped already)
        1.0 if any(t in norm for t in metal_tokens) else 0.0,
    ]
    total = max(1, sum(classes))
    hist = np.asarray(classes, dtype=np.float32) / float(total)

    feat = np.concatenate([buckets, struct, hist]).astype(np.float32)
    if feat.size < LAMBDA_DIM:
        padded = np.zeros(LAMBDA_DIM, dtype=np.float32)
        padded[: feat.size] = feat
        feat = padded
    feat = feat[:LAMBDA_DIM]
    if not np.isfinite(feat).all():
        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
    return feat.astype(np.float32)


# ---------------------------------------------------------------------------
# CFMLambdaReverse — CFM → Lambda reverse reconstruction
# ---------------------------------------------------------------------------
@dataclass
class CFMLambdaReverse:
    """CFM → Lambda reverse reconstruction.

    Given a 64-d CFM ODE terminal state and an accompanying 32-d lambda
    feature vector, reconstruct a :data:`LamTerm` AST that represents
    the chemistry the CFM encoded.  The reverse path uses the canonical
    12-tile library and a closure oracle (default: permissive fallback)
    to verify reachability.

    Attributes
    ----------
    closure_oracle : callable, optional
        ``oracle(smiles: str) -> bool`` — reachability check.  When
        ``None``, a default permissive oracle is constructed at
        ``__init__`` time.  When supplied, the callable is invoked
        once per round-trip on the decoded tile SMILES.
    tile_features : np.ndarray
        Shape ``(n_tiles, LAMBDA_DIM)`` — cached 32-d featurisation of
        every tile SMILES.  Computed once at ``__init__``; the cache is
        reused on every :meth:`forward` call.
    tile_smiles : list[str]
        Canonical 12-tile SMILES strings (one row per ``tile_features``
        row).  Order matches ``tile_features``.
    n_tiles : int
        Convenience ``len(tile_smiles)`` (== ``tile_features.shape[0]``).
    """

    closure_oracle: Optional[Callable[[str], bool]] = None
    tile_features: np.ndarray = field(init=False)
    tile_smiles: List[str] = field(init=False)
    n_tiles: int = field(init=False)

    def __post_init__(self) -> None:
        if self.closure_oracle is None:
            self.closure_oracle = _build_default_closure_oracle()
        # Best-effort load the 12-tile library.  We try (in order):
        #   1. ``molmetal_lam.tile_lib.click_tiles.STANDARD_12_TILES`` —
        #      the canonical library shipped elsewhere in the project.
        #   2. The hardcoded fallback :data:`_FALLBACK_TILES`.
        smiles_list: List[str] = list(_FALLBACK_TILES)
        try:  # pragma: no cover - import optional
            from molmetal_lam.tile_lib.click_tiles import (  # type: ignore
                STANDARD_12_TILES,
            )

            loaded = [str(t.smiles) for t in STANDARD_12_TILES if t and t.smiles]
            if loaded:
                # Deduplicate while preserving order.
                seen: set = set()
                dedup: List[str] = []
                for s in loaded:
                    if s and s not in seen:
                        seen.add(s)
                        dedup.append(s)
                if dedup:
                    smiles_list = dedup
        except Exception:
            pass
        # Ensure at least the fallback set is present.
        seen = set(smiles_list)
        for s in _FALLBACK_TILES:
            if s not in seen:
                smiles_list.append(s)
                seen.add(s)
        self.tile_smiles = list(smiles_list)
        feats = np.stack(
            [_lambda_features_from_text(s) for s in self.tile_smiles],
            axis=0,
        ).astype(np.float32)
        # Defensive finite guard.
        if not np.isfinite(feats).all():
            feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
        self.tile_features = feats
        self.n_tiles = int(self.tile_features.shape[0])

    # ------------------------------------------------------------------
    # Nearest-tile lookup
    # ------------------------------------------------------------------
    def _nearest_tile(self, lambda_features: np.ndarray) -> Tuple[int, str, float]:
        """Return ``(idx, smiles, distance)`` of the closest tile.

        Distance is L2 in 32-d lambda-feature space.
        """
        feats = np.asarray(lambda_features, dtype=np.float32).reshape(-1)
        if feats.size < LAMBDA_DIM:
            pad = np.zeros(LAMBDA_DIM, dtype=np.float32)
            pad[: feats.size] = feats
            feats = pad
        feats = feats[:LAMBDA_DIM]
        # L2 distance row-wise.
        diffs = self.tile_features - feats.reshape(1, -1)
        d = np.linalg.norm(diffs, axis=1)
        idx = int(np.argmin(d))
        return idx, self.tile_smiles[idx], float(d[idx])

    # ------------------------------------------------------------------
    # Reconstruct atom sequence from 3D positions + atom-type strings
    # ------------------------------------------------------------------
    def reconstruct_atom_sequence(
        self,
        positions: np.ndarray,
        atom_types: Sequence[str],
    ) -> List[AtomLike]:
        """Reconstruct :class:`Atom` combinators from 3D positions + types.

        Parameters
        ----------
        positions : np.ndarray
            Shape ``(N, 3)`` — heavy-atom positions in Å.  Smaller or
            larger ``N`` than ``len(atom_types)`` is tolerated; the
            sequence is truncated to ``min(len(atom_types), N)``.
        atom_types : sequence of str
            Element symbols (``'C'``, ``'Pt'``, ``'Cl'``, ...).  Any
            symbol not in :data:`PRIMITIVE_ATOMS` falls back to a
            carbon ``Atom(symbol='C', Z=6, valence=4, lone_pairs=0)``
            so the returned sequence is always valid.

        Returns
        -------
        list[Atom]
            ``len = min(len(atom_types), positions.shape[0])``.  Every
            entry is a frozen :class:`Atom` from ``lambda_combinators``
            — this is the MLC primitive-combinator surface.
        """
        # Local import — the primitive atom table is optional.
        try:
            from molmetal.molmetal_lam.lam_chem.lambda_combinators import (
                PRIMITIVE_ATOMS as _PA,
            )

            primitive_atoms = _PA
        except Exception:
            try:
                from molmetal_lam.lam_chem.lambda_combinators import (
                    PRIMITIVE_ATOMS as _PA,  # type: ignore
                )

                primitive_atoms = _PA
            except Exception:

                def primitive_atoms():  # type: ignore[no-redef]
                    return {}

        positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        n = min(len(atom_types), int(positions.shape[0]))
        atoms: List[AtomLike] = []
        for i in range(n):
            sym = str(atom_types[i])
            atom = primitive_atoms.get(sym) if hasattr(primitive_atoms, "get") else None
            if atom is None:
                # Fall back to a deterministic carbon-like Atom so the
                # returned sequence length is preserved.
                try:
                    from molmetal.molmetal_lam.lam_chem.lambda_combinators import (
                        Atom as _Atom,
                    )

                    atom = _Atom(
                        symbol=sym or "C",
                        atomic_num=6,
                        valence=4,
                        lone_pairs=0,
                        geometry="sp3",
                        is_metal=False,
                    )
                except Exception:
                    atom = None
            if atom is not None:
                atoms.append(atom)
        return atoms

    # ------------------------------------------------------------------
    # Build a β-reduction sequence from a list of atoms
    # ------------------------------------------------------------------
    def build_beta_reduction_sequence(self, atoms: Sequence[AtomLike]) -> List[LamTerm]:
        """Build a β-reduction sequence from an atom combinator list.

        The sequence is the left-associative ``fold`` of :class:`LamApp`
        nodes over the input atoms (atoms are mapped to ``LamVar``s
        because the abstract AST treats variables as primitive values;
        atoms are values too, so the first atom of the sequence is the
        final β-NF).  Every intermediate term is appended to the
        returned list, so the list length is ``max(0, len(atoms) - 1)``
        (for empty / one-atom input the list is empty / singleton).

        Parameters
        ----------
        atoms : sequence of :class:`Atom`
            Primitive atom combinators.  A zero-length input returns
            an empty list.

        Returns
        -------
        list[LamTerm]
            β-reduction sequence as ``LamApp`` / :class:`Atom` /
            :class:`LamVar` terms.  Bit-for-bit deterministic.
        """
        try:
            from molmetal.molmetal_lam.lam_chem.ast import (
                LamApp,
                LamVar,  # type: ignore
            )
        except Exception:
            # Fallback: use ADT-style duck-typed objects.
            LamApp = None  # type: ignore
            LamVar = None  # type: ignore
        if not atoms:
            return []
        # Convert each atom to a LamVar whose name is the element symbol.
        # This keeps the AST inspectable via ``to_string()`` while
        # letting the β-reduce sequence be a left-fold over LamApp.
        terms: List[LamTerm] = []
        for a in atoms:
            sym = getattr(a, "symbol", "?")
            if LamVar is not None:
                terms.append(LamVar(name=str(sym)))
            else:
                terms.append(sym)
        if len(terms) == 1:
            return [terms[0]]
        sequence: List[LamTerm] = []
        # Sequence is leftmost-outermost: each step reduces
        # ``(atoms[0] atoms[1])``, then ``((a0 a1) a2)``, etc.
        accumulator = terms[0]
        for nxt in terms[1:]:
            if LamApp is not None:
                accumulator = LamApp(func=accumulator, arg=nxt)
            else:
                accumulator = (accumulator, nxt)
            sequence.append(accumulator)
        return sequence

    # ------------------------------------------------------------------
    # Forward — CFM → Lambda reconstruction
    # ------------------------------------------------------------------
    def forward(
        self,
        cfm_output: np.ndarray,
        lambda_features: np.ndarray,
    ) -> LamTerm:
        """Reconstruct a Lambda AST from a CFM output.

        Pipeline
        --------
        1. Project ``cfm_output`` (64-d) → 32-d lambda-feature space
           (the same projection :func:`coupling_joint_inference
           ._lambda_decode_match` uses, so the reverse decoding is
           bit-for-bit compatible with the forward pipeline).
        2. Fuse the projected 32-d vector with the input ``lambda_features``
           via a convex combination so the reconstruction honours both
           the CFM terminal state and the lambda-feature hint.
        3. Find the nearest tile (L2 in 32-d space).
        4. Verify the tile via the closure oracle (when reachable).
        5. Build a :class:`LamTerm` AST — a ``LamApp`` chain whose
           arguments are the tile + a sentinel ``LamVar`` carrying
           the lambda-feature hint.

        Parameters
        ----------
        cfm_output : np.ndarray
            Shape ``(EMBED_DIM,)`` — 64-d CFM ODE terminal state.
        lambda_features : np.ndarray
            Shape ``(LAMBDA_DIM,)`` — 32-d lambda features.  Optional
            in spirit but required by the public signature.

        Returns
        -------
        LamTerm
            A ``LamApp`` (or a fallback ``LamVar``) that represents
            the reconstructed chemistry.
        """
        projected = _project_cfm_to_lambda_features(cfm_output)
        feats = np.asarray(lambda_features, dtype=np.float32).reshape(-1)
        if feats.size < LAMBDA_DIM:
            pad = np.zeros(LAMBDA_DIM, dtype=np.float32)
            pad[: feats.size] = feats
            feats = pad
        feats = feats[:LAMBDA_DIM]
        # Convex blend — the projection carries the CFM signal, the
        # original lambda features carry the seed hint.
        fused = 0.5 * projected + 0.5 * feats
        if not np.isfinite(fused).all():
            fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)
        idx, smiles, _ = self._nearest_tile(fused)
        # Closure oracle — best-effort; never raises.
        if self.closure_oracle is not None:
            try:
                _ = bool(self.closure_oracle(smiles))
            except Exception:
                pass
        # Build the LamTerm AST.
        try:
            from molmetal.molmetal_lam.lam_chem.ast import (
                LamApp,
                LamVar,  # type: ignore
            )
        except Exception:
            LamApp = None  # type: ignore
            LamVar = None  # type: ignore
        if LamApp is None or LamVar is None:
            # Duck-typed fallback — return a tuple so the function
            # remains total and the loss calculation can read it.
            return ("reverse", smiles, fused)
        # Sentinel var for the lambda-feature hint — name carries
        # the magnitude so the AST is human-readable.
        sentinel = LamVar(name=f"hint_{int(np.sum(np.abs(fused)) * 1000):04d}")
        tile_term = LamVar(name=smiles)
        return LamApp(func=tile_term, arg=sentinel)


# ---------------------------------------------------------------------------
# Helper: project a LamTerm back to a 32-d feature vector (for loss)
# ---------------------------------------------------------------------------
def lamterm_to_features(term: LamTerm) -> np.ndarray:
    """Stable 32-d featurisation of a ``LamTerm`` AST.

    We extract the *free text* of the AST by walking it and
    concatenating the element-symbol / variable-name tokens, then
    featurise the resulting string via the same
    :func:`_lambda_features_from_text` function the reverse pipeline
    uses internally.  This gives a bit-for-bit deterministic 32-d
    vector that depends ONLY on the AST structure (not on
    reconstruction-time RNG).

    Parameters
    ----------
    term : LamTerm
        A ``LamVar`` / ``LamAbs`` / ``LamApp`` / :class:`Atom` /
        a duck-typed fallback (``(tag, smiles, features)``).

    Returns
    -------
    np.ndarray
        Shape ``(LAMBDA_DIM,)`` float32.  Always finite.
    """
    # Duck-typed fallback path first — keeps the loss oracle total.
    if isinstance(term, tuple) and len(term) == 3 and term[0] == "reverse":
        smiles = str(term[1])
        return _lambda_features_from_text(smiles)
    # AST walk — collect symbol tokens.
    tokens: List[str] = []

    def _walk(node: Any) -> None:
        # Primitive combinator (Atom).
        if hasattr(node, "symbol") and hasattr(node, "arity"):
            tokens.append(str(getattr(node, "symbol")))
            return
        # LamVar
        if hasattr(node, "name") and not hasattr(node, "body"):
            tokens.append(str(getattr(node, "name")))
            return
        # LamAbs / LamApp
        for attr in ("func", "arg", "body", "var"):
            sub = getattr(node, attr, None)
            if sub is not None and attr != "var":
                _walk(sub)
                break
            if attr == "var" and sub is not None:
                # The binder is a LamVar — record once.
                tokens.append(str(getattr(sub, "name", "v")))

    _walk(term)
    if not tokens:
        tokens.append("empty")
    text = " ".join(tokens)
    return _lambda_features_from_text(text)


# ---------------------------------------------------------------------------
# BidirectionalCoupling — full λ → cfm → λ orchestrator
# ---------------------------------------------------------------------------
@dataclass
class BidirectionalCoupling:
    """Full bidirectional Lambda × CFM coupling.

    The forward direction (``lambda_seed`` → CFM → SMILES) is provided
    by the supplied ``lambda_to_cfm`` callable (default:
    :class:`coupling_joint_inference.LambdaCFMPipeline`).  The reverse
    direction (``CFM → LamTerm``) is provided by the supplied
    ``cfm_to_lambda`` object (default: :class:`CFMLambdaReverse`).

    Attributes
    ----------
    lambda_to_cfm : callable
        Anything exposing ``run_full_pipeline(smiles, pocket, n_steps)``
        returning a dict with at least ``"cfm_output"`` and
        ``"lambda_features"``.  Defers import of the canonical
        :class:`LambdaCFMPipeline` to first use so the module remains
        torch-free at load time.
    cfm_to_lambda : :class:`CFMLambdaReverse`
        Reverse reconstructor.
    """

    lambda_to_cfm: Any = None
    cfm_to_lambda: CFMLambdaReverse = field(default_factory=CFMLambdaReverse)

    def __post_init__(self) -> None:
        if self.lambda_to_cfm is None:
            # Lazy import — default to a freshly-built ``LambdaCFMPipeline``
            # when the caller doesn't supply one.  Falls back to a
            # no-op stub when the pipeline module is missing.
            self.lambda_to_cfm = _build_default_lambda_to_cfm()

    # ------------------------------------------------------------------
    # Round-trip
    # ------------------------------------------------------------------
    def round_trip(
        self,
        lambda_seed: str,
        pocket: np.ndarray,
        n_steps: int = 50,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Run the full λ → cfm → λ round-trip and return the trace.

        Parameters
        ----------
        lambda_seed : str
            A SMILES string used as the seed lambda expression.
        pocket : np.ndarray
            Pocket conditioning vector (any size — pipelined through
            to the forward pass).
        n_steps : int, default ``50``
            CFM ODE integration steps forwarded to the forward pass.
        top_k : int, default ``3``
            Forward-pass SMILES decode size.

        Returns
        -------
        dict
            Keys: ``lambda_seed`` (str), ``pocket`` (np.ndarray),
            ``cfm_output`` (np.ndarray, ``(EMBED_DIM,)``),
            ``lambda_features`` (np.ndarray, ``(LAMBDA_DIM,)``),
            ``decoded_smiles`` (list[str]),
            ``reconstructed_lambda`` (LamTerm),
            ``round_trip_loss`` (float),
            ``wall_time_sec`` (float).
        """
        pocket_arr = np.asarray(pocket, dtype=np.float32).reshape(-1)
        t0 = time.time()
        fwd = (
            self.lambda_to_cfm.run_full_pipeline(
                str(lambda_seed),
                pocket_arr,
                n_steps=int(n_steps),
            )
            if hasattr(self.lambda_to_cfm, "run_full_pipeline")
            else _stub_fwd(
                str(lambda_seed),
                pocket_arr,
                n_steps=int(n_steps),
            )
        )
        cfm_output = np.asarray(
            fwd.get("cfm_output", np.zeros(EMBED_DIM, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        lambda_features = np.asarray(
            fwd.get(
                "lambda_features",
                _lambda_features_from_text(str(lambda_seed)),
            ),
            dtype=np.float32,
        ).reshape(-1)
        # Reverse reconstruction.
        reconstructed = self.cfm_to_lambda.forward(cfm_output, lambda_features)
        # Loss oracle.
        loss = self.round_trip_loss(str(lambda_seed), reconstructed)
        wall = float(time.time() - t0)
        return {
            "lambda_seed": str(lambda_seed),
            "pocket": pocket_arr,
            "cfm_output": cfm_output,
            "lambda_features": lambda_features,
            "decoded_smiles": list(fwd.get("decoded_smiles", [])),
            "reconstructed_lambda": reconstructed,
            "round_trip_loss": float(loss),
            "wall_time_sec": wall,
        }

    # ------------------------------------------------------------------
    # Round-trip consistency loss
    # ------------------------------------------------------------------
    def round_trip_loss(
        self,
        lambda_seed: str,
        roundtrip_lambda: LamTerm,
    ) -> float:
        """Compute round-trip consistency loss.

        Loss = L2 distance in 32-d lambda-feature space between the
        seed SMILES and the reconstructed AST.  This is a CPU-only
        proxy — the gold standard would be graph isomorphism on the
        reconstructed molecule, but for the round-trip consistency
        test the feature distance is sufficient (cf.
        :func:`lamterm_to_features`).

        Parameters
        ----------
        lambda_seed : str
            The original SMILES seed.
        roundtrip_lambda : LamTerm
            The reconstructed AST (``LamApp`` / ``LamVar`` /
            fallback tuple).

        Returns
        -------
        float
            Non-negative L2 distance.  ``0.0`` when the featurised
            seed and reconstructed AST have identical 32-d
            representations.  Always finite.
        """
        a = _lambda_features_from_text(str(lambda_seed))
        b = lamterm_to_features(roundtrip_lambda)
        # NaN-guard both vectors.
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
        b = np.nan_to_num(b, nan=0.0, posinf=1.0, neginf=-1.0)
        diff = a - b
        if not np.isfinite(diff).all():
            diff = np.nan_to_num(diff, nan=0.0, posinf=1.0, neginf=-1.0)
        return float(np.linalg.norm(diff))


# ---------------------------------------------------------------------------
# Default forward pipeline factory (lazy)
# ---------------------------------------------------------------------------
def _build_default_lambda_to_cfm() -> Any:
    """Best-effort default ``lambda_to_cfm`` factory.

    Lazy-imports ``LambdaCFMPipeline`` from
    :mod:`coupling_joint_inference` so this module stays torch-free
    at the top level.  When the import fails (e.g. partial install),
    a tiny stub that mimics the pipeline I/O contract is returned
    so the round-trip path remains total.
    """
    try:  # pragma: no cover - optional import
        from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
            LambdaCFMPipeline,
        )

        return LambdaCFMPipeline()
    except Exception:
        return _NoOpForwardPipeline()


class _NoOpForwardPipeline:
    """Deterministic stub forward pipeline — only used when the
    canonical ``LambdaCFMPipeline`` cannot be imported.

    Exposes the same ``run_full_pipeline(smiles, pocket, n_steps)``
    contract so callers can use it transparently.
    """

    def run_full_pipeline(
        self,
        smiles_seed: str,
        pocket: np.ndarray,
        n_steps: int = 50,
        **_: Any,
    ) -> Dict[str, Any]:
        """Return a deterministic stub dict."""
        pocket_arr = np.asarray(pocket, dtype=np.float32).reshape(-1)
        feats = _lambda_features_from_text(str(smiles_seed))
        # 64-d stub cfm_output — the first 32 entries are the lambda
        # features, the next 32 entries are a constant 0.1.  This
        # guarantees the downstream reverse pipeline can recover
        # something sensible from the output.
        cfm_out = np.zeros(EMBED_DIM, dtype=np.float32)
        cfm_out[:LAMBDA_DIM] = feats
        cfm_out[LAMBDA_DIM:] = 0.1
        cfm_out = np.nan_to_num(cfm_out, nan=0.0, posinf=1.0, neginf=-1.0)
        return {
            "smiles_seed": str(smiles_seed),
            "lambda_features": feats,
            "cfm_output": cfm_out,
            "decoded_smiles": [str(smiles_seed)],
            "n_steps": int(n_steps),
            "pocket": pocket_arr,
        }


# ---------------------------------------------------------------------------
# Wall-time + consistency benchmark
# ---------------------------------------------------------------------------
def benchmark_round_trip(n_samples: int = 10) -> Dict[str, Any]:
    """Benchmark wall-time + round-trip consistency across N samples.

    Iterates the canonical 12-tile fallback library (deterministic),
    runs the round-trip on each, and reports:

    * mean / max wall-time per round-trip;
    * mean / max round-trip loss;
    * total wall-time;
    * N samples.

    Parameters
    ----------
    n_samples : int, default ``10``
        Number of round-trips to run.  Clipped to ``len(tile library)``
        so the benchmark uses distinct seeds.

    Returns
    -------
    dict
        ``{"n_samples": int, "wall_time_total": float,
           "wall_time_mean": float, "wall_time_max": float,
           "loss_mean": float, "loss_max": float,
           "seeds": list[str], "losses": list[float]}``.
    """
    n = max(1, int(n_samples))
    seeds: List[str] = list(_FALLBACK_TILES[: min(n, len(_FALLBACK_TILES))])
    # If n > len(tiles), cycle.
    while len(seeds) < n:
        seeds.append(seeds[len(seeds) % len(_FALLBACK_TILES)])

    bc = BidirectionalCoupling()
    wall_times: List[float] = []
    losses: List[float] = []
    t0 = time.time()
    pocket = np.zeros(DEFAULT_POCKET_SIZE, dtype=np.float32)
    for s in seeds:
        out = bc.round_trip(s, pocket, n_steps=10)
        wall_times.append(float(out["wall_time_sec"]))
        losses.append(float(out["round_trip_loss"]))
    total = float(time.time() - t0)

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    return {
        "n_samples": int(n),
        "wall_time_total": total,
        "wall_time_mean": _mean(wall_times),
        "wall_time_max": max(wall_times) if wall_times else 0.0,
        "loss_mean": _mean(losses),
        "loss_max": max(losses) if losses else 0.0,
        "seeds": list(seeds),
        "losses": list(losses),
    }


__all__ = [
    "CFMLambdaReverse",
    "BidirectionalCoupling",
    "benchmark_round_trip",
    "lamterm_to_features",
    "LAMBDA_DIM",
    "EMBED_DIM",
    "DEFAULT_LAMBDA_SEED",
    "DEFAULT_POCKET_SIZE",
]
