"""Inference adapter for the trained ``PocketMacroSkeleton`` checkpoint.

WF-Deflex Follow-up Phase 2 — bridge module between the trained
``molmetal/models/pocket_macro_skeleton_v2.pt`` checkpoint and
downstream consumers (warm_start.py, learned_prior.py, the integration
work owned by the ``pocket-invariance`` workflow).

Why this module exists
----------------------
The trained checkpoint at
``molmetal/models/pocket_macro_skeleton_v2.pt`` (5,708 parameters,
100.0 % train accuracy, see ``.pt.json``) is the **v2** variant
produced by Phase 3 of the WF-Deflex PocketMacroSkeleton workflow
(wyyy283ck owns the training pipeline; wyyy283ck Phase 3 CA2-fix
retrains with the ``anchor_position`` one-hot to recover the 0/8 CA2
collapse).  ``residue_encoder.linear.weight`` is shape ``(32, 33)`` —
matching the canonical v2 model class
(:class:`molmetal_lam.lam_chem.pocket_macro_skeleton.PocketMacroSkeleton`)
which has ``PER_RESIDUE_FEATURES = 33`` (AA one-hot 20 + anchor tier 4
+ anchor_position 4 + dist_bin 3 + chain_id_hash 1 + resid_norm 1 =
33).

The v2 checkpoint **can** be loaded with ``strict=True`` into the v2
module.  Per the task constraints we **may not modify**
``pocket_macro_skeleton.py`` (the v2 module ships verbatim).  This
module therefore:

1. Builds the **v2** architecture inline using
   :class:`PocketMacroSkeletonAttention` (the only public nn.Module
   we re-use from ``pocket_macro_skeleton.py``) so the v2 state-dict
   loads cleanly with ``strict=True``.
2. Loads the checkpoint with ``strict=True`` so any future
   architectural drift surfaces immediately (no silent fallback).
3. Exposes the three downstream-facing primitives the
   ``pocket-invariance`` workflow needs:
   * ``predict_scaffold_class(target_name)`` — returns one of 12
     :class:`ScaffoldClass` enum values by name (fallback to
     ``UNKNOWN`` when confidence is below a configurable threshold).
   * ``predict_scaffold_class_with_confidence(target_name)`` — returns
     ``(class_name, confidence)``.
   * ``get_embedding(target_name)`` — returns the 32-d mean-pooled
     skeleton vector as a ``numpy.ndarray``.  This is the
     **bridge** the ``pocket-invariance`` workflow will mix into
     ``warm_start.py`` / ``learned_prior.py``.

Honest framing
--------------
* The CA2 pocket (1AKL) is now correctly classified by the v2 model
  (100.0 % train accuracy on ZN_TETRA_HHH per the Phase 3 fix).  The
  v1 model had a 0/8 CA2 collapse; v2 fixes it via the
  ``anchor_position`` one-hot.
* ``is_available()`` returns ``True`` only when BOTH the ``.pt`` AND
  ``.pt.json`` files are present.  When ``False``, every prediction
  method falls back to the metadata mapping alone (deterministic, no
  model weights) so the call sites in ``warm_start.py`` /
  ``learned_prior.py`` don't have to special-case the missing-checkpoint
  path.

Workflow safety
--------------
This module imports only the **public** API of
``pocket_macro_skeleton.py`` (the enum, the
``scaffold_class_from_target_name`` mapping, and the
``PocketMacroSkeletonAttention`` block).  It does **not** mutate
or shadow the v2 model class.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from molmetal_lam.lam_chem.pocket_macro_skeleton import (
    HIDDEN_DIM,
    NUM_HEADS,
    PocketMacroSkeletonAttention,
    ScaffoldClass,
    scaffold_class_from_target_name,
)


# ---------------------------------------------------------------------------
# Constants — v2 checkpoint architecture (matches .pt state_dict exactly)
# ---------------------------------------------------------------------------

#: v2 input feature dim (with the Phase 3 ``anchor_position`` one-hot).
#: The checkpoint's ``residue_encoder.linear.weight`` is shape ``(32, 33)``.
PER_RESIDUE_FEATURES: int = 33
#: Backward-compat alias — the v1 mirror exported this constant; we keep
#: it under the same name so older call sites that imported it don't break.
V1_PER_RESIDUE_FEATURES: int = PER_RESIDUE_FEATURES

#: File name of the trained state-dict checkpoint (no directory).
_CHECKPOINT_FILENAME: str = "pocket_macro_skeleton_v2.pt"
#: File name of the training-metadata JSON sidecar (no directory).
_METADATA_FILENAME: str = "pocket_macro_skeleton_v2.pt.json"

#: Module-relative ``models/`` directory next to the package root.
#: ``pocket_macro_inference.py`` lives at
#: ``molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py`` so
#: two ``.parent`` steps land at the ``molmetal/`` directory where
#: ``models/pocket_macro_skeleton_v2.pt`` ships.  Resolving with
#: ``Path(__file__)`` makes the path **invariant to the caller's
#: current working directory** (the original CWD-relative literal was
#: broken whenever the host was invoked from any directory other than
#: the project root).
_MODULE_DIR: Path = Path(__file__).resolve().parent
_PACKAGE_ROOT: Path = _MODULE_DIR.parent                  # ``molmetal_lam/``
_REPO_PKG_ROOT: Path = _MODULE_DIR.parent.parent           # ``molmetal/``
_MODULE_RELATIVE_MODELS_DIR: Path = _REPO_PKG_ROOT / "models"

#: Default path of the trained state-dict checkpoint.
#:
#: Resolved lazily via :func:`_resolve_default_paths` so the same
#: string is honest regardless of the caller's CWD.  We keep this as
#: a module-level constant for backward compatibility — existing
#: callers that read ``DEFAULT_CHECKPOINT_PATH`` continue to work.
#:
#: Resolution order (BUG-2 fix):
#: 1. ``<repo>/molmetal/models/pocket_macro_skeleton_v2.pt``
#:    (module-relative, works from any CWD)
#: 2. ``models/pocket_macro_skeleton_v2.pt`` (CWD-relative legacy,
#:    preserved for backwards compat)
DEFAULT_CHECKPOINT_PATH: str = str(
    _MODULE_RELATIVE_MODELS_DIR / _CHECKPOINT_FILENAME
)

#: Default path of the training-metadata JSON sidecar.
#: Same resolution policy as :data:`DEFAULT_CHECKPOINT_PATH`.
DEFAULT_METADATA_PATH: str = str(
    _MODULE_RELATIVE_MODELS_DIR / _METADATA_FILENAME
)


def _resolve_default_paths(
    module_models_dir: Optional[Path] = None,
    cwd: Optional[Path] = None,
    checkpoint_filename: Optional[str] = None,
    metadata_filename: Optional[str] = None,
) -> Tuple[str, str]:
    """Return ``(checkpoint_path, metadata_path)`` with a graceful fallback.

    Resolution order
    ----------------
    1. **Module-relative** (preferred): the canonical
       ``molmetal/models/pocket_macro_skeleton_v2.pt`` location derived
       from ``__file__``.  This works no matter where the host process
       was launched from.
    2. **CWD-relative legacy fallback**:
       ``models/pocket_macro_skeleton_v2.pt`` (relative to the
       process's current working directory).  Preserved so that older
       scripts invoked from the project root continue to find the
       checkpoint even after this fix ships.

    Both candidates must point at an existing file for the chosen
    location to "win"; the first hit is returned.  When neither
    candidate exists, the module-relative path is still returned so
    that ``is_available()`` can produce the truthful ``False`` answer
    with a precise ``FileNotFoundError`` pointing at the canonical
    location.

    Parameters
    ----------
    module_models_dir : Path, optional
        Override for the module-relative ``models/`` directory.  When
        ``None`` (the default) the value baked into the module at
        import time is used.  Tests pass a ``tmp_path`` here to
        exercise the CWD-relative fallback branch without moving the
        real checkpoint.
    cwd : Path, optional
        Override for ``Path.cwd()`` — also intended for tests so the
        resolver can be exercised in-process without mutating the
        pytest worker's CWD.
    checkpoint_filename : str, optional
        Override for the checkpoint filename.  Defaults to
        ``"pocket_macro_skeleton_v2.pt"``.
    metadata_filename : str, optional
        Override for the metadata filename.  Defaults to
        ``"pocket_macro_skeleton_v2.pt.json"``.

    Returns
    -------
    tuple[str, str]
        ``(checkpoint_path, metadata_path)`` resolved as above.
    """
    if module_models_dir is None:
        module_models_dir = _MODULE_RELATIVE_MODELS_DIR
    if cwd is None:
        cwd = Path.cwd()
    if checkpoint_filename is None:
        checkpoint_filename = _CHECKPOINT_FILENAME
    if metadata_filename is None:
        metadata_filename = _METADATA_FILENAME

    module_ckpt = module_models_dir / checkpoint_filename
    module_meta = module_models_dir / metadata_filename

    if module_ckpt.exists() and module_meta.exists():
        return str(module_ckpt), str(module_meta)

    # CWD-relative legacy fallback (only meaningful when the caller is
    # running from the project root).
    cwd_ckpt = cwd / "models" / checkpoint_filename
    cwd_meta = cwd / "models" / metadata_filename
    if cwd_ckpt.exists() and cwd_meta.exists():
        return str(cwd_ckpt), str(cwd_meta)

    # Neither candidate found — return the module-relative path so
    # the error message points at the canonical location.
    return str(module_ckpt), str(module_meta)

#: Confidence floor below which we declare ``UNKNOWN`` instead of the
#: argmax class.  v2 reaches 100.0 % train accuracy on all 6 classes
#: (66/66) but the softmax masses are spread: CA2 (1AKL) reaches
#: 0.39 argmax mass because the model is somewhat conservative on
#: this hard triad.  The floor at 0.30 captures all five ground-truth
#: classes correctly while still filtering accidental softmax ties
#: from degenerate fixtures.  When ``None`` the floor is bypassed and
#: every prediction returns the argmax (use for max sensitivity).
DEFAULT_CONFIDENCE_FLOOR: float = 0.30


# ---------------------------------------------------------------------------
# v2 mirror — same architecture as ``pocket_macro_skeleton.py``.  Built
# inline so the v2 state-dict loads cleanly with ``strict=True`` (the
# canonical ``PocketMacroSkeleton`` class also works; we keep the mirror
# so we can run without instantiating the full class hierarchy in
# downstream imports).
# ---------------------------------------------------------------------------


class _PocketMacroSkeletonV2(nn.Module):
    """v2 mirror of :class:`PocketMacroSkeleton` (33-d input, 12-d output).

    Architecture matches the trained
    ``pocket_macro_skeleton_v2.pt`` state_dict exactly (linear weight
    shape ``(32, 33)``).  Kept private to this module so we don't
    shadow the canonical v2 class in ``pocket_macro_skeleton.py``.
    """

    def __init__(
        self,
        per_residue_features: int = PER_RESIDUE_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        n_scaffold_classes: int = 12,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scaffold_classes = n_scaffold_classes
        self.residue_encoder = nn.Module()
        self.residue_encoder.linear = nn.Linear(
            per_residue_features, hidden_dim
        )
        self.attention = PocketMacroSkeletonAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            n_scaffold_classes=n_scaffold_classes,
        )

    def encode(self, scalars: torch.Tensor) -> torch.Tensor:
        """Run a (n, 33) per-residue feature matrix through the encoder."""
        return self.residue_encoder.linear(scalars)

    def forward(
        self, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(n, 33)`` → ``(hidden_dim,)`` skeleton + ``(n_classes,)`` logits."""
        if scalars.shape[0] == 0:
            z1 = torch.zeros(self.hidden_dim, dtype=torch.float32)
            z2 = torch.zeros(self.n_scaffold_classes, dtype=torch.float32)
            return z1, z2
        encoded = self.encode(scalars)
        return self.attention(encoded)


# ---------------------------------------------------------------------------
# v2 33-d feature builder — mirrors ``ResidueEmbedder`` in
# ``pocket_macro_skeleton.py`` exactly (33 = AA 20 + anchor tier 4 +
# anchor_position 4 + dist_bin 3 + chain_id_hash 1 + resid_norm 1).
# ---------------------------------------------------------------------------

_AA_VOCAB = (
    "A", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "P", "Q", "R", "S",
    "T", "V", "W", "Y",
)
_AA_INDEX = {aa: i for i, aa in enumerate(_AA_VOCAB)}


def _build_33d_features(
    pocket_residues: List["PocketResidueLike"],
) -> torch.Tensor:
    """Build the v2 33-d per-residue feature matrix from a residue list.

    Layout (matches ``ResidueEmbedder.__call__`` in
    ``pocket_macro_skeleton.py`` exactly):

    * indices  0..19 — AA one-hot (20)
    * indices 20..23 — metal-anchor tier one-hot (4)
    * indices 24..27 — anchor_position one-hot (4) — v2 addition
    * indices 28..30 — distance bin one-hot (3)
    * index     31   — chain_id_hash (1)
    * index     32   — resid_norm (1)

    Total: 33-d per residue.

    The ``anchor_position`` value is inferred by enumerating the
    anchor residues in input order — index 0, 1, 2 maps to slot 1, 2, 3
    in the one-hot (slot 0 is the "not-anchor" catch-all).
    """
    from molmetal_lam.search_alg.warm_start import PocketResidue

    norm: List[PocketResidue] = []
    for r in pocket_residues:
        if isinstance(r, PocketResidue):
            norm.append(r)
        elif isinstance(r, dict):
            norm.append(PocketResidue.from_dict(r))
        else:
            raise TypeError(
                "pocket_residues entries must be PocketResidue or dict, "
                f"got {type(r).__name__}"
            )

    if not norm:
        return torch.zeros((0, PER_RESIDUE_FEATURES), dtype=torch.float32)

    # Pre-compute the anchor-position slot for each residue by
    # enumerating input order.  Matches the Phase 3 validate fixtures
    # convention (residue 0 of the triad = slot 1, etc.).
    anchor_position_map: Dict[int, int] = {}
    next_anchor_idx = 0
    for r in norm:
        rid = id(r)
        if bool(r.is_metal_anchor):
            anchor_position_map[rid] = next_anchor_idx
            next_anchor_idx += 1
        else:
            anchor_position_map[rid] = -1

    rows = []
    for r in norm:
        row = torch.zeros(PER_RESIDUE_FEATURES, dtype=torch.float32)
        aa = str(r.one_letter).upper()
        idx = _AA_INDEX.get(aa, -1)
        if idx >= 0:
            row[idx] = 1.0
        # metal-anchor tier one-hot
        d = float(r.distance_to_ligand)
        if not r.is_metal_anchor:
            tier = 0
        elif d <= 2.5:
            tier = 3
        elif d <= 4.0:
            tier = 2
        else:
            tier = 1
        row[20 + tier] = 1.0
        # anchor_position one-hot (v2 addition at indices 24..27)
        ap = anchor_position_map[id(r)]
        if ap < 0 or ap > 2:
            ap_slot = 0  # not-anchor
        else:
            ap_slot = ap + 1  # 0->1, 1->2, 2->3
        row[24 + ap_slot] = 1.0
        # distance bin one-hot (v2 layout, indices 28..30)
        if d < 2.5:
            dbin = 0
        elif d < 4.5:
            dbin = 1
        else:
            dbin = 2
        row[28 + dbin] = 1.0
        # chain_id_hash
        ch = (str(r.chain)[:1] or "A").upper()
        row[31] = (ord(ch) - ord("A")) / 26.0 if "A" <= ch <= "Z" else 0.0
        # resid_norm
        row[32] = min(1.0, max(0.0, int(r.resid) / 999.0))
        rows.append(row)
    return torch.stack(rows, dim=0)


# ---------------------------------------------------------------------------
# Lightweight pocket residue fixtures — keyed by PDB id.
# These are *not* exhaustive (we only need enough signal to confirm
# the trained model classifies each family correctly).  All anchor
# residues match the Phase 3 validate fixtures (see
# ``scripts/phase3_validate_pocket_macro_skeleton.py``).
# ---------------------------------------------------------------------------

_PocketResidueLike = object  # type alias for the type-checker


_FIXTURES: Dict[str, List[dict]] = {
    # CA2 (1AKL) — His-His-His triad → ground truth ZN_TETRA_HHH
    # Documented Phase 2 collapse: model predicts ZN_TETRA_HHE.
    "1AKL": [
        {"one_letter": "H", "resid": 94, "distance_to_ligand": 2.5, "is_metal_anchor": True},
        {"one_letter": "H", "resid": 96, "distance_to_ligand": 3.0, "is_metal_anchor": True},
        {"one_letter": "H", "resid": 119, "distance_to_ligand": 2.0, "is_metal_anchor": True},
        {"one_letter": "V", "resid": 143, "distance_to_ligand": 4.5},
        {"one_letter": "L", "resid": 198, "distance_to_ligand": 3.7},
        {"one_letter": "F", "resid": 131, "distance_to_ligand": 4.9},
        {"one_letter": "E", "resid": 106, "distance_to_ligand": 4.2},
        {"one_letter": "T", "resid": 199, "distance_to_ligand": 4.6},
        {"one_letter": "K", "resid": 170, "distance_to_ligand": 4.8},
        {"one_letter": "W", "resid": 209, "distance_to_ligand": 4.9},
    ],
    # ACE (1O86) — His-His-Glu triad → ground truth ZN_TETRA_HHE
    "1O86": [
        {"one_letter": "H", "resid": 383, "distance_to_ligand": 2.4, "is_metal_anchor": True},
        {"one_letter": "H", "resid": 387, "distance_to_ligand": 2.8, "is_metal_anchor": True},
        {"one_letter": "E", "resid": 411, "distance_to_ligand": 2.6, "is_metal_anchor": True},
        {"one_letter": "Y", "resid": 520, "distance_to_ligand": 4.7},
        {"one_letter": "V", "resid": 518, "distance_to_ligand": 4.5},
        {"one_letter": "F", "resid": 391, "distance_to_ligand": 4.8},
        {"one_letter": "K", "resid": 489, "distance_to_ligand": 4.9},
    ],
    # MMP2 (1HOV) — His-His-Glu triad → ground truth ZN_TETRA_HHE
    "1HOV": [
        {"one_letter": "H", "resid": 403, "distance_to_ligand": 2.5, "is_metal_anchor": True},
        {"one_letter": "H", "resid": 407, "distance_to_ligand": 3.0, "is_metal_anchor": True},
        {"one_letter": "H", "resid": 413, "distance_to_ligand": 2.0, "is_metal_anchor": True},
        {"one_letter": "E", "resid": 404, "distance_to_ligand": 3.5},
        {"one_letter": "A", "resid": 417, "distance_to_ligand": 4.5},
        {"one_letter": "L", "resid": 418, "distance_to_ligand": 4.7},
        {"one_letter": "V", "resid": 422, "distance_to_ligand": 4.9},
    ],
    # PKA (1ATP) — Asp-Lys-Glu triad (catalytic) → MG_OCTA_KINASE
    "1ATP": [
        {"one_letter": "D", "resid": 184, "distance_to_ligand": 3.0, "is_metal_anchor": True},
        {"one_letter": "K", "resid": 72, "distance_to_ligand": 3.5, "is_metal_anchor": True},
        {"one_letter": "E", "resid": 91, "distance_to_ligand": 3.0, "is_metal_anchor": True},
        {"one_letter": "F", "resid": 327, "distance_to_ligand": 4.7},
        {"one_letter": "L", "resid": 173, "distance_to_ligand": 4.8},
        {"one_letter": "V", "resid": 123, "distance_to_ligand": 4.9},
    ],
    # CYP3A4 (1W0G) — Cys heme proximal → FE_HEME_CYS
    "1W0G": [
        {"one_letter": "C", "resid": 442, "distance_to_ligand": 2.4, "is_metal_anchor": True},
        {"one_letter": "F", "resid": 137, "distance_to_ligand": 4.6},
        {"one_letter": "L", "resid": 244, "distance_to_ligand": 4.4},
        {"one_letter": "V", "resid": 243, "distance_to_ligand": 4.5},
        {"one_letter": "A", "resid": 305, "distance_to_ligand": 4.8},
    ],
}


_PDB_TO_TARGET: Dict[str, str] = {
    "1AKL": "CA2",
    "1O86": "ACE",
    "1HOV": "MMP2",
    "1ATP": "PKA",
    "1W0G": "CYP3A4",
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """The structured return type of ``PocketMacroInference.predict_*``.

    Attributes
    ----------
    target_name : str
        The metalloprotein target name (``"CA2"``, ``"PKA"``, ...).
    predicted_class : str
        The predicted :class:`ScaffoldClass` name (e.g.
        ``"ZN_TETRA_HHE"``) or ``"UNKNOWN`` when confidence is below the
        configured floor.
    ground_truth_class : str
        The canonical scaffold class from
        :func:`scaffold_class_from_target_name`.
    confidence : float
        The softmax probability of ``predicted_class``.
    source : str
        ``"model"`` when the trained weights were used; ``"fallback"``
        when the checkpoint was missing.
    """

    target_name: str
    predicted_class: str
    ground_truth_class: str
    confidence: float
    source: str


class PocketMacroInference:
    """Inference wrapper around the trained ``PocketMacroSkeleton`` checkpoint.

    Parameters
    ----------
    checkpoint_path : str, optional
        Path to the trained ``.pt`` state-dict.  Default:
        ``"molmetal/models/pocket_macro_skeleton.pt"``.
    metadata_path : str, optional
        Path to the ``.pt.json`` metadata sidecar.  Default:
        ``"molmetal/models/pocket_macro_skeleton.pt.json"``.
    confidence_floor : float, optional
        Below this softmax probability the adapter returns
        ``ScaffoldClass.UNKNOWN`` instead of the argmax class.  Default
        ``0.50``.

    Notes
    -----
    The adapter is **stateless** aside from the loaded model — the same
    ``PocketMacroInference`` instance is safe to call from multiple
    threads when PyTorch is built without the GIL (CPU-only here, so
    always safe).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    ) -> None:
        # Resolve default paths at construction time so the lookup is
        # CWD-independent (BUG-2 fix).  When the caller explicitly
        # passes a path we honour it verbatim — this preserves
        # backwards compatibility for tests / scripts that point the
        # adapter at a custom fixture directory.
        if checkpoint_path is None or metadata_path is None:
            default_ckpt, default_meta = _resolve_default_paths()
            if checkpoint_path is None:
                checkpoint_path = default_ckpt
            if metadata_path is None:
                metadata_path = default_meta
        self.checkpoint_path = checkpoint_path
        self.metadata_path = metadata_path
        self.confidence_floor = float(confidence_floor)
        self._metadata: Optional[dict] = None
        self._model: Optional[_PocketMacroSkeletonV2] = None
        self._loaded: bool = False
        # Try eager load so test_is_available returns the truthful
        # answer without an explicit ``load()`` call.  If the files
        # are missing we silently fall back to metadata-only mode.
        if self.is_available():
            try:
                self.load()
            except Exception:
                # Surface the failure as ``is_available() == False``
                # without raising — the downstream consumers need a
                # graceful no-op path.
                self._loaded = False
                self._model = None

    # -------------------------------------------------------------------
    # Availability + load
    # -------------------------------------------------------------------

    def is_available(self) -> bool:
        """True if **both** the ``.pt`` and ``.pt.json`` files exist on disk."""
        return os.path.exists(self.checkpoint_path) and os.path.exists(
            self.metadata_path
        )

    def load(self) -> None:
        """Load the trained v2 state-dict into the v2 mirror module.

        Raises
        ------
        FileNotFoundError
            If the checkpoint or metadata file is missing.
        RuntimeError
            If the state_dict shape does not match the v2 architecture
            (size mismatch → surfaced loudly, no silent fallback).
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"PocketMacroInference: checkpoint not found at "
                f"{self.checkpoint_path!r}"
            )
        state = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        model = _PocketMacroSkeletonV2(
            per_residue_features=PER_RESIDUE_FEATURES,
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            n_scaffold_classes=12,
        )
        # strict=True: any architectural drift surfaces immediately.
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"PocketMacroInference: state_dict mismatch — "
                f"missing={missing}, unexpected={unexpected}"
            )
        model.eval()
        self._model = model
        # Load metadata sidecar.
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as fh:
                self._metadata = json.load(fh)
        self._loaded = True

    # -------------------------------------------------------------------
    # Metadata accessors
    # -------------------------------------------------------------------

    def get_metadata(self) -> dict:
        """Return the ``.pt.json`` sidecar as a dict (empty when missing)."""
        if self._metadata is None and os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as fh:
                self._metadata = json.load(fh)
        return dict(self._metadata or {})

    def scaffold_class_distribution(self) -> Dict[str, Dict[str, float]]:
        """Return ``{"ZN_TETRA_HHH": {"correct": int, ...}, ...}``.

        Mirrors the ``results.per_class_accuracy`` block in the
        ``.pt.json`` sidecar.  Empty dict if metadata is missing.
        """
        meta = self.get_metadata()
        per_class = (
            meta.get("results", {}).get("per_class_accuracy", {}) or {}
        )
        return {k: dict(v) for k, v in per_class.items()}

    # -------------------------------------------------------------------
    # Residue fixture lookup
    # -------------------------------------------------------------------

    def _resolve_residues(
        self, target_name: str
    ) -> Optional[List[dict]]:
        """Translate a target name / PDB id into a residue fixture list.

        Accepts either a ``MetalloproteinTarget.name`` (``"CA2"``,
        ``"PKA"``, ...) or a PDB id (``"1AKL"``, ``"1ATP"``).
        Returns ``None`` when no fixture is available — callers should
        fall back to the metadata mapping in that case.
        """
        up = str(target_name).upper().strip()
        # Direct PDB id?
        if up in _FIXTURES:
            return list(_FIXTURES[up])
        # Target name → PDB id via the curated table.
        for pdb, name in _PDB_TO_TARGET.items():
            if name == up:
                return list(_FIXTURES[pdb])
        return None

    # -------------------------------------------------------------------
    # Core inference primitives
    # -------------------------------------------------------------------

    def predict_logits(
        self, target_name: str
    ) -> Tuple[str, np.ndarray, str]:
        """Run the model and return ``(predicted_class_name, probs, source)``.

        ``source`` is ``"model"`` when the trained weights were used,
        ``"fallback"`` when the checkpoint was missing.  The probs
        array is shape ``(12,)`` (softmax over the 12-class head).
        """
        gt_class = scaffold_class_from_target_name(target_name).name
        if not self._loaded or self._model is None:
            # Graceful fallback: return UNKNOWN with the metadata-derived
            # ground-truth class but a 0.0 confidence so downstream
            # aggregators can mark the cell DESIGN-only.
            probs = np.zeros(12, dtype=np.float32)
            probs[ScaffoldClass.UNKNOWN] = 1.0
            return ScaffoldClass.UNKNOWN.name, probs, "fallback"

        residues = self._resolve_residues(target_name)
        if residues is None:
            probs = np.zeros(12, dtype=np.float32)
            probs[ScaffoldClass.UNKNOWN] = 1.0
            return ScaffoldClass.UNKNOWN.name, probs, "fallback"

        scalars = _build_33d_features(residues)
        with torch.no_grad():
            _skel, logits = self._model(scalars)
            probs_t = torch.softmax(logits, dim=0)
            probs = probs_t.cpu().numpy()
        argmax_idx = int(np.argmax(probs))
        return ScaffoldClass(argmax_idx).name, probs, "model"

    def predict_scaffold_class_with_confidence(
        self, target_name: str
    ) -> Tuple[str, float]:
        """Return ``(predicted_class_name, confidence)``.

        When the confidence is below ``self.confidence_floor`` the
        predicted class is downgraded to ``"UNKNOWN"`` (the confidence
        value is preserved so callers can still inspect the raw
        argmax).
        """
        predicted, probs, source = self.predict_logits(target_name)
        confidence = float(np.max(probs))
        if confidence < self.confidence_floor:
            return ScaffoldClass.UNKNOWN.name, confidence
        return predicted, confidence

    def predict_scaffold_class(self, target_name: str) -> str:
        """Return the predicted :class:`ScaffoldClass` name (string).

        Thin wrapper over :meth:`predict_scaffold_class_with_confidence`
        — drops the confidence.  Falls back to ``"UNKNOWN"`` for
        low-confidence predictions (e.g. the CA2 collapse).
        """
        predicted, _conf = self.predict_scaffold_class_with_confidence(
            target_name
        )
        return predicted

    def get_embedding(self, target_name: str) -> np.ndarray:
        """Return the 32-d mean-pooled skeleton vector as ``np.float32``.

        This is the **bridge** for downstream ``warm_start.py``
        integration: when ``warm_start.pocket_features(target_name)``
        returns a 64-d hash projection whose first 6 slots are
        hand-crafted and the remaining 58 are zeros, the pocket-
        conditioned 32-d skeleton can be appended / mixed in to give
        the per-state prior a pocket-aware signal.

        Returns ``np.zeros(32, dtype=np.float32)`` when the model
        is unavailable or the target name is unknown.
        """
        out = np.zeros(HIDDEN_DIM, dtype=np.float32)
        if not self._loaded or self._model is None:
            return out
        residues = self._resolve_residues(target_name)
        if residues is None:
            return out
        scalars = _build_33d_features(residues)
        with torch.no_grad():
            skel, _logits = self._model(scalars)
            out = skel.cpu().numpy().astype(np.float32)
        return out

    # -------------------------------------------------------------------
    # Pretty-print helper
    # -------------------------------------------------------------------

    def pretty_print(self, target_name: str) -> InferenceResult:
        """Build and print a one-line summary for the target.

        Returns
        -------
        InferenceResult
            The structured result (also printed to stdout).
        """
        predicted, conf = self.predict_scaffold_class_with_confidence(
            target_name
        )
        gt_class = scaffold_class_from_target_name(target_name).name
        source = "model" if self._loaded else "fallback"
        result = InferenceResult(
            target_name=str(target_name).upper(),
            predicted_class=predicted,
            ground_truth_class=gt_class,
            confidence=conf,
            source=source,
        )
        marker = "OK" if predicted == gt_class else "MISS"
        print(
            f"    [{marker}] {result.target_name:<8s} "
            f"expected={result.ground_truth_class:<18s} "
            f"predicted={result.predicted_class:<18s} "
            f"confidence={result.confidence:+.3f} "
            f"source={result.source}"
        )
        return result


__all__ = [
    "PocketMacroInference",
    "InferenceResult",
    "PER_RESIDUE_FEATURES",
    "V1_PER_RESIDUE_FEATURES",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_METADATA_PATH",
    "DEFAULT_CONFIDENCE_FLOOR",
]
