"""Declarative feature specification (FIX-P1).

Before this module there were four feature builders in the MSRS scripts and no
way to say which one produced a matrix.  The audit counted the damage: the same
"ligand block" meant ECFP4 of the main fragment in ``msrs_code_levers`` and in
``msrs_surface_model``, a 172-d descriptor vector in ``msrs_172d_rebuild``, and
ECFP4 of the full complex in ``msrs_complex_features`` — three different objects
under one name, and only the last one matched the cohort.

A :class:`FeatureSpec` is the single description of a matrix: which ligand
column, how wide the fingerprint, which metal descriptors, how the cell line is
encoded, whether the assay-level blocks are included.  Its hash is a content
address, so an artifact can say exactly which matrix it scored instead of
naming a script that has since changed.

Presets reproduce the historical builders bit-for-bit; see
``tests/test_features_build.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Tuple

#: Fingerprint width used by every historical MSRS arm.
DEFAULT_N_BITS = 2048
DEFAULT_RADIUS = 2
DEFAULT_TOP_CELL_LINES = 40

#: Metal descriptors present in ``molmetal/data/metal_chemistry_5.csv``.
DEFAULT_METAL_COLUMNS: Tuple[str, ...] = ("ie1_kj", "ie2_kj", "pauling_en")


@dataclass(frozen=True)
class LigandSpec:
    """Which ligand object is featurized, and how."""

    #: ``ligand_canonical`` (full complex) or ``ligand_main_fragment`` (legacy).
    source: str = "ligand_canonical"
    n_bits: int = DEFAULT_N_BITS
    radius: int = DEFAULT_RADIUS
    #: Sub-blocks of :func:`molmetal.features.ligand.complex_ligand_blocks`.
    include: Tuple[str, ...] = ("canonical",)
    #: Optional 172-d descriptor table, keyed by the same SMILES as ``source``.
    descriptor_table: str = "none"  # "none" | "rdkit_172d"

    def __post_init__(self) -> None:
        if self.source not in ("ligand_canonical", "ligand_main_fragment"):
            raise ValueError(f"LigandSpec.source: unknown source {self.source!r}")
        if self.n_bits <= 0 or self.radius < 0:
            raise ValueError("LigandSpec: n_bits must be > 0 and radius >= 0")
        if self.descriptor_table not in ("none", "rdkit_172d"):
            raise ValueError(
                f"LigandSpec.descriptor_table: unknown {self.descriptor_table!r}"
            )


@dataclass(frozen=True)
class MetalSpec:
    """Metal one-hot plus tabulated periodic-table descriptors."""

    one_hot: bool = True
    columns: Tuple[str, ...] = DEFAULT_METAL_COLUMNS
    #: Path is recorded, not resolved, so the hash is machine-independent.
    descriptors_csv: str = "molmetal/data/metal_chemistry_5.csv"


@dataclass(frozen=True)
class CellSpec:
    """Cell-line encoding.

    ``top40`` is the historical one-hot with an ``other`` bucket; ``full``
    leaves every level to LightGBM's native categorical handler (AUD-F5).
    """

    mode: str = "top40"  # "top40" | "full" | "none"
    top_n: int = DEFAULT_TOP_CELL_LINES

    def __post_init__(self) -> None:
        if self.mode not in ("top40", "full", "none"):
            raise ValueError(f"CellSpec.mode: unknown mode {self.mode!r}")
        if self.mode == "top40" and self.top_n <= 0:
            raise ValueError("CellSpec.top_n must be > 0 in top40 mode")


@dataclass(frozen=True)
class AssaySpec:
    """Assay-level blocks that are not chemistry."""

    include_counterion: bool = False
    counterion_vocabulary: Tuple[str, ...] = ()
    include_n_fragments: bool = False


@dataclass(frozen=True)
class FeatureSpec:
    """One matrix, fully described."""

    ligand: LigandSpec = field(default_factory=LigandSpec)
    metal: MetalSpec = field(default_factory=MetalSpec)
    cell: CellSpec = field(default_factory=CellSpec)
    assay: AssaySpec = field(default_factory=AssaySpec)
    version: int = 1

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def hash(self) -> str:
        """Content address, stable across machines and dict ordering."""
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    # -- derived views -----------------------------------------------------
    @property
    def block_order(self) -> Tuple[str, ...]:
        blocks = []
        if self.ligand.include:
            blocks.append("ligand")
        if self.metal.one_hot or self.metal.columns:
            blocks.append("metal")
        if self.cell.mode != "none":
            blocks.append("cell")
        if self.assay.include_counterion:
            blocks.append("counterion")
        if self.assay.include_n_fragments:
            blocks.append("n_fragments")
        return tuple(blocks)

    def replace(self, **kwargs: Any) -> "FeatureSpec":
        return replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Presets — each reproduces a historical builder bit-for-bit.
# ---------------------------------------------------------------------------
#: ``msrs_code_levers.base_blocks`` / ``msrs_surface_model.build_design``:
#: ECFP4 of the main fragment + metal one-hot/descriptors + top-40 cell one-hot.
LEGACY_MAIN = FeatureSpec(
    ligand=LigandSpec(
        source="ligand_main_fragment", include=("main",), descriptor_table="none"
    ),
    cell=CellSpec(mode="top40"),
)

#: FIX-L1 default: ECFP4 of the full complex, everything else unchanged.
COMPLEX = FeatureSpec(
    ligand=LigandSpec(source="ligand_canonical", include=("canonical",)),
    cell=CellSpec(mode="top40"),
)

#: Full complex plus the assay-level blocks (counterion, component count).
COMPLEX_FULL = FeatureSpec(
    ligand=LigandSpec(source="ligand_canonical", include=("canonical",)),
    cell=CellSpec(mode="top40"),
    assay=AssaySpec(include_counterion=True, include_n_fragments=True),
)

PRESETS: Dict[str, FeatureSpec] = {
    "legacy_main": LEGACY_MAIN,
    "complex": COMPLEX,
    "complex_full": COMPLEX_FULL,
}


def get_preset(name: str) -> FeatureSpec:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown FeatureSpec preset {name!r}; have {sorted(PRESETS)}")
