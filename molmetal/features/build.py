"""One construction site for an MSRS design matrix (FIX-P1).

``build(frame, spec)`` is the only function that turns a cohort frame into a
feature matrix.  It composes blocks in a fixed order — ligand, metal, cell,
counterion, component count, descriptor table — and returns the slices, the
column names, and the spec hash that identifies the result.

The four historical builders are reproduced bit-for-bit by the presets in
:mod:`molmetal.features.spec`; ``tests/test_features_build.py`` asserts equality
against ``msrs_code_levers.base_blocks``, ``msrs_surface_model.build_design``
and the 172-d table.

Why a hash and not a script name: the audit found the same artifact label
("ligand block") attached to three different matrices, so "built by
msrs_surface_model.py" said nothing about *what* was built.  The hash is the
content address of the description, not of the data — it changes when the
design changes and stays put when the code around it is refactored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from molmetal.features.assay import counterion_block, n_fragments_block
from molmetal.features.cell import cell_block, cell_vocabulary
from molmetal.features.ligand import complex_ligand_blocks
from molmetal.features.metal import metal_block
from molmetal.features.spec import FeatureSpec

Block = Tuple[int, int]


@dataclass
class FeatureBundle:
    """A built matrix plus everything needed to describe it."""

    X: sparse.csr_matrix
    names: List[str]
    slices: Dict[str, Block]
    spec: FeatureSpec
    spec_hash: str
    n_rows: int

    def block(self, name: str) -> sparse.csr_matrix:
        start, stop = self.slices[name]
        return self.X[:, start:stop]

    def summary(self) -> Dict[str, object]:
        return {
            "spec_hash": self.spec_hash,
            "n_rows": self.n_rows,
            "n_features": int(self.X.shape[1]),
            "blocks": {k: [int(v[0]), int(v[1])] for k, v in self.slices.items()},
        }


def _ligand_block(
    frame: pd.DataFrame, spec: FeatureSpec
) -> Tuple[sparse.csr_matrix, List[str], Dict[str, Block]]:
    """ECFP4 of the declared ligand object.

    ``ligand_main_fragment`` reads the stored column verbatim so the legacy
    preset reproduces the historical matrix exactly; ``ligand_canonical`` goes
    through :func:`complex_ligand_blocks`, which recomputes the main component
    from the complex so the sub-blocks cannot drift from the cohort definition.
    """
    if spec.ligand.source == "ligand_main_fragment":
        from molmetal.scripts.msrs_surface_model import morgan_matrix

        smi = frame["ligand_main_fragment"].tolist()
        X = morgan_matrix(smi, n_bits=spec.ligand.n_bits, radius=spec.ligand.radius)
        names = [f"ecfp4_{i}" for i in range(X.shape[1])]
        return X, names, {"ligand": (0, X.shape[1])}

    return complex_ligand_blocks(
        frame,
        n_bits=spec.ligand.n_bits,
        radius=spec.ligand.radius,
        counterion_vocabulary=spec.assay.counterion_vocabulary,
        include=spec.ligand.include,
    )


def _descriptor_block(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    lookup: Optional[Mapping[str, np.ndarray]],
) -> Tuple[np.ndarray, List[str]]:
    from molmetal.features.descriptors import FEATURE_DIM, column_names, descriptor_table

    if lookup is None:
        values = frame[spec.ligand.source].astype(str).tolist()
        uniq = list(dict.fromkeys(values))
        table, _masks = descriptor_table(uniq)
        lookup = {s: table[i] for i, s in enumerate(uniq)}
    dense = np.full((len(frame), FEATURE_DIM), np.nan, dtype=np.float32)
    for i, smi in enumerate(frame[spec.ligand.source].astype(str)):
        vec = lookup.get(smi)
        if vec is not None:
            dense[i] = vec
    return dense, column_names()


def build(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    *,
    metal_descriptors: Optional[pd.DataFrame] = None,
    cell_vocabulary_override: Optional[Sequence[str]] = None,
    descriptor_lookup: Optional[Mapping[str, np.ndarray]] = None,
) -> FeatureBundle:
    """Compose the matrix described by ``spec``.

    ``cell_vocabulary_override`` exists for the legacy presets: the historical
    scripts derived the top-40 vocabulary from the *full* cohort and then built
    blocks for a subset.  Deriving it here from ``frame`` alone would silently
    change the columns whenever the frame is a slice.
    """
    parts: List[sparse.csr_matrix] = []
    names: List[str] = []
    slices: Dict[str, Block] = {}

    def add(key: str, block: sparse.csr_matrix, block_names: List[str]) -> None:
        if block.shape[1] != len(block_names):
            raise ValueError(
                f"{key}: {block.shape[1]} columns but {len(block_names)} names"
            )
        start = sum(p.shape[1] for p in parts)
        parts.append(sparse.csr_matrix(block))
        names.extend(block_names)
        slices[key] = (start, start + block.shape[1])

    lig, lig_names, lig_slices = _ligand_block(frame, spec)
    add("ligand", lig, lig_names)
    for key, (a, b) in lig_slices.items():
        if key != "ligand":
            slices[key] = (a, b)

    if spec.metal.one_hot or spec.metal.columns:
        if metal_descriptors is None:
            from molmetal.features.metal import load_descriptors

            metal_descriptors = load_descriptors(spec.metal.descriptors_csv)
        dense, met_names = metal_block(
            frame["metal"].astype(str).tolist(),
            metal_descriptors,
            columns=spec.metal.columns,
            one_hot=spec.metal.one_hot,
        )
        add("metal", sparse.csr_matrix(dense), met_names)

    if spec.cell.mode != "none":
        cells = frame["cell_line_key"].astype(str).tolist()
        if cell_vocabulary_override is not None:
            vocab = list(cell_vocabulary_override)
        else:
            vocab = cell_vocabulary(
                cells,
                top_n=spec.cell.top_n,
                mode="full" if spec.cell.mode == "full" else "top40",
            )
        block, cell_names = cell_block(cells, vocab)
        add("cell", block, cell_names)

    if spec.assay.include_counterion:
        raw = (
            frame["counterion"].tolist()
            if "counterion" in frame.columns
            else [""] * len(frame)
        )
        block, c_names = counterion_block(raw, spec.assay.counterion_vocabulary)
        add("counterion", block, c_names)

    if spec.assay.include_n_fragments:
        block, f_names = n_fragments_block(frame[spec.ligand.source].astype(str))
        add("n_fragments", block, f_names)

    if spec.ligand.descriptor_table != "none":
        dense, d_names = _descriptor_block(frame, spec, descriptor_lookup)
        add("descriptors", sparse.csr_matrix(dense), d_names)

    if not parts:
        raise ValueError("build: spec selected no blocks")

    X = sparse.hstack(parts, format="csr")
    if X.shape[1] != len(names):
        raise ValueError(f"build: {X.shape[1]} columns but {len(names)} names")

    return FeatureBundle(
        X=X,
        names=names,
        slices=slices,
        spec=spec,
        spec_hash=spec.hash(),
        n_rows=len(frame),
    )


def verify_bundle(bundle: FeatureBundle) -> None:
    """Cheap structural check: slices tile the matrix and the hash matches."""
    if bundle.spec_hash != bundle.spec.hash():
        raise ValueError("bundle.spec_hash does not match bundle.spec")
    covered = np.zeros(bundle.X.shape[1], dtype=bool)
    for key, (start, stop) in bundle.slices.items():
        if stop <= start:
            raise ValueError(f"slice {key!r} is empty")
        if covered[start:stop].any():
            raise ValueError(f"slice {key!r} overlaps another block")
        covered[start:stop] = True
    if not covered.all():
        raise ValueError("slices do not cover every column")


# ---------------------------------------------------------------------------
# FIX-P3 — artifact versioning
# ---------------------------------------------------------------------------
#: Bumped when the sidecar layout changes, not when the data changes.
ARTIFACT_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _meta_checksum(meta: Mapping[str, Any]) -> str:
    """Checksum over every field except itself, so an edited field is visible."""
    body = {k: v for k, v in meta.items() if k != "meta_checksum"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # pragma: no cover - git is optional at runtime
        return "unknown"


def write_artifact(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    artifact: str,
    bundle: Optional[FeatureBundle] = None,
    input_paths: Sequence[Path] = (),
    missing_value_policy: str = "nan",
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write ``<path>.npz`` plus a ``<path>.meta.json`` sidecar.

    The sidecar records the spec hash, the SHA-256 of every input file, the git
    commit, the missing-value policy and the hash of the npz itself.  A loader
    that calls :func:`verify_artifact` can then tell a stale artifact from a
    current one, and can tell a hand-edited one from either.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dict(arrays))

    meta: Dict[str, Any] = {
        "artifact": artifact,
        "artifact_version": ARTIFACT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "npz_sha256": _sha256_file(path),
        "npz_bytes": path.stat().st_size,
        "arrays": {
            k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)}
            for k, v in arrays.items()
        },
        "missing_value_policy": missing_value_policy,
        "inputs": [{"path": str(p), "sha256": _sha256_file(Path(p))} for p in input_paths],
        "spec_hash": bundle.spec_hash if bundle is not None else None,
        "spec": bundle.spec.to_dict() if bundle is not None else None,
        "n_rows": bundle.n_rows if bundle is not None else None,
        "n_features": int(bundle.X.shape[1]) if bundle is not None else None,
        "blocks": (
            {k: [int(v[0]), int(v[1])] for k, v in bundle.slices.items()}
            if bundle is not None
            else None
        ),
    }
    if extra:
        meta.update(dict(extra))
    meta["meta_checksum"] = _meta_checksum(meta)

    sidecar = path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return sidecar


def verify_artifact(
    path: Path, *, expected_spec: Optional[FeatureSpec] = None
) -> Dict[str, Any]:
    """Validate a sidecar and the npz it describes; return the metadata.

    Raises ``ValueError`` on: missing sidecar, edited metadata, modified npz,
    or a spec hash that does not match ``expected_spec``.
    """
    path = Path(path)
    sidecar = path.with_suffix(".meta.json")
    if not path.is_file():
        raise ValueError(f"artifact missing: {path}")
    if not sidecar.is_file():
        raise ValueError(f"artifact sidecar missing: {sidecar}")

    meta = json.loads(sidecar.read_text())
    recorded = meta.get("meta_checksum")
    if recorded != _meta_checksum(meta):
        raise ValueError(f"{sidecar}: metadata checksum mismatch (edited sidecar)")
    actual = _sha256_file(path)
    if actual != meta.get("npz_sha256"):
        raise ValueError(
            f"{path}: npz hash mismatch (expected {meta.get('npz_sha256')}, got {actual})"
        )
    if expected_spec is not None and meta.get("spec_hash") != expected_spec.hash():
        raise ValueError(
            f"{path}: spec hash {meta.get('spec_hash')} does not match "
            f"{expected_spec.hash()}"
        )
    return meta
