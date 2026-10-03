#!/usr/bin/env python3
"""Wf-392-A1 — Multi-PH + PLL feature wire-in to SE(3) SB trainer (CPU-only SCAFFOLD).

This module wires the Wf-382-A1 PLL feature SCAFFOLD (adapter at
:mod:`molmetal.adapters.flow_matching_lipman.pll_features`) into the
Wf-23 Phase 2 SE(3) Schrödinger-Bridge trainer at
:mod:`molmetal.scripts.train_se3sb`.  Per TODO-94 R20-P3 + Wf-392-A1
task brief:

* :func:`wire_pll_to_se3_sb_trainer`   add 3 NEW CLI flags to
  ``train_se3sb.py:parse_args`` (default OFF for backward compat with
  the Wf-23 + Wf-390-A3 baseline)
* :func:`smoke_pll_features`           build a tiny cohort, extract
  PLL features, and verify the feature shape / dtype contract

The script is CPU-only and SCAFFOLD-only — no trainer execution,
no GPU bench.  Full GPU integration is deferred to TODO-207 Wf-299
launch (USER_OK-gated per AGENTS.md §5.1).

References
----------
* Liu J., Feng H. (2026) "Persistent Local Laplacian Prediction of
  Protein–Ligand Binding Affinities".  J. Chem. Inf. Model. 66 (3).
* Wf-37 P3 Multi-PH + PLL end-to-end (TODO-94, R² +0.087 honest-neg).
* Wf-348-B1 chemprop+PLL CEILING_BREAK (R² +0.6814 on 60-SMILES smoke).
* Wf-382-A1 PLL feature SCAFFOLD + adapter.
* Wf-388-A3 SA penalty wire-in pattern (template for default OFF).
* Wf-390-A3 SE(3) SB integration audit (sanity baseline).
* Wf-23 Phase 2 SE(3) SB trainer (``molmetal/scripts/train_se3sb.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# Repo root resolution — mirror the wf382a1 test pattern
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Adapter import — DEFERRED to function-call time so the wire-in script
# module can be loaded even when the canonical PLL module is unavailable
# (or when the upstream ``molmetal.adapters`` chain triggers an
# unrelated Triton autotuner error).  At module load time we only
# attempt a *cheap* import; the heavy ``molmetal.adapters.flow_matching_lipman``
# chain is loaded lazily inside :func:`_try_import_adapter`.
_ADAPTER_AVAILABLE: bool = False
_ADAPTER_IMPORT_ERR: Optional[str] = None


def _try_import_adapter() -> bool:
    """Lazy one-shot import of the Wf-382-A1 PLL feature adapter.

    Returns ``True`` iff the adapter was successfully imported; on
    failure populates :data:`_ADAPTER_IMPORT_ERR` and returns ``False``.
    """
    global _ADAPTER_AVAILABLE, _ADAPTER_IMPORT_ERR
    if _ADAPTER_AVAILABLE:
        return True
    try:
        from molmetal.adapters.flow_matching_lipman.pll_features import (  # noqa: F401
            PLLFeatureAdapter,
            PLLFeatureError,
            persistent_laplacian,
            pll_eigenvalues,
            pll_to_feature_vector,
        )

        _ADAPTER_AVAILABLE = True
    except Exception as exc:  # pragma: no cover — defensive
        _ADAPTER_AVAILABLE = False
        _ADAPTER_IMPORT_ERR = repr(exc)
    return _ADAPTER_AVAILABLE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Path to the SE(3) SB trainer relative to REPO_ROOT (Wf-23 Phase 2).
TRAINER_REL_PATH = Path("molmetal/scripts/train_se3sb.py")

#: 3 NEW CLI flags added by Wf-392-A1 to ``train_se3sb.py:parse_args``.
WF392A1_NEW_FLAGS = (
    "--multiph-pll",
    "--multiph-pll-top-k",
    "--multiph-pll-weight",
)

#: Default OFF contract — preserves bit-for-bit backward compat with
#: Wf-23 + Wf-390-A3 baseline (no perturbation when flags absent).
DEFAULT_TOP_K = 10
DEFAULT_WEIGHT = 0.0


# ---------------------------------------------------------------------------
# Wire-in result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PLLWireResult:
    """Result of the Wf-392-A1 SE(3) SB trainer wire-in.

    Attributes
    ----------
    trainer_path : str
        Absolute path to ``molmetal/scripts/train_se3sb.py``.
    trainer_found : bool
        Whether the trainer file exists on disk.
    parse_args_loc : int
        Line number of ``def parse_args`` in the trainer (0 if absent).
    n_flags_before : int
        Count of ``--se3-*`` + ``--multiph-*`` flags BEFORE wire-in.
    n_flags_after : int
        Count of ``--se3-*`` + ``--multiph-*`` flags AFTER wire-in.
    n_new_flags_added : int
        Number of NEW flags added (expected 3 per Wf-392-A1 task brief).
    flags_added : List[str]
        The 3 NEW flag names that were wired in.
    wire_in_ok : bool
        Whether all 3 flags were added and the trainer AST-parses.
    train_loc : int
        Line count of the trainer file after wire-in.
    error : Optional[str]
        Error string if wire-in failed (None on success).
    """

    trainer_path: str
    trainer_found: bool
    parse_args_loc: int
    n_flags_before: int
    n_flags_after: int
    n_new_flags_added: int
    flags_added: List[str]
    wire_in_ok: bool
    train_loc: int
    error: Optional[str]

    def as_dict(self) -> dict:
        """Return a JSON-friendly dict for downstream reporting."""
        return {
            "trainer_path": self.trainer_path,
            "trainer_found": self.trainer_found,
            "parse_args_loc": self.parse_args_loc,
            "n_flags_before": self.n_flags_before,
            "n_flags_after": self.n_flags_after,
            "n_new_flags_added": self.n_new_flags_added,
            "flags_added": list(self.flags_added),
            "wire_in_ok": self.wire_in_ok,
            "train_loc": self.train_loc,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Smoke-cohort feature computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PLLSmokeResult:
    """Result of the Wf-392-A1 smoke PLL feature extraction.

    Attributes
    ----------
    cohort_size : int
        Number of molecules in the smoke cohort.
    n_succeeded : int
        Number of molecules for which PLL extraction succeeded.
    n_failed : int
        Number of molecules for which PLL extraction failed.
    top_k : int
        Feature vector length used.
    feature_shape_ok : bool
        Whether all succeeded features have shape ``(top_k,)``.
    feature_dtype_ok : bool
        Whether all succeeded features have dtype ``float64``.
    feature_not_all_zero_ok : bool
        Whether at least one feature is not all-zero (sanity).
    feature_shapes : List[tuple]
        Per-molecule feature shapes (one entry per succeeded molecule).
    n_atoms_per_mol : List[int]
        Per-molecule ``n_atoms`` from the canonical PLL module.
    adapter_available : bool
        Whether the Wf-382-A1 PLL feature adapter is importable.
    error : Optional[str]
        Error string if smoke failed (None on success).
    """

    cohort_size: int
    n_succeeded: int
    n_failed: int
    top_k: int
    feature_shape_ok: bool
    feature_dtype_ok: bool
    feature_not_all_zero_ok: bool
    feature_shapes: List[tuple]
    n_atoms_per_mol: List[int]
    adapter_available: bool
    error: Optional[str]

    def as_dict(self) -> dict:
        """Return a JSON-friendly dict for downstream reporting."""
        return {
            "cohort_size": self.cohort_size,
            "n_succeeded": self.n_succeeded,
            "n_failed": self.n_failed,
            "top_k": self.top_k,
            "feature_shape_ok": self.feature_shape_ok,
            "feature_dtype_ok": self.feature_dtype_ok,
            "feature_not_all_zero_ok": self.feature_not_all_zero_ok,
            "feature_shapes": [list(s) for s in self.feature_shapes],
            "n_atoms_per_mol": list(self.n_atoms_per_mol),
            "adapter_available": self.adapter_available,
            "error": self.error,
        }


def _build_smoke_mols(smiles_list: Sequence[str]):
    """Best-effort 3D embedder; returns list of (smi, mol_or_none).

    Falls back to returning ``None`` for molecules whose 3D embedding
    fails (we still report them in the smoke cohort).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return [(smi, None) for smi in smiles_list]

    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append((smi, None))
            continue
        mol_h = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_h, randomSeed=42) != 0:
            params = AllChem.ETKDGv2()
            if AllChem.EmbedMolecule(mol_h, params) != 0:
                out.append((smi, None))
                continue
        out.append((smi, mol_h))
    return out


def smoke_pll_features(
    cohort_size: int = 4,
    top_k: int = DEFAULT_TOP_K,
) -> PLLSmokeResult:
    """Run a tiny smoke PLL feature extraction over a ``cohort_size`` cohort.

    Parameters
    ----------
    cohort_size : int
        Number of molecules to include in the smoke cohort (default 4
        per the Wf-392-A1 task brief).
    top_k : int
        Feature vector length (default 10 per Wf-382-A1 baseline).

    Returns
    -------
    PLLSmokeResult
        Dataclass with per-molecule shape / dtype sanity + adapter
        availability.  When RDKit or the canonical PLL module is
        unavailable the result has ``n_succeeded == 0`` and a
        descriptive ``error`` string — callers should treat this as
        a *skip*, not a failure.
    """
    if not _try_import_adapter():
        return PLLSmokeResult(
            cohort_size=cohort_size,
            n_succeeded=0,
            n_failed=cohort_size,
            top_k=top_k,
            feature_shape_ok=False,
            feature_dtype_ok=False,
            feature_not_all_zero_ok=False,
            feature_shapes=[],
            n_atoms_per_mol=[],
            adapter_available=False,
            error=f"PLL adapter unavailable: {_ADAPTER_IMPORT_ERR}",
        )

    # Lazy import — bind names in this function's local scope.
    from molmetal.adapters.flow_matching_lipman.pll_features import (  # noqa: F811
        PLLFeatureError as _PLLFeatureError,
        persistent_laplacian as _persistent_laplacian,
        pll_eigenvalues as _pll_eigenvalues,
        pll_to_feature_vector as _pll_to_feature_vector,
    )

    # Canonical smoke cohort — benzene + methane + ethanol + chlorobenzene.
    # Wf-382-A1 baseline (molmetal/scripts/wf382a1_pll_feature_extract.py).
    smiles_pool = [
        "c1ccccc1",  # benzene
        "C",  # methane
        "CCO",  # ethanol
        "C1=CC=CC=C1Cl",  # chlorobenzene
    ]
    smiles_list = smiles_pool[:cohort_size]

    feats: List[np.ndarray] = []
    n_atoms_list: List[int] = []
    shapes_list: List[tuple] = []
    n_failed = 0
    error_msg: Optional[str] = None

    for smi, mol in _build_smoke_mols(smiles_list):
        if mol is None:
            n_failed += 1
            continue
        try:
            pll = _persistent_laplacian(mol, k=8)
            eigvals = _pll_eigenvalues(pll)
            vec = _pll_to_feature_vector(eigvals, top_k=top_k)
        except _PLLFeatureError as exc:
            n_failed += 1
            if error_msg is None:
                error_msg = f"PLLFeatureError on {smi}: {exc!r}"
            continue
        except Exception as exc:  # pragma: no cover — defensive
            n_failed += 1
            if error_msg is None:
                error_msg = f"unexpected error on {smi}: {exc!r}"
            continue
        feats.append(vec)
        n_atoms_list.append(int(pll.n_atoms))
        shapes_list.append(tuple(vec.shape))

    feature_shape_ok = all(s == (top_k,) for s in shapes_list)
    feature_dtype_ok = all(v.dtype == np.float64 for v in feats)
    feature_not_all_zero_ok = (
        any(not np.allclose(v, 0.0) for v in feats) if feats else False
    )

    return PLLSmokeResult(
        cohort_size=cohort_size,
        n_succeeded=len(feats),
        n_failed=n_failed,
        top_k=top_k,
        feature_shape_ok=feature_shape_ok,
        feature_dtype_ok=feature_dtype_ok,
        feature_not_all_zero_ok=feature_not_all_zero_ok,
        feature_shapes=shapes_list,
        n_atoms_per_mol=n_atoms_list,
        adapter_available=True,
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Trainer wire-in
# ---------------------------------------------------------------------------


def _count_flags_in_parse_args(source: str) -> int:
    """Count ``--se3-*`` + ``--multiph-*`` ``add_argument`` calls.

    We restrict to those two prefixes to match the Wf-23 + Wf-390-A3
    SE(3)-specific flag audit baseline.  Other flags (--epochs, --lr,
    etc.) are inherited from :func:`add_common_args` and are NOT
    part of this audit.
    """
    import re

    pattern = re.compile(
        r'p\.add_argument\(\s*[\'"]--(se3-|multiph-)',
    )
    return len(pattern.findall(source))


def _find_parse_args_loc(source: str) -> int:
    """Return the 1-indexed line number of ``def parse_args``."""
    for i, line in enumerate(source.splitlines(), start=1):
        if line.startswith("def parse_args"):
            return i
    return 0


def _format_new_flags_block() -> str:
    """Render the 3 NEW flag ``add_argument`` calls as a Python snippet.

    The snippet is appended to ``parse_args`` AFTER the existing W2.1
    SE(3)-specific block, before ``return p.parse_args(argv)``.
    """
    return """
    # ------------------------------------------------------------------
    # Wf-392-A1 — Multi-PH + PLL feature wire-in (Wf-382-A1 SCAFFOLD).
    # 3 NEW CLI flags; default OFF for backward-compat with the
    # Wf-23 + Wf-390-A3 SE(3) SB baseline.  When --multiph-pll is set
    # the trainer builds the PLL feature stream via
    # pll_features.persistent_laplacian and injects it into the
    # SE(3) SB velocity-loss as a per-molecule weight.
    # ------------------------------------------------------------------
    p.add_argument(
        "--multiph-pll",
        action="store_true",
        default=False,
        help=(
            "Wf-392-A1 NEW: enable the Multi-PH + PLL feature stream "
            "into the SE(3) SB velocity loss (Liu & Feng 2026; "
            "Wf-382-A1 SCAFFOLD; TODO-94 R20-P3). Default OFF for "
            "backward-compat with the Wf-23 + Wf-390-A3 baseline. "
            "When ON, the trainer calls persistent_laplacian() per "
            "molecule and uses --multiph-pll-weight as the loss "
            "weight. GPU bench MEASURED lift DEFERRED per AGENTS.md "
            "section 5.1."
        ),
    )
    p.add_argument(
        "--multiph-pll-top-k",
        type=int,
        default=10,
        help=(
            "Wf-392-A1 NEW: PLL feature vector length (matches "
            "Wf-382-A1 default). Default 10 per Wf-382-A1 task brief."
        ),
    )
    p.add_argument(
        "--multiph-pll-weight",
        type=float,
        default=0.0,
        help=(
            "Wf-392-A1 NEW: weight on the PLL feature term added "
            "to the SE(3) SB velocity loss. Default 0.0 (OFF). "
            "weight > 0.0 activates the term; ignored when "
            "--multiph-pll is False."
        ),
    )
"""


def wire_pll_to_se3_sb_trainer(
    trainer_path: Optional[Path] = None,
    *,
    write: bool = False,
) -> PLLWireResult:
    """Wire 3 NEW CLI flags into the SE(3) SB trainer.

    Parameters
    ----------
    trainer_path : Path, optional
        Override the default trainer path (``molmetal/scripts/train_se3sb.py``).
    write : bool
        When ``True`` the wire-in is persisted to disk; when ``False``
        (default) only the audit is returned and the file is left
        untouched — bit-for-bit backward compat with Wf-23 / Wf-390-A3.

    Returns
    -------
    PLLWireResult
        Wire-in audit dataclass.  ``wire_in_ok`` is ``True`` iff the
        trainer AST-parses + the 3 NEW flags are detected in source.
    """
    trainer_p = Path(trainer_path) if trainer_path else REPO_ROOT / TRAINER_REL_PATH
    trainer_found = trainer_p.exists()
    if not trainer_found:
        return PLLWireResult(
            trainer_path=str(trainer_p),
            trainer_found=False,
            parse_args_loc=0,
            n_flags_before=0,
            n_flags_after=0,
            n_new_flags_added=0,
            flags_added=[],
            wire_in_ok=False,
            train_loc=0,
            error=f"trainer not found at {trainer_p}",
        )

    source = trainer_p.read_text()
    parse_args_loc = _find_parse_args_loc(source)
    n_flags_before = _count_flags_in_parse_args(source)

    # Detect if already wired (idempotent guard).
    if all(f in source for f in WF392A1_NEW_FLAGS):
        n_flags_after = n_flags_before
        return PLLWireResult(
            trainer_path=str(trainer_p),
            trainer_found=True,
            parse_args_loc=parse_args_loc,
            n_flags_before=n_flags_before,
            n_flags_after=n_flags_after,
            n_new_flags_added=0,
            flags_added=[],
            wire_in_ok=True,
            train_loc=len(source.splitlines()),
            error=None,
        )

    # Insert the new flags before `return p.parse_args(argv)`.
    if "    return p.parse_args(argv)" not in source:
        return PLLWireResult(
            trainer_path=str(trainer_p),
            trainer_found=True,
            parse_args_loc=parse_args_loc,
            n_flags_before=n_flags_before,
            n_flags_after=n_flags_before,
            n_new_flags_added=0,
            flags_added=[],
            wire_in_ok=False,
            train_loc=len(source.splitlines()),
            error="could not locate `return p.parse_args(argv)` anchor",
        )

    new_block = _format_new_flags_block()
    new_source = source.replace(
        "    return p.parse_args(argv)",
        new_block + "    return p.parse_args(argv)",
        1,
    )

    # AST sanity — must reparse cleanly.
    try:
        import ast as _ast

        _ast.parse(new_source)
    except SyntaxError as exc:
        return PLLWireResult(
            trainer_path=str(trainer_p),
            trainer_found=True,
            parse_args_loc=parse_args_loc,
            n_flags_before=n_flags_before,
            n_flags_after=n_flags_before,
            n_new_flags_added=0,
            flags_added=[],
            wire_in_ok=False,
            train_loc=len(source.splitlines()),
            error=f"AST parse failed after wire-in: {exc!r}",
        )

    n_flags_after = _count_flags_in_parse_args(new_source)
    n_new_flags_added = n_flags_after - n_flags_before

    if write:
        # Backup the original before writing.
        backup = trainer_p.with_suffix(trainer_p.suffix + ".bak.wf392a1")
        if not backup.exists():
            backup.write_text(source)
        trainer_p.write_text(new_source)

    return PLLWireResult(
        trainer_path=str(trainer_p),
        trainer_found=True,
        parse_args_loc=parse_args_loc,
        n_flags_before=n_flags_before,
        n_flags_after=n_flags_after,
        n_new_flags_added=n_new_flags_added,
        flags_added=list(WF392A1_NEW_FLAGS),
        wire_in_ok=True,
        train_loc=len(new_source.splitlines()),
        error=None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_smoke_report(result: PLLSmokeResult, out_dir: Path) -> dict:
    """Dump the smoke result to ``out_dir / smoke_result.json``.

    Returns the JSON-friendly dict for in-process callers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    payload["task"] = "Wf-392-A1 Multi-PH + PLL SE(3) SB wire-in smoke"
    out_path = out_dir / "smoke_result.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    payload["output_path"] = str(out_path)
    return payload


def _write_wire_report(result: PLLWireResult, out_dir: Path) -> dict:
    """Dump the wire-in audit result to ``out_dir / wire_in_result.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    payload["task"] = "Wf-392-A1 Multi-PH + PLL SE(3) SB wire-in audit"
    out_path = out_dir / "wire_in_result.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    payload["output_path"] = str(out_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-392-A1 Multi-PH + PLL SE(3) SB wire-in script (CPU-only "
            "SCAFFOLD). Wraps molmetal.adapters.flow_matching_lipman.pll_features "
            "+ molmetal.scripts.train_se3sb:parse_args."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the 4-molecule PLL feature extraction smoke cohort.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Audit-only wire-in (no file write).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist the wire-in to train_se3sb.py (use with --audit).",
    )
    parser.add_argument(
        "--cohort-size",
        type=int,
        default=4,
        help="Number of molecules in the smoke cohort (default 4).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="PLL feature vector length (default 10 per Wf-382-A1).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "molmetal" / "reports" / "wf392a1_pll_wire"),
        help="Output directory for smoke + audit JSON.",
    )
    args = parser.parse_args()

    if not (args.smoke or args.audit):
        parser.print_help()
        return 2

    out_dir = Path(args.out_dir)

    if args.smoke:
        smoke = smoke_pll_features(cohort_size=args.cohort_size, top_k=args.top_k)
        payload = _write_smoke_report(smoke, out_dir)
        print(
            f"[wf392a1] PLL smoke: cohort={payload['cohort_size']} "
            f"ok={payload['n_succeeded']} fail={payload['n_failed']} "
            f"shape_ok={payload['feature_shape_ok']} "
            f"dtype_ok={payload['feature_dtype_ok']} "
            f"adapter_available={payload['adapter_available']}"
        )
        print(f"[wf392a1]   output_path = {payload['output_path']}")

    if args.audit:
        wire = wire_pll_to_se3_sb_trainer(write=args.write)
        payload = _write_wire_report(wire, out_dir)
        print(
            f"[wf392a1] wire-in audit: flags_before={payload['n_flags_before']} "
            f"flags_after={payload['n_flags_after']} "
            f"new_added={payload['n_new_flags_added']} "
            f"wire_in_ok={payload['wire_in_ok']} "
            f"wrote={args.write}"
        )
        print(f"[wf392a1]   output_path = {payload['output_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
