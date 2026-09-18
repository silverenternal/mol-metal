"""PoseBusters outer-gate runner — MMFF94-based physics/chemistry validity.

================================================================
Why this module exists (R3 audit)
================================================================
The existing adapter at
``molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py`` (the
``PoseBustersAdapter`` class) is the full ``posebusters`` API wrapper
and is used for batched validation of the *whole* candidate list.

For the closed-loop scoring wire we need a *lighter* entry point:

* ETKDGv3 → MMFF94 (NOT UFF) conformer pipeline — the MMFF94 window is
  what PoseBusters' reference geometries are tuned against, so
  pre-optimising with MMFF94 is the only path that hits the same
  validity window.  UFF falls outside it on triazoles / heterocycles.
* A **graceful skip** when ``posebusters`` itself is not installed
  (this is the canonical "no new deps" path — see TODO/environment.md
  ENV CONSTRAINTS).
* A **per-call dict** return so callers (the RewardAggregator channel
  + the closed-loop outer gate) can use it directly without going
  through the adapter's dataclass machinery.

Public API
----------
* :func:`check_posebusters` — one-shot check on a single SMILES.
* :func:`check_docked_pose` — validate supplied coordinates against a receptor.
* :func:`check_posebusters_batch` — bulk check on a SMILES list.
* :func:`pb_pass_rate` — fraction of ``passed`` over a list.
* :func:`posebusters_available` — probe for the package.

The module is intentionally **independent** of
``molmetal_lam.sbdd_env.posebusters_adapter`` so that callers do not
have to import the heavy adapter module when all they want is the
outer-gate pass/fail.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Optional-dep probes — keep failures from crashing the pipeline.
# ----------------------------------------------------------------
def _have_rdkit() -> bool:
    """Return ``True`` iff RDKit can be imported."""
    try:
        import rdkit  # noqa: F401  - probe only
        from rdkit import Chem  # noqa: F401
        return True
    except Exception:
        return False


def posebusters_available() -> bool:
    """Return ``True`` iff the ``posebusters`` package can be imported.

    Used by callers (and the closed-loop outer gate) to decide whether
    to fall back to the "skipped" path.  Also consumed by the unit
    test ``test_pb_runner_handles_missing_pkg`` so the missing-pkg
    contract is pinned regardless of the actual install state.
    """
    try:
        import posebusters  # noqa: F401
        return True
    except Exception:
        return False


# ----------------------------------------------------------------
# Conformers
# ----------------------------------------------------------------
def _embed_conformers(
    smiles: str,
    n_conformers: int,
    embed_method: str,
    optimize_method: str,
) -> List[Any]:
    """Embed ``n_conformers`` 3D conformers of ``smiles``.

    Pipeline:
        1. ``MolFromSmiles``  → parse SMILES
        2. ``AddHs``          → add explicit hydrogens
        3. ``EmbedMultipleConfs`` with the ETKDGv3 params (default)
        4. ``MMFF94OptimizeMolecule`` on each conformer — **MMFF94,
            NOT UFF** (see module docstring).  Falls back to UFF only
            when MMFF94 cannot parameterise the molecule.

    Returns a list of (mol_with_optimized_confs, conf_id) tuples.  The
    caller is responsible for invoking PoseBusters on the resulting
    molecule; we return the molecule object rather than the per-check
    result so the caller can inspect the geometry if needed.

    Returns an empty list when RDKit embedding fails — the caller
    (e.g. :func:`check_posebusters`) translates that into
    ``"failures": ["embed_failed"]``.
    """
    if not _have_rdkit():
        return []
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return []
    mol = Chem.AddHs(mol)

    # ETKDGv3 is the canonical RDKit embed method that matches the
    # PoseBusters reference window.  We accept "ETKDG" / "ETKDGv2" /
    # "ETKDGv3" but the default and the documented path is ETKDGv3.
    if embed_method.upper() == "ETKDG":
        params = AllChem.ETKDG()
    elif embed_method.upper() == "ETKDGV2":
        params = AllChem.ETKDGv2()
    else:
        params = AllChem.ETKDGv3()
    params.randomSeed = 42
    # ``EmbedMultipleConfs`` populates the molecule with up to
    # ``n_conformers`` 3D coordinates. Returns a vector of conformer IDs,
    # not an integer count; IDs need not be consecutive.
    try:
        conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_conformers), params=params))
    except Exception:
        conformer_ids = []
    if not conformer_ids:
        return []

    # MMFF94 optimisation (NOT UFF — see module docstring).
    if optimize_method.upper() == "UFF":
        optimizer = AllChem.UFFOptimizeMolecule
    else:
        optimizer = AllChem.MMFFOptimizeMolecule  # RDKit's default variant is MMFF94.
    for cid in conformer_ids:
        try:
            status = optimizer(mol, confId=cid, maxIters=200)
            # Status 0 = converged; non-zero = caller fall-back.
            if status != 0 and optimize_method.upper() != "UFF":
                try:
                    AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=200)
                except Exception:
                    pass
        except Exception:
            continue

    return [(mol, int(cid)) for cid in conformer_ids]


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------
def _binary_value(value):
    """Accept actual booleans only; NaN, pd.NA, strings and numeric 1 fail closed."""
    import numpy as np
    return bool(value) if isinstance(value, (bool, np.bool_)) else None


def _report_checks(df, config) -> Dict[str, Any]:
    """Reduce every configured binary check, including missing/object columns.

    PoseBusters renames chosen outputs and lowercases the resulting columns.
    Numeric diagnostics and loading checks not selected by the configuration
    (e.g. mol_true_loaded in dock mode) are not pass/fail requirements.
    """
    expected = []
    for module in config.get("modules", []):
        for output in module.get("chosen_binary_test_output", []):
            name = module.get("rename_outputs", {}).get(
                output, output + module.get("rename_suffix", ""))
            name = name.lower().replace(" ", "_")
            if name not in expected:
                expected.append(name)
    if not expected or df is None or len(df) == 0:
        return {"pb_valid": False, "checks": {key: None for key in expected},
                "row_checks": [], "missing_checks": expected,
                "failures": ["empty_report" if expected else "missing_check_schema"]}
    row_checks = [{key: _binary_value(row[key]) if key in df.columns else None
                   for key in expected} for _, row in df.iterrows()]
    missing = [key for key in expected if any(row[key] is None for row in row_checks)]
    checks = {key: (None if key in missing else all(row[key] for row in row_checks))
              for key in expected}
    failures = [key for key, value in checks.items() if value is not True]
    return {"pb_valid": not failures, "checks": checks, "row_checks": row_checks,
            "missing_checks": missing, "failures": failures}


def check_docked_pose(mol, receptor_path) -> Dict[str, Any]:
    """Check the supplied single-conformer pose with PoseBusters ``dock``.

    A copy of the original molecular graph and coordinates is passed directly
    as mol_pred, with receptor_path as mol_cond. This function never embeds,
    minimizes, adds hydrogens or changes input coordinates. PoseBusters may
    generate its own internal reference ensemble for the energy-ratio check;
    that does not replace the evaluated input pose. No reference ligand is
    supplied, so no redocking RMSD is implied.
    """
    result = {"pb_valid": False, "backend": "posebusters", "config": "dock",
              "status": "invalid_input", "checks": {}, "row_checks": [],
              "missing_checks": [], "failures": [], "skipped": False,
              "n_conformers": 0, "version": None,
              "coordinates_preserved": True}
    try:
        from rdkit import Chem
        import numpy as np
        if mol is None or not isinstance(mol, Chem.Mol) or not mol.GetNumAtoms():
            raise ValueError("missing_or_empty_molecule")
        if mol.GetNumConformers() != 1:
            raise ValueError("exactly_one_input_conformer_required")
        if not np.isfinite(mol.GetConformer().GetPositions()).all():
            raise ValueError("nonfinite_input_coordinates")
        receptor_path = os.fspath(receptor_path)
        if not os.path.isfile(receptor_path):
            raise ValueError("receptor_file_missing")
        pose = Chem.Mol(mol)  # preserve graph/conformer and protect the caller from backend mutation
        result["n_conformers"] = 1
    except Exception as exc:
        result["failures"] = [str(exc)]
        return result
    if not posebusters_available():
        result.update(pb_valid=None, skipped=True, status="unavailable",
                      failures=["posebusters_not_installed"])
        return result
    try:
        from importlib.metadata import version, PackageNotFoundError
        from posebusters import PoseBusters
        try:
            result["version"] = version("posebusters")
        except PackageNotFoundError:
            pass
        pb = PoseBusters(config="dock")
        df = pb.bust(mol_pred=pose, mol_cond=receptor_path, full_report=True)
        result.update(_report_checks(df, pb.config))
        result["status"] = "passed" if result["pb_valid"] else "failed"
    except Exception as exc:
        result.update(status="error", failures=["posebusters_bust_failed"], error=str(exc))
    return result


def check_posebusters(
    smiles: str,
    *,
    n_conformers: int = 5,
    embed_method: str = "ETKDGv3",
    optimize_method: str = "MMFF94",
) -> Dict[str, Any]:
    """Validate a single SMILES via PoseBusters under an MMFF94 window.

    Parameters
    ----------
    smiles : str
        Input SMILES.  Empty / unparsable strings return
        ``{"pb_valid": False, "failures": ["..."]}``.
    n_conformers : int, default 5
        Number of 3D conformers to embed.  PoseBusters validates the
        molecule on the conformer ensemble — 5 is the paper-grade
        default.
    embed_method : str, default ``"ETKDGv3"``
        RDKit embed method.  ``"ETKDG"``, ``"ETKDGv2"`` and
        ``"ETKDGv3"`` are accepted.
    optimize_method : str, default ``"MMFF94"``
        Geometry optimiser.  ``"MMFF94"`` is the canonical path —
        PoseBusters' reference MMFF94 window is what we want to be in.
        ``"UFF"`` is supported only as an explicit opt-out.

    Returns
    -------
    dict
        Always a dict.  Shape:

        * ``{"pb_valid": True|False, "n_conformers": int,
             "failures": [str, ...]}`` — full result.
        * ``{"pb_valid": None, "skipped": True,
             "reason": "..."}`` — ``posebusters`` is not installed.
        * ``{"pb_valid": False, "failures": ["embed_failed"]}`` —
            RDKit embedding failed (or RDKit itself is missing).

        The "skipped" branch is consumed by the closed-loop outer gate
        so the pipeline can record pass/fail in history even when PB
        is absent — the gate degrades gracefully, it does not crash.
    """
    # 1) Missing-pkg fast path — explicit so the closed-loop gate can
    # record this as a per-iter "skipped" status without invoking the
    # rest of the pipeline.
    if not posebusters_available():
        return {
            "pb_valid": None,
            "skipped": True,
            "reason": "posebusters not installed",
            "failures": [],
            "n_conformers": 0,
        }

    # 2) Empty / invalid SMILES — fail early with a parse failure.
    if not smiles or not isinstance(smiles, str):
        return {
            "pb_valid": False,
            "skipped": False,
            "failures": ["empty_smiles"],
            "n_conformers": 0,
        }

    # 3) Embed + optimise (MMFF94 by default).
    conformers = _embed_conformers(
        smiles,
        n_conformers=int(n_conformers),
        embed_method=embed_method,
        optimize_method=optimize_method,
    )
    if not conformers:
        return {
            "pb_valid": False,
            "skipped": False,
            "failures": ["embed_failed"],
            "n_conformers": 0,
        }

    mol, _first_cid = conformers[0]

    # 4) Run PoseBusters.  The bust() call works on the
    # conformer-equipped molecule; we use ``full_report=True`` so we
    # can also extract per-check failure names.
    try:
        from posebusters import PoseBusters  # type: ignore
        pb = PoseBusters(config="mol")
        df = pb.bust(mol, full_report=True)
    except Exception as exc:
        # PoseBusters raised — treat as a soft failure so the pipeline
        # degrades gracefully (e.g. when a required PB dataset is
        # missing on this install).
        log.debug("PoseBusters.bust failed: %s", exc)
        return {
            "pb_valid": False,
            "skipped": False,
            "failures": ["posebusters_bust_failed"],
            "n_conformers": int(len(conformers)),
        }

    # Configured binary checks must all be present and true, across every row.
    try:
        report = _report_checks(df, pb.config)
    except Exception:
        report = {"pb_valid": False, "checks": {}, "row_checks": [],
                  "missing_checks": [], "failures": ["report_parse_failed"]}
    return {
        **report,
        "backend": "posebusters", "config": "mol",
        "status": "passed" if report["pb_valid"] else "failed",
        "skipped": False,
        "n_conformers": int(len(conformers)),
    }


def check_posebusters_batch(
    smiles_list: List[str],
    *,
    n_conformers: int = 5,
    embed_method: str = "ETKDGv3",
    optimize_method: str = "MMFF94",
) -> List[Dict[str, Any]]:
    """Apply :func:`check_posebusters` to a list of SMILES.

    Each element is run sequentially — there is no parallelisation
    inside this function (the caller can fan out if needed).  Always
    returns one result dict per input element.
    """
    return [
        check_posebusters(
            s,
            n_conformers=n_conformers,
            embed_method=embed_method,
            optimize_method=optimize_method,
        )
        for s in (smiles_list or [])
    ]


def pb_pass_rate(results: List[Dict[str, Any]]) -> float:
    """Fraction of all input reports with an explicit passing verdict.

    Failed, skipped, missing, malformed and NaN verdicts stay in the
    denominator. Empty input retains the historical 0.0 return value.
    """
    if not results:
        return 0.0
    n_passed = sum(isinstance(r, dict) and _binary_value(r.get("pb_valid")) is True
                   and _binary_value(r.get("skipped", False)) is False for r in results)
    return float(n_passed) / len(results)


__all__ = [
    "check_posebusters",
    "check_docked_pose",
    "check_posebusters_batch",
    "pb_pass_rate",
    "posebusters_available",
]
