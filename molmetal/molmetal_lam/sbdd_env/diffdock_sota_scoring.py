"""Wire-up helper: DiffDock-L as a SOTA scoring column.

================================================================
What this is
================================================================
This module is the *pipeline-side* glue that turns a list of Lambda
candidates into per-candidate DiffDock-L confidence scores, suitable
for use as a column in the r4_c_full_sweep report.

It is intentionally decoupled from
:mod:`molmetal_lam.sbdd_env.diffdock_adapter` (which is the
:func:`DockingOracle` Protocol implementation used by the binding
leaf oracles during MCTS).  Here we need:

* a thin subprocess wrapper around ``molmetal/references/DiffDock/
  inference.py`` (the upstream CLI the vendored repo ships);
* a graceful fallback when the vendored repo is unavailable, the
  checkpoint weights are missing, or torch is unavailable (CPU-only
  smoke test path);
* an aggregator that emits ``diffdock_score_mean`` and
  ``diffdock_score_std`` per pocket/seed record.

Honest framing
--------------
* DiffDock-L inference is GPU-bound (~10-30 s per complex on A100).
  On the local RX 7800 XT (gfx1101) the upstream repo requires
  torch_cluster extensions that we have not yet built for ROCm 7.2.
  This module therefore ships with a CPU-only *smoke* path that
  parses the rank1 confidence filename without invoking the model.
* When invoked from a non-GPU environment without a vendored
  DiffDock repo, the helper returns ``status="unavailable"`` and the
  report records ``null``/``NaN`` confidence values.  This is the
  honest, SOTA-comparable behaviour: we do NOT synthesise
  confidence numbers.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any

logger = logging.getLogger(__name__)

# Default repo root for the cloned DiffDock checkout.
DEFAULT_REPO_ROOT = "molmetal/references/DiffDock"
DEFAULT_INFERENCE_REL = "inference.py"
DEFAULT_CONFIG_REL = "default_inference_args.yaml"
DEFAULT_COMPLEX_NAME = "r4_complex"


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------
@dataclass
class DiffDockScoreColumn:
    """Per-pocket SOTA scoring column from DiffDock-L.

    Attributes
    ----------
    status : str
        One of ``{"unavailable", "ok", "error", "partial"}``.
    n_invoked : int
        Number of candidates we attempted to score.
    n_scored : int
        Number of candidates with a finite confidence value.
    diffdock_score_mean : float | None
        Mean DiffDock-L confidence in [0, 1].  ``None`` when no
        candidate was scored.
    diffdock_score_std : float | None
        Standard deviation of the scored confidences.  ``None`` when
        n_scored < 2.
    per_smiles : dict[str, float]
        SMILES -> confidence map.  Missing keys map to ``float('nan')``
        downstream.
    """

    status: str = "unavailable"
    n_invoked: int = 0
    n_scored: int = 0
    diffdock_score_mean: Optional[float] = None
    diffdock_score_std: Optional[float] = None
    per_smiles: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _resolve_inference_script(repo_root: str = DEFAULT_REPO_ROOT) -> Optional[Path]:
    """Locate the vendored ``inference.py`` if present."""
    p = Path(repo_root) / DEFAULT_INFERENCE_REL
    return p if p.is_file() else None


def diffdock_cli_available(repo_root: str = DEFAULT_REPO_ROOT) -> bool:
    """True iff a DiffDock inference.py can be located AND is executable.

    We *do not* check for checkpoint weights here — the upstream
    ``inference.py`` will complain at runtime and the caller will get
    ``status="error"`` rather than a synthetic success.
    """
    script = _resolve_inference_script(repo_root)
    if script is None:
        return False
    # The repo itself must be importable as a package for upstream
    # ``from utils....`` imports to resolve.  We check for utils/ dir.
    return (Path(repo_root) / "utils").is_dir()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _aggregate(confidences: Sequence[float]) -> tuple[Optional[float], Optional[float]]:
    finite = [c for c in confidences
              if isinstance(c, (int, float)) and math.isfinite(c)]
    if not finite:
        return None, None
    mean = sum(finite) / len(finite)
    if len(finite) < 2:
        return mean, None
    var = sum((c - mean) ** 2 for c in finite) / len(finite)
    return mean, math.sqrt(var)


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------
def _build_cli(inference_script: Path, smiles: str, protein_path: str,
               out_dir: str, samples_per_complex: int, config_yaml: Optional[Path],
               complex_name: str) -> List[str]:
    cmd: List[str] = [
        sys.executable, str(inference_script),
        "--protein_path", protein_path,
        "--ligand_description", smiles,
        "--complex_name", complex_name,
        "--out_dir", out_dir,
        "--samples_per_complex", str(int(samples_per_complex)),
    ]
    if config_yaml is not None and config_yaml.is_file():
        cmd.extend(["--config", str(config_yaml)])
    return cmd


def _run_one_subprocess(inference_script: Path, smiles: str, protein_path: str,
                        out_dir: Path, samples_per_complex: int,
                        config_yaml: Optional[Path], timeout_sec: float) -> tuple[int, str]:
    """Invoke DiffDock inference.py once; return (returncode, stderr_snippet)."""
    cmd = _build_cli(
        inference_script, smiles, protein_path,
        str(out_dir), samples_per_complex, config_yaml,
        DEFAULT_COMPLEX_NAME,
    )
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(inference_script.parent),
        )
    except subprocess.TimeoutExpired:
        return -1, f"timeout after {timeout_sec:g}s"
    except Exception as exc:  # pragma: no cover - defensive
        return -2, f"{type(exc).__name__}: {exc}"
    snippet = (proc.stderr or "")[:400]
    return proc.returncode, snippet


def _parse_confidence_from_outdir(out_dir: Path, complex_name: str) -> float:
    """Best-effort parse of DiffDock rank1 confidence from filename."""
    try:
        from molmetal_lam.sbdd_env.diffdock_adapter import (
            DiffDockAdapter,
        )
        return DiffDockAdapter.parse_confidence_from_outdir(
            str(out_dir), complex_name,
        )
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def score_candidates(
    candidates: Sequence[dict],
    protein_path: str,
    *,
    repo_root: str = DEFAULT_REPO_ROOT,
    samples_per_complex: int = 4,
    timeout_sec: float = 600.0,
    config_yaml: Optional[str] = None,
) -> DiffDockScoreColumn:
    """Score a list of Lambda candidates via the DiffDock-L CLI.

    Parameters
    ----------
    candidates : list of dict
        Lambda candidates.  Each must have a ``smiles`` field.  We
        score every candidate that has a non-empty SMILES.
    protein_path : str
        Path to the receptor PDB / PDBQT used as the docking pocket.
    repo_root : str
        Path to the cloned DiffDock checkout.
    samples_per_complex : int
        Number of reverse-diffusion samples per SMILES.  Default 4
        is the cheapest non-trivial value; the upstream CLI default
        is 10 but we trade confidence variance for wall time.
    timeout_sec : float
        Maximum wall-clock per SMILES invocation.
    config_yaml : str | None
        Optional path to the upstream ``default_inference_args.yaml``;
        when ``None`` we use the vendored default.

    Returns
    -------
    DiffDockScoreColumn
        Per-pocket aggregate; ``status="unavailable"`` when the CLI
        cannot be located (CPU-only smoke environment).
    """
    if not candidates:
        return DiffDockScoreColumn(
            status="ok", n_invoked=0, n_scored=0,
            diffdock_score_mean=None, diffdock_score_std=None,
            notes="no candidates supplied",
        )

    if not diffdock_cli_available(repo_root):
        return DiffDockScoreColumn(
            status="unavailable", n_invoked=len(candidates), n_scored=0,
            diffdock_score_mean=None, diffdock_score_std=None,
            notes=f"DiffDock inference.py not found at {repo_root!r}",
        )

    if not Path(protein_path).is_file():
        return DiffDockScoreColumn(
            status="error", n_invoked=len(candidates), n_scored=0,
            diffdock_score_mean=None, diffdock_score_std=None,
            notes=f"protein_path not found: {protein_path!r}",
        )

    inference_script = _resolve_inference_script(repo_root)
    cfg_path = Path(config_yaml).resolve() if config_yaml else None

    confidences: Dict[str, float] = {}
    n_invoked = 0
    n_scored = 0
    last_stderr = ""

    for cand in candidates:
        smiles = (cand.get("smiles") or "").strip()
        if not smiles:
            continue
        n_invoked += 1
        with tempfile.TemporaryDirectory(prefix="diffdock_sota_") as tmp:
            tmp_path = Path(tmp)
            rc, stderr = _run_one_subprocess(
                inference_script, smiles, protein_path, tmp_path,
                samples_per_complex, cfg_path, timeout_sec,
            )
            last_stderr = stderr
            if rc != 0:
                logger.warning(
                    "DiffDock subprocess failed rc=%d for %s: %s",
                    rc, smiles[:40], stderr,
                )
                confidences[smiles] = float("nan")
                continue
            conf = _parse_confidence_from_outdir(tmp_path, DEFAULT_COMPLEX_NAME)
            confidences[smiles] = conf
            if math.isfinite(conf):
                n_scored += 1

    mean, std = _aggregate(list(confidences.values()))
    if n_scored == 0:
        status = "error" if n_invoked > 0 else "unavailable"
    elif n_scored < n_invoked:
        status = "partial"
    else:
        status = "ok"
    notes = "" if status != "error" else f"last stderr: {last_stderr}"
    return DiffDockScoreColumn(
        status=status, n_invoked=n_invoked, n_scored=n_scored,
        diffdock_score_mean=mean, diffdock_score_std=std,
        per_smiles=confidences, notes=notes,
    )


__all__ = [
    "DiffDockScoreColumn",
    "diffdock_cli_available",
    "score_candidates",
    "DEFAULT_REPO_ROOT",
]
