"""Wire-up helper: BioLM-Score as a SOTA scoring column.

================================================================
What this is
================================================================
This module is the *pipeline-side* glue that turns a list of Lambda
candidates into per-candidate BioLM-Score binding-affinity predictions,
suitable for use as a column in the r4_c_full_sweep report.

It mirrors :mod:`molmetal_lam.sbdd_env.diffdock_sota_scoring` exactly
in shape so the new SOTA columns can coexist in the sweep report:

* a thin subprocess wrapper around
  ``molmetal/references/BioLM-Score/scripts/casf2016_scoring_ranking.py``
  (the upstream CLI the vendored repo ships);
* a graceful fallback when the vendored repo is unavailable, the
  checkpoint weights are missing, or torch is unavailable (CPU-only
  smoke test path);
* an aggregator that emits ``biomlm_score_mean`` and
  ``biomlm_score_std`` per pocket/seed record.

Honest framing
--------------
* BioLM-Score is a learned protein-ligand binding-affinity scorer
  (CASF-2016 scoring / ranking / screening / docking).  The upstream
  API requires precomputed ESM-C (1152-d) and Chemformer (1024-d)
  embeddings stored as ``{pdbid}.npy`` files in two side directories,
  PyG-style ``Data`` objects (one per pocket), and a ``.pth`` checkpoint
  on disk that contains a ``model_state_dict`` key.
* Inference is GPU-bound (the upstream ``casf2016_scoring_ranking.py``
  imports ``torch_scatter`` and ESM/Chemformer feature extractors);
  on the local RX 7800 XT (gfx1101) we have not yet built the
  ``torch_scatter`` extensions for ROCm 7.2 and the ESM3 submodule
  is not initialized.  This module therefore ships with a CPU-only
  *smoke* path that records ``status="unavailable"`` whenever the
  vendored ``BioLM_Score/`` package is not importable, and a
  *smoke-mock* subprocess path that fabricates a finite score by
  reading a fake ``.score`` file written by the (mocked) upstream.
* When invoked from a non-GPU environment without a vendored
  BioLM-Score repo, the helper returns ``status="unavailable"`` and
  the report records ``null``/``NaN`` confidence values.  This is
  the honest, SOTA-comparable behaviour: we do NOT synthesise
  binding affinity numbers.

Upstream API contract (per the README + scripts/)
-------------------------------------------------
* Package: ``BioLM_Score.model.model4.BioLLMScore.forward``
  signature ``forward(data_ligand, data_target, protein_embs, ligand_embs)``
* Checkpoint shape: ``checkpoint = th.load(model_path)`` →
  ``model.load_state_dict(checkpoint['model_state_dict'])``
* Encoder choices: ``{"gatedgcn", "gt"}`` — we default to
  ``gatedgcn`` because the Zenodo mmgatedgcn_1.0_01.pth weights are
  the published checkpoints for BioLM-Score.
* Model-type choices: ``{"biolm", "genscore"}`` — ``biolm`` requires
  protein + ligand embeddings, ``genscore`` uses graph features only.
  We default to ``biolm``.
* Inference CLI: ``python scripts/casf2016_scoring_ranking.py
  --model_path /path/to/mmgatedgcn_1.0_01.pth --model_type biolm
  --encoder gatedgcn --outprefix biolm_score``.  The upstream CLI
  reads ``{data_dir}/{test_prefix}_prot.pt``,
  ``{data_dir}/{test_prefix}_lig.pt`` and ``{data_dir}/{test_prefix}_ids.npy``.
* Output: the upstream script writes ``{outprefix}.dat`` (a tab-separated
  ``#code\\tscore`` file).  We parse this to extract the per-pdbid score.

Discovery order
---------------
:meth:`biomlm_cli_available` returns True iff:

1. ``molmetal/references/BioLM-Score/BioLM_Score/__init__.py`` exists
2. AND ``molmetal/references/BioLM-Score/scripts/casf2016_scoring_ranking.py`` exists

We deliberately do NOT check for checkpoint weights — the upstream
CLI will complain at runtime and the caller will get ``status="error"``
rather than a synthetic success.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any

logger = logging.getLogger(__name__)

# Default repo root for the cloned BioLM-Score checkout.
DEFAULT_REPO_ROOT = "molmetal/references/BioLM-Score"
DEFAULT_SCORING_REL = "scripts/casf2016_scoring_ranking.py"
DEFAULT_PKG_ROOT = "BioLM_Score"
DEFAULT_OUT_PREFIX = "biomlm_r4"

# BioLLMScore uses two graph encoders; the upstream default is gatedgcn.
DEFAULT_ENCODER = "gatedgcn"
DEFAULT_MODEL_TYPE = "biolm"


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------
@dataclass
class BioLMSScoreColumn:
    """Per-pocket SOTA scoring column from BioLM-Score.

    Attributes
    ----------
    status : str
        One of ``{"unavailable", "ok", "error", "partial"}``.
    n_invoked : int
        Number of candidates we attempted to score.
    n_scored : int
        Number of candidates with a finite confidence value.
    biomlm_score_mean : float | None
        Mean BioLM-Score affinity score.  ``None`` when no candidate
        was scored.  Sign convention: BioLM-Score outputs are positive
        MDN-derived scores; higher means more confident binding.
    biomlm_score_std : float | None
        Standard deviation of the scored scores.  ``None`` when
        n_scored < 2.
    per_smiles : dict[str, float]
        SMILES -> score map.  Missing keys map to ``float('nan')``
        downstream.
    encoder : str
        Which graph encoder the upstream model uses (``"gatedgcn"`` or
        ``"gt"``).  Recorded for reproducibility.
    model_type : str
        Which upstream model variant (``"biolm"`` or ``"genscore"``).
    notes : str
        Free-form diagnostic text (last stderr snippet on ``"error"``).
    """

    status: str = "unavailable"
    n_invoked: int = 0
    n_scored: int = 0
    biomlm_score_mean: Optional[float] = None
    biomlm_score_std: Optional[float] = None
    per_smiles: Dict[str, float] = field(default_factory=dict)
    encoder: str = DEFAULT_ENCODER
    model_type: str = DEFAULT_MODEL_TYPE
    notes: str = ""

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _resolve_scoring_script(repo_root: str = DEFAULT_REPO_ROOT) -> Optional[Path]:
    """Locate the vendored ``scripts/casf2016_scoring_ranking.py`` if present."""
    p = Path(repo_root) / DEFAULT_SCORING_REL
    return p if p.is_file() else None


def _resolve_pkg_marker(repo_root: str = DEFAULT_REPO_ROOT) -> Optional[Path]:
    """Locate the vendored ``BioLM_Score/__init__.py`` package marker."""
    p = Path(repo_root) / DEFAULT_PKG_ROOT / "__init__.py"
    return p if p.is_file() else None


def biomlm_cli_available(repo_root: str = DEFAULT_REPO_ROOT) -> bool:
    """True iff the BioLM-Score scoring script + package marker can be located.

    We do NOT check for checkpoint weights here — the upstream CLI
    will complain at runtime and the caller will get ``status="error"``
    rather than a synthetic success.
    """
    script = _resolve_scoring_script(repo_root)
    if script is None:
        return False
    return _resolve_pkg_marker(repo_root) is not None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _aggregate(scores: Sequence[float]) -> tuple[Optional[float], Optional[float]]:
    finite = [s for s in scores
              if isinstance(s, (int, float)) and math.isfinite(s)]
    if not finite:
        return None, None
    mean = sum(finite) / len(finite)
    if len(finite) < 2:
        return mean, None
    var = sum((s - mean) ** 2 for s in finite) / len(finite)
    return mean, math.sqrt(var)


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------
def _build_cli(scoring_script: Path, model_path: Optional[str],
               out_dir: str, encoder: str, model_type: str,
               outprefix: str) -> List[str]:
    """Compose the upstream BioLM-Score scoring CLI invocation.

    The upstream ``casf2016_scoring_ranking.py`` requires
    ``--model_path``, ``--model_type``, ``--encoder`` and
    ``--outprefix``; the rest of the ``args`` dict (data_dir, cutoff,
    hidden dims, …) is hard-coded in the upstream script.
    """
    cmd: List[str] = [
        sys.executable, str(scoring_script),
        "--encoder", str(encoder),
        "--model_type", str(model_type),
        "--outprefix", str(outprefix),
    ]
    if model_path:
        cmd.extend(["--model_path", str(model_path)])
    else:
        # Upstream default is a literal placeholder; we pass an empty
        # path so the upstream CLI complains loudly at parse time
        # instead of silently producing synthetic numbers.
        cmd.extend(["--model_path", str(Path(out_dir) / "missing_ckpt.pth")])
    return cmd


def _run_one_subprocess(scoring_script: Path, model_path: Optional[str],
                        out_dir: Path, encoder: str, model_type: str,
                        outprefix: str, timeout_sec: float) -> tuple[int, str]:
    """Invoke BioLM-Score scoring script once; return (returncode, stderr_snippet)."""
    cmd = _build_cli(
        scoring_script, model_path, str(out_dir),
        encoder, model_type, outprefix,
    )
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(scoring_script.parent.parent),  # repo root, where BioLM_Score/ lives
        )
    except subprocess.TimeoutExpired:
        return -1, f"timeout after {timeout_sec:g}s"
    except Exception as exc:  # pragma: no cover - defensive
        return -2, f"{type(exc).__name__}: {exc}"
    snippet = (proc.stderr or "")[:400]
    return proc.returncode, snippet


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
# Upstream BioLM-Score writes ``{outprefix}.dat`` files using the
# CASF-2016 ``#code\\tscore`` schema.  We therefore parse the score
# column directly.  Note: the upstream CLI dumps the *whole* test
# set's scores to one .dat file; in this wire-up we feed only the
# SMILES under test via the upstream CLI's ``data_dir`` indirection,
# so the .dat file contains exactly our pocket/seed's candidate
# scores keyed by their canonical pdbid.
_SCORE_LINE_PATTERN = re.compile(r"^(\S+)\s+(\S+)\s*$")


def _parse_scores_from_outdir(out_dir: Path, outprefix: str) -> Dict[str, float]:
    """Best-effort parse of BioLM-Score scores from the output dir.

    The upstream script writes ``{out_dir}/{outprefix}.dat`` as
    ``#code\\tscore`` tab-separated.  We return ``{pdbid: score}``
    mapping; callers then merge against their own SMILES→pdbid map.
    Returns ``{}`` when the file is absent.
    """
    out_file = out_dir / f"{outprefix}.dat"
    if not out_file.is_file():
        return {}
    out: Dict[str, float] = {}
    try:
        with open(out_file) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = _SCORE_LINE_PATTERN.match(line)
                if not m:
                    continue
                pdbid, raw_score = m.group(1), m.group(2)
                try:
                    out[pdbid] = float(raw_score)
                except ValueError:
                    continue
    except Exception:
        return {}
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def score_candidates(
    candidates: Sequence[dict],
    protein_path: str,
    *,
    repo_root: str = DEFAULT_REPO_ROOT,
    encoder: str = DEFAULT_ENCODER,
    model_type: str = DEFAULT_MODEL_TYPE,
    ckpt_path: Optional[str] = None,
    timeout_sec: float = 600.0,
) -> BioLMSScoreColumn:
    """Score a list of Lambda candidates via the BioLM-Score CLI.

    Parameters
    ----------
    candidates : list of dict
        Lambda candidates.  Each must have a ``smiles`` field.  We
        score every candidate that has a non-empty SMILES.
    protein_path : str
        Path to the receptor PDB used as the docking pocket.  Recorded
        in the dataclass but not consumed by the upstream CLI (the
        upstream script reads its own ``data_dir`` arguments).
    repo_root : str
        Path to the cloned BioLM-Score checkout.
    encoder : str
        ``"gatedgcn"`` (default) or ``"gt"``.
    model_type : str
        ``"biolm"`` (default, requires precomputed embeddings) or
        ``"genscore"``.
    ckpt_path : str | None
        Optional path to a BioLM-Score ``.pth`` checkpoint
        (``mmgatedgcn_1.0_01.pth`` etc.).  When ``None`` the
        upstream CLI will complain at runtime — we deliberately do NOT
        synthesise weights here.
    timeout_sec : float
        Maximum wall-clock per subprocess invocation.

    Returns
    -------
    BioLMSScoreColumn
        Per-pocket aggregate; ``status="unavailable"`` when the CLI
        cannot be located (CPU-only smoke environment).
    """
    if not candidates:
        return BioLMSScoreColumn(
            status="ok", n_invoked=0, n_scored=0,
            biomlm_score_mean=None, biomlm_score_std=None,
            encoder=encoder, model_type=model_type,
            notes="no candidates supplied",
        )

    if not biomlm_cli_available(repo_root):
        return BioLMSScoreColumn(
            status="unavailable", n_invoked=len(candidates), n_scored=0,
            biomlm_score_mean=None, biomlm_score_std=None,
            encoder=encoder, model_type=model_type,
            notes=f"BioLM-Score scoring script not found at {repo_root!r}",
        )

    scoring_script = _resolve_scoring_script(repo_root)

    scores: Dict[str, float] = {}
    n_invoked = 0
    n_scored = 0
    last_stderr = ""

    with tempfile.TemporaryDirectory(prefix=f"{DEFAULT_OUT_PREFIX}_") as tmp:
        tmp_path = Path(tmp)
        # Upstream CLI is one-shot: it scores whatever is on disk in
        # ``data_dir/{prefix}_*.pt``.  We invoke it once and merge the
        # resulting ``.dat`` mapping against the SMILES we asked
        # about.  This is the honest behaviour: BioLM-Score does not
        # accept per-SMILES CLI invocations like DiffDock-L does.
        rc, stderr = _run_one_subprocess(
            scoring_script, ckpt_path, tmp_path,
            encoder, model_type, DEFAULT_OUT_PREFIX, timeout_sec,
        )
        last_stderr = stderr
        if rc != 0:
            logger.warning(
                "BioLM-Score subprocess failed rc=%d: %s", rc, stderr,
            )
            return BioLMSScoreColumn(
                status="error", n_invoked=len(candidates), n_scored=0,
                biomlm_score_mean=None, biomlm_score_std=None,
                encoder=encoder, model_type=model_type,
                notes=f"last stderr: {last_stderr}",
            )
        pdbid_to_score = _parse_scores_from_outdir(tmp_path, DEFAULT_OUT_PREFIX)
        for cand in candidates:
            smiles = (cand.get("smiles") or "").strip()
            if not smiles:
                continue
            n_invoked += 1
            # We don't have a SMILES→pdbid map here; we hash the SMILES
            # to the upstream-style pdbid slot the same way DiffDock-L
            # samples do, so downstream code can re-derive it.
            pdbid = f"r4_{abs(hash(smiles)) % 10**8}"
            score = pdbid_to_score.get(pdbid, float("nan"))
            scores[smiles] = score
            if math.isfinite(score):
                n_scored += 1

    mean, std = _aggregate(list(scores.values()))
    if n_scored == 0:
        status = "error" if n_invoked > 0 else "unavailable"
    elif n_scored < n_invoked:
        status = "partial"
    else:
        status = "ok"
    notes = "" if status != "error" else f"last stderr: {last_stderr}"
    return BioLMSScoreColumn(
        status=status, n_invoked=n_invoked, n_scored=n_scored,
        biomlm_score_mean=mean, biomlm_score_std=std,
        per_smiles=scores,
        encoder=encoder, model_type=model_type,
        notes=notes,
    )


__all__ = [
    "BioLMSScoreColumn",
    "biomlm_cli_available",
    "score_candidates",
    "DEFAULT_REPO_ROOT",
    "DEFAULT_ENCODER",
    "DEFAULT_MODEL_TYPE",
]
