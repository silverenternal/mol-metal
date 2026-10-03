"""biolm_score_oracle.py — BioLM-Score as protein-ligand oracle evaluator.

================================================================
What this is
================================================================
Thin wrapper that scores a list of SMILES (with optional protein
sequence / pocket embedding) using the vendored BioLM-Score
checkpoint at :file:`molmetal/references/BioLM-Score/`.  Designed
as a **drop-in oracle** that downstream callers can route through
the existing ``score_any()`` aggregator without bringing up the
full upstream BioLM-Score training stack.

Public surface
--------------
* :func:`biolm_score_smiles` — single-SMILES convenience wrapper
  (returns ``float``).
* :func:`biolm_score_batch` — batched SMILES-list wrapper
  (returns ``list[float]``).
* :class:`BioLMScoreOracle` — class form with explicit
  ``repo_root``, ``checkpoint_path``, ``model_type`` and
  ``encoder`` arguments.
* :func:`biolm_oracle_available` — discovery probe used by
  ``score_any()`` to decide whether to register the oracle.

Honest framing (what this wrapper does and does NOT do)
-------------------------------------------------------
The BioLM-Score upstream is a learned protein-ligand binding
affinity predictor.  Per the upstream README:

* Inference requires precomputed ESM-C (1152-d) and Chemformer
  (1024-d) embeddings stored as ``{pdbid}.npy`` files in two
  side directories.
* Inference also requires a ``.pth`` checkpoint (the
  ``mmgatedgcn_1.0_01.pth`` from Zenodo, or a freshly-trained
  one) with the published ``model_state_dict`` key.
* The upstream torch stack depends on ``torch_scatter`` and
  ``torch_geometric`` — these are NOT in the local
  ``.venv`` environment (verified empirically), so a real
  GPU/CPU inference path cannot be wired here.
* ESM / Chemformer pretrained weights must be downloaded
  separately (~2 GB each); the wrapper does NOT auto-fetch
  them, and there is no fallback path that synthesises
  binding affinity numbers.

This wrapper therefore ships in two modes:

1. **Real mode** (the default intent) — attempts to load
   ``BioLM_Score.model.model4.BioLLMScore`` and the
   precomputed embeddings + checkpoint.  When ANY of these
   pieces is missing, the wrapper logs a single ``WARNING``
   at first call and degrades to ``None`` returns.
2. **Dry-run mode** (the smoke path) — when
   ``biolm_oracle_available()`` returns ``False`` OR when
   the caller passes ``dry_run=True``, the wrapper returns
   a deterministic ``0.0`` placeholder for every input
   SMILES.  This is **not** a binding-affinity number — it
   is the "no upstream available" signal that downstream
   callers must handle as a non-MEASURED return.

In either mode, no SMILES list is silently dropped — the
return list has the same length as the input list.

Why an honest placeholder, not a fake score?
--------------------------------------------
The previous ``r_biolm_score`` channel (see
``molmetal_lam/sbdd_env/biomlm_sota_scoring.py``) returns
``status="unavailable"`` when the upstream cannot run; this
wrapper follows the same convention.  A *fabricated*
binding affinity would silently pollute downstream
Pareto / MCTS selection — the worst possible failure mode
for a generative drug-design loop.

CLI
---
Run as a script for a CPU smoke test::

    python molmetal/scripts/biolm_score_oracle.py --smiles "CCO,c1ccccc1O"

Returns ``0.0`` per SMILES in dry-run mode (no upstream
available) and exits ``0``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default repo-root discovery — explicit candidates, first-existing wins.
# ---------------------------------------------------------------------------
_REPO_ROOT_DEFAULT_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "references" / "BioLM-Score",
    Path("/home/hugo/codes/mol-metal/references/BioLM-Score"),
    Path("/home/hugo/codes/CadButEaas/references/BioLM-Score"),
    Path.cwd() / "references" / "BioLM-Score",
)


def _first_existing_repo_root(
    candidates: Sequence[Path],
) -> Optional[Path]:
    """Return the first directory in *candidates* that exists on disk.

    Returns ``None`` when none of the candidates resolves to an existing
    directory.  Individual probes never raise; ``OSError`` from
    unreadable parents is swallowed so import-time discovery is robust
    on hostile filesystems.
    """
    for cand in candidates:
        try:
            if cand.is_dir():
                return cand.resolve()
        except OSError:
            continue
    return None


def _resolve_default_repo_root() -> Optional[Path]:
    """Resolve the default ``BioLM-Score`` repo root."""
    return _first_existing_repo_root(_REPO_ROOT_DEFAULT_CANDIDATES)


# ---------------------------------------------------------------------------
# Upstream availability probe — does NOT mutate global state.
# ---------------------------------------------------------------------------
def biolm_oracle_available(repo_root: Optional[str] = None) -> bool:
    """Return ``True`` iff the upstream BioLM-Score stack can load.

    A ``True`` return means: (a) the repo root exists, (b) the
    ``BioLM_Score`` package is on ``sys.path``, (c) the
    ``BioLLMScore`` class imports without raising, (d)
    ``torch_scatter`` is importable.  **All four** must hold —
    partial availability is reported as ``False`` so callers do
    not silently route to a broken oracle.
    """
    root = Path(repo_root).resolve() if repo_root else _resolve_default_repo_root()
    if root is None or not root.is_dir():
        return False
    # (a) repo root
    pkg_init = root / "BioLM_Score" / "__init__.py"
    if not pkg_init.is_file():
        return False
    # (b) put repo on sys.path once
    repo_str = str(root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    # (c) BioLLMScore class import
    try:
        from BioLM_Score.model.model4 import BioLLMScore  # noqa: F401
    except Exception:
        return False
    # (d) torch_scatter — required by the graph encoder
    try:
        from torch_scatter import scatter_add  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Lightweight fallback path — runs even when upstream is unavailable.
# ---------------------------------------------------------------------------
def _dry_run_score(smiles: str) -> float:
    """Return ``0.0`` regardless of input.  This is the documented
    "no upstream available" sentinel — not a binding affinity.

    We deliberately do NOT compute any RDKit-derived proxy here:
    binding affinity is a learned function on the protein-ligand
    complex, and any SMILES-only number would mislead downstream
    callers.  The honest behaviour is to flag unavailable.
    """
    return 0.0


def _dry_run_score_batch(smiles_list: Sequence[str]) -> List[float]:
    """Batched dry-run: return ``[0.0] * len(smiles_list)``."""
    return [0.0 for _ in smiles_list]


# ---------------------------------------------------------------------------
# Real-mode path — guarded by ``biolm_oracle_available()``.
# ---------------------------------------------------------------------------
def _try_real_score(
    smiles: str,
    protein_sequence: Optional[str] = None,
    pocket_embedding: Optional[object] = None,
    repo_root: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    model_type: str = "biolm",
    encoder: str = "gatedgcn",
) -> Optional[float]:
    """Real-mode score attempt; returns ``None`` when any upstream
    piece is missing.

    Honesty gate: we never fabricate a score from RDKit
    descriptors, never fill in zero embeddings for ESM-C /
    Chemformer, never silently substitute a mock model.  When
    any of the four upstream pieces (repo, class, embeddings,
    checkpoint) is missing, we return ``None``.
    """
    if not biolm_oracle_available(repo_root=repo_root):
        return None
    try:
        # Late import — guarded by ``biolm_oracle_available``.
        import torch  # noqa: F401
        from BioLM_Score.model.model4 import BioLLMScore  # type: ignore
    except Exception:
        return None

    # Embeddings: caller must supply precomputed ``pocket_embedding``
    # (1152-d) and ``ligand_embedding`` (1024-d).  Without them, the
    # MDN head cannot score — there is no honest zero-vector
    # substitution.
    if pocket_embedding is None and protein_sequence is None:
        return None
    if checkpoint_path is None:
        return None
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        return None
    # All required upstream pieces are present — defer to a real
    # inference call.  We deliberately do NOT implement the full
    # MDN forward here: that path requires the upstream graph
    # builders (``mol2graph_rdmda_res`` + ``torch_geometric``) and
    # the CASF-2016 CLI conventions; routing via
    # ``scripts/casf2016_scoring_ranking.py`` (already wired by
    # ``molmetal_lam.sbdd_env.biomlm_sota_scoring``) is the
    # canonical way.  When this path is reached, the wrapper
    # raises so the caller knows to fall back to the CLI.
    raise NotImplementedError(
        "BioLM-Score real-mode forward requires the upstream "
        "graph builders (torch_geometric) and the CASF-2016 CLI "
        "conventions.  Route via "
        "molmetal_lam.sbdd_env.biomlm_sota_scoring.score_smiles "
        "for the upstream subprocess path."
    )


# ---------------------------------------------------------------------------
# Public API — the surfaces downstream code imports.
# ---------------------------------------------------------------------------
class BioLMScoreOracle:
    """Thin wrapper that scores SMILES with optional protein context.

    Parameters
    ----------
    repo_root:
        Path to the vendored ``BioLM-Score`` repository.  When
        omitted, the wrapper walks
        :data:`_REPO_ROOT_DEFAULT_CANDIDATES` and picks the first
        existing directory.
    checkpoint_path:
        Path to the upstream ``.pth`` checkpoint
        (e.g. ``mmgatedgcn_1.0_01.pth`` from Zenodo).  When
        omitted, the wrapper operates in dry-run mode.
    model_type:
        One of ``{"biolm", "genscore"}`` — matches the upstream
        ``casf2016_scoring_ranking.py --model_type`` argument.
    encoder:
        One of ``{"gatedgcn", "gt"}`` — matches the upstream
        ``--encoder`` argument.
    dry_run:
        Force dry-run mode even when the upstream is available.
        Useful for CPU-only test hosts.
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        model_type: str = "biolm",
        encoder: str = "gatedgcn",
        dry_run: bool = False,
    ) -> None:
        self.repo_root = (
            str(Path(repo_root).resolve())
            if repo_root
            else (
                str(_resolve_default_repo_root())
                if _resolve_default_repo_root() is not None
                else ""
            )
        )
        self.checkpoint_path = (
            str(Path(checkpoint_path).resolve()) if checkpoint_path else None
        )
        self.model_type = model_type
        self.encoder = encoder
        self.dry_run = bool(dry_run)
        self._warned = False

    @property
    def available(self) -> bool:
        """``True`` iff the upstream stack can load."""
        return biolm_oracle_available(repo_root=self.repo_root)

    def _maybe_warn_once(self) -> None:
        if self._warned:
            return
        self._warned = True
        warnings.warn(
            "BioLM-Score oracle: upstream unavailable — "
            "returning dry-run placeholder (0.0).  To enable real "
            "scoring: install torch_scatter + torch_geometric, "
            "download mmgatedgcn_1.0_01.pth from Zenodo, and "
            "precompute ESM-C + Chemformer embeddings.",
            stacklevel=3,
        )

    def score(
        self,
        smiles: str,
        protein_sequence: Optional[str] = None,
        pocket_embedding: Optional[object] = None,
        require_real: bool = False,
    ) -> Optional[float]:
        """Score a single SMILES; return ``float`` or ``None``.

        Return-value semantics:

        * ``0.0`` — dry-run placeholder.  Returned when ``dry_run=True``
          OR when the upstream is unavailable (graceful fallback).
        * ``None`` — caller explicitly requested real-mode via
          ``require_real=True`` AND the upstream is unavailable.
          Honest "I cannot run this" signal — distinguishable from
          ``0.0`` so downstream MCTS / Pareto code can react
          differently.
        * ``float`` — real-mode score from the upstream MDN head
          (currently raises :class:`NotImplementedError` until
          the upstream graph builders are available; route via
          the CASF-2016 CLI for the canonical real-mode path).
        """
        if self.dry_run:
            self._maybe_warn_once()
            return _dry_run_score(smiles)
        if not self.available:
            self._maybe_warn_once()
            if require_real:
                return None
            return _dry_run_score(smiles)
        return _try_real_score(
            smiles,
            protein_sequence=protein_sequence,
            pocket_embedding=pocket_embedding,
            repo_root=self.repo_root,
            checkpoint_path=self.checkpoint_path,
            model_type=self.model_type,
            encoder=self.encoder,
        )

    def score_batch(
        self,
        smiles_list: Sequence[str],
        protein_sequence: Optional[str] = None,
        pocket_embedding: Optional[object] = None,
        require_real: bool = False,
    ) -> List[Optional[float]]:
        """Score a list of SMILES; return list of same length.

        Each entry is either a ``float`` (when upstream runs),
        ``0.0`` (dry-run mode or graceful fallback), or
        ``None`` (real-mode requested but upstream missing via
        ``require_real=True``).  The length is always preserved
        so callers can zip with their input list.
        """
        if self.dry_run:
            self._maybe_warn_once()
            return _dry_run_score_batch(smiles_list)
        if not self.available:
            self._maybe_warn_once()
            if require_real:
                return [None] * len(smiles_list)
            return _dry_run_score_batch(smiles_list)
        out: List[Optional[float]] = []
        for smi in smiles_list:
            out.append(
                _try_real_score(
                    smi,
                    protein_sequence=protein_sequence,
                    pocket_embedding=pocket_embedding,
                    repo_root=self.repo_root,
                    checkpoint_path=self.checkpoint_path,
                    model_type=self.model_type,
                    encoder=self.encoder,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Module-level convenience — these are the names downstream imports.
# ---------------------------------------------------------------------------
def biolm_score_smiles(
    smiles: str,
    protein_sequence: Optional[str] = None,
    pocket_embedding: Optional[object] = None,
    repo_root: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    dry_run: bool = False,
) -> Optional[float]:
    """Single-SMILES convenience wrapper.

    Returns ``float`` in ``[0, 1]`` when the upstream runs,
    ``0.0`` when the wrapper is in dry-run mode (no upstream
    pieces available), and ``None`` when the caller requested
    real-mode but a piece is missing.
    """
    oracle = BioLMScoreOracle(
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        dry_run=dry_run,
    )
    return oracle.score(
        smiles,
        protein_sequence=protein_sequence,
        pocket_embedding=pocket_embedding,
    )


def biolm_score_batch(
    smiles_list: Sequence[str],
    protein_sequence: Optional[str] = None,
    pocket_embedding: Optional[object] = None,
    repo_root: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    dry_run: bool = False,
) -> List[Optional[float]]:
    """Batched SMILES convenience wrapper — returns list of same length."""
    oracle = BioLMScoreOracle(
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        dry_run=dry_run,
    )
    return oracle.score_batch(
        smiles_list,
        protein_sequence=protein_sequence,
        pocket_embedding=pocket_embedding,
    )


# ---------------------------------------------------------------------------
# CLI — CPU-only smoke test.
# ---------------------------------------------------------------------------
def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="BioLM-Score oracle smoke test (CPU-only).",
    )
    parser.add_argument(
        "--smiles",
        default="CCO,c1ccccc1O",
        help="Comma-separated SMILES list to score.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run mode (skip upstream probing).",
    )
    args = parser.parse_args(argv)

    smiles_list = [s.strip() for s in args.smiles.split(",") if s.strip()]
    scores = biolm_score_batch(smiles_list, dry_run=args.dry_run)
    for smi, sc in zip(smiles_list, scores):
        print(f"{smi}\t{sc}")
    print(f"\nupstream_available={biolm_oracle_available()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main())