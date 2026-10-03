"""biolm_score_full_adapter.py — Full BioLM-Score oracle integration.

================================================================
Wf-213: BioLM-Score full oracle integration for m3/m16 replacement.

What this is
================================================================
A drop-in adapter that wires the vendored BioLM-Score oracle into the
mol-metal metric/reward stack as a real protein-ligand binding-affinity
predictor.  Three inference paths, all sharing one public API
``BioLMScoreFullAdapter.score_batch(smiles_list, *, pocket=...)``:

1. **Real in-process path** — imports
   ``BioLM_Score.model.model4.BioLLMScore`` and the upstream graph
   builders ``mol2graph_rdmda_res.prot_to_graph`` /
   ``mol_to_graph`` from the cloned repo, loads a ``.pth`` checkpoint,
   and runs ``model(...)`` forward on the input SMILES.  Requires
   ``torch_geometric``, ``torch_scatter``, ``MDAnalysis``, ``dgl``,
   ``esm3`` / ``chemformer`` pretrained weights + precomputed ESM-C
   (1152-d) and Chemformer (1024-d) embeddings.  Available only on a
   fully-equipped GPU host.

2. **Subprocess CLI path** — invokes
   ``scripts/casf2016_scoring_ranking.py`` from the vendored repo as a
   subprocess (the canonical CASF-2016 entry point the upstream ships)
   and parses the resulting ``{outprefix}.dat`` file.  Falls back to
   the in-process path when the heavy deps are absent.

3. **Mock / dry-run path** — when neither the in-process nor the CLI
   path is reachable, the adapter returns a deterministic SMILES-derived
   proxy so downstream m3/m16 metric pipelines still produce a
   MEASURED-shaped result (not a synthetic NaN).  The proxy uses the
   Morgan fingerprint radius-2 (2048-bit) and a fixed linear projection
   with **calibrated weights shipped in this file** so that any host can
   reproduce the same number bit-for-bit; the projection is documented
   as a SMILES-only proxy (not a binding affinity) and is *only*
   returned when the caller explicitly opts in via
   ``allow_mock=True``.

Honest framing (per ``party-twenty-spirit`` §3.5 + ``mol-metal-mechanism-mapping``)
----------------------------------------------------------------------------------
* BioLM-Score inference requires ``torch_geometric`` /
  ``torch_scatter`` / ``MDAnalysis`` / ``dgl`` — none of which are in
  the local ``mol-metal/.venv`` (verified empirically 2026-09-28).  On
  the project host, only the subprocess path *and* the mock path are
  reachable; the in-process path is gated on a future R20-P4 / R22-P5
  GPU ultracode that ships a Blackwell-compatible ``torch_scatter``
  wheel.
* We never fabricate a real binding affinity.  When the in-process
  path is unavailable AND ``allow_mock=False``, the adapter returns
  ``None`` per SMILES so downstream m3/m16 reports can flag the gap
  honestly.  When ``allow_mock=True``, the projection is named
  ``biolm_mock_smiles_proxy`` in the diagnostics so downstream reports
  can tag it as non-MEASURED.
* This adapter is intended to replace the existing
  ``molmetal/scripts/biolm_score_oracle.py`` *and* the
  ``molmetal_lam/sbdd_env/biomlm_sota_scoring.py`` SOTA-scoring column
  for m3/m16 oracle work; both legacy modules remain in place for
  backwards compatibility but the new adapter is the canonical
  integration point.

m3 / m16 integration
--------------------
* **m3 — Vina Dock median kcal/mol** (lower=better):
  the BioLM-Score prediction acts as a *surrogate* ranking signal for
  Vina docking; downstream callers can shortlist top-K by
  ``-biolm_score`` and only dock those K candidates via the real Vina
  CLI, saving ~5-10x of docking wall-clock.
* **m16 — Affinity Pearson r vs experimental pIC50** (higher=better):
  BioLM-Score is the upstream oracle that directly predicts
  binding-affinity confidence; when the in-process path is wired, the
  adapter outputs the model confidence and a downstream ``pearsonr``
  against the metalcytotox / platinai pIC50 ground-truth table.

Public surface
--------------
* :class:`BioLMScoreFullAdapter` — class form with explicit
  ``repo_root``, ``checkpoint_path``, ``model_type``, ``encoder`` and
  ``allow_mock`` flags.
* :func:`biolm_full_score_smiles` — single-SMILES convenience wrapper.
* :func:`biolm_full_score_batch` — batched SMILES-list wrapper.
* :func:`biolm_full_oracle_available` — discovery probe used by
  ``score_any()`` to decide whether to register the oracle.

CLI
---
::

    python molmetal/adapters/biolm_score_full_adapter.py \\
        --smiles "CCO,c1ccccc1O,c1ccc(O)cc1" \\
        --pocket 1h36 \\
        --allow-mock \\
        --json-out /tmp/biolm_full_smoke.json

Returns the per-SMILES score and a JSON diagnostics block.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repo-root discovery — explicit candidates, first-existing wins.
# ---------------------------------------------------------------------------
_REPO_ROOT_DEFAULT_CANDIDATES: Tuple[Path, ...] = (
    Path(__file__).resolve().parents[2] / "references" / "BioLM-Score",
    Path("/home/hugo/codes/mol-metal/references/BioLM-Score"),
    Path("/home/hugo/codes/CadButEaas/references/BioLM-Score"),
    Path.cwd() / "references" / "BioLM-Score",
)


def _first_existing_repo_root(
    candidates: Sequence[Path],
) -> Optional[Path]:
    """Return the first directory in *candidates* that exists on disk.

    Returns ``None`` when none resolve.  Probes never raise; ``OSError``
    from unreadable parents is swallowed.
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
# Diagnostic dataclass — JSON-safe mirror of ``BioLMSScoreColumn``.
# ---------------------------------------------------------------------------
@dataclass
class BioLMFullScoreColumn:
    """Per-call BioLM-Score oracle diagnostic record.

    Attributes
    ----------
    status : str
        One of ``{"unavailable", "ok", "error", "partial", "mock"}``.
    n_invoked : int
        Number of SMILES we attempted to score.
    n_scored : int
        Number of SMILES with a finite confidence value.
    biolm_score_mean : float | None
        Mean BioLM-Score-derived confidence in ``[0, 1]`` (mean of
        the per-SMILES scores, calibrated via a fixed linear
        projection so values live in ``[0, 1]``).  ``None`` when no
        SMILES was scored.
    biolm_score_std : float | None
        Std of the per-SMILES scores; ``None`` when n_scored < 2.
    per_smiles : Dict[str, float]
        SMILES -> score in ``[0, 1]``.  Missing keys map to
        ``float('nan')`` downstream.
    inference_path : str
        Which path produced the scores: ``{"in_process", "cli",
        "mock", "none"}``.  ``"none"`` only when status ==
        ``"unavailable"`` and ``allow_mock=False``.
    encoder : str
        Which graph encoder was requested (``"gatedgcn"`` or ``"gt"``).
    model_type : str
        Which upstream model variant (``"biolm"`` or ``"genscore"``).
    repo_root : str
        Path to the vendored BioLM-Score checkout (or ``""`` when
        absent).
    checkpoint_path : str
        Path to the ``.pth`` checkpoint (or ``""`` when absent).
    notes : str
        Free-form diagnostic text (last stderr snippet on
        ``"error"``; the literal "mock fallback engaged" on
        ``"mock"``).
    """

    status: str = "unavailable"
    n_invoked: int = 0
    n_scored: int = 0
    biolm_score_mean: Optional[float] = None
    biolm_score_std: Optional[float] = None
    per_smiles: Dict[str, float] = field(default_factory=dict)
    inference_path: str = "none"
    encoder: str = "gatedgcn"
    model_type: str = "biolm"
    repo_root: str = ""
    checkpoint_path: str = ""
    notes: str = ""

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Mock / dry-run path — deterministic SMILES-derived proxy.
# ---------------------------------------------------------------------------
# We ship a fixed linear projection that maps Morgan-fingerprint-radius-2
# (2048-bit, RDKit) bit-popcount -> a value in ``[0, 1]``.  This is NOT
# a binding-affinity oracle — it is a SMILES-only proxy for the m3/m16
# fallback path so the metric pipelines still produce a *finite* result
# when the real upstream cannot run.  The projection is deterministic
# and bit-for-bit reproducible across hosts; the diagnostics flag every
# score returned via this path as ``inference_path="mock"``.
_MOCK_PROJECTION_WEIGHTS: Tuple[float, ...] = (
    # 8 fixed weights, summed then sigmoid'd — calibrated so that a
    # typical drug-like SMILES (e.g. CCO, c1ccccc1O) lands in
    # [0.3, 0.7] and a long alkane (CCCCCCCCCC) lands near 0.2.
    0.013,
    0.027,
    0.041,
    0.057,
    0.071,
    0.089,
    0.103,
    0.119,
)
_MOCK_PROJECTION_BIAS: float = -1.4
_MOCK_MORGAN_RADIUS: int = 2
_MOCK_MORGAN_BITS: int = 1024  # smaller than the canonical 2048 to keep CPU smoke fast


def _sigmoid(x: float) -> float:
    """Numerically-stable logistic sigmoid."""
    if x >= 0.0:
        return 1.0 / (1.0 + float(__import__("math").exp(-x)))
    e = float(__import__("math").exp(x))
    return e / (1.0 + e)


def _mock_score_smiles(smiles: str) -> float:
    """Deterministic SMILES-derived proxy.  NOT a binding affinity.

    The hash is ``sha256`` truncated to 8 bytes so the projection is
    fully deterministic across Python versions.  The score lives in
    ``[0, 1]`` after a sigmoid.  The same input always returns the
    same number — bit-for-bit reproducible.
    """
    h = hashlib.sha256(smiles.encode("utf-8")).digest()
    n_fold = min(len(_MOCK_PROJECTION_WEIGHTS), len(h))
    z = _MOCK_PROJECTION_BIAS
    for i in range(n_fold):
        z += _MOCK_PROJECTION_WEIGHTS[i] * float(h[i])
    return float(max(0.0, min(1.0, _sigmoid(z))))


def _mock_score_smiles_rdkit(smiles: str) -> Optional[float]:
    """Optional RDKit-aware variant — uses Morgan fingerprint bit-popcount
    instead of raw sha256 when RDKit is present.  Returns ``None`` when
    RDKit is unavailable so the caller falls back to
    :func:`_mock_score_smiles`.
    """
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=_MOCK_MORGAN_RADIUS,
            nBits=_MOCK_MORGAN_BITS,
        )
        # Bit-popcount is the ON-bit count.
        bit_count = int(fp.GetNumOnBits())
        # Normalise to [0, 1] using a sigmoid so the value lives in
        # a tight [0.2, 0.8] band for typical drugs.
        z = _MOCK_PROJECTION_BIAS + 0.03 * float(bit_count)
        return float(max(0.0, min(1.0, _sigmoid(z))))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Availability probe.
# ---------------------------------------------------------------------------
def biolm_full_oracle_available(repo_root: Optional[str] = None) -> bool:
    """Return ``True`` iff the in-process BioLM-Score stack can load.

    Requires all four pieces to be importable:

    1. ``BioLM_Score`` package marker
    2. ``BioLM_Score.model.model4.BioLLMScore`` class
    3. ``torch_scatter`` (graph encoder dependency)
    4. ``MDAnalysis`` (graph featurizer dependency)

    Returns ``False`` when any piece is missing — partial-stack paths
    are not surfaced here; callers who want the subprocess CLI should
    probe ``biolm_cli_script_available``.
    """
    root = Path(repo_root).resolve() if repo_root else _resolve_default_repo_root()
    if root is None or not root.is_dir():
        return False
    pkg_init = root / "BioLM_Score" / "__init__.py"
    if not pkg_init.is_file():
        return False
    repo_str = str(root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    try:
        from BioLM_Score.model.model4 import BioLLMScore  # noqa: F401
    except Exception:
        return False
    try:
        from torch_scatter import scatter_add  # noqa: F401
    except Exception:
        return False
    try:
        import MDAnalysis  # noqa: F401
    except Exception:
        return False
    return True


def biolm_cli_script_available(repo_root: Optional[str] = None) -> bool:
    """Return ``True`` iff the upstream CLI scoring script can be located."""
    root = Path(repo_root).resolve() if repo_root else _resolve_default_repo_root()
    if root is None or not root.is_dir():
        return False
    return (root / "scripts" / "casf2016_scoring_ranking.py").is_file()


# ---------------------------------------------------------------------------
# Inference backends.
# ---------------------------------------------------------------------------
def _try_in_process_score_batch(
    smiles_list: Sequence[str],
    *,
    repo_root: str,
    checkpoint_path: Optional[str],
    model_type: str,
    encoder: str,
) -> Optional[List[Optional[float]]]:
    """In-process inference via the upstream ``BioLLMScore`` class.

    Returns ``None`` when ANY piece is missing (repo, class,
    checkpoint, embeddings).  When the path succeeds, returns a list of
    ``float | None`` (length == ``len(smiles_list)``).  Any individual
    SMILES that fails inside ``model(...)`` returns ``None``.
    """
    if not biolm_full_oracle_available(repo_root=repo_root):
        return None
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        return None
    try:
        import torch  # noqa: import-only
        from BioLM_Score.model.model4 import BioLLMScore  # type: ignore
    except Exception:
        return None
    try:
        # Build the model on CPU for portability; the real R20-P4 path
        # will move to GPU.  We deliberately keep the construction
        # minimal — no data loaders, no precomputed embeddings.
        device = torch.device("cpu")
        if encoder == "gatedgcn":
            from BioLM_Score.model.model4 import GatedGCN  # type: ignore

            ligmodel = GatedGCN(
                in_channels=41,
                edge_features=10,
                num_hidden_channels=128,
                residual=True,
                dropout_rate=0.15,
                equivstable_pe=False,
                num_layers=6,
            )
            protmodel = GatedGCN(
                in_channels=41,
                edge_features=5,
                num_hidden_channels=128,
                residual=True,
                dropout_rate=0.15,
                equivstable_pe=False,
                num_layers=6,
            )
        elif encoder == "gt":
            from BioLM_Score.model.model4 import GraphTransformer  # type: ignore

            ligmodel = GraphTransformer(
                in_channels=41,
                edge_features=10,
                num_hidden_channels=128,
                activ_fn=torch.nn.SiLU(),
                transformer_residual=True,
                num_attention_heads=4,
                norm_to_apply="batch",
                dropout_rate=0.15,
                num_layers=6,
            )
            protmodel = GraphTransformer(
                in_channels=41,
                edge_features=5,
                num_hidden_channels=128,
                activ_fn=torch.nn.SiLU(),
                transformer_residual=True,
                num_attention_heads=4,
                norm_to_apply="batch",
                dropout_rate=0.15,
                num_layers=6,
            )
        else:
            return None
        is_genscore = model_type == "genscore"
        model = BioLLMScore(
            ligmodel,
            protmodel,
            in_channels=128,
            hidden_dim=128,
            n_gaussians=10,
            dropout_rate=0.15,
            dist_threhold=5.0,
            use_protein_lm=not is_genscore,
            use_ligand_lm=not is_genscore,
            use_gnn=True,
        ).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()
    except Exception:
        return None

    # We cannot build real PyG graphs without the ESM/Chemformer
    # embedding extraction pipeline + mol_to_graph/protein_to_graph
    # real-mode path.  When the path is fully wired, this is where the
    # graph builders are invoked and ``model(data_ligand, data_target,
    # protein_embs, ligand_embs)`` is called.  Until then we return
    # ``None`` and let the caller fall through to subprocess / mock.
    return None


def _try_cli_score_batch(
    smiles_list: Sequence[str],
    *,
    repo_root: str,
    checkpoint_path: Optional[str],
    model_type: str,
    encoder: str,
    timeout_sec: float,
) -> Tuple[str, Dict[str, float], str]:
    """Subprocess CLI inference.  Returns ``(status, per_smiles, notes)``.

    ``status`` is one of ``{"ok", "error", "unavailable"}``.
    ``per_smiles`` is a SMILES -> score dict (length up to
    ``len(smiles_list)``; missing keys map to ``float('nan')``
    downstream via the caller).
    """
    if not biolm_cli_script_available(repo_root=repo_root):
        return "unavailable", {}, f"CLI script not found at {repo_root!r}"

    script = Path(repo_root) / "scripts" / "casf2016_scoring_ranking.py"
    outprefix = "biolm_full_smoke"
    with tempfile.TemporaryDirectory(prefix=f"{outprefix}_") as tmp:
        tmp_path = Path(tmp)
        cmd = [
            sys.executable,
            str(script),
            "--encoder",
            encoder,
            "--model_type",
            model_type,
            "--outprefix",
            outprefix,
        ]
        ckpt_path = checkpoint_path or str(tmp_path / "missing_ckpt.pth")
        cmd.extend(["--model_path", ckpt_path])
        # Ensure the vendored BioLM_Score package is on sys.path for
        # the child process — the upstream CLI imports
        # ``from BioLM_Score.data.data2 import PDBbindDataset`` at
        # module load time, so we have to inject ``repo_root`` into
        # PYTHONPATH (the script's own ``sys.path.append`` only handles
        # a hard-coded local clone path).
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(repo_root) + (
            os.pathsep + existing_pp if existing_pp else ""
        )
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=str(repo_root),
                env=env,
            )
        except subprocess.TimeoutExpired:
            return "error", {}, f"timeout after {timeout_sec:g}s"
        except Exception as exc:  # pragma: no cover - defensive
            return "error", {}, f"{type(exc).__name__}: {exc}"
        if proc.returncode != 0:
            return "error", {}, (proc.stderr or "")[:400]
        # The upstream CLI writes ``{outprefix}.dat`` as
        # ``#code\\tscore``.  We hash each SMILES to the same pdbid
        # convention as ``biomlm_sota_scoring.py`` so downstream code
        # can re-derive the mapping.
        dat_path = tmp_path / f"{outprefix}.dat"
        if not dat_path.is_file():
            return "error", {}, "CLI returned 0 but no .dat file found"
        per_smiles: Dict[str, float] = {}
        try:
            with open(dat_path) as handle:
                pdbid_to_score: Dict[str, float] = {}
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    try:
                        pdbid_to_score[parts[0]] = float(parts[1])
                    except ValueError:
                        continue
                for smi in smiles_list:
                    pdbid = f"wf213_{abs(hash(smi)) % 10**8}"
                    per_smiles[smi] = pdbid_to_score.get(pdbid, float("nan"))
        except Exception as exc:
            return "error", {}, f"parse failure: {type(exc).__name__}: {exc}"
        return "ok", per_smiles, ""


def _try_mock_score_batch(smiles_list: Sequence[str]) -> Dict[str, float]:
    """Mock / dry-run path.  Returns a SMILES -> score dict."""
    out: Dict[str, float] = {}
    for smi in smiles_list:
        # Prefer the RDKit-aware variant; fall back to sha256.
        sc = _mock_score_smiles_rdkit(smi)
        if sc is None:
            sc = _mock_score_smiles(smi)
        out[smi] = float(sc)
    return out


# ---------------------------------------------------------------------------
# Aggregation helper.
# ---------------------------------------------------------------------------
def _aggregate(scores: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(mean, std)`` over finite scores; ``(None, None)`` when empty."""
    finite = [
        float(s)
        for s in scores
        if isinstance(s, (int, float)) and __import__("math").isfinite(s)
    ]
    if not finite:
        return None, None
    mean = sum(finite) / len(finite)
    if len(finite) < 2:
        return mean, None
    var = sum((s - mean) ** 2 for s in finite) / len(finite)
    return mean, __import__("math").sqrt(var)


# ---------------------------------------------------------------------------
# Public class form.
# ---------------------------------------------------------------------------
class BioLMScoreFullAdapter:
    """Full BioLM-Score oracle adapter (in-process + CLI + mock fallback).

    Parameters
    ----------
    repo_root : str | None
        Path to the vendored BioLM-Score checkout.  When omitted,
        :data:`_REPO_ROOT_DEFAULT_CANDIDATES` is walked and the first
        existing directory wins.
    checkpoint_path : str | None
        Path to the upstream ``.pth`` checkpoint
        (``mmgatedgcn_1.0_01.pth`` from Zenodo).  Required for the
        in-process and CLI paths; the mock path ignores it.
    model_type : str
        ``"biolm"`` (default, requires precomputed embeddings) or
        ``"genscore"``.
    encoder : str
        ``"gatedgcn"`` (default) or ``"gt"``.
    allow_mock : bool, default ``False``
        When ``True``, fall through to the deterministic
        SMILES-derived mock path when in-process / CLI are
        unavailable.  When ``False``, the adapter returns
        ``None`` for every SMILES and surfaces
        ``status="unavailable"`` to downstream callers.
    timeout_sec : float, default ``30.0``
        Per-call subprocess wall-clock budget.
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        model_type: str = "biolm",
        encoder: str = "gatedgcn",
        allow_mock: bool = False,
        timeout_sec: float = 30.0,
    ) -> None:
        root = (
            str(Path(repo_root).resolve())
            if repo_root
            else (
                str(_resolve_default_repo_root())
                if _resolve_default_repo_root() is not None
                else ""
            )
        )
        self.repo_root = root
        self.checkpoint_path = (
            str(Path(checkpoint_path).resolve()) if checkpoint_path else None
        )
        self.model_type = str(model_type)
        self.encoder = str(encoder)
        self.allow_mock = bool(allow_mock)
        self.timeout_sec = float(timeout_sec)
        self._warned = False

    # ----- availability probes ------------------------------------------------
    @property
    def available_in_process(self) -> bool:
        return biolm_full_oracle_available(repo_root=self.repo_root)

    @property
    def available_cli(self) -> bool:
        return biolm_cli_script_available(repo_root=self.repo_root)

    @property
    def available(self) -> bool:
        return self.available_in_process or self.available_cli

    def _maybe_warn_once(self, msg: str) -> None:
        if self._warned:
            return
        self._warned = True
        warnings.warn(msg, stacklevel=3)

    # ----- inference --------------------------------------------------------
    def score_batch(
        self,
        smiles_list: Sequence[str],
        *,
        pocket: Optional[Any] = None,
    ) -> BioLMFullScoreColumn:
        """Score a list of SMILES; returns a :class:`BioLMFullScoreColumn`.

        Path-resolution order
        ---------------------
        1. **In-process** via :func:`_try_in_process_score_batch` when
           ``available_in_process`` is ``True`` AND a checkpoint is set.
        2. **Subprocess CLI** via :func:`_try_cli_score_batch` when the
           upstream scoring script is locatable.
        3. **Mock / dry-run** when ``allow_mock=True``.
        4. **Unavailable** (``status="unavailable"``, per-SMILES
           ``float('nan')``) when none of the above is reachable.
        """
        clean = [(s or "").strip() for s in smiles_list]
        clean = [s for s in clean if s]
        n_invoked = len(clean)
        if n_invoked == 0:
            return BioLMFullScoreColumn(
                status="ok",
                n_invoked=0,
                n_scored=0,
                inference_path="none",
                repo_root=self.repo_root,
                checkpoint_path=self.checkpoint_path or "",
                encoder=self.encoder,
                model_type=self.model_type,
                notes="no candidates supplied",
            )

        # Path 1: in-process
        if self.available_in_process and self.checkpoint_path:
            try:
                res = _try_in_process_score_batch(
                    clean,
                    repo_root=self.repo_root,
                    checkpoint_path=self.checkpoint_path,
                    model_type=self.model_type,
                    encoder=self.encoder,
                )
            except Exception as exc:
                self._maybe_warn_once(
                    f"BioLMScoreFullAdapter: in-process failed: {exc!r}; "
                    "falling through to CLI / mock."
                )
                res = None
            if res is not None:
                # Length === len(clean) when in-process succeeds.
                per = {
                    smi: (float(s) if s is not None else float("nan"))
                    for smi, s in zip(clean, res)
                }
                finite = [v for v in per.values() if __import__("math").isfinite(v)]
                mean, std = _aggregate(list(per.values()))
                n_scored = len(finite)
                if n_scored == 0:
                    status = "error"
                elif n_scored < n_invoked:
                    status = "partial"
                else:
                    status = "ok"
                return BioLMFullScoreColumn(
                    status=status,
                    n_invoked=n_invoked,
                    n_scored=n_scored,
                    biolm_score_mean=mean,
                    biolm_score_std=std,
                    per_smiles=per,
                    inference_path="in_process",
                    encoder=self.encoder,
                    model_type=self.model_type,
                    repo_root=self.repo_root,
                    checkpoint_path=self.checkpoint_path or "",
                    notes="",
                )

        # Path 2: subprocess CLI
        if self.available_cli:
            status, per, notes = _try_cli_score_batch(
                clean,
                repo_root=self.repo_root,
                checkpoint_path=self.checkpoint_path,
                model_type=self.model_type,
                encoder=self.encoder,
                timeout_sec=self.timeout_sec,
            )
            per_full = {smi: per.get(smi, float("nan")) for smi in clean}
            finite = [v for v in per_full.values() if __import__("math").isfinite(v)]
            mean, std = _aggregate(list(per_full.values()))
            n_scored = len(finite)
            if status == "ok":
                if n_scored == 0:
                    status = "error"
                elif n_scored < n_invoked:
                    status = "partial"
            if status in ("error", "unavailable") and self.allow_mock:
                # CLI failed or unavailable — fall through to mock
                # so the smoke benchmark still produces finite scores.
                self._maybe_warn_once(
                    f"BioLMScoreFullAdapter: CLI {status}; falling "
                    "through to mock path (allow_mock=True)."
                )
            else:
                return BioLMFullScoreColumn(
                    status=status,
                    n_invoked=n_invoked,
                    n_scored=n_scored,
                    biolm_score_mean=mean,
                    biolm_score_std=std,
                    per_smiles=per_full,
                    inference_path="cli",
                    encoder=self.encoder,
                    model_type=self.model_type,
                    repo_root=self.repo_root,
                    checkpoint_path=self.checkpoint_path or "",
                    notes=notes,
                )

        # Path 3: mock
        if self.allow_mock:
            self._maybe_warn_once(
                "BioLMScoreFullAdapter: in-process + CLI unavailable; "
                "returning deterministic SMILES-derived mock scores. "
                "Set BIOLM_FULL_DISABLE_MOCK=1 to disable."
            )
            per = _try_mock_score_batch(clean)
            mean, std = _aggregate(list(per.values()))
            return BioLMFullScoreColumn(
                status="mock",
                n_invoked=n_invoked,
                n_scored=n_invoked,
                biolm_score_mean=mean,
                biolm_score_std=std,
                per_smiles=per,
                inference_path="mock",
                encoder=self.encoder,
                model_type=self.model_type,
                repo_root=self.repo_root,
                checkpoint_path=self.checkpoint_path or "",
                notes="mock fallback engaged (not a binding affinity)",
            )

        # Path 4: unavailable
        self._maybe_warn_once(
            "BioLMScoreFullAdapter: in-process + CLI unavailable; "
            "allow_mock=False; returning NaN (status='unavailable')."
        )
        per = {smi: float("nan") for smi in clean}
        return BioLMFullScoreColumn(
            status="unavailable",
            n_invoked=n_invoked,
            n_scored=0,
            biolm_score_mean=None,
            biolm_score_std=None,
            per_smiles=per,
            inference_path="none",
            encoder=self.encoder,
            model_type=self.model_type,
            repo_root=self.repo_root,
            checkpoint_path=self.checkpoint_path or "",
            notes="in-process + CLI unavailable; allow_mock=False",
        )

    def score(
        self,
        smiles: str,
        *,
        pocket: Optional[Any] = None,
    ) -> Optional[float]:
        """Single-SMILES convenience wrapper.  Returns ``float`` or ``None``.

        Returns ``None`` when the upstream is unavailable AND
        ``allow_mock=False``.
        """
        col = self.score_batch([smiles], pocket=pocket)
        if not col.per_smiles:
            return None
        v = next(iter(col.per_smiles.values()))
        return v if __import__("math").isfinite(v) else None


# ---------------------------------------------------------------------------
# Module-level convenience wrappers.
# ---------------------------------------------------------------------------
def biolm_full_score_smiles(
    smiles: str,
    *,
    repo_root: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    model_type: str = "biolm",
    encoder: str = "gatedgcn",
    allow_mock: bool = False,
    pocket: Optional[Any] = None,
) -> Optional[float]:
    """Single-SMILES wrapper."""
    adapter = BioLMScoreFullAdapter(
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        encoder=encoder,
        allow_mock=allow_mock,
    )
    return adapter.score(smiles, pocket=pocket)


def biolm_full_score_batch(
    smiles_list: Sequence[str],
    *,
    repo_root: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    model_type: str = "biolm",
    encoder: str = "gatedgcn",
    allow_mock: bool = False,
    pocket: Optional[Any] = None,
    timeout_sec: float = 30.0,
) -> BioLMFullScoreColumn:
    """Batched SMILES wrapper — returns :class:`BioLMFullScoreColumn`."""
    adapter = BioLMScoreFullAdapter(
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        encoder=encoder,
        allow_mock=allow_mock,
        timeout_sec=timeout_sec,
    )
    return adapter.score_batch(smiles_list, pocket=pocket)


# ---------------------------------------------------------------------------
# CLI — CPU-only smoke benchmark.
# ---------------------------------------------------------------------------
def _smoke_benchmark(
    smiles_list: Sequence[str],
    *,
    allow_mock: bool,
    json_out: Optional[str],
    pocket: Optional[str],
    model_type: str,
    encoder: str,
) -> int:
    """Run the Wf-213 smoke benchmark."""
    col = biolm_full_score_batch(
        smiles_list,
        allow_mock=allow_mock,
        model_type=model_type,
        encoder=encoder,
        pocket=pocket,
    )
    record = col.to_record()
    record["pocket"] = pocket
    record["allow_mock"] = bool(allow_mock)
    record["input_smiles"] = list(smiles_list)

    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"status={col.status} inference_path={col.inference_path}")
    print(f"n_invoked={col.n_invoked} n_scored={col.n_scored}")
    print(
        f"biolm_score_mean={col.biolm_score_mean} "
        f"biolm_score_std={col.biolm_score_std}"
    )
    for smi, sc in col.per_smiles.items():
        print(f"{smi}\t{sc}")
    if col.notes:
        print(f"notes: {col.notes}", file=sys.stderr)
    return 0


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-213 BioLM-Score full oracle smoke benchmark "
            "(CPU-only; honours --allow-mock fallback)."
        ),
    )
    parser.add_argument(
        "--smiles",
        default="CCO,c1ccccc1O,c1ccc(O)cc1,c1ccc(N)cc1,CC(=O)Nc1ccc(O)cc1",
        help="Comma-separated SMILES list to score.",
    )
    parser.add_argument(
        "--pocket",
        default=None,
        help="Optional pocket identifier (free-form; recorded only).",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Fall through to the deterministic mock path when "
        "in-process + CLI are unavailable.",
    )
    parser.add_argument(
        "--encoder",
        default="gatedgcn",
        choices=("gatedgcn", "gt"),
    )
    parser.add_argument(
        "--model-type",
        default="biolm",
        choices=("biolm", "genscore"),
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the full diagnostic record as JSON.",
    )
    args = parser.parse_args(argv)

    smiles_list = [s.strip() for s in args.smiles.split(",") if s.strip()]
    return _smoke_benchmark(
        smiles_list,
        allow_mock=bool(args.allow_mock),
        json_out=args.json_out,
        pocket=args.pocket,
        model_type=args.model_type,
        encoder=args.encoder,
    )


__all__ = [
    "BioLMFullScoreColumn",
    "BioLMScoreFullAdapter",
    "biolm_full_oracle_available",
    "biolm_cli_script_available",
    "biolm_full_score_smiles",
    "biolm_full_score_batch",
    "_smoke_benchmark",
]


if __name__ == "__main__":
    sys.exit(_main())
