"""Wire-up helper: FlowDock as a SOTA scoring column.

================================================================
What this is
================================================================
This module is the *pipeline-side* glue that turns a list of Lambda
candidates into per-candidate FlowDock confidence scores, suitable
for use as a column in the r4_c_full_sweep report.

It mirrors :mod:`molmetal_lam.sbdd_env.diffdock_sota_scoring` exactly
in shape so the two SOTA columns can coexist in the sweep report.

* a thin subprocess wrapper around ``molmetal/references/FlowDock/
  flowdock/sample.py`` (the upstream CLI the vendored repo ships);
* a graceful fallback when the vendored repo is unavailable, the
  checkpoint weights are missing, or torch is unavailable (CPU-only
  smoke test path);
* an aggregator that emits ``flowdock_score_mean`` and
  ``flowdock_score_std`` per pocket/seed record.

Honest framing
--------------
* FlowDock inference is GPU-bound; the upstream repo also depends on
  ``lightning``, ``hydra``, ``esm``, and ``torch_scatter``.  None of
  these are easy to build on ROCm 7.2 / RX 7800 XT (gfx1101) — the
  situation is even harder than DiffDock-L because the FlowDock
  ``flash_attn`` path also depends on NVIDIA-specific kernels.
* This module therefore ships with a CPU-only *smoke* path that
  records ``status="unavailable"`` whenever the vendored
  ``flowdock/sample.py`` is not importable.  We do NOT synthesise
  confidence numbers — the report records ``null``/``NaN`` for the
  FlowDock column whenever the binary / weights are missing.
* FlowDock is the **preferred** L-1 binding oracle over DiffDock-L
  on consumer GPUs because it ships a CPU-fallback ODE solver
  (40 ODE steps).  When DiffDock's ``torch_cluster`` /
  ``torch_scatter`` extensions are unavailable but FlowDock's
  inference code is, this column can still produce real numbers.
  That capability is a *future* integration point — for now the
  column is *reporting-only*, mirroring the DiffDock-L column.

Discovery order
---------------
:meth:`flowdock_cli_available` returns True iff:

1. ``molmetal/references/FlowDock/flowdock/sample.py`` exists
2. AND there is a ``flowdock/__init__.py`` package marker (so the
   vendored repo is at least package-shaped).

We deliberately do NOT check for checkpoint weights — the upstream
``sample.py`` will complain at runtime and the caller will get
``status="error"`` rather than a synthetic success.
"""

from __future__ import annotations

import logging
import math
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any

logger = logging.getLogger(__name__)

# Default repo root for the cloned FlowDock checkout.
DEFAULT_REPO_ROOT = "molmetal/references/FlowDock"
DEFAULT_SAMPLE_REL = "flowdock/sample.py"
DEFAULT_SAMPLE_PKG = "flowdock.sample"
DEFAULT_OUT_PREFIX = "flowdock_r4"

# Hydra uses dotlist-style CLI overrides.  We pass them as
# ``key=value`` so the user never has to write a YAML.
_FLOWDOCK_OVERRIDES = (
    "task_name=sample",
    "tags=[r4_flowdock]",
    "sampling_task=batched_structure_sampling",
)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------
@dataclass
class FlowDockScoreColumn:
    """Per-pocket SOTA scoring column from FlowDock.

    Attributes
    ----------
    status : str
        One of ``{"unavailable", "ok", "error", "partial"}``.
    n_invoked : int
        Number of candidates we attempted to score.
    n_scored : int
        Number of candidates with a finite confidence value.
    flowdock_score_mean : float | None
        Mean FlowDock confidence.  ``None`` when no candidate was
        scored.  Sign convention: FlowDock confidences are *logits* /
        negative values (lower = more confident) by analogy with
        DiffDock-L; we record them verbatim.
    flowdock_score_std : float | None
        Standard deviation of the scored confidences.  ``None`` when
        n_scored < 2.
    per_smiles : dict[str, float]
        SMILES -> confidence map.  Missing keys map to ``float('nan')``
        downstream.
    """

    status: str = "unavailable"
    n_invoked: int = 0
    n_scored: int = 0
    flowdock_score_mean: Optional[float] = None
    flowdock_score_std: Optional[float] = None
    per_smiles: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _resolve_sample_script(repo_root: str = DEFAULT_REPO_ROOT) -> Optional[Path]:
    """Locate the vendored ``flowdock/sample.py`` if present."""
    p = Path(repo_root) / DEFAULT_SAMPLE_REL
    return p if p.is_file() else None


def flowdock_cli_available(repo_root: str = DEFAULT_REPO_ROOT) -> bool:
    """True iff a FlowDock sample.py + flowdock package marker can be located.

    We do NOT check for checkpoint weights here — the upstream
    ``sample.py`` will complain at runtime and the caller will get
    ``status="error"`` rather than a synthetic success.
    """
    script = _resolve_sample_script(repo_root)
    if script is None:
        return False
    # The repo itself must look like a Python package.
    pkg_root = Path(repo_root) / "flowdock"
    return pkg_root.is_dir() and (pkg_root / "__init__.py").is_file()


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
def _build_cli(sample_script: Path, smiles: str, protein_path: str,
               out_dir: str, ckpt_path: Optional[str],
               n_samples: int, num_steps: int,
               device: str = "cpu") -> List[str]:
    """Compose the upstream Hydra CLI invocation as a dotlist.

    Upstream ``flowdock.sample`` is Hydra-driven so flags take the
    ``+key=value`` form.  We pin ``sampling_task`` to the batched
    structural sampler and let the user override ``n_samples`` /
    ``num_steps`` / ``device`` via keyword args.
    """
    cmd: List[str] = [
        sys.executable, str(sample_script),
        # Disable rich logging that confuses stdout parsing.
        "hydra.run.dir=" + str(Path(out_dir) / "hydra"),
        "hydra.job.chdir=False",
        "hydra.output_subdir=null",
        "extras.print_config=false",
        # Mandatory upstream arguments.
        f"+input_receptor={protein_path}",
        f"+input_ligand={smiles}",
        f"+out_path={out_dir}",
        f"+n_samples={int(n_samples)}",
        f"+num_steps={int(num_steps)}",
        f"+device={device}",
    ]
    for kv in _FLOWDOCK_OVERRIDES:
        cmd.append(f"+{kv}")
    if ckpt_path:
        cmd.append(f"+ckpt_path={ckpt_path}")
    return cmd


def _run_one_subprocess(sample_script: Path, smiles: str, protein_path: str,
                        out_dir: Path, n_samples: int, num_steps: int,
                        ckpt_path: Optional[str], timeout_sec: float) -> tuple[int, str]:
    """Invoke FlowDock sample.py once; return (returncode, stderr_snippet)."""
    cmd = _build_cli(
        sample_script, smiles, protein_path,
        str(out_dir), ckpt_path, n_samples, num_steps,
    )
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(sample_script.parent.parent),  # repo root, where flowdock/ lives
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
# Upstream FlowDock writes ``rank1_confidence*.sdf`` files using the
# SAME naming convention as DiffDock-L (FlowDock was forked from
# DiffDock-L and preserves the output schema).  We therefore reuse
# :meth:`DiffDockAdapter.parse_confidence_from_outdir` to avoid
# duplicating the regex.
_RANK_CONF_PATTERN = re.compile(r"rank(\d+)_confidence(-?\d+\.?\d*)")


def _parse_confidence_from_outdir(out_dir: Path, sample_id: str) -> float:
    """Best-effort parse of FlowDock rank1 confidence from the output dir.

    Returns ``float('nan')`` when the rank1 file is absent.  The
    helper intentionally reuses the DiffDock-L filename parser
    because the two repos share the same SDF naming convention.
    """
    try:
        from molmetal_lam.sbdd_env.diffdock_adapter import DiffDockAdapter
        return DiffDockAdapter.parse_confidence_from_outdir(
            str(out_dir), sample_id,
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
    n_samples: int = 4,
    num_steps: int = 40,
    ckpt_path: Optional[str] = None,
    device: str = "cpu",
    timeout_sec: float = 600.0,
) -> FlowDockScoreColumn:
    """Score a list of Lambda candidates via the FlowDock CLI.

    Parameters
    ----------
    candidates : list of dict
        Lambda candidates.  Each must have a ``smiles`` field.  We
        score every candidate that has a non-empty SMILES.
    protein_path : str
        Path to the receptor PDB used as the docking pocket.
    repo_root : str
        Path to the cloned FlowDock checkout.
    n_samples : int
        Number of reverse-flow samples per SMILES (matches the
        upstream ``n_samples`` flag).  Default 4 is the cheapest
        non-trivial value.
    num_steps : int
        Number of ODE steps for the SE(3) flow (upstream default 40).
    ckpt_path : str | None
        Optional path to a FlowDock checkpoint.  When ``None`` the
        upstream CLI will complain at runtime — we deliberately do NOT
        synthesise weights here.
    device : str
        ``"cpu"`` or ``"cuda"``.  Default ``"cpu"`` so the wire-up is
        usable in CI / smoke-test environments without a GPU.  Note:
        the *preferred* L-1 binding path is CPU because FlowDock has
        a CPU-fallback ODE solver; for the *scoring column* the
        cheaper CPU path is acceptable.
    timeout_sec : float
        Maximum wall-clock per SMILES invocation.

    Returns
    -------
    FlowDockScoreColumn
        Per-pocket aggregate; ``status="unavailable"`` when the CLI
        cannot be located (CPU-only smoke environment).
    """
    if not candidates:
        return FlowDockScoreColumn(
            status="ok", n_invoked=0, n_scored=0,
            flowdock_score_mean=None, flowdock_score_std=None,
            notes="no candidates supplied",
        )

    if not flowdock_cli_available(repo_root):
        return FlowDockScoreColumn(
            status="unavailable", n_invoked=len(candidates), n_scored=0,
            flowdock_score_mean=None, flowdock_score_std=None,
            notes=f"FlowDock sample.py not found at {repo_root!r}",
        )

    if not Path(protein_path).is_file():
        return FlowDockScoreColumn(
            status="error", n_invoked=len(candidates), n_scored=0,
            flowdock_score_mean=None, flowdock_score_std=None,
            notes=f"protein_path not found: {protein_path!r}",
        )

    sample_script = _resolve_sample_script(repo_root)

    confidences: Dict[str, float] = {}
    n_invoked = 0
    n_scored = 0
    last_stderr = ""

    for cand in candidates:
        smiles = (cand.get("smiles") or "").strip()
        if not smiles:
            continue
        n_invoked += 1
        with tempfile.TemporaryDirectory(prefix=f"{DEFAULT_OUT_PREFIX}_") as tmp:
            tmp_path = Path(tmp)
            sample_id = f"r4_{abs(hash(smiles)) % 10**8}"
            rc, stderr = _run_one_subprocess(
                sample_script, smiles, protein_path, tmp_path,
                n_samples, num_steps, ckpt_path, timeout_sec,
            )
            last_stderr = stderr
            if rc != 0:
                logger.warning(
                    "FlowDock subprocess failed rc=%d for %s: %s",
                    rc, smiles[:40], stderr,
                )
                confidences[smiles] = float("nan")
                continue
            conf = _parse_confidence_from_outdir(tmp_path, sample_id)
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
    return FlowDockScoreColumn(
        status=status, n_invoked=n_invoked, n_scored=n_scored,
        flowdock_score_mean=mean, flowdock_score_std=std,
        per_smiles=confidences, notes=notes,
    )


__all__ = [
    "FlowDockScoreColumn",
    "flowdock_cli_available",
    "score_candidates",
    "DEFAULT_REPO_ROOT",
]
