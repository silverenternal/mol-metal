"""DiffDock-L adapter — Protocol-shaped binding oracle.

================================================================
What this is
================================================================
Adapter that wraps a DiffDock-L binary (or, when absent, exposes the
*Protocol* shape used by :func:`molmetal_lam.binding.types.typecheck`).

* DiffDock-L (Corso et al. 2024, arXiv:2403.05784) is a diffusion-based
  docking model.  We invoke it through a subprocess wrapper because the
  upstream repo has its own CLI (``inference.py``) and a heavy
  PyTorch / PyG dependency graph that we *do not* want pulled into the
  molmetal MLC import path.
* When no DiffDock binary is discoverable, :meth:`DiffDockAdapter.dock`
  raises :class:`AdapterUnavailable` so the caller can transparently
  fall back to the fingerprint stub.  This is the only safe behaviour
  in CI environments (ROCm 7.2 + Triton 3.8 + ROCm-flavoured torch)
  where the DiffDock torch_cluster extensions refuse to build.

Public API
----------
* :class:`DockResult`     — ``(rmsd_A, vina_kcal, confidence)`` dataclass
* :class:`AdapterUnavailable` — raised when the underlying binary
  cannot be located
* :class:`DiffDockAdapter` — the adapter

Discovery order
---------------
:meth:`DiffDockAdapter.is_available` returns True iff any of these
exist (first hit wins):

1. ``$DIFFDOCK_BIN`` env var pointing at a CLI / python wrapper
2. A vendored repo at ``molmetal/references/DiffDock`` (only checked
   for *directory* presence — full inference is a Phase-2 task)
3. A ``diffdock`` or ``diffdock-l`` binary on ``$PATH``
4. A python module ``diffdock`` importable

If the upstream DiffDock is unavailable, :meth:`dock` raises
:class:`AdapterUnavailable`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses + exceptions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DockResult:
    """Single docking oracle output.

    Attributes
    ----------
    rmsd_A : float
        Root-mean-square deviation of the predicted pose vs. the
        reference (in Angstrom).  Lower is better.  ``float('nan')``
        when the adapter cannot compute RMSD (e.g. no reference pose).
    vina_kcal : float
        AutoDock Vina score (in kcal/mol, negative = favourable).
        Lower (more negative) is better.  ``float('nan')`` when not
        available.
    confidence : float
        DiffDock-L confidence in [0, 1].  Higher is better.
    """

    rmsd_A: float
    vina_kcal: float
    confidence: float


class AdapterUnavailable(RuntimeError):
    """Raised when a docking oracle backend cannot be located.

    Callers MUST catch this and fall back to a cheaper proxy (the
    existing fingerprint stub).  Never let an unavailable backend
    crash MCTS.
    """


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
class DockingOracle(Protocol):
    """Protocol signature used by the binding layer.

    The fingerprint stub and the DiffDock / FlowDock adapters all
    satisfy this shape so :func:`typecheck` can call whichever is
    available.
    """

    name: str

    def dock(
        self,
        smiles: str,
        pocket_pdb: str,
        exhaustiveness: int = 16,
    ) -> DockResult:
        """Dock a ligand SMILES into a pocket PDB.

        Parameters
        ----------
        smiles : str
            The candidate ligand (canonical SMILES preferred).
        pocket_pdb : str
            Path to a PDB file containing the binding pocket.
        exhaustiveness : int, default 16
            Search-space sampling budget.  DiffDock-L uses this to
            decide how many reverse-diffusion samples to draw.

        Returns
        -------
        DockResult
            ``(rmsd_A, vina_kcal, confidence)`` of the best pose.
        """
        ...

    def is_available(self) -> bool:
        """True iff the underlying backend is reachable."""
        ...


# ---------------------------------------------------------------------------
# DiffDock-L adapter
# ---------------------------------------------------------------------------
DEFAULT_DIFFDOCK_BIN_ENV = "DIFFDOCK_BIN"
DEFAULT_DIFFDOCK_REPO = "molmetal/references/DiffDock"
DEFAULT_DIFFDOCK_CANDIDATES = ("diffdock", "diffdock-l", "DiffDock-L")


class DiffDockAdapter:
    """Adapter for DiffDock-L (Corso et al. 2024).

    Discovers a DiffDock CLI / vendored repo and exposes a
    :class:`DockingOracle` shape.  When the binary is missing,
    :meth:`dock` raises :class:`AdapterUnavailable` so the caller can
    fall back to the fingerprint stub.

    Parameters
    ----------
    binary : str, optional
        Explicit path to the DiffDock CLI.  When omitted we search
        ``$DIFFDOCK_BIN`` first, then the default candidate names on
        ``$PATH``, then a vendored repo path.
    repo_path : str, optional
        Path to a vendored DiffDock checkout.  Used as a *directory*
        probe only — real inference is Phase-2.
    python_module : str, default ``"diffdock"``
        Python module name to attempt import as a final fallback.
    timeout_sec : float, default 300.0
        Maximum wall-clock per ``dock`` invocation.  DiffDock-L inference
        is GPU-bound (~10-30 s on A100) so 5 minutes is generous.
    """

    def __init__(
        self,
        binary: Optional[str] = None,
        repo_path: str = DEFAULT_DIFFDOCK_REPO,
        python_module: str = "diffdock",
        timeout_sec: float = 300.0,
    ) -> None:
        self.binary = binary
        self.repo_path = repo_path
        self.python_module = python_module
        self.timeout_sec = float(timeout_sec)
        self._resolved: Optional[str] = self._discover_binary()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "DiffDock-L"

    def is_available(self) -> bool:
        """True iff a DiffDock-L backend is reachable.

        We probe, in order:
        1. explicit ``binary`` argument
        2. ``$DIFFDOCK_BIN`` env var
        3. candidate names on ``$PATH``
        4. vendored repo directory
        5. python ``diffdock`` module importable
        """
        if self._resolved is not None:
            return True
        # Re-probe — discovery may have been lazy at __init__ time.
        self._resolved = self._discover_binary()
        return self._resolved is not None

    def _discover_binary(self) -> Optional[str]:
        # 1. explicit override
        if self.binary:
            if os.path.isfile(self.binary) and os.access(self.binary, os.X_OK):
                return self.binary
            if shutil.which(self.binary):
                return shutil.which(self.binary)
        # 2. env override
        env_bin = os.environ.get(DEFAULT_DIFFDOCK_BIN_ENV)
        if env_bin and (os.path.isfile(env_bin) or shutil.which(env_bin)):
            return env_bin or shutil.which(env_bin)
        # 3. PATH lookup
        for cand in DEFAULT_DIFFDOCK_CANDIDATES:
            hit = shutil.which(cand)
            if hit:
                return hit
        # 4. vendored repo — only the directory check; we never import
        if os.path.isdir(self.repo_path):
            return self.repo_path
        # 5. python module
        try:
            __import__(self.python_module)
            return self.python_module
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def dock(
        self,
        smiles: str,
        pocket_pdb: str,
        exhaustiveness: int = 16,
    ) -> DockResult:
        """Run DiffDock-L and return the best pose as a :class:`DockResult`.

        Raises
        ------
        AdapterUnavailable
            When no DiffDock backend can be discovered.
        subprocess.TimeoutExpired
            When the underlying CLI exceeds :attr:`timeout_sec`.
        """
        if not self.is_available():
            raise AdapterUnavailable(
                "DiffDock-L backend unavailable "
                f"(binary={self.binary!r}, repo={self.repo_path!r})"
            )
        # We support two execution modes:
        #   - "binary" : the discovered entry point is a CLI / python file
        #   - "module" : the discovered entry point is a python module name
        # For Phase-1 the binary mode is implemented; the module mode is
        # a defensive fallback that delegates to the same parsing path.
        with tempfile.TemporaryDirectory() as tmp:
            out_json = os.path.join(tmp, "diffdock_result.json")
            cmd = self._build_cmd(smiles, pocket_pdb, exhaustiveness, out_json)
            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                logger.warning("DiffDock-L timed out after %.1fs", self.timeout_sec)
                raise
            if proc.returncode != 0:
                raise RuntimeError(
                    f"DiffDock-L failed (rc={proc.returncode}): {proc.stderr[:500]}"
                )
            return self._parse_output(out_json, smiles)

    def _build_cmd(
        self,
        smiles: str,
        pocket_pdb: str,
        exhaustiveness: int,
        out_json: str,
    ) -> list:
        """Build the CLI invocation.

        The default invocation mirrors the upstream ``inference.py`` CLI::

            python -m diffdock.inference \\
                --protein_path <pocket_pdb> \\
                --ligand <smiles> \\
                --out_dir <tmp> \\
                --samples <exhaustiveness> \\
                --no_final_step_noise
        """
        # When the discovery result is a python module, we invoke it
        # via ``python -m``.  When it is a path, we invoke it directly.
        resolved = self._resolved or ""
        if resolved.endswith(".py") or os.path.isfile(resolved):
            return [
                resolved,
                "--protein_path", pocket_pdb,
                "--ligand", smiles,
                "--out_dir", os.path.dirname(out_json),
                "--samples", str(int(exhaustiveness)),
                "--out_json", out_json,
            ]
        # Default: treat as a CLI binary on PATH.
        return [
            resolved,
            "--protein_path", pocket_pdb,
            "--ligand", smiles,
            "--out_dir", os.path.dirname(out_json),
            "--samples", str(int(exhaustiveness)),
            "--out_json", out_json,
        ]

    @staticmethod
    def _parse_output(out_json: str, smiles: str) -> DockResult:
        """Parse the JSON written by the DiffDock-L CLI.

        Tolerates missing fields — defaults to ``float('nan')`` so the
        downstream :func:`typecheck` can degrade gracefully.
        """
        try:
            with open(out_json, "r", encoding="utf-8") as fh:
                payload: Any = json.load(fh)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("DiffDock-L output parse failed: %s", exc)
            return DockResult(rmsd_A=float("nan"), vina_kcal=float("nan"), confidence=0.0)
        # DiffDock-L writes a list of poses; the first is the best by
        # confidence.  Fall back to a flat dict if the schema differs.
        if isinstance(payload, list) and payload:
            payload = payload[0]
        if not isinstance(payload, dict):
            return DockResult(rmsd_A=float("nan"), vina_kcal=float("nan"), confidence=0.0)
        return DockResult(
            rmsd_A=float(payload.get("rmsd", float("nan"))),
            vina_kcal=float(payload.get("vina_score", float("nan"))),
            confidence=float(payload.get("confidence", 0.0)),
        )

    @staticmethod
    def parse_confidence_from_outdir(out_dir: str, complex_name: str) -> float:
        """Parse the highest-confidence value from a DiffDock-L output dir.

        DiffDock-L's ``inference.py`` writes SDF files of the form
        ``rank{N}_confidence{X.XX}.sdf``.  When confidence model output
        exists, the *first* (rank 1) confidence is the best by definition;
        we read it from the filename rather than reparsing the SDF.
        Returns ``float('nan')`` when no rank1 file exists.

        This helper is used by the r4 sweep pipeline to record DiffDock
        confidence as a SOTA scoring column without requiring a
        confidence JSON file.
        """
        out_path = Path(out_dir) / complex_name
        if not out_path.is_dir():
            return float("nan")
        rank1 = sorted(out_path.glob("rank1_confidence*.sdf"))
        if rank1:
            # Filename is e.g. "rank1_confidence-1.25.sdf"
            stem = rank1[0].stem  # "rank1_confidence-1.25"
            try:
                value = stem.split("confidence", 1)[1]
                return float(value)
            except (IndexError, ValueError):
                return float("nan")
        # No rank1 file — check for any rank file (no confidence)
        any_rank = sorted(out_path.glob("rank*.sdf"))
        if any_rank:
            return float("nan")
        return float("nan")


# ---------------------------------------------------------------------------
# Protocol conformance hint
# ---------------------------------------------------------------------------
def _assert_protocol_conformance() -> None:
    """Static check that the adapter shape matches :class:`DockingOracle`."""
    adapter: DockingOracle = DiffDockAdapter()  # type: ignore[assignment]
    _ = adapter.name
    _ = adapter.dock
    _ = adapter.is_available


__all__ = [
    "DockResult",
    "AdapterUnavailable",
    "DockingOracle",
    "DiffDockAdapter",
]
