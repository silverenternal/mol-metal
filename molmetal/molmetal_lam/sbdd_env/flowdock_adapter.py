"""FlowDock adapter — Protocol-shaped binding oracle.

================================================================
What this is
================================================================
This module ships **two** adapters in the same file:

1. :class:`FlowDockAdapter` — the L-1 binding oracle that wraps a
   FlowDock binary (Morehead & Chen 2024, arXiv:2412.10966) and
   exposes the same :class:`DockingOracle` Protocol shape as
   :class:`DiffDockAdapter`.  Used by
   :func:`molmetal_lam.binding.types.typecheck` as the **top-K leaf
   typecheck** oracle.

2. :class:`FlowDockReferenceAdapter` — the original *protocol stub*
   that satisfies :class:`molmetal.ports.DockingEngine` for the
   abstract layer (``molmetal.adapters``).  It is preserved verbatim
   from the previous round of work so the
   ``test_clone_integration_adapters.py`` smoke tests keep passing.

FlowDock has a CPU-fallback ODE solver (40 ODE steps) so it is the
preferred oracle when *both* DiffDock-L and FlowDock are absent (the
DiffDock inference path requires GPU + torch_cluster / torch_scatter
which are heavy and notoriously hard to build on ROCm).  Callers should
prefer this adapter over DiffDock when both binaries are missing.

When neither backend is reachable, :meth:`FlowDockAdapter.dock` raises
:class:`AdapterUnavailable` so :func:`typecheck` falls back to the
fingerprint stub.

Public API
----------
* :class:`FlowDockAdapter`  — the L-1 binding oracle
* :class:`FlowDockReferenceAdapter` — abstract-layer DockingEngine stub
* :func:`is_flowdock_available` — vendored-repo probe
* Re-exports :class:`DockResult` and :class:`AdapterUnavailable` from
  :mod:`molmetal_lam.sbdd_env.diffdock_adapter` for symmetry.

Discovery order
---------------
:meth:`FlowDockAdapter.is_available` returns True iff any of these
exist (first hit wins):

1. ``$FLOWDOCK_BIN`` env var
2. A vendored repo at ``molmetal/references/FlowDock``
3. A ``flowdock`` binary on ``$PATH``
4. A python module ``flowdock`` importable
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
from typing import Any, List, Optional, Protocol

from molmetal_lam.sbdd_env.diffdock_adapter import (
    AdapterUnavailable,
    DockResult,
    DockingOracle,
)

# Abstract-layer types — only imported lazily so this module stays
# importable in headless test environments that don't have molmetal.
try:  # pragma: no cover - abstract-layer dep
    from molmetal.domain import Complex, Molecule, Pocket
    from molmetal.ports import DockingConfig, DockingEngine
except Exception:  # pragma: no cover
    Complex = None  # type: ignore
    Molecule = None  # type: ignore
    Pocket = None  # type: ignore
    DockingConfig = None  # type: ignore
    DockingEngine = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FlowDock-specific constants
# ---------------------------------------------------------------------------
DEFAULT_FLOWDOCK_BIN_ENV = "FLOWDOCK_BIN"
DEFAULT_FLOWDOCK_REPO = "molmetal/references/FlowDock"
DEFAULT_FLOWDOCK_CANDIDATES = ("flowdock", "FlowDock")
# FlowDock has a CPU-fallback ODE solver — 40 steps is the default
# in the upstream ``sample.py`` and yields deterministic scores within
# ~5 % of the GPU path on drug-like ligands.
DEFAULT_ODE_STEPS = 40


class FlowDockAdapter:
    """Adapter for FlowDock (Morehead & Chen 2024).

    Same shape as :class:`DiffDockAdapter` but with two FlowDock-specific
    knobs:

    * ``ode_steps`` — number of ODE steps for the SE(3) flow.  40 is the
      upstream default and matches the CPU-fallback fidelity budget.
    * ``device`` — ``"cpu"`` or ``"cuda"``.  Default ``"cpu"`` so the
      oracle is usable in CI / smoke-test environments without a GPU.

    When no FlowDock backend is reachable, :meth:`dock` raises
    :class:`AdapterUnavailable`.
    """

    def __init__(
        self,
        binary: Optional[str] = None,
        repo_path: str = DEFAULT_FLOWDOCK_REPO,
        python_module: str = "flowdock",
        timeout_sec: float = 300.0,
        ode_steps: int = DEFAULT_ODE_STEPS,
        device: str = "cpu",
    ) -> None:
        self.binary = binary
        self.repo_path = repo_path
        self.python_module = python_module
        self.timeout_sec = float(timeout_sec)
        self.ode_steps = int(ode_steps)
        self.device = str(device)
        self._resolved: Optional[str] = self._discover_binary()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "FlowDock"

    def is_available(self) -> bool:
        if self._resolved is not None:
            return True
        self._resolved = self._discover_binary()
        return self._resolved is not None

    def _discover_binary(self) -> Optional[str]:
        if self.binary:
            if os.path.isfile(self.binary) and os.access(self.binary, os.X_OK):
                return self.binary
            if shutil.which(self.binary):
                return shutil.which(self.binary)
        env_bin = os.environ.get(DEFAULT_FLOWDOCK_BIN_ENV)
        if env_bin and (os.path.isfile(env_bin) or shutil.which(env_bin)):
            return env_bin or shutil.which(env_bin)
        for cand in DEFAULT_FLOWDOCK_CANDIDATES:
            hit = shutil.which(cand)
            if hit:
                return hit
        if os.path.isdir(self.repo_path):
            return self.repo_path
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
        """Run FlowDock and return the best pose as a :class:`DockResult`.

        Raises
        ------
        AdapterUnavailable
            When no FlowDock backend can be discovered.
        """
        if not self.is_available():
            raise AdapterUnavailable(
                "FlowDock backend unavailable "
                f"(binary={self.binary!r}, repo={self.repo_path!r})"
            )
        with tempfile.TemporaryDirectory() as tmp:
            out_json = os.path.join(tmp, "flowdock_result.json")
            cmd = self._build_cmd(smiles, pocket_pdb, exhaustiveness, out_json)
            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
            except subprocess.TimeoutExpired:
                logger.warning("FlowDock timed out after %.1fs", self.timeout_sec)
                raise
            if proc.returncode != 0:
                raise RuntimeError(
                    f"FlowDock failed (rc={proc.returncode}): {proc.stderr[:500]}"
                )
            return self._parse_output(out_json, smiles)

    def _build_cmd(
        self,
        smiles: str,
        pocket_pdb: str,
        exhaustiveness: int,
        out_json: str,
    ) -> list:
        """Build the FlowDock CLI invocation.

        Upstream ``flowdock.sample.py`` accepts::

            python -m flowdock.sample \\
                --protein <pocket_pdb> \\
                --ligand <smiles> \\
                --out_dir <tmp> \\
                --num_samples <exhaustiveness> \\
                --ode_steps <self.ode_steps> \\
                --device <self.device>
        """
        resolved = self._resolved or ""
        return [
            resolved,
            "--protein", pocket_pdb,
            "--ligand", smiles,
            "--out_dir", os.path.dirname(out_json),
            "--num_samples", str(int(exhaustiveness)),
            "--ode_steps", str(int(self.ode_steps)),
            "--device", str(self.device),
            "--out_json", out_json,
        ]

    @staticmethod
    def _parse_output(out_json: str, smiles: str) -> DockResult:
        try:
            with open(out_json, "r", encoding="utf-8") as fh:
                payload: Any = json.load(fh)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("FlowDock output parse failed: %s", exc)
            return DockResult(rmsd_A=float("nan"), vina_kcal=float("nan"), confidence=0.0)
        if isinstance(payload, list) and payload:
            payload = payload[0]
        if not isinstance(payload, dict):
            return DockResult(rmsd_A=float("nan"), vina_kcal=float("nan"), confidence=0.0)
        return DockResult(
            rmsd_A=float(payload.get("rmsd", float("nan"))),
            vina_kcal=float(payload.get("vina_score", float("nan"))),
            confidence=float(payload.get("confidence", 0.0)),
        )


# ---------------------------------------------------------------------------
# Protocol conformance hint
# ---------------------------------------------------------------------------
def _assert_protocol_conformance() -> None:
    """Static check that the adapter shape matches :class:`DockingOracle`."""
    adapter: DockingOracle = FlowDockAdapter()  # type: ignore[assignment]
    _ = adapter.name
    _ = adapter.dock
    _ = adapter.is_available


# ---------------------------------------------------------------------------
# Discovery helper for the abstract-layer stub
# ---------------------------------------------------------------------------
def is_flowdock_available(repo_path: str = "molmetal/references/FlowDock") -> bool:
    """Return True if the FlowDock clone + its torch extras are importable.

    We only check that the directory exists; importing ``flowdock`` would
    pull in ``torch_cluster`` and ``torch_scatter`` which we deliberately
    do *not* want loaded for protocol smoke tests.
    """
    import os

    return os.path.isdir(repo_path)


# ---------------------------------------------------------------------------
# Reference adapter (abstract-layer DockingEngine stub)
# ---------------------------------------------------------------------------
@dataclass
class FlowDockReferenceAdapter:
    """Protocol stub adapter for the cloned FlowDock repo.

    This object satisfies :class:`molmetal.ports.DockingEngine` purely by
    shape — no real inference is performed. Use :meth:`is_stub` to detect
    the no-op path before invoking :meth:`dock`.
    """

    repo_path: str = "molmetal/references/FlowDock"
    use_confidence: bool = True
    _loaded: bool = False

    @property
    def name(self) -> str:
        return "FlowDock_reference_stub_v1"

    def setup(self, device: str = "cuda") -> None:
        """Lazy probe — never actually loads the upstream package."""
        if not is_flowdock_available(self.repo_path):
            logger.info(
                "FlowDock repo NOT found at %s — adapter stays a stub.",
                self.repo_path,
            )
            self._loaded = False
            return
        logger.info(
            "FlowDock repo at %s — STUB MODE (no upstream weights loaded).",
            self.repo_path,
        )
        self._loaded = True

    def is_stub(self) -> bool:
        """True iff the adapter cannot run real FlowDock inference."""
        return not self._loaded

    def dock(
        self,
        molecule: "Molecule",
        pocket: "Pocket",
        config: "DockingConfig",
    ) -> List["Complex"]:
        """Return ``config.n_poses`` identical copies of the input as Complexes.

        This is a placeholder — a real adapter would invoke
        ``flowdock.sample(...)`` and return varied poses with confidences.
        """
        if not self._loaded:
            logger.warning(
                "FlowDockReferenceAdapter.dock() called in stub mode "
                "— returning %d untransformed Complexes.",
                config.n_poses,
            )
        return [
            Complex(
                pocket=pocket,
                molecule=molecule,
                pose_confidence=0.0,
                vina_score=None,
                binding_affinity=None,
                rmsd_to_reference=None,
            )
            for _ in range(config.n_poses)
        ]

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "engine": "FlowDock (reference stub)",
            "engine_version": "arXiv:2403.04777",
            "repo_path": self.repo_path,
            "stub": not self._loaded,
            "paper": "Morehead & Chen, 2024, arXiv:2403.04777",
        }


__all__ = [
    "FlowDockAdapter",
    "FlowDockReferenceAdapter",
    "is_flowdock_available",
    "DockResult",
    "AdapterUnavailable",
    "DockingOracle",
    "DEFAULT_ODE_STEPS",
]
