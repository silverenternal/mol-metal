"""DiffDock adapter — diffusion-based docking (Corso et al. ICLR 2023).

This is a Phase-1 **skeleton** adapter.  It satisfies the
:class:`DockingEngine` port from ``molmetal.ports`` so that the rest of the
framework (orchestration, tests, baselines) can be developed and benchmarked
against a uniform interface.

PHASE 1 (this file)
-------------------
* Implements the full ``DockingEngine`` Protocol.
* :meth:`setup` performs no I/O when ``checkpoint_path`` is ``None`` and
  records only ``device`` + a ``self._loaded = False`` flag.
* :meth:`dock` returns a STUB: ``config.n_poses`` copies of the molecule
  with small random SE(3) perturbations and a uniformly random
  ``vina_score`` in the realistic ``[-12, -4]`` kcal/mol range, plus a
  random ``pose_confidence`` in ``[0, 1]``.
* The actual DiffDock inference loop is **not** invoked — that is Phase 2.

PHASE 2 (TODO)
--------------
When a real ``checkpoint_path`` is provided, :meth:`setup` will:

* add the vendored ``molmetal/references/DiffDock`` repo to ``sys.path``
* import ``src.diffdock_model`` (or whichever module the upstream repo
  exposes) and instantiate the confidence model + score model
* load state dicts from ``checkpoint_path``
* move the models to ``self._device`` and ``eval()`` mode

:meth:`dock` will then:

1. Convert the :class:`Molecule` to an RDKit ``Mol`` with a 3D conformer
   (via :meth:`Molecule.to_rdkit`).
2. Run the DiffDock confidence model for ``config.n_samples`` reverse-
   diffusion steps to propose a ligand pose in the binding site.
3. Re-rank the samples with the DiffDock **confidence** model and keep
   the top-``config.n_poses``.
4. Wrap each pose as a :class:`Complex`, attaching ``pose_confidence``
   from the confidence model and an optional ``vina_score`` computed by
   an external Vina call (only if ``DiffDock --vina_full`` was selected).

The signature of the real call (illustrative):

    >>> import sys
    >>> sys.path.insert(0, str(repo_path))
    >>> from src.diffdock_model import ConfidenceModel  # upstream API
    >>> from src.protein import load_pocket_from_pocket
    >>> from inference import run_diffdock_inference
    >>> poses = run_diffdock_inference(
    ...     protein=pocket.to_rdkit(),
    ...     ligand=mol.to_rdkit(),
    ...     model=self._model,
    ...     n_samples=config.n_samples,
    ...     n_steps=config.n_steps,
    ...     device=self._device,
    ... )

Until that path is wired up, ``checkpoint_path=None`` triggers the STUB.

References
----------
Corso, G., Stärk, H., Jing, B., Barzilay, R., Jaakkola, T. (2023).
"DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking."
ICLR 2023.  https://arxiv.org/abs/2210.01776
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import torch

from molmetal.domain import Complex, Molecule, Pocket
from molmetal.ports import DockingConfig, DockingEngine


__all__ = ["DiffDockAdapter"]


# ---------------------------------------------------------------------------
# DiffDockAdapter
# ---------------------------------------------------------------------------
class DiffDockAdapter(DockingEngine):
    """Adapter wrapping DiffDock (Corso et al., ICLR 2023).

    Parameters
    ----------
    repo_path : str | Path | None
        Filesystem path to the vendored DiffDock repository
        (``molmetal/references/DiffDock`` by default).  The path is only
        added to ``sys.path`` in Phase 2; in Phase 1 it is recorded but
        never imported from.
    checkpoint_path : str | Path | None
        Path to a ``.pt`` file holding DiffDock model weights.  When
        ``None`` the adapter runs in STUB mode — :meth:`dock` returns
        random-but-realistic poses with no model loaded.  This lets unit
        tests exercise the interface without GPUs or downloads.

    Attributes
    ----------
    name : str
        ``"DiffDock_v1"`` — stable identifier for bookkeeping / logs.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "DiffDock_v1"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        repo_path: str | Path = "molmetal/references/DiffDock",
        checkpoint_path: Optional[str | Path] = None,
    ) -> None:
        self._repo_path = Path(repo_path)
        self._checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self._device: str = "cpu"
        # Phase 2 will hold the loaded torch ``nn.Module`` here.
        self._model: Optional[object] = None
        # True after a real model has been loaded.  In STUB mode this
        # stays False and :meth:`dock` falls back to random poses.
        self._loaded: bool = False
        # Local RNG so successive calls are reproducible from a seed.
        self._seed = 42
        self._rng = torch.Generator(device="cpu").manual_seed(self._seed)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, device: str = "cuda") -> None:
        """Configure device + (Phase 2) load model weights.

        Phase-1 behaviour: in STUB mode (``checkpoint_path is None``) we
        simply record ``device`` and re-seed the RNG; no I/O is performed
        and no tensors are allocated.  This keeps unit tests fast.

        Phase-2 behaviour (TODO): when ``checkpoint_path`` is provided,
        we will:

        1. Append ``self._repo_path`` to ``sys.path``.
        2. Import ``src.diffdock_model`` (or the equivalent upstream
           module) and instantiate the confidence + score models.
        3. Load state dicts via ``torch.load(self._checkpoint_path, ...)``.
        4. Move the models to ``device`` and call ``.eval()``.
        5. Set ``self._loaded = True`` so :meth:`dock` switches from the
           STUB path to the real inference path.

        Any ImportError / FileNotFoundError will propagate so that the
        orchestrator can decide whether to fall back to the STUB.
        """
        self._device = device
        # Re-seed the RNG so successive dock() calls are deterministic.
        self._rng = torch.Generator(device="cpu").manual_seed(self._seed)

        if self._checkpoint_path is None:
            # STUB mode — nothing to load.
            self._loaded = False
            return

        # ----------------------------------------------------------------
        # TODO(Phase 2): real model load.
        #
        #   import sys
        #   if str(self._repo_path) not in sys.path:
        #       sys.path.insert(0, str(self._repo_path))
        #
        #   # Import the upstream DiffDock modules.  Names below are
        #   # illustrative — verify against the actual vendored tree at
        #   # molmetal/references/DiffDock before implementing.
        #   from src.diffdock_model import ConfidenceModel, ScoreModel  # noqa
        #
        #   confidence = ConfidenceModel.load_from_checkpoint(
        #       self._checkpoint_path / "confidence.ckpt"
        #   )
        #   score = ScoreModel.load_from_checkpoint(
        #       self._checkpoint_path / "score.ckpt"
        #   )
        #
        #   self._model = {
        #       "confidence": confidence.to(device).eval(),
        #       "score": score.to(device).eval(),
        #   }
        #   self._loaded = True
        # ----------------------------------------------------------------
        raise NotImplementedError(
            "DiffDockAdapter: real model loading is Phase 2. "
            "Pass checkpoint_path=None to use STUB mode."
        )

    # ------------------------------------------------------------------
    # Dock
    # ------------------------------------------------------------------
    def dock(
        self,
        molecule: Molecule,
        pocket: Pocket,
        config: DockingConfig,
    ) -> List[Complex]:
        """Predict ``n_poses`` binding poses for ``molecule`` in ``pocket``.

        Phase-1 behaviour (STUB): produces ``n_poses`` independent random
        SE(3) perturbations of the molecule's coordinates and assigns a
        uniformly random ``vina_score`` in the realistic ``[-12, -4]``
        kcal/mol range plus a ``pose_confidence`` in ``[0, 1]``.  This is
        enough to exercise the orchestration code paths and to verify
        downstream property / scoring adapters.

        Phase-2 behaviour (TODO): invoke DiffDock's reverse-diffusion
        sampler with ``config.n_samples`` ligand poses, re-rank by the
        confidence model, and keep the top-``config.n_poses``.  See the
        module docstring for the expected call structure.
        """
        n_poses = int(config.n_poses)
        # Per-call sub-seed so successive dock() calls differ.
        sub = torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item()
        gen = torch.Generator(device="cpu").manual_seed(sub)

        # ----------------------------------------------------------------
        # STUB: random SO(3) rotation via QR of N(0,1).
        # ----------------------------------------------------------------
        a = torch.randn(n_poses, 3, 3, generator=gen)
        q, r = torch.linalg.qr(a)
        # Ensure det = +1 (proper rotation, not reflection).
        det = torch.linalg.det(q)
        sign = torch.sign(det)
        sign[sign == 0] = 1.0
        q = q * sign.view(-1, 1, 1)

        # Random translation N(0, 2) Å — small perturbations only.
        t = torch.randn(n_poses, 3, generator=gen) * 2.0

        coords = molecule.coords  # (N, 3)
        # (n_poses, N, 3) = (n_poses, N, 3) @ (3, 3)^T + (n_poses, 1, 3)
        rotated = coords.unsqueeze(0) @ q.transpose(1, 2) + t.unsqueeze(1)

        # Realistic vina_score range: U(-12, -4) kcal/mol.
        vina = -4.0 - 8.0 * torch.rand(n_poses, generator=gen)

        # pose_confidence: U(0, 1).
        confidence = torch.rand(n_poses, generator=gen)

        complexes: List[Complex] = []
        for i in range(n_poses):
            transformed = Molecule(
                coords=rotated[i],
                atom_types=molecule.atom_types,
                bonds=molecule.bonds,
                bond_types=molecule.bond_types,
                formal_charges=molecule.formal_charges,
                smiles=molecule.smiles,
                qed=molecule.qed,
                sa_score=molecule.sa_score,
                logp=molecule.logp,
            )
            complexes.append(
                Complex(
                    pocket=pocket,
                    molecule=transformed,
                    pose_confidence=float(confidence[i].item()),
                    vina_score=float(vina[i].item()),
                    binding_affinity=None,
                    rmsd_to_reference=None,
                )
            )
        return complexes

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        """Reproducibility + provenance info for benchmarking."""
        return {
            "model": self.name,
            "type": "diffdock",
            "stub": not self._loaded,
            "repo_path": str(self._repo_path),
            "checkpoint_path": (
                str(self._checkpoint_path)
                if self._checkpoint_path is not None
                else None
            ),
            "device": self._device,
            "paper": "Corso et al., ICLR 2023 (DiffDock)",
            "arxiv": "2210.01776",
            "description": (
                "DiffDockAdapter wraps Corso et al.'s diffusion-based "
                "docking model. Phase-1 ships a deterministic STUB that "
                "returns n_poses random SE(3)-perturbed copies of the "
                "input molecule with realistic vina_score / "
                "pose_confidence ranges; Phase 2 will invoke the real "
                "DiffDock sampler when checkpoint_path is provided."
            ),
        }
