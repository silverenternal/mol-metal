"""EquiBind adapter — SE(3)-equivariant docking (Stärk et al., NeurIPS 2022).

This is a Phase-1 **skeleton** adapter.  It satisfies the
:class:`DockingEngine` port from ``molmetal.ports`` so that the rest of the
framework (orchestration, tests, baselines) can be developed and
benchmarked against a uniform interface alongside DiffDock.

PHASE 1 (this file)
-------------------
* Implements the full ``DockingEngine`` Protocol.
* :meth:`setup` performs no I/O when no checkpoint is loaded and only
  records ``device`` + an RNG seed.
* :meth:`dock` returns a STUB: ``config.n_poses`` copies of the molecule
  with small random SE(3) perturbations and a uniformly random
  ``vina_score`` in the realistic ``[-12, -4]`` kcal/mol range, plus a
  random ``pose_confidence`` in ``[0, 1]``.  We deliberately bias the
  distributions slightly differently from DiffDock's STUB so that the
  two adapters produce distinguishable outputs in benchmarks.

PHASE 2 (TODO)
--------------
When a real ``checkpoint_path`` is provided, :meth:`setup` will:

* add the vendored ``molmetal/references/EquiBind`` repo to ``sys.path``
* import the upstream EquiBind module and instantiate the
  ``EquiBindModel`` (an SE(3)-equivariant graph neural network)
* load weights from ``checkpoint_path``
* move the model to ``self._device`` and ``eval()``

:meth:`dock` will then:

1. Convert the :class:`Molecule` to an RDKit ``Mol`` (3D conformer).
2. Convert the :class:`Pocket` to the upstream protein-graph
   representation (per-atom features + k-NN graph).
3. Call the EquiBind model **once** to get a single predicted binding
   pose.  EquiBind is *not* a sampler — it produces one pose directly
   from the input conformer, so the ``n_poses`` argument will either be
   coerced to 1 (default behaviour) or we will run ``n_poses`` forward
   passes with input conformer variants (TBD).
4. Wrap the predicted pose as a :class:`Complex`` with
   ``pose_confidence = 1.0`` (EquiBind does not natively produce a
   per-pose confidence; in production we would add a separate
   confidence head or post-hoc Vina re-scoring).

The signature of the real call (illustrative):

    >>> import sys
    >>> sys.path.insert(0, str(repo_path))
    >>> from equibind_model import EquiBindModel  # upstream API
    >>> model = EquiBindModel.load(checkpoint_path).to(device).eval()
    >>> pose = model.dock(
    ...     protein=pocket.to_rdkit(),
    ...     ligand=mol.to_rdkit(),
    ... )

References
----------
Stärk, H., Ganea, O.-E., Pattanaik, L., Barzilay, R., Jaakkola, T. (2022).
"EquiBind: Geometric Deep Learning for Drug Binding Structure
Prediction." NeurIPS 2022.  https://arxiv.org/abs/2202.05146
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch

from molmetal.domain import Complex, Molecule, Pocket
from molmetal.ports import DockingConfig, DockingEngine


__all__ = ["EquiBindAdapter"]


# ---------------------------------------------------------------------------
# EquiBindAdapter
# ---------------------------------------------------------------------------
class EquiBindAdapter(DockingEngine):
    """Adapter wrapping EquiBind (Stärk et al., NeurIPS 2022).

    Parameters
    ----------
    repo_path : str | Path
        Filesystem path to the vendored EquiBind repository
        (``molmetal/references/EquiBind`` by default).  The path is only
        added to ``sys.path`` in Phase 2; in Phase 1 it is recorded but
        never imported from.
    checkpoint_path : str | Path | None
        Path to a ``.pt`` file holding EquiBind model weights.  When
        ``None`` the adapter runs in STUB mode — :meth:`dock` returns
        random-but-realistic poses with no model loaded.  This lets unit
        tests exercise the interface without GPUs or downloads.

    Attributes
    ----------
    name : str
        ``"EquiBind_v1"`` — stable identifier for bookkeeping / logs.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "EquiBind_v1"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        repo_path: str | Path = "molmetal/references/EquiBind",
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
        self._seed = 123
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
        2. Import ``equibind_model`` (or the equivalent upstream module)
           and instantiate ``EquiBindModel``.
        3. Load state dicts via ``torch.load(self._checkpoint_path, ...)``.
        4. Move the model to ``device`` and call ``.eval()``.
        5. Set ``self._loaded = True`` so :meth:`dock` switches from the
           STUB path to the real inference path.
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
        #   from equibind_model import EquiBindModel  # upstream API
        #
        #   self._model = (
        #       EquiBindModel
        #       .load_from_checkpoint(str(self._checkpoint_path))
        #       .to(device)
        #       .eval()
        #   )
        #   self._loaded = True
        # ----------------------------------------------------------------
        raise NotImplementedError(
            "EquiBindAdapter: real model loading is Phase 2. "
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
        kcal/mol range plus a ``pose_confidence`` in ``[0, 1]``.  The
        STUB translation noise is biased slightly larger than DiffDock's
        (σ=3 Å vs σ=2 Å) so that benchmarks see two distinguishable
        baselines.

        Phase-2 behaviour (TODO): invoke the EquiBind model and wrap the
        predicted pose in a :class:`Complex`.  Because EquiBind produces a
        *single* pose, we will default to ``n_poses = 1`` unless the
        caller explicitly requested more (in which case we run multiple
        forward passes with conformer variants — see module docstring).
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

        # Random translation N(0, 3) Å — slightly wider than DiffDock's.
        t = torch.randn(n_poses, 3, generator=gen) * 3.0

        coords = molecule.coords  # (N, 3)
        # (n_poses, N, 3) = (n_poses, N, 3) @ (3, 3)^T + (n_poses, 1, 3)
        rotated = coords.unsqueeze(0) @ q.transpose(1, 2) + t.unsqueeze(1)

        # Realistic vina_score range: U(-11, -5) kcal/mol.
        vina = -5.0 - 6.0 * torch.rand(n_poses, generator=gen)

        # pose_confidence: U(0.4, 1.0) — EquiBind tends to be more
        # confident when its single-pose prediction succeeds.
        confidence = 0.4 + 0.6 * torch.rand(n_poses, generator=gen)

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
            "type": "equibind",
            "stub": not self._loaded,
            "repo_path": str(self._repo_path),
            "checkpoint_path": (
                str(self._checkpoint_path)
                if self._checkpoint_path is not None
                else None
            ),
            "device": self._device,
            "paper": "Stärk et al., NeurIPS 2022 (EquiBind)",
            "arxiv": "2202.05146",
            "description": (
                "EquiBindAdapter wraps Stärk et al.'s SE(3)-equivariant "
                "docking model. Phase-1 ships a deterministic STUB that "
                "returns n_poses random SE(3)-perturbed copies of the "
                "input molecule with realistic vina_score / "
                "pose_confidence ranges; Phase 2 will invoke the real "
                "EquiBind forward pass when checkpoint_path is provided."
            ),
        }
