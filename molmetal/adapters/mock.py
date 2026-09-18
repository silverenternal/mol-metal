"""Mock adapters for molmetal ports — used in unit tests and orchestration smoke tests.

These are *inline* mocks that do NOT import anything from
``molmetal.orchestration`` (intentional: the orchestration module may
depend on these mocks, so the mocks must stay importable without it).

Four mocks are provided:

* :class:`MockGenerator`   — implements :class:`MoleculeGenerator`
* :class:`MockDocker`      — implements :class:`DockingEngine`
* :class:`MockPredictor`   — implements :class:`PropertyPredictor`
* :class:`MockScorer`      — implements :class:`ScoringFunction`

All randomness uses a local ``torch.Generator`` so test runs are
deterministic when ``seed`` is passed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from molmetal.domain import Complex, Molecule, Pocket
from molmetal.ports import (
    DockingConfig,
    DockingEngine,
    GenerationConfig,
    MoleculeGenerator,
    PropertyPrediction,
    PropertyPredictor,
    ScoredCandidate,
    ScoringFunction,
)


__all__ = [
    "MockGenerator",
    "MockDocker",
    "MockPredictor",
    "MockScorer",
]


# ---------------------------------------------------------------------------
# Mock MoleculeGenerator
# ---------------------------------------------------------------------------
class MockGenerator(MoleculeGenerator):
    """Deterministic mock generator — produces ``n_samples`` random 8-atom mols.

    Each generated ``Molecule`` has:
      * coords   ~ N(0, 1)                          shape (8, 3)
      * atom_types drawn from {6, 7, 8}             shape (8,)
      * bonds    chain (0-1, 1-2, …, 6-7)          7 single bonds
      * bond_types = 1                             shape (7,)
      * formal_charges = 0                         shape (8,)
      * smiles = ""

    :meth:`train_step` returns ``1.0 / (1 + global_step)`` — a strictly
    decreasing sequence that the orchestrator can use to verify that
    "training" is happening.
    """

    @property
    def name(self) -> str:
        return "MockGenerator_v0"

    def __init__(self, seed: int = 42) -> None:
        self._seed = int(seed)
        self._rng = torch.Generator(device="cpu").manual_seed(self._seed)
        self._global_step = 0
        self._device = "cpu"

    # ------------------------------------------------------------------
    def setup(self, device: str = "cpu") -> None:
        # Re-seed on setup so successive calls are reproducible.
        self._rng = torch.Generator(device="cpu").manual_seed(self._seed)
        self._device = device

    # ------------------------------------------------------------------
    def generate(
        self,
        pocket: Pocket,
        config: GenerationConfig,
    ) -> List[Molecule]:
        n_samples = int(config.n_samples)
        n_atoms = 8
        # Per-call sub-seed so successive generate() calls differ.
        sub = torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item()

        # coords ~ N(0, 1)
        gen = torch.Generator(device="cpu").manual_seed(sub)
        coords = torch.randn(n_samples, n_atoms, 3, generator=gen)
        # sample atom_types from {6, 7, 8} = {C, N, O}
        pool = torch.tensor([6, 7, 8], dtype=torch.long)
        idx = torch.randint(0, 3, (n_samples, n_atoms), generator=gen)
        atom_types = pool[idx]
        # Build the chain of 7 single bonds: edges (0-1, 1-2, …, 6-7)
        # Stored undirected (both directions) like the real Molecule
        # convention used in ``rdkit_io._mol_to_molecule``.
        edge_pairs = torch.tensor([[i, i + 1] for i in range(n_atoms - 1)], dtype=torch.long)
        # Duplicate edges for symmetry
        bonds_single = torch.cat([edge_pairs, edge_pairs.flip(1)], dim=0)  # (14, 2)
        bonds = bonds_single.t().contiguous()  # (2, 14)
        bond_types = torch.ones(bonds.shape[1], dtype=torch.long)  # all single
        formal_charges = torch.zeros(n_atoms, dtype=torch.long)

        mols: List[Molecule] = []
        for i in range(n_samples):
            mols.append(
                Molecule(
                    coords=coords[i],
                    atom_types=atom_types[i],
                    bonds=bonds,
                    bond_types=bond_types,
                    formal_charges=formal_charges,
                    smiles="",
                )
            )
        return mols

    # ------------------------------------------------------------------
    def train_step(
        self,
        pocket: Pocket,
        mols: List[Molecule],
    ) -> float:
        # Synthetic strictly-decreasing loss: 1/(1+step).
        self._global_step += 1
        return 1.0 / (1.0 + self._global_step)

    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        return {
            "model": self.name,
            "type": "mock",
            "n_atoms_per_mol": 8,
            "atom_pool": [6, 7, 8],
            "description": (
                "Deterministic mock generator. Returns n_samples random 8-atom "
                "molecules with N(0,1) coords and atom types in {C,N,O}. "
                "train_step returns 1/(1+global_step)."
            ),
        }


# ---------------------------------------------------------------------------
# Mock DockingEngine
# ---------------------------------------------------------------------------
class MockDocker(DockingEngine):
    """Random-rotation+translation pose sampler with a constant-ish vina score.

    For each call to :meth:`dock`, ``n_poses`` independent random SE(3)
    transforms are applied to the molecule's coordinates.  The
    :class:`Complex` objects returned share the *same* molecule object
    (the Molecule dataclass is frozen, so we can't mutate it; the
    rotation is applied to a *copy*).
    """

    @property
    def name(self) -> str:
        return "MockDocker_v0"

    def __init__(self, seed: int = 42) -> None:
        self._seed = int(seed)
        self._rng = torch.Generator(device="cpu").manual_seed(self._seed)
        self._device = "cpu"

    # ------------------------------------------------------------------
    def setup(self, device: str = "cpu") -> None:
        self._rng = torch.Generator(device="cpu").manual_seed(self._seed)
        self._device = device

    # ------------------------------------------------------------------
    def dock(
        self,
        molecule: Molecule,
        pocket: Pocket,
        config: DockingConfig,
    ) -> List[Complex]:
        n_poses = int(config.n_poses)
        sub = torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item()
        gen = torch.Generator(device="cpu").manual_seed(sub)

        # Random rotation matrices via QR of N(0,1)
        a = torch.randn(n_poses, 3, 3, generator=gen)
        q, r = torch.linalg.qr(a)
        # Ensure proper rotation (det = +1): flip sign on rows where det < 0
        det = torch.linalg.det(q)
        sign = torch.sign(det)
        sign[sign == 0] = 1.0
        q = q * sign.view(-1, 1, 1)

        # Random translations N(0, 5)
        t = torch.randn(n_poses, 3, generator=gen) * 5.0

        coords = molecule.coords  # (N, 3)
        # (n_poses, N, 3) = (n_poses, 1, 3) + (n_poses, N, 3) @ (3, 3)^T
        rotated = coords.unsqueeze(0) @ q.transpose(1, 2) + t.unsqueeze(1)

        # vina_score = -5.0 + uniform(-1, 1) per pose
        vina = -5.0 + (torch.rand(n_poses, generator=gen) * 2.0 - 1.0)

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
                    pose_confidence=0.5,
                    vina_score=float(vina[i].item()),
                    binding_affinity=None,
                    rmsd_to_reference=None,
                )
            )
        return complexes

    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        return {
            "model": self.name,
            "type": "mock",
            "description": (
                "Random SE(3) pose sampler. Returns n_poses transformed copies "
                "of the input molecule with vina_score ~ Uniform(-6, -4)."
            ),
        }


# ---------------------------------------------------------------------------
# Mock PropertyPredictor
# ---------------------------------------------------------------------------
class MockPredictor(PropertyPredictor):
    """Predictor that uses RDKit for QED/logP and synthesises the rest."""

    @property
    def name(self) -> str:
        return "MockPredictor_v0"

    def __init__(self, seed: int = 42) -> None:
        self._seed = int(seed)
        self._rng = torch.Generator(device="cpu").manual_seed(self._seed)
        self._device = "cpu"

    # ------------------------------------------------------------------
    def setup(self, device: str = "cpu") -> None:
        self._rng = torch.Generator(device="cpu").manual_seed(self._seed)
        self._device = device

    # ------------------------------------------------------------------
    def predict(
        self,
        molecule: Molecule,
        complex: Optional[Complex] = None,
    ) -> PropertyPrediction:
        sub = torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item()
        gen = torch.Generator(device="cpu").manual_seed(sub)

        # QED: try RDKit via Molecule.to_rdkit(); fallback to 0.5
        qed = 0.5
        logp = 0.0
        mol_weight = 0.0
        tpsa = 0.0
        num_h_donors = 0
        num_h_acceptors = 0
        num_rotatable_bonds = 0

        if molecule.smiles:
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors

                mol = Chem.MolFromSmiles(molecule.smiles)
                if mol is not None:
                    qed = float(Descriptors.qed(mol))
                    logp = float(Crippen.MolLogP(mol))
                    mol_weight = float(Descriptors.MolWt(mol))
                    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
                    num_h_donors = int(Lipinski.NumHDonors(mol))
                    num_h_acceptors = int(Lipinski.NumHAcceptors(mol))
                    num_rotatable_bonds = int(Lipinski.NumRotatableBonds(mol))
            except Exception:
                # RDKit may not be installed, or SMILES invalid — fallback.
                qed = 0.5
                logp = 0.0
        # else: empty smiles → keep defaults above

        # sa_score: random in [0.3, 0.9]
        sa_score = float(0.3 + 0.6 * torch.rand(1, generator=gen).item())

        # binding_affinity_pic50: 5.0 + N(0, 1) * 0.5
        binding_pic50 = float(5.0 + 0.5 * torch.randn(1, generator=gen).item())

        return PropertyPrediction(
            qed=qed,
            sa_score=sa_score,
            logp=logp,
            mol_weight=mol_weight,
            tpsa=tpsa,
            num_h_donors=num_h_donors,
            num_h_acceptors=num_h_acceptors,
            num_rotatable_bonds=num_rotatable_bonds,
            binding_affinity_pic50=binding_pic50,
            metal_binding_score=None,
        )

    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        return {
            "model": self.name,
            "type": "mock",
            "description": (
                "Uses RDKit QED/logP when smiles is provided, else falls back "
                "to 0.5/0.0. sa_score ~ Uniform(0.3, 0.9), "
                "binding_affinity_pic50 ~ N(5.0, 0.5)."
            ),
        }


# ---------------------------------------------------------------------------
# Mock ScoringFunction (default-weights)
# ---------------------------------------------------------------------------
class MockScorer(ScoringFunction):
    """Weighted-sum scorer.

    positive weights:
        qed, sa_score, binding_affinity_pic50
    negative weight:
        vina_score   (more negative → better, so we subtract its value)

    The combined score is::

        combined = w_qed * qed
                 + w_sa  * sa_score
                 + w_pic50 * binding_affinity_pic50
                 - w_vina  * vina_score

    All weights default to 1.0; rank is 1-indexed and assigned after
    sorting by ``combined_score`` descending.
    """

    @property
    def name(self) -> str:
        return "MockScorer_v0"

    def __init__(
        self,
        w_qed: float = 1.0,
        w_sa: float = 1.0,
        w_pic50: float = 1.0,
        w_vina: float = 1.0,
    ) -> None:
        self.w_qed = float(w_qed)
        self.w_sa = float(w_sa)
        self.w_pic50 = float(w_pic50)
        self.w_vina = float(w_vina)

    # ------------------------------------------------------------------
    def setup(self) -> None:
        # Nothing to load — the scorer is purely arithmetic.
        return None

    # ------------------------------------------------------------------
    def score(
        self,
        candidates: List[Tuple[Molecule, Optional[Complex], PropertyPrediction]],
    ) -> List[ScoredCandidate]:
        scored: List[ScoredCandidate] = []
        for mol, cmpl, prop in candidates:
            vina = cmpl.vina_score if (cmpl is not None and cmpl.vina_score is not None) else 0.0
            binding_pic50 = (
                prop.binding_affinity_pic50
                if prop.binding_affinity_pic50 is not None
                else 0.0
            )
            combined = (
                self.w_qed * float(prop.qed)
                + self.w_sa * float(prop.sa_score)
                + self.w_pic50 * float(binding_pic50)
                - self.w_vina * float(vina)
            )
            scored.append(
                ScoredCandidate(
                    molecule=mol,
                    complex=cmpl,
                    property_pred=prop,
                    combined_score=combined,
                    rank=0,  # filled in after sort
                )
            )

        # Sort descending by combined_score; 1-indexed rank
        scored.sort(key=lambda c: c.combined_score, reverse=True)
        for i, c in enumerate(scored):
            scored[i] = ScoredCandidate(
                molecule=c.molecule,
                complex=c.complex,
                property_pred=c.property_pred,
                combined_score=c.combined_score,
                rank=i + 1,
            )
        return scored

    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        return {
            "model": self.name,
            "type": "mock",
            "weights": {
                "qed": self.w_qed,
                "sa_score": self.w_sa,
                "binding_affinity_pic50": self.w_pic50,
                "vina_score": -self.w_vina,  # negative in the formula
            },
            "description": (
                "Linear weighted sum: qed, sa_score, binding_affinity_pic50 "
                "(positive); vina_score (negative, more negative = better)."
            ),
        }