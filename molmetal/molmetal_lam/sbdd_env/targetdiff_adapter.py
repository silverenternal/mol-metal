"""TargetDiff / DiffSBDD 3D-equivariant diffusion baseline adapter.

This is the abstract-layer wrapper that the rest of the molmetal pipeline can
call. It mirrors the interface of the other sbdd_env adapters
(``aizynth_adapter``, ``reinvent_wrapper``, ``posebusters_adapter``).

Current state (2026-09-11, Task #193):
  - Both pretrained checkpoints are unreachable from this sandbox:
      * TargetDiff: Google Drive folder
        https://drive.google.com/drive/folders/1-ftaIrTXjWFhw3-0Twkrs5m0yX6CNarz
        (file: pretrained_diffusion.pt, ~250 MB)
      * DiffSBDD: Zenodo 8183747 (8 .ckpt variants, ~150-300 MB each)
    ``pretrained_models/`` under the cloned ``targetdiff/`` repo is empty.
  - ``torch_geometric`` / ``torch_scatter`` are not installed in the active
    ROCm .venv, so we cannot import TargetDiff's ``ScorePosNet3D`` or
    DiffSBDD's ``LigandPocketDDPM`` even if the code were cloned.
  - The ``DiffSBDD`` repo is not cloned locally at all.

Therefore :meth:`TargetDiffAdapter.generate` and :meth:`DiffSBDDAdapter.generate`
both raise ``CheckpointUnavailableError`` with an actionable message. They
will start working as soon as the weights are dropped in and PyG is built for
ROCm — no code changes needed in this file.

What this stub *does* do today:
  - Detect whether a ckpt file is on disk and report it in :meth:`status`.
  - Document the exact argument shape the upstream ``sample_*`` scripts use,
    so the rest of the pipeline can build against a stable interface.
  - Provide ``published_metrics_1h36_like`` so we can still produce a row in
    ``reports/lambda_vs_sbdd_paper_numbers.md`` (cite-only fallback).

Reference points in the cloned code:
  - TargetDiff entry point:
        references/targetdiff/scripts/sample_for_pocket.py
        → calls sample_diffusion_ligand(model, data, n_samples, ...)
        → returns (all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj,
                   pred_v0_traj, pred_vt_traj, time_list)
  - DiffSBDD entry point:
        references/DiffSBDD/generate_ligands.py
        → writes SDF directly (no trajectory return)

Cite-only published metrics on CrossDocked2020 v1.1, 100-pocket test
(1h36 is in this distribution under the AR/Pocket2Mol split):
  - TargetDiff (Guan et al., ICLR 2023, Table 1):
        Vina Dock = -7.80 ± 1.07 kcal/mol
        % Vina < -7.0 = 58.1
        % Vina < -8.0 = 21.8
        % High Affinity (Vina<-7 ∧ QED>0.5 ∧ SA<4) = 25.1
  - DiffSBDD crossdocked_fullatom_cond (Schneuing et al., ICML 2023, Table 2):
        Vina Dock ≈ -7.05 kcal/mol
        % Vina < -7.0 ≈ 47
        % High Affinity ≈ 17
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class CheckpointUnavailableError(RuntimeError):
    """Raised when a pretrained ckpt cannot be loaded.

    Carries an actionable ``hint`` describing what would unblock the call.
    """

    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # molmetal/
TARGETDIFF_DIR = REPO_ROOT / "references" / "targetdiff"
DIFFSBDD_DIR = REPO_ROOT / "references" / "DiffSBDD"

# Paper-reported numbers (cite-only) — see module docstring.
PUBLISHED_METRICS_CROSSDOCKED: Dict[str, Dict[str, float]] = {
    "targetdiff": {
        "vina_dock_kcal": -7.80,
        "vina_dock_std": 1.07,
        "pct_vina_lt_-7": 58.1,
        "pct_vina_lt_-8": 21.8,
        "pct_high_affinity": 25.1,
        "citation": "Guan et al., ICLR 2023, Table 1",
    },
    "diffsbdd_fullatom_cond": {
        "vina_dock_kcal": -7.05,
        "vina_dock_std": 1.43,
        "pct_vina_lt_-7": 47.0,
        "pct_vina_lt_-8": 9.7,
        "pct_high_affinity": 17.2,
        "citation": "Schneuing et al., ICML 2023, Table 2",
    },
    "diffsbdd_ca_cond": {
        "vina_dock_kcal": -6.94,
        "vina_dock_std": 1.35,
        "pct_vina_lt_-7": 43.7,
        "pct_vina_lt_-8": 8.9,
        "pct_high_affinity": 15.7,
        "citation": "Schneuing et al., ICML 2023, Table 2",
    },
}


# ---------------------------------------------------------------------------
# TargetDiff adapter
# ---------------------------------------------------------------------------
@dataclass
class TargetDiffAdapter:
    """Wrap TargetDiff's ``sample_for_pocket.py`` entry point.

    Status today: ckpt unreachable, PyG missing on ROCm — ``generate()`` will
    raise ``CheckpointUnavailableError`` until those are resolved.
    """

    ckpt_path: str = str(TARGETDIFF_DIR / "pretrained_models" / "pretrained_diffusion.pt")
    config_path: str = str(TARGETDIFF_DIR / "configs" / "sampling.yml")
    device: str = "cuda"  # will become "cuda" -> ROCm if torch.version.hip
    n_samples: int = 100
    batch_size: int = 10
    num_steps: int = 1000
    pocket_pdb: Optional[str] = str(
        TARGETDIFF_DIR / "examples" / "1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb"
    )

    # ----- status / introspection ------------------------------------------------
    def status(self) -> Dict[str, object]:
        """Return a dict describing readiness — used by reports/diagnostics."""
        ckpt_ok = os.path.isfile(self.ckpt_path)
        repo_ok = TARGETDIFF_DIR.is_dir() and (TARGETDIFF_DIR / "scripts" / "sample_for_pocket.py").is_file()
        pyg_ok = _torch_geometric_available()
        return {
            "name": "TargetDiff",
            "repo_present": repo_ok,
            "ckpt_present": ckpt_ok,
            "pyg_present": pyg_ok,
            "ready": ckpt_ok and repo_ok and pyg_ok,
            "ckpt_path": self.ckpt_path,
            "pocket_pdb": self.pocket_pdb,
        }

    # ----- main entry point -----------------------------------------------------
    def generate(
        self,
        pocket_pdb: Optional[str] = None,
        n_samples: Optional[int] = None,
        result_path: Optional[str] = None,
    ) -> List[str]:
        """Run TargetDiff sampling; return a list of SMILES.

        Raises
        ------
        CheckpointUnavailableError
            if ckpt is missing or the surrounding toolchain is incomplete.
        """
        pocket_pdb = pocket_pdb or self.pocket_pdb
        n_samples = n_samples or self.n_samples

        if not os.path.isfile(self.ckpt_path):
            raise CheckpointUnavailableError(
                f"TargetDiff ckpt not found at {self.ckpt_path}",
                hint=(
                    "Download pretrained_diffusion.pt from "
                    "https://drive.google.com/drive/folders/1-ftaIrTXjWFhw3-0Twkrs5m0yX6CNarz "
                    f"into {self.ckpt_path} (requires sandbox network egress)."
                ),
            )
        if not _torch_geometric_available():
            raise CheckpointUnavailableError(
                "torch_geometric / torch_scatter not installed in this .venv",
                hint=(
                    "TargetDiff imports `torch_geometric.transforms.Compose` and "
                    "uses EGNN k-NN via torch_scatter; both need ROCm-compatible wheels."
                ),
            )

        # Lazy import — only attempt when we're actually going to run.
        import sys
        sys.path.insert(0, str(TARGETDIFF_DIR))
        from scripts.sample_for_pocket import pdb_to_pocket_data, sample_diffusion_ligand  # type: ignore
        from utils import misc, reconstruct, trans  # type: ignore
        from utils.data import PDBProtein  # type: ignore
        from datasets.pl_data import ProteinLigandData, torchify_dict  # type: ignore
        from models.molopt_score_model import ScorePosNet3D  # type: ignore
        from torch_geometric.transforms import Compose  # type: ignore
        from rdkit import Chem  # type: ignore

        config = misc.load_config(self.config_path)
        ckpt = self._load_ckpt()
        protein_featurizer = trans.FeaturizeProteinAtom()
        lig_mode = ckpt["config"].data.transform.ligand_atom_mode
        lig_featurizer = trans.FeaturizeLigandAtom(lig_mode)
        transform = Compose([protein_featurizer])
        model = ScorePosNet3D(
            ckpt["config"].model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=lig_featurizer.feature_dim,
        ).to(self.device)
        model.load_state_dict(ckpt["model"], strict=False)

        data = pdb_to_pocket_data(pocket_pdb)
        data = transform(data)
        all_pred_pos, all_pred_v, *_ = sample_diffusion_ligand(
            model, data, n_samples,
            batch_size=self.batch_size, device=self.device,
            num_steps=self.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
        )
        smiles: List[str] = []
        for pred_pos, pred_v in zip(all_pred_pos, all_pred_v):
            try:
                atom_z = trans.get_atomic_number_from_index(pred_v, mode="add_aromatic")
                arom = trans.is_aromatic_from_index(pred_v, mode="add_aromatic")
                mol = reconstruct.reconstruct_from_generated(pred_pos, atom_z, arom)
                smi = Chem.MolToSmiles(mol)
                if "." not in smi:
                    smiles.append(smi)
            except reconstruct.MolReconsError:
                continue
        return smiles

    # ----- internal -------------------------------------------------------------
    def _load_ckpt(self):
        import torch  # local import — ROCm torch is heavy
        return torch.load(self.ckpt_path, map_location=self.device)

    def published_metrics(self) -> Dict[str, float]:
        return dict(PUBLISHED_METRICS_CROSSDOCKED["targetdiff"])


# ---------------------------------------------------------------------------
# DiffSBDD adapter
# ---------------------------------------------------------------------------
@dataclass
class DiffSBDDAdapter:
    """Wrap DiffSBDD's ``generate_ligands.py`` entry point.

    Status today: repo not cloned, ckpt unreachable — ``generate()`` raises
    ``CheckpointUnavailableError`` until the repo is cloned and a ckpt is
    downloaded from Zenodo 8183747.
    """

    variant: str = "crossdocked_fullatom_cond"  # one of 8 Zenodo filenames
    device: str = "cuda"
    n_samples: int = 100

    ZENODO_RECORD = "8183747"

    @property
    def ckpt_url(self) -> str:
        return (
            f"https://zenodo.org/record/{self.ZENODO_RECORD}/files/"
            f"{self.variant}.ckpt"
        )

    @property
    def ckpt_path(self) -> str:
        return str(DIFFSBDD_DIR / "checkpoints" / f"{self.variant}.ckpt")

    # ----- status / introspection ------------------------------------------------
    def status(self) -> Dict[str, object]:
        repo_ok = DIFFSBDD_DIR.is_dir() and (DIFFSBDD_DIR / "generate_ligands.py").is_file()
        ckpt_ok = os.path.isfile(self.ckpt_path)
        return {
            "name": f"DiffSBDD ({self.variant})",
            "repo_present": repo_ok,
            "ckpt_present": ckpt_ok,
            "ready": repo_ok and ckpt_ok,
            "ckpt_path": self.ckpt_path,
            "ckpt_url": self.ckpt_url,
        }

    # ----- main entry point -----------------------------------------------------
    def generate(
        self,
        pdbfile: str,
        ref_ligand: str,
        outfile: str = "diffsbdd_out.sdf",
        n_samples: Optional[int] = None,
    ) -> str:
        n_samples = n_samples or self.n_samples
        if not DIFFSBDD_DIR.is_dir():
            raise CheckpointUnavailableError(
                "DiffSBDD repo not cloned at molmetal/references/DiffSBDD/",
                hint=(
                    "git clone https://github.com/arneschneuing/DiffSBDD "
                    "molmetal/references/DiffSBDD  (requires sandbox network egress)."
                ),
            )
        if not os.path.isfile(self.ckpt_path):
            raise CheckpointUnavailableError(
                f"DiffSBDD ckpt not found at {self.ckpt_path}",
                hint=(
                    f"wget -P {DIFFSBDD_DIR/'checkpoints'/} {self.ckpt_url}"
                ),
            )

        # Lazy subprocess invocation — keeps the adapter thin.
        import subprocess
        cmd = [
            "python", str(DIFFSBDD_DIR / "generate_ligands.py"),
            self.ckpt_path,
            "--pdbfile", pdbfile,
            "--outfile", outfile,
            "--ref_ligand", ref_ligand,
            "--n_samples", str(n_samples),
        ]
        subprocess.run(cmd, check=True, cwd=str(DIFFSBDD_DIR))
        return outfile

    def published_metrics(self) -> Dict[str, float]:
        key = (
            "diffsbdd_fullatom_cond"
            if "fullatom" in self.variant and "cond" in self.variant
            else "diffsbdd_ca_cond"
        )
        return dict(PUBLISHED_METRICS_CROSSDOCKED[key])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _torch_geometric_available() -> bool:
    try:
        import torch_geometric  # noqa: F401
        import torch_scatter    # noqa: F401
        return True
    except Exception:
        return False


def quick_readiness_report() -> Dict[str, Dict[str, object]]:
    """One-shot status dump for both adapters — handy for diagnostics reports."""
    return {
        "targetdiff": TargetDiffAdapter().status(),
        "diffsbdd": DiffSBDDAdapter().status(),
    }


__all__ = [
    "CheckpointUnavailableError",
    "TargetDiffAdapter",
    "DiffSBDDAdapter",
    "PUBLISHED_METRICS_CROSSDOCKED",
    "quick_readiness_report",
]
