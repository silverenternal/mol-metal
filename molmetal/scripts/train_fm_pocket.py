"""Train pocket-conditioned Flow Matching on CrossDocked2020 (MMP2/MMP9 subset).

Steps
-----
1. Load CrossDockedDataset with the dual-EOCD zip reader.
   Filter for MMP2 / MMP9 targets (from PDB header or ligand name).
   Fall back to 100 random pairs if no MMP pockets found.
2. Build a PocketConditionedVelocityField — EGNN that takes
   (ligand_coords, ligand_atom_types, pocket_coords, pocket_atom_types, t)
   and predicts the conditional flow velocity.
3. Train 30 epochs, batch_size=8 (heavy 3D pocket+ligand graphs).
4. Save checkpoint to molmetal/checkpoints/fm_pocket.pt.
5. Generate 100 candidates from the first pocket in the selected set.
6. Score with RDKitPropertyPredictor (QED, MolLogP, MolWt, TPSA) and
   the Morgan-FP → pIC50 regressor trained on MetalCytoToxDB (Ru subset
   as proxy for drug-likeness).
7. Write molmetal/reports/fm_pocket_eval.md.

Usage
-----
    source .venv/bin/activate
    cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.scripts.train_fm_pocket --epochs 30
"""

from __future__ import annotations

import argparse
import io
import os
import struct
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import zlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.flow_matching_lipman import (
    _import_fm_lib,
    LipmanFlowMatchingAdapter,
)
from molmetal.adapters.rdkit_predictor import RDKitPropertyPredictor
from molmetal.data.crossdocked import CrossDockedDataset
from molmetal.data.featurize import MorganFingerprinter
from molmetal.domain import Molecule, Pocket
from molmetal.models._scatter import scatter_sum_legacy as _scatter_sum
from molmetal.ports import GenerationConfig, PropertyPrediction
from molmetal.utils.device import get_device, verify_rocm_active
from triton_kernels import fused_silu_mlp as _fused_silu_mlp
from triton_kernels.config import triton_config
import torch.nn.functional as F


class _MaybeFusedSiLUMLP(nn.Module):
    """``Linear -> SiLU -> Linear`` wrapper gated by :data:`triton_config`.

    Drop-in replacement for ``nn.Sequential(Linear, SiLU, Linear)`` that
    preserves the parameter layout (``linear1.*`` / ``linear2.*``) so
    existing state-dicts load unchanged.  When :func:`triton_config.use_fused_mlp`
    says the fused kernel is the right choice for the input shape the
    forward routes through :func:`triton_kernels.fused_silu_mlp`;
    otherwise the pure-PyTorch ``nn.SiLU`` + two :class:`nn.Linear`
    chain is used and the output is bit-exact with the legacy
    ``nn.Sequential``.

    Local copy of :class:`molmetal.adapters.egnn_rocm._MaybeFusedSiLUMLP`
    so this script does not pick up the adapter's private symbol.  Kept
    identical to make the two call sites share behaviour.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if triton_config.use_fused_mlp(self.in_dim, self.hidden_dim):
            w1 = self.linear1.weight.t().contiguous()
            w2 = self.linear2.weight.t().contiguous()
            return _fused_silu_mlp(x, w1, self.linear1.bias, w2, self.linear2.bias)
        return F.silu(self.linear1(x)) @ self.linear2.weight.t() + self.linear2.bias

REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
CHECKPOINT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints"
CACHE_DIR = Path("/mnt/storage/data/molmetal/3d_cache")


# ---------------------------------------------------------------------------
# Pocket-conditioned velocity field
# ---------------------------------------------------------------------------


class EquivariantGraphConv(nn.Module):
    """SE(3)-equivariant graph conv (same as egcn_rocm.py)."""

    def __init__(self, in_node_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        # TritonConfig-gated fused path; the trailing SiLU / Sigmoid
        # activations are applied in :meth:`forward` (the fused kernel
        # only covers the ``Linear -> SiLU -> Linear`` triplet).
        self.mlp = _MaybeFusedSiLUMLP(
            in_dim=in_node_dim * 2 + 1,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
        )
        self.coord_mlp = _MaybeFusedSiLUMLP(
            in_dim=in_node_dim * 2 + 1,
            hidden_dim=hidden_dim,
            out_dim=1,
        )
        self._h_proj = nn.Linear(in_node_dim, hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_nodes = h.size(0)
        src, dst = edge_index
        diff = x[src] - x[dst]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
        dir_vec = diff / dist
        h_i = h[src]
        h_j = h[dst]
        mlp_in = torch.cat([h_i, h_j, dist], dim=-1)
        msg = F.silu(self.mlp(mlp_in))
        coord_scale = torch.sigmoid(self.coord_mlp(mlp_in))
        agg_h = _scatter_sum(msg, dst, dim=0, dim_size=n_nodes)
        h_out = self._h_proj(h) + agg_h
        coord_msg = dir_vec * coord_scale
        agg_x = _scatter_sum(coord_msg, dst, dim=0, dim_size=n_nodes)
        x_out = x + agg_x
        return h_out, x_out


class PocketConditionedVelocityField(nn.Module):
    """Velocity field v(x_t, t | pocket) using a shared EGNN for ligand + pocket.

    The ligand and pocket atoms are processed by separate EGNN stacks whose
    outputs are concatenated and projected to a per-atom velocity.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 3,
        max_atomic_number: int = 100,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_atomic_number = max_atomic_number

        # TritonConfig-gated fused path for ``Linear -> SiLU -> Linear``.
        self.time_mlp = _MaybeFusedSiLUMLP(
            in_dim=1,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
        )
        self.ligand_atom_embed = nn.Embedding(max_atomic_number, hidden_dim)
        self.pocket_atom_embed = nn.Embedding(max_atomic_number, hidden_dim)

        self.ligand_layers = nn.ModuleList(
            [EquivariantGraphConv(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.pocket_layers = nn.ModuleList(
            [EquivariantGraphConv(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        # Cross-attention: ligand nodes attend to pocket context
        # TritonConfig-gated fused path.
        self.cross_mlp = _MaybeFusedSiLUMLP(
            in_dim=hidden_dim * 2 + 1,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
        )

        # Final velocity head (zero-init)
        self.vel_head = nn.Linear(hidden_dim, 3, bias=False)
        nn.init.zeros_(self.vel_head.weight)

    def forward(
        self,
        ligand_coords: torch.Tensor,      # (B, N_lig, 3)
        ligand_atom_types: torch.Tensor,  # (B, N_lig)
        pocket_coords: torch.Tensor,       # (B, N_poc, 3)
        pocket_atom_types: torch.Tensor,   # (B, N_poc)
        edge_index_lig: torch.Tensor,      # (B, 2, E_lig)  — not used; per-sample edges built internally
        edge_index_poc: torch.Tensor,      # (B, 2, E_poc)  — not used
        t: torch.Tensor,                   # (B,) or (B, 1)
        ligand_mask: torch.Tensor,         # (B, N_lig)
    ) -> torch.Tensor:
        """Predict per-atom velocity for ligand atoms (B, N_lig, 3).

        Processes each sample in the batch independently since EquivariantGraphConv
        is a per-graph operation.
        """
        b, n_lig = ligand_coords.shape[:2]
        device = ligand_coords.device

        # Broadcast time
        if t.dim() == 0:
            t = t.view(1, 1).expand(b, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t)  # (B, H)

        # Build per-sample edge indices for ligand and pocket
        # Each sample may have different n_atoms, so we use the full n_lig/n_poc
        n_poc = pocket_coords.shape[1]

        # Process each sample independently
        v_list = []
        last_v_list = []

        for i in range(b):
            # Get sample data
            x_lig_i = ligand_coords[i]                    # (N_lig_i, 3)
            h_lig_i = self.ligand_atom_embed(ligand_atom_types[i:i+1].squeeze(0))  # (N_lig_i, H)
            t_emb_i = t_emb[i]                             # (H,)
            mask_i = ligand_mask[i]                        # (N_lig_i,)
            n_lig_i = int(mask_i.sum().item())

            if n_lig_i < 1:
                v_list.append(torch.zeros(n_lig, 3, device=device))
                continue

            x_lig_masked = x_lig_i[:n_lig_i]
            h_lig_masked = h_lig_i[:n_lig_i] + t_emb_i    # (N_lig_i, H)

            x_poc_i = pocket_coords[i]                     # (N_poc, 3)
            h_poc_i = self.pocket_atom_embed(pocket_atom_types[i:i+1].squeeze(0))  # (N_poc, H)
            n_poc_i = x_poc_i.shape[0]

            # Build edges for this sample
            if n_lig_i > 1:
                idx = torch.arange(n_lig_i, device=device)
                src = idx.view(1, n_lig_i, 1).expand(1, n_lig_i, n_lig_i)
                dst = idx.view(1, 1, n_lig_i).expand(1, n_lig_i, n_lig_i)
                mask_e = src != dst
                ei_lig = torch.cat([src[mask_e].view(1, -1), dst[mask_e].view(1, -1)], dim=0)
            else:
                ei_lig = torch.zeros(2, 0, dtype=torch.long, device=device)

            if n_poc_i > 1:
                idx = torch.arange(n_poc_i, device=device)
                src = idx.view(1, n_poc_i, 1).expand(1, n_poc_i, n_poc_i)
                dst = idx.view(1, 1, n_poc_i).expand(1, n_poc_i, n_poc_i)
                mask_e = src != dst
                ei_poc = torch.cat([src[mask_e].view(1, -1), dst[mask_e].view(1, -1)], dim=0)
            else:
                ei_poc = torch.zeros(2, 0, dtype=torch.long, device=device)

            last_v = torch.zeros(n_lig_i, 3, device=device)

            # Process ligand
            for layer in self.ligand_layers:
                h_lig_masked, last_v = layer(h_lig_masked, x_lig_masked, ei_lig)

            # Process pocket
            h_poc_masked = h_poc_i + t_emb_i
            for layer in self.pocket_layers:
                h_poc_masked, _ = layer(h_poc_masked, x_poc_i, ei_poc)

            # Pool pocket: (H,)
            h_poc_pool = h_poc_masked.sum(dim=0) / max(n_poc_i, 1)

            # Combine
            h_combined = h_lig_masked + h_poc_pool  # (N_lig_i, H)
            v_i = self.vel_head(h_combined) + last_v  # (N_lig_i, 3)

            # Pad back to max_n_lig
            v_padded = torch.zeros(n_lig, 3, device=device)
            v_padded[:n_lig_i] = v_i
            v_list.append(v_padded)

        return torch.stack(v_list, dim=0)  # (B, N_lig, 3)


# ---------------------------------------------------------------------------
# Pocket dataset wrapper
# ---------------------------------------------------------------------------

class PocketDataModule:
    """Wraps CrossDockedDataset and parses PDB files to extract pockets."""

    def __init__(
        self,
        archive_path: Path,
        extracted_dir: Path,
        rmsd_max: float = 1.0,
        heavy_atoms_range: tuple = (3, 50),
        split: str = "train",
        filter_mmp: bool = True,
        max_pairs: int = 500,
        seed: int = 42,
    ):
        self.archive_path = archive_path
        self.extracted_dir = extracted_dir
        self.rmsd_max = rmsd_max
        self.heavy_atoms_range = heavy_atoms_range
        self.split = split
        self.filter_mmp = filter_mmp
        self.max_pairs = max_pairs
        self.seed = seed

        self._dataset = None
        self._pairs = []
        self._pockets = []
        self._ligands = []

    def _try_native_zip_extract(self, archive_path: Path, member: str, out_path: Path) -> bool:
        import zipfile
        try:
            with zipfile.ZipFile(archive_path, "r") as z:
                with z.open(member) as src, open(out_path, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 22)
                        if not chunk:
                            break
                        dst.write(chunk)
            return out_path.exists() and out_path.stat().st_size > 0
        except Exception:
            return False

    def _stream_extract_member(self, archive_path: Path, target_name: str, out_path: Path) -> bool:
        from molmetal.data.crossdocked import (
            _parse_cd_entry, _read_central_directory, _read_first_eocd,
        )
        import zipfile
        cd = _read_central_directory(archive_path)
        match = None
        for entry in cd:
            if entry["fname"].rstrip("/").endswith(target_name):
                match = entry
                break
        if match is None:
            return False
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(archive_path, "rb") as f:
            f.seek(match["local_offset"])
            lh = f.read(30)
            if lh[:4] != b"PK\x03\x04":
                return False
            (
                ver, flags, method, mtime, mdate, crc,
                lh_csize, lh_usize, fnlen, exlen,
            ) = struct.unpack("<HHHHHIIIHH", lh[4:30])
            f.seek(match["local_offset"] + 30 + fnlen + exlen)
            csize = match["csize"]
            if csize == 0:
                return False
            if method == 8:
                z = zlib.decompressobj(-15)
                remaining = csize
                with open(out_path, "wb") as dst:
                    while remaining > 0:
                        chunk = f.read(min(1 << 22, remaining))
                        if not chunk:
                            break
                        out = z.decompress(chunk)
                        if out:
                            dst.write(out)
                        remaining -= len(chunk)
                    tail = z.flush()
                    if tail:
                        dst.write(tail)
                return True
            elif method == 0:
                with open(out_path, "wb") as dst:
                    remaining = csize
                    while remaining > 0:
                        chunk = f.read(min(1 << 22, remaining))
                        if not chunk:
                            break
                        dst.write(chunk)
                        remaining -= len(chunk)
                return True
        return False

    def _parse_pdb_header(self, pdb_path: str) -> str | None:
        """Extract protein name from PDB header (REMARK lines or HEADER)."""
        try:
            if not os.path.exists(pdb_path):
                return None
            with open(pdb_path, "r") as f:
                for line in f:
                    if line.startswith("HEADER"):
                        return line[10:60].strip().lower()
                    if line.startswith("COMPND"):
                        # Look for MMP in the line
                        if "MMP" in line.upper():
                            return "mmp"
            return None
        except Exception:
            return None

    def _load_pocket_from_pdb(self, pdb_path: str) -> Pocket | None:
        """Extract pocket atoms from a PDB file (CA atoms within radius)."""
        try:
            if not os.path.exists(pdb_path):
                return None
            coords_list = []
            atom_types_list = []
            residue_ids_list = []
            chain_ids_list = []
            center = torch.zeros(3)
            res_idx = 0

            with open(pdb_path, "r") as f:
                for line in f:
                    if not line.startswith("ATOM") and not line.startswith("HETATM"):
                        continue
                    # Only CA atoms for speed
                    if line[12:16].strip() != "CA":
                        continue
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    res_id = int(line[22:26].strip())
                    chain = line[21:22].strip()
                    elem = line[76:78].strip().capitalize()
                    if not elem:
                        elem = line[12:14].strip().capitalize()
                    atom_num = getattr(__import__("rdkit" if False else "rdkit.Chem", fromlist=["Atom"]), "Atom")(
                        0
                    ).GetAtomicNumber() if False else 0

                    # Use element to atomic number
                    elem_map = {"C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "FE": 26, "ZN": 30}
                    atomic_num = elem_map.get(elem, 6)

                    coords_list.append([x, y, z])
                    atom_types_list.append(atomic_num)
                    residue_ids_list.append(res_idx)
                    chain_ids_list.append(ord(chain) if chain else 0)
                    center[0] += x
                    center[1] += y
                    center[2] += z
                    res_idx += 1

            if len(coords_list) < 3:
                return None

            center = center / len(coords_list)
            coords = torch.tensor(coords_list, dtype=torch.float32)
            atom_types = torch.tensor(atom_types_list, dtype=torch.long)
            mask = torch.ones(len(coords_list), dtype=torch.bool)

            pdb_id = os.path.basename(pdb_path).replace(".pdb", "")
            return Pocket(
                pdb_id=pdb_id,
                coords=coords,
                atom_types=atom_types,
                residue_ids=torch.tensor(residue_ids_list, dtype=torch.long),
                chain_ids=torch.tensor(chain_ids_list, dtype=torch.long),
                mask=mask,
                center=center,
                radius=6.0,
            )
        except Exception as e:
            return None

    def _load_ligand_molecule(self, sdf_path: str) -> Molecule | None:
        """Load a ligand from SDF file into a Molecule."""
        try:
            from rdkit import Chem
            if not os.path.exists(sdf_path):
                return None
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = next(suppl, None)
            if mol is None:
                return None
            # Get first conformer
            conf = mol.GetConformer(0) if mol.GetNumConformers() > 0 else None
            if conf is None:
                return None
            coords = []
            atom_types = []
            for atom in mol.GetAtoms():
                atom_types.append(atom.GetAtomicNum())
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            bonds = []
            bond_types = []
            for bond in mol.GetBonds():
                bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                bond_types.append(bond.GetBondTypeAsDouble())
            if not bonds:
                bonds = [[0, 0]]
                bond_types = [0.0]
            # Compute canonical SMILES
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                smiles = ""
            return Molecule(
                coords=torch.tensor(coords, dtype=torch.float32),
                atom_types=torch.tensor(atom_types, dtype=torch.long),
                bonds=torch.tensor(bonds, dtype=torch.long).t().contiguous(),
                bond_types=torch.tensor(bond_types, dtype=torch.long),
                formal_charges=torch.zeros(len(atom_types), dtype=torch.long),
                smiles=smiles,
            )
        except Exception:
            return None

    def setup(self):
        """Load CrossDockedDataset and filter for MMP2/MMP9."""
        from molmetal.data.crossdocked import DEFAULT_SPLIT_NAME, DEFAULT_TARBALL_NAME

        extracted_dir = Path(self.extracted_dir)
        split_path = extracted_dir / DEFAULT_SPLIT_NAME
        tarball_path = extracted_dir / DEFAULT_TARBALL_NAME

        # Extract split file and tarball if needed
        if not split_path.exists() or not tarball_path.exists():
            if self.archive_path.exists():
                ok_split = self._try_native_zip_extract(
                    self.archive_path, f"crossdocked/{DEFAULT_SPLIT_NAME}", split_path
                )
                ok_tar = self._try_native_zip_extract(
                    self.archive_path, f"crossdocked/{DEFAULT_TARBALL_NAME}", tarball_path
                )
                if not ok_split:
                    self._stream_extract_member(self.archive_path, DEFAULT_SPLIT_NAME, split_path)
                if not ok_tar:
                    self._stream_extract_member(self.archive_path, DEFAULT_TARBALL_NAME, tarball_path)

        if not split_path.exists():
            print(f"[pocket_fm] ERROR: split file not found at {split_path}")
            return

        # Load the dataset (auto_extract=True so _pairs is populated)
        self._dataset = CrossDockedDataset(
            archive_path=self.archive_path,
            extracted_dir=self.extracted_dir,
            rmsd_max=self.rmsd_max,
            heavy_atoms_range=self.heavy_atoms_range,
            split=self.split,
            auto_extract=True,
        )

        # Extract tarball to get actual files
        tarball_path = Path(self.extracted_dir) / DEFAULT_TARBALL_NAME
        # The tarball extracts to crossdocked_pocket10/ subdirectory
        extract_dir = Path(self.extracted_dir) / "extracted" / "crossdocked_pocket10"
        if not extract_dir.exists():
            extract_dir.parent.mkdir(parents=True, exist_ok=True)
            import tarfile
            print(f"[pocket_fm] Extracting tarball to {extract_dir.parent} ...")
            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall(extract_dir.parent)
            print(f"[pocket_fm] Tarball extracted.")

        # Iterate through dataset and filter for MMP or random
        np.random.seed(self.seed)
        all_pairs = []
        mmp_pairs = []

        print(f"[pocket_fm] Scanning {len(self._dataset)} pairs for MMP targets...")
        for idx in range(len(self._dataset)):
            item = self._dataset[idx]
            pocket_rel = item["pocket_pdb_relpath"]
            ligand_rel = item["ligand_sdf_relpath"]

            # Resolve actual paths in the extracted tarball
            pocket_path = str(extract_dir / pocket_rel)
            ligand_path = str(extract_dir / ligand_rel)

            if not os.path.exists(pocket_path) or not os.path.exists(ligand_path):
                continue

            # Check for MMP in PDB header or filename
            protein_name = self._parse_pdb_header(pocket_path) or ""
            is_mmp = "mmp" in protein_name.lower() or "mmp" in ligand_rel.lower()

            if is_mmp:
                mmp_pairs.append((pocket_path, ligand_path, pocket_rel))
            else:
                all_pairs.append((pocket_path, ligand_path, pocket_rel))

            if idx % 10000 == 0 and idx > 0:
                print(f"  scanned {idx} pairs, MMP candidates: {len(mmp_pairs)}")

        print(f"[pocket_fm] Found {len(mmp_pairs)} MMP pairs, {len(all_pairs)} total non-MMP")

        # Use MMP pairs if available, otherwise fallback
        if len(mmp_pairs) >= 10 and self.filter_mmp:
            self._pairs = mmp_pairs[: self.max_pairs]
            print(f"[pocket_fm] Using {len(self._pairs)} MMP pairs")
        else:
            # Fallback: random 100 pairs
            fallback_n = min(self.max_pairs, len(all_pairs))
            self._pairs = list(np.random.RandomState(self.seed).choice(
                all_pairs, size=fallback_n, replace=False
            ))
            print(f"[pocket_fm] Fallback: using {len(self._pairs)} random pairs")

        # Pre-load pockets and ligands
        print(f"[pocket_fm] Loading {len(self._pairs)} pocket-ligand pairs...")
        for i, (pocket_path, ligand_path, pocket_rel) in enumerate(self._pairs):
            if i % 100 == 0 and i > 0:
                print(f"  loaded {i}/{len(self._pairs)}")
            pocket = self._load_pocket_from_pdb(pocket_path)
            ligand = self._load_ligand_molecule(ligand_path)
            if pocket is not None and ligand is not None:
                self._pockets.append(pocket)
                self._ligands.append(ligand)

        print(f"[pocket_fm] Loaded {len(self._pockets)} valid pocket-ligand pairs")

    def __len__(self) -> int:
        return len(self._pockets)

    def __getitem__(self, idx: int) -> tuple[Pocket, Molecule]:
        return self._pockets[idx], self._ligands[idx]


# ---------------------------------------------------------------------------
# pIC50 regressor (Morgan FP → pIC50) — same as train_fm_cytotox.py
# ---------------------------------------------------------------------------

class MorganPic50Regressor:
    def __init__(self, n_bits: int = 2048, hidden: int = 256, seed: int = 42):
        self.n_bits = n_bits
        self.hidden = hidden
        self.seed = seed
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MorganPic50Regressor":
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        self._model = MLPRegressor(
            hidden_layer_sizes=(self.hidden, self.hidden // 2),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.seed,
            verbose=False,
        )
        self._model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scaler.transform(X)
        return self._model.predict(Xs)


# ---------------------------------------------------------------------------
# Plot loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(losses: list[float], out_path: Path, title: str = "Pocket FM Training Loss") -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, marker="o", markersize=3, linewidth=1, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CFM Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[pocket_fm] Wrote loss curve to {out_path}")


# ---------------------------------------------------------------------------
# Evaluate generated molecules
# ---------------------------------------------------------------------------

def evaluate_molecules(
    mols: list[Molecule | None],
    predictor: RDKitPropertyPredictor,
    regressor: MorganPic50Regressor,
    morgan: MorganFingerprinter,
) -> dict:
    qed_list, logp_list, mw_list, tpsa_list = [], [], [], []
    pic50_pred_list = []
    valid_count = 0

    for mol in mols:
        if mol is None:
            continue
        smiles = getattr(mol, "smiles", None) or ""
        if not smiles:
            continue
        try:
            prop: PropertyPrediction = predictor.predict(mol)
        except Exception:
            continue
        qed_list.append(prop.qed)
        logp_list.append(prop.logp)
        mw_list.append(prop.mol_weight)
        tpsa_list.append(prop.tpsa)

        try:
            import rdkit.Chem
            fp = morgan.fingerprint_mol(predictor._coerce_to_mol(mol, rdkit.Chem))
            if fp is not None:
                pic50 = float(regressor.predict(fp.reshape(1, -1))[0])
                pic50_pred_list.append(pic50)
        except Exception:
            pass
        valid_count += 1

    qed_arr = np.array(qed_list, dtype=np.float32)
    pic50_arr = np.array(pic50_pred_list, dtype=np.float32)

    return {
        "n_valid": valid_count,
        "qed_mean": float(np.mean(qed_arr)) if len(qed_arr) > 0 else 0.0,
        "qed_std": float(np.std(qed_arr)) if len(qed_arr) > 0 else 0.0,
        "qed_median": float(np.median(qed_arr)) if len(qed_arr) > 0 else 0.0,
        "logp_mean": float(np.mean(logp_list)) if logp_list else 0.0,
        "mw_mean": float(np.mean(mw_list)) if mw_list else 0.0,
        "tpsa_mean": float(np.mean(tpsa_list)) if tpsa_list else 0.0,
        "frac_druglike": float(np.mean(qed_arr >= 0.5)) if len(qed_arr) > 0 else 0.0,
        "pic50_pred_mean": float(np.mean(pic50_arr)) if len(pic50_arr) > 0 else 0.0,
        "pic50_pred_std": float(np.std(pic50_arr)) if len(pic50_arr) > 0 else 0.0,
        "hit_rate_pic50_6": float(np.mean(pic50_arr >= 6.0)) if len(pic50_arr) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------

def write_report(
    args: argparse.Namespace,
    losses: list[float],
    eval_metrics: dict,
    ref_metrics: dict,
    train_time: float,
    gpu_used: bool,
    n_pairs: int,
    checkpoint_path: Path,
    report_path: Path,
) -> None:
    init_loss = losses[0] if losses else float("nan")
    final_loss = losses[-1] if losses else float("nan")
    content = f"""# Pocket-Conditioned Flow Matching — CrossDocked2020 Evaluation

## Experiment Config

| Parameter | Value |
|-----------|-------|
| Epochs | {args.epochs} |
| Batch size | {args.batch} |
| Learning rate | {args.lr} |
| Hidden dim | {args.hidden_dim} |
| EGNN layers | {args.n_layers} |
| Max atoms per ligand (OOM guard) | {args.max_atoms} |
| Max atoms per pocket (OOM guard) | {args.max_pocket_atoms} |
| Generation budget | {args.n_gen} |
| Seed | {args.seed} |
| Dataset | CrossDocked2020 |
| Target filter | {'MMP2/MMP9' if args.filter_mmp else 'Random'} |

## Dataset

- **Pairs loaded**: {n_pairs}
- **Split**: {args.split}

## Training

- **Initial loss**: {init_loss:.6f}
- **Final loss**: {final_loss:.6f}
- **Loss ratio (final/initial)**: {final_loss/init_loss:.4f}
- **Wall-clock (training)**: {train_time:.1f}s
- **GPU used**: {gpu_used}

## Generation Quality

> **Phase-0 limitation**: ``generate()`` produces placeholder atoms (random
> element types, no bonds, no SMILES). Full ligand decoding is Phase 1 work.
> The **training reference** column shows the distribution of the actual
> training ligands (ground truth).

| Metric | Generated (Phase-0 placeholder) | Training Reference |
|--------|--------------------------------|--------------------|
| Valid molecules | {eval_metrics['n_valid']} / {args.n_gen} | {ref_metrics['n_valid']} |
| QED mean | {eval_metrics['qed_mean']:.4f} | {ref_metrics['qed_mean']:.4f} |
| QED std | {eval_metrics['qed_std']:.4f} | {ref_metrics['qed_std']:.4f} |
| Fraction drug-like (QED ≥ 0.5) | {eval_metrics['frac_druglike']:.3f} | {ref_metrics['frac_druglike']:.3f} |
| MolLogP mean | {eval_metrics['logp_mean']:.4f} | {ref_metrics['logp_mean']:.4f} |
| MolWt mean | {eval_metrics['mw_mean']:.1f} | {ref_metrics['mw_mean']:.1f} |
| TPSA mean | {eval_metrics['tpsa_mean']:.2f} | {ref_metrics['tpsa_mean']:.2f} |
| Predicted pIC50 mean | {eval_metrics['pic50_pred_mean']:.4f} | {ref_metrics['pic50_pred_mean']:.4f} |
| Predicted pIC50 std | {eval_metrics['pic50_pred_std']:.4f} | {ref_metrics['pic50_pred_std']:.4f} |
| Hit rate @ pIC50 ≥ 6 | {eval_metrics['hit_rate_pic50_6']:.3f} | {ref_metrics['hit_rate_pic50_6']:.3f} |

## Checkpoint

Saved to: `{checkpoint_path}`

## Training Loss Curve

![Loss curve](./fm_pocket_train_loss.png)

## Notes

- Pocket-conditioned velocity field: shared EGNN for ligand + pocket,
  cross-context via pocket → ligand global pooling.
- pIC50 regressor = MLP on Morgan FP (radius=2, 2048 bits), trained on
  Ru subset of MetalCytoToxDB as a proxy for drug-likeness.
- Phase-1: replace placeholder generation with SMILES/element decoder.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content)
    print(f"[pocket_fm] Wrote report to {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pocket-conditioned FM on CrossDocked2020.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8, help="Batch size (keep small for heavy 3D)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--max-atoms", type=int, default=40, help="Max ligand atoms")
    p.add_argument("--max-pocket-atoms", type=int, default=200, help="Max pocket atoms")
    p.add_argument("--n-gen", type=int, default=100, help="Molecules to generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--filter-mmp", type=int, default=1, help="Filter for MMP targets (1=yes, 0=no)")
    p.add_argument("--max-pairs", type=int, default=500, help="Max pairs to load")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--checkpoint-out", default=None)
    p.add_argument("--plot-out", default=None)
    p.add_argument("--report-out", default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.filter_mmp = bool(args.filter_mmp)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- 1. Load dataset ---------------------------------------------------
    print("[pocket_fm] Loading CrossDocked2020 dataset...")
    t0 = time.time()

    archive_path = Path("/mnt/storage/data/molmetal/CrossDocked2020_cascadediff.zip")
    extracted_dir = Path("/mnt/storage/data/molmetal/crossdocked")

    dm = PocketDataModule(
        archive_path=archive_path,
        extracted_dir=extracted_dir,
        rmsd_max=1.0,
        heavy_atoms_range=(3, 50),
        split=args.split,
        filter_mmp=args.filter_mmp,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    dm.setup()

    n_pairs = len(dm)
    if n_pairs == 0:
        print("[pocket_fm] ERROR: no valid pocket-ligand pairs loaded")
        return 1
    print(f"[pocket_fm] {n_pairs} pairs loaded in {time.time()-t0:.1f}s")

    # --- 2. Build velocity field + FM infrastructure -----------------------
    print("[pocket_fm] Initialising PocketConditionedVelocityField...")
    device_info = verify_rocm_active()
    device = get_device()
    print(f"[pocket_fm]   device = {device}")

    model = PocketConditionedVelocityField(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Flow matching path (from facebookresearch/flow_matching)
    AffineProbPath, CondOTScheduler, ODESolver, ModelWrapper = _import_fm_lib(
        str(PROJECT_ROOT / "molmetal" / "references" / "flow_matching")
    )
    scheduler = CondOTScheduler()
    path = AffineProbPath(scheduler=scheduler)

    # --- 3. Training loop --------------------------------------------------
    print(f"[pocket_fm] Training for {args.epochs} epochs (batch={args.batch})...")
    t_train = time.time()
    losses: list[float] = []

    n_batches_per_epoch = max(1, n_pairs // args.batch)

    for epoch in range(args.epochs):
        epoch_losses = []
        order = np.random.permutation(n_pairs)

        for batch_start in range(0, n_pairs, args.batch):
            batch_idx = order[batch_start:batch_start + args.batch]
            if len(batch_idx) == 0:
                continue

            # Collect batch
            batch_pockets = []
            batch_ligands = []
            max_lig_atoms = 0
            max_poc_atoms = 0

            for idx in batch_idx:
                pocket, ligand = dm[idx]
                if ligand.n_atoms > args.max_atoms:
                    continue
                batch_pockets.append(pocket)
                batch_ligands.append(ligand)
                max_lig_atoms = max(max_lig_atoms, ligand.n_atoms)
                max_poc_atoms = max(max_poc_atoms, pocket.n_atoms)

            if not batch_ligands:
                continue

            b = len(batch_ligands)

            # Build padded tensors
            x_1 = torch.zeros(b, max_lig_atoms, 3, device=device)
            atom_types_lig = torch.zeros(b, max_lig_atoms, dtype=torch.long, device=device)
            ligand_mask = torch.zeros(b, max_lig_atoms, dtype=torch.bool, device=device)

            x_pocket = torch.zeros(b, max_poc_atoms, 3, device=device)
            atom_types_poc = torch.zeros(b, max_poc_atoms, dtype=torch.long, device=device)
            pocket_mask = torch.zeros(b, max_poc_atoms, dtype=torch.bool, device=device)

            for i, (pocket, ligand) in enumerate(zip(batch_pockets, batch_ligands)):
                n_l = ligand.n_atoms
                n_p = pocket.n_atoms
                x_1[i, :n_l] = ligand.coords.to(device)
                atom_types_lig[i, :n_l] = ligand.atom_types.to(device)
                ligand_mask[i, :n_l] = True
                x_pocket[i, :n_p] = pocket.coords.to(device)
                atom_types_poc[i, :n_p] = pocket.atom_types.to(device)
                pocket_mask[i, :n_p] = True

            # Edge indices — dummy (model builds its own per-sample edges internally)
            edge_index_lig = torch.zeros(b, 2, 0, dtype=torch.long, device=device)
            edge_index_poc = torch.zeros(b, 2, 0, dtype=torch.long, device=device)

            # Sample noise + time
            x_0 = torch.randn_like(x_1)
            t = torch.rand(b, device=device)

            # Sample path
            path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)

            # Velocity prediction
            v_pred = model(
                ligand_coords=path_sample.x_t,
                ligand_atom_types=atom_types_lig,
                pocket_coords=x_pocket,
                pocket_atom_types=atom_types_poc,
                edge_index_lig=edge_index_lig,
                edge_index_poc=edge_index_poc,
                t=t,
                ligand_mask=ligand_mask,
            )

            # CFM loss
            mask_3d = ligand_mask.unsqueeze(-1).to(v_pred.dtype)
            per_atom_loss = (v_pred - path_sample.dx_t) ** 2 * mask_3d
            denom = mask_3d.sum().clamp(min=1.0)
            loss = per_atom_loss.sum() / denom

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        losses.append(avg_loss)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}: loss={avg_loss:.6f}")

    train_time = time.time() - t_train
    gpu_used = str(device).startswith("cuda") or str(device).startswith("hip")
    print(f"[pocket_fm] Training done in {train_time:.1f}s ({train_time/args.epochs:.2f}s/epoch)")

    # --- 4. Save checkpoint ------------------------------------------------
    ckpt_out = Path(args.checkpoint_out) if args.checkpoint_out else CHECKPOINT_DIR / "fm_pocket.pt"
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "losses": losses,
        "args": vars(args),
        "n_pairs": n_pairs,
    }
    torch.save(ckpt, str(ckpt_out))
    print(f"[pocket_fm] Saved checkpoint to {ckpt_out}")

    # --- 5. Plot loss curve ------------------------------------------------
    plot_out = Path(args.plot_out) if args.plot_out else REPORTS_DIR / "fm_pocket_train_loss.png"
    plot_loss_curve(losses, plot_out)

    # --- 6. Generate from first pocket -------------------------------------
    print(f"[pocket_fm] Generating {args.n_gen} molecules from first pocket...")
    if len(dm) == 0:
        print("[pocket_fm] ERROR: no pairs available for generation")
        return 1

    first_pocket, _ = dm[0]
    first_pocket_dev = first_pocket.to(device)

    # Use a fixed ligand size (median of training)
    median_lig_atoms = int(np.median([dm[i][1].n_atoms for i in range(min(100, len(dm)))]))
    n_gen_atoms = max(8, min(median_lig_atoms, 20))

    gen_atom_types = torch.randint(1, 10, (args.n_gen, n_gen_atoms), device=device)
    gen_x_0 = torch.randn(args.n_gen, n_gen_atoms, 3, device=device)
    t_grid = torch.linspace(0.0, 1.0, 51, device=device)

    # Build pocket context tensors
    n_poc = first_pocket_dev.n_atoms
    gen_pocket_coords = first_pocket_dev.coords.unsqueeze(0).expand(args.n_gen, n_poc, -1)
    gen_pocket_atom_types = first_pocket_dev.atom_types.unsqueeze(0).expand(args.n_gen, n_poc)

    # Dummy edge indices — model builds its own per-sample edges internally
    edge_index_lig_gen = torch.zeros(args.n_gen, 2, 0, dtype=torch.long, device=device)
    edge_index_poc_gen = torch.zeros(args.n_gen, 2, 0, dtype=torch.long, device=device)

    # ligand_mask for generation: all atoms are real (no padding)
    gen_ligand_mask = torch.ones(args.n_gen, n_gen_atoms, dtype=torch.bool, device=device)

    from functools import partial

    def velocity_model(x, t, atom_types, pocket_coords, pocket_atom_types, edge_index_lig, edge_index_poc, ligand_mask):
        # torchdiffeq passes x=... as keyword arg; flow_matching ModelWrapper forwards t=t
        return model(
            ligand_coords=x,
            ligand_atom_types=atom_types,
            pocket_coords=pocket_coords,
            pocket_atom_types=pocket_atom_types,
            edge_index_lig=edge_index_lig,
            edge_index_poc=edge_index_poc,
            t=t,
            ligand_mask=ligand_mask,
        )

    velocity_fn = partial(
        velocity_model,
        atom_types=gen_atom_types,
        pocket_coords=gen_pocket_coords,
        pocket_atom_types=gen_pocket_atom_types,
        edge_index_lig=edge_index_lig_gen,
        edge_index_poc=edge_index_poc_gen,
        ligand_mask=gen_ligand_mask,
    )
    wrapper = ModelWrapper(model=velocity_fn)
    solver = ODESolver(velocity_model=wrapper)

    x_final = solver.sample(
        x_init=gen_x_0,
        step_size=1.0 / 50,
        method="euler",
        time_grid=t_grid,
    )
    if x_final.dim() == 4:
        x_final = x_final[-1]

    gen_mols = []
    for i in range(args.n_gen):
        gen_mols.append(Molecule(
            coords=x_final[i].detach().cpu(),
            atom_types=gen_atom_types[i].detach().cpu(),
            bonds=torch.zeros(2, 0, dtype=torch.long),
            bond_types=torch.zeros(0, dtype=torch.long),
            formal_charges=torch.zeros(n_gen_atoms, dtype=torch.long),
            smiles="",
        ))

    print(f"[pocket_fm] Generated {len(gen_mols)} raw molecules (placeholder atoms)")

    # --- 7. Score with RDKitPropertyPredictor ------------------------------
    print("[pocket_fm] Scoring with RDKitPropertyPredictor...")
    predictor = RDKitPropertyPredictor()
    predictor.setup(device="cpu")

    # --- 8. Train pIC50 regressor on Ru subset (proxy for drug-likeness) ---
    print("[pocket_fm] Training Morgan-FP → pIC50 regressor on Ru subset...")
    try:
        from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
        flt = CytotoxFilter(time_threshold=24.0, ic50_min=0.01, metal_whitelist=["Ru"], compute_pic50=True, compute_active=True)
        ds_ru = MetalCytotoxDataset.from_csv(filters=flt)
        train_smiles_ru = []
        train_pic50_ru = []
        for i in range(len(ds_ru)):
            row = ds_ru[i]
            if row["smiles"] and not np.isnan(row["pIC50"]):
                train_smiles_ru.append(row["smiles"])
                train_pic50_ru.append(row["pIC50"])
        train_pic50_ru = np.array(train_pic50_ru, dtype=np.float32)
        print(f"[pocket_fm]   {len(train_pic50_ru)} Ru training molecules with valid pIC50")
    except Exception as e:
        print(f"[pocket_fm] WARNING: could not load Ru dataset ({e}), using random regressor")
        train_smiles_ru = ["CCO", "c1ccccc1", "CC(=O)O"]
        train_pic50_ru = np.array([5.0, 4.0, 6.0], dtype=np.float32)

    morgan = MorganFingerprinter(radius=2, n_bits=2048)
    if train_smiles_ru:
        X_train_ru = morgan(train_smiles_ru)
        regressor = MorganPic50Regressor(seed=args.seed)
        regressor.fit(X_train_ru, train_pic50_ru)
    else:
        regressor = None

    # --- 9. Evaluate -------------------------------------------------------
    eval_metrics = evaluate_molecules(gen_mols, predictor, regressor, morgan)

    # Reference: score training ligands
    ref_mols = [dm[i][1] for i in range(min(1000, len(dm)))]
    ref_metrics = evaluate_molecules(ref_mols, predictor, regressor, morgan)

    print(f"[pocket_fm] Generated: valid={eval_metrics['n_valid']}, QED_mean={eval_metrics['qed_mean']:.3f}")
    print(f"[pocket_fm] Reference: valid={ref_metrics['n_valid']}, QED_mean={ref_metrics['qed_mean']:.3f}")

    # --- 10. Write report --------------------------------------------------
    report_path = Path(args.report_out) if args.report_out else REPORTS_DIR / "fm_pocket_eval.md"
    write_report(
        args=args,
        losses=losses,
        eval_metrics=eval_metrics,
        ref_metrics=ref_metrics,
        train_time=train_time,
        gpu_used=gpu_used,
        n_pairs=n_pairs,
        checkpoint_path=ckpt_out,
        report_path=report_path,
    )

    total_time = time.time() - t0
    print(f"\n[pocket_fm] DONE total_time={total_time:.1f}s  GPU={gpu_used}")
    print(f"[pocket_fm] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
