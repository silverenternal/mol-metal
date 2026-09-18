#!/usr/bin/env -S uv run --script
"""Validate locally extracted CrossDocked test pairs and write their manifest.

--dry-run validates all paths without overwriting the current manifest.
Incomplete local data is an error; placeholder pocket IDs are never silently
substituted for the test split.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


LOG = logging.getLogger("stage_crossdocked100")
ROOT = Path(__file__).resolve().parents[2]


def download_primary() -> Path | None:
    """Download a CrossDocked archive from known mirrors, returning its path."""
    urls = (
        "https://bits.csb.pitt.edu/files/crossdocked2020/CrossDocked2020.tar.gz",
        "https://huggingface.co/datasets/Paulino/crossdocked2020/resolve/main/CrossDocked2020.tar.gz",
        "https://zenodo.org/records/10359592/files/CrossDocked2020.tar.gz",
    )
    destination = ROOT / "molmetal" / "data" / "CrossDocked2020.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            LOG.info("Trying CrossDocked mirror: %s", url)
            with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as out:
                out.write(response.read())
            if destination.stat().st_size > 0:
                return destination
        except Exception as exc:  # network and mirror-specific failures are expected
            LOG.warning("Mirror failed (%s): %s", url, exc)
            destination.unlink(missing_ok=True)
    return None


def find_cached() -> Path | None:
    """Search project and user cache locations for a CrossDocked dataset."""
    candidates = [
        Path("/mnt/storage/data/molmetal"),
        ROOT / "molmetal" / "data", ROOT / "data",
        Path.home() / ".cache" / "crossdocked",
    ]
    patterns = ("*crossdock*", "*CrossDocked*", "*.tar.gz", "*.zip")
    for directory in candidates:
        if not directory.exists():
            continue
        for pattern in patterns:
            for path in directory.rglob(pattern):
                if path.is_file() or path.is_dir():
                    return path
    return None


def cached_test_pairs(dataset_root: Path) -> list[dict] | None:
    """Build a manifest from a locally staged DiffDock split, if present."""
    split_path = dataset_root / "crossdocked" / "split_by_name.pt"
    if not split_path.exists():
        return None
    try:
        import torch

        split = torch.load(split_path, weights_only=True)
        pairs = split.get("test", []) if isinstance(split, dict) else []
        if len(pairs) < 100:
            return None
        roots = [dataset_root / "crossdocked" / "extracted" / "crossdocked_pocket10",
                 dataset_root / "crossdocked" / "crossdocked_pocket10",
                 dataset_root / "crossdocked"]
        extracted = next((root for root in roots if all(
            (root / rec).is_file() and (root / lig).is_file()
            for rec, lig in pairs[:100]
        )), None)
        if extracted is None:
            raise FileNotFoundError("No extraction root contains all 100 test receptor/ligand pairs")
        if len(set(map(tuple, pairs[:100]))) != 100:
            raise ValueError("Test split contains duplicate receptor/ligand pairs")
        return [
            {
                "pocket_id": f"test_{i:03d}",
                "receptor_path": str((extracted / rec).resolve()),
                "ligand_path": str((extracted / lig).resolve()),
                "ref_path": "",
                "source": "CrossDocked2020_local",
            }
            for i, (rec, lig) in enumerate(pairs[:100])
        ]
    except Exception as exc:
        LOG.warning("Could not read staged split %s: %s", split_path, exc)
        return None


def fallback_pockets() -> list[dict]:
    """Return ten reproducible pocket records used when data is unavailable."""
    ids = ("1h36", "830c", "MMP2", "MMP13", "CA2", "1hbv", "1ca2", "4ptb", "1m17", "1di8")
    return [
        {"pocket_id": pid, "receptor_path": "", "ligand_path": "", "ref_path": "",
         "metal_atoms": "", "n_atoms": "", "n_residues": "", "source": "fallback"}
        for pid in ids
    ]


def parse_pocket(path: Path) -> dict:
    """Parse a PDB or SDF with RDKit and report atoms, metals, and residues."""
    try:
        from rdkit import Chem
    except ImportError:
        LOG.warning("RDKit is unavailable; returning empty structural counts for %s", path)
        return {"metal_atoms": "", "n_atoms": "", "n_residues": ""}
    suffix = path.suffix.lower()
    mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False) if suffix == ".pdb" else Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit could not parse {path}")
    metals = [a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol().upper() in {"ZN", "FE", "MG", "MN", "CA", "CU", "CO", "NI", "NA", "K"}]
    residues = {(a.GetPDBResidueInfo().GetChainId(), a.GetPDBResidueInfo().GetResidueNumber()) for a in mol.GetAtoms() if a.GetPDBResidueInfo()}
    return {"metal_atoms": ";".join(metals), "n_atoms": mol.GetNumAtoms(), "n_residues": len(residues)}


def write_manifest(pockets: list[dict], out: Path) -> None:
    """Write pocket records as a CSV manifest and echo it to stdout."""
    fields = ["pocket_id", "receptor_path", "ligand_path", "ref_path", "metal_atoms", "n_atoms", "n_residues", "source"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: pocket.get(key, "") for key in fields} for pocket in pockets)
    with out.open(newline="") as handle:
        print(handle.read(), end="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate local pairs without writing output")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/storage/data/molmetal"))
    parser.add_argument("--out", type=Path, default=ROOT / "molmetal/data/crossdocked100_manifest.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pockets = cached_test_pairs(args.data_root)
    if pockets is None:
        LOG.error("No complete local 100-pair test split; existing manifest preserved")
        return 1
    LOG.info("Validated %d test pairs / %d files under %s", len(pockets), 2 * len(pockets), args.data_root)
    if args.dry_run:
        return 0
    write_manifest(pockets, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
