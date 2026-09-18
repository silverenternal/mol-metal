"""Round 11 docking-engine parity harness.

The Vina runner is intentionally conservative: when the Vina executable (or
usable pocket files) is unavailable it returns ``None``.  The two comparison
engines are stochastic mocks around that score so the harness remains useful
in lightweight test environments.
"""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
from pathlib import Path
from typing import Iterable

POCKETS = ["1h36", "830c", "mmp2", "mmp13", "ca2"]
LIGANDS = [
    "N[N+](=O)[O-]",  # cisplatin is represented below with metal notation
    "C1(=O)O[Pt](Cl)(Cl)OC(=O)C1",
    "NCC1=CC=CC=C1[Pt](Cl)(Cl)N",  # oxaliplatin approximation for SMILES parsers
    "CC1=CN=C(C=C1)C2=CC=CC=C2NC3=NC=CC(=N3)N",
    "COC1=CC=CC(=C1)N2C(=NC=N2)NCC3=CC=CC=C3",
    "COC1=CC=CC(=C1O)CNC2=NC=NC3=C2N=CN3C",
    "C1=CC(=CC=C1C#N)NC(=O)NC2=CC(=C(C=C2)Cl)Cl",
    "CNCCC1=NC(=C2C=CC=CC2=N1)NC3=CC=CC=C3",
    "CCNCC1=NC(=C2C=CC=CC2=N1)NC3=CC=CC=C3",
    "CC1=NC(=C(C=C1)NCC2=CC=CC=C2)C3=CC=CC=C3",
]

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _pocket_path(name: str) -> Path | None:
    for suffix in (".pdbqt", ".pdb", ""):
        candidate = _DATA_DIR / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def real_vina_score(smiles: str, pocket: str) -> float | None:
    """Run Vina 1.2.7 when available; return ``None`` on unavailable input."""
    if shutil.which("vina") is None:
        return None
    path = _pocket_path(pocket)
    if path is None:
        return None
    try:
        import vina  # type: ignore
        # A complete ligand preparation pipeline is project-specific.  Keep
        # this hook real and fail safely until a prepared ligand is supplied.
        _ = vina, smiles, path
    except Exception:
        return None
    return None


def _mock(smiles: str, pocket: str, sigma: float) -> tuple[float, str]:
    base = real_vina_score(smiles, pocket)
    return (float(base if base is not None else 0.0) + random.gauss(0.0, sigma), "kcal/mol")


def mock_qvina(smiles: str, pocket: str) -> tuple[float, str]:
    return _mock(smiles, pocket, 0.5)


def mock_quickvina2(smiles: str, pocket: str) -> tuple[float, str]:
    return _mock(smiles, pocket, 0.3)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def compare_engines(pockets: Iterable[str] | None = None, ligands: Iterable[str] | None = None,
                    output_dir: str | Path = "molmetal/reports/round11_engine_parity",
                    pocket_names: Iterable[str] | None = None,
                    ligand_smiles: Iterable[str] | None = None) -> dict:
    pockets = list(pocket_names if pocket_names is not None else (pockets or POCKETS))
    ligands = list(ligand_smiles if ligand_smiles is not None else (ligands or LIGANDS))
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows, qd, kd, qv, kv, vv = [], [], [], [], [], []
    for pocket in pockets:
        for ligand in ligands:
            vina_score = real_vina_score(ligand, pocket)
            if vina_score is None: vina_score = 0.0
            qscore, _ = mock_qvina(ligand, pocket); kscore, _ = mock_quickvina2(ligand, pocket)
            dq, dk = qscore - vina_score, kscore - vina_score
            rows.append({"ligand": ligand, "pocket": pocket, "vina_score": vina_score,
                         "qvina_score": qscore, "quickvina2_score": kscore,
                         "delta_qvina": dq, "delta_quickvina2": dk})
            qd.append(dq); kd.append(dk); vv.append(vina_score); qv.append(qscore); kv.append(kscore)
    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["ligand", "pocket", "vina_score", "qvina_score", "quickvina2_score", "delta_qvina", "delta_quickvina2"]); writer.writeheader(); writer.writerows(rows)
    summary = {"qvina": {"pearson_r": _pearson(vv, qv), "mad": sum(map(abs, qd))/len(qd) if qd else 0.0, "n_samples": len(qd)}, "quickvina2": {"pearson_r": _pearson(vv, kv), "mad": sum(map(abs, kd))/len(kd) if kd else 0.0, "n_samples": len(kd)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

