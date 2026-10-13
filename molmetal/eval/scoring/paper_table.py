"""Canonical paper-table evaluator for Track E (m14 + 8 ADMET + m4/m5/m7).

Track E (Wf-60231cff-03e-15, m14-admet-paper-align) — the proposed
canonical evaluator that wraps three orthogonal metric axes into a
single JSON for the paper §X table:

  1. **m14  per-atom decode ratio**  — :mod:`molmetal.eval.scoring.decode_gate`.
     Replaces the silent ``SanitizeMol``-only gate with the
     ``GetMolFrags``-aware floor at ``largest_frac_min``.
  2. **8 ADMET proxy**              — :mod:`molmetal.eval.scoring.admet8`.
     SA + QED + logP + TPSA + HBD + HBA + RotB + aromatic_rings
     (standalone per-mol emissions, disjoint from
     ``metal_aware_admet.MetalAwareADMET``).
  3. **m4/m5/m7 paper alignment**   — reads existing JSONs
     (m5 ``platinai_mcf7_pred.json``, m7 ``diversity_tanimoto.json``,
     m4 ``sa_mean.json``) and emits a single canonical row.

Honest-negative preservation
----------------------------
m5 MCF7 pIC50 (``platinai_mcf7_pred.json::value=null``,
``PENDING_GPU_RETRAIN``) is **preserved verbatim** in the canonical
output — the GPU retrain required to wire r_platinai into
``RewardAggregator`` was NOT executed (TODO-F2).  m7 IntDiv1's
``~0.017 BELOW training distribution`` honest-negative framing is
also preserved verbatim.

Inputs
------
- ``smiles_csv`` (optional path) — generated SMILES cohort.  When
  provided, m14 + ADMET8 are computed on the cohort on-the-fly.
- Existing JSONs on disk under ``metrics/by_metric/`` are read
  unconditionally and their values echoed (with their own honest-
  negative framings) into the canonical output.

Output JSON schema (single file per cohort)::

    {
      "cohort_id": str,
      "n_smiles": int,
      "m14_decode": {
         "per_atom_validity": float,
         "rdkit_strict_proxy": float,
         "n_multi_frag_rejects": int,
         "largest_frac_min": float,
         "min_atoms": int,
      },
      "admet8": { ... },                    # see admet8.compute_admet8_panel
      "m4_sa": { "value": float | null, "source": "...", "status": "..." },
      "m5_mcf7_pic50": { "value": float | null, "source": "...", "status": "..." },
      "m7_intdiv1": {
         "tanimoto_mean": float | null,
         "source": "...",
         "status": "...",
         "honest_negative": "...",
      },
      "protocol_version": "paper_table_e/2026-10-13",
    }

Lit anchors
-----------
- All inputs cite the production methodologist (`metal_aware_admet`,
  `decode_gate`, `admet8`, `sa_mean.json`, `platinai_mcf7_pred.json`,
  `diversity_tanimoto.json`).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from molmetal.eval.scoring import admet8 as _admet8
from molmetal.eval.scoring import decode_gate as _decode_gate


PROTOCOL_VERSION = "paper_table_e/2026-10-13"


# ---------------------------------------------------------------------------
# Side-JSON readers — preserve honest-negative verbatim
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_smiles_csv(path: Path) -> List[str]:
    """Best-effort reader for a generated-SMILES CSV.

    Looks for the first column with the header ``"smiles"``; falls back
    to taking the first column if no ``smiles`` header is found.
    Empty strings / ``None`` / non-string values are filtered.
    """
    if not path.exists():
        return []
    out: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
        # Find the SMILES column.
        col = 0
        for i, h in enumerate(header):
            if h.strip().lower() == "smiles":
                col = i
                break
        for row in reader:
            if not row or col >= len(row):
                continue
            v = row[col].strip()
            if v:
                out.append(v)
    return out


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------
@dataclass
class PaperTableRow:
    """Per-mol row emitted into the canonical JSON."""

    cohort_id: str
    n_smiles: int
    m14_decode: Dict[str, Any]
    admet8: Dict[str, Any]
    m4_sa: Dict[str, Any]
    m5_mcf7_pic50: Dict[str, Any]
    m7_intdiv1: Dict[str, Any]
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "n_smiles": int(self.n_smiles),
            "m14_decode": dict(self.m14_decode),
            "admet8": dict(self.admet8),
            "m4_sa": dict(self.m4_sa),
            "m5_mcf7_pic50": dict(self.m5_mcf7_pic50),
            "m7_intdiv1": dict(self.m7_intdiv1),
            "protocol_version": self.protocol_version,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_paper_table(
    cohort_id: str,
    smiles: Optional[Iterable[str]] = None,
    *,
    metrics_dir: Path,
    smiles_csv: Optional[Path] = None,
    largest_frac_min: float = 0.80,
    min_atoms: int = 2,
) -> PaperTableRow:
    """Build the canonical Track-E paper-table row.

    Parameters
    ----------
    cohort_id : str
        Used as ``cohort_id`` in the output dict.  Must be one of
        :data:`COHORT_KEYS` or a custom descriptor string.
    smiles : iterable[str], optional
        Generated-SMILES cohort.  If ``None`` and ``smiles_csv`` is
        provided, the CSV is loaded.  If both ``None``, the
        m14-decode + ADMET8 axes emit zero-cohort sentinels; m4/m5/m7
        side-JSON values still populate from disk.
    metrics_dir : Path
        Root of ``metrics/by_metric/*.json`` side-files.
    smiles_csv : Path, optional
        Optional CSV with a ``smiles`` column to load the cohort from.
    largest_frac_min, min_atoms
        Forwarded to :func:`decode_gate.gate_ratio` and
        :func:`decode_gate.gate_smiles`.

    Returns
    -------
    :class:`PaperTableRow`
        Always returns (no exceptions on missing side-JSONs; the
        honest-negative framings are emitted as ``status="MISSING"``).
    """
    # Coalesce smiles from CSV / direct input.
    if smiles is None and smiles_csv is not None:
        smiles = _read_smiles_csv(smiles_csv)
    smiles_list = list(smiles) if smiles is not None else []
    n_smiles = len(smiles_list)

    # m14 per-atom decode + ADMET8 (only if we have a cohort).
    if n_smiles > 0:
        m14 = _decode_gate.gate_ratio(
            smiles_list,
            largest_frac_min=largest_frac_min,
            min_atoms=min_atoms,
        )
        a8 = _admet8.compute_admet8_panel(smiles_list, cohort_tag=cohort_id)
        # RDKit-strict proxy (legacy behaviour) for side-by-side.
        rs = _decode_gate.gate_ratio(
            smiles_list, largest_frac_min=0.0, min_atoms=0,
        )
        m14_out = {
            "per_atom_validity": m14["per_atom_validity"],
            "rdkit_strict_proxy": rs["per_atom_validity"],
            "n_multi_frag_rejects": m14["multi_frag_rejects"],
            "n_min_atoms_rejects": m14["min_atoms_rejects"],
            "mean_largest_frac": m14["mean_largest_frac"],
            "mean_n_atoms": m14["mean_n_atoms"],
            "largest_frac_min": m14["largest_frac_min"],
            "min_atoms": m14["min_atoms"],
        }
        admet8_out = dict(a8)
    else:
        m14_out = {
            "per_atom_validity": 0.0,
            "rdkit_strict_proxy": 0.0,
            "n_multi_frag_rejects": 0,
            "n_min_atoms_rejects": 0,
            "mean_largest_frac": 0.0,
            "mean_n_atoms": 0.0,
            "largest_frac_min": float(largest_frac_min),
            "min_atoms": int(min_atoms),
            "cohort_empty": True,
        }
        admet8_out = {
            "cohort_id": cohort_id,
            "n_input": 0,
            "n_ok": 0,
            "cohort_empty": True,
            "protocol_version": "admet8/2026-10-13",
        }

    # m4 / m5 / m7 — read existing side-JSONs verbatim.
    m4 = _read_json(metrics_dir / "sa_mean.json")
    if m4 is None:
        m4_out = {
            "value": None,
            "source": "metrics/by_metric/sa_mean.json",
            "status": "MISSING",
            "honest_negative": "sa_mean.json not on disk",
        }
    else:
        # ``measurements`` is a list; surface the last MEASURED entry.
        last_meas = None
        for m in reversed(m4.get("measurements", [])):
            if m.get("status") == "MEASURED":
                last_meas = m
                break
        m4_out = {
            "value": last_meas.get("value") if last_meas else None,
            "n_samples": last_meas.get("n_samples") if last_meas else None,
            "source_run": last_meas.get("run") if last_meas else None,
            "source": "metrics/by_metric/sa_mean.json",
            "status": "MEASURED" if last_meas else m4.get("aggregate", {}).get("status", "UNKNOWN"),
            "honest_negative": (
                last_meas.get("note", "") if last_meas else
                m4.get("honest_framing", "")
            ),
        }

    m5 = _read_json(metrics_dir / "platinai_mcf7_pred.json")
    if m5 is None:
        m5_out = {
            "value": None,
            "source": "metrics/by_metric/platinai_mcf7_pred.json",
            "status": "MISSING",
            "n_cells_measured": 0,
            "n_cells_design": 10,
        }
    else:
        agg = m5.get("aggregate", {})
        m5_out = {
            "value": m5.get("aggregate", {}).get("best_measured"),
            "n_cells_design": agg.get("n_cells_design", 10),
            "n_cells_measured": agg.get("n_cells_measured", 0),
            "source": "metrics/by_metric/platinai_mcf7_pred.json",
            "status": agg.get("status", "UNKNOWN"),
            "honest_negative": m5.get("honest_framing", ""),
        }

    m7 = _read_json(metrics_dir / "diversity_tanimoto.json")
    if m7 is None:
        m7_out = {
            "tanimoto_mean": None,
            "source": "metrics/by_metric/diversity_tanimoto.json",
            "status": "MISSING",
        }
    else:
        last_meas = None
        for m in reversed(m7.get("measurements", [])):
            if m.get("status") == "MEASURED":
                last_meas = m
                break
        m7_out = {
            "tanimoto_mean": last_meas.get("value") if last_meas else None,
            "n_samples": last_meas.get("n_samples") if last_meas else None,
            "source_run": last_meas.get("run") if last_meas else None,
            "source": "metrics/by_metric/diversity_tanimoto.json",
            "status": "MEASURED" if last_meas else "UNKNOWN",
            "honest_negative": (
                last_meas.get("note", "") if last_meas else
                m7.get("honest_framing", "")
            ),
        }

    return PaperTableRow(
        cohort_id=str(cohort_id),
        n_smiles=n_smiles,
        m14_decode=m14_out,
        admet8=admet8_out,
        m4_sa=m4_out,
        m5_mcf7_pic50=m5_out,
        m7_intdiv1=m7_out,
    )


def write_paper_table_json(row: PaperTableRow, out_path: Path) -> Path:
    """Atomic-write helper (per §5.6).

    Writes ``row.to_dict()`` to ``out_path`` using a temp-file +
    ``os.replace`` round trip so the resulting file is never
    half-written even if the process is killed mid-write.
    """
    import os
    import tempfile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=out_path.name, suffix=".tmp", dir=str(out_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(row.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return out_path


# Canonical cohort keys — preserved for use by the CLI / paper §X labels.
COHORT_KEYS: List[str] = [
    "platinai_pt_subset_5k",
    "nci60_9_cell",
    "drugood",
    "lambda_pilot_5x1",
    "lambda_path_a_10x3",
]


AVAILABLE: bool = _decode_gate.AVAILABLE and _admet8.AVAILABLE

__all__ = [
    "PaperTableRow",
    "build_paper_table",
    "write_paper_table_json",
    "COHORT_KEYS",
    "PROTOCOL_VERSION",
    "AVAILABLE",
]


# ---------------------------------------------------------------------------
# Smoke (per §5.7)
# ---------------------------------------------------------------------------
def _smoke() -> int:
    """Self-test: returns 0 on PASS, 1 on FAIL.

    Tests:
      1. Empty cohort → row has n_smiles=0; side-JSON fields reflect
         whatever's on disk (``MISSING`` if the worktree has no
         ``metrics/by_metric/sa_mean.json``).
      2. Synthetic 4-mol cohort → m14 + ADMET8 populated, m4/m5/m7
         side-JSONs reflected verbatim (status preserved).
      3. ``write_paper_table_json`` round-trips the row.
    """
    # Test 1
    repo_root = Path(__file__).resolve().parents[3]
    metrics_dir = repo_root / "metrics" / "by_metric"
    if not metrics_dir.exists():
        # In a worktree that lacks the metrics directory, fall back
        # to a synthetic dir; the test asserts the structure, not
        # the data.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            metrics_dir = Path(td)
    r1 = build_paper_table(
        cohort_id="smoke_empty",
        smiles=[],
        metrics_dir=metrics_dir,
    )
    assert r1.n_smiles == 0
    assert r1.m14_decode["cohort_empty"] is True
    assert r1.admet8["cohort_empty"] is True
    # Test 2
    cohort = ["CCO", "c1ccccc1", "C.C.C.C", "nonsense"]
    r2 = build_paper_table(
        cohort_id="smoke_4mol",
        smiles=cohort,
        metrics_dir=metrics_dir,
    )
    assert r2.n_smiles == 4
    assert r2.m14_decode["per_atom_validity"] >= 0.0
    assert r2.admet8["n_ok"] >= 1
    # m4 / m5 / m7 status keys always present
    for k in ("status", "source"):
        assert k in r2.m4_sa, r2.m4_sa
        assert k in r2.m5_mcf7_pic50, r2.m5_mcf7_pic50
        assert k in r2.m7_intdiv1, r2.m7_intdiv1
    # CCO should pass per-atom; C.C.C.C should fail multi-fragment.
    assert r2.m14_decode["per_atom_validity"] < 1.0, r2.m14_decode
    # Test 3
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "paper_table_smoke.json"
        write_paper_table_json(r2, out_path)
        assert out_path.exists()
        with out_path.open("r") as f:
            rd = json.load(f)
        assert rd["cohort_id"] == "smoke_4mol"
        assert rd["n_smiles"] == 4
        assert rd["protocol_version"] == PROTOCOL_VERSION
    print(f"[paper_table] SMOKE PASS  3/3  n_smiles={r2.n_smiles}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(_smoke())
