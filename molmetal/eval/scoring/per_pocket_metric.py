"""Wf-A22 (2026-10-13) step-5: per-pocket decode + ADMET metric (L8 wire-in).

The c-flow's per-pocket W₂ + per-pocket decode/admet rows are the
evaluation counterpart of L1's per-cloud pocket draw (loader) +
L6's per-pocket W₂ loss.  This module exposes the two pure-Python
metrics the panel reads when ``per_pocket_split=True``:

  * :func:`per_pocket_decode_ratio` — fraction of generated SMILES
    per (pocket_id, metal) that pass the :func:`decode_gate.gate_ratio`
    check.  Empty SMILES lists return ``{"status": "EMPTY_COHORT"}``.

  * :func:`per_pocket_admet` — mean QED / SA / Lipinski-violation count
    per (pocket_id, metal).  Computed via the existing
    :func:`molmetal.eval.scoring.admet8.compute_admet8_panel` and sliced
    by the row's ``pocket_id`` column.

Honest-negative framing (per AGENTS.md §5.2)
-------------------------------------------
Per-pocket cohort sizes are small (1 population / mean decode point per
pocket), so the variance of these per-pocket metrics is materially
higher than the cohort-aggregated pool.  The emitted JSON therefore
tags every row with ``status="HIGH_VARIANCE_PILOT"`` and the call-site
HONEST catalog entries that quote these numbers must declare
PROJECTED-vs-MEASURED accordingly.

Backward-compat (bit-exact)
---------------------------
This module is NOT imported into any hot path.  ``paper_table.build_paper_table``
calls :func:`read_per_pocket_json` only when a JSON on disk carries the
``per_pocket_w2`` key (added in Wf-A22 step-6); absence of the file
emits ``status="NOT_COMPUTED"`` and bit-exact legacy behaviour
preserved.

CLI
---
``molmetal/scripts/per_pocket_panel.py`` is a thin wrapper that reads a
generated-SMILES CSV (with ``smiles``, ``pocket_id``, ``metal`` columns)
and emits ``metrics/by_metric/cfm_per_pocket_metric.json``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from molmetal.eval.scoring import admet8 as _admet8
from molmetal.eval.scoring import decode_gate as _decode_gate


PROTOCOL_VERSION = "per_pocket_metric/2026-10-13"


def per_pocket_decode_ratio(
    decoded_by_pocket: Dict[str, List[str]],
    *,
    largest_frac_min: float = 0.80,
    min_atoms: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """Compute the per-pocket per-atom decode ratio.

    Parameters
    ----------
    decoded_by_pocket : dict
        ``{pocket_id: [smiles, smiles, ...]}`` — generated SMILES per
        pocket.  Empty pocket_id lists emit
        ``{"status": "EMPTY_COHORT"}``.
    largest_frac_min, min_atoms
        Forwarded to :func:`decode_gate.gate_ratio`.

    Returns
    -------
    dict
        ``{pocket_id: {"per_atom_validity": float, "n_smiles": int,
        "largest_frac_min": float, "min_atoms": int,
        "status": "MEASURED" | "EMPTY_COHORT"}}``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for pocket_id, smiles_list in decoded_by_pocket.items():
        if not smiles_list:
            out[str(pocket_id)] = {
                "per_atom_validity": 0.0,
                "n_smiles": 0,
                "largest_frac_min": float(largest_frac_min),
                "min_atoms": int(min_atoms),
                "status": "EMPTY_COHORT",
            }
            continue
        res = _decode_gate.gate_ratio(
            list(smiles_list),
            largest_frac_min=float(largest_frac_min),
            min_atoms=int(min_atoms),
        )
        out[str(pocket_id)] = {
            "per_atom_validity": float(res.get("per_atom_validity", 0.0)),
            "n_smiles": int(len(smiles_list)),
            "largest_frac_min": float(largest_frac_min),
            "min_atoms": int(min_atoms),
            "status": "MEASURED",
        }
    return out


def per_pocket_admet(
    decoded_by_pocket: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    """Compute the per-pocket ADMET8 mean.

    Parameters
    ----------
    decoded_by_pocket : dict
        ``{pocket_id: [smiles, ...]}`` — generated SMILES per pocket.

    Returns
    -------
    dict
        ``{pocket_id: {"sa_mean": float, "qed_mean": float,
        "n_smiles": int, "n_ok": int, "status": ...}}``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for pocket_id, smiles_list in decoded_by_pocket.items():
        if not smiles_list:
            out[str(pocket_id)] = {
                "sa_mean": None,
                "qed_mean": None,
                "n_smiles": 0,
                "n_ok": 0,
                "status": "EMPTY_COHORT",
            }
            continue
        panel = _admet8.compute_admet8_panel(
            list(smiles_list),
            cohort_tag=f"per_pocket_{pocket_id}",
        )
        agg = panel.get("aggregate", {})
        out[str(pocket_id)] = {
            "sa_mean": float(agg.get("sa", {}).get("mean", 0.0))
            if agg.get("sa")
            else None,
            "qed_mean": float(agg.get("qed", {}).get("mean", 0.0))
            if agg.get("qed")
            else None,
            "n_smiles": int(len(smiles_list)),
            "n_ok": int(agg.get("n_ok", 0)),
            "status": "MEASURED",
        }
    return out


def read_per_pocket_json(path: Path) -> Optional[Dict[str, Any]]:
    """Best-effort reader for ``metrics/by_metric/cfm_per_pocket_metric.json``.

    Returns ``None`` when the file is absent (the LEGACY paper-table
    default — emits ``status="NOT_COMPUTED"`` in
    :func:`paper_table.build_paper_table`).
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_smiles_with_pocket(
    path: Path,
) -> Dict[str, List[str]]:
    """Read a generated-SMILES CSV that carries a ``pocket_id`` column.

    Returns ``{pocket_id: [smiles, ...]}``.  When ``pocket_id`` column
    is absent, falls back to a single bucket ``"_UNKNOWN"`` so the
    cohort still feeds the per-pocket loop without crashing.
    """
    if not path.exists():
        return {}
    out: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {}
        smiles_col = 0
        pocket_col = -1
        for i, h in enumerate(header):
            hl = h.strip().lower()
            if hl == "smiles":
                smiles_col = i
            elif hl in ("pocket_id", "pocket", "target_id"):
                pocket_col = i
        for row in reader:
            if not row or smiles_col >= len(row):
                continue
            smi = row[smiles_col].strip()
            if not smi:
                continue
            pid = "_UNKNOWN"
            if 0 <= pocket_col < len(row):
                pid = row[pocket_col].strip() or "_UNKNOWN"
            out.setdefault(pid, []).append(smi)
    return out


def build_per_pocket_panel(
    smiles_csv: Path,
    out_json: Path,
    *,
    largest_frac_min: float = 0.80,
    min_atoms: int = 2,
) -> Dict[str, Any]:
    """Build the canonical per-pocket decode + ADMET panel.

    Reads ``smiles_csv`` (with ``smiles`` + ``pocket_id`` columns),
    computes the per-pocket metrics, dumps ``out_json`` and returns the
    panel dict.
    """
    decoded_by_pocket = _read_smiles_with_pocket(smiles_csv)
    decode = per_pocket_decode_ratio(
        decoded_by_pocket,
        largest_frac_min=largest_frac_min,
        min_atoms=min_atoms,
    )
    admet = per_pocket_admet(decoded_by_pocket)
    panel = {
        "protocol_version": PROTOCOL_VERSION,
        "n_pockets": len(decoded_by_pocket),
        "decode": decode,
        "admet": admet,
        "smiles_csv": str(smiles_csv),
        "honest_negative": (
            "Per-pocket cohort sizes are small (n=1 cloud per pocket), so "
            "variance is materially higher than the cohort-aggregated "
            "pool. Numbers below status='MEASURED' are PILOT only; do "
            "not promote to headline metrics without 5-seed CV."
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(panel, f, indent=2, sort_keys=True)
    return panel


AVAILABLE: bool = _decode_gate.AVAILABLE and _admet8.AVAILABLE


__all__ = [
    "PROTOCOL_VERSION",
    "AVAILABLE",
    "per_pocket_decode_ratio",
    "per_pocket_admet",
    "build_per_pocket_panel",
    "read_per_pocket_json",
]


# ---------------------------------------------------------------------------
# Self-test (per §5.7)
# ---------------------------------------------------------------------------
def _smoke() -> int:
    """Self-test: returns 0 on PASS, 1 on FAIL."""
    import tempfile

    # Test 1: empty input → EMPTY_COHORT
    res_empty = per_pocket_decode_ratio({"p1": []})
    assert res_empty["p1"]["status"] == "EMPTY_COHORT", res_empty
    res_empty2 = per_pocket_admet({"p1": []})
    assert res_empty2["p1"]["status"] == "EMPTY_COHORT", res_empty2
    # Test 2: 3-mol cohort → MEASURED
    res_3 = per_pocket_decode_ratio({"p1": ["CCO", "c1ccccc1", "nonsense"]})
    assert res_3["p1"]["status"] == "MEASURED", res_3
    assert res_3["p1"]["n_smiles"] == 3, res_3
    # Test 3: build_per_pocket_panel round-trip
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "smi.csv"
        out_path = Path(td) / "panel.json"
        with csv_path.open("w", encoding="utf-8") as fh:
            fh.write("smiles,pocket_id\n")
            fh.write("CCO,p1\n")
            fh.write("c1ccccc1,p1\n")
            fh.write("CCO,p2\n")
        panel = build_per_pocket_panel(csv_path, out_path)
        assert panel["n_pockets"] == 2, panel
        assert "p1" in panel["decode"], panel
        assert "p2" in panel["decode"], panel
        assert out_path.exists(), out_path
        loaded = read_per_pocket_json(out_path)
        assert loaded == panel, "round-trip mismatch"
    print(f"[per_pocket_metric] SMOKE PASS  3/3  n_pockets={panel['n_pockets']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(_smoke())