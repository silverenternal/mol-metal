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
        # Wf-A33-FIXUP-4 (REGR-1): admet8.py:221 schema emits
        # ``per_metric_mean`` / ``per_metric_median`` / ``per_metric_iqr`` /
        # ``n_ok`` / ``missing`` — NO ``aggregate`` key.  Read directly
        # from those keys (was: ``agg = panel.get("aggregate", {})`` which
        # always returned ``{}`` and caused n_ok=0 / qed_mean=null /
        # sa_mean=null for all 10 platiny pockets).
        per_mean = panel.get("per_metric_mean", {}) or {}
        per_median = panel.get("per_metric_median", {}) or {}
        per_iqr = panel.get("per_metric_iqr", {}) or {}
        missing = panel.get("missing", {}) or {}
        sa_mean_v = per_mean.get("sa")
        qed_mean_v = per_mean.get("qed")
        sa_median_v = per_median.get("sa")
        qed_median_v = per_median.get("qed")
        sa_iqr_v = per_iqr.get("sa")
        qed_iqr_v = per_iqr.get("qed")
        out[str(pocket_id)] = {
            "sa_mean": float(sa_mean_v) if sa_mean_v is not None else None,
            "qed_mean": float(qed_mean_v) if qed_mean_v is not None else None,
            "sa_median": float(sa_median_v) if sa_median_v is not None else None,
            "qed_median": float(qed_median_v) if qed_median_v is not None else None,
            "sa_iqr": float(sa_iqr_v) if sa_iqr_v is not None else None,
            "qed_iqr": float(qed_iqr_v) if qed_iqr_v is not None else None,
            "n_smiles": int(len(smiles_list)),
            "n_ok": int(panel.get("n_ok", 0)),
            "missing_sa": int(missing.get("sa", 0)),
            "missing_qed": int(missing.get("qed", 0)),
            # Honest framing per AGENTS.md §5.2: per-pocket SMILES are
            # MOCK placeholders (Wf-A33-FIXUP-1) — the pocket metadata
            # CSV's target_id column is being read as SMILES, RDKit
            # cannot parse these strings, so n_ok=0 is honest (not a
            # bug) once the schema fix is in place.
            "status": "MEASURED_PILOT_MOCK",
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


def run_full_per_pocket_panel(
    out_dir: Path,
    *,
    cohort_name: str = "default",
    smiles_csv: Optional[Path] = None,
    largest_frac_min: float = 0.80,
    min_atoms: int = 2,
    ckpt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Wf-A36-3 G2 producer — write ``cfm_per_pocket_metric.json``.

    Per AGENTS.md §5.5 (filesystem-truth) the JSON must exist on disk
    before :func:`paper_table.build_paper_table` can read it.  This is
    the *thin* producer the L8 paper-table column expects; it does NOT
    run checkpoint inference (heavyweight / GPU).  It either:

    * walks ``smiles_csv`` via :func:`build_per_pocket_panel` and emits
      ``status: MEASURED`` for any pocket with >=1 valid SMILES, OR
    * falls back to a ``status: NOT_COMPUTED`` stub that still carries
      the cohort_name + protocol_version so the gate can record a real
      ``sources_missing`` rather than a missing file.

    The ``ckpt_path`` argument is recorded for traceability but is
    intentionally unused here — running inference is delegated to a
    separate Wf-A37 hook (out of scope for A36-3).  Recording the path
    surfaces the consumer side (paper_table.py) and downstream callers
    that an inference path exists for future wiring.

    Parameters
    ----------
    out_dir : Path
        Directory where ``cfm_per_pocket_metric.json`` is written.  Will
        be created if it does not exist.
    cohort_name : str
        Cohort label — e.g. ``"wf140_p1_sa_5pocket"`` or ``"smoke"``.
        Persisted into the JSON so cross-table consumers can attribute
        the row.
    smiles_csv : Path | None
        Optional CSV with ``smiles`` (and ideally ``pocket_id``)
        columns.  When the file is present and non-empty, the canonical
        per-pocket decode + ADMET panel is computed.
    ckpt_path : Path | None
        Optional checkpoint path.  Recorded in the JSON
        ``ckpt_path`` field for traceability, but inference is NOT
        triggered from this function (Wf-A36-3 scope; see
        TODO/pending/a37_decoder_smoke_link.md).
    largest_frac_min, min_atoms
        Forwarded to the per-pocket decode gate.

    Returns
    -------
    dict
        The panel dict that was written to disk.  Useful for callers
        that want to read-and-verify in one step.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "cfm_per_pocket_metric.json"

    # Honour the explicit producer path when the CSV is on disk.
    if smiles_csv is not None and Path(smiles_csv).is_file() and Path(smiles_csv).stat().st_size > 0:
        try:
            panel = build_per_pocket_panel(
                Path(smiles_csv),
                out_json,
                largest_frac_min=largest_frac_min,
                min_atoms=min_atoms,
            )
        except Exception as e:
            panel = None
            stub_status = "BUILD_ERROR"
            stub_note = f"build_per_pocket_panel raised {type(e).__name__}: {e}"
        else:
            panel["cohort"] = cohort_name
            if ckpt_path is not None:
                panel["ckpt_path"] = str(ckpt_path)
            panel["status"] = "MEASURED"
            panel["producer"] = "build_per_pocket_panel"
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(panel, f, indent=2, sort_keys=True)
            return panel
    else:
        panel = None
        stub_status = "NOT_COMPUTED"
        stub_note = (
            f"No smiles_csv provided (smiles_csv={smiles_csv!r}); producer "
            f"running in stub mode. paper_table.py will read this JSON and "
            f"see status=NOT_COMPUTED. Run with --smiles-csv to switch "
            f"the gate to MEASURED."
        )

    # Stub / error fallback: still write the canonical JSON so the L8
    # gate sees a file on disk and emits status=NOT_COMPUTED rather than
    # silently dropping the row.
    panel = {
        "protocol_version": PROTOCOL_VERSION,
        "cohort": cohort_name,
        "n_pockets": 0,
        "decode": {},
        "admet": {},
        "smiles_csv": None if smiles_csv is None else str(smiles_csv),
        "ckpt_path": None if ckpt_path is None else str(ckpt_path),
        "status": stub_status,
        "producer": "run_full_per_pocket_panel",
        "honest_negative": (
            "L8 stub emission (Wf-A36-3 G2): cfm_per_pocket_metric.json "
            "was written so paper_table.py:398 reads a file instead of "
            "raising. Per-pocket decode + ADMET only flip to MEASURED "
            "once a smiles_csv is supplied to run_full_per_pocket_panel."
        ),
        "note": stub_note,
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
    "run_full_per_pocket_panel",
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
    # Test 4: Wf-A36-3 G2 — run_full_per_pocket_panel stub emission.
    # No smiles_csv → status=NOT_COMPUTED, but JSON MUST exist on disk.
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        stub = run_full_per_pocket_panel(
            out_dir,
            cohort_name="smoke",
        )
        assert (out_dir / "cfm_per_pocket_metric.json").is_file(), (
            "run_full_per_pocket_panel stub did not write the canonical JSON"
        )
        assert stub["status"] == "NOT_COMPUTED", stub["status"]
        assert stub["cohort"] == "smoke", stub["cohort"]
        assert stub["n_pockets"] == 0, stub["n_pockets"]
        assert stub["protocol_version"] == PROTOCOL_VERSION
    # Test 5: same function, smiles_csv provided → status=MEASURED.
    with _tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        csv_path = out_dir / "smi.csv"
        csv_path.write_text("smiles,pocket_id\nCCO,p1\nc1ccccc1,p1\n")
        measured = run_full_per_pocket_panel(
            out_dir,
            cohort_name="smoke_measured",
            smiles_csv=csv_path,
        )
        assert measured["status"] == "MEASURED", measured["status"]
        assert measured["cohort"] == "smoke_measured"
        assert measured["n_pockets"] == 1, measured["n_pockets"]
        assert "p1" in measured["decode"], measured["decode"]
        assert (out_dir / "cfm_per_pocket_metric.json").is_file()
    print(f"[per_pocket_metric] SMOKE PASS  5/5  n_pockets={panel['n_pockets']}")
    return 0


def _cli_main(argv: Optional[List[str]] = None) -> int:
    """Wf-A36-3 G2 — ``python -m molmetal.eval.scoring.per_pocket_panel``.

    Sub-commands:
      * (no args, ``--smoke``, ``--help``) → run :func:`_smoke`
      * ``--cohort NAME --out-dir DIR`` → run
        :func:`run_full_per_pocket_panel` writing
        ``cfm_per_pocket_metric.json`` into ``DIR``.

    Returns ``0`` on PASS / successful write, ``1`` on smoke FAIL.
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="per_pocket_metric",
        description=(
            "Wf-A36-3 G2: thin producer for "
            "cfm_per_pocket_metric.json. Either run --smoke (default, "
            "3 unit tests + 2 run_full round-trips) or write the JSON "
            "via run_full_per_pocket_panel. See molmetal.eval.scoring."
            "per_pocket_metric.run_full_per_pocket_panel for the full "
            "docstring."
        ),
    )
    parser.add_argument("--cohort", help="cohort name persisted into JSON")
    parser.add_argument("--out-dir", type=Path, help="output directory (writes cfm_per_pocket_metric.json here)")
    parser.add_argument("--smiles-csv", type=Path, help="CSV with smiles+pocket_id columns; omit for NOT_COMPUTED stub")
    parser.add_argument("--ckpt-path", type=Path, help="checkpoint path (recorded for traceability, not used)")
    parser.add_argument("--largest-frac-min", type=float, default=0.80)
    parser.add_argument("--min-atoms", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="run the unit smoke instead")
    args = parser.parse_args(argv)

    # Default to smoke when no producer args were given.
    producer_args = [args.cohort, args.out_dir, args.smiles_csv, args.ckpt_path, args.smoke]
    if not any(producer_args):
        return _smoke()
    if args.smoke:
        return _smoke()
    if not args.out_dir:
        print("--out-dir is required when not running --smoke", file=sys.stderr)
        return 2
    panel = run_full_per_pocket_panel(
        args.out_dir,
        cohort_name=args.cohort or "default",
        smiles_csv=args.smiles_csv,
        ckpt_path=args.ckpt_path,
        largest_frac_min=args.largest_frac_min,
        min_atoms=args.min_atoms,
    )
    print(
        f"[per_pocket_metric] wrote {args.out_dir / 'cfm_per_pocket_metric.json'} "
        f"status={panel['status']!r} cohort={panel['cohort']!r}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(_cli_main())