"""Wf-253-D — m6_kappa Shape Kappa Index validation against MetalCytoToxDB labelled pIC50.

================================================================================
Wf-253-D / TODO-130 G2 — close the m6_kappa "do NOT claim without validation"
honest-negative on the Shape Kappa Index emitted by Wf-252-Apply F2.
================================================================================

Background
----------
Wf-252-Apply F2 (M6Kappa) shipped ``metrics/by_metric/m6_kappa.json`` showing
the Shape Kappa Index mean = 0.4719 on the 449-SMILES metallo-drug cohort.
The honest_framing note explicitly said:

    "do NOT claim it as a learned metallodrug score without validation
     against MetalCytoToxDB labelled pIC50"

This script closes that honest-negative by:

  1. Loading MetalCytoToxDB labelled pIC50 (via ``molmetal/data/metal_lomo.csv``,
     which is the Krasnov 2026 labelled export used by Wf-119 / Wf-124 / Wf-200
     v3 training — 5460 rows, 5 metals {Ru, Ir, Os, Rh, Re}, no pIC50 missing).
  2. Computing :func:`m6_kappa_from_smiles` for each SMILES.
  3. Computing Pearson r + Spearman r vs labelled pIC50 (and 95% CI via
     Fisher-z) — globally and per-metal.
  4. Emitting ``molmetal/reports/wf253d_m6_kappa_validation/`` with:
       - m6_kappa_validation.json (FLAT metrics/by_metric schema)
       - m6_kappa_validation.md (human-readable report)

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* **m6_kappa is a topological index, not a learned score.** A near-zero
  Pearson r is the EXPECTED outcome (shape complexity ≠ cytotoxicity);
  we report honestly regardless of sign.
* PlatinAI pIC50 is **synthetic** (per ``TODO/02_metrics/``); MetalCytoToxDB
  labelled pIC50 via metal_lomo.csv is the better validation cohort.
* Per-metal splits reduce cohort size — we report per-metal + global
  + per-cell-line (where available) to surface cohort-fragile signals.
* The Shape Kappa Index is bounded [0, 1] by the linear-min-max clip in
  ``metrics_v2._normalise_kappa``; this clipping is documented and shared
  with paper §6 Limitations.
* The validation result is appended to ``metrics/by_metric/m6_kappa.json``
  as a ``metalcytotox_validation`` block — the existing
  ``honest_framing_note`` field is preserved verbatim so paper §3
  doesn't drift.

Usage
-----

    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \\
        molmetal/reports/wf253d_m6_kappa_validation/m6_kappa_validate.py

Writes ``molmetal/reports/wf253d_m6_kappa_validation/m6_kappa_validation.json``
and ``m6_kappa_validation.md``.
"""

from __future__ import annotations

import csv
import json
import math
import os as _os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root on sys.path so we can import molmetal.* modules.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in _os.sys.path:  # type: ignore[attr-defined]
    _os.sys.path.insert(0, str(PROJECT_ROOT))  # type: ignore[attr-defined]

try:
    from molmetal.molmetal_lam.sbdd_env.metrics_v2 import (
        _RDKIT_AVAILABLE,
        m6_kappa_from_smiles,
        m6_kappa_mean,
    )
    _IMPORT_OK = True
except Exception as exc:  # pragma: no cover — best-effort
    _IMPORT_OK = False
    _IMPORT_ERR = exc
    m6_kappa_from_smiles = None  # type: ignore
    m6_kappa_mean = None  # type: ignore


REPORT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf253d_m6_kappa_validation"
DATA_CSV = PROJECT_ROOT / "molmetal" / "data" / "metal_lomo.csv"
EXISTING_M6_KAPPA_JSON = PROJECT_ROOT / "metrics" / "by_metric" / "m6_kappa.json"

#: Canonical MetalCytoToxDB labelled pIC50 cohort.  metal_lomo.csv is
#: the Krasnov 2026 / MetalCytoToxDB labelled export used by Wf-119 /
#: Wf-124 / Wf-200 v3 training.  5460 rows × 5 metals {Ru, Ir, Os,
#: Rh, Re}, all pIC50 populated.
METALS = ("Ru", "Ir", "Os", "Rh", "Re")
#: SMILES column name in metal_lomo.csv (matches the Wf-119 audit).
SMILES_COL = "SMILES_Ligands"
#: pIC50 column name in metal_lomo.csv.
PIC50_COL = "pIC50"
#: Metal column name.
METAL_COL = "Metal"


def _load_labelled_pic50() -> List[Dict]:
    """Read metal_lomo.csv and return list of dicts with smiles / metal / pIC50.

    Skips rows with empty / NaN pIC50 — though metal_lomo.csv has no
    NaN pIC50 rows by construction.
    """
    if not DATA_CSV.exists():
        return []
    out: List[Dict] = []
    with open(DATA_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = (row.get(SMILES_COL) or "").strip()
            metal = (row.get(METAL_COL) or "").strip()
            pic50_raw = (row.get(PIC50_COL) or "").strip()
            if not smi or not metal or not pic50_raw:
                continue
            try:
                pic50 = float(pic50_raw)
            except ValueError:
                continue
            if not math.isfinite(pic50):
                continue
            out.append(
                {"smiles": smi, "metal": metal, "pIC50": pic50}
            )
    return out


def _pearson_r(x: List[float], y: List[float]) -> Optional[float]:
    """Pearson r for paired lists; None on degenerate inputs."""
    n = len(x)
    if n < 3 or n != len(y):
        return None
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = sum((xi - mx) ** 2 for xi in x)
    den_y = sum((yi - my) ** 2 for yi in y)
    if den_x <= 0 or den_y <= 0:
        return None
    den = math.sqrt(den_x * den_y)
    if den == 0:
        return None
    return float(num / den)


def _spearman_rho(x: List[float], y: List[float]) -> Optional[float]:
    """Spearman rank-correlation coefficient; None on degenerate inputs.

    Computes Pearson r on the rank-transformed x / y.  Tied ranks are
    handled by mean-rank averaging (scipy convention, simple impl).
    """
    n = len(x)
    if n < 3 or n != len(y):
        return None

    def _rank(values: List[float]) -> List[float]:
        # Sort by value; tied values get the mean rank.
        indexed = sorted(enumerate(values), key=lambda kv: kv[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            mean_rank = (i + j) / 2.0 + 1.0  # 1-indexed ranks
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = mean_rank
            i = j + 1
        return ranks

    return _pearson_r(_rank(x), _rank(y))


def _fisher_z_ci(r: float, n: int, z_alpha: float = 1.96) -> Optional[Tuple[float, float]]:
    """95% CI for Pearson r via Fisher-z transform.

    Returns ``(lo, hi)`` as floats, or None on degenerate inputs.
    """
    if not math.isfinite(r) or n < 4:
        return None
    # Clamp r to avoid atanh(±1) -> inf.
    r_clamped = max(min(r, 0.999999), -0.999999)
    z = math.atanh(r_clamped)
    se = 1.0 / math.sqrt(n - 3)
    lo_z = z - z_alpha * se
    hi_z = z + z_alpha * se
    return (math.tanh(lo_z), math.tanh(hi_z))


def _summarise_block(rows: List[Dict]) -> Dict:
    """Compute m6_kappa + pIC50 + correlation for a block of rows.

    Returns a dict with n / n_parsed / pearson_r / spearman_rho /
    m6_kappa_mean / pIC50_mean + 95% CI on r.
    """
    n_input = len(rows)
    if n_input == 0:
        return {
            "n": 0,
            "n_parsed": 0,
            "pearson_r": None,
            "pearson_ci_lo": None,
            "pearson_ci_hi": None,
            "spearman_rho": None,
            "m6_kappa_mean": None,
            "m6_kappa_std": None,
            "pIC50_mean": None,
            "pIC50_std": None,
        }
    kappa_vals: List[float] = []
    pic50_vals: List[float] = []
    for row in rows:
        k = m6_kappa_from_smiles(row["smiles"]) if m6_kappa_from_smiles else None
        if isinstance(k, float):
            kappa_vals.append(k)
            pic50_vals.append(float(row["pIC50"]))
    n_parsed = len(kappa_vals)
    if n_parsed < 3:
        return {
            "n": n_input,
            "n_parsed": n_parsed,
            "pearson_r": None,
            "pearson_ci_lo": None,
            "pearson_ci_hi": None,
            "spearman_rho": None,
            "m6_kappa_mean": (
                sum(kappa_vals) / len(kappa_vals) if kappa_vals else None
            ),
            "m6_kappa_std": None,
            "pIC50_mean": (
                sum(pic50_vals) / len(pic50_vals) if pic50_vals else None
            ),
            "pIC50_std": None,
        }
    pr = _pearson_r(kappa_vals, pic50_vals)
    sr = _spearman_rho(kappa_vals, pic50_vals)
    ci = _fisher_z_ci(pr, n_parsed) if pr is not None else None
    kappa_mean = sum(kappa_vals) / n_parsed
    kappa_var = sum((k - kappa_mean) ** 2 for k in kappa_vals) / n_parsed
    pic50_mean = sum(pic50_vals) / n_parsed
    pic50_var = sum((p - pic50_mean) ** 2 for p in pic50_vals) / n_parsed
    return {
        "n": n_input,
        "n_parsed": n_parsed,
        "pearson_r": pr,
        "pearson_ci_lo": ci[0] if ci is not None else None,
        "pearson_ci_hi": ci[1] if ci is not None else None,
        "spearman_rho": sr,
        "m6_kappa_mean": float(kappa_mean),
        "m6_kappa_std": float(math.sqrt(kappa_var)),
        "pIC50_mean": float(pic50_mean),
        "pIC50_std": float(math.sqrt(pic50_var)),
    }


def _verdict(global_block: Dict) -> str:
    """Honest verdict based on global Pearson r + n."""
    pr = global_block.get("pearson_r")
    n = global_block.get("n_parsed", 0)
    if pr is None:
        return "INSUFFICIENT_DATA"
    abs_r = abs(pr)
    if n < 30:
        return "INSUFFICIENT_N_HONEST_NEGATIVE"
    if abs_r < 0.05:
        return "NEGATIVE_NO_CORRELATION (expected: Shape Kappa is a topological index, not a cytotoxicity predictor)"
    elif abs_r < 0.15:
        return "NEGATIVE_WEAK (Shape Kappa has negligible linear association with pIC50)"
    elif abs_r < 0.30:
        return "AMBIGUOUS_WEAK (modest association; near the noise floor for n~1k metallo drugs)"
    else:
        return "POSITIVE_MODERATE (unexpected: Shape Kappa Index does carry pIC50 signal)"


def main() -> int:
    if not _IMPORT_OK:
        print(
            f"[wf253d-m6] metrics_v2 import failed: {_IMPORT_ERR}",
            flush=True,
        )
        return 2

    if not _RDKIT_AVAILABLE:
        print(
            "[wf253d-m6] RDKit not available — skipping validation.",
            flush=True,
        )
        return 2

    rows = _load_labelled_pic50()
    n_total = len(rows)

    if n_total == 0:
        print(
            f"[wf253d-m6] No labelled data loaded from {DATA_CSV}",
            flush=True,
        )
        return 2

    # Global summary.
    global_block = _summarise_block(rows)

    # Per-metal summary.
    per_metal: Dict[str, Dict] = {}
    for m in METALS:
        m_rows = [r for r in rows if r["metal"] == m]
        per_metal[m] = _summarise_block(m_rows)

    verdict = _verdict(global_block)

    out = {
        "metric": "m6_kappa_validation",
        "description": (
            "Validation of the Wf-252 Shape Kappa Index (m6_kappa) "
            "against MetalCytoToxDB labelled pIC50 (Krasnov 2026 "
            "export via metal_lomo.csv, 5460 rows × 5 metals "
            "{Ru, Ir, Os, Rh, Re}).  Reports Pearson r + Spearman "
            "rho globally and per-metal.  Closes the Wf-252 "
            "honest-negative 'do NOT claim without validation' "
            "framing on m6_kappa."
        ),
        "unit": "Pearson r + Spearman rho (–1..+1, 0 = no correlation)",
        "scope": (
            "MetalCytoToxDB labelled pIC50 cohort via "
            "molmetal/data/metal_lomo.csv (Krasnov 2026 export)"
        ),
        "source": (
            "molmetal/reports/wf253d_m6_kappa_validation/"
            "m6_kappa_validate.py"
        ),
        "data_source_csv": str(DATA_CSV),
        "n_total_rows": int(n_total),
        "n_metals": int(len(METALS)),
        "global": global_block,
        "per_metal": per_metal,
        "verdict": verdict,
        "status": "MEASURED" if global_block.get("pearson_r") is not None else "DEFERRED",
        "timestamp": "2026-09-28",
        "honest_framing_note": (
            "Wf-253-D m6_kappa validation closes the Wf-252 "
            "honest-negative on the Shape Kappa Index.  MetalCytoToxDB "
            "labelled pIC50 cohort is the Krasnov 2026 export via "
            "molmetal/data/metal_lomo.csv (5460 rows × 5 metals, no "
            "pIC50 missing).  Pearson r + Spearman rho computed globally "
            "and per-metal; verdict surfaced honestly (a near-zero r "
            "is the EXPECTED outcome — Shape Kappa is a topological "
            "index, not a cytotoxicity predictor).  The Shape Kappa "
            "Index should be reported in paper §3 as a side-by-side "
            "topological complement to RDKit QED, NOT as a learned "
            "metallodrug score."
        ),
        "wf_tag": "Wf-253-D / TODO-130 G2",
        "wf_design_ref": (
            "molmetal/reports/wf253d_m6_kappa_validation/"
            "m6_kappa_validate.py"
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "m6_kappa_validation.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"[wf253d-m6] wrote {json_path}", flush=True)

    # Human-readable markdown summary.
    md_lines: List[str] = []
    md_lines.append("# m6_kappa Shape Kappa Index — MetalCytoToxDB pIC50 Validation")
    md_lines.append("")
    md_lines.append(
        "**Wf-253-D / TODO-130 G2** — closes the Wf-252 honest-negative "
        "(`do NOT claim as learned metallodrug score without validation`)."
    )
    md_lines.append("")
    md_lines.append("## Global summary")
    md_lines.append("")
    md_lines.append(f"- N_total rows: **{n_total}**")
    md_lines.append(
        f"- N_parsed (RDKit canonical SMILES): **{global_block['n_parsed']}**"
    )
    pr = global_block.get("pearson_r")
    sr = global_block.get("spearman_rho")
    md_lines.append(
        f"- Pearson r (m6_kappa vs pIC50): **{pr:.4f}**"
        if pr is not None else "- Pearson r: None"
    )
    if global_block.get("pearson_ci_lo") is not None:
        md_lines.append(
            f"  - 95% CI: [{global_block['pearson_ci_lo']:.4f}, "
            f"{global_block['pearson_ci_hi']:.4f}]"
        )
    md_lines.append(
        f"- Spearman rho: **{sr:.4f}**"
        if sr is not None else "- Spearman rho: None"
    )
    md_lines.append(
        f"- m6_kappa mean: **{global_block['m6_kappa_mean']:.4f}** "
        f"(std {global_block['m6_kappa_std']:.4f})"
    )
    md_lines.append(
        f"- pIC50 mean: **{global_block['pIC50_mean']:.4f}** "
        f"(std {global_block['pIC50_std']:.4f})"
    )
    md_lines.append("")
    md_lines.append("## Per-metal breakdown")
    md_lines.append("")
    md_lines.append("| metal | n | Pearson r | 95% CI | Spearman rho | m6_kappa mean | pIC50 mean |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in METALS:
        b = per_metal[m]
        pr_m = b.get("pearson_r")
        sr_m = b.get("spearman_rho")
        pr_s = f"{pr_m:.4f}" if pr_m is not None else "—"
        ci_lo = b.get("pearson_ci_lo")
        ci_hi = b.get("pearson_ci_hi")
        ci_s = (
            f"[{ci_lo:.4f}, {ci_hi:.4f}]"
            if ci_lo is not None and ci_hi is not None
            else "—"
        )
        sr_s = f"{sr_m:.4f}" if sr_m is not None else "—"
        mk = b.get("m6_kappa_mean")
        pi = b.get("pIC50_mean")
        mk_s = f"{mk:.4f}" if mk is not None else "—"
        pi_s = f"{pi:.4f}" if pi is not None else "—"
        md_lines.append(
            f"| {m} | {b.get('n_parsed', 0)} | {pr_s} | {ci_s} | "
            f"{sr_s} | {mk_s} | {pi_s} |"
        )
    md_lines.append("")
    md_lines.append("## Verdict")
    md_lines.append("")
    md_lines.append(f"> **{verdict}**")
    md_lines.append("")
    md_lines.append(
        "Honest framing: the Shape Kappa Index is a *normalised topological "
        "index*, not a learned metallodrug score.  A near-zero Pearson r is "
        "the EXPECTED outcome — the metric is reported in paper §3 as a "
        "side-by-side complement to RDKit Bickerton 2012 QED, NOT as a "
        "cytotoxicity predictor.  This validation explicitly closes the "
        "Wf-252 `do NOT claim without validation` honest-negative so the "
        "metric can ship in paper §3 with full disclosure."
    )
    md_lines.append("")
    md_path = REPORT_DIR / "m6_kappa_validation.md"
    md_path.write_text("\n".join(md_lines))
    print(f"[wf253d-m6] wrote {md_path}", flush=True)

    # Append to metrics/by_metric/m6_kappa.json as a metalcytotox_validation block.
    if EXISTING_M6_KAPPA_JSON.exists():
        try:
            existing = json.loads(EXISTING_M6_KAPPA_JSON.read_text())
        except Exception:
            existing = {}
        existing["metalcytotox_validation"] = {
            "source_script": (
                "molmetal/reports/wf253d_m6_kappa_validation/"
                "m6_kappa_validate.py"
            ),
            "data_source_csv": str(DATA_CSV),
            "n_total_rows": int(n_total),
            "global_pearson_r": global_block.get("pearson_r"),
            "global_spearman_rho": global_block.get("spearman_rho"),
            "verdict": verdict,
            "per_metal_pearson_r": {
                m: per_metal[m].get("pearson_r") for m in METALS
            },
            "per_metal_spearman_rho": {
                m: per_metal[m].get("spearman_rho") for m in METALS
            },
            "honest_framing": (
                "m6_kappa validated against MetalCytoToxDB labelled pIC50 "
                "(Krasnov 2026 export via metal_lomo.csv).  Shape Kappa "
                "Index is a topological index, not a learned metallodrug "
                "score — the global Pearson r is the verdict band honestly "
                "rather than over-claiming."
            ),
        }
        EXISTING_M6_KAPPA_JSON.write_text(json.dumps(existing, indent=2))
        print(
            f"[wf253d-m6] appended metalcytotox_validation block to "
            f"{EXISTING_M6_KAPPA_JSON}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())