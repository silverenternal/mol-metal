# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wf-R43 m16 — D-MPNN pIC50 Pt checkpoint-gap auditor (CPU-only, no training).

Background
----------
``metrics/by_metric/dmpnn_ckpt_status.json`` inventories the D-MPNN pIC50
checkpoints on disk.  As of Wf-R43 it reports ``n_checkpoints = 25``
across **five** metals (Ru / Ir / Rh / Os / Re) x seeds 0-4.  Pt is
absent, so ``m16`` is tracked in ``TODO/INDEX.md`` section 17.1 as a
red ``MOCK_NEEDS_REMEASURE`` item ("m16 ckpt GAP — D-MPNN pIC50 Pt
missing").

Why the gap cannot simply be trained away
------------------------------------------
The 5-metal inventory is built from ``MetalCytoToxDB``-derived CSVs, and
those CSVs carry **no Pt rows at all** (MEASURED on disk,
``.data/molmetal/MetalCytoToxDB.csv`` 26,801 rows: Ru 19135 / Ir 4546 /
Rh 1134 / Os 1118 / Re 868 — Pt count 0).  The only Pt-labelled pIC50
cohort reachable from the trainer is ``--dataset platinai_4cell``
(``.data/molmetal/platinai_4cell.csv``, 4,078 rows), and its labels are
**SYNTHETIC** by construction — ``molmetal/data/platinai_4cell.meta.json``
records ``"framing": "synthetic IC50 labels for pipeline test; NOT
directly comparable to Rusanov 2025 Table S4-S5"``, generated as a
per-cell-line offset of ``-log10(IC50_nM)`` around
``base_log10_ic50_nm = 3.95`` with ``sigma = 0.45``.  Training a Pt
checkpoint on it would produce a number that *looks* like the missing
metric while being a restatement of the generative model that made the
labels — precisely the fabrication AGENTS.md section 5.2 forbids.

What this module does
---------------------
It is a **fail-loud guard**, not a trainer.  Given a ckpt-inventory JSON
it answers a single question: *may m16 be reported as SHIPPED?*  It
returns FAIL while the Pt gap is open, so no downstream surface
(paper draft, INDEX row, wave closeout) can quietly inherit a
SHIPPED-forever m16.  It never writes metrics and never trains.

The gate is opt-in via ``WF43_M16_PT_CKPT=1`` (the Wf-R35 P-A precedent)
so registering it does not change the verdict of an existing pre-flight
run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OPT_IN_ENV = "WF43_M16_PT_CKPT"

#: Metals the D-MPNN pIC50 inventory is expected to cover.  Pt is
#: deliberately included: its absence from the inventory is the m16 gap.
EXPECTED_METALS: Tuple[str, ...] = ("Pt", "Ru", "Ir", "Rh", "Os", "Re")

#: Metric key this gate guards (matches ``TODO/INDEX.md`` section 17.1).
METRIC_KEY = "m16"

#: Files consulted to prove whether a *real-labelled* Pt cohort exists.
REAL_PT_COHORT_CANDIDATES: Tuple[str, ...] = (
    ".data/molmetal/MetalCytoToxDB.csv",
    ".data/molmetal/metalcytotox_5metal_multitask_v4.csv",
    "molmetal/data/nci60_precious_9cell.csv",
)


def _metals_in_inventory(payload: Dict[str, Any]) -> List[str]:
    """Return the metals actually inventoried, from either JSON shape.

    Two on-disk shapes have shipped over time:

    * ``per_metal_summary`` — a dict keyed by metal (the current shape).
    * ``checkpoints`` — a list of per-file records carrying ``"metal"``.

    Both are read so the guard keeps working across the Wf-109 -> Wf-120
    JSON evolution instead of silently reporting a false PASS when the
    shape it expects is absent.
    """
    summary = payload.get("per_metal_summary")
    if isinstance(summary, dict) and summary:
        return sorted(str(k) for k in summary)

    ckpts = payload.get("checkpoints")
    if isinstance(ckpts, list):
        found = {
            str(c["metal"])
            for c in ckpts
            if isinstance(c, dict) and c.get("metal") is not None
        }
        return sorted(found)

    return []


def _real_pt_cohorts(repo_root: Path) -> List[str]:
    """Return candidate Pt cohorts that actually contain Pt rows.

    A candidate only counts if its metal column holds at least one Pt
    row.  This is what separates a REAL Pt cohort from the synthetic
    ``platinai_4cell.csv`` (which is Pt-only but carries synthetic
    labels, and which is therefore *not* listed here — it is covered by
    the SYNTHETIC check instead).
    """
    hits: List[str] = []
    for rel in REAL_PT_COHORT_CANDIDATES:
        path = repo_root / rel
        if not path.is_file():
            continue
        try:
            import pandas as pd

            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        metal_cols = [c for c in df.columns if str(c).lower() == "metal"]
        if not metal_cols:
            continue
        if df[metal_cols[0]].astype(str).str.strip().str.lower().eq("pt").any():
            hits.append(rel)
    return hits


def _platinai_labels_are_synthetic(repo_root: Path) -> Optional[bool]:
    """True/False if the PlatinAI 4-cell framing is readable, else None."""
    meta = repo_root / "molmetal" / "data" / "platinai_4cell.meta.json"
    if not meta.is_file():
        return None
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    framing = str(payload.get("framing", "")).lower()
    return "synthetic" in framing


def audit_m16(
    inventory_path: Path,
    *,
    repo_root: Optional[Path] = None,
    expected_metals: Tuple[str, ...] = EXPECTED_METALS,
) -> Dict[str, Any]:
    """Audit the D-MPNN pIC50 ckpt inventory for the m16 Pt gap.

    Returns a dict with ``status`` in {PASS, FAIL, SKIP}, an
    ``evidence`` string, and the MEASURED fields the verdict rests on
    (``metals_inventoried``, ``metals_missing``, ``n_checkpoints``,
    ``real_pt_cohorts``).  No file is written and nothing is trained.
    """
    repo_root = repo_root or inventory_path.resolve().parents[2]

    if not inventory_path.is_file():
        return {
            "status": "SKIP",
            "evidence": f"ckpt inventory absent ({inventory_path}) — nothing to audit",
            "metals_inventoried": [],
            "metals_missing": list(expected_metals),
            "n_checkpoints": None,
            "real_pt_cohorts": [],
        }

    try:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    except Exception as exc:  # malformed JSON must not crash the gate
        return {
            "status": "FAIL",
            "evidence": f"ckpt inventory unreadable ({exc})",
            "metals_inventoried": [],
            "metals_missing": list(expected_metals),
            "n_checkpoints": None,
            "real_pt_cohorts": [],
        }

    inventoried = _metals_in_inventory(payload)
    missing = [m for m in expected_metals if m not in inventoried]
    n_ckpts = payload.get("n_checkpoints")

    if not missing:
        return {
            "status": "PASS",
            "evidence": (
                f"all {len(expected_metals)} expected metals inventoried "
                f"({', '.join(inventoried)}); n_checkpoints={n_ckpts}"
            ),
            "metals_inventoried": inventoried,
            "metals_missing": [],
            "n_checkpoints": n_ckpts,
            "real_pt_cohorts": _real_pt_cohorts(repo_root),
        }

    real_pt = _real_pt_cohorts(repo_root)
    synthetic = _platinai_labels_are_synthetic(repo_root)

    evidence = (
        f"{METRIC_KEY} ckpt GAP: inventoried {len(inventoried)} metals "
        f"({', '.join(inventoried) or 'none'}); missing "
        f"{', '.join(missing)}; n_checkpoints={n_ckpts}. "
        f"REAL Pt cohorts on disk: {len(real_pt)}"
        + (f" ({', '.join(real_pt)})" if real_pt else "")
        + ". "
        + (
            "platinai_4cell labels are SYNTHETIC per "
            "molmetal/data/platinai_4cell.meta.json, so training a Pt "
            "checkpoint on them would restate the label generator — NOT a fix."
            if synthetic
            else "platinai_4cell framing unreadable; treat Pt labels as UNVERIFIED."
        )
        + " Closing m16 needs a real-labelled Pt pIC50 cohort "
        "(external dep: Rusanov 2025 Tables S4-S5 are login-walled), so "
        f"{METRIC_KEY} stays HONEST_NEGATIVE_MOCK_CKPT_GAP, not SHIPPED."
    )

    return {
        "status": "FAIL",
        "evidence": evidence,
        "metals_inventoried": inventoried,
        "metals_missing": missing,
        "n_checkpoints": n_ckpts,
        "real_pt_cohorts": real_pt,
        "platinai_labels_synthetic": synthetic,
    }


def gate_m16_pt_ckpt_gap(
    name: str,
    priority: str,
    pipeline: str,
    inventory_path: Path,
    *,
    repo_root: Optional[Path] = None,
    opt_in: bool = True,
) -> Any:
    """Pre-flight gate wrapper around :func:`audit_m16`.

    ``opt_in=True`` (the registered default) emits SKIP unless
    ``WF43_M16_PT_CKPT=1``, following the Wf-R35 P-A precedent: the gate
    is *registered* and auditable, but adding it does not change the
    PASS/FAIL verdict of an existing pre-flight run.
    """
    import time

    from molmetal.scripts.pre_flight_check import GateResult

    t0 = time.perf_counter()
    if opt_in and os.environ.get(OPT_IN_ENV, "").strip() not in (
        "1",
        "true",
        "TRUE",
        "yes",
    ):
        return GateResult(
            name,
            priority,
            pipeline,
            "SKIP",
            f"{name}: opt-in gate — set {OPT_IN_ENV}=1 to enforce the {METRIC_KEY} "
            "D-MPNN pIC50 Pt ckpt-gap guard (AGENTS.md §5.6/§5.7 new-gate "
            "opt-in pattern)",
            time.perf_counter() - t0,
        )

    result = audit_m16(inventory_path, repo_root=repo_root)
    return GateResult(
        name,
        priority,
        pipeline,
        result["status"],
        f"{name}: {result['evidence']}",
        time.perf_counter() - t0,
    )
