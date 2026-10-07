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

"""Wf-R43 m1 — Ru-AUC paper-protocol COHORT-COMPARABILITY guard (CPU-only).

Background
----------
``metrics/by_metric/metalcytotox_ru_auc.json`` carries ``m1`` as
``MEASURED_PARTIAL_LIFT`` with a value of 0.7613 (scaffold split, seed 0,
30 epochs) against a Krasnov-2026 gate target of 0.86 (``gap_to_gate``
-0.0987).  Under ``paper_protocol_measurement`` it also carries a
DOI-grouped 10-fold CV number: ``mean_auc`` 0.7717 +/- 0.0380, compared
against the paper's 0.810 +/- 0.024.

That comparison is **not a reproduction**, and the metric file already says
so in its own ``honest-negative``: the CV ran on a cohort of 15,799 rows /
4,793 complexes / **591 cell lines**, while the paper's Ru ML set is
**23** cell lines / 11,294 rows / 4,399 complexes.  95% of the residual row
gap sits in the 568 cell lines outside the top-23-by-volume subset.  So the
0.7717 is a cross-cohort estimate and the -0.038 shortfall against 0.810 is
not evidence that the model is worse than the paper's LightGBM; it is
partly evidence that the two models were scored on different data.

Why this module exists
----------------------
The honest-negative is currently a *string inside a JSON file*.  Nothing
stops a downstream surface (paper draft, INDEX row, wave closeout,
pre-flight summary) from quoting "m1 0.7717 vs paper 0.810, -0.0383" as a
like-for-like comparison.  This module makes the comparison auditable: it
answers one question —

    may m1 be reported as a Krasnov-COMPARABLE measurement?

— and answers **FAIL** until a 10-fold DOI-CV measurement on a
23-cell-line cohort is actually on record.  It never trains, never writes a
metric, and never touches ``paper/*``.

What is already available for the re-measurement
-----------------------------------------------
The cohort machinery is implemented and CPU-reachable today:

  * ``molmetal.data.cytotox.build_paper_cohort_indices(df, metal="Ru",
    cell_lines=...)`` — paper funnel + optional cell-line restriction.
  * ``molmetal.data.cytotox.top_cell_lines(df, 23)`` — the 23 lines with the
    most rows, tie-broken on NAME so the choice does not depend on CSV order.
  * ``molmetal/scripts/run_m1_paper_cv.py --top-cell-lines 23`` — the
    DOI-split 10-fold CV driver (the default is already 23).

MEASURED on ``.data/molmetal/MetalCytoToxDB.csv`` (md5
f270dc907dbdfede953063a14f0825a7, Zenodo v2 10.5281/zenodo.17106822): the
top-23 subset is **11,517 rows / 4,502 complexes / 23 cell lines** against
the paper's 11,294 / 4,399 / 23 — within 2% on rows and 9% on cell lines.
So the remediation is *not* blocked on data; it is blocked only on the GPU
sweep (10 folds x ~700 s ~ 2 h, AGENTS.md §5.1 USER_OK-gated).

Status semantics
----------------
PASS  a 23-cell-line, 10-fold, all-finite AUC record is on disk
FAIL  the recorded paper-protocol measurement is on a different cohort
      (the cross-cohort case), or its folds are incomplete / non-finite
SKIP  the metric file is absent (nothing to audit, not a crash)

Gate wiring
-----------
Opt-in via ``WF43_M1_PAPER_COHORT=1`` (the Wf-R35 P-A precedent, same as
``p3_wf_r43_m16_pt_ckpt_gap`` in this same wave), so registering the gate
moves ``EXPECTED_GATE_COUNTS["all"]`` 114 -> 115 without changing the
PASS/FAIL verdict of any existing pre-flight run.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

OPT_IN_ENV = "WF43_M1_PAPER_COHORT"
GATE_NAME = "p3_wf_r43_m1_paper_cohort"
METRIC_KEY = "m1"

#: The paper's Ru ML set is 23 cell lines.  m1 is comparable only at 23.
PAPER_CELL_LINES = 23
#: Krasnov-2026, DOI-split 10-fold CV, LightGBM (ChemRxiv v2 preprint).
PAPER_MEAN_AUC = 0.810
PAPER_STD_AUC = 0.024
#: Row / complex tolerances vs the paper's published funnel.
ROW_TOLERANCE_PCT = 10.0
COMPLEX_TOLERANCE_PCT = 15.0
#: Full 10-fold CV, no folds dropped.
EXPECTED_FOLDS = 10


def _is_finite_number(value: Any) -> bool:
    """True only for a real finite number (``NaN``/``None``/``str`` fail)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _measure_top_n_cohort(
    csv_path: Path, n_cell_lines: int
) -> Optional[Dict[str, Any]]:
    """MEASURE the top-``n`` Ru cohort straight from the release CSV.

    Returns ``None`` when the CSV (or pandas) is unavailable, so the guard
    degrades to the recorded-JSON audit instead of raising.  This is the
    CPU-reachable half of the remediation: it proves the 23-line cohort is
    constructible today, independent of any GPU run.
    """
    try:
        import pandas as pd  # local import: keeps the gate import cheap

        from molmetal.data import cytotox
    except Exception:  # pragma: no cover - import-time degradation
        return None
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
        probe = df.loc[cytotox.build_paper_cohort_indices(df, metal="Ru")]
        lines = cytotox.top_cell_lines(probe, n_cell_lines)
        sub = df.loc[
            cytotox.build_paper_cohort_indices(df, metal="Ru", cell_lines=lines)
        ]
    except Exception:  # pragma: no cover - malformed CSV degrades, not crash
        return None
    return {
        "n_rows": int(len(sub)),
        "n_complexes": int(sub["SMILES_Ligands"].nunique()),
        "n_cell_lines": int(sub["Cell_line"].nunique()),
        "cell_lines": list(lines),
    }


def audit_m1_paper_cohort(
    metric_path: Path,
    *,
    repo_root: Optional[Path] = None,
    expected_cell_lines: int = PAPER_CELL_LINES,
    paper_rows: int = 11294,
    paper_complexes: int = 4399,
    recompute_cohort: bool = True,
) -> Dict[str, Any]:
    """Audit whether ``m1`` may be reported as Krasnov-comparable.

    Reads ``metrics/by_metric/metalcytotox_ru_auc.json``.  The recorded
    ``paper_protocol_measurement`` block is the object under audit: it is the
    number a downstream surface would quote against the paper's 0.810.

    Returns a dict with ``status`` in {PASS, FAIL, SKIP}, a human-readable
    ``evidence`` string, and the structured findings.  Never raises on
    malformed input — a broken record FAILs loudly instead of crashing the
    pre-flight run.
    """
    result: Dict[str, Any] = {
        "status": "FAIL",
        "metric": METRIC_KEY,
        "evidence": "",
        "recorded_cohort": None,
        "observed_cell_lines": None,
        "expected_cell_lines": int(expected_cell_lines),
        "folds_run": 0,
        "per_fold_auc_finite": 0,
        "mean_auc": None,
        "cross_cohort": True,
        "top_n_cohort": None,
    }

    if not metric_path.is_file():
        result["status"] = "SKIP"
        result["evidence"] = (
            f"{METRIC_KEY}: no record at {metric_path} — nothing to audit "
            "(absent metric is SKIP, not FAIL)"
        )
        return result

    try:
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - any decode/read failure
        result["evidence"] = (
            f"{METRIC_KEY}: {metric_path.name} is unreadable ({exc.__class__.__name__}: "
            f"{exc}) — FAIL loud rather than silently inheriting the "
            "cross-cohort number (AGENTS.md §5.2)"
        )
        return result

    measurement = payload.get("paper_protocol_measurement")
    if not isinstance(measurement, dict):
        result["evidence"] = (
            f"{METRIC_KEY}: no paper_protocol_measurement block in "
            f"{metric_path.name} — there is NO Krasnov-comparable CV on record, "
            f"so m1 may not be compared to the paper's {PAPER_MEAN_AUC}. "
            f"Shipped scaffold-split value stays "
            f"{payload.get('value')} (status {payload.get('status')})"
        )
        result["mean_auc"] = payload.get("value")
        return result

    cohort = measurement.get("cohort") or {}
    observed_lines = cohort.get("cell_lines")
    observed_rows = cohort.get("rows")
    observed_complexes = cohort.get("complexes")
    result["recorded_cohort"] = {
        "rows": observed_rows,
        "complexes": observed_complexes,
        "cell_lines": observed_lines,
    }
    result["observed_cell_lines"] = observed_lines

    per_fold = list(measurement.get("per_fold_auc") or [])
    result["folds_run"] = int(measurement.get("folds_run", len(per_fold)) or 0)
    result["per_fold_auc_finite"] = sum(1 for a in per_fold if _is_finite_number(a))
    mean_auc = measurement.get("mean_auc")
    result["mean_auc"] = mean_auc if _is_finite_number(mean_auc) else None

    if recompute_cohort and repo_root is not None:
        csv_path = (
            Path(os.environ.get("MOLMETAL_DATA_DIR", repo_root / ".data" / "molmetal"))
            / "MetalCytoToxDB.csv"
        )
        result["top_n_cohort"] = _measure_top_n_cohort(csv_path, expected_cell_lines)

    # ---- verdict ---------------------------------------------------------
    if not isinstance(observed_lines, int):
        result["status"] = "FAIL"
        result["evidence"] = (
            f"{METRIC_KEY}: recorded paper-protocol cohort declares no "
            f"cell_lines count — the cohort scope is untraceable, so the "
            f"{mean_auc} CV number cannot be compared to the paper "
            f"({PAPER_MEAN_AUC} +/- {PAPER_STD_AUC})"
        )
        return result

    if observed_lines != expected_cell_lines:
        result["cross_cohort"] = True
        result["evidence"] = (
            f"{METRIC_KEY}: CROSS-COHORT — the recorded DOI-CV "
            f"(mean_auc={mean_auc}, {result['folds_run']} folds) was measured "
            f"on {observed_lines} cell lines / {observed_rows} rows / "
            f"{observed_complexes} complexes; the paper's Ru ML set is "
            f"{expected_cell_lines} lines / {paper_rows} rows / "
            f"{paper_complexes} complexes. The delta vs the paper's "
            f"{PAPER_MEAN_AUC} +/- {PAPER_STD_AUC} is therefore NOT a "
            f"like-for-like comparison and may not be quoted as one."
            + _remediation_sentence(result)
        )
        return result

    # cohort scope matches — now the folds must be complete and finite.
    result["cross_cohort"] = False
    if result["folds_run"] != EXPECTED_FOLDS or result["per_fold_auc_finite"] != len(
        per_fold
    ):
        result["evidence"] = (
            f"{METRIC_KEY}: cohort scope is paper-comparable "
            f"({observed_lines} cell lines) but the CV is incomplete — "
            f"{result['folds_run']}/{EXPECTED_FOLDS} folds run, "
            f"{result['per_fold_auc_finite']}/{len(per_fold)} finite AUCs. "
            f"A partial sweep cannot support a comparison to "
            f"{PAPER_MEAN_AUC}"
        )
        return result

    if isinstance(observed_rows, int) and isinstance(observed_complexes, int):
        row_delta = abs(observed_rows - paper_rows) / max(paper_rows, 1) * 100.0
        cx_delta = abs(observed_complexes - paper_complexes) / max(
            paper_complexes, 1
        ) * 100.0
        if row_delta > ROW_TOLERANCE_PCT or cx_delta > COMPLEX_TOLERANCE_PCT:
            result["evidence"] = (
                f"{METRIC_KEY}: cell-line scope matches ({observed_lines}) but "
                f"the cohort volume does not — {observed_rows} rows "
                f"({row_delta:+.1f}% vs {paper_rows}) / {observed_complexes} "
                f"complexes ({cx_delta:+.1f}% vs {paper_complexes}). Comparison "
                f"to the paper's {PAPER_MEAN_AUC} remains cross-cohort."
            )
            return result

    result["status"] = "PASS"
    result["evidence"] = (
        f"{METRIC_KEY}: paper-comparable — DOI-CV on {observed_lines} cell "
        f"lines / {observed_rows} rows / {observed_complexes} complexes, "
        f"{result['folds_run']}/{EXPECTED_FOLDS} folds, "
        f"mean_auc={mean_auc} +/- {measurement.get('std_auc')} vs paper "
        f"{PAPER_MEAN_AUC} +/- {PAPER_STD_AUC}"
    )
    return result


def _remediation_sentence(result: Dict[str, Any]) -> str:
    """Trailing evidence: what the GPU re-measurement would have to produce."""
    top = result.get("top_n_cohort")
    if not top:
        return (
            " Remediation: re-run molmetal/scripts/run_m1_paper_cv.py "
            f"--top-cell-lines {result['expected_cell_lines']} (GPU, ~2 h for "
            f"{EXPECTED_FOLDS} folds; USER_OK-gated per AGENTS.md §5.1)."
        )
    return (
        " Remediation cohort is constructible TODAY (CPU, MEASURED on the "
        f"release CSV): top-{result['expected_cell_lines']} = {top['n_rows']} "
        f"rows / {top['n_complexes']} complexes vs the paper's "
        f"{11294} / {4399}. Re-run "
        f"molmetal/scripts/run_m1_paper_cv.py --top-cell-lines "
        f"{result['expected_cell_lines']} --device cuda (10 folds x ~700 s "
        "~ 2 h GPU, USER_OK-gated per AGENTS.md §5.1); no data blocker "
        "remains."
    )


def gate_m1_paper_cohort(
    name: str,
    priority: str,
    pipeline: str,
    metric_path: Path,
    *,
    repo_root: Optional[Path] = None,
    opt_in: bool = True,
) -> Any:
    """Pre-flight gate wrapper around :func:`audit_m1_paper_cohort`.

    ``opt_in=True`` (the registered default) emits SKIP unless
    ``WF43_M1_PAPER_COHORT=1``, following the Wf-R35 P-A precedent used by
    ``p3_wf_r43_m16_pt_ckpt_gap`` in this same wave: the gate is registered
    and auditable, but adding it does not change the PASS/FAIL verdict of an
    existing pre-flight run.
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
            "Ru-AUC paper-cohort comparability guard (Wf-R35 P-A / Wf-R43 m16 "
            "opt-in pattern)",
            time.perf_counter() - t0,
        )

    result = audit_m1_paper_cohort(metric_path, repo_root=repo_root)
    return GateResult(
        name,
        priority,
        pipeline,
        result["status"],
        f"{name}: {result['evidence']}",
        time.perf_counter() - t0,
    )


__all__: List[str] = [
    "EXPECTED_FOLDS",
    "GATE_NAME",
    "METRIC_KEY",
    "OPT_IN_ENV",
    "PAPER_CELL_LINES",
    "PAPER_MEAN_AUC",
    "PAPER_STD_AUC",
    "audit_m1_paper_cohort",
    "gate_m1_paper_cohort",
]
