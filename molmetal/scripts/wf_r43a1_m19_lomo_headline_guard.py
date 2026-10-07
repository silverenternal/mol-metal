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

"""Wf-R43 m19 — LOMO (Rh+Re) HEADLINE-USABILITY guard (CPU-only).

Background
----------
``metrics/by_metric/metal_lomo_auc_wasserstein.json`` carries ``m19`` as a
mean test ROC-AUC of **0.7577 +/- 0.0924** over 7 runs (2 folds: Rh 0.8798,
Re 0.7088), against an INTERNAL ``GATE_AUC`` of 0.7.  On its face that is a
clean PASS_KRASNOV +0.020 on the Rh fold.

It is not a real-data-only measurement.  Per Wf-R33 m19 cohort-resolution
(MEASURED, resolved to source):

  * the Rh cohort actually fed the run was
    ``molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg_normalized.json``
    (``cohort_resolution_verdict = "WF139P7_SYNTHETIC_LABELS"``), NOT the
    real-labelled ``wf387a2`` cohort the metric file previously pointed at;
  * ``n_synthetic_rh_appended = 500`` identically on all 7 runs;
  * ``append_synthetic_rh_to_dataset`` hard-codes the potency labels —
    active rows IC50=0.1 / pIC50=7.0, inactive rows IC50=10.0 / pIC50=5.0 —
    so the multitask REGRESSION head is supervised on fabricated targets;
  * the synthetic block is appended to the shared 5-metal pool **before** the
    LOMO split, so it contaminates the Rh train fold AND enters the Re train
    fold too.  Only the held-out TEST fold of each metal is clean, which is
    why the number is still held-out but is not real-data-only.

MEASURED corroboration (CPU, this module's own audit): the 5-metal pool
``metalcytotox_5metal_multitask_v4.csv`` holds 26,632 rows and the wave's
``run.log`` records ``n=27132`` — exactly 26,632 + 500.  The guard re-measures
both numbers on every run and reports ``pool_rows_match`` rather than trusting
this paragraph, so the contamination is arithmetic, not an inference from a
prose note.

Why this module exists
----------------------
The disqualification is currently a *string inside a JSON file*.  Nothing
stops a downstream surface (paper draft, INDEX row, closeout, pre-flight
summary) from quoting "m19 LOMO 0.7577, Rh fold +0.020 over Krasnov" as a
real-data result.  This module makes that auditable: it answers one
question —

    may m19 be quoted as a real-data LOMO headline?

— and answers **FAIL** until a real-only wave (n_synthetic_rh_appended == 0)
covering all five folds is on record.  It never trains, never writes a
metric, and never touches ``paper/*``.

Status semantics
----------------
PASS  a real-only LOMO record is on disk: zero synthetic rows appended, every
      per-run AUC finite, and every metal in the pool carrying a fold
FAIL  the recorded headline is contaminated (``n_synthetic_rh_appended > 0``
      or ``real_only_fold_value`` is null or the fold count is short), or any
      per-run AUC is non-finite / malformed
SKIP  the metric file is absent (nothing to audit, not a crash)

Gate wiring
-----------
Opt-in via ``WF43_M19_LOMO_HEADLINE=1`` (the Wf-R35 P-A precedent, same as
``p3_wf_r43_m16_pt_ckpt_gap`` / ``p3_wf_r43_m1_paper_cohort`` /
``p3_wf_r43_m2_cohort_independence`` in this same wave), so registering the
gate moves ``EXPECTED_GATE_COUNTS["all"]`` 117 -> 118 without changing the
verdict of any existing pre-flight run.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

OPT_IN_ENV = "WF43_M19_LOMO_HEADLINE"
GATE_NAME = "p3_wf_r43_m19_lomo_headline"
METRIC_KEY = "m19"

#: The multitask pool this LOMO wave splits over.  A real headline must
#: carry a fold for EVERY metal in the pool, not a 2-of-5 subset.
EXPECTED_METALS_IN_POOL = ("Ir", "Os", "Re", "Rh", "Ru")
#: The recorded wave covers Rh + Re only (n_folds = 2).
EXPECTED_FOLDS = len(EXPECTED_METALS_IN_POOL)

#: INDEX §17.1 cites these JSON keys for the m19 row.  They were absent from
#: the metric file at audit time (MEASURED, grep -c == 0 for each), so the
#: INDEX row pointed at fields that did not exist.  The guard reports the
#: dangling citation in its evidence rather than silently inheriting it.
INDEX_CITED_KEYS = (
    "usable_as_headline",
    "r17_baseline_reference",
    "disqualifying_reasons",
)


def _is_finite_number(value: Any) -> bool:
    """True only for a real finite number (``NaN``/``None``/``str`` fail)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _measure_pool_rows(csv_path: Path) -> Optional[Dict[str, Any]]:
    """MEASURE the real 5-metal pool row count + per-metal counts from CSV.

    This is what makes the contamination arithmetic checkable rather than
    asserted: the wave's ``n=27132`` must equal ``pool_rows + 500`` for the
    appended-synthetic story to hold.  Returns ``None`` when pandas / the CSV
    is unavailable so the guard degrades to the recorded-JSON audit instead of
    raising (same posture as the m1 guard's ``_measure_top_n_cohort``).
    """
    try:
        import pandas as pd  # local import: keeps the gate import cheap
    except Exception:  # pragma: no cover - import-time degradation
        return None
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:  # pragma: no cover - malformed CSV degrades, not crash
        return None
    if "metal" not in df.columns:
        return None
    counts = df["metal"].value_counts().to_dict()
    return {
        "n_rows": int(len(df)),
        "per_metal": {str(k): int(v) for k, v in counts.items()},
    }


def _wave_observed_rows(run_log: Path, limit_mb: float = 8.0) -> Optional[int]:
    """MEASURE the ``n=`` row count the wave actually trained on.

    Reads only the head of ``run.log`` (bounded, never the whole file) and
    returns the last ``n=<int>`` recorded.  ``None`` when the log is absent or
    carries no such line, so the corroboration degrades rather than raising.
    """
    try:
        if not run_log.is_file() or run_log.stat().st_size > limit_mb * 1024 * 1024:
            return None
        with run_log.open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(256 * 1024)
    except Exception:  # pragma: no cover - unreadable log degrades, not crash
        return None
    found = re.findall(r"\bn=(\d+)\b", head)
    return int(found[-1]) if found else None


def audit_m19_lomo_headline(
    metric_path: Path,
    *,
    repo_root: Optional[Path] = None,
    expected_folds: int = EXPECTED_FOLDS,
    expected_metals: tuple = EXPECTED_METALS_IN_POOL,
) -> Dict[str, Any]:
    """Audit whether ``m19`` may be quoted as a real-data LOMO headline.

    Reads ``metrics/by_metric/metal_lomo_auc_wasserstein.json``.  The recorded
    ``value`` / ``per_run_auc`` / ``n_synthetic_rh_appended`` / fold set are
    the objects under audit: they are what a downstream surface would quote
    as "LOMO transfer AUC".

    Returns a dict with ``status`` in {PASS, FAIL, SKIP}, a human-readable
    ``evidence`` string, and the structured findings.  Never raises on
    malformed input — a broken record FAILs loudly instead of crashing the
    pre-flight run (AGENTS.md §5.2).
    """
    result: Dict[str, Any] = {
        "status": "FAIL",
        "metric": METRIC_KEY,
        "evidence": "",
        "value": None,
        "n_runs": 0,
        "per_run_auc_finite": 0,
        "n_synthetic_rh_appended": None,
        "real_only_fold_value": None,
        "cohort_resolution_verdict": None,
        "folds_covered": 0,
        "expected_folds": int(expected_folds),
        "metals_missing": [],
        "pool_rows": None,
        "observed_rows": None,
        "pool_rows_match": None,
        "dangling_index_keys": [],
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
            f"{METRIC_KEY}: {metric_path.name} is unreadable "
            f"({exc.__class__.__name__}: {exc}) — FAIL loud rather than "
            "silently inheriting a contaminated headline (AGENTS.md §5.2)"
        )
        return result

    value = payload.get("value")
    result["value"] = value if _is_finite_number(value) else None
    result["n_synthetic_rh_appended"] = payload.get("n_synthetic_rh_appended")
    result["real_only_fold_value"] = payload.get("real_only_fold_value")
    result["cohort_resolution_verdict"] = payload.get("cohort_resolution_verdict")

    per_run = list(payload.get("per_run_auc") or [])
    result["n_runs"] = int(payload.get("n_runs", len(per_run)) or 0)
    result["per_run_auc_finite"] = sum(1 for a in per_run if _is_finite_number(a))

    metals = [str(m) for m in (payload.get("metals") or [])]
    result["folds_covered"] = len(metals)
    result["metals_missing"] = sorted(set(expected_metals) - set(metals))

    result["dangling_index_keys"] = [k for k in INDEX_CITED_KEYS if k not in payload]

    if repo_root is not None:
        root = Path(repo_root)
        data_dir = Path(
            os.environ.get("MOLMETAL_DATA_DIR", root / ".data" / "molmetal")
        )
        pool = _measure_pool_rows(
            data_dir / "metalcytotox_5metal_multitask_v4.csv"
        )
        if pool is not None:
            result["pool_rows"] = pool["n_rows"]
        result["observed_rows"] = _wave_observed_rows(
            root / "molmetal" / "reports" / "wf131_p3_deferred_runs" / "run.log"
        )
        appended_val = result["n_synthetic_rh_appended"]
        if (
            result["pool_rows"] is not None
            and _is_finite_number(appended_val)
            and result["observed_rows"] is not None
        ):
            result["pool_rows_match"] = (
                result["observed_rows"]
                == result["pool_rows"] + int(appended_val)
            )

    # ---- verdict ---------------------------------------------------------
    if result["value"] is None:
        result["evidence"] = (
            f"{METRIC_KEY}: {metric_path.name} records no finite headline "
            "value — FAIL loud rather than publishing a non-measurement"
        )
        return result

    if result["per_run_auc_finite"] != len(per_run) or (
        result["n_runs"] != len(per_run)
    ):
        result["evidence"] = (
            f"{METRIC_KEY}: the headline {result['value']} is not backed by a "
            f"complete run sweep — {result['per_run_auc_finite']}/{len(per_run)} "
            f"finite AUCs across {result['n_runs']}/{len(per_run)} recorded runs. "
            "A partial or non-finite sweep may not be quoted as a headline."
        )
        return result

    appended = result["n_synthetic_rh_appended"]
    if _is_finite_number(appended) and int(appended) > 0:
        pool_clause = ""
        if result["pool_rows"] is not None:
            pool_clause = (
                f" MEASURED corroboration: the 5-metal pool holds "
                f"{result['pool_rows']} rows"
            )
            if result["observed_rows"] is not None:
                pool_clause += (
                    f" and the wave's run.log records n={result['observed_rows']}"
                    f" = {result['pool_rows']} + {int(appended)}"
                )
            pool_clause += (
                " — the contamination is arithmetic, not an inference from a "
                "prose note."
            )
        result["evidence"] = (
            f"{METRIC_KEY}: NOT REAL-DATA-ONLY — the headline "
            f"{result['value']} pools {result['folds_covered']} folds "
            f"(n_synthetic_rh_appended={int(appended)}) fed by "
            f"cohort_resolution_verdict="
            f"{result['cohort_resolution_verdict']}, i.e. "
            "SYNTHETIC Rh rows appended to the shared pool BEFORE the LOMO "
            "split, so they contaminate the Rh train fold and enter the Re "
            "train fold too. Only the held-out TEST fold of each metal is "
            "clean, which is why the number is held-out but not "
            "real-data-only. The synthetic rows also carry hard-coded "
            "potency labels (active IC50=0.1/pIC50=7.0, inactive "
            "IC50=10.0/pIC50=5.0), so the multitask regression head trains "
            "on fabricated targets and no pIC50 from this wave is a "
            f"measurement.{pool_clause}"
            + _remediation_sentence(result, appended)
        )
        return result

    if result["real_only_fold_value"] is None:
        result["evidence"] = (
            f"{METRIC_KEY}: no synthetic rows are recorded as appended, but "
            "real_only_fold_value is null — there is still no real-only fold "
            "to quote, so the headline is not established as a real-data "
            "measurement."
        )
        return result

    if not _is_finite_number(result["real_only_fold_value"]):
        result["evidence"] = (
            f"{METRIC_KEY}: real_only_fold_value is present but non-finite "
            f"({result['real_only_fold_value']!r}) — FAIL loud rather than "
            "publishing a non-measurement"
        )
        return result

    if result["folds_covered"] != expected_folds or result["metals_missing"]:
        result["evidence"] = (
            f"{METRIC_KEY}: the cohort is clean (n_synthetic_rh_appended=0, "
            f"real_only_fold_value={result['real_only_fold_value']}) but the "
            f"sweep covers only {result['folds_covered']}/{expected_folds} "
            f"metals (missing: {', '.join(result['metals_missing']) or 'none'}). "
            "A 2-of-5 LOMO subset may not be quoted as the leave-one-metal-out "
            "transfer result."
        )
        return result

    result["status"] = "PASS"
    result["evidence"] = (
        f"{METRIC_KEY}: real-data headline established — headline "
        f"{result['value']} over {result['n_runs']} finite runs, "
        f"n_synthetic_rh_appended=0, real_only_fold_value="
        f"{result['real_only_fold_value']}, all {expected_folds} metals covered"
    )
    if result["dangling_index_keys"]:
        result["evidence"] += (
            f" (note: INDEX §17.1 cites absent JSON keys "
            f"{result['dangling_index_keys']})"
        )
    return result


def _remediation_sentence(result: Dict[str, Any], appended: Any) -> str:
    """Trailing evidence: what the real-only re-measurement must produce."""
    return (
        " Remediation: a real-labelled Rh replacement cohort IS on disk "
        "today (CPU) — "
        "molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json, "
        "MEASURED 500 compounds (250 active / 250 inactive) with "
        "is_synthetic=false, no_wf139_p7_carryover=true, "
        "scaffold_leak_count=0. What remains is the GPU half: re-run "
        "molmetal/scripts/train_metal_lomo_wasserstein.py --rh-cohort-json "
        "<real cohort> (or '') so n_synthetic_rh_appended=0, across all 5 "
        "metals x >=7 seeds at 50 epochs; USER_OK-gated per AGENTS.md §5.1, "
        "no launch performed by this guard."
    )


def gate_m19_lomo_headline(
    name: str,
    priority: str,
    pipeline: str,
    metric_path: Path,
    *,
    repo_root: Optional[Path] = None,
    opt_in: bool = True,
) -> Any:
    """Pre-flight gate wrapper around :func:`audit_m19_lomo_headline`.

    ``opt_in=True`` (the registered default) emits SKIP unless
    ``WF43_M19_LOMO_HEADLINE=1``, following the Wf-R35 P-A precedent used by
    ``p3_wf_r43_m16_pt_ckpt_gap``, ``p3_wf_r43_m1_paper_cohort`` and
    ``p3_wf_r43_m2_cohort_independence`` in this same wave: the gate is
    registered and auditable, but adding it does not change the PASS/FAIL
    verdict of an existing pre-flight run.
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
            f"{name}: opt-in gate — set {OPT_IN_ENV}=1 to enforce the "
            f"{METRIC_KEY} LOMO real-data headline guard (Wf-R35 P-A / "
            "Wf-R43 m16+m1+m2 opt-in pattern)",
            time.perf_counter() - t0,
        )

    result = audit_m19_lomo_headline(metric_path, repo_root=repo_root)
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
    "EXPECTED_METALS_IN_POOL",
    "GATE_NAME",
    "INDEX_CITED_KEYS",
    "METRIC_KEY",
    "OPT_IN_ENV",
    "audit_m19_lomo_headline",
    "gate_m19_lomo_headline",
]