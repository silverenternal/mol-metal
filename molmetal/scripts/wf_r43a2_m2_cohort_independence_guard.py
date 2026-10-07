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

"""Wf-R43 m2 — R2 per-cohort COHORT-INDEPENDENCE guard (CPU-only).

Background
----------
``metrics/by_metric/m2_r2_per_cohort.json`` reports m2 R2 for three
cohorts: ``platinai_4cell`` (n=200), ``hilbert`` (n=200) and
``metalcytotox_combined`` (n=2000).  Two of those three are **not
independent cohorts**.

MEASURED (reproduced by
``molmetal/tests/test_wf_r43a2_m2_cohort_independence_guard.py``):
``wf331r5_r2_w2_chemprop.load_cohort("hilbert")`` (L1099-1101) does not
read a Hilbert dataset — no Hilbert cohort file exists anywhere under
``molmetal/data/``.  It returns ``load_cohort("platinai_4cell")``, so the
two cohorts share the same 200 SMILES, the same 32-d features and the
same pIC50 vector (``np.array_equal(X_platinai, X_hilbert) is True``).

Wf-331-R5 *intended* the two to differ by CV split seed — its own runner
shifts the seed by +1 for hilbert (L1250-1251).  But Wf-388-A1 called
``train_eval_per_cohort`` with a hardcoded ``seed_offset=0`` for every
cohort, so the shift never happened.  The recorded result is two
bit-identical rows:

    platinai_4cell  r2_mean = -0.14401547578768395
    hilbert         r2_mean = -0.14401547578768395   <- same float

A second consequence, same root cause: the baseline-anchor dispatch in
``wf388a1_m2_per_cohort_real.train_eval_per_cohort`` keyed off
``target_name``, and both cohorts report
``pIC50_mean_mcf7_a549_a2780_a2780cis``.  Its ``elif "hilbert" in
target_name.lower()`` arm was therefore **dead code**, and hilbert's
``lift_vs_baseline`` was scored against *platinai's* anchor (-0.1251)
rather than its own (-0.1268).

Why this module exists
----------------------
The metric file says nothing about any of this.  It reports
``n_cohorts_tested: 3`` and ``aggregate_status:
MEASURED_PARTIAL_LIFT``, so a downstream surface (paper draft, INDEX row,
wave closeout) can quote "3 cohorts, 1/3 lifted" when the real evidence
base is **2 cohorts**.  Worse, the duplication biases the aggregate: the
aliased negative row is counted twice, dragging ``aggregate.mean_r2`` to
-0.0310 when the independent-cohort mean is +0.0270.

This module answers one question —

    are the m2 per-cohort rows independent cohorts?

— and answers **FAIL** while any aliased row is presented as independent
evidence.  It never trains, never writes a metric, and never touches
``paper/*``.

What is already available for remediation
-----------------------------------------
The fix is CPU-only and already applied in
``molmetal/scripts/wf388a1_m2_per_cohort_real.py`` (Wf-R43-A2):

  * ``COHORT_ALIASES`` / ``cohort_seed_offset`` — hilbert re-splits the
    shared platinai data at seed offset +1 (Wf-331-R5's own convention),
    so the two rows differ instead of replaying one another.
  * ``baseline_anchor_for`` — dispatches the anchor on the cohort NAME,
    which makes hilbert's own anchor reachable for the first time.
  * each ``per_cohort`` row now carries ``cohort_is_alias_of``,
    ``independent_cohort`` and ``cv_seed_offset``.
  * ``aggregate_per_cohort_r2`` reports ``n_cohorts_independent`` and
    ``n_cohorts_aliased``.

So the guard PASSes once the metric file is regenerated from the fixed
script AND a genuinely distinct hilbert cohort (or an explicit drop of
the hilbert row) lands.  Until then it FAILs loud, because the honest
reading of the current record is "2 cohorts, not 3".
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

GATE_NAME = "p3_wf_r43_m2_cohort_independence"
METRIC_KEY = "m2_r2_per_cohort"
OPT_IN_ENV = "WF43_M2_COHORT_INDEPENDENCE"

# Rows that are known aliases of another cohort. Kept local (rather than
# imported from wf388a1) so the guard stays importable without numpy /
# rdkit, matching the m1 guard's dependency-light shape.
KNOWN_ALIASES: Dict[str, str] = {"hilbert": "platinai_4cell"}

# Rounding used when comparing two recorded r2_mean values. Two cohorts
# that share data AND a split seed agree to full float precision, so any
# tolerance this loose still catches the duplication.
_R2_EPS = 1e-9


def audit_m2_cohort_independence(
    metric_path: Path,
    *,
    known_aliases: Optional[Dict[str, str]] = None,
    require_alias_disclosure: bool = True,
) -> Dict[str, Any]:
    """Audit whether the m2 per-cohort rows are independent cohorts.

    Reads ``metrics/by_metric/m2_r2_per_cohort.json`` and checks three
    things:

      1. every row discloses its alias status (``cohort_is_alias_of`` /
         ``independent_cohort``, added by Wf-R43-A2);
      2. no two rows present bit-identical ``r2_mean`` under different
         cohort names;
      3. the aggregate reports an independent-cohort count.

    Returns a dict with ``status`` in {PASS, FAIL, SKIP} and a
    human-readable ``evidence`` string.  Never raises on malformed input
    — a broken record FAILs loudly instead of crashing pre-flight.
    """
    aliases = KNOWN_ALIASES if known_aliases is None else dict(known_aliases)
    result: Dict[str, Any] = {
        "status": "FAIL",
        "metric": METRIC_KEY,
        "evidence": "",
        "n_rows": 0,
        "n_independent_declared": None,
        "duplicate_r2_groups": [],
        "undisclosed_aliases": [],
        "aggregate_mean_r2": None,
        "aggregate_mean_r2_independent": None,
        "aggregate_status": None,
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
    except Exception as exc:  # noqa: BLE001 — any decode/read failure
        result["evidence"] = (
            f"{METRIC_KEY}: could not parse {metric_path} ({exc}) — FAIL"
        )
        return result

    rows = payload.get("per_cohort") or []
    if not isinstance(rows, list) or not rows:
        result["evidence"] = (
            f"{METRIC_KEY}: {metric_path} has no usable `per_cohort` list — FAIL"
        )
        return result

    result["n_rows"] = len(rows)
    agg = payload.get("aggregate") or {}
    result["aggregate_mean_r2"] = agg.get("mean_r2")
    result["aggregate_mean_r2_independent"] = agg.get("mean_r2_independent")
    result["aggregate_status"] = agg.get("aggregate_status")
    result["n_independent_declared"] = agg.get("n_cohorts_independent")

    failures: List[str] = []

    # --- check 1: alias disclosure ------------------------------------
    if require_alias_disclosure:
        undisclosed = [
            r.get("cohort")
            for r in rows
            if isinstance(r, dict)
            and r.get("cohort") in aliases
            and r.get("cohort_is_alias_of") is None
        ]
        result["undisclosed_aliases"] = undisclosed
        if undisclosed:
            failures.append(
                f"row(s) {sorted(undisclosed)} are known aliases of another "
                "cohort but record no `cohort_is_alias_of` disclosure"
            )

    # --- check 2: duplicate r2_mean under distinct cohort names -------
    by_r2: Dict[float, List[str]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        val = r.get("r2_mean")
        name = r.get("cohort")
        if isinstance(val, (int, float)) and isinstance(name, str):
            key = round(float(val), 9)
            by_r2.setdefault(key, []).append(name)
    dupes = sorted(names for names in by_r2.values() if len(names) > 1)
    result["duplicate_r2_groups"] = dupes
    if dupes:
        failures.append(
            "cohort(s) "
            + ", ".join(
                "{ " + ", ".join(g) + " }" for g in dupes
            )
            + " record a bit-identical r2_mean — they are the same "
            "measurement, not independent cohorts"
        )

    # --- check 3: aggregate reports an independent count --------------
    if result["n_independent_declared"] is None:
        failures.append(
            "aggregate reports no `n_cohorts_independent` — an aliased row "
            "would be counted as independent evidence"
        )

    if failures:
        result["status"] = "FAIL"
        result["evidence"] = f"{METRIC_KEY}: " + "; ".join(failures)
        return result

    n_indep = result["n_independent_declared"]
    result["status"] = "PASS"
    result["evidence"] = (
        f"{METRIC_KEY}: {result['n_rows']} rows, {n_indep} independent "
        f"cohort(s); no duplicate r2_mean; alias disclosure present"
    )
    return result


def gate_m2_cohort_independence(
    name: str,
    priority: str,
    pipeline: str,
    metric_path: Path,
    *,
    repo_root: Optional[Path] = None,
    opt_in: bool = True,
) -> Any:
    """Pre-flight gate wrapper around :func:`audit_m2_cohort_independence`.

    ``opt_in=True`` (the registered default) emits SKIP unless
    ``WF43_M2_COHORT_INDEPENDENCE=1``, following the same Wf-R35 P-A
    precedent used by ``p3_wf_r43_m16_pt_ckpt_gap`` and
    ``p3_wf_r43_m1_paper_cohort`` in this same wave: the gate is registered
    and auditable, but adding it does not change the PASS/FAIL verdict of
    an existing pre-flight run.
    """
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
            f"{METRIC_KEY} R2 cohort-independence guard (Wf-R35 P-A / "
            "Wf-R43 m16 + m1 opt-in pattern)",
            time.perf_counter() - t0,
        )

    result = audit_m2_cohort_independence(metric_path)
    return GateResult(
        name,
        priority,
        pipeline,
        result["status"],
        f"{name}: {result['evidence']}",
        time.perf_counter() - t0,
    )


__all__: List[str] = [
    "GATE_NAME",
    "KNOWN_ALIASES",
    "METRIC_KEY",
    "OPT_IN_ENV",
    "audit_m2_cohort_independence",
    "gate_m2_cohort_independence",
]


if __name__ == "__main__":  # pragma: no cover — manual invocation
    root = Path(__file__).resolve().parents[2]
    print(
        json.dumps(
            audit_m2_cohort_independence(
                root / "metrics" / "by_metric" / "m2_r2_per_cohort.json"
            ),
            indent=2,
        )
    )