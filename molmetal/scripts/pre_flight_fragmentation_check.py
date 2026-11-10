#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wf-A128 — a STANDALONE fragmentation pre-flight gate (CPU-only).

WHY THIS EXISTS
---------------
The fragmentation verdict IS computed in this repository, but it is a REPORT
COLUMN, not a launch gate:

* ``molmetal/scripts/wf_cfm_run_census.py`` builds it at
  ``_fragmentation_verdict`` (the verdict builder) and records it per run in
  ``metrics/by_metric/cfm_run_census.json`` under ``fragmentation_gate`` (the
  run's last epoch) and ``fragmentation_gate_last_measured`` (the last epoch
  that carried a measurement);
* ``molmetal/scripts/cfm_fragmentation_gate.py`` holds the thresholds
  (``DEFAULT_MAX_FRAGMENTED_RATIO = 0.50``,
  ``DEFAULT_MIN_LARGEST_FRAGMENT_FRAC = 0.90``) and a standalone pool scorer;
* ``molmetal/scripts/pre_flight_check.py`` has NO fragmentation check at all,
  and the module-gate ``gate_m11`` needs ``--pool-csv`` passed explicitly.

THE TALLY IS RECOMPUTED AT RUN TIME — THIS MODULE STATES NO COUNTS
------------------------------------------------------------------
The verdict counts depend on WHICH artifact the gate is pointed at, and this
repository holds more than one shape of the census artifact:

* the WORKING-TREE artifact ``metrics/by_metric/cfm_run_census.json`` — the
  one this module reads by default — carries a verdict block per run;
* the artifact at ``HEAD`` may carry FEWER runs and NO verdict block at all,
  in which case every run is SKIP / ``field_absent`` and the overall verdict
  is SKIP;
* a REGENERATED artifact (the census builder re-run) applies the Wf-A127
  nulling rule, so a run whose last epoch scored no rows publishes SKIP /
  ``denominator_empty`` instead of a FAIL on a placeholder ``0.0``, and
  publishes ``fragmentation_gate_last_measured`` BESIDE it.

So the honest way to quote a tally is to RUN the gate and read the report:

    .venv/bin/python molmetal/scripts/pre_flight_fragmentation_check.py \
        --census metrics/by_metric/cfm_run_census.json --out /tmp/frag.json

WHAT THE GATE IS FOR
--------------------
A verdict that is recorded but never consulted consumes NOTHING: a run can
train for hours and publish a FAIL pool with no gate in the way.  That is
the hole this file closes.

THE ONE RULE THAT MATTERS MOST
-------------------------------
**A gate that FAILs on missing data is worse than no gate** — it is exactly
how this repository ended up with 27 meaningless FAILs.  So this check:

* **SKIPs** (never FAILs) when its input is absent (``field_absent``);
* **SKIPs** (never FAILs) when the measurement's denominator is empty
  (``denominator_empty``) — the eval drew nothing, or it drew samples and
  scored none of them;
* **FAILs** only on a rate that was actually measured.

THE BASIS VOCABULARY IS SHARED BY CONSTRUCTION
-----------------------------------------------
The verdict is produced by the census builder's OWN function
(``molmetal.scripts.wf_cfm_run_census._fragmentation_verdict``), not by a
copy of it, so the four basis values — ``field_absent`` /
``denominator_empty`` / ``end_to_end`` / ``conditional`` — and the two
thresholds cannot drift apart from the artifact they are compared against.
The thresholds are imported from ``cfm_fragmentation_gate`` by that function,
so they are the SAME two constants.

A basis OUTSIDE that vocabulary is not a crash: it is reported as its own
basis and the gate keeps going.  A gate that raises on an unexpected field is
a gate that cannot be pointed at an artifact it has not seen before.

THE LAST MEASURED EPOCH, NOT THE LAST EPOCH
-------------------------------------------
A run's LAST epoch is not always the epoch that carries a measurement: the
generation eval can draw samples and score NONE of them, and the rate over
the scored rows is then 0/0 — undefined, not 0.0.  The census builder
publishes ``fragmentation_gate_last_measured`` beside the last epoch's
verdict for exactly that case, so that "nothing measured is lost when the
run's last epoch is unmeasured".  This gate reads BOTH blocks and judges the
last MEASURED epoch, so a measured FAIL is never hidden behind an unmeasured
last epoch.  In ``--run-dir`` epoch mode the same rule is applied to the
run's own ``per_epoch_metrics.csv``: the gate falls back to the most recent
epoch that carries a non-empty denominator.

WHAT IT READS
-------------
``--pool-csv PATH``
    One decoded pool (``generated_smiles.csv``).  The pool is RE-SCORED with
    the trainer's own ``decode_chemistry`` (via
    ``cfm_fragmentation_gate.score_smiles_pool``), so the fragment definition
    cannot drift from the one the metrics use.

``--run-dir PATH``
    A run directory: its ``generated_smiles.csv`` if it has one, otherwise its
    ``per_epoch_metrics.csv`` (the trainer's own columns).

``--census PATH``
    A ``cfm_run_census.json`` artifact: every run in it is re-scored and the
    per-run verdicts are aggregated.

THE DENOMINATOR IS STATED, ALWAYS
----------------------------------
A pool file holds the molecules that SURVIVED, so a rate over its rows is a
rate over the survivors.  The end-to-end rate is over the number of samples
the pool was DRAWN from.  When that count is not available the end-to-end
denominator is the pool's own row count and says so; when it IS available but
comes from ANOTHER run's artifact (the free-generation panel), the verdict
falls back to the CONDITIONAL rate over the pool's own parsed rows and the
foreign denominator is named in the reasons.  A rate over somebody else's
population is not this pool's rate.

USAGE
-----
    # one pool
    .venv/bin/python molmetal/scripts/pre_flight_fragmentation_check.py \
        --pool-csv molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv

    # every run in the census artifact
    .venv/bin/python molmetal/scripts/pre_flight_fragmentation_check.py \
        --census metrics/by_metric/cfm_run_census.json

    # no mode at all -> the repository's own census artifact
    # (metrics/by_metric/cfm_run_census.json)
    .venv/bin/python molmetal/scripts/pre_flight_fragmentation_check.py

    # machine-readable report
    .venv/bin/python molmetal/scripts/pre_flight_fragmentation_check.py \
        --run-dir molmetal/reports/cfm_c7_fullgen_2026_10_20 \
        --out /tmp/fragmentation_report.json

CPU-only and RDKit-only.  No GPU, no model, no sampler.  Exit codes follow
the project's gate convention: 0 = PASS, 1 = FAIL, 2 = SKIP.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: The census builder's basis vocabulary.  The verdict itself is produced by
#: the census builder's own function, so these four values are the artifact's
#: values, not a re-statement of them.
BASIS_VOCABULARY = ("field_absent", "denominator_empty", "end_to_end", "conditional")

#: Exit codes — the same convention as the project's other gates.
EXIT_PASS, EXIT_FAIL, EXIT_SKIP = 0, 1, 2

#: The repository's own census artifact — the DEFAULT the gate reads when
#: no mode is named on the command line.  Resolved from this file's own
#: location, so it points at the repo the script lives in.
DEFAULT_CENSUS = REPO / "metrics" / "by_metric" / "cfm_run_census.json"


def _verdict_builder():
    """The census builder's OWN verdict function.

    Imported lazily so this module stays importable without the census's
    module-level work, and so the basis vocabulary is shared by
    CONSTRUCTION: this is the same function object the census artifact was
    built with, not a re-implementation of it.
    """
    from molmetal.scripts.wf_cfm_run_census import _fragmentation_verdict

    return _fragmentation_verdict


def _reasons_of(verdict: Dict[str, Any]) -> List[str]:
    """Every reason string a verdict carries, in one list.

    The census builder emits ``reason`` (singular) on the SKIP path and
    ``reasons`` (plural) on the measured path; a caller must not have to
    know which.
    """
    out: List[str] = []
    single = verdict.get("reason")
    if isinstance(single, str) and single:
        out.append(single)
    many = verdict.get("reasons")
    if isinstance(many, (list, tuple)):
        out.extend(str(r) for r in many)
    return out


def _skip(
    base: Dict[str, Any],
    basis: str,
    reason: str,
    **extra: Any,
) -> Dict[str, Any]:
    """A SKIP verdict with an explicit basis.

    SKIP is the answer for "there is nothing to judge", never FAIL — see the
    module docstring.

    A basis OUTSIDE the shared vocabulary is DEGRADED, never fatal: it is
    reported as its own basis, flagged with ``basis_in_vocabulary: False``,
    and named in the reasons.  Asserting the vocabulary here would make the
    gate crash on an artifact that carries a field this module has not seen
    — which is the opposite of what a gate is for.
    """
    basis = str(basis)
    in_vocabulary = basis in BASIS_VOCABULARY
    reasons = [reason]
    if not in_vocabulary:
        reasons.append(
            f"the basis {basis!r} is NOT one of the census builder's "
            f"vocabulary ({', '.join(BASIS_VOCABULARY)}); it is reported as "
            f"its own basis instead of crashing the gate"
        )
    base.update(
        {
            "status": "SKIP",
            "basis": basis,
            "basis_in_vocabulary": in_vocabulary,
            "reasons": reasons,
            "measured_rate": None,
            "fragmented_ratio": None,
        }
    )
    base.update(extra)
    return base


def _measured(
    base: Dict[str, Any],
    *,
    single_fragment_ratio: float,
    largest_fragment_frac: Optional[float],
    validity: Optional[float],
    field_present: bool,
    gen_validity_n: Optional[int] = None,
    gen_validity_val: Optional[int] = None,
    extra_reasons: Optional[Sequence[str]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """A verdict on a MEASURED rate, via the census builder's own function.

    ``validity`` is ``None`` when the run carries no parse rate, which makes
    the verdict CONDITIONAL — the number over the parseable rows only.  That
    is the builder's own vocabulary and it is not the same number as the
    end-to-end one, so the basis travels with the verdict.
    """
    verdict = _verdict_builder()(
        single_fragment_ratio,
        largest_fragment_frac,
        validity,
        field_present,
        gen_validity_n=gen_validity_n,
        gen_validity_val=gen_validity_val,
    )
    reasons = _reasons_of(verdict)
    for r in extra_reasons or ():
        reasons.append(str(r))
    basis = verdict.get("basis")
    base.update(
        {
            "status": verdict.get("status"),
            "basis": basis,
            "basis_in_vocabulary": basis in BASIS_VOCABULARY,
            "reasons": reasons,
            "measured_rate": verdict.get("measured_rate"),
            "fragmented_ratio": verdict.get("fragmented_ratio"),
            "single_fragment_ratio_conditional": verdict.get(
                "single_fragment_ratio_conditional"
            ),
            "thresholds": verdict.get("thresholds"),
        }
    )
    base.update(extra)
    return base


def _denominator_is_empty(validity: Optional[float]) -> bool:
    """True when the parse rate says the measurement's denominator is empty.

    ``gen_validity`` is the parse rate over the samples the eval DREW; a
    value of exactly 0.0 means the eval scored NONE of them, so every
    chemistry rate over those rows is 0/0 and UNDEFINED.  The trainer's own
    convention since Wf-A114/T85 is that such a rate is ``None``, never
    ``0.0`` — and the census's own normalisation applies that rule to the
    rows it reads.  This check reads RAW artifacts, so it applies the same
    rule here.

    MEASURED 2026-10-31 on the WORKING-TREE artifact
    ``metrics/by_metric/cfm_run_census.json``: two runs carry
    ``gen_validity_end == 0.0`` — ``cfm_bond_head_30ep_onecycle_2026_10_09``
    and ``cfm_fm_control_2026_10_09`` — and that artifact publishes BOTH as a
    FAIL on a placeholder ``0.0``.  A REGENERATED artifact (the census
    builder re-run under the Wf-A127 nulling rule) publishes them as SKIP /
    ``denominator_empty`` instead, and publishes
    ``fragmentation_gate_last_measured`` beside them.  This check applies
    the same rule to BOTH shapes, so the verdict does not depend on which
    artifact the gate is pointed at.
    """
    if validity is None:
        return False
    try:
        return float(validity) == 0.0
    except (TypeError, ValueError):
        return False


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    """One numeric cell of a CSV row, or None when absent / unparseable."""
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _epoch_row_measured(row: Dict[str, str]) -> bool:
    """True when this epoch row carries a rate with a NON-EMPTY denominator.

    This is the census builder's OWN rule
    (``molmetal.scripts.wf_cfm_run_census._normalise_row``): a chemistry
    rate is undefined when the generation eval drew NOTHING
    (``gen_validity_n == 0``) or scored no rows (``gen_validity_val == 0``).
    A row that does not carry the count column at all says nothing about
    its denominator, so its value is kept — the same "only PROVED emptiness
    nulls a value" rule the census applies.

    Used to find the LAST epoch that carries a measurement, so that a run
    whose last epoch is unmeasured is judged on the epoch that WAS measured
    instead of being reported as "nothing to judge".
    """
    if _f(row, "gen_single_fragment_ratio") is None:
        return False
    n = _f(row, "gen_validity_n")
    if n is not None and n <= 0:
        return False
    v = _f(row, "gen_validity_val")
    if v is not None and v <= 0:
        return False
    return not _denominator_is_empty(_f(row, "gen_validity"))


def _csv_connectivity_claims(
    rows: List[Dict[str, str]], decode_chemistry
) -> Dict[str, Any]:
    """What the pool's OWN connectivity columns claim about rows RDKit refuses.

    MEASURED 2026-10-31: the pool
    ``molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`` (12,669
    rows) has exactly **1** row RDKit cannot parse, and that row carries
    ``n_frags_raw = 1`` and ``largest_frag_frac = 1.0`` — a fabricated
    "perfect single fragment" on a string that is not a molecule at all.  The
    trainer now writes the sentinel ``-1`` / ``-1.0`` for such rows
    (``molmetal/scripts/train_cfm_yuelbond_wasserstein.py``), so the pools
    written AFTER that fix are honest; the older ones are not.

    This is an ADVISORY: it is measured and reported, and it does NOT change
    the verdict.  The verdict is computed from the re-scored pool, never from
    the CSV's own columns.
    """
    out: Dict[str, Any] = {
        "has_n_frags_raw_column": bool(rows) and "n_frags_raw" in (rows[0] or {}),
        "n_rows": len(rows),
        "n_sentinel_unknown": 0,
        "n_claimed_single_fragment_on_unparseable": 0,
        "examples": [],
    }
    if not out["has_n_frags_raw_column"]:
        return out
    for i, r in enumerate(rows):
        n_frags = _f(r, "n_frags_raw")
        if n_frags is None:
            continue
        if n_frags < 0:
            out["n_sentinel_unknown"] += 1
            continue
        smiles = (r.get("smiles") or "").strip()
        if not smiles:
            continue
        if decode_chemistry(smiles) is not None:
            continue
        out["n_claimed_single_fragment_on_unparseable"] += 1
        if len(out["examples"]) < 5:
            out["examples"].append(
                {
                    "row_index": i,
                    "n_frags_raw": n_frags,
                    "largest_frag_frac": _f(r, "largest_frag_frac"),
                    "smiles_head": smiles[:120],
                }
            )
    out["note"] = (
        "rows whose 'n_frags_raw' claims a fragment count but whose SMILES "
        "RDKit refuses: the claim is not knowable for those rows.  The "
        "trainer writes the sentinel -1 / -1.0 for them since Wf-A114/T85; "
        "older pools still carry the fabricated 1 / 1.0.  ADVISORY ONLY — the "
        "verdict below is computed from the re-scored pool, never from these "
        "columns."
    )
    return out


def check_pool_csv(csv_path: Path) -> Dict[str, Any]:
    """Score one decoded pool and return a verdict block."""
    from molmetal.scripts.cfm_fragmentation_gate import (
        DEFAULT_MAX_FRAGMENTED_RATIO,
        DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
        pool_attempts,
        score_smiles_pool,
    )

    path = Path(csv_path)
    base: Dict[str, Any] = {
        "mode": "pool_csv",
        "path": str(path),
        "exists": path.is_file(),
        "thresholds": {
            "max_fragmented_ratio": DEFAULT_MAX_FRAGMENTED_RATIO,
            "min_largest_fragment_frac": DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
        },
    }
    if not path.is_file():
        return _skip(
            base,
            "field_absent",
            f"the pool CSV {path} does not exist, so there is no "
            f"fragmentation measurement to judge",
        )
    try:
        rows = list(
            csv.DictReader(path.open(newline="", encoding="utf-8", errors="replace"))
        )
    except OSError as exc:
        return _skip(
            base,
            "field_absent",
            f"the pool CSV {path} could not be read ({type(exc).__name__}: "
            f"{exc}), so there is no fragmentation measurement to judge",
        )

    smiles = [(r.get("smiles") or "") for r in rows]
    attempts = pool_attempts(path)
    n_attempts = int(attempts["n_attempted"]) if attempts else None
    attempts_source = str(attempts["source"]) if attempts else None
    pool = score_smiles_pool(
        smiles,
        n_attempts=n_attempts,
        attempts_source=attempts_source,
    )
    base["pool"] = {
        "n_input": pool.get("n_input"),
        "n_scored": pool.get("n_scored"),
        "n_empty": pool.get("n_empty"),
        "n_unparseable": pool.get("n_unparseable"),
        "n_single": pool.get("n_single"),
        "single_fragment_ratio": pool.get("single_fragment_ratio"),
        "single_molecule_rate": pool.get("single_molecule_rate"),
        "largest_fragment_frac_mean": pool.get("largest_fragment_frac_mean"),
        "n_fragments_mean": pool.get("n_fragments_mean"),
        "end_to_end_denominator": pool.get("end_to_end_denominator"),
        "end_to_end_denominator_source": pool.get("end_to_end_denominator_source"),
    }
    base["pool_attempts_source"] = attempts_source

    n_input = int(pool.get("n_input") or 0)
    n_scored = int(pool.get("n_scored") or 0)
    if n_input == 0:
        return _skip(
            base,
            "field_absent",
            f"the pool {path} holds no rows, so no connectivity measurement "
            f"was ever recorded for it",
        )
    if n_scored == 0:
        return _skip(
            base,
            "denominator_empty",
            f"the pool {path} holds {n_input} row(s) but RDKit parsed none "
            f"of them, so the connectivity rate's denominator is empty "
            f"(0/{n_input}) and the rate is undefined, not 0.0",
        )

    denominator_source = str(pool.get("end_to_end_denominator_source") or "")
    own_rows = denominator_source.startswith("the pool file's own rows")
    extra: List[str] = []
    validity: Optional[float] = None
    if own_rows:
        # The end-to-end rate over the pool's OWN rows:
        # (n_scored / n_input) * (n_single / n_scored) == n_single / n_input.
        validity = n_scored / n_input
    else:
        extra.append(
            f"the end-to-end denominator ({pool.get('end_to_end_denominator')}) "
            f"comes from ANOTHER run's artifact ({denominator_source}), so it "
            f"is not this pool's own population: the verdict is made on the "
            f"CONDITIONAL rate over the pool's own parsed rows"
        )
    verdict = _measured(
        base,
        single_fragment_ratio=float(pool["single_fragment_ratio"]),
        largest_fragment_frac=pool.get("largest_fragment_frac_mean"),
        validity=validity,
        field_present=True,
        extra_reasons=extra,
        single_molecule_rate=pool.get("single_molecule_rate"),
        end_to_end_denominator=pool.get("end_to_end_denominator"),
        end_to_end_denominator_source=denominator_source,
    )
    # The cross-check against the pool's own columns + the advisory on the
    # rows the columns misdescribe.
    from molmetal.scripts.cfm_fragmentation_gate import _trainer_decode_chemistry

    decode_chemistry = _trainer_decode_chemistry()
    mism = 0
    checked = 0
    per_row = pool.get("rows") or []
    row_indexes = pool.get("row_indexes") or []
    for chem, i in zip(per_row, row_indexes):
        csv_frag = _f(rows[i], "n_frags_raw")
        if csv_frag is None:
            continue
        checked += 1
        if abs(csv_frag - float(chem.get("n_fragments", 0) or 0)) > 1e-9:
            mism += 1
    base["csv_column_cross_check"] = {"n_checked": checked, "n_mismatch": mism}
    base["csv_connectivity_claims"] = _csv_connectivity_claims(rows, decode_chemistry)
    return verdict


def check_run_dir(run_dir: Path) -> Dict[str, Any]:
    """Score a run directory: its pool if it has one, else its epoch columns."""
    d = Path(run_dir)
    base: Dict[str, Any] = {"mode": "run_dir", "path": str(d), "exists": d.is_dir()}
    if not d.is_dir():
        return _skip(
            base,
            "field_absent",
            f"the run directory {d} does not exist, so there is no "
            f"fragmentation measurement to judge",
        )
    pool_csv = d / "generated_smiles.csv"
    if pool_csv.is_file():
        out = check_pool_csv(pool_csv)
        out["mode"] = "run_dir"
        out["run_dir"] = str(d)
        return out
    metrics_csv = d / "per_epoch_metrics.csv"
    if not metrics_csv.is_file():
        return _skip(
            base,
            "field_absent",
            f"the run directory {d} holds neither generated_smiles.csv nor "
            f"per_epoch_metrics.csv, so there is no fragmentation "
            f"measurement to judge",
        )
    return check_epoch_columns(metrics_csv)


def _row_facts(row: Dict[str, str]) -> Dict[str, Any]:
    """The six numbers of one epoch row, in one shape."""
    n_drawn = _f(row, "gen_validity_n")
    n_scored = _f(row, "gen_validity_val")
    return {
        "epoch": _f(row, "epoch"),
        "gen_validity": _f(row, "gen_validity"),
        "gen_validity_n": int(n_drawn) if n_drawn is not None else None,
        "gen_validity_val": int(n_scored) if n_scored is not None else None,
        "gen_single_fragment_ratio": _f(row, "gen_single_fragment_ratio"),
        "gen_largest_fragment_frac": _f(row, "gen_largest_fragment_frac"),
    }


def check_epoch_columns(metrics_csv: Path) -> Dict[str, Any]:
    """Score a ``per_epoch_metrics.csv`` on the trainer's columns.

    The trainer's columns are the same three numbers the census reads:
    ``gen_single_fragment_ratio`` (over the PARSED rows),
    ``gen_largest_fragment_frac`` and ``gen_validity`` (the parse rate over
    the DRAWN rows).

    THE LAST MEASURED EPOCH, NOT THE LAST EPOCH
    ---------------------------------------------
    The last row of the file is not always the last row that carries a
    measurement: the generation eval can draw samples and score NONE of
    them, and the trainer then writes its ``0.0`` placeholder.  Judging
    only the last row would report "nothing to judge" for a run that DID
    measure a connectivity rate at an earlier epoch — a measured FAIL
    hidden behind an unmeasured epoch.  So the gate walks BACK from the
    last row and takes the most recent epoch whose denominator is not
    empty, and says which epoch it judged.
    """
    from molmetal.scripts.cfm_fragmentation_gate import (
        DEFAULT_MAX_FRAGMENTED_RATIO,
        DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
    )

    path = Path(metrics_csv)
    base: Dict[str, Any] = {
        "mode": "epoch_columns",
        "path": str(path),
        "exists": path.is_file(),
        "thresholds": {
            "max_fragmented_ratio": DEFAULT_MAX_FRAGMENTED_RATIO,
            "min_largest_fragment_frac": DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
        },
    }
    if not path.is_file():
        return _skip(
            base,
            "field_absent",
            f"the per-epoch metrics CSV {path} does not exist, so there is "
            f"no fragmentation measurement to judge",
        )
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        return _skip(
            base,
            "field_absent",
            f"the per-epoch metrics CSV {path} could not be read "
            f"({type(exc).__name__}: {exc})",
        )
    if not rows:
        return _skip(
            base,
            "field_absent",
            f"the per-epoch metrics CSV {path} holds no row, so no "
            f"fragmentation measurement was ever recorded for it",
        )
    base["n_rows"] = len(rows)
    last_row = rows[-1]
    judged = None
    for candidate in reversed(rows):
        if _epoch_row_measured(candidate):
            judged = candidate
            break
    used_last_measured = judged is not None and judged is not last_row
    row = judged if judged is not None else last_row
    base["last_row"] = _row_facts(last_row)
    if used_last_measured:
        base["epoch_source"] = (
            "the LAST epoch that carried a measured denominator (the file's "
            "last row carries an empty denominator)"
        )
    elif judged is not None:
        base["epoch_source"] = "the file's last row (it carries a measurement)"
    else:
        base["epoch_source"] = (
            "the file's last row (NO epoch in the file carries a measured "
            "denominator)"
        )
    header = set(row.keys())
    single = _f(row, "gen_single_fragment_ratio")
    largest = _f(row, "gen_largest_fragment_frac")
    validity = _f(row, "gen_validity")
    n_drawn = _f(row, "gen_validity_n")
    n_scored = _f(row, "gen_validity_val")
    field_present = "gen_single_fragment_ratio" in header
    base["row"] = _row_facts(row)
    if single is None:
        if not field_present:
            return _skip(
                base,
                "field_absent",
                f"the row of {path} carries no 'gen_single_fragment_ratio' "
                f"column, so the run never recorded a connectivity rate",
            )
        return _skip(
            base,
            "denominator_empty",
            f"the row of {path} carries 'gen_single_fragment_ratio' but its "
            f"value is undefined (gen_validity_n="
            f"{int(n_drawn) if n_drawn is not None else None}, "
            f"gen_validity_val="
            f"{int(n_scored) if n_scored is not None else None}), so the "
            f"rate's denominator is empty",
        )
    if _denominator_is_empty(validity):
        return _skip(
            base,
            "denominator_empty",
            f"the row of {path} carries gen_validity == 0.0, so the eval "
            f"scored NONE of the samples it drew (gen_validity_n="
            f"{int(n_drawn) if n_drawn is not None else None}, "
            f"gen_validity_val="
            f"{int(n_scored) if n_scored is not None else None}) and the "
            f"chemistry rate is 0/0 — undefined, not 0.0",
        )
    verdict = _measured(
        base,
        single_fragment_ratio=float(single),
        largest_fragment_frac=largest,
        validity=validity,
        field_present=True,
        gen_validity_n=int(n_drawn) if n_drawn is not None else None,
        gen_validity_val=int(n_scored) if n_scored is not None else None,
    )
    if used_last_measured:
        verdict["reasons"] = list(verdict.get("reasons") or []) + [
            f"the verdict is made at epoch {base['row']['epoch']}, the LAST "
            f"epoch that carried a measured denominator; the file's last row "
            f"(epoch {base['last_row']['epoch']}) carries an empty one, so "
            f"nothing measured is lost"
        ]
    return verdict


def _recorded_measured_block(
    name: str,
    recorded: Dict[str, Any],
    last_measured_block: Dict[str, Any],
    source: str,
    current_thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    """The artifact's own verdict at the run's LAST MEASURED epoch.

    The census builder publishes ``fragmentation_gate_last_measured`` so that
    "nothing measured is lost when the run's last epoch is unmeasured"
    (``molmetal/scripts/wf_cfm_run_census.py``).  That block was produced by
    the census builder's OWN ``_fragmentation_verdict`` on a rate whose
    denominator is NOT empty, so it is already a judgeable verdict in the
    shared vocabulary — it is used as it stands, because the block does not
    carry the ``largest_fragment_frac`` the threshold check needs and
    re-deriving it would mean re-implementing the builder.

    The thresholds the block was built with are compared against the ones in
    force now, and a mismatch is reported rather than hidden.
    """
    reasons = _reasons_of(last_measured_block)
    basis = last_measured_block.get("basis")
    recorded_thresholds = last_measured_block.get("thresholds") or {}
    if recorded_thresholds and recorded_thresholds != current_thresholds:
        reasons.append(
            f"the artifact's verdict was built with the thresholds "
            f"{recorded_thresholds}, and the thresholds in force now are "
            f"{current_thresholds}; the artifact's verdict is reported as it "
            f"stands"
        )
    reasons.append(
        f"the run's LAST epoch carries no judgeable rate (verdict "
        f"{recorded.get('status')} / {recorded.get('basis')}), so the verdict "
        f"reported here is the one at the last MEASURED epoch "
        f"(epoch {last_measured_block.get('epoch')})"
    )
    return {
        "run": name,
        "recorded_status": recorded.get("status"),
        "recorded_basis": recorded.get("basis"),
        "status": last_measured_block.get("status"),
        "basis": basis,
        "basis_in_vocabulary": basis in BASIS_VOCABULARY,
        "reasons": reasons,
        "measured_rate": last_measured_block.get("measured_rate"),
        "fragmented_ratio": last_measured_block.get("fragmented_ratio"),
        "single_fragment_ratio_conditional": last_measured_block.get(
            "single_fragment_ratio_conditional"
        ),
        "thresholds": last_measured_block.get("thresholds"),
        "epoch": last_measured_block.get("epoch"),
        "resourced_from": source,
    }


def check_census(census_path: Path) -> Dict[str, Any]:
    """Re-score every run in a ``cfm_run_census.json`` artifact.

    The artifact's OWN verdict block is the record of what the census builder
    decided, and it is the ONLY place the ``field_absent`` /
    ``denominator_empty`` distinction survives: the artifact publishes the
    NORMALISED rates, and a normalised rate of ``None`` cannot say whether
    the column was absent or its denominator was empty.  So:

    * a run the artifact records as SKIP keeps the artifact's SKIP and its
      basis — UNLESS the artifact also publishes a judgeable verdict at the
      run's last MEASURED epoch, which then wins (see below);
    * a run the artifact records as PASS / FAIL is RE-SCORED here, with the
      empty-denominator rule applied — which is where the artifact's two
      FAIL-on-a-placeholder-0.0 runs move to SKIP / ``denominator_empty``.

    THE LAST MEASURED EPOCH IS NOT HIDDEN
    --------------------------------------
    A run's last epoch is not always the epoch that carries a measurement.
    The census builder publishes ``fragmentation_gate_last_measured`` beside
    the last epoch's verdict precisely so that "nothing measured is lost when
    the run's last epoch is unmeasured", and this gate reads it: when the
    last epoch's verdict is a SKIP and the artifact carries a PASS / FAIL at
    the last MEASURED epoch, THAT verdict is the one reported.  Otherwise a
    measured FAIL would be hidden behind an unmeasured last epoch, which is
    the same blindness this gate exists to remove.

    The comparison against the artifact is made against the SAME block the
    gate judged, so "the gate disagrees with the artifact" always means the
    two verdicts are comparable.

    Every disagreement with the artifact is reported.
    """
    from molmetal.scripts.cfm_fragmentation_gate import (
        DEFAULT_MAX_FRAGMENTED_RATIO,
        DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
    )

    path = Path(census_path)
    base: Dict[str, Any] = {
        "mode": "census",
        "path": str(path),
        "exists": path.is_file(),
        "thresholds": {
            "max_fragmented_ratio": DEFAULT_MAX_FRAGMENTED_RATIO,
            "min_largest_fragment_frac": DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
        },
    }
    if not path.is_file():
        return _skip(
            base,
            "field_absent",
            f"the census artifact {path} does not exist, so there is no "
            f"fragmentation measurement to judge",
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _skip(
            base,
            "field_absent",
            f"the census artifact {path} could not be read "
            f"({type(exc).__name__}: {exc})",
        )
    runs = doc.get("runs")
    if not isinstance(runs, list) or not runs:
        return _skip(
            base,
            "field_absent",
            f"the census artifact {path} carries no runs, so there is no "
            f"fragmentation measurement to judge",
        )

    per_run: List[Dict[str, Any]] = []
    disagreements: List[Dict[str, Any]] = []
    n_pass = n_fail = n_skip = 0
    n_last_measured = 0
    basis_counts: Dict[str, int] = {}
    for run in runs:
        name = str(run.get("run") or run.get("run_path") or "?")
        recorded = run.get("fragmentation_gate") or {}
        recorded_lm = run.get("fragmentation_gate_last_measured") or {}
        rec_status = recorded.get("status")
        rec_basis = recorded.get("basis")
        lm_status = recorded_lm.get("status")
        # The artifact's EFFECTIVE verdict: the last epoch's, unless that is
        # a SKIP and the artifact also publishes a judgeable verdict at the
        # last MEASURED epoch.
        used_last_measured = rec_status == "SKIP" and lm_status in ("PASS", "FAIL")
        if used_last_measured:
            effective = recorded_lm
            effective_source = (
                "the census artifact's own verdict at its last MEASURED epoch"
            )
        else:
            effective = recorded
            effective_source = "the census artifact's own verdict at its last epoch"
        eff_status = effective.get("status")
        eff_basis = effective.get("basis")
        single = run.get("gen_single_fragment_ratio_end")
        largest = run.get("gen_largest_fragment_frac_end")
        validity = run.get("gen_validity_end")
        last_measured = bool(run.get("gen_eval_ran_on_last_epoch"))
        if eff_status == "SKIP":
            block = _skip(
                {
                    "run": name,
                    "recorded_status": rec_status,
                    "recorded_basis": rec_basis,
                    "recorded_last_measured_status": lm_status,
                    "recorded_last_measured_basis": recorded_lm.get("basis"),
                },
                str(eff_basis or "field_absent"),
                str(effective.get("reason") or "the census recorded a SKIP"),
                resourced_from=effective_source,
            )
        elif used_last_measured:
            n_last_measured += 1
            block = _recorded_measured_block(
                name,
                recorded,
                recorded_lm,
                effective_source,
                dict(base["thresholds"]),
            )
        else:
            if single is None:
                block = _skip(
                    {"run": name, "recorded_status": rec_status},
                    "field_absent",
                    f"the run carries no 'gen_single_fragment_ratio' value at "
                    f"its last measured epoch",
                )
            elif _denominator_is_empty(validity):
                block = _skip(
                    {"run": name, "recorded_status": rec_status},
                    "denominator_empty",
                    f"gen_validity_end == 0.0, so the eval scored NONE of the "
                    f"samples it drew and the chemistry rate is 0/0 — "
                    f"undefined, not 0.0",
                )
            else:
                block = _measured(
                    {
                        "run": name,
                        "recorded_status": rec_status,
                        "recorded_basis": rec_basis,
                    },
                    single_fragment_ratio=float(single),
                    largest_fragment_frac=largest,
                    validity=validity if last_measured else None,
                    field_present=True,
                )
                block["resourced_from"] = "re-scored from the artifact's rates"
        status = str(block.get("status"))
        basis = str(block.get("basis"))
        basis_counts[basis] = basis_counts.get(basis, 0) + 1
        if status == "PASS":
            n_pass += 1
        elif status == "FAIL":
            n_fail += 1
        else:
            n_skip += 1
        if eff_status and eff_status != status:
            disagreements.append(
                {
                    "run": name,
                    "recorded_status": eff_status,
                    "recorded_basis": eff_basis,
                    "recorded_source": effective_source,
                    "rescored_status": status,
                    "rescored_basis": basis,
                }
            )
        per_run.append(block)

    overall = "FAIL" if n_fail else ("SKIP" if n_skip else "PASS")
    base.update(
        {
            "status": overall,
            "basis": None,
            "reasons": [
                f"{n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP over "
                f"{len(per_run)} run(s)",
            ],
            "n_runs": len(per_run),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_skip": n_skip,
            "basis_counts": basis_counts,
            "n_judged_at_last_measured_epoch": n_last_measured,
            "n_disagreements_with_artifact": len(disagreements),
            "disagreements_with_artifact": disagreements,
            "runs": per_run,
        }
    )
    return base


def _overall(blocks: List[Dict[str, Any]]) -> str:
    """The overall verdict of several checks.

    FAIL dominates, then SKIP, then PASS: a check that could not be made is
    not a pass.
    """
    statuses = {str(b.get("status")) for b in blocks}
    if "FAIL" in statuses:
        return "FAIL"
    if "SKIP" in statuses:
        return "SKIP"
    return "PASS"


def _exit_code(status: str) -> int:
    return {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL}.get(status, EXIT_SKIP)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pre_flight_fragmentation_check.py",
        description=(
            "Wf-A128: standalone CPU-only fragmentation gate.  PASS / FAIL / "
            "SKIP with the census builder's own basis vocabulary "
            "(field_absent / denominator_empty / end_to_end / conditional) "
            "and the same thresholds.  SKIP, never FAIL, on absent input or "
            "an empty denominator."
        ),
    )
    p.add_argument("--pool-csv", type=Path, default=None, help="one decoded pool CSV")
    p.add_argument("--run-dir", type=Path, default=None, help="a run directory")
    p.add_argument(
        "--census",
        type=Path,
        default=None,
        help=(
            "a cfm_run_census.json artifact to re-score run by run; with no "
            "mode named at all the repository's own artifact "
            f"({DEFAULT_CENSUS}) is used"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the JSON report here (atomic write, AGENTS.md §5.6)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the per-check stdout lines (the JSON report is unaffected)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    blocks: List[Dict[str, Any]] = []
    if args.pool_csv is not None:
        blocks.append(check_pool_csv(args.pool_csv))
    if args.run_dir is not None:
        blocks.append(check_run_dir(args.run_dir))
    if args.census is not None:
        blocks.append(check_census(args.census))
    elif args.pool_csv is None and args.run_dir is None:
        # No mode named: the repository's own census artifact is the default.
        blocks.append(check_census(DEFAULT_CENSUS))
    if not blocks:
        raise SystemExit(
            "pre_flight_fragmentation_check: one of --pool-csv / --run-dir / "
            "--census is required."
        )

    overall = _overall(blocks)
    report = {
        "check": "p3_cfm_fragmentation_preflight",
        "wave": "Wf-A128",
        "status": overall,
        "basis_vocabulary": list(BASIS_VOCABULARY),
        "n_checks": len(blocks),
        "checks": blocks,
    }
    if not args.quiet:
        for b in blocks:
            print(
                f"[wf-a128-frag] {b.get('status')}: {b.get('path')} "
                f"(basis={b.get('basis')})"
            )
        print(f"[wf-a128-frag] OVERALL {overall}")
    if args.out is not None:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

        atomic_write_file(args.out, json.dumps(report, indent=2, default=str))
    return _exit_code(overall)


if __name__ == "__main__":
    raise SystemExit(main())
