# SPDX-License-Identifier: Apache-2.0
"""THE POOL-EPOCH RULE -- the ONE owner of "at which epoch was this pool's
generation-quality number measured?".

THE RULE (verbatim)
-------------------
"The generation-quality number published for a pool is the resolved
POPULATION's aggregate value AT THE POOL'S OWN LAST EPOCH -- the aggregate row
that carries gen_validity_n, never the per-seed gen_eval rows."

WHY THIS MODULE EXISTS
-----------------------
``molmetal/scripts/cfm_fragmentation_gate.py::score_run_dir`` read
``<run>/per_epoch_metrics.csv`` with ``epoch=None``, i.e. the LAST ROW, and
published that number as ``verdict.gen_validity``.  A pool is a set of
MOLECULES, so the only epoch at which a number describes that pool is the
epoch the pool was drawn at.  Reading the run's last row instead published a
validity measured at a DIFFERENT epoch than the pool's own molecules.

MEASURED on disk (Wf-A120, 2026-10-31) -- three of the thirteen pools:

    run                            pool epoch  published today   correct value
    cfm_fullcorpus_2026_10_10           10       0.9140625       0.3255208333333333
    cfm_fullcorpus_b64_2026_10_11        8       0.0859375       0.3763020833333333
    cfm_objective_terms_2026_10_11       5       0.07421875      0.234375

For ``cfm_fullcorpus_2026_10_10`` the published 0.9140625 is a CLOBBERED
1-epoch CPU smoke (n_train 64) that overwrote ``per_epoch_metrics.csv``; the
pool's own epoch 10 belongs to the 14-epoch run of record, whose aggregate row
at epoch 10 reads 250/768.  The other ten pools were already aligned.

ONE RESOLUTION, ONE OWNER (Wf-A121 / I1)
----------------------------------------
This module used to hold TWO policies -- the GATE's (``resolve_population`` /
``pool_epoch_validity``) and the C7 AUDIT's (``resolve_population_census`` /
``generation_validity_at_epoch``) -- and they disagreed about the SAME run
directory:

  * the GATE needed a ``done`` event to segment a run, so a stream with NO
    ``done`` event resolved to NOTHING and published no number, with the
    reason "this run directory has no events.jsonl, so no population can be
    resolved from the event stream" -- FALSE for a directory that HAS an
    ``events.jsonl``;
  * the C7 AUDIT treated such a stream as ONE population ("fallback:
    un_segmented_event_stream"), because an un-segmented stream is not the
    ABSENCE of a record, it IS the record.

MEASURED: ``cfm_c8_fullgen_2026_10_24``'s ``events.jsonl`` holds three
``train_epoch`` rows and NO ``done`` event, so the census segments it into
zero populations; the pool's own epoch-2 aggregate reads 704/704 = 1.0 and is
that pool's only surviving record.

Both policies now live behind ONE resolution, :func:`_resolve_population`.
The unified rule, in this order:

  a. the census's published run of record
     (``census_row["integrity"]["run_of_record_segment_index"]``) when that
     segment exists;
  b. else the LONGEST population (the census's own rule,
     ``molmetal/scripts/wf_cfm_run_census.py::_run_of_record_index``);
  c. else the segment that CONTAINS the pool's own epoch;
  d. else -- the stream holds NO ``done`` event, so it cannot be segmented --
     the WHOLE STREAM IS ONE POPULATION (the rule the c7 audit already
     applied).

THE AMBIGUITY GUARD
-------------------
In case (d) the whole stream is ONE population, so a row at the pool's own
epoch is that population's row.  But if the stream holds MORE THAN ONE
aggregate (``train_epoch``) row at the pool's own epoch, the pool's epoch
CANNOT be attributed to one population: no number is published, and the
reason says exactly how many rows were found.  A row is never picked
silently.

WHICH POPULATION'S EPOCHS
--------------------------
A segment can hold MORE epochs than its ``done`` event ran: the LEADING
extra epochs are a KILLED PRE-RESUME ATTEMPT that wrote into the same
directory.  MEASURED: ``cfm_c7_fullgen_2026_10_20``'s segment holds 18
epochs (0..17) while its ``done`` event ran 14 -- and the 14 are exactly
epochs 4..17.  The population's OWN epochs are therefore the LAST
``n_epochs_run`` epochs of its segment -- the census's own rule, reused here
rather than re-derived.

The run of record's cohort is attached to the population, resolved by the
cohort owner (``molmetal/scripts/cfm_cohort_recovery.py``'s ``recover_cohort``)
so a reader can see WHICH launch a number belongs to -- MEASURED:
``cfm_fullcorpus_2026_10_10``'s run of record is the 26715/192 run and the
clobbering smoke is the 64/32 one.

WHETHER THE RECORDED PER-EPOCH FILE IS THE RUN OF RECORD'S OWN FILE
--------------------------------------------------------------------
This used to be re-derived here by comparing epoch SETS, which contradicted
the census for a resumed run: ``cfm_c7_fullgen_2026_10_20``'s file holds
epochs {4..17} while its segment holds {0..17} (the leading epochs are the
killed pre-resume attempt), so the local test said "not the run of record"
and the cohort came out "unattested" -- while the census's own answer was
``n_train 26715 / n_val 192`` from ``per_epoch_metrics.json:config``.  The
census's answer is now taken as the census's answer
(``molmetal/scripts/wf_cfm_run_census.py``'s
``file_is_run_of_record = not resolved and verdict != "unmatched_conflict"``),
and the local epoch-set test is only the fallback for a run the census does
not carry.

The pool's KIND is attached too, by the pool-kind owner
(``molmetal/scripts/cfm_pool_kind.py``'s ``classify_pool``): a reconstruction
pool's number and a generation pool's number are different measurements.

THE AGGREGATE ROW, NOT THE PER-SEED ROWS
-----------------------------------------
``events.jsonl`` carries three kinds of row.  ``train_epoch`` is the AGGREGATE
row (one per epoch, carrying ``gen_validity`` / ``gen_validity_val`` /
``gen_validity_n`` / ``gen_single_fragment_ratio``); ``gen_eval`` is one row
per (epoch, seed) and is NOT an aggregate -- the rule never uses it.

NO GPU.  This module only reads files.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts.cfm_cohort_recovery import (  # noqa: E402
    COHORT_SOURCE_UNATTESTED,
    recover_cohort,
)
from molmetal.scripts.cfm_pool_kind import classify_pool  # noqa: E402
from molmetal.scripts.wf_cfm_run_census import (  # noqa: E402
    # The census's OWN segmentation semantics: ``_read_events`` returns the
    # event stream ALREADY segmented by ``_event_segments`` (one segment per
    # ``done`` event), and ``_run_of_record_index`` picks the run of record
    # (the longest population, earliest tie).  They are reused rather than
    # re-implemented so the population this module resolves is the SAME
    # population the SSOT resolves.
    _read_events,
    _run_of_record_index,
)

__all__ = [
    "POOL_EPOCH_RULE",
    "RECORDED_SERIES",
    "EVENT_STREAM",
    "pool_last_epoch",
    "resolve_population",
    "pool_epoch_validity",
    "census_row",
    "resolve_population_census",
    "generation_validity_at_epoch",
    "pool_epoch_recorded_row",
    "run_last_epoch_recorded_row",
    "pool_population_composition",
    "SOURCE_CENSUS_RUN_OF_RECORD",
    "SOURCE_SEGMENT_CONTAINS_POOL_EPOCH",
    "SOURCE_UNSEGMENTED_EVENT_STREAM",
    "SOURCE_AMBIGUOUS_POOL_EPOCH",
]

#: The rule, verbatim.  ONE owner: the gate and the audit both read it.
POOL_EPOCH_RULE = (
    "The generation-quality number published for a pool is the resolved "
    "POPULATION's aggregate value AT THE POOL'S OWN LAST EPOCH -- the "
    "aggregate row that carries gen_validity_n, never the per-seed gen_eval "
    "rows."
)

#: The run's RECORDED per-epoch validity series.
RECORDED_SERIES = "per_epoch_metrics.csv"

#: The append-only event stream: EVERY population that ever wrote here.
EVENT_STREAM = "events.jsonl"

# ---------------------------------------------------------------------------
# The population SOURCES.  These strings are what the c7 artifact publishes
# (``population_source``) and what the fragmentation audit publishes
# (``pool_epoch_validity.population.source``), so they are part of the
# artifact's contract: the first three keep their existing spelling.
# ---------------------------------------------------------------------------

#: The census's own published run of record (segment index) was used.
SOURCE_CENSUS_RUN_OF_RECORD = "census:run_of_record_segment_index"

#: The population was taken because it HOLDS the pool's own epoch (the
#: census does not carry this run, or its run of record does not hold it).
SOURCE_SEGMENT_CONTAINS_POOL_EPOCH = "fallback:segment_contains_pool_epoch"

#: The stream carries no ``done`` event, so it cannot be segmented into
#: populations: the WHOLE STREAM is one population.
SOURCE_UNSEGMENTED_EVENT_STREAM = "fallback:un_segmented_event_stream"

#: The un-segmented stream holds more than one aggregate row at the pool's
#: own epoch: the pool's epoch cannot be attributed to one population and NO
#: number is published.
SOURCE_AMBIGUOUS_POOL_EPOCH = "ambiguous:multiple_aggregate_rows_at_pool_epoch"


# ---------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------


def _num(value: Any) -> Optional[float]:
    """One numeric value from a JSON scalar or a CSV cell; None if absent."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _epoch_of(row: Dict[str, Any]) -> Optional[int]:
    """The ``epoch`` of one row (an event dict or a CSV row)."""
    raw = row.get("epoch")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Every row of a CSV, or an empty list when it cannot be read."""
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def _read_recorded_row(run_dir: Path, epoch: Optional[int]) -> Optional[Dict[str, str]]:
    """One row of the run's RECORDED per-epoch series.

    ``epoch=None`` means "the LAST row", which is what the gate used to read
    unconditionally -- the defect this module exists to fix.
    """
    path = Path(run_dir) / RECORDED_SERIES
    if not path.is_file():
        return None
    rows = _read_csv_rows(path)
    if not rows:
        return None
    if epoch is None:
        return rows[-1]
    for row in rows:
        if _epoch_of(row) == epoch:
            return row
    return None


def _recorded_epochs(run_dir: Path) -> List[int]:
    """Every epoch the run's recorded per-epoch series carries."""
    path = Path(run_dir) / RECORDED_SERIES
    if not path.is_file():
        return []
    out: List[int] = []
    for row in _read_csv_rows(path):
        ep = _epoch_of(row)
        if ep is not None:
            out.append(ep)
    return out


def _config_cohort(run_dir: Path) -> Optional[Tuple[Any, Any]]:
    """The ``(n_train, n_val)`` in the recorded file's own config block."""
    path = Path(run_dir) / "per_epoch_metrics.json"
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    config = blob.get("config") or {}
    n_train = config.get("n_train")
    n_val = config.get("n_val")
    if n_train is None or n_val is None:
        return None
    return (n_train, n_val)


def _validity_of(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """``(value, n)`` of one aggregate row's generation validity.

    The exact quotient ``gen_validity_val / gen_validity_n`` is preferred over
    the ROUNDED ``gen_validity`` column (the CSV stores it at 4 decimals), so
    the published number is the measurement and not its rendering.
    """
    n = _num(row.get("gen_validity_n"))
    val = _num(row.get("gen_validity_val"))
    if val is not None and n:
        return val / n, n
    return _num(row.get("gen_validity")), n


def _population_own_epochs(
    seg: Dict[str, Any], published: Optional[Sequence[int]] = None
) -> List[int]:
    """The population's OWN epochs: the LAST ``n_epochs_run`` of its segment.

    A segment can hold MORE epochs than its ``done`` event ran -- the LEADING
    extra epochs are a killed pre-resume attempt that wrote into the same
    directory.  MEASURED: ``cfm_c7_fullgen_2026_10_20``'s segment holds 18
    epochs (0..17) while its ``done`` event ran 14, and the 14 are exactly
    epochs 4..17.

    This is the census's OWN rule (``wf_cfm_run_census.py``'s
    ``_population_decision``: ``R = seg_epochs[-n_run:]``); it is reused, not
    re-derived.  When the census published the population's epoch list, that
    list is used as-is.
    """
    if published:
        return sorted({int(e) for e in published})
    idxs = sorted({int(e) for e in (seg.get("epoch_indices") or [])})
    n = seg.get("n_epochs_run")
    n = int(n) if isinstance(n, (int, float)) else 0
    if n and n < len(idxs):
        return idxs[-n:]
    return idxs


# ---------------------------------------------------------------------------
# THE ONE RESOLUTION
# ---------------------------------------------------------------------------


def _segments(run_dir: Path) -> List[Dict[str, Any]]:
    """The run's populations, in the census's own segmentation semantics."""
    try:
        events = _read_events(Path(run_dir))
    except Exception:
        return []
    if not events:
        return []
    return list(events.get("segments") or [])


def _segment_with_epoch(
    segs: List[Dict[str, Any]], epoch: Optional[int]
) -> Optional[Dict[str, Any]]:
    """The segment whose epoch set CONTAINS ``epoch``."""
    if epoch is None:
        return None
    for seg in segs:
        if epoch in (seg.get("epoch_indices") or []):
            return seg
    return None


def _event_stream(run_dir: Path) -> Optional[Dict[str, Any]]:
    """The census's own read of the run's event stream.

    ``None`` when the directory has no ``events.jsonl`` at all -- which is a
    DIFFERENT fact from "the stream exists but carries no ``done`` event".
    """
    try:
        return _read_events(Path(run_dir))
    except Exception:
        return None


def _resolve_population(
    run_dir: Path,
    pool_epoch: Optional[int],
    census_row: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """THE ONE POPULATION RESOLUTION -- the gate and the c7 audit both use it.

    The unified rule, in this order:

      a. the census's published run of record
         (``integrity.run_of_record_segment_index``) when that segment exists;
      b. else the LONGEST population (the census's own rule);
      c. else the segment that CONTAINS the pool's own epoch;
      d. else -- the stream holds NO ``done`` event -- the WHOLE STREAM is
         ONE population.

    Every published reason string is TRUE: "no events.jsonl" is said only of
    a directory that has none, and "carries no done event" only of a stream
    that exists but cannot be segmented.

    Returns ``(population, pool_epoch_row)``: the resolution dict and the
    resolved population's aggregate row AT THE POOL'S OWN EPOCH (``None``
    when there is none).  The dict carries the c7 audit's keys (``rows``,
    ``segment_index``, ``epochs``, ``source``, ``reason``) AND the gate's
    keys (``resolution``, ``epoch_indices``, ``n_epochs_run``, ``cohort``,
    ``metrics_file_is_run_of_record``, ...), so ONE resolution serves both.
    """
    run_dir = Path(run_dir)
    census_row = census_row or {}
    integrity = census_row.get("integrity") or {}

    stream = _event_stream(run_dir)
    has_stream_file = (run_dir / EVENT_STREAM).is_file()
    segs = list((stream or {}).get("segments") or [])
    train_epoch = list((stream or {}).get("train_epoch") or [])

    # (a) the census's published run of record.
    ro_seg: Optional[Dict[str, Any]] = None
    from_census = False
    published_idx = integrity.get("run_of_record_segment_index")
    if isinstance(published_idx, int):
        ro_seg = next((s for s in segs if s.get("index") == published_idx), None)
        from_census = ro_seg is not None
    # (b) else the LONGEST population -- the census's own rule.
    if ro_seg is None and segs:
        ro_idx = _run_of_record_index(segs)
        ro_seg = next((s for s in segs if s.get("index") == ro_idx), None)

    chosen: Optional[Dict[str, Any]] = None
    population_rows: List[Dict[str, Any]] = []
    population_epochs: List[int] = []
    segment_index: Optional[int] = None
    n_epochs_run: Optional[int] = None
    resolution = "unresolved"
    source = ""
    reason = ""
    notes: List[str] = []
    ambiguous = False

    if ro_seg is not None:
        ro_epochs = _population_own_epochs(
            ro_seg,
            published=(
                integrity.get("run_of_record_epoch_indices") if from_census else None
            ),
        )
        ro_rows = [
            r for r in (ro_seg.get("rows") or []) if _epoch_of(r) in set(ro_epochs)
        ]
        if pool_epoch is None or pool_epoch in set(ro_epochs):
            chosen = ro_seg
            population_epochs = ro_epochs
            population_rows = ro_rows
            resolution = "run_of_record"
            source = (
                SOURCE_CENSUS_RUN_OF_RECORD
                if from_census
                else SOURCE_SEGMENT_CONTAINS_POOL_EPOCH
            )
            if from_census:
                reason = (
                    f"the census resolves this run's run of record to segment "
                    f"{ro_seg.get('index')} of {len(segs)}; its own epochs are "
                    f"{ro_epochs}, and the pool's own epoch {pool_epoch} is "
                    f"one of them"
                )
            else:
                reason = (
                    f"the census does not carry this run, so the population "
                    f"that HOLDS the pool's own epoch ({pool_epoch}) was "
                    f"taken: segment {ro_seg.get('index')} is the run of "
                    f"record (the longest population, {len(ro_epochs)} epoch(s)"
                    f" {ro_epochs})"
                )
        else:
            # (c) the segment that CONTAINS the pool's own epoch.
            containing = _segment_with_epoch(segs, pool_epoch)
            if containing is not None:
                chosen = containing
                population_epochs = sorted(
                    {int(e) for e in (containing.get("epoch_indices") or [])}
                )
                population_rows = [
                    r
                    for r in (containing.get("rows") or [])
                    if _epoch_of(r) in set(population_epochs)
                ]
                resolution = "segment_containing_pool_epoch"
                source = SOURCE_SEGMENT_CONTAINS_POOL_EPOCH
                reason = (
                    f"the pool's own epoch {pool_epoch} is NOT one of the run "
                    f"of record's epochs {ro_epochs}, but it IS one of segment "
                    f"{containing.get('index')}'s (n_epochs_run="
                    f"{containing.get('n_epochs_run')}); the value is therefore "
                    f"read from THAT population, because the run of record's "
                    f"aggregate row does not exist at the pool's epoch"
                )
            else:
                chosen = ro_seg
                population_epochs = ro_epochs
                population_rows = ro_rows
                resolution = "run_of_record"
                source = (
                    SOURCE_CENSUS_RUN_OF_RECORD
                    if from_census
                    else SOURCE_SEGMENT_CONTAINS_POOL_EPOCH
                )
                reason = (
                    f"the pool's own epoch {pool_epoch} is not one of the run "
                    f"of record's epochs {ro_epochs} and no segment of this "
                    f"run directory holds it, so the run of record's "
                    f"aggregate row does not exist at the pool's epoch"
                )
                notes.append(
                    f"the pool's own epoch {pool_epoch} is held by no "
                    f"population of this run directory"
                )
    elif has_stream_file:
        # (d) the stream holds NO ``done`` event, so it cannot be segmented
        #     into populations: the WHOLE STREAM is ONE population.
        if not train_epoch:
            source = "events:no_aggregate_row"
            reason = (
                "the event stream carries no aggregate (train_epoch) row, so "
                "no population can be resolved from it"
            )
        else:
            population_epochs = sorted(
                {e for e in (_epoch_of(r) for r in train_epoch) if e is not None}
            )
            population_rows = list(train_epoch)
            at_pool = (
                [r for r in train_epoch if _epoch_of(r) == pool_epoch]
                if pool_epoch is not None
                else []
            )
            if len(at_pool) > 1:
                # THE AMBIGUITY GUARD: the pool's epoch cannot be attributed
                # to ONE population, so NO number is published.
                ambiguous = True
                population_rows = []
                resolution = "ambiguous"
                source = SOURCE_AMBIGUOUS_POOL_EPOCH
                reason = (
                    f"the event stream holds {len(at_pool)} aggregate rows at "
                    f"epoch {pool_epoch}, so the pool's epoch cannot be "
                    f"attributed to one population"
                )
            else:
                resolution = "un_segmented_event_stream"
                source = SOURCE_UNSEGMENTED_EVENT_STREAM
                reason = (
                    "the event stream carries no done event, so it cannot be "
                    "segmented into populations; the whole stream is ONE "
                    f"population with epoch(s) {population_epochs}"
                )
    else:
        source = "no events.jsonl"
        reason = (
            "this run directory has no events.jsonl, so no population can be "
            "resolved from the event stream"
        )

    if chosen is not None:
        segment_index = chosen.get("index")
        n_epochs_run = chosen.get("n_epochs_run")

    # The aggregate row AT THE POOL'S OWN EPOCH -- the row that carries
    # gen_validity_n.  More than one row at that epoch is AMBIGUOUS: the
    # pool's epoch cannot be attributed to one population, so no row is
    # picked.
    pool_epoch_row: Optional[Dict[str, Any]] = None
    candidates = (
        [r for r in population_rows if _epoch_of(r) == pool_epoch]
        if pool_epoch is not None
        else []
    )
    if len(candidates) == 1:
        pool_epoch_row = candidates[0]
    elif len(candidates) > 1:
        ambiguous = True
        population_rows = []
        pool_epoch_row = None
        resolution = "ambiguous"
        source = SOURCE_AMBIGUOUS_POOL_EPOCH
        reason = (
            f"the resolved population holds {len(candidates)} aggregate rows "
            f"at epoch {pool_epoch}, so the pool's epoch cannot be attributed "
            f"to one population"
        )

    # The run of record's cohort, by the cohort owner.  It is what says WHICH
    # launch a number belongs to: populations in one directory do NOT share a
    # cohort.
    ro_ts: List[Any] = []
    if ro_seg is not None:
        ro_epochs = _population_own_epochs(
            ro_seg,
            published=(
                integrity.get("run_of_record_epoch_indices") if from_census else None
            ),
        )
        ro_ts = [
            r.get("ts")
            for r in (ro_seg.get("rows") or [])
            if _epoch_of(r) in set(ro_epochs)
        ]
    elif population_rows:
        # The whole (un-segmented) stream IS the population.
        ro_ts = [r.get("ts") for r in population_rows]
    metrics_is_ro = _metrics_file_is_run_of_record(
        run_dir,
        population_epochs if resolution != "unresolved" else None,
        census_row=census_row,
    )
    cohort = recover_cohort(
        run_dir,
        ro_ts,
        config_cohort=(
            _config_cohort(run_dir) if metrics_is_ro is True else None
        ),
    )

    population = {
        # -- the c7 audit's keys (its contract) ---------------------------
        "rows": population_rows,
        "segment_index": segment_index,
        "epochs": list(population_epochs),
        "source": source,
        "reason": reason,
        # -- the gate's keys ----------------------------------------------
        "resolution": resolution,
        "epoch_indices": list(population_epochs),
        "population_epochs": list(population_epochs),
        "n_epochs_run": n_epochs_run,
        "contains_pool_epoch": bool(
            pool_epoch is not None and pool_epoch in set(population_epochs)
        ),
        "run_of_record_segment_index": (
            ro_seg.get("index") if ro_seg is not None else None
        ),
        "run_of_record_epoch_indices": (
            _population_own_epochs(
                ro_seg,
                published=(
                    integrity.get("run_of_record_epoch_indices")
                    if from_census
                    else None
                ),
            )
            if ro_seg is not None
            else []
        ),
        "metrics_file_is_run_of_record": metrics_is_ro,
        "cohort": {
            "n_train": cohort.get("n_train"),
            "n_val": cohort.get("n_val"),
            "source": cohort.get("source"),
            "reason": cohort.get("reason"),
            "unattested": cohort.get("source") == COHORT_SOURCE_UNATTESTED,
        },
        "reason": reason,
        # -- the new, additive keys ---------------------------------------
        "ambiguous": ambiguous,
        "pool_epoch_row": (
            _recorded_row_dict(
                pool_epoch_row,
                (
                    f"events.jsonl:train_epoch[epoch={_epoch_of(pool_epoch_row)}]"
                    + (
                        f" of segment {segment_index}"
                        if segment_index is not None
                        else " of the whole (un-segmented) event stream"
                    )
                ),
                source,
            )
            if pool_epoch_row is not None
            else None
        ),
        "notes": notes,
        "events_present": has_stream_file,
        "n_segments": len(segs),
    }
    if cohort.get("source") == COHORT_SOURCE_UNATTESTED and not (
        (cohort.get("evidence") or {}).get("n_logs")
    ) and not (run_dir / "per_epoch_metrics.json").is_file():
        # The cohort owner's own prose names a per-epoch metrics file that
        # this directory does not have; the note states what is actually
        # missing.
        population["cohort"]["note"] = (
            "this directory holds no run log and no per-epoch metrics file, "
            "so no artifact on disk ties a cohort to the resolved population"
        )
    return population, pool_epoch_row


def resolve_population(
    run_dir: Path,
    pool_epoch: Optional[int],
    census_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Which POPULATION of ``run_dir`` the pool's own epoch belongs to.

    Returns the ONE resolution dict (see :func:`_resolve_population`).  Its
    ``resolution`` is one of:

    ``run_of_record``
        the resolved population is the run of record and it holds the pool's
        epoch;
    ``segment_containing_pool_epoch``
        the fallback -- another population of this directory holds it;
    ``un_segmented_event_stream``
        the stream carries no ``done`` event, so the WHOLE STREAM is one
        population (and it holds the pool's epoch);
    ``ambiguous``
        more than one aggregate row sits at the pool's own epoch, so no
        number may be published;
    ``unresolved``
        nothing on disk ties the pool's epoch to a population.  The caller
        then has NO number to publish; it must say so, never invent one.
    """
    population, _row = _resolve_population(
        run_dir, pool_epoch, census_row=census_row
    )
    # The RAW event rows are the c7 audit's contract (``population["rows"]``);
    # the gate publishes this dict, and publishing every aggregate row of the
    # population would bloat every artifact that carries it.
    population.pop("rows", None)
    return population


def resolve_population_census(
    run_dir: Path,
    census: Dict[str, Any],
    pool_epoch: Optional[int],
) -> Dict[str, Any]:
    """Which POPULATION of the event stream the pool's numbers belong to.

    THE SAME RESOLUTION as :func:`resolve_population` -- the c7 audit and the
    fragmentation gate ask one question ("which population is this pool's
    number about?") and must get one answer.  This wrapper only looks the
    run's census row up for the caller.

    The keys the c7 caller reads -- ``rows``, ``segment_index``, ``epochs``,
    ``source``, ``reason`` -- and the source strings it already emits
    (``census:run_of_record_segment_index``,
    ``fallback:segment_contains_pool_epoch``,
    ``fallback:un_segmented_event_stream``) are unchanged.

    Wf-A115 P2 (v1.6) -- the event stream is append-only and a directory can
    hold several populations (a real run, then a smoke that overwrote the
    metrics file), so the LAST ``gen_validity`` in the file is NOT the run of
    record's.  The segmentation and the run-of-record decision belong to the
    census (``_event_segments`` / ``_population_decision``); this reuses them.
    """
    run_dir = Path(run_dir)
    row = census_row(census, run_dir.name)
    population, _pool_epoch_row = _resolve_population(
        run_dir, pool_epoch, census_row=row
    )
    return population


def _metrics_file_is_run_of_record(
    run_dir: Path,
    population_epochs: Optional[Sequence[int]],
    census_row: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    """Is the run's recorded per-epoch file the RUN OF RECORD's own file?

    When the census carries the run, the census's OWN answer is taken
    (``molmetal/scripts/wf_cfm_run_census.py``'s
    ``file_is_run_of_record = not resolved and verdict != "unmatched_conflict"``,
    published as ``integrity.population_resolved`` / ``integrity.
    population_verdict``) -- this module does not re-derive a decision the
    census already made.

    The local epoch-set test is only the FALLBACK, for a run the census does
    not carry: True when the file's epoch set EQUALS the population's own
    epoch set.  MEASURED: ``cfm_fullcorpus_2026_10_10``'s recorded file holds
    ONE row (epoch 0) while its run of record ran 14 epochs (0..13) -- the
    file is a CLOBBERED 1-epoch CPU smoke, not the run of record.

    ``None`` when there is no population to compare against.
    """
    integrity = (census_row or {}).get("integrity") or {}
    if integrity:
        return bool(
            (not integrity.get("population_resolved"))
            and (integrity.get("population_verdict") != "unmatched_conflict")
        )
    if population_epochs is None:
        return None
    recorded = _recorded_epochs(run_dir)
    if not recorded:
        # No recorded series at all: there is nothing to judge.
        return None
    return sorted(set(recorded)) == sorted(set(int(e) for e in population_epochs))


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------


def pool_last_epoch(csv_path: Path) -> Optional[int]:
    """The pool's OWN last epoch: the max of the pool CSV's ``epoch`` column.

    NOT the run's last epoch.  A pool file holds the rows of the epoch(s) it
    was written for, and the run may have trained on for longer -- or, the
    other way round, a later population may have overwritten the run's
    per-epoch record.  MEASURED: ``cfm_fullcorpus_2026_10_10``'s pool holds
    epoch 10 while its recorded per-epoch series holds a single epoch-0 row.
    """
    path = Path(csv_path)
    best: Optional[int] = None
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                raw = (row.get("epoch") or "").strip()
                if not raw:
                    continue
                try:
                    epoch = int(float(raw))
                except (TypeError, ValueError):
                    continue
                if best is None or epoch > best:
                    best = epoch
    except OSError:
        return None
    return best


def pool_epoch_validity(
    run_dir: Path,
    csv_path: Path,
    census_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The aggregate generation validity AT THE POOL'S OWN LAST EPOCH.

    Returns a dict with the keys ``epoch``, ``value``, ``n``, ``source``,
    ``population`` and ``superseded`` (plus ``pool_kind``,
    ``recorded_series`` and ``notes``).

    ``value``
        the resolved population's aggregate ``gen_validity`` at the pool's own
        last epoch -- ``None`` when nothing on disk answers that question.
        NEVER a fallback to another epoch.
    ``superseded``
        the value the run's recorded per-epoch series' LAST ROW carried, with
        the reason it was superseded -- ``None`` when there is no such row.

    The rule realigns a RECORDED number; it does not create one.  A run with
    no recorded per-epoch series records no validity, so ``value`` is ``None``
    for it unless the append-only event stream holds the aggregate row (in
    which case the event stream IS the run's surviving record).
    """
    run_dir = Path(run_dir)
    csv_path = Path(csv_path)
    epoch = pool_last_epoch(csv_path)
    population, pool_row = _resolve_population(
        run_dir, epoch, census_row=census_row
    )

    out: Dict[str, Any] = {
        "epoch": epoch,
        "value": None,
        "n": None,
        "source": None,
        "population": population,
        "superseded": None,
        "rule": POOL_EPOCH_RULE,
        "pool_kind": None,
        "recorded_series": (run_dir / RECORDED_SERIES).is_file(),
        "notes": list(population.get("notes") or []),
    }

    try:
        out["pool_kind"] = classify_pool(csv_path).kind
    except Exception:
        out["pool_kind"] = None

    if epoch is None:
        out["notes"].append(
            "the pool file carries no usable epoch column, so the epoch at "
            "which its generation quality was measured is unknown"
        )
        return out

    # 1. the resolved population's AGGREGATE row (events.jsonl's train_epoch
    #    rows -- append-only, and the run of record's own record).
    row: Optional[Dict[str, Any]] = pool_row
    if row is not None:
        seg = population.get("segment_index")
        where = (
            f"of segment {seg}"
            if seg is not None
            else "of the whole (un-segmented) event stream"
        )
        out["source"] = (
            f"events.jsonl:train_epoch[epoch={epoch}] {where} -- the "
            f"aggregate row that carries gen_validity_n"
        )
    elif population.get("ambiguous"):
        out["notes"].append(
            f"no validity is published for this pool: {population.get('reason')}"
        )
    else:
        # 2. the run's recorded per-epoch series, when it is the run of
        #    record's own file.
        if population.get("metrics_file_is_run_of_record") is not False:
            row = _read_recorded_row(run_dir, epoch)
            if row is not None:
                out["source"] = (
                    f"{RECORDED_SERIES}:epoch={epoch} -- the aggregate row "
                    f"that carries gen_validity_n"
                )
        if row is None:
            out["notes"].append(
                f"no aggregate row carries the pool's own epoch {epoch}: the "
                f"resolved population holds no train_epoch row for it and the "
                f"run's recorded per-epoch series does not either"
            )

    if row is not None:
        value, n = _validity_of(row)
        out["value"] = value
        out["n"] = int(n) if n is not None and float(n).is_integer() else n
    elif not population.get("ambiguous"):
        out["notes"].append(
            "no validity is published for this pool: nothing on disk measures "
            "it at the pool's own epoch, and a number from another epoch is "
            "not the same measurement"
        )

    # What the run's recorded series' LAST ROW carried -- the number this
    # rule supersedes.
    last_row = _read_recorded_row(run_dir, None)
    if last_row is not None:
        old_value, old_n = _validity_of(last_row)
        old_epoch = _epoch_of(last_row)
        if old_epoch == epoch:
            why = (
                f"the recorded series' last row IS the pool's own epoch "
                f"({epoch}), so the number did not change"
            )
        elif old_value == out["value"]:
            why = (
                f"the recorded series' last row is at epoch {old_epoch}, not "
                f"the pool's own epoch {epoch}, but it carries the same value"
            )
        else:
            why = (
                f"the recorded series' last row is at epoch {old_epoch}, not "
                f"the pool's own epoch {epoch}"
            )
            if population.get("metrics_file_is_run_of_record") is False:
                why += (
                    "; that file is NOT the run of record's own file "
                    "(the census's own cross-check says so), so its last row "
                    "measures a DIFFERENT population"
                )
            why += (
                f"; the value was re-read at the pool's own epoch from the "
                f"resolved population ({population.get('resolution')})"
            )
        out["superseded"] = {
            "value": old_value,
            "epoch": old_epoch,
            "n": int(old_n) if old_n is not None and float(old_n).is_integer() else old_n,
            "source": str(run_dir / RECORDED_SERIES),
            "changed": bool(
                old_value is not None
                and out["value"] is not None
                and old_value != out["value"]
            ),
            "reason": why,
        }

    return out


# ---------------------------------------------------------------------------
# THE RECORDED NUMBERS, EACH AT ITS OWN EPOCH (Wf-A121 / I1, R2)
# ---------------------------------------------------------------------------


def _row_at_epoch(
    rows: Sequence[Dict[str, Any]], epoch: Optional[int]
) -> Optional[Dict[str, Any]]:
    """The one aggregate row at ``epoch``, or ``None``."""
    if epoch is None:
        return None
    candidates = [r for r in rows if _epoch_of(r) == epoch]
    if len(candidates) != 1:
        return None
    return candidates[0]


def _recorded_row_dict(
    row: Dict[str, Any], source: str, population_source: str
) -> Dict[str, Any]:
    """One aggregate row as a published, self-labelling dict."""
    value, n = _validity_of(row)
    return {
        "epoch": _epoch_of(row),
        "gen_validity": value,
        "gen_validity_n": int(n) if n is not None and float(n).is_integer() else n,
        "gen_single_fragment_ratio": _num(row.get("gen_single_fragment_ratio")),
        "gen_largest_fragment_frac": _num(row.get("gen_largest_fragment_frac")),
        "source": source,
        "population_source": population_source,
    }


def pool_epoch_recorded_row(
    run_dir: Path,
    csv_path: Path,
    census_row: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """The resolved population's aggregate row AT THE POOL'S OWN EPOCH.

    THE helper a caller needs to publish a RECORDED number beside a
    RE-DERIVED one: the recorded number must be read at the pool's own epoch,
    not at the run's last epoch, or the two numbers describe different
    models.

    The row carries BOTH ``gen_validity`` / ``gen_validity_n`` AND
    ``gen_single_fragment_ratio`` (the events' ``train_epoch`` rows carry it
    too -- MEASURED), plus the ``epoch`` it was measured at and the
    ``source`` it was read from.

    Returns ``None`` when nothing on disk answers the question.
    """
    run_dir = Path(run_dir)
    csv_path = Path(csv_path)
    epoch = pool_last_epoch(csv_path)
    population, row = _resolve_population(
        run_dir, epoch, census_row=census_row
    )
    if row is not None:
        seg = population.get("segment_index")
        where = (
            f"of segment {seg}"
            if seg is not None
            else "of the whole (un-segmented) event stream"
        )
        return _recorded_row_dict(
            row,
            f"events.jsonl:train_epoch[epoch={epoch}] {where}",
            str(population.get("source")),
        )
    # The run's recorded per-epoch series, when it is the run of record's
    # own file.
    if population.get("metrics_file_is_run_of_record") is not False:
        recorded = _read_recorded_row(run_dir, epoch)
        if recorded is not None:
            return _recorded_row_dict(
                recorded,
                f"{RECORDED_SERIES}:epoch={epoch}",
                str(population.get("source")),
            )
    return None


def run_last_epoch_recorded_row(
    run_dir: Path,
    csv_path: Optional[Path] = None,
    census_row: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """The run of record's LAST epoch aggregate row, with its epoch.

    The companion of :func:`pool_epoch_recorded_row`: it publishes the OTHER
    number, the one a last-epoch reader would quote, each labelled with the
    epoch it was measured at.  MEASURED: for ``cfm_objective_terms_2026_10_11``
    the last-epoch row reads ``gen_single_fragment_ratio`` 0.8771929824561403
    at epoch 8, while the pool's own epoch 5 reads 0.6666666666666666 -- two
    different models, so a reader needs both, each labelled.

    Returns ``None`` when the run has no aggregate row at all.
    """
    run_dir = Path(run_dir)
    epoch = pool_last_epoch(csv_path) if csv_path is not None else None
    population, _row = _resolve_population(
        run_dir, epoch, census_row=census_row
    )
    rows = list(population.get("rows") or [])
    if rows:
        last_epoch = max(
            (e for e in (_epoch_of(r) for r in rows) if e is not None),
            default=None,
        )
        row = _row_at_epoch(rows, last_epoch)
        if row is not None:
            seg = population.get("segment_index")
            where = (
                f"of segment {seg}"
                if seg is not None
                else "of the whole (un-segmented) event stream"
            )
            return _recorded_row_dict(
                row,
                f"events.jsonl:train_epoch[epoch={last_epoch}] {where}",
                str(population.get("source")),
            )
    recorded = _read_recorded_row(run_dir, None)
    if recorded is not None:
        return _recorded_row_dict(
            recorded,
            f"{RECORDED_SERIES}:last row",
            str(population.get("source")),
        )
    return None


def pool_population_composition(
    run_dir: Path,
    csv_path: Path,
    census_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The pool's own COMPOSITION: which epochs, how many rows, how many lie
    OUTSIDE the run of record.

    Published ADDITIVELY: it changes no existing number.  It says, in one
    place, how much of the pool belongs to the run of record and how much to
    an earlier, killed attempt that wrote into the same directory.

    MEASURED: ``cfm_c7_fullgen_2026_10_20``'s pool holds 12669 rows over
    epochs 0..17, of which epochs 0..3 (2816 rows) belong to a KILLED
    PRE-RESUME ATTEMPT and the run of record ran epochs 4..17.
    """
    run_dir = Path(run_dir)
    csv_path = Path(csv_path)
    epoch = pool_last_epoch(csv_path)
    population, _row = _resolve_population(
        run_dir, epoch, census_row=census_row
    )

    counts: Counter = Counter()
    n_rows_total = 0
    try:
        with csv_path.open(newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                n_rows_total += 1
                ep = _epoch_of({"epoch": row.get("epoch")})
                if ep is not None:
                    counts[ep] += 1
    except OSError:
        pass

    ro_epochs = list(population.get("population_epochs") or [])
    ro_set = set(ro_epochs)
    n_outside = sum(n for ep, n in counts.items() if ep not in ro_set)
    outside_epochs = sorted(ep for ep in counts if ep not in ro_set)

    if n_outside:
        note = (
            f"{n_outside} of the pool's {n_rows_total} row(s) sit at epoch(s) "
            f"{outside_epochs}, which the run of record did NOT run (it ran "
            f"{ro_epochs}): they belong to an earlier, KILLED attempt that "
            f"wrote into the same directory, so the pool's molecules are not "
            f"all the run of record's"
        )
    elif n_rows_total:
        note = (
            f"every one of the pool's {n_rows_total} row(s) sits at an epoch "
            f"the run of record ran ({ro_epochs})"
        )
    else:
        note = "the pool file holds no row with a usable epoch column"

    return {
        "pool_epochs": sorted(counts),
        "pool_epoch": epoch,
        "n_rows_total": n_rows_total,
        "n_rows_at_pool_epoch": counts.get(epoch, 0) if epoch is not None else 0,
        "run_of_record_epochs": ro_epochs,
        "n_rows_outside_run_of_record": n_outside,
        "note": note,
        "population_source": population.get("source"),
    }


# ---------------------------------------------------------------------------
# THE C7 AUDIT'S ACCESSORS
# ---------------------------------------------------------------------------


def census_row(census: Dict[str, Any], run_name: str) -> Optional[Dict[str, Any]]:
    """The census's own row for one run, or ``None``.

    The run census (``metrics/by_metric/cfm_run_census.json``) is the SSOT
    of which run wrote what; this is the accessor the epoch rule and the
    cohort resolution both reach it through.
    """
    for row in census.get("runs") or []:
        if isinstance(row, dict) and row.get("run") == run_name:
            return row
    return None


def _read_event_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """Every readable row of a run's ``events.jsonl``.

    A malformed line is SKIPPED, not fatal: the census does the same, so a
    census that dies on one bad line reads nothing.
    """
    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


def generation_validity_at_epoch(
    run_dir: Path,
    epoch: Optional[int] = None,
    rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """The ``gen_validity`` of the trainer, with its sample count.

    THE EPOCH RULE (v1.6): the value published on a c7 pool row is the
    resolved population's value AT THE POOL'S OWN LAST EPOCH -- the epoch
    whose molecules the pool actually contains.  The SA / QED / novelty on
    that row describe THAT epoch, so a validity taken from another epoch
    describes a different model.  Within the population the AGGREGATE row is
    taken -- the one carrying ``gen_validity_n`` -- never the per-seed rows
    (which carry no ``gen_validity_n``).

    THE AMBIGUITY GUARD: when MORE THAN ONE aggregate row sits at the
    requested epoch, the epoch cannot be attributed to one population and
    ``None`` is returned -- a row is never picked silently.

    MEASURED: ``cfm_fullcorpus_2026_10_10``'s pool is epoch 10 and its run of
    record's epoch-10 aggregate is 0.3255208333333333 (n=768); the value the
    old "last logged row" rule published was the epoch-0 row of a LATER
    1-epoch CPU smoke that wrote into the same directory (0.9140625, n=128).

    Parameters
    ----------
    epoch:
        the pool's own last epoch.  When it is given, the aggregate row AT
        that epoch is returned.  When it is ``None`` the LEGACY behaviour
        applies -- the last row of the file that carries a ``gen_validity`` --
        which is what the pre-v1.6 artifact published and what a caller with
        no pool epoch can still ask for.  The artifact never uses it.
    rows:
        the resolved population's event rows.  Read from ``events.jsonl`` when
        not given.
    """
    if rows is None:
        rows = _read_event_rows(run_dir)
    if epoch is None:
        last: Optional[Dict[str, Any]] = None
        for row in rows:
            if "gen_validity" in row:
                last = {
                    "value": float(row["gen_validity"]),
                    "n": row.get("gen_validity_n"),
                    "epoch": row.get("epoch"),
                }
        return last
    want = int(epoch)
    candidates = [
        row
        for row in rows
        if _epoch_of(row) == want
        and "gen_validity" in row
        and row.get("gen_validity_n") is not None
    ]
    if len(candidates) != 1:
        # No row, or an AMBIGUOUS epoch: never pick one silently.
        return None
    row = candidates[0]
    return {
        "value": float(row["gen_validity"]),
        "n": row.get("gen_validity_n"),
        "epoch": want,
    }
