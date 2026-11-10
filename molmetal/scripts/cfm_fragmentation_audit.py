# SPDX-License-Identifier: Apache-2.0
"""Project-wide audit of the CFM fragmentation failure (CPU-only, RDKit-only).

WHY THIS EXISTS
---------------
``gen_validity`` cannot tell a molecule from a pile of loose atoms.  It counts
a sample as valid when RDKit can sanitise the string, and ``C.C.C.C``
sanitises.  MEASURED on ``molmetal/reports/cfm_c8_fullgen_2026_10_24``:
``gen_validity`` read **1.000** at all three epochs it completed while
``gen_single_fragment_ratio`` read **0.203 / 0.200 / 0.348**.

This script produces the project-wide record of that failure:

  * it re-scores EVERY generation pool on disk with the trainer's OWN
    ``decode_chemistry``, so the numbers here are derived from the SMILES
    themselves rather than copied from a column a run wrote about itself;
  * it cross-checks each run's recorded ``n_frags_raw`` column against the
    re-derivation, so a run that mis-recorded itself is visible;
  * it joins the result with the SSOT (``cfm_run_census.json``) so the pools
    and the run-level rows can be read against each other.

THE TWO POOL KINDS ARE NOT THE SAME MEASUREMENT
-----------------------------------------------
Not every ``generated_smiles.csv`` on disk is the generator's output.
``molmetal/scripts/wf_cfm_decode_dump_smiles.py`` wrote a pool with a
``metal_z`` column in TWO eras: before 2026-10-12 it loaded REAL tmQM
geometries and ran only the bond head (a RECONSTRUCTION); from 2026-10-12 it
calls the trainer's own ODE sampler (FREE GENERATION).  The trainer's own
free-generation path writes ``n_frags_raw``.  This script classifies each
pool with the ONE owner of that rule
(``molmetal/scripts/cfm_pool_kind.py`` -- which reads the CONTENT of a
``metal_z`` pool, because the header alone cannot decide it) and never mixes
the two:

  * ``generation``    — the sampler's output.  The fragmentation rate of the
    model.  THIS is the number the audit is about.
  * ``reconstruction`` — the bond head applied to real geometry.  This
    measures the DECODER's fidelity to real chemistry and says nothing about
    the generator.  MEASURED contrast on ``cfm_fullcorpus_b64_2026_10_11`` at
    that pool's own epoch (**8**): the RECONSTRUCTION pool re-derives
    ``single_fragment_ratio`` **0.75** over its own **40** rows (**30** of
    them one connected molecule), while the SAME run's trainer-side
    generation eval recorded ``gen_single_fragment_ratio``
    **0.5986159169550173** over the PARSED rows of the **768** samples it
    drew and ``gen_validity`` **0.3763020833333333** over ALL **768** of
    them (``events.jsonl``'s ``train_epoch`` row for epoch **8**).  Those are
    two different measurements of two different things, and averaging them
    would be a category error.

Read it with the direction of the metric in mind: ``single_fragment_ratio``
and ``single_molecule_rate`` are the fraction of the pool that is ONE
connected molecule, so HIGHER IS BETTER.  ``fragmented_ratio`` is the
complement, so LOWER IS BETTER.

TWO DENOMINATORS
----------------
``single_fragment_ratio`` is conditioned on the row being PARSEABLE -- the
trainer's own denominator.  ``single_molecule_rate`` is over ALL samples the
pool was drawn from: a string RDKit refuses is not a molecule and must not
leave the denominator.  MEASURED on ``cfm_fullcorpus_b64_2026_10_11`` at
epoch 11: the conditional number is 0.7879 and the end-to-end number is
0.0859 x 0.7879 = 0.0677.  The conditional number overstates the generator by
more than 10x.

AND THE END-TO-END DENOMINATOR IS NOT ALWAYS THE RUN'S EVENT COUNT
--------------------------------------------------------------------
Two populations are called "the number of samples drawn", and they are not
the same:

  * the TRAINER's epoch-level draws (``events.jsonl`` -> ``gen_validity_n``),
    which is what the trainer's own columns are measured against;
  * the decode-dump script's OWN attempts, which is what the pool file it
    wrote is a sample of.

MEASURED on ``cfm_fullcorpus_2026_10_10``: the trainer drew 768 samples at
epoch 10, but the pool on disk holds 66 survivors of the decode-dump's own
**192** attempts (``metrics/by_metric/cfm_free_generation_panel_seed0.json``
-> ``generation.n_attempted``).  The pool's end-to-end rate is therefore
64/192 = **0.3333**, not 64/768.  Every run row records which denominator was
used (``end_to_end_denominator_source``), because a rate without its
denominator is unreadable.

AT WHICH EPOCH (Wf-A120)
------------------------
A pool is a set of MOLECULES, so the only epoch at which a generation-quality
number describes it is the epoch the pool was drawn at.  The gate used to read
the run's per-epoch series' LAST ROW and publish that, so three of the
thirteen pools published a validity measured at a DIFFERENT epoch than their
own molecules.  The rule now has ONE owner
(``molmetal/scripts/cfm_pool_epoch_rule.py``):

    "The generation-quality number published for a pool is the resolved
    POPULATION's aggregate value AT THE POOL'S OWN LAST EPOCH -- the
    aggregate row that carries gen_validity_n, never the per-seed gen_eval
    rows."

Every run row publishes ``pool_epoch`` and ``pool_epoch_validity`` (the value,
its ``n``, its source, the resolved population and the superseded value with
the reason it was superseded) next to the verdict.

``census.gen_validity_end`` is UNCHANGED by this rule: it is the RUN's
last-epoch value, which is a different and correctly-labelled number.

THE CROSS-CHECK IS AT THE POOL'S EPOCH TOO (Wf-A121 / I2)
---------------------------------------------------------
``recorded_vs_rederived`` used to put the census's ``gen_single_fragment_
ratio_end`` -- the run's LAST-epoch aggregate -- beside the re-derivation,
which is over the pool's OWN rows.  Two DIFFERENT epochs were compared, so
the difference said nothing about the pool.  The recorded side is now read
AT THE POOL'S OWN EPOCH, so both sides describe the same epoch; the old
value is kept beside it, labelled with its own epoch, because it is still a
correctly-labelled number of its own.  The BLOCK'S PRESENCE follows the
recorded side at the pool's own epoch: a run the census does not carry still
gets its cross-check when that recorded row is on disk.

THE LAST-EPOCH SIDE IS READ FROM THE RUN ITSELF (Wf-A123 DEFECT B)
------------------------------------------------------------------
When the CENSUS's own ``gen_single_fragment_ratio_end`` column does not
cover a run, the last-epoch value is read from the run's OWN
``events.jsonl`` (its last ``train_epoch`` row) instead of being published as
null.  MEASURED 2026-10-31: the two census-less runs DO carry a last-epoch
aggregate -- ``cfm_c7_fullgen_ep4_partial_2026_10_20`` epoch 3
(0.08380681818181818) and ``cfm_c8_fullgen_2026_10_24`` epoch 2
(0.34801136363636365) -- so the previous null asserted an absence that disk
contradicts.  The two numbers stay DISTINCTLY labelled: the pool-epoch value
and the last-epoch value are separate fields, each with its own source, and
``recorded_last_epoch_equals_pool_epoch`` says explicitly when they happen
to be the same epoch (they are, for both census-less runs -- two readings of
one epoch, not one copied from the other).

THE CENSUS READ OUTCOME IS PUBLISHED (Wf-A123 DEFECT C)
--------------------------------------------------------
``census_read`` states whether the census was read (``ok`` / ``missing`` /
``parse_error``), how many rows were read, and the parse error verbatim.  A
census that could not be parsed is therefore never indistinguishable from a
census that simply has no row for a run.

THE POOL'S OWN COMPOSITION (Wf-A121 / I2, additive)
--------------------------------------------------
Every run row publishes ``pool_population_composition``: the epochs the pool
covers, how many rows it holds, and how many of them lie OUTSIDE the run of
record.  MEASURED: ``cfm_c7_fullgen_2026_10_20``'s pool holds 12669 rows
over epochs 0..17, of which 2816 rows (epochs 0..3) belong to a KILLED
PRE-RESUME ATTEMPT while the run of record ran epochs 4..17.  It changes no
existing number.

USAGE
-----
    .venv/bin/python molmetal/scripts/cfm_fragmentation_audit.py
    .venv/bin/python molmetal/scripts/cfm_fragmentation_audit.py \
        --out metrics/by_metric/cfm_fragmentation_audit.json

CPU-only and RDKit-only.  No GPU, no model, no sampler.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts.cfm_fragmentation_gate import (  # noqa: E402
    DEFAULT_MAX_FRAGMENTED_RATIO,
    DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
    VALIDITY_SATURATION,
    evaluate,
    pool_attempts,
    score_run_dir,
)
from molmetal.scripts.cfm_pool_kind import (  # noqa: E402
    GENERATION_MARKER,
    MEASUREMENT_MEANING,
    POOL_KIND_RULE,
    RECONSTRUCTION_MARKER,
    UnknownPoolKindError,
    VERDICT_SCOPE,
    classify_pool,
    require_known_kind,
)
from molmetal.scripts.cfm_pool_epoch_rule import (  # noqa: E402
    pool_epoch_recorded_row,
    pool_population_composition,
    run_last_epoch_recorded_row,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

REPORTS = REPO / "molmetal" / "reports"
CENSUS = REPO / "metrics" / "by_metric" / "cfm_run_census.json"

#: The pool-kind rule (the markers, the classifier and the per-kind
#: SEMANTICS -- ``MEASUREMENT_MEANING`` / ``VERDICT_SCOPE``) has ONE owner:
#: ``molmetal/scripts/cfm_pool_kind.py``.  They are imported above rather
#: than redefined here, so the audit and the C7 audit cannot disagree about
#: what a reconstruction pool's number MEANS.


def _repo_relative(path: Path) -> str:
    """``path`` relative to the repo root when it is inside it, else absolute."""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except (ValueError, OSError):
        return str(path)


def _census_by_run() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """The SSOT's per-run rows, keyed by run name, AND the READ OUTCOME.

    Wf-A123 DEFECT C: this used to catch EVERY exception and return ``{}``,
    so a census that could not be read was indistinguishable from a census
    that simply carries no row for a run.  The consequence was silent: every
    ``gen_validity_end`` went null, the blind list shrank, and the published
    reason said "the census carries no row" about a run whose row merely
    failed to parse.

    The outcome is therefore PUBLISHED, not swallowed:

    ``census_present``      -- the census file exists
    ``census_read``         -- ``ok`` | ``missing`` | ``parse_error``
    ``census_rows_read``    -- how many rows were actually read
    ``census_error``        -- the parse error, verbatim, when there was one

    A caller can now tell "the census has no row for this run" (read ok, N
    rows) from "the census could not be read" (parse_error).
    """
    status: Dict[str, Any] = {
        "census_path": _repo_relative(CENSUS),
        "census_present": CENSUS.is_file(),
        "census_read": "ok",
        "census_rows_read": 0,
        "census_error": None,
    }
    if not CENSUS.is_file():
        status["census_read"] = "missing"
        return {}, status
    try:
        art = json.loads(CENSUS.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — the outcome is PUBLISHED below
        status["census_read"] = "parse_error"
        status["census_error"] = f"{type(exc).__name__}: {exc}"
        return {}, status
    rows = {r["run"]: r for r in (art.get("runs") or []) if r.get("run")}
    status["census_rows_read"] = len(rows)
    return rows, status


def _run_dirs() -> List[Path]:
    """Every report directory that holds a generation pool.

    Discovered with a single ``iterdir`` + ``is_file`` check per candidate —
    this repository is 201 GB and must never be walked recursively.
    """
    if not REPORTS.is_dir():
        return []
    out = []
    for d in sorted(REPORTS.iterdir()):
        if not d.is_dir():
            continue
        if (d / "generated_smiles.csv").is_file():
            out.append(d)
    return out


def _sample_counts(run_dir: Path) -> Dict[int, float]:
    """``epoch -> samples drawn``, from the run's own events.

    The trainer's ``generated_smiles.csv`` holds RDKit-strict SURVIVORS ONLY
    (trainer :4093 gates the append on a non-empty ``smi``, and ``smi`` is
    ``""`` for every molecule the decoder failed to sanitise).  So the number
    of rows in the file is NOT the number of molecules drawn, and a rate over
    the file's rows is a rate over the survivors.

    MEASURED: ``cfm_pcgrad_off_2026_10_15`` has 1175 CSV rows against 1440
    samples drawn — 265 samples (18.4%) produced no SMILES at all and are
    not in the file.  ``cfm_c8_fullgen_2026_10_24`` has 2112 rows against
    2112 drawn — nothing dropped, because every one of its samples parsed.
    """
    out: Dict[int, float] = {}
    ev = run_dir / "events.jsonl"
    if not ev.is_file():
        return out
    try:
        for line in ev.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "gen_validity_n" not in row:
                continue
            try:
                ep = int(float(row.get("epoch", 0)))
                n = float(row["gen_validity_n"])
            except (TypeError, ValueError):
                continue
            out[ep] = n
    except OSError:
        return {}
    return out


def _pool_epochs(csv_path: Path) -> set:
    """The epochs the pool file actually covers.

    A run's ``events.jsonl`` can hold many more epochs than the pool file
    does — ``cfm_ode_link_probe`` has 12 pool rows over 2 epochs while its
    events span dozens.  Summing the drawn count over epochs the pool does
    not cover would divide by molecules that were never in the pool.
    """
    import csv as _csv

    eps = set()
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                try:
                    eps.add(int(float(row.get("epoch", ""))))
                except (TypeError, ValueError):
                    continue
    except OSError:
        return eps
    return eps


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    median = vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0
    return {
        "n": n,
        "min": vals[0],
        "max": vals[-1],
        "mean": sum(vals) / n,
        "median": median,
    }


def audit_run(
    d: Path,
    census_row: Optional[Dict[str, Any]],
    *,
    strict_pool_kind: bool = False,
    census_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Re-score one run's pool and join it with the census row.

    ``census_status`` is the READ OUTCOME published by :func:`_census_by_run`
    (how many rows were read, whether the census could be read at all).  It
    is carried into the run's own row so that a reader of ONE run can tell
    "the census has no row for this run" from "the census could not be
    read" -- Wf-A123 DEFECT C.
    """
    csv_path = d / "generated_smiles.csv"
    if strict_pool_kind:
        # Strict mode REFUSES an unclassified pool instead of quietly
        # bucketing it: the caller sees UnknownPoolKindError, not a number.
        kind = require_known_kind(csv_path)
    else:
        kind = classify_pool(csv_path)
    pipeline = kind.kind
    # The end-to-end denominator, when it is known from outside the pool
    # file: the decode-dump script's OWN attempt count.  It is a DIFFERENT
    # population from the trainer's epoch-level draws in events.jsonl, and
    # for a pool the decode-dump wrote it is the right one.
    attempts = pool_attempts(csv_path) if pipeline == "generation" else None
    scored = score_run_dir(
        d,
        n_attempts=(attempts or {}).get("n_attempted"),
        attempts_source=(attempts or {}).get("source"),
        # The census row lets the POOL-EPOCH RULE resolve WHICH population of
        # this run directory the pool belongs to (the census's own ts-attested
        # run of record) instead of guessing from the directory alone.
        census_row=census_row,
    )
    verdict = evaluate(scored)
    pool = scored.get("pool") or {}
    src = (scored.get("sources") or {}).get("generated_smiles_csv") or {}
    # THE POOL-EPOCH RULE's provenance, published next to the number so a
    # reader can check it without re-deriving it.
    pool_epoch = (scored.get("sources") or {}).get("pool_epoch_validity") or {}
    row = census_row or {}

    n_single = pool.get("n_single")
    # The end-to-end rate needs the number of molecules DRAWN, which is not
    # the number of rows in the file (the file holds survivors only).  Only
    # the epochs the POOL actually covers may be summed: a run's events.jsonl
    # can hold many more epochs than the pool file does.
    counts = _sample_counts(d)
    pool_epochs = _pool_epochs(csv_path)
    if pool_epochs and counts:
        counts = {ep: n for ep, n in counts.items() if ep in pool_epochs}
    total_drawn = sum(counts.values()) if counts else None
    e2e: Optional[float] = None
    e2e_note = None
    denominator_source = None
    if pipeline == "reconstruction":
        e2e_note = (
            "not applicable: a reconstruction pool's rows are real "
            "geometries the bond head decoded, not molecules the sampler drew"
        )
    elif pipeline != "generation":
        # UNKNOWN: the pool's provenance is not established, so its rows
        # cannot be attributed to the sampler either.  Prose only -- e2e
        # stays None in this branch exactly as it does for reconstruction.
        e2e_note = (
            "not applicable: the pool's provenance is not established, so "
            "which code path wrote it is unknown; its rows cannot be "
            "attributed to the sampler (reason: " + kind.reason + ")"
        )
    elif attempts:
        # The pool's OWN attempt count: the honest denominator for a pool the
        # decode-dump wrote, and a DIFFERENT population from the trainer's
        # epoch-level draws.
        denominator_source = attempts.get("source")
        e2e = n_single / attempts["n_attempted"] if (n_single is not None and attempts["n_attempted"]) else None
        e2e_note = (
            "n_single / the decode-dump's OWN attempt count for this pool ("
            + str(attempts.get("source"))
            + "); the pool file holds only the RDKit-strict survivors of "
            "those attempts, so the survival rate IS part of the measurement "
            "and the trainer's epoch-level draw count (a different "
            "population) is not used here"
        )
    elif not counts:
        e2e_note = (
            "not computable: the run has no events.jsonl, so the number of "
            "samples drawn is unknown and the pool's row count (survivors "
            "only) cannot stand in for it"
        )
    elif n_single is None:
        e2e_note = "not computable: no scored rows"
    else:
        e2e = n_single / total_drawn if total_drawn else None
        denominator_source = "events.jsonl:gen_validity_n (the trainer's epoch-level draws)"
        e2e_note = (
            "n_single / samples drawn, where the drawn count is the sum of "
            "gen_validity_n over the run's epochs (events.jsonl); the pool "
            "file holds only the RDKit-strict survivors of those samples"
        )
        n_pool_rows = src.get("n_rows") or 0
        if total_drawn and n_pool_rows / total_drawn < 0.5:
            # The pool and the events are not describing the same sample set:
            # the file holds far fewer rows than the run drew, so dividing by
            # the drawn count would compare a pool against molecules that
            # were never in it.  MEASURED: cfm_ode_link_probe has 12 pool
            # rows against 1536 samples drawn (0.8%).
            e2e = None
            e2e_note = (
                f"not computable: the pool holds {n_pool_rows} rows against "
                f"{total_drawn:.0f} samples drawn "
                f"({n_pool_rows / total_drawn:.1%}), so the file is not a "
                "sample of that run's generation and the two denominators do "
                "not describe the same set of molecules"
            )

    # R2 -- THE CROSS-CHECK AT THE POOL'S OWN EPOCH (Wf-A121 / I2).
    #
    # The recorded side used to be the census's ``gen_single_fragment_ratio_end``
    # -- the run's LAST-epoch aggregate -- set beside the re-derivation, which
    # is over the pool's OWN rows.  Two DIFFERENT epochs were being compared
    # (MEASURED: cfm_objective_terms_2026_10_11's recorded value is its
    # epoch-8 aggregate while its pool is epoch 5), so the difference said
    # nothing about the pool.
    #
    # The recorded side is now read AT THE POOL'S OWN EPOCH, by the rule's
    # owner, so both sides describe the same epoch.  The old value is kept
    # beside it, labelled with its own epoch, because it is still a
    # correctly-labelled number of its own (the census's END value) and it
    # must not be silently dropped.
    #
    # The BLOCK'S PRESENCE is now keyed on the RECORDED SIDE AT THE POOL'S
    # OWN EPOCH (and the pool being re-derived) instead of on the census
    # carrying a last-epoch value.  The census's last-epoch value is a
    # DIFFERENT epoch's number, so a run the census does not carry used to
    # lose its cross-check even when the recorded side at the pool's own
    # epoch was on disk.  No run that has the block today loses it: every
    # one of them has a recorded row at the pool's own epoch.
    recorded_last = row.get("gen_single_fragment_ratio_end")
    rederived = pool.get("single_fragment_ratio")
    agreement = None
    at_pool = pool_epoch_recorded_row(d, csv_path, census_row=census_row)
    recorded_at = (at_pool or {}).get("gen_single_fragment_ratio")
    if recorded_at is not None and rederived is not None:
        recorded_at_epoch = (at_pool or {}).get("epoch")
        if recorded_at_epoch is None:
            recorded_at_epoch = pool_epoch.get("epoch")
        recorded_source = (at_pool or {}).get("source")
        # The epoch of the OLD value.  The census states it as
        # ``integrity.epoch_index_end``; the rule's owner reads the same
        # epoch off the run's last aggregate row, and the two agree on every
        # run the census carries.
        if recorded_last is None:
            # Wf-A123 DEFECT B: the CENSUS's own last-epoch column does not
            # cover this run, but the run's OWN events.jsonl carries the
            # last-epoch aggregate.  Reading it from there is what keeps the
            # published value true; publishing a null here asserted an
            # absence that disk contradicts (MEASURED: the census-less runs
            # ``cfm_c7_fullgen_ep4_partial_2026_10_20`` and
            # ``cfm_c8_fullgen_2026_10_24`` DO carry a last train_epoch row).
            # The census's MISSING COLUMN is what is reported as missing --
            # never the value's existence on disk.
            last_row = run_last_epoch_recorded_row(
                d, csv_path, census_row=census_row
            )
            last_value = (last_row or {}).get("gen_single_fragment_ratio")
            if last_value is not None:
                recorded_last = last_value
                recorded_last_epoch = (last_row or {}).get("epoch")
                recorded_last_source = (
                    f"{(last_row or {}).get('source')} -- the run's OWN "
                    "last-epoch aggregate row, read because the census's "
                    "gen_single_fragment_ratio_end column does not cover this "
                    "run"
                )
            else:
                # Neither the census NOR the run's own events.jsonl carries
                # a last-epoch aggregate.  Say exactly that.
                recorded_last_epoch = None
                if row:
                    recorded_last_source = (
                        "the census carries a row for this run but no last-epoch "
                        "gen_single_fragment_ratio, and the run's own "
                        "events.jsonl carries no last-epoch aggregate row "
                        "either, so no last-epoch value is published for it"
                    )
                else:
                    recorded_last_source = (
                        "the census carries no row for this run, and the run's "
                        "own events.jsonl carries no last-epoch aggregate row "
                        "either, so no last-epoch value is published for it"
                    )
        else:
            recorded_last_epoch = ((row.get("integrity") or {}).get("epoch_index_end"))
            if recorded_last_epoch is None:
                last_row = run_last_epoch_recorded_row(
                    d, csv_path, census_row=census_row
                )
                recorded_last_epoch = (last_row or {}).get("epoch")
                recorded_last_source = (
                    "the run's recorded per-epoch series' LAST aggregate row, "
                    "read by the pool-epoch rule's owner"
                )
            else:
                recorded_last_source = (
                    f"cfm_run_census.json:runs[{d.name}].gen_single_fragment_"
                    "ratio_end -- the run's LAST-epoch aggregate; its epoch is "
                    "the census's own integrity.epoch_index_end"
                )
        abs_diff = (
            abs(float(recorded_at) - float(rederived))
            if (recorded_at is not None and rederived is not None)
            else None
        )
        agreement = {
            # -- the recorded side, AT THE POOL'S OWN EPOCH --------------
            "recorded_single_fragment_ratio": recorded_at,
            "recorded_at_epoch": recorded_at_epoch,
            "recorded_source": recorded_source,
            # -- the recorded side, at the run's LAST epoch (the old value) -
            "recorded_last_epoch_single_fragment_ratio": recorded_last,
            "recorded_last_epoch": recorded_last_epoch,
            "recorded_last_epoch_source": recorded_last_source,
            # -- are the pool's own epoch and the run's last epoch the SAME
            #    epoch?  Wf-A123 DEFECT B: they CAN be (MEASURED: for both
            #    census-less runs they are), and saying so keeps the two
            #    numbers DISTINCTLY labelled instead of implying the
            #    last-epoch value was copied from the pool-epoch one.
            "recorded_last_epoch_equals_pool_epoch": (
                recorded_last_epoch is not None
                and recorded_at_epoch is not None
                and int(recorded_last_epoch) == int(recorded_at_epoch)
            ),
            # -- the re-derived side (UNCHANGED) --------------------------
            "rederived_single_fragment_ratio": rederived,
            "abs_diff": abs_diff,
            "note": (
                "BOTH sides of this cross-check now refer to the POOL'S OWN "
                "EPOCH: the recorded value is the resolved population's "
                "aggregate row at that epoch, and the re-derivation is over "
                "every SMILES row of the pool file -- which, for a multi-epoch "
                "pool, spans EVERY epoch the pool covers, not only its last "
                "one.  A non-zero difference is therefore expected and is NOT "
                "a mismatch by itself.  For a RECONSTRUCTION pool the two "
                "numbers are not even the same measurement, so the difference "
                "carries no information about the generator.  The run's "
                "LAST-epoch value is kept beside this one, labelled with its "
                "own epoch, because it is a DIFFERENT measurement: it "
                "describes the run's final epoch, not the epoch these "
                "molecules were drawn at.  When the run's last epoch IS the "
                "pool's own epoch (MEASURED: it is, for both census-less "
                "runs) the two measurements coincide BY CONSTRUCTION -- they "
                "are two readings of the same epoch, not one copied from the "
                "other -- and recorded_last_epoch_equals_pool_epoch says so "
                "explicitly."
            ),
        }

    # R4 -- THE POOL'S OWN COMPOSITION (additive).
    #
    # A pool is a set of MOLECULES, and a run directory can hold several
    # populations.  This says, in one place, how much of the pool belongs to
    # the run of record and how much to an earlier, killed attempt that wrote
    # into the same directory.  MEASURED:
    # cfm_c7_fullgen_2026_10_20's pool holds 12669 rows over epochs 0..17, of
    # which 2816 rows (epochs 0..3) belong to a KILLED PRE-RESUME ATTEMPT
    # while the run of record ran epochs 4..17.
    #
    # It changes no existing number: it is published next to them.
    try:
        composition = dict(
            pool_population_composition(d, csv_path, census_row=census_row)
        )
    except Exception as exc:  # a composition must never cost a run its row
        composition = {
            "error": f"{type(exc).__name__}: {exc}",
            "note": (
                "the pool's composition could not be resolved; the run's own "
                "numbers are unaffected"
            ),
        }
    # The resolution is published WITH the composition, because the
    # composition's own note assumes the run of record is known.  For a run
    # with no resolvable population the note's "KILLED attempt" wording does
    # not apply, and the resolution says so.
    resolution = (pool_epoch.get("population") or {}).get("resolution")
    composition["population_resolution"] = resolution
    if resolution in (None, "unresolved"):
        # The composition's own note says the rows "belong to an earlier,
        # KILLED attempt that wrote into the same directory".  For a run
        # directory with NO resolvable population that claim does not hold --
        # nothing ran there that a record can attest -- so the attribution is
        # stated here rather than left to a reader to unpick.
        composition["attribution_note"] = (
            f"no population of this run directory could be resolved (the "
            f"population's own resolution is {resolution!r}), so the pool's "
            f"rows cannot be attributed to a run of record at all: the "
            f"composition's 'KILLED attempt' wording does not apply here"
        )

    verdict = dict(verdict)
    verdict["pipeline"] = pipeline
    verdict["applies_to"] = VERDICT_SCOPE[pipeline]
    verdict["measurement_meaning"] = MEASUREMENT_MEANING[pipeline]

    try:
        run_dir_str = str(d.relative_to(REPO))
    except ValueError:
        # A run directory OUTSIDE the repository (a synthetic fixture under
        # tmp_path) has no repo-relative path; the absolute one is published
        # instead of crashing the run.
        run_dir_str = str(d)

    return {
        "run": d.name,
        "run_dir": run_dir_str,
        "pipeline": pipeline,
        # The signal that DECIDED the kind, straight from the rule owner.
        # It is NOT re-derived here: a content-decided pool's marker is
        # "sanitized+metal_z", not the header marker.
        "pipeline_marker": kind.marker,
        # WHY this pool was classified the way it was.  For an unknown pool
        # this is the only place the reason survives -- the row is bucketed
        # as "unknown" but never silently: ``applies_to`` stays
        # "unclassified" and this string says which markers were missing.
        "pool_kind_reason": kind.reason,
        "measurement_meaning": MEASUREMENT_MEANING[pipeline],
        "applies_to": VERDICT_SCOPE[pipeline],
        "n_rows_in_csv": src.get("n_rows"),
        "n_scored": pool.get("n_scored"),
        "n_unparseable": pool.get("n_unparseable"),
        "n_empty": pool.get("n_empty"),
        "n_single": n_single,
        # The trainer's OWN epoch-level draw count.  It is NOT the end-to-end
        # denominator for a pool the decode-dump wrote (that pool is a sample
        # of the decode-dump's attempts, not of the trainer's draws).
        "samples_drawn": total_drawn,
        "samples_drawn_source": "events.jsonl:gen_validity_n (the trainer's epoch-level draws)",
        "end_to_end_denominator": (
            attempts["n_attempted"]
            if attempts
            else (total_drawn if pipeline == "generation" else None)
        ),
        "end_to_end_denominator_source": denominator_source,
        "single_fragment_ratio": rederived,
        # The rate over the pool FILE's own rows (its survivors).  It is NOT
        # the end-to-end rate: a file the decode-dump wrote holds only the
        # RDKit-strict survivors of the samples that were drawn, so this
        # number describes the FILE.  Both denominators are published side by
        # side, each with its source, so neither can be read as the other.
        "single_molecule_rate_over_pool_rows": (
            (n_single / pool["n_input"])
            if (n_single is not None and pool.get("n_input"))
            else None
        ),
        # The gate's own end-to-end rate: over the pool's rows, or over the
        # decode-dump's attempt count when the pool is one of its files.
        "single_molecule_rate_over_declared_denominator": pool.get(
            "single_molecule_rate"
        ),
        "single_molecule_rate_end_to_end": e2e,
        "end_to_end_note": e2e_note,
        "fragmented_ratio": pool.get("fragmented_ratio"),
        "largest_fragment_frac_mean": pool.get("largest_fragment_frac_mean"),
        "n_fragments_mean": pool.get("n_fragments_mean"),
        "mean_heavy_atoms": pool.get("mean_heavy_atoms"),
        "metal_ratio": pool.get("metal_ratio"),
        "csv_column_cross_check": src.get("csv_column_cross_check"),
        # AT WHICH EPOCH this pool's generation quality was measured.  The
        # rule's ONE owner is molmetal/scripts/cfm_pool_epoch_rule.py.
        "pool_epoch": pool_epoch.get("epoch"),
        "pool_epoch_validity": {
            "value": pool_epoch.get("value"),
            "n": pool_epoch.get("n"),
            "source": pool_epoch.get("source"),
            "population": pool_epoch.get("population"),
            "superseded": pool_epoch.get("superseded"),
            "notes": pool_epoch.get("notes"),
        },
        # The pool's own COMPOSITION: which epochs it covers, how many rows
        # lie OUTSIDE the run of record.  Additive -- it changes no number
        # above (Wf-A121 / I2).
        "pool_population_composition": composition,
        "verdict": verdict,
        "census": {
            # Wf-A123 DEFECT C: the READ OUTCOME, so a null in this block is
            # never ambiguous between "the census has no row for this run"
            # and "the census could not be read".
            "census_read": (census_status or {}).get("census_read", "unknown"),
            "census_rows_read": (census_status or {}).get("census_rows_read"),
            "census_row_present": bool(row),
            "gen_validity_end": row.get("gen_validity_end"),
            "gen_validity_max": row.get("gen_validity_max"),
            "gen_single_fragment_ratio_end": row.get("gen_single_fragment_ratio_end"),
            "gen_single_fragment_ratio_max": row.get("gen_single_fragment_ratio_max"),
            "gen_single_molecule_rate_end": row.get("gen_single_molecule_rate_end"),
            "gen_largest_fragment_frac_end": row.get("gen_largest_fragment_frac_end"),
            "gen_molecule_validity_end": row.get("gen_molecule_validity_end"),
            "decode_single_fragment_ratio_end": row.get(
                "decode_single_fragment_ratio_end"
            ),
            "n_epochs_run": row.get("n_epochs_run"),
            "gate_metric": row.get("gate_metric"),
        },
        "recorded_vs_rederived": agreement,
    }


def project_generation_baseline(census: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The generator-side baseline across EVERY run in the SSOT.

    This is the answer to "in most cases the molecules are fragments", and it
    comes from the runs' own trainer-side generation eval, so it covers the
    runs that have no pool file on disk.

    Wf-A115 — the HEADLINE is over the CLEAN runs (contiguous-from-zero
    epoch history AND no nan train_loss), matching the census's own
    ``fragmentation_summary``.  The all-runs distribution is kept beside it:
    the two differ, and the difference is the degenerate runs, which are
    NAMED.
    """
    cond: List[float] = []
    e2e: List[float] = []
    e2e_clean: List[float] = []
    best: List[float] = []
    best_clean: List[float] = []
    validity: List[float] = []
    with_both = 0
    degenerate: List[str] = []
    for row in census.values():
        c = row.get("gen_single_fragment_ratio_end")
        v = row.get("gen_validity_end")
        integ = row.get("integrity") or {}
        clean = bool(
            integ.get("epoch_history_contiguous_from_zero")
            and not integ.get("nan_train_loss")
        )
        if not clean:
            degenerate.append(str(row.get("run")))
        if c is not None:
            cond.append(float(c))
        if v is not None:
            validity.append(float(v))
        # The census already pairs the two columns ROW BY ROW, so its rate
        # is preferred over a re-multiplication here.
        rate = row.get("gen_single_molecule_rate_end")
        if rate is not None:
            with_both += 1
            e2e.append(float(rate))
            if clean:
                e2e_clean.append(float(rate))
        elif c is not None and v is not None:
            with_both += 1
            e2e.append(float(v) * float(c))
            if clean:
                e2e_clean.append(float(v) * float(c))
        b = row.get("gen_single_molecule_rate_best")
        if b is not None:
            best.append(float(b))
            if clean:
                best_clean.append(float(b))
    return {
        "n_runs_in_the_ssot": len(census),
        "n_runs_with_a_conditional_number": len(cond),
        "n_runs_with_both_columns": with_both,
        "gen_single_fragment_ratio_end": _stats(cond),
        "gen_validity_end": _stats(validity),
        # Wf-A115 — the headline: CLEAN runs only.
        "gen_single_molecule_rate_end": _stats(e2e_clean),
        # The run's BEST epoch by the same end-to-end rate.  A last-epoch
        # number alone understates a run whose training oscillates: 2 runs
        # reach a real validity at some epoch and then collapse to 0.0 at
        # their last epoch, and 8 of 27 reach >= 0.50 at their best epoch.
        # A max over a noisy epoch series is optimistically biased, so this
        # is NOT interchangeable with the last-epoch figure — it is the
        # number a run that early-stops at its best epoch would deliver.
        "gen_single_molecule_rate_best": _stats(best_clean),
        "population": (
            "CLEAN runs only: contiguous-from-zero epoch history AND no nan "
            "train_loss. The all-runs distribution is in "
            "`gen_single_molecule_rate_end_all_runs`; the excluded runs are "
            "named in `degenerate_runs_excluded_from_the_headline`"
        ),
        # Wf-A115 — the same numbers over ALL runs, so nothing measured is
        # lost and the reader can see what the clean subset changes.
        "gen_single_molecule_rate_end_all_runs": _stats(e2e),
        "gen_single_molecule_rate_best_all_runs": _stats(best),
        "degenerate_runs_excluded_from_the_headline": sorted(degenerate),
        "n_end_to_end_below_0.50": sum(1 for x in e2e_clean if x < 0.50),
        "n_end_to_end_below_0.10": sum(1 for x in e2e_clean if x < 0.10),
        "n_best_epoch_at_or_above_0.50": sum(1 for x in best_clean if x >= 0.50),
        "n_conditional_below_0.50": sum(1 for x in cond if x < 0.50),
        "n_validity_saturated_but_no_molecule": sum(
            1
            for row in census.values()
            if (row.get("gen_validity_end") or 0) >= VALIDITY_SATURATION
            and (row.get("gen_single_molecule_rate_end") or 1.0) < 0.50
        ),
        "runs_that_collapse_at_their_last_epoch": [
            row.get("run")
            for row in census.values()
            if row.get("gen_validity_end") == 0.0
            and (row.get("gen_validity_max") or 0.0) > 0.5
        ],
        "basis": (
            "the runs' own trainer-side generation eval: "
            "gen_validity_end x gen_single_fragment_ratio_end is the "
            "end-to-end rate of ONE connected molecule per sample"
        ),
        # Wf-A115 — the pointer that was missing: the number that was
        # published as the project baseline is WITHDRAWN.
        "retracted_numbers": {
            "0.572276953125": (
                "WITHDRAWN. It is cfm_r11_eval256's end-to-end rate — a run "
                "with ONE recorded epoch (index 19) of a declared 20, "
                "resumed above epoch 0, train_loss nan. The JSON value of "
                "the same cell is 0.572265625; 0.572276953125 is the "
                "4-decimal CSV rendering. It is the max of the ALL-RUNS "
                "distribution, not of the clean-run headline."
            ),
        },
        "cohort_axis_is_unresolved": (
            "the n_train axis (256 vs 26715) is UNRESOLVED: the ordering "
            "flips with the READING and with the BUDGET, so no cohort median "
            "is a bar. See metrics/by_metric/cfm_run_census.json -> "
            "budget_matched.ordering_flips and by_n_train."
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Project-wide CFM fragmentation audit (CPU-only)."
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "metrics" / "by_metric" / "cfm_fragmentation_audit.json",
    )
    args = ap.parse_args(argv)

    # Strict mode REFUSES an unclassified pool instead of bucketing it.
    # The switch is the environment variable (repo precedent:
    # WFA2_CONFIG_STRICT=1, WF343C4_GATE=1) -- NOT a new argparse flag, so
    # the default launch path is unchanged.
    strict_pool_kind = os.environ.get("CFM_POOL_KIND_STRICT", "") == "1"

    census, census_status = _census_by_run()
    runs = []
    for d in _run_dirs():
        try:
            runs.append(
                audit_run(
                    d,
                    census.get(d.name),
                    strict_pool_kind=strict_pool_kind,
                    census_status=census_status,
                )
            )
        except UnknownPoolKindError as exc:
            # Strict mode REFUSES an unclassified pool: the run is reported as
            # a refusal, never bucketed into a number.
            runs.append(
                {
                    "run": d.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "refusal": "pool_kind_unknown",
                    "pool_kind_reason": exc.reason,
                }
            )
        except Exception as exc:  # a broken pool must not kill the audit
            runs.append({"run": d.name, "error": f"{type(exc).__name__}: {exc}"})

    with_pool = [r for r in runs if "error" not in r]
    gen = [r for r in with_pool if r.get("pipeline") == "generation"]
    rec = [r for r in with_pool if r.get("pipeline") == "reconstruction"]
    unk = [r for r in with_pool if r.get("pipeline") == "unknown"]
    # Wf-A114 verify:B D3 -- a refusal is a ROW, and the summary said
    # nothing about it: `n_runs_with_a_pool` silently read 12 when 13 runs
    # were considered and one was REFUSED (strict mode) or errored.  The
    # refusal is now counted and named, so the reader can tell "12 pools"
    # from "13 runs, one refused".
    refused = [r for r in runs if r.get("refusal")]
    errored = [r for r in runs if "error" in r and not r.get("refusal")]

    def _side(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        cond = [
            float(r["single_fragment_ratio"])
            for r in rows
            if r.get("single_fragment_ratio") is not None
        ]
        e2e = [
            float(r["single_molecule_rate_end_to_end"])
            for r in rows
            if r.get("single_molecule_rate_end_to_end") is not None
        ]
        return {
            "n_pools": len(rows),
            "single_fragment_ratio": _stats(cond),
            "single_molecule_rate_end_to_end": _stats(e2e),
            # WHICH denominator each end-to-end rate used.  Two populations
            # are in play (the trainer's epoch-level draws and a decode-dump
            # pool's own attempts), so the reader must be able to tell them
            # apart without reading every run row.
            "end_to_end_denominator_sources": sorted(
                {
                    str(r.get("end_to_end_denominator_source"))
                    for r in rows
                    if r.get("end_to_end_denominator_source")
                }
            ),
            "gate_pass": sum(
                1 for r in rows if (r.get("verdict") or {}).get("status") == "PASS"
            ),
            "gate_fail": sum(
                1 for r in rows if (r.get("verdict") or {}).get("status") == "FAIL"
            ),
            "gate_skip": sum(
                1 for r in rows if (r.get("verdict") or {}).get("status") == "SKIP"
            ),
        }

    generation_side = _side(gen)
    reconstruction_side = _side(rec)
    unknown_side = _side(unk)

    # The blind-metric signature, project-wide, over GENERATION pools only:
    # validity saturated while the pool is fragmented.
    blind = []
    for r in gen:
        v = (r.get("verdict") or {})
        gv = (r.get("census") or {}).get("gen_validity_end")
        if (
            v.get("blind_validity_signature")
            or (
                gv is not None
                and float(gv) >= VALIDITY_SATURATION
                and r.get("fragmented_ratio") is not None
                and float(r["fragmented_ratio"]) > DEFAULT_MAX_FRAGMENTED_RATIO
            )
        ):
            pev = r.get("pool_epoch_validity") or {}
            blind.append(
                {
                    "run": r["run"],
                    # UNCHANGED: the CENSUS's last-epoch value.  It stays
                    # census-sourced and stays null where the census has no
                    # row for the run.
                    "gen_validity_end": gv,
                    "fragmented_ratio": r.get("fragmented_ratio"),
                    "single_fragment_ratio": r.get("single_fragment_ratio"),
                    # Wf-A121C -- this column used to read a key the run row
                    # does NOT carry, so it was null for EVERY entry while
                    # the run's OWN published end-to-end rate sat one level
                    # down in the verdict block.  It now carries the run's
                    # own published number (the verdict's), not a
                    # re-derivation.
                    "single_molecule_rate": v.get("single_molecule_rate"),
                    # The three LABELLED rates, side by side, so a reader can
                    # never guess which denominator a rate is over.
                    "single_molecule_rate_over_pool_rows": r.get(
                        "single_molecule_rate_over_pool_rows"
                    ),
                    "single_molecule_rate_over_declared_denominator": r.get(
                        "single_molecule_rate_over_declared_denominator"
                    ),
                    "single_molecule_rate_end_to_end": r.get(
                        "single_molecule_rate_end_to_end"
                    ),
                    # Wf-A121C -- the pool-epoch validity the per-run verdict
                    # ALREADY publishes, added here under its own names:
                    # the value, its n, and the epoch it was measured at.
                    "gen_validity_at_pool_epoch": pev.get("value"),
                    "gen_validity_n": pev.get("n"),
                    "pool_epoch": r.get("pool_epoch"),
                }
            )

    # Cross-check: does the re-derivation agree with the run's own column?
    checked = [
        r for r in with_pool
        if (r.get("csv_column_cross_check") or {}).get("n_checked")
    ]
    mismatches = [
        {
            "run": r["run"],
            "pipeline": r.get("pipeline"),
            "n_checked": r["csv_column_cross_check"]["n_checked"],
            "n_mismatch": r["csv_column_cross_check"]["n_mismatch"],
        }
        for r in checked
        if r["csv_column_cross_check"]["n_mismatch"]
    ]

    art = {
        "metric": "cfm_fragmentation_audit",
        "tag": "MEASURED",
        "date": "2026-10-30",
        "wave": "Wf-A114",
        "purpose": (
            "Project-wide record of the CFM fragmentation failure: every "
            "generation pool on disk re-scored with the trainer's own "
            "decode_chemistry, joined with the SSOT's per-run rows, and "
            "SPLIT by which code path wrote the pool so a decoder "
            "reconstruction is never reported as a generator measurement."
        ),
        "direction": (
            "single_fragment_ratio (over the parsed rows) and "
            "single_molecule_rate (over ALL rows) are the fraction of the "
            "pool that is ONE connected molecule -- HIGHER IS BETTER.  "
            "fragmented_ratio is the complement -- LOWER IS BETTER."
        ),
        "thresholds": {
            "max_fragmented_ratio": DEFAULT_MAX_FRAGMENTED_RATIO,
            "min_largest_fragment_frac": DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
            "validity_saturation": VALIDITY_SATURATION,
            "basis": (
                "declared policy, not a discovered optimum; the measured "
                "calibration anchor is the decoder's own single_fragment_ratio "
                "on REAL clouds (0.9375 on the c8 val cohort) against the same "
                "run's generation side (0.348)"
            ),
        },
        "pipeline_classification": {
            "rule": POOL_KIND_RULE,
            "markers": {
                "generation": GENERATION_MARKER,
                "reconstruction": RECONSTRUCTION_MARKER,
            },
            "canonical_source": "molmetal/scripts/cfm_pool_kind.py",
            "measurement_meaning": MEASUREMENT_MEANING,
            "verdict_scope": VERDICT_SCOPE,
            "strict_pool_kind": strict_pool_kind,
        },
        "summary": {
            "n_runs_considered": len(runs),
            "n_runs_with_a_pool": len(with_pool),
            "n_runs_refused": len(refused),
            "refused_runs": sorted(r["run"] for r in refused),
            "n_runs_errored": len(errored),
            "errored_runs": sorted(r["run"] for r in errored),
            "n_generation_pools": len(gen),
            "n_reconstruction_pools": len(rec),
            "n_unknown_pools": len(unk),
            "generation_side": generation_side,
            "reconstruction_side": reconstruction_side,
            "unknown_side": unknown_side,
            "n_blind_validity_signature": len(blind),
            "n_pools_with_csv_cross_check": len(checked),
            "n_pools_with_csv_mismatch": len(mismatches),
        },
        "project_generation_baseline": project_generation_baseline(census),
        # Wf-A123 DEFECT C: the census READ OUTCOME, published.  Before this,
        # a census that could not be parsed and a census that simply had no
        # row for a run were indistinguishable -- both produced {}.
        "census_read": census_status,
        "blind_validity_signature": blind,
        "csv_cross_check_mismatches": mismatches,
        "runs": runs,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(str(args.out), json.dumps(art, indent=2))
    print(f"[cfm-frag-audit] {len(with_pool)} run(s) with a pool")
    print(
        f"  pools: generation={len(gen)}  reconstruction={len(rec)}  "
        f"unknown={len(unk)}"
    )
    # Wf-A115 — a run that REFUSED to be scored is a REFUSAL IS A SUCCESS
    # (the pool kind is not classifiable), not a missing run.  It is
    # counted here so "13 runs with a pool" and "42 runs considered" can be
    # reconciled by a reader.
    if refused or errored:
        print(
            f"  refused={len(refused)}  errored={len(errored)}  "
            f"(a refusal is a success: the pool kind is not classifiable)"
        )
    gs = generation_side
    print(
        f"  GENERATION pools  single_fragment_ratio(over survivors) "
        f"min={gs['single_fragment_ratio']['min']} "
        f"max={gs['single_fragment_ratio']['max']} "
        f"mean={gs['single_fragment_ratio']['mean']}"
    )
    print(
        f"  GENERATION pools  single_molecule_rate_end_to_end(per sample "
        f"drawn) min={gs['single_molecule_rate_end_to_end']['min']} "
        f"max={gs['single_molecule_rate_end_to_end']['max']} "
        f"mean={gs['single_molecule_rate_end_to_end']['mean']}"
    )
    print(
        f"  gate (generation pools)  PASS={gs['gate_pass']}  "
        f"FAIL={gs['gate_fail']}  SKIP={gs['gate_skip']}   "
        f"blind-validity signature={len(blind)}"
    )
    base = art["project_generation_baseline"]
    print(
        f"  project baseline over "
        f"{base['gen_single_molecule_rate_end']['n']} CLEAN run(s) of "
        f"{base['n_runs_with_both_columns']} run(s) with both columns: "
        f"end-to-end single-molecule rate "
        f"median={base['gen_single_molecule_rate_end']['median']}  "
        f"below 0.50: {base['n_end_to_end_below_0.50']}"
    )
    if mismatches:
        print(f"  csv column mismatches: {mismatches}")
    print(f"[cfm-frag-audit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
