# SPDX-License-Identifier: Apache-2.0
"""Honest-fragmentation gate for CFM generation pools (CPU-only).

WHY THIS EXISTS
---------------
``gen_validity`` cannot tell a molecule from a pile of loose atoms.  It counts
a sample as valid when RDKit can sanitise the string, and ``C.C.C.C``
sanitises.  MEASURED on ``molmetal/reports/cfm_c8_fullgen_2026_10_24``:
``gen_validity`` read **1.000** at all three epochs while
``gen_single_fragment_ratio`` read **0.203 / 0.200 / 0.348** — i.e. 65-80% of
the "valid" molecules were not one connected molecule at all.  The same
metric reached 1.000 on a pool whose largest fragment held 94.68% of the heavy
atoms, i.e. one blob plus a few orphans.

That blindness is not new and it is not c8's: of the 43 CFM runs on disk that
wrote ``per_epoch_metrics.csv``, 41 carry an honest
``gen_single_fragment_ratio``, and it ranges from 0.0000 to 0.8772 — the
higher the number the BETTER, because it is the fraction of the pool that is a
single connected fragment.  The SSOT (``metrics/by_metric/cfm_run_census.json``)
recorded none of it, and its ``gate_metric`` was never a fragmentation metric,
so nothing in the project's own machinery could see the failure.

This gate is the missing piece: it scores a generation pool for connectivity
and returns PASS / FAIL / SKIP with exit codes 0 / 1 / 2, in the same shape as
the other project gates (``wf343c3_data_quality_check.py``).

WHAT IT MEASURES
----------------
Per decoded sample, using the trainer's OWN ``decode_chemistry`` so the
definition cannot drift from the one the metrics use:

  * ``n_fragments``         — number of connected components
  * ``largest_fragment_frac`` — largest fragment's heavy atoms / total heavy atoms
  * ``has_metal``           — the corpus is metal complexes

and aggregated over the pool:

  * ``single_fragment_ratio``   — of the PARSED rows, the fraction that is ONE
    fragment (HIGHER IS BETTER).  This is the trainer's own denominator, so it
    is the number comparable to its ``gen_single_fragment_ratio`` column.
  * ``single_molecule_rate``    — of ALL rows, the fraction that is ONE
    connected, RDKit-parseable molecule (HIGHER IS BETTER).  This is the
    honest end-to-end rate, and it is the one the verdict uses: a string
    RDKit refuses is not a molecule, so conditioning on "it parsed" hides
    the failure instead of measuring it.
  * ``fragmented_ratio``        — 1 - the rate the verdict used (LOWER IS BETTER)
  * ``largest_fragment_frac_mean``

WHY TWO DENOMINATORS
--------------------
The two differ exactly when some samples do not parse, and they say different
things.  MEASURED on the three full-corpus runs: ``cfm_fullcorpus_b64``
recorded ``gen_validity`` 0.0859 with ``gen_single_fragment_ratio`` 0.7879 at
epoch 11, i.e. 78.8% of the *parseable* samples were one molecule, but only
0.0859 x 0.7879 = **0.0677** of all samples were — the other 91.4% were
strings RDKit rejected outright.  Reading the conditional number alone
overstates the generator by more than 10x.  The end-to-end rate is what the
model actually produced.

AND THE DENOMINATOR IS NOT ALWAYS THE POOL'S OWN ROWS
-----------------------------------------------------
A pool file holds the molecules that SURVIVED, so its row count is a rate over
the survivors.  For the trainer's own pools the honest denominator is the
run's epoch-level draw count (``events.jsonl``).  For a pool written by the
decode-dump script it is that script's OWN attempt count: the pool of record
(``cfm_fullcorpus_2026_10_10``) holds 66 survivors of **192** attempts, so
the end-to-end rate over it is 64/192 = **0.3333**, not 64/66 = 0.9697.  The
two denominators are different populations and mixing them is a category
error, so the denominator that was used is recorded next to every rate
(``end_to_end_denominator`` / ``end_to_end_denominator_source``).

THE THRESHOLDS ARE POLICY, NOT DISCOVERED OPTIMA
------------------------------------------------
``--max-fragmented-ratio`` (default 0.50) and ``--min-largest-fragment-frac``
(default 0.90) are a declared bar, not a measured knee.  The one calibration
anchor that IS measured is the decoder's own behaviour on REAL clouds: on the
c8 val cohort the same decoder read ``decode_single_fragment_ratio`` 0.9375
(``decode_fragments_mean`` 1.068), i.e. it turns real metal-complex geometry
into a connected molecule 93.75% of the time.  The generation side of the same
run read 0.348.  The gap between those two numbers is the thing this gate
exists to make impossible to ignore, and the thresholds are set far below the
decoder's reference so that a pool only passes when it is closer to the
decoder-on-real-clouds behaviour than to the prior.

Read the thresholds as "the bar this project declares", and change them
deliberately — with the change recorded — rather than as an empirical optimum.

USAGE
-----
    # score a run directory (reads generated_smiles.csv and/or per_epoch_metrics.csv)
    .venv/bin/python molmetal/scripts/cfm_fragmentation_gate.py \
        --run-dir molmetal/reports/cfm_c8_fullgen_2026_10_24 --epoch 2

    # score a bare SMILES pool
    .venv/bin/python molmetal/scripts/cfm_fragmentation_gate.py \
        --smiles-csv molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv

    # machine-readable verdict
    .venv/bin/python molmetal/scripts/cfm_fragmentation_gate.py \
        --run-dir <dir> --out metrics/by_metric/<name>.json

CPU-only and RDKit-only.  No GPU, no model, no sampler.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts.cfm_pool_epoch_rule import (  # noqa: E402
    POOL_EPOCH_RULE,
    pool_epoch_validity,
)
from molmetal.scripts.cfm_pool_kind import classify_pool  # noqa: E402

#: Default bar: at most this fraction of the pool may be multi-fragment.
DEFAULT_MAX_FRAGMENTED_RATIO = 0.50

#: Default bar: the mean largest fragment must hold at least this share of
#: the heavy atoms.
DEFAULT_MIN_LARGEST_FRAGMENT_FRAC = 0.90

#: A validity this high with a fragmented pool is the blind-metric signature.
VALIDITY_SATURATION = 0.99

#: The free-generation panel artifact: the decode-dump's OWN attempt count for
#: the pool of record.  The trainer's ``events.jsonl`` records a DIFFERENT
#: population (the trainer's own epoch-level draws), so the two denominators
#: must never be mixed.
FREE_GENERATION_PANEL = (
    REPO / "metrics" / "by_metric" / "cfm_free_generation_panel_seed0.json"
)


def _mentions_run(text: str, run_name: str) -> bool:
    """True when ``text`` names ``run_name`` as a WHOLE token.

    A plain substring test would also match ``cfm_fullcorpus_2026_10_10_v2``,
    so the name is bounded by non-identifier characters on both sides.
    """
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(run_name)}(?![A-Za-z0-9_])",
            text or "",
        )
    )


def pool_attempts(csv_path: Path) -> Optional[Dict[str, Any]]:
    """The number of SAMPLES the pool was drawn from, and where it came from.

    Two sources, in order:

    1. **the pool's own ``n_attempted`` column** -- the decode-dump writer
       stamps it on every row it writes, so it is the writer's own count and
       needs no external artifact;
    2. **the free-generation panel artifact**, when it names THIS run -- the
       panel records the decode-dump's own attempt count (192 for the pool of
       record), which is a different population from the trainer's
       epoch-level draws in ``events.jsonl``.

    Returns ``None`` when neither source is available: the caller then falls
    back to the pool's own row count and says so.
    """
    path = Path(csv_path)
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            if "n_attempted" in header:
                for row in reader:
                    raw = (row.get("n_attempted") or "").strip()
                    try:
                        n_attempted = int(float(raw))
                    except (TypeError, ValueError):
                        break
                    if n_attempted > 0:
                        return {
                            "n_attempted": n_attempted,
                            "source": "the pool's own 'n_attempted' column",
                            "source_path": str(path),
                        }
                    break
    except OSError:
        pass

    if not FREE_GENERATION_PANEL.is_file():
        return None
    try:
        panel = json.loads(FREE_GENERATION_PANEL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    run_name = path.parent.name
    named = _mentions_run(str(panel.get("checkpoint") or ""), run_name) or (
        _mentions_run(str(panel.get("how") or ""), run_name)
    )
    if not named:
        return None
    n_attempted = ((panel.get("generation") or {}) or {}).get("n_attempted")
    try:
        n_attempted = int(n_attempted)
    except (TypeError, ValueError):
        return None
    if n_attempted <= 0:
        return None
    return {
        "n_attempted": n_attempted,
        "source": (
            "metrics/by_metric/cfm_free_generation_panel_seed0.json:"
            "generation.n_attempted"
        ),
        "source_path": str(FREE_GENERATION_PANEL),
        "note": (
            "the decode-dump script's OWN attempt count for this run -- a "
            "different population from the trainer's epoch-level draws in "
            "events.jsonl"
        ),
    }


def _trainer_decode_chemistry():
    """The trainer's own fragment facts, so the definition cannot drift.

    Imported lazily: the trainer module is ~19.9k LOC and the gate must stay
    usable as a standalone scoring tool.
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import decode_chemistry

    return decode_chemistry


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    try:
        v = float(row.get(key, "") or "nan")
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def score_smiles_pool(
    smiles: Sequence[str],
    *,
    expected_metal_z: Optional[frozenset] = None,
    n_attempts: Optional[int] = None,
    attempts_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Score decoded SMILES for connectivity.

    Returns the aggregate + the per-sample facts needed to explain a verdict.
    ``unparseable`` counts strings RDKit refused; they are NOT silently
    dropped — a pool that is 40% unparseable is a different failure from a
    pool that is 40% fragmented, and the gate must be able to say which.

    ``n_attempts`` is the number of SAMPLES the pool was drawn from, when that
    is known from outside the file (the decode-dump's own attempt count).  It
    replaces the pool's row count as the end-to-end denominator: the file
    holds the survivors only, so a rate over its rows is a rate over the
    survivors.  ``attempts_source`` names where the number came from and
    travels with the rate.
    """
    decode_chemistry = _trainer_decode_chemistry()
    rows: List[Dict[str, Any]] = []
    row_indexes: List[int] = []
    unparseable = 0
    empty = 0
    n_input = len(smiles)
    for i, s in enumerate(smiles):
        s = (s or "").strip()
        if not s:
            empty += 1
            continue
        chem = decode_chemistry(s, expected_metal_z=expected_metal_z)
        if chem is None:
            unparseable += 1
            continue
        rows.append(chem)
        # The index of the INPUT string each scored row came from.  Without
        # it a caller cannot line the scored rows back up with the source
        # rows: unparseable and empty strings are skipped, so the two
        # sequences have different lengths and a positional zip silently
        # MISALIGNS everything after the first skip.  MEASURED 2026-10-30:
        # the c7 pool has one unparseable row at index 10370, and the naive
        # zip reported 2299 "mismatches" against a re-derivation that in fact
        # agrees on all 12668 scored rows.
        row_indexes.append(i)
    n = len(rows)
    # The end-to-end denominator.  Normally the pool's own rows; when the
    # pool's own attempt count is known, THAT is the denominator (the file
    # holds the survivors only).
    use_attempts = bool(n_attempts and int(n_attempts) > 0)
    denominator = int(n_attempts) if use_attempts else n_input
    denominator_source = (
        attempts_source
        if use_attempts
        else "the pool file's own rows (n_input)"
    )
    if n == 0:
        return {
            "n_input": n_input,
            "n_scored": 0,
            "n_empty": empty,
            "n_unparseable": unparseable,
            "n_single": 0,
            "single_fragment_ratio": None,
            "single_molecule_rate": None,
            "fragmented_ratio": None,
            "largest_fragment_frac_mean": None,
            "metal_ratio": None,
            "mean_heavy_atoms": None,
            "n_fragments_mean": None,
            "end_to_end_denominator": denominator,
            "end_to_end_denominator_source": denominator_source,
            "rows": [],
            "row_indexes": [],
        }
    single = sum(1 for r in rows if int(r.get("n_fragments", 0) or 0) == 1)
    # The end-to-end rate: ONE connected, parseable molecule per SAMPLE.
    # The denominator is the number of samples drawn, so an unparseable or
    # empty string counts AGAINST the pool — it is not a molecule and it does
    # not get to leave the denominator just because it failed to parse.
    rate = single / denominator if denominator else None
    return {
        "n_input": n_input,
        "n_scored": n,
        "n_empty": empty,
        "n_unparseable": unparseable,
        "n_single": single,
        "row_indexes": row_indexes,
        "single_fragment_ratio": single / n,
        "single_molecule_rate": rate,
        "fragmented_ratio": (1.0 - rate) if rate is not None else None,
        "largest_fragment_frac_mean": sum(
            float(r.get("largest_fragment_frac", 0.0) or 0.0) for r in rows
        )
        / n,
        "metal_ratio": sum(1 for r in rows if r.get("has_metal")) / n,
        "mean_heavy_atoms": sum(int(r.get("n_heavy", 0) or 0) for r in rows) / n,
        "n_fragments_mean": sum(int(r.get("n_fragments", 0) or 0) for r in rows) / n,
        "end_to_end_denominator": denominator,
        "end_to_end_denominator_source": denominator_source,
        "rows": rows,
    }


def read_generated_smiles(path: Path, epoch: Optional[int] = None) -> List[Dict[str, str]]:
    """Read a ``generated_smiles.csv``, optionally restricted to one epoch."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if epoch is not None:
        rows = [r for r in rows if _f(r, "epoch") == float(epoch)]
    return rows


def read_epoch_metrics(path: Path, epoch: Optional[int] = None) -> Optional[Dict[str, str]]:
    """Last (or requested) row of a ``per_epoch_metrics.csv``."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return None
    if epoch is None:
        return rows[-1]
    for r in rows:
        if _f(r, "epoch") == float(epoch):
            return r
    return None


def score_run_dir(
    run_dir: Path,
    *,
    epoch: Optional[int] = None,
    n_attempts: Optional[int] = None,
    attempts_source: Optional[str] = None,
    census_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score one run directory from whatever honest artefacts it has.

    Two sources, in priority order:
      1. ``generated_smiles.csv`` — the pool itself, re-scored with the
         trainer's definition (the CSV's own ``n_frags_raw`` /
         ``largest_frag_frac`` columns are cross-checked against it).
      2. ``per_epoch_metrics.csv`` — the trainer's own aggregate, used when no
         SMILES pool exists.

    ``n_attempts`` / ``attempts_source`` carry the number of samples the
    pool was drawn from when that is known from outside the file; see
    :func:`pool_attempts`.

    ``census_row`` is the run's row in ``cfm_run_census.json``; it lets the
    pool-epoch rule resolve WHICH population of the run directory the pool
    belongs to (the census's own ts-attested run of record).  It is optional
    and only ever NARROWS the resolution — the rule falls back to the event
    stream's own segmentation without it.

    THE POOL-EPOCH RULE
    -------------------
    A pool is a set of MOLECULES, so the only epoch at which a validity
    number describes it is the epoch the pool was drawn at.  The rule's ONE
    owner is ``molmetal/scripts/cfm_pool_epoch_rule.py``:

        "The generation-quality number published for a pool is the resolved
        POPULATION's aggregate value AT THE POOL'S OWN LAST EPOCH -- the
        aggregate row that carries gen_validity_n, never the per-seed
        gen_eval rows."

    Before this rule the gate read ``per_epoch_metrics.csv``'s LAST ROW and
    published that.  MEASURED (Wf-A120): three of the thirteen pools on disk
    published a validity measured at a different epoch than their own
    molecules.
    """
    out: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "epoch": epoch,
        "sources": {},
    }
    smiles_csv = run_dir / "generated_smiles.csv"
    metrics_csv = run_dir / "per_epoch_metrics.csv"

    if smiles_csv.is_file():
        rows = read_generated_smiles(smiles_csv, epoch)
        pool = score_smiles_pool(
            [r.get("smiles", "") for r in rows],
            n_attempts=n_attempts,
            attempts_source=attempts_source,
        )
        out["sources"]["generated_smiles_csv"] = {
            "path": str(smiles_csv),
            "n_rows": len(rows),
        }
        # The pool's KIND: which code path wrote this CSV.  The gate used to
        # be pool-kind BLIND -- it scored a pool without ever recording
        # whether the rows were the sampler's free generation or the bond
        # head's reconstruction of real geometry.  The two mean DIFFERENT
        # things (see molmetal/scripts/cfm_pool_kind.py), so a score without
        # its kind is a number nobody can read.  Metadata only: the gate's
        # numbers come from score_smiles_pool, not from the header.
        kind = classify_pool(smiles_csv)
        out["sources"]["generated_smiles_csv"].update(
            {
                "pool_kind": kind.kind,
                "pool_kind_marker": kind.marker,
                "pool_kind_reason": kind.reason,
            }
        )
        # Cross-check: the CSV's own columns must agree with the re-derivation.
        # The scored rows are lined up by INPUT INDEX, not by position: the
        # pool skips empty and unparseable strings, so a positional zip
        # misaligns every row after the first skip (see score_smiles_pool).
        mism = 0
        checked = 0
        per_row = list(pool.pop("rows", []))
        indexes = list(pool.pop("row_indexes", []))
        for chem, i in zip(per_row, indexes):
            r = rows[i]
            csv_frag = _f(r, "n_frags_raw")
            if csv_frag is None:
                continue
            checked += 1
            if abs(csv_frag - float(chem.get("n_fragments", 0) or 0)) > 1e-9:
                mism += 1
        out["sources"]["generated_smiles_csv"].update(
            {
                "n_rows_scored": pool["n_scored"],
                "n_unparseable": pool["n_unparseable"],
                "csv_column_cross_check": {
                    "n_checked": checked,
                    "n_mismatch": mism,
                },
                **pool,
            }
        )
        out["pool"] = out["sources"]["generated_smiles_csv"]
        # THE POOL-EPOCH RULE.  The pool's own last epoch decides at which
        # epoch the run's generation quality may be published for it; the
        # rule's owner resolves the population and the aggregate row.
        pev = pool_epoch_validity(
            run_dir,
            smiles_csv,
            census_row=census_row,
        )
        # The resolved population's RAW aggregate rows are NOT published.
        # The rule's own public API drops them for exactly this reason
        # ("the gate publishes this dict, and publishing every aggregate row
        # of the population would bloat every artifact that carries it"),
        # and the one row that matters -- the aggregate row AT the pool's own
        # epoch -- is published separately as ``population["pool_epoch_row"]``.
        # Without this the audit artifact grows ~5x on every run.
        population = pev.get("population")
        if isinstance(population, dict):
            population.pop("rows", None)
        out["sources"]["pool_epoch_validity"] = pev
    else:
        out["sources"]["generated_smiles_csv"] = None

    if metrics_csv.is_file():
        row = read_epoch_metrics(metrics_csv, epoch)
        if row is not None:
            out["sources"]["per_epoch_metrics_csv"] = {
                "path": str(metrics_csv),
                "epoch": _f(row, "epoch"),
                "gen_validity": _f(row, "gen_validity"),
                "gen_validity_val": _f(row, "gen_validity_val"),
                "gen_validity_n": _f(row, "gen_validity_n"),
                "gen_single_fragment_ratio": _f(row, "gen_single_fragment_ratio"),
                "gen_largest_fragment_frac": _f(row, "gen_largest_fragment_frac"),
                "gen_molecule_validity": _f(row, "gen_molecule_validity"),
                "decode_single_fragment_ratio": _f(row, "decode_single_fragment_ratio"),
                "decode_fragments_mean": _f(row, "decode_fragments_mean"),
                "decode_ratio": _f(row, "decode_ratio"),
            }
    else:
        out["sources"]["per_epoch_metrics_csv"] = None

    return out


def evaluate(
    scored: Dict[str, Any],
    *,
    max_fragmented_ratio: float = DEFAULT_MAX_FRAGMENTED_RATIO,
    min_largest_fragment_frac: float = DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
) -> Dict[str, Any]:
    """Turn a scored run into a verdict + the reasons behind it.

    PASS / FAIL / SKIP, with the same exit-code convention as the other
    project gates: 0 = PASS, 1 = FAIL, 2 = SKIP.
    """
    pool = scored.get("pool")
    met = (scored.get("sources") or {}).get("per_epoch_metrics_csv")
    # THE POOL-EPOCH RULE's verdict input.  ``None`` when the run was not
    # scored from a pool (the ``--smiles-csv`` path), and it is IGNORED when
    # the caller asked for one specific epoch -- an explicit epoch is the
    # caller's own choice and the rule does not override it.
    pool_ep = (scored.get("sources") or {}).get("pool_epoch_validity") or {}
    rule_applies = bool(pool_ep) and scored.get("epoch") is None

    # Three numbers, in increasing honesty:
    #   single_fragment_ratio — over the PARSED rows.  This is the trainer's
    #     own denominator, so it is the number comparable to its
    #     `gen_single_fragment_ratio` column.
    #   single_molecule_rate  — over ALL samples.  The honest end-to-end rate.
    #   validity              — the RDKit parse rate, the blind metric.
    #
    # The verdict is driven by the end-to-end rate whenever it exists, and it
    # says which basis it used, so a conditional number is never mistaken for
    # an end-to-end one.
    single: Optional[float] = None
    rate: Optional[float] = None
    largest: Optional[float] = None
    n_scored: Optional[int] = None
    basis: Optional[str] = None

    if pool and pool.get("n_scored"):
        single = pool.get("single_fragment_ratio")
        rate = pool.get("single_molecule_rate")
        largest = pool.get("largest_fragment_frac_mean")
        n_scored = pool.get("n_scored")
        basis = "pool_end_to_end"
    elif met:
        single = met.get("gen_single_fragment_ratio")
        largest = met.get("gen_largest_fragment_frac")
        n_scored = met.get("gen_validity_n")
        validity_col = met.get("gen_validity")
        if validity_col is not None and single is not None:
            # gen_validity is the parse rate over the same samples, so the
            # product is the end-to-end single-molecule rate without needing
            # the pool itself.
            rate = validity_col * single
            basis = "trainer_columns_derived"
        else:
            basis = "trainer_conditional_column"

    if single is None and rate is None:
        return {
            "status": "SKIP",
            "reasons": [
                "no honest fragmentation number available: neither a "
                "generated_smiles.csv pool nor a gen_single_fragment_ratio "
                "column in per_epoch_metrics.csv"
            ],
            "single_fragment_ratio": None,
            "single_molecule_rate": None,
            "fragmented_ratio": None,
            "largest_fragment_frac_mean": None,
            "n_scored": n_scored,
            "basis": None,
            "end_to_end_denominator": (pool or {}).get("end_to_end_denominator"),
            "end_to_end_denominator_source": (pool or {}).get(
                "end_to_end_denominator_source"
            ),
            "gen_validity": None,
            "gen_validity_provenance": {
                "rule": POOL_EPOCH_RULE,
                "epoch": pool_ep.get("epoch"),
                "n": None,
                "source": None,
                "superseded": None,
                "population": pool_ep.get("population"),
                "notes": ["no fragmentation number available, so no validity "
                          "is published either"],
            },
            "blind_validity_signature": False,
            "decoder_reference_single_fragment_ratio": (
                met.get("decode_single_fragment_ratio") if met else None
            ),
        }

    # The rate the verdict is made on.  The end-to-end rate when we have it;
    # otherwise the conditional one, explicitly labelled as such.
    measured = rate if rate is not None else single
    fragmented = 1.0 - measured
    reasons: List[str] = []
    status = "PASS"
    if fragmented > max_fragmented_ratio:
        status = "FAIL"
        reasons.append(
            f"fragmented_ratio {fragmented:.4f} > {max_fragmented_ratio:.4f}: "
            f"{fragmented * 100:.1f}% of the pool is NOT one connected molecule"
        )
    if largest is not None and largest < min_largest_fragment_frac:
        status = "FAIL"
        reasons.append(
            f"largest_fragment_frac_mean {largest:.4f} < "
            f"{min_largest_fragment_frac:.4f}: the pool's dominant fragment "
            f"does not hold the atoms"
        )

    # The blind-metric signature: validity saturated while the pool is
    # fragmented.  Reported separately because it is the thing that let this
    # survive for 41 runs.
    #
    # THE POOL-EPOCH RULE decides WHICH validity this is.  When the rule
    # applies, the published number is the resolved population's aggregate at
    # the POOL'S OWN LAST EPOCH -- and when no aggregate row carries that
    # epoch, NO number is published: a validity measured at another epoch is
    # not this pool's validity.
    validity = None
    validity_provenance: Dict[str, Any] = {
        "rule": POOL_EPOCH_RULE,
        "epoch": None,
        "n": None,
        "source": None,
        "superseded": None,
        "population": None,
        "notes": [],
    }
    if rule_applies:
        validity = pool_ep.get("value")
        validity_provenance.update(
            {
                "epoch": pool_ep.get("epoch"),
                "n": pool_ep.get("n"),
                "source": pool_ep.get("source"),
                "superseded": pool_ep.get("superseded"),
                "population": pool_ep.get("population"),
                "notes": list(pool_ep.get("notes") or []),
            }
        )
        if validity is None and pool_ep.get("epoch") is not None:
            # The pool's epoch is known and NOTHING on disk measures the
            # generation quality there.  Publishing the run's last-row value
            # would publish a number measured at a different epoch, so the
            # honest answer is no number.
            validity_provenance["notes"].append(
                "no aggregate row carries the pool's own epoch, so NO validity "
                "is published: the run's last-row value measures a different "
                "epoch than this pool's molecules"
            )
    elif met:
        v, n = met.get("gen_validity_val"), met.get("gen_validity_n")
        if v is not None and n:
            validity = v / n
        elif met.get("gen_validity") is not None:
            validity = met.get("gen_validity")
        # THE LABELLED NO-POOL FALLBACK (Wf-A121 / I2, R8).
        #
        # The rule does not apply here, so the published validity is NOT a
        # pool-epoch value.  WHICH non-pool-epoch value it is depends on why
        # the rule did not apply:
        #
        #   * an EXPLICIT epoch was requested -- the value is that epoch's
        #     own aggregate row, which the caller asked for by name;
        #   * there is NO pool -- the value is the run's RECORDED per-epoch
        #     series' LAST row, i.e. the run's FINAL epoch.  It is labelled
        #     as such, because a reader would otherwise take it for a
        #     pool-epoch number like every other published validity.
        #
        # No published number changes here: this branch is not exercised by
        # any pool-bearing run (every pool-bearing run takes the rule's
        # branch above).
        met_epoch = met.get("epoch")
        if scored.get("epoch") is not None:
            epoch_role = "the_requested_epoch"
            role_note = (
                f"the caller requested epoch {scored.get('epoch')}, so the "
                f"pool-epoch rule does not apply and the published validity "
                f"is that epoch's own aggregate row"
            )
        else:
            epoch_role = "recorded_series_last_row"
            role_note = (
                f"there is no pool for this score, so the published validity "
                f"is the run's RECORDED per-epoch series' LAST row (epoch "
                f"{met_epoch}) -- NOT a pool-epoch value: no pool's own epoch "
                f"exists here to measure it at"
            )
        validity_provenance.update(
            {
                "epoch": met_epoch,
                "n": met.get("gen_validity_n"),
                "source": met.get("path"),
                # WHICH epoch-role the published value has.  "pool_epoch" is
                # what the rule's branch publishes; these two are the
                # labelled fallbacks.
                "epoch_role": epoch_role,
                "notes": [
                    "the pool-epoch rule does not apply to this score (no "
                    "pool, or an explicit epoch was requested)",
                    role_note,
                ],
            }
        )
    blind_validity = bool(
        validity is not None
        and validity >= VALIDITY_SATURATION
        and fragmented > max_fragmented_ratio
    )
    if blind_validity:
        reasons.append(
            f"BLIND-METRIC SIGNATURE: gen_validity {validity:.4f} is saturated "
            f"while {fragmented * 100:.1f}% of the pool is fragmented — the "
            f"sanitisation test cannot see this failure"
        )

    if status == "PASS" and not reasons:
        reasons.append(
            f"fragmented_ratio {fragmented:.4f} <= {max_fragmented_ratio:.4f} "
            f"and largest_fragment_frac_mean {largest:.4f} >= "
            f"{min_largest_fragment_frac:.4f}"
            if largest is not None
            else f"fragmented_ratio {fragmented:.4f} <= {max_fragmented_ratio:.4f}"
        )

    # The basis is part of the verdict, not a footnote: a conditional rate
    # that reads 0.79 and an end-to-end rate that reads 0.07 are the same
    # pool seen through two denominators, and only one of them is what the
    # model produced.
    if basis == "pool_end_to_end":
        reasons.append(
            f"basis: pool_end_to_end — single_molecule_rate {measured:.4f} "
            f"(one connected molecule per sample) over "
            f"{pool.get('end_to_end_denominator', pool.get('n_input'))} "
            f"samples (denominator: "
            f"{pool.get('end_to_end_denominator_source')}); the conditional "
            f"single_fragment_ratio over the parsed rows is {single:.4f}"
        )
    elif basis == "trainer_columns_derived":
        reasons.append(
            f"basis: trainer_columns_derived — gen_validity x "
            f"gen_single_fragment_ratio = {measured:.4f} end-to-end "
            f"(no pool on disk to re-score)"
        )
    else:
        reasons.append(
            "basis: trainer_conditional_column — NO end-to-end rate available; "
            f"the conditional single_fragment_ratio {measured:.4f} is over "
            "the PARSED rows only and overstates the generator"
        )

    return {
        "status": status,
        "reasons": reasons,
        "single_fragment_ratio": single,
        "single_molecule_rate": rate,
        "fragmented_ratio": fragmented,
        "largest_fragment_frac_mean": largest,
        "n_scored": n_scored,
        "basis": basis,
        # WHICH denominator the end-to-end rate used.  It is not always the
        # pool's own row count: a pool written by the decode-dump script holds
        # the survivors of its own draws, and its own attempt count is the
        # honest denominator.
        "end_to_end_denominator": (pool or {}).get("end_to_end_denominator"),
        "end_to_end_denominator_source": (pool or {}).get(
            "end_to_end_denominator_source"
        ),
        "gen_validity": validity,
        # WHERE the published validity was measured, so a reader can check it
        # without re-deriving it: the epoch, the n (gen_validity_n), the
        # source string, and -- where a number was replaced -- the superseded
        # value and the reason it was superseded.
        "gen_validity_provenance": validity_provenance,
        "blind_validity_signature": blind_validity,
        "decoder_reference_single_fragment_ratio": (
            met.get("decode_single_fragment_ratio") if met else None
        ),
    }


def _atomic_write(path: Path, text: str) -> None:
    """Write through the project's atomic-write protocol (§5.6)."""
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(str(path), text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Honest-fragmentation gate for CFM generation pools (CPU-only)."
    )
    ap.add_argument("--run-dir", type=Path, help="a report directory to score")
    ap.add_argument("--smiles-csv", type=Path, help="a generated_smiles.csv to score")
    ap.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="restrict to one epoch (default: the last epoch present)",
    )
    ap.add_argument(
        "--max-fragmented-ratio",
        type=float,
        default=DEFAULT_MAX_FRAGMENTED_RATIO,
        help="FAIL if more than this fraction of the pool is multi-fragment",
    )
    ap.add_argument(
        "--min-largest-fragment-frac",
        type=float,
        default=DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
        help="FAIL if the mean largest fragment holds less than this share",
    )
    ap.add_argument("--out", type=Path, default=None, help="write the verdict JSON here")
    ap.add_argument(
        "--strict", action="store_true", help="treat SKIP as FAIL (exit 1)"
    )
    args = ap.parse_args(argv)

    if args.run_dir is None and args.smiles_csv is None:
        ap.error("one of --run-dir / --smiles-csv is required")

    if args.run_dir is not None:
        attempts = pool_attempts(args.run_dir / "generated_smiles.csv")
        scored = score_run_dir(
            args.run_dir,
            epoch=args.epoch,
            n_attempts=(attempts or {}).get("n_attempted"),
            attempts_source=(attempts or {}).get("source"),
        )
    else:
        rows = read_generated_smiles(args.smiles_csv, args.epoch)
        attempts = pool_attempts(args.smiles_csv)
        pool = score_smiles_pool(
            [r.get("smiles", "") for r in rows],
            n_attempts=(attempts or {}).get("n_attempted"),
            attempts_source=(attempts or {}).get("source"),
        )
        # Same pool-kind metadata as the --run-dir path: the gate must never
        # emit a score without saying which pipeline produced the pool.
        kind = classify_pool(args.smiles_csv)
        scored = {
            "run_dir": str(args.smiles_csv),
            "epoch": args.epoch,
            "sources": {
                "generated_smiles_csv": {
                    "path": str(args.smiles_csv),
                    "pool_kind": kind.kind,
                    "pool_kind_marker": kind.marker,
                    "pool_kind_reason": kind.reason,
                    **pool,
                }
            },
            "pool": scored_pool_alias(pool),
        }

    verdict = evaluate(
        scored,
        max_fragmented_ratio=args.max_fragmented_ratio,
        min_largest_fragment_frac=args.min_largest_fragment_frac,
    )
    artifact = {
        "metric": "cfm_fragmentation_gate",
        "tag": "MEASURED",
        "purpose": (
            "Score a CFM generation pool for connectivity. gen_validity cannot "
            "see fragmentation (a pile of loose atoms sanitises), so this gate "
            "is the only place the failure is visible."
        ),
        "thresholds": {
            "max_fragmented_ratio": args.max_fragmented_ratio,
            "min_largest_fragment_frac": args.min_largest_fragment_frac,
            "basis": (
                "declared policy, not a discovered optimum; the measured "
                "calibration anchor is the decoder's own single_fragment_ratio "
                "on REAL clouds (0.9375 on the c8 val cohort) against the same "
                "run's generation side (0.348)"
            ),
        },
        "verdict": verdict,
        "scored": scored,
    }

    status = verdict["status"]
    print(f"[cfm-frag-gate] {status}")
    for r in verdict["reasons"]:
        print(f"  - {r}")
    rate = verdict.get("single_molecule_rate")
    cond = verdict.get("single_fragment_ratio")
    if rate is not None:
        print(
            f"  single_molecule_rate={rate:.4f} (end-to-end, one connected "
            f"molecule per sample)  fragmented_ratio={verdict['fragmented_ratio']:.4f}"
        )
        denominator = verdict.get("end_to_end_denominator")
        if denominator is not None:
            print(
                f"  end-to-end denominator: {denominator} "
                f"({verdict.get('end_to_end_denominator_source')})"
            )
    if cond is not None:
        print(
            f"  single_fragment_ratio={cond:.4f} (over the PARSED rows — the "
            f"trainer's own denominator)"
        )
    if verdict.get("largest_fragment_frac_mean") is not None:
        print(
            f"  largest_fragment_frac_mean="
            f"{verdict['largest_fragment_frac_mean']:.4f}"
        )
    ref = verdict.get("decoder_reference_single_fragment_ratio")
    if ref is not None:
        print(f"  decoder reference on REAL clouds: single_fragment_ratio={ref:.4f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(args.out, json.dumps(artifact, indent=2))
        print(f"[cfm-frag-gate] wrote {args.out}")

    if status == "SKIP":
        return 1 if args.strict else 2
    return 0 if status == "PASS" else 1


def scored_pool_alias(pool: Dict[str, Any]) -> Dict[str, Any]:
    """The pool dict without the per-sample rows (they are bulky)."""
    return {k: v for k, v in pool.items() if k not in ("rows", "row_indexes")}


if __name__ == "__main__":
    raise SystemExit(main())
