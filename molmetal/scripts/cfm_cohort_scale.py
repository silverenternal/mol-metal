#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ONE owner for the CFM cohort-SCALE rule (CPU-only; the census is the only I/O).

WHY THIS MODULE EXISTS
-----------------------
Two audits measured the SAME generator over pools whose TRAINING cohorts
differ by two orders of magnitude, and neither of them said so in the number
it published:

  * ``molmetal/scripts/c7_scale_matched_pools.py`` held the thresholds
    (``FULL_CORPUS_N_TRAIN = 26715``, ``SMALL_TRAIN_CUTOFF = 1024``) and the
    classifier (``tier_of``), so the C7 audit COULD talk about cohort scale;
  * ``molmetal/scripts/cfm_sa_yield_audit.py`` had no tier axis at all, so
    its headline ``combined`` rate pooled the n_train=26715 panel of record
    with two n_train=256 pools and published the result as THE generator's
    rate.

Two owners of one rule is not a style problem: the two audits cannot AGREE or
DISAGREE about cohort scale, they can only silently differ -- and a reader of
the SA artifact had no way to tell that its denominator mixed a full-corpus
pool with two small-cohort pools.  This module is the ONE owner of the rule;
``c7_scale_matched_pools.py`` imports from here (a later phase wires that up;
until then the two copies are byte-equal in behaviour and this module says so
in :func:`tier_of`).

THE RULE, AS CODE
------------------
``tier_of(n_train)`` returns:

  * ``"unknown"``     -- ``n_train`` is ``None`` (the cohort was not
    established; a rate over an unknown cohort is not comparable to any
    known-cohort rate, and is never silently treated as one);
  * ``"full_corpus"`` -- ``n_train >= 26715`` (the whole corpus);
  * ``"small"``       -- ``n_train < 1024``;
  * ``"mid"``         -- everything else.

The semantics are IDENTICAL to the c7 rule that predates this module:
``full_corpus`` if ``n_train >= FULL_CORPUS_N_TRAIN`` else ``small`` if
``n_train < SMALL_TRAIN_CUTOFF`` else ``mid``, and ``unknown`` when
``n_train`` is ``None``.  The boundaries are inclusive on the full-corpus
side and exclusive on the small side: 26715 -> ``full_corpus``,
26714 -> ``mid``, 1024 -> ``mid``, 1023 -> ``small``.

WHERE n_train COMES FROM
------------------------
The run census (``metrics/by_metric/cfm_run_census.json``) records, per run,
the ``n_train`` of the training cohort that produced it.  A pool of generated
SMILES lives under ``molmetal/reports/<run>/generated_smiles.csv``, so the
run directory name IS the run name the census is keyed by.  :func:`census_n_train`
looks the run up and returns its ``n_train`` (``None`` when the run is not in
the census, or when the census recorded a null cohort); :func:`cohort_scale_of_run`
turns that into the ``(n_train, tier)`` pair every consumer labels its
numbers with.

A RUN NOT IN THE CENSUS IS NOT AN ERROR
----------------------------------------
It is a pool whose training cohort was never recorded.  The honest answer is
``tier="unknown"`` -- NOT a guess, and NOT the pool's path or date silently
promoted to a cohort size.  The caller records the provenance string that
says exactly which of the two happened (``not in the census`` vs ``recorded
as null``), so the difference is visible in the artifact.

USAGE
-----
    from molmetal.scripts.cfm_cohort_scale import cohort_scale_of_run, tier_of

    n_train, tier = cohort_scale_of_run("cfm_fullcorpus_2026_10_10")
    # -> (26715, "full_corpus")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: The full corpus: 26,715 training complexes.  A run trained on it is a
#: ``full_corpus`` cohort, whatever else it did.
FULL_CORPUS_N_TRAIN = 26715

#: A run trained on fewer than this many complexes is a ``small`` cohort.
SMALL_TRAIN_CUTOFF = 1024

#: Every tier this module can return, in reporting order (largest cohort
#: first, ``unknown`` last because it is not a cohort).
TIERS = ("full_corpus", "mid", "small", "unknown")

#: One-line statement of the rule, for payloads and docs.
TIER_RULE = (
    f"full_corpus if n_train >= {FULL_CORPUS_N_TRAIN}; small if "
    f"n_train < {SMALL_TRAIN_CUTOFF}; else mid; unknown if n_train is None"
)

#: Where the rule lives.  It has ONE owner; consumers import it.
COHORT_SCALE_SOURCE = "molmetal/scripts/cfm_cohort_scale.py"

#: The run census: the artifact that records each run's training cohort.
CENSUS_PATH = PROJECT_ROOT / "metrics" / "by_metric" / "cfm_run_census.json"

#: The census is one file and is read once per process; the cache is keyed by
#: the resolved path so a test can point the lookup at its own census.
_CENSUS_CACHE: Dict[str, Dict[str, Optional[int]]] = {}


def tier_of(n_train: Optional[int]) -> str:
    """The cohort tier of a training cohort of size ``n_train``.

    ``None`` -> ``"unknown"``; ``>= 26715`` -> ``"full_corpus"``;
    ``< 1024`` -> ``"small"``; otherwise ``"mid"``.
    """
    if n_train is None:
        return "unknown"
    return (
        "full_corpus"
        if n_train >= FULL_CORPUS_N_TRAIN
        else ("small" if n_train < SMALL_TRAIN_CUTOFF else "mid")
    )


def _census_index(census_path: Optional[Path] = None) -> Dict[str, Optional[int]]:
    """``{run_name: n_train}`` from the run census (cached per path)."""
    path = Path(census_path) if census_path is not None else CENSUS_PATH
    key = str(path)
    cached = _CENSUS_CACHE.get(key)
    if cached is not None:
        return cached
    index: Dict[str, Optional[int]] = {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        # A missing or unreadable census is NOT an error: it means no run has
        # an established cohort, so every tier is "unknown".
        _CENSUS_CACHE[key] = index
        return index
    for row in doc.get("runs") or []:
        if not isinstance(row, dict):
            continue
        run = row.get("run")
        if not isinstance(run, str) or run in index:
            continue
        n_train = row.get("n_train")
        index[run] = n_train if isinstance(n_train, int) else None
    _CENSUS_CACHE[key] = index
    return index


def census_n_train(
    run_name: Optional[str], census_path: Optional[Path] = None
) -> Optional[int]:
    """The training cohort size of ``run_name``, or ``None`` if unknown."""
    if not run_name:
        return None
    return _census_index(census_path).get(run_name)


def cohort_scale_of_run(
    run_name: Optional[str], census_path: Optional[Path] = None
) -> Tuple[Optional[int], str]:
    """``(n_train, tier)`` for ``run_name``.

    ``n_train`` is ``None`` when the census does not record it; the tier is
    then ``"unknown"`` -- never a guess.
    """
    n_train = census_n_train(run_name, census_path)
    return n_train, tier_of(n_train)


def n_train_source(run_name: Optional[str], census_path: Optional[Path] = None) -> str:
    """The provenance of a pool's ``n_train`` -- what a reader must trust.

    It names the census AND the exact record the number came from, or it
    says why there is no number.
    """
    path = Path(census_path) if census_path is not None else CENSUS_PATH
    try:
        shown = path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = path
    if not run_name:
        return f"unknown: no run name (the pool path does not name a run)"
    index = _census_index(census_path)
    if run_name not in index:
        return (
            f"unknown: run {run_name!r} is not in {shown}; its training "
            f"cohort was never recorded"
        )
    if index[run_name] is None:
        return (
            f"{shown}:runs[run={run_name}].n_train is null -- the cohort was "
            f"recorded as unknown"
        )
    return f"{shown}:runs[run={run_name}].n_train"


def run_name_of_pool(pool_path: Path) -> Optional[str]:
    """The run name a pool belongs to, from its ``reports/<run>/`` directory.

    ``molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv`` ->
    ``"cfm_fullcorpus_2026_10_10"``, which is the key the census uses.  A
    pool that does not live under a run directory has no run name (``None``)
    and therefore no established cohort.
    """
    parent = Path(pool_path).parent
    if not parent.name:
        return None
    return parent.name


def cohort_scale_fields(
    run_name: Optional[str], census_path: Optional[Path] = None
) -> Dict[str, object]:
    """The cohort-scale labelling that travels NEXT TO the numbers.

    Every consumer gets the same three fields, so a reader who sees only a
    rate can still tell which training cohort the denominator came from.
    """
    n_train, tier = cohort_scale_of_run(run_name, census_path)
    return {
        "tier": tier,
        "n_train": n_train,
        "n_train_source": n_train_source(run_name, census_path),
        "run_name": run_name,
        "cohort_scale_source": COHORT_SCALE_SOURCE,
        "cohort_scale_rule": TIER_RULE,
    }
