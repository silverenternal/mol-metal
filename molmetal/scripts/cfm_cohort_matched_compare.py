# SPDX-License-Identifier: Apache-2.0
"""A cohort- and budget-MATCHED comparison of two CFM runs' end-to-end
generation rate -- and a refusal machine for every comparison that is not.

WHY THIS SCRIPT EXISTS
-----------------------
The CFM line's generation quality was never measured on the right denominator.
``gen_validity`` is an RDKit PARSE rate (a run of "C.C.C" parses), and
``gen_single_fragment_ratio`` is conditioned on the PARSED ROWS ONLY.  The
honest end-to-end rate is therefore::

    e2e = (gen_validity_val / gen_validity_n) * gen_single_fragment_ratio

i.e. the share of ALL generated samples that are ONE connected molecule.

Three blind spots made the published numbers untrustworthy, and this script
closes the fourth:

  1. the reported metric cannot see fragmentation;
  2. the early-stop criterion on disk (``bond_type_f1``) is blind to
     connectivity, so the runs that stopped early stopped on the wrong axis;
  3. the criterion that would see it (``molecule_validity``) was never
     MEASURED -- no ``per_epoch_metrics.csv`` on disk carries that column;
  4. (this script) a number quoted from one cohort, at one epoch, at one
     budget was compared against a number from another cohort, at another
     epoch, at another budget.  A cohort MEDIAN is not a bar: it is a
     summary of runs that did not train the same amount, on the same data.

WHAT A COMPARISON MUST MATCH (seven axes, all of them refusals)
---------------------------------------------------------------
  AXIS 1  ``cohort``      -- the RUN OF RECORD's ``n_train`` AND ``n_val``
                              equal.  (A cohort is the PAIR.  MEASURED: the
                              corpus holds 13 distinct ``(n_train, n_val)``
                              pairs, and the ``n_train=26715`` label alone
                              mixes ``n_val=192`` (8 runs) with ``n_val=6714``
                              (5 runs) -- two different validation protocols
                              under one label.)  The cohort is NOT read out of
                              the per-epoch file's own ``config`` block: that
                              file is written ONCE to a FIXED path, so a later
                              run launched into the same ``--out-dir``
                              overwrites it with a DIFFERENT run's config.
                              MEASURED: ``cfm_fullcorpus_2026_10_10`` holds a
                              14-epoch run of record at 26715/192 beside two
                              1-epoch CPU smokes at 64/32 whose ``done``
                              events are LATER, and the surviving file is the
                              SMOKE's.  The cohort therefore comes from the
                              ONE ts-attested owner -- the census's own
                              resolver (``events.jsonl`` + ``cfm_cohort_
                              recovery``) -- and never from the clobbered
                              ``config`` block.
  AXIS 2  ``epoch_budget``-- both runs must have a RECORDED row at the
                              compared epoch index.  A run that stopped at
                              epoch 4 never contributes an epoch-10 value.
  AXIS 2b ``not_measured``-- the contributing row must satisfy
                              ``gen_validity_n > 0``.  The trainer writes
                              ``gen_validity`` / ``gen_single_fragment_ratio``
                              as ``0.0`` when the generation eval did NOT
                              run, which is indistinguishable from a
                              measured 0.  ``gen_eval_n`` is NOT the flag
                              (MEASURED: 11 rows carry ``gen_eval_n == 0``
                              while ``gen_validity_n > 0``).
  AXIS 3  ``reading``     -- one reading for both arms.  The ``best``
                              reading (a max over a noisy series, which the
                              census itself labels "optimistically biased")
                              is allowed ONLY when both runs' recorded epoch
                              sets are IDENTICAL.
  AXIS 4  ``history``     -- each run's recorded epoch set must be contiguous
                              AND start at 0, so epoch index means the same
                              training amount; and ``train_loss`` must never be
                              NaN (or absent) ANYWHERE IN THE RUN'S WHOLE
                              recorded history, not only on the contributing
                              row (Wf-A114 verify:D D3: a run whose loss went
                              nan at epoch 3 and recovered by epoch 5 has a
                              broken history whatever row is compared).
  AXIS 5  ``provenance``  -- the compared number must come from the
                              GENERATION pool (never the reconstruction
                              pool), the two runs' pool construction must
                              match at the compared epoch (``gen_solver``,
                              ``gen_eval_n``, ``gen_eval_seeds``), and the
                              run's on-disk pool must be a GENERATION pool
                              per ``cfm_pool_kind`` (the ONE owner of that
                              rule).
  AXIS 7  ``population``  -- the compared EPOCH SERIES must be the RUN OF
                              RECORD's own record.  The per-epoch file is
                              written ONCE to a FIXED path, so it can hold a
                              DIFFERENT population's series under this run's
                              name; comparing it publishes another
                              population's epochs as the run's.  The
                              cross-check is the census's own resolver, not a
                              re-implementation: the verdict names the
                              population it found and why.

A REFUSAL IS A SUCCESS.  The script writes ``verdict: "REFUSED"`` with the
list of unmatched axes and the nearest valid pair it could find on disk.

READ THE JSON, NEVER THE CSV
----------------------------
``per_epoch_metrics.csv`` rounds every rate to 4 decimals.  MEASURED:
``molmetal/reports/cfm_r11_eval256/per_epoch_metrics.csv`` stores
``gen_single_fragment_ratio=0.8277`` where the JSON stores
``0.827683615819209``, so the CSV product is 0.572276953125 and the JSON
product is 0.572265625 (delta 1.13e-5).  A bar that disagrees with its own
artifact cannot be checked, so this script reads the JSON only.

THE BAR IS INTEGRITY-GATED (Wf-A114 verify:D D1)
-----------------------------------------------
``--bar`` used to scan every run x every epoch for a matching value and, on a
hit, print ``bar = <value> MEASURED <path>``.  That re-minted a WITHDRAWN
number: ``cfm_r11_eval256`` records exactly ONE epoch (index 19) out of a
declared 20, its history is not contiguous from zero and its ``train_loss`` is
``nan`` -- and its epoch-19 end-to-end rate is ``0.572265625``, the number the
census withdrew.  A trace to disk is therefore NOT enough: the run that
carries the value must also pass the ``history`` axis.  A match on a
degenerate run is reported as ``REJECTED`` with the reason, never as
``MEASURED``.

CPU-ONLY.  No training, no sampling, no CUDA.  Standard library only, plus
``molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file`` (AGENTS.md
§5.6) and ``molmetal/scripts/cfm_pool_kind.py`` (Wf-A114 cluster C).

CLI
---
argparse is justified: AGENTS.md §5.11 binds LAUNCHES (``scripts/launch_*.sh``)
and this script trains nothing and has no hyperparameter, so a YAML would carry
none.  Every flag is a SELECTOR (which run, which reading, where to write),
never a training knob.

    --run-a / --run-b      run names under --reports-root
    --reports-root         default: molmetal/reports
    --reading {epoch,best} default: epoch (the fixed, budget-matched index)
    --epoch N              override the compared epoch INDEX (K-1 of the
                            budget rule; the compared budget is N+1)
    --out-dir              default:
                            molmetal/reports/wfa114_cohort_matched_compare/<a>__vs__<b>
    --bar FLOAT            optional threshold, reported with a label and
                            traced to disk when a run on disk carries it
    --census PATH          optional cross-check ONLY (never a gate)

Exit codes follow AGENTS.md §4: 0 = verdict, 1 = refused, 2 = skip.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import wf_cfm_run_census as _census  # noqa: E402
from molmetal.scripts.cfm_cohort_recovery import (  # noqa: E402
    COHORT_SOURCE_UNATTESTED,
    recover_cohort,
)
from molmetal.scripts.cfm_pool_kind import (  # noqa: E402  (Wf-A114 cluster C)
    KIND_GENERATION,
    KIND_RECONSTRUCTION,
    classify_pool,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

#: Default root of the run corpus.
DEFAULT_REPORTS_ROOT = REPO_ROOT / "molmetal" / "reports"

#: Default output directory parent (AGENTS.md reports-dir convention).
DEFAULT_OUT_PARENT = REPO_ROOT / "molmetal" / "reports" / "wfa114_cohort_matched_compare"

#: The axis names, as they appear in ``unmatched_axes``.
AXIS_COHORT = "cohort"
AXIS_EPOCH_BUDGET = "epoch_budget"
AXIS_NOT_MEASURED = "not_measured"
AXIS_READING = "reading"
AXIS_HISTORY = "history"
AXIS_PROVENANCE = "provenance"
#: Wf-A116b -- the compared epoch series must be the RUN OF RECORD's own.
AXIS_POPULATION = "population"

ALL_AXES = (
    AXIS_COHORT,
    AXIS_EPOCH_BUDGET,
    AXIS_NOT_MEASURED,
    AXIS_READING,
    AXIS_HISTORY,
    AXIS_PROVENANCE,
    AXIS_POPULATION,
)

#: The ONLY fields the verdict's number is derived from.  The reconstruction
#: pool (``decode_ratio_val`` / ``decode_single_fragment_ratio``) is the bond
#: head applied to REAL geometry -- it is the DECODER's fidelity, never the
#: generator's quality, so it is never read for a verdict.
GENERATION_FIELDS = ("gen_validity_val", "gen_validity_n", "gen_single_fragment_ratio")
RECONSTRUCTION_FIELDS = ("decode_ratio_val", "decode_single_fragment_ratio")

#: Pool-construction fields that must match between the two arms.
POOL_CONSTRUCTION_FIELDS = ("gen_solver", "gen_eval_n", "gen_eval_seeds")

#: Candidate names for a run's on-disk generation pool.  ``per_epoch_metrics.csv``
#: is NOT a pool and must never be classified as one.
POOL_CSV_PATTERNS = ("*smiles*.csv", "*pool*.csv", "*generated*.csv")

#: The connectivity columns a GENERATION pool may carry.  The trainer's own
#: path writes ``n_frags_raw``; the decode-dump script writes ``n_fragments``
#: from the same ``decode_chemistry`` definition.  A pool written by the
#: decode-dump path is a generation pool too (the 2026-10-12+ ODE path), so
#: its connectivity must be readable -- otherwise the corroboration silently
#: skips it.
CONNECTIVITY_COLUMNS = ("n_frags_raw", "n_fragments")

#: The caveat that travels with every verdict.  ``gen_eval_n`` is filled in
#: from the compared rows so the pool size quoted is the pool size MEASURED.
CAVEAT_TEMPLATE = (
    "one run is one run: no error bar, finite generation pool "
    "(gen_eval_n={pool_n}) => sampling noise of unknown sign; 2 seeds is still n=2"
)

#: The caveat used when no compared row is available (a refusal).
CAVEAT = (
    "one run is one run: no error bar, finite generation pool "
    "(gen_eval_n=see the contributing row) => sampling noise of unknown sign; "
    "2 seeds is still n=2"
)


def build_caveat(rows: Dict[str, Optional[Dict[str, Any]]]) -> str:
    """The caveat with the MEASURED pool size of the compared arms."""
    sizes = []
    for row in rows.values():
        if not row:
            continue
        n = _num(row.get("gen_eval_n"))
        if n is not None:
            sizes.append(int(n))
    if not sizes:
        return CAVEAT
    pool_n = sizes[0] if len(set(sizes)) == 1 else "/".join(str(s) for s in sizes)
    return CAVEAT_TEMPLATE.format(pool_n=pool_n)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _num(value: Any) -> Optional[float]:
    """One numeric value, NaN-aware.  ``None`` for absent / unparseable / nan.

    A value of ``0`` or ``0.0`` is falsy in Python, so an ``or "nan"`` style
    fallback would LOSE a measured zero.  This never does.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() == "nan":
            return None
        try:
            x = float(s)
        except ValueError:
            return None
    else:
        try:
            x = float(value)
        except (TypeError, ValueError):
            return None
    return None if x != x else x


def _is_nan(value: Any) -> bool:
    """True when the value is nan, absent, or not a finite number at all.

    An unparseable cell is treated as nan: a loss that cannot be read is not a
    verified finite loss.
    """
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "nan"):
            return True
        try:
            x = float(s)
        except ValueError:
            return True
        return x != x
    if value is None:
        return True
    try:
        x = float(value)
    except (TypeError, ValueError):
        return True
    return x != x


def _is_absent(value: Any) -> bool:
    """True when the key holds nothing at all (missing key or an empty cell)."""
    return value is None or (isinstance(value, str) and value.strip() == "")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    """Repo-relative POSIX path when possible, else the absolute path."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# the run record
# --------------------------------------------------------------------------
@dataclass
class RunRecord:
    """One run directory, read from ``per_epoch_metrics.json`` (FULL precision).

    Wf-A116b -- the COHORT here is the run of RECORD's cohort, NOT the
    ``config`` block of whichever run wrote the file last.  ``n_train`` /
    ``n_val`` answer the RESOLVED cohort; ``config_n_train`` /
    ``config_n_val`` answer the raw block, so the payload can show both and
    a reader can see when they disagree.
    """

    name: str
    path: Path
    config: Dict[str, Any] = field(default_factory=dict)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    by_epoch: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    epochs: List[int] = field(default_factory=list)
    source: str = ""
    error: Optional[str] = None
    # -- the run of record's cohort, and where it came from ---------------
    cohort: Tuple[Any, Any] = (None, None)
    cohort_source: str = ""
    cohort_reason: str = ""
    cohort_evidence: Optional[Dict[str, Any]] = None
    # -- the population cross-check (the census's OWN resolver) -----------
    population_verdict: str = ""
    population_reason: str = ""
    per_epoch_file_is_run_of_record: bool = True

    @property
    def n_train(self) -> Any:
        """The run of record's ``n_train`` (resolved, never the config block)."""
        return self.cohort[0]

    @property
    def n_val(self) -> Any:
        """The run of record's ``n_val`` (resolved, never the config block)."""
        return self.cohort[1]

    @property
    def config_n_train(self) -> Any:
        """The ``n_train`` in the on-disk file's own ``config`` block."""
        return self.config.get("n_train")

    @property
    def config_n_val(self) -> Any:
        """The ``n_val`` in the on-disk file's own ``config`` block."""
        return self.config.get("n_val")

    @property
    def config_cohort_agrees_with_resolved(self) -> bool:
        """Does the on-disk ``config`` block state the RESOLVED cohort?

        ``False`` is the clobbering signature: the file that survived at the
        fixed path was written by a DIFFERENT population than the run of
        record, so its config block describes that other population.
        """
        return (
            self.config_n_train == self.n_train
            and self.config_n_val == self.n_val
        )

    @property
    def last_epoch(self) -> Optional[int]:
        return self.epochs[-1] if self.epochs else None

    def row_at(self, epoch: int) -> Optional[Dict[str, Any]]:
        return self.by_epoch.get(epoch)

    def history_contiguous_from_zero(self) -> bool:
        if not self.epochs:
            return False
        if self.epochs[0] != 0:
            return False
        return self.epochs == list(range(0, self.epochs[-1] + 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run": self.name,
            "run_path": _rel(self.path),
            "source": self.source,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "epochs_recorded": list(self.epochs),
            "n_epochs_recorded": len(self.epochs),
            "last_epoch": self.last_epoch,
            "declared_epochs": self.config.get("epochs"),
            "history_contiguous_from_zero": self.history_contiguous_from_zero(),
            # -- Wf-A116b: where the cohort came from, and whether the
            #    on-disk per-epoch file IS the run of record's record.
            "cohort_source": self.cohort_source,
            "cohort_reason": self.cohort_reason,
            "cohort_evidence": self.cohort_evidence,
            "config_n_train": self.config_n_train,
            "config_n_val": self.config_n_val,
            "config_cohort_agrees_with_resolved": self.config_cohort_agrees_with_resolved,
            "population_verdict": self.population_verdict,
            "population_reason": self.population_reason,
            "per_epoch_file_is_run_of_record": self.per_epoch_file_is_run_of_record,
        }


class RunNotFound(RuntimeError):
    """The run directory has no ``per_epoch_metrics.json``."""


def resolve_run_of_record(
    path: Path,
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """The run of record's cohort, and whether the on-disk file IS its record.

    Wf-A116b -- ``per_epoch_metrics.json`` is written ONCE at the END of
    training, to a FIXED path.  A later run launched into the same
    ``--out-dir`` therefore OVERWRITES it, and the survivor is whichever
    finished LAST: its ``config`` block, its epoch series and everything
    derived from them belong to THAT run, not to the run the directory is
    named for.  MEASURED: ``molmetal/reports/cfm_fullcorpus_2026_10_10``
    holds a 14-epoch run of record at cohort 26715/192 and, in the SAME
    directory, two 1-epoch CPU smokes at 64/32 whose ``done`` events are
    LATER; the surviving ``per_epoch_metrics.json`` is the SMOKE's.

    The resolution is NOT re-implemented here.  The census owns the
    population segmentation, and this calls its resolver
    (``wf_cfm_run_census._read_events`` + ``._population_decision``); the
    cohort itself comes from ``cfm_cohort_recovery.recover_cohort``, whose
    rule is "a run log attests the run of record iff it echoes EVERY ``ts``
    of that population".  Nothing in the census is written to, so no census
    number changes; this only READS.

    The census's own definition of "the per-epoch file IS the run of
    record" is reused verbatim: the file is the run of record's record iff
    the population cross-check did not RESOLVE the run to another
    population and did not REFUSE it (``unmatched_conflict``).  Only then is
    the file's own ``config`` block admitted as evidence about the cohort;
    otherwise the cohort must come from a ts-matched log or it is reported
    as unattested -- never filled in from the clobbered block.
    """
    raw_rows = [r for r in rows if isinstance(r, dict)]
    try:
        # The census normalises a row before comparing it; this harness reads
        # the JSON (never the CSV), so the JSON normalisation is the one
        # that applies here.
        norm_rows = [_census._normalise_row(r, "json") for r in raw_rows]
        ev = _census._read_events(path)
        decision = _census._population_decision(norm_rows, ev)
    except Exception as exc:  # pragma: no cover -- defensive
        return {
            "cohort": (None, None),
            "cohort_source": COHORT_SOURCE_UNATTESTED,
            "cohort_reason": (
                f"the population cross-check could not be run ({exc!r}), so no "
                f"cohort is asserted for this run and the run's own series is "
                f"not published as the run of record's"
            ),
            "cohort_evidence": None,
            "population_verdict": "resolver_failed",
            "population_reason": f"the population cross-check raised {exc!r}",
            "per_epoch_file_is_run_of_record": False,
        }

    verdict = str(decision.get("verdict") or "")
    resolved = bool(decision.get("resolved"))
    reason = str(decision.get("reason") or "")
    file_is_run_of_record = (not resolved) and (verdict != "unmatched_conflict")

    cohort = recover_cohort(
        path,
        list(decision.get("run_of_record_ts") or []),
        config_cohort=(
            (config.get("n_train"), config.get("n_val"))
            if file_is_run_of_record
            else None
        ),
    )
    return {
        "cohort": (cohort.get("n_train"), cohort.get("n_val")),
        "cohort_source": str(cohort.get("source") or ""),
        "cohort_reason": str(cohort.get("reason") or ""),
        "cohort_evidence": cohort.get("evidence"),
        "population_verdict": verdict,
        "population_reason": reason,
        "per_epoch_file_is_run_of_record": file_is_run_of_record,
    }


def load_run(reports_root: Path, name: str) -> RunRecord:
    """Read one run directory.  The JSON is the only accepted source."""
    if not name or name.strip() == "":
        raise RunNotFound("empty run name")
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise RunNotFound(f"run name escapes the reports root: {name!r}")
    path = reports_root.joinpath(*parts)
    json_path = path / "per_epoch_metrics.json"
    if not json_path.is_file():
        raise RunNotFound(
            f"{_rel(json_path)} does not exist -- this harness reads the JSON "
            f"(full precision), never per_epoch_metrics.csv (4 dp rounded)"
        )
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunNotFound(f"{_rel(json_path)} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunNotFound(f"{_rel(json_path)} is not a JSON object")

    config = payload.get("config") or {}
    if not isinstance(config, dict):
        config = {}
    rows = payload.get("per_epoch_metrics") or []
    if not isinstance(rows, list):
        rows = []

    by_epoch: Dict[int, Dict[str, Any]] = {}
    epochs: List[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        epoch = _num(row.get("epoch"))
        if epoch is None:
            continue
        epoch_i = int(epoch)
        by_epoch[epoch_i] = row
        epochs.append(epoch_i)
    epochs = sorted(set(epochs))

    resolution = resolve_run_of_record(path, rows, config)

    return RunRecord(
        name=name,
        path=path,
        config=config,
        rows=[r for r in rows if isinstance(r, dict)],
        by_epoch=by_epoch,
        epochs=epochs,
        source=_rel(json_path),
        cohort=resolution["cohort"],
        cohort_source=resolution["cohort_source"],
        cohort_reason=resolution["cohort_reason"],
        cohort_evidence=resolution["cohort_evidence"],
        population_verdict=resolution["population_verdict"],
        population_reason=resolution["population_reason"],
        per_epoch_file_is_run_of_record=resolution["per_epoch_file_is_run_of_record"],
    )


def e2e_generation(row: Optional[Dict[str, Any]]) -> Optional[float]:
    """The honest end-to-end rate of ONE per-epoch row.

    ``(gen_validity_val / gen_validity_n) * gen_single_fragment_ratio`` --
    the share of ALL generated samples that are one connected molecule.

    Only the GENERATION family is read.  The reconstruction family
    (``decode_ratio_val`` / ``decode_single_fragment_ratio``) is never
    touched: that is the bond head applied to real geometry, i.e. the
    DECODER's fidelity, not the generator's quality.

    Returns ``None`` when the row was not measured (``gen_validity_n == 0``
    or absent) -- never 0.0, which is what the trainer writes as a default.
    """
    if not row:
        return None
    n = _num(row.get("gen_validity_n"))
    v = _num(row.get("gen_validity_val"))
    c = _num(row.get("gen_single_fragment_ratio"))
    if n is None or n <= 0:
        return None
    if v is None or c is None:
        return None
    return (v / n) * c


# --------------------------------------------------------------------------
# the run's integrity (the WHOLE history, not one row)
# --------------------------------------------------------------------------
def run_integrity(run: RunRecord) -> Dict[str, Any]:
    """Everything the run's OWN recorded history says about its integrity.

    Wf-A114 verify:D D3 -- the criterion is "``train_loss`` is NEVER nan",
    and "never" means anywhere in the recorded history, not only on the row
    that contributes the compared number.  A run whose loss went nan at
    epoch 3 and recovered by epoch 5 has a broken history whatever row is
    compared: the epoch index of such a run does not mean the same training
    amount as a sibling's.

    The census makes the same call: ``_is_clean(run)`` is
    ``integrity.epoch_history_contiguous_from_zero AND NOT
    integrity.nan_train_loss`` over the run's whole series.
    """
    epochs = list(run.epochs)
    first = epochs[0] if epochs else None
    last = run.last_epoch
    gaps: List[int] = []
    if epochs:
        gaps = sorted(set(range(first, last + 1)) - set(epochs))

    nan_epochs: List[int] = []
    missing_epochs: List[int] = []
    for epoch in epochs:
        row = run.row_at(epoch) or {}
        tl = row.get("train_loss")
        if _is_absent(tl):
            missing_epochs.append(epoch)
        elif _is_nan(tl):
            nan_epochs.append(epoch)

    contiguous = run.history_contiguous_from_zero()
    reasons: List[str] = []
    if not epochs:
        reasons.append(f"{run.name} has no recorded epoch")
    else:
        if first != 0:
            reasons.append(
                f"{run.name}'s recorded epochs start at {first}, not 0: "
                f"{first} epoch(s) are invisible, so its epoch index is not "
                f"the same training amount as another run's"
            )
        elif gaps:
            reasons.append(
                f"{run.name}'s recorded epochs are not contiguous (missing "
                f"{gaps[:10]})"
            )
    if nan_epochs:
        reasons.append(
            f"{run.name}'s recorded epoch(s) {nan_epochs[:10]} carry "
            f"train_loss = nan (the whole history is scanned, not only the "
            f"contributing row)"
        )
    if missing_epochs:
        reasons.append(
            f"{run.name}'s recorded epoch(s) {missing_epochs[:10]} carry no "
            f"train_loss at all, so the run's integrity is not established"
        )

    return {
        "run": run.name,
        "epochs_recorded": len(epochs),
        "first_epoch": first,
        "last_epoch": last,
        "contiguous_from_zero": contiguous,
        "missing_epochs_in_range": gaps,
        "nan_train_loss_epochs": nan_epochs,
        "missing_train_loss_epochs": missing_epochs,
        "ok": not reasons,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------
# the pool-kind cross-check (Wf-A114 cluster C)
# --------------------------------------------------------------------------
def find_pool_csv(run_dir: Path) -> Optional[Path]:
    """The run's on-disk generation pool, if it left one.

    Only files whose NAME says pool/smiles/generated are considered.
    ``per_epoch_metrics.csv`` is a metrics file, never a pool, and must not
    be classified as one.
    """
    if not run_dir.is_dir():
        return None
    seen: List[Path] = []
    for pattern in POOL_CSV_PATTERNS:
        for candidate in sorted(run_dir.glob(pattern)):
            if candidate.is_file() and candidate not in seen:
                seen.append(candidate)
    if not seen:
        return None
    # A pool that the shared rule can actually classify wins over one it
    # cannot; among equals, the first in the pattern order wins.
    for candidate in seen:
        kind = classify_pool(candidate)
        if kind.kind in (KIND_GENERATION, KIND_RECONSTRUCTION):
            return candidate
    return seen[0]


def pool_kind_record(run: RunRecord) -> Dict[str, Any]:
    """Classify the run's on-disk pool with the ONE shared rule."""
    pool = find_pool_csv(run.path)
    if pool is None:
        return {
            "pool_csv": None,
            "pool_kind": "absent",
            "pool_marker": None,
            "pool_reason": (
                "the run directory holds no on-disk pool CSV; the compared "
                "number comes from the per-epoch generation path and needs no "
                "pool file"
            ),
        }
    kind = classify_pool(pool)
    return {
        "pool_csv": _rel(pool),
        "pool_kind": kind.kind,
        "pool_marker": kind.marker,
        "pool_reason": kind.reason,
        "pool_header": list(kind.header),
    }


def pool_single_fragment_fraction(pool_csv: Path, epoch: int) -> Optional[float]:
    """The single-fragment fraction of a GENERATION pool at one epoch.

    Read from the pool's own connectivity column: ``n_frags_raw`` (the
    trainer's free-generation path) or ``n_fragments`` (the decode-dump
    script's path -- same ``decode_chemistry`` definition, different column
    name).  Returns ``None`` when the pool carries no rows for that epoch, or
    no connectivity column at all.
    """
    import csv  # local import: only needed for the corroboration path

    total = 0
    single = 0
    try:
        with pool_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                raw = None
                for column in CONNECTIVITY_COLUMNS:
                    if column in row:
                        raw = row.get(column)
                        break
                if raw is None:
                    continue
                if str(row.get("epoch")) != str(epoch):
                    continue
                try:
                    n_frag = int(float(raw))
                except (TypeError, ValueError):
                    continue
                total += 1
                if n_frag == 1:
                    single += 1
    except OSError:
        return None
    if total == 0:
        return None
    return single / total


# --------------------------------------------------------------------------
# the five axes
# --------------------------------------------------------------------------
@dataclass
class AxisResult:
    axis: str
    ok: bool
    detail: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis": self.axis,
            "status": "PASS" if self.ok else "FAIL",
            "detail": self.detail,
            **({"evidence": self.extra} if self.extra else {}),
        }


def axis_cohort(a: RunRecord, b: RunRecord) -> AxisResult:
    """AXIS 1 -- the two runs must have trained on the same cohort."""
    evidence: Dict[str, Any] = {
        "n_train": {a.name: a.n_train, b.name: b.n_train},
        "n_val": {a.name: a.n_val, b.name: b.n_val},
    }
    problems: List[str] = []
    for key in ("n_train", "n_val"):
        va = getattr(a, key)
        vb = getattr(b, key)
        if va is None or vb is None:
            problems.append(f"{key} is not recorded (A={va!r}, B={vb!r})")
        elif va != vb:
            problems.append(f"{key} differs (A={va}, B={vb})")
    if problems:
        return AxisResult(
            AXIS_COHORT,
            False,
            "the two runs are not the same cohort: " + "; ".join(problems),
            evidence,
        )
    return AxisResult(
        AXIS_COHORT,
        True,
        f"both runs trained on n_train={a.n_train}, n_val={a.n_val}",
        evidence,
    )


def axis_population(a: RunRecord, b: RunRecord) -> AxisResult:
    """AXIS 7 -- the compared epoch series must be the RUN OF RECORD's own.

    Wf-A116b -- ``per_epoch_metrics.json`` is written ONCE at the END of
    training, to a FIXED path.  A later run launched into the same
    ``--out-dir`` overwrites it, and the file that survives is the LAST
    writer's: its ``config`` block, its epoch series and everything derived
    from them belong to THAT population.  Comparing that file's series under
    this run's name publishes another population's epochs as the run's, so
    the comparison is REFUSED and the population that was found is named.

    The verdict is the census's own (``_population_decision``), not a
    re-implementation of it -- see ``resolve_run_of_record``.
    """
    evidence: Dict[str, Any] = {}
    problems: List[str] = []
    for run in (a, b):
        evidence[run.name] = {
            "per_epoch_file_is_run_of_record": run.per_epoch_file_is_run_of_record,
            "population_verdict": run.population_verdict,
            "population_reason": run.population_reason,
            "cohort_source": run.cohort_source,
            "cohort_reason": run.cohort_reason,
            "cohort": [run.n_train, run.n_val],
            "config_cohort": [run.config_n_train, run.config_n_val],
        }
        if not run.per_epoch_file_is_run_of_record:
            problems.append(
                f"{run.name}'s on-disk per_epoch_metrics.json is NOT the run of "
                f"record's own record (population verdict: "
                f"{run.population_verdict or 'unknown'}): "
                f"{run.population_reason}"
            )
    if problems:
        return AxisResult(AXIS_POPULATION, False, "; ".join(problems), evidence)
    return AxisResult(
        AXIS_POPULATION,
        True,
        "each run's on-disk per-epoch file IS its run of record's own record, "
        "so the compared epoch series belongs to the run it is named for",
        evidence,
    )


def axis_epoch_budget(
    a: RunRecord, b: RunRecord, index: int
) -> AxisResult:
    """AXIS 2 -- both runs must have a RECORDED row at the compared index."""
    evidence = {
        "compared_epoch_index": index,
        "compared_budget_epochs": index + 1,
        "recorded_epochs": {a.name: list(a.epochs), b.name: list(b.epochs)},
    }
    missing = []
    for run in (a, b):
        if run.row_at(index) is None:
            last = run.last_epoch
            missing.append(
                f"{run.name} has no recorded row at epoch {index} "
                f"(its recorded epochs run {run.epochs[0] if run.epochs else '-'}..{last})"
            )
    if missing:
        return AxisResult(
            AXIS_EPOCH_BUDGET,
            False,
            "the compared epoch index is not reached by both runs: " + "; ".join(missing),
            evidence,
        )
    return AxisResult(
        AXIS_EPOCH_BUDGET,
        True,
        f"both runs recorded a row at epoch {index} (compared budget = {index + 1} epochs)",
        evidence,
    )


def axis_not_measured(rows: Dict[str, Optional[Dict[str, Any]]]) -> AxisResult:
    """AXIS 2b -- the contributing row must CARRY a measurement."""
    evidence: Dict[str, Any] = {}
    problems: List[str] = []
    for name, row in rows.items():
        n = _num(row.get("gen_validity_n")) if row else None
        evidence[name] = {
            "gen_validity_n": None if n is None else int(n),
            "gen_validity_val": row.get("gen_validity_val") if row else None,
            "gen_single_fragment_ratio": row.get("gen_single_fragment_ratio") if row else None,
        }
        if n is None or n <= 0:
            problems.append(
                f"{name}'s contributing row carries gen_validity_n="
                f"{'absent' if n is None else int(n)}: the generation eval did NOT "
                f"run that epoch, so the 0.0000 in gen_validity / "
                f"gen_single_fragment_ratio is the trainer's DEFAULT, not a "
                f"measurement"
            )
    if problems:
        return AxisResult(AXIS_NOT_MEASURED, False, "; ".join(problems), evidence)
    return AxisResult(
        AXIS_NOT_MEASURED,
        True,
        "the contributing row of each run carries a generation measurement "
        "(gen_validity_n > 0)",
        evidence,
    )


def axis_reading(
    a: RunRecord,
    b: RunRecord,
    reading: str,
    compared_epochs: Optional[Dict[str, Optional[int]]] = None,
) -> AxisResult:
    """AXIS 3 -- one reading for both arms.

    Wf-A114 verify:D D2 -- under the ``best`` reading each arm contributes
    its OWN maximum over the SAME searched window, so the two compared
    numbers need not sit at the same epoch.  The per-arm epoch is recorded
    as evidence, and the payload's ``compared_epochs`` names it -- the
    payload must never claim that both numbers came from the budget index.
    """
    evidence = {
        "reading": reading,
        "epoch_set_a": list(a.epochs),
        "epoch_set_b": list(b.epochs),
        "epoch_sets_identical": list(a.epochs) == list(b.epochs),
    }
    if compared_epochs:
        evidence["compared_epoch_a"] = compared_epochs.get(a.name)
        evidence["compared_epoch_b"] = compared_epochs.get(b.name)
    if reading != "best":
        return AxisResult(
            AXIS_READING,
            True,
            "the fixed-epoch reading compares one index for both arms, so the "
            "two series need not cover the same epochs",
            evidence,
        )
    if list(a.epochs) != list(b.epochs):
        return AxisResult(
            AXIS_READING,
            False,
            "the best-epoch reading needs both runs to cover the SAME epochs: "
            f"{a.name} has {len(a.epochs)} recorded epochs, {b.name} has "
            f"{len(b.epochs)}; a max over a noisy series is only comparable "
            "when the series cover the same range",
            evidence,
        )
    ea = evidence.get("compared_epoch_a")
    eb = evidence.get("compared_epoch_b")
    same_epoch = (ea is not None and ea == eb)
    return AxisResult(
        AXIS_READING,
        True,
        "both runs cover the same recorded epochs, so their best epochs are "
        "comparable under the same reading"
        + (
            ""
            if same_epoch
            else (
                f"; the two maxima sit at DIFFERENT epochs (A@{ea}, B@{eb}) -- "
                "each arm's own epoch is recorded, and the payload never "
                "claims both numbers came from the budget index"
            )
        ),
        evidence,
    )


def axis_history(
    a: RunRecord, b: RunRecord, rows: Dict[str, Optional[Dict[str, Any]]]
) -> AxisResult:
    """AXIS 4 -- intact, contiguous-from-zero history and no NaN loss.

    Wf-A114 verify:D D3 -- the whole recorded history is scanned for a nan
    (or absent) ``train_loss``, not only the contributing row.
    """
    evidence: Dict[str, Any] = {}
    problems: List[str] = []
    for run in (a, b):
        integ = run_integrity(run)
        evidence[run.name] = {
            "first_epoch": integ["first_epoch"],
            "last_epoch": integ["last_epoch"],
            "contiguous_from_zero": integ["contiguous_from_zero"],
            "missing_epochs_in_range": integ["missing_epochs_in_range"],
            "nan_train_loss_epochs": integ["nan_train_loss_epochs"],
            "missing_train_loss_epochs": integ["missing_train_loss_epochs"],
            "scanned_epochs": integ["epochs_recorded"],
        }
        problems.extend(integ["reasons"])
    if problems:
        return AxisResult(AXIS_HISTORY, False, "; ".join(problems), evidence)
    return AxisResult(
        AXIS_HISTORY,
        True,
        "both runs recorded a contiguous epoch history from 0 and no epoch in "
        "either run's whole history carries a nan or an absent train_loss",
        evidence,
    )


def axis_provenance(
    a: RunRecord,
    b: RunRecord,
    rows: Dict[str, Optional[Dict[str, Any]]],
    compared_epochs: Dict[str, Optional[int]],
) -> AxisResult:
    """AXIS 5 -- generation pool, matching construction, pool kind."""
    evidence: Dict[str, Any] = {
        "metric_family": "generation",
        "generation_fields_read": list(GENERATION_FIELDS),
        "reconstruction_fields_not_read": list(RECONSTRUCTION_FIELDS),
    }
    problems: List[str] = []

    # (a) the pool construction must match between the two arms.
    differing: List[str] = []
    for field_name in POOL_CONSTRUCTION_FIELDS:
        va = rows[a.name].get(field_name) if rows.get(a.name) else None
        vb = rows[b.name].get(field_name) if rows.get(b.name) else None
        evidence.setdefault("pool_construction", {})[field_name] = {
            a.name: va,
            b.name: vb,
        }
        if va != vb:
            differing.append(f"{field_name} ({va!r} vs {vb!r})")
    if differing:
        problems.append(
            "the two runs' pools were not constructed the same way: "
            + ", ".join(differing)
        )

    # (b) the run's own on-disk pool must be a GENERATION pool (cluster C).
    kinds: Dict[str, Any] = {}
    for run in (a, b):
        record = pool_kind_record(run)
        kinds[run.name] = record
        if record["pool_kind"] == KIND_RECONSTRUCTION:
            problems.append(
                f"{run.name}'s on-disk pool is a bond-head RECONSTRUCTION "
                f"(marker {record['pool_marker']!r}; {record['pool_reason']}), "
                f"so its connectivity number is decoder fidelity on real "
                f"geometry, not the generator's quality"
            )
        elif record["pool_kind"] not in (KIND_GENERATION, "absent"):
            problems.append(
                f"{run.name}'s on-disk pool is UNCLASSIFIED ({record['pool_reason']}), "
                f"so its provenance is not established"
            )
    evidence["pool_kind"] = kinds

    # (c) corroboration: a GENERATION pool must AGREE with the per-epoch row.
    for run in (a, b):
        record = kinds.get(run.name, {})
        if record.get("pool_kind") != KIND_GENERATION:
            continue
        epoch = compared_epochs.get(run.name)
        row = rows.get(run.name)
        if epoch is None or row is None:
            continue
        pool_csv = run.path / Path(record["pool_csv"]).name
        observed = pool_single_fragment_fraction(pool_csv, epoch)
        expected = _num(row.get("gen_single_fragment_ratio"))
        if observed is None:
            record["corroboration"] = {
                "status": "NOT_CHECKED",
                "reason": (
                    f"the pool carries no rows for epoch {epoch} in a "
                    f"connectivity column ({' / '.join(CONNECTIVITY_COLUMNS)})"
                ),
            }
            continue
        if expected is None:
            record["corroboration"] = {
                "status": "NOT_CHECKED",
                "reason": "the per-epoch row carries no gen_single_fragment_ratio",
            }
            continue
        agrees = math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-9)
        record["corroboration"] = {
            "status": "AGREES" if agrees else "DISAGREES",
            "epoch": epoch,
            "pool_single_fragment_fraction": observed,
            "row_gen_single_fragment_ratio": expected,
        }
        if not agrees:
            problems.append(
                f"{run.name}'s on-disk GENERATION pool and its per-epoch row "
                f"DISAGREE at epoch {epoch} (pool {observed!r} vs row "
                f"{expected!r}), so the compared number's provenance is not "
                f"established"
            )

    if problems:
        return AxisResult(AXIS_PROVENANCE, False, "; ".join(problems), evidence)
    return AxisResult(
        AXIS_PROVENANCE,
        True,
        "both arms read the GENERATION pool, their pool construction matches, "
        "and no on-disk pool contradicts the per-epoch metric",
        evidence,
    )


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------
@dataclass
class Comparison:
    reading: str
    epoch_index: int
    rows: Dict[str, Optional[Dict[str, Any]]]
    compared_epochs: Dict[str, Optional[int]]
    axes: Dict[str, AxisResult]
    values: Dict[str, Optional[float]]

    @property
    def unmatched_axes(self) -> List[str]:
        return [name for name in ALL_AXES if not self.axes[name].ok]

    @property
    def ok(self) -> bool:
        return not self.unmatched_axes


def compare_runs(
    a: RunRecord,
    b: RunRecord,
    reading: str = "epoch",
    epoch_index: Optional[int] = None,
) -> Comparison:
    """Run the seven axes over two runs.  Never raises for a mismatch."""
    # The budget rule: K = min(last_A, last_B) + 1, compared at K-1.
    lasts = [r.last_epoch for r in (a, b) if r.last_epoch is not None]
    budget_index = (min(lasts) if lasts else 0)
    if epoch_index is None:
        index = budget_index
    else:
        index = int(epoch_index)

    rows: Dict[str, Optional[Dict[str, Any]]] = {}
    compared_epochs: Dict[str, Optional[int]] = {}
    values: Dict[str, Optional[float]] = {}

    if reading == "best":
        # Wf-A114 verify:D D2 -- both arms search the SAME window (epochs
        # 0..index, the budget rule's window).  Each arm still contributes
        # its OWN maximum, so the two compared numbers need not sit at the
        # same epoch; the per-arm epoch is recorded and the payload says
        # which epoch each number came from.
        for run in (a, b):
            best_row = None
            best_value = None
            for epoch in run.epochs:
                if epoch > index:
                    continue
                row = run.row_at(epoch)
                value = e2e_generation(row)
                if value is None:
                    continue
                if best_value is None or value > best_value:
                    best_value = value
                    best_row = row
            rows[run.name] = best_row
            best_epoch = _num(best_row.get("epoch")) if best_row else None
            compared_epochs[run.name] = int(best_epoch) if best_epoch is not None else None
            values[run.name] = best_value
    else:
        for run in (a, b):
            row = run.row_at(index)
            rows[run.name] = row
            compared_epochs[run.name] = index
            values[run.name] = e2e_generation(row)

    axes = {
        AXIS_COHORT: axis_cohort(a, b),
        AXIS_EPOCH_BUDGET: axis_epoch_budget(a, b, index),
        AXIS_NOT_MEASURED: axis_not_measured(rows),
        AXIS_READING: axis_reading(a, b, reading, compared_epochs),
        AXIS_HISTORY: axis_history(a, b, rows),
        AXIS_PROVENANCE: axis_provenance(a, b, rows, compared_epochs),
        AXIS_POPULATION: axis_population(a, b),
    }
    return Comparison(
        reading=reading,
        epoch_index=index,
        rows=rows,
        compared_epochs=compared_epochs,
        axes=axes,
        values=values,
    )


# --------------------------------------------------------------------------
# the nearest valid pair on disk (the suggestion on a refusal)
# --------------------------------------------------------------------------
def discover_runs(reports_root: Path) -> List[RunRecord]:
    """Every run directory under the reports root that has a per-epoch JSON."""
    found: List[RunRecord] = []
    if not reports_root.is_dir():
        return found
    for json_path in sorted(reports_root.rglob("per_epoch_metrics.json")):
        name = json_path.parent.relative_to(reports_root).as_posix()
        try:
            found.append(load_run(reports_root, name))
        except RunNotFound:
            continue
    return found


def find_suggested_pair(
    reports_root: Path,
    a: RunRecord,
    b: RunRecord,
    reading: str,
    exclude: Sequence[str] = (),
) -> Optional[Dict[str, Any]]:
    """The nearest valid pair on disk in B's cohort, with its epoch index.

    Preference order:
      1. a run X in B's cohort that ALSO sits in A's cohort (the true
         nearest substitute for A) and pairs validly with B;
      2. any run X in B's cohort that pairs validly with B;
      3. the mirror image, a run X in A's cohort that pairs validly with A.
    Within a tier the pair with the LARGEST compared epoch index wins -- the
    most matched training budget.
    """
    excluded = set(exclude)
    pool = discover_runs(reports_root)
    b_cohort = (b.n_train, b.n_val)
    a_cohort = (a.n_train, a.n_val)

    def _tier(candidate_cohort: Tuple[Any, Any], anchor: RunRecord) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for other in pool:
            if other.name in excluded or other.name == anchor.name:
                continue
            if (other.n_train, other.n_val) != candidate_cohort:
                continue
            result = compare_runs(other, anchor, reading=reading)
            if not result.ok:
                continue
            out.append(
                {
                    "run_a": other.name,
                    "run_b": anchor.name,
                    "epoch_index": result.epoch_index,
                    "compared_budget_epochs": result.epoch_index + 1,
                    "e2e_a": result.values.get(other.name),
                    "e2e_b": result.values.get(anchor.name),
                    "n_train": other.n_train,
                    "n_val": other.n_val,
                }
            )
        out.sort(key=lambda d: (-d["epoch_index"], d["run_a"]))
        return out

    for tier_cohort in (a_cohort if a_cohort == b_cohort else None, b_cohort):
        if tier_cohort is None:
            continue
        candidates = _tier(tier_cohort, b)
        if candidates:
            best = dict(candidates[0])
            best["why"] = (
                f"the nearest valid pair on disk in {b.name}'s cohort "
                f"(n_train={b.n_train}, n_val={b.n_val})"
            )
            return best

    candidates = _tier(a_cohort, a)
    if candidates:
        best = dict(candidates[0])
        best["why"] = (
            f"no valid pair exists in {b.name}'s cohort; the nearest valid "
            f"pair on disk in {a.name}'s cohort is offered instead"
        )
        return best
    return None


# --------------------------------------------------------------------------
# the census cross-check (a cross-check only, NEVER a gate)
# --------------------------------------------------------------------------
def census_crosscheck(census_path: Path, runs: Sequence[RunRecord]) -> Dict[str, Any]:
    """Compare the harness's end-to-end formula against the census artifact.

    The census records ``gen_single_molecule_rate_end`` -- the LAST epoch's
    product.  This cross-check recomputes the same thing from the JSON and
    reports the delta.  It NEVER changes the verdict.
    """
    out: Dict[str, Any] = {
        "census_path": _rel(census_path),
        "role": "cross-check only, never a gate",
        "runs": {},
    }
    if not census_path.is_file():
        out["status"] = "SKIP"
        out["reason"] = f"{_rel(census_path)} does not exist"
        return out
    try:
        payload = json.loads(census_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        out["status"] = "SKIP"
        out["reason"] = f"unreadable: {exc}"
        return out
    census_rows = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(census_rows, list):
        out["status"] = "SKIP"
        out["reason"] = "the census artifact carries no 'runs' list"
        return out
    by_run = {}
    for row in census_rows:
        if isinstance(row, dict) and isinstance(row.get("run"), str):
            by_run[row["run"]] = row
    for run in runs:
        entry: Dict[str, Any] = {}
        census_row = by_run.get(run.name)
        if census_row is None:
            entry["status"] = "NOT_IN_CENSUS"
        else:
            census_value = _num(census_row.get("gen_single_molecule_rate_end"))
            last_row = run.row_at(run.last_epoch) if run.last_epoch is not None else None
            harness_value = e2e_generation(last_row)
            entry["census_field"] = "gen_single_molecule_rate_end"
            entry["census_epoch"] = (
                census_row.get("fragmentation_gate", {}) or {}
            ).get("epoch")
            entry["census_value"] = census_value
            entry["harness_value_at_last_epoch"] = harness_value
            if census_value is None or harness_value is None:
                entry["status"] = "NOT_COMPARABLE"
            else:
                delta = harness_value - census_value
                entry["delta"] = delta
                entry["agrees_within_1e-9"] = abs(delta) <= 1e-9
                entry["status"] = "AGREES" if abs(delta) <= 1e-9 else "DIFFERS"
                entry["note"] = (
                    "the census multiplies the trainer's rounded gen_validity "
                    "rate by gen_single_fragment_ratio; this harness divides "
                    "gen_validity_val by gen_validity_n, so a last-bit delta "
                    "is expected and is reported, not hidden"
                )
        out["runs"][run.name] = entry
    out["status"] = "DONE"
    return out


# --------------------------------------------------------------------------
# the bar (a threshold, traced to disk when a run on disk carries it)
# --------------------------------------------------------------------------
def trace_bar_to_disk(bar: float, reports_root: Path) -> Optional[Dict[str, Any]]:
    """Find a run on disk whose end-to-end rate equals the bar.

    A bar that cannot be traced to a file path on disk is a PROJECTED
    threshold, not a measurement (AGENTS.md §5.2).

    Wf-A114 verify:D D1 -- being ON DISK is not enough.  The trace is
    INTEGRITY-GATED: the run that carries the value must also pass the
    ``history`` axis (contiguous from zero, no nan / absent ``train_loss``
    anywhere in its whole history).  Without that gate the scan re-minted
    the WITHDRAWN number: ``cfm_r11_eval256`` records exactly ONE epoch
    (index 19) out of a declared 20, its history does not start at zero and
    its ``train_loss`` is ``nan`` -- and its epoch-19 end-to-end rate is
    ``0.572265625``, the number the census withdrew.  A match on a
    degenerate run is reported as ``REJECTED`` (with the run, the epoch and
    the reason), never as ``MEASURED``.

    Returns ``None`` only when NO run on disk carries the value at all.
    """
    matches: List[Dict[str, Any]] = []
    for run in discover_runs(reports_root):
        integrity = run_integrity(run)
        for epoch in run.epochs:
            value = e2e_generation(run.row_at(epoch))
            if value is None:
                continue
            if math.isclose(value, bar, rel_tol=0.0, abs_tol=1e-12):
                matches.append(
                    {
                        "run": run.name,
                        "epoch": epoch,
                        "value": value,
                        "source": f"{run.source} -> per_epoch_metrics[epoch=={epoch}]",
                        "integrity": integrity,
                    }
                )
    if not matches:
        return None

    clean = [m for m in matches if m["integrity"]["ok"]]
    if clean:
        best = dict(clean[0])
        best["status"] = "TRACED"
        best["note"] = (
            "the value is on disk in a run whose recorded history is intact "
            "(contiguous from zero, no nan train_loss)"
        )
        return best

    # Every match sits on a degenerate run -> REJECTED, and say which one and
    # why.  The number EXISTS on disk; what it is not is a measurement.
    rejected = sorted(matches, key=lambda m: (m["run"], m["epoch"]))
    first = rejected[0]
    return {
        "status": "REJECTED",
        "run": first["run"],
        "epoch": first["epoch"],
        "value": first["value"],
        "source": first["source"],
        "integrity": first["integrity"],
        "reason": "; ".join(first["integrity"]["reasons"]) or "the run's history is not intact",
        "rejected_matches": [
            {
                "run": m["run"],
                "epoch": m["epoch"],
                "value": m["value"],
                "source": m["source"],
                "reason": "; ".join(m["integrity"]["reasons"]),
            }
            for m in rejected
        ],
        "note": (
            "this value IS on disk, but the run that carries it has a broken "
            "recorded history, so it is not a measurement of anything -- it "
            "is reported as REJECTED and never as MEASURED"
        ),
    }


# --------------------------------------------------------------------------
# payload rendering
# --------------------------------------------------------------------------
def build_payload(
    a: RunRecord,
    b: RunRecord,
    result: Comparison,
    reports_root: Path,
    out_dir: Path,
    bar: Optional[float],
    census: Optional[Dict[str, Any]],
    suggestion: Optional[Dict[str, Any]],
    reading_requested: str,
) -> Dict[str, Any]:
    verdict = "VERDICT" if result.ok else "REFUSED"
    payload: Dict[str, Any] = {
        "metric": "cfm_cohort_matched_compare",
        "schema_version": 2,
        "generated_at": _iso_now(),
        "verdict": verdict,
        "reading": result.reading,
        "reading_requested": reading_requested,
        "epoch_index": result.epoch_index,
        "compared_budget_epochs": result.epoch_index + 1,
        "metric_family": "generation",
        "e2e_formula": "(gen_validity_val / gen_validity_n) * gen_single_fragment_ratio",
        "runs": {
            "a": a.to_dict(),
            "b": b.to_dict(),
        },
        "axes": {name: axis.to_dict() for name, axis in result.axes.items()},
        "unmatched_axes": result.unmatched_axes,
        "sources": {
            "a": a.source,
            "b": b.source,
        },
        "reports_root": _rel(reports_root),
        "out_dir": _rel(out_dir),
        "caveat": build_caveat(result.rows),
        "suggested_matched_comparison": suggestion,
    }

    # Wf-A114 verify:D D2 -- the epoch index means different things under the
    # two readings, and the payload used to claim one meaning for both.
    #   epoch reading: the compared index for BOTH arms.
    #   best  reading: the SEARCH WINDOW's last index; each arm contributes
    #                  its OWN maximum inside that window, so the two numbers
    #                  need not sit at the same epoch.
    payload["compared_epochs"] = {
        a.name: result.compared_epochs.get(a.name),
        b.name: result.compared_epochs.get(b.name),
    }
    if result.reading == "best":
        payload["epoch_index_meaning"] = (
            "the SEARCH WINDOW's last index: under the 'best' reading each "
            "arm contributes its own maximum over epochs 0..epoch_index, and "
            "the two maxima need not sit at the same epoch -- the per-arm "
            "epoch is in `compared_epochs`"
        )
        payload["searched_epoch_range"] = [0, result.epoch_index]
    else:
        payload["epoch_index_meaning"] = (
            "the compared epoch index for BOTH arms"
        )
        payload["searched_epoch_range"] = [result.epoch_index, result.epoch_index]

    if result.ok:
        va = result.values.get(a.name)
        vb = result.values.get(b.name)
        epoch_a = result.compared_epochs.get(a.name)
        epoch_b = result.compared_epochs.get(b.name)
        delta = None if (va is None or vb is None) else vb - va
        payload["e2e"] = {
            "a": {
                "value": va,
                "label": "MEASURED",
                "source": f"{a.source} -> per_epoch_metrics[epoch=={epoch_a}]",
                "epoch": epoch_a,
                "gen_validity_n": (
                    _num(result.rows[a.name].get("gen_validity_n"))
                    if result.rows.get(a.name)
                    else None
                ),
            },
            "b": {
                "value": vb,
                "label": "MEASURED",
                "source": f"{b.source} -> per_epoch_metrics[epoch=={epoch_b}]",
                "epoch": epoch_b,
                "gen_validity_n": (
                    _num(result.rows[b.name].get("gen_validity_n"))
                    if result.rows.get(b.name)
                    else None
                ),
            },
            "delta_b_minus_a": {
                "value": delta,
                "label": "MEASURED",
                "source": (
                    f"{b.source} -> per_epoch_metrics[epoch=={epoch_b}] minus "
                    f"{a.source} -> per_epoch_metrics[epoch=={epoch_a}]"
                ),
            },
        }
        if result.reading == "best":
            payload["e2e"]["note"] = (
                "the best reading is a MAX over a noisy series (the census "
                "labels it 'optimistically biased'); it is reported as a "
                "secondary reading and never as a bar"
            )
    else:
        payload["e2e"] = None
        payload["refusal"] = {
            "reason": (
                "the two runs do not match on: " + ", ".join(result.unmatched_axes)
            ),
            "axes": {
                name: result.axes[name].detail
                for name in result.unmatched_axes
            },
        }

    if bar is not None:
        traced = trace_bar_to_disk(bar, reports_root)
        if traced is None:
            bar_payload: Dict[str, Any] = {
                "value": bar,
                "label": "PROJECTED",
                "source": (
                    "user-supplied --bar; no run on disk carries this "
                    "end-to-end rate in its full-precision per_epoch_metrics.json"
                ),
                "note": (
                    "a bar is a THRESHOLD, not a measurement; it is traced to "
                    "disk when a run on disk carries it.  MEASURED: the CSV "
                    "rendering of a run rounds every rate to 4 decimals, so a "
                    "value that exists only in the CSV's rounded form is NOT "
                    "traced here -- this harness reads the JSON only"
                ),
            }
        elif traced["status"] == "REJECTED":
            bar_payload = {
                "value": bar,
                "label": "REJECTED",
                "source": traced["source"],
                "note": traced["note"],
                "rejection_reason": traced["reason"],
                "rejected_matches": traced["rejected_matches"],
            }
        else:
            bar_payload = {
                "value": bar,
                "label": "MEASURED",
                "source": traced["source"],
                "note": traced["note"],
            }
            bar_payload["traced_to"] = {
                "run": traced["run"],
                "epoch": traced["epoch"],
                "value": traced["value"],
                "integrity": {
                    k: v
                    for k, v in traced["integrity"].items()
                    if k != "reasons"
                },
            }
        if result.ok:
            vb = result.values.get(b.name)
            va = result.values.get(a.name)
            bar_payload["b_meets_bar"] = bool(vb is not None and vb > bar)
            bar_payload["a_meets_bar"] = bool(va is not None and va > bar)
            bar_payload["arm_roles"] = (
                "A is the REFERENCE arm, B is the CANDIDATE arm; "
                "'meets the bar' means the arm's end-to-end rate is strictly "
                "above the threshold"
            )
        payload["bar"] = bar_payload
    else:
        payload["bar"] = None

    if census is not None:
        payload["census_crosscheck"] = census
    return payload


def render_markdown(payload: Dict[str, Any]) -> str:
    """The human-readable twin of ``compare.json``."""
    lines: List[str] = []
    verdict = payload["verdict"]
    runs = payload["runs"]
    lines.append(f"# CFM cohort-matched comparison — {verdict}")
    lines.append("")
    lines.append(f"- **A**: `{runs['a']['run']}` ({payload['sources']['a']})")
    lines.append(f"- **B**: `{runs['b']['run']}` ({payload['sources']['b']})")
    lines.append(
        f"- **cohort**: n_train={runs['a']['n_train']}, n_val={runs['a']['n_val']}"
    )
    if payload["reading"] == "best":
        rng = payload.get("searched_epoch_range") or [payload["epoch_index"]] * 2
        lines.append(
            f"- **reading**: `{payload['reading']}` — each arm contributes its "
            f"own maximum over the searched window epochs {rng[0]}..{rng[1]} "
            f"(budget = {payload['compared_budget_epochs']} epochs); the two "
            f"maxima need not sit at the same epoch"
        )
    else:
        lines.append(
            f"- **reading**: `{payload['reading']}` at epoch index "
            f"{payload['epoch_index']} (compared budget = "
            f"{payload['compared_budget_epochs']} epochs)"
        )
    lines.append(f"- **formula**: `{payload['e2e_formula']}`")
    lines.append("")
    lines.append("## Axes")
    lines.append("")
    lines.append("| axis | status | detail |")
    lines.append("|---|---|---|")
    for name in ALL_AXES:
        axis = payload["axes"][name]
        lines.append(f"| `{name}` | {axis['status']} | {axis['detail']} |")
    lines.append("")
    if payload.get("e2e"):
        e2e = payload["e2e"]
        lines.append("## Result")
        lines.append("")
        lines.append(f"- e2e(A) = **{e2e['a']['value']!r}** (MEASURED, epoch {e2e['a']['epoch']})")
        lines.append(f"- e2e(B) = **{e2e['b']['value']!r}** (MEASURED, epoch {e2e['b']['epoch']})")
        lines.append(
            f"- delta (B - A) = **{e2e['delta_b_minus_a']['value']!r}** (MEASURED)"
        )
        lines.append(f"- source: `{e2e['delta_b_minus_a']['source']}`")
        if e2e.get("note"):
            lines.append(f"- note: {e2e['note']}")
    else:
        lines.append("## Refusal")
        lines.append("")
        lines.append(f"- unmatched axes: {', '.join(payload['unmatched_axes'])}")
        for name, detail in payload["refusal"]["axes"].items():
            lines.append(f"  - `{name}`: {detail}")
    lines.append("")
    if payload.get("bar"):
        bar = payload["bar"]
        lines.append("## Bar")
        lines.append("")
        lines.append(f"- bar = **{bar['value']!r}** ({bar['label']}) — {bar['source']}")
        if bar.get("rejection_reason"):
            lines.append(f"- REJECTED because: {bar['rejection_reason']}")
        if bar.get("b_meets_bar") is not None:
            lines.append(
                f"- arm B (the CANDIDATE) meets the bar: "
                f"**{bar['b_meets_bar']}**; arm A (the REFERENCE) meets the "
                f"bar: **{bar.get('a_meets_bar')}**"
            )
        lines.append("")
    if payload.get("suggested_matched_comparison"):
        sug = payload["suggested_matched_comparison"]
        lines.append("## Nearest valid pair on disk")
        lines.append("")
        lines.append(f"- `{sug['run_a']}` vs `{sug['run_b']}` at epoch index {sug['epoch_index']}")
        lines.append(f"- e2e(A) = {sug.get('e2e_a')!r}, e2e(B) = {sug.get('e2e_b')!r}")
        lines.append(f"- {sug['why']}")
        lines.append("")
    lines.append("## Caveat")
    lines.append("")
    lines.append(f"- {payload['caveat']}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cfm_cohort_matched_compare.py",
        description=(
            "Cohort- and budget-matched comparison of two CFM runs' "
            "end-to-end generation rate.  A comparison that does not match on "
            "all seven axes (cohort, epoch_budget, not_measured, reading, "
            "history, provenance, population) is REFUSED, not reported."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-a", required=True, help="run A name under --reports-root")
    parser.add_argument("--run-b", required=True, help="run B name under --reports-root")
    parser.add_argument(
        "--reports-root",
        default=str(DEFAULT_REPORTS_ROOT),
        help="root of the run corpus (default: molmetal/reports)",
    )
    parser.add_argument(
        "--reading",
        choices=("epoch", "best"),
        default="epoch",
        help=(
            "epoch = the fixed budget-matched index (default); "
            "best = the max over the run's own series, allowed only when both "
            "runs recorded the same epochs"
        ),
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help=(
            "override the compared epoch INDEX (the budget rule compares at "
            "min(last_A, last_B); the compared budget is index+1 epochs)"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: <reports-root>/wfa114_cohort_matched_compare/<a>__vs__<b>)",
    )
    parser.add_argument(
        "--bar",
        type=float,
        default=None,
        help="optional threshold to report alongside the verdict (traced to disk when possible)",
    )
    parser.add_argument(
        "--census",
        default=None,
        help="optional census artifact for a CROSS-CHECK ONLY (never a gate)",
    )
    return parser.parse_args(argv)


def default_out_dir(reports_root: Path, run_a: str, run_b: str) -> Path:
    leaf = f"{run_a.replace('/', '__')}__vs__{run_b.replace('/', '__')}"
    parent = DEFAULT_OUT_PARENT if reports_root == DEFAULT_REPORTS_ROOT else reports_root / "wfa114_cohort_matched_compare"
    return parent / leaf


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    reports_root = Path(args.reports_root).expanduser().resolve()

    try:
        run_a = load_run(reports_root, args.run_a)
        run_b = load_run(reports_root, args.run_b)
    except RunNotFound as exc:
        print(f"SKIP SKIP  {exc}")
        return 2

    reading = args.reading
    if args.epoch is not None and reading == "best":
        print(
            "NOTE  --epoch pins the comparison to a fixed index, so the 'best' "
            "reading is not used"
        )
        reading = "epoch"

    result = compare_runs(run_a, run_b, reading=reading, epoch_index=args.epoch)

    suggestion: Optional[Dict[str, Any]] = None
    if not result.ok:
        suggestion = find_suggested_pair(
            reports_root, run_a, run_b, reading=reading, exclude=(run_a.name, run_b.name)
        )

    census_payload: Optional[Dict[str, Any]] = None
    if args.census:
        census_payload = census_crosscheck(Path(args.census).expanduser(), [run_a, run_b])

    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir
        else default_out_dir(reports_root, run_a.name, run_b.name)
    )

    payload = build_payload(
        run_a,
        run_b,
        result,
        reports_root=reports_root,
        out_dir=out_dir,
        bar=args.bar,
        census=census_payload,
        suggestion=suggestion,
        reading_requested=args.reading,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "compare.json"
    md_path = out_dir / "compare.md"
    atomic_write_file(json_path, json.dumps(payload, indent=2, sort_keys=False) + "\n")
    atomic_write_file(md_path, render_markdown(payload))

    # ---- stdout (AGENTS.md §4 invariant: PASS / FAIL / SKIP) ----
    if payload["verdict"] == "VERDICT":
        e2e = payload["e2e"]
        if result.reading == "best":
            print(
                f"VERDICT PASS  {run_a.name} vs {run_b.name}  "
                f"reading=best (searched window 0..{result.epoch_index}, "
                f"budget={result.epoch_index + 1} epochs)  "
                f"compared_epochs={payload['compared_epochs']}"
            )
        else:
            print(
                f"VERDICT PASS  {run_a.name} vs {run_b.name}  "
                f"reading={result.reading} epoch_index={result.epoch_index} "
                f"(budget={result.epoch_index + 1} epochs)"
            )
        print(f"  e2e(A) = {e2e['a']['value']!r}  MEASURED  {e2e['a']['source']}")
        print(f"  e2e(B) = {e2e['b']['value']!r}  MEASURED  {e2e['b']['source']}")
        print(
            f"  delta(B-A) = {e2e['delta_b_minus_a']['value']!r}  MEASURED  "
            f"{e2e['delta_b_minus_a']['source']}"
        )
    else:
        print(
            f"REFUSED FAIL  {run_a.name} vs {run_b.name}  "
            f"unmatched_axes={payload['unmatched_axes']}"
        )
        for name in payload["unmatched_axes"]:
            print(f"  [FAIL] {name}: {result.axes[name].detail}")
    for name in ALL_AXES:
        if name in payload["unmatched_axes"]:
            continue
        print(f"  [PASS] {name}: {result.axes[name].detail}")
    if payload.get("bar"):
        bar = payload["bar"]
        print(f"  bar = {bar['value']!r}  {bar['label']}  {bar['source']}")
        if bar.get("rejection_reason"):
            print(f"  bar REJECTED (not a measurement): {bar['rejection_reason']}")
        if bar.get("b_meets_bar") is not None:
            print(
                f"  arm B (candidate) meets bar: {bar['b_meets_bar']}; "
                f"arm A (reference) meets bar: {bar.get('a_meets_bar')}"
            )
    if suggestion:
        print(
            f"  suggested matched pair: {suggestion['run_a']} vs "
            f"{suggestion['run_b']} at epoch index {suggestion['epoch_index']}"
        )
    print(f"  wrote {_rel(json_path)} and {_rel(md_path)}")
    print(f"  caveat: {payload['caveat']}")

    return 0 if payload["verdict"] == "VERDICT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
