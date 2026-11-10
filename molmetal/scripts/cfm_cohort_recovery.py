# SPDX-License-Identifier: Apache-2.0
"""Recover a CFM run's COHORT ``(n_train, n_val)`` from the artifacts that survive.

WHY THIS MODULE EXISTS
-----------------------
``wf_cfm_run_census.py`` reads a run's cohort out of
``<run_dir>/per_epoch_metrics.json``'s ``config`` block.  That file is written
ONCE, at the END of training, to a FIXED path, so two runs launched into the
same ``--out-dir`` CLOBBER each other's record and the survivor is whichever
finished LAST.  Wf-A114E fixed the *epoch series* by re-deriving it from the
append-only ``events.jsonl``; the *cohort* was left behind, and a run whose
per-epoch file was clobbered ended up with ``n_train = None`` -- dropped out of
every cohort axis, including the one it belongs to.

The cohort is recoverable, but ONLY from an artifact that can be tied to the
RUN OF RECORD.  This module implements that tie.

THE ATTESTATION RULE
---------------------
A run directory can hold SEVERAL populations (a real run, then a smoke that
overwrote the metrics file).  A log file on disk is the log of ONE of them.
The tie is made by TIME: the trainer stamps every ``train_epoch`` row with a
``ts``, and the log echoes the SAME rows (the trainer prints the whole event
dict).  So:

    a log attests the run of record  <=>  it echoes EVERY ``ts`` of the
                                          run-of-record population

Full coverage, not partial: a log that echoes only SOME of the run of
record's epochs is a DIFFERENT population's log (or a truncated one), and its
cohort line belongs to that other population.

WHY "FULL COVERAGE" IS NOT A TECHNICALITY
-----------------------------------------
MEASURED on disk: ``molmetal/reports/cfm_fullcorpus_2026_10_10`` holds THREE
populations -- a 14-epoch run (``n_train=26715, n_val=192``), then two 1-epoch
CPU smokes (``n_train=64, n_val=32``).  The surviving ``master.log`` attests
the 14-epoch one; the surviving ``per_epoch_metrics.json`` is the 1-epoch
smoke's.  So **populations in one directory do NOT share a cohort**, and a
sibling population's cohort line is NOT evidence about the run of record.

The same holds for the other three clobbered runs: in
``cfm_round3_retrain/run`` the surviving ``train.log`` echoes the 4-epoch
population's epochs (its ``gen_validity_n`` is 768), while the run of record
is the 20-epoch population (``gen_validity_n`` 192) -- a DIFFERENT launch with
a DIFFERENT evaluation width.  For those three runs no log, no config
snapshot and no checkpoint survives for the run of record, so the cohort is
reported as UNATTESTED rather than guessed.

SOURCES, IN ORDER
-----------------
``run_log:ts_matched``
    the log that echoes the run of record's epochs and states the cohort.
``per_epoch_metrics.json:config``
    the metrics file's own config block -- only when the file IS the run of
    record (the census decides that separately).
``unattested``
    nothing on disk ties a cohort to the run of record.  ``n_train`` and
    ``n_val`` stay ``None`` and the reason says why.

NO GPU.  This module only reads files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "COHORT_SOURCE_LOG",
    "COHORT_SOURCE_CONFIG",
    "COHORT_SOURCE_UNATTESTED",
    "METRICS_FILE",
    "scan_run_logs",
    "recover_cohort",
]

#: The per-epoch metrics file a later run into the SAME directory can
#: overwrite (see this module's docstring).  Its ABSENCE is a fact about
#: the directory, and the reason string may only state facts on disk.
METRICS_FILE = "per_epoch_metrics.json"

#: The cohort came from the run log that echoes the run of record's epochs.
COHORT_SOURCE_LOG = "run_log:ts_matched"

#: The cohort came from the metrics file's own config block.
COHORT_SOURCE_CONFIG = "per_epoch_metrics.json:config"

#: No artifact on disk ties a cohort to the run of record.
COHORT_SOURCE_UNATTESTED = "unattested"

#: The trainer's own cohort line (see ``train_cfm_yuelbond_wasserstein.py``).
#: MEASURED: ``[Wf-124 P2] building cohorts: train=26715, val=192``.
_COHORT_RE = re.compile(r"building cohorts:\s*train=(\d+),\s*val=(\d+)")

#: The trainer's post-build echo: ``[Wf-124 P2] cohort built: train=256 val=192``.
_BUILT_RE = re.compile(r"cohort built:\s*train=(\d+)\s+val=(\d+)")

#: The loader's own count: ``256 cloud(s) from 1280 candidate(s)``.  The train
#: cohort is printed first and the validation cohort second.
_CLOUD_RE = re.compile(r"(\d+)\s+cloud\(s\)\s+from\s+(\d+)\s+candidate\(s\)")

#: Log-like files that may hold a run's stdout.
_LOG_SUFFIXES = (".log", ".out", ".err")

#: The event kind whose rows carry the ``ts`` that ties a log to a population.
_TRAIN_EPOCH = "train_epoch"


def _log_paths(run_dir: Path) -> List[Path]:
    """Every log-like file directly inside ``run_dir``, sorted by name."""
    out: List[Path] = []
    try:
        entries = sorted(run_dir.iterdir())
    except OSError:
        return out
    for p in entries:
        if not p.is_file():
            continue
        if p.name == "nohup.out" or p.suffix in _LOG_SUFFIXES:
            out.append(p)
    return out


def scan_run_logs(run_dir: Path) -> List[Dict[str, object]]:
    """One record per log-like file in ``run_dir``.

    Each record carries:

    ``file``
        the log's file name.
    ``cohort``
        ``(n_train, n_val)`` from the trainer's ``building cohorts`` line, or
        ``None`` when the log does not state it.
    ``cohort_built``
        the same pair from the ``cohort built`` echo, or ``None``.
    ``cloud_counts``
        the loader's ``(clouds, candidates)`` pairs, in the order printed:
        the TRAIN cohort first, the VALIDATION cohort second.
    ``epoch_ts``
        the ``ts`` of every ``train_epoch`` row the log echoes.
    ``n_epoch_rows``
        how many ``train_epoch`` rows the log echoes.
    """
    records: List[Dict[str, object]] = []
    for path in _log_paths(Path(run_dir)):
        cohort: Optional[Tuple[int, int]] = None
        built: Optional[Tuple[int, int]] = None
        clouds: List[Tuple[int, int]] = []
        ts_values: Set[float] = set()
        n_rows = 0
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = _COHORT_RE.search(line)
                    if m:
                        cohort = (int(m.group(1)), int(m.group(2)))
                    m = _BUILT_RE.search(line)
                    if m:
                        built = (int(m.group(1)), int(m.group(2)))
                    for cm in _CLOUD_RE.finditer(line):
                        clouds.append((int(cm.group(1)), int(cm.group(2))))
                    if _TRAIN_EPOCH in line:
                        row = _row_ts(line)
                        if row is not None:
                            ts_values.add(row)
                            n_rows += 1
        except OSError:
            continue
        records.append({
            "file": path.name,
            "cohort": list(cohort) if cohort else None,
            "cohort_built": list(built) if built else None,
            "cloud_counts": [list(c) for c in clouds],
            "epoch_ts": ts_values,
            "n_epoch_rows": n_rows,
        })
    return records


def _row_ts(line: str) -> Optional[float]:
    """The ``ts`` of a ``train_epoch`` JSON row, or None when it has none.

    The log line is the trainer's own ``json.dumps`` of the event dict, so it
    is valid JSON -- but a log can be truncated mid-line, and a census that
    dies on one bad line reads nothing.
    """
    start = line.find("{")
    if start < 0:
        return None
    try:
        obj = json.loads(line[start:])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    ts = obj.get("ts")
    if isinstance(ts, (int, float)):
        return float(ts)
    return None


def _log_attests(record: Dict[str, object], population_ts: Set[float]) -> bool:
    """Does this log echo EVERY epoch of the run-of-record population?"""
    if not population_ts:
        return False
    log_ts = record.get("epoch_ts") or set()
    return population_ts.issubset(log_ts)


def _cohort_from_record(record: Dict[str, object]) -> Tuple[Optional[Tuple[int, int]], str]:
    """The cohort a log states, and whether the log's two statements agree."""
    cohort = record.get("cohort")
    built = record.get("cohort_built")
    if cohort and built:
        if tuple(cohort) != tuple(built):
            return None, (
                "the log states two DIFFERENT cohorts: 'building cohorts' "
                f"says {tuple(cohort)} and 'cohort built' says {tuple(built)}"
            )
    if cohort:
        return tuple(cohort), ""  # type: ignore[return-value]
    if built:
        return tuple(built), ""  # type: ignore[return-value]
    clouds = record.get("cloud_counts") or []
    if len(clouds) >= 2:
        return (int(clouds[0][0]), int(clouds[1][0])), ""
    return None, "the log states no cohort"


def _metrics_file_epochs(path: Path) -> Optional[Set[int]]:
    """The epoch indices the per-epoch metrics file on disk records.

    ``None`` when the file cannot be read as a per-epoch series.
    """
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(blob, dict):
        return None
    rows = blob.get("per_epoch_metrics")
    if not isinstance(rows, list):
        return None
    eps: Set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        ep = row.get("epoch")
        if isinstance(ep, bool) or not isinstance(ep, (int, float)):
            continue
        eps.add(int(ep))
    return eps


def _population_epochs(run_dir: Path, population_ts: Set[float]) -> Optional[Set[int]]:
    """The epoch indices the RUN OF RECORD's own rows carry.

    The run of record's population is identified by its ``ts`` values, so its
    epoch set is recovered by matching those ``ts`` against the run
    directory's OWN event stream.  ``None`` when the directory holds no
    event stream to match against.
    """
    events = Path(run_dir) / "events.jsonl"
    if not events.is_file():
        return None
    eps: Set[int] = set()
    seen = False
    try:
        with events.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                ts = row.get("ts")
                ep = row.get("epoch")
                if isinstance(ts, bool) or not isinstance(ts, (int, float)):
                    continue
                if isinstance(ep, bool) or not isinstance(ep, (int, float)):
                    continue
                seen = True
                if float(ts) in population_ts:
                    eps.add(int(ep))
    except OSError:
        return None
    return eps if seen else None


def recover_cohort(
    run_dir: Path,
    population_ts: Iterable[float],
    config_cohort: Optional[Tuple[Any, Any]] = None,
) -> Dict[str, object]:
    """The cohort of the RUN OF RECORD of ``run_dir``, with its provenance.

    Parameters
    ----------
    run_dir:
        the run directory.
    population_ts:
        the ``ts`` of every ``train_epoch`` row of the run-of-record
        population.  Empty means "no population is known" -- then only
        ``config_cohort`` can answer.
    config_cohort:
        the ``(n_train, n_val)`` read from the metrics file's config block.
        The caller passes it ONLY when that file IS the run of record.

    Returns
    -------
    dict
        ``n_train``, ``n_val``, ``source``, ``reason`` and ``evidence``.
    """
    population_ts = {float(t) for t in population_ts if isinstance(t, (int, float))}
    logs = scan_run_logs(Path(run_dir))

    for record in logs:
        if not _log_attests(record, population_ts):
            continue
        cohort, problem = _cohort_from_record(record)
        if cohort is None:
            return {
                "n_train": None,
                "n_val": None,
                "source": COHORT_SOURCE_UNATTESTED,
                "reason": (
                    f"the run log {record['file']} echoes every epoch of the "
                    f"run of record ({len(population_ts)} epochs), but "
                    f"{problem}, so no cohort can be read from it"
                ),
                "evidence": {
                    "log": record["file"],
                    "epochs_matched": len(population_ts),
                    "cohort": None,
                },
            }
        return {
            "n_train": int(cohort[0]),
            "n_val": int(cohort[1]),
            "source": COHORT_SOURCE_LOG,
            "reason": (
                f"the run log {record['file']} echoes every epoch of the run "
                f"of record ({len(population_ts)} of {len(population_ts)} "
                f"timestamps matched) and states the cohort it built"
            ),
            "evidence": {
                "log": record["file"],
                "epochs_matched": len(population_ts),
                "cohort": [int(cohort[0]), int(cohort[1])],
                "cohort_built": record.get("cohort_built"),
                "cloud_counts": record.get("cloud_counts"),
            },
        }

    if population_ts and logs:
        detail = "; ".join(
            f"{r['file']} echoes {len(r.get('epoch_ts') or set())} epoch(s)"
            for r in logs
        )
        reason = (
            f"no log in this directory echoes every epoch of the run of "
            f"record ({len(population_ts)} epochs): {detail}. A sibling "
            f"population's log does NOT attest the run of record's cohort -- "
            f"populations in one directory do not share a cohort (MEASURED: "
            f"cfm_fullcorpus_2026_10_10 holds a 26715/192 run beside a "
            f"64/32 CPU smoke)"
        )
    elif population_ts:
        # The reason may state ONLY what is ON DISK.  The old wording ("the
        # per-epoch metrics file was overwritten by a later run") named a
        # file that need not exist at all: MEASURED,
        # ``cfm_c7_fullgen_ep4_partial_2026_10_20`` holds NO
        # ``per_epoch_metrics.json``, so the clause named a file that is not
        # there.  The clause is now kept only for a file that IS on disk and
        # records a DIFFERENT set of epochs than the run of record.
        metrics_path = Path(run_dir) / METRICS_FILE
        if not metrics_path.is_file():
            reason = (
                "this run directory has no log file and no per-epoch metrics "
                "file, so no artifact on disk ties a cohort to the run of "
                "record"
            )
        else:
            file_epochs = _metrics_file_epochs(metrics_path)
            ro_epochs = _population_epochs(Path(run_dir), population_ts)
            if (
                file_epochs is not None
                and ro_epochs is not None
                and file_epochs != ro_epochs
            ):
                reason = (
                    "this run directory has no log file, and the per-epoch "
                    "metrics file on disk records a different set of epochs "
                    "than the run of record (it was overwritten by a later "
                    "run), so no artifact ties a cohort to the run of record"
                )
            else:
                reason = (
                    "this run directory has no log file, and the per-epoch "
                    "metrics file on disk does not attest the run of record, "
                    "so no artifact ties a cohort to the run of record"
                )
    else:
        reason = (
            "no population is known for this run, so no log can be tied to a "
            "run of record"
        )

    if config_cohort is not None:
        n_train, n_val = config_cohort
        if n_train is not None and n_val is not None:
            return {
                "n_train": n_train,
                "n_val": n_val,
                "source": COHORT_SOURCE_CONFIG,
                "reason": (
                    "the cohort comes from the per-epoch metrics file's own "
                    "config block, which is the run of record"
                ),
                "evidence": {"config_block": [n_train, n_val]},
            }

    return {
        "n_train": None,
        "n_val": None,
        "source": COHORT_SOURCE_UNATTESTED,
        "reason": reason,
        "evidence": {
            "n_logs": len(logs),
            "logs": [
                {
                    "file": r["file"],
                    "n_epoch_rows": r.get("n_epoch_rows"),
                    "cohort": r.get("cohort"),
                }
                for r in logs
            ],
        },
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="the run directory to inspect")
    args = ap.parse_args(argv)

    d = Path(args.run_dir)
    logs = scan_run_logs(d)
    print(f"run_dir: {d}")
    for rec in logs:
        print(
            f"  log {rec['file']}: cohort={rec['cohort']} "
            f"built={rec['cohort_built']} epoch_rows={rec['n_epoch_rows']} "
            f"clouds={rec['cloud_counts']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
