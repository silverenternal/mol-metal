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

"""Run ledger — append-only provenance for every metric value.

WHY THIS EXISTS
---------------
``metrics/by_metric/*.json`` records a "current best" for each metric but keeps
NO history: there is no way to tell *when* a number was produced or *from which*
git / data / config state.  ``registry/metrics.yaml`` (the SSOT) carries a
``source_json_sha256`` and a ``last_refreshed`` stamp, but it is regenerated in
place — the previous value is overwritten.  So "which number is real, and when
did it become real?" cannot be answered from disk.

The fix is a single append-only ledger.  Every time a metric value is recorded,
one JSON line is appended to ``molmetal/reports/run_ledger.jsonl`` carrying the
value, the source JSON path, and the provenance hashes (git SHAs + data hash +
config hash) that identify the reproducibility state.  The file is *never*
rewritten; ``latest()`` and ``history()`` read it back.  ``staleness()`` then
answers "is the registry's current number still backed by the newest run?" by
comparing the ledger's latest entry against each registry row.

This is the executable-provenance counterpart to AGENTS.md §5.2 (honest framing
— every number traces to a run) + §5.5 (filesystem-truth) + §5.6 (atomic write,
append-only here).

HONEST FRAMING (AGENTS.md §5.2): every hash written here is MEASURED on the host
at record time; the seed pass copies values verbatim from the source JSON named
beside them and marks them `seed=True`.  Nothing is PROJECTED.  Seeding does NOT
invent history — it records exactly one baseline entry per registry metric from
the current on-disk state.

APPEND-ONLY CONTRACT: ``record_run`` opens the ledger in ``"a"`` mode and writes
after ``fsync``; it never truncates or rewrites.  A single write of one line is
atomic enough for O_APPEND on POSIX (writes < PIPE_BUF are not interleaved).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - PyYAML is a hard dep of this repo
    yaml = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl"
REGISTRY_METRICS_PATH = REPO_ROOT / "registry" / "metrics.yaml"
REGISTRY_DATASETS_PATH = REPO_ROOT / "registry" / "datasets.yaml"
BY_METRIC_DIR = REPO_ROOT / "metrics" / "by_metric"

#: env var that turns the staleness pre-flight gate from report-only (SKIP) to
#: enforced.  Opt-in per the AGENTS.md §5.6/§5.7 new-gate pattern.
RUN_LEDGER_ENFORCE_ENV = "WF_RUN_LEDGER"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git_sha(repo_root: Path = REPO_ROOT) -> Optional[str]:
    """Return the current HEAD sha256-style commit id, or None outside git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None


def _resolve_source_json(source_json: str, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a source_json string to an absolute path.

    Accepts a repo-relative path (``metrics/by_metric/x.json``) or a bare stem
    (``x`` / ``x.json``), which resolves under ``metrics/by_metric/``.
    """
    p = Path(source_json)
    if p.is_absolute():
        return p
    if "/" in source_json or p.suffix == ".json":
        return repo_root / p
    return BY_METRIC_DIR / f"{source_json}.json"


def _resolve_data_source(data_source: str, repo_root: Path = REPO_ROOT) -> Optional[Path]:
    """Resolve a data_source string (possibly prose) to an on-disk file.

    ``data_source`` in the metric JSONs is often a human sentence, e.g.
    ``"MetalCytoToxDB.csv (precious-metal subset)"``.  Take the leading token
    up to whitespace / paren and try to resolve it against a few known roots;
    return the first that exists, else None.  Never fabricate a path.
    """
    if not data_source:
        return None
    token = str(data_source).strip().split()[0].split("(")[0].strip().strip(",;")
    if not token:
        return None
    candidates = [
        Path(token),
        repo_root / token,
        repo_root / ".data" / "molmetal" / Path(token).name,
        repo_root / "molmetal" / "data" / Path(token).name,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _append_line(entry: Dict[str, Any], ledger_path: Path = LEDGER_PATH) -> Path:
    """Atomically append ONE JSON line to the ledger.  Never rewrites."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n"
    # O_APPEND: a single write of a short line is not interleaved with other
    # appenders; fsync flushes before we return so a crash cannot lose it.
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    return ledger_path


def record_run(
    metric_id: str,
    value: Any,
    *,
    source_json: str,
    git_sha: Optional[str] = None,
    data_sha: Optional[str] = None,
    config_sha: Optional[str] = None,
    data_source: Optional[str] = None,
    config_path: Optional[str] = None,
    protocol_id: Optional[str] = None,
    host: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    source_json_sha256: Optional[str] = None,
    timestamp: Optional[str] = None,
    seed: bool = False,
    ledger_path: Path = LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Append ONE entry to the ledger and return the entry that was written.

    ``git_sha`` defaults to ``git rev-parse HEAD`` when not supplied; a
    ``None`` result (outside a git tree) is recorded as ``null`` rather than
    fabricated.  ``source_json_sha256`` defaults to the sha256 of ``source_json``.
    ``data_sha`` / ``config_sha`` are used verbatim when supplied; otherwise they
    are derived by hashing ``data_source`` (a metric ``data_source`` string, a
    possibly-prose value — its leading path token is resolved against the repo)
    and ``config_path`` when those resolve to an on-disk file.  An unresolvable
    data_source / config_path leaves the hash ``None`` rather than fabricated.
    """
    if not metric_id:
        raise ValueError("record_run requires a metric_id")
    src_path = _resolve_source_json(source_json, repo_root=repo_root)
    if source_json_sha256 is None:
        source_json_sha256 = _sha256_file(src_path)
    if git_sha is None:
        git_sha = _git_sha(repo_root)
    if data_sha is None and data_source:
        dp = _resolve_data_source(data_source, repo_root=repo_root)
        if dp is not None:
            data_sha = _sha256_file(dp)
    if config_sha is None and config_path:
        cp = Path(config_path)
        if not cp.is_absolute():
            cp = repo_root / cp
        if cp.is_file():
            config_sha = _sha256_file(cp)
    entry: Dict[str, Any] = {
        "metric_id": metric_id,
        "value": value,
        "source_json": source_json,
        "source_json_sha256": source_json_sha256,
        "git_sha": git_sha,
        "data_sha": data_sha,
        "config_sha": config_sha,
        "protocol_id": protocol_id,
        "host": host if host is not None else os.uname().nodename,
        "timestamp": timestamp if timestamp is not None else _now_utc(),
        "seed": bool(seed),
    }
    if extra:
        entry["extra"] = extra
    _append_line(entry, ledger_path=ledger_path)
    return entry


def _iter_entries(ledger_path: Path = LEDGER_PATH) -> List[Dict[str, Any]]:
    """Read every entry.  Skips blank lines; a corrupt line is surfaced, not
    silently dropped (a corrupt ledger is a real problem, not noise)."""
    if not ledger_path.is_file():
        return []
    entries: List[Dict[str, Any]] = []
    for lineno, raw in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            entries.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"corrupt JSON on {ledger_path}:{lineno}: {exc}"
            ) from exc
    return entries


def latest(metric_id: str, ledger_path: Path = LEDGER_PATH) -> Optional[Dict[str, Any]]:
    """Return the most recently appended entry for ``metric_id``, or None."""
    found: Optional[Dict[str, Any]] = None
    for entry in _iter_entries(ledger_path):
        if entry.get("metric_id") == metric_id:
            found = entry
    return found


def history(metric_id: str, ledger_path: Path = LEDGER_PATH) -> List[Dict[str, Any]]:
    """Return every entry for ``metric_id``, in append order (oldest first)."""
    return [e for e in _iter_entries(ledger_path) if e.get("metric_id") == metric_id]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def staleness(
    registry_metrics_path: Path = REGISTRY_METRICS_PATH,
    *,
    ledger_path: Path = LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Dict[str, Any]]:
    """Compare the ledger's latest entry per metric against the registry row.

    For each metric in ``registry/metrics.yaml``:
      - ``no_run``      — no ledger entry exists for this metric.
      - ``stale``       — the registry's ``source_json_sha256`` differs from the
                          ledger's latest ``source_json_sha256`` **and the value
                          differs too** (the registry was refreshed from a source
                          the ledger has not seen, or the source changed since
                          the ledger entry was written).
      - ``reformatted`` — the hashes differ but the recorded value is identical:
                          the JSON gained a field, lost whitespace or was
                          re-serialised. The number is still the one the ledger
                          saw, so this is not a provenance gap.
      - ``current``     — the latest ledger entry's source_json hash matches the
                          registry row.

    ``reformatted`` exists because MEASURED 2026-10-21 three of four "stale"
    rows (m7_diversity, m2_r2_per_cell, m21_nci60_9cell) carried byte-identical
    values and only the file had been rewritten. A staleness gate that cries
    wolf on a reformat is a gate people stop reading, and the one row that had
    really changed (l5_hilbert_r2, -5.6625 -> -6.7126) was hidden among them.

    Returns ``{metric_id: {status, ledger_timestamp, ledger_source_json_sha256,
    registry_source_json_sha256, ledger_value, registry_value}}``.
    """
    reg = _load_yaml(registry_metrics_path)
    out: Dict[str, Dict[str, Any]] = {}
    for mid, rec in (reg.get("metrics") or {}).items():
        reg_sha = rec.get("source_json_sha256")
        entry = latest(mid, ledger_path=ledger_path)
        info: Dict[str, Any] = {
            "status": "no_run",
            "ledger_timestamp": None,
            "ledger_source_json_sha256": None,
            "registry_source_json_sha256": reg_sha,
            "ledger_value": None,
            "registry_value": rec.get("value"),
        }
        if entry is not None:
            info["ledger_timestamp"] = entry.get("timestamp")
            info["ledger_source_json_sha256"] = entry.get("source_json_sha256")
            info["ledger_value"] = entry.get("value")
            if entry.get("source_json_sha256") == reg_sha:
                info["status"] = "current"
            elif _same_value(entry.get("value"), rec.get("value")):
                info["status"] = "reformatted"
            else:
                info["status"] = "stale"
        out[mid] = info
    return out


def _same_value(a: Any, b: Any) -> bool:
    """True when two recorded values are the same number, list or mapping.

    Compared through a canonical JSON dump rather than ``==``: ``0.123`` and
    ``"0.123"`` are not the same measurement, while ``1`` and ``1.0`` are, and
    a nested dict must compare by content rather than by identity.
    """

    def canon(v: Any) -> str:
        try:
            return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return repr(v)

    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return canon(a) == canon(b)


def staleness_summary(
    registry_metrics_path: Path = REGISTRY_METRICS_PATH,
    *,
    ledger_path: Path = LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, int]:
    """Counts of current / reformatted / stale / no_run across the registry metrics."""
    counts = {"current": 0, "reformatted": 0, "stale": 0, "no_run": 0}
    for info in staleness(
        registry_metrics_path, ledger_path=ledger_path, repo_root=repo_root
    ).values():
        counts[info["status"]] = counts.get(info["status"], 0) + 1
    return counts


def seed_from_disk(
    registry_metrics_path: Path = REGISTRY_METRICS_PATH,
    *,
    registry_protocols_path: Optional[Path] = None,
    ledger_path: Path = LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
    skip_existing: bool = True,
) -> List[Dict[str, Any]]:
    """Seed the ledger with ONE baseline entry per registry metric.

    For each metric in ``registry/metrics.yaml`` append exactly one entry built
    from the CURRENT on-disk state: the value + sha256 recorded in the registry
    row and the source JSON's own ``timestamp`` (falling back to the registry's
    ``last_refreshed`` stamp, then ``now``).  ``data_sha`` is derived from the
    source JSON's ``data_source``; ``config_sha`` from the metric's
    ``trainer_config`` when ``registry/protocols.yaml`` names one.  This records
    the baseline and does NOT invent history.  With ``skip_existing`` (default) a
    metric that already has a ledger entry is left untouched, so re-running the
    seed never duplicates the baseline.
    """
    reg = _load_yaml(registry_metrics_path)
    protocols_path = (
        registry_protocols_path
        if registry_protocols_path is not None
        else registry_metrics_path.parent / "protocols.yaml"
    )
    protos = (_load_yaml(protocols_path).get("protocols") or {})
    written: List[Dict[str, Any]] = []
    already = {e.get("metric_id") for e in _iter_entries(ledger_path)}
    for mid, rec in (reg.get("metrics") or {}).items():
        if skip_existing and mid in already:
            continue
        src = rec.get("source_json", "")
        src_path = _resolve_source_json(src, repo_root=repo_root)
        payload = _read_json(src_path)
        ts = (
            payload.get("timestamp")
            or rec.get("metric_timestamp")
            or rec.get("last_refreshed")
            or _now_utc()
        )
        trainer_cfg = (protos.get(mid) or {}).get("trainer_config")
        entry = record_run(
            mid,
            rec.get("value"),
            source_json=src,
            source_json_sha256=rec.get("source_json_sha256"),
            data_source=payload.get("data_source"),
            config_path=trainer_cfg,
            timestamp=ts,
            seed=True,
            ledger_path=ledger_path,
            repo_root=repo_root,
            extra={"value_key": rec.get("value_key"), "data_source": payload.get("data_source")},
        )
        written.append(entry)
    return written


def check_ledger(
    registry_metrics_path: Path = REGISTRY_METRICS_PATH,
    *,
    ledger_path: Path = LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Report-only staleness check.  Returns {status, counts, detail}."""
    info = staleness(registry_metrics_path, ledger_path=ledger_path, repo_root=repo_root)
    counts = {"current": 0, "reformatted": 0, "stale": 0, "no_run": 0}
    for v in info.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    return {
        # `reformatted` is not a provenance gap: the value the ledger saw is
        # still the value on disk. Only a changed value or a missing run is.
        "status": "PASS" if counts["stale"] == 0 and counts["no_run"] == 0 else "WARN",
        "counts": counts,
        "detail": info,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "status"
    if cmd == "seed":
        written = seed_from_disk()
        print(f"seeded {len(written)} entr{'y' if len(written)==1 else 'ies'} into {LEDGER_PATH}")
        return 0
    if cmd == "status":
        rep = check_ledger()
        c = rep["counts"]
        print(
            f"ledger: current={c['current']} reformatted={c['reformatted']} "
            f"stale={c['stale']} no_run={c['no_run']}"
        )
        for mid, info in rep["detail"].items():
            if info["status"] != "current":
                print(f"    {info['status']:6s} {mid}: ledger_ts={info['ledger_timestamp']}")
        print(f"LEDGER: {rep['status']}")
        return 0
    if cmd == "latest":
        if len(args) < 2:
            print("usage: run_ledger latest <metric_id>", file=sys.stderr)
            return 2
        e = latest(args[1])
        print(json.dumps(e, indent=2, ensure_ascii=False) if e else "no entry")
        return 0 if e else 1
    if cmd == "history":
        if len(args) < 2:
            print("usage: run_ledger history <metric_id>", file=sys.stderr)
            return 2
        hs = history(args[1])
        print(json.dumps(hs, indent=2, ensure_ascii=False))
        return 0
    print(f"unknown command {cmd!r}; use 'seed', 'status', 'latest', 'history'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
