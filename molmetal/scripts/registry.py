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

"""Registry — the machine-readable single source of truth for the project.

WHY THIS EXISTS
---------------
The recurring failure mode is *agent misjudgement on status refresh*: an agent
asked "is the dataset ready / is the protocol still locked / is this metric
current?" re-reads prose docs and re-derives the answer by inference.  When the
inference chain breaks (a stale snapshot, a doc that lags disk, a lost file),
the agent reports a FALSE wrong answer — "dataset not ready", "protocol lost",
"metric stale" — and every downstream decision inherits it.

The fix is to stop asking agents to *reason* about status and let them *query*
it against the filesystem.  Three registry files carry the facts, and
``registry_check.py`` verifies each fact against disk by content hash:

  registry/datasets.yaml   dataset id -> path, sha256, n_rows, schema_cols,
                           split_protocol, source_paper
  registry/protocols.yaml  metric id -> baseline, alignment_level, dataset_id,
                           split, n_folds, protocol_lock_ref
  registry/metrics.yaml    metric id -> current_value, source_json,
                           source_json_sha256, target, verdict,
                           measured_projected, last_refreshed

A fact is TRUE iff the hash on disk matches the hash recorded here.  "Ready" is
not a claim; it is a sha256 comparison.  This is the executable-contract
upgrade of AGENTS.md §5.5 (filesystem-truth) + §5.6 (atomic write).

This module is the GENERATOR + CHECKER in one place:
  - ``build_registry(repo_root)`` harvests datasets.yaml from
    configs/data_checksums.json + configs/locks/protocol_lock.yaml, and
    metrics.yaml from metrics/by_metric/*.json, then reports what it could not
    cover so the gap is explicit rather than silent.
  - ``check_registry(repo_root, phase=...)`` verifies every entry against disk
    and returns PASS / FAIL / SKIP per layer (report-only by default; the
    pre-flight gate may promote to enforced).

HONEST FRAMING (AGENTS.md §5.2): every sha256 written here is MEASURED on the
host at generation time; every ``value`` is copied verbatim from the source
JSON named beside it.  Nothing is PROJECTED.  A missing source JSON is recorded
as ``source_json_missing: true`` rather than silently dropped.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make ``molmetal.*`` importable when this file is run directly as
# ``python molmetal/scripts/registry.py``. Without this, only the script's own
# directory is on sys.path and ``generate()``'s
# ``from molmetal.scripts.wf386a3_atomic_write import atomic_write_file``
# raises ModuleNotFoundError — while ``check()`` keeps working, because it
# imports nothing from the package. That asymmetry is what makes the bug easy
# to miss: the read path looks healthy and only the write path breaks.
# Measured 2026-10-10; canonical pattern copied from pre_flight_check.py:69-71.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - PyYAML is a hard dep of this repo
    yaml = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO_ROOT / "registry"
DATASETS_PATH = REGISTRY_DIR / "datasets.yaml"
PROTOCOLS_PATH = REGISTRY_DIR / "protocols.yaml"
METRICS_PATH = REGISTRY_DIR / "metrics.yaml"

DATA_CHECKSUMS_PATH = REPO_ROOT / "configs" / "data_checksums.json"
PROTOCOL_LOCK_PATH = REPO_ROOT / "configs" / "locks" / "protocol_lock.yaml"
BY_METRIC_DIR = REPO_ROOT / "metrics" / "by_metric"

# The three paper lines (per TODO/03_plans/PROTOCOL_RECONCILIATION_AND_RE_PLAN_2026_10_07.md).
# Each metric id below is a *registry* id; the value maps it to the on-disk
# metrics/by_metric/<file>.json that carries its current number.
LINE_METRICS: Dict[str, Dict[str, str]] = {
    "line1_cfm": {
        "m14_decode_ratio": "decode_ratio",
        "cfm_bond_accuracy": "cfm_bond_accuracy",
        "m3_vina_dock": "vina_kcal_per_mol",
        "m7_diversity": "m7_diversity",
    },
    "line2_dmpnn": {
        "m1_ru_auc": "metalcytotox_ru_auc",
        "m19_lomo": "metal_lomo_auc_wasserstein",
        # m2 is an R², so it must read an R² file.  Until 2026-10-20 this
        # pointed at platinai_4cell_auc.json, whose unit is ROC-AUC: the gap
        # table then compared 0.7105 (AUC) against the R² target 0.85 and
        # reported a 0.14 shortfall where the real one is 0.88.
        "m2_r2_per_cell": "m2_r2_per_cohort",
        "m21_nci60_9cell": "nci60_precious_9cell_auc",
    },
    "line3_lambda_mcts": {
        "l5_hilbert_r2": "hilbert_4cell_r2",
        # Was `molmetal/reports/wf67_p4_2_closure_r20/reachability_overall`, a
        # path .gitignore:145 ignores — so the paper line's L6 number had no
        # source in version control.  The metric file now carries the headline
        # and names the report as its (untracked) raw artifact.
        "l6_closure": "l6_closure",
        "m6_kappa": "m6_kappa",
    },
}

METRIC_TARGETS: Dict[str, Dict[str, Any]] = {
    "m14_decode_ratio": {
        "value_key": "aggregate.best_measured_real",
        "target": 0.50,
        "direction": ">=",
        "verdict": "PASS",
        "remains": "nothing for the gate (real 0.9980 / synth 1.000)",
    },
    "cfm_bond_accuracy": {
        "target": None,
        "direction": None,
        "verdict": "HONEST_NEGATIVE",
        "remains": "no scalar emitted on disk; one GPU probe to score it",
    },
    "m3_vina_dock": {
        # The only REAL number on disk is the cisplatin reference mean.  It was
        # stored under ``v3_cohort_median_REAL``, a name that claims it is the
        # generated-molecule cohort median — it is not, and its own ``_source``
        # string says so.  The −7.782 "V3 cohort median" is PROJECTED
        # arithmetic (333_final_synthesis.md §0), not a docking run.
        "value_key": "cisplatin_reference_mean_kcal_per_mol",
        "target": -8.5,
        "direction": "<=",
        "verdict": "HONEST_NEGATIVE",
        "remains": "Pt+2 parameterisation (BLOCKED, Wf-299B); the REAL value here is the cisplatin positive control, not a generated-molecule score",
    },
    "m7_diversity": {
        # No target: the metric file explicitly refuses one
        # (target.note = "NO target is asserted"), so inventing 0.65 here
        # manufactured a PASS the file never claimed.  No verdict override
        # either — the file's own MEASURED_NOT_A_CLAIM must survive.
        "value_key": "aggregate.best_measured",
        "target": None,
        "direction": None,
        "remains": "no target is asserted at n=37-47 (see m7_diversity.json::target.note); GPU launch L1 would convert PROJECTED->MEASURED, not make the number comparable",
    },
    "m1_ru_auc": {
        "target": 0.86,
        "direction": ">=",
        "verdict": "HONEST_NEGATIVE",
        "remains": "cohort honest-negative (591 vs 23 cell lines); no launch closes it",
    },
    "m19_lomo": {
        "target": 0.70,
        "direction": ">=",
        "verdict": "HONEST_NEGATIVE",
        "remains": "P-A drift ALREADY FIXED on disk (real-only 5-fold); doc retag only",
    },
    "m2_r2_per_cell": {
        "value_key": "aggregate.mean_r2",
        "target": 0.85,
        "direction": ">=",
        "verdict": "HONEST_NEGATIVE",
        "remains": "closed structural (Wf-369-A2); REAL cohorts per Wf-388-A1 (2 of 3 independent — hilbert is aliased); no GPU",
    },
    "m21_nci60_9cell": {
        "target": 0.70,
        "direction": ">=",
        "verdict": "PASS",
        "remains": "nothing for gate; scaffold split enforced in code",
    },
    "l5_hilbert_r2": {
        "target": 0.85,
        "direction": ">=",
        "verdict": "HONEST_NEGATIVE",
        "remains": "architectural ceiling (~0.68); drop-from-S3 or metric redefinition; NO GPU",
    },
    "l6_closure": {
        "value_key": "aggregate.closure_reachability_rate",
        "target": 0.95,
        "direction": ">=",
        "verdict": "HONEST_NEGATIVE",
        "remains": "volume fix (75->100/metal, CPU); reachability ceiling stays",
    },
    "m6_kappa": {
        "target": 0.60,
        "direction": ">=",
        "verdict": "HONEST_NEGATIVE",
        "remains": "drop-from-S3 per Wf-389-A1; same family as L5",
    },
}


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _strip_generated_at(text: str) -> str:
    """Drop timestamp header lines so a regenerate that only moves the clock
    compares equal to what is already on disk (avoids needless rewrites)."""
    return "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith(("generated_at:", "    last_refreshed:"))
    )


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_datasets(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Harvest the dataset layer from configs/data_checksums.json + protocol_lock.

    Every entry is a *measured* copy: the sha256 is the one already frozen in
    data_checksums.json (or read live when the file is present and only the lock
    declares it).  Nothing is invented.
    """
    out: Dict[str, Any] = {}
    checks = _load_json(DATA_CHECKSUMS_PATH)
    for rel_path, rec in (checks.get("files") or {}).items():
        out[_dataset_id(rel_path)] = {
            "path": rel_path,
            "sha256": rec.get("sha256"),
            "n_lines": rec.get("n_lines"),
            "frozen_at": rec.get("frozen_at"),
            "source": "configs/data_checksums.json",
        }

    lock = _load_yaml(PROTOCOL_LOCK_PATH)
    for metric_key, entry in (lock.get("metrics") or {}).items():
        for item in entry.get("accepted_dataset_md5") or []:
            if not isinstance(item, dict):
                continue
            p = item.get("path")
            if not p:
                continue
            did = _dataset_id(p)
            if did in out:
                out[did].setdefault("locked_for", []).append(metric_key)
                continue
            live = Path(p)
            if not live.is_absolute():
                live = repo_root / live
            out[did] = {
                # The lock records md5, not sha256; keep it as the lock's own
                # md5 (a different but equally checkable digest) and note it.
                "path": _rel(live) if live.is_absolute() else p,
                "md5": item.get("md5"),
                "sha256": _sha256_file(live),
                "why": item.get("why"),
                "source": "configs/locks/protocol_lock.yaml",
                "locked_for": [metric_key],
            }
    return out


def _dataset_id(path: str) -> str:
    return Path(path).name


def build_protocols(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Harvest the protocol layer from configs/locks/protocol_lock.yaml."""
    lock_path = PROTOCOL_LOCK_PATH
    lock_sha = _sha256_file(lock_path)
    out: Dict[str, Any] = {}
    lock = _load_yaml(lock_path)
    for metric_key, entry in (lock.get("metrics") or {}).items():
        out[metric_key] = {
            "description": entry.get("description"),
            "baseline_paper": entry.get("baseline_paper"),
            "baseline_source": entry.get("baseline_source"),
            "alignment_level": entry.get("alignment_level"),
            "dataset_id": _dataset_id(
                (entry.get("accepted_dataset_md5") or [{}])[0].get("path", "")
            ),
            "split": entry.get("split"),
            "n_folds": entry.get("n_folds"),
            "seeds": entry.get("seeds"),
            "epochs": entry.get("epochs"),
            "trainer_config": entry.get("trainer_config"),
            "protocol_lock_sha256": lock_sha,
            "protocol_lock_ref": _rel(lock_path),
        }
    return out


def _metric_timestamp(payload: Dict[str, Any]) -> Any:
    """The date a metric file records, under any of the three keys in use.

    ``timestamp`` (105 files), ``date`` (60) and ``date_utc`` (3) all mean the
    same thing; reading only the first left 63 dated files with a null
    timestamp in the registry.
    """
    for key in ("timestamp", "date", "date_utc"):
        v = payload.get(key)
        if v:
            return v
    return None


def _n_seeds(payload: Dict[str, Any]) -> Optional[float]:
    v = payload.get("n_seeds")
    return _coerce_float(v) if _coerce_float(v) is not None else None


def build_metrics(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Harvest the metric layer from metrics/by_metric/*.json.

    Two passes per Wf-A36-3 G1 (2026-10-14):

    1. The three paper-line metrics (LINE_METRICS) get the full
       ``target`` / ``direction`` / ``verdict`` wiring from
       :data:`METRIC_TARGETS`.
    2. **Every other JSON under ``metrics/by_metric/``** also gets a
       registry row — without the targets wiring, but with the
       sha256, value, status, scope, and verdict pulled verbatim from
       the on-disk JSON.  The pre-A36-3 behaviour was to drop these
       into an ``uncovered_metric_files`` list and leave them out of
       the registry, which made 240+ JSONs invisible to the registry
       check.

    The ``metric_id`` for pass 2 is taken from the JSON's own
    ``metric`` field, falling back to the file stem.  Pass-1 ids keep
    precedence when a file is named under both ``LINE_METRICS`` and
    the disk glob.
    """
    out: Dict[str, Any] = {}
    covered_files: set = set()
    for _line, mapping in LINE_METRICS.items():
        for metric_id, src in mapping.items():
            covered_files.add(src)
            # A src may be a bare stem (under metrics/by_metric/) or a
            # repo-relative path (e.g. a report JSON outside by_metric/).
            if "/" in src:
                src_path = REPO_ROOT / f"{src}.json"
            else:
                src_path = BY_METRIC_DIR / f"{src}.json"
            payload = _load_json(src_path)
            exists = src_path.is_file()
            rec: Dict[str, Any] = {
                "source_json": _rel(src_path),
                "source_json_sha256": _sha256_file(src_path),
                "source_json_missing": not exists,
                "value": _extract_value(payload, METRIC_TARGETS.get(metric_id, {}).get("value_key")),
                "status": payload.get("status"),
                "verdict": payload.get("verdict"),
                "scope": payload.get("scope"),
                "metric_timestamp": _metric_timestamp(payload),
                "registry_role": "paper_line",
                "last_refreshed": _now_utc(),
            }
            seeds = _n_seeds(payload)
            if seeds is not None:
                rec["n_seeds"] = seeds
            # A MEASURED label with no scalar is invisible to any target
            # comparison.  Flag it rather than letting `value: null` read as
            # "no data" when the file claims to have measured something.
            rec["value_missing"] = rec["value"] is None and str(
                rec.get("status") or ""
            ).startswith("MEASURED")
            rec.update(METRIC_TARGETS.get(metric_id, {}))
            out[metric_id] = rec
    # Wf-A36-3 G1 (2026-10-14): walk every JSON under metrics/by_metric/
    # and add a registry row for each one not already covered by
    # LINE_METRICS.  No METRIC_TARGETS wiring for these — they live in
    # the registry because they exist on disk, not because they map to
    # a paper line.
    for src_path in sorted(BY_METRIC_DIR.glob("*.json")):
        stem = src_path.stem
        if stem in covered_files:
            continue
        covered_files.add(stem)
        payload = _load_json(src_path)
        exists = src_path.is_file()
        metric_id = payload.get("metric") if isinstance(payload, dict) else None
        if not metric_id or not isinstance(metric_id, str):
            metric_id = stem
        # Avoid clobbering a LINE_METRICS entry when the JSON's
        # ``metric`` field happens to equal a paper-line id.  Several files
        # share a ``metric`` field (7 vina_score_* files all say
        # "vina_score"), so name the variant after its file stem — an id like
        # ``vina_score__disk`` says nothing about which file it came from.
        if metric_id in out:
            metric_id = stem if stem not in out else f"{metric_id}__disk"
        rec = {
            "source_json": _rel(src_path),
            "source_json_sha256": _sha256_file(src_path),
            "source_json_missing": not exists,
            "value": _extract_value(payload),
            "status": payload.get("status") if isinstance(payload, dict) else None,
            "verdict": payload.get("verdict") if isinstance(payload, dict) else None,
            "scope": payload.get("scope") if isinstance(payload, dict) else None,
            "metric_timestamp": _metric_timestamp(payload) if isinstance(payload, dict) else None,
            "registry_role": "artifact",
            "last_refreshed": _now_utc(),
            "registry_origin": "wf_a36_3_g1_disk_walk",
        }
        seeds = _n_seeds(payload) if isinstance(payload, dict) else None
        if seeds is not None:
            rec["n_seeds"] = seeds
        rec["value_missing"] = rec["value"] is None and str(
            rec.get("status") or ""
        ).startswith("MEASURED")
        out[metric_id] = rec
    return out, covered_files


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    layer: str
    status: str  # PASS / FAIL / SKIP
    checks: List[Dict[str, str]] = field(default_factory=list)
    evidence: str = ""


def check_datasets(repo_root: Path = REPO_ROOT) -> CheckResult:
    reg = _load_yaml(DATASETS_PATH)
    if not reg:
        return CheckResult("datasets", "SKIP", [], f"{DATASETS_PATH} absent/empty")
    checks: List[Dict[str, str]] = []
    bad = 0
    for did, rec in (reg.get("datasets") or {}).items():
        p = repo_root / rec["path"]
        if not p.is_file():
            checks.append({"id": did, "status": "FAIL", "detail": "file missing on disk"})
            bad += 1
            continue
        if rec.get("sha256"):
            live = _sha256_file(p)
            if live != rec["sha256"]:
                checks.append(
                    {"id": did, "status": "FAIL", "detail": f"sha256 drift: {rec['sha256'][:12]} -> {(live or 'nil')[:12]}"}
                )
                bad += 1
                continue
        checks.append({"id": did, "status": "PASS", "detail": "sha256 match"})
    status = "FAIL" if bad else "PASS"
    return CheckResult("datasets", status, checks, f"{len(checks)-bad}/{len(checks)} datasets verified")


def check_protocols(repo_root: Path = REPO_ROOT) -> CheckResult:
    reg = _load_yaml(PROTOCOLS_PATH)
    if not reg:
        return CheckResult("protocols", "SKIP", [], f"{PROTOCOLS_PATH} absent/empty")
    live_lock_sha = _sha256_file(PROTOCOL_LOCK_PATH)
    checks: List[Dict[str, str]] = []
    bad = 0
    for mid, rec in (reg.get("protocols") or {}).items():
        if live_lock_sha is None:
            checks.append({"id": mid, "status": "FAIL", "detail": "protocol_lock.yaml missing"})
            bad += 1
            continue
        if rec.get("protocol_lock_sha256") != live_lock_sha:
            checks.append(
                {"id": mid, "status": "FAIL", "detail": "protocol_lock.yaml changed since registry build"}
            )
            bad += 1
            continue
        checks.append({"id": mid, "status": "PASS", "detail": "lock hash intact"})
    status = "FAIL" if bad else "PASS"
    return CheckResult("protocols", status, checks, f"{len(checks)-bad}/{len(checks)} protocol locks verified")


def check_metrics(repo_root: Path = REPO_ROOT) -> CheckResult:
    reg = _load_yaml(METRICS_PATH)
    if not reg:
        return CheckResult("metrics", "SKIP", [], f"{METRICS_PATH} absent/empty")
    checks: List[Dict[str, str]] = []
    bad = 0
    stale = 0
    for mid, rec in (reg.get("metrics") or {}).items():
        src = repo_root / rec["source_json"]
        if rec.get("source_json_missing") or not src.is_file():
            checks.append({"id": mid, "status": "FAIL", "detail": "source_json missing on disk"})
            bad += 1
            continue
        live = _sha256_file(src)
        if live != rec.get("source_json_sha256"):
            checks.append({"id": mid, "status": "STALE", "detail": "source_json changed since last_refreshed"})
            stale += 1
            continue
        checks.append({"id": mid, "status": "PASS", "detail": "source_json hash current"})
    status = "FAIL" if bad else ("STALE" if stale else "PASS")
    rows = list((reg.get("metrics") or {}).values())
    unlabelled_paper = sum(
        1
        for r in rows
        if r.get("status") is None and r.get("registry_role") == "paper_line"
    )
    unlabelled_artifact = sum(
        1
        for r in rows
        if r.get("status") is None and r.get("registry_role") != "paper_line"
    )
    undated = sum(1 for r in rows if r.get("metric_timestamp") is None)
    value_missing = sum(1 for r in rows if r.get("value_missing"))
    return CheckResult(
        "metrics",
        status,
        checks,
        f"{len(checks)-bad-stale} current, {stale} stale, {bad} missing; "
        f"{unlabelled_paper} paper-line and {unlabelled_artifact} artifact rows "
        f"without a status; {undated} without a date; "
        f"{value_missing} labelled MEASURED with no scalar",
    )


def check_all(repo_root: Path = REPO_ROOT) -> Dict[str, CheckResult]:
    return {
        "datasets": check_datasets(repo_root),
        "protocols": check_protocols(repo_root),
        "metrics": check_metrics(repo_root),
    }


def _coerce_float(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# The metric JSONs store their headline under different keys depending on the
# wave that produced them.  Try the common ones in order so the registry value
# tracks the real number rather than reading a misleading None.
_VALUE_KEYS = (
    "value",
    "value_overall_mean_auc",
    "mean_auc",
    "aggregate_combined_r2_mean",
    "reachability_rate",
)

# Scalar keys tried inside a dict-valued ``value`` block, most specific first.
_VALUE_DICT_KEYS = (
    "mean",
    "mean_A",  # com_shift_* / rigid_fragment_rmsd_*: mean in Angstrom
    "rate",
    "score_kcal_mol",
    "median",
    "value",
    "auc_mean",
    "r2_mean",
    "pearson_r",
    "accuracy",
    "f1",
)


def _extract_value(payload: Dict[str, Any], value_key: Optional[str] = None) -> Any:
    """Pull the headline number from an arbitrary metric JSON.

    ``value_key`` (from METRIC_TARGETS) may be a dotted path like
    ``aggregate.best_measured_real``; it wins when present and finite.
    Otherwise fall through the common top-level + nested keys so the registry
    value tracks the real number rather than a misleading None.
    """
    if value_key:
        node: Any = payload
        for part in value_key.split("."):
            if isinstance(node, dict):
                node = node.get(part)
            else:
                node = None
                break
        if _coerce_float(node) is not None:
            return node
    for key in _VALUE_KEYS:
        v = payload.get(key)
        if _coerce_float(v) is not None:
            return v
    # nested aggregate.value / aggregate.best_measured
    agg = payload.get("aggregate")
    if isinstance(agg, dict):
        for key in ("value", "best_measured", "mean_r2", "reachability_rate", "cisplatin_mean"):
            v = agg.get(key)
            if _coerce_float(v) is not None:
                return v
    raw = payload.get("value")
    if isinstance(raw, dict):
        # Some files nest their headline inside ``value`` — e.g.
        # vina_high_affinity.value.rate, vina_reference.value.score_kcal_mol.
        # Returning the dict made the registry's ``value`` column non-numeric
        # for 19 rows, which no consumer can compare against a target.
        for key in _VALUE_DICT_KEYS:
            v = raw.get(key)
            if _coerce_float(v) is not None:
                return v
        return None
    return raw


def gap_table(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Compute the current-vs-target gap per paper line from the registry.

    The gap is DERIVED (current value - target, read from metrics.yaml), not
    hand-maintained, so it cannot drift from the metric it describes.  Returns
    {line: {"metrics": [...], "at_target": N, "total": M}}.
    """
    reg = _load_yaml(METRICS_PATH)
    metrics = reg.get("metrics") or {}
    out: Dict[str, Any] = {}
    for line, mapping in LINE_METRICS.items():
        rows = []
        at_target = 0
        for metric_id in mapping:
            rec = metrics.get(metric_id) or {}
            cur = _coerce_float(rec.get("value"))
            target = _coerce_float(rec.get("target"))
            direction = rec.get("direction")
            gap: Optional[float] = None
            hit = None
            if cur is not None and target is not None and direction:
                if direction == ">=":
                    gap = round(cur - target, 4)
                    hit = cur >= target
                elif direction == "<=":
                    gap = round(target - cur, 4)
                    hit = cur <= target
            if hit:
                at_target += 1
            rows.append(
                {
                    "metric": metric_id,
                    "value": rec.get("value"),
                    "target": rec.get("target"),
                    "gap": gap,
                    "verdict": rec.get("verdict"),
                    "remains": rec.get("remains"),
                }
            )
        out[line] = {"metrics": rows, "at_target": at_target, "total": len(rows)}
    return out


# ---------------------------------------------------------------------------
# Generate (writes registry/*.yaml via atomic write per AGENTS.md §5.6)
# ---------------------------------------------------------------------------


def generate(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Build + write the three registry files. Returns a coverage report."""
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    datasets = build_datasets(repo_root)
    protocols = build_protocols(repo_root)
    metrics, covered = build_metrics(repo_root)

    all_metric_files = {p.stem for p in BY_METRIC_DIR.glob("*.json")}
    covered_stems = {Path(s).stem for s in covered}
    uncovered = sorted(all_metric_files - covered_stems)

    datasets_doc = {
        "schema": "molmetal-registry-datasets-v1",
        "generated_at": _now_utc(),
        "generator": "python -m molmetal.scripts.registry generate",
        "note": "Machine-readable dataset facts. A dataset is READY iff sha256(path) == sha256 here.",
        "datasets": datasets,
    }
    protocols_doc = {
        "schema": "molmetal-registry-protocols-v1",
        "generated_at": _now_utc(),
        "generator": "python -m molmetal.scripts.registry generate",
        "note": "Machine-readable protocol facts. A protocol is LOCKED iff protocol_lock_sha256 still matches disk.",
        "protocols": protocols,
    }
    metrics_doc = {
        "schema": "molmetal-registry-metrics-v1",
        "generated_at": _now_utc(),
        "generator": "python -m molmetal.scripts.registry generate",
        "note": "Machine-readable metric facts. A metric is CURRENT iff source_json_sha256 still matches disk.",
        "lines": list(LINE_METRICS.keys()),
        "metrics": metrics,
        "uncovered_metric_files": uncovered,
    }

    for path, doc in (
        (DATASETS_PATH, datasets_doc),
        (PROTOCOLS_PATH, protocols_doc),
        (METRICS_PATH, metrics_doc),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
        # Skip a rewrite that would change nothing but the timestamp header, so a
        # read-only caller (e.g. the test suite) does not dirty the tracked file.
        if path.is_file() and _strip_generated_at(path.read_text(encoding="utf-8")) == _strip_generated_at(text):
            continue
        atomic_write_file(path, text)

    return {
        "datasets": len(datasets),
        "protocols": len(protocols),
        "metrics": len(metrics),
        "uncovered_metric_files": len(uncovered),
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "check"
    if cmd == "generate":
        rep = generate()
        print(f"registry generated: {rep}")
        return 0
    if cmd == "check":
        results = check_all()
        worst = "PASS"
        for layer in ("datasets", "protocols", "metrics"):
            r = results[layer]
            print(f"[{r.status}] {layer}: {r.evidence}")
            for c in r.checks:
                if c["status"] != "PASS":
                    print(f"    {c['status']:5s} {c['id']}: {c['detail']}")
            if r.status == "FAIL":
                worst = "FAIL"
            elif r.status == "STALE" and worst == "PASS":
                worst = "STALE"
        print(f"REGISTRY: {worst}")
        return 0 if worst == "PASS" else 1
    if cmd == "gaps":
        table = gap_table()
        for line, info in table.items():
            print(f"== {line}: {info['at_target']}/{info['total']} at target ==")
            for row in info["metrics"]:
                gap = row["gap"]
                gap_s = "n/a" if gap is None else f"{gap:+.4f}"
                print(
                    f"   {row['metric']:20s} value={str(row['value'])[:10]:>10s} "
                    f"target={str(row['target']):>6s} gap={gap_s:>9s} "
                    f"{row['verdict']}"
                )
        return 0
    print(f"unknown command {cmd!r}; use 'generate', 'check' or 'gaps'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
