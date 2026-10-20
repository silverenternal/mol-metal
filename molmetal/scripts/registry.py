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
import re
import subprocess
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


_TOP_LEVEL_DATE_KEYS = ("timestamp", "date", "date_utc")

# A key is a date claim when its name says so.  Searching *values* for a date
# instead would pick up citation years and target dates out of prose — 119 of
# the 142 undated files contain a date somewhere, but only 23 contain one under
# a key that claims to be a date.
_DATE_KEY_RE = re.compile(r"(^|_)(date|timestamp|time|at)($|_)", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _walk_date_keys(obj: Any, path: str = ""):
    """Yield ``(iso_date, json_path)`` for date-named keys holding an ISO date."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if isinstance(v, str) and _DATE_KEY_RE.search(str(k)):
                m = _ISO_DATE_RE.search(v)
                if m:
                    yield m.group(0), p
            yield from _walk_date_keys(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_date_keys(v, f"{path}[{i}]")


def _metric_timestamp(payload: Dict[str, Any]) -> tuple:
    """(date, where it came from) for a metric payload.

    ``timestamp`` (105 files), ``date`` (60) and ``date_utc`` (3) all mean the
    same thing at the top level; reading only the first left 63 dated files
    with a null timestamp.  When no top-level key exists, the latest date under
    a *date-named* key is used and its JSON path is recorded, so the value is
    checkable rather than inferred.
    """
    for key in _TOP_LEVEL_DATE_KEYS:
        v = payload.get(key)
        if v:
            return v, key
    hits = sorted(_walk_date_keys(payload), key=lambda h: h[0])
    if hits:
        date, path = hits[-1]
        return date, f"nested:{path}"
    return None, None


def _vcs_added_dates(repo_root: Path = REPO_ROOT) -> Dict[str, str]:
    """repo-relative metric path -> the date it first entered mainline history.

    This is a PROV ``generatedAtTime`` for the *file entity*, not the date the
    measurement was made (Git2PROV maps commit time to the activity that
    produced the entity).  The registry keeps the two apart: ``metric_timestamp``
    is the measurement date, ``date_added_to_repo`` is this one.

    ``--first-parent`` rather than ``--diff-filter=A``: 20 metric files reached
    mainline through a merge commit, and ``git log --diff-filter=A`` shows no
    file changes for a merge, so those files looked like they had never been
    committed.  The oldest commit on the first-parent chain that names the path
    is the one that introduced it.
    """
    try:
        out = subprocess.run(
            [
                "git", "log", "--first-parent", "--name-only", "--format=%cs",
                "--", "metrics/by_metric",
            ],
            cwd=str(repo_root),
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.CalledProcessError):
        return {}
    seen: Dict[str, str] = {}
    current: Optional[str] = None
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", line):
            current = line
        elif current:
            # git log is newest-first; overwriting leaves the OLDEST add event.
            seen[line] = current
    return seen


def _n_seeds(payload: Dict[str, Any]) -> Optional[float]:
    v = payload.get("n_seeds")
    return _coerce_float(v) if _coerce_float(v) is not None else None


# ---------------------------------------------------------------------------
# Citability rule — see docs/METRIC_CITABILITY_RULE.md
#
# Anchored on the ML Reproducibility Checklist v2.0 (Pineau et al., 2020):
# for every reported result, state "the exact number of training and
# evaluation runs" and give "central tendency (e.g. mean) & variation (e.g.
# error bars)".  A std of a single run is not variation — R's ``sd()`` returns
# NA for a length-one vector, and so does this rule.
# ---------------------------------------------------------------------------

# Status prefixes that describe something other than a measurement of this
# metric: a projection, a blocked run, a synthetic stand-in, a placeholder.
_NON_QUOTABLE_STATUS_PREFIXES = (
    "PROJECTED",
    "DEFERRED",
    "BLOCKED",
    "REINDEX_PENDING",
    "SUPERSEDED",
    "UNMEASURED",
    "DESIGN",
    "CHECKPOINT",
    "WIRED",
    "PARTIAL",
    "MEASURED_SYNTHETIC_LABELS",
    "MEASURED_PILOT_MOCK",
)

# Statuses that ARE a measurement but whose own text says it must not be read
# as a result.  decode_ratio = 0.998 means the decoder almost never abstains —
# it is not evidence of quality; m7 IntDiv1 is a property of the generated
# cohort, not a claim about the method.  Quotable with a caveat, never a table
# headline.
_NON_HEADLINE_STATUSES = (
    "MEASURED_NOT_A_CLAIM",
    "MEASURED_BUT_NOT_A_QUALITY_METRIC",
)

# A value averaged over real + synthetic folds is not a measurement of the real
# task.  m19 is exactly this: real_only_fold = 0.381, synthetic fold = 0.880,
# headline 0.5575 — the mean of two things that are not the same task.
_SYNTHETIC_VALUE_FLAGS = ("value_is_half_synthetic", "value_is_synthetic")

_DATASET_KEYS = ("dataset", "data_source", "cohort", "cohorts_tested")

# A file can carry its status under a different name.  Reading only `status`
# counted 44 artifacts as unlabelled whose `tag` is literally "MEASURED".
_STATUS_ALIAS_KEYS = ("verdict", "honest_framing", "tag")

# ...but only a *token* is a status.  `verdict` is also used for narrative
# findings ("THE HEAD IS NOT THE RING DEFECT. On clean geometry ..."), and
# copying that into `status` would put a sentence in a column that is supposed
# to hold one word from a controlled vocabulary -- a claim of provenance that
# the value does not support.  Narrative verdicts are surfaced separately.
#
# The vocabulary is a prefix list, not an enumeration: MEASURED_HONEST_NEGATIVE,
# MEASURED_NOT_A_CLAIM, PARTIAL_LIFT_v5, DESIGN+1-PILOT and
# DEFERRED_receptors_missing are all legitimate and all share a stem.  It is
# consulted only when a token has to be *cut out* of a longer string, where the
# risk is grabbing the first word of a sentence ("THE HEAD IS NOT ...").
_STATUS_VOCAB_PREFIXES = (
    "MEASURED", "PROJECTED", "DEFERRED", "BLOCKED", "SUPERSEDED", "UNMEASURED",
    "DESIGN", "CHECKPOINT", "WIRED", "PARTIAL", "VERIFIED", "EXPECTED",
    "INCOMPLETE", "NOT_A", "PROVENANCE", "RETRACTED", "FAILED", "PASS",
    # outcome words the artifact files use as their only declaration
    "POSITIVE", "NEGATIVE", "KILL", "STABLE", "TBD", "ROOT_CAUSE", "SPRINT",
    "PRODUCTION", "PENDING", "RESOLVED", "CONFIRMED", "REFUTED", "MIXED",
    "INCONCLUSIVE",
)


def _is_status_token(value: Any) -> bool:
    """True for a single-word status: no whitespace, starts with a capital.

    Structural rather than vocabulary-based, because the project's artifact
    verdicts include KILL / POSITIVE / NEGATIVE / STABLE / TBD alongside the
    MEASURED_* family, and a closed list would keep rejecting legitimate ones.
    """
    if not isinstance(value, str):
        return False
    s = value.strip()
    return 3 <= len(s) <= 40 and s[0].isupper() and not any(c.isspace() for c in s)


def _is_vocab_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper().startswith(
        _STATUS_VOCAB_PREFIXES
    )


# ---------------------------------------------------------------------------
# Supersession.  A metric file may say it replaces, or was replaced by, another
# one -- PROV-O's prov:wasRevisionOf, and the practice a governed semantic layer
# calls "version history: see what the metric meant last quarter".  The claim
# has to resolve: `dmpnn_within_metal_ir_cap512_pf` declared
# `superseded_by: metrics/by_metric/dmpnn_within_metal_cap512_pf.json`, a file
# that has never existed, so nothing said the old number was dead.
# ---------------------------------------------------------------------------

_SUPERSESSION_KEYS = ("supersedes", "superseded_by")


def _resolve_metric_ref(ref: Any) -> Optional[str]:
    """Normalise a path or a bare id to a ``metrics/by_metric`` stem.

    Returns None for anything that is not a single path-like token -- prose and
    dicts are not pointers, and treating them as pointers is how a field ends
    up asserting a relationship that cannot be checked.
    """
    if not isinstance(ref, str):
        return None
    s = ref.strip()
    if not s or any(c.isspace() for c in s):
        return None
    if s.endswith(".json"):
        s = s[: -len(".json")]
    s = s.rsplit("/", 1)[-1]
    return s or None


def _ref_exists(ref: Optional[str]) -> bool:
    return bool(ref) and (BY_METRIC_DIR / f"{ref}.json").is_file()


def _supersession_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Row columns for a supersession declaration, with the pointer resolved.

    ``supersedes_dangling`` is the column that matters: it says the file claims
    to replace something that is not there, which is a claim no reader can
    follow and no gate would otherwise catch.
    """
    out: Dict[str, Any] = {}
    sup_by = _resolve_metric_ref(payload.get("superseded_by"))
    if sup_by:
        out["superseded_by"] = sup_by
        out["superseded_by_resolved"] = _ref_exists(sup_by)
    sup = _resolve_metric_ref(payload.get("supersedes"))
    if sup:
        out["supersedes"] = sup
        out["supersedes_dangling"] = not _ref_exists(sup)
    if isinstance(payload.get("definition_status"), str):
        out["definition_status"] = payload["definition_status"].strip()
    return out

# Some verdicts lead with the token and then explain it: "MEASURED — 0.7 on 10
# curated SMILES", "VERIFIED_LIFT — 0.000 -> 0.1065".  The token before the
# separator is the status; the rest is the finding.  A sentence that merely
# starts with a capital word ("THE HEAD IS NOT THE RING DEFECT.") has no
# separator after that word and is correctly left as narrative.
_LEADING_STATUS_RE = re.compile(r"^([A-Z][A-Z0-9_+\-]{2,})\s*(?:[\u2014\u2013]|--|:|\()")


def _status_with_source(payload: Dict[str, Any]) -> tuple:
    """(status, status_source, narrative_alias_key).

    ``status`` is a token or None.  ``narrative_alias_key`` names a
    ``verdict``/``honest_framing``/``tag`` that carries prose instead, so the
    check can distinguish "nothing was concluded" from "a conclusion was
    written, in the wrong field".
    """
    v = payload.get("status")
    if isinstance(v, str) and v.strip():
        return v.strip(), "status", None
    narrative = None
    for k in _STATUS_ALIAS_KEYS:
        v = payload.get(k)
        if not isinstance(v, str) or not v.strip():
            continue
        token = v.strip()
        if _is_status_token(token):
            return token, k, None
        m = _LEADING_STATUS_RE.match(token)
        if m and _is_vocab_token(m.group(1)):
            return m.group(1), f"{k}:leading_token", None
        if narrative is None:
            narrative = k
    return None, None, narrative


def _tracked_paths() -> Optional[set]:
    """Repo-relative paths known to git, or None when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {p for p in out.decode("utf-8", "replace").split("\0") if p}


def _declares_dataset(payload: Dict[str, Any]) -> bool:
    return any(payload.get(k) for k in _DATASET_KEYS)


def _declares_split(payload: Dict[str, Any]) -> bool:
    if payload.get("split"):
        return True
    protocol = payload.get("protocol")
    if isinstance(protocol, str):
        low = protocol.lower()
        return "fold" in low or "split" in low
    return False


def _run_count(payload: Dict[str, Any]) -> Optional[float]:
    """The number of training/evaluation runs, however the file names it.

    The checklist asks for "the exact number of training and evaluation runs";
    files record that as n_seeds, n_runs or n_seeds_total, at the top level or
    inside ``aggregate``.
    """
    for key in ("n_seeds", "n_runs", "n_seeds_total"):
        n = _coerce_float(payload.get(key))
        if n is not None:
            return n
    seeds = payload.get("seeds")
    if isinstance(seeds, list) and seeds:
        return float(len(seeds))
    agg = payload.get("aggregate")
    if isinstance(agg, dict):
        for key in ("n_runs_measured", "n_seeds", "n_runs"):
            n = _coerce_float(agg.get(key))
            if n is not None:
                return n
    return None


# Only std fields that describe run-to-run variation count as an error bar.
# A bare ``std`` is usually the spread of the per-sample distribution
# (m6_kappa: 449 molecules), which is a property of the cohort, not evidence
# that a single run was repeated.  Flagging those would be a false positive.
_RUN_STD_NAME_HINTS = ("value_std", "seed", "run")


def _std_fields(payload: Dict[str, Any]) -> List[str]:
    out = []
    for k, v in payload.items():
        if _coerce_float(v) is None:
            continue
        low = k.lower()
        if "std" in low and any(h in low for h in _RUN_STD_NAME_HINTS):
            out.append(k)
    return out


def assess_citability(
    payload: Dict[str, Any],
    value: Any,
    source_rel: Optional[str] = None,
    tracked: Optional[set] = None,
) -> Dict[str, Any]:
    """Apply the citability rule to one metric payload.

    Two levels, because "is this number real?" and "is this number fit for a
    headline?" are different questions:

    ``quotable``      the number is a real measurement of this metric, with a
                      source that exists in version control.
    ``headline_grade`` quotable AND the protocol is disclosed well enough to
                      put the number in a table: run count, variation when
                      n > 1, evaluation dataset and split.

    Returns a dict with both booleans and the reasons each failed.
    """
    not_quotable: List[str] = []
    status = (_status_with_source(payload)[0] or "").strip()
    if not status:
        not_quotable.append("no status")
    elif not _is_status_token(status):
        # An explicit `status` is authoritative, but a paragraph there is not a
        # status: no prefix rule can read it, so it cannot be classified.
        not_quotable.append(
            f"status {status[:48]!r} is prose, not a status token"
        )
    elif status.upper().startswith(_NON_QUOTABLE_STATUS_PREFIXES):
        not_quotable.append(
            f"status {status.split()[0]!r} is not a measurement of this metric"
        )
    # A superseded measurement is not a current claim, whatever its `status`
    # says.  Without this, `status: MEASURED` + `superseded_by: ...` stayed
    # quotable and the supersession was decorative.
    superseded_by = _resolve_metric_ref(payload.get("superseded_by"))
    if superseded_by:
        not_quotable.append(
            f"superseded by {superseded_by}"
            + ("" if _ref_exists(superseded_by) else " (which does not exist)")
        )
    if value is None:
        declared = payload.get("no_headline_reason")
        if isinstance(declared, str) and declared:
            not_quotable.append(f"no scalar headline by design: {declared}")
        else:
            not_quotable.append("no scalar headline")
    if payload.get("cisplatin_reference_mean_is_control") or payload.get(
        "value_is_control"
    ):
        not_quotable.append("value is a positive control, not a model score")
    for flag in _SYNTHETIC_VALUE_FLAGS:
        if payload.get(flag):
            not_quotable.append(
                f"value mixes real and synthetic folds ({flag}); "
                "see the file's honest-framing note for the real-only value"
            )
            break
    if tracked is not None and source_rel is not None and source_rel not in tracked:
        not_quotable.append("source file is not tracked by git")

    not_headline: List[str] = []
    if status.upper().startswith(_NON_HEADLINE_STATUSES):
        not_headline.append(
            f"status {status.split()[0]!r}: the file itself declines the claim"
        )
    # A redefinition breaks comparability with everything measured before it.
    # `m7_diversity` used to be "count of wired-in CLI flags" and is now MOSES
    # IntDiv1; the old values live in the same file under supersedes.old_values.
    definition_status = payload.get("definition_status")
    if isinstance(definition_status, str) and definition_status.upper().startswith(
        ("REDEFINED", "RECONSTRUCTED", "RENAMED")
    ):
        not_headline.append(
            f"definition was changed ({definition_status.split()[0]}); "
            "values measured before the change are not comparable"
        )
    runs = _run_count(payload)
    if runs is None:
        not_headline.append("run count not declared (n_seeds/seeds/n_runs)")
    elif runs == 1 and _std_fields(payload):
        not_headline.append(
            f"std reported for a single run ({', '.join(_std_fields(payload))})"
        )
    if runs is not None and runs < 2:
        not_headline.append(
            "single run: quote as a single-run measurement, not a table mean"
        )
    if not _declares_dataset(payload):
        not_headline.append("evaluation dataset not declared")
    if not _declares_split(payload):
        not_headline.append("split not declared")

    quotable = not not_quotable
    caveat = payload.get("headline_caveat")
    return {
        "quotable": quotable,
        "headline_grade": quotable and not not_headline,
        "not_quotable": not_quotable,
        "not_headline": not_headline,
        "headline_caveat": (
            caveat.strip() if isinstance(caveat, str) and caveat.strip() else None
        ),
    }


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
    tracked = _tracked_paths()
    vcs_added = _vcs_added_dates(repo_root)
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
            status, status_source, narrative = _status_with_source(payload)
            stamp, stamp_source = _metric_timestamp(payload)
            value, value_basis = _extract_value_basis(
                payload, METRIC_TARGETS.get(metric_id, {}).get("value_key")
            )
            rec: Dict[str, Any] = {
                "source_json": _rel(src_path),
                "source_json_sha256": _sha256_file(src_path),
                "source_json_missing": not exists,
                "value": value,
                "value_basis": value_basis,
                "status": status,
                "status_source": status_source,
                "status_narrative_source": narrative,
                "verdict": payload.get("verdict"),
                "scope": payload.get("scope"),
                "metric_timestamp": stamp,
                "metric_timestamp_source": stamp_source,
                "registry_role": "paper_line",
                "last_refreshed": _now_utc(),
            }
            added = vcs_added.get(_rel(src_path))
            if added:
                rec["date_added_to_repo"] = added
            seeds = _n_seeds(payload)
            if seeds is not None:
                rec["n_seeds"] = seeds
            # A MEASURED label with no scalar is invisible to any target
            # comparison.  Flag it rather than letting `value: null` read as
            # "no data" when the file claims to have measured something.  A
            # file that declares `no_headline_reason` has answered the
            # question, so it is not a gap.
            rec["value_missing"] = (
                rec["value"] is None
                and str(rec.get("status") or "").startswith("MEASURED")
                and not payload.get("no_headline_reason")
            )
            rec["carries_claim"] = rec["value"] is not None
            rec.update(_supersession_fields(payload))
            if rec["value_missing"]:
                rec["has_numeric_leaf"] = _has_numeric_leaf(payload)
            rec.update(METRIC_TARGETS.get(metric_id, {}))
            rec.update(
                assess_citability(payload, rec["value"], _rel(src_path), tracked)
            )
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
        status, status_source, narrative = (
            _status_with_source(payload) if isinstance(payload, dict) else (None, None, None)
        )
        stamp, stamp_source = (
            _metric_timestamp(payload) if isinstance(payload, dict) else (None, None)
        )
        value, value_basis = _extract_value_basis(
            payload if isinstance(payload, dict) else {}
        )
        rec = {
            "source_json": _rel(src_path),
            "source_json_sha256": _sha256_file(src_path),
            "source_json_missing": not exists,
            "value": value,
            "value_basis": value_basis,
            "status": status,
            "status_source": status_source,
            "status_narrative_source": narrative,
            "verdict": payload.get("verdict") if isinstance(payload, dict) else None,
            "scope": payload.get("scope") if isinstance(payload, dict) else None,
            "metric_timestamp": stamp,
            "metric_timestamp_source": stamp_source,
            "registry_role": "artifact",
            "last_refreshed": _now_utc(),
            "registry_origin": "wf_a36_3_g1_disk_walk",
        }
        added = vcs_added.get(_rel(src_path))
        if added:
            rec["date_added_to_repo"] = added
        seeds = _n_seeds(payload) if isinstance(payload, dict) else None
        if seeds is not None:
            rec["n_seeds"] = seeds
        rec["value_missing"] = (
            rec["value"] is None
            and str(rec.get("status") or "").startswith("MEASURED")
            and not (isinstance(payload, dict) and payload.get("no_headline_reason"))
        )
        rec["carries_claim"] = rec["value"] is not None
        rec.update(_supersession_fields(payload if isinstance(payload, dict) else {}))
        if rec["value_missing"]:
            rec["has_numeric_leaf"] = _has_numeric_leaf(payload)
        rec.update(
            assess_citability(
                payload if isinstance(payload, dict) else {},
                rec["value"],
                _rel(src_path),
                tracked,
            )
        )
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
    # A status qualifies a *claim*.  A probe log or a design note that carries
    # no scalar has nothing to qualify, so counting it as "unlabelled" made the
    # evidence string read like 170 broken claims when it was 12.
    unlabelled_artifact = sum(
        1
        for r in rows
        if r.get("status") is None
        and r.get("registry_role") != "paper_line"
        and r.get("carries_claim")
    )
    narrative_only = sum(
        1
        for r in rows
        if r.get("status") is None and r.get("status_narrative_source")
    )
    undated = sum(1 for r in rows if r.get("metric_timestamp") is None)
    value_missing = sum(1 for r in rows if r.get("value_missing"))
    undated_paper = sum(
        1 for r in rows if r.get("metric_timestamp") is None and r.get("registry_role") == "paper_line"
    )
    undated_artifact = undated - undated_paper
    # W3C PROV ``generatedAtTime`` for the file entity: a file with no
    # measurement date still has a commit date, which is checkable provenance.
    undated_no_vcs = sum(
        1
        for r in rows
        if r.get("metric_timestamp") is None
        and r.get("registry_role") != "paper_line"
        and not r.get("date_added_to_repo")
    )
    paper_rows = [r for r in rows if r.get("registry_role") == "paper_line"]
    value_missing_paper = sum(
        1 for r in paper_rows if r.get("value_missing")
    )
    missing_art = [r for r in rows if r.get("value_missing") and r.get("registry_role") != "paper_line"]
    missing_unstandardised = sum(1 for r in missing_art if r.get("has_numeric_leaf"))
    missing_none = len(missing_art) - missing_unstandardised
    dangling = sum(
        1
        for r in rows
        if r.get("supersedes_dangling") or r.get("superseded_by_resolved") is False
    )
    quotable = sum(1 for r in paper_rows if r.get("quotable"))
    headline_grade = sum(1 for r in paper_rows if r.get("headline_grade"))
    return CheckResult(
        "metrics",
        status,
        checks,
        f"{len(checks)-bad-stale} current, {stale} stale, {bad} missing | "
        f"{len(paper_rows)} paper-line ({quotable} quotable, "
        f"{headline_grade} headline-grade) | "
        f"{unlabelled_paper} paper-line + {unlabelled_artifact} artifact claims "
        f"without a status ({narrative_only} hold a narrative verdict instead) | "
        f"{undated_paper} paper-line + {undated_artifact} artifact rows record no "
        f"measurement date ({undated_artifact - undated_no_vcs} of the artifacts "
        f"carry a commit date instead, {undated_no_vcs} carry neither) | "
        f"{value_missing_paper} paper-line + {len(missing_art)} artifact rows "
        f"labelled MEASURED with no standardised scalar "
        f"({missing_unstandardised} hold numbers under a file-specific key, "
        f"{missing_none} hold none) | {dangling} dangling supersession pointers",
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


def _has_numeric_leaf(obj: Any) -> bool:
    """True if any non-bool number appears anywhere in the payload.

    Separates two things that ``value: null`` was conflating: a file whose
    headline simply is not at a standard key (numbers exist, the extractor
    cannot know which one is the claim) versus a decision/probe record that
    carries no numbers at all.
    """
    if isinstance(obj, bool):
        return False
    if isinstance(obj, (int, float)):
        return True
    if isinstance(obj, dict):
        return any(_has_numeric_leaf(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_numeric_leaf(v) for v in obj)
    return False


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


def _extract_value_basis(
    payload: Dict[str, Any], value_key: Optional[str] = None
) -> tuple:
    """(value, basis) — the headline number and the JSON path it came from.

    ``basis`` exists because a ``value`` column with no provenance cannot be
    audited: ``sa_mean`` read 2.2884 from ``aggregate.best_measured``, which is
    the *best* of five pockets (n=14), while the file's own five-pocket mean is
    2.4486.  The ML Reproducibility Checklist asks for central tendency, and a
    best-of is not one.  Recording where the number came from makes that
    visible instead of leaving a best case sitting in a column called ``value``.

    ``value_key`` (from METRIC_TARGETS) may be a dotted path like
    ``aggregate.best_measured_real``; it wins when present and finite.
    Otherwise a file may declare its own ``headline_key`` (same dotted-path
    form) — the escape hatch for files whose headline lives in a list of arms
    or a per-metal block, where no generic key order can know which entry is
    the headline.  Otherwise fall through the common top-level + nested keys.
    """
    declared_by = None
    if value_key is None:
        declared = payload.get("headline_key")
        if isinstance(declared, str) and declared:
            value_key = declared
            declared_by = "headline_key"
    if value_key:
        node: Any = payload
        for part in value_key.split("."):
            if isinstance(node, dict):
                node = node.get(part)
            else:
                node = None
                break
        if _coerce_float(node) is not None:
            return node, declared_by or f"value_key:{value_key}"
    for key in _VALUE_KEYS:
        v = payload.get(key)
        if _coerce_float(v) is not None:
            return v, key
    agg = payload.get("aggregate")
    if isinstance(agg, dict):
        for key in ("value", "mean", "mean_r2", "reachability_rate", "cisplatin_mean", "best_measured"):
            v = agg.get(key)
            if _coerce_float(v) is not None:
                return v, f"aggregate.{key}"
    raw = payload.get("value")
    if isinstance(raw, dict):
        # Some files nest their headline inside ``value`` — e.g.
        # vina_high_affinity.value.rate, vina_reference.value.score_kcal_mol.
        # Returning the dict made the registry's ``value`` column non-numeric
        # for 19 rows, which no consumer can compare against a target.
        for key in _VALUE_DICT_KEYS:
            v = raw.get(key)
            if _coerce_float(v) is not None:
                return v, f"value.{key}"
        return None, None
    return raw, "value" if raw is not None else None


def _extract_value(payload: Dict[str, Any], value_key: Optional[str] = None) -> Any:
    """The headline number only; see :func:`_extract_value_basis`."""
    return _extract_value_basis(payload, value_key)[0]


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


def _preserve_underived_datasets(
    datasets: Dict[str, Any], path: Path = DATASETS_PATH
) -> List[str]:
    """Carry over rows the builder cannot re-derive, instead of deleting them.

    ``build_datasets`` reads two sources (data_checksums.json, protocol_lock).
    Ten ``rusanov_2025_*`` rows were hand-added in b76ef187 and no source
    declares them, so a plain regenerate shrank the registry 21 -> 11 while
    ``check_datasets`` still reported 11/11 PASS — a smaller registry that
    looks green.  A generator must not delete what it cannot re-derive.
    """
    if not path.is_file():
        return []
    try:
        prev = (_load_yaml(path).get("datasets") or {})
    except Exception:  # noqa: BLE001 — a corrupt previous file must not block a rebuild
        return []
    kept: List[str] = []
    for did, row in prev.items():
        if did in datasets or not isinstance(row, dict):
            continue
        row = dict(row)
        row["preserved_from"] = "previous registry/datasets.yaml"
        row["preserved_reason"] = (
            "no generator source (configs/data_checksums.json or "
            "configs/locks/protocol_lock.yaml) declares this row; generate() "
            "must not delete what it cannot re-derive"
        )
        datasets[did] = row
        kept.append(did)
    return kept


# The dataset registry's field documentation lives in a comment header.  A
# yaml.safe_dump rewrite silently deleted it, so the file that explains itself
# became the one file that did not.  Emit it from here instead.
_DATASETS_HEADER = """\
# mol-metal dataset registry
#
# Each entry: {path, md5, sha256, why, source, locked_for}
# - path: relative to repo root
# - md5/sha256: file content hash (MEASURED at registration)
# - why: one-paragraph intent + caveats + join keys
# - source: yaml or pdf or zip or 'cistern' or 'manual'
# - locked_for: list of metric/protocol IDs that depend on this dataset
#
# Regenerate via: .venv/bin/python molmetal/scripts/registry.py check
# Update via: .venv/bin/python molmetal/scripts/registry.py generate
#
"""


def generate(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Build + write the three registry files. Returns a coverage report."""
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file
    datasets = build_datasets(repo_root)
    preserved = _preserve_underived_datasets(datasets)
    if preserved:
        print(
            f"registry: preserved {len(preserved)} dataset row(s) with no generator "
            f"source: {', '.join(sorted(preserved))}",
            file=sys.stderr,
        )
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
        if path == DATASETS_PATH:
            text = _DATASETS_HEADER + text
        # Skip a rewrite that would change nothing but the timestamp header, so a
        # read-only caller (e.g. the test suite) does not dirty the tracked file.
        if path.is_file() and _strip_generated_at(path.read_text(encoding="utf-8")) == _strip_generated_at(text):
            continue
        atomic_write_file(path, text)

    return {
        "datasets": len(datasets),
        "datasets_preserved": len(preserved),
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
    if cmd == "cite":
        reg = _load_yaml(METRICS_PATH)
        rows = reg.get("metrics") or {}
        shown = 0
        for mid, r in rows.items():
            if r.get("registry_role") != "paper_line":
                continue
            shown += 1
            if r.get("headline_grade"):
                mark = "CITE"
            elif r.get("quotable"):
                mark = "NUM "
            else:
                mark = "NO  "
            value = r.get("value")
            value_s = "None" if value is None else f"{value:+.4f}" if isinstance(value, float) else str(value)
            print(f"[{mark}] {mid:20s} value={value_s:>10s} target={r.get('target')}")
            caveat = r.get("headline_caveat")
            if caveat:
                print(f"        caveat: {caveat}")
            for reason in r.get("not_quotable") or []:
                print(f"        not quotable: {reason}")
            for reason in r.get("not_headline") or []:
                print(f"        not headline: {reason}")
        print(f"-- {shown} paper-line metrics; CITE = table-grade, NUM = quotable with caveats, NO = do not quote")
        print("   rule: docs/METRIC_CITABILITY_RULE.md")
        return 0
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
    print(
        f"unknown command {cmd!r}; use 'generate', 'check', 'gaps' or 'cite'",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
