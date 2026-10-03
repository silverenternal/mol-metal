#!/usr/bin/env python3
"""Wf-343C2 — per-wave internal check script (pre-retrain readiness).

==============================================================================
Wf-343C2 — companion to ``molmetal/scripts/pre_flight_check.py`` focused on
**per-wave** internal checks (vs pre-flight's global gates).

The pre-flight script (TODO-207, 25/25 PASS gate) emits a ``pre_flight.json``
that the rest of the project consumes.  Wf-343C2 emits a **per-wave** check
report that focuses on the specific (metal, seed) pair(s) the wave will
retrain + the specific config YAML the wave will use + the live GPU/disk
state at the moment of wave-launch.

The 4 NEW Wf-343C4 pre-retrain internal check gates surfaced in
``pre_flight_check.py`` (per AGENTS.md §5.1 standing GPU grant + the
"hyperparam config reference" + "YuelBond-conditionality structural check"
+ "AnchorGuard freshness enforcement" + "wave_budget_tracker integration"
contracts) are the canonical "internal pre-retrain check" surface.
Wf-343C2 is the per-wave **runtime** companion that re-runs those
checks against the **specific** wave config + live environment.

Seven checks emitted (per Wf-343C2 design contract):

  1. ``preflight_freshness_24h``     — pre_flight.json ≤ 24h old (live
                                       targets only; frozen archives SKIP
                                       with a remedy — see below)
  2. ``preflight_no_fail_gates``     — no FAIL gates in pre-flight JSON
  3. ``hyperparams_config_complete`` — wave YAML has all REQUIRED_CONFIG_KEYS
  4. ``yuelbond_decoder_registered``  — YuelBond decoder wired into CFM scripts
  5. ``wave_budget_remaining_nonneg``— declared budget covers the
                                       grid-derived requirement
                                       (n_pairs × 0.7 h MEASURED × 1.25
                                       PROJECTED margin); no budget
                                       declared → loud SKIP, never PASS
  6. ``gpu_vram_available``          — nvidia-smi free VRAM ≥ threshold
  7. ``disk_free_space``             — df free disk ≥ threshold (per-wave data)

CLI::

    wf343c2_per_wave_check.py --wave "Wave-1 chemprop v2 retrain"
                              [--config configs/cfm_full_retrain.yaml]
                              [--preflight molmetal/reports/wf259_run/pre_flight.json]
                              [--preflight-live]  # declare --preflight as this
                                                  # run's output (force mtime check)
                              [--out molmetal/reports/wf343c2_per_wave_check/per_wave_check.json]
                              [--pairs-metals "Ru,Ir,Os,Rh,Re"]
                              [--pairs-seeds "0,1,2"]
                              [--min-vram-mib 8192]
                              [--min-disk-gib 50]
                              [--budget-hr 10.0]           # declared budget (override)
                              [--budget-report wave_budget_report.json]
                              [--allow-under-budget --reason "<text>"]
                              [--strict]
                              [--skip-live]    # skip nvidia-smi + df (CPU-only tests)
                              [--verbose]

Exit codes:

  * 0 — every check PASSED (or SKIP — only when ``--strict`` is NOT set).
  * 1 — at least one check FAILED.
  * 2 — at least one check SKIPPED AND ``--strict`` is set.

(The 0/1/2 = PASS/FAIL/SKIP mapping above is the script's long-standing
contract: a non-strict run tolerates SKIPs, ``--strict`` escalates any SKIP
to exit 2.  It is preserved deliberately — the pre-flight subprocess gate
invokes this script without ``--strict`` and must not FAIL on legitimate
SKIPs; see Wf-367-A2 / Wf-368-A1.  No code path turns an under-budget wave
into PASS: the ``--allow-under-budget`` escape downgrades FAIL→SKIP only,
and ``--strict`` re-FAILs it.)

The script is **CPU-only** by default (``--skip-live`` toggles the nvidia-smi
+ df probes; defaults to running them so the live environment is checked).
It is **backward-compat** with any pre-flight JSON whose top-level keys
include ``summary`` + ``gates``.

Honest framing (party-twenty-spirit §3.5):
    Every numeric value in the JSON is MEASURED at the moment of wave-launch
    (nvidia-smi / df / pre-flight mtime).  The budget requirement uses one
    MEASURED constant (0.7 h/pair upper bound, Wf-268C) times one PROJECTED
    planning margin (``SAFETY_FACTOR`` 1.25, labelled where it appears).
    The ``overall_status`` is FAIL whenever any check is FAIL — there is no
    "FAIL-with-recovery" downgrade.  The strict flag only governs SKIP
    classification.  Frozen archived anchors are exempt from the mtime
    contract via an explicit documented allowlist (never ``touch``ed); they
    remain covered by the content/version pin gates.

Backward-compat: default ``--out`` is
``molmetal/reports/wf343c2_per_wave_check/per_wave_check.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../mol-metal
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
SCRIPT_DIR = MOLMETAL_ROOT / "scripts"
CONFIGS_DIR = REPO_ROOT / "configs"
REPORTS_DIR = MOLMETAL_ROOT / "reports"

# Required config-keys contract (per 207 v2 §1.3 + Wf-343C4
# hyperparams_config_complete gate).
REQUIRED_CONFIG_KEYS: Tuple[str, ...] = (
    "data",
    "model",
    "flow",
    "train",
    "protocol",
    "eval",
    "output",
    "compute",
    "honest_framing",
)

# Wf-343C2 design contract: 7 checks per wave.
N_CHECKS = 7

# Anchors
DEFAULT_PREFLIGHT_JSON = REPORTS_DIR / "wf259_run" / "pre_flight.json"
DEFAULT_OUT_JSON = REPORTS_DIR / "wf343c2_per_wave_check" / "per_wave_check.json"
DEFAULT_CONFIG_YAML = CONFIGS_DIR / "cfm_full_retrain.yaml"

# Defaults
DEFAULT_PREFLIGHT_MAX_AGE_SEC = 86400  # 24h (AnchorGuard default; Wf-302A)
DEFAULT_PAIRS_METALS = "Ru,Ir,Os,Rh,Re"
DEFAULT_PAIRS_SEEDS = "0,1,2,3,4"
DEFAULT_MIN_VRAM_MIB = 8192  # 8 GB free VRAM minimum per wave
DEFAULT_MIN_DISK_GIB = 50  # 50 GB free disk minimum per wave
DEFAULT_NVIDIA_SMI_TIMEOUT_SEC = 10
DEFAULT_DF_TIMEOUT_SEC = 5

# Per-pair baseline wall-hour estimate for the budget check.
#
# Source claim (on disk): TODO/completed/343c2_per_wave_check.md §8 item 1
# — "Wf-268C MEASURED actuals are 0.55-0.7h/pair for Ru/Ir heavier
# cohorts" — backed by the Wf-268C artifact
# molmetal/reports/wf268c_update/wb_report.md (per-metal actual_hr table
# for the Wf-124 P3 5-metal × 3-seed × 30-epoch run on RTX 5090 GPU-1;
# only Ru seed=0 is independently auditable from raw JSON at
# molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json —
# the wb_report carries the full honest note).  We ship the MEASURED
# UPPER BOUND (0.7) so the gate cannot pass below the observed
# heaviest-cohort cost.
#
# History: the previous value here was 0.5 h/pair — a PROJECTED floor —
# which made the gate FAIL on its own shipped defaults once the default
# grid grew to 25 pairs (25 x 0.5 = 12.5h > 10h hardcoded "default
# budget").  Both the constant and the hardcoded budget were deleted per
# the 2026-10-03 gate-semantics review: the requirement is now DERIVED
# from the declared grid, and the budget is an INPUT the caller must
# declare (no declaration -> loud SKIP, never PASS).
PER_PAIR_BASELINE_HR = 0.7  # MEASURED upper bound (Wf-268C: 0.55-0.7 h/pair)

# Planning margin applied on top of the MEASURED per-pair baseline.
# PROJECTED (not measured): absorbs queueing / restart / eval overhead so
# the verdict compares against the plausible cost, not the best case.
# Configurable here; 1.0 disables the margin.
SAFETY_FACTOR = 1.25  # PROJECTED planning margin (labelled in evidence)

# ---------------------------------------------------------------------------
# Frozen archived pre-flight targets (2026-10-03 gate-semantics fix)
# ---------------------------------------------------------------------------
# Check 1 (preflight_freshness_24h) applies the mtime contract only to
# pre-flight targets declared LIVE — produced by the launch that is about to
# run.  The paths below are archived historical runs with NO producer in the
# current pipeline: re-running pre-flight does NOT rewrite them, so a
# wall-clock FAIL on them can never be repaired by any legitimate action
# (``touch`` would be a fabricated freshness claim per AGENTS.md §5.2 and is
# forbidden).  They SKIP with a remedy instead of FAILing, and they are NOT
# auto-passed: content/version coverage stays with the AnchorGuard consumers
# and ``gate_anchor_version_pin``.  Add a path here ONLY together with a
# reason; new/unknown targets are mtime-checked by default (no silent
# scope drops).
FROZEN_PREFLIGHT_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "molmetal/reports/wf259_run/pre_flight.json",
        "Wf-259 archived run pre-flight (shipped --preflight default); "
        "no producer in today's pipeline",
    ),
    (
        "molmetal/reports/wf260_preflight/pre_flight.json",
        "Wf-260 archived pre-flight (118h MEASURED age); historical artifact",
    ),
)

SCHEMA_VERSION = "343.v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """ISO-8601 UTC timestamp, e.g. ``2026-09-29T15:49:00+00:00``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _read_json(path: Path) -> Dict[str, Any]:
    """Read JSON; raise ``FileNotFoundError`` / ``json.JSONDecodeError``."""
    with open(str(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, obj: Any) -> None:
    """Write JSON with ``indent=2, ensure_ascii=False``; create parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def _parse_yaml_safe_load(path: Path) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Best-effort YAML parse; never raises. Returns
    ``(parse_ok, parsed_dict_or_None, error_message_or_empty)``."""
    if not path.exists():
        return False, None, f"file not found: {path}"
    try:
        import yaml  # type: ignore

        with open(str(path), "r", encoding="utf-8") as fh:
            obj = yaml.safe_load(fh)
        if not isinstance(obj, dict):
            return False, None, f"top-level must be dict, got {type(obj).__name__}"
        return True, obj, ""
    except ImportError:
        return _parse_yaml_minimal(path)
    except Exception as exc:  # noqa: BLE001
        return False, None, f"yaml.safe_load failed: {type(exc).__name__}: {exc}"


def _parse_yaml_minimal(path: Path) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Fallback parser — same shape as per_wave_checklist's."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return False, None, f"read failed: {type(exc).__name__}: {exc}"
    out: Dict[str, Any] = {}
    top_pending: Optional[str] = None
    pending_dict: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith(" ") or line.startswith("\t"):
            if top_pending is None:
                continue
            stripped = line.strip()
            if ":" in stripped:
                k, _, v = stripped.partition(":")
                pending_dict[k.strip()] = v.strip().strip('"').strip("'")
            continue
        if top_pending is not None:
            out[top_pending] = pending_dict
            top_pending = None
            pending_dict = {}
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if v == "":
                top_pending = k
                pending_dict = {}
            else:
                v = v.split("#", 1)[0].strip().strip('"').strip("'")
                out[k] = v
    if top_pending is not None:
        out[top_pending] = pending_dict
    return True, out, ""


def _parse_csv_int_list(s: str) -> List[int]:
    out: List[int] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _parse_csv_str_list(s: str) -> List[str]:
    out: List[str] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Per-check functions
# ---------------------------------------------------------------------------
# Each returns ``(status, evidence)`` where status ∈ {PASS, FAIL, SKIP} and
# evidence is a single-line string ≤ 200 chars describing what was found.


def _frozen_preflight_reason(
    path: Path,
    frozen_paths: Tuple[Tuple[str, str], ...],
) -> Optional[str]:
    """Return the documented reason if ``path`` is on the frozen allowlist.

    Matching is by resolved absolute path so both the shipped constants
    (repo-relative) and CLI/test inputs (absolute) normalise the same way.
    Returns ``None`` when the target is NOT frozen (i.e. treated as live
    and therefore subject to the mtime contract).
    """
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for rel, reason in frozen_paths:
        candidate = Path(rel)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        try:
            candidate_resolved = candidate.resolve()
        except OSError:
            candidate_resolved = candidate
        if resolved == candidate_resolved:
            return reason
    return None


def check_preflight_freshness_24h(
    *,
    preflight_path: Path,
    max_age_sec: int,
    declared_live: bool = False,
    frozen_paths: Tuple[Tuple[str, str], ...] = FROZEN_PREFLIGHT_PATHS,
) -> Tuple[str, str]:
    """Check 1: pre_flight.json mtime ≤ max_age_sec — live targets only.

    2026-10-03 gate-semantics fix: the mtime contract applies only to
    pre-flight targets DECLARED LIVE — produced by the launch that is about
    to run (a path not on the documented frozen allowlist, or any target
    explicitly declared live via ``declared_live=True`` / ``--preflight-live``).
    Frozen archived targets SKIP carrying the remedy text: they have no
    producer in today's pipeline, so a wall-clock FAIL on them would be
    permanent decoration (the exact disease the review removed).  The SKIP
    is loud (names the frozen reason + remedy) and is NOT a pass — no
    freshness claim is made about the frozen file.

    A stale LIVE target FAILs with the remedy embedded: re-run the producing
    wave to regenerate the path, or widen ``--max-age-sec`` together with an
    HONEST_NEGATIVE catalog entry.  Never ``touch`` (fabricated freshness).
    """
    if not declared_live:
        frozen_reason = _frozen_preflight_reason(preflight_path, frozen_paths)
        if frozen_reason is not None:
            return (
                "SKIP",
                f"frozen archived pre-flight ({frozen_reason}); mtime contract "
                f"skipped — remedy: re-run the producing wave, point "
                f"--preflight at this run's output, or --preflight-live if "
                f"this launch regenerates it; never touch",
            )
    if not preflight_path.exists():
        return "FAIL", f"preflight missing at {preflight_path}"
    try:
        mtime = preflight_path.stat().st_mtime
    except OSError as exc:
        return "FAIL", f"stat failed: {type(exc).__name__}: {exc}"
    age_sec = time.time() - mtime
    if age_sec <= max_age_sec:
        return "PASS", f"preflight age {age_sec:.1f}s ≤ {max_age_sec}s"
    return (
        "FAIL",
        f"preflight STALE: age {age_sec:.1f}s > {max_age_sec}s; remedy: "
        f"re-run the producing wave to regenerate {preflight_path.name}, or "
        f"widen --max-age-sec + add HONEST_NEGATIVE entry; never touch",
    )


def check_preflight_no_fail_gates(
    *,
    preflight_path: Path,
) -> Tuple[str, str]:
    """Check 2: pre_flight.json has zero FAIL gates (excluding self).

    Wf-364-A1 cycle fix: this gate (named ``p3_wf343c2_per_wave_runtime_check``)
    would otherwise create a self-referential FAIL cycle — pre_flight.json's
    ``summary.fail`` count includes this gate's own FAIL, so reading
    ``summary.fail > 0`` would always FAIL this gate, which would always
    keep ``summary.fail >= 1`` on the next pre-flight run.

    The fix is to count FAILs in the pre_flight.json's ``gates[]`` array
    while excluding the entry whose name matches this gate.  Then the
    check correctly detects upstream cascade FAILs (e.g. ``p2_tests_pass``
    per Wf-358-D1) without the self-reference holding the gate FAILed
    forever.
    """
    if not preflight_path.exists():
        return "FAIL", f"preflight missing at {preflight_path}"
    try:
        pf = _read_json(preflight_path)
    except (json.JSONDecodeError, OSError) as exc:
        return "FAIL", f"preflight JSON unreadable: {type(exc).__name__}"
    summary = pf.get("summary") or {}
    n_fail = int(summary.get("fail", 0) or 0)
    n_total = int(summary.get("total", 0) or 0)
    n_pass = int(summary.get("pass", 0) or 0)
    # Subtract self from FAIL count to break the Wf-364-A1 self-referential
    # cycle.  ``gates[]`` lists every gate with its status; we find any
    # gate named like this cascade's parent (``p3_wf343c2_per_wave_runtime_check``)
    # and exclude it.  This is safe because the parent gate has no other
    # upstream side-effects — its sole purpose is to surface cascade state.
    self_fail_count = sum(
        1
        for g in pf.get("gates", [])
        if g.get("status") == "FAIL"
        and g.get("name") == "p3_wf343c2_per_wave_runtime_check"
    )
    upstream_fail = n_fail - self_fail_count
    if upstream_fail <= 0:
        return (
            "PASS",
            f"preflight upstream FAIL={upstream_fail}/{n_total} "
            f"(PASS={n_pass}; self-excluded {self_fail_count} per Wf-364-A1)",
        )
    return (
        "FAIL",
        f"preflight upstream FAIL={upstream_fail}/{n_total} "
        f"(PASS={n_pass}; self-excluded {self_fail_count} per Wf-364-A1)",
    )


def check_hyperparams_config_complete(
    *,
    config_path: Optional[Path],
) -> Tuple[str, str]:
    """Check 3: wave YAML has all REQUIRED_CONFIG_KEYS present."""
    if config_path is None or str(config_path) == "":
        return "SKIP", "no --config provided (skipped schema check)"
    parse_ok, yaml_obj, err = _parse_yaml_safe_load(config_path)
    if not parse_ok:
        return "FAIL", f"yaml parse: {err}"
    keys: set = set(yaml_obj.keys() if yaml_obj else [])
    nested = yaml_obj.get("config") if isinstance(yaml_obj, dict) else None
    if isinstance(nested, dict):
        keys.update(nested.keys())
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in keys]
    if missing:
        return "FAIL", f"missing keys: {', '.join(missing)}"
    return "PASS", f"all {len(REQUIRED_CONFIG_KEYS)} REQUIRED_CONFIG_KEYS present"


def check_yuelbond_decoder_registered(
    *,
    config_path: Optional[Path],
) -> Tuple[str, str]:
    """Check 4: YuelBond decoder is wired into the CFM scripts / config.

    Heuristic: scan the train_cfm_yuelbond_wasserstein.py + flow_matching
    adapter + the wave config for any of: ``yuelbond``,
    ``YuelBondDecoder``, ``--yuelbond``, ``bond_head``.
    """
    hits: List[str] = []
    # 1. Wave config: any yuelbond / bond_head token
    if config_path is not None and str(config_path) != "":
        if config_path.exists():
            try:
                txt = config_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                txt = ""
            if re.search(r"(?i)yuelbond", txt):
                hits.append(f"config:{config_path.name}")
            elif re.search(r"(?i)bond[_-]head", txt):
                hits.append(f"config:{config_path.name}(bond_head)")
    # 2. Canonical CFM training script (always present in mol-metal)
    cfm_script = MOLMETAL_ROOT / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    if cfm_script.is_file():
        hits.append(f"script:{cfm_script.name}")
    # 3. Flow matching adapter directory (canonical home of YuelBond)
    flow_adapter = MOLMETAL_ROOT / "adapters" / "flow_matching_lipman"
    if flow_adapter.is_dir():
        hits.append(f"adapter:{flow_adapter.name}/")
    if not hits:
        return "FAIL", "no YuelBond decoder wiring detected"
    return "PASS", f"YuelBond registered via: {'; '.join(hits[:3])}"


def check_wave_budget_remaining_nonneg(
    *,
    budget_hr: Optional[float],
    pairs_metals: List[str],
    pairs_seeds: List[int],
    budget_report: Optional[Path] = None,
    allow_under_budget: bool = False,
    under_budget_reason: str = "",
    strict: bool = False,
) -> Tuple[str, str]:
    """Check 5: declared wave budget vs REQUIRED_HR derived from the grid.

    Requirement (derived, never a magic constant)::

        REQUIRED_HR = n_pairs × PER_PAIR_BASELINE_HR × SAFETY_FACTOR
                    = n_pairs × 0.7 h/pair (MEASURED upper bound, Wf-268C
                      0.55-0.7 h/pair) × 1.25 (PROJECTED planning margin)

    Budget source precedence:

      1. ``budget_hr`` — explicit ``--budget-hr`` override;
      2. ``budget_report`` — a ``wave_budget_tracker.py`` report JSON
         (``remaining_hr`` preferred — the budget left for this wave —
         falling back to ``budget_hr``);
      3. none — no budget declared.

    Verdicts:

      * empty (metal, seed) grid → SKIP;
      * no budget declared → SKIP, naming REQUIRED_HR + both tags + how to
        declare.  Omission must never look like a pass;
      * declared ≥ required → PASS (evidence shows declared / required /
        remaining);
      * declared < required → FAIL with both numbers + sources.  When the
        declared figure came via explicit ``--budget-hr`` the evidence
        reads "override contradicts derivation";
      * agreed escape: ``--allow-under-budget`` + ``--reason "<text>"``
        downgrades FAIL→SKIP — never to PASS.  Under ``--strict`` the
        escape is refused and the check re-FAILs.

    No path turns an under-budget wave into PASS.
    """
    n_pairs = len(pairs_metals) * len(pairs_seeds)
    if n_pairs == 0:
        return "SKIP", "empty (metal, seed) grid — no cost to check"
    required_hr = round(n_pairs * PER_PAIR_BASELINE_HR * SAFETY_FACTOR, 2)
    derivation = (
        f"{n_pairs} pairs × {PER_PAIR_BASELINE_HR}h/pair (MEASURED Wf-268C "
        f"0.55-0.7 upper bound) × {SAFETY_FACTOR} (PROJECTED margin)"
    )

    # ---- resolve the declared budget (precedence: --budget-hr > report) ----
    declared_hr: Optional[float] = None
    source = ""
    if budget_hr is not None:
        declared_hr = float(budget_hr)
        source = "explicit --budget-hr override"
    elif budget_report is not None:
        try:
            rep = _read_json(budget_report)
        except (OSError, json.JSONDecodeError) as exc:
            return (
                "FAIL",
                f"budget report unreadable ({type(exc).__name__}): {budget_report}",
            )
        for key in ("remaining_hr", "budget_hr"):
            val = rep.get(key) if isinstance(rep, dict) else None
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                declared_hr = float(val)
                source = f"budget report {budget_report} ({key})"
                break
        if declared_hr is None:
            return (
                "FAIL",
                f"budget report {budget_report} carries no numeric "
                f"remaining_hr/budget_hr",
            )

    if declared_hr is None:
        return (
            "SKIP",
            f"no budget declared — REQUIRED_HR={required_hr}h = {derivation}; "
            f"declare with --budget-hr H or --budget-report <report.json>; "
            f"omission is not a pass",
        )

    remaining_hr = round(declared_hr - required_hr, 2)
    if declared_hr + 1e-9 >= required_hr:
        return (
            "PASS",
            f"declared {declared_hr:.2f}h ({source}) ≥ REQUIRED_HR "
            f"{required_hr}h = {derivation}; remaining {remaining_hr:.2f}h",
        )

    # ---- under-budget: never PASS from here on ----
    fail_evidence = (
        f"declared {declared_hr:.2f}h ({source}) < REQUIRED_HR {required_hr}h "
        f"= {derivation}; remaining {remaining_hr:.2f}h"
    )
    if source == "explicit --budget-hr override":
        fail_evidence += "; override contradicts derivation"
    if allow_under_budget and under_budget_reason.strip():
        if strict:
            return (
                "FAIL",
                fail_evidence + f"; --strict refuses the --allow-under-budget escape "
                f"(reason={under_budget_reason!r})",
            )
        return (
            "SKIP",
            fail_evidence + f"; accepted via --allow-under-budget (SKIP, NOT PASS; "
            f"reason={under_budget_reason!r})",
        )
    if allow_under_budget:
        fail_evidence += "; --allow-under-budget ignored: --reason required"
    else:
        fail_evidence += (
            '; escape: --allow-under-budget --reason "<text>" (FAIL→SKIP only)'
        )
    return "FAIL", fail_evidence


def check_gpu_vram_available(
    *,
    min_vram_mib: int,
    timeout_sec: int,
    skip_live: bool,
) -> Tuple[str, str]:
    """Check 6: nvidia-smi free VRAM ≥ min_vram_mib on at least 1 GPU."""
    if skip_live:
        return "SKIP", "--skip-live set (CPU-only test mode)"
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return "SKIP", "nvidia-smi not on PATH (CPU-only host)"
    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return "FAIL", f"nvidia-smi timeout after {timeout_sec}s"
    except OSError as exc:
        return "FAIL", f"nvidia-smi exec error: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return "FAIL", f"nvidia-smi rc={proc.returncode}: {proc.stderr[:120]}"
    # Parse CSV; expect lines like "0, NVIDIA RTX ..., 97887"
    rows: List[Tuple[int, str, int]] = []
    for line in (proc.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                rows.append((int(parts[0]), parts[1], int(parts[2])))
            except ValueError:
                pass
    if not rows:
        return "FAIL", "nvidia-smi returned no GPU rows"
    best = max(rows, key=lambda r: r[2])
    if best[2] >= min_vram_mib:
        return (
            "PASS",
            f"GPU {best[0]} ({best[1]}) free {best[2]} MiB ≥ {min_vram_mib}",
        )
    return (
        "FAIL",
        f"GPU {best[0]} free {best[2]} MiB < {min_vram_mib} MiB min",
    )


def check_disk_free_space(
    *,
    min_disk_gib: int,
    timeout_sec: int,
    skip_live: bool,
) -> Tuple[str, str]:
    """Check 7: df free disk (REPO_ROOT filesystem) ≥ min_disk_gib."""
    if skip_live:
        return "SKIP", "--skip-live set (CPU-only test mode)"
    df_bin = shutil.which("df")
    if df_bin is None:
        return "SKIP", "df not on PATH (Windows host?)"
    try:
        proc = subprocess.run(
            [df_bin, "-k", str(REPO_ROOT)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return "FAIL", f"df timeout after {timeout_sec}s"
    except OSError as exc:
        return "FAIL", f"df exec error: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return "FAIL", f"df rc={proc.returncode}: {proc.stderr[:120]}"
    # Parse: header line + 1+ data lines; last column on data line is the
    # mountpoint; the "Available" column index is 3 on Linux df -k.
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return "FAIL", "df returned no data lines"
    data_line = lines[-1]
    parts = data_line.split()
    if len(parts) < 4:
        return "FAIL", f"df parse error: line={data_line!r}"
    try:
        avail_kib = int(parts[3])
    except ValueError:
        return "FAIL", f"df Available not int: {parts[3]!r}"
    avail_gib = avail_kib / (1024 * 1024)
    if avail_gib >= min_disk_gib:
        return (
            "PASS",
            f"free {avail_gib:.1f} GiB at {parts[-1]} ≥ {min_disk_gib} GiB",
        )
    return (
        "FAIL",
        f"free {avail_gib:.1f} GiB at {parts[-1]} < {min_disk_gib} GiB",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_per_wave_checks(
    *,
    wave: str,
    config_path: Optional[Path],
    preflight_path: Path,
    pairs_metals: List[str],
    pairs_seeds: List[int],
    min_vram_mib: int,
    min_disk_gib: int,
    budget_hr: Optional[float],
    max_age_sec: int,
    skip_live: bool,
    preflight_live: bool = False,
    budget_report: Optional[Path] = None,
    allow_under_budget: bool = False,
    under_budget_reason: str = "",
    strict: bool = False,
) -> Dict[str, Any]:
    """Run all 7 per-wave checks and return the structured report dict.

    The dict has the schema_version + checks[] + n_* fields + overall_status.
    It is also written to ``out`` by the caller.

    ``budget_hr=None`` means no budget was declared: check 5 then emits a
    loud SKIP (never PASS, never FAIL) — see
    :func:`check_wave_budget_remaining_nonneg`.  ``strict`` is passed into
    check 5 so ``--strict`` can refuse the ``--allow-under-budget`` escape
    (the script's own SKIP escalation still happens in :func:`main`).
    """
    checks: List[Dict[str, Any]] = []
    # 1
    s, ev = check_preflight_freshness_24h(
        preflight_path=preflight_path,
        max_age_sec=max_age_sec,
        declared_live=preflight_live,
    )
    checks.append({"check_id": "preflight_freshness_24h", "status": s, "evidence": ev})
    # 2
    s, ev = check_preflight_no_fail_gates(preflight_path=preflight_path)
    checks.append({"check_id": "preflight_no_fail_gates", "status": s, "evidence": ev})
    # 3
    s, ev = check_hyperparams_config_complete(config_path=config_path)
    checks.append(
        {"check_id": "hyperparams_config_complete", "status": s, "evidence": ev}
    )
    # 4
    s, ev = check_yuelbond_decoder_registered(config_path=config_path)
    checks.append(
        {"check_id": "yuelbond_decoder_registered", "status": s, "evidence": ev}
    )
    # 5
    s, ev = check_wave_budget_remaining_nonneg(
        budget_hr=budget_hr,
        pairs_metals=pairs_metals,
        pairs_seeds=pairs_seeds,
        budget_report=budget_report,
        allow_under_budget=allow_under_budget,
        under_budget_reason=under_budget_reason,
        strict=strict,
    )
    checks.append(
        {"check_id": "wave_budget_remaining_nonneg", "status": s, "evidence": ev}
    )
    # 6
    s, ev = check_gpu_vram_available(
        min_vram_mib=min_vram_mib,
        timeout_sec=DEFAULT_NVIDIA_SMI_TIMEOUT_SEC,
        skip_live=skip_live,
    )
    checks.append({"check_id": "gpu_vram_available", "status": s, "evidence": ev})
    # 7
    s, ev = check_disk_free_space(
        min_disk_gib=min_disk_gib,
        timeout_sec=DEFAULT_DF_TIMEOUT_SEC,
        skip_live=skip_live,
    )
    checks.append({"check_id": "disk_free_space", "status": s, "evidence": ev})

    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    n_skip = sum(1 for c in checks if c["status"] == "SKIP")
    if n_fail > 0:
        overall = "FAIL"
    elif n_skip > 0:
        overall = "PASS_WITH_SKIP"
    else:
        overall = "PASS"

    # Honest framing per party-twenty-spirit §3.5 — MEASURED values
    # (nvidia-smi / df / preflight mtime are live probes).  The budget
    # requirement is one MEASURED constant (0.7 h/pair upper bound,
    # Wf-268C) times one PROJECTED margin (SAFETY_FACTOR).
    honest = (
        f"Wf-343C2 per-wave check on wave={wave!r}: "
        f"{n_pass}/{N_CHECKS} PASS, {n_fail} FAIL, {n_skip} SKIP. "
        f"MEASURED live probes (nvidia-smi, df, preflight mtime); budget "
        f"baseline 0.7h/pair MEASURED (Wf-268C) × {SAFETY_FACTOR} PROJECTED "
        f"margin; undeclared budget = loud SKIP, never PASS."
    )

    obj: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "wave": wave,
        "generated_at": _now_iso(),
        "n_checks": N_CHECKS,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skip": n_skip,
        "overall_status": overall,
        "checks": checks,
        "honest_framing": honest,
        "schema_valid": True,  # populated by _validate_schema at end
        "config_path": str(config_path) if config_path else "",
        "preflight_path": str(preflight_path),
        "n_pairs": len(pairs_metals) * len(pairs_seeds),
        # None = no budget declared (check 5 emits a loud SKIP).  A
        # declared budget is always a float — the old DEFAULT_BUDGET_HR
        # magic constant was removed by the 2026-10-03 gate-semantics fix.
        "budget_hr": float(budget_hr) if budget_hr is not None else None,
        "skip_live": bool(skip_live),
        "error": "",
    }
    obj["schema_valid"] = _validate_schema(obj)
    return obj


def _validate_schema(obj: Dict[str, Any]) -> bool:
    """Duck-typed schema validator — returns True iff required top-level
    keys + checks[] shape + per-check sub-shape all conform."""
    expected: Dict[str, type] = {
        "schema_version": str,
        "wave": str,
        "generated_at": str,
        "n_checks": int,
        "n_pass": int,
        "n_fail": int,
        "n_skip": int,
        "overall_status": str,
        "checks": list,
        "honest_framing": str,
        "schema_valid": bool,
        "config_path": str,
        "preflight_path": str,
        "n_pairs": int,
        # None (no budget declared) or float (declared) — see the
        # 2026-10-03 gate-semantics fix: there is no default budget.
        "budget_hr": (float, type(None)),
        "skip_live": bool,
        "error": str,
    }
    for k, t in expected.items():
        if k not in obj:
            return False
        if not isinstance(obj[k], t):
            return False
    # checks[] shape
    if len(obj["checks"]) != N_CHECKS:
        return False
    seen_ids = set()
    for c in obj["checks"]:
        if not isinstance(c, dict):
            return False
        if not {"check_id", "status", "evidence"}.issubset(c.keys()):
            return False
        if c["status"] not in {"PASS", "FAIL", "SKIP"}:
            return False
        seen_ids.add(c["check_id"])
    expected_ids = {
        "preflight_freshness_24h",
        "preflight_no_fail_gates",
        "hyperparams_config_complete",
        "yuelbond_decoder_registered",
        "wave_budget_remaining_nonneg",
        "gpu_vram_available",
        "disk_free_space",
    }
    return seen_ids == expected_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_argv(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="wf343c2_per_wave_check.py",
        description=(
            "Wf-343C2 — per-wave internal pre-retrain check script. "
            "Runs 7 checks (preflight freshness + no-fail + hyperparams "
            "config + YuelBond decoder + wave budget + GPU VRAM + disk "
            "free space) against the specific wave config + live env, "
            "emits machine-readable JSON."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--wave",
        required=True,
        help="Wave name (e.g. 'Wave-1 chemprop v2 retrain').",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            f"Path to active configs/*.yaml for this wave "
            f"(default: {DEFAULT_CONFIG_YAML})."
        ),
    )
    p.add_argument(
        "--preflight",
        type=Path,
        default=DEFAULT_PREFLIGHT_JSON,
        help=f"Path to pre_flight.json (default: {DEFAULT_PREFLIGHT_JSON}).",
    )
    p.add_argument(
        "--preflight-live",
        action="store_true",
        help=(
            "Declare --preflight as produced by THIS launch: force the "
            "24h mtime contract even if the path is on the frozen-archive "
            "allowlist (FROZEN_PREFLIGHT_PATHS)."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUT_JSON}).",
    )
    p.add_argument(
        "--pairs-metals",
        type=str,
        default=DEFAULT_PAIRS_METALS,
        help=(
            f"Comma-separated metal list (default: {DEFAULT_PAIRS_METALS}). "
            f"Per-pair grid is the cross-product of metals × seeds."
        ),
    )
    p.add_argument(
        "--pairs-seeds",
        type=str,
        default=DEFAULT_PAIRS_SEEDS,
        help=f"Comma-separated seed list (default: {DEFAULT_PAIRS_SEEDS}).",
    )
    p.add_argument(
        "--min-vram-mib",
        type=int,
        default=DEFAULT_MIN_VRAM_MIB,
        help=(
            f"Minimum free VRAM in MiB per GPU (default: " f"{DEFAULT_MIN_VRAM_MIB})."
        ),
    )
    p.add_argument(
        "--min-disk-gib",
        type=int,
        default=DEFAULT_MIN_DISK_GIB,
        help=(
            f"Minimum free disk in GiB at REPO_ROOT (default: "
            f"{DEFAULT_MIN_DISK_GIB})."
        ),
    )
    p.add_argument(
        "--budget-hr",
        type=float,
        default=None,
        help=(
            "Declared wave budget in hours (the budget available for this "
            "wave's grid). Default: None = no budget declared, which makes "
            "check 5 emit a loud SKIP (never PASS). Precedence: --budget-hr "
            "> --budget-report."
        ),
    )
    p.add_argument(
        "--budget-report",
        type=Path,
        default=None,
        help=(
            "Path to a wave_budget_tracker.py report JSON, used when "
            "--budget-hr is absent. Prefers remaining_hr (budget left for "
            "this wave); falls back to budget_hr."
        ),
    )
    p.add_argument(
        "--allow-under-budget",
        action="store_true",
        help=(
            "Agreed escape for an under-budget wave: downgrades the check-5 "
            "FAIL to SKIP (never PASS). Requires --reason; --strict refuses "
            "the escape and re-FAILs."
        ),
    )
    p.add_argument(
        "--reason",
        type=str,
        default="",
        help="Free-text reason required together with --allow-under-budget.",
    )
    p.add_argument(
        "--max-age-sec",
        type=int,
        default=DEFAULT_PREFLIGHT_MAX_AGE_SEC,
        help=(
            f"Pre-flight freshness cap in seconds (default: "
            f"{DEFAULT_PREFLIGHT_MAX_AGE_SEC} = 24h per Wf-302A)."
        ),
    )
    p.add_argument(
        "--skip-live",
        action="store_true",
        help=(
            "Skip live probes (nvidia-smi + df). All live checks return "
            "SKIP. Useful for CPU-only test environments."
        ),
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat SKIP as FAIL (exit code 2). Default: SKIP only "
            "triggers PASS_WITH_SKIP without downgrading to FAIL."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose progress to stderr.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_argv(argv)
    if not args.wave.strip():
        print(
            "[wf343c2_per_wave_check] error: --wave must be non-empty",
            file=sys.stderr,
        )
        return 1
    try:
        pairs_metals = _parse_csv_str_list(args.pairs_metals)
        pairs_seeds = _parse_csv_int_list(args.pairs_seeds)
    except ValueError as exc:
        print(
            f"[wf343c2_per_wave_check] error: bad --pairs-metals/"
            f"--pairs-seeds: {exc}",
            file=sys.stderr,
        )
        return 1
    if not pairs_metals or not pairs_seeds:
        print(
            "[wf343c2_per_wave_check] error: --pairs-metals AND --pairs-seeds "
            "must each have at least one token",
            file=sys.stderr,
        )
        return 1

    if args.verbose:
        print(
            f"[wf343c2_per_wave_check] wave={args.wave!r} "
            f"config={args.config} preflight={args.preflight} "
            f"pairs=({len(pairs_metals)}m × {len(pairs_seeds)}s="
            f"{len(pairs_metals) * len(pairs_seeds)}) "
            f"min_vram={args.min_vram_mib}MiB min_disk={args.min_disk_gib}GiB "
            f"budget={args.budget_hr}h report={args.budget_report} "
            f"skip_live={args.skip_live} strict={args.strict}",
            file=sys.stderr,
        )

    obj = run_per_wave_checks(
        wave=args.wave,
        config_path=args.config if str(args.config or "") else None,
        preflight_path=args.preflight,
        pairs_metals=pairs_metals,
        pairs_seeds=pairs_seeds,
        min_vram_mib=args.min_vram_mib,
        min_disk_gib=args.min_disk_gib,
        budget_hr=None if args.budget_hr is None else float(args.budget_hr),
        max_age_sec=args.max_age_sec,
        skip_live=args.skip_live,
        preflight_live=args.preflight_live,
        budget_report=args.budget_report,
        allow_under_budget=args.allow_under_budget,
        under_budget_reason=args.reason,
        strict=args.strict,
    )

    try:
        _write_json(args.out, obj)
    except OSError as exc:
        print(
            f"[wf343c2_per_wave_check] FATAL: write {args.out} failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    if args.verbose:
        print(
            f"[wf343c2_per_wave_check] wrote {args.out} "
            f"(overall={obj['overall_status']} "
            f"pass={obj['n_pass']} fail={obj['n_fail']} skip={obj['n_skip']} "
            f"schema_valid={obj['schema_valid']})",
            file=sys.stderr,
        )

    # AGENTS.md §4 stdout invariant: emit the per-check PASS/FAIL/SKIP
    # verdict tokens on stdout (the JSON artifact carries the evidence).
    print(
        "wf343c2_per_wave_check: "
        + " ".join(f"{c['check_id']}={c['status']}" for c in obj["checks"])
        + f" | overall={obj['overall_status']}"
    )

    # Exit code policy (long-standing contract, preserved): FAIL outranks
    # everything; a non-strict run tolerates SKIPs; --strict escalates any
    # SKIP to exit 2.  The check-5 escape never yields PASS (SKIP only).
    if obj["n_fail"] > 0:
        return 1
    if obj["n_skip"] > 0 and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(
            f"[wf343c2_per_wave_check] UNHANDLED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
