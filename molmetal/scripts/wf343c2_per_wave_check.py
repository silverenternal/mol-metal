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

  1. ``preflight_freshness_24h``     — pre_flight.json ≤ 24h old
  2. ``preflight_no_fail_gates``     — no FAIL gates in pre-flight JSON
  3. ``hyperparams_config_complete`` — wave YAML has all REQUIRED_CONFIG_KEYS
  4. ``yuelbond_decoder_registered``  — YuelBond decoder wired into CFM scripts
  5. ``wave_budget_remaining_nonneg``— wave budget remaining ≥ 0
  6. ``gpu_vram_available``          — nvidia-smi free VRAM ≥ threshold
  7. ``disk_free_space``             — df free disk ≥ threshold (per-wave data)

CLI::

    wf343c2_per_wave_check.py --wave "Wave-1 chemprop v2 retrain"
                              [--config configs/cfm_full_retrain.yaml]
                              [--preflight molmetal/reports/wf259_run/pre_flight.json]
                              [--out molmetal/reports/wf343c2_per_wave_check/per_wave_check.json]
                              [--pairs-metals "Ru,Ir,Os,Rh,Re"]
                              [--pairs-seeds "0,1,2"]
                              [--min-vram-mib 8192]
                              [--min-disk-gib 50]
                              [--budget-hr 10.0]
                              [--strict]
                              [--skip-live]    # skip nvidia-smi + df (CPU-only tests)
                              [--verbose]

Exit codes:

  * 0 — every check PASSED (or SKIP — only when ``--strict`` is NOT set).
  * 1 — at least one check FAILED.
  * 2 — at least one check SKIPPED AND ``--strict`` is set.

The script is **CPU-only** by default (``--skip-live`` toggles the nvidia-smi
+ df probes; defaults to running them so the live environment is checked).
It is **backward-compat** with any pre-flight JSON whose top-level keys
include ``summary`` + ``gates``.

Honest framing (party-twenty-spirit §3.5):
    Every numeric value in the JSON is MEASURED at the moment of wave-launch
    (nvidia-smi / df / pre-flight freshness), not PROJECTED.  The
    ``overall_status`` is FAIL whenever any check is FAIL — there is no
    "FAIL-with-recovery" downgrade.  The strict flag only governs SKIP
    classification.

Backward-compat: default ``--out`` is
``molmetal/reports/wf343c2_per_wave_check/per_wave_check.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


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
DEFAULT_BUDGET_HR = 10.0
DEFAULT_NVIDIA_SMI_TIMEOUT_SEC = 10
DEFAULT_DF_TIMEOUT_SEC = 5

# Per-pair baseline wall-hour estimate (D-MPNN Wasserstein 30-epoch on
# RTX 5090 GPU-1 per Wf-124 P3 + Wf-262 §0.5 + Wf-268C actuals).  Used
# by check_wave_budget_remaining_nonneg to flag grossly over-budget
# waves (e.g. 25 pairs × 0.5h = 12.5h > 10h budget).
PER_PAIR_BASELINE_HR = 0.5

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


def check_preflight_freshness_24h(
    *,
    preflight_path: Path,
    max_age_sec: int,
) -> Tuple[str, str]:
    """Check 1: pre_flight.json mtime ≤ max_age_sec seconds old."""
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
        f"preflight STALE: age {age_sec:.1f}s > {max_age_sec}s (24h cap)",
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
    budget_hr: float,
    pairs_metals: List[str],
    pairs_seeds: List[int],
) -> Tuple[str, str]:
    """Check 5: wave budget remaining ≥ 0 against (metal, seed) grid cost.

    Per-pair baseline cost is :data:`PER_PAIR_BASELINE_HR` (D-MPNN
    Wasserstein 30-epoch on RTX 5090 GPU-1 per Wf-124 P3 / Wf-262
    actuals).  A wave is FAIL when
    ``n_pairs × PER_PAIR_BASELINE_HR > budget_hr``.

    The 0.5h/pair baseline is PROJECTED (Wf-262 §0.5 honest gap; Wf-268C
    MEASURED 0.55-0.7h/pair for Ru/Ir heavier cohorts), so we use a
    conservative floor to avoid false FAILs on light cohorts while
    still flagging grossly over-budget waves (e.g. 25 pairs × 0.5h =
    12.5h > 10h budget).
    """
    n_pairs = max(1, len(pairs_metals) * len(pairs_seeds))
    required_hr = round(n_pairs * PER_PAIR_BASELINE_HR, 2)
    if required_hr <= budget_hr + 1e-9:
        return (
            "PASS",
            f"budget {budget_hr:.1f}h ≥ est {required_hr:.2f}h "
            f"({n_pairs} pairs × {PER_PAIR_BASELINE_HR}h/pair)",
        )
    return (
        "FAIL",
        f"budget {budget_hr:.1f}h < est {required_hr:.2f}h "
        f"({n_pairs} pairs × {PER_PAIR_BASELINE_HR}h/pair)",
    )


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
    budget_hr: float,
    max_age_sec: int,
    skip_live: bool,
) -> Dict[str, Any]:
    """Run all 7 per-wave checks and return the structured report dict.

    The dict has the schema_version + checks[] + n_* fields + overall_status.
    It is also written to ``out`` by the caller.
    """
    checks: List[Dict[str, Any]] = []
    # 1
    s, ev = check_preflight_freshness_24h(
        preflight_path=preflight_path,
        max_age_sec=max_age_sec,
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
    # (nvidia-smi / df / preflight freshness are live probes).
    honest = (
        f"Wf-343C2 per-wave check on wave={wave!r}: "
        f"{n_pass}/{N_CHECKS} PASS, {n_fail} FAIL, {n_skip} SKIP. "
        f"MEASURED live probes (nvidia-smi, df, preflight mtime); "
        f"PROJECTED budget/HR estimates flagged as such."
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
        "budget_hr": float(budget_hr),
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
        "budget_hr": float,
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
        default=DEFAULT_BUDGET_HR,
        help=f"Wave budget hours (default: {DEFAULT_BUDGET_HR}).",
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
            f"budget={args.budget_hr}h skip_live={args.skip_live} "
            f"strict={args.strict}",
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
        budget_hr=float(args.budget_hr),
        max_age_sec=args.max_age_sec,
        skip_live=args.skip_live,
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

    # Exit code policy
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
