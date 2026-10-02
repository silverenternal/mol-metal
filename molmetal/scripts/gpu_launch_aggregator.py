#!/usr/bin/env python3
"""Wf-360-A3 — GPU launch dashboard aggregator.

Reads the Wf-355 launch wrapper outputs (plan.json, launch.log, master.log,
per-pair training_summary.json) and produces a unified master status JSON
at molmetal/reports/gpu_launch_dashboard.json.

Owner:       Hugo Lee + ultracode agent (Wf-360-A3)
Date:        2026-09-29
Origin:      Wf-360-A3 GPU launch dashboard scaffold (per Wf-357-C3)
Updated:     Wf-364-B2 (UX polish + activation gate; --status-summary, --activate)
             + Wf-363-B2 (budget remaining indicator; budget_status field,
                          per-launcher remaining_hr vs 10h session cap +
                          DEFERRED_TEMP wave aggregate from wave_budget_tracker)
Parents:
  - scripts/gpu_launch_dashboard.sh        (read-only text dashboard)
  - scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh  (Wf-355-A1)
  - scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh             (Wf-355-A2)
  - scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh     (Wf-355-A3)
  - scripts/launch_wf131_p3_deferred_runs.sh                   (Wf-355-B1)
  - scripts/launch_wf146_deferred_runs.sh                      (Wf-355-B2)
  - scripts/launch_wf124_p3_g5.sh                              (TODO-181 G5)
  - scripts/recover_g6_failed_tuples.sh                        (Wf-337-G1 G6 recovery)
  - molmetal/scripts/wave_budget_tracker.py                    (DEFERRED_TEMP rebalance)

HONEST FRAMING (party-twenty-spirit §3.5):
  - Aggregator is READ-ONLY; does NOT modify any launcher output.
  - All status values traced to on-disk files (plan.json, launch.log,
    master.log, training_summary.json, run.log).
  - USER_OK gate not bypassed; this is a reporting-only scaffold.
  - Output schema version: 1.0; extend via increments to dashboard_schema_version.
  - Wf-364-B2: activation gate --activate flag (default = READ-ONLY scaffold).

Usage
-----
  python molmetal/scripts/gpu_launch_aggregator.py
  python molmetal/scripts/gpu_launch_aggregator.py --out /path/to/output.json
  python molmetal/scripts/gpu_launch_aggregator.py --launcher wf355_a3
  python molmetal/scripts/gpu_launch_aggregator.py --status-summary
  python molmetal/scripts/gpu_launch_aggregator.py --budget-summary
  python molmetal/scripts/gpu_launch_aggregator.py --activate
  python molmetal/scripts/gpu_launch_aggregator.py --stdout | jq .
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = "/home/hugo/codes/mol-metal"
DASHBOARD_OUTPUT = f"{PROJECT_ROOT}/molmetal/reports/gpu_launch_dashboard.json"

# Wf-363-B2: per-session budget cap (per Wf-262 §7.1 budget discipline;
# 10h × N waves is the standing convention for `wave_budget_tracker.py`
# DEFAULT_BUDGET_HR). Surfaced here so the dashboard's per-launcher
# `remaining_hr` computation has a single source of truth.
SESSION_CAP_HR = 10.0

# Wave budget report path (Wf-262 / TODO-208). Read-only; aggregator
# does NOT modify this file. The DEFERRED_TEMP rebalance figure is
# extracted from here for the `budget_status.deferred_temp_total_hr`
# field below.
WAVE_BUDGET_REPORT_JSON = (
    f"{PROJECT_ROOT}/molmetal/reports/wf262_wave_budget/wave_budget_report.json"
)

LAUNCHERS = [
    ("wf355_a1", f"{PROJECT_ROOT}/molmetal/reports/wf310e_r20_p3_drugood_full_launch",
     f"{PROJECT_ROOT}/scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh", 10.0),
    ("wf355_a2", f"{PROJECT_ROOT}/molmetal/reports/wf355a2_r20_p4_bindingdb",
     f"{PROJECT_ROOT}/scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh", 7.0),
    ("wf355_a3", f"{PROJECT_ROOT}/molmetal/reports/wf355a3_r20_p5_tmqm_plat",
     f"{PROJECT_ROOT}/scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh", 12.0),
    ("wf355_b1", f"{PROJECT_ROOT}/molmetal/reports/wf131_p3_deferred_runs",
     f"{PROJECT_ROOT}/scripts/launch_wf131_p3_deferred_runs.sh", 5.0),
    ("wf355_b2", f"{PROJECT_ROOT}/molmetal/reports/wf146_chemprop_lomo",
     f"{PROJECT_ROOT}/scripts/launch_wf146_deferred_runs.sh", 3.0),
    ("g5_retrain", f"{PROJECT_ROOT}/results/wf124_p3_50epoch_full_sweep_g5",
     f"{PROJECT_ROOT}/scripts/launch_wf124_p3_g5.sh", 5.0),
    ("g6_recover", f"{PROJECT_ROOT}/results/wf124_p3_per_innovation_ablation_g6",
     f"{PROJECT_ROOT}/scripts/recover_g6_failed_tuples.sh", 11.4),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_iso_mtime(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def discover_pairs(root: str) -> list[str]:
    """Discover per-pair directories under <root>/<Metal>/seed<N> or <root>/<cfg>/<Metal>/seed<N>."""
    pairs = []
    p = Path(root)
    if not p.exists():
        return pairs
    for d in p.glob("[A-Z][a-z]/seed[0-9]*"):
        if d.is_dir():
            pairs.append(str(d))
    for d in p.glob("*/*/seed[0-9]*"):
        if d.is_dir():
            pairs.append(str(d))
    return sorted(set(pairs))


def pair_status(pdir: str) -> str:
    """Detect pair status from training_summary.json (multi-shape) or run.log.

    Returns one of: PLANNED, RUNNING, COMPLETED, FAILED, UNKNOWN, STALE.
    Aliases OK→COMPLETED for backward compatibility with Wf-360-A3 status enum.
    """
    summary = Path(pdir) / "training_summary.json"
    runlog = Path(pdir) / "run.log"
    if summary.exists():
        try:
            with open(summary) as f:
                d = json.load(f)
            st = d.get("status")
            if not st:
                pm = d.get("per_metal", {})
                if pm:
                    for _, mdata in pm.items():
                        if isinstance(mdata, dict) and mdata.get("status"):
                            st = mdata["status"]
                            break
            if not st:
                ppt = d.get("per_pair_table", [])
                if isinstance(ppt, list) and ppt:
                    ok = sum(1 for r in ppt if isinstance(r, dict) and r.get("status") == "OK")
                    failed = sum(1 for r in ppt if isinstance(r, dict) and r.get("status") == "FAILED")
                    if failed > 0:
                        st = "PARTIAL"
                    elif ok > 0:
                        st = "OK"
            if not st and d.get("elapsed_seconds_total"):
                if runlog.exists():
                    text = runlog.read_text(errors="ignore")
                    if "DONE total elapsed=" in text or re.search(r"best_epoch=\d+", text):
                        st = "OK"
                    elif "FAILED" in text or "Traceback" in text:
                        st = "FAILED"
            st = (st or "UNKNOWN").upper()
            # Normalize: OK → COMPLETED; ALL_SEEDS_FAILED → FAILED
            if st == "OK":
                return "COMPLETED"
            if st == "ALL_SEEDS_FAILED":
                return "FAILED"
            return st
        except Exception:
            return "UNKNOWN"
    if runlog.exists():
        # Wf-363-B1: even when no summary.json, peek at run.log content
        # for completion / failure signatures BEFORE falling back to age.
        try:
            text = runlog.read_text(errors="ignore")
            if "DONE total elapsed=" in text or re.search(r"best_epoch=\d+", text):
                return "COMPLETED"
            if "FAILED" in text or "Traceback" in text:
                return "FAILED"
        except Exception:
            pass
        age_min = (datetime.now().timestamp() - os.path.getmtime(runlog)) / 60
        return "RUNNING" if age_min < 30 else "STALE"
    # No artifacts at all: pair is planned but not started.
    return "PLANNED"


def parse_pair_identity(pdir: str) -> tuple[str, str | None]:
    """Parse (metal, seed) from a per-pair directory path.

    Recognizes two shapes:
      - <root>/<Metal>/seed<N>          (g5_retrain)
      - <root>/<cfg>/<Metal>/seed<N>    (g6_recover: no_dann/Ir/seed0)

    Returns ("UNKNOWN", None) if neither pattern matches.
    """
    parts = Path(pdir).parts
    seed = None
    metal = "UNKNOWN"
    for part in reversed(parts):
        m = re.match(r"^seed(\d+)$", part)
        if m:
            seed = m.group(1)
            continue
        # metal-name detection: starts with uppercase letter followed by lowercase
        if re.match(r"^[A-Z][a-z]$", part) and metal == "UNKNOWN":
            metal = part
    if seed is None:
        return (metal, None)
    return (metal, seed)


def pair_progress_detail(pdir: str) -> dict:
    """Per-pair completion tracking dict for the Wf-363-B1 schema.

    Schema:
      {
        "metal":        "Ru",
        "seed":         "0",
        "status":       "RUNNING" | "COMPLETED" | "FAILED" | "PLANNED" | "STALE" | "UNKNOWN",
        "wall_elapsed_s": int (since first artifact mtime in dir, 0 if no artifact),
        "last_log_ts":   ISO-8601 UTC string or None,
      }
    """
    metal, seed = parse_pair_identity(pdir)
    status = pair_status(pdir)
    runlog = Path(pdir) / "run.log"
    summary = Path(pdir) / "training_summary.json"
    # wall_elapsed_s = elapsed since the earliest artifact in this pair dir
    # (proxy for "how long has this pair been running / completed since")
    wall_elapsed_s = 0
    last_ts = None
    artifacts = []
    for art in (runlog, summary):
        if art.exists():
            artifacts.append(art)
    if artifacts:
        mtimes = [os.path.getmtime(str(a)) for a in artifacts]
        first_ts = min(mtimes)
        last_ts = max(mtimes)
        wall_elapsed_s = int(datetime.now().timestamp() - first_ts)
    last_log_ts = (
        datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if last_ts is not None else None
    )
    return {
        "metal": metal,
        "seed": seed,
        "status": status,
        "wall_elapsed_s": wall_elapsed_s,
        "last_log_ts": last_log_ts,
    }


VALID_PAIR_STATUSES = {
    "PLANNED", "RUNNING", "COMPLETED", "FAILED", "STALE", "UNKNOWN",
}


def launcher_status(root: str) -> str:
    """Detect launcher-level status from per-pair aggregates + root files."""
    pairs = discover_pairs(root)
    if pairs:
        statuses = [pair_status(p) for p in pairs]
        if any(s == "RUNNING" for s in statuses):
            return "RUNNING"
        ok = sum(1 for s in statuses if s in ("OK", "COMPLETED"))
        failed = sum(1 for s in statuses if s in ("FAILED", "ALL_SEEDS_FAILED"))
        if ok + failed == len(statuses):
            return "COMPLETED" if failed == 0 else "PARTIAL"
    plan_json = Path(root) / "plan.json"
    if plan_json.exists():
        try:
            with open(plan_json) as f:
                d = json.load(f)
            ps = (d.get("status") or "").upper()
            if ps in ("COMPLETED", "DRY_RUN_OK", "PLAN_ONLY", "FAILED"):
                return ps
        except Exception:
            pass
    summary = Path(root) / "training_summary.json"
    if summary.exists():
        try:
            with open(summary) as f:
                d = json.load(f)
            if d.get("todo_182_g6_total_tuples_planned") or d.get("todo_181_g5_total_pairs"):
                return "PARTIAL"
        except Exception:
            pass
    launch_log = Path(root) / "launch.log"
    if not launch_log.exists():
        launch_log = Path(root) / "master.log"
    if launch_log.exists():
        try:
            text = launch_log.read_text(errors="ignore")
            if "DRY-RUN:" in text:
                return "DRY_RUN_OK"
            age_hr = (datetime.now().timestamp() - os.path.getmtime(launch_log)) / 3600
            return "STALE" if age_hr > 24 else "READY"
        except Exception:
            pass
    return "NO_PLAN"


def aggregate_launcher(name: str, root: str, script: str, wall_proj: float,
                        include_per_pair: bool = False) -> dict:
    """Build the per-launcher JSON entry.

    When include_per_pair=True, also emit a per_pair_progress array with
    [{metal, seed, status, wall_elapsed_s, last_log_ts}] entries
    (Wf-363-B1 per-pair completion tracking).
    """
    status = launcher_status(root)
    pairs = discover_pairs(root)
    pair_statuses = [pair_status(p) for p in pairs]
    ok = sum(1 for s in pair_statuses if s in ("OK", "COMPLETED"))
    failed = sum(1 for s in pair_statuses if s in ("FAILED", "ALL_SEEDS_FAILED"))
    running = sum(1 for s in pair_statuses if s == "RUNNING")
    pending = sum(1 for s in pair_statuses if s in ("PENDING", "STALE", "UNKNOWN"))
    plan_total = 0
    plan_json = Path(root) / "plan.json"
    if plan_json.exists():
        try:
            with open(plan_json) as f:
                plan_total = (json.load(f)).get("total_runs", 0) or 0
        except Exception:
            pass
    launch_log = Path(root) / "launch.log"
    if not launch_log.exists():
        launch_log = Path(root) / "master.log"
    elapsed_sec = int(datetime.now().timestamp() - os.path.getmtime(str(launch_log))) if launch_log.exists() else 0
    entry = {
        "name": name,
        "script": script,
        "root": root,
        "status": status,
        "wall_projected_hr": wall_proj,
        "last_activity": now_iso_mtime(str(launch_log)),
        "elapsed_sec": elapsed_sec,
        "pair_progress": {
            "total_pairs": plan_total if plan_total > 0 else len(pairs),
            "discovered_pairs": len(pairs),
            "ok": ok,
            "failed": failed,
            "running": running,
            "pending": pending,
        },
        "activation_gated": True,
        "user_ok_required": True,
    }
    if include_per_pair:
        entry["per_pair_progress"] = [pair_progress_detail(p) for p in pairs]
    return entry


def aggregate(filter_launcher: str | None = None,
              include_per_pair: bool = False) -> dict:
    """Build the full dashboard JSON.

    include_per_pair=True emits a per_pair_progress array per launcher
    (Wf-363-B1 per-pair completion tracking).

    Wf-363-B2: also emits `budget_status` field with per-launcher
    remaining_hr vs the 10h session cap, plus the DEFERRED_TEMP wave
    aggregate from wave_budget_tracker.py (per Wf-360-A1 rebalance).
    """
    entries = []
    for name, root, script, wall_proj in LAUNCHERS:
        if filter_launcher and name != filter_launcher:
            continue
        entries.append(aggregate_launcher(name, root, script, wall_proj,
                                          include_per_pair=include_per_pair))
    return {
        "dashboard_generated_at": now_iso(),
        "dashboard_schema_version": "1.0",
        "wf_origin": "Wf-360-A3 GPU launch dashboard scaffold",
        "wf_origin_update": "Wf-364-B2: activation gate + status-summary; "
                            "Wf-363-B1: per-pair completion tracking; "
                            "Wf-363-B2: budget remaining indicator (budget_status)",
        "activation_gated": True,
        "user_ok_required": True,
        "launcher_count": len(entries),
        "launchers": entries,
        "budget_status": compute_budget_status(entries, session_cap_hr=SESSION_CAP_HR),
    }


def format_status_summary(result: dict) -> str:
    """Compact TSV status summary: NAME\\tSTATUS\\tPAIRS\\tWALL_PROJ_HR."""
    lines = ["NAME\tSTATUS\tPAIRS\tWALL_PROJ_HR"]
    for entry in result["launchers"]:
        pp = entry["pair_progress"]
        pair_str = f"{pp['ok']}/{pp['total_pairs']}"
        lines.append(f"{entry['name']}\t{entry['status']}\t{pair_str}\t{entry['wall_projected_hr']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wf-363-B2 — budget remaining indicator
# ---------------------------------------------------------------------------

def _load_wave_budget_report() -> dict | None:
    """Read wave_budget_tracker.py's report JSON (READ-ONLY).

    Returns the parsed dict, or None when the file is missing or invalid.
    The aggregator never writes to this file; it only consumes it to derive
    the `deferred_temp_total_hr` rebalance figure per Wf-360-A1.
    """
    path = Path(WAVE_BUDGET_REPORT_JSON)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _classify_deferred_class_safe(wave_name: str) -> str:
    """Classify a wave's deferred class via wave_budget_tracker.

    Falls back to the conservative default "DEFERRED_TEMP" when the
    import fails (per wave_budget_tracker.classify_deferred_class
    default behaviour — never silently mis-classifies a USER_OK-gated
    wave as permanent). This keeps the dashboard self-contained even
    when wave_budget_tracker is not on PYTHONPATH.
    """
    try:
        # Defer import to function-call time so the aggregator can
        # still be imported in minimal test environments.
        import importlib.util as importlib_util
        spec = importlib_util.spec_from_file_location(
            "wave_budget_tracker",
            f"{PROJECT_ROOT}/molmetal/scripts/wave_budget_tracker.py",
        )
        if spec is None or spec.loader is None:
            return "DEFERRED_TEMP"
        mod = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = mod.classify_deferred_class(wave_name)
        return cls or "DEFERRED_TEMP"
    except Exception:
        return "DEFERRED_TEMP"


def _deferred_temp_total_hr() -> tuple[float, list[dict]]:
    """Compute DEFERRED_TEMP total wall-hours from the wave budget report.

    Walks every wave in the report, filters to status='DEFERRED' +
    deferred_class=='DEFERRED_TEMP', and aggregates actual_hr (when
    available, MEASURED) or estimated_hr (PROJECTED fallback). Returns
    (total_hr, list_of_per_wave_dicts).

    Honest framing (party-twenty-spirit §3.5):
      - Per-wave actual_hr takes priority over estimated_hr (MEASURED
        values are authoritative when present).
      - Waves missing actual_hr fall back to estimated_hr (PROJECTED).
      - Each per-wave dict carries an `evidence` string so the source
        of the figure is auditable downstream.
    """
    report = _load_wave_budget_report()
    if not report:
        return 0.0, []
    waves = report.get("waves", []) or []
    total_hr = 0.0
    per_wave: list[dict] = []
    for w in waves:
        if (w.get("status") or "").upper() != "DEFERRED":
            continue
        name = w.get("wave_name") or ""
        if _classify_deferred_class_safe(name) != "DEFERRED_TEMP":
            continue
        actual = w.get("actual_hr")
        estimated = w.get("estimated_hr") or 0.0
        # MEASURED > PROJECTED: actual_hr wins when present (non-null + > 0).
        if actual is not None and float(actual) > 0.0:
            hr = float(actual)
            evidence = "MEASURED_actual_hr"
        else:
            hr = float(estimated)
            evidence = "PROJECTED_estimated_hr"
        total_hr += hr
        per_wave.append({
            "wave_name": name,
            "deferred_class": "DEFERRED_TEMP",
            "hr": round(hr, 2),
            "evidence": evidence,
        })
    return round(total_hr, 6), per_wave


def compute_budget_status(entries: list[dict],
                          session_cap_hr: float = SESSION_CAP_HR) -> dict:
    """Compute the per-launcher + cross-launcher budget remaining state.

    Output schema (Wf-363-B2):
      {
        "session_cap_hr": 10.0,
        "per_launcher": [
            {
              "launcher": "<name>",
              "budget_hr": <wall_projected_hr>,
              "remaining_hr": <session_cap_hr - budget_hr>,
              "status": "WITHIN" | "OVER",
              "status_evidence": "<human-readable>",
              "wall_projected_hr": <budget_hr>,
              "pair_progress_share": <float in [0,1]>,
              "launcher_status": "<STATUS>"
            },
            ...
        ],
        "total_projected_hr": <sum budget_hr>,
        "total_remaining_hr": <session_cap_hr - total_projected_hr>,
        "deferred_temp_total_hr": <hr>,
        "deferred_temp_waves": [...],
        "deferred_temp_remaining_hr": <session_cap_hr - deferred_temp_total_hr>,
        "honest_framing": "<5-line summary>"
      }

    Per Wf-262 §7.1 budget discipline: each launcher's `remaining_hr` is
    computed against the 10h SESSION_CAP_HR (so a single launcher that
    exceeds 10h individually is flagged `OVER` regardless of how many
    other launchers exist). The cross-launcher `total_remaining_hr`
    figures the SUM of all per-launcher budgets vs the session cap, so
    the operator can see at-a-glance whether the session fits.

    Honest framing per party-twenty-spirit §3.5:
      - All per-launcher figures use the registered wall_projected_hr
        (PROJECTED by Wf-355; not re-measured here).
      - DEFERRED_TEMP aggregate MEASURED > PROJECTED per-wave.
    """
    per_launcher: list[dict] = []
    total_projected_hr = 0.0
    for e in entries:
        budget_hr = float(e.get("wall_projected_hr", 0.0) or 0.0)
        remaining_hr = round(session_cap_hr - budget_hr, 6)
        status = "WITHIN" if remaining_hr >= -1e-9 else "OVER"
        # Per-launcher pair-progress share: completed-pairs / total-pairs
        # (or 0.0 when no plan exists yet).
        pp = e.get("pair_progress", {}) or {}
        total_pairs = pp.get("total_pairs", 0) or 0
        ok_pairs = pp.get("ok", 0) or 0
        share = (ok_pairs / total_pairs) if total_pairs > 0 else 0.0
        per_launcher.append({
            "launcher": e.get("name", "?"),
            "budget_hr": round(budget_hr, 6),
            "remaining_hr": remaining_hr,
            "status": status,
            "status_evidence": (
                f"{round(budget_hr, 2)}h projected vs {session_cap_hr}h session cap "
                f"= {round(remaining_hr, 2)}h remaining"
            ),
            "wall_projected_hr": round(budget_hr, 6),
            "pair_progress_share": round(share, 6),
            "launcher_status": e.get("status", "UNKNOWN"),
        })
        total_projected_hr += budget_hr

    deferred_temp_hr, deferred_temp_waves = _deferred_temp_total_hr()
    total_remaining_hr = round(session_cap_hr - total_projected_hr, 6)
    deferred_temp_remaining_hr = round(session_cap_hr - deferred_temp_hr, 6)

    over_launchers = [pl["launcher"] for pl in per_launcher if pl["status"] == "OVER"]
    within_launchers = [pl["launcher"] for pl in per_launcher if pl["status"] == "WITHIN"]
    honest_lines = [
        f"Session cap = {session_cap_hr}h (per Wf-262 §7.1 budget discipline).",
        f"Per-launcher: {len(within_launchers)} WITHIN ({', '.join(within_launchers) or 'none'}), "
        f"{len(over_launchers)} OVER ({', '.join(over_launchers) or 'none'}).",
        f"Sum of all per-launcher budgets = {round(total_projected_hr, 2)}h "
        f"vs {session_cap_hr}h cap = {round(total_remaining_hr, 2)}h total remaining.",
        f"DEFERRED_TEMP rebalance (per Wf-360-A1): {round(deferred_temp_hr, 2)}h "
        f"across {len(deferred_temp_waves)} wave(s); "
        f"{round(deferred_temp_remaining_hr, 2)}h headroom.",
    ]
    if over_launchers:
        honest_lines.append(
            f"HONEST-NEGATIVE: {len(over_launchers)} launcher(s) individually exceed "
            f"the {session_cap_hr}h session cap and would need cross-session scheduling "
            f"or a USER_OK waiver."
        )
    else:
        honest_lines.append(
            "All launchers individually fit within the 10h session cap."
        )

    return {
        "session_cap_hr": session_cap_hr,
        "per_launcher": per_launcher,
        "total_projected_hr": round(total_projected_hr, 6),
        "total_remaining_hr": total_remaining_hr,
        "deferred_temp_total_hr": round(deferred_temp_hr, 6),
        "deferred_temp_waves": deferred_temp_waves,
        "deferred_temp_remaining_hr": deferred_temp_remaining_hr,
        "honest_framing": " | ".join(honest_lines),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wf-360-A3 GPU launch dashboard aggregator")
    parser.add_argument("--out", default=DASHBOARD_OUTPUT,
                        help=f"Output JSON path (default: {DASHBOARD_OUTPUT})")
    parser.add_argument("--launcher", default=None,
                        help="Filter to a single launcher name (e.g. wf355_a3)")
    parser.add_argument("--stdout", action="store_true",
                        help="Print JSON to stdout INSTEAD of writing to --out")
    parser.add_argument("--status-summary", action="store_true",
                        help="Emit compact TSV status summary (NAME\\tSTATUS\\tPAIRS\\tWALL_PROJ_HR) "
                             "instead of full JSON. Default = off (full JSON).")
    parser.add_argument("--activate", action="store_true",
                        help="ACTIVATION GATE: enable LIVE mode (default = READ-ONLY scaffold). "
                             "Live mode is USER_OK-gated per AGENTS.md §5.1; the operator must "
                             "explicitly pass --activate to opt in. Output is tagged with "
                             "'dashboard_mode' = LIVE vs READ-ONLY_SCAFFOLD.")
    parser.add_argument("--per-pair-detail", action="store_true",
                        help="Wf-363-B1: emit per_pair_progress array per launcher "
                             "[{metal, seed, status, wall_elapsed_s, last_log_ts}]. "
                             "Use with --json or --stdout for machine-readable output.")
    parser.add_argument("--budget-summary", action="store_true",
                        help="Wf-363-B2: emit ONLY the budget_status JSON "
                             "(per-launcher remaining_hr vs 10h session cap + "
                             "DEFERRED_TEMP aggregate). Default = off (full JSON).")
    args = parser.parse_args(argv)
    result = aggregate(filter_launcher=args.launcher,
                        include_per_pair=args.per_pair_detail)
    # Activation gate: tag the output with dashboard_mode (LIVE vs READ-ONLY_SCAFFOLD).
    # Default = READ-ONLY_SCAFFOLD; LIVE only when --activate is explicitly set.
    result["dashboard_mode"] = "LIVE" if args.activate else "READ-ONLY_SCAFFOLD"
    result["activation_flag_set"] = bool(args.activate)
    if args.budget_summary:
        # Wf-363-B2: emit just the budget_status field as compact JSON.
        text = json.dumps(result.get("budget_status", {}), indent=2)
    elif args.status_summary:
        text = format_status_summary(result)
    else:
        text = json.dumps(result, indent=2)
    if args.stdout:
        print(text)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"[{now_iso()}] wrote dashboard → {args.out} "
              f"({len(result['launchers'])} launchers, mode={result['dashboard_mode']})",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())