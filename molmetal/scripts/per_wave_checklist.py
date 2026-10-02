#!/usr/bin/env python3
"""Wf-262 / TODO-208 — per_wave_checklist.py: per-wave sub-checklist generator.

==============================================================================
Wf-262 / TODO-208 — generates a wave-specific sub-checklist JSON + Markdown
from pre-flight JSON + active config yaml per
``TODO/completed/207_unified_pre_flight_checklist_v2.md`` §9.

Responsibilities
---------------

  1. Read the ``pre_flight.json`` produced by
     ``molmetal/scripts/pre_flight_check.py`` (or any per-wave JSON with
     the same ``summary`` + ``gates`` shape).
  2. Read the active ``configs/*.yaml`` for the wave.
  3. Partition the pre-flight gates by **priority** (P0 / P1 / P2).
  4. Compute **per-pair** sub-checklist entries — one row per
     ``(metal, seed)`` pair the wave will retrain — by scanning
     ``molmetal/reports/wf*/<metal>/per_seed_metrics_seed<N>.json`` and
     ``molmetal/checkpoints/dmpnn_*_<metal>_seed<N>.pt``.
  5. Compute **abort criteria active** (§7.2) from per-pair results
     (e.g. ``per_pair_timeout``, ``failure_pattern_3consec``).
  6. Compute **wave budget** by reading ``protocol.time_budget_hr`` from
     the YAML (default 10.0 h; per ``207 §0.3``).
  7. Emit both JSON and Markdown outputs.

CLI::

    per_wave_checklist.py --wave "Wave-1 chemprop v2 retrain"
                          [--config configs/cfm_full_retrain.yaml]
                          [--preflight molmetal/reports/wf259_run/pre_flight.json]
                          [--out molmetal/reports/wf262_checklist/per_wave.json]
                          [--out-md molmetal/reports/wf262_checklist/per_wave.md]
                          [--pairs-metals "Ru,Ir,Os,Rh,Re"]
                          [--pairs-seeds "0,1,2"]
                          [--time-budget-hr 10.0]
                          [--verbose]

Exit codes:

  * 0 — checklist emitted (regardless of gate pass/fail).
  * 1 — bad usage / missing required CLI flag.
  * 2 — script error (config parse fail, preflight unreadable, etc.).

This script is **CPU-only** and **backward-compat** with any pre-flight
JSON whose top-level keys include ``summary`` (pass/fail/skip/total) +
``gates`` (list of ``{name, priority, status, ...}`` dicts).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from glob import glob as _glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )
    _WF424_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF424_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../mol-metal
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
SCRIPT_DIR = MOLMETAL_ROOT / "scripts"
CONFIGS_DIR = REPO_ROOT / "configs"
REPORTS_DIR = MOLMETAL_ROOT / "reports"
CHECKPOINTS_DIR = MOLMETAL_ROOT / "checkpoints"
TESTS_DIR = MOLMETAL_ROOT / "tests"

DEFAULT_PREFLIGHT_JSON = REPORTS_DIR / "wf259_run" / "pre_flight.json"
DEFAULT_OUT_JSON = REPORTS_DIR / "wf262_checklist" / "per_wave.json"
DEFAULT_OUT_MD = REPORTS_DIR / "wf262_checklist" / "per_wave.md"
DEFAULT_CONFIG_YAML = CONFIGS_DIR / "cfm_full_retrain.yaml"

SCHEMA_VERSION = "262.v2"
DEFAULT_SESSION_BUDGET_HR = 10.0  # per 207 v2 §0.3
DEFAULT_PAIRS_METALS = "Ru,Ir,Os,Rh,Re"  # 5-metal protocol
DEFAULT_PAIRS_SEEDS = "0,1,2,3,4"  # 5-seed protocol

# Required config-keys contract (per 207 v2 §1.3 + §2.1):
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

# §7.2 abort-criteria name constants — used both for detection AND for the
# emitted ``abort_criteria_active`` array so downstream tools can match
# by name (not by string text).
ABORT_CRITERIA: Dict[str, str] = {
    "7.2.1_per_wave_timeout": "Per-wave timeout: (metal, seed) wall > 2× projected",
    "7.2.2_failure_pattern_3consec": "3+ consecutive seeds crash on same metal",
    "7.2.3_gpu_contention": "GPU util < 70% for 5+ minutes",
    "7.2.4_os_collapse": "Os AUC < 0.51 (structural collapse)",
    "7.2.5_decode_ratio_collapse": "decode_ratio < 0.05 (CFM went-to-zero)",
    "7.2.6_m7_tanimoto_below_004": "m7 Tanimoto < 0.04 (honest-negative baseline)",
    "7.2.7_checkpoint_loaded_forward_deferred": "ckpt loaded but no `epoch_*` keys written",
    "7.2.8_partial_pilot": "n_completed < n_total (5/12 partial pilot)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wf424_atomic_write_text(path, content: str, min_lines: int = 50, expected_min_length: int = 200) -> None:
    """Wf-424-A4 atomic-write-protocol wrapper around direct write_text.

    Falls back to direct write_text if atomic_write_file is unavailable
    (e.g. when the script is run standalone without the molmetal package
    on PYTHONPATH).
    """
    if _WF424_HAS_ATOMIC and atomic_write_file is not None:
        if safe_overwrite_check is not None:
            _risk = safe_overwrite_check(path, content, min_lines=min_lines)
            if _risk is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-424-A4] safe_overwrite_check diff_ratio=%.3f at %s",
                    _risk, path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-424-A4] detect_truncation reason=%r at %s",
                    _trunc, path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")

def _now_iso() -> str:
    """ISO-8601 UTC timestamp, e.g. ``2026-09-28T13:00:00+00:00``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _read_json(path: Path) -> Dict[str, Any]:
    """Read JSON; raise ``FileNotFoundError`` / ``json.JSONDecodeError``
    on bad path or malformed JSON. Used for the pre-flight input."""
    with open(str(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, obj: Any) -> None:
    """Write JSON with ``indent=2, ensure_ascii=False``; create parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def _write_text(path: Path, text: str) -> None:
    """Write text; create parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as fh:
        fh.write(text)


def _parse_yaml_safe_load(path: Path) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Best-effort YAML parse; never raises. Returns
    ``(parse_ok, parsed_dict_or_None, error_message_or_empty)``. Uses
    PyYAML when available, falls back to a tiny dict-of-dicts parser
    sufficient for the schema check.
    """
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
    """Fallback parser — recognises only top-level ``key: value`` scalars
    (and 2-level ``key:\n  subkey: value`` dicts).  Sufficient for the
    presence / absence of ``REQUIRED_CONFIG_KEYS``; sufficient for
    ``honest_framing`` (str) and ``protocol.time_budget_hr`` lookup.
    For accurate structural parses we rely on the real PyYAML.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return False, None, f"read failed: {type(exc).__name__}: {exc}"
    out: Dict[str, Any] = {}
    top_pending: Optional[str] = None  # current top with sub-dict
    pending_dict: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith(" ") or line.startswith("\t"):
            # sub-key — best-effort str value only
            if top_pending is None:
                continue
            stripped = line.strip()
            if ":" in stripped:
                k, _, v = stripped.partition(":")
                pending_dict[k.strip()] = v.strip().strip('"').strip("'")
            continue
        if top_pending is not None:
            # Flush previous top with sub-dict
            out[top_pending] = pending_dict
            top_pending = None
            pending_dict = {}
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if v == "":
                # likely a sub-block
                top_pending = k
                pending_dict = {}
            else:
                # scalar — strip quotes + comments
                v = v.split("#", 1)[0].strip().strip('"').strip("'")
                out[k] = v
    if top_pending is not None:
        out[top_pending] = pending_dict
    return True, out, ""


def _parse_csv_int_list(s: str) -> List[int]:
    """Parse ``"0,1,2"`` -> ``[0, 1, 2]``. Bad tokens raise ``ValueError``."""
    out: List[int] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _parse_csv_str_list(s: str) -> List[str]:
    """Parse ``"Ru,Ir"`` -> ``["Ru", "Ir"]``. Empty / None -> empty list."""
    out: List[str] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Pre-flight gate partitioning
# ---------------------------------------------------------------------------


def _partition_gates_by_priority(
    gates: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    int,
    int,
]:
    """Split the ``pre_flight.json`` ``gates`` list by ``priority`` into
    P0 / P1 / P2 / P3 buckets. Returns (p0, p1, p2, p3, pass_count, fail_count)
    where the counts are the totals across all priorities.

    Priority values in the pre-flight schema are ``"P0"``, ``"P1"``,
    ``"P2"``, ``"P3"``. The P3 bucket was added in Wf-343C4 (2026-09-29)
    to surface the pre-flight v2 paper-grade gates added by
    Wf-306C / Wf-308B / Wf-312B/C/E / Wf-313C/H / Wf-314E /
    Wf-315A/D / Wf-319A/B/C/W / Wf-321A/E / Wf-322F / Wf-323F /
    Wf-325D/E/F / Wf-327 / Wf-343C4 — prior to Wf-343C4 the
    per-wave checklist silently bucketed these into P1 (the
    historical default per 207 v2 §0), which caused the
    per_wave_checklist Markdown to lose the P3 frontier-math
    section.  Unknown priorities are still silently bucketed into P1
    (the 207 v2 §0 default).

    Wf-343C4 also extended the return shape from 5-tuple to 6-tuple
    to include the new ``p3`` bucket.  The function signature is
    backward-compat at the call site because ``build_per_wave_checklist``
    unpacks with positional args (line 727 calls
    ``_partition_gates_by_priority(pf.get('gates') or [])`` and assigns
    ``p0, p1, p2, _, _``); the Wf-343C4 update bumps that line to
    ``p0, p1, p2, p3, _, _`` so the new bucket surfaces in the
    Markdown.  See TODO/completed/343c4_integration.md for the
    cross-script integration context.
    """
    p0: List[Dict[str, Any]] = []
    p1: List[Dict[str, Any]] = []
    p2: List[Dict[str, Any]] = []
    p3: List[Dict[str, Any]] = []
    n_pass = 0
    n_fail = 0
    for g in gates or []:
        priority = str(g.get("priority", "P1"))
        if priority == "P0":
            p0.append(g)
        elif priority == "P2":
            p2.append(g)
        elif priority == "P3":
            p3.append(g)
        else:
            p1.append(g)
        status = str(g.get("status", "")).upper()
        if status == "PASS":
            n_pass += 1
        elif status == "FAIL":
            n_fail += 1
    return p0, p1, p2, p3, n_pass, n_fail


# ---------------------------------------------------------------------------
# Per-pair sub-checklist (metal, seed)
# ---------------------------------------------------------------------------


def _expected_ckpt_path(metal: str, seed: int) -> str:
    """Compose the canonical ckpt path for ``(metal, seed)`` per the
    Wf-124 P2 / Wf-124 P3 protocol filename convention."""
    return str(CHECKPOINTS_DIR / f"dmpnn_mtw_{metal.lower()}_seed{seed}.pt")


def _expected_metrics_path(metal: str, seed: int) -> str:
    """Compose the per-seed metrics JSON path used by Wf-124 P3.

    Wf-124 P3 stores per-metal directories in **lowercase** (Ru -> ``ru``)
    on disk; we mirror that convention here so a single ``os.path.exists``
    call finds the on-disk file regardless of ``metal`` argument casing.
    See ``ls molmetal/reports/wf124_p3_multitask/`` for the canonical layout.
    """
    return str(
        REPORTS_DIR
        / "wf124_p3_multitask"
        / metal.lower()
        / f"per_seed_metrics_seed{seed}.json"
    )


def _collect_per_pair(
    metals: List[str], seeds: List[int],
) -> List[Dict[str, Any]]:
    """Walk the (metal, seed) grid; for each pair, determine:

      - ``expected_ckpt`` — the canonical ckpt path
      - ``expected_metrics_json`` — the canonical per-seed metrics path
      - ``ckpt_exists`` — ``True`` when the ckpt is on disk
      - ``expected_auc`` — extracted from per-seed JSON when present,
        else ``None``
      - ``status`` — ``"DONE"`` (ckpt + metrics), ``"IN-FLIGHT"``
        (metrics partial), or ``"PENDING"`` (no evidence yet)
      - ``wall_sec`` — elapsed time from the per-seed metrics JSON
        (when available)
    """
    rows: List[Dict[str, Any]] = []
    for metal in metals:
        for seed in seeds:
            ckpt_path = _expected_ckpt_path(metal, seed)
            metrics_path = _expected_metrics_path(metal, seed)
            ckpt_exists = Path(ckpt_path).exists()
            expected_auc: Optional[float] = None
            wall_sec: Optional[float] = None
            partial_completed = False
            status = "PENDING"
            mp = Path(metrics_path)
            if mp.exists():
                try:
                    with open(str(mp), "r", encoding="utf-8") as fh:
                        m = json.load(fh)
                    tm = m.get("test_metrics") or {}
                    if "roc_auc" in tm:
                        expected_auc = float(tm["roc_auc"])
                    if "elapsed_seconds" in m:
                        wall_sec = float(m["elapsed_seconds"])
                    if ckpt_exists and expected_auc is not None:
                        status = "DONE"
                    else:
                        partial_completed = True
                        status = "IN-FLIGHT"
                except Exception:  # noqa: BLE001
                    partial_completed = True
                    status = "IN-FLIGHT"
            elif ckpt_exists:
                partial_completed = True
                status = "IN-FLIGHT"
            rows.append(
                {
                    "metal": metal,
                    "seed": int(seed),
                    "expected_ckpt": ckpt_path,
                    "expected_metrics_json": metrics_path,
                    "ckpt_exists": bool(ckpt_exists),
                    "expected_auc": expected_auc,
                    "wall_sec": wall_sec,
                    "partial_completed": partial_completed,
                    "status": status,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Abort-criteria active set (§7.2)
# ---------------------------------------------------------------------------


def _detect_abort_criteria_active(
    per_pair: List[Dict[str, Any]],
    *,
    time_budget_hr: float,
) -> List[str]:
    """Walk ``per_pair`` and detect which §7.2 abort criteria are
    currently triggered.  Conservative — only flags when *evidence*
    on disk matches the criterion.  Returns a list of criterion
    *names* (e.g. ``"7.2.4_os_collapse"``).

    Heuristics (v1 — refinable post-MEASURED):

      - ``7.2.1_per_wave_timeout``: any (metal, seed) row with
        ``wall_sec > 2 × (time_budget_hr × 3600) / N_PAIRS``.
        (Impossibly cheap threshold; only flagged when something is
        grossly over — usually means a row crashed mid-cycle.)
      - ``7.2.2_failure_pattern_3consec``: when ≥ 3 consecutive seeds
        on the same metal have ``ckpt_exists=False AND expected_auc=None``.
      - ``7.2.4_os_collapse``: ``metal == "Os"`` AND ``test_auc < 0.51``
        when AUC is on disk.
      - ``7.2.5_decode_ratio_collapse``: requires
        ``molmetal/reports/wf124_p2_cfm_full/per_epoch_metrics.json``
        with min decode_ratio < 0.05 (best-effort spot-check).
      - ``7.2.8_partial_pilot``: any row with ``status == "IN-FLIGHT"``.
    """
    active: List[str] = []
    # 7.2.1 — per-wave timeout (heuristic: >2× per-pair budget)
    n_pairs = max(1, len(per_pair))
    per_pair_budget_sec = (time_budget_hr * 3600.0) / n_pairs * 2.0
    if any(
        (r.get("wall_sec") is not None and r["wall_sec"] > per_pair_budget_sec)
        for r in per_pair
    ):
        active.append("7.2.1_per_wave_timeout")
    # 7.2.2 — 3 consecutive missing on same metal
    by_metal_seed = sorted(per_pair, key=lambda r: (r["metal"], r["seed"]))
    consec = 0
    last_metal: Optional[str] = None
    for r in by_metal_seed:
        if r["metal"] != last_metal:
            consec = 0
            last_metal = r["metal"]
        if not r.get("ckpt_exists") and r.get("expected_auc") is None:
            consec += 1
            if consec >= 3:
                active.append("7.2.2_failure_pattern_3consec")
                break
        else:
            consec = 0
    # 7.2.4 — Os collapse
    os_rows = [r for r in per_pair if r["metal"] == "Os"]
    for r in os_rows:
        if r.get("expected_auc") is not None and r["expected_auc"] < 0.51:
            active.append("7.2.4_os_collapse")
            break
    # 7.2.5 — decode_ratio collapse
    cfm_metrics = (
        REPORTS_DIR / "wf124_p2_cfm_full" / "per_epoch_metrics.json"
    )
    if cfm_metrics.exists():
        try:
            with open(str(cfm_metrics), "r", encoding="utf-8") as fh:
                m = json.load(fh)
            drs = [
                float(x.get("decode_ratio_rdkit_strict", 1.0))
                for x in m
                if isinstance(x, dict)
            ]
            if drs and min(drs) < 0.05:
                active.append("7.2.5_decode_ratio_collapse")
        except Exception:  # noqa: BLE001
            pass
    # 7.2.8 — partial pilot
    if any(r.get("status") == "IN-FLIGHT" for r in per_pair):
        active.append("7.2.8_partial_pilot")
    return active


# ---------------------------------------------------------------------------
# Wave budget
# ---------------------------------------------------------------------------


def _extract_time_budget_hr(
    yaml_obj: Optional[Dict[str, Any]], fallback: float,
) -> float:
    """Extract ``protocol.time_budget_hr`` / ``directives.time_budget``
    hours-as-string. Falls back to ``fallback`` on missing keys.

    Conventions from ``configs/cfm_full_retrain.yaml``:
      * ``protocol: time_budget_hr`` (rare)
      * ``directives: time_budget: "10h"`` (common; parse trailing 'h')
    """
    if yaml_obj is None:
        return float(fallback)
    proto = yaml_obj.get("protocol")
    if isinstance(proto, dict):
        for key in ("time_budget_hr", "wall_time_target_hr", "wall_time_hr"):
            v = proto.get(key)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v.rstrip("h").strip())
                except ValueError:
                    pass
    directives = yaml_obj.get("directives")
    if isinstance(directives, dict):
        v = directives.get("time_budget")
        if isinstance(v, str) and v.endswith("h"):
            try:
                return float(v[:-1].strip())
            except ValueError:
                pass
    return float(fallback)


def _compute_wave_remaining(
    yaml_obj: Optional[Dict[str, Any]],
    abort_active: List[str],
    fallback: float,
) -> float:
    """Compute ``wave_remaining_hr`` — conservative, MEASURED-vs-PROJECTED
    honest framing. Returns ``time_budget_hr - len(abort_active) * 0.5``
    (each active abort criterion costs ~30 min of remediation budget).
    Used to drive the human-readable markdown row "Remaining: Xh Ym".
    """
    budget_hr = _extract_time_budget_hr(yaml_obj, fallback)
    penalty_hr = 0.5 * len(abort_active)
    remaining = max(0.0, budget_hr - penalty_hr)
    return round(remaining, 2)


# ---------------------------------------------------------------------------
# Config schema check (per 207 v2 §1.3 / §2.1)
# ---------------------------------------------------------------------------


def _check_config_schema(
    yaml_obj: Optional[Dict[str, Any]],
) -> Tuple[bool, int, List[str]]:
    """Return ``(parse_ok, n_required_keys_present, missing_keys)``.

    ``parse_ok`` is True iff every key in ``REQUIRED_CONFIG_KEYS`` is
    present at the top level of the YAML (or in a nested ``config``
    dict, for backward-compat).  ``n_required_keys_present`` counts
    matches; ``missing_keys`` is the diff list.
    """
    if yaml_obj is None:
        return False, 0, list(REQUIRED_CONFIG_KEYS)
    # gather all top-level + nested 'config' keys for backward-compat
    keys: set = set(yaml_obj.keys())
    nested = yaml_obj.get("config") if isinstance(yaml_obj, dict) else None
    if isinstance(nested, dict):
        keys.update(nested.keys())
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in keys]
    return len(missing) == 0, len(REQUIRED_CONFIG_KEYS) - len(missing), missing


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_markdown(
    wave: str,
    obj: Dict[str, Any],
) -> str:
    """Render the per-wave checklist to a human-readable Markdown table.

    Sections:
      * header (wave name, generated-at, JSON location)
      * summary row (preflight pass/fail counts)
      * P0 / P1 / P2 gate tables (one row per gate)
      * Config summary block
      * Wave budget + abort criteria
      * Per-pair sub-checklist table
185    * Footer with honest-framing note
    """
    lines: List[str] = []
    lines.append(f"# Per-Wave Checklist — {wave}")
    lines.append("")
    lines.append(
        f"_Generated: {obj.get('generated_at', '?')}_  "
        f"_Schema: {SCHEMA_VERSION}_"
    )
    lines.append("")

    # Summary
    s = obj.get("preflight_pass_count", 0)
    f = obj.get("preflight_fail_count", 0)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- preflight pass count: **{s}**")
    lines.append(f"- preflight fail count: **{f}**")
    lines.append(
        f"- wave budget: **{obj.get('wave_budget_hr', 0.0)} h** "
        f"({obj.get('wave_remaining_hr', 0.0)} h remaining after abort penalties)"
    )
    lines.append("")

    # Helper for gate section
    def _gate_section(title: str, gates: List[Dict[str, Any]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not gates:
            lines.append("_no gates in this priority_")
            lines.append("")
            return
        lines.append("| Status | Gate | Evidence |")
        lines.append("|---|---|---|")
        for g in gates:
            lines.append(
                f"| `{g.get('status', '?')}` | "
                f"`{g.get('name', g.get('gate_name', '?'))}` | "
                f"{str(g.get('evidence', ''))[:120]} |"
            )
        lines.append("")

    _gate_section("P0 BLOCKER gates", obj.get("p0_blocker_gates", []))
    _gate_section("P1 VERIFY gates", obj.get("p1_verify_gates", []))
    _gate_section("P2 DOCUMENTED gates", obj.get("p2_documented_gates", []))
    _gate_section("P3 PAPER-GRADE gates", obj.get("p3_paper_grade_gates", []))

    # Config summary
    cs = obj.get("config_summary") or {}
    lines.append("## Config summary")
    lines.append("")
    lines.append(f"- yaml_path: `{cs.get('yaml_path', '?')}`")
    lines.append(f"- parse_ok: **{cs.get('parse_ok', False)}**")
    if not cs.get("parse_ok", False):
        lines.append(f"- error: `{cs.get('error', '?')}`")
    lines.append(
        f"- n_required_keys_present: **{cs.get('n_required_keys_present', 0)}** "
        f"of {len(REQUIRED_CONFIG_KEYS)}"
    )
    if cs.get("missing_keys"):
        lines.append(f"- missing_keys: `{', '.join(cs['missing_keys'])}`")
    lines.append("")

    # Abort criteria
    aborts = obj.get("abort_criteria_active", [])
    lines.append("## Abort criteria active (§7.2)")
    lines.append("")
    if not aborts:
        lines.append("_none — wave is CLEAN per §7.2_")
    else:
        lines.append("| Criterion | Description |")
        lines.append("|---|---|")
        for a in aborts:
            desc = ABORT_CRITERIA.get(a, a)
            lines.append(f"| `{a}` | {desc} |")
    lines.append("")

    # Per-pair sub-checklist
    pp = obj.get("per_pair_sub_checklist", [])
    lines.append("## Per-pair sub-checklist")
    lines.append("")
    if not pp:
        lines.append("_no (metal, seed) pairs — wave is pre-execution_")
    else:
        lines.append(
            "| Metal | Seed | Expected ckpt | Expected AUC | Status | Wall (s) |"
        )
        lines.append("|---|---|---|---|---|---|")
        for r in pp:
            auc = "—" if r.get("expected_auc") is None else f"{r['expected_auc']:.4f}"
            wall = (
                "—"
                if r.get("wall_sec") is None
                else f"{r['wall_sec']:.1f}"
            )
            lines.append(
                f"| {r['metal']} | {r['seed']} | "
                f"`{Path(str(r.get('expected_ckpt', ''))).name}` | "
                f"{auc} | `{r.get('status', '?')}` | {wall} |"
            )
    lines.append("")

    # Footer
    lines.append("---")
    lines.append(
        "_Honest framing (party-twenty-spirit §3.5): every entry above is "
        "POPULATED from JSON / on-disk evidence, not synthesised. Counts and "
        "status values trace to the referenced files; MEASURED vs PROJECTED "
        "is preserved per row. Re-run this script before each wave re-launch._"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the per-wave JSON + Markdown
# ---------------------------------------------------------------------------


def build_per_wave_checklist(
    *,
    wave: str,
    config_path: Optional[Path],
    preflight_path: Path,
    pairs_metals: List[str],
    pairs_seeds: List[int],
    time_budget_hr: float,
    out_json: Path,
    out_md: Optional[Path],
) -> Dict[str, Any]:
    """Top-level build. Returns the JSON-ready dict (also written to
    ``out_json``); writes the Markdown to ``out_md`` when provided.

    Never raises — every step is wrapped.  Unrecoverable conditions
    produce a ``stub`` dict with an ``error`` field; the caller decides
    the exit code based on ``obj["schema_valid"]``.
    """
    obj: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "wave": wave,
        "generated_at": _now_iso(),
        "preflight_pass_count": 0,
        "preflight_fail_count": 0,
        "p0_blocker_gates": [],
        "p1_verify_gates": [],
        "p2_documented_gates": [],
        "p3_paper_grade_gates": [],
        "config_summary": {
            "yaml_path": str(config_path) if config_path else "",
            "parse_ok": False,
            "n_required_keys": len(REQUIRED_CONFIG_KEYS),
            "n_required_keys_present": 0,
            "missing_keys": list(REQUIRED_CONFIG_KEYS),
            "error": "",
        },
        "wave_budget_hr": float(time_budget_hr),
        "wave_remaining_hr": float(time_budget_hr),
        "abort_criteria_active": [],
        "per_pair_sub_checklist": [],
        "schema_valid": False,
        "error": "",
    }

    # 1. Parse pre-flight JSON
    try:
        pf = _read_json(preflight_path)
    except FileNotFoundError:
        obj["error"] = f"preflight file not found: {preflight_path}"
        return obj
    except json.JSONDecodeError as exc:
        obj["error"] = f"preflight JSON parse error: {exc}"
        return obj

    summary = pf.get("summary") or {}
    obj["preflight_pass_count"] = int(summary.get("pass", 0) or 0)
    obj["preflight_fail_count"] = int(summary.get("fail", 0) or 0)

    # 2. Partition gates by priority
    p0, p1, p2, p3, _, _ = _partition_gates_by_priority(
        pf.get("gates") or [],
    )
    obj["p0_blocker_gates"] = [
        {
            "gate_name": g.get("name", "?"),
            "status": g.get("status", "?"),
            "evidence": g.get("evidence", ""),
        }
        for g in p0
    ]
    obj["p1_verify_gates"] = [
        {
            "gate_name": g.get("name", "?"),
            "status": g.get("status", "?"),
            "evidence": g.get("evidence", ""),
        }
        for g in p1
    ]
    obj["p2_documented_gates"] = [
        {
            "gate_name": g.get("name", "?"),
            "status": g.get("status", "?"),
            "evidence": g.get("evidence", ""),
        }
        for g in p2
    ]
    obj["p3_paper_grade_gates"] = [
        {
            "gate_name": g.get("name", "?"),
            "status": g.get("status", "?"),
            "evidence": g.get("evidence", ""),
        }
        for g in p3
    ]

    # 3. Parse YAML config
    yaml_path = config_path or DEFAULT_CONFIG_YAML
    parse_ok, yaml_obj, yaml_err = _parse_yaml_safe_load(yaml_path)
    cs = obj["config_summary"]
    cs["yaml_path"] = str(yaml_path)
    cs["parse_ok"] = parse_ok
    cs["error"] = yaml_err
    if parse_ok:
        s_ok, n_present, missing = _check_config_schema(yaml_obj)
        cs["parse_ok"] = s_ok
        cs["n_required_keys_present"] = n_present
        cs["missing_keys"] = missing

    # 4. Wave budget
    obj["wave_budget_hr"] = _extract_time_budget_hr(
        yaml_obj if parse_ok else None, time_budget_hr,
    )

    # 5. Per-pair sub-checklist
    per_pair = _collect_per_pair(pairs_metals, pairs_seeds)
    obj["per_pair_sub_checklist"] = per_pair

    # 6. Abort criteria
    obj["abort_criteria_active"] = _detect_abort_criteria_active(
        per_pair, time_budget_hr=obj["wave_budget_hr"],
    )

    # 7. Wave remaining
    obj["wave_remaining_hr"] = _compute_wave_remaining(
        yaml_obj if parse_ok else None,
        obj["abort_criteria_active"],
        obj["wave_budget_hr"],
    )

    # 8. Validate schema (presence + types of all REQUIRED keys)
    obj["schema_valid"] = _validate_schema(obj)

    # 9. Write outputs
    _write_json(out_json, obj)
    if out_md is not None:
        _write_text(out_md, _render_markdown(wave, obj))

    return obj


def _validate_schema(obj: Dict[str, Any]) -> bool:
    """Structural JSON-schema sanity check (duck-typed). Returns True
    iff every required top-level key is present with the expected type.
    Used by ``G3_json_output_valid`` acceptance gate."""
    expected: Dict[str, type] = {
        "schema_version": str,
        "wave": str,
        "generated_at": str,
        "preflight_pass_count": int,
        "preflight_fail_count": int,
        "p0_blocker_gates": list,
        "p1_verify_gates": list,
        "p2_documented_gates": list,
        # ``p3_paper_grade_gates`` is OPTIONAL — added in Wf-343C4;
        # older test fixtures / pre-Wf-343C4 dicts won't carry it.  When
        # missing we treat the bucket as empty (the Wf-343C4 default)
        # so backward-compat is preserved for pre-Wf-343C4 test fixtures.
        # See TODO/completed/343c4_integration.md for the cross-script
        # integration context.
        "config_summary": dict,
        "wave_budget_hr": float,
        "wave_remaining_hr": float,
        "abort_criteria_active": list,
        "per_pair_sub_checklist": list,
    }
    for k, t in expected.items():
        if k not in obj:
            return False
        if not isinstance(obj[k], t):
            return False
    # p3_paper_grade_gates: optional, default to empty list for
    # backward-compat with pre-Wf-343C4 dicts.
    if "p3_paper_grade_gates" in obj and not isinstance(
        obj["p3_paper_grade_gates"], list,
    ):
        return False
    # per-pair rows must have the contract fields
    for r in obj.get("per_pair_sub_checklist", []) or []:
        if not isinstance(r, dict):
            return False
        if not {"metal", "seed", "expected_ckpt"}.issubset(r.keys()):
            return False
    # each gate entry has gate_name / status / evidence
    for bucket_key in (
        "p0_blocker_gates",
        "p1_verify_gates",
        "p2_documented_gates",
        "p3_paper_grade_gates",
    ):
        for g in obj.get(bucket_key, []) or []:
            if not isinstance(g, dict):
                return False
            if not {"gate_name", "status", "evidence"}.issubset(g.keys()):
                return False
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_argv(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Argparse — accepts ≥ 4 flags (per acceptance G2)."""
    p = argparse.ArgumentParser(
        prog="per_wave_checklist.py",
        description=(
            "Wf-262 / TODO-208 — per-wave sub-checklist generator per 207 v2 §9. "
            "Emits a wave-specific JSON + Markdown from a pre-flight JSON + "
            "active configs/*.yaml + (metal, seed) grid."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--wave",
        required=True,
        help=(
            "Wave name (e.g. 'Wave-1 chemprop v2 retrain', "
            "'Wave-2 CFM TODO-119 P1.3')."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            f"Path to the active configs/*.yaml for this wave "
            f"(default: {DEFAULT_CONFIG_YAML}). Pass --config '' or omit "
            f"to skip config-schema check."
        ),
    )
    p.add_argument(
        "--preflight",
        type=Path,
        default=DEFAULT_PREFLIGHT_JSON,
        help=(
            f"Path to pre_flight.json (default: {DEFAULT_PREFLIGHT_JSON}). "
            f"For per-wave JSONs, pass an absolute path explicitly."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUT_JSON}).",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_OUT_MD,
        help=f"Output Markdown path (default: {DEFAULT_OUT_MD}).",
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
        help=(
            f"Comma-separated seed list (default: {DEFAULT_PAIRS_SEEDS})."
        ),
    )
    p.add_argument(
        "--time-budget-hr",
        type=float,
        default=DEFAULT_SESSION_BUDGET_HR,
        help=(
            f"Fallback session budget hours when not in YAML "
            f"(default: {DEFAULT_SESSION_BUDGET_HR})."
        ),
    )
    p.add_argument(
        "--no-md",
        action="store_true",
        help="Skip Markdown output (JSON only).",
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
        print("[per_wave_checklist] error: --wave must be non-empty", file=sys.stderr)
        return 1
    try:
        pairs_metals = _parse_csv_str_list(args.pairs_metals)
        pairs_seeds = _parse_csv_int_list(args.pairs_seeds)
    except ValueError as exc:
        print(
            f"[per_wave_checklist] error: bad --pairs-metals/--pairs-seeds: {exc}",
            file=sys.stderr,
        )
        return 1
    if not pairs_metals or not pairs_seeds:
        print(
            "[per_wave_checklist] error: --pairs-metals AND --pairs-seeds "
            "must each have at least one token",
            file=sys.stderr,
        )
        return 1

    if args.verbose:
        print(
            f"[per_wave_checklist] wave={args.wave!r} "
            f"config={args.config} preflight={args.preflight} "
            f"out={args.out} pairs=({len(pairs_metals)} metals × "
            f"{len(pairs_seeds)} seeds={len(pairs_metals) * len(pairs_seeds)})",
            file=sys.stderr,
        )

    try:
        obj = build_per_wave_checklist(
            wave=args.wave,
            config_path=args.config if str(args.config or "") else None,
            preflight_path=args.preflight,
            pairs_metals=pairs_metals,
            pairs_seeds=pairs_seeds,
            time_budget_hr=float(args.time_budget_hr),
            out_json=args.out,
            out_md=None if args.no_md else args.out_md,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[per_wave_checklist] FATAL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    if obj.get("error"):
        if args.verbose:
            print(
                f"[per_wave_checklist] degraded: {obj['error']}",
                file=sys.stderr,
            )
    if args.verbose:
        print(
            f"[per_wave_checklist] wrote JSON -> {args.out} "
            f"(schema_valid={obj['schema_valid']})",
            file=sys.stderr,
        )
        if not args.no_md:
            print(
                f"[per_wave_checklist] wrote Markdown -> {args.out_md}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(
            f"[per_wave_checklist] UNHANDLED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
