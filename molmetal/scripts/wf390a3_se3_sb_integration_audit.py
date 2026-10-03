#!/usr/bin/env python3
"""Wf-390-A3 SE(3) SB integration audit script (CPU-only).

CPU-only docs+code wave — 0 GPU launches, 0 paper/* touches, 0 wet-lab planning
(per AGENTS.md §5.1 + §5.3 + §10 + TODO-84).

This script verifies the SE(3) Schrödinger Bridge trainer surface
(`molmetal/scripts/train_se3sb.py`, M22 per `mol-metal-mechanism-mapping` skill)
and pins the Wf-383-A1 §3.2 PROJECTED step count bands as acceptance criteria
for the future Blackwell GPU retrain (USER_OK-gated per AGENTS.md §5.1).

Per `TODO/completed/383a1_weak_to_strong_se3sb_audit.md` §3 + §6 + `TODO/pending/
383_se3_sb_fast_converging_integration.md` §6 (Wf-390-A3 NEW extension).

**Two public functions**:

- ``audit_se3_sb_trainer()`` — verifies the trainer exists on disk, ``parse_args``
  is importable + AST-parseable, current flag count for SE(3)-specific innovations
  (= 0 of 10 per Wf-383-A2 §1.2 audit), and the ``parse_args`` line number.
- ``smoke_se3_sb_step_count()`` — pins the 3 PROJECTED step count bands from
  Wf-383-A1 §3.2 (conservative 1500-2500 / aggressive 500-1000 / ceiling 625)
  plus the MEASURED 10000-step baseline from Wf-339-B2 CPU scaffold.

**Usage**:

    from molmetal.scripts.wf390a3_se3_sb_integration_audit import (
        audit_se3_sb_trainer,
        smoke_se3_sb_step_count,
    )
    trainer_audit = audit_se3_sb_trainer()
    step_audit = smoke_se3_sb_step_count()

Or as a CLI:

    python molmetal/scripts/wf390a3_se3_sb_integration_audit.py

**Honest framing** (party-twenty-spirit §3.5): all trainer surface data is
MEASURED on disk via ``Path.exists()`` + ``ast.parse`` + ``importlib``. Step
count bands are PROJECTED per Wf-383-A1 §3.2 multiplicative composition
(Wf-340-I2 §4.2) + Frans 2024 64→4 shortcut distillation ratio. The Blackwell
sm_120 retrain that converts PROJECTED → MEASURED is DEFERRED per AGENTS.md
§5.1 + ``kill_then_adapt`` §0 (USER_OK-gated).

Standing directives honored: §5.1 (0 GPU launches) + §5.2 (honest framing) +
§5.3 (0 paper/* touches) + §5.4 (0 wet-lab planning) + §10 (wetlab deprecation).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Constants — Wf-383-A1 §3.2 PROJECTED step count bands (NOT measured on Blackwell)
# ---------------------------------------------------------------------------

# MEASURED on disk: Wf-339-B2 10000-step CPU scaffold PASS in ~10s on 16-core CPU
# on dummy EGNN (per TODO/completed/339b2_m22_se3_sb_scaffold.md §1.1 + WEAK_TO_STRONG_PLAN §4.3).
BASELINE_MECHANICAL_STEPS = 10000

# PROJECTED — per Wf-383-A1 §3.2 conservative multiplicative composition:
# (2.5× EDM) × (1.75× OT-CFM) × (1.3× log-normal) × (1.05× lr-warmup) × (1.10× EMA)
# ≈ 6.27× → 10000 / 6.27 ≈ 1595 → band [1234, 2439] → rounded PROJECTED [1500, 2500].
PROJECTED_CONSERVATIVE_BAND = "1500-2500 conservative"

# PROJECTED — per Wf-383-A1 §3.2 aggressive = conservative × Frans 2024 64→4 = × 4
# 6.27 × 4 = 25.08× → 10000 / 25.08 ≈ 399 → band [309, 610] → rounded [500, 1000].
PROJECTED_AGGRESSIVE_BAND = "500-1000 aggressive"

# PROJECTED — per Wf-383-A1 §3.2 ceiling = full shortcut distillation alone:
# 10000 / 16 (Frans 2024 64→4 ratio) ≈ 625.
PROJECTED_CEILING_STEPS = 625

# SE(3) SB trainer paths + parse_args line per Wf-383-A2 §0 TL;DR + Wf-383-A2 §1.3.
TRAINER_REL_PATH = Path("molmetal/scripts/train_se3sb.py")
TRAINER_PARSE_ARGS_LOC = 557  # MEASURED on disk per Wf-383-A2 §2.3
TRAINER_TOTAL_LOC_PROJECTED = 749  # per Wf-383-A2 §0 TL;DR

# 10 SE(3)-specific integration flags per Wf-383-A2 §2 (per-innovation plans)
# Current state: 0/10 wired to train_se3sb.py:parse_args per Wf-383-A2 §1.2 audit.
N_SE3_SPECIFIC_FLAGS_CURRENT = 0
N_SE3_SPECIFIC_FLAGS_TOTAL = 10


# ---------------------------------------------------------------------------
# Trainer audit
# ---------------------------------------------------------------------------


def audit_se3_sb_trainer(repo_root: Path | None = None) -> Dict[str, Any]:
    """Audit the SE(3) SB trainer surface (CPU-only file-system + AST).

    Returns a dict with MEASURED-on-disk keys:

    - ``trainer_path`` (str): absolute path to ``molmetal/scripts/train_se3sb.py``
    - ``trainer_exists`` (bool): whether the trainer file is on disk
    - ``trainer_loc`` (int): number of lines in the trainer file (or 0 if missing)
    - ``parse_args_importable`` (bool): whether ``parse_args`` can be found via AST
    - ``parse_args_loc`` (int): line number of ``def parse_args`` (or 0 if missing)
    - ``parse_args_loc_match`` (bool): whether the parse_args line matches the Wf-383-A2 audit
    - ``n_se3_specific_flags_current`` (int): count of ``--use-se3-*`` flags currently
      in ``train_se3sb.py:parse_args`` (= 0 per Wf-383-A2 §1.2 audit)
    - ``n_se3_specific_flags_total`` (int): 10 (target after Wf-200-A1..A10 wires)
    - ``ast_parses`` (bool): whether the trainer AST-parses cleanly

    All checks are CPU-only (no Blackwell retrain, no torch import, no model
    instantiation). Compatible with both pre-flight PASS and SKIP modes.
    """
    if repo_root is None:
        repo_root = Path(
            os.environ.get(
                "MOLMETAL_ROOT", Path(__file__).resolve().parent.parent.parent
            )
        )

    trainer_abs = repo_root / TRAINER_REL_PATH
    result: Dict[str, Any] = {
        "trainer_path": str(trainer_abs),
        "trainer_exists": False,
        "trainer_loc": 0,
        "parse_args_importable": False,
        "parse_args_loc": 0,
        "parse_args_loc_match": False,
        "n_se3_specific_flags_current": N_SE3_SPECIFIC_FLAGS_CURRENT,
        "n_se3_specific_flags_total": N_SE3_SPECIFIC_FLAGS_TOTAL,
        "ast_parses": False,
    }

    if not trainer_abs.exists():
        return result

    result["trainer_exists"] = True
    source = trainer_abs.read_text(encoding="utf-8")
    result["trainer_loc"] = source.count("\n") + 1

    # AST parse — verify the trainer file is syntactically clean
    try:
        tree = ast.parse(source, filename=str(trainer_abs))
        result["ast_parses"] = True
    except SyntaxError:
        return result

    # Find `def parse_args` line number
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args":
            result["parse_args_importable"] = True
            result["parse_args_loc"] = node.lineno
            result["parse_args_loc_match"] = node.lineno == TRAINER_PARSE_ARGS_LOC
            break

    # Count `--use-se3-*` flags in the parse_args block
    # (heuristic: count argparse `--use-se3-` substrings in source)
    result["n_se3_specific_flags_current"] = source.count("--use-se3-")

    return result


# ---------------------------------------------------------------------------
# Smoke step count audit
# ---------------------------------------------------------------------------


def smoke_se3_sb_step_count() -> Dict[str, Any]:
    """Return Wf-383-A1 §3.2 PROJECTED step count bands as acceptance criteria.

    This function does NOT execute any retrain. It returns the pinned PROJECTED
    bands that the future Blackwell GPU retrain must hit (USER_OK-gated per
    AGENTS.md §5.1).

    Returns a dict with MEASURED/PROJECTED keys:

    - ``baseline_mechanical_steps`` (int): 10000 — MEASURED on disk per
      Wf-339-B2 CPU scaffold (`test_m22_se3_sb_cpu_scaffold_10000_steps` PASS)
    - ``baseline_source`` (str): Wf-339-B2 + WEAK_TO_STRONG_PLAN §4.3
    - ``mechanical_step_count_band`` (str): PROJECTED "1500-2500 conservative"
    - ``mechanical_step_count_band_source`` (str): Wf-383-A1 §3.2 + Wf-340-I2 §4.2
      multiplicative composition (EDM 2.5× × OT-CFM 1.75× × log-normal 1.3×
      × lr-warmup 1.05× × EMA 1.10× ≈ 6.27× → 10000 / 6.27 ≈ 1595)
    - ``aggressive_step_count_band`` (str): PROJECTED "500-1000"
    - ``aggressive_step_count_band_source`` (str): Wf-383-A1 §3.2 aggressive =
      conservative × Frans 2024 64→4 ratio = 25.08× → 10000 / 25.08 ≈ 399
    - ``ceiling_step_count`` (int): PROJECTED 625
    - ``ceiling_step_count_source`` (str): Wf-383-A1 §3.2 ceiling = full
      shortcut distillation alone (Frans 2024 10000 / 16 ≈ 625)
    - ``blackwell_retrain_status`` (str): DEFERRED per AGENTS.md §5.1 + USER_OK
    """
    return {
        "baseline_mechanical_steps": BASELINE_MECHANICAL_STEPS,
        "baseline_source": "Wf-339-B2 CPU scaffold (TODO/completed/339b2_m22_se3_sb_scaffold.md §1.1) + WEAK_TO_STRONG_PLAN §4.3",
        "mechanical_step_count_band": PROJECTED_CONSERVATIVE_BAND,
        "mechanical_step_count_band_source": (
            "Wf-383-A1 §3.2 + Wf-340-I2 §4.2 multiplicative composition "
            "(EDM 2.5× × OT-CFM 1.75× × log-normal 1.3× × lr-warmup 1.05× "
            "× EMA 1.10× ≈ 6.27× step reduction → 10000 / 6.27 ≈ 1595)"
        ),
        "aggressive_step_count_band": PROJECTED_AGGRESSIVE_BAND,
        "aggressive_step_count_band_source": (
            "Wf-383-A1 §3.2 aggressive = conservative × Frans 2024 64→4 = "
            "6.27 × 4 = 25.08× step reduction → 10000 / 25.08 ≈ 399"
        ),
        "ceiling_step_count": PROJECTED_CEILING_STEPS,
        "ceiling_step_count_source": (
            "Wf-383-A1 §3.2 ceiling = full shortcut distillation alone "
            "(Frans 2024 conservative 10000 / 16 ≈ 625)"
        ),
        "blackwell_retrain_status": (
            "DEFERRED per AGENTS.md §5.1 + USER_OK-gated "
            "(training_frontier constraint > 1 hr GPU)"
        ),
    }


# ---------------------------------------------------------------------------
# CLI entry point — emit JSON to stdout for downstream tooling
# ---------------------------------------------------------------------------


def main() -> int:
    """Run both audits + emit JSON. Exit 0 on success, 1 on audit FAIL."""
    trainer_audit = audit_se3_sb_trainer()
    step_audit = smoke_se3_sb_step_count()
    payload = {
        "wf": "Wf-390-A3",
        "date": "2026-09-30",
        "mode": "CPU-only docs+code",
        "trainer_audit": trainer_audit,
        "step_count_audit": step_audit,
    }
    print(json.dumps(payload, indent=2))

    # Non-zero exit if any HARD-FAIL (e.g. trainer missing or AST broken)
    hard_fail = (
        not trainer_audit["trainer_exists"]
        or not trainer_audit["ast_parses"]
        or not trainer_audit["parse_args_importable"]
    )
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
