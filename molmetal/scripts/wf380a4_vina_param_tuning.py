#!/usr/bin/env python3
"""Wf-380-A4 — m3 Vina docking parameter tuning script.

================================================================
Why this script exists
================================================================
TODO-203 (m3 Vina docking tuning: exhaustiveness + n_poses) calls
for a 3-level Vina exhaustiveness ladder (low / medium / high) that
scales the standard Vina 1.2.7 budget bracket used by Wf-121 P3 /
Wf-143 baseline (ex=8) up to Wf-141 / Wf-142 lift target (ex=32)
and a Wf-380-A4 PROJECTED ceiling (ex=64).

This script:

1. Loads the 3-level config from
   ``molmetal/configs/vina_docking_params.yaml``.
2. Provides ``tune_docking(cohort, level)`` — a stub that simulates
   a per-(pocket, seed) Vina dock at the requested exhaustiveness
   level.  It returns a deterministic synthetic kcal/mol per pocket
   (NOT a real Vina dock — real docking is BLOCKED per TODO-207
   and AGENTS.md §5.4 until the user OK = 1 grant is given).
3. Provides ``per_pocket_median_kcal(cohort, level)`` — median
   kcal/mol across (pocket x seeds) for a single level.
4. Provides ``aggregate_median(cohort_results)`` — overall cohort
   median across all levels.
5. Emits ``molmetal/reports/wf380a4_vina_tuning_smoke/result.json``
   with the per-level median kcal/mol + the smoke-validation verdict.

The smoke values follow the Wf-141 §0 PROJECTED lift model:

  low    -> -7.0 kcal/mol (canonical Wf-121 / Wf-143 baseline)
  medium -> -7.5 kcal/mol (-0.5 lift vs low; Wf-141 target)
  high   -> -8.0 kcal/mol (-1.0 lift vs low; Wf-380-A4 ceiling)

HONEST FRAMING (party-twenty-spirit §3.5)
-----------------------------------------
* This script does NOT run real Vina docking.  The per-level
  median kcal/mol values are SYNTHETIC PLACEHOLDERS that mirror
  the Wf-141 §0 PROJECTED lift curve.
* Real Vina docking requires the Vina 1.2.7 binary + pocket
  PDBQT receptors on disk + USER_OK = 1 grant per AGENTS.md §5.4.
  The Wf-299 CFM 3-wave launch (TODO-207) is the canonical
  parent workflow for real validation.
* Pt+2 UFF substitution is delegated to Wf-325B MEASURED_GENERIC_ATOM
  path; this script does not attempt the GPU metal-coordination
  term (BLOCKED per Wf-299B / AGENTS.md §5.4).

Usage
-----
::

    .venv/bin/python molmetal/scripts/wf380a4_vina_param_tuning.py \\
        --config molmetal/configs/vina_docking_params.yaml \\
        --out molmetal/reports/wf380a4_vina_tuning_smoke/result.json

CLI flags
---------
- ``--config <path>`` : path to the YAML config (default = canonical path).
- ``--out <path>`` : path to the result JSON (default = canonical path).
- ``--level {low,medium,high}`` : optional — run a single level only.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("wf380a4_vina_param_tuning")

# Wf-414-A3 migration: atomic write + safe_overwrite_check + detect_truncation
# closes the half-merged-file failure mode for the result.json Write site
# (per AGENTS.md §5.5/§5.6 standing directives). The shared atomic_write_file
# primitive at molmetal/scripts/wf386a3_atomic_write.py is the canonical
# I/O substrate for all closeout-drafting scripts (Wf-386-A3 §6.1 adoption
# sequence step 4).
try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )
    _WF414_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF414_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = (
    ROOT / "molmetal" / "configs" / "vina_docking_params.yaml"
)
DEFAULT_OUT_PATH = (
    ROOT / "molmetal" / "reports" / "wf380a4_vina_tuning_smoke" / "result.json"
)

# Canonical 5-pocket Wf-143 / Wf-325B cohort.
COHORT_POCKETS_DEFAULT: Tuple[str, ...] = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")
COHORT_SEEDS_DEFAULT: Tuple[int, ...] = (0, 1, 2)
CISPLATIN_SMILES = "Cl[Pt](Cl)(N)N"
INDUSTRY_CUTOFF_KCAL_MOL = -7.0


# ---------------------------------------------------------------------------
# Minimal YAML loader (no PyYAML dependency required at script runtime).
# This is a SHALLOW loader sufficient for the canonical config layout —
# if the config gains nested maps beyond the documented schema, switch
# to ``import yaml``.  Kept as stdlib-only for CPU-only environments
# where ``pyyaml`` may be missing.
# ---------------------------------------------------------------------------
def _load_yaml_simple(path: Path) -> Dict[str, Any]:
    """Minimal YAML loader for the canonical vina_docking_params.yaml schema.

    Supports:
      - top-level ``key: value`` mappings
      - ``key:`` followed by indented children (2-space indent)
      - inline ``[a, b, c]`` lists
      - inline ``- a - b`` lists
      - scalar types: int, float, str, bool, None

    Anything fancier falls back to parsing as a string.  This is
    intentional — the Wf-380-A4 config is hand-authored and flat.
    """
    raw = path.read_text(encoding="utf-8")
    out: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, out)]
    for line_no, raw_line in enumerate(raw.splitlines(), 1):
        # Skip blank lines and full-line comments.
        stripped = raw_line.split("#", 1)[0].rstrip()
        if not stripped:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        content = stripped.strip()
        # Pop stack until we are at the correct parent indent.
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else out
        if content.startswith("- "):
            # Inline list item — append to most-recent key on parent.
            value_str = content[2:].strip()
            value = _parse_scalar(value_str)
            # find the most-recent key with a list under it
            if isinstance(parent, list):
                parent.append(value)
                stack.append((indent, {}))  # placeholder for children
            continue
        if ":" in content:
            key, _, value_str = content.partition(":")
            key = key.strip()
            value_str = value_str.strip()
            if value_str == "":
                # nested map or list begins on next lines
                new_dict: Dict[str, Any] = {}
                parent[key] = new_dict
                stack.append((indent, new_dict))
            elif value_str.startswith("[") and value_str.endswith("]"):
                items = [_parse_scalar(x.strip()) for x in value_str[1:-1].split(",") if x.strip()]
                parent[key] = items
            else:
                parent[key] = _parse_scalar(value_str)
    return out


def _parse_scalar(text: str) -> Any:
    """Parse a single scalar token from the YAML config."""
    text = text.strip()
    if not text:
        return ""
    if text in ("true", "True", "TRUE", "yes"):
        return True
    if text in ("false", "False", "FALSE", "no"):
        return False
    if text in ("null", "None", "~"):
        return None
    # quoted string
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    # int
    try:
        return int(text)
    except ValueError:
        pass
    # float
    try:
        val = float(text)
        return val
    except ValueError:
        pass
    return text


def load_vina_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the 3-level Vina config from disk.

    Returns the parsed YAML dict with keys ``vina_engine``, ``levels``,
    ``cohort``, ``smoke_validation``, ``honest_framing``,
    ``cross_references``.
    """
    LOG.info("Loading Vina config from %s", config_path)
    cfg = _load_yaml_simple(config_path)
    if "levels" not in cfg or not isinstance(cfg["levels"], dict):
        raise RuntimeError(
            f"Wf-380-A4 config missing 'levels' mapping at {config_path}"
        )
    required_levels = ("low", "medium", "high")
    for lvl in required_levels:
        if lvl not in cfg["levels"]:
            raise RuntimeError(
                f"Wf-380-A4 config missing level '{lvl}' at {config_path}"
            )
    LOG.info("Config loaded with %d levels", len(cfg["levels"]))
    return cfg


# ---------------------------------------------------------------------------
# Per-(pocket, seed) synthetic kcal/mol generator
# ---------------------------------------------------------------------------
# Per Wf-141 §0 + Wf-380-A4 PROJECTED lift curve, the synthetic
# kcal/mol per (pocket, seed) at level `level_name` is:
#
#   base[level] + per_pocket_offset[pocket] + seed_jitter
#
# where:
#   base[low]    = -7.0  (Wf-121 / Wf-143 baseline)
#   base[medium] = -7.5  (Wf-141 lift target)
#   base[high]   = -8.0  (Wf-380-A4 PROJECTED ceiling)
#
# Per-pocket offsets mirror the Wf-143 MEASURED per-pocket ordering:
#   1h36 (most negative): -0.30 kcal/mol
#   3eml: -0.05
#   3ny8 (most negative): -0.40
#   4aaw: -0.10
#   4rlu: -0.15
#
# Seed jitter is +/- 0.02 kcal/mol (deterministic, hash-stable) to
# capture the standard Vina Monte-Carlo noise floor.
# ---------------------------------------------------------------------------
BASE_KCAL_PER_LEVEL: Dict[str, float] = {
    "low": -7.0,
    "medium": -7.5,
    "high": -8.0,
}
PER_POCKET_OFFSET_KCAL: Dict[str, float] = {
    "1h36": -0.30,
    "3eml": -0.05,
    "3ny8": -0.40,
    "4aaw": -0.10,
    "4rlu": -0.15,
}
SEED_JITTER_KCAL = 0.02


def _seed_jitter(pocket: str, seed: int) -> float:
    """Deterministic +/- SEED_JITTER_KCAL jitter per (pocket, seed).

    Uses ``hashlib.blake2b`` (NOT Python's built-in ``hash()`` which is
    non-deterministic across processes due to PYTHONHASHSEED randomization
    — see PEP 456).  blake2b is hash-stable across processes, so the
    smoke report JSON is bit-for-bit reproducible across runs.
    """
    import hashlib
    h_bytes = hashlib.blake2b(
        f"{pocket}|{seed}".encode("utf-8"), digest_size=4
    ).digest()
    h = int.from_bytes(h_bytes, "big")
    sign = -1.0 if (h % 2 == 0) else 1.0
    return sign * SEED_JITTER_KCAL


def tune_docking(
    cohort: Dict[str, Any],
    level: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the synthetic per-(pocket, seed) kcal/mol generator at `level`.

    Parameters
    ----------
    cohort : dict
        The cohort mapping with keys ``pockets`` (list[str]) and
        ``seeds`` (list[int]).  Defaults to the canonical Wf-143
        5-pocket × 3-seed cohort if not provided.
    level : str
        One of ``"low"``, ``"medium"``, ``"high"``.

    Returns
    -------
    dict with keys:
        level: str
        pockets: list[str]
        seeds: list[int]
        per_pocket_seed_kcal: dict[pocket -> list[float]]
        per_pocket_median_kcal: dict[pocket -> float]
        cohort_median_kcal: float
        status: str  ("SYNTHETIC_PLACEHOLDER")
        source: str  ("Wf-141 §0 PROJECTED lift model")
    """
    if level not in BASE_KCAL_PER_LEVEL:
        raise ValueError(
            f"Unknown level '{level}'; expected one of {list(BASE_KCAL_PER_LEVEL)}"
        )
    pockets = list(cohort.get("pockets", COHORT_POCKETS_DEFAULT))
    seeds = list(cohort.get("seeds", COHORT_SEEDS_DEFAULT))
    base = BASE_KCAL_PER_LEVEL[level]
    per_pocket_seed_kcal: Dict[str, List[float]] = {}
    per_pocket_median_kcal: Dict[str, float] = {}
    for pocket in pockets:
        offset = PER_POCKET_OFFSET_KCAL.get(pocket, 0.0)
        vals = [
            round(base + offset + _seed_jitter(pocket, seed), 4) for seed in seeds
        ]
        per_pocket_seed_kcal[pocket] = vals
        per_pocket_median_kcal[pocket] = round(statistics.median(vals), 4)
    cohort_median = round(
        statistics.median(per_pocket_median_kcal.values()), 4
    )
    return {
        "level": level,
        "pockets": pockets,
        "seeds": seeds,
        "per_pocket_seed_kcal": per_pocket_seed_kcal,
        "per_pocket_median_kcal": per_pocket_median_kcal,
        "cohort_median_kcal": cohort_median,
        "status": "SYNTHETIC_PLACEHOLDER",
        "source": "Wf-141 §0 PROJECTED lift model (Wf-380-A4 smoke)",
    }


def per_pocket_median_kcal(
    cohort: Dict[str, Any], level: str
) -> Dict[str, float]:
    """Return per-pocket median kcal/mol at `level` (thin wrapper)."""
    return tune_docking(cohort, level)["per_pocket_median_kcal"]


def aggregate_median(cohort_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-level cohort_results into an overall verdict.

    Parameters
    ----------
    cohort_results : dict
        Mapping from level name (low / medium / high) to the dict
        produced by ``tune_docking(...)``.

    Returns
    -------
    dict with keys:
        per_level_cohort_median: dict[level -> float]
        overall_median_kcal: float (median across all level medians)
        lift_vs_low: dict[level -> float]   (negative = more binding)
        expected_pass_at_industry_cutoff: dict[level -> bool]
        verdict: str ("MEASURED" / "PROJECTED" / "HONEST_NEGATIVE")
    """
    per_level_median = {
        lvl: res["cohort_median_kcal"] for lvl, res in cohort_results.items()
    }
    overall_median = round(statistics.median(per_level_median.values()), 4)
    lift = {
        lvl: round(per_level_median[lvl] - per_level_median["low"], 4)
        for lvl in per_level_median
    }
    expected_pass = {
        lvl: per_level_median[lvl] <= INDUSTRY_CUTOFF_KCAL_MOL
        for lvl in per_level_median
    }
    return {
        "per_level_cohort_median": per_level_median,
        "overall_median_kcal": overall_median,
        "lift_vs_low": lift,
        "expected_pass_at_industry_cutoff_-7_0": expected_pass,
        "verdict": "SYNTHETIC_PLACEHOLDER",
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def write_smoke_report(
    out_path: Path,
    config: Dict[str, Any],
    cohort_results: Dict[str, Dict[str, Any]],
    aggregate: Dict[str, Any],
) -> Path:
    """Write the smoke report JSON to `out_path`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "_workflow": "Wf-380-A4",
        "_phase": "P1",
        "_title": "m3 Vina docking parameter tuning — 3-level smoke (CPU-only)",
        "_generated_utc": _utc_now_iso(),
        "_hardware": "CPU (no Vina binary launch; synthetic per-level placeholder)",
        "_config_path": str(DEFAULT_CONFIG_PATH),
        "_n_pockets": len(cohort_results["low"]["pockets"]),
        "_n_seeds": len(cohort_results["low"]["seeds"]),
        "_smiles": CISPLATIN_SMILES,
        "_industry_cutoff_kcal_mol": INDUSTRY_CUTOFF_KCAL_MOL,
        "_cohort": cohort_results["low"]["pockets"],
        "_per_level": cohort_results,
        "_aggregate": aggregate,
        "_honest_framing": {
            "status": "SYNTHETIC_PLACEHOLDER",
            "real_validation_blocker": (
                "Vina binary + pocket PDBQT + Wf-299 CFM 3-wave GPU launch "
                "(TODO-207, USER_OK-gated per AGENTS.md §5.1 / §5.4)"
            ),
            "per_level_lift_curve_source": "Wf-141 §0 PROJECTED lift model",
            "claims": [
                "3-level config covers standard Vina 1.2.7 budget bracket",
                "low (ex=8) preserves backward compat with Wf-121 / Wf-143",
                "medium (ex=32) matches Wf-141 / Wf-142 lift target",
                "high (ex=64) is Wf-380-A4 PROJECTED ceiling",
            ],
            "do_not_claim": [
                "Real per-level kcal/mol as MEASURED until Vina binary docks",
                "m3 Vina Dock metric lift as MEASURED",
                "Pt+2 kcal/mol as MEASURED (BLOCKED per Wf-301B / Wf-325B)",
            ],
        },
        "_cross_references": [
            "TODO/completed/380a4_vina_param_tuning.md",
            "molmetal/configs/vina_docking_params.yaml",
            "molmetal/tests/test_wf380a4_vina_params.py",
            "TODO/pending/207_wf299_cfm_3wave_launch.md",
            "TODO/completed/141_vina_exhaustiveness_8_to_32_lift.md",
            "TODO/completed/325b_vina_pt_zn_substitution.md",
        ],
    }
    # Wf-414-A3 migration: atomic write + safe_overwrite_check + detect_truncation.
    _payload = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if _WF414_HAS_ATOMIC:
        _risk = safe_overwrite_check(out_path, _payload, min_lines=50)
        if _risk is not None:
            LOG.warning(
                "[Wf-380-A4] safe_overwrite_check ratio=%.3f at %s",
                _risk, out_path,
            )
        atomic_write_file(out_path, _payload, encoding="utf-8")
        _trunc = detect_truncation(_payload, expected_min_length=200)
        if _trunc is not None:
            LOG.warning(
                "[Wf-380-A4] detect_truncation reason=%r at %s",
                _trunc, out_path,
            )
    else:  # pragma: no cover
        out_path.write_text(_payload, encoding="utf-8")
    LOG.info("Wrote smoke report to %s", out_path)
    return out_path


def _utc_now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wf-380-A4: 3-level Vina docking parameter tuning (smoke)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to vina_docking_params.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"Path to result JSON (default: {DEFAULT_OUT_PATH})",
    )
    parser.add_argument(
        "--level",
        choices=("low", "medium", "high"),
        default=None,
        help="If given, run only this level (default: all 3 levels)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_vina_config(args.config)
    cohort = config.get("cohort", {})
    cohort_pockets = cohort.get("pockets", list(COHORT_POCKETS_DEFAULT))
    cohort_seeds = cohort.get("seeds", list(COHORT_SEEDS_DEFAULT))
    cohort_dict = {"pockets": cohort_pockets, "seeds": cohort_seeds}

    levels_to_run = (args.level,) if args.level else ("low", "medium", "high")
    cohort_results: Dict[str, Dict[str, Any]] = {
        lvl: tune_docking(cohort_dict, lvl, config) for lvl in levels_to_run
    }
    aggregate = aggregate_median(cohort_results)

    out_path = write_smoke_report(args.out, config, cohort_results, aggregate)

    LOG.info(
        "Per-level cohort median kcal/mol: %s",
        aggregate["per_level_cohort_median"],
    )
    LOG.info("Overall median: %.4f kcal/mol", aggregate["overall_median_kcal"])
    LOG.info("Lift vs low: %s", aggregate["lift_vs_low"])
    LOG.info("Smoke report written to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())