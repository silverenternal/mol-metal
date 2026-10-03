#!/usr/bin/env python3
"""Wf-381-A1 — m3 Vina docking parameter tuning script (follow-up wave).

================================================================
Why this script exists
================================================================
TODO-203 (m3 Vina docking tuning: exhaustiveness + n_poses) calls
for a 3-level Vina exhaustiveness ladder (low / medium / high) that
scales the standard Vina 1.2.7 budget bracket used by Wf-121 P3 /
Wf-143 baseline (ex=8) up to Wf-141 / Wf-142 lift target (ex=32)
and a Wf-380-A4 PROJECTED ceiling (ex=64).

This script is the Wf-381-A1 follow-up wave variant that RE-USES
the canonical Wf-380-A4 3-level config
(``molmetal/configs/vina_docking_params.yaml``) but provides a
clean re-implementation of the 3-function API:

1. ``tune_docking(cohort, level, config=None)`` — synthetic per-
   (pocket, seed) kcal/mol generator with the same Wf-141 §0
   PROJECTED lift curve; returns a dict with explicit MEASURED-vs-
   PROJECTED tag on the ``status`` field.
2. ``per_pocket_median_kcal(cohort, level)`` — thin wrapper.
3. ``aggregate_median(cohort_results)`` — overall verdict across
   levels (per-level cohort median + lift-vs-low + expected pass
   at industry cutoff + verdict tag).

Then it writes a smoke report at
``molmetal/reports/wf381a1_vina_tuning_smoke/result.json`` with
the canonical Wf-381-A1 schema.

HONEST FRAMING (party-twenty-spirit §3.5)
-----------------------------------------
* This script does NOT run real Vina docking.  The per-level
  median kcal/mol values are SYNTHETIC PLACEHOLDERS that mirror
  the Wf-141 §0 PROJECTED lift curve (low=-7.17, medium=-7.67,
  high=-8.17 kcal/mol cohort median).
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

    .venv/bin/python molmetal/scripts/wf381a1_vina_param_tuning.py \\
        --config molmetal/configs/vina_docking_params.yaml \\
        --out molmetal/reports/wf381a1_vina_tuning_smoke/result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("wf381a1_vina_param_tuning")

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

DEFAULT_CONFIG_PATH = ROOT / "molmetal" / "configs" / "vina_docking_params.yaml"
DEFAULT_OUT_PATH = (
    ROOT / "molmetal" / "reports" / "wf381a1_vina_tuning_smoke" / "result.json"
)

COHORT_POCKETS_DEFAULT: Tuple[str, ...] = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")
COHORT_SEEDS_DEFAULT: Tuple[int, ...] = (0, 1, 2)
CISPLATIN_SMILES = "Cl[Pt](Cl)(N)N"
INDUSTRY_CUTOFF_KCAL_MOL = -7.0

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


# ---------------------------------------------------------------------------
# Config loader (subset of Wf-380-A4 — sufficient for the smoke run)
# ---------------------------------------------------------------------------
def load_vina_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the 3-level Vina config from disk.

    The Wf-381-A1 script imports the Wf-380-A4 loader to keep schema
    parity.  Falls back to a stdlib-only minimal parser if the
    Wf-380-A4 module is not importable.
    """
    LOG.info("Loading Vina config from %s (Wf-381-A1)", config_path)
    try:
        from wf380a4_vina_param_tuning import load_vina_config as _load

        cfg = _load(config_path)
    except Exception:
        # Stdlib fallback (very minimal — sufficient for the canonical config)
        cfg = _minimal_yaml(config_path)
    for lvl in ("low", "medium", "high"):
        if lvl not in cfg.get("levels", {}):
            raise RuntimeError(
                f"Wf-381-A1 config missing level '{lvl}' at {config_path}"
            )
    return cfg


def _minimal_yaml(path: Path) -> Dict[str, Any]:
    """Minimal stdlib YAML loader sufficient for the canonical config."""
    raw = path.read_text(encoding="utf-8").splitlines()
    out: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, out)]
    for line in raw:
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = stripped.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if ":" not in content:
            continue
        key, _, value_str = content.partition(":")
        key = key.strip()
        value_str = value_str.strip()
        if value_str == "":
            new_dict: Dict[str, Any] = {}
            parent[key] = new_dict
            stack.append((indent, new_dict))
        elif value_str.startswith("[") and value_str.endswith("]"):
            items = [
                x.strip().strip("'\"") for x in value_str[1:-1].split(",") if x.strip()
            ]
            parent[key] = items
        else:
            try:
                parent[key] = int(value_str)
            except ValueError:
                try:
                    parent[key] = float(value_str)
                except ValueError:
                    parent[key] = value_str.strip("'\"")
    return out


# ---------------------------------------------------------------------------
# Synthetic kcal/mol generator
# ---------------------------------------------------------------------------
def _seed_jitter(pocket: str, seed: int) -> float:
    """Deterministic +/- SEED_JITTER_KCAL jitter per (pocket, seed)."""
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
    """Synthetic per-(pocket, seed) kcal/mol generator at `level`.

    Parameters
    ----------
    cohort : dict
        The cohort mapping with keys ``pockets`` and ``seeds``.
    level : str
        One of ``"low"``, ``"medium"``, ``"high"``.
    config : dict, optional
        The Wf-381-A1 parsed config (used for tag discipline only).

    Returns
    -------
    dict with keys:
        level, pockets, seeds, per_pocket_seed_kcal,
        per_pocket_median_kcal, cohort_median_kcal,
        status ("SYNTHETIC_PLACEHOLDER"), source,
        tag ("PROJECTED" / "MEASURED")
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
        vals = [round(base + offset + _seed_jitter(pocket, seed), 4) for seed in seeds]
        per_pocket_seed_kcal[pocket] = vals
        per_pocket_median_kcal[pocket] = round(statistics.median(vals), 4)
    cohort_median = round(statistics.median(per_pocket_median_kcal.values()), 4)
    return {
        "level": level,
        "pockets": pockets,
        "seeds": seeds,
        "per_pocket_seed_kcal": per_pocket_seed_kcal,
        "per_pocket_median_kcal": per_pocket_median_kcal,
        "cohort_median_kcal": cohort_median,
        "status": "SYNTHETIC_PLACEHOLDER",
        "source": "Wf-141 §0 PROJECTED lift model (Wf-381-A1)",
        "tag": "PROJECTED",
    }


def per_pocket_median_kcal(cohort: Dict[str, Any], level: str) -> Dict[str, float]:
    """Return per-pocket median kcal/mol at `level` (thin wrapper)."""
    return tune_docking(cohort, level)["per_pocket_median_kcal"]


def aggregate_median(cohort_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-level cohort_results into an overall verdict.

    Returns a dict with keys:
        per_level_cohort_median, overall_median_kcal,
        lift_vs_low, expected_pass_at_industry_cutoff_-7_0, verdict.
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
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def write_smoke_report(
    out_path: Path,
    config: Dict[str, Any],
    cohort_results: Dict[str, Dict[str, Any]],
    aggregate: Dict[str, Any],
) -> Path:
    """Write the Wf-381-A1 smoke report JSON to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "_workflow": "Wf-381-A1",
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
                "medium < low by ~-0.5 kcal/mol (Wf-141 PROJECTED lift)",
                "high < medium by ~-0.5 kcal/mol (Pareto frontier ceiling)",
            ],
            "do_not_claim": [
                "Real per-level kcal/mol as MEASURED until Vina binary actually docks",
                "m3 Vina Dock metric lift as MEASURED (only PROJECTED per Wf-141 §0)",
            ],
        },
        "_cross_references": [
            "TODO/pending/203_m3_vina_docking_param_tuning.md",
            "molmetal/scripts/wf381a1_vina_param_tuning.py",
            "molmetal/tests/test_wf381a1_vina_params.py",
            "molmetal/configs/vina_docking_params.yaml",
            "molmetal/reports/wf380a4_vina_tuning_smoke/result.json",
            "TODO/completed/380a4_vina_param_tuning.md",
        ],
    }
    # Wf-414-A3 migration: atomic write + safe_overwrite_check + detect_truncation.
    _payload = json.dumps(payload, indent=2)
    if _WF414_HAS_ATOMIC:
        _risk = safe_overwrite_check(out_path, _payload, min_lines=50)
        if _risk is not None:
            LOG.warning(
                "[Wf-381-A1] safe_overwrite_check ratio=%.3f at %s",
                _risk,
                out_path,
            )
        atomic_write_file(out_path, _payload, encoding="utf-8")
        _trunc = detect_truncation(_payload, expected_min_length=200)
        if _trunc is not None:
            LOG.warning(
                "[Wf-381-A1] detect_truncation reason=%r at %s",
                _trunc,
                out_path,
            )
    else:  # pragma: no cover
        out_path.write_text(_payload, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to Vina config YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"Path to smoke report JSON (default: {DEFAULT_OUT_PATH})",
    )
    parser.add_argument(
        "--level",
        choices=("low", "medium", "high"),
        default=None,
        help="Optional single-level mode (default: all 3 levels)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    config = load_vina_config(args.config)
    cohort = {
        "pockets": config.get("cohort", {}).get(
            "pockets", list(COHORT_POCKETS_DEFAULT)
        ),
        "seeds": config.get("cohort", {}).get("seeds", list(COHORT_SEEDS_DEFAULT)),
    }
    levels = (args.level,) if args.level else ("low", "medium", "high")
    cohort_results = {lvl: tune_docking(cohort, lvl, config) for lvl in levels}
    aggregate = aggregate_median(cohort_results)
    write_smoke_report(args.out, config, cohort_results, aggregate)
    LOG.info("Wf-381-A1 smoke report written to %s", args.out)
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
