#!/usr/bin/env python3
"""FIX-E1 — one scoreboard, two splits, no exceptions.

Every MSRS arm is a ligand-level predictor evaluated with GroupKFold.  Which
column defines a group changes the answer: on the *same* arm, grouping by
``ligand_main_fragment`` scored 0.3366 while grouping by ``ligand_canonical``
scored 0.3972 — 0.0606 of pure split artifact.  The pre-audit scoreboard only
ever reported the fragment-grouped number, and because 5,861 canonical
complexes collapse onto 2,846 main fragments, that is the *optimistic* of the
two when the deployment unit is the complex.

So a report that carries only one split is not a weaker result, it is an
uninterpretable one.  This module makes that a hard error:

* :func:`validate_payload` returns every reason a report is not dual-split.
* :func:`write_report` refuses to write an artifact that fails validation.
* ``python -m molmetal.scripts.msrs_dual_split`` re-checks artifacts on disk
  and is wired into pre-flight as ``p3_msrs_dual_split_check``.

Artifacts opt in by being listed in :data:`DUAL_SPLIT_ARTIFACTS` or by
carrying ``"requires_dual_split": true`` at the top level.  Historical
artifacts written before this rule (decision policy, variance decomposition,
…) are out of scope: they either do not report per-arm R2 at all or report a
single protocol by design.

Usage::

    .venv/bin/python -m molmetal.scripts.msrs_dual_split
    .venv/bin/python -m molmetal.scripts.msrs_dual_split --metrics-dir metrics/by_metric
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

#: Grouping column is the main fragment (the old, optimistic protocol).
SPLIT_FRAGMENT_GROUPED = "fragment_grouped"
#: Grouping column is the full complex (the deployment unit).
SPLIT_CANONICAL_GROUPED = "canonical_grouped"

#: Both must be present, always, in this order.
REQUIRED_SPLITS: tuple[str, ...] = (SPLIT_FRAGMENT_GROUPED, SPLIT_CANONICAL_GROUPED)

#: Legacy spellings accepted on read, normalised on write.
SPLIT_ALIASES: Dict[str, str] = {
    "fragment": SPLIT_FRAGMENT_GROUPED,
    "fragment_grouped": SPLIT_FRAGMENT_GROUPED,
    "main_fragment_grouped": SPLIT_FRAGMENT_GROUPED,
    "canonical": SPLIT_CANONICAL_GROUPED,
    "canonical_grouped": SPLIT_CANONICAL_GROUPED,
    "complex_grouped": SPLIT_CANONICAL_GROUPED,
    "ligand_canonical_grouped": SPLIT_CANONICAL_GROUPED,
}

#: Artifacts that MUST be dual-split.  New arm scripts append their name here
#: in the same commit that introduces them — that is the point of the registry.
DUAL_SPLIT_ARTIFACTS: tuple[str, ...] = (
    "msrs_complex_features",
    "msrs_metal_block",
    "msrs_coordination",
    "msrs_dmpnn_surface",
)


class DualSplitError(ValueError):
    """Raised when an artifact is written without both splits."""


def normalize_splits(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the split table under canonical keys.

    Accepts ``payload["splits"]`` (preferred) or split tables at the top
    level.  Legacy keys (``fragment`` / ``canonical``) are renamed.
    """
    raw = payload.get("splits")
    if not isinstance(raw, Mapping):
        raw = {
            k: payload[k]
            for k in list(SPLIT_ALIASES)
            if isinstance(payload.get(k), Mapping)
        }
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        out[SPLIT_ALIASES.get(str(key), str(key))] = value
    return out


def _arm_names(table: Any) -> List[str]:
    return sorted(table.keys()) if isinstance(table, Mapping) else []


def validate_payload(
    payload: Mapping[str, Any], *, path: Optional[Path] = None
) -> List[str]:
    """Every reason ``payload`` is not a usable dual-split report.

    Returns an empty list when the report is valid.  The checks are ordered so
    the first message is the most actionable one.
    """
    where = f"{path}: " if path is not None else ""
    splits = normalize_splits(payload)
    problems: List[str] = []

    missing = [s for s in REQUIRED_SPLITS if s not in splits]
    if missing:
        problems.append(
            f"{where}missing split(s) {missing}; a single-split R2 is not "
            f"interpretable (the same arm moves ~0.06 between protocols)"
        )
        return problems

    for split in REQUIRED_SPLITS:
        table = splits[split]
        if not isinstance(table, Mapping) or not table:
            problems.append(f"{where}split '{split}' has no entries")

    if problems:
        return problems

    # Two report shapes exist: an arm table (``{arm: {r2: ...}}``, the feature
    # A/Bs) and a protocol report (decision-layer metrics keyed by block name).
    # Enforce the arm contract only when the payload actually looks like one;
    # a protocol report is still required to carry both splits, which the
    # presence check above already guarantees.
    arm_like = any(
        isinstance(entry, Mapping) and "r2" in entry
        for table in (splits[s] for s in REQUIRED_SPLITS)
        for entry in table.values()
    )
    if not arm_like:
        return []

    for split in REQUIRED_SPLITS:
        for arm, entry in splits[split].items():
            if not isinstance(entry, Mapping):
                problems.append(f"{where}{split}/{arm}: entry is not a mapping")
            elif "r2" not in entry:
                problems.append(f"{where}{split}/{arm}: no 'r2'")
            elif not isinstance(entry["r2"], (int, float)):
                problems.append(f"{where}{split}/{arm}: 'r2' is not numeric")

    if problems:
        return problems

    frag_arms = _arm_names(splits[SPLIT_FRAGMENT_GROUPED])
    canon_arms = _arm_names(splits[SPLIT_CANONICAL_GROUPED])
    if frag_arms != canon_arms:
        problems.append(
            f"{where}arm sets differ between splits: "
            f"fragment_grouped={frag_arms} canonical_grouped={canon_arms}"
        )
    return problems


def requires_dual_split(payload: Mapping[str, Any]) -> bool:
    """Is this artifact in scope for the gate?"""
    if payload.get("requires_dual_split") is True:
        return True
    name = payload.get("artifact")
    return isinstance(name, str) and name in DUAL_SPLIT_ARTIFACTS


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    """Validate then atomically write.  Raises :class:`DualSplitError`."""
    problems = validate_payload(payload, path=Path(path))
    if problems:
        raise DualSplitError("; ".join(problems))
    atomic_write_file(
        path, json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n"
    )


def check_directory(metrics_dir: Path) -> tuple[int, List[str], List[str]]:
    """Validate every in-scope artifact in ``metrics_dir``.

    Returns ``(n_checked, passed, failed)``.
    """
    checked = 0
    passed: List[str] = []
    failed: List[str] = []
    for path in sorted(Path(metrics_dir).glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping) or not requires_dual_split(payload):
            continue
        checked += 1
        problems = validate_payload(payload, path=path)
        if problems:
            failed.extend(problems)
        else:
            passed.append(path.name)
    return checked, passed, failed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    checked, passed, failed = check_directory(args.metrics_dir)
    if args.json:
        print(
            json.dumps(
                {"checked": checked, "passed": passed, "failed": failed},
                indent=2,
            )
        )
    else:
        for name in passed:
            print(f"PASS  {name}")
        for problem in failed:
            print(f"FAIL  {problem}")

    if failed:
        return 1
    if checked == 0:
        print("SKIP  no dual-split artifacts found")
        return 2
    print(f"PASS  {checked} dual-split artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
