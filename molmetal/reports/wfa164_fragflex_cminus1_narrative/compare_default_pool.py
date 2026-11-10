#!/usr/bin/env python3
"""Wf-A164 — compare the DEFAULT-pool artifact before and after the repair.

Reads
  OLD: ``molmetal/reports/wfa142_fragflex_cminus1/probe.json``
       (produced by the pre-repair tool; ``cohort_detail`` was a literal)
  NEW: ``molmetal/reports/wfa164_fragflex_cminus1_narrative/default_pool_probe.json``
       (produced by the repaired tool, same pool, same cohort, same seed)

Writes ``default_pool_diff.json``: a leaf-by-leaf diff, the narrative text
of both artifacts, and the verdict on whether the DEFAULT-pool artifact's
TEXT changes under the repaired tool.

CPU-only; reads two JSON files and writes one.  No GPU, no trainer.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wfa142_fragflex_cminus1_probe import (  # noqa: E402
    NARRATIVE_PATHS,
    _leaf_map,
    _resolve,
)

OLD = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wfa142_fragflex_cminus1"
    / "probe.json"
)
NEW = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wfa164_fragflex_cminus1_narrative"
    / "default_pool_probe.json"
)
OUT = Path(__file__).resolve().parent / "default_pool_diff.json"

#: Fields whose value is a wall-clock timestamp / duration: they differ
#: between any two runs and say nothing about the repair.
TIMING_FIELDS = {"generated_utc", "wall_seconds", "wall_seconds_total"}

#: The literals the pre-repair tool hard-coded (true for the DEFAULT pool).
STALE_LITERALS = ("704", "12,669", "18 epochs", "epoch 17")


def is_timing(path: str) -> bool:
    return path.rsplit(".", 1)[-1] in TIMING_FIELDS


def narrative(artifact) -> dict:
    """The artifact's prose leaves, resolved through NARRATIVE_PATHS."""
    leaves = _leaf_map(artifact)
    prose = set()
    for p in NARRATIVE_PATHS:
        prose.update(_resolve(artifact, p))
    return {
        p: leaves[p] for p in sorted(prose) if isinstance(leaves.get(p), str)
    }


def main() -> int:
    old = json.loads(OLD.read_text())
    new = json.loads(NEW.read_text())
    old_leaves = _leaf_map(old)
    new_leaves = _leaf_map(new)

    only_old = sorted(set(old_leaves) - set(new_leaves))
    only_new = sorted(set(new_leaves) - set(old_leaves))
    changed = []
    unchanged = 0
    for path in sorted(set(old_leaves) & set(new_leaves)):
        if is_timing(path):
            continue
        if old_leaves[path] != new_leaves[path]:
            changed.append(
                {
                    "path": path,
                    "old": old_leaves[path],
                    "new": new_leaves[path],
                }
            )
        else:
            unchanged += 1

    old_prose = narrative(old)
    new_prose = narrative(new)
    text_changed = {
        p: {"old": old_prose.get(p), "new": new_prose.get(p)}
        for p in sorted(set(old_prose) | set(new_prose))
        if old_prose.get(p) != new_prose.get(p)
    }
    new_prose_text = "\n".join(new_prose.values())

    numeric_changed = [c for c in changed if isinstance(c["old"], (int, float))
                       and not isinstance(c["old"], bool)]
    string_changed = [c for c in changed if isinstance(c["old"], str)]

    payload = {
        "artifact": "wfa164_default_pool_diff",
        "tag": "MEASURED",
        "wave": "Wf-A164",
        "question": (
            "Does the DEFAULT-pool artifact's TEXT change under the repaired "
            "tool?"
        ),
        "old_artifact": str(OLD),
        "new_artifact": str(NEW),
        "pool": new.get("pool"),
        "pool_sha256_old": old.get("pool_sha256"),
        "pool_sha256_new": new.get("pool_sha256"),
        "pool_sha256_identical": old.get("pool_sha256") == new.get("pool_sha256"),
        "cohort": new.get("cohort"),
        "seed_old": old.get("seed"),
        "seed_new": new.get("seed"),
        "n_rows_in_cohort_old": old.get("n_rows_in_cohort"),
        "n_rows_in_cohort_new": new.get("n_rows_in_cohort"),
        "n_leaf_paths_old": len(old_leaves),
        "n_leaf_paths_new": len(new_leaves),
        "n_leaf_paths_unchanged": unchanged,
        "n_leaf_paths_changed": len(changed),
        "n_numeric_leaves_changed": len(numeric_changed),
        "n_string_leaves_changed": len(string_changed),
        "changed_leaves": changed,
        "leaves_only_in_old": only_old,
        "leaves_only_in_new": only_new,
        "narrative_paths": list(NARRATIVE_PATHS),
        "narrative_old": old_prose,
        "narrative_new": new_prose,
        "narrative_text_changed": text_changed,
        "narrative_text_changed_count": len(text_changed),
        "stale_literals_in_new_narrative": {
            lit: (lit in new_prose_text) for lit in STALE_LITERALS
        },
        "stale_literals_in_old_narrative": {
            lit: (lit in "\n".join(old_prose.values()))
            for lit in STALE_LITERALS
        },
        "verdict": {
            "default_pool_text_changes": bool(text_changed),
            "default_pool_numeric_results_change": bool(numeric_changed),
            "cohort_detail_old": old.get("cohort_detail"),
            "cohort_detail_new": new.get("cohort_detail"),
            "cohort_detail_identical": old.get("cohort_detail")
            == new.get("cohort_detail"),
            "status_new": new.get("status"),
            "narrative_self_consistency_pass": new.get(
                "narrative_self_consistency", {}
            ).get("pass"),
            "cohort_detail_recheck_matches": new.get(
                "cohort_detail_recheck", {}
            ).get("matches"),
        },
    }
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(OUT, json.dumps(payload, indent=2, sort_keys=False))
    print(f"wrote {OUT}")
    print(
        "text_changed={} numeric_changed={} added={} removed={}".format(
            len(text_changed),
            len(numeric_changed),
            len(only_new),
            len(only_old),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
