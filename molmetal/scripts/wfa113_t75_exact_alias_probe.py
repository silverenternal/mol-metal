#!/usr/bin/env python
"""Wf-A113 (T75) — is the bond "exact" pair a decomposition or a duplicate?

``evaluate_bond_accuracy`` emits both ``bond_exact_match`` and
``bond_exact_type_rate``.  Until T75 the two were incremented in the SAME
branch under the SAME condition, so every archived artefact carried one number
under two names — and a run could not say whether it had failed on
connectivity or on bond order.

This probe reads every ``per_epoch_metrics.json`` under ``results/`` and
``molmetal/reports/`` and counts how often the two keys agree.  It also reads
the bench's definition so the two vocabularies are stated side by side:

* bench ``exact``       = bond SET only  -> trainer ``bond_set_exact_rate``
* bench ``exact_type``  = set + order    -> trainer ``bond_exact_type_rate``
                                            and its alias ``bond_exact_match``

Writes ``metrics/by_metric/wfa113_t75_exact_match_alias.json``.

Usage:
    .venv/bin/python molmetal/scripts/wfa113_t75_exact_alias_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "metrics" / "by_metric" / "wfa113_t75_exact_match_alias.json"


def _rows() -> list[dict]:
    seen = []
    for base in ("results", "molmetal/reports"):
        for path in sorted((ROOT / base).rglob("per_epoch_metrics.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            rows = payload if isinstance(payload, list) else payload.get(
                "per_epoch_metrics", []
            )
            for row in rows:
                if "bond_exact_match" not in row or "bond_exact_type_rate" not in row:
                    continue
                seen.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "epoch": row.get("epoch"),
                        "bond_exact_match": float(row["bond_exact_match"]),
                        "bond_exact_type_rate": float(row["bond_exact_type_rate"]),
                    }
                )
    return seen


def main() -> int:
    rows = _rows()
    identical = [
        r for r in rows if abs(r["bond_exact_match"] - r["bond_exact_type_rate"]) < 1e-12
    ]
    differing = [r for r in rows if r not in identical]
    nonzero = [r for r in rows if r["bond_exact_match"] > 0.0]
    payload = {
        "wave": "Wf-A113",
        "task": "T75",
        "date": "2026-10-28",
        "question": (
            "are `bond_exact_match` and `bond_exact_type_rate` a decomposition "
            "or the same number printed twice?"
        ),
        "measured": {
            "n_rows": len(rows),
            "n_files": len({r["path"] for r in rows}),
            "n_identical": len(identical),
            "n_differing": len(differing),
            "n_nonzero_exact_match": len(nonzero),
            "max_exact_match": max((r["bond_exact_match"] for r in rows), default=0.0),
        },
        "verdict": (
            "identical in every row — the two keys were one number under two "
            "names, and the trainer had no set-only quantity at all"
        )
        if not differing
        else "the two keys differ somewhere",
        "definitions": {
            "bench_exact": "bond SET only (connectivity), bench_tmqm_bond_perception.score_bonds",
            "bench_exact_type": "bond set AND every bond order",
            "trainer_bond_set_exact_rate": "bench_exact (added by T75)",
            "trainer_bond_exact_type_rate": "bench_exact_type",
            "trainer_bond_exact_match": "alias of bond_exact_type_rate (strict reading of 'exact match')",
        },
        "fix": (
            "split the counter: `exact_set` is incremented on set equality "
            "alone, `exact_type` keeps the per-class `all(...)` guard; the "
            "set-only rate is emitted as `bond_set_exact_rate` and carried "
            "through EpochMetrics, events.jsonl and the per-epoch CSV"
        ),
        "test": "molmetal/tests/test_wfa113_t75_exact_match_decomposition.py",
        "script": "molmetal/scripts/wfa113_t75_exact_alias_probe.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["measured"], indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
