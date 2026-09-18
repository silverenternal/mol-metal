"""Measure ETKDGv3 embed failure rate per tile in the fragment pool.

Usage:
    python -m scripts.measure_fragment_library

Reports per-category embed failure rates plus the final tile count by
category. Designed to be runnable from any working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make molmetal importable regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molmetal_lam.tile_lib.fragment_pool import (
    FRAGMENT_POOL_AZIDES,
    FRAGMENT_POOL_ALKYNES,
    FRAGMENT_POOL_DIENES,
    FRAGMENT_POOL_THOLS,
    fragments_from_chembl_reactive,
    l6_fragment_pool_metrics,
)


def main() -> int:
    """Run validation, print summary, return 0 on success."""
    print("=" * 60)
    print("Fragment library embed-failure report")
    print("=" * 60)

    # Run full pool build first so counters populate.
    pool = fragments_from_chembl_reactive()

    by_category = {
        "azide":  [s for s in FRAGMENT_POOL_AZIDES],
        "alkyne": [s for s in FRAGMENT_POOL_ALKYNES],
        "diene":  [s for s in FRAGMENT_POOL_DIENES],
        "thiol":  [s for s in FRAGMENT_POOL_THOLS],
    }

    # Bucket the emitted tiles by primary tag.
    emitted_by_category = {"azide": 0, "alkyne": 0, "diene": 0, "thiol": 0}
    for t in pool:
        for cat in emitted_by_category:
            if t.has_group(cat) or (cat == "alkyne" and (t.has_group("terminal_alkyne") or t.has_group("cyclooctyne"))):
                emitted_by_category[cat] += 1
                break
            if cat == "diene" and (t.has_group("diene") or t.has_group("dienophile")):
                emitted_by_category[cat] += 1
                break
            if cat == "thiol" and t.has_group("thiol"):
                emitted_by_category[cat] += 1
                break

    metrics = l6_fragment_pool_metrics()
    print()
    print(f"{'category':<10} {'input':>6} {'valid':>6} {'embed_fail':>11} {'mw_fail':>8} {'logp_fail':>10} {'fail_%':>8}")
    print("-" * 60)
    total_in, total_valid = 0, 0
    for cat, n_input in [(k, len(v)) for k, v in by_category.items()]:
        m = metrics[cat]
        fails = m["embed_fail"] + m["mw_fail"] + m["logp_fail"]
        fail_pct = (fails / n_input * 100.0) if n_input else 0.0
        print(
            f"{cat:<10} {m['input']:>6} {m['valid']:>6} {m['embed_fail']:>11} "
            f"{m['mw_fail']:>8} {m['logp_fail']:>10} {fail_pct:>7.1f}%"
        )
        total_in += m["input"]
        total_valid += m["valid"]
    print("-" * 60)
    overall_fail_pct = ((total_in - total_valid) / total_in * 100.0) if total_in else 0.0
    print(f"{'TOTAL':<10} {total_in:>6} {total_valid:>6} {'':<11} {'':<8} {'':<10} {overall_fail_pct:>7.1f}%")

    print()
    print("Emitted tiles per category:")
    for cat, n in emitted_by_category.items():
        print(f"  {cat:<8} {n:>4}")
    print(f"  {'TOTAL':<8} {len(pool):>4}")

    print()
    print(f"Total pool size: {len(pool)} tiles (target: >= 200)")
    return 0 if len(pool) >= 200 else 1


if __name__ == "__main__":
    sys.exit(main())
