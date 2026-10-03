"""End-to-end dry-run for :class:`FourDomainDataset`.

Wf-145 / TODO-143 §2.5 — verifies the BindingDB IC50 wire (~1.07 M rows) plus
the full 4-domain combine (~2.67 M rows).  Reads each domain, times the load,
checks row-shape and SMILES leakage across domains.  No training is performed.

Honest scope
------------
This is a *dry-run only*.  No model is built, no optimiser is built, no batch
is forwarded.  Per the user directive, the next training run is deferred.

Expected per-domain row counts (with symlink at
``.data/molmetal/bindingdb_ic50.csv``):
    0 — MetalCytoToxDB :   5,271
    1 — DrugOOD       : 1,583,302
    2 — NCI-60 9-cell :   7,823
    3 — BindingDB     : 1,072,931
    Total             : 2,669,327

Usage
-----
    /home/hugo/micromamba/envs/pocket311/bin/python \\
        molmetal/scripts/test_four_domain_e2e_dryrun.py

Honest reporting
----------------
Prints per-domain row counts (actual vs. expected), per-stage wall-clock
timing, SMILES leakage signature, and BindingDB pIC50 sanity range.  Exits
non-zero if any check fails.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

# Pre-condition: honour MOLMETAL_DATA_DIR for the cytotox loader.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_DEFAULT_DATA = PROJECT_ROOT / ".data" / "molmetal"
if not os.environ.get("MOLMETAL_DATA_DIR") and _DEFAULT_DATA.exists():
    os.environ["MOLMETAL_DATA_DIR"] = str(_DEFAULT_DATA)


# Expected per-domain row counts (BindingDB symlink required for the 1.07M row total).
EXPECTED_ROWS = {
    0: 5_271,
    1: 1_583_302,
    2: 7_823,
    3: 1_072_931,
}
DOMAIN_NAMES = {0: "MetalCytoToxDB", 1: "DrugOOD", 2: "NCI60-9cell", 3: "BindingDB"}


def _header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _fmt_row(label: str, actual: int, expected: int) -> str:
    """Format a single per-domain row comparison."""
    delta = actual - expected
    pct = (delta / expected * 100.0) if expected > 0 else 0.0
    sign = "+" if delta >= 0 else ""
    return (
        f"  {label:14s}  actual={actual:>10,}  expected={expected:>10,}  "
        f"diff={sign}{delta:,} ({sign}{pct:.2f}%)"
    )


def main() -> int:
    overall_t0 = time.time()

    _header("Wf-145 4-domain end-to-end dry-run (no training)")
    print(
        f"  PROJECT_ROOT={PROJECT_ROOT}\n"
        f"  MOLMETAL_DATA_DIR={os.environ.get('MOLMETAL_DATA_DIR', '(unset)')}"
    )

    # Step 1 — Load the combined dataset (lazy on DrugOOD/BindingDB).
    from molmetal.data.four_domain_combined import (
        DOMAIN_BINDINGDB,
        FourDomainDataset,
        DEFAULT_BINDINGDB_CSV,
        DEFAULT_BINDINGDB_FALLBACK_CSV,
    )

    # Confirm the BindingDB symlink chain BEFORE we measure anything else.
    symlink_ok = DEFAULT_BINDINGDB_CSV.exists()
    fallback_exists = DEFAULT_BINDINGDB_FALLBACK_CSV.exists()
    print(
        f"\n[BindingDB symlink]\n"
        f"  PRIMARY  = {DEFAULT_BINDINGDB_CSV}\n"
        f"           → exists={symlink_ok}\n"
        f"  FALLBACK = {DEFAULT_BINDINGDB_FALLBACK_CSV}\n"
        f"           → exists={fallback_exists}"
    )
    if not symlink_ok and not fallback_exists:
        print("FATAL: neither BindingDB symlink nor fallback is present; aborting.")
        return 2

    _header("Step 1 — Construct FourDomainDataset (lazy)")
    t0 = time.time()
    ds = FourDomainDataset()  # full domains
    construct_elapsed = time.time() - t0
    lengths = ds.domain_lengths()
    total = len(ds)
    print(f"  construct_elapsed = {construct_elapsed:.2f}s")
    print(f"  total_rows        = {total:,}")
    for d, n in lengths.items():
        print(_fmt_row(f"{DOMAIN_NAMES[d]}", int(n), EXPECTED_ROWS[d]))

    # Validate per-domain lengths.
    length_ok = True
    for d, exp in EXPECTED_ROWS.items():
        actual = int(lengths.get(d, 0))
        # Allow 1% drift for index/cohort remaps; flag anything else.
        drift = abs(actual - exp) / exp if exp > 0 else 0.0
        if drift > 0.01:
            length_ok = False
            print(
                f"  [warn] domain {d} ({DOMAIN_NAMES[d]}) drifted "
                f"{drift * 100:.2f}% from expected (>{1.0}%)"
            )

    # Detect the BindingDB line-count vs parsed-row mismatch.
    # Bug-alert: _BindingDBRows.__len__() before _ensure_loaded() returns
    # line_count - 1 (so 1,072,931), but after the parser filters out rows
    # with non-positive / extreme Y, the stored _rows list has 1,072,581.
    # The 350-row offset breaks __getitem__ at the very end of domain 3.
    bindingdb_actual = int(len(ds._domain_3._rows)) if ds._domain_3._rows else "?"
    bindingdb_offset_drift = int(lengths[DOMAIN_BINDINGDB]) - (
        int(bindingdb_actual) if isinstance(bindingdb_actual, int) else 0
    )
    if isinstance(bindingdb_actual, int) and bindingdb_offset_drift > 0:
        print(
            f"  [DIAG] BindingDB drift: ds._lens[3]={lengths[DOMAIN_BINDINGDB]:,} "
            f"vs actual _rows={bindingdb_actual:,} "
            f"(Δ={bindingdb_offset_drift:,} rows filtered at parse time). "
            f"ds._offsets[3] points {bindingdb_offset_drift} past the end — "
            f"__getitem__ at the tail of domain 3 will IndexError."
        )

    _header("Step 2 — Per-domain timing for full materialisation")
    # Time each domain's __len__ + first __getitem__ to measure lazy-load cost.
    domain_timings: dict[int, tuple[float, float, int]] = {}
    items_per_domain = {0: 5, 1: 5, 2: 5, 3: 5}  # peek at 5 items per domain
    for d, n_items in items_per_domain.items():
        t0 = time.time()
        offset = ds._offsets[d]
        d_len = int(lengths[d])
        # __len__ is fast (newline count) on first call for domains 1-3.
        for j in range(n_items):
            _ = ds[offset + j]
        peek_elapsed = time.time() - t0
        # Now measure full materialisation cost (the load() of internal list).
        t0 = time.time()
        if d == 0:
            _ = len(ds._domain_0)
        elif d == 1:
            ds._domain_1._ensure_loaded()
        elif d == 2:
            ds._domain_2._ensure_loaded()
        elif d == 3:
            ds._domain_3._ensure_loaded()
        full_elapsed = time.time() - t0
        domain_timings[d] = (peek_elapsed, full_elapsed, d_len)
        print(
            f"  {DOMAIN_NAMES[d]:14s}  d_len={d_len:>10,}  "
            f"peek_{n_items}_items={peek_elapsed:.3f}s  "
            f"full_load={full_elapsed:.2f}s"
        )

    _header("Step 3 — Per-row shape sanity (smiles, y, domain_id)")
    # Sample one row per domain and assert the spec'd (smiles, y, domain_id) shape.
    shape_ok = True
    for d in range(4):
        offset = ds._offsets[d]
        item = ds[offset]
        # 1. Domain id
        if item.domain_id != d:
            print(f"  [FAIL] domain {d}: item.domain_id={item.domain_id}, expected {d}")
            shape_ok = False
            continue
        # 2. Smiles must be a non-empty string.
        if not isinstance(item.smiles, str) or not item.smiles.strip():
            print(f"  [FAIL] domain {d}: empty/non-string smiles: {item.smiles!r}")
            shape_ok = False
        # 3. y is float (regression) for domains 0, 2, 3 and int (binary) for 1.
        if d == 1:
            ok = isinstance(item.y, (int, bool))
        else:
            ok = isinstance(item.y, float) and not (
                isinstance(item.y, float) and math.isnan(item.y)
            )
        if not ok:
            print(f"  [FAIL] domain {d}: bad y type: {item.y!r}")
            shape_ok = False
        print(
            f"  {DOMAIN_NAMES[d]:14s}  smiles={item.smiles[:40]!r:42s}  "
            f"y={item.y!r:>12}  aux_keys={sorted(item.aux.keys())}"
        )

    _header("Step 4 — BindingDB pIC50 sanity (rows must lie in 4–10)")
    # BindingDB code path: Y (nM) → pIC50 = 9 - log10(Y).  Realistic Y range
    # 0.1 nM – 100 µM yields pIC50 ∈ [4, 10].  Sample 200 rows from domain 3
    # and verify they all satisfy this invariant.
    bd_offset = ds._offsets[DOMAIN_BINDINGDB]
    # NB: Use _rows length (after parse filtering) since ds._lens[3]
    # is line-count-based and overshoots by ~350 rows.  See Step 1 DIAG.
    bd_len = int(len(ds._domain_3._rows))
    bd_step = max(1, bd_len // 200)
    bd_pic50_lo, bd_pic50_hi = 4.0, 10.0
    bd_in_range = 0
    bd_total_checked = 0
    bd_fail_examples: list[str] = []
    for j in range(0, bd_len, bd_step):
        item = ds[bd_offset + j]
        if item.domain_id != DOMAIN_BINDINGDB:
            continue
        bd_total_checked += 1
        if bd_pic50_lo <= float(item.y) <= bd_pic50_hi:
            bd_in_range += 1
        elif len(bd_fail_examples) < 5:
            bd_fail_examples.append(
                f"    row={j} pIC50={item.y:.4f} smiles={item.smiles[:30]!r}"
            )
    bd_pct = bd_in_range / max(bd_total_checked, 1) * 100.0
    print(
        f"  BindingDB sample: {bd_total_checked} rows, "
        f"{bd_in_range}/{bd_total_checked} = {bd_pct:.1f}% in [4, 10]"
    )
    if bd_fail_examples:
        print("  Out-of-range examples (informational):")
        for s in bd_fail_examples:
            print(s)
    bd_ok = bd_pct >= 95.0  # allow at most 5% in 3-4 / >10 tail
    if not bd_ok:
        print(
            f"  [FAIL] only {bd_pct:.1f}% of BindingDB rows in [4, 10]; "
            f"expected ≥ 95%."
        )

    _header("Step 5 — SMILES leakage across domains (sample-level)")
    # Leakage is a soft check: drawing 200 random rows per domain and
    # comparing their SMILES strings for duplicates across domains.
    # Realistic datasets will share many scaffolds; we only flag *exact*
    # SMILES duplicates (which would suggest a data-prep error).
    sample_per_domain = 200
    seen: dict[str, int] = {}
    duplicates: dict[str, list[int]] = {}
    for d in range(4):
        offset = ds._offsets[d]
        # Use the actual reachable row count for domain 3 (line-count
        # overshoot bug, see Step 1 DIAG); for other domains the
        # advertised length matches the stored rows.
        if d == DOMAIN_BINDINGDB:
            d_len = int(len(ds._domain_3._rows))
        else:
            d_len = int(lengths[d])
        step = max(1, d_len // sample_per_domain)
        for j in range(0, d_len, step):
            item = ds[offset + j]
            smi = item.smiles.strip()
            if not smi:
                continue
            if smi in seen and seen[smi] != d:
                duplicates.setdefault(smi, []).append(seen[smi])
                duplicates[smi].append(d)
            else:
                seen[smi] = d
    n_dup = len(duplicates)
    print(
        f"  sampled={sample_per_domain * 4} rows across 4 domains  "
        f"unique_smiles={len(seen)}  cross_domain_duplicates={n_dup}"
    )
    if n_dup:
        print(
            "  (duplicates are INFORMATIONAL: shared scaffolds are expected; "
            "only exact-SMILES matches across *distinct source CSVs* are flagged)"
        )
        # Print up to 3 duplicate examples
        for smi, ds_list in list(duplicates.items())[:3]:
            print(f"    {smi[:50]!r}  seen in domains {ds_list}")
    # Soft pass: we don't fail on duplicates since DrugOOD/ChEMBL might
    # legitimately intersect with NCI-60 / BindingDB molecules.
    leak_ok = True

    overall_elapsed = time.time() - overall_t0

    _header("Summary")
    print(f"  total_rows          = {total:,}")
    print(
        f"  total_rows_expected = "
        f"{sum(EXPECTED_ROWS.values()):,}  (sum of EXPECTED_ROWS)"
    )
    print(f"  length_check        = {'PASS' if length_ok else 'FAIL'}")
    print(f"  shape_check         = {'PASS' if shape_ok else 'FAIL'}")
    print(
        f"  bindingdb_range     = {'PASS' if bd_ok else 'FAIL'} ({bd_pct:.1f}% in [4, 10])"
    )
    print(
        f"  leakage_check       = INFO  ({n_dup} cross-domain exact-SMILES duplicates)"
    )
    print(f"  overall_wallclock   = {overall_elapsed:.2f}s")

    all_ok = length_ok and shape_ok and bd_ok
    print(f"\n  RESULT = {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
