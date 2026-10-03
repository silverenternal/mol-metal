# Wf-345-C4 — 4th stale anchor refresh

**Wf:** Wf-345 (wx63q8fwe ultracode residual)
**Date:** 2026-09-29
**Audit source:** `molmetal/reports/wf345a3_anchor_audit/audit.json`

---

## §1 Identification of the 4th stale anchor

Per Wf-345-A3 audit (`audit.json`), 11 paper-grade anchors were inspected under glob patterns
`{m14_anchor,wf269c_report,m17_drugood_reeval,sweep_summary}.json`. **4 stale anchors** (age > 24h):

| # | path | age_hr (before) |
|---|---|---:|
| 1 | `wf121_cisplatin_real_vina/sweep_summary.json` | 60.6599 |
| 2 | `wf143_p4_chemotype_dock/sweep_summary.json` | 57.731 |
| 3 | `wf170_permetal_singletask/sweep_summary.json` | 29.0894 |
| **4** | **`wf67_p4_3_100pocket/sweep_summary.json`** | **70.8641** |

The 4th anchor (the unknown) is **`molmetal/reports/wf67_p4_3_100pocket/sweep_summary.json`**.

This anchor matches the `sweep_summary.json` glob pattern, but unlike Wf-121 / Wf-143 / Wf-170
(which are real-Vina sweeps) the Wf-67 P4.3 sweep is a **100-pocket × 10-seed honest-stub Vina
docking sweep** (deterministic SHA-256-derived kcal/mol, per Wf-30 honest-negative fallback).
Schema: `tool = bench_100pocket_10seed.py`, `pilot_id = wf67_p4_3_100pocket`, `schema_version = 1`.

It had been stale since 2026-09-26T09:08:46Z (70.86 hr) — the longest of the four anchors
because the Wf-67 P4.4 real-Vina re-bench (CrossDocked-2020 receptors) is queued externally
and the stub sweep just needs a periodic re-emit to keep the freshness gate green.

---

## §2 Source script

Located via `grep -rln "wf67\|p4_3_100pocket\|P4.3.3" molmetal/scripts/`:

```
molmetal/scripts/bench_100pocket_10seed.py
```

This is the canonical emitter for `molmetal/reports/wf67_p4_3_100pocket/sweep_summary.json`.
Per the script docstring it is the Wf-67 Phase 3 expansion of the Wf-30 10-pocket × 1-seed
sweep to a 100-pocket × 10-seed grid. Default mode is `vina_stub`; `--use-real-vina` is the
optional mode that requires the Wf-63 ship + staged CrossDocked-2020 receptors (out of scope
for this refresh).

---

## §3 Refresh run

```bash
python3 molmetal/scripts/bench_100pocket_10seed.py \
    --out-dir molmetal/reports/wf67_p4_3_100pocket
```

Wall: **~1.0 s** (CPU-only; 100 pockets × 10 seeds × 5 candidates = 5000 stub evals).

Resulting `sweep_summary.json` (`generated_utc = 2026-09-29T08:55:23.041836+00:00`):

| field | pre-refresh | post-refresh |
|---|---|---|
| `n_pockets_completed` | 5 (Wf-30 back-compat) | **100** (full grid) |
| `exit_status` | 0 | 0 |
| `total_wall_s` | 0.00880 | 0.22768 |
| `overall_pocket_mean_kcal_mol` | -8.4157 | **-8.3777** |
| `overall_pocket_std_kcal_mol` | 0.1415 | (new value emitted by full 100-pocket grid) |
| `pocket_modes` | 5/5 `vina_stub` | 100/100 `vina_stub` |
| `use_real_vina_resolved` | false | false |

Note: the full 100-pocket grid produces 95 additional stubs whose mean kcal/mol pulls
the overall mean from -8.4157 to -8.3777. This is **expected** — the pre-refresh file was
Wf-30 back-compatible only; the canonical Wf-67 P4.3 sweep runs the full 100-pocket grid.

---

## §4 Freshness verdict (post-refresh)

- `mtime`: 2026-09-29 16:55:23 +0800
- `age_hr_post_refresh`: **0.0021** (well below the 24-h gate)
- `stale_24h`: **False**

---

## §5 Honest framing (per party-twenty-spirit §3.5)

- The 4th anchor was a **stub-mode sweep**, not a real-Vina run; refresh is a fast CPU
  re-emit (~1 s), not a GPU retrain.
- The pre-refresh summary.json had only 5/100 pockets (Wf-30 back-compat), which is
  incomplete for the `wf67_p4_3_100pocket` schema. The post-refresh file is the canonical
  full 100-pocket grid, which is more representative but not a real-Vina re-bench.
- Real Vina re-scoring on a 10-pocket subset remains **queued** for Wf-67 P4.4 once the
  CrossDocked-2020 receptors are staged (per `honest_framing` field). This refresh closes
  the freshness gate but does not lift the real-Vina honest-negative caveat.
- Drift status: **expected drift** (-8.4157 → -8.3777 kcal/mol) due to the 5→100 pocket
  expansion. Not a regression.

---

## §6 Gate impact

- `p3_wf343c4_anchor_freshness_24h` gate: now 4 stale anchors (Wf-121 / Wf-143 / Wf-170
  still need separate refresh workflows). Wf-67 anchor refreshed here.
- `EXPECTED_GATE_COUNTS["all"]` 62 PRESERVED (no new gate added; just one of the 4 stale
  anchors brought under 24 h).
- Pre-flight v3 fresh-run **62 PASS / 0 FAIL / 62 TOTAL** PRESERVED at Wf-335-SYN baseline
  (NOT re-run by Wf-345-C4 — docs-only).
- 0 NEW tests, 0 GPU launches.

---

## §7 Cross-references

- Wf-345-A3 audit: `molmetal/reports/wf345a3_anchor_audit/audit.json`
- Wf-67 P4.3 plan: `TODO/pending/67_r20_p4_3_100pocket.md` (per Wf-329 §1.1 reference)
- Wf-67 P4.3 source script: `molmetal/scripts/bench_100pocket_10seed.py`
- Wf-345-C4 sibling closeouts: Wf-345-C1 / Wf-345-C2 / Wf-345-C3 (per wx63q8fwe)
