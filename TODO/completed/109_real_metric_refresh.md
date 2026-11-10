# TODO-109 — Wf-87 REAL Metric Refresh Close-out

**Date**: 2026-09-26
**Workflow**: Wf-87 Phase 1+2+3 — per-metric disk audit + real JSON promotion + INDEX §17 refresh
**Owner**: Hugo Lee + Wf-87 ultracode agent
**Parent**: `TODO/INDEX.md` §17 (TODO-89 5-Category Audit) + Wf-72 Phase 3 (`molmetal/scripts/audit_5cat_extend.py`) + Wf-84 P3 (`molmetal/reports/wf84_dmpnn_ckpt/training_summary.json`)
**Status**: **SHIP** — 17/17 promotion tests PASS + 10/10 STRONG_REAL cross-check PASS + 22/22 cross-ref files exist + 8 NEW central metric files
**Constraint**: paper-gate (TODO-84) honored — no `paper/*` touched; party-twenty-spirit §3.5 (0-tolerance honest framing) applied throughout

---

## §1 TL;DR

Wf-87 ships a 3-phase **real metric refresh** of `metrics/by_metric/` + INDEX §17, taking the per-metric ledger from "stale R17 close-out" to a 2026-09-26 audit-anchored state with 8 NEW promoted JSON files:

| Phase | What | Output |
|---|---|---|
| **Wf-87 P1** | Per-metric disk audit (125 JSON files × 26 conceptual metrics) | `molmetal/reports/wf87_p1_disk_audit/per_metric_status.json` + `synthesis.md` |
| **Wf-87 P2** | Promote 8 NEW MEASURED JSONs from `molmetal/reports/wf*/` into `metrics/by_metric/` | `molmetal/reports/wf87_p2_promotion/promotion_log.json` + 8 NEW central files |
| **Wf-87 P3** | Verify INDEX §17 STRONG_REAL counts + cross-ref files | `molmetal/reports/wf87_index_17_real/verify_index_17.{json,md}` |

**§17 STRONG_REAL count stays 10** (no fabricated bumps); §17 cross-ref ledger **22/22 file-exists PASS**; 8 NEW central metric files add **PROJECTED-free real data** to the central ledger for downstream consumers (Wf-44, Wf-49, Wf-83/85/86 etc.).

---

## §2 Per-deliverable breakdown

### §2.1 Wf-87 Phase 1 — Per-metric disk audit

- **Script**: `molmetal/scripts/audit_disk_metrics.py` (per `wf87_p1_disk_audit/synthesis.md`)
- **Output**: `molmetal/reports/wf87_p1_disk_audit/per_metric_status.json` (per-metric row: `metric_id`, `category_bucket`, `on_disk_state`, `measured_count`, `projected_count`, `deferred_count`, `latest_value`, `mtime_utc`, `primary_source`, `fallback_source`)
- **Aggregate**: 125 files scanned / 26 conceptual metrics audited; on-disk distribution = 19 MEASURED / 4 REVIEW_NEEDED / 2 DEFERRED / 1 DEFERRED_HOST_GAP
- **Honest finding**: the 3 DEFERRED rows (`diversity_subpocket` / `steric_clash_pb_dock` / `vina_*_test_001`) carry explicit `deferred_reason` per Wf-61 §1.4-1.6 close-out (host gap, not algorithm gap)
- **Note**: `m15 RMSD` and `m3 Vina Dock stub` flagged REVIEW_NEEDED — need follow-up wf (R20 plan items)

### §2.2 Wf-87 Phase 2 — Real JSON promotion to `metrics/by_metric/`

- **Promoter script**: `molmetal/scripts/promote_metrics.py` (~960 LOC)
- **Test file**: `molmetal/tests/test_promote_metrics.py` (~395 LOC, 17 tests PASS in 0.37s)
- **Output**: 8 NEW JSON files in `metrics/by_metric/` + `molmetal/reports/wf87_p2_promotion/promotion_log.json` (14 KB)
- **Aggregate**: 8 promoted / 0 skipped / 0 errors / 0 collisions (first-run)

| # | Target file | Source(s) | n_sources | Headline value | Status |
|--:|---|---|--:|---|---|
| 1 | `hilbert_4cell_r2.json` | `wf83_p1_phase2_4cell/{mcf7,a549,a2780,a2780cis}/per_seed_metrics.json` | 4 | aggregate_combined_r2_mean = **−6.7126** — **(SUPERSEDED 2026-10-31: the committed artifact's value is −5.662524090882017; see §5.1)** | MEASURED_BELOW_GATE |
| 2 | `platinai_combined_r2.json` | `wf77_p1_hilbert_baseline/platinai_4cell` + `wf77_p2_astr_baseline/platinai_4cell` + `wf79_p2_3way_pll/platinai_4cell` | 3 | baseline_astr_r2_mean = **−0.4028** | MEASURED |
| 3 | `metalcytotox_combined_r2.json` | `wf77_p2_astr_baseline/metalcytotox_5metal` + `wf79_p2_3way_pll/metalcytotox_5metal` | 2 | per-method baseline **−0.2841** | MEASURED |
| 4 | `tmqm_combined_r2.json` | `wf71_p3_wf49_cross_round2/tmqm_dblock/per_cohort_metrics` + `wf71_p3_wf49_cross_round2/cross_dataset_summary` | 2 | cross_dataset_macro_r2_mean = **+0.1248** | MEASURED |
| 5 | `ood_per_metal_reachability.json` | `wf73_p2_ood_permetal/{Ru,Ir,Os,Rh,Re}/ood_reachability` | 5 | aggregate_ood_pass_rate = **1.000** | MEASURED |
| 6 | `ood_per_element_reachability.json` | `wf73_p3_ood_perelement/{Co,Ni,Cu,Au,Hg}/ood_reachability` | 5 | aggregate_ood_pass_rate = **0.650** | MEASURED |
| 7 | `dmpnn_ckpt_status.json` | `.ckpt/` directory scan (incl. Wf-84 P3 Ru/Ir/Rh/Os/Re × 2-seed = 10 NEW ckpts) | 0 | n_checkpoints = **12** | MEASURED |
| 8 | `vina_status.json` | `vina_kcal_per_mol.json` + filesystem probe (Wf-84 P1+P2 host probe) | 0 | vina_binary_available = **false** (Wf-84 P2 wrapper in place; auto-detect pending) | MEASURED |

### §2.3 Wf-87 Phase 3 — INDEX §17 verification

- **Verifier**: `molmetal/scripts/verify_index_17_real.py`
- **Output**: `molmetal/reports/wf87_index_17_real/verify_index_17.{json,md}`
- **STRONG_REAL cross-check**: 10 claimed (m11 / m13 / m14 / m17 / m21 / m22 / m23 / m24 / ring_size / metallodrug_vc) ↔ 10 on-disk MEASURED = **PASS**
- **Per-metric on-disk audit**: 25 expected / 25 existing / 24 with value / 0 missing
- **Cross-ref audit**: 25 expected / 0 missing (all 22 files in §17.4 + 3 §17.5 helper files exist)

### §2.4 §17 INDEX refresh — minimal touch

Per `party-twenty-spirit §3.5`, INDEX §17 stays at **STRONG_REAL = 10 / WEAK_REAL = 11 / MOCK = 5+ / DEFERRED = 3** counts (no fabricated bumps). The 8 NEW promoted files are categorized as **cross-dataset / ckpt-inventory / status-probe** supporting files, not new conceptual metrics — they backfill the existing 22-25 conceptual metric surface, not extend it.

**§17 cross-ref refresh**: added 3 new cross-ref files per Wf-87 close-out:
- `molmetal/reports/wf87_p1_disk_audit/synthesis.md` (per-metric disk audit)
- `molmetal/reports/wf87_p2_promotion/synthesis.md` (8 NEW promoted files)
- `molmetal/reports/wf87_index_17_real/verify_index_17.{json,md}` (§17 STRONG_REAL cross-check + 22/22 cross-ref PASS)

---

## §3 MEASURED vs PROJECTED

### §3.1 MEASURED on disk (Wf-87 cycle deliverables)

- **125** files scanned in `metrics/by_metric/`
- **26** conceptual metrics audited (22 baseline + 3 NEW from R20/R21 + 1 NEW from Wf-86 MetalCytoToxDB lift)
- **8** NEW promoted JSON files (each carrying MEASURED / MEASURED_BELOW_GATE status, NOT PROJECTED)
- **17** promoter tests PASS (`molmetal/tests/test_promote_metrics.py`)
- **22/22** §17.4 cross-ref files exist (100% PASS)
- **10/10** §17.1 STRONG_REAL on-disk PASS
- **0** GPU retrain executed (CPU-only docs phase)
- **0** paper touched (TODO-84 gate honored)

### §3.2 PROJECTED — what R20 must deliver to lift STRONG_REAL count

Per `party-twenty-spirit §3.5`, **STRONG_REAL count = 7 → 9-11** is PROJECTED, NOT MEASURED. The lift depends on R20 GPU retrain results that have NOT executed:

| Path | Condition for STRONG_REAL promotion | R20 plan item | Today status |
|---|---|---|---|
| m1 Ru AUC → STRONG | retrain lifts ≥ 0.86 (vs 0.7123 R17 baseline) | R20·P2 | **0.630** Wf-22 honest-negative; 5-seed × 50-epoch × {se3sb, tmqm} DEFERRED |
| m2 Pt 4-cell AUC → STRONG | 5-seed × 50-epoch lifts ≥ 0.85 / 0.70 Rusanov | R20·P3 | **0.648 / 0.584 / 0.630** (3 of 4 below gate); wet-lab IC50 dependent |
| m16 Affinity r → STRONG | 5-seed × 30-epoch dual-cohort lift > 0.207 R17 + 0.572 ridge ceiling | R20·P5 | **−0.040** Wf-32 smoke (1-seed × 1-epoch PlatinAI 4-cell, HONEST-NEGATIVE) |
| m19 LOMO → STRONG | Rh 0.368 bottleneck lift via 5-metal multitask pretrain | R20·P6 | **0.535** unchanged; DEFERRED to R20 GPU-0 sweep |

**Projected STRONG_REAL count = 11** (10 current + 1 from Wf-86 MetalCytoToxDB lift if MEASURED R² ≥ 0 lands in R20 GPU retrain — currently still PROJECTED). The 5 NEW files promoted today (hilbert_4cell_r2 / platinai_combined_r2 / metalcytotox_combined_r2 / tmqm_combined_r2 / ood_per_metal_reachability / ood_per_element_reachability / dmpnn_ckpt_status / vina_status) are **STATUS files, not conceptual metrics**, so they do not bump §17 STRONG_REAL count.

### §3.3 Wf-84 P3 anchor (companion to Wf-87 — same cycle)

`molmetal/reports/wf84_dmpnn_ckpt/training_summary.json` ships 5-metal × 2-seed D-MPNN pIC50 retrain results:

| metal | test pearson | test R² | test RMSE | n_test | ckpts |
|---|--:|--:|--:|--:|---|
| Ru | **+0.276** ± 0.086 | +0.084 | 4.118 | 796 | 2 |
| Ir | +0.016 ± 0.043 | +0.002 | 4.354 | 301 | 2 |
| Rh | −0.067 ± 0.344 | +0.123 | 4.091 | 38 | 2 |
| Os | +0.056 ± 0.280 | +0.081 | 5.305 | 144 | 2 |
| Re | +0.025 ± 0.030 | +0.002 | 4.072 | 216 | 2 |
| **aggregate** | **+0.061** | **+0.058** | — | 1595 | 10 |

These values are surfaced in `metrics/by_metric/pearson_r_pic50.json` (latest entries) + `metrics/by_metric/dmpnn_ckpt_status.json` (NEW, 12 ckpts total including 10 from Wf-84 + 2 from earlier affinity_platinai run). Wf-84 is **MEASURED** but **cohort-not-comparable** with R17 ridge baseline (0.572 on HeLa48h vs Wf-84 0.276 on MetalCytoToxDB Ru) — apples-to-oranges per `pearson_r_pic50.json` `scope_caveat`.

### §3.4 Wf-85 / Wf-86 anchor (companion to Wf-87)

- **Wf-85 P1** (`molmetal/reports/wf85_p1_hilbert_only_real/`) — Hilbert-only on PlatinAI 4-cell CPU vs CUDA, 5 seeds × 4 cell lines
- **Wf-85 P2** (`molmetal/reports/wf85_p2_astr_only_real/`) — ASTSR-only on PlatinAI 4-cell
- **Wf-85 P3** (`molmetal/reports/wf85_p3_hilbert_astr_combined_real/`) — Hilbert+ASTSR convex blend on PlatinAI 4-cell
- **Wf-86 P1** (`molmetal/reports/wf86_p1_metalcytotox_real/`) — REAL cross-dataset R² on MetalCytoToxDB 5-metal (scaffold split, 5 seeds)
- **Wf-86 P2** (`molmetal/reports/wf86_p2_tmqm_real/`) — REAL cross-dataset R² on tmQM d-block 5-element (5-fold CV)
- **Wf-86 P3** (`molmetal/reports/wf86_p3_platinai_4cell_real/`) — REAL Hilbert+ASTSR combined R² on PlatinAI 4-cell (5-seed × strict-scaffold)

Wf-85 P3 / Wf-86 P1-P3 all surface as **MEASURED_BELOW_GATE / MEASURED_HONEST_NEGATIVE_R2** values in `hilbert_4cell_r2.json` + `platinai_combined_r2.json` + `metalcytotox_combined_r2.json` + `tmqm_combined_r2.json` — honest-negative R² is **load-bearing for §3.5 honest framing**, NOT hidden.

---

## §4 Before vs After

### §4.1 §17 ledger (per-metric state)

| Aspect | Before Wf-87 (R17 close-out baseline) | After Wf-87 (2026-09-26) |
|---|---|---|
| STRONG_REAL count | 10 (10 conceptual metrics × MEASURED) | **10** (no fabricated bumps; same 10 conceptual metrics) |
| WEAK_REAL count | 11 | **11** (unchanged) |
| MOCK count | 5 | **5** (unchanged) |
| DEFERRED_HOST_GAP | 3 | **3** (unchanged) |
| Total central `metrics/by_metric/` files | ~117 (R17 close-out) | **125** (+8 NEW from Wf-87 P2) |
| Per-metric audit ledger | none | `molmetal/reports/wf87_p1_disk_audit/per_metric_status.json` (NEW) |
| §17 STRONG_REAL cross-check | not verified | `molmetal/reports/wf87_index_17_real/verify_index_17.{json,md}` — **10/10 PASS** |
| §17 cross-ref file-existence | not verified | **22/22 PASS** (0 missing) |
| Wf-84 5-metal D-MPNN pIC50 results | not in central ledger | surfaced via `pearson_r_pic50.json` + `dmpnn_ckpt_status.json` |
| Wf-85/86 PlatinAI 4-cell + tmQM 5-element cross-dataset R² | not in central ledger | surfaced via `hilbert_4cell_r2.json` + `platinai_combined_r2.json` + `metalcytotox_combined_r2.json` + `tmqm_combined_r2.json` |
| Wf-73 P2/P3 OOD reachability | not in central ledger | surfaced via `ood_per_metal_reachability.json` + `ood_per_element_reachability.json` |

### §4.2 §17 metric refresh today (on-disk facts)

| # | What changed | Source of truth |
|--:|---|---|
| 1 | **8 NEW files in `metrics/by_metric/`** | `molmetal/reports/wf87_p2_promotion/promotion_log.json` (n_promoted=8) |
| 2 | **125 files scanned** in `metrics/by_metric/` | `molmetal/reports/wf87_p1_disk_audit/per_metric_status.json` (`files_in_by_metric_dir: 125`) |
| 3 | **10/10 STRONG_REAL PASS** | `molmetal/reports/wf87_index_17_real/verify_index_17.md` (`STRONG_REAL cross-check: PASS`) |
| 4 | **22/22 cross-ref PASS** | `molmetal/reports/wf87_index_17_real/verify_index_17.md` (`Missing: 0`) |
| 5 | **17/17 promoter tests PASS** | `molmetal/tests/test_promote_metrics.py` |
| 6 | **3 §17.4 cross-ref entries added** (Wf-87 P1+P2+P3 outputs) | `TODO/INDEX.md` §17.4 |
| 7 | **§17.1 counts unchanged** (no fabricated STRONG_REAL bumps) | `TODO/INDEX.md` §17.1 (counts_claimed_in_index == actual) |

---

## §5 Honest framing (party-twenty-spirit §3.5)

Per `party-twenty-spirit` §3.5 (0-tolerance + 不掩盖):

| Finding | Evidence |
|---|---|
| **§17 STRONG_REAL count NOT bumped from 10 → 11+** | The 8 NEW promoted files are status / inventory / cross-dataset R² aggregators, NOT new conceptual metrics with measurable lift. Bumping §17 count without R20 GPU MEASURED would be **fabrication**. |
| **Wf-84 5-metal D-MPNN pIC50 PEARSON ≠ R17 ridge 0.572** | `pearson_r_pic50.json` `scope_caveat` explicitly states: cohort differ (MetalCytoToxDB 5-metal vs HeLa48h), per-metal headline (Ru 0.276) is best, aggregate is +0.061. Apples-to-oranges. |
| **Wf-85/86 honest-negative R² preserved** | `hilbert_4cell_r2.json` ships aggregate = **−6.7126** **(SUPERSEDED 2026-10-31 → −5.662524090882017; the framing below still holds)** with `status: MEASURED_BELOW_GATE` + `verdict: MEASURED_HONEST_NEGATIVE_R2_ARCHITECTURAL_CEILING` (the cell quoted the older truncated `MEASURED_HONEST_NEGATIVE_R2`; the artifact's verdict has carried the `_ARCHITECTURAL_CEILING` suffix in every commit that touches the file). NOT hidden behind a pseudo-positive summary. |
| **dmpnn_ckpt_status.json is a directory listing** | 12 ckpt metadata entries (10 NEW from Wf-84 + 2 from earlier affinity run). NOT inference results — downstream consumers must still load `metal_lomo_auc.json` / `platinai_3cell_resistance.json` for AUC values. |
| **vina_status.json reports host probe** | `meeko_installed: false`, `vina_binary_available: false` on this host (Wf-84 P2 wrapper in place at `molmetal/bin/vina` but bench-level auto-detect pending). NOT a method-level result. |
| **Wf-84 Wf-86 lifts require GPU retrain to land in §17** | Per `Wf-44 P1 root-cause catalog`: 8 root causes ranked, 2 CRITICAL (mean collapse + feature mismatch). R20-P2 Wf-40 launch gate applies catalog top-down. |
| **No paper touched** | Per `TODO/pending/84_no_paper_task_without_authorization.md` — no `paper/sections/*.tex`, no `paper/main.tex`, no reviewer rebuttal. |
| **0 GPU retrain executed today** | Wf-87 P1+P2+P3 are all CPU-only docs / file-move / JSON aggregation / tests. |

### §5.1 CORRECTION (2026-10-31) — the promoted headline value was −6.7126, and that value exists in NO commit

**The Wf-87 P2 promotion itself happened; only the VALUE attribution in the
two cells marked SUPERSEDED above was wrong.** This note does **not** rewrite
the promotion record — it corrects the record of **what the promotion ran
with**.

**MEASURED 2026-10-31 (`git` + the artifact on disk):**

  * `metrics/by_metric/hilbert_4cell_r2.json` carries
    `aggregate_combined_r2_mean` = **−5.662524090882017** at **HEAD** and in
    **EVERY** commit that touches the file — `9b12e6f5`, `c289aa42`,
    `fe209f61`, `293f8fb1`, `eb70929d`, `9a5e5c6f`, `7e30d6ea`, `6d3b309b`
    (**8 commits, 8/8 the same value**).
  * **−6.7126 exists in NO commit of that file**
    (`git log --all -S"-6.7126" -- metrics/by_metric/hilbert_4cell_r2.json` →
    **no hits**). The promoted value came from an **UNCOMMITTED working tree**.
  * **The ledger's own attribution of the change to commit `7e30d6ea` was
    FALSE.** That commit's diff on the JSON is **+49 / −0** — an appended
    `l5_feature_ceiling_probe_2026_10_08` block — and it **did not touch the
    headline**. The false row is `molmetal/reports/run_ledger.jsonl` line 23
    (`"value": -6.712634861296495`, `"superseded_value": -5.662524090882017`).
  * **The registry row `l5_hilbert_r2` was re-pointed by Wf-A123** and now
    publishes the committed value: `registry/metrics.yaml` →
    `l5_hilbert_r2` → `value: -5.662524090882017`,
    `value_basis: aggregate_combined_r2_mean`, `status: MEASURED_BELOW_GATE`,
    `source_json_sha256: 422a04b4321d2dcaa9ef9aee54b01d5b5963801adca2c0808c4c086f250f949e`
    (the committed file's sha256), `last_refreshed: 2026-10-30T20:20:22+00:00`.
    The ledger's own correction row (line 410) records the same re-pointing.
    **The repair is UNCOMMITTED** — see `docs/HONEST_NEGATIVE_CATALOG.md`
    §6.211 (net +1, cumulative 265 → 266).

**Why the honest-negative framing still holds.** −5.6625 is **still far below
any gate** — a negative held-out R² is still a negative R², so
`status: MEASURED_BELOW_GATE` and
`verdict: MEASURED_HONEST_NEGATIVE_R2_ARCHITECTURAL_CEILING` remain correct
and the framing is **unchanged**. Only the **magnitude** was wrong: −6.7126 was
**1.05 units** worse than the committed −5.6625. **The citability status did
not change**: `registry.py cite l5_hilbert_r2` returns **[NO]**.

**What is NOT corrected.** The **promotion itself happened**: the targets
listed in §2.2 exist on disk, and the **other seven rows of §2.2** and the
rest of §5 are **untouched**. (Separately: the promoter's on-disk
`promotion_log.json` was later **overwritten by a re-run** and now records
`promoted_count: 0` / `skipped_count: 9` with `resolution: preserved` on the
targets — the files were preserved, not re-promoted. That log is gitignored
(`molmetal/reports/**/*.json`); it is **outside this correction**.)

**0 GPU** was used (read-only `git` + JSON reads).

---

## §6 Cross-references

- **`molmetal/reports/wf87_p1_disk_audit/synthesis.md`** — Wf-87 P1 close-out (125-file scan, 26 conceptual metrics × 5 categories, on-disk state distribution)
- **`molmetal/reports/wf87_p1_disk_audit/per_metric_status.json`** — Wf-87 P1 per-metric audit JSON
- **`molmetal/reports/wf87_p2_promotion/synthesis.md`** — Wf-87 P2 close-out (8 NEW promoted files + collision + no-PROJECTED-leak + MEASURED_BELOW_GATE preservation)
- **`molmetal/reports/wf87_p2_promotion/promotion_log.json`** — Wf-87 P2 promotion log (n_promoted=8, n_skipped=0, n_errors=0, n_collisions=0)
- **`molmetal/reports/wf87_index_17_real/verify_index_17.json`** — Wf-87 P3 §17 STRONG_REAL cross-check JSON (10/10 PASS + 22/22 cross-ref PASS)
- **`molmetal/reports/wf87_index_17_real/verify_index_17.md`** — Wf-87 P3 close-out report
- **`molmetal/scripts/promote_metrics.py`** — NEW promoter script (~960 LOC)
- **`molmetal/scripts/verify_index_17_real.py`** — NEW §17 verifier script
- **`molmetal/scripts/audit_disk_metrics.py`** — Wf-87 P1 audit script (per `wf87_p1_disk_audit/synthesis.md` reference)
- **`molmetal/tests/test_promote_metrics.py`** — NEW promoter test file (~395 LOC, 17 tests)
- **`TODO/INDEX.md` §17** — TODO-89 5-Category Audit (10 STRONG_REAL / 11 WEAK_REAL / 5 MOCK / 3 DEFERRED)
- **`TODO/INDEX.md` §17.4** — cross-references (22 files, all exist)
- **`TODO/INDEX.md` §17.5** — TODO 100-107 R22 plan rows
- **`TODO/03_plans/LAMBDA_ROADMAP.md`** §1.2 + §2.4 + §4.2 (L1-L12 trajectory)
- **`TODO/completed/100_wf71_p3_wf49_cross_round2.md`** — Wf-71 P3 ship (cross-dataset macro R² +0.1248 source for `tmqm_combined_r2.json`)
- **`TODO/completed/85_mock_audit.md`** (Wf-42) — 8 critical + 23 major + 98 minor mock inventory (mocks flagged REVIEW_NEEDED in Wf-87 P1)
- **`TODO/completed/87l_undefined_metrics_definition.md`** (Wf-61) — 6 UNDEFINED metrics close (m6 + m9 + m20 MEASURED; 3 DESIGN_ONLY/DEFERRED host-gap → kept on disk with explicit `deferred_reason`)
- **`TODO/completed/87_r2_debug.md`** (Wf-44) — 5-phase R² debug close (R20-P2 Wf-40 launch gate)
- **`TODO/completed/89_wf43_routing_policy.md`** (Wf-43 P2) — 27 routed metrics + 4 MOCK_DELETED
- **`TODO/completed/91_wf49_r2_root_fix.md`** (Wf-49) — Phase 1+2 RDKit 2D descriptor swap
- **`TODO/completed/r20_status_ledger_2026_09_26.md`** (Wf-70 Phase 1) — 50 sub-tasks × 8 R20-P × 3 verdicts
- **`TODO/completed/100_r20_r21_consolidation.md`** (Wf-70 Phase 2) — L1-L8 → L1-L12 + INDEX §17 sweep
- **`TODO/pending/84_no_paper_task_without_authorization.md`** — paper-gate honored
- **`TODO/pending/r22_l13_l16_gap_analysis.md`** — 4 candidate Lambda gaps L13-L16 (R22 plan)
- **`TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md`** — 6 R22-P plan items (4 active + 2 reserved)
- **`.claude/skills/party-twenty-spirit/SKILL.md` §3.5** — 0-tolerance honest framing anchor (applied throughout Wf-87)
- **`molmetal/reports/wf84_dmpnn_ckpt/training_summary.json`** — Wf-84 P3 5-metal × 2-seed D-MPNN pIC50 retrain (Ru 0.276 best per-metal; aggregate +0.061; 10 ckpts at `molmetal/checkpoints/dmpnn_pic50_{metal}_seed{0,1}.pt`)
- **`molmetal/reports/wf84_dmpnn_ckpt/diagnostic_synthesis.md`** — Wf-84 P3 diagnostic synthesis (5-metal scaffold split + per-metal ckpt breakdown)
- **`molmetal/reports/wf85_p1_hilbert_only_real/synthesis.md`** — Wf-85 P1 Hilbert-only PlatinAI 4-cell
- **`molmetal/reports/wf85_p2_astr_only_real/synthesis.md`** — Wf-85 P2 ASTSR-only PlatinAI 4-cell
- **`molmetal/reports/wf85_p3_hilbert_astr_combined_real/synthesis.md`** — Wf-85 P3 Hilbert+ASTSR convex blend PlatinAI 4-cell
- **`molmetal/reports/wf86_p1_metalcytotox_real/`** — Wf-86 P1 REAL MetalCytoToxDB 5-metal cross-dataset R²
- **`molmetal/reports/wf86_p2_tmqm_real/synthesis.md`** — Wf-86 P2 REAL tmQM d-block 5-element cross-dataset R² (aggregate −0.1974)
- **`molmetal/reports/wf86_p3_platinai_4cell_real/synthesis.md`** — Wf-86 P3 REAL PlatinAI 4-cell combined R² (aggregate −1.5030)
- **`molmetal/scripts/audit_5cat_extend.py`** — Wf-72 Phase 3 22 → 25 metric audit extension

---

## §7 Projected STRONG_REAL candidates (R20 follow-up — NOT MEASURED today)

Per `party-twenty-spirit §3.5`, the following 4 metrics are STRONG_REAL **candidates** (not yet MEASURED-on-disk today) pending R20 GPU retrain:

| Candidate | R17 baseline | R20 plan | Current status |
|---|--:|---|---|
| **m1 Ru AUC** | 0.7123 (1-seed) | R20·P2 Wf-40 (5-seed × 50-epoch × {se3sb, tmqm}) | Wf-22 0.630 honest-negative; encoder arch mismatch from Wf-29 blocks lift |
| **m2 Pt 4-cell AUC** | 0.648 A2780cis / 0.584 MCF7 / 0.630 A549 | R20·P3 wet-lab IC50 dep | Signal-absence bottleneck per Wf-44 P1; wet-lab outreach R20-P6 |
| **m16 Affinity r** | 0.207 (HeLa48h) + 0.572 ridge ceiling | R20·P5 Wf-40 dual-cohort | Wf-32 smoke −0.040 on PlatinAI 4-cell (cohort-differ); no lift expected |
| **m19 Multi-metal LOMO** | 0.535 (Rh 0.368 bottleneck) | R20·P6 Wf-40 5-metal multitask pretrain | DEFERRED GPU-0 sweep |

**STRONG_REAL count today = 10** (MEASURED on disk). **STRONG_REAL count projected = 11** (only if Wf-86 P1 MetalCytoToxDB cross-dataset macro R² is formally added to §17 — currently surfaced via `metalcytotox_combined_r2.json` but NOT yet promoted to a §17.1 STRONG_REAL row). Per Wf-87 §3.2, the count bump happens ONLY when MEASURED R² ≥ 0.0 lands in central ledger — currently still PROJECTED for R20-P2.

---

## §8 Final verdict

**Wf-87 = ✅ SHIP**:
- 8 NEW promoted JSON files in `metrics/by_metric/` (≥5 NEW gate met)
- 17/17 promoter tests PASS
- 10/10 §17 STRONG_REAL cross-check PASS
- 22/22 §17.4 cross-ref file-existence PASS
- 0 fabricated MEASURED numbers (party-twenty-spirit §3.5 honored)
- 0 GPU retrain executed (CPU-only docs / file-move / aggregation phase)
- 0 paper touched (TODO-84 gate honored)
- §17.1 STRONG_REAL count stays 10 (no bump; honest)

**STRONG_REAL → STRONG_REAL count = 10 → 10 (MEASURED) + 1 PROJECTED (Wf-86 P1 MetalCytoToxDB lift if R20 GPU R² ≥ 0.0 lands)**. Path to STRONG_REAL = 11 is **Wf-86 R20 GPU sweep**, not Wf-87 docs cycle.

**Date**: 2026-09-26 (Wf-87 close-out, post-Wf-84/85/86 real-data sweep)
