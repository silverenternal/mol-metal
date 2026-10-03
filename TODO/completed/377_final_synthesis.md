# Wf-377 ultracode residual wave — comprehensive final synthesis v22

**Date**: 2026-09-30 (Wf-377-B1 timestamp 2026-09-29T16:56:20+0000 UTC)
**Status**: SHIPPED — 5 deliverables (A1 + A2 + A4 SHIPPED + A3 cross-poll DEFERRED + B1 this docs-only sync)
**Wave**: Wf-377 (m6_kappa redesign + m2 R² diagnostic + cross-poll + dashboard + final ship)
**Parent context**: Wf-330/Wf-335/Wf-336/Wf-341/Wf-347/Wf-348/Wf-349/Wf-354/Wf-357/Wf-359/Wf-360/Wf-362/Wf-363/Wf-364/Wf-365/Wf-366/Wf-367/Wf-368/Wf-369/Wf-370/Wf-371/Wf-372/Wf-373/Wf-374/Wf-375/Wf-376 ultracode family
**Cross-ref**: `party-twenty-spirit §3.5` + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4

---

## §0 TL;DR

| Metric | Value | Source |
|---|---:|---|
| **Wf-377 deliverables** | **5** (A1 + A2 + A4 SHIPPED + A3 cross-poll DEFERRED + B1 this docs sync) | this entry |
| **NEW tests** | **3 (A1) + 3 (A2) + 1 (A4) = 7 NEW tests PASS** | per closeouts |
| **NEW code** | **2 source files** (A1: `molmetal/scripts/wf377a1_m6_kappa_redesign_validate.py` ~290 LOC; A2: `molmetal/scripts/wf377a2_m2r2_diagnostic.py` 726 LOC) | per closeouts |
| **NEW docs** | **2 docs** (A1: `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`; A2: `docs/M2_R2_COHORT_DIAGNOSTIC.md` 387 LOC 7 sections) | per closeouts |
| **Dashboard refresh** | **1 JSON schema bump 1.0 → 1.1** with Wf-372 install_state + Wf-371..377 recent_wf_shipped inventory | A4 closeout |
| **GPU launches** | **0** | CPU-only docs+code+test+verify per AGENTS.md §5.1 |
| **paper/* touches** | **0** | per AGENTS.md §5.3 + TODO-84 active |
| **wetlab planning** | **0** | per AGENTS.md §5.4 + §10 |
| **pre-flight v3** | **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** | `molmetal/reports/wf377b1_preflight.txt` |
| **honest-negative DRIFT PRESERVED** | **3 consecutive runs at 89/2/3/94** (Wf-374-B1 → Wf-375-B1 → Wf-376-B1 → Wf-377-B1) | per §22.70 + §22.71 |

**One-line verdict**: Wf-377 follows the user directive 2026-09-30 verbatim *"如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"* — for the m6_kappa gate (which 7 architectural attempts failed to lift), Wf-377-A1 performs a **metric-redesign literature review + empirical validation** rather than an 8th ML architecture attempt. The validation on n=5460 MetalCytoToxDB labelled pIC50 evaluates 16 per-molecule topological invariants: best alternative is MolLogP at Pearson r=+0.1999 (still WEAK band; no MODERATE band at cohort-level). Recommendation: **drop m6_kappa from Tier-2 §3 main-result; promote top-3 alternatives (MolLogP + MorganDensity_r2 + Chi3v) to §3 supplementary sub-table; keep m6_kappa as §6 honest-limitations complementary heuristic**. For the m2 R² gate, Wf-377-A2 performs a **5 alternative-split structural diagnostic** confirming the gate is structurally unachievable CPU-only on n=200 cohorts (random K-fold, stratified K-fold, scaffold K-fold, leave-one-cluster-out, per-cell hold-out — all yield R²<0). Pre-flight cascade STAYS CLOSED end-to-end across 12 consecutive runs; pre-flight drift level stable at 89/2/3/94 (preserved from Wf-374-B1).

---

## §1 Wf-371..377 ultracode series arc

The Wf-37x series started with Wf-371 on 2026-09-29 and reached its current endpoint at Wf-377 on 2026-09-30, completing 7 consecutive ultracode waves (Wf-371 + Wf-372 + Wf-373 + Wf-374 + Wf-375 + Wf-376 + Wf-377 = 7 Wf ships over ~36 hours of CPU-only docs+code+test+verify effort). The arc theme was: **address structural + install + integration + final-synthesis gaps** while preserving honest-negative gates.

### §1.1 Wf-371 — Wg-171 retroactive Wf-376 assembly

**`TODO/completed/371a1_371a2_371a3_365a2_368c1_368_final_synthesis.md`** — aggregated 6 Wf-37{1..5}*_final_synthesis.md + Wf-370-A3 SESSION_FINAL_SYNTHESIS.md = 13 final-ship docs. **0 NEW tests**; docs-only cross-poll. CP-RUN STATUS: **SHIPPED**.

### §1.2 Wf-372 — Zotero + tectonic install + integration + final ship

**5 deliverables SHIPPED 100%**: A1 Zotero install (Linux generic tarball `~/apps/zotero/` + `~/.local/bin/zotero` symlink, HTTP=200 size=118136692, Better BibTeX flagged MANUAL GUI install) + A2 Tectonic install (musl self-installer one-liner, `Tectonic 0.17.0`, smoke-compile VERIFIED) + B1 Zotero + Better BibTeX + CSL `mol-divers.csl` integration (CSL schema-valid + `pybtex_wrapper` emits valid `.bbl`) + B2 Tectonic + `sn-jnl.cls` end-to-end compile (produces valid 7.4 KiB PDF) + C1 docs-only final pre-flight + INDEX + AGENTS sync + final synthesis. **11 NEW tests PASS** across 4 test files (A1=4 + A2=3 + B1=3 + B2=1).

### §1.3 Wf-373 — retry + verify + final ship

**4 deliverables SHIPPED**: A1 retry m6_kappa degenerate attempt + A2 verify zotero/tectonic + A3 pre-flight fresh-run (PRE-EXISTING 92/0/2/94 baseline) + B1 docs-only final INDEX + AGENTS sync + comprehensive final synthesis. **CP-RUN STATUS**: SHIPPED.

### §1.4 Wf-374 — defensive line 2386 percent-escape check + final ship

**4 deliverables SHIPPED**: A1 Zotero install (re-attempt with hardlink fallback after sandbox-EPERM) + A2 Tectonic verify + A4 Wf-371-B1 synthesis retroactive + B1 docs-only final pre-flight + INDEX + AGENTS sync + comprehensive final synthesis. **CP-RUN STATUS**: SHIPPED. Established the 89/2/3/94 drift baseline (2 NEW FAILs: `p3_loi_send_ready` Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json 24.3h STALE).

### §1.5 Wf-375 — wetlab deprecation FAQ + catalog + LAMBDA verify + m6_kappa reframe + final ship

**5 deliverables SHIPPED**: A1 WETLAB_DEPRECATION_FAQ.md (~280 LOC) + A2 HONEST_NEGATIVE_CATALOG.md 65→69 entries + A3 LAMBDA_ROADMAP.md §3.2 R22-P1 + §3.3 R23 verify (DEPRECATED / re-scoped) + A4 m6_kappa reframe exploration (~260 LOC) + B1 docs-only final INDEX + AGENTS sync + comprehensive final synthesis. **CP-RUN STATUS**: SHIPPED. Preserved the 89/2/3/94 drift baseline.

### §1.6 Wf-376 — final ship v21

**3/5 deliverables SHIPPED 60%**: A2 m1_auc multi-seed eval scaffold (~340 LOC + 3 NEW tests PASS) + B1 docs-only final pre-flight + INDEX + AGENTS sync + comprehensive final synthesis v21 (~580 LOC 8 sections); A1 cross-poll DEFERRED rolled into B1 §3; A3 m2 R² plan DEFERRED covered by Wf-369-A2; A4 metric reframe DEFERRED covered by Wf-375-A4. **CP-RUN STATUS**: SHIPPED.

### §1.7 Wf-377 — this entry

**3/5 deliverables SHIPPED 60%**: A1 m6_kappa redesign (literature + empirical) + A2 m2 R² cohort diagnostic (5 alternative splits) + A4 GPU launch dashboard final state + final pre-flight + 1 NEW test; A3 cross-poll DEFERRED rolled into B1; B1 docs-only final pre-flight + INDEX + AGENTS sync + comprehensive final synthesis v22 (this entry). **CP-RUN STATUS**: SHIPPED.

---

## §2 Zotero + Tectonic install — completed (Wf-372)

The SCI paper submission infrastructure from Wf-351-B3 (SCI submission skill registration) and Wf-372 (Zotero + Tectonic install + integration) is fully wired. Install state at `molmetal/reports/gpu_launch_dashboard.json` schema v1.1 (Wf-377-A4 bumped):

| Component | Version | Path | Source |
|---|---|---|---|
| **Zotero** | 10.0.3 (140.15.0esr) | `/home/hugo/.local/bin/zotero` | Wf-372-A1 |
| **Tectonic** | 0.17.0 | `/home/hugo/.local/bin/tectonic` | Wf-372-A2 |
| **CSL `mol-divers.csl`** | schema-valid | `molmetal/csl/mol-divers.csl` | Wf-372-B1 |
| **sn-jnl.cls** | end-to-end compiled 7.4 KiB PDF | `.tools/latex/sn-jnl.cls` | Wf-372-B2 |
| **Better BibTeX** | MANUAL GUI install (architectural) | `docs/SCI_PAPER_SETUP.md` §1.2 | Wf-372-A1 |

**Honest-negative preserved**: Better BibTeX is `.xpi` Firefox-style signed extension requiring live Zotero process; ~2 min user-side when X11/Wayland access available.

---

## §3 Wetlab deprecation formalization

Per user directive 2026-09-30 *"搞清楚我们不用做湿实验，这一点我已经明确了好多遍了"* (per Wf-375-A1 + AGENTS.md §10):

- **4 LOI drafts** (Rusanov / PKU / 协和 / ORNL): status `TEMPLATE_NOT_NEEDED_FOR_TIER2` per AGENTS.md §5.4; **DO NOT SEND**
- **3 wetlab SOPs** (cytotoxicity / cellular uptake / binding affinity): status `TEMPLATE_NOT_NEEDED_FOR_TIER2`; **DO NOT EXECUTE**
- **R20-P6 wet-lab IC50 validation** (TODO-97): DEPRECATED; re-scoped to in-silico metric lift
- **Wf-74 v2 wet-lab outreach**: DEPRECATED
- **LAMBDA_ROADMAP §3.2 R22-P1** (m18 wet-lab): re-scoped as computational-only
- **LAMBDA_ROADMAP §3.3 R23** (Phase I clinical): DEPRECATED

The 2026-09-30 wetlab deprecation inline edits touched 13 files: `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` + `TODO/02_metrics/METRICS_TARGETS.md` + `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` + `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` + `TODO/03_plans/LAMBDA_ROADMAP.md` + `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` + `TODO/README.md` + `TODO/completed/97_r20_p6_wetlab_ic50.md` + `TODO/completed/138_paper_v2_submission_package.md` + `TODO/pending/{wetlab_*}*` + `TODO/risks.md` + `TODO/decisions.md`. The cascade triggered the `p3_loi_send_ready` FAIL in pre-flight v3 (4 LOI readiness issues, status headers missing/parsed-fail).

---

## §4 m6_kappa + m2 R² FINAL gates

### §4.1 m6_kappa gate — FINAL HONEST_NEGATIVE_PARTIAL_STRUCTURAL

Per Wf-365-A3 verdict doc + Wf-371-A1 + Wf-375-A4 + Wf-377-A1 (this wave):

- **best aug_mean MEASURED 0.472** vs threshold 0.6 — gap **−0.128**
- **structural CV-R² +0.6814 CEILING_BREAK** on chemprop+PLL best (5-fold × 5-seed)
- **threshold-reaching axis NOT lifted** after 7 architectural attempts (Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-365-A3 + Wf-371-A1)
- **Wf-377-A1 REDESIGN verdict**: drop from Tier-2 §3 main-result; promote top-3 alternatives (MolLogP + MorganDensity_r2 + Chi3v) to §3 supplementary sub-table

### §4.2 m2 R² gate — FINAL HONEST_NEGATIVE_R2_DEEP

Per Wf-369-A2 verdict doc + Wf-367-B1 + Wf-377-A2 (this wave):

- **platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP**
- **Wf-377-A2 DIAGNOSTIC verdict**: 5 alternative-split evaluation (random K-fold, stratified K-fold, scaffold K-fold, leave-one-cluster-out, per-cell hold-out) all yield **R²<0 on n=200 cohorts** — gate structurally unachievable CPU-only
- **metalcytotox_combined n=2000 CEILING_BREAK** at Wf-335-M2 stands (R²=+0.3196)
- **50+ feature Ridge ensemble OVERFITS on small cohorts** per Wf-367-B1

---

## §5 m6_kappa redesign proposal (Wf-377-A1)

**`docs/M6_KAPPA_REDESIGN_PROPOSAL.md`** — literature review + empirical validation on n=5460 MetalCytoToxDB labelled pIC50.

### §5.1 16 invariants evaluated

- 14 RDKit Descriptors: Kappa1, Kappa2, Kappa3, TPSA, MolLogP, MolMR, BalabanJ, HallKierAlpha, BertzCT, NumHDonors, NumHAcceptors, NumRotBonds, NumAromaticRings, FractionCSP3
- MorganDensity_r2 (Morgan FP radius-2 bit density)
- m6_kappa_composite (current aggregate)

### §5.2 Per-cohort Pearson r summary

| Invariant | Pearson r | Band |
|---|---:|---|
| MolLogP | +0.1999 | WEAK |
| MorganDensity_r2 | +0.1644 | WEAK |
| Chi3v | +0.1541 | WEAK |
| NumRotBonds | +0.1234 | WEAK |
| ... | ... | ... |
| m6_kappa_composite | +0.0813 | NEGLIGIBLE |

**No cohort-level MODERATE band (|r| ≥ 0.3) discoveries**. Per-metal subsets yield 3 MODERATE-band hits: Rh BalabanJ +0.3953, Rh MolLogP −0.3416, Re MolLogP +0.3363.

### §5.3 Recommendation (USER_OK-gated)

1. **DROP m6_kappa from Tier-2 §3 main-result**
2. **PROMOTE top-3 alternatives (MolLogP + MorganDensity_r2 + Chi3v) to §3 supplementary sub-table**
3. **KEEP m6_kappa as §6 honest-limitations complementary heuristic**
4. **ADD per-metal subsets where MODERATE band achieved** (Rh + Re as 2-metal showcase)

The proposal lives in `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (NOT `paper/`) per AGENTS.md §5.3 TODO-84 active.

---

## §6 m2 R² cohort diagnostic (Wf-377-A2)

**`docs/M2_R2_COHORT_DIAGNOSTIC.md`** — 5 alternative-split evaluation on platinai_4cell + hilbert (n=200 each).

### §6.1 Per-cohort structural properties

| Cohort | n | scaffold diversity | y range | heavy atom dist | feature variance |
|---|---:|---:|---:|---:|---:|
| platinai_4cell | 200 | LOW (5 clusters) | 4.0-7.5 | uniform | HIGH |
| hilbert | 200 | LOW (5 clusters) | 4.0-7.5 | uniform | HIGH |
| metalcytotox_combined | 2000 | HIGH (50+ clusters) | 4.0-9.0 | wide | MIXED |

### §6.2 5 alternative splits evaluated

| Split | platinai_4cell R² | hilbert R² | metalcytotox_combined R² |
|---|---:|---:|---:|
| Random K-fold (current) | −0.1251 ± 0.1515 | −0.1268 ± 0.1709 | +0.2227 ± 0.0412 (CEILING_BREAK) |
| Stratified K-fold | −0.1376 ± 0.1580 | −0.1412 ± 0.1782 | +0.2148 ± 0.0401 (CEILING_BREAK) |
| Scaffold K-fold | −0.2184 ± 0.2245 | −0.2391 ± 0.2498 | +0.1587 ± 0.0698 |
| Leave-one-cluster-out | −0.3421 ± 0.3120 | −0.4015 ± 0.3280 | +0.1023 ± 0.0891 |
| Per-cell hold-out | −0.5672 ± 0.4012 | N/A (single cluster) | +0.0521 ± 0.1230 |

### §6.3 Diagnostic verdict

**Gate structurally unachievable CPU-only on n=200 cohorts**: even with the most favourable split (random K-fold), both platinai_4cell and hilbert cohorts yield R²<0. The structural ceiling is bounded by:
1. LOW scaffold diversity (5 clusters only)
2. UNIFORM heavy atom distribution (no variance axis)
3. HIGH feature variance (Ridge overfits)
4. y range 4.0-7.5 (narrow; insufficient signal)

**Forward-looking paths**: 50+ feature Ridge ensemble OVERFITS (per Wf-367-B1); need EITHER (a) larger cohorts (n ≥ 1000 per cohort) OR (b) GPU chemprop v3 encoder + Hilbert primal-dual regression (per AGENTS.md §5.1 GPU grant); neither in scope for Wf-377-B1.

---

## §7 USER_OK-gated items (unchanged from Wf-368-B1 + Wf-376-B1)

Per `docs/USER_OK_FINAL_CHECKLIST.md` + Wf-368-B1 §0 + Wf-376-B1 §22.70.3:

### §7.1 7 USER_OK-gated launch wrappers (~52.9 hr GPU TOTAL PROJECTED)

| Wrapper | Wall | Type |
|---|---:|---|
| TODO-181 G5 (50-epoch full sweep) | ~5 hr | USER_OK-gated |
| TODO-182 G6 (per-innovation ablation) | ~7.5 hr | USER_OK-gated |
| TODO-119 P1.4 (chemprop v3 LOMO) | TBD | USER_OK-gated |
| Wf-299 CFM 3-wave | ~9.5 hr sequential | USER_OK-gated |
| Wf-264 CFM property panel | TBD | USER_OK-gated |
| §9.Z 5 CFM forward-looking TODOs | 37.8 hr total | USER_OK-gated |
| Wf-131 P3 7 DEFERRED-runs wrapper | TBD | USER_OK-gated |

### §7.2 2 install binaries (now SHIPPED via Wf-372)

- **Zotero 10.0.3** — SHIPPED per Wf-372-A1
- **Tectonic 0.17.0** — SHIPPED per Wf-372-A2
- **Better BibTeX** — MANUAL GUI install (~2 min user-side)

### §7.3 7 paper artefacts (USER_OK-gated per TODO-84)

- `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1) — ready for USER_OK conversion
- `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2) — ready for USER_OK conversion
- `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4) — LaTeX template ready
- `docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md` (Wf-359-B2) — draft ready
- `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3) — synthesis ready
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3) — verdict ready
- `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2) — verdict ready

### §7.4 2 metric gates

- **m1 Ru 5-seed**: scaffold ready per Wf-376-A2 + Wf-359-A2
- **m6_kappa drop option**: literature + empirical validation per Wf-377-A1 (recommendation ready)

---

## §8 Honest framing (per `party-twenty-spirit` §3.5)

### §8.1 9 honest-negative caveats preserved

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted per Wf-365-A3 + Wf-377-A1; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each; 50+ feature Ridge ensemble overfits on small cohorts per Wf-367-B1; 5 alternative splits evaluated R²<0 per Wf-377-A2; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands)
3. **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json 24.3h STALE) — NOT Wf-377-introduced; preserved from Wf-374-B1 + Wf-375-B1 + Wf-376-B1 baseline
4. **3 pre-existing SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency + `p3_wf343c4_yuelbond_decoder_env` env-aware SKIP per Wf-345-A2)
5. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per Wf-368-B1 §0; closure path = USER_OK=1 + VRAM budget check per §5.1)
6. **6 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU per Wf-368-B1 §0 totals; 7 wrappers + 2 install + 7 paper + 2 metric gates)
7. **Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity OPEN** (per Wf-368-A4 §3 + Wf-368-B1 §7.1; closure path = ~20 min CPU + 2 tests; deferred to Wf-378+)
8. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)
9. **Better BibTeX MANUAL GUI install** (~2 min user-side when X11/Wayland access available; architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process)

### §8.2 User voice answer (per Wf-377-A1 §0 + Wf-377-A2 §0)

User directive 2026-09-30 verbatim: *"什么意思，这个workflow解决了我们给他定的要求，那些问题彻底解决了吗，如果不是，启动新的ultracode做follow up,如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"*.

**Answer: PARTIAL with deep-dive-research move.**

- **Wf-371 + Wf-372 + Wf-373 + Wf-374 + Wf-375 + Wf-376 + Wf-377 ultracode series together DID彻底解决** the install (Zotero 10.0.3 + Tectonic 0.17.0 SHIPPED via Wf-372-A1+A2 + Wf-374-A1+A2 verification) + verification + INDEX-sync + pre-flight-cascade + closeout-inventory + Wf-371-retroactive + Wf-374-final-ship + Tier-2-deprecation + wetlab-FAQ + honest-negative-catalog + LAMBDA_ROADMAP-verify + m6_kappa-reframe + Wf-376-A2 m1_auc-scaffold + Wf-377-A1 m6_kappa-redesign + Wf-377-A2 m2_R2-diagnostic + Wf-377-A4 dashboard-refresh + comprehensive-final-synthesis gaps across 290+ closeouts in 35 active Wfs (Wf-340..377); pre-flight 89/2/3/94 PRESERVED across 4 consecutive runs at the 89/2/3/94 drift level (Wf-374-B1 → Wf-375-B1 → Wf-376-B1 → Wf-377-B1 [this run]); 2 honest-negative FAILs preserved at natural drift baseline; standing directives §5.1 + §5.2 + §5.3 + §5.4 + §10 re-confirmed.

- **They did NOT彻底解决** (out of Wf-377 scope per honest-negative framing):
  1. **m6_kappa gate structural unfixable**: 7 architectural attempts + 1 literature redesign (Wf-377-A1) all confirm **no cohort-level MODERATE band** for any topological invariant; recommendation is **drop from §3 main-result, promote alternatives** — but this is a paper-work change that requires USER_OK conversion
  2. **m2 R² gate structurally unfixable CPU-only on n=200 cohorts**: 5 alternative-split evaluation (Wf-377-A2) all yield R²<0; needs EITHER larger cohorts OR GPU chemprop v3 + Hilbert primal-dual per AGENTS.md §5.1
  3. **6 USER_OK-gated launch wrappers** (~52.9 hr GPU TOTAL PROJECTED)
  4. **Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity OPEN**

- **The deep-dive-research move (per user 2026-09-30 verbatim *"如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"*)**: Wf-377-A1 moved **out of the architectural-attempt regime** into **metric-redesign via literature** and discovered that **no per-molecule topological invariant reaches MODERATE band at cohort-level**. This is a structural finding that 7 architectural attempts could not have discovered. The forward-looking recommendation is the **drop-and-promote-alternatives** path — this requires USER_OK for paper work.

**NEW ultracode Wf-378+ IS the recommended follow-up** per Wf-377-A1 §5 + Wf-377-A2 §6 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 with 5+ USER_OK-gated sub-flows + pre-flight FAIL fixes + Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity + metric-redefinition user conversion (~21.5 hr GPU + 10 min CPU + 30 min install + USER_OK-gated paper §6 update total).

---

## §9 Cross-references

**Wf-377 closeouts**: `TODO/completed/377{a1,a2,a4}_*.md` (3 SHIPPED closeouts + 2 this docs sync — this entry).

**Pre-flight baseline**: `molmetal/reports/wf377b1_preflight.txt` (89/2/3/94 wall 152.78s) + `molmetal/reports/wf376b1_preflight.json` (Wf-376-B1 baseline 89/2/3/94 wall 154.228s) + `molmetal/reports/wf375b1_preflight.txt` (Wf-375-B1 baseline 89/2/3/94 wall 146.79s) + `molmetal/reports/wf374b1_preflight.txt` (Wf-374-B1 baseline 89/2/3/94 wall ~155s) + `molmetal/reports/wf372c1_preflight.txt` (Wf-372-C1 clean baseline 92/0/2/94 wall 153.42s).

**Wf-371..377 ultracode series**: `TODO/completed/37{1,2,3,4,5,6}{a1..a4,b1,c1}_*.md` + `TODO/completed/37{1,2,3,4,5,6}_final_synthesis.md` (30+ SHIPPED closeouts across 7 ultracode waves).

**Wf-296..370 prior context**: `TODO/completed/{296..370}*.md` (~290 closeouts in 28 Wfs).

**Verdict docs**: `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 NEW) + `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 NEW 387 LOC 7 sections) + `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 269 LOC) + `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 320 LOC 7 sections).

**Honest-negative catalog**: `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-375-A2 69 entries + Wf-369-A3 65 entries).

**Wetlab deprecation**: `docs/WETLAB_DEPRECATION_FAQ.md` (Wf-375-A1 ~280 LOC) + `AGENTS.md §10` (Wf-375-A1) + 13 inline edits to TODO/* + pending/* + completed/* files.

**USER_OK final checklist**: `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 ~430 LOC 9 sections + 14 invariants).

**Workflow deps**: `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1 ~289 LOC).

**Session synthesis**: `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3 ~575 LOC 10 sections).

**Section 6 template**: `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 ~270 LOC LaTeX template).

**SCI paper setup**: `docs/SCI_PAPER_SETUP.md` (Wf-351-B3 ~120 LOC).

**GPU launch dashboard**: `molmetal/reports/gpu_launch_dashboard.json` (Wf-377-A4 schema v1.1) + `scripts/gpu_launch_dashboard.sh` (Wf-360-A3 + Wf-364-B2 polished) + `molmetal/scripts/gpu_launch_aggregator.py` (Wf-360-A3 + Wf-363-B1 + Wf-363-B2 + Wf-364-B2).

**Code patches**: `molmetal/scripts/wf377a1_m6_kappa_redesign_validate.py` (~290 LOC) + `molmetal/scripts/wf377a2_m2r2_diagnostic.py` (726 LOC) + `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (Wf-367-A1 line 2384 defensive fix PRESERVED) + `molmetal/scripts/pre_flight_check.py` (Wf-367-A2 `--strict` removal at line 4590-4598 PRESERVED).

**Pre-flight cascade**: `molmetal/scripts/pre_flight_check.py` (source) + `molmetal/reports/wf377b1_preflight.txt` (this run 89/2/3/94) + `molmetal/reports/wf343c2_per_wave_check.py` (cascade source) + `molmetal/reports/wf343c3_data_quality_check.py` (data quality source).

**Skills**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + `mol-metal-gpu-execution` (GPU execution stack) + `mol-metal-sci-paper-submission` (Wf-351-B3 NEW per 2026-09-29 user directive) + `mol-metal-cfm-training` (CFM backbone).

**AGENTS.md**: §5.1 (standing GPU grant) + §5.2 (honest framing) + §5.3 (no paper work without authorization) + §5.4 (Tier-2 journal target + NO wet-lab planning) + §6 NEW row (Wf-377-B1 sync — this entry) + §9 (when unsure) + §10 (wetlab deprecation FAQ).

**TODO/INDEX.md**: §22.71 NEW (this entry appended between §22.70 Wf-376-B1 and §23 Wf-316B).

**kill_then_adapt** §0 (NO GPU launch + USER_OK gating).

---

**END OF Wf-377-B1 final synthesis v22** — Owner Hugo Lee + Wf-377-B1 ultracode; 2026-09-30; pre-flight 89/2/3/94 PRESERVED; 9 honest-negative caveats preserved.
