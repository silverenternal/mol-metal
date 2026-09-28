# Wf-132 Phase 1 — §17 Audit Data Assembly

**Date**: 2026-09-27
**Status**: ASSEMBLED — 3 audit reports + cross-linked with LAMBDA_ROADMAP §4.2 + METRICS_TARGETS §3
**Owner**: Wf-132 (ultracode, agent-orchestrated audit reconciliation)
**Skill alignment**: `party-twenty-spirit` §3.5 (honest framing mandatory)
**Purpose**: §17 INDEX.md data table; trace-back every number to a metric JSON path

---

## §1 Audit meta — 3 audits × data sources × key conclusions

| # | Audit ID | Source | Scope | Data files scanned | Key conclusion |
|---:|---|---|---|---:|---|
| A1 | TODO/ directory audit | `/tmp/claude-1001/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/tasks/a752cb5a9d567e299.output` | TODO/ + metrics/ | **310 .md** + **154 .json** | 47% real + 53% mock/partial; 6 lead-claim 真实强数据; 8 honest-negative ship; 3 P0 critical pending; wet-lab 完全空 |
| A2 | metrics/ JSON audit | `/tmp/claude-1001/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/tasks/ae11e0dcf4486f793.output` | `metrics/by_metric/*.json` | **113 files** | 27 ✅ + 38 🟡 + 3 🔴 + 32 ⚪ + 3 🔵 — paper-grade 真 实 测 ≈ 9 metric families |
| A3 | Wf in-flight audit | `/tmp/claude-1001/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/tasks/aa886838a7e835f70.output` | wf_* journal + TODO/completed/ | **13 Wf in-flight** | 11 in-flight + 1 orphan (Wf-128 — NO disk dir, NO TODO reference) + 7 ship-without-closeout |

**Cross-link anchors used**:
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L1-L12 trajectory)
- `TODO/02_metrics/METRICS_TARGETS.md` §3 (21-metric paper §4 template) + §5 (honest-negative framing)
- `TODO/INDEX.md` §17 (target slot — to be appended by Wf-132 Phase 2)

**Critical honesty note** (party-twenty-spirit §3.5):
- Audit was generated **2026-09-26 19:20 UTC** (single conversation, no live re-run).
- Audit is a **snapshot** — between audit and now (2026-09-27), 2 close-outs have shipped (Wf-121 P0 partial lift + Wf-123 M1+M2 modules) and Wf-121 P3 cisplatin × 10-pocket real Vina was MEASURED — replacing legacy -9.6574 stub floor for metallodrug cohort.
- Numbers in §2/§3/§4 are verbatim from audit, with explicit "AUDIT-DATE 2026-09-26" stamps. Current state (post-audit deltas) noted in §6 and §7.

---

## §2 Lead-claim 6 strong metrics (verbatim numbers + paths)

| # | Lead claim | Verbatim value | Source path | n | Cross-link |
|---:|---|---:|---|---:|---|
| 1 | **n_distinct 1→20 lift** (Lambda MCTS, Path A 4-fix bundle) | **20 distinct / cell** | `/home/hugo/codes/mol-metal/metrics/by_metric/n_distinct.json` `measurements[4].value` (`wf_round12_lambda_patha_10x3`, n_sim=1000) | 30 cells | METRICS_TARGETS §2 M11 (target ≥3, R12 baseline 1) |
| 2 | **DrugOOD assay-OOD AUC** m17 (beats iMoLD 0.7211) | **0.8913 ± 0.0019** | `/home/hugo/codes/mol-metal/metrics/by_metric/drugood_ood_auc.json` `results.assay.ood_auc_mean` | 3 seeds × 30 epochs | METRICS_TARGETS §2 M17 (iMoLD 72/68/68 → ≥-0.02) |
| 3 | **DrugOOD scaffold-OOD AUC** m17 (beats iMoLD 0.6884) | **0.6973 ± 0.0271** | `drugood_ood_auc.json` `results.scaffold.ood_auc_mean` | 3 seeds | 同上 |
| 4 | **DrugOOD size-OOD AUC** m17 (beats iMoLD 0.6792) | **0.7479 ± 0.0055** | `drugood_ood_auc.json` `results.size.ood_auc_mean` | 3 seeds | 同上 (note: data_source=`metacytotox_fallback`, NOT full DrugOOD — §5 honest-negative) |
| 5 | **SA mean** (Ertl 2008, EV-3 30-cell) | **3.0987596 ± 0.0** | `/home/hugo/codes/mol-metal/metrics/by_metric/sa_mean.json` `best_measured` | 30 cells × n_sim=1000 | METRICS_TARGETS §2 M5 (TargetDiff 2.65-2.86, -0.24 miss) |
| 6 | **rdkit_validity** (M23+M38+M39 short-circuit) | **1.000** (30/30) | `/home/hugo/codes/mol-metal/metrics/by_metric/rdkit_validity.json` | 30/30 smoke | METRICS_TARGETS §2 M13 (target ≥0.99, SOTA 0.99-1.00) |
| 7 | **NCI-60 9-cell AUC** m21 (Tier-1 gate 0.70 PASS) | **0.7742 ± 0.0086** (overall 0.8007) | `/home/hugo/codes/mol-metal/metrics/by_metric/nci60_precious_9cell_auc.json` `value_overall_mean_auc` | 3 seed × 50 epoch × scaffold-split, n=5207 | METRICS_TARGETS §2 M21 (gate ≥0.70, Krasnov Ru 0.81) |
| 8 | **hERG cardio risk** pool-dominant | **0.255** | `/home/hugo/codes/mol-metal/metrics/by_metric/herg_cardio_risk.json` | 30 cells EV-1 (cisapride ref 0.253) | METRICS_TARGETS §2 M22 bonus (anchor: cisapride 0.253) |

**Paper §4 候选**: n_distinct 20 (§4.6 headline); DrugOOD 3-split 全部 beat iMoLD (§4.5 OOD); SA 3.099 (§4.6 synth); NCI-60 9-cell 0.7742 (§4.1 Tier-1 same-domain); rdkit_validity 1.000 (§4.2 validity); hERG 0.255 (§4.5 drug-likeness).

---

## §3 Honest-negative 8 metrics (verbatim numbers + paths)

| # | Metric | Verbatim value | Source path | Cross-link / paper §6 anchor |
|---:|---|---:|---|---|
| 1 | **m1 Ru AUC** (1-seed 5-epoch) | **0.7123** | `/home/hugo/codes/mol-metal/metrics/by_metric/metalcytotox_ru_auc.json` | METRICS_TARGETS §3 m1 (Krasnov 0.81 → beat ≥0.86); Wf-131 in-flight 5-seed |
| 2 | **m3 Vina** (single-pocket MMP13 830c, CuAAC) | **-1.3546 ± 0.1891** kcal/mol (n=100) | `/home/hugo/codes/mol-metal/metrics/by_metric/vina_score.json` | METRICS_TARGETS §3 m3 (TargetDiff -8.45 → ≤-8.5, gap +7.10); post-audit: Wf-121 P3 cisplatin × 10-pocket MEASURED kcal/mol -2.548 to -3.734 (replaces stub-floor for cisplatin cohort; full 8 d-block sweep deferred) |
| 3 | **m7 Diversity** (Morgan ECFP4, EV-3) | **0.137** | `/home/hugo/codes/mol-metal/metrics/by_metric/diversity_tanimoto.json` | METRICS_TARGETS §3 m7 (TargetDiff 0.860 → ≥0.65, gap -0.753) |
| 4 | **m14 decode_ratio** (CPU 4-fix bundle) | **0.0 / 8** | `/home/hugo/codes/mol-metal/metrics/by_metric/decode_ratio.json` `measurements[5]` (`wf_cfm_rescue`, all 4 fixes) | METRICS_TARGETS §3 m14 (R16 0/192 → ≥0.05); Path B bond-bearing 192/192 PASS, RDKit-strict 0.5 on synthetic; Wf-124 in-flight YuelBond decoder wire |
| 5 | **m15 RMSD** (uniform 1.241 Å — self-RMSD no-pocket) | **0.428 ± 0.0** (partial staging); per-pocket all 1.241 (uniform upper-bound) | `/home/hugo/codes/mol-metal/metrics/by_metric/vina_score_test_001.json` (sibling) | METRICS_TARGETS §3 m15 (DeltaDock 47.4% → ≥25%); Wf-126 P2 RMSD SE(3) bridge ship ✅ (honest framing: self-RMSD no-pocket, NOT staging) |
| 6 | **m19 LOMO** (5-fold, Rh bottleneck) | **0.5346 ± 0.110** | `/home/hugo/codes/mol-metal/metrics/by_metric/metal_lomo_auc.json` | METRICS_TARGETS §3 m19 (gate 0.70 → ≥0.70, gap -0.166); Wf-125 + Wf-121 in-flight |
| 7 | **L5 Hilbert+ASTSR R²** (40 seeds) | **-6.71** | `/home/hugo/codes/mol-metal/metrics/by_metric/hilbert_4cell_r2.json` | LAMBDA_ROADMAP §4.2 L5 (R20-P2 target ≥0.85 on real pIC50); Wf-127 ship 结论 "12-d RDKit signal floor — 不是 L2/tune"; Wf-122 L2 fix in-flight |
| 8 | **L10 3-way R²** (Wf-79 baseline) | **-0.36** (synthetic) | `/home/hugo/codes/mol-metal/metrics/by_metric/platinai_combined_r2.json` (sibling Wf-79) | LAMBDA_ROADMAP §4.2 L10 (R20-P4 target ≥+0.30); **Wf-128 orphan — 完全无人救** |

**Paper §6 mandatory citations** (party-twenty-spirit §3.5): all 8 above must appear in §6 Limitations + Honest-Negative. m1/m3/m7/m19 are "real-measured below gate"; m14/m15 are "fix attempted, honest negative"; L5/L10 are "structural signal floor — 12-d feature insufficient even with full feature stack".

---

## §4 Mock / stub / contaminated 3 metrics (root cause)

| # | Metric | Verbatim value | Source path | Root cause |
|---:|---|---:|---|---|
| 1 | **vina_status -9.6574 kcal/mol** (stub-floor) | `last_real=-9.6574`; `stub_floor_artifact: true` | `/home/hugo/codes/mol-metal/metrics/by_metric/vina_status.json` | `dock_10pockets_se3sb.py:506-527` `_stub_score()` returns `-8.0 + uniform(-2, 2) - 0.05*dist`; meko 0.8.0 拒绝 Pt/Au/Ru → 8 d-block 全 stub-floor. **POST-AUDIT FIX (Wf-121)**: 8 SMARTS rules ship; cisplatin × 10 pockets MEASURED -2.548 to -3.734 kcal/mol; full 8 d-block sweep deferred 4-6 wk R20 GPU. |
| 2 | **decode_ratio Path A 0/192** (CPU architecture ceiling) | `value=0.0, n_samples=192` (`wf_cfm_path_b_decoder_rework` Path A) | `/home/hugo/codes/mol-metal/metrics/by_metric/decode_ratio.json` `measurements[2]` | 4-fix bundle 修了 diversity attractor, **未修 EGNN velocity architecture** (overvalent atoms → RDKit sanitization fail). Path B bond-bearing 192/192 PASS, RDKit-strict 0.5 on synthetic; real CFM 0/8 architecture ceiling. Wf-124 YuelBond decoder wire in-flight. |
| 3 | **coord_geometry_proxy + 3 MOCK_DELETED aliases** | MOCK_DELETED; silent `None` through `_safe_external_score` | `metrics_v2.py:865-884`; 12 stale JSONs (`phototherapy_activity`, `ptiv_reduction_potential`, `metalloprotein_affinity` × 3 each lambda_mcts/pool/pt_drug) | **Wf-119 cleanup wave 已删 12 stale JSONs** (per TODO-114 P1 + `wf881f9c1d-ca2` audit3 §1 row 8); `metrics_v2.py:865-884` silent None fallback 仍 active (TODO-114 ship 后改 raise AdapterUnavailable). **POST-AUDIT**: 3 MOCK_DELETED files 删除. |

**Total honest-acknowledgement**: paper §4.2 不要列 vina_status -9.6574; §6 footnote 即可。paper §6 必须写 "vina_status -9.6574 = Wf-30 P4 close stub-floor, meko 拒 Pt/Au/Ru → cisplatin × 10 pockets real Vina -2.548 to -3.734 kcal/mol 替换 (Wf-121 P3)"。

---

## §5 P0/P1 critical pending (3 P0 + 2 P1)

| # | Priority | Pending TODO | Block on | Status (audit 2026-09-26 → today 2026-09-27) |
|---:|---|---|---|---|
| 1 | **P0** | `TODO/pending/119_decode_ratio_strengthen.md` | Wf-124 (YuelBond decoder wire) | in-flight (1 sub-agent, 0 result) |
| 2 | **P0** | `TODO/pending/110_meko_metal_atom_extension.md` | Wf-121 → **PARTIAL LIFT ship** (`TODO/completed/121_meko_metal_atom_extension_closeout.md` 2026-09-27); cisplatin × 10 pockets MEASURED; 8 d-block Pt-group SMARTS ship; **full 8 d-block sweep deferred 4-6 wk R20 GPU** |
| 3 | **P0** | `TODO/pending/116_wf74_wetlab_outreach_partial.md` | Wf-74 v2 close-out rewrite (silent overstatement fix — claim 3 LOI + 3 SOP, only 1 doc on disk) | **open as of 2026-09-27** (audit3 §1 row 8 = Wf-119 close-out wave SHIPPED 4 sub-agents, **但 Wf-74 v2 close-out doc 未 ship**); party-twenty-spirit §3.5 violation |
| 4 | **P1** | `TODO/pending/114_mock_deleted_external_scorers_cleanup.md` | `metrics_v2.py:865-884` silent None → raise AdapterUnavailable; 12 stale JSON delete | **PARTIAL** (12 stale JSON 已 ship delete via Wf-119; code-level raise still TODO-114 in pending/) |
| 5 | **P1** | `TODO/pending/118_score_router_hygiene.md` | grep audit + `score_any` clean + `metrics_v2.py` invariant | in-flight (preventive hygiene) |

**Risk**: P0 #3 (Wf-74 v2) 是唯一还未 ship 的 P0 critical,直接违反 party-twenty-spirit §3.5,P0 #1 #2 已 partial-lift。

---

## §6 Wf in-flight 11 status + 1 orphan (Wf-128)

### §6.1 11 in-flight workflows

| # | Wf | Sub-agent count | Status | Ship target metric | Audit3 row |
|---:|---|---:|---|---|---:|
| 1 | Wf-119 close-out wave | 4/4 ✅ | **DONE** | Wf-56 v2 reclass + Wf-74 v2 + 12 stale JSON cleanup + 6 wetlab docs | audit3 §1 row 8 |
| 2 | Wf-120 D-MPNN seed integrity | 3/4 result | P1 retrain + P3 guard ship ✅; P2 5-seed expansion 14/15 ckpts DEFERRED | m1 Rh seed stability (sha256 audit) | row 5 |
| 3 | Wf-121 Meeko + Vina metal + cisplatin | 2/3 result | Meeko SMARTS + adapter ship; **P3 cisplatin × 10 pockets MEASURED ✅** (post-audit ship 2026-09-26) | m3 Vina cisplatin cohort (replaces stub-floor) | row 7 |
| 4 | Wf-122 L2 + ASTSR guard + bench | 2/3 result | L2 patch + ASTR guard ship; P3 post-fix PlatinAI 4-cell re-bench running | L5 Hilbert R² floor lift | row 4 |
| 5 | Wf-123 M1 wasserstein + M2 closure bias | 1/2 result | M1 wasserstein_transfer_loss + M2 closure_oversample **ship** ✅ (`123_master_plan_wasserstein_closure_closeout.md` 2026-09-27); pocket_closure_bias running | m1/m5/m6 strengthen master | row 6 |
| 6 | Wf-124 TODO-119 YuelBond decoder wire | 0/1 result | in-flight | m14 decode_ratio lift | row 13 |
| 7 | Wf-125 Wasserstein + Rh closure oversample | 0/1 result | `wf125_p1_rh_oversample/synthesis.md` PASS (14.92% 6/6 PASS) — **report ship, close-out 未写** | m19 LOMO Rh bottleneck | row 12 |
| 8 | Wf-126 5-pocket staging + SE(3) + SA/Vina | 2/3 result | P1 5/5 READY ✅; P2 RMSD SE(3) bridge ✅; P3 SA + closure-bias + per-pocket Vina running | m5 SA + m15 RMSD + per-pocket Vina | row 11 |
| 9 | Wf-127 PlatinAI L2/ASTR diagnostics | 1/2 result | l2/astr/init sweep ship; L2 guard running | L5 Hilbert R² diagnostic | row 1 |
| 10 | Wf-129 L2 variance MCTS + replay learner | 1/2 result | bench_variance_mcts_wasserstein ship; bench did NOT execute (no torch host) — replay-learner in-flight | L7 Multi-PH + PLL R² | row 2 |
| 11 | Wf-130 Catherine 30 + closure-30×5000 | 1/2 result | 30 metals × 6/6 PASS audit ship ✅ (`130_wf130_p1_catherine_30.md`); closure-30×5000 running | L1 (shipped) + L6 + L12 | row 3 |
| 12 | Wf-131 m1 Ru + scaffold 172-d | 0/2 result | 2 sub-agent in-flight; smoke `wf131_smoke/ru` seed0 AUC=0.688 (**REGRESSION** from 0.7123 — needs ≥5-seed) | m1 Ru AUC + m5/m6 scaffold | row 9, 10 |

### §6.2 1 orphan — Wf-128 (L10 3-way R²)

| Item | Value |
|---|---|
| Failing metric | L10 3-way R² = **-0.36** (Wf-79 baseline) |
| Target | LAMBDA_ROADMAP §4.2 L10 R20-P4 target ≥+0.30 on real pIC50 |
| Disk evidence | **NO `wf128_*` dir** under `molmetal/reports/`; NO Wf-128 reference anywhere in `TODO/` or `journal.jsonl` |
| Verdict | **🔴 完全孤儿** — audit3 §4 row 1 confirmed; needs Wf-132 immediate spawn |

---

## §7 Ship-without-closeout 7 list (audit-snapshot, with post-audit deltas)

Per audit3 §2 verbatim list (audit date 2026-09-26 19:20 UTC):

| Wf | Disk ship status | TODO/completed/ close-out | Post-audit delta (2026-09-27) |
|---|---|---|---|
| Wf-121 | ✅ 3/3 done | ❌ 无 | **✅ SHIP 2026-09-27** (`TODO/completed/121_meko_metal_atom_extension_closeout.md`) — P0 partial lift, cisplatin × 10 pockets MEASURED |
| Wf-122 | ✅ 3/3 done (l2_verification, guard_verification, a2780/a549/mcf7 post-fix) | ❌ 无 | still ❌ (close-out not yet written; L2 fix + ASTSR guard on disk) |
| Wf-123 | ✅ 2/2 done (wasserstein ship) | ❌ 无 | **✅ SHIP 2026-09-27** (`TODO/completed/123_master_plan_wasserstein_closure_closeout.md`) — M1+M2 wasserstein + closure-bias master |
| Wf-125 | ✅ 1/1 (synthesis.md + synthetic_rh_cohort.json + wasserstein_decay.json) | ❌ 无 | still ❌ |
| Wf-126 | ⚙️ 2/3 done (P1+P2 ship; P3 SA + per-pocket Vina still running) | ❌ 无 | still ❌ (P3 closure pending) |
| Wf-127 | ⚙️ 1/2 done (l2/astr/init sweep + synthesis ship; L2 guard running) | ❌ 无 | still ❌ |
| Wf-129 | ⚙️ 1/2 done (`synthesis.md` + `per_seed_metrics.json` + `seeds/` ship; replay-learner running) | ❌ 无 | still ❌ |

**Net ship-without-closeout as of 2026-09-27**: **5** (Wf-122, Wf-125, Wf-126, Wf-127, Wf-129) — reduced from audit3's 7 after Wf-121 + Wf-123 closed out 2026-09-27.

**Bonus (audit3 didn't list)**: Wf-131 (smoke + smoke5 + p1_ru_auc_multitask on disk, **0 result** still, but synthesis.md + ru/ written) — qualifies as ship-without-closeout for the smoke cohort only. Wf-124 (`yuelbond_decoder.py` not yet shipped) and Wf-130 P2 (closure-30×5000 in-flight) do NOT qualify as "shipped without close-out" — they are still in-flight.

---

## §8 Health ratio 47% / 53% + verdict

### §8.1 Health ratio (verbatim from A1 §7.1 + A2 §1)

| Category | Count | Ratio | Verdict |
|---|---:|---:|---|
| **Real strong + lead-claim** (A1 §2 + A2 ✅ + A3 ship-cohort) | 6 lead-claim + ~9 paper-grade families + 4 post-audit ship (Wf-121/123 + Wf-130 + Wf-120) ≈ 14-16 metrics | **~47%** | n_distinct 20, DrugOOD 3-split, NCI-60 9-cell, SA 3.099, rdkit_validity 1.0, hERG 0.255, Ru 0.7123 (1-seed) — paper §4 anchor |
| **Mock / partial / honest-negative** (A1 §3+§6 + A2 🟡+🔴+⚪) | 8 honest-negative + 3 contaminated + 11+ partial ≈ 22-25 metrics | **~53%** | decode 0/8, L5 -6.71, L10 -0.36, m3 -1.355 (single-pocket), m7 diversity 0.137, m15 RMSD 1.241, vina_status -9.6574 (replaced for cisplatin), coord_geometry MOCK_DELETED |

**Audit verdict (verbatim from A1 §7.1)**: "47% real + 53% mock/partial,对 paper 准备来说属于**'诚实但不够'**:real 部分已可放 §4,但 paper §6 必须写 8+ honest-negative"。

### §8.2 Post-audit uplift (2026-09-26 → 2026-09-27)

| Item | Before audit | After audit | Δ |
|---|---|---|---|
| **vina_status -9.6574** stub-floor | active (stub_floor_artifact) | **cisplatin × 10-pocket real Vina** -2.548 to -3.734 MEASURED ✅ | stub-floor partially lifted for cisplatin cohort |
| **Wf-119 cleanup** | 12 stale MOCK_DELETED JSONs active | 12 stale JSONs deleted ✅ (Wf-119 ship) | MOCK_DELETED 8 files → 0 (coord_geometry_proxy + 3 aliases each) |
| **Wf-121 close-out** | no close-out (TODO/pending/110) | **P0 partial lift** (`121_meko_metal_atom_extension_closeout.md`) | P0 #2 partial ship |
| **Wf-123 close-out** | no close-out | M1+M2 wasserstein + closure-bias master ship ✅ | innovation roadmap lifted |
| **Health ratio** | 47% / 53% | **~49% / 51%** (rough delta +2%) | honest improvement, NOT a substantive paper-grade lift |

### §8.3 Final verdict (party-twenty-spirit §3.5 compliant)

1. **Real data quality**: 6 lead-claim 真实强数据 (n_distinct 20, DrugOOD 3-split 0.891/0.697/0.748, SA 3.099, rdkit_validity 1.0, NCI-60 9-cell 0.7742, hERG 0.255) — paper §4 anchor sufficient.
2. **Honest-negative mandatory**: 8 honest-negative (m1/m3/m7/m14/m15/m19/L5/L10) + 3 contaminated (vina_status stub-floor / decode_ratio architecture / coord_geometry MOCK_DELETED) — paper §6 must cite verbatim.
3. **P0 critical**: 3 P0 critical pending — TODO-119 (Wf-124 in-flight), TODO-110 (Wf-121 P0 partial-lift ship, full 8 d-block sweep deferred), TODO-116 (Wf-74 v2 close-out — **唯一 P0 未 ship**).
4. **Wf-128 完全孤儿** — L10 3-way R² 无人救; Wf-132 must spawn this first per audit3 §5 row 1 + this §8.
5. **Wet-lab 完全空** — 0 LOI sent, 0 IC50, paper §5 future work 必写。
6. **Overall**: **47% real + 53% honest-negative** is paper-acceptable under party-twenty-spirit §3.5 (honesty first) but **not yet Tier-1 journal headline** — needs R20 GPU retrain (5-seed m1, full 8 d-block Vina, L5/L10 architectural lift) for substantive lift to ≥65% real.

---

## §9 Source paths (absolute, for §17 INDEX.md cross-link)

### §9.1 Metric JSON sources (verbatim §2/§3/§4)
- `/home/hugo/codes/mol-metal/metrics/by_metric/n_distinct.json`
- `/home/hugo/codes/mol-metal/metrics/by_metric/drugood_ood_auc.json`
- `/home/hugo/codes/mol-metal/metrics/by_metric/sa_mean.json`
- `/home/hugo/codes/mol-metal/metrics/by_metric/rdkit_validity.json`
- `/home/hugo/codes/mol-metal/metrics/by_metric/nci60_precious_9cell_auc.json`
- `/home/hugo/codes/mol-metal/metrics/by_metric/herg_cardio_risk.json`
- `/home/hugo/codes/mol-metal/metrics/by_metric/metalcytotox_ru_auc.json`
- `/home/hugo/codes/mol-metal/metrics/by_metric/vina_score.json`
- `/home/hugo/codes/mol-metal/metrics/by_metric/vina_kcal_per_mol.json` (post-audit: cisplatin × 10-pocket MEASURED)
- `/home/hugo/codes/mol-metal/metrics/by_metric/vina_status.json` (post-audit: stub_floor partially lifted for cisplatin)
- `/home/hugo/codes/mol-metal/metrics/by_metric/decode_ratio.json`
- `/home/hugo/codes/mol-metal/metrics/by_metric/diversity_tanimoto.json`
- `/home/hugo/codes/mol-metal/metrics/by_metric/metal_lomo_auc.json`
- `/home/hugo/codes/mol-metal/metrics/by_metric/hilbert_4cell_r2.json`
- `/home/hugo/codes/mol-metal/metrics/by_metric/platinai_combined_r2.json`
- `/home/hugo/codes/mol-metal/metrics/by_metric/dmpnn_ckpt_status.json`

### §9.2 Plan + target sources (cross-link anchors)
- `/home/hugo/codes/mol-metal/TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L1-L12 trajectory, R20-R25)
- `/home/hugo/codes/mol-metal/TODO/02_metrics/METRICS_TARGETS.md` §3 (paper §4 template, 21 metrics)
- `/home/hugo/codes/mol-metal/TODO/INDEX.md` §17 (target slot — to be appended by Wf-132 Phase 2)

### §9.3 In-flight workflow sources (§6)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf119_cleanup/`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf120_p1_rh_retrain/CLOSE_OUT.md`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf121_cisplatin_real_vina/synthesis.md`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf121_p1_atom_types/`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf121_p2_adapter/`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf122_p{1,2,3}_*/`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf123_p1_wasserstein/`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf124_p1_yuelbond/` (in-flight)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf125_p1_rh_oversample/synthesis.md`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf126_p{1,2,3}_*/`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf127_p{1,2}_*/`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf129_p1_variance_wasserstein/`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf130_p1_catherine_30/`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf131_{smoke,smoke5,smoke10,p1_ru_auc_multitask}/`

### §9.4 Close-out files (post-audit ship 2026-09-27)
- `/home/hugo/codes/mol-metal/TODO/completed/120_dmpnn_seed_integrity.md`
- `/home/hugo/codes/mol-metal/TODO/completed/121_meko_metal_atom_extension_closeout.md`
- `/home/hugo/codes/mol-metal/TODO/completed/123_master_plan_wasserstein_closure_closeout.md`
- `/home/hugo/codes/mol-metal/TODO/completed/130_wf130_p1_catherine_30.md`

---

## §10 Audit disclaimer (party-twenty-spirit §3.5)

- **No fabrication**: every number in §2/§3/§4 is verbatim from the 3 audit reports or directly from the metric JSON path cited.
- **Audit-snapshot**: audit was generated 2026-09-26 19:20 UTC; post-audit deltas (Wf-121 P3 ship + Wf-119 cleanup + Wf-121/Wf-123 close-out) explicitly noted in §6 and §7.
- **Cross-link integrity**: LAMBDA_ROADMAP §4.2 L1-L12 trajectory referenced in §3 (L5/L10); METRICS_TARGETS §3 referenced in §2/§3 with paper §4 anchor.
- **Honest-negative mandatory**: 8 honest-negative (§3) + 3 contaminated (§4) must appear in paper §6 verbatim.
- **P0 unresolved**: TODO-116 (Wf-74 v2 close-out) is the single P0 not yet shipped — direct party-twenty-spirit §3.5 violation pending fix.
- **Wf-128 orphan**: L10 3-way R²无人救 — must be addressed by Wf-132 spawn (per audit3 §5 row 1 priority).

---

**Word count**: ~2,800 (300-line cap respected; this is dense by design).
**Total lines**: 230.
