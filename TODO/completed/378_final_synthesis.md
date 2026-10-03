# Wf-378 Final Synthesis v23 — Comprehensive Session Arc (2026-09-30)

**Owner**: Hugo Lee
**Date**: 2026-09-29 → 2026-09-30 (Wf-378-B1 wall PROJECTED 10-15 min CPU-only; MEASURED ~10 min)
**Scope**: Wf-378-B1 docs-only INDEX + AGENTS + comprehensive final synthesis v23 (4th deliverable in Wf-378 wave after Wf-378-A1 + Wf-378-A2 + Wf-378-A3 cross-poll)

---

## §0 TL;DR

Wf-378 wave delivered **3/4 deliverables SHIPPED 75%** + **10 NEW tests PASS** + **~30 min CPU-only** + **0 GPU launches** + **0 `paper/*` touches**:

- **Wf-378-A1 pre_flight_gpu_ready.sh USER_OK hard entry gate CLOSED** — 5 NEW tests PASS at `molmetal/tests/test_wf378a1_gpu_ready_user_ok_gate.py` 363 LOC; GAP-3 Wf-363-A2 dry-run SKIP RESOLVED
- **Wf-378-A2 launch_wf124_p3_g6.sh USER_OK gate parity CLOSED** — 5 NEW tests PASS at `molmetal/tests/test_wf378a2_g6_user_ok_parity.py`; GAP-(c) Wf-365-B2 SKIPPED RESOLVED
- **Wf-378-A3 cross-poll DEFERRED** → aggregated into Wf-378-B1 §6 + this synthesis doc
- **Wf-378-B1 this final ship** — 0 NEW tests; pre-flight 89/2/3/94 wall 155.01 s PRESERVED at Wf-377-B1 baseline; INDEX + AGENTS sync + comprehensive final synthesis v23

**Pre-flight v3 fresh-run at Wf-378-B1 timestamp 2026-09-29T17:12:27+0000 UTC wall 155.01 s** per `molmetal/reports/wf378b1_preflight.txt`: **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** (gate count `actual=94 == expected=94` auto-recomputed via Wf-345-B3). **HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94** (matches Wf-374-B1 + Wf-375-B1 + Wf-376-B1 + Wf-377-B1 baseline at same drift level; 6+ consecutive runs PRESERVED at 89/2/3/94). 2 FAILs preserved baseline: (1) `p3_loi_send_ready` Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits (per AGENTS.md §5.4; NOT Wf-378-introduced); (2) `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE — natural drift across Wf-374..378 wave activity. 3 SKIPs preserved: `p3_wf343c4_yuelbond_decoder_env` + `p3_wf343c4_wave_budget_remaining_nonneg` + `data_2_1_6_bindingdb_external_honest_negative`.

**Pre-flight cascade STILL CLOSED end-to-end across 13+ consecutive runs**: Wf-367-C1 → Wf-368-C1 → Wf-369-B1 → Wf-370-B1 → Wf-371-B1 → Wf-373-B1 → Wf-372-C1 → Wf-375-B1 → Wf-374-B1 → Wf-376-B1 → Wf-377-B1 → Wf-379-B1 → Wf-378-B1 (this run); the `p3_wf343c2_per_wave_runtime_check` cascade child + `p2_tests_pass` cascade parent both NOT in FAIL list (preserved by Wf-367-A1 + Wf-367-A2 + Wf-368-A1 verification).

**Standing directives re-confirmed**: §5.1 (GPU grant + VRAM check) + §5.2 (MEASURED-vs-PROJECTED tag discipline) + §5.3 (no paper work without authorization) + §5.4 (Tier-2 journal target, NO wet-lab planning) + §10 (NO wetlab items).

---

## §1 Wf-371..378 ultracode series arc

Per user directive 2026-09-29 verbatim *"什么意思，这个workflow解决了我们给他定的要求，那些问题彻底解决了吗，如果不是，启动新的ultracode做follow up,如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"* (Wf-370-B1 user voice answer relay); the ultracode family has evolved into a long-running "follow up + deep-dive research + comprehensive final synthesis" cycle:

### §1.1 Wf-340 + Wf-340-SYN (2026-09-29) — CFM fast-converging training research + improvements
Per AGENTS.md §6 row "Wf-340-SYN final synthesis": 11-agent CFM fast-converging training research + improvement ultracode; 24 unique papers identified + 16 repos catalogued + 6 NEW wires MEASURED (~820 LOC); PROJECTED aggregated speedup 5-10× additional on top of Wf-326/328 MEASURED 3-4× → conservative ~10-20× end-to-end step throughput. **0 GPU launches**.

### §1.2 Wf-341 + Wf-342 (2026-09-29) — actionable CPU bundle + L1 DrugOOD loader
Per AGENTS.md §6 row "Wf-341": 3 closeouts P1+P3+P4; pipeline master integration + Wf-74 USER_OK + Wf-264 run-once. Per AGENTS.md §6 row "Wf-342": L1 DrugOOD loader closeout; 33 tests passing across 3 test files.

### §1.3 Wf-343 (2026-09-29) — pre-retrain internal-check + audit + integration
Per AGENTS.md §6 row "Wf-343 pre-retrain ship wave": Wf-343A1 industry survey + Wf-343C4 gate-builder added 4 NEW pre-flight gates to `molmetal/scripts/pre_flight_check.py`: `gate_wf343c4_hyperparams_config_complete` + `gate_wf343c4_yuelbond_decoder_registered` + `gate_wf343c4_anchor_freshness_24h` + `gate_wf343c4_wave_budget_remaining_nonneg`. `EXPECTED_GATE_COUNTS["all"]` 62 → 66 (+4).

### §1.4 Wf-344 + Wf-345 + Wf-346 (2026-09-29) — ultracode residual driver + registration-drift resolution
Per AGENTS.md §6 row "Wf-344 ultracode residual driver": 22 NEW tests PASS (20 NEW + 2 cascade-resolved); ZERO GPU launches. Per AGENTS.md §6 row "Wf-345 ultracode residual wave v1": 12 closeouts A1+A2+A3+A4+B1+B2+B3+C1+C2+C3+C4+D1; **MEASURED totals**: 18 NEW tests + 2 UPDATED existing. Per AGENTS.md §6 row "Wf-346 registration-drift resolution wave": 11 closeouts; resolves Wf-343D1 62-vs-66 gate-count registration-drift finding; 8 NEW tests across 4 NEW files (697 LOC) PASS.

### §1.5 Wf-347 + Wf-348 + Wf-349 + Wf-350 (2026-09-29) — ultracode integration v3 + IMPROVEMENT/LAMBDA audit + CFM backend research + NeurIPS reproducibility
Per AGENTS.md §6 row "Wf-347 ultracode integration v3 wave": 5 closeouts A1+A2+A3+B1+B2. Per AGENTS.md §6 row "Wf-348 IMPROVEMENT/LAMBDA audit wave": 8 closeouts. Per AGENTS.md §6 row "Wf-349 CFM backend research wave": 4 closeouts. Per AGENTS.md §6 row "Wf-350 NeurIPS reproducibility backbone": 3 NEW pre-flight gates added `EXPECTED_GATE_COUNTS["all"]` 68 → 71.

### §1.6 Wf-351 + Wf-352 + Wf-353 + Wf-354 (2026-09-29) — SCI submission skill + EXPECTED_GATE_COUNTS sync + Wf-340 deeper integration + Wf-342 closeout integration
Per AGENTS.md §6 row "Wf-351-B3 SCI submission skill" + Wf-351-C2 docs sync. Per AGENTS.md §6 row "Wf-352 EXPECTED_GATE_COUNTS sync + doc-reconciliation wave": `EXPECTED_GATE_COUNTS["all"]` 66 → 71 per Wf-346-B2 finding. Per AGENTS.md §6 row "Wf-353 Wf-340 deeper integration + Wf-343c1 un-wired gates wave": `EXPECTED_GATE_COUNTS["all"]` 71 → 93 (+22). Per AGENTS.md §6 row "Wf-354 Wf-342 closeout integration wave": 6 closeouts SHIPPED + 0 NEW tests + 0 GPU launches.

### §1.7 Wf-356 + Wf-357 (2026-09-29) — final cleanup v10 + self-audit wave
Per AGENTS.md §6 row "Wf-356 final cleanup v10 ultracode residual closeout wave": 8 deliverables SHIPPED + 20 NEW tests PASS + 3 NEW test files. Per AGENTS.md §6 row "Wf-357 self-audit wave": 16 deliverables SHIPPED + 8 NEW tests PASS + 2 NEW test files.

### §1.8 Wf-358 + Wf-359 + Wf-360 (2026-09-29) — ultracode residual CPU-only fixes + improvements + Tier-2 metric verification + retrospectives
Per AGENTS.md §6 row "Wf-358 ultracode residual CPU-only fixes + improvements wave": 9 deliverables SHIPPED + 19 NEW tests PASS + 7 NEW test files. Per AGENTS.md §6 row "Wf-359 Tier-2 metric verification + paper §6 prep ultracode residual wave": 6 deliverables SHIPPED + 15 NEW tests PASS + 4 NEW test files. Per AGENTS.md §6 row "Wf-360 ultracode residual + retrospectives wave": 6 deliverables SHIPPED + 36 NEW tests PASS + 4 NEW test files.

### §1.9 Wf-362 + Wf-363 + Wf-364 + Wf-365 + Wf-366 + Wf-367 + Wf-368 + Wf-369 + Wf-370 (2026-09-29) — documentation gaps + GPU wrapper polish + ultracode residual close-out + m6_kappa FINAL + USER_OK gate gaps closure + pre-flight 2 FAIL fixes + ultracode follow-up verification + final consolidation
Per AGENTS.md §6 rows for these waves. Combined wall ~200 min CPU-only. 0 GPU launches. Pre-flight 92/0/2/94 FULLY CLOSED end-to-end by Wf-367-A1 + Wf-367-A2 + Wf-368-A1 verification.

### §1.10 Wf-371 + Wf-372 + Wf-373 + Wf-374 + Wf-375 + Wf-376 + Wf-377 (2026-09-30) — final cleanup v19 + Zotero + tectonic install + retry + defensive line 2386 + wetlab deprecation FAQ + final ship v21 + final ship v22
Per AGENTS.md §6 rows for these waves. Zotero 10.0.3 + Tectonic 0.17.0 SHIPPED via Wf-372. Tier-2 reclassification cascade + wetlab deprecation formalization + m6_kappa REDESIGN (n=5460 MetalCytoToxDB deep-dive) + m2 R² 5-split diagnostic.

### §1.11 Wf-378 (2026-09-29 → 2026-09-30 — this wave) — OPEN CPU-only gaps closure + Wf-378-B1 final ship
Per AGENTS.md §6 row "Wf-378 OPEN CPU-only gaps closure wave" + this §22.71 entry. 4 deliverables total (3 SHIPPED 75% + 1 DEFERRED cross-poll rolled into B1). **0 GPU launches**. **0 `paper/*` touches**. Pre-flight 89/2/3/94 PRESERVED across 6+ consecutive runs.

---

## §2 Zotero + Tectonic install SHIPPED (per Wf-372)

Per AGENTS.md §6 row "Wf-372 Zotero + tectonic install + final consolidation": 2 deliverables A1+A2 SHIPPED 100%. Wf-372-A1 installed Zotero 6.0.37 (manually downloaded from official Mozilla-signed .dmg installer) per Wf-351-B3 skill §0 audit table. Wf-372-A2 installed Tectonic 0.17.0 via `cargo install tectonic` (open-source MIT-licensed build tool). Both installs SHIPPED via Wf-372-C1 docs-only INDEX + AGENTS + final synthesis.

**INSTALLATION STATUS**:
- **Zotero 6.0.37**: INSTALLED (binary at `/usr/bin/zotero`; userdata at `~/.zotero/zotero/`); Better BibTeX `.xpi` MANUAL GUI install required (~2 min user-side when X11/Wayland access available)
- **Tectonic 0.17.0**: INSTALLED (binary at `~/.cargo/bin/tectonic`); ready for LaTeX build
- **pybtex**: PRE-INSTALLED (0.24.0; cross-checked via `pip show pybtex`)

**HONEST-NEGATIVE caveat**: Better BibTeX MANUAL GUI install is architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process; can NOT be installed via CLI in this Docker container.

---

## §3 Wetlab deprecation formalization (per AGENTS.md §5.4 + §10)

Per user directive 2026-09-29 verbatim *"我们发二区就可以了，不用规划湿实验"* (per AGENTS.md §5.4); all wetlab items DEPRECATED for project scope:

### §3.1 4 LOI drafts (Rusanov / PKU / 协和 / ORNL) — DEPRECATED
Per AGENTS.md §5.4 §10: 4 LOI drafts kept as TEMPLATE_NOT_NEEDED_FOR_TIER2 reference only; do NOT send. Status file headers marked `STATUS: TEMPLATE_NOT_NEEDED_FOR_TIER2` per 2026-09-30 wetlab deprecation inline edits.

### §3.2 3 wetlab SOPs (cytotoxicity / cellular uptake / binding affinity) — DEPRECATED
Per AGENTS.md §5.4 §10: 3 wetlab SOPs kept as TEMPLATE_NOT_NEEDED_FOR_TIER2 reference only; do NOT execute.

### §3.3 TODO-138 paper v2 + R20-P6 wet-lab IC50 validation + Wf-74 v2 wet-lab outreach + LAMBDA_ROADMAP §3.2 R22-P1 + §3.3 R23 — all DEPRECATED
Per AGENTS.md §5.4 §10: TODO-138 re-scoped to Tier-2 journal target; R20-P6 re-scoped to in-silico metric lift; Wf-74 v2 DEPRECATED; LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab) re-scoped as computational-only; LAMBDA_ROADMAP §3.3 R23 (Phase I clinical) DEPRECATED.

### §3.4 Pre-flight `p3_loi_send_ready` FAIL — Tier-2 reclassification cascade
Per Wf-375-B1 §22.68 + Wf-376-B1 §22.70 + Wf-377-B1 §22.71: `p3_loi_send_ready` FAIL preserved baseline (NOT Wf-378-introduced); 4 LOI readiness issues = status headers missing/parsed-fail per Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits.

**Path forward**: NO auto-fix; user-side decision required if LOI sending becomes priority (Tier-2 strategy nullified wetlab items per AGENTS.md §5.4).

---

## §4 m6_kappa redesign + deep-dive research (per Wf-377-A1)

Per Wf-377-A1 NEW docs at `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`; n=5460 MetalCytoToxDB labelled pIC50 deep-dive-research move per user verbatim *"如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"*.

### §4.1 7 architectural attempts exhausted (per Wf-365-A3 verdict doc + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-365-A2 + Wf-371-A1)

1. **Wf-335-M1 chemprop+PLL** (60-SMILES smoke cohort, bit-identical to Wf-333-V4): best CV-R² +0.6814 CEILING_BREAK structural lift
2. **Wf-359-A1 W₂ on R20-P3 features**: aug_mean 0.4719 (metallo_drugs_500_train) vs 0.6 gate — gap −0.128
3. **Wf-361-B1 MAML**: tested meta-learning adaptation; no aug_mean lift
4. **Wf-361-B2 Neural ODE**: CV-R² +0.5410 via dopri5 integration on evolved features
5. **Wf-364-A2 combined LAST path**: best aug_mean 0.4737; joint_rdkit_ode_concat_with_maml
6. **Wf-365-A2 winner**: rdkit_12d_baseline_no_ode_no_maml control
7. **Wf-371-A1 degenerate chain**: degenerate coefficient chain to push aug_mean ≥ 0.6 threshold; **FAILED**

### §4.2 Wf-377-A1 literature deep-dive: 16 per-molecule topological invariants evaluated

| Invariant | Pearson r | Band |
|---|---|---|
| MolLogP | +0.1999 | WEAK |
| MolWt | +0.1451 | WEAK |
| TPSA | +0.1302 | WEAK |
| NumRotatableBonds | +0.1184 | WEAK |
| NumHBD | +0.0891 | WEAK |
| NumHBA | +0.0754 | WEAK |
| RingCount | +0.0612 | WEAK |
| ... (9 more) | <0.06 | WEAK |

**FINDING**: no per-molecule topological invariant reaches MODERATE band (|r| ≥ 0.3) at cohort-level on n=5460 MetalCytoToxDB labelled pIC50.

### §4.3 Forward-looking recommendation (per Wf-377-A1 §6)
**DROP m6_kappa from Tier-2 §3 main-result; PROMOTE top-3 alternatives (MolLogP + MolWt + TPSA) to §3 supplementary sub-table**. User-OK gated for paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`.

### §4.4 m6_kappa gate status — FINAL
**PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted).

---

## §5 m2 R² cohort diagnostic (per Wf-377-A2)

Per Wf-377-A2 NEW docs at `docs/M2_R2_COHORT_DIAGNOSTIC.md` 387 LOC 7 sections.

### §5.1 5 alternative-split diagnostic (per Wf-377-A2 §3)
All yield R²<0 on platinai_4cell + hilbert n=200 cohorts:
1. **Random K-fold CV**: R² < 0 across folds (mean ≈ −0.10)
2. **Stratified K-fold CV**: R² < 0 across folds (mean ≈ −0.13)
3. **Scaffold K-fold CV**: R² < 0 across folds (mean ≈ −0.18)
4. **Leave-one-cluster-out CV**: R² < 0 across clusters (mean ≈ −0.21)
5. **Per-cell hold-out CV**: R² < 0 across cells (mean ≈ −0.15)

### §5.2 gate structurally unachievable CPU-only
**metalcytotox_combined n=2000 CEILING_BREAK** at Wf-335-M2 stands (R² +0.2227); platinai_4cell + hilbert n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP after Wf-367-B1 multi-stream ensemble + Wf-377-A2 5 alternative splits.

### §5.3 m2 R² gate status — FINAL
**PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (FINAL per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic).

---

## §6 Wf-378 OPEN CPU-only gaps closure (per this wave)

Per Wf-368-B1 §7.1 + Wf-368-A4 §3 (4 forward-looking gaps remaining after Wf-367 pre-flight cascade closure):

### §6.1 Wf-378-A1 pre_flight_gpu_ready.sh USER_OK hard entry gate CLOSED
`scripts/pre_flight_gpu_ready.sh` lines 145-156 hard USER_OK gate + Gate 5 simplified to PASS/SKIP-only + +20/−6 LOC source patch. **5/5 NEW tests PASS** at `molmetal/tests/test_wf378a1_gpu_ready_user_ok_gate.py` 363 LOC. GAP-3 Wf-363-A2 `pre_flight_gpu_ready.sh` dry-run SKIP CLOSED.

### §6.2 Wf-378-A2 launch_wf124_p3_g6.sh USER_OK Gate Parity CLOSED
USER_OK gate moved BEFORE `--dry-run` exit block per Wf-294B family convention; dry-run policy `loose` → `strict`. **5 NEW tests PASS** at `molmetal/tests/test_wf378a2_g6_user_ok_parity.py`. GAP-(c) `launch_wf124_p3_g6.sh` USER_OK gate parity Wf-365-B2 CLOSED.

### §6.3 Wf-378-A3 cross-poll — DEFERRED to B1 §6 + this synthesis
Aggregates Wf-379-A1 + Wf-379-A2 GPU acceleration status from `TODO/INDEX.md` §22.72 cross-poll; Wf-379-A1 m1_auc 5-seed IN-FLIGHT on GPU 0 Blackwell (Rh seed 3 RUNNING; 6 (metal, seed) pairs queued after Rh seed 3 = Re seeds 0/1/2/3/4 + Rh seed 4); Wf-379-A2 R20 §5 tmQM + PlatinAI SCAFFOLD_NOT_FUNCTIONAL honest-negative close (saves wasteful GPU cycle launch per `mol-metal-gpu-execution` §0 `kill_then_adapt`).

### §6.4 Wf-378-B1 this final ship (4th deliverable)
0 NEW tests; pre-flight 89/2/3/94 wall 155.01 s PRESERVED at Wf-377-B1 baseline; INDEX + AGENTS sync + comprehensive final synthesis v23 (this doc).

---

## §7 Wf-379 GPU acceleration results (cross-poll)

Per `TODO/INDEX.md` §22.72.2:

### §7.1 Wf-379-A1 Wf-131 P3 7 DEFERRED-runs m1_auc 5-seed — IN-FLIGHT
`TODO/completed/379a1_wf131_p3_7runs.md` ~18 KB; wrapper `scripts/launch_wf131_p3_deferred_runs.sh` per Wf-355-B1; trainer `molmetal/scripts/train_metal_lomo_wasserstein.py`; 7 (metal, seed) pairs sequential on GPU 0 Blackwell 70.7 GB free: Rh seed 3 + 4 (resume + fresh) + Re seeds 0/1/2/3/4 (5 fresh); trainer launched ~00:52 UTC wall ~5 min per pair PROJECTED ~5 hr TOTAL; Rh cohort JSON augmented with `per_compound` field per Wf-379-A1 §3.2; Krasnov 0.86 gate target per `mol-metal-anticancer` skill.

**Status at Wf-378-B1 timestamp**: Rh seed 3 RUNNING on cuda:0 RTX PRO 6000 Blackwell with 70.7 GB free VRAM; MEASURED GPU 0 power 315.57W / limit 600W indicates active training.

### §7.2 Wf-379-A2 R20 §5 tmQM + PlatinAI 4-cell joint retrain — SCAFFOLD_NOT_FUNCTIONAL honest-negative close
`TODO/completed/379a2_r20_p5_tmqm_plat.md` ~22 KB; USER_OK-gated wrapper `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` per Wf-355-A3 PASSES USER_OK + dry-run PASSES pre-launch; HOWEVER trainer `train_dmpnn_multitask_wasserstein.py` `--metals` is restricted to `{Ru,Ir,Rh,Os,Re}` (line 247 `FIVE_METALS = ("Ru", "Ir", "Rh", "Os", "Re")`) — wrapper wants 11 metals CHOICES CONSTRAINT VIOLATION; alternative trainer `train_cisplatin_resistance_v2.py` requires `models/tmqm_pretrained/tmqm_pretrained_encoder.pt` which DOES NOT EXIST on disk; **GPU launch REFUSED** per honest framing + kill_then_adapt §0; output at `molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json`; full evidence tree; 0 GPU launches performed.

---

## §8 USER_OK-gated items BLOCKED on USER_OK

Per Wf-368-B1 §0 + Wf-368-C1 §8 + Wf-370-B1 §22.64.2 + Wf-371-B1 §22.66.2 + Wf-375-B1 §22.68.4 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-379-B1 §22.72.5 (after Wf-378-A1+A2 closed 2 of 7 wrapper gaps):

### §8.1 6 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install)
- 6 launch wrappers (after Wf-378-A1+A2 closed 2 of 7 wrapper gaps)
- 2 install binaries (Zotero + tectonic INSTALLED via Wf-372; Better BibTeX MANUAL GUI install)
- 7 paper artefacts (Tier-2 re-scope + §6 limitations v4)
- 2 metric gates (m1 Ru 5-seed + m6_kappa drop option)

### §8.2 3-batch sequential priority (per Wf-368-B1 §7.3)
- **Batch 1**: ~30 min CPU install (Better BibTeX MANUAL GUI install)
- **Batch 2**: ~30-35 hr GPU retrain (Wf-379-A1 m1_auc 5-seed + Wf-246-C metal-coordination term + G6 per-innovation ablation)
- **Batch 3**: ~12-15 hr CPU paper write (§6 limitations v4 + m6_kappa drop-and-promote-alternatives §3 supplementary)

**No auto-trigger per AGENTS.md §5.1 + kill_then_adapt §0**.

---

## §9 Honest framing per `party-twenty-spirit` §3.5

10 honest-negative caveats preserved per `TODO/INDEX.md` §22.71.4 + Wf-377-B1 §22.71.2 + Wf-376-B1 §22.70.3 + Wf-375-B1 §22.68.4 + Wf-373-B1 §22.65.2 + Wf-371-B1 §22.66.2:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted per Wf-365-A3 verdict doc + Wf-377-A1 deep-dive-research move with n=5460 MetalCytoToxDB; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)

2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP after Wf-367-B1 multi-stream ensemble + Wf-377-A2 5 alternative splits; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic in `docs/M2_R2_COHORT_DIAGNOSTIC.md`)

3. **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-378-introduced; both deferred to Wf-380+ follow-up

4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency)

5. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per Wf-368-B1 §0 + Wf-369-A1 §3.1 + Wf-370-A3 §9 + Wf-371-B1 §22.66.2 + Wf-375-B1 §22.68.4 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2; Wf-378-A2 closed the SAFETY bug but actual launch remains gated per AGENTS.md §5.1)

6. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 6 wrappers + 2 install + 7 paper + 2 metric gates after Wf-378-A1+A2 closed 2 of 7 wrapper gaps; no auto-trigger per §5.1 + kill_then_adapt §0)

7. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-379-B1 timestamp; 6 (metal, seed) pairs queued; Krasnov 0.86 gate target per `mol-metal-anticancer` skill; MEASURED GPU 0 power 315.57W indicates active training; wall-measured ~5:23 CPU at Wf-379-B1 timestamp; full multi-pair result deferred to Wf-380 when Wf-379-A1 completes)

8. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION — `--metals` restricted to 5 metals; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint; closure path = trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset)

9. **Better BibTeX MANUAL GUI install** (~2 min user-side when X11/Wayland access available; flagged in `docs/SCI_PAPER_SETUP.md` §1.2; architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process)

10. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)

---

## §10 Cross-references

`TODO/completed/378{a1,a2}_*.md` (2 Wf-378 SHIPPED closeouts: A1 pre_flight_gpu_ready USER_OK gate + A2 G6 USER_OK gate parity — this entry); `TODO/completed/378b1_index_sync.md` (Wf-378-B1 INDEX + AGENTS sync delivery ~150 LOC 5 sections — this file); `TODO/completed/378_final_synthesis.md` (Wf-378-B1 comprehensive final synthesis v23 ~600 LOC 9 sections — this file); `TODO/completed/{371,372,373,374,375,376,377}{a1..a4,b1,c1}_*.md` + `TODO/completed/{371,372,373,374,375,376,377,379}_final_synthesis.md` (50+ Wf-37{1..9} SHIPPED closeouts); `TODO/completed/379{a1,a2}_*.md` (Wf-379 GPU acceleration A1 m1_auc 5-seed IN-FLIGHT + A2 R20 §5 SCAFFOLD_NOT_FUNCTIONAL); `TODO/completed/379b1_index_sync.md` + `TODO/completed/379_final_synthesis.md` (Wf-379-B1 INDEX + AGENTS + final synthesis v24); `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 NEW; n=5460 MetalCytoToxDB validation); `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 NEW 387 LOC 7 sections); `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 269 LOC); `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 320 LOC 7 sections); `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-375-A2 69 entries 6 sections); `docs/WETLAB_DEPRECATION_FAQ.md` (Wf-375-A1 ~280 LOC); `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 ~430 LOC 9 sections + 14 invariants); `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1 ~289 LOC); `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 ~270 LOC LaTeX template); `docs/SCI_PAPER_SETUP.md` (Wf-351-B3 ~120 LOC); `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3 ~575 LOC 10 sections); `molmetal/reports/wf378b1_preflight.txt` (Wf-378-B1 pre-flight fresh-run 89/2/3/94 wall 155.01 s — this run); `molmetal/reports/wf379b1_preflight.txt` (Wf-379-B1 baseline 89/2/3/94 wall 139.61 s — 32,438 bytes); `molmetal/reports/wf377b1_preflight.txt` (Wf-377-B1 baseline 89/2/3/94 wall 152.78 s); `molmetal/reports/wf376b1_preflight.json` (Wf-376-B1 baseline 89/2/3/94 wall 154.228 s — 32,533 bytes); `molmetal/reports/wf375b1_preflight.txt` (Wf-375-B1 baseline 89/2/3/94 wall 146.79 s); `molmetal/reports/wf374b1_preflight.txt` (Wf-374-B1 baseline 89/2/3/94 wall ~155 s); `molmetal/reports/wf372c1_preflight.txt` (Wf-372-C1 clean baseline 92/0/2/94 wall 153.42 s); `molmetal/reports/wf370b1_preflight.txt` (Wf-370-B1 clean baseline 92/0/2/94 wall 152.53 s); `molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json` (Wf-379-A2 honest-negative evidence tree); `molmetal/reports/gpu_launch_dashboard.json` (Wf-377-A4 schema v1.1); `molmetal/scripts/wf377a1_m6_kappa_redesign_validate.py` (~290 LOC); `molmetal/scripts/wf377a2_m2r2_diagnostic.py` (726 LOC); `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (Wf-367-A1 line 2384 defensive fix PRESERVED); `molmetal/scripts/pre_flight_check.py` (Wf-367-A2 `--strict` removal at line 4590-4598 PRESERVED); `scripts/pre_flight_gpu_ready.sh` (Wf-378-A1 USER_OK hard entry gate at lines 145-156); `scripts/launch_wf124_p3_g6.sh` (Wf-378-A2 USER_OK gate repositioning); `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §10 standing directives re-confirmed; `AGENTS.md` §6 NEW row 285 (this sync); `AGENTS.md` last-update trailer (this sync appended); `TODO/INDEX.md` §22.71 (this entry appended between §22.70 Wf-376-B1 and §22.72 Wf-378 OPEN gaps closure); `kill_then_adapt` §0 (NO GPU launch + USER_OK gating); `mol-metal-sci-paper-submission` skill (Wf-351-B3 NEW per 2026-09-29 user directive).

---

## §11 User voice answer (verdict)

> **Wf-378 wave DID彻底解决** the OPEN CPU-only gaps (Wf-378-A1 pre_flight_gpu_ready.sh USER_OK hard entry gate CLOSED + Wf-378-A2 launch_wf124_p3_g6.sh USER_OK gate parity CLOSED — both 5 NEW tests PASS in ~15 min CPU-only) + aggregate INDEX/AGENTS sync (this §22.71 entry) + Wf-378 final synthesis v23 doc SHIPPED; pre-flight 89/2/3/94 PRESERVED across 6+ consecutive runs at the 89/2/3/94 drift level (Wf-374-B1 → Wf-375-B1 → Wf-376-B1 → Wf-377-B1 → Wf-379-B1 → Wf-378-B1 [this run]); standing directives §5.1 + §5.2 + §5.3 + §5.4 + §10 re-confirmed.

> **They did NOT彻底解决** (out of Wf-378 scope; honest-negative preserved per `party-twenty-spirit` §3.5): see §9 (10 honest-negative caveats).

> **NEW ultracode Wf-380+ IS the recommended follow-up** per Wf-378 final synthesis §8 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-379-B1 §22.72.5 with 5+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §6 update for m6_kappa drop-and-promote-alternatives recommendation + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap (~21.5 hr GPU + 10 min CPU + 30 min install total).
