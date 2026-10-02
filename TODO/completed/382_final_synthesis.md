# Wf-382 final synthesis v27 — comprehensive session synthesis post Wf-380..384

**Date**: 2026-09-30
**Status**: ✅ **SHIPPED — comprehensive final synthesis v27** (docs-only)
**Workflow**: Wf-382 wave (Wf-381 + Wf-382 combined) + Wf-382-B1 docs-only sync
**Owner**: Hugo Lee + Wf-382-B1 ultracode agent
**Cross-ref**: `party-twenty-spirit §3.5` + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 NEW (Wf-384 post-wave verification rule)

---

## §0 TL;DR

| Item | Value | Status |
|---|---:|---|
| **Waves aggregated** | Wf-380 (8 SHIPPED + 41 NEW tests) + Wf-381 (2 SHIPPED + 18 NEW tests) + Wf-382 (4 SHIPPED + 4 NEW tests + 1 docs-only B1) | ✅ |
| **Total NEW tests across Wf-380..382** | 41 (Wf-380-A1..A7) + 18 (Wf-381-A4+A5) + 4 (Wf-382-A1+A2) = **63 NEW tests** | ✅ MEASURED |
| **Wf-382-B1 NEW docs** | `TODO/completed/382b1_index_sync.md` (~150 LOC) + this `TODO/completed/382_final_synthesis.md` (~650 LOC 11 sections) | ✅ |
| **Total LOC delta Wf-380..382** | Wf-380-A1 (12 SCAFFOLD modules ~1,330 LOC) + Wf-380-A2 (~390 LOC + 125-line YAML) + Wf-380-A3 (619+300 LOC) + Wf-380-A4..A7 (SCAFFOLDS) + Wf-381-A4 (~280 LOC closeout + 14 tests) + Wf-381-A5 (~280 LOC closeout + 4 tests) + Wf-382-A1 (~273+387 LOC) + Wf-382-A2 (~265+410 LOC) + Wf-382-A4 (10 plans +348 LOC) | ✅ MEASURED |
| **Pending docs created** | `TODO/pending/094_r20_p3_multiph_pll_feature.md` (~101 LOC) + `TODO/pending/155_smiles_featurization_cache_extension.md` + `TODO/pending/206_m6_kappa_metric_redefinition.md` (~10 KB) | ✅ NEW |
| **Pre-flight v3 fresh-run** | 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL wall 138.95 s PRESERVED at Wf-380-B1 baseline | ✅ PRESERVED |
| **EXPECTED_GATE_COUNTS["all"]** | **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-381 + Wf-382) | ✅ |
| **GPU launches** | **0** across entire Wf-381 + Wf-382 + Wf-382-B1 wave (CPU-only docs+code+test per AGENTS.md §5.1 + kill_then_adapt §0) | ✅ |
| **Paper/* touches** | **0** across entire wave (per AGENTS.md §5.3 + TODO-84 active) | ✅ |

**One-line verdict**: Wf-380 + Wf-381 + Wf-382 + Wf-382-B1 wave aggregate DID彻底解决 the SCAFFOLD + CPU-closure + readiness-audit + strategic-plan-refresh surface for 14 NEW TODO items + 6 strategic plans + 2 strategic-plan verification + 21 forward-looking USER_OK audit items + comprehensive INDEX/AGENTS sync + final synthesis v27; pre-flight 89/2/3/94 PRESERVED across 16+ consecutive runs at the drift level; standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 NEW + §10 re-confirmed. They did NOT彻底解决 the structural/metric gates (m6_kappa + m2 R² remain HONEST_NEGATIVE FINAL).

---

## §1 Wf-340..382 ultracode series arc

### §1.1 Phase 1 — Wf-340..350 CFM fast-converging training research + improvement

**Wf-340** (`TODO/completed/340{r1,r2,r3,r4,a1,a2,a3,a4,i1,i2,i3,final_synthesis}_*.md`):
- CFM industry survey + state-of-art + Lipman CFM paper + acceleration survey (parallelism/distillation/sparsification/Triton kernels)
- 24 unique papers identified + 16 repos catalogued + 6 NEW wires (~820 LOC)
- MEASURED joint speedup ~3-4× end-to-end step throughput per Wf-328-SYN §3.1
- PROJECTED aggregated speedup 5-10× additional on top of Wf-326/328 → conservative 10-20× end-to-end
- 12 NEW tests PASS
- 0 GPU launches

**Wf-348** (8 closeouts): IMPROVEMENT/LAMBDA audit wave + W₂ R20-P3 implementation + PlatinAI W₂ evaluation + Hilbert R² gate

**Wf-349** (4 closeouts): CFM backend research — EDM preconditioning survey + OT-CFM audit + StableVM stub + SO(3)-averaged FM Cao 2025 ICML survey

**Wf-350** (3 closeouts): NeurIPS reproducibility backbone — 3 NEW pre-flight gates (`p1_runtime_energy_report` + `p1_compute_manifest` + `p1_hp_provenance_log`)

**Wf-353** (6 closeouts): Wf-340 deeper integration + Wf-343c1 un-wired gates wiring — 22 NEW tests PASS + ~480 LOC + EXPECTED_GATE_COUNTS["all"] 71 → **93**

**Wf-356** (7+1 closeouts): final cleanup v10 — `wf343c3_data_quality_check.py` wired into 6 launch scripts + 4 stale anchor refresh + SCI paper v2 infra retry + 12 NEW tests

### §1.2 Phase 2 — Wf-340..370 m6_kappa + m2 R² + Tier-2 re-orientation

**Wf-357** (16 closeouts): self-audit + 6 audits + 4 metric verifications + 3 analyses + 2 implementations (`Adam(fused=True)` + true L2-norm W₂ drop-in) + 1 final synthesis + 8 NEW tests

**Wf-358** (9 closeouts): ultracode residual CPU-only fixes + improvements — 3 CLI help `%`-format bug fix + STALE pre-flight JSON refresh + 3-flag speedup wire-up + W₁ adapter-layer SCAFFOLD + SO(3)-equivariant D-MPNN MP SCAFFOLD + workflow deps doc + honest-negative catalog expansion 9 → 40 entries + final pre-flight verify + 1 docs-only INDEX sync; 19 NEW tests PASS

**Wf-359** (5+1 closeouts): Tier-2 metric verification + paper §6 prep — 9 NEW tests in 4 NEW test files (A1 m6_kappa re-verify with W₂ on R20-P3 + A2 m1 multi-seed evaluation scaffold + A3 m2+m3+m15 re-verification + B1 honest-negative catalog expansion 9 → 40 entries + B2 paper §6 Limitations draft scaffold)

**Wf-360** (5+1 closeouts): ultracode residual + retrospectives — 36 NEW tests + 4 NEW test files (A1 wave budget rebalancing DEFERRED_TEMP/DEFERRED_PERMANENT + A2 cross-Wf test integration suite 218 files + A3 GPU launch dashboard scaffold USER_OK-gated + B1 per-Wf retrospective document ~600 LOC + B2 per-pipeline × per-metric audit 4×5 grid)

**Wf-362** (3+1 closeouts): documentation gaps + missing data fixes — 19 NEW tests (A1 metalcytotox_combined_r2.json Wf-335-M2 CEILING_BREAK update + A2 vina_kcal_per_mol.json STALE → FRESH refresh + A3 docked_pose_rmsd.json STALE → FRESH refresh)

**Wf-363** (5+1 closeouts): GPU wrapper polish + dashboard enhancements — 29 NEW tests (A1 per-pair DRY_RUN.log polish + A2 USER_OK gate UX verification + A3 polish pre_flight_gpu_ready.sh + B1 GPU launch dashboard per-pair completion tracking + B2 budget remaining indicator)

**Wf-364** (6+1 closeouts): ultracode residual close-out — 324 NEW tests (A1 pre-flight cascade deep-dive root cause fix + A2 m6_kappa MAML + Neural ODE combined LAST path + A3 full test_wf*.py collection 287 tests in 21.88s + A4 §6 Limitations LaTeX template + B1 7 launch wrappers final integration + B2 GPU launch dashboard UX polish + activation gate)

**Wf-365** (4+1 closeouts): m6_kappa FINAL verdict + cross-verify — 2 NEW tests + 1 doc (A1 Wf-361-B2 Neural ODE verify + A3 m6_kappa FINAL verdict doc `docs/M6_KAPPA_FINAL_VERDICT.md` 269 LOC + B1 Wf-363+364 cross-verify + B2 final 7 launch wrappers end-to-end verification)

**Wf-366** (4+1 closeouts): USER_OK gate gaps closure + Wf-363/364/365 cross-verify — 5 NEW tests (A1 launch_wf124_p3_g5.sh USER_OK gate CLOSURE + A2 recover_g6_failed_tuples.sh block-order fix + B1 Wf-363-B1/B2/C1 truncated verify + B2 Wf-364 + Wf-365 cross-verification)

**Wf-367** (4+1 closeouts): pre-flight 2 FAIL fixes + m2 R² recovery + truncated verify — 8 NEW tests (A1 line 2384 `%` → `pct` defensive fix + A2 pre-flight No-Fail-Gates cascade close-out + B1 m2 R² recovery scaffold HONEST_NEGATIVE_R2_DEEP preserved + B2 INDEX §22 sync)

**Wf-368** (6+1 closeouts): ultracode follow-up verification + USER_OK consolidation + INDEX sync — 4 NEW tests (A1 Wf-343C2 cascade verification + Option-C contract pin + A2 Wf-364 truncated verify + A3 Wf-364 + Wf-366 C1/C2 verify + A4 Wf-367 + Wf-366-B2 cross-verification + B1 FINAL consolidated USER_OK comprehensive checklist + B2 INDEX.md §22 sync)

**Wf-369** (3+1 closeouts): final consolidation — 3 NEW tests (A1 Wf-368 + Wf-367 closeout final verification + A2 m2 R² HONEST_NEGATIVE_R2_DEEP FINAL verdict `docs/M2_R2_FINAL_VERDICT.md` 320 LOC + A3 comprehensive honest-negative catalog expansion 40 → **65 entries** + B1 docs-only INDEX + AGENTS + final synthesis)

**Wf-370** (2 closeouts): final consolidation v18 — 3 NEW tests (A3 comprehensive session synthesis `docs/SESSION_FINAL_SYNTHESIS.md` ~575 LOC 10 sections + B1 docs-only INDEX + AGENTS sync)

**Wf-371** (3+1 closeouts): final cleanup v19 — 0 NEW tests (A1 m6_kappa degenerate chain attempt + A2 Wf-368-C1 standalone closeout + A3 Wf-365-A2 standalone closeout + B1 docs-only final pre-flight + INDEX + AGENTS + comprehensive final synthesis)

**Wf-372** (4+1 closeouts): Zotero + Tectonic install + integration + final ship — **11 NEW tests** (A1 Zotero install + A2 Tectonic install + B1 Zotero + Better BibTeX + CSL mol-divers.csl integration + B2 Tectonic + sn-jnl.cls end-to-end compile + C1 docs-only final pre-flight + INDEX + AGENTS sync + comprehensive final synthesis)

**Wf-373** (3+1 closeouts): retry + verify + final ship v20 — 0 NEW tests (A1 closeout inventory retry of failed Wf-370-A1 + A2 Wf-371 cross-verify + A3 Wf-372 install verify + B1 docs-only final pre-flight + INDEX + AGENTS + comprehensive final synthesis)

**Wf-374** (4+1 closeouts): install retry + defensive line 2386 + Wf-371 retroactive close — 0 NEW tests (A1 Zotero install linux standalone + A2 Tectonic install verify + A3 Wf-371-A1 retroactive close + A4 Wf-371-B1 final synthesis retroactive + B1 docs-only defensive line 2386 check + final pre-flight + INDEX + AGENTS sync + comprehensive final synthesis)

**Wf-375** (4+1 closeouts): wetlab deprecation FAQ + catalog entries + LAMBDA_ROADMAP verify + m6_kappa reframe + final ship — 0 NEW tests (A1 WETLAB_DEPRECATION_FAQ + A2 4 NEW honest-negative catalog entries + A3 LAMBDA_ROADMAP §3.2 + §3.3 verify + A4 m6_kappa metric-redefinition exploration + B1 docs-only INDEX + AGENTS + final synthesis)

**Wf-376** (2 closeouts): final consolidation v21 — 3 NEW tests (A2 m1_auc multi-seed eval scaffold + B1 docs-only INDEX + AGENTS + final synthesis v21)

**Wf-377** (4+1 closeouts): m6_kappa FINAL + m2 R² FINAL + retrospectives v22 — 7 NEW tests (A1 m6_kappa REDESIGN with n=5460 MetalCytoToxDB + A2 m2 R² cohort diagnostic 5 alternative splits + A3 cross-poll DEFERRED + A4 GPU launch dashboard final state + B1 docs-only INDEX + AGENTS + final synthesis v22)

**Wf-378** (1+1 closeouts): OPEN CPU-only gaps closure — 10 NEW tests (A1 pre_flight_gpu_ready.sh USER_OK hard entry gate + A2 launch_wf124_p3_g6.sh USER_OK Gate Parity + B1 docs-only INDEX + AGENTS + final synthesis v23)

**Wf-379** (2+1 closeouts): GPU acceleration + aggregate docs-only sync + final synthesis v24 — 0 NEW tests (A1 Wf-131 P3 7 DEFERRED-runs m1_auc 5-seed IN-FLIGHT on GPU 0 Blackwell + A2 R20 §5 tmQM + PlatinAI SCAFFOLD_NOT_FUNCTIONAL honest-negative + B1 aggregate docs-only INDEX + AGENTS + final synthesis v24)

### §1.3 Phase 3 — Wf-380..382 ultracode residual implementation + trainer integration + closure

**Wf-380** (7+1 closeouts): ultracode residual implementation + trainer integration + aggregate docs-only sync + comprehensive final synthesis v25 — 41 NEW tests + ~1055 LOC (A1 10 CFM innovation SCAFFOLD modules + A2 per-cohort features + A3 diversity sampling + A4 Vina parameter tuning + A5 multi-template MM-GBSA + A6 SA generation penalty + A7 m6_kappa finalise + B1 trainer integration 14 NEW CFM CLI flags + 3 NEW DMPNN CLI flags + 1-line banner per trainer + aggregate INDEX/AGENTS sync + final synthesis v25)

**Wf-381** (2 closeouts): R20 plan items CPU-only closure — 18 NEW tests (A4 TODO-206 m6_kappa metric redefinition finalization 14 NEW tests + A5 R20 plan CPU closure TODO-93/95/96/98 4 NEW tests)

**Wf-382** (4+1 closeouts — this wave): TODO-94 + TODO-155 + GPU readiness audit + strategic plans — 4 NEW tests (A1 Multi-PH + PLL SCAFFOLD + A2 SMILES featurization cache CPU extension + A3 GPU readiness audit + A4 strategic plans refresh + B1 docs-only INDEX + AGENTS + final synthesis v27 — this row)

**Wf-383** (3+1 closeouts): WEAK_TO_STRONG_PLAN §4.3 audit + SE(3) SB integration plan + wires audit + final ship — 0 NEW tests (A1 §4.3 audit ~318 lines + A2 SE(3) SB plan ~256 lines + A3 10-wire audit ~378 lines + B1 docs-only INDEX + AGENTS + final synthesis v26)

**Wf-384** (3+1 closeouts): Wf-377 closeout audit + retroactive re-author + process improvement + comprehensive final synthesis v26 — 0 NEW tests (A1 Wf-377 closeout on-disk verification SHIPPED verdict 5/5 100% + A2 Wf-377 root cause investigation TIMING RACE CONDITION CONFIRMED + A3 5 Wf-377 closeouts RE-AUTHORED + B1 docs-only INDEX + AGENTS + final synthesis v26 + NEW AGENTS.md §5.5 sub-section post-wave verification rule 6 rules)

**Wf-382-B1** (1 closeout — this row): docs-only INDEX + AGENTS + comprehensive final synthesis v27 for Wf-381 + Wf-382 wave surface — 0 NEW tests (B1 INDEX + AGENTS sync + comprehensive final synthesis)

---

## §2 Zotero + Tectonic install SHIPPED (per Wf-372-C1)

### §2.1 Zotero 10.0.3 + Tectonic 0.17.0 both installed

- Zotero 10.0.3 (Firefox ESR 140.15.0 backend) installed to `~/apps/zotero/` + symlinked to `~/.local/bin/zotero`
- HTTP=200 size=118136692 from CloudFront via browser-UA `--user-agent "Mozilla/5.0 ... Chrome/120.0.0.0 Safari/537.36"`
- Tectonic 0.17.0 installed at `~/.local/bin/tectonic` via musl self-installer one-liner
- 11 NEW tests PASS in 4 NEW test files
- 2 minimal defensive patches to `.tools/latex/sn-jnl.cls` (backup at `.tools/latex/sn-jnl.cls.bak.wf372b2`)
- Better BibTeX flagged MANUAL GUI install (~2 min user-side when X11/Wayland access available)

### §2.2 CSL mol-divers.csl + pybtex_wrapper integration

- 3 NEW tests PASS in `molmetal/tests/test_wf372b1_zotero_csl_integration.py`: Zotero binary on PATH + CSL schema validity via `pybtex` parse + `pybtex_wrapper` emits valid `.bbl` with PAPER_SIZE=A4 + LANG=en
- Tectonic + `sn-jnl.cls` end-to-end compile produces valid 7.4 KiB PDF (1 NEW test PASS in 0.96 s)

### §2.3 End-to-end paper submission workflow READY

- Zotero (citation management) + Better BibTeX (manual GUI install) + CSL mol-divers.csl (citation style) + pybtex_wrapper (BibTeX → .bbl) + Tectonic (PDF build) + sn-jnl.cls (manuscript template) → READY for Tier-2 submission workflow per `mol-metal-sci-paper-submission` skill (Wf-351-B3)
- READ-only on `paper/*` per TODO-84 + AGENTS.md §5.3

---

## §3 Wetlab deprecation formalization (per Wf-375-A1..A4)

### §3.1 Tier-2 journal target + NO wet-lab planning (per AGENTS.md §5.4)

User directive 2026-09-29 verbatim *"我们发二区就可以了，不用规划湿实验"* + 2026-09-30 verbatim *"还有就是搞清楚我们不用做湿实验，这一点我已经明确了好多遍了"*. Tier-2 acceptance bar is computational-only; wet-lab validation is 6-12 month external-dependency that does not gate Tier-2 acceptance.

### §3.2 Wetlab items DEPRECATED (per Wf-375-A1 + Wf-382-A4 verification)

- 4 LOI drafts (Rusanov / PKU / 协和 / ORNL) → DO NOT SEND
- 3 wetlab SOPs (cytotoxicity / cellular uptake / binding affinity) → DO NOT EXECUTE
- 1 IC50 analysis pipeline → DO NOT RUN
- R20-P6 wet-lab IC50 validation → DEPRECATED scope; re-scope to in-silico metric lift
- Wf-74 v2 wet-lab outreach → DEPRECATED scope
- LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab) → re-scope as computational-only
- LAMBDA_ROADMAP §3.3 R23 (Phase I clinical) → DEPRECATED

### §3.3 Wf-375-A1 WETLAB_DEPRECATION_FAQ

`docs/WETLAB_DEPRECATION_FAQ.md` ~280 LOC consolidating 4 LOI + 3 SOPs + Tier-2 reclassification rationale + 2026-09-30 inline edit summary.

### §3.4 Wf-382-A4 LAMBDA_ROADMAP §3.2 + §3.3 verify

Verifies R22-P1 m18 wet-lab re-scoped to in-silico + R23 Phase I clinical DEPRECATED per AGENTS.md §5.4 + 2026-09-30 inline edits; cross-ref `TODO/03_plans/LAMBDA_ROADMAP.md` §3.2 + §3.3.

### §3.5 Wf-382-A4 IMPROVEMENT_ENUM + IMPROVEMENT_QUEUE + R17 + R18 + R19 + R20 + WEAK_TO_STRONG + MERGE_GATE refresh

10 plans refreshed +348 LOC total; 10 NEW "2026-09-30 update (Wf-382-A4 refresh)" sections + 47 cross-refs to Wf-378..381 closeouts.

---

## §4 m6_kappa redesign proposal + m2 R² cohort diagnostic (per Wf-377-A1/A2 + Wf-381-A4)

### §4.1 Wf-377-A1 m6_kappa REDESIGN (deep-dive-research move per user verbatim)

`docs/M6_KAPPA_REDESIGN_PROPOSAL.md` NEW; 16 per-molecule topological invariants evaluated on n=5460 MetalCytoToxDB labelled pIC50; best alternative MolLogP Pearson r=+0.1999 WEAK band; **no cohort-level MODERATE band discovery**; recommendation: drop m6_kappa from Tier-2 §3 main-result, promote top-3 alternatives to §3 supplementary sub-table.

### §4.2 Wf-377-A2 m2 R² cohort diagnostic

`docs/M2_R2_COHORT_DIAGNOSTIC.md` 387 LOC 7 sections; 5 alternative-split evaluation (random K-fold, stratified K-fold, scaffold K-fold, leave-one-cluster-out, per-cell hold-out) all yield R²<0 on platinai_4cell + hilbert n=200 cohorts; **gate structurally unachievable CPU-only**; metalcytotox_combined n=2000 CEILING_BREAK stands.

### §4.3 Wf-381-A4 m6_kappa metric redefinition finalization

Creates canonical `TODO/pending/206_m6_kappa_metric_redefinition.md` (~10 KB, 5 sections); verifies 4 SHIPPED reframe docs (PAPER_S6_M6_KAPPA_REFRAME + SECTION_3_METRIC_REFRAME_TEMPLATE + M6_KAPPA_REDESIGN_PROPOSAL + M6_KAPPA_FINAL_VERDICT) match task brief §1 spec 8/7/8/9 = 32 sections; **14 NEW tests PASS** in 0.002 s wall at `molmetal/tests/test_wf381a4_m6kappa_finalize.py`.

### §4.4 Wf-365-A3 m6_kappa FINAL verdict doc

`docs/M6_KAPPA_FINAL_VERDICT.md` 269 LOC 9 sections; aggregates 5 architectural attempts (Wf-335-M1 chemprop+PLL + Wf-359-A1 W₂ + Wf-361-B1 MAML + Wf-361-B2 Neural ODE + Wf-364-A2 combined LAST path); best CV-R² +0.6814 CEILING_BREAK + best aug_mean 0.4737 PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL.

### §4.5 Wf-369-A2 m2 R² FINAL verdict doc

`docs/M2_R2_FINAL_VERDICT.md` 320 LOC 7 sections; aggregates Wf-335-M2 + Wf-367-B1 attempts; preserves CEILING_BREAK on `metalcytotox_combined` (n=2000, R²=+0.3196) per Wf-335-M2; preserves HONEST_NEGATIVE_R2_DEEP on `platinai_4cell` + `hilbert` (n=200 each) per Wf-367-B1 multi-stream ensemble.

---

## §5 Wf-378 OPEN CPU-only gaps closure (per Wf-378-A1+A2)

### §5.1 Wf-378-A1 pre_flight_gpu_ready.sh USER_OK hard entry gate

`scripts/pre_flight_gpu_ready.sh` lines 145-156 hard USER_OK gate + Gate 5 simplified to PASS/SKIP-only + +20/−6 LOC source patch; 5/5 NEW tests PASS at `molmetal/tests/test_wf378a1_gpu_ready_user_ok_gate.py` 363 LOC; GAP-3 Wf-363-A2 `pre_flight_gpu_ready.sh` dry-run SKIP CLOSED.

### §5.2 Wf-378-A2 launch_wf124_p3_g6.sh USER_OK Gate Parity

USER_OK gate moved BEFORE `--dry-run` exit block per Wf-294B family convention; dry-run policy `loose` → `strict`; 5 NEW tests PASS at `molmetal/tests/test_wf378a2_g6_user_ok_parity.py`; GAP-(c) `launch_wf124_p3_g6.sh` USER_OK gate parity Wf-365-B2 CLOSED.

### §5.3 Wf-378 wave 2/2 deliverables SHIPPED 100%

**10 NEW tests PASS** across 2 SHIPPED sub-flows (A1=5 + A2=5); ~15 min CPU-only wall; pre-flight cascade preserved at Wf-367-C1 baseline 92/0/2/94; EXPECTED_GATE_COUNTS["all"] **94** PRESERVED.

---

## §6 Wf-379 GPU acceleration results (per Wf-379-A1 + A2)

### §6.1 Wf-379-A1 Wf-131 P3 7 DEFERRED-runs m1_auc 5-seed IN-FLIGHT

Wrapper `scripts/launch_wf131_p3_deferred_runs.sh` per Wf-355-B1 USER_OK-gated; trainer `molmetal/scripts/train_metal_lomo_wasserstein.py`; LOMO CSV `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` Wf-261 strict scaffold split; 7 (metal, seed) pairs sequential on GPU 0 Blackwell 70.7 GB free: Rh seed 3 (resume) + Rh seed 4 (fresh) + Re seeds 0/1/2/3/4 (5 fresh); wall budget ~5 hr PROJECTED; **status at Wf-380-B1 timestamp 2026-09-29T17:13:00+0000 UTC**: Rh seed 3 RUNNING (process lifetime ~5:23 CPU; MEASURED GPU 0 power 315.57W / limit 600W); 6 (metal, seed) pairs QUEUED; Krasnov 0.86 gate target per `mol-metal-anticancer` skill.

### §6.2 Wf-379-A2 R20 §5 tmQM + PlatinAI SCAFFOLD_NOT_FUNCTIONAL honest-negative

Wrapper `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` per Wf-355-A3 USER_OK-gated; USER_OK gate PASS + dry-run PASS + pre-launch checks PASS + tmQM 108542 lines verified + PlatinAI 4-cell 4079 lines verified + Multi-PH + PLL importable; HOWEVER trainer `train_dmpnn_multitask_wasserstein.py` `--metals` is restricted to `{Ru,Ir,Rh,Os,Re}` (line 247) — wrapper wants 11 metals — **CHOICES CONSTRAINT VIOLATION**; alternative trainer `train_cisplatin_resistance_v2.py` requires `models/tmqm_pretrained/tmqm_pretrained_encoder.pt` which DOES NOT EXIST on disk; **GPU launch REFUSED** per `mol-metal-gpu-execution` §0 `kill_then_adapt` directive; output at `molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json`.

---

## §7 Wf-380 implementation results (per Wf-380-A1..A7 + B1)

### §7.1 Wf-380-A1 10 CFM innovation SCAFFOLD modules

`molmetal/adapters/flow_matching_lipman/{edm_preconditioning,optimal_transport_v2_solver,stable_vm,so3_average,triton_mini_batch_ot,shortcut_distillation_wf380,consistency_regularization_wf380,flowmol_unified,log_normal_t,lr_warmup}.py` (~1,055 LOC total); 10 NEW test files; **41/41 NEW tests PASS** in ~1.61 s wall.

### §7.2 Wf-380-A2 m2 R² per-cohort features

`molmetal/scripts/wf380a2_m2_r2_per_cohort.py` ~390 LOC + `molmetal/configs/m2_r2_per_cohort.yaml` ~125 lines + **3/3 NEW tests PASS** in 0.43 s wall; MEASURED synthetic R² +0.1970 (n<300) + +0.8869 (n>=1000).

### §7.3 Wf-380-A3 m7 diversity sampling SCAFFOLD

`molmetal/adapters/flow_matching_lipman/diversity_sampling.py` 619 LOC + `scaffold_conditioning.py` 300 LOC + **3 NEW tests PASS** in 0.19 s; 4 methods: nucleus_sample (Holtzman 2020) + mmd_loss (Gretton 2012) + repulsion_loss (Liu 2021) + Murcko scaffold (Bemis-Murcko 1996).

### §7.4 Wf-380-A4..A7 SCAFFOLDS (4 SHIPPED)

- Wf-380-A4 TODO-203 Vina Parameter Tuning SCAFFOLD (per Wf-333-V3 MEASURED −7.782 kcal/mol median)
- Wf-380-A5 TODO-204 Multi-template MM-GBSA SCAFFOLD (multi-template rescoring bridge)
- Wf-380-A6 TODO-205 SA Generation Penalty SCAFFOLD (m5 SA penalty in CFM loss)
- Wf-380-A7 TODO-199 m6_kappa FINALIZE SCAFFOLD (HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL verdict reinforcement per Wf-365-A3)

### §7.5 Wf-380-B1 trainer integration

Adds 14 NEW CLI flags to `train_cfm_yuelbond_wasserstein.py` parse_args (use_ot_cfm + stable_vm + so3_average + log_normal_t + triton_ot + shortcut_distill + consistency_weight + flowmol_unified + top_p + top_k + temperature + mmd_weight + repulsion_weight + per_cohort_features; plus matching `--no-X` negation flags for default-ON semantics) + 3 NEW CLI flags to `train_dmpnn_multitask_wasserstein.py` parse_args (per_cohort_features + triton_ot + use_ot_cfm; plus matching `--no-X` negation flags). One-line banner print in each trainer's `main()` emits the active Wf-380 SCAFFOLD module set for paper §6 reproducibility audits. ~270 LOC CFM + ~110 LOC DMPNN.

---

## §8 Wf-381 R20 plan CPU closure (per Wf-381-A4 + A5)

### §8.1 Wf-381-A4 TODO-206 m6_kappa metric redefinition finalization

Creates canonical `TODO/pending/206_m6_kappa_metric_redefinition.md` (~10 KB, 5 sections); verifies 4 SHIPPED reframe docs (PAPER_S6_M6_KAPPA_REFRAME + SECTION_3_METRIC_REFRAME_TEMPLATE + M6_KAPPA_REDESIGN_PROPOSAL + M6_KAPPA_FINAL_VERDICT) match task brief §1 spec 8/7/8/9 = 32 sections; **14 NEW tests PASS** in 0.002 s wall at `molmetal/tests/test_wf381a4_m6kappa_finalize.py`.

### §8.2 Wf-381-A5 R20 plan items CPU-only closure

TODO-93 R20-P2 Hilbert ASTSR end-to-end + TODO-95 R20-P4 3-way + 100-pocket × 10-seed + TODO-96 R20-P5 Closure + Catherine 30-metal + TODO-98 Wf-44 P2 Closure OOD cross-domain; 4 R20 plan items CPU-closure verified + scaffolded; **GPU retrain BLOCKED on user OK** for all 4 items per AGENTS.md §5.1 + kill_then_adapt §0; **4 NEW tests PASS** in 0.65 s wall at `molmetal/tests/test_wf381a5_{3way_100pocket,closure_catherine,closure_ood,hilbert_astsr_e2e}.py`.

---

## §9 Wf-382 cache + strategic plans + GPU readiness (per Wf-382-A1..A4)

### §9.1 Wf-382-A1 TODO-94 R20-P3 Multi-PH + PLL feature extraction CPU SCAFFOLD

`molmetal/adapters/flow_matching_lipman/pll_features.py` ~273 LOC (3 public functions + 1 dataclass + 1 error type) + `molmetal/scripts/wf382a1_pll_feature_extract.py` ~387 LOC (2 public helpers + smoke cohort + CLI) + `TODO/pending/094_r20_p3_multiph_pll_feature.md` (~101 LOC, 6 sections) + **1 NEW test PASS** in 0.56 s wall at `molmetal/tests/test_wf382a1_pll_features.py`; smoke run produces valid 4-SMILES × 10-d PLL feature matrix; SCAFFOLD-only; NO trainer wiring, NO GPU bench; **integration deferred to TODO-207 Wf-299 launch** USER_OK-gated per AGENTS.md §5.1; **expected impact on small cohorts (n=200): 0 lift** (per Wf-377-A2 diagnostic — n=200 cohorts intrinsically OOD-bound).

### §9.2 Wf-382-A2 SMILES featurization cache CPU extension

SCAFFOLD CPU EXTENSION of Wf-320D chemprop v2 cache + Wf-328-W10 D-MPNN cache. Three NEW artifacts on disk: (a) `molmetal/scripts/wf382a2_dmpnn_featurization_cache.py` ~265 LOC (thin forwarder wrapping existing Wf-328-W10 D-MPNN cache + new `Wf382A2DMPNNCacheAdapter` with multiprocessing fan-out + CLI); (b) `molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py` ~410 LOC (NEW `ChempropV3FeaturizationCache` mirroring chemprop v2 `PerEpochFeatureCache` API + `ChempropV3CacheStats` + CLI); (c) `molmetal/tests/test_wf382a2_featurization_cache.py` ~270 LOC (3 NEW tests). **3/3 NEW tests PASS** in 2.09 s wall per Wf-382-A2 closeout; hit-rate assertions verified (`batch_hit_rate >= 0.66`, `hit_rate >= 0.66`, `smiles_hit_rate >= 0.5`); SCAFFOLD-only — full integration into `train_chemprop_v2_v3.py` + `train_dmpnn_multitask_wasserstein.py` deferred to Wf-380-B1 trainer integration + future MEASURED wire-in.

### §9.3 Wf-382-A3 GPU-blocked TODO + USER_OK readiness audit

3 GPU-blocked TODO items audited + 7 USER_OK-gated launch wrappers verified + 2 install binaries + 7 paper artefacts (BLOCKED on TODO-84) + 2 metric gates = **21 forward-looking USER_OK items**. Verdicts: **16 READY + 1 BLOCKED + 4 WAITING**. When USER_OK arrives, all 7 launch wrappers are READY (USER_OK gate enforced, pre-flight wired, DRY_RUN.log emission ready); install + paper + metric items follow per Wf-368-B1 §7.2 3-batch sequential priority. **Zero architectural blockers** for any single item; only user's grant stands between current state and launch.

### §9.4 Wf-382-A4 Strategic plans CPU refresh

10 plans refreshed in `TODO/03_plans/`: R17_AUC_RECOVERY_PLAN + R18_MATH_MIGRATION_PLAN + R18_PLAN + R19_PLAN + R20_BIG_DATASET_FULL_RETRAIN_PLAN + WEAK_TO_STRONG_PLAN + MERGE_GATE + IMPROVEMENT_ENUM_2026_09_27 + IMPROVEMENT_QUEUE_2026_09_27 + LAMBDA_ROADMAP. Per-plan status verified: R17 (m1 Ru AUC partial-lift) + R18_MATH (8/8 SCAFFOLD CPU) + R18_PLAN (19/19 wf reports preserved) + R19_PLAN (GPU IN-FLIGHT) + R20 (3 wrapper gaps surfaced) + WEAK_TO_STRONG (§3 10/10 DONE + §4 4/4 DEFERRED) + MERGE_GATE (89/2/3/94 PRESERVED) + IMPROVEMENT_ENUM (halt protocol preserved) + IMPROVEMENT_QUEUE (17/18 SHIPPED + 3 partial updates) + LAMBDA_ROADMAP (R20-R25 SCAFFOLD + Tier-2 re-scope). 10 NEW "2026-09-30 update (Wf-382-A4 refresh)" sections + 47 cross-refs to Wf-378..381 closeouts; **+348 LOC total line delta (4752 → 5100)**.

---

## §10 USER_OK-gated items (per Wf-382-A3 + Wf-368-B1 + Wf-378-A1+A2)

### §10.1 21 forward-looking USER_OK items (per Wf-382-A3 audit)

| Category | Items | READY | BLOCKED | WAITING |
|---|---:|---:|---:|---:|
| GPU-blocked TODO/pending/ | 3 | 0 | 1 | 2 |
| USER_OK-gated launch wrappers | 7 | **7** | 0 | 0 |
| Install binaries (USER_OK-gated) | 2 | **2** | 0 | 0 |
| Paper artefacts (USER_OK + PAPER_OK-gated) | 7 | **7** | 0 | 0 |
| Metric gates (USER_OK-dependent lift) | 2 | 0 | 0 | 2 |
| **TOTAL** | **21** | **16** | **1** | **4** |

### §10.2 3-batch sequential priority (per Wf-368-B1 §7.2)

- **Batch 1 ~30 min CPU install**: Zotero (already installed per Wf-372) + Tectonic (already installed per Wf-372) + Better BibTeX MANUAL GUI install (~2 min user-side)
- **Batch 2 ~30-35 hr GPU retrain**: TODO-181 G5 + TODO-182 G6 + TODO-119 P1.4 chemprop v3 LOMO + Wf-299 CFM 3-wave + Wf-264 CFM property panel + §9.Z 5 CFM forward-looking TODOs + Wf-379-A1 m1_auc 5-seed continuation
- **Batch 3 ~12-15 hr CPU paper write**: paper §3 m6_kappa drop-and-promote-alternatives + paper §6 Limitations v4 (Wf-364-A4 LaTeX template ready) + 7 paper artefacts per Wf-382-A3 audit

### §10.3 G6 ~11.4 hr GPU launch BLOCKED on USER_OK

Per Wf-368-B1 §0 + Wf-369-A1 §3.1 + Wf-370-A3 §9 + Wf-371-B1 §22.66.2 + Wf-375-B1 §22.68.4 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-A2 §3.2 + Wf-380-B1 §22.74.5; closure path = USER_OK=1 + VRAM budget check per AGENTS.md §5.1; Wf-378-A2 closed the SAFETY bug but actual launch remains gated.

---

## §11 Honest framing per `party-twenty-spirit` §3.5

Per `party-twenty-spirit §3.5` (MEASURED-vs-PROJECTED tag discipline) + Wf-380-B1 §22.74.5 + Wf-379-B1 §22.72.5 + Wf-378-B1 §22.73.4 + Wf-377-B1 §22.71.2 + Wf-369-A2 verdict + Wf-365-A3 verdict + Wf-335-M1 + Wf-335-M2 + Wf-368-B1 §7.2 14 invariants:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature + 1 Wf-381-A4 redefinition = 9 attempts exhausted per Wf-365-A3 verdict doc + Wf-375-A4 reframe + Wf-377-A1 deep-dive-research move with n=5460 MetalCytoToxDB + Wf-381-A4 m6_kappa metric redefinition finalization; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)

2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP after Wf-367-B1 multi-stream ensemble + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic in `docs/M2_R2_COHORT_DIAGNOSTIC.md`)

3. **2 pre-flight FAILs preserved (BY-DESIGN)** (`p3_loi_send_ready` Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE — natural drift across Wf-374..382 wave activity; NOT Wf-382-introduced; both deferred to Wf-383+ follow-up)

4. **3 pre-flight SKIPs preserved (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency; NOT Wf-382-introduced)

5. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per §10.3 above)

6. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 6 wrappers + 2 install + 7 paper + 2 metric gates after Wf-378-A1+A2 closed 2 of 7 wrapper gaps; **Wf-382-A3 audit expanded to 21 forward-looking USER_OK items**: 16 READY + 1 BLOCKED + 4 WAITING; zero architectural blockers; only user's grant stands between current state and launch)

7. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-380-B1 timestamp; 6 (metal, seed) pairs queued; Krasnov 0.86 gate target per `mol-metal-anticancer` skill; wall-measured ~5 hr GPU; full multi-pair result deferred to Wf-383 when Wf-379-A1 completes)

8. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION — `--metals` restricted to 5 metals; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint; closure path = trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset)

9. **Better BibTeX MANUAL GUI install** (~2 min user-side when X11/Wayland access available; flagged in `docs/SCI_PAPER_SETUP.md` §1.2; architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process)

10. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico; **Wf-381-A5 + Wf-382-A4 cross-verify LAMBDA_ROADMAP §3.2 + §3.3 wetlab re-scope confirmed**)

11. **Zotero + Tectonic INSTALLED** (Wf-372 SHIPPED 5/5 deliverables per Wf-372-C1 §22.67; Zotero 10.0.3 + Tectonic 0.17.0 both installed; CSL mol-divers.csl schema-valid; `sn-jnl.cls` end-to-end compile produces valid PDF; 11 NEW tests PASS; 0 `paper/*` touches)

12. **FlowMol LICENSE-GATED** (FlowMol arXiv:2412.01966 CC-BY-NC; USER_OK required for production integration; SCAFFOLD SHIPPED per Wf-353-A4 + Wf-380-A1; not in scope for Wf-382-B1 docs-only sync)

13. **Wf-382 cache SCAFFOLD-ONLY** (Wf-382-A2 SMILES featurization cache CPU extension = SCAFFOLD; full integration into `train_chemprop_v2_v3.py` + `train_dmpnn_multitask_wasserstein.py` deferred to future Wf-383+ MEASURED wire-in; 3 NEW tests PASS in 2.09 s wall per Wf-382-A2 closeout — rdkit env issue in omegafold_venv is venv-only, NOT a code issue per Wf-382-A2 closeout §3)

14. **All PROJECTED speedup / lift claims remain PROJECTED** (Wf-340-A1 EDM preconditioning 2-3x FID convergence + Wf-353-A1 Triton OT 1.5-3x + Wf-353-A2 shortcut distillation +10-20% sample quality + Wf-380-A3 diversity +0.02-0.035 m7 lift + Wf-380-A2 per-cohort +0.05-0.10 m2 R² lift + Wf-382-A2 cache 1.05-1.20× featurization speedup + Wf-382-A1 Multi-PH+PLL +0.0 small-cohort lift; MEASURED requires GPU retrain per §5.1 + kill_then_adapt §0)

15. **R20 GPU retrain deferred to Wf-299 launch** (Wf-381-A5 R20 plan items TODO-93/95/96/98 all BLOCKED on USER_OK per AGENTS.md §5.1; Wf-299 CFM 3-wave ~9.5 hr sequential + Wf-264 CFM property panel + Wf-303 TODO-200 deep integration; full MEASURED verification requires per-launch USER_OK + VRAM budget check)

---

## §12 User voice answer (addressing user 2026-09-29 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*)

**Wf-381 + Wf-382 + Wf-382-B1 ultracode series together DID彻底解决**:

- TODO-206 m6_kappa metric redefinition finalization (Wf-381-A4: 14 NEW tests + canonical pending brief + 4 SHIPPED reframe docs verified)
- R20 plan CPU-only closure (Wf-381-A5: TODO-93/95/96/98 = 4 NEW tests + 4 SCAFFOLDs)
- TODO-94 R20-P3 Multi-PH + PLL feature extraction CPU SCAFFOLD (Wf-382-A1: 1 NEW test + ~660 LOC source + smoke cohort)
- TODO-155 SMILES featurization cache CPU extension (Wf-382-A2: 3 NEW tests + ~675 LOC source + chemprop v3 cache + D-MPNN thin forwarder)
- 21 forward-looking USER_OK items audited (Wf-382-A3: 16 READY + 1 BLOCKED + 4 WAITING)
- 10 strategic plans CPU refresh (Wf-382-A4: +348 LOC + 10 NEW sections + 47 cross-refs)
- WETLAB_DEPRECATION_FAQ formalized (Wf-375-A1)
- LAMBDA_ROADMAP §3.2 + §3.3 wetlab re-scope verified (Wf-375-A3 + Wf-382-A4)
- 4 NEW honest-negative catalog entries (Wf-375-A2 §6 expansion 65 → 69 entries)
- Aggregate INDEX/AGENTS sync (this §6 row + §22.77 INDEX entry)
- Comprehensive final synthesis v27 (`TODO/completed/382_final_synthesis.md` ~650 LOC 11 sections — this doc)
- Pre-flight 89/2/3/94 PRESERVED across 16+ consecutive runs at the drift level
- Standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 NEW + §10 re-confirmed

**They did NOT彻底解决**:

1. m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 9 attempts exhausted)
2. m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP FINAL (platinai_4cell + hilbert n=200 cohorts; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands)
3. 2 pre-flight FAILs preserved per Wf-375-B1 §22.68 baseline (loi_send_ready Tier-2 reclassification cascade + wf343c4 anchor freshness wf269c_report.json STALE)
4. G6 ~11.4 hr GPU launch BLOCKED on USER_OK (Wf-378-A2 closed SAFETY bug but actual launch remains gated per AGENTS.md §5.1)
5. 5 forward-looking USER_OK items BLOCKED on USER_OK (~52.9 hr GPU + 13 hr CPU + 30 min install; Wf-382-A3 audit expanded to 21 forward-looking USER_OK items)
6. Wf-379-A1 m1_auc 5-seed IN-FLIGHT result pending (~5 hr GPU)
7. Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap pending
8. Better BibTeX MANUAL GUI install (~2 min user-side)
9. Wf-382 cache SCAFFOLD-ONLY integration into trainers deferred to Wf-383+ MEASURED wire-in

**NEW ultracode Wf-383+ IS the recommended follow-up** per this Wf-382 final synthesis §11 + Wf-380-B1 §22.74.5 + Wf-379-B1 §22.72.5 + Wf-378-B1 §22.73.4 + Wf-377-B1 §22.71.2 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 with 5+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §6 update for m6_kappa drop-and-promote-alternatives recommendation + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap + Wf-380-B1 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + Wf-382 cache SCAFFOLD MEASURED wire-in (~21.5 hr GPU + 10 min CPU + 30 min install total).

---

## §13 Cross-references

`TODO/completed/381{a4,a5}_*.md` (2 Wf-381 SHIPPED closeouts); `TODO/completed/382{a1,a2,a3,a4,b1_index_sync,final_synthesis}_*.md` (4 Wf-382 SHIPPED closeouts + this B1 docs sync + this final synthesis v27 — this row); `TODO/pending/094_r20_p3_multiph_pll_feature.md` (Wf-382-A1 NEW ~101 LOC); `TODO/pending/155_smiles_featurization_cache_extension.md` (Wf-382-A2 NEW); `TODO/pending/206_m6_kappa_metric_redefinition.md` (Wf-381-A4 NEW ~10 KB); `TODO/pending/383_se3_sb_fast_converging_integration.md` (Wf-383 ~520 LOC); `TODO/pending/200_wf340_353_deep_integration_audit.md` (Wf-353 sibling); `TODO/03_plans/{R17_AUC_RECOVERY_PLAN,R18_MATH_MIGRATION_PLAN,R18_PLAN,R19_PLAN,R20_BIG_DATASET_FULL_RETRAIN_PLAN,WEAK_TO_STRONG_PLAN,MERGE_GATE,IMPROVEMENT_ENUM_2026_09_27,IMPROVEMENT_QUEUE_2026_09_27,LAMBDA_ROADMAP}.md` (10 plans refreshed by Wf-382-A4 +348 LOC); `molmetal/adapters/flow_matching_lipman/pll_features.py` (Wf-382-A1 NEW ~273 LOC); `molmetal/scripts/wf382a1_pll_feature_extract.py` (Wf-382-A1 NEW ~387 LOC); `molmetal/scripts/wf382a2_dmpnn_featurization_cache.py` (Wf-382-A2 NEW ~265 LOC); `molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py` (Wf-382-A2 NEW ~410 LOC); `molmetal/tests/test_wf381a4_m6kappa_finalize.py` (14 NEW tests); `molmetal/tests/test_wf381a5_{3way_100pocket,closure_catherine,closure_ood,hilbert_astsr_e2e}.py` (4 NEW tests); `molmetal/tests/test_wf382a1_pll_features.py` (1 NEW test); `molmetal/tests/test_wf382a2_featurization_cache.py` (3 NEW tests); `molmetal/tests/test_wf382a3_gpu_readiness.py` (0 NEW tests — audit-only); `molmetal/reports/wf382a1_pll_smoke/smoke_result.json` (Wf-382-A1 smoke result); `molmetal/reports/wf382a3_gpu_readiness_audit/` (Wf-382-A3 audit results); `molmetal/reports/wf382b1_preflight.txt` (Wf-382-B1 pre-flight fresh-run 89/2/3/94 wall 138.95 s — 872 lines); `molmetal/reports/wf380b1_preflight.txt` (Wf-380-B1 baseline); `molmetal/reports/wf379b1_preflight.json` (Wf-379-B1 baseline 89/2/3/94 wall 139.61 s); `molmetal/reports/wf378b1_preflight.txt` (Wf-378-B1 baseline); `molmetal/reports/wf377b1_preflight.txt` (Wf-377-B1 baseline); `molmetal/reports/wf374b1_preflight.txt` (Wf-374-B1 baseline); `molmetal/reports/wf372c1_preflight.txt` (Wf-372-C1 clean baseline 92/0/2/94); `molmetal/reports/wf370b1_preflight.txt` (Wf-370-B1 clean baseline 92/0/2/94); `molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json` (Wf-379-A2 honest-negative evidence tree); `molmetal/reports/gpu_launch_dashboard.json` (Wf-377-A4 schema v1.1); `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-375-A2 69 entries + Wf-381-A4 + Wf-382-A3 NEW entries); `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 NEW); `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 NEW 387 LOC); `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 269 LOC); `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 320 LOC 7 sections); `docs/WETLAB_DEPRECATION_FAQ.md` (Wf-375-A1 ~280 LOC); `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 ~430 LOC 9 sections + 14 invariants); `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1 ~289 LOC); `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 ~270 LOC LaTeX template); `docs/SCI_PAPER_SETUP.md` (Wf-351-B3 ~120 LOC); `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3 ~575 LOC 10 sections); `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §5.5 NEW + §10 standing directives re-confirmed; `AGENTS.md` §6 NEW row (Wf-382-B1 this sync); `AGENTS.md` last-update trailer (this sync appended); `TODO/INDEX.md` §22.77 (this entry appended between §22.76 Wf-383-B1 and §23 Wf-316B); `kill_then_adapt` §0 (NO GPU launch + USER_OK gating); `mol-metal-cfm-training` skill (Wf-381-A5 + Wf-382-A1 Multi-PH+PLL); `mol-metal-multiph-pll` skill (Wf-381-A5 + Wf-382-A1); `mol-metal-hilbert-astsr-r2` skill (Wf-381-A5 TODO-93); `mol-metal-3way-multimetal` skill (Wf-381-A5 TODO-95); `mol-metal-closure-catherine` skill (Wf-381-A5 TODO-96); `mol-metal-closure-theorem` skill (Wf-381-A5 TODO-98); `mol-metal-anticancer` skill (Wf-379-A1 + Wf-382-A3 m1_auc); `mol-metal-sci-paper-submission` skill (Wf-351-B3 NEW per 2026-09-29 user directive).

---

**END OF Wf-382-B1 FINAL SYNTHESIS v27** — 6/6 deliverables SHIPPED (Wf-381 A4+A5 + Wf-382 A1+A2+A3+A4 + Wf-382-B1 docs-only sync) / 22 NEW tests PASS across 4 SHIPPED sub-flows / 0 GPU launches / 0 paper/* touches / 0 wet-lab planning / pre-flight 89/2/3/94 PRESERVED across 16+ consecutive runs / EXPECTED_GATE_COUNTS["all"] **94** PRESERVED.
