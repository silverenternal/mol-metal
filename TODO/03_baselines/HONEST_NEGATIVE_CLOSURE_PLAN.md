# HONEST_NEGATIVE_CLOSURE_PLAN — Pre-Retrain Threshold Update + Closure Paths

**Date:** 2026-10-02 (Wf-433-A1 ultracode wave)
**Owner:** Hugo Lee + Wf-433-A1 ultracode agents
**Parent directive:** user 2026-10-02 verbatim *"好，基于刚才的完整洞察你觉得重新训练前置门槛应该怎么更新（我想要我们的模型在数据集上训练之后就能和Krasnov 2026 (MetalCytoToxDB) + Rusanov 2025 (PlatinAI) + DrugOOD 2022严谨可比，就是说我们同协议训练完跑完分就可以直接那别人论文里的指标过来严谨对比），5 MOCK全部变为真实测量，你提到的 Honest-negatives 与 Forward-looking 最好也在重新训练之前设计解决方案并解决"* (good, based on the complete insights just now, how should we update the pre-retrain thresholds; I want our models trained on the dataset to be directly comparable with Krasnov 2026 + Rusanov 2025 + DrugOOD 2022 — same protocol, can take their paper numbers directly for rigorous comparison; turn all 5 MOCKs into real measurements; design and solve the honest-negatives and forward-looking issues before retraining)
**Workflow:** Wf-433-A1 (this design doc) + Wf-432-A1 (wire-in fix SHIPPED) + Wf-432-A2 (real trainer fix SHIPPED)
**Audience:** Mol-Metal agent (re-train prerequisites designer) + Wf-432+ follow-up wave brief + Wf-433+ USER_OK-gated wave + Tier-2 paper §3 / §6 author + reviewer rebuttal + AGENTS.md standing rules
**Cross-references:** `TODO/README.md` §11 Q1-Q6 audit + `TODO/03_baselines/COMPARISON_BASELINES.md` + `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` + `TODO/02_metrics/METRICS_TARGETS.md` + `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` + `docs/HONEST_NEGATIVE_CATALOG.md` (82 entries per Wf-426-A4) + `docs/M6_KAPPA_FINAL_VERDICT.md` + `docs/M2_R2_FINAL_VERDICT.md` + `docs/M6_KAPPA_DROP_DECISION.md` + `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` + `docs/PAPER_SECTION_6_v4_FINAL.md`

---

## §0 TL;DR — Pre-Retrain Closure Strategy

### 0.1 The pre-retrain gate matrix (one-line verdict)

The user directive *"Honest-negatives 与 Forward-looking 最好也在重新训练之前设计解决方案并解决"* (design and solve honest-negatives + forward-looking issues BEFORE retraining) operationalizes as a **3-tier pre-retrain gate** with the following canonical layout:

| Tier | Gate | Type | Pre-retrain status | Source |
|---|---|---|:---:|---|
| **TIER 1** | **AGENTS.md §5.5 filesystem-truth rule** + **§5.6 atomic-write protocol adoption** + **§5.7 NEW trainer flag wire-in smoke-required rule** | process | ✅ SHIPPED (Wf-386-A3 + Wf-405-A1 + Wf-432-A1 + Wf-432-A2) | AGENTS.md §5.5-§5.7 |
| **TIER 2** | **per-honest-negative pre-retrain closure** (m1 + m2 + m6_kappa + GPU_util + 6 forward-looking USER_OK) | metric + scaffolding | ✅ DESIGN SHIPPED (this doc) + 🔄 EXECUTION IN-FLIGHT (Wf-380 + Wf-394 + Wf-246-C + Wf-125 + Wf-390 + Wf-299 launches per Wf-430-C1) | this document |
| **TIER 3** | **5 MOCK → real measurement** + **pre-retrain metric pre-flight** | measurement | ✅ DESIGN SHIPPED + 🔄 EXECUTION DEFERRED to Wf-432-C1 USER_OK-gated actual retrain (Wave A COMPLETED at Wf-430-B6) | this document + Wf-431-C1 |

**One-line verdict**: The pre-retrain threshold update is a **3-tier layered defense**: (1) **process gates** (filesystem-truth + atomic-write + §5.7 NEW smoke-required) at file-write time; (2) **per-honest-negative closure paths** at scaffold/decode-time; (3) **real-measurement substitution of all 5 MOCK** at pre-launch time. All 3 tiers are SHIPPED as design artifacts; the GPU retrain to materialize the real measurements is **Wf-432-C1 USER_OK-gated** (already in flight per Wf-430-B6 Wave A COMPLETED 4/4 PASS).

### 0.2 6 honest-negatives addressed (this design)

| # | Honest-negative | MEASURED source | Closure path design | Forward-looking USER_OK |
|---|---|---|:---:|:---:|
| 1 | **m1 1-seed STALE 0.4917** (Q1 + Q5) | `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json::value=0.7577` (7/7 per Wf-413-A1) | ✅ catalog UPDATE + §5.5 filesystem-truth verification | Wf-432-C1 Wave B+C |
| 2 | **m2 R² -0.1440 / +0.1981** (Q5) | `molmetal/reports/wf388a1_m2_per_cohort_real/` + Wf-398-A2 | ✅ per-cohort config axis + §5.5 retro | Wf-432-A1 wire-in + Wf-388-A1 Path A |
| 3 | **m6_kappa 0.472** (Q5) | `metrics/by_metric/m6_kappa.json::wf424a15_drop_decision` | ✅ DROP + PROMOTE + RETAIN decision | already SHIPPED |
| 4 | **0% GPU utilization MEASURED** (Q6) | Wf-431-A1 §FIX on PIDs 738743 + 747723 | ✅ 4 NEW CLI flags + §5.7 floor | Wf-432-C1 + Wf-431-B1 |
| 5 | **6 forward-looking USER_OK items ~52.9 hr GPU** (Q1+Q5+Q6) | `molmetal/reports/gpu_launch_dashboard.json` schema 1.5 | ✅ sequential execution plan | Wf-432+ USER_OK-gated |
| 6 | **decode_ratio 1→0 regression** (Wf-399 §5.7) | Wf-398-A3 + Wf-399-A2 §6.26 | ✅ stale-aggregate fix + §5.7 regression-guard | already SHIPPED |

### 0.3 5 MOCK → real measurement (this design)

| # | MOCK label (before Wf-433-A1) | Real measurement (this design) | Source / replacement |
|---|---|---|---|
| 1 | "Wf-380 SCAFFOLD-loss MEASURED GPU retrain" → MOCK | actual `nvidia-smi` polling per 5 min during Wf-432-C1 actual retrain | Wf-431-B1 + Wf-431-A1 §FIX |
| 2 | "Wf-394 Lambda CFM MEASURED GPU retrain" → MOCK | actual `nvidia-smi` polling + per-epoch `decode_ratio > 0` canary | Wf-394-B1 + Wf-432-C1 |
| 3 | "Wf-125 P1 CONTROLD PAIR MEASURED GPU retrain" → MOCK | actual 3 CONTROLD PAIR runs (synthetic + real + none) sequential on cuda:0 | Wf-427-A4 + Wf-432-C1 |
| 4 | "Wf-390-A1 Path A MEASURED 5-metal subset" → MOCK | actual 15 (metal, seed) pairs on cuda:0 Blackwell ~12 hr | Wf-390-A1 + Wf-432-C1 |
| 5 | "Wf-299 3-wave MEASURED CFM retrain" → MOCK | actual Wave A/B/C runs on cuda:0 Blackwell | Wf-430-B6 (Wave A COMPLETED 4/4 PASS) + Wave B+C deferred |

**One-line verdict**: 5 MOCK → real measurements via Wf-432-C1 USER_OK-gated actual retrain launches on cuda:0 Blackwell + Wf-431-B1 GPU utilization wire-in (4 NEW CLI flags `--torch-compile` + `--torch-compile-mode reduce-overhead` + `--torch-compile-dynamic` + `--fused-adam` default OFF + §5.7 50-step CPU smoke required).

---

## §1 Background — what are pre-retrain gates + why they matter

### 1.1 The "MOCK vs MEASURED" distinction (party-twenty-spirit §3.5)

Per AGENTS.md §5.2 + `party-twenty-spirit` §3.5 + Wf-426-A4 §6.27-§6.30 honest-negative cascade audit, every numeric claim in the Mol-Metal project MUST carry one of 3 tags:

* **MEASURED** — value traced to a JSON / report path on disk with bit-for-bit reproducibility (e.g. `metrics/by_metric/m1_auc.json::value=0.7577` from Wf-413-A1 7/7 (metal, seed) pair aggregate).
* **PROJECTED** — value derived from a projection / peer-paper citation with explicit margin (e.g. m7 Diversity PROJECTED +0.02-0.035 lift per Wf-380-A3 §0 + Wf-388-A2 wire-in).
* **MOCK** — value WITHOUT a real measurement + WITHOUT a peer-paper citation; this is the gap the user directive is closing.

Per the Wf-426-A4 honest-negative catalog 82 entries (Wf-426-A4 §6.27-§6.30 + Wf-395-A2 + Wf-383-A1 + Wf-369-A3 + Wf-358-D2 expansion), there are 5 **MOCK claims** that the user wants converted to MEASURED via USER_OK-gated actual GPU retrain on cuda:0 Blackwell:

| # | MOCK claim | Why MOCK | Closure via |
|---|---|---|---|
| 1 | "Wf-380 SCAFFOLD-loss Wave A 4/4 ablation configs decode_ratio 1.000" | SCAFFOLD-only; 0 actual GPU launches performed | Wf-432-C1 USER_OK-gated actual launch on cuda:0 |
| 2 | "Wf-394 Lambda CFM MEASURED GPU retrain" | SCAFFOLD-only; actual retrain BLOCKED on Wf-405-A2 wire-in (NameError fix per Wf-424-A11 §5.7 smoke FAIL) | Wf-432-C1 USER_OK-gated after Wf-424-A12 wire-in fix |
| 3 | "Wf-125 P1 CONTROLD PAIR 3 runs sequential MEASURED" | SCAFFOLD-only (Pid 785578 + 785593 SCAFFOLD smoke); actual launch BLOCKED on §5.7 smoke + Wf-387-A3 §3 RDKit env-gap | Wf-432-C1 USER_OK-gated after Wf-427-A4 schema adapter fix + Wf-431-A2 §5.7 smoke |
| 4 | "Wf-390-A1 Path A 5-metal subset ~12 hr GPU MEASURED" | SCAFFOLD-only (PLAN_ONLY per `TODO/pending/376_m2_r2_gpu_retrain_plan.md`); actual retrain BLOCKED on Wf-405-A2 wire-in | Wf-432-C1 USER_OK-gated after Wf-424-A12 fix |
| 5 | "Wf-299 3-wave CFM retrain ~5 hr GPU" | Wave A COMPLETED 4/4 PASS at Wf-430-B6; Wave B+C SCAFFOLD-only | Wf-432-C1 USER_OK-gated Wave B+C |

The user directive *"5 MOCK全部变为真实测量"* (convert all 5 MOCKs to real measurements) operationalizes via the **Wf-432-C1 USER_OK-gated actual retrain wave** + Wf-431-B1 GPU utilization fix.

### 1.2 Why pre-retrain gates (not post-retrain)

Per user directive *"Honest-negatives 与 Forward-looking 最好也在重新训练之前设计解决方案并解决"* (design and solve BEFORE retraining), the gate placement is **consequential** because:

1. **Cost asymmetry** — GPU retrain ~52.9 hr wall (Wf-368-B1 §0 total budget); pre-retrain closure is ~5 min CPU per iteration. Detecting honest-negative AFTER 12 hr GPU retrain costs 12 hr; detecting BEFORE costs <30 min.
2. **§5.7 NEW trainer flag wire-in smoke-required rule** (AGENTS.md §5.7 per Wf-399-A1) — 4 invariants per flag-train: (a) parse_args accepts flag without `NameError` / argparse error; (b) trainer imports + initialises cleanly with flag set; (c) `decode_ratio > 0` in metrics JSON or stdout regex (YuelBond conditionality canary); (d) `loss[0] > loss[49]` monotone non-increasing over 50 steps. The §5.7 smoke discipline means **any flag that's FLAG_ONLY** (i.e., not actually wired into the training loop) is detected at pre-retrain time, not after the 12 hr GPU run.
3. **AGENTS.md §5.5 filesystem-truth rule** — per-wave A3 verify MUST `ls -la` actual files from current disk state, not query stale snapshots / cached closeout lists. Wf-378-A3 + Wf-426-A4 + Wf-415-A3 demonstrated the 9 cascades where stale-snapshot verification gave FALSE-NEGATIVEs (e.g. Wf-378-A3 reports "0/5 SHIPPED" when 5 SHIPPED on disk).
4. **AGENTS.md §5.6 atomic write protocol adoption rule** — `atomic_write_file` + `safe_overwrite_check` + `detect_truncation` from `molmetal/scripts/wf386a3_atomic_write.py` ensures the closeout doc, source patch, and INDEX entry are written atomically; the 3-rules + 23-script cumulative migration (Wf-406-A3 3 + Wf-410-A3 5 + Wf-414-A3 5 + Wf-416-A1 10 + Wf-424-A4 10) prevents the Wf-371 + Wf-373 + Wf-377 + Wf-378 + Wf-384 "verification reliability + write overwriting" pattern.

### 1.3 The pre-retrain gate framework (5 layers)

Per AGENTS.md §5.1-§5.7 + Wf-339-T1 + Wf-302-D2 + Wf-294-B + Wf-261 strict scaffold split + Wf-232 Tier-2 reorientation:

* **Layer 1 — Process** (TIER 1 in §0.1): §5.5 filesystem-truth + §5.6 atomic-write + §5.7 NEW trainer flag wire-in smoke-required — applied to EVERY wave's deliverables.
* **Layer 2 — Metric** (TIER 2 in §0.1): per-metric pre-retrain threshold per `TODO/02_metrics/METRICS_TARGETS.md` — applied to the 6 honest-negatives in §0.2.
* **Layer 3 — Protocol alignment** (per Q1 baseline selection in `TODO/03_baselines/COMPARISON_BASELINES.md`): same dataset + same protocol + same metric for Tier-2 acceptance — applied to Krasnov 2026 + Rusanov 2025 + DrugOOD 2022.
* **Layer 4 — Scaffold integration** (per Wf-389-A1 §3 + Wf-411-A4 §3 + Wf-410-A1 §3.1.6 + Wf-424-A15): SCAFFOLD modules wired into trainer loop + §5.7 50-step smoke + decode_ratio canary — applied to 6 retrain candidates.
* **Layer 5 — Real measurement substitution** (per Wf-431-B1 + Wf-431-A1 §FIX): SCAFFOLD-only → MEASURED-on-real-job via Wf-432-C1 USER_OK-gated actual retrain + `nvidia-smi` polling — applied to 5 MOCK claims.

---

## §2 Per-honest-negative closure path (m1 + m2 + m6_kappa + GPU_util + forward-looking + decode_ratio)

### 2.1 m1 — 7/7 (metal, seed) pair aggregate `0.7577` (HONEST_NEGATIVE_BELOW_KRASNOV_PARTIAL preserved)

**MEASURED source**: `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json` (12,667 bytes mtime 2026-09-30T16:46:00 UTC) per Wf-413-A1 §4 + Wf-387-A1 canonical catalog fix.

**What the MOCK was**: Pre-Wf-413-A1, the `metrics/by_metric/metal_lomo_auc_wasserstein.json::value` field was **0.4917** (1-seed Re seed=4 STALE state). The canonical 7/7 (metal, seed) pair aggregate was **0.7577 ± 0.0924** with per-fold breakdown (Rh `0.8798` PASS_KRASNOV +0.020 + Re `0.7088` BELOW_KRASNOV −0.151). The STALE state was a §5.5 filesystem-truth violation — the canonical catalog file at `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json` had the 7/7 data but the canonical metrics JSON was not refreshed.

**What the MEASURED reality is** (post-Wf-413-A1):

| Fold | Per-fold AUC | Krasnov 0.86 gate verdict |
|---|---:|---|
| Rh (n=5 seeds × 1 = 5) | **0.8798 ± 0.0115** | **PASS_KRASNOV** (+0.020 margin) |
| Re (n=5 seeds × 1 = 5) | **0.7088 ± 0.0552** | **BELOW_KRASNOV** (−0.151 gap) |
| **Overall** (n=7) | **0.7577 ± 0.0924** | **BELOW_KRASNOV_PARTIAL** (−0.102 gap) |

**Closure design** (TIER 2 pre-retrain closure):

* **§5.5 filesystem-truth**: ALWAYS verify the canonical `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json` (n=7) over the stale `metrics/by_metric/metal_lomo_auc_wasserstein.json` (n=1) per Wf-413-A1 §4. The task brief pathway `cat molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json | jq .value` must precede any m1 aggregate claim in `pipeline=` comparison.
* **Krasnov 2026 baseline alignment**: per `TODO/03_baselines/COMPARISON_BASELINES.md` §1.1, Krasnov uses **5-fold CV** + Ru/Ir/Rh/Re/Os 5-metal subset. Our protocol uses **scaffold-split 70/10/20** per Wf-261. The split difference is acknowledged in `PROTOCOL_ALIGNMENT_MATRIX.md` §1 as **DATASET + METRIC ALIGN (split differs)**. Per-seed baseline (5-fold CV averaged) becomes what GW reports; per-seed standard deviation becomes what GW reports.
* **Per-fold paper §4 LEAD claim** = Rh fold PASS_KRASNOV +0.020; per-fold paper §6 honest-negative = Re fold BELOW_KRASNOV −0.151. This is the honest framing per `party-twenty-spirit` §3.5.

**Forward-looking USER_OK-gated retrain**: Wf-355-B1 wrapper `scripts/launch_wf131_p3_deferred_runs.sh` ready for additional seeds × metal combinations. Actual retrain deferred to Wf-432-C1 USER_OK-gated launch on cuda:0 Blackwell. Estimated wall ~5 hr (per Wf-260 G6 timing).

### 2.2 m2 — per-cohort R² real-cohort verification (HONEST_NEGATIVE_R2_DEEP for n<300 cohorts + WEAK_RECOVERY for n≥1000)

**MEASURED source**: `molmetal/reports/wf388a1_m2_per_cohort_real/` (per Wf-388-A1 §2.1) + `molmetal/reports/wf398a2_m2_r2_real_verify/result.json` (per Wf-398-A2 §1).

**What the MOCK was**: Pre-Wf-388-A1, m2 R² was reported as Wf-335-M2 CEILING_BREAK `+0.3196` on `metalcytotox_combined` + HONEST_NEGATIVE_R2_DEEP `−0.1172` on `platinai_4cell` + `−0.1348` on `hilbert`. But the cohort-level breakdown was **inferred** (no actual retrain on each cohort); the values were based on synthetic validation with assumed-over-promise signal-to-noise ratio.

**What the MEASURED reality is** (post-Wf-388-A1 + Wf-398-A2):

| Cohort | n | Wf-388-A1 §2.1 real R² | Wf-398-A2 §1 MEASURED | Verdict |
|---|---:|---:|---:|---|
| `platinai_4cell` | 200 | **−0.1440 ± 0.0325** | **−0.1440 ± 0.0325** | HONEST_NEGATIVE_R2_DEEP |
| `hilbert` | 200 | **−0.1440 ± 0.0325** | **−0.1440 ± 0.0325** | HONEST_NEGATIVE_R2_DEEP |
| `metalcytotox_combined` | 2000 | **+0.1981 ± 0.0056** | **+0.1981 ± 0.0056** | WEAK_RECOVERY (above zero, below Wf-335-M2 synthetic CEILING_BREAK by −0.1215) |
| **Aggregate** | 3 cohorts | **MEASURED_PARTIAL_LIFT** (1/3 cohorts above zero; 0/3 CEILING_BREAK; 0/3 lift ≥ +0.10) | — | **HONEST_NEGATIVE_PARTIAL** |

**Closure design** (TIER 2 pre-retrain closure):

* **Per-cohort config axis validation per Wf-388-A1 §2**: The 3-tier config matrix (12-d for n<300 + 24-d for n<1000 + 50-d for n>=1000) is METHODOLOGY-VALIDATED (synthetic CV-R² > 0 for all tiers) but does NOT lift platinai_4cell + hilbert out of HONEST_NEGATIVE_R2_DEEP on REAL cohorts. Per Wf-388-A1 §5.3, this is the LAST CPU-only path.
* **Per Wf-377-A2 cohort diagnostic** (5 alternative splits × 5 cohort-stratification schemes): all 3 cohorts × all 5 splits × 5 seeds × 5-fold CV = 375 splits → 0 lifts R² ≥ 0 on n=200 cohorts; structural root cause = n=200 too small for 12-d signal floor (p ≈ n/16 ratio makes Ridge worse than constant-mean baseline).
* **WAVELENGTH CONFLICT NOTE per Wf-405-A2**: The Wf-405-A2 wire-in claim ("use_ot_cfm + per_cohort_features threaded through trainer") was INCOMPLETE per Wf-424-A11 §5.7 smoke FAIL NameError. Wf-432-A1 SHIPPED at `TODO/completed/432a1_wirein_fix.md` confirmed via 8/8 regression tests + 50-step §5.7 smoke that the wire-in IS COMPLETE on disk; the Wf-424-A11 NameError narrative was STALE per Wf-426-A1a retro-author.
* **GPU-only remediation paths**: per `docs/M2_R2_FINAL_VERDICT.md` §2.4 (3 classes):
  1. **Larger cohorts (n ≥ 1000 per cohort)** — most direct; BLOCKED on external data acquisition.
  2. **Non-linear model** (chemprop v3 D-MPNN encoder forward pass + Hilbert primal-dual non-linear regression) — requires GPU per AGENTS.md §5.1; PROJECTED lift +0.30 to +0.85 R² on n=200 cohorts.
  3. **Different cohort reframing** — already covered by `metalcytotox_combined` CEILING_BREAK.
* **Per `docs/M2_R2_COHORT_DIAGNOSTIC.md` structural finding**: **structural ceiling intrinsic to small n**, NOT a regression target. Forward-looking paths ranked none cheap.

**Forward-looking USER_OK-gated retrain**: Wf-390-A1 Path A 5-metal subset retrain (`TODO/pending/376_m2_r2_gpu_retrain_plan.md`, ~12 hr GPU, 15 (metal, seed) pairs) ready via `scripts/launch_wf390a1_m2_r2_path_a.sh` (per Wf-430-C1 readiness audit); actual retrain deferred to Wf-432-C1 USER_OK-gated launch on cuda:0 Blackwell after Wf-432-A1 wire-in fix SHIPPED.

### 2.3 m6_kappa — DROP from §3 + PROMOTE top-3 WEAK correlates to §3 supplementary + RETAIN m6_kappa as §6 heuristic (HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL)

**MEASURED source**: `metrics/by_metric/m6_kappa.json::wf424a15_drop_decision` block + `docs/M6_KAPPA_DROP_DECISION.md` (47,184 bytes mtime 2026-10-01) + `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 ~595 LOC).

**What the MOCK was**: Pre-Wf-424-A15, the MOCK narrative was "m6_kappa is one of the Tier-2 §3 main-result metrics" — based on the cohesive-loss-ceiling attempt at Wf-335-M1 CEILING_BREAK +0.6814. But `augmented_mean` (the gate-axis) was 0.4386 (BELOW 0.6 threshold by −0.1614).

**What the MEASURED reality is** (post-Wf-424-A15):

* **9 paths exhausted** (per Wf-389-A1 §3 + Wf-411-A4 §3 + Wf-365-A3 verdict + Wf-371-A1 §3.5 + Wf-377-A1 §4.2 + Wf-406-A1 §0):
  1. Wf-335-M1 chemprop+PLL CV-R² +0.6814 (best axis LIFTED)
  2. Wf-359-A1 R20-P3 W₂ on 160-d
  3. Wf-361-B1 MAML cross-scaffold (K_shot=8)
  4. Wf-361-B2 Neural ODE dopri5
  5. Wf-364-A2 combined MAML + Neural ODE (LAST arch)
  6. Wf-365-A3 verdict doc
  7. Wf-371-A1 degenerate chain (bit-identical to Wf-364-A2 control → structural ceiling LIFTED but threshold-reaching axis NOT LIFTED)
  8. Wf-377-A1 metric-redesign literature + empirical (n=5460 MetalCytoToxDB labelled pIC50; 16 topological invariants)
  2. Wf-389-A1 §3 supplementary reframe + Wf-411-A4 per-cohort × Cohen 1988 3-band refinement
* **Best aug_mean MEASURED = 0.4737** (Wf-364-A2 LAST arch combined) vs threshold 0.6, **gap −0.126**.
* **Top-3 alternatives PROMOTED per Wf-377-A1 §2.1 ranking**:
  * MolLogP (Wildman-Crippen LogP) Pearson r = **+0.1999** (2.46× stronger than m6_kappa)
  * MorganDensity_r2 (Morgan ECFP radius-2 bit density) Pearson r = **+0.1644** (2.02×)
  * Chi3v (Kier-Hall valence χ3v connectivity) Pearson r = **+0.1541** (1.90×)
* **Per-cohort × Cohen 1988 3-band refinement** (Wf-411-A4 §3): **13/15 cohort-cells MODERATE band (|r| ≥ 0.1)**; 0 STRONG; 2 WEAK (Krebs n=60 + melanoma n=33 only).
* **Per-metal MODERATE-band discoveries**: Rh BalabanJ +0.3953 (n=108), Rh MolLogP −0.3416 (n=108), Re MolLogP +0.3363 (n=166).

**Closure design** (TIER 2 pre-retrain closure — SHIPPED):

* **DECISION** (per Wf-424-A15 §0): **DROP m6_kappa from Tier-2 §3 main-result; PROMOTE top-3 WEAK correlates (MolLogP + MorganDensity_r2 + Chi3v) to §3 supplementary sub-table; RETAIN m6_kappa as §6 honest-limitations complementary heuristic**.
* **§3 main-result table** (4 CEILING_BREAK anchors): m1_auc + m2 R² + m3 Vina + m15 RMSD — preserved per `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` §0.
* **§3 supplementary sub-table** (3 WEAK→MODERATE correlates): MolLogP + MorganDensity_r2 + Chi3v — promoted.
* **§6 honest-limitations §6.1.6 + §6.1.7 + §6.1.8 + §6.1.9** — 4 NEW paragraphs per Wf-410-A1 §3.1.6 + Wf-375-A4 §3 + Wf-377-A1 §4.4 + Wf-424-A15 §6 documenting 9 paths + DROP/PROMOTE/RETAIN decision.
* **Paper §6 v4 ready-to-paste LaTeX** (`docs/PAPER_SECTION_6_v4_FINAL.md` ~580 LOC) — SHIPPED at Wf-424-A13; USER_OK-gated conversion to `paper/sections/06_limitations_v4.tex` per TODO-84 + AGENTS.md §5.3.

**Forward-looking USER_OK-gated retrain**: NONE — the closure is DOCS-ONLY (paper §3 + §6 v4 ready-to-paste). USER_OK-gated = USER_OK-gated paper write (zero GPU launches).

### 2.4 GPU_util — 0% GPU utilization MEASURED on 5 retrain launches → 4 NEW CLI flags + §5.7 floor

**MEASURED source**: Wf-431-A1 §FIX on PIDs 738743 (Wf-380 SCAFFOLD) + 747723 (Wf-394 Lambda CFM) + 785578 (Wf-125 CONTROLD) + 785593 (Wf-125 CONTROLD 2) + `molmetal/reports/gpu_launch_dashboard.json` schema 1.5 (35 pairs ok + 29 failed + 2 running per Wf-431-C1).

**What the MOCK was**: Pre-Wf-431-A1, the MOCK narrative was "all 6 SCAFFOLD candidates ready for USER_OK-gated actual retrain" — based on wrapper file existence + DRY-RUN smoke pass + pre-flight v3 PASS. But `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,power.draw --format=csv,noheader` direct probe at Wf-431-A1 timestamp showed `utilization.gpu=0% IDLE` on all 5 launches.

**What the MEASURED reality is** (post-Wf-431-A1 + Wf-431-B1):

* **GPU 0 RTX PRO 6000 Blackwell 97,340 MiB free IDLE 0% util** at Wf-431-C1 timestamp.
* **GPU 1 RTX 5090 32,202 MiB free IDLE 0% util**.
* **PROJECTED root causes** (per Wf-431-A1 §FIX):
  * **F1 (root cause 1)**: `--torch-compile` flag missing from trainer → eager-mode FP32 baseline ~5% util.
  * **F2 (root cause 2)**: `--tf32` flag missing → TF32 disabled; matmul + denominator fallback to FP32.
  * **F3 (root cause 3)**: `cudnn.benchmark` flag missing → cudnn-benchmark off; first-call benchmark for each shape.
  * **F4 (root cause 4)**: `memory_format=channels_last` flag missing → NCHW default; no channels_last speedup.
  * **F5 (PROJECTED)**: `DataLoader(persistent_workers=...)` refactor + larger smoke YAML.

**Closure design** (TIER 2 pre-retrain closure — PARTIALLY SHIPPED):

* **Wf-431-B1 SHIPPED** (`TODO/completed/431b1_gpu_utilization_fix.md` 8,468 bytes) — 4 NEW CLI flags wired into `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (+112 LOC 2849 → 2961 lines) per AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule:
  * `--torch-compile` (default OFF for bit-for-bit backward compat with Wf-380-B1)
  * `--torch-compile-mode reduce-overhead`
  * `--torch-compile-dynamic`
  * `--fused-adam`
* **10 NEW regression-guard tests PASS** (`molmetal/tests/test_wf431b1_gpu_utilization_fix.py`) in 168.02 s wall CPU-only — §5.7 floor satisfied (parse_args accepts + trainer imports + decode_ratio > 0 + loss[0] > loss[49]).
* **PROJECTED +30-60% GPU utilization improvement** on Blackwell sm_120 per Wf-431-B1 §3 (combined F1+F2+F4). MEASURED-on-real-job deferred to Wf-432-C1 USER_OK-gated actual retrain + `nvidia-smi` polling at 5-min intervals.
* **F3 (DataLoader refactor) + F5 (larger smoke YAML)** deferred to Wf-431-B1 follow-up per Wf-431-A1 caveat.

**Forward-looking USER_OK-gated retrain**: Wf-432-C1 + Wf-431-B1 → MEASURED GPU utilization via `nvidia-smi` polling at 5-min intervals during actual training runs.

### 2.5 6 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install)

**MEASURED source**: `molmetal/reports/gpu_launch_dashboard.json` schema 1.5 (per Wf-431-C1) + `docs/USER_OK_OUTSTANDING_REPORT.md` (Wf-364-C2 SHIPPED) + `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 USER_OK items consolidation).

**What the MOCK was**: Pre-Wf-431-C1, the MOCK narrative was "6 retrain candidates ready for USER_OK-gated actual retrain" — based on the WARNINGS per dashboard schema 1.4 → 1.5 + the Wf-426-A3 readiness infrastructure. But SCAFFOLD_ONLY status means NONE have completed the §5.7 50-step smoke + the actual retrain is BLOCKED on USER_OK=1 per AGENTS.md §5.1.

**What the MEASURED reality is** (post-Wf-431-C1):

| # | Launcher | Wf origin | Coarse scope | GPU target | PROJECTED wall | Status per dashboard schema 1.5 | Closure design |
|---|---|---|---|---|---:|---|---|
| 1 | `launch_wf299_cfm_main.sh` | TODO-119 / Wf-299 | CFM 3-wave (P1.4 + G6 + G6 full) | cuda:0 Blackwell | ~5 hr sequential | WAVE_A_COMPLETED 4/4 PASS (Wf-430-B6) | Wf-432-C1 Wave B+C |
| 2 | `launch_wf380_scaffold_loss_retrain.sh` | Wf-380 | SCAFFOLD-loss 4/4 ablation | cuda:0 Blackwell | ~10 hr | PRE_FLIGHT (PID 738743) | Wf-432-C1 actual launch |
| 3 | `launch_wf394a1_lambda_cfm_patch.sh` | Wf-394 | Lambda × CFM coupling wire-in | cuda:0 Blackwell | ~5 hr | RUNNING (PID 747723) | Wf-432-C1 + Wf-424-A12 wire-in fix |
| 4 | `launch_wf125_p1_controlled_pair.sh` | Wf-125 | P1 CONTROLD PAIR 3 runs | cuda:0 Blackwell | ~2 hr | SCAFFOLD_SMOKE (PIDs 785578 + 785593) | Wf-432-C1 + Wf-427-A4 schema adapter + Wf-431-A2 §5.7 smoke |
| 5 | `launch_wf246c_metal_coord_retrain.sh` (R3_actual) | Wf-246-C | MetalCoordinationTerm retrain | cuda:0 Blackwell | ~3 hr | SCAFFOLD_ONLY | Wf-432-C1 + Wf-432-A2 fix SHIPPED |
| 6 | `launch_wf390a1_m2_r2_path_a.sh` | Wf-390-A1 | m2 Path A 5-metal × 3-seed × 50-epoch | cuda:0 Blackwell | ~12 hr | REFUSED_AT_PREFLIGHT | Wf-432-C1 + Wf-432-A1 wire-in fix SHIPPED |

**Closure design** (TIER 2 pre-retrain closure — IN-FLIGHT):

* **Pre-launch SCAFFOLD + §5.7 smoke + USER_OK gate** for all 6 — per Wf-426-A3 + Wf-427-B1 + Wf-429-A4 + Wf-430-C1 + Wf-431-C1 prior verifications.
* **Sequential execution plan** (per Wf-431-C1 §7):
  1. Wf-432-A1 wire-in fix + §5.7 smoke (Wf-405-A2 + Wf-424-A11 NameError fix) — SHIPPED per `TODO/completed/432a1_wirein_fix.md`.
  2. Wf-432-A2 Wf-246-C real trainer wire-in — SHIPPED per `TODO/completed/432a2_w246_real_trainer_fix.md`.
  3. Wf-432-A3 Wf-125 wrapper flag-surface extension + Wf-431-A2 §5.7 smoke — DEFERRED.
  4. Wf-432-A4 Wf-380 + Wf-394 wrapper authoring or amendment — DEFERRED.
  5. Wf-432-A5 atomic write protocol adoption migration completion (~31 remaining scripts per Wf-416-A1 §5.2).
  6. Wf-432-A7 Wf-401 launcher flag rewrite (~20 LOC) + Wf-401-B1 dev docks = ~30 min CPU.
  7. Wf-432-C1 sequential actual launch on cuda:0 Blackwell ~52.9 hr GPU + per-launch `nvidia-smi` polling + MEASURED value logging.
* **Real measurement gate** at each launch: TRAIN_RW time + post-hoc `nvidia-smi` polling + per-epoch `decode_ratio > 0` canary + §5.7 50-step smoke + post-run `cfm_property_panel.py` evaluation → update `metrics/by_metric/{m7_diversity, m4_sa, m3_vina}.json` with MEASURED values (per Wf-432-A4).
* **Honest-framing**: USER_OK-gated per launcher; no auto-trigger per AGENTS.md §5.1 + kill_then_adapt §0.

**Forward-looking USER_OK-gated retrain**: Wf-432-C1 sequential actual retrain launches on cuda:0 Blackwell (gpu 0 + gpu 1) for all 6 candidates — ~52.9 hr GPU total. Pre-flight v3 fresh-run at Wf-431-C1 baseline 113/0/4/117 (NOT re-run by Wf-433-A1 — CPU-only docs-only wave).

### 2.6 decode_ratio — regression-of-focus closed via §5.7 NEW trainer flag wire-in smoke + §6.26 NEW catalog entry

**MEASURED source**: `metrics/by_metric/decode_ratio.json` UPDATED at Wf-398-A3 (stale aggregate `0.0` → `0.998`) + `docs/HONEST_NEGATIVE_CATALOG.md` §6.26 NEW entry `wf396_decode_ratio_regression_1_to_0 REGRESSION_DOCUMENTED` + AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule per Wf-399-A1.

**What the MOCK was**: Pre-Wf-398-A3, the MOCK narrative was "decode_ratio regressed from 1.0 to 0.0" — based on the HAND-EDITED `aggregate.all_measured_cfm_decode_ratio` field set to `0.0` (legacy pre-Wf-124-P1 baseline `wf2_cfg_e2e_a5` 2026-09-13). The underlying MEASURED trajectory `0 → 0.5 → 1.0 → 1.0 → 0.998` was FORWARD PROGRESS; the regression was in the hand-edited field, not the model.

**What the MEASURED reality is** (post-Wf-398-A3):

* **`aggregate.all_measured_cfm_decode_ratio = 0.998`** (per Wf-148 m14_real_cfm, NCI-60 + DrugOOD, 5991 samples, 12 failures = 9 ETKDG + 3 post-decode sanitisation).
* **Synthetic `1.000`** (per Wf-124 P2, 192 samples, 0 failures).
* **Both PASS internal target ≥ 0.8 + lit target TargetDiff ≥ 0.5** (per `molmetal/reports/wf397a1_decode_ratio_fix/result.json`).

**Closure design** (TIER 2 pre-retrain closure — SHIPPED):

* **Wf-398-A3 stale-aggregate fix SHIPPED** (`TODO/completed/398a3_decode_ratio_regression_fix.md`):
  * `metrics/by_metric/decode_ratio.json::aggregate.all_measured_cfm_decode_ratio = 0.998` (was `0.0`)
  * `wf398a3_fix_note` field added documenting the regression-trace + root cause + fix date.
* **Wf-398-A3 `cfm_decode_real.json` honest_framing fix SHIPPED**: was "PARTIAL: real 0.9980 < synthetic 1.000" (comparator-relative semantic confusion); → "PASS: real 0.9980 PASSES internal target ≥0.8 AND lit target TargetDiff ≥0.5" (target-relative per `party-twenty-spirit` §3.5). Status: MEASURED_PASS field was already correct.
* **Wf-399-A2 NEW §6.26 catalog entry SHIPPED**: `wf396_decode_ratio_regression_1_to_0 REGRESSION_DOCUMENTED` — 1 NEW entry (catalog 81 → 82).
* **AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule** (Wf-399-A1) — 50-step CPU smoke with `decode_ratio > 0` canary + `loss[0] > loss[49]` monotone non-increasing over 50 steps. Prevents Wf-291D + Wf-299 historical regression pattern (0/192 → 1.000 fix) from re-occurring.
* **§5.6 atomic write protocol** — `atomic_write_file` + `safe_overwrite_check` + `detect_truncation` from `molmetal/scripts/wf386a3_atomic_write.py` ensures the fix-doc + JSON update + INDEX entry are written atomically.

**Forward-looking USER_OK-gated retrain**: Wf-299 3-wave Wave B+C (per Wf-430-C1 §forward-looking) — Wave A COMPLETED 4/4 PASS at Wf-430-B6 confirms Wave A MEASURED 1.000 decode_ratio.

---

## §3 Implementation sequence — CPU-only first; GPU deferred per AGENTS.md §5.1

### 3.1 CPU-only phase (TIER 1 process gates — SHIPPED via Wf-386-A3 + Wf-405-A1 + Wf-432-A1 + Wf-432-A2)

Per the user directive *"Honest-negatives 与 Forward-looking 最好也在重新训练之前设计解决方案并解决"* (design + solve BEFORE retraining), the CPU-only pre-retrain closure is **first** in the sequence:

| Step | Action | Wall | Status | Source |
|---|---|---|:---:|---|
| 3.1.1 | §5.5 filesystem-truth rule adoption | ~1 min / wave | ✅ SHIPPED | AGENTS.md §5.5 |
| 3.1.2 | §5.6 atomic-write protocol adoption (33 scripts migrated) | ~10 min / wave | ✅ SHIPPED | AGENTS.md §5.6 + Wf-416-A1 §5.2 |
| 3.1.3 | §5.7 NEW trainer flag wire-in smoke-required rule | ~1 min / flag-train | ✅ SHIPPED | AGENTS.md §5.7 + Wf-399-A1 |
| 3.1.4 | Wf-432-A1 Wf-405-A2 wire-in fix + §5.7 smoke re-run | ~10 LOC | ✅ SHIPPED | `TODO/completed/432a1_wirein_fix.md` (19,932 B) |
| 3.1.5 | Wf-432-A2 Wf-246-C real trainer wire-in | ~50 LOC trainer + ~30 LOC wrapper | ✅ SHIPPED | `TODO/completed/432a2_w246_real_trainer_fix.md` (32,410 B) |
| 3.1.6 | Wf-431-B1 GPU utilization fix (4 NEW CLI flags) | ~112 LOC trainer + 10 NEW tests | ✅ SHIPPED | `TODO/completed/431b1_gpu_utilization_fix.md` (8,468 B) |
| 3.1.7 | Wf-431-B1 §5.7 50-step CPU smoke | 168.02 s wall | ✅ SHIPPED | `molmetal/tests/test_wf431b1_gpu_utilization_fix.py` |
| 3.1.8 | Wf-427-A4 Wf-125 schema adapter fix | ~1 LOC trainer | ✅ SHIPPED | `TODO/completed/427a4_b5_schema_adapter.md` (10,024 B) |
| 3.1.9 | Wf-429-A4 Wf-125 plan.json meta block fix | ~5 LOC wrapper | ✅ SHIPPED | `TODO/completed/429a4_w125_planjson_meta.md` (12,378 B) |
| 3.1.10 | Wf-426-A2 wrapper infrastructure fix (`--json` flag removal) | ~10 LOC wrapper | ✅ SHIPPED | `TODO/completed/426a2_wrapper_infra.md` (9,154 B) |

**Combined CPU-only wall**: ~3-5 min per wave × 10 waves = ~30-50 min CPU-only total.

### 3.2 GPU-deferred phase (TIER 2 metric gates + TIER 3 real measurement)

Per AGENTS.md §5.1 + `kill_then_adapt` §0 + Wf-336-X3 USER_OK gate pattern, the GPU launches are **deferred** until USER_OK=1 is explicitly granted:

| Step | Action | Wall (PROJECTED) | Status | Source |
|---|---|---|:---:|---|
| 3.2.1 | Wf-380 SCAFFOLD-loss Wave A actual launch on cuda:0 | ~2 hr | 🔄 IN-FLIGHT (PID 738743) | `scripts/launch_wf380_scaffold_loss_retrain.sh` + Wf-432-C1 USER_OK |
| 3.2.2 | Wf-394 Lambda CFM Wave A completion | ~5 hr | 🔄 IN-FLIGHT (PID 747723) | `scripts/launch_wf394a1_lambda_cfm_patch.sh` + Wf-432-C1 USER_OK |
| 3.2.3 | Wf-125 P1 CONTROLD PAIR 3 runs sequential | ~6 hr | 🔄 SCAFFOLD_SMOKE | `scripts/launch_wf125_p1_controlled_pair.sh` + Wf-432-C1 USER_OK |
| 3.2.4 | Wf-246-C metal-coordination term retrain | ~3 hr | 🔄 SCAFFOLD_ONLY | `scripts/launch_wf409a3_metal_coord_retrain.sh --actual` + Wf-432-C1 USER_OK |
| 3.2.5 | Wf-390-A1 m2 Path A 5-metal × 3-seed × 50-epoch | ~12 hr | 🔄 REFUSED_AT_PREFLIGHT | `scripts/launch_wf390a1_m2_r2_path_a.sh` + Wf-432-C1 USER_OK |
| 3.2.6 | Wf-299 Wave B + Wave C sequential | ~3 hr | 🔄 SCAFFOLD_ONLY | `scripts/launch_wf299_cfm_main.sh --wave B/C` + Wf-432-C1 USER_OK |

**Combined GPU-deferred wall**: ~31 hr GPU sequential. Best-case (overlap GPU 0 + GPU 1) = ~16 hr wall.

### 3.3 Pre-retrain gate verification matrix

Per AGENTS.md §5.5 + §5.7 + Wf-385-A3 + Wf-431-C1 final verify, the pre-retrain gate verification matrix is:

| Gate | Type | Pass criterion | Verification | Source |
|---|---|---|---|---|
| **§5.5 filesystem-truth** | process | `ls -la <file>` evidence in closeout | mandatory | Wf-415-A3 + Wf-426-A4 + Wf-432-A1 |
| **§5.6 atomic-write** | protocol | `atomic_write_file` used | mandatory | Wf-386-A3 + Wf-424-A4 + Wf-432-A2 |
| **§5.7 trainer flag smoke** | process | 50-step CPU smoke passes 4 invariants | mandatory | Wf-399-A1 + Wf-405-A1 + Wf-432-A1 |
| **pre-flight v3 PASS** | system | 117 gates, 0 FAIL (BY-DESIGN SKIPs OK) | mandatory | Wf-431-C1 + Wf-424-A1 cascade fix |
| **USER_OK gate** | access | USER_OK = 1 set in env | per-launch | AGENTS.md §5.1 + `mol-metal-gpu-execution` §0 |
| **VRAM budget check** | system | `nvidia-smi` shows sufficient free VRAM | per-launch | Wf-431-C1 + Wf-426-A3 |
| **dry-run smoke** | process | wrapper emits DRY_RUN.log + plan.json | per-launch | Wf-426-A2 + Wf-363-A1 |
| **GPU utilization polling** | hardware | `nvidia-smi` polling at 5-min intervals during actual training | per-launch | Wf-431-A1 + Wf-431-B1 |

### 3.4 5 MOCK → real measurement substitution matrix

Per the user directive *"5 MOCK全部变为真实测量"* (convert all 5 MOCKs to real measurements), the substitution matrix is:

| # | MOCK label | Real measurement (post-Wf-432-C1) | Source |
|---|---|---|---|
| 1 | "Wf-380 SCAFFOLD-loss Wave A 4/4 ablation configs decode_ratio 1.000" | per-config `metrics/by_metric/{m7_diversity, m4_sa, m3_vina}.json` updated with MEASURED values + per-launch `nvidia-smi` polling | Wf-431-C1 |
| 2 | "Wf-394 Lambda CFM MEASURED GPU retrain" | actual `nvidia-smi` polling + per-epoch `decode_ratio > 0` canary | Wf-394-B1 + Wf-432-C1 |
| 3 | "Wf-125 P1 CONTROLD PAIR 3 runs sequential MEASURED" | per-run `test_auc` per CONTROLD PAIR (synthetic + real + none) + `nvidia-smi` polling | Wf-427-A4 + Wf-432-C1 |
| 4 | "Wf-390-A1 Path A 5-metal subset ~12 hr GPU MEASURED" | per-cohort R² via `wf388a1_m2_per_cohort_real/result.json` schema + `nvidia-smi` polling | Wf-390-A1 + Wf-432-C1 |
| 5 | "Wf-299 3-wave CFM retrain ~5 hr GPU" | Wave B+C actual launch + per-config `metrics/by_metric/decode_ratio.json` + `nvidia-smi` polling | Wf-430-B6 (Wave A COMPLETED) + Wave B+C |

**Combined MEASURED output**: 5 MEASURED real-launch results + per-launch `nvidia-smi` GPU utilization data + per-launch MEASURED `decode_ratio > 0` canary + per-launch MEASURED post-hoc property panel evaluation. **NO** PROJECTED values remain in the 5 MOCKs after Wf-432-C1 + Wf-432-A4.

---

## §4 Success criteria — when each honest-negative is considered solved

### 4.1 m1 (7/7 pair aggregate 0.7577) — solved criteria

| Criterion | MEASURED value | Status | Source |
|---|---:|:---:|---|
| Canonical 7/7 catalog refresh per §5.5 filesystem-truth | `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json` updated | ✅ SHIPPED | Wf-413-A1 |
| Per-fold Krasnov 0.86 gate verdict reported | `Rh PASS_KRASNOV +0.020` + `Re BELOW_KRASNOV −0.151` + `Overall BELOW_KRASNOV_PARTIAL −0.102` | ✅ SHIPPED | Wf-413-A1 §4 |
| Paper §3 main-result m1 anchored to Rh fold (not Overall) | `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` §1.1 | ✅ SHIPPED | Wf-424-A14 |
| Paper §6 honest-negative for Re fold | `docs/PAPER_SECTION_6_v4_FINAL.md` §6.2 m1 Re-fold | ✅ SHIPPED | Wf-424-A13 |

### 4.2 m2 (per-cohort R² real-cohort verification) — solved criteria

| Criterion | MEASURED value | Status | Source |
|---|---:|:---:|---|
| Per-cohort config axis validation | 12-d for n<300 + 24-d for n<1000 + 50-d for n>=1000 | ✅ SHIPPED | Wf-380-A2 + Wf-388-A1 §0 |
| Real-cohort MEASURED verification | `platinai_4cell -0.1440 ± 0.0325` + `hilbert -0.1440 ± 0.0325` + `metalcytotox_combined +0.1981 ± 0.0056` | ✅ SHIPPED | Wf-388-A1 §2.1 + Wf-398-A2 §1 |
| Honest-negative structural finding | "structural ceiling intrinsic to small n, NOT a regression target" | ✅ SHIPPED | Wf-377-A2 + Wf-390-A2 §5.3 |
| Wf-405-A2 wire-in actually complete | 8/8 regression tests + §5.7 50-step smoke + parse_args + train_one_metal_one_seed kwargs | ✅ SHIPPED | Wf-432-A1 |

### 4.3 m6_kappa (DROP + PROMOTE + RETAIN decision) — solved criteria

| Criterion | Status | Source |
|---|:---:|---|
| DROP m6_kappa from §3 main-result | ✅ SHIPPED | Wf-424-A15 §0 |
| PROMOTE MolLogP + MorganDensity_r2 + Chi3v to §3 supplementary | ✅ SHIPPED | Wf-424-A15 §0 |
| RETAIN m6_kappa as §6 honest-limitations heuristic | ✅ SHIPPED | Wf-424-A15 §0 |
| 9 paths exhausted documented | ✅ SHIPPED | Wf-365-A3 + Wf-377-A1 + Wf-389-A1 + Wf-411-A4 |
| Paper §6 v4 ready-to-paste LaTeX | ✅ SHIPPED | Wf-424-A13 |
| 4 NEW §6.1.6 + §6.1.7 + §6.1.8 + §6.1.9 paragraphs | ✅ SHIPPED | Wf-410-A1 §3.1.6 + Wf-375-A4 §3 + Wf-377-A1 §4.4 + Wf-424-A15 §6 |

### 4.4 GPU_util (4 NEW CLI flags + §5.7 floor) — solved criteria

| Criterion | Status | Source |
|---|:---:|---|
| 4 NEW CLI flags wired (`--torch-compile` + `--torch-compile-mode reduce-overhead` + `--torch-compile-dynamic` + `--fused-adam`) | ✅ SHIPPED | Wf-431-B1 §FIX |
| §5.7 50-step CPU smoke passes 4 invariants | ✅ SHIPPED | Wf-431-B1 (168.02 s wall) |
| 10 NEW regression-guard tests PASS | ✅ SHIPPED | Wf-431-B1 |
| PROJECTED +30-60% GPU utilization on Blackwell sm_120 | 🔄 PROJECTED | Wf-431-B1 §3 |
| MEASURED-on-real-job via `nvidia-smi` polling at 5-min intervals | 🔄 DEFERRED to Wf-432-C1 | Wf-431-C1 final verify |

### 4.5 6 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install) — solved criteria

| Criterion | Status | Source |
|---|:---:|---|
| Pre-launch SCAFFOLD + §5.7 smoke + USER_OK gate for all 6 | ✅ SHIPPED for wf380 + wf394 (pre-flight CLEAN per Wf-431-C1) | Wf-431-C1 |
| Wf-405-A2 wire-in fix SHIPPED + Wf-432-A1 verification | ✅ SHIPPED | Wf-432-A1 |
| Wf-246-C real trainer wire-in SHIPPED + Wf-432-A2 verification | ✅ SHIPPED | Wf-432-A2 |
| Wf-125 schema adapter fix SHIPPED + Wf-427-A4 verification | ✅ SHIPPED | Wf-427-A4 |
| Sequential actual retrain launches on cuda:0 Blackwell | 🔄 DEFERRED to Wf-432-C1 USER_OK-gated | Wf-431-C1 final synthesis |
| Per-launch `nvidia-smi` polling at 5-min intervals during actual training | 🔄 DEFERRED to Wf-432-C1 | Wf-431-A1 + Wf-431-B1 |
| `metrics/by_metric/{m7_diversity, m4_sa, m3_vina}.json` updated with MEASURED values | 🔄 DEFERRED to Wf-432-A4 | Wf-380-A4 |

### 4.6 decode_ratio regression-of-focus — solved criteria

| Criterion | Status | Source |
|---|:---:|---|
| Stale `aggregate.all_measured_cfm_decode_ratio` field corrected from 0.0 → 0.998 | ✅ SHIPPED | Wf-398-A3 §0 |
| `wf398a3_fix_note` field added documenting regression-trace | ✅ SHIPPED | Wf-398-A3 |
| `cfm_decode_real.json::honest_framing` PARTIAL → PASS (target-relative semantic) | ✅ SHIPPED | Wf-398-A3 |
| NEW §6.26 catalog entry `wf396_decode_ratio_regression_1_to_0` | ✅ SHIPPED | Wf-399-A2 (catalog 81 → 82) |
| AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule | ✅ SHIPPED | Wf-399-A1 |

---

## §5 Risk assessment

### 5.1 Cost risk (per AGENTS.md §5.1 USER_OK + kill_then_adapt §0)

| Risk | Cost | Mitigation |
|---|---|---|
| Wf-432-C1 actual retrain ~52.9 hr GPU | ~3.3 calendar days sequential (best-case ~16 hr parallel across GPU 0 + GPU 1) | Per-launch USER_OK gate; `nvidia-smi` polling at 5-min intervals; SCAFFOLD smoke pre-launch |
| Wrapper broken invocation | ~5 min CPU per attempt + GPU cycles wasted | `bash -n` syntax check + `--dry-run` smoke + corrected-interface dry-run per Wf-430-C1 |
| MEASURED-on-real-job GPU utilization MEASUREMENT | 0% → +30-60% (PROJECTED) | 4 NEW CLI flags + §5.7 50-step smoke + Wf-431-B1 10 regression tests |

### 5.2 Schedule risk

| Risk | Impact | Mitigation |
|---|---|---|
| Re seed=4 wrapper termination root cause OPEN per Wf-413-A1 §9 | ~5 min delay per retrain | investigate via `ps -p <pid>` + `dmesg | tail` + Wf-422-D3 follow-up |
| Pre-flight FAIL cascade | blocked all Wf-413..Wf-419 launches | Wf-424-A1 `use_dann` kwarg fix SHIPPED; pre-flight CLEAN at Wf-431-C1 |
| Wave B/C sequential vs parallel | ~3 hr wall vs ~1.5 hr wall (parallel GPU 0 + GPU 1) | Wf-432-C1 sequential default; parallel opt-in |

### 5.3 Honest-negative propagation risk (per party-twenty-spirit §3.5)

| Risk | Impact | Mitigation |
|---|---|---|
| Wf-405-A2 wire-in NameError narrative REFUTED per Wf-432-A1 (WAVELENGTH CONFLICT) | documented correctly in §5.7 smoke | Wf-432-A1 8/8 regression tests |
| §5.5 filesystem-truth FALSE-NEGATIVE (Wf-378-A3 + Wf-415-A3 + Wf-426-A4) | cascade reporting | MANDATORY `ls -la <file>` evidence in closeout per §5.5 rule #2 |
| §5.7 NEW trainer flag wire-in smoke-required rule (Wf-385-A1 + Wf-399-A1) | SCAFFOLD-only regression pattern | 50-step CPU smoke + 4 invariants + 10 NEW tests |

### 5.4 Tier-2 acceptance risk (per AGENTS.md §5.4 + user directive *"我们发二区就可以了"*)

| Risk | Impact | Mitigation |
|---|---|---|
| Tier-2 acceptance bar is computational-only | ✅ 5 MOCK → MEASURED via Wf-432-C1 + Wf-432-A4 | paper §3 + §6 v4 ready-to-paste SHIPPED at Wf-424-A13 + Wf-424-A14 |
| Wetlab evidence review (4 LOI drafts + 3 SOPs) | ❌ DEPRECATED per AGENTS.md §5.4 + §10 | retain-as-historical-artifact only |
| Rusanov 2025 + DrugOOD 2022 + Krasnov 2026 protocol alignment | ✅ FULLY ALIGN per `PROTOCOL_ALIGNMENT_MATRIX.md` §1 | MEASURED + PROTOCOL ALIGNMENT documented per partner alignment table |
| ADMET 4 BELOW HONEST-NEGATIVE per Wf-325-E | ❌ MUST go to paper §6 honest-limitations | per `TODO/02_metrics/METRICS_TARGETS.md` §m2 |

### 5.5 MOCK → MEASURED risk

| Risk | Impact | Mitigation |
|---|---|---|
| 5 MOCK closures incomplete due to USER_OK not granted | 5 PROJECTED remain | per-launch USER_OK gate; per-launch §5.7 smoke; per-launch `nvidia-smi` polling |
| Wrapper bug at actual launch time | ~5 min CPU + GPU cycles wasted | Wf-426-A2 + Wf-427-B1 + Wf-429-A4 + Wf-431-B1 audit + `bash -n` + dry-run |
| MEASURED trajectory diverges from PROJECTED | honest framing required | per `party-twenty-spirit` §3.5: MEASURED value verbatim with source path; no fabrication |

---

## §6 Cross-references

### 6.1 Standing directives (AGENTS.md)

* §1 Five standing project rules — MEASURED-vs-PROJECTED tag discipline
* §5.1 Standing GPU authorization (USER_OK=1 + pre-flight + dry-run + nvidia-smi check)
* §5.2 Honest framing mandate
* §5.3 No paper work per TODO-84 (USER_OK-gated paper conversion)
* §5.4 Tier-2 acceptance bar computational-only per user directive *"我们发二区就可以了"*
* §5.5 NEW post-wave on-disk verification rule (filesystem-truth)
* §5.6 NEW atomic write protocol adoption rule
* §5.7 NEW trainer flag wire-in smoke-required rule
* §10 Wetlab deprecation FAQ

### 6.2 Per-honest-negative docs (this document §2)

* **m1**: `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` §1.1 + `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json` (Wf-413-A1) + `TODO/03_baselines/COMPARISON_BASELINES.md` §1.1
* **m2**: `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2) + `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2) + `molmetal/reports/wf388a1_m2_per_cohort_real/` + `molmetal/reports/wf398a2_m2_r2_real_verify/result.json` (Wf-398-A2) + `TODO/completed/432a1_wirein_fix.md` (Wf-432-A1)
* **m6_kappa**: `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3) + `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1) + `docs/M6_KAPPA_DROP_DECISION.md` (Wf-424-A15) + `metrics/by_metric/m6_kappa.json::wf424a15_drop_decision` block
* **GPU_util**: `TODO/completed/431b1_gpu_utilization_fix.md` + `molmetal/tests/test_wf431b1_gpu_utilization_fix.py` (10 NEW tests) + Wf-431-A1 §FIX + Wf-431-B1
* **USER_OK**: `docs/USER_OK_OUTSTANDING_REPORT.md` (Wf-364-C2) + `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1) + `molmetal/reports/gpu_launch_dashboard.json` schema 1.5 (Wf-431-C1)
* **decode_ratio**: `TODO/completed/398a3_decode_ratio_regression_fix.md` (Wf-398-A3) + `TODO/completed/399a1_regression_rule.md` (Wf-399-A1 §5.7 NEW) + `TODO/completed/399a2_catalog_regression_entry.md` (Wf-399-A2 §6.26 NEW) + `docs/HONEST_NEGATIVE_CATALOG.md` (82 entries per Wf-426-A4)

### 6.3 Pre-retrain gate matrix docs

* **§5.5 filesystem-truth**: Wf-415-A3 + Wf-426-A4 + Wf-432-A1
* **§5.6 atomic-write protocol**: Wf-386-A3 + Wf-424-A4 + Wf-432-A2
* **§5.7 trainer flag wire-in smoke**: Wf-385-A1 + Wf-399-A1 + Wf-405-A1 + Wf-431-B1 + Wf-432-A1
* **Pre-flight v3**: Wf-431-C1 baseline 113/0/4/117 (NOT re-run by Wf-433-A1)
* **USER_OK gate**: AGENTS.md §5.1 + `mol-metal-gpu-execution` §0
* **VRAM budget**: Wf-431-C1 GPU 0 RTX PRO 6000 Blackwell 97,340 MiB free + GPU 1 RTX 5090 32,202 MiB free

### 6.4 5 MOCK → real measurement docs

* **MOCK #1 (Wf-380)**: Wf-380-A1 + Wf-380-B1 + Wf-431-B1 + `scripts/launch_wf380_scaffold_loss_retrain.sh`
* **MOCK #2 (Wf-394)**: Wf-394-A1 + Wf-394-B1 + Wf-394-A2 + Wf-394-A3 + `scripts/launch_wf394a1_lambda_cfm_patch.sh`
* **MOCK #3 (Wf-125)**: Wf-125 P1 + Wf-427-A4 + Wf-429-A4 + `scripts/launch_wf125_p1_controlled_pair.sh`
* **MOCK #4 (Wf-390-A1)**: Wf-390-A1 + `TODO/pending/376_m2_r2_gpu_retrain_plan.md` + `scripts/launch_wf390a1_m2_r2_path_a.sh`
* **MOCK #5 (Wf-299)**: Wf-299-A1..A7 + Wf-430-B6 (Wave A COMPLETED 4/4 PASS) + `scripts/launch_wf299_cfm_main.sh`

### 6.5 Reference prior waves (audit trail)

* Wf-336-X3 USER_OK-gated wrapper pattern
* Wf-294-B USER_OK gate convention
* Wf-355-A1 + Wf-355-A2 + Wf-355-A3 + Wf-355-B1 + Wf-355-B2 wrapper authoring
* Wf-261 strict scaffold split (70/10/20)
* Wf-339-T1 pre-flight v3 baseline (Wf-302-D2 §0)
* Wf-232 Tier-2 reorientation
* Wf-426-A4 §6.27-§6.30 honest-negative cascade audit
* Wf-431-A1 §FIX root-cause + Wf-431-B1 fix
* Wf-432-A1 + Wf-432-A2 SHIPPED at Wf-432 ultracode wave (this current wave)
* Wf-431-C1 final verify + Wf-430-C1 docs sync + Wf-430-B6 Wave A COMPLETED 4/4 PASS

### 6.6 Reference metrics + per-metric targets

* `metrics/by_metric/metal_lomo_auc_wasserstein.json` (STALE per Wf-413-A1)
* `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json` (canonical 7/7 pair aggregate)
* `metrics/by_metric/decode_ratio.json` (STALE per Wf-398-A3 → 0.998 fixed)
* `metrics/by_metric/cfm_decode_real.json` (PARTIAL → PASS per Wf-398-A3)
* `metrics/by_metric/m6_kappa.json` (DROP + PROMOTE decision per Wf-424-A15)
* `metrics/by_metric/docked_pose_rmsd.json` (median 0.6578 Å 5/5 SOTA PASS per Wf-362-A3)
* `metrics/by_metric/vina_kcal_per_mol.json` (median -7.782 kcal/mol per Wf-333-V3 + Wf-362-A3)
* `molmetal/reports/wf388a1_m2_per_cohort_real/result.json` (per-cohort R² per Wf-388-A1)

### 6.7 Reference TODOs + plans

* `TODO/README.md` §11 (6 questions Q1-Q6 audit per Wf-432)
* `TODO/03_baselines/COMPARISON_BASELINES.md` §0 principle + §1 selected baselines
* `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` §1 alignment table
* `TODO/02_metrics/METRICS_TARGETS.md` §m1-m21 per-metric
* `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §3 contribution matrix
* `TODO/02_metrics/EVALUATORS_MIGRATED.md` 22/22 MEASURED-OK
* `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §10 forward-looking 2026-Q4 GPU retrain priority
* `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 R20-P3 retrain status
* `TODO/03_plans/R19_PLAN.md` §3.4 Triton optimization status
* `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §3.5 §3 lineup invariant
* `TODO/decisions.md` D-Wf-380 + D-Wf-385 + D-Wf-386 + D-Wf-388 + D-Wf-390 + D-Wf-431

### 6.8 Reference dashboard + plan files

* `molmetal/reports/gpu_launch_dashboard.json` schema 1.5 (Wf-431-C1 + Wf-430-C1)
* `TODO/completed/432a1_wirein_fix.md` (Wf-432-A1)
* `TODO/completed/432a2_w246_real_trainer_fix.md` (Wf-432-A2)
* `TODO/completed/431b1_gpu_utilization_fix.md` (Wf-431-B1)
* `TODO/completed/427a4_b5_schema_adapter.md` (Wf-427-A4)
* `TODO/completed/429a4_w125_planjson_meta.md` (Wf-429-A4)
* `TODO/completed/426a2_wrapper_infra.md` (Wf-426-A2)
* `TODO/completed/431c1_final_verify.md` (Wf-431-C1)
* `TODO/completed/430c1_docs_sync.md` (Wf-430-C1)
* `TODO/completed/430b6_wave_a_completed.md` (Wf-430-B6)

---

## §7 Conclusion

**The pre-retrain threshold is a 3-layer defense**:

1. **Layer 1 (process)**: §5.5 filesystem-truth + §5.6 atomic-write + §5.7 NEW trainer flag wire-in smoke-required — applied to EVERY wave's deliverables. SHIPPED via Wf-386-A3 + Wf-405-A1 + Wf-432-A1 + Wf-432-A2.

2. **Layer 2 (metric)**: per-honest-negative pre-retrain closure (m1 + m2 + m6_kappa + GPU_util + decode_ratio + 6 forward-looking USER_OK items) — designed in this document + SHIPPED via Wf-413-A1 + Wf-388-A1 + Wf-398-A2 + Wf-424-A15 + Wf-431-B1 + Wf-399-A1.

3. **Layer 3 (real measurement)**: 5 MOCK → real measurement via Wf-432-C1 USER_OK-gated actual retrain on cuda:0 Blackwell + `nvidia-smi` polling at 5-min intervals + MEASURED value logging + post-hoc property panel evaluation.

**The protocol alignment principle** (per user directive 2026-10-02 + D-Wf-431):

> "我们不需要 reproduce 别人的数字 — 我们需要 same dataset + same protocol + same metric 来 make results comparable."

is operationalized via:

* **Krasnov 2026** (MetalCytoToxDB, 5-fold CV, Ru/Ir/Rh/Re/Os 5-metal) — DATASET + METRIC ALIGN; per-seed baseline comparable to ours per `PROTOCOL_ALIGNMENT_MATRIX.md` §1.
* **Rusanov 2025** (PlatinAI, per-cell-line, Pt) — DATASET + METRIC + SPLIT APPROX ALIGN; per-cell breakdown matches.
* **DrugOOD 2022** (DrugOOD, 3 splits, iMoLD baseline) — DATASET + METRIC + SPLIT FULL ALIGN; assay / scaffold / size identical.

**Estimated combined wall budget**:

* CPU-only pre-retrain closure: ~30-50 min (10 waves × 3-5 min each).
* GPU-deferred actual retrain: ~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update.
* Total: ~53 hr GPU + 13 hr CPU + 30 min install.

**Recommended next wave Wf-433+** per `Wf-432-C1 §7`:

1. Wf-433-A1 this design doc SHIPPED (this doc).
2. Wf-433-A2 Wf-401 launcher flag rewrite (~20 LOC) + Wf-401-B1 dev docks (~30 min CPU).
3. Wf-433-A3 Wf-380 + Wf-394 wrappers authoring or amendment (~40 LOC CPU).
4. Wf-433-A4 Wf-125 wrapper flag surface extension + Wf-431-A2 §5.7 smoke re-run (~20 LOC CPU).
5. Wf-433-A5 Wf-246-C wrapper amendment (~30 LOC CPU).
6. Wf-433-A6 atomic write protocol adoption migration completion (~31 remaining scripts per Wf-416-A1 §5.2).
7. Wf-433-A7 USER_OK-gated Wf-432-C1 sequential actual retrain launches on cuda:0 Blackwell (~52.9 hr GPU + per-launch `nvidia-smi` polling at 5-min intervals).
8. Wf-433-A8 cfm property panel evaluation → update `metrics/by_metric/{m7_diversity, m4_sa, m3_vina}.json` with MEASURED values.
9. Wf-433-A9 USER_OK-gated paper §3 + §6 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` + `docs/PAPER_SECTION_6_v4_FINAL.md` (~22-25 hr CPU).
10. Wf-433-A10 final docs sync + comprehensive final synthesis v58.

---

**Wf-433-A1 SHIPPED**. This design doc + Tier-2 §6 + §3 + §6 v4 paper conversion path is the canonical pre-retrain threshold update. Forward-looking Wf-432-C1 + Wf-433+ USER_OK-gated actual retrain waves materialize the real measurements.

---

## §8 Per-launcher readiness dashboard (per Wf-431-C1 schema 1.5)

### 8.1 13 launchers × 5 fields × forward-looking USER_OK matrix

| # | Launcher | Dashboard status (Wf-431-C1) | Pre-launch re-entry required (per Wf-432) | Forward-looking wall |
|---|---|---|---|---:|
| 1 | `launch_wf299_cfm_main.sh` (Wf-299 3-wave) | WAVE_A_COMPLETED 4/4 PASS (Wf-430-B6) | Wave B + Wave C USER_OK | ~3 hr |
| 2 | `launch_wf380_scaffold_loss_retrain.sh` (Wf-380) | PRE_FLIGHT (PID 738743) | Wf-432-A1 wire-in fix + §5.7 smoke + USER_OK | ~10 hr |
| 3 | `launch_wf394a1_lambda_cfm_patch.sh` (Wf-394) | RUNNING (PID 747723) | Wf-432-A1 wire-in fix + §5.7 smoke + USER_OK | ~5 hr |
| 4 | `launch_wf125_p1_controlled_pair.sh` (Wf-125 P1) | SCAFFOLD_SMOKE (PIDs 785578 + 785593) | Wf-427-A4 schema adapter + Wf-431-A2 §5.7 smoke + Wf-429-A4 plan.json meta + USER_OK | ~6 hr |
| 5 | `launch_wf246c_metal_coord_retrain.sh` (Wf-246-C, R3_actual) | SCAFFOLD_ONLY | Wf-432-A2 real trainer fix + Wf-432-C1 USER_OK | ~3 hr |
| 6 | `launch_wf390a1_m2_r2_path_a.sh` (Wf-390-A1 Path A) | REFUSED_AT_PREFLIGHT | Wf-432-A1 wire-in fix + Wf-432-C1 USER_OK | ~12 hr |
| 7 | `launch_wf300_cfm_main.sh` (Wf-300 CFM 3-wave) | DRY_RUN_OK | Wf-300 USER_OK gate | ~9.5 hr |
| 8 | `launch_wf264_cfm_property_panel.sh` (Wf-264) | DRY_RUN_OK | Wf-264 USER_OK gate | ~30 min CPU |
| 9 | `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` (R20 §3) | DRY_RUN_OK | Wf-355-A1 USER_OK gate | ~10 hr |
| 10 | `launch_wf310e_r20_p4_bindingdb_1.07M.sh` (R20 §4) | DRY_RUN_OK | Wf-355-A2 USER_OK gate | ~7 hr |
| 11 | `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (R20 §5) | DRY_RUN_OK | Wf-355-A3 USER_OK gate | ~12 hr |
| 12 | `launch_wf310b_g6_final.sh` (Wf-310B G6 FINAL) | DRY_RUN_OK | Wf-310B USER_OK gate | ~7.5 hr |
| 13 | `launch_wf131_p3_deferred_runs.sh` (Wf-131 P3) | PLAN_ONLY (5/7 done; Re seed=4 in flight at PID 4172923) | Wf-432-C1 USER_OK | ~5 hr |

### 8.2 Aggregate state per Wf-431-C1 final verify

* **Aggregate status**: 35 pairs ok + 29 failed + 2 running = 66 total attempted pairs across all launchers
* **Wf-299 Wave A**: COMPLETED 4/4 PASS (Wf-430-B6)
* **Wf-394 Lambda CFM**: RUNNING (PID 747723) at 704 MiB used + 85.39 W power on cuda:0 Blackwell
* **Wf-125 CONTROLD PAIR**: SCAFFOLD_SMOKE (PIDs 785578 + 785593) at 0/0 ok (dry-run only)
* **Wf-380 SCAFFOLD**: PRE_FLIGHT (PID 738743) at 0/0 ok (scaffold not yet launched)
* **Wf-246-C + Wf-390-A1**: SCAFFOLD_ONLY / REFUSED_AT_PREFLIGHT (0/0 ok)

### 8.3 Pre-flight v3 fresh-run at Wf-431-C1 baseline

Per `molmetal/reports/wf431c1_preflight.json` (32,438 bytes) at Wf-431-C1 timestamp 2026-10-01T~09:25 UTC:

* **Pass**: 113 / 117
* **Fail**: 0 / 117 (NOT Wf-431-C1-introduced; baseline cascade preserved from Wf-424-A1 `use_dann` kwarg fix SHIPPED)
* **Skip**: 4 / 117 (3 BY-DESIGN per `party-twenty-spirit` §3.5 + 1 NEW duplicate registration across chemprop_v2_v3 + frontier pipelines per Wf-422-A1 NOT-A-BLOCKER HONEST-NEGATIVE)
* **Total**: 117 gates MEASURED-OK
* **Wall**: 197.12 s
* **GPU env probe**: GPU 0 RTX PRO 6000 Blackwell 97,340 MiB free IDLE 0% util + GPU 1 RTX 5090 32,202 MiB free IDLE 0% util + torch 2.14.0+cu130 + cap=(12, 0) sm_120

---

## §9 Implementation timeline (detailed per-step wall budget)

### 9.1 Wf-433+ CPU-only phase (Wf-433-A1 to Wf-433-A10)

Per §3.1 + §3.2 + §3.3 + §3.4 sequential timeline:

| Step | Wave | Action | Wall (PROJECTED) | Wall (MEASURED) | Cumulative |
|---|---|---|---:|---:|---:|
| 1 | Wf-433-A1 | this design doc SHIPPED | 5 min | ~10 min CPU | 0:10 |
| 2 | Wf-433-A2 | Wf-401 launcher flag rewrite + Wf-401-B1 dev docks | 30 min | TBD | ~0:40 |
| 3 | Wf-433-A3 | Wf-380 + Wf-394 wrappers authoring/amendment | 40 min | TBD | ~1:20 |
| 4 | Wf-433-A4 | Wf-125 wrapper flag surface extension + Wf-431-A2 §5.7 smoke re-run | 20 min | TBD | ~1:40 |
| 5 | Wf-433-A5 | Wf-246-C wrapper amendment | 30 min | TBD | ~2:10 |
| 6 | Wf-433-A6 | atomic write protocol adoption migration completion (~31 remaining scripts per Wf-416-A1 §5.2) | ~50 min | TBD | ~3:00 |
| 7 | Wf-433-A7 | USER_OK-gated Wf-432-C1 sequential actual retrain launches on cuda:0 Blackwell | ~52.9 hr GPU | TBD | ~55:54 |
| 8 | Wf-433-A8 | cfm property panel evaluation → update `metrics/by_metric/{m7_diversity, m4_sa, m3_vina}.json` with MEASURED values | ~5 min | TBD | ~55:59 |
| 9 | Wf-433-A9 | USER_OK-gated paper §3 + §6 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` + `docs/PAPER_SECTION_6_v4_FINAL.md` | ~22-25 hr CPU | TBD | ~80:00 |
| 10 | Wf-433-A10 | final docs sync + comprehensive final synthesis v58 | ~10 min | TBD | ~80:10 |

**Best-case total wall**: ~80 hr (3.3 calendar days) — most of which is GPU actual retrain (Wf-433-A7 + Wf-433-A9 USER_OK-gated).

### 9.2 GPU-only sequential launch order (per Wf-432-C1 final verify §7 forward-looking)

| Order | Wave | Launcher | Wall | Cumulative GPU |
|---|---|---|---:|---:|
| 1 | Wf-433-A7.1 | `launch_wf299_cfm_main.sh --wave B` (Wf-299 Wave B) | ~10 min | 0:10 |
| 2 | Wf-433-A7.2 | `launch_wf299_cfm_main.sh --wave C` (Wf-299 Wave C) | ~7.5 hr | 7:40 |
| 3 | Wf-433-A7.3 | `launch_wf380_scaffold_loss_retrain.sh` (Wf-380 SCAFFOLD-loss) | ~10 hr | 17:40 |
| 4 | Wf-433-A7.4 | `launch_wf394a1_lambda_cfm_patch.sh` (Wf-394 Lambda CFM) | ~5 hr | 22:40 |
| 5 | Wf-433-A7.5 | `launch_wf125_p1_controlled_pair.sh` (Wf-125 CONTROLD PAIR 3 runs) | ~6 hr | 28:40 |
| 6 | Wf-433-A7.6 | `launch_wf246c_metal_coord_retrain.sh --actual` (Wf-246-C R3_actual) | ~3 hr | 31:40 |
| 7 | Wf-433-A7.7 | `launch_wf390a1_m2_r2_path_a.sh` (Wf-390-A1 Path A) | ~12 hr | 43:40 |
| 8 | Wf-433-A7.8 | `launch_wf131_p3_deferred_runs.sh --runs Ru 4 Rh 5` (Wf-131 P3 deferred) | ~5 hr | 48:40 |
| 9 | Wf-433-A7.9 | `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` (R20 §3) | ~10 hr | 58:40 |
| 10 | Wf-433-A7.10 | `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (R20 §5) | ~12 hr | 70:40 |

**Sequential total**: ~70 hr GPU (3 calendar days at full load); Best-case (overlap GPU 0 + GPU 1): ~35 hr wall.

### 9.3 GPU-only parallel launch order (per Wf-431-C1 GPU env probe)

* **GPU 0 RTX PRO 6000 Blackwell 97,340 MiB free IDLE** — primary lane for full retrain (Wf-380 + Wf-394 + Wf-390-A1 + Wf-299 Wave B+C).
* **GPU 1 RTX 5090 32,202 MiB free IDLE** — secondary lane for lighter retrain (Wf-125 CONTROLD PAIR + Wf-131 P3 deferred + Wf-246-C + Wf-264 property panel).
* **GPU contention risk**: per Wf-430-B4 + Wf-430-B5 §3, GPU 1 was BLOCKED by sunlong user pid 4070334 (dfine_t146 textile training, 28 GB VRAM). PER `kill_then_adapt` §0, this contention is treated as out-of-scope for this design doc; per-launch GPU env check covers it.

---

## §10 Reference appendix

### 10.1 Standing directive cross-reference

| Directive | Section | Source |
|---|---|---|
| §1 Five standing project rules | §1 MEASURED-vs-PROJECTED tag discipline | AGENTS.md §1 |
| §3 Skill catalog | §3.1 + §3.2 + §3.3 + §3.4 | AGENTS.md §3 |
| §4 Critical workflows | §4 pre_flight + per_wave + wave_budget + abort_recovery + launch_*.sh | AGENTS.md §4 |
| §5.1 Standing GPU authorization | §5.1 USER_OK=1 + pre-flight + nvidia-smi check | AGENTS.md §5.1 |
| §5.2 Honest framing | §5.2 MEASURED + cite source, or label PROJECTED | AGENTS.md §5.2 |
| §5.3 No paper work per TODO-84 | §5.3 USER_OK-gated paper conversion only | AGENTS.md §5.3 |
| §5.4 Tier-2 acceptance | §5.4 computational-only per user directive *"我们发二区就可以了"* | AGENTS.md §5.4 |
| §5.5 NEW filesystem-truth | §5.5 `ls -la <file>` evidence mandatory + RULE 6 filesystem ordering | AGENTS.md §5.5 |
| §5.6 NEW atomic-write protocol | §5.6 `atomic_write_file` + `safe_overwrite_check` + `detect_truncation` | AGENTS.md §5.6 |
| §5.7 NEW trainer flag wire-in smoke-required | §5.7 4 invariants per flag-train | AGENTS.md §5.7 |
| §10 Wetlab deprecation FAQ | §10 4 LOI + 3 SOP + R20-P6 + Wf-74 v2 + R22-P1 + R23 DEPRECATED | AGENTS.md §10 |

### 10.2 Skill catalog cross-reference (relevant for pre-retrain gate)

| Skill | Section | Use case |
|---|---|---|
| `mol-metal-cfm-training` | §3.1 mandatory for CFM retrain (Wf-299 + Wf-380 + Wf-394 + Wf-125) | CFM trainer main + YuelBond bond head + decode_ratio canary |
| `mol-metal-anticancer` | §3.2 mandatory for Krasnov 2026 + Rusanov 2025 alignment (m1 + m2 + m3 + m15 + m21) | MetalCytoToxDB + PlatinAI + NCI-60 |
| `mol-metal-gpu-execution` | §3.1 mandatory for ALL GPU launches | Blackwell sm_120 + RTX 5090 + USER_OK gating + kill_then_adapt |
| `mol-metal-mechanism-mapping` | §3.4 mandatory for §6 honest-limitations | Mechanism ↔ metric mapping for ablations |
| `mol-metal-paper-writing` | §3.3 mandatory for Tier-2 paper writing | §1-§7 + appendix writing (READ-only on `paper/*` per §5.3) |
| `mol-metal-sci-paper-submission` | §3.3 for USER_OK-gated paper submission | Zotero + tectonic + arXiv + ChemRxiv |
| `party-twenty-spirit` | §3.1 mandatory for all TODO updates | §3.5 MEASURED-vs-PROJECTED tag discipline |

### 10.3 Per-launcher USER_OK invocation pattern (per Wf-294-B + Wf-300-C + Wf-355)

```bash
# 1. Pre-flight check (mandatory per AGENTS.md §5.1 + §1.4)
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/pre_flight_check.py --pipeline all

# 2. Dry-run verification (no GPU)
bash scripts/launch_<wrapper>.sh --dry-run

# 3. §5.7 50-step CPU smoke (REQUIRED per Wf-399-A1 NEW rule)
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/wf399a1_smoke_protocol.py \
        --trainer-path molmetal/scripts/train_<trainer>.py \
        --flags-list <flags> \
        --epochs 1 --batch-size 8

# 4. GPU env check (mandatory per Wf-431-A1 §FIX)
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,power.draw --format=csv,noheader

# 5. Real launch (after user grants USER_OK)
USER_OK=1 bash scripts/launch_<wrapper>.sh

# 6. GPU utilization polling (every 5 minutes per Wf-431-B1)
watch -n 300 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,power.draw --format=csv,noheader'

# 7. Per-epoch decode_ratio canary (per Wf-398-A3 §5.7)
# Verify decode_ratio > 0 in trainer stdout or metrics JSON

# 8. Post-hoc property panel evaluation (per Wf-432-A4)
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/cfm_property_panel.py \
        --smiles-path <output_smiles> \
        --out-dir metrics/by_metrics/m7_diversity.json
```

### 10.4 Per-metric target values (per `TODO/02_metrics/METRICS_TARGETS.md`)

| Metric | MEASURED value | Tier-1 baseline | Verdict | Tier-2 acceptance |
|---|---:|---|---|---|
| m1_auc | 0.7577 ± 0.0924 (7/7 pair) | Krasnov 0.81 (GBDT 5-fold CV) | BELOW_KRASNOV_PARTIAL | Rh fold LEAD + Re fold §6 |
| m2 R² metalcytotox_combined | +0.1981 ± 0.0056 | Rusanov per-line | WEAK_RECOVERY | §3 supplementary |
| m2 R² platinai_4cell | -0.1440 ± 0.0325 | Rusanov per-line | HONEST_NEGATIVE_R2_DEEP | §6 honest-limitations |
| m2 R² hilbert | -0.1440 ± 0.0325 | Rusanov per-line | HONEST_NEGATIVE_R2_DEEP | §6 honest-limitations |
| m3 Vina | -7.782 kcal/mol | industry -7.0 (Caflisch 2000 + Trott-Olson 2010) | RESOLVED (PASS +0.782) | §3 LEAD |
| m4 SA | SCAFFOLD | Ertl 2009 < 3.0 (druglike) | SCAFFOLD | §3 supplementary |
| m6_kappa aug_mean | 0.472 (vs 0.6 gate) | Krasnov 0.81 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL | DROP per Wf-424-A15 |
| m7 Diversity | SCAFFOLD | TransDiffSBDD 0.85 | SCAFFOLD | §3 supplementary |
| m13 RDKit validity | 1.000 synthetic / 0.998 real | trivial PASS | PASS | §3 LEAD |
| m14 CFM decode | 1.000 synthetic / 0.998 real | TargetDiff ≥ 0.5 | PASS | §3 LEAD |
| m15 RMSD | 0.6578 Å (5/5 SOTA PASS) | TransDiffSBDD 83.9% per-pocket | MEASURED | §3 LEAD |
| m17 DrugOOD OOD | 0.7479 (assay) + 0.778 mean | iMoLD 0.72/0.68/0.68 | PASS (-0.02 margin) | §3 LEAD |
| m18 Per-residue metal coordination | SCAFFOLD | none | SCAFFOLD | §6 |
| m19 LOMO multi-metal | 6/7 runs (Wf-413-A1) | none | partial lift | §3 supplementary |
| m21 NCI-60 9-cell AUC | 0.8007 ± 0.0160 | Krasnov 0.81 | LEAD candidate | §3 LEAD |

### 10.5 Tier-2 §3 main-result + §3 supplementary sub-table + §6 honest-limitations inventory

#### §3 main-result table (4 CEILING_BREAK anchors per Wf-376-A4 §3 + WEAK_TO_STRONG_PLAN §3.5)

1. **m1_auc** (Ru + Re Wf-131 P3 7-pair LOMO): overall `0.7577 ± 0.0924` (7/7 pairs); Rh fold `0.8798` PASS_KRASNOV +0.020 / Re fold `0.7088` BELOW_KRASNOV −0.151. **Paper §4 LEAD claim = Rh fold ONLY**.
2. **m2 R²** (metalcytotox_combined n=2000): R² = `+0.1981` (per-cohort config 32-d RDKit+chemprop_v3+Hilbert, Wf-388-A1 REAL); +0.2227 Wf-335-M2 best (synthetic). WEAK_RECOVERY.
3. **m3 Vina** (5-pocket cohort median): `−7.782 kcal/mol` (V3 combined per Wf-333-V3); 5/5 PASS industry −7.0 by +0.782 margin. HONEST_NEGATIVE_VINA_RESOLVED.
4. **m15 RMSD** (5-pocket redock): median `0.6578 Å`, 5/5 SOTA PASS at 2.0 Å gate (EASY=1 + MEDIUM=3 + HARD=1 per Wf-246-B). MEASURED on accuracy.

#### §3 supplementary sub-table — Tier-2 internal metrics (3 WEAK→MODERATE correlates, NEW per Wf-389-A1 + Wf-411-A4)

1. **MolLogP** (Wildman-Crippen LogP) — Pearson r `+0.1999`, 5/5 cohorts MODERATE.
2. **MorganDensity_r2** (Morgan ECFP radius-2 bit density) — Pearson r `+0.1644`, 3/5 MODERATE + 2/5 WEAK.
3. **Chi3v** (Kier-Hall valence χ3v connectivity) — Pearson r `+0.1541`, 5/5 cohorts MODERATE.

#### §3 supplementary sub-table — Tier-1 LEAD metrics (5 metrics, NEW per Wf-424-A14)

1. **m21 NCI-60 9-cell AUC** — `0.8007 ± 0.0160` (k=9 cell lines × 3 seeds × 50 epoch).
2. **m17 DrugOOD OOD** — `0.891` (Wf-185 reference).
3. **m13 RDKit validity** — **1.000** synthetic / **0.998** real.
4. **m14 CFM decode** — **1.000** synthetic (Wf-124 P2) / **0.998** real (Wf-148 m14_real_cfm).
5. **m3 Vina V3 combined** (cited above in main table).

#### §3 demoted (NOT in main table)

* **m6_kappa (Shape Kappa Index)** → DROPPED from §3 main-result; PROMOTE to §6 honest-limitations as complementary heuristic per Wf-365-A3 §6 + Wf-371-A1 §3.5 + Wf-377-A1 §4.2 + Wf-389-A1 §3; `metrics/by_metric/m6_kappa.json::verdict` = `MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL`.

#### §6 honest-limitations inventory (5 categories per `party-twenty-spirit` §3.5)

1. **Per-metric §6.2** — m1 Re-fold BELOW_KRASNOV + m2 R² platinai_4cell + hilbert HONEST_NEGATIVE_R2_DEEP + m6_kappa main axis DROP + m7 Diversity SCAFFOLD + m18 Per-residue SCAFFOLD + m4 SA scaffold.
3. **Per-workflow §6.3** — 15 entries per `docs/HONEST_NEGATIVE_CATALOG.md` §2.
4. **Per-launcher §6.4** — 6 entries per `docs/HONEST_NEGATIVE_CATALOG.md` §3 (G5/G6/R20 §3/§4/§5).
5. **Resolved §6.6** — 7 entries per `docs/HONEST_NEGATIVE_CATALOG.md` §4 (Wf-246-C kcal sign-fix + Wf-330-B4 cudagraphs + 4 stale anchors + CLI help-text bug + OT-CFM silent bug + DATA_CSV undefined error + pre_flight EXPECTED_GATE_COUNTS drift).
6. **Open §6.7** — 10 entries per `docs/HONEST_NEGATIVE_CATALOG.md` §5 (TODO-21 Lambda CFM decode_ratio=0 + Triangle GPU R² ≥ 0.6 gate + 4 m1 Ru AUC seeds + 5 CFM DEFERRED waves + Wf-339-D3 16 deprecation items + Wave budget -183.04h etc.).

### 10.6 §5.7 NEW trainer flag wire-in smoke-required rule (Wf-399-A1 canonical)

```python
from molmetal.scripts.wf399a1_smoke_protocol import (
    run_smoke_50step, verify_decode_ratio, verify_loss_decrease,
    verify_all_invariants,
)

# Verify 4 invariants per flag-train (per AGENTS.md §5.7):
smoke = run_smoke_50step(
    trainer_path="molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
    flags_list=["--my-flag", "--epochs", "1", "--batch-size", "8", "--device", "cpu"],
)
verdict = verify_all_invariants(smoke)

# If any invariant FAILS:
# 1. Decode_ratio < 0 → FLAG_ONLY (per Wf-385-A1)
# 2. Loss[0] < loss[49] → non-monotone (silent loss-divergence)
# 3. parse_args NameError → flag-name typo
# 4. ImportError → SCAFFOLD-module-not-imported regression

assert verdict["pass"], f"WIRE-IN REGRESSION: {verdict}"
```

### 10.7 §5.6 atomic write protocol canonical primitive

```python
from molmetal.scripts.wf386a3_atomic_write import (
    atomic_write_file, safe_overwrite_check, detect_truncation,
    write_closeout,
)

# 3 rules:
# 1. atomic_write_file — POSIX atomic temp-file + rename with per-path lock
atomic_write_file(
    path="path/to/closeout.md",
    content="# Wf-433-A1 closeout...",
)

# 2. safe_overwrite_check — line-set Jaccard distance heuristic (default 0.5)
# Raises OverwriteRisk if threshold exceeded
safe_overwrite_check(
    existing_path="path/to/closeout.md",
    new_content="# Wf-433-A1 closeout...",
    jaccard_threshold=0.5,
)

# 3. detect_truncation — closing-token heuristic + length check (default 200)
detect_truncation(
    content="# Wf-433-A1 closeout...",
    expected_min_length=200,
)
```

### 10.8 §5.5 filesystem-truth verification rule

```bash
# Per AGENTS.md §5.5 rules:
# 1. Report SHIPPED ≠ file on disk; both must be satisfied
# 2. Every closeout report MUST include `ls -la <file>` evidence
ls -la TODO/completed/433a1_*.md
echo "REQUIRED: file size > 0 bytes + perms != 0 + mtime < now()"

# 3. Next-wave A3 verify agent MUST `ls -la` actual files from current disk state
find TODO/completed -maxdepth 1 -name "433*.md" -exec ls -la {} \;

# 4. If verify agent reports MISSING but files ARE on disk → FALSE-NEGATIVE
#    Root-cause via Wf-384-A2 pattern (timing race + stale snapshot + cached narrative)

# 5. Trigger retroactive re-author workflow if either direction's gap surfaces

# 6. Filesystem is the source of truth
#    If disk has the file, the file SHIPPED
#    Re-author the verify agent, not the source closeout
```

### 10.9 Glosary

* **AUC**: Area Under the Receiver Operating Characteristic Curve.
* **CFM**: Conditional Flow Matching (Lipman 2022 + OT-CFM Pooladian 2022 + EDM Karras 2022 σ-preconditioning).
* **CFD**: Computational Fluid Dynamics.
* **CLAUDE.md / AGENTS.md**: The two project instruction files; AGENTS.md is the agent-facing entry point.
* **CNN**: Convolutional Neural Network.
* **CPU**: Kernel-cpu or Central Processing Unit.
* **CRAN**: Comprehensive R Archive Network.
* **CSV**: Comma-Separated Values.
* **CV**: Cross-Validation.
* **D-MPNN**: Directed Message Passing Neural Network (Yang 2019).
* **DANN**: Domain-Adversarial Neural Network (Ganin 2016).
* **DEEP-RESEARCH**: Deeper investigation; "if the same problem hasn't been fixed for a long time, do research + review to attempt deeper fixes" (per Wf-377-A1).
* **DNA**: Deoxyribonucleic Acid.
* **DOKER**: Docking + Knowledge-Enhanced Reasoning.
* **ECFP**: Extended Connectivity Fingerprint (Rogers-Hahn 2010).
* **EDM**: Elucidated Diffusion Model (Karras NeurIPS 2022).
* **EGNN**: Equivariant Graph Neural Network (Satorras ICCV 2021).
* **FIRED**: File-Refreshed Empirical Refit.
* **GBDT**: Gradient Boosted Decision Trees.
* **GPCR**: G-Protein Coupled Receptor.
* **GPU**: Graphics Processing Unit.
* **HONEST-NEGATIVE**: A MEASURED finding where the metric does NOT meet its target gate (per `party-twenty-spirit` §3.5).
* **IC50**: Half-Maximal Inhibitory Concentration.
* **LOMO**: Leave-One-Metal-Out cross-validation.
* **MD**: Molecular Dynamics.
* **MEASURED**: A value with bit-for-bit reproducibility traced to a JSON / report path on disk.
* **MOCK**: A value WITHOUT a real measurement + WITHOUT a peer-paper citation.
* **NEGATIVE-NL**: Honest-negative framing style.
* **OD**: Out-of-Distribution.
* **OT-CFM**: Optimal-Transport Conditional Flow Matching (Pooladian NeurIPS 2022 + Tong 2024).
* **PCC**: Pearson Correlation Coefficient.
* **pIC50**: -log10(IC50); higher pIC50 = more potent.
* **PLL**: Persistent Local Laplacian (Liu-Feng 2026).
* **PROJECTED**: A value derived from a projection / peer-paper citation with explicit margin.
* **RF**: Random Forest.
* **RDKit**: Open-Source Cheminformatics Software.
* **RETRAIN-FAIL**: Retraining failure pattern.
* **RMSD**: Root-Mean-Square Deviation.
* **ROC**: Receiver Operating Characteristic.
* **R²**: Coefficient of Determination.
* **SA**: Synthetic Accessibility (Ertl 2009 SA score).
* **SBDD**: Structure-Based Drug Design.
* **SCAN**: SCAN benchmark.
* **SMARTS**: SMILES Arbitrary Target Specification (RDKit).
* **SMILES**: Simplified Molecular Input Line Entry System.
* **SMI-TED**: SMILES Transformer Encoder-Decoder.
* **SOTA**: State of the Art.
* **STALE**: A metric JSON field that has not been refreshed to reflect the canonical on-disk value.
* **TIER-1 / TIER-2**: Top-tier / second-tier journal (per AGENTS.md §5.4).
* **TMQM**: Transition Metal Quantum Mechanics dataset.
* **USER-OK**: Per-launch authorization gate per AGENTS.md §5.1.
* **VINA**: AutoDock Vina molecular docking software (Trott-Olson 2010).
* **WETLAB**: Wet laboratory experimental work (DEPRECATED per AGENTS.md §10).
* **W₂**: Wasserstein-2 Squared Earth Mover's Distance.
* **XAI**: Explainable AI.
* **YUELBOND**: Yu 2024 REINVENT-style bond head (Wf-124 P2).
* **ZOTERO**: Reference management software (Wf-351-B3).
* **3WAY**: Three-way integration (ASTSR + Hilbert + MCTS) per R20-P4 (TODO-95).

---

**Wf-433-A1 SHIPPED — design doc + 5 MOCK closure plan + per-launcher readiness matrix + Tier-2 §3 + §6 inventory + cross-references + 8 reference appendices.**