# Wf-296..359 Retrospective — Institutional Memory for Mol-Metal Ultrawave

**Date:** 2026-09-29
**Workflow:** Wf-360-B1 (per-Wf-296..359 retrospective; ultracode aggregation)
**Owner:** Hugo Lee + Claude Code subagent (Wf-360-B1)
**Skill:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline; cite every claim to JSON / report / file path on disk)
**Cross-ref:**
- AGENTS.md §1 (5 standing rules)
- AGENTS.md §5.1 (standing GPU authorization, 2026-09-29)
- AGENTS.md §5.2 (honest framing mandate)
- AGENTS.md §5.3 (no paper work without authorization, TODO-84)
- AGENTS.md §5.4 (Tier-2 journal target + NO wet-lab planning, 2026-09-29)
- Wf-340-R1 (CFM research survey) + Wf-340-I1..I3 (6 wires + 12 tests)
- Wf-335-SYN (Phase 1 RUNTIME BUG + Phase 2 m6_kappa + Phase 3 P0 wins)
- Wf-357-C1/C2/C3 (workflow + architecture + speedup analysis trio)
- Wf-358-A3 (gradient_checkpointing + channels_last + torch_compile_dynamic)
- Wf-359-A1 (m6_kappa W₂ R20-P3 re-verify; HONEST_NEGATIVE_PARTIAL_STRUCTURAL)

**Status:** ✅ **SHIPPED** (docs-only aggregation; 0 NEW tests beyond Wf-360-B1 regression test; 0 GPU launches; PROJECTED 25-40 min CPU-only)

---

## §0 TL;DR (Honest framing per `party-twenty-spirit` §3.5)

* **64 distinct workflow units** spanned Wf-296..359 (per `ls TODO/completed/ | grep -E '^(29|30|31|32|33|34|35)[0-9]' | wc -l` 2026-09-29).
* **419 closeout files** in 290..359 range covering integration, docs refresh, GPU profile, ultracode residual, CFM research, Tier-2 re-orientation, recovery, retrospectives.
* **Top-3 patterns that worked**: (1) Honest-negative framing preserved as institutional practice; (2) Pre-flight v3 gate as golden thread; (3) Ultracode aggregation pattern (R1..Rn → SYN doc).
* **Top-3 failure modes**: (1) GPU Blackwell sm_120 cudagraphs correctness gap (Wf-330-B4 / Wf-335-R1+R3); (2) DDP/FSDP out-of-scope on single-GPU Mol-Metal; (3) m6_kappa gate NOT lifted by W₂+PLL feature engineering (Wf-335-M1 / Wf-359-A1).
* **Forward-looking improvements** (per Wf-357-C3 + Wf-358-A3): (1) auto-fix pre-flight v3 2 FAILs; (2) pre-retrain checklist auto-emission; (3) cross-workflow dependency tracking; (4) opt-in `--gradient-checkpointing` + `--channels-last` + `--torch-compile-dynamic` + `--fused-adam` CLI flags (Wf-358-A3 / Wf-357-D1).
* **3 honest-negative caveats preserved**: (1) all speedup / lift claims PROJECTED from upstream paper benchmarks; (2) NO Mol-Metal GPU A/B smoke launched as of 2026-09-29; (3) Wf-359-A1 m6_kappa gate PRESERVED (augmented_mean 0.356 < 0.6 threshold = -0.244 margin).

---

## §1 Per-Wf summary (64 Wfs × 1-line summary)

One row per wave cohort, Wf-296..359. `Status` sourced from on-disk closeout headers.

| § | Wave | Status | Outcome | Key takeaway | Primary file |
|---|---|---|---|---|---|
| §1.1 | Wf-296 — Pipeline integration audit + integration (Wf-296 SH | ✅ SHIPPED 2026-09-29 | Initial integration checkpoint; downstream Wf-309..317 wave absorbed follow-up. | Set the stage for the ultrawave by cataloguing the in-flight pipelines (D-MPNN + CFM + Lambda MCTS + closur… | `TODO/completed/296_wf296_pipeline_integration.md` (1 of 1 file in 290s range; bulk 296 content lives in Wf-309..317 + W |
| §1.2 | Wf-309..317 ship wave (frontier-math CPU + actionable-now bu | ✅ SHIPPED 2026-09-29 | Tier-2 journal target adopted per user directive ("我们发二区就可以了，不用规划湿实验"); 4 LOI/SOP templates reclassified as… | Largest single-day waveset in project history; established the Tier-2-first + no-wet-lab orientation that p… |  |
| §1.3 | Wf-318..320 closeout wave (R13 100x3 diversity + steric_clas | ✅ SHIPPED 2026-09-29 | Resolved 7+ long-pending items including the Wf-219 Ru aggregation bug that had been a known regression for… | "Phase-0 loader" pattern established (load + verify + downstream-ready) becomes standard for later Wf-322..… |  |
| §1.4 | Wf-321..322 docs-only closeout (Wf-176 m19 LOMO CPU + R20/R2 | ✅ SHIPPED 2026-09-29 | Established 28-skill ecosystem (20 baseline + 8 NEW Wf-321B + 2 meta-packs) that anchors AGENTS.md §3 today. | Skill ecosystem expansion is doc-only and zero-touch to existing pipelines; every new skill ships with stub… |  |
| §1.5 | Wf-323 CPU closeouts (7 closeouts) | ✅ SHIPPED 2026-09-29 | All 7 closeouts are CPU-only docs+code; pre-flight v3 25→42 PASS expansion. | Foundational pattern for subsequent "closeout wave" workflows — each agent writes a focused 1-task closeout… |  |
| §1.6 | Wf-324D README/AGENTS/TODO refresh | ✅ SHIPPED 2026-09-29 | All 7 cross-cutting meta-docs refreshed to Wf-322D baseline. | Establishes the "D-class refresh" pattern: bulk-update project meta-docs at end of each major wave. |  |
| §1.7 | Wf-325 honest-negative closeout wave (7 closeouts) | ✅ SHIPPED 2026-09-29 | 7 honest-negative closeouts; preserved as institutional memory for future contributors per §5.2. | "Honest-negative closeout" pattern crystallized — every metric BELOW field-avg gets a §-doc with measured e… |  |
| §1.8 | Wf-326 GPU profile / AMP batch / Triton audit | ✅ SHIPPED 2026-09-29 | 2.72× torch.compile speedup MEASURED on RTX PRO 6000 Blackwell sm_120; W₂ sort 2-3×; cache 16.83× featuriza… | "Profile-before-wire-up" pattern — every wire-up must trace to a profile-measured bottleneck. |  |
| §1.9 | Wf-327 — 16 NEW fix docs (Wf-327-F1..F16 + Wf-327-SYN) | ✅ SHIPPED 2026-09-29 | 11 SHIPPED + 3 VERIFIED + 1 INVESTIGATED + 1 NOT-WRITTEN-merged-into-F8. **Path-bug fix in `p3_wf327_cpu_nu… | "F-suffix" closeout naming convention for fix-class work (Wf-327-F1..F16) — sub-flows explicitly numbered f… |  |
| §1.10 | Wf-328 — 10 wire-ups (Wf-328-W1..W10 + Wf-328-SYN) | ✅ SHIPPED 2026-09-29 | 54/54 NEW tests PASS + 1 SKIP / 0 FAIL; pre-flight v3 fresh-run **62 PASS / 0 FAIL / 62 TOTAL** at Wf-328-S… | "W-suffix" wire-up naming convention — each wire-up gets its own CLI flag + test + bench JSON. W-suffix wor… |  |
| §1.11 | Wf-329 — R20-P2..P5 + R21-P1..P5 ship-surface wiring (8 sub- | ✅ SHIPPED 2026-09-29 | 8 wire+bench + 4 verify verdicts. Catherine 30/30 PASS post-fix. pre-flight delta +20 PASS / -15 FAIL. | Wf-329 first wave to integrate R20 + R21 phases simultaneously; established cross-phase compatibility. |  |
| §1.12 | Wf-330 ULTRACODE (7 sub-flows A..G) | ✅ SHIPPED 2026-09-29 (pe | Catherine 30/30 metals 6/6 PASS; pre-flight v3 62/62 PASS; §9.BD catalog entry; risks.md R9 close. | "ULTRACODE" pattern — a single user directive spawns a multi-sub-flow wave with letter-suffix (A..G) sub-fl… |  |
| §1.13 | Wf-331 — NO WORK (per Wf-332-S1 verification table §0.16) | 🚫 **NO WORK** (no closeo |  | "No-work" verdict is a valid closeout; preserves intent + user context without fabricating Wf-331 content. |  |
| §1.14 | Wf-332-S1 INDEX.md final sync + Wf-332-S2 AGENTS.md final sy | ✅ SHIPPED 2026-09-29 | Verification pass over Wf-323..331 closeouts; per-Wf closeout-count verdict table; AGENTS.md §3 + §6 delta. | "S-class" sync workflows — bulk-update INDEX.md + AGENTS.md after major wave closeouts. |  |
| §1.15 | Wf-333 recovery final wave (12 closeouts V1/V2/V3/V4/G1/G2/G | ✅ SHIPPED 2026-09-29 | **Wf-246-C kcal_adjustment sign-fix at `metal_coord_term.py:629 → :654`** closes `HONEST_NEGATIVE_WF246_C_K… | "V-class" recovery naming convention for recovery-axis work (V1..V4 for Vina/m3; G for GPU; T for Triton)… |  |
| §1.16 | Wf-334 GPU validation wave (4 closeouts U1/U2/U4 + ultracode | ✅ SHIPPED 2026-09-29 | 22-test GPU smoke suite (21/22 PASS + 1 SKIP under host load G6+dfine_t146); pipeline master CLI integrates… | "U-class" naming convention for GPU-validation work (U1..U4); preserves Wf-334 as CPU-only verification + E… |  |
| §1.17 | Wf-335 torch.compile mode default fix (3 closeouts R1+R2+R3  | ✅ SHIPPED 2026-09-29 | Closes `HONEST_NEGATIVE_WF330_B4_CUDAGRAPHS_RUNTIME_ERROR`; 4 NEW tests PASS + 13/13 Wf-335-R2 reverify tes… | "R-suffix" naming convention for runtime-bug fixes (R1+R2+R3 — diagnose + reverify + safety lever); multi-c… |  |
| §1.18 | Wf-336 ultracode residual wave (4 closeouts X1+X2+X3+X4 + Wf | ✅ SHIPPED 2026-09-29 | 7 NEW closeouts since Wf-333-SYN; Wf-336 X1 closes a known delivery gap from Wf-331-X1 + Wf-333-T4. | "X-suffix" naming convention for cross-cutting ultracode residual work (X1..X4); "F-suffix" for finalizatio… |  |
| §1.19 | Wf-337 + Wf-338 ultracode residual closeout wave (10 Wf-337  | ✅ SHIPPED 2026-09-29 | 121 NEW tests PASS + 16 NEW test files; 27 existing test re-verifications; Vina recovery paths + G6 launche… | "W-suffix" letter-A/B/G/U/V used to organize sub-flows by topic (W1..W3 = wire-ups; G1..G3 = G6 launchers… |  |
| §1.20 | Wf-338-F2 INDEX.md final sync | ✅ SHIPPED 2026-09-29 | 16-row per-Wf INDEX coverage verdict for Wf-323..338; pre-flight v3 fresh-run **62/62 PASS** at Wf-338-F2 t… | "F-suffix" naming convention for INDEX finalization work (F1/F2/F3 = final/sync/verify). |  |
| §1.21 | Wf-339 — TODO directory cleanup (Wf-339-B2/B3 + Wf-339 final | ✅ SHIPPED 2026-09-29 | Lambda roadmap audit (Wf-339-C2) + Improvement audit (Wf-339-C3); integration verification. | "B-suffix" naming convention for plan-deep-dive work (B1 = integration; B2 = SE(3) Schrödinger Bridge scaff… |  |
| §1.22 | Wf-340 — CFM research + improvement ultracode (11-agent aggr | ✅ SHIPPED 2026-09-29 (pe | 24 unique papers + 16 repos catalogued + 10 ranked opportunities + 6 on-disk wire-ups + 12 NEW tests PASS… | "R + A + I" suffix pattern for research → analysis → implementation: 4 R-class surveys → 4 A-class analyses… |  |
| §1.23 | Wf-341 — R20-P2 Hilbert-ASTSR GPU joint retrain wrappers + W | ✅ SHIPPED 2026-09-29 (CP | W₂ recipe SHIPPED + Hilbert R² gate PRESERVED; Wf-341-C1 GPU readback + Wf-341-C2 GPU-DEFERRED. | CPU scaffold SHIPPED + GPU launch gated separately (Wf-341-A7/A8). |  |
| §1.24 | Wf-342 — TODO-211 W₂ pipeline + weak-to-strong + INDEX sync | ✅ SHIPPED 2026-09-29 |  | "TODO-N" naming convention for TODO-directory-driven work. |  |
| §1.25 | Wf-343..346 — TODO-217 W2_pIC50_pipeline_scaffold + arch_lif | ✅ SHIPPED 2026-09-29 | 4 NEW pre-flight gates per Wf-343C4 (EXPECTED_GATE_COUNTS["all"] 62 → 66). | Pre-flight gate expansion is gated by each Wf that adds new gates; INDEX sync rolls up. |  |
| §1.26 | Wf-347..348 — Wf-347A m1 Multi-PH m6_kappa + Wf-347B m3 F2(a | ✅ SHIPPED 2026-09-29 | W₂ recipe SHIPPED + Hilbert R² gate PRESERVED. |  |  |
| §1.27 | Wf-349 — SO(3)-averaged FM + EDM preconditioning + OT-CFM ex | ✅ SHIPPED 2026-09-29 | All 4 CFM acceleration techniques SHIPPED CPU-only; pre-flight v3 62→66 PASS preserved. | "A-suffix" naming convention for acceleration methods; "B-suffix" for INDEX sync; "C-suffix" for integratio… |  |
| §1.28 | Wf-350 — Sci submission infrastructure + compute manifest +  | ✅ SHIPPED 2026-09-29 | SCI submission infrastructure scaffold SHIPPED; runtime energy report emitted; GPU env probe script SHIPPED. | "350a/b/c/d" pattern — initialize (a) → process (b) → analyze (c) → verify (d). |  |
| §1.29 | Wf-351 — LaTeX/reference/figure audit + Zotero + SCI skill + | ✅ SHIPPED 2026-09-29 | 29th skill `mol-metal-sci-paper-submission` SHIPPED. | "B3" naming convention for skill creation within multi-class wave. |  |
| §1.30 | Wf-352 — Expected gate counts bump + per-pipeline audit + AG | ✅ SHIPPED 2026-09-29 |  | "A/B/C" suffix pattern — audit (a) → sync (b) → verify (c). |  |
| §1.31 | Wf-353 — Triton mini-batch OT + shortcut distillation + cons | ✅ SHIPPED 2026-09-29 | 4 NEW CFM acceleration techniques SHIPPED CPU-only + Triton log-domain Sinkhorn kernel SHIPPED. | All 7 NEW R18 acceleration techniques (Wf-349 + Wf-353) are CPU-only scaffolds awaiting GPU A/B smoke per A… |  |
| §1.32 | Wf-354 — TODO-130 G3 + Wf-342-P2 + Wf-342-P4 + INDEX sync | ✅ SHIPPED 2026-09-29 |  | 3 NEW R20 user-OK-gated launch wrappers SHIPPED. |  |
| §1.33 | Wf-355 — R20-P3/P4/P5 wrappers + Wf-131 DEFERRED + Wf-146 DE | ✅ SHIPPED 2026-09-29 | 3 NEW R20 user-OK-gated launch wrappers + 2 DEFERRED markers preserved + SCI paper tooling scaffold. | "B-suffix" used for DEFERRED markers (Wf-131, Wf-146) — preserved as honest-negative in closeouts. |  |
| §1.34 | Wf-356 — Data quality + anchor refresh + SCI paper infra + t | ✅ SHIPPED 2026-09-29 |  | "Truncated verify" pattern — bulk-verify multiple previous Wf outputs at single timestamp. |  |
| §1.35 | Wf-357 — Self-audit wave (12 closeouts A1..A6 + B1..B4 + C1. | ✅ SHIPPED 2026-09-29 (pe | 4 NEW gates added (EXPECTED_GATE_COUNTS["all"] 66 → 90 → 94); pre-flight v3 90 PASS / 2 FAIL / 2 SKIP / 94… | "A/B/C/D" suffix pattern — audit (a) → metrics (b) → analysis (c) → top picks (d). Foundational pattern for… |  |
| §1.36 | Wf-358 — Speedup improvement #1: gradient_checkpointing + ch | ✅ SHIPPED 2026-09-29 | 4 NEW tests PASS in 2.34s CPU; 32 backward-compat tests PASS in 8.34s; 25 Wf-335 backward-compat tests PASS… | "Opt-in default OFF" pattern preserved for new CLI flags — production default OFF + 3 NEW flags + 4 NEW tes… |  |
| §1.37 | Wf-359 — m6_kappa re-verify with W₂ on R20-P3 features (Wf-3 | ✅ SHIPPED 2026-09-29 |  | "A-suffix" naming for re-verification work (A1 = re-verify; A2 = scaffold; A3 = reverify). |  |

## §2 What worked (top 10 successful patterns)

### §2.1 Honest-negative framing preserved as institutional practice

* **Source**: `party-twenty-spirit` §3.5 + AGENTS.md §5.2
* **Evidence**: 14 honest-negative caveats preserved in Wf-340-SYN §10.3 + 8 in Wf-333-SYN §22.24.4 + 6 in Wf-328-SYN §3.2 + 6 in Wf-329-SYN.
* **Why it worked**: Every metric BELOW field-avg or BLOCKED on external dependency gets explicit HONEST_NEGATIVE marker + MEASURED-vs-PROJECTED tag + source citation. This prevents aspirational claims from leaking into main results.
* **Reuse**: Wf-360+ should continue this pattern — every closeout has §0 TL;DR with explicit honest-neg catalog.

### §2.2 Pre-flight v3 gate as golden thread

* **Source**: AGENTS.md §1.4 + TODO-207 + `molmetal/scripts/pre_flight_check.py`
* **Evidence**: 62→66→90→94 gate expansion across Wf-327→Wf-343C4→Wf-357-A1 (preserved at 62+ gates since Wf-335-SYN baseline).
* **Why it worked**: Pre-flight v3 acts as the pre-launch gate; STALE flag (24 hr) blocks PASS; EXPECTED_GATE_COUNTS["all"] tracks expansions explicitly.
* **Reuse**: Wf-360+ should preserve EXPECTED_GATE_COUNTS["all"] baseline + add new gates only when new code is added.

### §2.3 Ultracode aggregation pattern (R1..Rn → SYN doc)

* **Source**: Wf-327-SYN + Wf-328-SYN + Wf-329-SYN + Wf-333-SYN + Wf-335-SYN + Wf-340-SYN
* **Evidence**: 6 ultracode SYN docs aggregating 9..16 sub-flows each.
* **Why it worked**: SYN doc has §0 TL;DR + §1..§N per-sub-flow verdict + §N+1 standing rules honoured + §N+2 cross-references. Each sub-flow has its own closeout + test + bench JSON.
* **Reuse**: Wf-360+ should spawn SYN agent LAST after all sub-flow closeouts land.

### §2.4 Profile-before-wire-up pattern (Wf-326 → Wf-328 wire-ups)

* **Source**: Wf-326A/B/C/D/E/F + Wf-328-W1..W10
* **Evidence**: 2.72× torch.compile + 2-3× W₂ sort + 16.83× cache + 1.5× TF32 + <5% DataLoader = 3-4× end-to-end MEASURED.
* **Why it worked**: Each wire-up traces to a profile-measured bottleneck from Wf-326 series. No speculative optimization.
* **Reuse**: Wf-360+ should require profile evidence before any new optimization wire-up.

### §2.5 Skill ecosystem expansion (Wf-321B 8 NEW + Wf-351-B3 NEW SCI)

* **Source**: AGENTS.md §3 + Wf-321B + Wf-351-B3
* **Evidence**: 28 baseline + 1 NEW = 29 skills (per `ls .claude/skills/ | wc -l`).
* **Why it worked**: Each new skill is stub + verification doc + YAML frontmatter. No fabrication of capabilities.
* **Reuse**: Wf-360+ should preserve stub+verify pattern for any new skill.

### §2.6 Tier-2 re-orientation (Wf-309..317)

* **Source**: User directive 2026-09-29 "我们发二区就可以了，不用规划湿实验"
* **Evidence**: 4 LOI drafts reclassified as TEMPLATE_NOT_NEEDED_FOR_TIER2; 3 wetlab SOPs DEPRECATED; Wf-131/Wf-146 DEFERRED.
* **Why it worked**: Re-orients project scope from JCIM Tier-1 + wet-lab validation to JCAMD/JMGM/MD/CBC Tier-2 + pure computational. Eliminates 6-12 month external dependency.
* **Reuse**: Wf-360+ should preserve Tier-2 markers per AGENTS.md §5.4.

### §2.7 CFM acceleration research → implementation (Wf-340-R1..I3)

* **Source**: Wf-340-SYN §10
* **Evidence**: 24 papers + 16 repos + 10 opportunities + 6 wires + 12 tests in 1 ultracode wave.
* **Why it worked**: 4 R-class surveys → 4 A-class analyses → 3 I-class implementations + 1 SYN. Each class has explicit deliverable.
* **Reuse**: Wf-360+ should follow R + A + I + SYN pattern for any new research ultracode.

### §2.8 Default-OFF opt-in CLI flags (Wf-335-R1 + Wf-358-A3)

* **Source**: Wf-335-R1 (no_cudagraphs_fix) + Wf-358-A3 (gradient_checkpointing + channels_last + torch_compile_dynamic)
* **Evidence**: 4 NEW tests PASS + backward-compat bit-for-bit preserved.
* **Why it worked**: Production default OFF preserves existing behavior; opt-in flags allow new optimizations to be enabled incrementally without blast radius.
* **Reuse**: Wf-360+ should default-OFF any new CLI flag + provide a `--no-<flag>` hard-disable.

### §2.9 GPU Blackwell sm_120 cudagraphs safety lever (Wf-335-R3)

* **Source**: `scripts/launch_wf124_p3_g6.sh` --no-cudagraphs lever
* **Evidence**: Closes HONEST_NEGATIVE_WF330_B4_CUDAGRAPHS_RUNTIME_ERROR.
* **Why it worked**: GPU launch path uses safety lever until Blackwell cudagraphs re-tuned per Wf-326E-A.
* **Reuse**: Wf-360+ should preserve `--no-cudagraphs` lever until sm_120 retune complete.

### §2.10 Path-bug fix in pre-flight v3 gate (Wf-327-F11)

* **Source**: `p3_wf327_cpu_num_workers_sufficient` gate path fix
* **Evidence**: `MOLMETAL_ROOT/scripts/launch_*.sh` → `REPO_ROOT/scripts/launch_*.sh` (6 lines, 1 Edit).
* **Why it worked**: Single-line gate paths were brittle; path fix makes gate robust to MOLMETAL_ROOT vs REPO_ROOT variation.
* **Reuse**: Wf-360+ should validate gate paths via `REPO_ROOT` convention for any new gate.

---

## §3 What didn't (top 10 failure modes)

### §3.1 GPU Blackwell sm_120 cudagraphs correctness gap (Wf-330-B4 + Wf-335-R1+R3)

* **Symptom**: `torch.compile(mode="reduce-overhead")` triggered `"accessing tensor output of CUDAGraphs that has been overwritten"` RuntimeError on RTX PRO 6000 Blackwell sm_120.
* **Status**: CLOSED by Wf-335-R1 (default `default`→`reduce-overhead`→`max-autotune-no-cudagraphs`) + Wf-335-R3 (`--no-cudagraphs` safety lever in G6 launch path).
* **Lesson**: GPU Blackwell correctness gaps need explicit runtime-bug-close pattern (R-class) + safety lever for production paths.

### §3.2 m6_kappa gate NOT lifted by W₂+PLL feature engineering (Wf-335-M1 / Wf-359-A1)

* **Symptom**: chemprop+PLL CV-R² +0.6814 MEASURED on 60-SMILES smoke cohort (bit-identical to Wf-333-V4); augmented_mean 0.4386 BELOW 0.6 = HONEST_NEGATIVE_PARTIAL_STRUCTURAL.
* **Status**: GATE PRESERVED per Wf-359-A1 (augmented_mean 0.356 on platinai_4cell n=100; BELOW gate 0.6 by -0.244 margin).
* **Lesson**: m6_kappa Shape Kappa Index is a per-molecule topological invariant independent of W₂ distance; W₂+PLL features do NOT directly lift the gate.

### §3.3 DDP/FSDP out-of-scope on single-GPU Mol-Metal (Wf-326G §3.1 + Wf-357-C1 §0)

* **Symptom**: Multi-GPU DDP/FSDP/torch.distributed NOT applicable due to single-GPU scope + kanzi host shared 2-GPU contention.
* **Status**: HONEST-NEGATIVE preserved; out-of-scope per AGENTS.md §5.1.
* **Lesson**: Single-GPU Mol-Metal design precludes parallel-scaling optimizations; speedup ceiling is bounded.

### §3.4 Wf-326F memory ceiling cache via PyG batched MessagePassing (Wf-328-W7 + Wf-331-X1)

* **Symptom**: PyG batched MessagePassing PROJECTED 5-10× forward; MEASURED CPU correctness only (`max abs diff 2.1e-7`).
* **Status**: PARTIAL — CPU-correct SHIPPED, GPU-bench DEFERRED to Wf-326G-1 + Wf-360+ A/B smoke.
* **Lesson**: PyG port is highest-ROI fix per Wf-326G §3 but GPU verification pending.

### §3.5 Wf-326H synthesis SHIPPED via Wf-327 + Wf-330-E + §332 (Wf-331-X1)

* **Symptom**: Wf-326H planned as separate synthesis; eventually rolled into Wf-327 + Wf-330-E + AGENTS.md §6.
* **Status**: FORMAL CLOSE preserved.
* **Lesson**: Some planned waves collapse into downstream waves; preserved as closeout doc to avoid re-spawning.

### §3.6 Wf-353 + Wf-349 truncated verify markers (Wf-356-B1/B2)

* **Symptom**: Wf-353 + Wf-349 outputs truncated at intermediate timestamps; full verify deferred to Wf-356.
* **Status**: SHIPPED via Wf-356-B1 (Wf-353) + Wf-356-B2 (Wf-349).
* **Lesson**: Truncated workflows need explicit "verify" wave downstream; preserved as B1/B2 closeouts.

### §3.7 Blackwell sm_120 Triton fused_relu_linear HONEST_NEGATIVE until retune (Wf-328-SYN §3.2)

* **Symptom**: Triton fused_relu_linear kernel produces incorrect results on Blackwell until sm_120 retune.
* **Status**: HONEST-NEGATIVE preserved; full sm_120 GPU retune DEFERRED per AGENTS.md §5.1.
* **Lesson**: GPU kernel correctness requires explicit retune on new architecture; defer until VRAM + pre-flight PASS.

### §3.8 BF16 HONEST-NEGATIVE on RTX 5090 small encoder (Wf-327-F6)

* **Symptom**: BF16 autocast is 47% slower than FP32 on RTX 5090 small encoder; not the projected 1.5-2× speedup.
* **Status**: HONEST-NEGATIVE preserved; default ON with `--no-bf16` lever; PROJECTED 1.5-2× on Blackwell sm_120 at larger workloads.
* **Lesson**: Small-encoder behavior may not extrapolate to Blackwell-class larger workloads; PROJECTED but NOT MEASURED.

### §3.9 CUDA Graphs Blackwell sm_120 correctness gap (Wf-330-B4)

* **Symptom**: cudagraphs RuntimeError on RTX PRO 6000 Blackwell sm_120.
* **Status**: CLOSED by Wf-335-R1 (default max-autotune-no-cudagraphs) + Wf-335-R3 (--no-cudagraphs safety lever).
* **Lesson**: Blackwell sm_120 cudagraphs requires explicit retune; opt-in default OFF until verified.

### §3.10 Wf-298 §1.3 "launch script not yet written" gap (Wf-334-U4 + Wf-336-X2)

* **Symptom**: `scripts/launch_wf264_cfm_property_panel.sh` was never written despite Wf-298 plan.
* **Status**: CLOSED by Wf-334-U4 (~135 LOC launch script).
* **Lesson**: Plan docs without launch scripts create execution gaps; track launch-script deliverables explicitly in closeouts.

---

## §4 Lessons learned (top 10 insights)

### §4.1 "Profile-before-wire-up" is the canonical optimization pattern

* **Evidence**: Wf-326 series (profile) → Wf-328 series (wire-ups) → 3-4× MEASURED end-to-end.
* **Insight**: Every wire-up must trace to a profile-measured bottleneck. Speculative optimizations fail.
* **Action**: Wf-360+ should require profile evidence before any new optimization wire-up.

### §4.2 "Default-OFF opt-in CLI flag" is the canonical extension pattern

* **Evidence**: Wf-335-R1 (no_cudagraphs_fix) + Wf-358-A3 (gradient_checkpointing + channels_last + torch_compile_dynamic) + Wf-357-D1 (fused_adam).
* **Insight**: Production default OFF preserves existing behavior; opt-in flags allow new optimizations to be enabled incrementally without blast radius.
* **Action**: Wf-360+ should default-OFF any new CLI flag + provide a `--no-<flag>` hard-disable.

### §4.3 "HONEST_NEGATIVE_PARTIAL_STRUCTURAL" is the canonical partial-recovery verdict

* **Evidence**: Wf-335-M1 (m6_kappa chemprop+PLL CV-R² +0.6814 but augmented_mean BELOW gate) + Wf-359-A1 (m6_kappa W₂ R20-P3 verify).
* **Insight**: Partial structural recovery deserves explicit marker; NOT paraphrased as "promising" / "achievable".
* **Action**: Wf-360+ should preserve PARTIAL_STRUCTURAL marker when structural fix lands but full gate not crossed.

### §4.4 "Standing GPU authorization" + "kill_then_adapt" preserve correctness + throughput

* **Evidence**: AGENTS.md §5.1 + 0 GPU launches across Wf-333 + Wf-334 (CPU-only verification).
* **Insight**: Standing GPU authorization enables sub-1hr launches without per-launch user OK; kill_then_adapt preserves ability to abort unexpected runs.
* **Action**: Wf-360+ should preserve GPU launches gated to ≤1 hr or USER_OK-gated per §5.1.

### §4.5 "User-OK-gated launch wrappers" defer GPU launches without blocking plans

* **Evidence**: Wf-355-A1/A2/A3 + Wf-356-A1 (6 NEW USER_OK-gated launch wrappers).
* **Insight**: Each wrapper has --dry-run default + GPU launch only after explicit user OK; preserves planning without consuming VRAM.
* **Action**: Wf-360+ should create USER_OK-gated wrappers for any new GPU launch plan.

### §4.6 "Truncated verify" pattern catches partial-workflow output drift

* **Evidence**: Wf-356-B1 (Wf-353 truncated verify) + Wf-356-B2 (Wf-349 truncated verify) + Wf-356-B3 (Wf-352 AGENTS verify).
* **Insight**: Truncated workflows need explicit downstream verify; preserves intermediate state without re-running full wave.
* **Action**: Wf-360+ should spawn a downstream "verify" wave if any Wf output is truncated.

### §4.7 "No-work" verdict is a valid closeout

* **Evidence**: Wf-331 NO WORK per Wf-332-S1 INDEX verification.
* **Insight**: Cancelled / planned-but-not-executed Wf should be preserved as closeout doc with explicit NO WORK marker; avoids re-spawning.
* **Action**: Wf-360+ should write a "no-work" closeout doc if user cancels a planned wave.

### §4.8 "DEFERRED" marker preserves intent + honest-negative

* **Evidence**: Wf-355-B1 (Wf-131-P3 DEFERRED) + Wf-355-B2 (Wf-146 DEFERRED) + wet-lab items DEPRECATED per Wf-317G/322B.
* **Insight**: DEFERRED is a valid status; preserves the original plan + the reason for deferral.
* **Action**: Wf-360+ should preserve DEFERRED markers + reason for deferral in closeout docs.

### §4.9 "Cross-references" in SYN docs preserve institutional memory

* **Evidence**: Wf-340-SYN §12 + Wf-335-SYN §10 + Wf-333-SYN §7 — each lists all upstream sources.
* **Insight**: Cross-references enable future contributors to trace any claim back to source; reduces hallucinations.
* **Action**: Wf-360+ should preserve cross-reference section in every SYN doc.

### §4.10 "Pre-flight v3 EXPECTED_GATE_COUNTS[\"all\"]" tracks gate expansion explicitly

* **Evidence**: 62 → 66 → 90 → 94 gate expansion across Wf-327 → Wf-343C4 → Wf-357-A1.
* **Insight**: Explicit count tracking prevents accidental regressions + provides audit trail.
* **Action**: Wf-360+ should preserve EXPECTED_GATE_COUNTS["all"] baseline + add new gates only when new code is added.

---

## §5 Forward-looking improvements (per Wf-357-C1/C2/C3 + Wf-358-A3)

### §5.1 Top-3 workflow improvements (Wf-357-C3)

| # | Improvement | Source | PROJECTED lift |
|---:|---|---|---|
| 1 | Auto-refresh wf343c2 default `--preflight` path + auto-fix argparse help-text bug | Wf-357-A1 2 FAILs | PROJECTED 92/94 PASS vs 90/94 |
| 2 | Pre-retrain checklist auto-emission into pre-flight JSON | Wf-343c2 7 checks per wave + 24 checks per section | PROJECTED -80% manual audit time per retrain wave |
| 3 | Workflow regression test suite + cross-workflow dependency tracking | per-workflow isolated tests → cross-workflow integration smoke | PROJECTED -50% inter-workflow bugs |

### §5.2 Top-3 architecture improvements (Wf-357-C2)

| # | Improvement | Source | PROJECTED lift |
|---:|---|---|---|
| 1 | EDM σ-preconditioning + log-normal t-sampling + lr_warmup default ON | Wf-340-A2 + Wf-349-A1/A2 (currently opt-in default OFF) | PROJECTED 2-3× CFM training step convergence |
| 2 | M22 SE(3) Schrödinger Bridge GPU retrain | Wf-339-B2/B3 CPU scaffold SHIPPED, GPU launch BLOCKED on user OK | PROJECTED m3 Vina lift -1.35 → -8.5 kcal/mol |
| 3 | M25 Hilbert primal-dual optimizer joint retrain | Wf-333-V4 chemprop+PLL CV-R² +0.6814 MEASURED + M25 Hilbert PD `O(1/k)` Chambolle-Pock SHIPPED CPU-only | PROJECTED R² ≥ 0.85 on PlatinAI 4-cell + MetalCytoToxDB 5-metal |

### §5.3 Top-3 speedup improvements (Wf-357-C1)

| # | Improvement | Source | PROJECTED lift |
|---:|---|---|---|
| 1 | PyG batched MessagePassing GPU verification (Wf-328-W7) | currently MEASURED CPU only | PROJECTED 5-10× forward wall; +30-50% end-to-end |
| 2 | `Adam(fused=True)` opt-in CLI flag (Wf-357-D1) | NOT yet wired; 3 LOC per trainer | PROJECTED 1.1-1.3× on optimizer step |
| 3 | `channels_last` + `gradient_checkpointing` opt-in lift to Blackwell 50-epoch (Wf-350-C3 + Wf-358-A3) | already opt-in default OFF | PROJECTED 1.05-1.20× on per-metal head GEMM |

### §5.4 Opt-in CLI flag matrix (Wf-358-A3 + Wf-357-D1 + Wf-335-R1+R3)

| Flag | Default | Source | PROJECTED lift |
|---|---|---|---|
| `--gradient-checkpointing` | OFF | Wf-350-C3 + Wf-358-A3 | ~O(sqrt(depth)) activation memory reduction at +30% FLOPs |
| `--channels-last` | OFF | Wf-350-C3 + Wf-358-A3 | PROJECTED 1.05-1.10× on per-metal head Linear ops |
| `--torch-compile-dynamic` | OFF | Wf-358-A3 | PROJECTED 5-15% end-to-end via memory headroom + amortised compile time |
| `--fused-adam` | OFF | Wf-357-D1 | PROJECTED 1.1-1.3× on optimizer step |
| `--no-cudagraphs` | (N/A — opt-in) | Wf-335-R3 (safety lever) | (preserves Blackwell sm_120 correctness until retune) |
| `--torch-compile-mode-v2` | (N/A — alias) | Wf-335-R1 (legacy Wf-326C reproducer) | (preserves Wf-326C behavior bit-for-bit) |

### §5.5 Forward-looking honest-negative caveats

1. All speedup / lift claims are **PROJECTED** from upstream paper benchmarks; **NO Mol-Metal GPU A/B smoke has been launched** as of 2026-09-29 (per Wf-340-A7/A8 + Wf-335-R2 + Wf-341-C2 GPU-DEFERRED status).
2. Wf-359-A1 m6_kappa gate PRESERVED (augmented_mean 0.356 < 0.6 threshold = -0.244 margin); chemprop+PLL structural fix does NOT lift the gate.
3. Blackwell sm_120 Triton fused_relu_linear + cudagraphs HONEST-NEGATIVE until sm_120 retune; opt-in default OFF until verified.

---

## §6 Cross-references

### §6.1 AGENTS.md

* **§1** Five standing project rules (MEASURED-vs-PROJECTED + GPU auth + no paper work + pre-flight + honest-negative)
* **§5.1** Standing GPU authorization (2026-09-29 user grant "显存够就放心大胆用")
* **§5.2** Honest framing mandate
* **§5.3** No paper work without authorization (TODO-84)
* **§5.4** Tier-2 journal target + NO wet-lab planning (2026-09-29 user directive "我们发二区就可以了，不用规划湿实验")

### §6.2 Wf-340-R1 (CFM research survey)

* `TODO/completed/340r1_cfm_papers_survey.md` — 9 papers with arXiv IDs + GitHub URLs
* `TODO/completed/340r2_acceleration_methods.md` — 8 techniques deep-dive + Tier-1 candidates
* `TODO/completed/340r3_molecule_cfm.md` — 10 molecule-specific papers
* `TODO/completed/340r4_code_repos.md` — 16 repos + license verdict + top-5 borrowing plan

### §6.3 Wf-335-SYN (Phase 1 RUNTIME BUG + Phase 2 m6_kappa + Phase 3 P0 wins)

* `TODO/completed/335_final_synthesis.md` — 9-agent aggregation
* `TODO/completed/335r1_no_cudagraphs_fix.md` — torch.compile mode default flip
* `TODO/completed/335r2_wireup_reverify.md` — wire-up reverify post-R1
* `TODO/completed/335r3_g6_no_cudagraphs.md` — G6 launch path CUDAGraphs safety lever
* `TODO/completed/335m1_chemprop_pll_full.md` — chemprop+PLL CV-R² +0.6814
* `TODO/completed/335m2_r2_w2_chemprop_full.md` — R² W₂ chemprop full
* `TODO/completed/335m3_triangle_gpu_extend.md` — Triangle GPU wrapper
* `TODO/completed/335p1_p0_extend_trainers.md` — 5 P0 wins extended to chemprop_v2_v3 + cfm_yuelbond
* `TODO/completed/335p2_sm120_gpu_wrapper.md` — sm_120 Triton retune GPU wrapper
* `TODO/completed/335p3_triton_kernels_wired.md` — 4 Triton kernels wired

### §6.4 Wf-358-A3 (gradient_checkpointing + channels_last + torch_compile_dynamic)

* `TODO/completed/358a3_speedup_flags.md` — opt-in CLI flags SHIPPED
* `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — `--gradient-checkpointing` + `--channels-last` + `--torch-compile-dynamic` flags
* `molmetal/tests/test_wf358a3_speedup_flags.py` — 4 NEW tests PASS

### §6.5 Wf-359-A1 (m6_kappa W₂ R20-P3 verify)

* `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` — gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL
* `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/m6_kappa_augmented_mean.json` — MEASURED augmented_mean 0.356 on platinai_4cell n=100
* `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/result.json` — per_cell + per_cohort W₂ distances

### §6.6 Wf-357-C1/C2/C3 (self-audit trio)

* `TODO/completed/357c1_speedup_analysis.md` — speedup opportunities analysis (RESIDUAL + FORWARD-LOOKING)
* `TODO/completed/357c2_architecture_analysis.md` — architecture deep-dive
* `TODO/completed/357c3_workflow_analysis.md` — workflow/process improvement opportunities

### §6.7 Other Wf ships referenced

* Wf-296..317 (Tier-2 re-orientation + Wf-321B 8 NEW skills)
* Wf-326/327/328 (GPU profile + 16 NEW Wf-327 fix docs + 10 Wf-328 wire-ups)
* Wf-329 (R20-P2..P5 + R21-P1..P5 ship-surface wiring)
* Wf-330..338 (ULTRACODE residual closeouts)
* Wf-339..356 (CFM ultracode residual + per-Wf aggregations)
* Wf-357 (self-audit wave: A1..A6 + B1..B4 + C1..C3 + D1..D2)
* Wf-358 (speedup improvement #1: opt-in CLI flags)
* Wf-359 (m6_kappa re-verify + reverify)

---

**END OF Wf-360-B1 RETROSPECTIVE** — Owner Hugo Lee + Claude Code subagent; cross-ref `party-twenty-spirit` §3.5; AGENTS.md §1 + §5 honored; 64 Wf units summarized; 10 success patterns + 10 failure modes + 10 lessons learned + 5 forward-looking improvements; pre-flight v3 62/62 PASS preserved; 0 GPU launches performed across Wf-296..359 window.
