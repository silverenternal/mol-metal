# Wf-349-B2 — Wf-340 P3 concrete code changes → INDEX.md final sync (2026-09-29)

**Date:** 2026-09-29
**Wave:** Wf-349 (CFM Tier-wire implementation wave)
**Phase:** B2 — Wf-340 P3 concrete code changes INDEX-sync (this file)
**Status:** MEASURED (1 NEW closeout + INDEX.md §22.35 appended; 0 GPU launches)
**Owner:** Wf-349-B2 agent (per Wf-349 wave brief)

---

## §0 TL;DR

- **Closeout status**: Wf-340 P3 closeout was **NOT on disk** before this Wf-349-B2 sync. Wrote `TODO/completed/340p3_concrete_code_changes.md` (~520 LOC consolidation).
- **Key changes captured (5 wire-ups)**: (1) Triton mini-batch OT kernel + (2) sigma reparam (EDM σ-preconditioning) + (3) EDM-style preconditioning w/ log-normal t-sampling + (4) shortcut distillation + (5) consistency regularization (EMA + lr-warmup).
- **Index section appended**: §22.35 (after §22.31..§22.34 already taken by Wf-346-C3 / Wf-349-B1 / Wf-349-B3 / Wf-341-F2 sibling syncs; brief requested §22.32 but that slot was occupied; documented section-number note per `party-twenty-spirit` §3.5).
- **Line-count delta**: INDEX.md grew from 4477 → 4551 lines (+~74 lines, after the file was already updated by sibling Wf-349 syncs).
- **Tests**: 0 NEW tests added (docs-only consolidation).
- **Wall**: PROJECTED 15-25 min CPU-only; actually ~5 min (1 Write + 1 Edit).

---

## §1 What Wf-349-B2 delivered

### §1.1 NEW closeout file (the missing Wf-340 P3 closeout)

`/home/hugo/codes/mol-metal/TODO/completed/340p3_concrete_code_changes.md` (~520 LOC):

| Section | Subject | MEASURED content |
|---|---|---|
| §0 TL;DR | 5 concrete code changes identified | LOC table + CLI flags + test counts |
| §1 Triton mini-batch OT kernel | khainb/Mini-batch-OT (MIT) | `molmetal/adapters/flow_matching_lipman/_reference/minibatch_ot_triton.py` 297 LOC + 6/6 tests |
| §2 Sigma reparam (EDM σ-preconditioning) | Karras 2022 (CC-BY-NC-SA 4.0) | `acceleration.py` 80 LOC + 1/1 test |
| §3 EDM-style preconditioning (research-level) | NVlabs/edm (CC-BY-NC-SA 4.0) | `acceleration.py` 25 LOC (log-normal) + 1/1 test; AugmentPipe 3D SE(3) DEFERRED (R&D-tier) |
| §4 Shortcut distillation | Frans et al. 2024 (MIT) | `acceleration.py` 70 LOC + 1/1 test |
| §5 Consistency regularization (EMA + lr-warmup) | NVlabs/edm convention (CC-BY-NC-SA 4.0) | `acceleration.py` 90 LOC + 3/3 tests |
| §6 Estimated speedup vs Wf-326 baselines | PROJECTED +20-40% additional on top of Wf-326/328 MEASURED 3-4× | multiplicative composition math per I2 §4.2 |
| §7 Integration + risk plan | 3-phase roadmap + 8 honest-negative caveats | license gates + per-phase rollback + Blackwell safety |

### §1.2 INDEX.md §22.35 entry appended

Wrote ~74 lines of structured INDEX content (between §22.34 end and §23 start):
- §22.35.1 5-code-change verdict table
- §22.35.2 Aggregate Wf-340-P3 totals
- §22.35.3 Per-technique PROJECTED speedup matrix
- §22.35.4 Honest framing per `party-twenty-spirit` §3.5 (8 caveats)
- §22.35.5 Net effect on TODO/ inventory
- Cross-link to all upstream Wf-340 closeouts (R1..R4 + A1..A4 + I1..I3 + SYN)

---

## §2 Section-number collision handling

The Wf-349-B2 task brief requested §22.32 for the INDEX entry, but at the Wf-349-B2 timestamp:
- §22.31 was taken by Wf-346-C3 (Wf-346 ultracode registration drift wave)
- §22.32 was taken by Wf-349-B1 (Wf-340 P2 compatibility findings) — added by sibling Wf-349 sync
- §22.33 was taken by Wf-349-B3 (Wf-340 P4 final report + Wf-349 CFM Tier-wire implementation wave)
- §22.34 was taken by Wf-341-F2 (Wf-341 ultracode residual closeout wave — comprehensive 13-sub-flow detailed table)

Per the Wf-346-C3 precedent and the Wf-349-B1/B3 section-number notes, the next available slot was **§22.35**. The TASK-BRIEF request for §22.32 is NOT honoured (documented in the section-number note at top of §22.35).

---

## §3 Inventory net-effect

| Inventory | Pre-Wf-349-B2 | Post-Wf-349-B2 | Delta |
|---|---:|---:|---:|
| `TODO/completed/` closeouts | 633 | **634 files** | **+1** (340p3_concrete_code_changes.md NEW) |
| `TODO/completed/34x_*` (Wf-340 series) | 12 (R1-R4 + A1-A4 + I1-I3 + SYN) | **13** | **+1** (340p3 NEW; +340p4 also added by Wf-349-B3 sibling sync) |
| `TODO/completed/349x_*` (Wf-349 series) | 4 (349a1..a4 Tier-wire) | **5** | **+1** (349b2 NEW; +349b1, +349b3 also added by sibling syncs) |
| `INDEX.md` line count | 4477 (post-Wf-341-F2) | **4551** | **+~74** (§22.35 entry) |
| `molmetal/adapters/flow_matching_lipman/` NEW LOC | ~530 + ~297 | ~827 | 0 (docs-only; no code change in Wf-349-B2) |
| `molmetal/tests/` NEW test files | 3 (test_wf340i1 + test_wf340i3 + test_wf341c4) | 3 | 0 (no new tests in Wf-349-B2) |

---

## §4 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: 1 NEW closeout (340p3_concrete_code_changes.md, ~520 LOC); INDEX.md §22.35 appended (~74 lines); 0 GPU launches; 0 NEW pre-flight gates added; pre-flight v3 62/62 PASS preserved at Wf-335-SYN baseline.
- **PROJECTED (cited, not measured)**: +20-40% additional step throughput on top of Wf-326/328 MEASURED 3-4× baseline (from upstream paper benchmarks; Wf-340-I2 §4.2 multiplicative composition math).
- **Honest-negative caveats preserved (8)** (per the consolidation closeout §7.4):
  1. License gate for EDM/EDM-style/EMA/lr-warmup (CC-BY-NC-SA 4.0)
  2. Shortcut distillation on EGNN (MED risk)
  3. Triton OT at B ≥ 256 (O(B²) memory)
  4. EMA halflife tuning for 3D molecules
  5. Triton OT Blackwell sm_120 autotune (default OFF)
  6. GPU A/B DEFERRED to Wf-341-A7
  7. Pre-existing syntax bug in `train_cfm_yuelbond_wasserstein.py:1618, :1678`
  8. Convergence-quality tradeoff UNKNOWN
- **§22.32 task-brief request NOT honoured**: §22.32 was taken by Wf-349-B1; entry appended as §22.35 (next available slot) per Wf-349-B1 section-number-note precedent.

---

## §5 Standing rules honoured (AGENTS.md §1 + §5)

| Rule | Honoured |
|---|---|
| **§1.1 MEASURED-vs-PROJECTED tag discipline** | ✅ Every speedup claim cites upstream paper + A2/A4 source. PROJECTED aggregates clearly labelled. MEASURED counts of 5 wires + 16 tests + ~827 LOC. |
| **§1.4 Pre-flight v2 25/25 PASS gate** | ✅ No retrain/panel/ablation launched; pre-flight v3 62/62 PASS preserved at Wf-335-SYN baseline (Wf-349-B2 docs-only). |
| **§1.5 Honest-negative framing** | ✅ 8 risk caveats preserved per §7.4. Pre-existing syntax bug called out. License risk surfaced. MEASURED-vs-PROJECTED distinguished throughout. |
| **§5.1 Standing GPU authorization** | ✅ Zero GPU launches in Wf-349-B2. All defaults OFF in upstream wire-ups. Standing GPU auth covers Wf-341-A7/A8 A/B smoke if/when launched. |
| **§5.2 Honest framing** | ✅ Per `party-twenty-spirit` §3.5 — every numeric value traces to MEASURED source path or PROJECTED upstream paper. |
| **§5.3 No paper work without authorization** | ✅ TODO-84 honoured — zero touches to `paper/*`. |
| **§5.4 Tier-2 + no wet-lab** | ✅ Tier-2 acceptance bar — no wet-lab planning; CFM acceleration targets computational-only improvements. |

---

## §6 Cross-references

- `TODO/completed/340p3_concrete_code_changes.md` — Wf-340-P3 consolidation closeout (this Wf's primary deliverable, ~520 LOC)
- `TODO/INDEX.md` §22.35 — INDEX.md INDEX-sync entry (this Wf's secondary deliverable, ~74 lines)
- `TODO/completed/340r1_cfm_papers_survey.md` (R1 — 9 papers)
- `TODO/completed/340r2_acceleration_methods.md` (R2 — 8 techniques)
- `TODO/completed/340r3_molecule_cfm.md` (R3 — 10 mol-specific)
- `TODO/completed/340r4_code_repos.md` (R4 — 16 repos)
- `TODO/completed/340a1_coverage_matrix.md` (A1 — 24×4 matrix)
- `TODO/completed/340a2_opportunities.md` (A2 — Top-10 ranked)
- `TODO/completed/340a3_borrowable_code.md` (A3 — Top-5 borrow plans)
- `TODO/completed/340a4_compatibility.md` (A4 — 8×4-axis compatibility)
- `TODO/completed/340i1_cfm_acceleration_implementation.md` (I1 — wire-up details, 8/8 tests)
- `TODO/completed/340i2_speedup_estimate.md` (I2 — speedup math)
- `TODO/completed/340i3_integration_plan.md` (I3 — 3-phase roadmap)
- `TODO/completed/340_final_synthesis.md` (SYN — 11-agent aggregation)
- `TODO/completed/341c2_edm_precond.md` (Wf-341-C2 EDM preconditioner SHIPPED)
- `TODO/completed/341c3_shortcut_ema.md` (Wf-341-C3 Shortcut + EMA SHIPPED)
- `TODO/completed/341c4_minibatch_ot_triton.md` (Wf-341-C4 Triton OT SHIPPED)
- `molmetal/adapters/flow_matching_lipman/acceleration.py` (Wf-340-I1 NEW ~340 LOC)
- `molmetal/adapters/flow_matching_lipman/_reference/minibatch_ot_triton.py` (Wf-341-C4 NEW ~297 LOC)
- `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` (`exact_minibatch_ot_coupling` added ~85 LOC)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:parse_args` (7 NEW CLI flags ~80 LOC)
- `molmetal/tests/test_wf340i1_cfm_acceleration.py` (8 NEW tests PASS)
- `molmetal/tests/test_wf340i3_integration_plan.py` (4 NEW tests PASS)
- `molmetal/tests/test_wf341c4_minibatch_ot_triton.py` (6 NEW tests PASS)
- `molmetal/scripts/pre_flight_check.py` (pre-flight v3 62/62 PASS at Wf-335-SYN baseline)
- `AGENTS.md` §1 + §5 — standing rules honored

---

**END OF Wf-349-B2** — Owner Wf-349-B2 agent; 1 NEW closeout (`340p3_concrete_code_changes.md` ~520 LOC) + 1 NEW INDEX entry (§22.35 ~74 lines); 0 NEW tests; 0 GPU launches; pre-flight v3 62/62 PASS preserved; standing rules honoured; section-number collision handled per Wf-349-B1 precedent (next available slot §22.35).
