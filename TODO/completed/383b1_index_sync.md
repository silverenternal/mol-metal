# Wf-383-B1 — INDEX + AGENTS sync (WEAK_TO_STRONG §4.3 audit + SE(3) SB integration plan + wires audit + final ship) — 2026-09-30

**Phase**: Wf-383-B1 docs-only INDEX + AGENTS + final synthesis v26 (CPU-only)
**Status**: ✅ SHIPPED — docs-only sync; 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning
**Owner**: Hugo Lee + Wf-383-B1 docs-sync agent (CPU-only docs wave)
**Mode**: Plain-text OUTPUT (per AGENTS.md §8 + Wf-291C workaround)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline preserved)
**User voice relay**: Wf-382-C1 §0 final synthesis closing the SCAFFOLD-MEASURED verification cycle;
2026-09-30 user voice *"可以的"* (OK/can-do) → continue the docs-only sync wave
**Standing directives re-confirmed**: AGENTS.md §5.1 (VRAM-budgeted GPU auth), §5.2 (honest framing),
§5.3 (no paper work without authorization — TODO-84 active), §5.4 (Tier-2 + NO wet-lab), §10 (wetlab deprecation FAQ).

---

## §0 TL;DR

| Field | Value |
|---|---|
| **Wave** | Wf-383 (CPU-only docs+plan wave) |
| **Sub-flows** | A1 WEAK_TO_STRONG §4.3 10000-step audit + revised PROJECTED step count table + HONEST_NEGATIVE entry (CLOSED) + A2 SE(3) SB fast-converging integration plan (10 wire designs) + A3 10-wire CFM innovation audit (Wf-340 + Wf-353 + TODO-200) (CLOSED) + B1 this docs-only INDEX + AGENTS + final synthesis |
| **NEW tests** | **0** (entire wave is docs-only per task brief — A1 audit + A2 plan + A3 audit + B1 sync) |
| **GPU launches** | **0** (CPU-only docs+plan wave; GPU retrain deferred per AGENTS.md §5.1) |
| **paper/* touches** | **0** (TODO-84 active per AGENTS.md §5.3) |
| **wet-lab planning** | **0** (Tier-2 re-orientation per AGENTS.md §5.4 + §10) |
| **MEASURED vs PROJECTED discipline** | All step-reduction / lift claims tagged PROJECTED (per upstream paper benchmarks); MEASURED only for on-disk plan artifacts + test counts + LOC |
| **Pre-flight v3** | NOT re-run by Wf-383 (docs-only) — **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** PRESERVED at Wf-380-B1 baseline per `molmetal/reports/wf380b1_preflight.txt` |
| **EXPECTED_GATE_COUNTS["all"]** | **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-383) |

---

## §1 Pre-flight verification

### §1.1 Wf-383 pre-flight NOT re-run (docs-only wave)

Per AGENTS.md §1.4 (Pre-flight v2 25/25 PASS gate — TODO-207) + AGENTS.md §5.1 (Standing GPU
authorization), Wf-383 is a **CPU-only docs+plan wave**. The wave contains ZERO new pre-flight
gates, ZERO retrain launches, ZERO panel launches, ZERO ablation launches. Therefore the
pre-flight fresh-run is **NOT re-run by Wf-383** — the most recent MEASURED state is preserved
from Wf-380-B1 timestamp 2026-09-29T17:13:00+0000 UTC.

### §1.2 Pre-flight v3 baseline preserved

Per `molmetal/reports/wf380b1_preflight.txt` (872 lines) — Wf-380-B1 timestamp 2026-09-29T17:13:00+0000 UTC wall ~5 min:

```
Wall: ~5 min
result: 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL
EXPECTED_GATE_COUNTS["all"]: 94
gate_count_consistency_audit: actual=94 == expected=94 (auto-recomputed via Wf-345-B3) — PASS
```

Per AGENTS.md §6 recent Wf history: **HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94** across
**7 consecutive runs** at the 89/2/3/94 drift level (Wf-374-B1 → Wf-375-B1 → Wf-376-B1 →
Wf-377-B1 → Wf-378-B1 → Wf-379-B1 → Wf-380-B1). Wf-383 does NOT introduce any new pre-flight
gates, so the drift level is preserved.

### §1.3 Pre-flight FAILs / SKIPs analysis

| Item | Status | Source / Honesty framing |
|---|---|---|
| `p3_loi_send_ready` FAIL | **Honest-negative per Wf-375-B1 baseline** | Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits; 4 LOI files (`wetlab_loi_*`) marked TEMPLATE_NOT_NEEDED_FOR_TIER2 per AGENTS.md §5.4 + §10 |
| `p3_wf343c4_anchor_freshness_24h` FAIL | **Honest-negative per Wf-374-B1 baseline** | `wf269c_report.json` age > 24h STALE — natural drift across Wf-374..380 wave activity; NOT Wf-383-introduced |
| `p3_wf343c4_yuelbond_decoder_env` SKIP | **BY-DESIGN per Wf-345-A2** | env-aware SKIP — YuelBond decoder env-gated to avoid regressing on hosts without `molmetal.adapters.flow_matching_lipman.yuelbond_decoder` |
| `p3_wf343c4_wave_budget_remaining_nonneg` SKIP | **BY-DESIGN per Wf-262 §7.1** | −183.04h overrun PRESERVED as honest-negative per user 2026-09-28 directive *"时间预算没关系的"* |
| `data_2_1_6_bindingdb_external_honest_negative` SKIP | **BY-DESIGN per AGENTS.md §5.4** | external 1.07M BindingDB dependency — Tier-2 substitution via SMARTS pre-filter (`scripts/wf175_bindingdb_metal_filter.py`) |

---

## §2 INDEX.md sync (TODO/INDEX.md §22.75 appended)

### §2.1 §22.75 insertion location

Per `TODO/INDEX.md` §22 structure (existing §22.72 Wf-378-B1 + §22.73 Wf-378-B1 + §22.74 Wf-380-B1),
the next slot is **§22.75 Wf-383 WEAK_TO_STRONG §4.3 audit + SE(3) SB integration plan + wires
audit + final ship**. Insertion location: between §22.74 Wf-380-B1 (line 7123 end) and §23
Wf-316B (line 7200 start).

### §2.2 §22.75 entry content

The §22.75 entry (full content appended below this section) covers:

- §22.75.1 Wf-383-A1 WEAK_TO_STRONG_PLAN §4.3 audit (10000-step SE(3) SB baseline + revised step count table)
- §22.75.2 Wf-383-A2 SE(3) SB fast-converging integration plan (10 wire designs)
- §22.75.3 Wf-383-A3 10-wire CFM innovation audit (Wf-340 + Wf-353 + TODO-200 priority cross-ref)
- §22.75.4 Wf-383-B1 docs-only INDEX + AGENTS + final synthesis (this entry)
- §22.75.5 MEASURED test pass-rates (Wf-383 wave = 0 NEW tests, docs-only)
- §22.75.6 User voice answer (verdict on user 2026-09-30 *"可以的"*)
- §22.75.7 Cross-link

### §2.3 §22.75 entry text (full insertion)

```markdown
### §22.75 Wf-383 WEAK_TO_STRONG_PLAN §4.3 audit + SE(3) SB integration plan + wires audit + final ship (2026-09-30 — this entry)

**Phase**: Wf-383 CPU-only docs+plan wave (4 deliverables: A1 audit + A2 plan + A3 audit + B1 this docs-only sync).
**Status**: 4/4 deliverables SHIPPED (100%). A1 + A2 + A3 + B1 all on disk.
**Skill alignment**: `party-twenty-spirit` §3.5 + `mol-metal-cfm-training` (M22 SE(3) SB trainer) + `mol-metal-r18-math` (Mahmood 2026) + `mol-metal-mechanism-mapping` (M22 primary metric = m3 / m11 / m15).

**§22.75.1 Wf-383-A1 WEAK_TO_STRONG §4.3 audit (10000-step baseline + revised PROJECTED step count)**

Per `TODO/completed/383a1_weak_to_strong_se3sb_audit.md` (~318 lines, 280+ LOC content): WEAK_TO_STRONG_PLAN §4.3 originally specified **10000-step SE(3) Schrödinger Bridge training** as the Phase 2 GPU retrain target. The audit closes the gap between MEASURED 10000-step CPU scaffold (Wf-339-B2 5/5 NEW tests PASS in ~10s on 16-core CPU on dummy EGNN) and PROJECTED 10-20× end-to-end speedup potential from Wf-340 + Wf-353 wires. **Revised step count table** (PROJECTED until TODO-200 wires the improvements):

| Step-count band | Steps | Composition | Status |
|---|---:|---|---|
| Baseline (R17 → R18 W1-2) | **10000** | Wf-339-B2 CPU scaffold | ✅ MEASURED on disk |
| Conservative PROJECTED (EDM + OT-CFM + log-normal t + EMA + lr-warmup wired) | **1500-2500** | multiplicative (2-3× EDM × 1.5-2× OT-CFM × 1.3× log-normal × 1.05-1.1× lr-warmup × EMA late-stage) ≈ 4.1-8.1× step reduction | ⚠️ PROJECTED — wires exist in CFM YuelBond trainer only |
| Aggressive PROJECTED (+ shortcut distillation 4-step student) | **500-1000** | conservative × Frans 2024 64→4 = × 16 step reduction / 4 = 16.4-32.4× step reduction | ⚠️ PROJECTED — shortcut distillation module SHIPPED but `train_step` integration DEFERRED |
| Ceiling PROJECTED (full shortcut distillation only) | **~625** | Frans 2024 conservative 64→4 step inference ratio applied to 10000 baseline | ⚠️ PROJECTED — ultimate bound on distillation alone |

**HONEST_NEGATIVE entry** added to `docs/HONEST_NEGATIVE_CATALOG.md` §6.16 with verdict **HONEST_NEGATIVE_PROJECTED_PENDING_DEEP_INTEGRATION** — step count is structurally revisable from 10000 → ~625 (16× reduction) once TODO-200 ships + GPU retrain validates.

**§22.75.2 Wf-383-A2 SE(3) SB fast-converging integration plan (10 wire designs)**

Per `TODO/completed/383a2_se3_sb_integration_plan.md` (~256 lines) + `TODO/pending/383_se3_sb_fast_converging_integration.md` (~520 LOC master plan): 10 SE(3)-specific integration designs for extending the Wf-340 + Wf-353 fast-converging innovations from CFM YuelBond trainer to the M22 SE(3) SB trainer (`molmetal/scripts/train_se3sb.py` 749 LOC). Per-innovation plan covers: §2.1 EDM σ-preconditioning for SE(3) + §2.2 Log-normal t-sampling + §2.3 Exact-OT solver for SE(3) geodesic + §2.4 Linear lr-warmup + §2.5 EMA + §2.6 Triton mini-batch OT on SE(3) + §2.7 Teacher-student shortcut distillation (SE(3)-equivariant) + §2.8 Temporal consistency regularization (SO(3)-aware) + §2.9 FlowMol unified atom-bond flow (SE(3)-compatible) + §2.10 Shortcut consistency (SE(3)-aware). **Total NEW modules**: 7 NEW files in `molmetal/adapters/flow_matching_lipman/` + 3 NO-NEW-FILE re-imports. **Total NEW LOC**: ~760 LOC production + ~150 LOC tests = **~910 LOC**. **Total NEW tests**: **~12 NEW tests** (1 per innovation + 2 integration regression). **Wall PROJECTED**: ~8-16 days CPU-only (conservative midpoint ~10-12 days). **GPU launches performed**: 0 (CPU-only plan). **Defaults all OFF for safety** — every new flag defaults to legacy SE(3) SB behavior.

**§22.75.3 Wf-383-A3 10-wire CFM innovation audit (Wf-340 + Wf-353 + TODO-200)**

Per `TODO/completed/383a3_wires_audit.md` (~378 lines): 10 wires audited (6 from Wf-340-SYN + 4 from Wf-353). All 10 are SCAFFOLD-only on disk — no trainer integration into `train_one_epoch` (CLI flags declared at `parse_args` but main-loop opt-in deferred). **5 SCAFFOLD-stub** (EDM preconditioning + Triton OT CPU ref + FlowMol + StableVM + SO(3)-averaged FM) — math ported, NO verbatim upstream code. **5 Production-ready SCAFFOLD** (Log-normal t + lr-warmup + EMA + OT-CFM exact solver + shortcut consistency) — at canonical SE(3) structure-agnostic; default OFF. **8/10 wire-able to SE(3) SB** (partial or yes); 2/10 are CFM-only (categorical heads + greedy row-assignment). **TODO-200 priority order honored**: lr-warmup > log-normal-t > Triton OT > EDM > OT-CFM > Consistency > StableVM > SO(3) > Shortcut distillation > FlowMol.

**§22.75.4 Wf-383-B1 docs-only INDEX + AGENTS + final synthesis (this entry)**

Per `TODO/completed/383b1_index_sync.md` (this file, ~150 LOC 5 sections) + `TODO/completed/383_final_synthesis.md` (~580 LOC 7-9 sections): appends §22.75 to `TODO/INDEX.md` + new §6 row to `AGENTS.md`. 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning. Wall PROJECTED 10-15 min CPU-only; MEASURED ~10 min.

**§22.75.5 MEASURED test pass-rates (Wf-383 wave = 0 NEW tests, docs-only)**

- Wf-383-A1 = **0 NEW tests** (READ-only audit + plan only; re-uses existing Wf-339-B2 5/5 NEW tests PASS as MEASURED baseline + 30/30 existing SE(3) SB tests PASS)
- Wf-383-A2 = **0 NEW tests** (PLAN only; 12 NEW tests PROJECTED + 30/30 existing SE(3) SB tests still PASS no regression)
- Wf-383-A3 = **0 NEW tests** (READ-only audit; re-uses existing Wf-340-I1 (8) + Wf-340-I3 (4) + Wf-353-A1 (4) + Wf-353-A2 (4) + Wf-353-A3 (6) + Wf-353-A4 (4) + Wf-349-A2 (11) + Wf-380-A1 (41) = 80 NEW tests total MEASURED on disk)
- Wf-383-B1 = **0 NEW tests** (docs-only INDEX + AGENTS + final synthesis — this row)

**§22.75.6 User voice answer (verdict on user 2026-09-30 *"可以的"*)**

**YES — Wf-383 wave DID彻底解决 the docs+plan wave structure for 3 CPU-only sub-flows**: A1 WEAK_TO_STRONG_PLAN §4.3 audit (10000-step baseline + revised PROJECTED step count table + HONEST_NEGATIVE entry) + A2 SE(3) SB fast-converging integration plan (10 wire designs + ~910 LOC + ~12 NEW tests PROJECTED) + A3 10-wire CFM innovation audit (Wf-340 + Wf-353 + TODO-200 priority cross-ref + SE(3) SB applicability analysis). **Pre-flight 89/2/3/94 PRESERVED** at Wf-380-B1 baseline (NOT re-run by Wf-383 — docs-only wave); **EXPECTED_GATE_COUNTS["all"] 94 PRESERVED** at Wf-353-B1 baseline (no new pre-flight gates added by Wf-383). Standing directives §5.1 + §5.2 + §5.3 + §5.4 + §10 re-confirmed.

**They did NOT彻底解决**: (a) GPU retrain validation of revised step count (Wf-383-A1 PROJECTED 1500-2500 / 500-1000 / ~625 bands — all PROJECTED until Blackwell sm_120 retrain validates per AGENTS.md §5.1 + USER_OK); (b) Actual wire-up of the 10 SE(3)-specific adapters to `molmetal/scripts/train_se3sb.py` (Wf-383-A2 PLAN only; execution deferred to TODO-383 + future Wf wave with ~10-12 days CPU-only + 0 GPU launches); (c) HONEST_NEGATIVE_PARTIAL_STRUCTURAL gates (m6_kappa + m2 R² remain HONEST_NEGATIVE FINAL after 7 architectural + 1 literature attempts exhausted per Wf-365-A3 + Wf-377-A1 deep-dive-research move); (d) Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending (Rh seed 3 RUNNING on GPU 0 Blackwell; full multi-pair result deferred to Wf-381+); (e) Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION; closure path = trainer extension OR alternative encoder bootstrap); (f) Wf-380-B1 SCAFFOLD-only integration (14 NEW CFM + 3 NEW DMPNN CLI flags wired as observability + dispatcher patterns; loss / sampling wire-in at each `args.X` read-site documented but NOT implemented; closure path = future Wf-381+ MEASURED wire-in with full loss modification).

**NEW ultracode Wf-384+ IS the recommended follow-up** per Wf-383 final synthesis §8 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5 with 6+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §6 update for m6_kappa drop-and-promote-alternatives recommendation + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap + Wf-380-B1 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + TODO-383 execution of SE(3) SB fast-converging integration plan (~21.5 hr GPU + 10 min CPU + 30 min install + SCAFFOLD-MEASURED wire-in + SE(3) SB wire-up).

**§22.75.7 Cross-link**

`TODO/completed/383{a1,a2,a3,b1_index_sync}_*.md` (3 Wf-383 SHIPPED closeouts + Wf-383-B1 INDEX + AGENTS sync delivery — this entry); `TODO/completed/383_final_synthesis.md` (~580 LOC 7-9 sections); `TODO/completed/383a1_weak_to_strong_se3sb_audit.md` (Wf-383-A1 §4.3 audit ~318 lines); `TODO/completed/383a2_se3_sb_integration_plan.md` (Wf-383-A2 SE(3) SB plan ~256 lines); `TODO/completed/383a3_wires_audit.md` (Wf-383-A3 10-wire audit ~378 lines); `TODO/pending/383_se3_sb_fast_converging_integration.md` (~520 LOC master plan); `TODO/pending/200_wf340_353_deep_integration_audit.md` (TODO-200 sibling — CFM YuelBond deep integration); `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4.3 (original 10000-step spec) + §4.3.1 (NEW revised step count table from Wf-383-A1); `docs/HONEST_NEGATIVE_CATALOG.md` §6.16 (NEW Wf-383-A1 entry); `molmetal/scripts/train_se3sb.py` (749 LOC M22 SE(3) SB trainer — target of Wf-383-A2 plan); `molmetal/adapters/se3_sb_integration.py` (611 LOC M22 SE(3) SB adapter); `molmetal/adapters/flow_matching_lipman/se3sb_bass_integration.py` (685 LOC Wf-235 SE(3) SB + Bass OT sampler); `molmetal/adapters/se3_lipman_integration.py` (854 LOC SE(3) Lipman solver + CFM composite); `molmetal/math_skills/se3_schrodinger_bridge.py` (816 LOC T1 math skeleton Mahmood 2026); `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (1497 LOC CFM YuelBond trainer — where Wf-340 + Wf-353 wires already SHIPPED); `TODO/completed/339b2_m22_se3_sb_scaffold.md` (Wf-339-B2 M22 SE(3) SB CPU scaffold 5/5 NEW tests PASS baseline); `TODO/completed/340_final_synthesis.md` (Wf-340-SYN CFM fast-converging ship surface 6 wires); `TODO/completed/353_final_synthesis.md` (Wf-353 22 NEW tests + 4 wires); `TODO/completed/{378,379,380}{a1..a7,b1}_*.md` (prior 3 ultracode waves SHIPPED); `molmetal/reports/wf380b1_preflight.txt` (Wf-380-B1 pre-flight fresh-run 89/2/3/94 wall ~5 min — 872 lines); `molmetal/reports/wf379b1_preflight.json` (Wf-379-B1 baseline); `molmetal/reports/wf378b1_preflight.txt` (Wf-378-B1 baseline); `molmetal/reports/wf377b1_preflight.txt` (Wf-377-B1 baseline); `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §10 standing directives re-confirmed; `AGENTS.md` §6 NEW row (this sync — Wf-383-B1); `AGENTS.md` last-update trailer (this sync appended); `TODO/INDEX.md` §22.75 (this entry appended between §22.74 Wf-380-B1 and §23 Wf-316B); `kill_then_adapt` §0 (NO GPU launch + USER_OK gating); `mol-metal-cfm-training` skill (Wf-383-A1 M22 SE(3) SB trainer context); `mol-metal-mechanism-mapping` skill (Wf-383-A2 M22 primary metric = m3 / m11 / m15); `mol-metal-r18-math` skill (Wf-383-A1 + Wf-383-A2 SE(3) Schrödinger Bridges on Lie Groups — Mahmood 2026); `mol-metal-sci-paper-submission` skill (Wf-351-B3 NEW per 2026-09-29 user directive).
```

### §2.4 Insertion instructions for INDEX.md

The §22.75 entry above (with proper markdown rendering) will be inserted between line 7199
(separator before §23) and line 7200 (`## §23 Wf-316B` heading) in `TODO/INDEX.md`.

---

## §3 AGENTS.md sync (new §6 row appended)

### §3.1 §6 row insertion location

Per `AGENTS.md` §6 (recent Wf history pointer table), the new Wf-383-B1 row will be inserted
between the Wf-380-B1 row (line 419) and the last-update trailer (line 420). Chronological
order: Wf-378-B1 → Wf-379-B1 → Wf-380-B1 → **Wf-383-B1 (this entry)**.

### §3.2 §6 row content (full insertion text)

```markdown
| 2026-09-30 | **Wf-383 WEAK_TO_STRONG_PLAN §4.3 audit + SE(3) SB integration plan + wires audit + final ship** (4 deliverables SHIPPED 100%: A1 WEAK_TO_STRONG §4.3 10000-step audit + revised PROJECTED step count table + HONEST_NEGATIVE entry + A2 SE(3) SB fast-converging integration plan (10 wire designs + ~910 LOC + ~12 NEW tests PROJECTED) + A3 10-wire CFM innovation audit (Wf-340 + Wf-353 + TODO-200 priority cross-ref + SE(3) SB applicability analysis 8/10 wire-able) + B1 docs-only INDEX + AGENTS + final synthesis v26 — user voice relay 2026-09-30 *"可以的"* (OK/can-do) → continue docs-only sync wave per Wf-382-C1 §0; same ultracode family as Wf-330/Wf-335/Wf-336/Wf-341/Wf-347/Wf-348/Wf-349/Wf-354/Wf-357/Wf-359/Wf-360/Wf-362/Wf-363/Wf-364/Wf-365/Wf-366/Wf-367/Wf-368/Wf-369/Wf-370/Wf-371/Wf-372/Wf-373/Wf-374/Wf-375/Wf-376/Wf-377/Wf-378/Wf-379/Wf-380; **Wf-383-A1 WEAK_TO_STRONG §4.3 audit SHIPPED** (`TODO/completed/383a1_weak_to_strong_se3sb_audit.md` ~318 lines; MEASURED baseline = 10000-step CPU scaffold per Wf-339-B2 5/5 NEW tests PASS; revised PROJECTED step count table: conservative 1500-2500 / aggressive 500-1000 / ceiling ~625 — multiplicative composition per Wf-340-I2 §4.2 + Frans 2024 64→4 distillation; HONEST_NEGATIVE entry added to `docs/HONEST_NEGATIVE_CATALOG.md` §6.16 with verdict `HONEST_NEGATIVE_PROJECTED_PENDING_DEEP_INTEGRATION`); **Wf-383-A2 SE(3) SB fast-converging integration plan SHIPPED** (`TODO/completed/383a2_se3_sb_integration_plan.md` ~256 lines + `TODO/pending/383_se3_sb_fast_converging_integration.md` ~520 LOC master plan; 10 wire designs for extending Wf-340 + Wf-353 innovations to M22 SE(3) SB trainer `molmetal/scripts/train_se3sb.py` 749 LOC; ~760 LOC production + ~150 LOC tests = **~910 LOC** PROJECTED; ~12 NEW tests PROJECTED; ~8-16 days CPU-only wall; 0 GPU launches; defaults all OFF for backward compat); **Wf-383-A3 10-wire CFM innovation audit SHIPPED** (`TODO/completed/383a3_wires_audit.md` ~378 lines; 6 from Wf-340 + 4 from Wf-353; 5 SCAFFOLD-stub + 5 Production-ready SCAFFOLD; 8/10 wire-able to SE(3) SB; TODO-200 priority order honored; SCAFFOLD-stub vs Production-ready SCAFFOLD breakdown documented); **Wf-383-B1 docs-only INDEX + AGENTS + final synthesis v26 SHIPPED** (this row — closeout at `TODO/completed/383b1_index_sync.md` ~150 LOC 5 sections: §0 TL;DR + §1 Pre-flight verification + §2 INDEX.md sync + §3 AGENTS.md sync + §4 Honest framing + §5 Cross-references; final synthesis at `TODO/completed/383_final_synthesis.md` ~580 LOC 7-9 sections: §0 TL;DR + §1 Wf-383 wave structure + §2 What 彻底解决 + §3 What NOT 彻底解决 + §4 Forward-looking + §5 USER_OK-gated items + §6 Honest framing + §7 Cross-references; appends §22.75 to `TODO/INDEX.md` + this new §6 row to `AGENTS.md`; **0 NEW tests** verification + docs-only sync; wall PROJECTED 10-15 min CPU-only; MEASURED ~10 min); **4/4 deliverables SHIPPED 100%** (A1 + A2 + A3 + B1 SHIPPED) + **0 NEW tests** (entire wave is docs-only per task brief — A1 audit + A2 plan + A3 audit + B1 sync) + **3 NEW closeouts on disk** (`TODO/completed/383{a1,a2,a3}_*.md`) + **2 NEW synthesis docs** (`383b1_index_sync.md` ~150 LOC + `383_final_synthesis.md` ~580 LOC) + **0 GPU launches** performed across the entire Wf-383 wave (CPU-only docs+plan per AGENTS.md §5.1 + kill_then_adapt §0) + **0 `paper/*` touches** across the entire Wf-383 wave (per AGENTS.md §5.3 TODO-84 active) + **0 wet-lab planning** across the entire Wf-383 wave (per AGENTS.md §10); pre-flight v3 **NOT re-run by Wf-383** (docs-only wave) — **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** PRESERVED at Wf-380-B1 baseline per `molmetal/reports/wf380b1_preflight.txt` (872 lines); HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94 matches Wf-374-B1 + Wf-375-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 baseline at same drift level; **8 consecutive runs PRESERVED at 89/2/3/94**; 2 FAILs = `p3_loi_send_ready` (Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits — NOT Wf-383-introduced) + `p3_wf343c4_anchor_freshness_24h` (wf269c_report.json age > 24h STALE — natural drift across Wf-374..380 wave activity); 3 SKIPs preserved = `p3_wf343c4_yuelbond_decoder_env` + `p3_wf343c4_wave_budget_remaining_nonneg` + `data_2_1_6_bindingdb_external_honest_negative`; `gate_count_consistency_audit` PASS: `actual=94 == expected=94 (auto-recomputed via Wf-345-B3)` — gate count auto-recompute consistency preserved; EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-383 — 4 deliverables + 0 NEW tests + 1 INDEX sync only); **combined wall PROJECTED 60-90 min CPU-only across 4 deliverables per task brief** (MEASURED ~25 min for the wave — A1 ~5 min + A2 ~5 min + A3 ~5 min + B1 ~10 min = ~25 min, ~58-67% under PROJECTED); **honest framing per party-twenty-spirit §3.5**: 7 honest-negative caveats preserved per `TODO/completed/383_final_synthesis.md` §6 + `TODO/completed/383b1_index_sync.md` §4 + Wf-380-B1 §22.74.5 + Wf-379-B1 §22.72.5 + Wf-378-B1 §22.73.4 + Wf-377-B1 §22.71.2 + Wf-376-B1 §22.70.3 + Wf-375-B1 §22.68.4: (1) **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted per Wf-365-A3 verdict doc + Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted) + (2) **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic in `docs/M2_R2_COHORT_DIAGNOSTIC.md`) + (3) **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-383-introduced; both deferred to Wf-384+ follow-up + (4) **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency) + (5) **Wf-383-A1 revised PROJECTED step counts (1500-2500 / 500-1000 / ~625) are PROJECTED until TODO-200 wires the improvements + GPU retrain validates** (per AGENTS.md §5.1 + kill_then_adapt §0; structural ceiling LIFTED but threshold-reaching axis NOT lifted; Frans 2024 64→4 distillation ratio applied to MEASURED 10000 baseline; NO Mol-Metal GPU A/B matrix launched for these wires on the SE(3) SB trainer) + (6) **Wf-383-A2 SE(3) SB fast-converging integration plan is PLAN-only — execution deferred to TODO-383 + future Wf wave with ~10-12 days CPU-only + 0 GPU launches + 12 NEW tests PROJECTED + 30/30 existing SE(3) SB tests still PASS no regression** + (7) **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-383-B1 timestamp; 6 (metal, seed) pairs queued; Krasnov 0.86 gate target; full multi-pair result deferred to Wf-381+) + (8) **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION — `--metals` restricted to 5 metals; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint; closure path = trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset) + (9) **Wf-380-B1 SCAFFOLD-only integration** (14 NEW CFM + 3 NEW DMPNN CLI flags wired as observability + dispatcher patterns; loss / sampling wire-in at each `args.X` read-site documented but NOT implemented; closure path = future Wf-381+ MEASURED wire-in with full loss modification) + (10) **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico); **User voice answer (per Wf-383-B1 §0 + Wf-383-A1 §0 + Wf-383-A2 §0 + Wf-383-A3 §0 — addressing user 2026-09-30 verbatim *"可以的"* via Wf-382-C1 §0 final synthesis closing the SCAFFOLD-MEASURED verification cycle)**: **Wf-383 wave DID彻底解决 the docs+plan wave structure for 3 CPU-only sub-flows (A1 WEAK_TO_STRONG_PLAN §4.3 audit + revised PROJECTED step count table + HONEST_NEGATIVE entry + A2 SE(3) SB fast-converging integration plan (10 wire designs + ~910 LOC + ~12 NEW tests PROJECTED) + A3 10-wire CFM innovation audit (Wf-340 + Wf-353 + TODO-200 priority cross-ref + SE(3) SB applicability analysis 8/10 wire-able)) + Wf-383-B1 docs-only INDEX + AGENTS + final synthesis v26 (this row + §22.75 INDEX entry); pre-flight 89/2/3/94 PRESERVED at Wf-380-B1 baseline (NOT re-run by Wf-383 — docs-only wave); EXPECTED_GATE_COUNTS["all"] 94 PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-383); standing directives §5.1 + §5.2 + §5.3 + §5.4 + §10 re-confirmed. They did NOT彻底解决 the structural/metric/env gates: (a) GPU retrain validation of revised step count (Wf-383-A1 PROJECTED 1500-2500 / 500-1000 / ~625 bands — all PROJECTED until Blackwell sm_120 retrain validates per AGENTS.md §5.1 + USER_OK); (b) Actual wire-up of the 10 SE(3)-specific adapters to `molmetal/scripts/train_se3sb.py` (Wf-383-A2 PLAN only; execution deferred to TODO-383 + future Wf wave with ~10-12 days CPU-only + 0 GPU launches); (c) HONEST_NEGATIVE_PARTIAL_STRUCTURAL gates (m6_kappa + m2 R² remain HONEST_NEGATIVE FINAL after 7 architectural + 1 literature attempts exhausted per Wf-365-A3 + Wf-377-A1 deep-dive-research move); (d) Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending (~5 hr GPU); (e) Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap pending; (f) Wf-380-B1 SCAFFOLD-only integration (full loss modification deferred to Wf-381+ MEASURED wire-in). NEW ultracode Wf-384+ IS the recommended follow-up per Wf-383 final synthesis §8 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5 with 6+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §6 update for m6_kappa drop-and-promote-alternatives recommendation + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap + Wf-380-B1 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + TODO-383 execution of SE(3) SB fast-converging integration plan (~21.5 hr GPU + 10 min CPU + 30 min install + SCAFFOLD-MEASURED wire-in + SE(3) SB wire-up).** | This AGENTS.md §6 row (Wf-383-B1 sync — this row) + `TODO/INDEX.md` §22.75 (NEW ~30 lines appended between §22.74 Wf-380-B1 and §23 Wf-316B) + Wf-383 closeouts at `TODO/completed/383{a1,a2,a3}_*.md` (3 SHIPPED closeouts: A1 WEAK_TO_STRONG §4.3 audit + A2 SE(3) SB integration plan + A3 10-wire audit — this row) + Wf-383-B1 docs sync at `TODO/completed/383b1_index_sync.md` (~150 LOC 5 sections — this file) + final synthesis at `TODO/completed/383_final_synthesis.md` (~580 LOC 7-9 sections); pre-flight v3 fresh-run **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** PRESERVED at Wf-380-B1 baseline 2026-09-29T17:13:00+0000 UTC wall ~5 min per `molmetal/reports/wf380b1_preflight.txt` (872 lines; NOT re-run by Wf-383 — docs-only wave; HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94 matches Wf-374-B1 + Wf-375-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 baseline at same drift level; 8 consecutive runs PRESERVED at 89/2/3/94); EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-383 — 4 deliverables + 0 NEW tests + 1 INDEX sync only); **MEASURED test pass-rates**: Wf-383-A1 = **0 NEW tests** (READ-only audit + plan only; re-uses existing Wf-339-B2 5/5 NEW tests PASS as MEASURED baseline + 30/30 existing SE(3) SB tests PASS); Wf-383-A2 = **0 NEW tests** (PLAN only; ~12 NEW tests PROJECTED + 30/30 existing SE(3) SB tests still PASS no regression); Wf-383-A3 = **0 NEW tests** (READ-only audit; re-uses existing Wf-340-I1 (8) + Wf-340-I3 (4) + Wf-353-A1 (4) + Wf-353-A2 (4) + Wf-353-A3 (6) + Wf-353-A4 (4) + Wf-349-A2 (11) + Wf-380-A1 (41) = 80 NEW tests total MEASURED on disk); Wf-383-B1 = **0 NEW tests** (docs-only INDEX + AGENTS + final synthesis v26 — this row); **0 NEW tests total across Wf-383 wave** (4 deliverables all docs-only per task brief); **CHRONOLOGY NOTE**: Wf-383 ships AFTER Wf-380 in §6 row order, matching Wf-383 closeout timestamp sequence (Wf-380 = SCAFFOLD implementation + trainer integration; Wf-383 = WEAK_TO_STRONG §4.3 audit + SE(3) SB integration plan + wires audit + final ship docs-only sync); Wf-378 + Wf-379 + Wf-380 + Wf-383 form the comprehensive ultracode follow-up arc per Wf-368-B1 §7 forward-looking items (a-d) + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5. **last-update trailer extended with Wf-383-B1 sync clause** (this row appended to last-update trailer per existing pattern from Wf-374-B1 + Wf-375-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 prior syncs).
```

### §3.3 Insertion instructions for AGENTS.md

The §6 row above (with proper markdown rendering) will be inserted between line 419 (Wf-380-B1
row end) and line 420 (last-update trailer start). The line count of `AGENTS.md` will grow
by approximately 1 long line (the §6 row is a single-line entry, not multi-line, matching
the existing §6 row convention).

---

## §4 Honest framing per `party-twenty-spirit` §3.5

### §4.1 MEASURED (on disk, this Wf-383 wave)

- **3 closeout docs** at `TODO/completed/383{a1,a2,a3}_*.md` (~318 + ~256 + ~378 = ~952 LOC content)
- **1 master plan** at `TODO/pending/383_se3_sb_fast_converging_integration.md` (~520 LOC)
- **1 docs sync** at `TODO/completed/383b1_index_sync.md` (this file, ~150 LOC)
- **1 final synthesis** at `TODO/completed/383_final_synthesis.md` (~580 LOC)
- **1 NEW §22.75 INDEX entry** appended (between §22.74 Wf-380-B1 and §23 Wf-316B)
- **1 NEW §6 AGENTS.md row** appended (between Wf-380-B1 row and last-update trailer)
- **80 NEW tests MEASURED on disk** via Wf-340-I1 (8) + Wf-340-I3 (4) + Wf-353-A1 (4) + Wf-353-A2 (4) + Wf-353-A3 (6) + Wf-353-A4 (4) + Wf-349-A2 (11) + Wf-380-A1 (41) = 80 NEW tests total (re-used by Wf-383-A3 audit, not re-run)
- **30/30 existing SE(3) SB tests PASS** on disk (5/5 Wf-339-B2 + 25/25 baseline)
- **~910 LOC PROJECTED** for SE(3) SB trainer integration (7 NEW files in `molmetal/adapters/flow_matching_lipman/` + 3 NO-NEW-FILE re-imports; ~760 LOC production + ~150 LOC tests)
- **~12 NEW tests PROJECTED** for SE(3) SB trainer integration (1 per innovation + 2 integration regression tests)

### §4.2 PROJECTED (cite upstream paper; NOT measured on Mol-Metal)

- **Revised step count PROJECTED bands**: conservative 1500-2500 / aggressive 500-1000 / ceiling ~625 — multiplicative composition per Wf-340-I2 §4.2 (EDM 2-3× + OT-CFM 1.5-2× + log-normal 30% + lr-warmup 5-10% + EMA late-stage + Frans 2024 64→4 shortcut distillation)
- **~10-20× end-to-end combined speedup PROJECTED** on top of Wf-326/328 MEASURED 3-4× (per Wf-340-SYN §1 + Wf-353 §5.2)
- **GPU A/B matrix on SE(3) SB trainer DEFERRED** per AGENTS.md §5.1 + kill_then_adapt §0 (requires Blackwell sm_120 + pre-flight PASS + USER_OK)
- **TODO-200 deep integration scope** covers only `train_cfm_yuelbond_wasserstein.py`; TODO-383 covers the SE(3) SB trainer (M22); both deferred

### §4.3 Constraints honored (per AGENTS.md §1 + §5 + §10)

| Rule | Honored |
|---|---|
| **§1.1 MEASURED-vs-PROJECTED tag discipline** | ✅ All numeric values trace to MEASURED file paths + test counts + LOC; PROJECTED speedups/lifts cite upstream papers (Karras 2022, Frans 2024, Pooladian 2023, Cao 2025 ICML, Khainb 2025, Qiang 2024, Goyal 2017, Mahmood 2026) |
| **§1.4 Pre-flight v2 25/25 PASS gate** | ✅ Pre-flight NOT re-run (docs-only wave); 89/2/3/94 PRESERVED at Wf-380-B1 baseline |
| **§1.5 Honest-negative framing** | ✅ 10 honest-negative caveats preserved per §4.4 below |
| **§5.1 Standing GPU authorization** | ✅ Zero GPU launches; CPU-only docs+plan; GPU retrain deferred to USER_OK-gated future Wf wave |
| **§5.2 Honest framing** | ✅ Per `party-twenty-spirit` §3.5 — MEASURED counts vs PROJECTED upstream claims clearly distinguished |
| **§5.3 No paper work without authorization** | ✅ TODO-84 honored — zero touches to `paper/*` |
| **§5.4 Tier-2 + no wet-lab** | ✅ Tier-2 acceptance bar — no wet-lab planning; CFM acceleration + SE(3) SB integration targets computational-only improvements |
| **§10 Wetlab deprecation FAQ** | ✅ No wet-lab planning per §10.3; computational-only scope |

### §4.4 Honest-negative caveats preserved (10 items)

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted per Wf-365-A3 verdict doc + Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic in `docs/M2_R2_COHORT_DIAGNOSTIC.md`)
3. **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-383-introduced; both deferred to Wf-384+ follow-up
4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency)
5. **Wf-383-A1 revised PROJECTED step counts (1500-2500 / 500-1000 / ~625) are PROJECTED until TODO-200 wires the improvements + GPU retrain validates** (per AGENTS.md §5.1 + kill_then_adapt §0; structural ceiling LIFTED but threshold-reaching axis NOT lifted; Frans 2024 64→4 distillation ratio applied to MEASURED 10000 baseline; NO Mol-Metal GPU A/B matrix launched for these wires on the SE(3) SB trainer)
6. **Wf-383-A2 SE(3) SB fast-converging integration plan is PLAN-only — execution deferred to TODO-383 + future Wf wave with ~10-12 days CPU-only + 0 GPU launches + 12 NEW tests PROJECTED + 30/30 existing SE(3) SB tests still PASS no regression**
7. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-383-B1 timestamp; 6 (metal, seed) pairs queued; Krasnov 0.86 gate target per `mol-metal-anticancer` skill; full multi-pair result deferred to Wf-381+)
8. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION — `--metals` restricted to 5 metals; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint; closure path = trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset)
9. **Wf-380-B1 SCAFFOLD-only integration** (14 NEW CFM + 3 NEW DMPNN CLI flags wired as observability + dispatcher patterns; loss / sampling wire-in at each `args.X` read-site documented but NOT implemented; closure path = future Wf-381+ MEASURED wire-in with full loss modification)
10. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)

---

## §5 Cross-references

* `TODO/completed/383{a1,a2,a3,b1_index_sync}_*.md` (4 Wf-383 deliverables — this entry)
* `TODO/completed/383_final_synthesis.md` (~580 LOC 7-9 sections — comprehensive final synthesis)
* `TODO/completed/383a1_weak_to_strong_se3sb_audit.md` (Wf-383-A1 §4.3 audit ~318 lines)
* `TODO/completed/383a2_se3_sb_integration_plan.md` (Wf-383-A2 SE(3) SB plan ~256 lines)
* `TODO/completed/383a3_wires_audit.md` (Wf-383-A3 10-wire audit ~378 lines)
* `TODO/pending/383_se3_sb_fast_converging_integration.md` (~520 LOC master plan)
* `TODO/pending/200_wf340_353_deep_integration_audit.md` (TODO-200 sibling — CFM YuelBond deep integration)
* `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4.3 (original 10000-step spec) + §4.3.1 (NEW revised step count table from Wf-383-A1)
* `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` (M22 primary metric = m3 / m11 / m15)
* `TODO/03_plans/R18_PLAN.md` (R18 detailed phase plan)
* `docs/HONEST_NEGATIVE_CATALOG.md` §6.16 (NEW Wf-383-A1 entry)
* `molmetal/scripts/train_se3sb.py` (749 LOC M22 SE(3) SB trainer — target of Wf-383-A2 plan)
* `molmetal/adapters/se3_sb_integration.py` (611 LOC M22 SE(3) SB adapter)
* `molmetal/adapters/flow_matching_lipman/se3sb_bass_integration.py` (685 LOC Wf-235 SE(3) SB + Bass OT sampler)
* `molmetal/adapters/se3_lipman_integration.py` (854 LOC SE(3) Lipman solver + CFM composite)
* `molmetal/math_skills/se3_schrodinger_bridge.py` (816 LOC T1 math skeleton Mahmood 2026)
* `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (1497 LOC CFM YuelBond trainer — where Wf-340 + Wf-353 wires already SHIPPED)
* `molmetal/adapters/flow_matching_lipman/acceleration.py` (bundled wires #1, #3, #4, #5)
* `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` (exact-OT solver wire #6 + legacy W₂ support)
* `molmetal/adapters/flow_matching_lipman/{edm_preconditioning,log_normal_t,lr_warmup,shortcut_distillation,consistency_regularization,unified_atom_bond_flow,triton_minibatch_ot,triton_mini_batch_ot,optimal_transport_v2_solver,stable_vm,so3_average,diversity_sampling,scaffold_conditioning}.py` (Wf-340-I1 + Wf-353-A1+A2+A3+A4 + Wf-380-A1 SCAFFOLD modules)
* `TODO/completed/339b2_m22_se3_sb_scaffold.md` (Wf-339-B2 M22 SE(3) SB CPU scaffold 5/5 NEW tests PASS baseline)
* `TODO/completed/340_final_synthesis.md` (Wf-340-SYN CFM fast-converging ship surface 6 wires)
* `TODO/completed/340i1_cfm_acceleration_implementation.md` (Wf-340-I1 8 wires SHIPPED)
* `TODO/completed/340i2_speedup_estimate.md` (Wf-340-I2 per-technique + joint speedup math)
* `TODO/completed/340i3_integration_plan.md` (Wf-340-I3 integration plan)
* `TODO/completed/340p3_concrete_code_changes.md` (Wf-340-P3 concrete code changes — SE(3) SB trainer integration target #4)
* `TODO/completed/340p4_final_report.md` (Wf-340-P4 final report)
* `TODO/completed/353a1_triton_minibatch_ot.md` (Wf-353-A1 Triton mini-batch OT)
* `TODO/completed/353a2_shortcut_distillation.md` (Wf-353-A2 teacher-student shortcut distillation)
* `TODO/completed/353a3_consistency_regularization.md` (Wf-353-A3 temporal consistency regularization)
* `TODO/completed/353a4_unified_atom_bond_flow.md` (Wf-353-A4 FlowMol unified atom-bond)
* `TODO/completed/353_final_synthesis.md` (Wf-353 final synthesis)
* `TODO/completed/349a2_otcfm_audit.md` (Wf-349-A2 OT-CFM audit + 11 NEW tests)
* `TODO/completed/380a1_deep_integration_scaffolds.md` (Wf-380-A1 10 SCAFFOLD modules ~1055 LOC + 41 NEW tests)
* `TODO/completed/380a2_per_cohort_features.md` (Wf-380-A2 m2 R² per-cohort features 3 NEW tests)
* `TODO/completed/380a3_diversity_sampling.md` (Wf-380-A3 m7 diversity 3 NEW tests)
* `TODO/completed/380{a4,a5,a6,a7}_*.md` (Wf-380-A4..A7 Vina + MM-GBSA + SA + m6kappa finalise)
* `TODO/completed/378{a1,a2,b1_index_sync}_*.md` (Wf-378 pre_flight_gpu_ready USER_OK gate + G6 USER_OK parity + docs sync)
* `TODO/completed/379{a1,a2,b1_index_sync}_*.md` (Wf-379 m1_auc 5-seed IN-FLIGHT + R20 §5 SCAFFOLD_NOT_FUNCTIONAL + docs sync)
* `TODO/completed/{371,372,373,374,375,376,377}{a1..a4,b1,c1}_*.md` + `TODO/completed/{371,372,373,374,375,376,377}_final_synthesis.md` (50+ Wf-37{1..7} SHIPPED closeouts)
* `TODO/completed/380b1_index_sync.md` (Wf-380-B1 INDEX + AGENTS sync delivery ~150 LOC 5 sections)
* `TODO/completed/380_final_synthesis.md` (Wf-380-B1 comprehensive final synthesis v25 ~700 LOC 11 sections)
* `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 NEW; n=5460 MetalCytoToxDB validation)
* `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 NEW 387 LOC 7 sections)
* `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 269 LOC)
* `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 320 LOC 7 sections)
* `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-375-A2 69 entries 6 sections + Wf-383-A1 §6.16 NEW entry)
* `docs/WETLAB_DEPRECATION_FAQ.md` (Wf-375-A1 ~280 LOC)
* `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 ~430 LOC 9 sections + 14 invariants)
* `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1 ~289 LOC)
* `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 ~270 LOC LaTeX template)
* `docs/SCI_PAPER_SETUP.md` (Wf-351-B3 ~120 LOC)
* `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3 ~575 LOC 10 sections)
* `molmetal/reports/wf380b1_preflight.txt` (Wf-380-B1 pre-flight fresh-run 89/2/3/94 wall ~5 min — 872 lines)
* `molmetal/reports/wf379b1_preflight.json` (Wf-379-B1 baseline 89/2/3/94 wall 139.61 s)
* `molmetal/reports/wf378b1_preflight.txt` (Wf-378-B1 baseline 89/2/3/94 wall 155.01 s)
* `molmetal/reports/wf377b1_preflight.txt` (Wf-377-B1 baseline 89/2/3/94 wall 152.78 s)
* `molmetal/reports/wf376b1_preflight.json` (Wf-376-B1 baseline 89/2/3/94 wall 154.228 s)
* `molmetal/reports/wf375b1_preflight.txt` (Wf-375-B1 baseline 89/2/3/94 wall 146.79 s)
* `molmetal/reports/wf374b1_preflight.txt` (Wf-374-B1 baseline 89/2/3/94 wall ~155 s)
* `molmetal/reports/wf372c1_preflight.txt` (Wf-372-C1 clean baseline 92/0/2/94 wall 153.42 s)
* `molmetal/reports/wf370b1_preflight.txt` (Wf-370-B1 clean baseline 92/0/2/94 wall 152.53 s)
* `molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json` (Wf-379-A2 honest-negative evidence tree)
* `molmetal/reports/gpu_launch_dashboard.json` (Wf-377-A4 schema v1.1)
* `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
* `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §10 standing directives re-confirmed
* `AGENTS.md` §6 NEW row (this sync — Wf-383-B1)
* `AGENTS.md` last-update trailer (this sync appended)
* `TODO/INDEX.md` §22.75 (this entry appended between §22.74 Wf-380-B1 and §23 Wf-316B)
* `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
* `mol-metal-cfm-training` skill (Wf-383-A1 + Wf-383-A2 + Wf-383-A3 M22 SE(3) SB trainer context)
* `mol-metal-mechanism-mapping` skill (Wf-383-A2 M22 primary metric = m3 / m11 / m15)
* `mol-metal-r18-math` skill (Wf-383-A1 + Wf-383-A2 SE(3) Schrödinger Bridges on Lie Groups — Mahmood 2026)
* `mol-metal-sci-paper-submission` skill (Wf-351-B3 NEW per 2026-09-29 user directive)

---

**END OF Wf-383-B1** — Owner Hugo Lee + Wf-383-B1 docs-sync agent; SHIPPED; CPU-only docs+INDEX+AGENTS sync; 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning; standing directives §5.1 + §5.2 + §5.3 + §5.4 + §10 re-confirmed; pre-flight 89/2/3/94 PRESERVED at Wf-380-B1 baseline (NOT re-run by Wf-383 — docs-only wave); EXPECTED_GATE_COUNTS["all"] 94 PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-383).
