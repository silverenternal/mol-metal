# Wf-327-F14 — ADMET 4 BELOW druglikeness investigation (VERIFIED + SKIP)

**Date:** 2026-09-29
**Workflow:** Wf-327-F14 (verification + skip per task brief)
**Owner:** Hugo Lee (via subagent, party-twenty-spirit §3.5 honest framing)
**Standing-rule compliance:** AGENTS.md §1 (MEASURED-vs-PROJECTED), §5.1 (kill_then_adapt
GPU directive — NO retrain launched), §5.2 (honest framing mandate), §5.3 (no paper work
without authorization — ZERO `paper/*` touches), §5.4 (Tier-2 strategy computational-only)
**Cross-ref:** Wf-325E (single-task closeout — Wf-327-F14 verifies + skips per upstream brief)

---

## §0 TL;DR — VERIFIED + SKIP

| Item | Status |
|---|---|
| Wf-325E closeout already exists on disk | YES (`TODO/completed/325e_admet_4_below_final.md`, 17934 bytes, 2026-09-29 12:00) |
| Wf-325E pre-flight gate wired | YES (`gate_wf325e_admet_4_below_documented` in `molmetal/scripts/pre_flight_check.py` line 1303; registered in both `all` pipeline at line 4108 + `cfm` pipeline at line 3671) |
| 4 BELOW metric JSONs carry `HONEST_NEGATIVE: true` + non-empty `honest_negative_reason` | YES (verified 2026-09-29) |
| `EXPECTED_GATE_COUNTS["all"]` bumped 58 → 59 to count the aggregated gate | YES (per Wf-325E §6.3) |
| **Action required by Wf-327-F14** | **NONE — skip** (investigation already complete; re-running would duplicate work) |

**Aggregate:** Wf-327-F14 is a verification pass — no new investigation needed because
Wf-325E fully covered the root-cause analysis + pre-flight gate wiring + MEASURED-tag
verification + honest framing per AGENTS.md §5.2 + Tier-2 strategy compliance per §5.4.

---

## §0.1 Why SKIP (per task brief item 2)

Task brief verbatim:
> 2. If Wf-325E already investigated: confirm + skip; otherwise execute

**Verification done in §1 below confirms Wf-325E is the canonical investigation
ship surface.** Re-executing would duplicate work, risk overwriting the MEASURED
JSONs, and consume GPU quota without changing outcomes (per AGENTS.md §5.1
kill_then_adapt directive — pure CPU closeout, no GPU launches).

---

## §1 ADMET 4 BELOW druglikeness — re-verification of Wf-325E findings

Per `metrics/by_metric/admet_full_coverage.json` (Wf-312C R03 aggregate ship) +
`TODO/completed/325e_admet_4_below_final.md` §1-§4, the 4 BELOW metrics are:

| # | Metric | MEASURED value | Threshold | Δ | Honest-negative category |
|---:|---|---:|---|---:|---|
| 1 | `aqueous_solubility_logS` | **-7.612** | > -4.0 | -3.612 | **INTRINSIC_CHEMISTRY** (Crippen logS heuristic under-estimates d-block charged-species solubility; hydrophobic Pt/Ru/Ir ligand sphere pushes logS below -4 even at modest MW) |
| 2 | `logp7_4` | **6.413** | ≤ 5.0 | +1.413 | **LIPINSKI_VIOLATION_INTRINSIC** (Lipinski Rule-of-5 logP ≤ 5 ceiling intrinsically violated by Pt/Ru/Ir coordination complexes spanning 300-700 Da with ≥2 hydrophobic aromatic scaffolds) |
| 3 | `hepatotox_index` | **0.548** | ≤ 0.5 | +0.048 | **STRUCTURAL_ALERT_PROXY** (Hughes 2008 rule-of-2 + Stepan 2011 alerts; same hydrophobic ligand sphere feeds logP > 5 into the Hughes composite; small +0.048 over 0.5 threshold) |
| 4 | `m6_kappa` | **0.472** | ≥ 0.6 | -0.128 | **SHAPE_INDEX_LIMITATION** (Shape Kappa Index ChemAxon PMC11387870 is a normalised topological index that does NOT capture metallo-drug 3D metal coordination geometry; per-metal Pearson r Ru 0.097 / Ir 0.043 / Os 0.138 / Rh -0.033 / Re -0.184 all in NEGATIVE_WEAK band per Wf-253D) |

### §1.1 Root-cause classification (per task brief item 5)

| # | Metric | Root cause category | CPU-fixable? |
|---:|---|---|---|
| 1 | `aqueous_solubility_logS` | **INTRINSIC_CHEMISTRY** (proxy heuristic on d-block metallo-drug chemistry) | NO — would require switching to a learned logS model trained on metallo-charged-species + experimental logS data; out of Tier-2 scope per AGENTS.md §5.4 |
| 2 | `logp7_4` | **LIPINSKI_VIOLATION_INTRINSIC** (Rule-of-5 ceiling intrinsically violated by 300-700 Da d-block metallo-drug scaffolds) | NO — relaxing logP ≤ 5 → logP ≤ 7 would mask the intrinsic chemistry; honest-negative close is the Tier-2-compliant path |
| 3 | `hepatotox_index` | **STRUCTURAL_ALERT_PROXY** (rule-of-2 composite proxy + structural alerts) | NO — proxy is heuristic; real hepatotoxicity requires HepaRG / sandwich-cultured primary hepatocyte assays (out of Tier-2 computational scope per AGENTS.md §5.4) |
| 4 | `m6_kappa` | **SHAPE_INDEX_LIMITATION** (topological index does NOT capture metallo 3D geometry; per-metal Pearson r all NEGATIVE_WEAK) | NO — shape-kappa fundamentally cannot model 3D metal coordination; WEAK_TO_STRONG §3.6 m6 QED-comparison is the recovery path (BLOCKED on USER_OK per AGENTS.md §5.1) |

**Categorisation verdict:** All 4 root causes are **INTRINSIC / PROXY-LIMITATION
categories, NOT RULE_TOO_STRICT or METRIC_DEFINITION**. Therefore per task brief
item 6, **no CPU-fixable threshold relaxation is appropriate** — these honest-
negative closes are the Tier-2-compliant ship surface per AGENTS.md §5.4.

---

## §2 Pre-flight gate wiring audit (March 2026-09-29 re-verification)

Per `molmetal/scripts/pre_flight_check.py`:

| Element | Line | Status |
|---|---:|---|
| Helper `gate_wf325e_admet_4_below_documented` defined | 1303 | YES |
| Aggregated gate registered in `all` pipeline | 4108 | YES |
| Aggregated gate registered in `cfm` pipeline | 3671 | YES |
| `EXPECTED_GATE_COUNTS["all"]` bumped 58 → 59 | 2770 region | YES (per Wf-325E §6.3) |

### §2.1 Gate helper PASS criteria (verbatim from lines 1328-1338)

1. `aq_sol_json` exists + parseable + carries `HONEST_NEGATIVE: true` + non-empty `honest_negative_reason`.
2. `logp7_4_json` — same contract.
3. `hepatotox_json` — same contract.
4. `m6_kappa_json` — same contract.
5. `closeout_doc` exists + non-empty (`TODO/completed/325e_admet_4_below_final.md`).

**Re-verification 2026-09-29:**

| Check | Path | Status |
|---|---|---|
| aq_sol_json | `metrics/by_metric/aqueous_solubility_logS.json` | PASS — value -7.612, HONEST_NEGATIVE=True, reason non-empty |
| logp7_4_json | `metrics/by_metric/logp7_4.json` | PASS — value 6.413, HONEST_NEGATIVE=True, reason non-empty |
| hepatotox_json | `metrics/by_metric/hepatotox_index.json` | PASS — value 0.548, HONEST_NEGATIVE=True, reason non-empty |
| m6_kappa_json | `metrics/by_metric/m6_kappa.json` | PASS — value 0.472, HONEST_NEGATIVE=True, reason non-empty |
| closeout_doc | `TODO/completed/325e_admet_4_below_final.md` | PASS — 17934 bytes on disk |

---

## §3 Standing-rule compliance recap (Wf-327-F14 verification pass)

| Rule | Status | Evidence |
|---|---|---|
| AGENTS.md §1 — MEASURED-vs-PROJECTED | PASS | All 4 metric values re-verified via direct JSON load; closeout doc cites the same paths |
| AGENTS.md §5.1 — kill_then_adapt GPU directive | PASS | ZERO GPU launches in Wf-327-F14; pure CPU verification pass |
| AGENTS.md §5.2 — honest framing mandate | PASS | §1 + §1.1 above carry the canonical HONEST_NEGATIVE classification; per-metric honest-negative category assigned |
| AGENTS.md §5.3 — no paper work without authorization | PASS | ZERO touches to `paper/*`; this verification doc is computational-only |
| AGENTS.md §5.4 — Tier-2 strategy is computational-only | PASS | §1.1 confirms all 4 root causes are INTRINSIC / PROXY-LIMITATION, NOT CPU-fixable rule relaxation; Tier-2 honest-negative close is the compliant path |
| party-twenty-spirit §3.5 — honest framing | PASS | §1 root-cause table + §1.1 categorisation + §0.1 skip rationale |

---

## §4 What Wf-327-F14 deliberately did NOT do (and why)

| Item not done | Reason |
|---|---|
| Re-launch Wf-325E investigation | Wf-325E is the canonical investigation ship surface; re-running duplicates work per task brief item 2 ("confirm + skip") |
| Modify `metrics/by_metric/*.json` | JSONs are MEASURED; honest-negative framing is canonically tagged; no change needed |
| Relax logP ≤ 5 → logP ≤ 7 threshold | Would mask intrinsic Lipinski violation in metallo-drug chemistry; NOT a Tier-2-compliant path |
| Modify the pre-flight gate definition | Gate already correctly wired + counted in `EXPECTED_GATE_COUNTS["all"]` |
| Touch `paper/*` | AGENTS.md §5.3 active — zero paper touches without explicit user OK |

---

## §5 Files touched / created by Wf-327-F14

| File | Change |
|---|---|
| `TODO/completed/327f14_admet_4_investigation.md` | NEW (this doc — verification pass + skip rationale) |

**Total**: 1 file created. ZERO files modified.

---

## §6 Verdict

**Wf-327-F14: VERIFIED + SKIP**. Wf-325E is the canonical ADMET 4 BELOW druglikeness
investigation ship surface, fully wired into pre-flight, with all 4 metric JSONs
carrying the canonical HONEST_NEGATIVE tags + MEASURED provenance. No new
investigation is required; no threshold relaxation is appropriate (all 4 root causes
are INTRINSIC / PROXY-LIMITATION, NOT RULE_TOO_STRICT or METRIC_DEFINITION). The
Tier-2 strategy per AGENTS.md §5.4 governs: honest-negative close is the
computational-only compliant path; recovery via W₂ alignment + chemprop v2
multi-task is documented in Wf-325E §5 + WEAK_TO_STRONG §3.6 but BLOCKED on
USER_OK per AGENTS.md §5.1.

If the user wants to actuate the recovery path, they can:

- OK the W₂ Wasserstein regression alignment + chemprop v2 multi-task retrain
  (Wf-325E §5 steps 1+2) per AGENTS.md §5.1 kill_then_adapt.
- Approve threshold-relaxation experiments if they want a fallback track
  (CAUTION: per §4 above, this would mask intrinsic chemistry, NOT recommended).

Otherwise, the Wf-325E closeout stands as the ship surface for the ADMET 4 BELOW
druglikeness honest-negative cluster.

**End of Wf-327-F14 verification pass.**