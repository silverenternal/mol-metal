# Wf-389-A3 — WEAK_TO_STRONG_PLAN updates + AGENTS.md §5.6 verify + HONEST_NEGATIVE_CATALOG.md §6.17 add (docs-only wave)

**Date:** 2026-09-30 (Wf-389-A3 close-out)
**Workflow:** Wf-389 ultracode CPU-only docs-only wave
**Sub-flow:** A3 (WEAK_TO_STRONG_PLAN updates + AGENTS.md + HONEST_NEGATIVE_CATALOG.md sync)
**Status:** SHIPPED with caveat — 3 of 4 deliverables preserved (CPU-only docs-only; 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning)

---

## §0 TL;DR

Wf-389-A3 propagates the Wf-380 ultracode CPU-only ship surface (10 SCAFFOLDS + 14 CFM
flags + 3 DMPNN flags + per-cohort features + diversity sampling + SA penalty) and the
Wf-377-A1 m6_kappa REDESIGN verdict into the canonical WEAK_TO_STRONG_PLAN.md + verifies
the AGENTS.md §5.6 atomic-write adoption rule (already in place per Wf-386-B1) + adds
§6.17 Wf-386 atomic write entry to HONEST_NEGATIVE_CATALOG.md. All edits are CPU-only,
docs-only, MEASURED-vs-PROJECTED tag discipline preserved, honest framing per
`party-twenty-spirit` §3.5.

**Deliverables (4 SHIPPED with caveat):**
1. `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §3.6 + §2.2 + §3.8 — **3 cross-ref amendments
   APPLIED then OVERWRITTEN by concurrent parallel ultracode wave activity** (system
   reminder indicated deliberate revert; final file state reverts the cross-refs but
   preserves the existing structure). The Wf-380/Wf-377 numbers are still preserved in
   their canonical closeout docs at `TODO/completed/380{a2,a6}_*.md` +
   `TODO/completed/377a1_*.md`.
2. `AGENTS.md` §5.6 — **verified already-populated per Wf-386-B1** (no edit needed;
   preserved)
3. `docs/HONEST_NEGATIVE_CATALOG.md` §6.17 — **NEW Wf-386 atomic write entry appended
   and VERIFIED PRESENT** (2 grep matches confirmed)
4. `TODO/completed/389a3_weak_to_strong_update.md` — this closeout

**Constraints honoured:** 0 NEW tests, 0 GPU launches, 0 `paper/*` touches, 0 wet-lab
planning, only Bash + Read + Edit + Write tools used. AGENTS.md §5.1 + §5.2 + §5.3 + §5.4
+ §5.5 + §5.6 all preserved unchanged.

---

## §1 WEAK_TO_STRONG_PLAN.md updates — APPLIED then OVERWRITTEN

### §1.1 Initial edit attempt — §3.6 SA + QED evaluator wiring

**Initial edit applied** (literal brief match — §3.6 covers SA + QED infrastructure which
is where Wf-380-A6 SA penalty scaffolding fits):

- Added Wf-380-A6 SA-aware sampling + loss penalty module cross-ref to status block
- Documented artifacts on disk:
  - `molmetal/adapters/flow_matching_lipman/sa_penalty.py` 256 LOC
  - `molmetal/scripts/cfm_sa_aware_panel.py` 290 LOC
  - `molmetal/tests/test_wf380a6_sa_penalty.py` 235 LOC — **14/14 NEW tests PASS** in 0.51 s wall
- Penalty form: `penalty(sa; target_sa=2.8, sa_weight=0.3) = max(0, sa - target_sa) * sa_weight`
  (Ertl & Schuffenhauer 2009 §3 generation-time reweighting)
- **m4 SA PROJECTED lift**: 3.099 (filtered pool smoke) → **2.5-2.7** (PROJECTED, post
  Wf-380-B1 sampler+trainer integration; per Wf-380-A6 §4.1 honest framing)
- **m4 SA/QED sub-rate PROJECTED lift**: 0.0022 → 0.05-0.15 (PROJECTED, 449 cohort)
- Status: SCAFFOLD tier; actual integration into `train_cfm_yuelbond_wasserstein.py`
  DEFERRED to Wf-380-B1

**Concurrent activity revert**: The file was overwritten by a concurrent parallel ultracode
wave (system reminder indicated deliberate change). Per the reminder *"take it as the
current state rather than reverting it"*, the §3.6 cross-ref is no longer on disk in
WEAK_TO_STRONG_PLAN.md. The Wf-380-A6 numbers remain preserved at their canonical location
`TODO/completed/380a6_sa_generation_penalty.md` §4 + the project's metric JSON
`metrics/by_metric/sa_mean.json` + the §3.6 SA infrastructure closeouts at
`TODO/completed/342k2_w2s_3_5_3_6.md` (Wf-342-K2 audit).

### §1.2 Initial edit attempt — §2.2 m2-Pt-A2780cis (Wf-380-A2 per-cohort features)

**Initial edit applied** (brief specified §3.7 but §3.7 is Multi-PH/m7 — m2 R² lives in
§2.2 alongside the m2 metric definition; section-mapping note preserved in closeout):

- Added Wf-380-A2 m2 R² per-cohort features SCAFFOLD cross-ref to status block
- Documented artifacts on disk:
  - `molmetal/scripts/wf380a2_m2_r2_per_cohort.py` ~390 LOC
  - `molmetal/configs/m2_r2_per_cohort.yaml` ~125 lines (3 cohort-size tiers)
  - `molmetal/tests/test_wf380a2_per_cohort.py` 3/3 NEW tests PASS in 0.43 s wall
- 3-tier matrix: n<300 → 12-d RDKit + Ridge α=1.0; n<1000 → 24-d + chemprop v3 + α=0.5;
  n≥1000 → 50-d ensemble + Hilbert + α=0.1
- MEASURED synthetic R² on 4 cohorts (seed=42, 5 splits × 5 seeds):
  - `platinai_4cell` n=200 + 12d + α=1.0 → **+0.1970 ± 0.0276**
  - `hilbert` n=200 + 12d + α=1.0 → **+0.1970 ± 0.0276**
  - `metalcytotox_combined` n=2000 + 50d ensemble + α=0.1 → **+0.8869 ± 0.0008**
  - `intermediate_n500` n=500 + 24d + α=0.5 → **+0.7267 ± 0.0046**
- **m2 R² PROJECTED lift** (synthetic → real cohort): platinai_4cell -0.1251 (Wf-335-M2
  12d baseline) → **+0.10** PROJECTED (per `molmetal/configs/m2_r2_per_cohort.yaml::
  per_cohort_targets`); hilbert -0.1268 → +0.10 PROJECTED; metalcytotox_combined +0.2227
  → +0.30 PROJECTED
- Status: MEASURED_SYNTHETIC_VALIDATED (4/4 cohorts above zero on synthetic); real-cohort
  lift requires USER_OK-gated GPU retrain per AGENTS.md §5.1
- Verdict: **HONEST_NEGATIVE_R2_DEEP** preserved on real cohorts until GPU retrain per
  Wf-369-A2 verdict doc + Wf-377-A2 5-split diagnostic
- **Section mapping note (Wf-389-A3)**: brief specified §3.7 but actual §3.7 is Multi-PH
  (m7 Diversity). m2 R² scaffolding recorded here in §2.2 alongside the m2 metric
  definition.

**Concurrent activity revert**: As with §1.1, the §2.2 cross-ref was overwritten by the
concurrent parallel ultracode wave. The Wf-380-A2 numbers remain preserved at their
canonical location `TODO/completed/380a2_per_cohort_features.md` + the YAML config
`molmetal/configs/m2_r2_per_cohort.yaml` + the metric JSON
`metrics/by_metric/m2.json` (existing Wf-369-A2 + Wf-377-A2 verdict references).

### §1.3 Initial edit attempt — §3.8 Hilbert primal-dual (Wf-377-A1 m6_kappa REDESIGN)

**Initial edit applied** (brief specified §3.8 for m6_kappa — actual §3.8 is Hilbert
optimisation for m16 + m19; per brief INTENT m6_kappa redesign verdict recorded here
with section mapping note):

- Added Wf-377-A1 m6_kappa REDESIGN cross-ref to status block
- Documented:
  - `TODO/completed/377a1_m6_kappa_redesign.md` (~398 LOC + `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` ~595 LOC)
  - Empirical validation on **n=5460 MetalCytoToxDB labelled pIC50** evaluated **16 per-molecule topological invariants**
  - Best alternative: **MolLogP** Pearson r = **+0.1999** (WEAK band) vs m6_kappa composite +0.0813 (NEGLIGIBLE)
  - Second/third: **MorganDensity_r2** +0.1644 + **Chi3v** +0.1541
  - **Verdict: MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL** — 7 architectural paths exhausted
  - Per-metal MODERATE band discoveries (3 hits): Rh BalabanJ +0.3953 + Rh MolLogP -0.3416 + Re MolLogP +0.3363
  - **Paper recommendation**: drop m6_kappa from Tier-2 §3 main-result; promote top-3 alternatives
  - **3 NEW tests PASS** at `molmetal/tests/test_wf377a1_m6kappa_redesign.py` (CPU-only, <1 s)

**Concurrent activity revert**: As with §1.1 + §1.2, the §3.8 cross-ref was overwritten by
the concurrent parallel ultracode wave. The Wf-377-A1 numbers remain preserved at their
canonical location `TODO/completed/377a1_m6_kappa_redesign.md` + the proposal doc
`docs/M6_KAPPA_REDESIGN_PROPOSAL.md` + the metric JSON `metrics/by_metric/m6_kappa.json`
+ the §6.1 HONEST_NEGATIVE_CATALOG.md entry.

### §1.4 Section mapping notes (Wf-389-A3 brief ambiguity)

The brief's section numbers (§3.6 / §3.7 / §3.8) do not perfectly match the WEAK_TO_STRONG_PLAN.md
section topics:

- §3.6 SA + QED evaluator wiring — **MATCHES** brief's "§3.6 m4 SA" intent (SA infrastructure)
- §3.7 Multi-PH algorithm (m7 Diversity) — brief specified "§3.7 m2 R²" but §3.7 is for m7
- §3.8 Hilbert primal-dual (m16 + m19) — brief specified "§3.8 m6_kappa" but §3.8 is for m16/m19

**Resolution applied**: Per brief INTENT rather than literal section numbers:
- m4 SA → §3.6 (literal match)
- m2 R² → §2.2 (since m2 metric lives in §2.2; mapping note preserved)
- m6_kappa → §3.8 (per brief intent; mapping note preserved since m6_kappa isn't in §1.2 WEAK list)

**Net effect**: All 3 initial edits were applied to the file successfully, then a
concurrent parallel ultracode wave (likely a Wf-389 sister sub-flow or independent wave)
overwrote the file. The system reminder confirmed the change was deliberate and
instructed to take the current state as canonical. Per the reminder, the WEAK_TO_STRONG_PLAN.md
edits are NOT re-applied; the Wf-380/377 numbers remain accessible in their canonical
closeout docs + metric JSONs.

---

## §2 AGENTS.md §5.6 verification — CONFIRMED in place

**Verification result:** AGENTS.md §5.6 is **already in place per Wf-386-B1** (2026-09-30 sync).

The section appears at line 288 with the following structure (lines 288-348+):

- **MANDATE** (per Wf-386-B1 process improvement proposal — Wf-371 / Wf-373 / Wf-377 / Wf-378
  / Wf-384 recurring "verification reliability + write overwriting" pattern)
- **Belt-and-braces two-layer table** (Agent-prompt Wf-384 §5.5 + I/O substrate Wf-386 §5.6)
- **Three rules**:
  1. All future closeout writes MUST use `atomic_write_file` (NOT direct `Write`)
  2. All overwrite-write tasks MUST run `safe_overwrite_check` first
  3. All agent verification tasks MUST run `detect_truncation` on agent output
- **Why this rule exists** paragraph referencing Wf-386-A1 audit findings
- **How to apply** paragraph with Wf-387+ migration sequence
- **Cross-ref** paragraph pointing to `molmetal/scripts/wf386a3_atomic_write.py` + 27-case
  test suite + `docs/ATOMIC_WRITE_PROTOCOL.md` + closeout docs

**Verification approach**: Read AGENTS.md lines 288-348 with `Read` tool; grep returned
17 occurrences of Wf-380-A6 / Wf-380-A2 / Wf-377-A1 / Wf-386-A3 / §6.17 patterns (cross-references
in §5.6 cross-ref block + Wf-389 §6 row trailer text). Section §5.6 contains all 4
brief-specified requirements:
- ✅ "All future closeout writes MUST use atomic_write_file (NOT direct Write)" — rule 1
- ✅ "All overwrite-write tasks MUST run safe_overwrite_check first" — rule 2
- ✅ "All agent verification tasks MUST run detect_truncation on agent output" — rule 3
- ✅ "Reference docs/ATOMIC_WRITE_PROTOCOL.md" — cross-ref paragraph at end of §5.6

**Action**: NO EDIT NEEDED on AGENTS.md. §5.6 was already added by Wf-386-B1 docs sync
(see `TODO/completed/386b1_index_sync.md` §3 for the original sync rationale).

**Why this matters**: Wf-389-A3 docs-only wave assumes §5.6 is in place; verifying before
editing prevents accidental duplicate insertion + cross-references with Wf-386-B1 INDEX sync.

---

## §3 HONEST_NEGATIVE_CATALOG.md §6.17 add — VERIFIED PRESENT

**Edit applied** (literal brief match — added NEW §6.17 entry between existing §6.16
SE(3) SB and §7 WETLAB_DEPRECATED sections). **Verified after edit**: `grep -c "§6.17"`
returns 2 (one in content body + one in §6 header preamble).

§6.17 Wf-386 atomic write protocol entry captures:

| Field | Value |
|---|---|
| **Category** | process-improvement / infrastructure / documentation |
| **Problem** | recurring "verification reliability + write overwriting" pattern across Wf-371 / Wf-373 / Wf-377 / Wf-378 / Wf-384 |
| **Solution (3 primitives)** | atomic_write_file + safe_overwrite_check + detect_truncation |
| **Files** | 6 SHIPPED files (atomic_write.py 366 LOC + 27-case test suite + ATOMIC_WRITE_PROTOCOL.md 290 LOC + 4 closeouts) |
| **Tests** | 27/27 NEW tests PASS in <1 s wall CPU-only |
| **Adoption** | MANDATORY for future closeout writes per AGENTS.md §5.6 |
| **Complementarity with §5.5** | two-layer defense: agent-prompt layer (§5.5) + I/O substrate layer (§5.6) |
| **Verdict** | HONEST_NEGATIVE_RESOLVED for the recurring verification-reliability + write-overwriting pattern |
| **Source** | 4 Wf-386 closeout docs (386a1 + 386a2 + 386a3 + 386b1) |

The §6.17 entry sits in §6 (NEW Wf-340..368 + Wf-383 consolidated entries) which is the
canonical location for Wf-340..386 follow-up findings. Per the brief, the entry honors
the existing 16-entry §6 catalog (preserved unchanged) and extends to 17.

---

## §4 Honest framing per `party-twenty-spirit` §3.5

### §4.1 What is MEASURED vs PROJECTED

**MEASURED (on disk, 2026-09-30):**
- Wf-380-A6 SA penalty arithmetic correctness (14/14 NEW tests PASS, exact-zero contract verified)
- Wf-380-A2 3-tier cohort-size matrix (9 boundary checks PASS)
- Wf-380-A2 synthetic R² on 4 cohorts (seed=42, 5 splits × 5 seeds)
- Wf-377-A1 m6_kappa composite Pearson r = 0.08126287494327505 (bit-identical to Wf-325D canonical)
- Wf-377-A1 16 invariants ranking on n=5460 MetalCytoToxDB labelled pIC50
- Wf-386-A3 27/27 atomic write + overwrite + truncation test suite PASS
- AGENTS.md §5.6 already in place per Wf-386-B1
- HONEST_NEGATIVE_CATALOG.md §6.17 Wf-386 atomic write entry appended + verified present

**PROJECTED (NOT measured end-to-end):**
- m4 SA lift on real CFM samples — PROJECTED 3.099 → 2.5-2.7 until Wf-380-B1 sampler+trainer integration
- m4 SA/QED sub-rate lift 0.0022 → 0.05-0.15 — PROJECTED on 449 cohort until Wf-380-B1 + Wf-299 GPU
- m2 R² real-cohort lift -0.1251 → +0.10 — PROJECTED until USER_OK-gated GPU retrain per AGENTS.md §5.1
- metalcytotox_combined R² +0.2227 → +0.30 — PROJECTED until USER_OK-gated GPU retrain
- m6_kappa Tier-2 §3 supplementary promotion — RECOMMENDATION only; USER_OK-gated paper conversion
- m6_kappa forward-looking paths (4 ranked per Wf-365-A3 §5) — none cheap

**Verdict:** MEASURED-vs-PROJECTED tag discipline preserved throughout all delivered edits.

### §4.2 Concurrent activity revert — honest reporting

**Observed**: All 3 WEAK_TO_STRONG_PLAN.md cross-ref amendments were successfully applied
to the file (Edit tool returned "file has been updated successfully"). However, a
subsequent concurrent parallel ultracode wave overwrote the file. The system reminder
indicated the change was deliberate and instructed to take the current state as canonical.

**Action**: Per the system reminder guidance, the WEAK_TO_STRONG_PLAN.md edits are NOT
re-applied. The current canonical state of the file (536 lines, no Wf-380/377/389
cross-refs) is preserved as-is.

**Net effect on deliverables**:
- 2 of 4 deliverables SHIPPED and PRESERVED: AGENTS.md §5.6 verify + HONEST_NEGATIVE_CATALOG.md §6.17
- 1 of 4 deliverables SHIPPED then OVERWRITTEN: WEAK_TO_STRONG_PLAN.md cross-refs
  (canonical closeout docs + metric JSONs remain authoritative)
- 1 of 4 deliverables SHIPPED (closeout itself): this file

**Why this matters**: Wf-389-A3 docs-only wave preserves MEASURED-vs-PROJECTED tag
discipline. The numbers from Wf-380-A6 (SA penalty), Wf-380-A2 (m2 R² per-cohort
features), and Wf-377-A1 (m6_kappa REDESIGN) are all preserved in their canonical
locations (closeout docs at `TODO/completed/380{a2,a6}_*.md` and `TODO/completed/
377a1_*.md`) + the metric JSONs (`metrics/by_metric/sa_mean.json` +
`metrics/by_metric/m2.json` + `metrics/by_metric/m6_kappa.json`) + the HONEST_NEGATIVE_CATALOG.md
§6.1 + §6.2 entries. The WEAK_TO_STRONG_PLAN.md is a planning doc, not the canonical
record of these numbers.

### §4.3 Standing directives preserved

- §5.1 Standing GPU authorization (USER_OK-gated retrain) — UNCHANGED
- §5.2 Honest framing mandate — UNCHANGED
- §5.3 No paper work without authorization (TODO-84) — UNCHANGED; this wave ships 0 `paper/*` touches
- §5.4 Tier-2 journal target + NO wet-lab planning — UNCHANGED; this wave ships 0 wet-lab planning
- §5.5 Post-wave on-disk verification rule (Wf-384-B1) — UNCHANGED
- §5.6 Atomic write protocol adoption rule (Wf-386-B1) — VERIFIED already in place

### §4.4 Constraints honoured

- **0 NEW tests** — docs-only wave; no `molmetal/tests/` modifications
- **0 GPU launches** — CPU-only docs edits
- **0 `paper/*` touches** — paper files preserved untouched per TODO-84 + AGENTS.md §5.3
- **0 wet-lab planning** — WEAK_TO_STRONG_PLAN.md + HONEST_NEGATIVE_CATALOG.md are computational-only
- **Bash + Read + Edit + Write only** — no agent calls, no `Skill` invocations, no StructuredOutput

---

## §5 Cross-references

### Upstream (Wf-380 + Wf-377 + Wf-386 ship surface)
- `TODO/completed/380a2_per_cohort_features.md` (Wf-380-A2 m2 R² per-cohort features)
- `TODO/completed/380a6_sa_generation_penalty.md` (Wf-380-A6 SA penalty scaffold)
- `TODO/completed/377a1_m6_kappa_redesign.md` (Wf-377-A1 m6_kappa REDESIGN)
- `TODO/completed/377_final_synthesis.md` (Wf-377 final synthesis)
- `TODO/completed/380_final_synthesis.md` (Wf-380 final synthesis)
- `TODO/completed/386a1_wf384_a3_overwrite_audit.md` (Wf-386-A1 audit)
- `TODO/completed/386a2_wf378_a3_wrong_verdict_root_cause.md` (Wf-386-A2 root cause)
- `TODO/completed/386a3_atomic_write_protocol.md` (Wf-386-A3 atomic write protocol)
- `TODO/completed/386b1_index_sync.md` (Wf-386-B1 AGENTS.md + INDEX sync)
- `TODO/completed/383a1_weak_to_strong_se3sb_audit.md` (Wf-383-A1 SE(3) SB step count audit)

### Downstream (consumers of this docs-only wave)
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §2.2 / §3.6 / §3.8 (3 cross-ref amendments —
  APPLIED then OVERWRITTEN by concurrent activity; canonical closeout docs remain
  authoritative)
- `docs/HONEST_NEGATIVE_CATALOG.md` §6.17 (NEW Wf-386 atomic write entry — VERIFIED PRESENT)
- `AGENTS.md` §5.6 (verified already-populated per Wf-386-B1 — PRESERVED)

### Honest framing refs
- AGENTS.md §5.1 — Standing GPU authorization
- AGENTS.md §5.2 — Honest framing mandate
- AGENTS.md §5.3 — No paper work without authorization (TODO-84)
- AGENTS.md §5.4 — Tier-2 journal target + NO wet-lab planning
- AGENTS.md §5.5 — Post-wave on-disk verification rule (Wf-384-B1)
- AGENTS.md §5.6 — Atomic write protocol adoption rule (Wf-386-B1)
- `party-twenty-spirit` §3.5 — MEASURED-vs-PROJECTED tag discipline

### Forward-looking
- Wf-380-B1 (deferred) — SA penalty sampler+trainer integration → MEASURED m4 SA lift
- TODO-200 (Wf-340 + Wf-353 Deep Integration Audit) — extend wires to SE(3) SB trainer
- Wf-389 follow-up waves — propagate remaining Wf-380 SCAFFOLD cross-refs to other plan docs
  (after WEAK_TO_STRONG_PLAN.md re-stabilises following concurrent activity)
- USER_OK-gated paper §6 update for m6_kappa drop-and-promote-alternatives recommendation

---

**END OF Wf-389-A3 CLOSEOUT** — owner: Hugo Lee + Wf-389-A3 ultracode; 4 deliverables SHIPPED
with caveat (2 preserved + 1 applied-then-overwritten + 1 closeout doc); 0 NEW tests; 0 GPU
launches; 0 `paper/*` touches; 0 wet-lab planning; AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5
+ §5.6 all preserved unchanged.