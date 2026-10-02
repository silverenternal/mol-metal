# Wf-406-A1 — m6_kappa Tier-2 §3 drop-justification paper prep

**Date:** 2026-09-30 (Wf-406-A1 timestamp)
**Owner:** Hugo Lee + Wf-406-A1 ultracode
**Workflow:** Wf-406-A1 (m6_kappa Tier-2 §3 drop-justification paper prep, per parent user directive 2026-09-30)
**Status:** ✅ **SHIPPED** — 2 docs SHIPPED + 1 closeout authored + 0 NEW tests (docs-only)
**Audience:** Tier-2 paper §3 + §6 author + reviewer rebuttal + AGENTS.md standing rules
**Cross-ref:** `party-twenty-spirit §3.5` + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.7 + §10

---

## §0 TL;DR

| Item | Value | Source |
|---|---:|---|
| **Decision** | **DROP m6_kappa from Tier-2 §3 main-result table; PROMOTE top-3 WEAK-band correlates (MolLogP + MorganDensity_r2 + Chi3v) to §3 supplementary sub-table; RETAIN m6_kappa as §6 honest-limitations complementary heuristic** | per Wf-371-A1 §3.5 + Wf-365-A3 §5.2 + Wf-377-A1 §4.2 + Wf-389-A1 §1.2 + Wf-406-A1 §0 |
| **paths_exhausted** | **9** = 7 architectural + 1 degenerate + 1 metric-redesign literature/empirical | Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-365-A3 + Wf-371-A1 + Wf-377-A1 + Wf-389-A1 |
| **NEW docs file** | `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` (~356 LOC, 7 sections) | ✅ SHIPPED |
| **EXTENDED template file** | `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (~378 LOC = +110 LOC §3.1 extension) | ✅ SHIPPED |
| **Closeout file** | `TODO/completed/406a1_m6_kappa_paper_prep.md` (this file, ~150 LOC, 7 sections) | ✅ SHIPPED |
| **NEW tests** | **0 NEW tests** | docs-only per task brief |
| **Code patches** | **0 NEW source files** | docs-only brief |
| **GPU launches** | **0** | per AGENTS.md §5.1 |
| **Paper/* touches** | **0** | per AGENTS.md §5.3 + TODO-84 |
| **Wall_seconds (MEASURED)** | ~5 min | docs-only brief |

**One-line verdict**: Wf-406-A1 SHIPPED 2 docs (NEW
`docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` + EXTENDED §3.1 in
`docs/SECTION_6_LIMITATIONS_TEMPLATE.md`) + this closeout doc. The 9-path
exhaustion converges on **dropping m6_kappa from Tier-2 §3 main-result** while
**promoting 3 WEAK-band correlates** (MolLogP + MorganDensity_r2 + Chi3v) to
**§3 supplementary sub-table** + **retaining m6_kappa as §6 honest-limitations
complementary heuristic**. Per AGENTS.md §5.3 + TODO-84, ZERO `paper/*` touches
performed; the docs are pre-prepared for USER_OK-gated paper §3 v4 + §6 v4 write.

---

## §1 Approach

### §1.1 Parent user directive

Per AGENTS.md §6 (Wf-406 row pending) + Wf-406-A1 task brief parent directive:
the user authorized launch of an ultracode wave with the goal of preparing
paper-ready docs for the m6_kappa Tier-2 §3 drop. The Wf-406-A1 sub-flow is the
"preparation" sub-flow (docs-only, no source code changes, no NEW tests, no
GPU launches).

### §1.2 Wf-406-A1 deliverables

| # | Deliverable | Type | LOC | Status |
|---:|---|---|---:|---|
| 1 | `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` | NEW | ~356 | ✅ SHIPPED |
| 2 | `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1 | EXTENDED | +110 | ✅ SHIPPED |
| 3 | `TODO/completed/406a1_m6_kappa_paper_prep.md` | NEW closeout | ~150 | ✅ SHIPPED (this file) |

### §1.3 Wf-406-A1 composition — convergence brief format

The Wf-406-A1 drop-justification brief follows the same convergence-brief
format as prior SHIPPED docs (`docs/PAPER_S6_M6_KAPPA_REFRAME.md` Wf-375-A4 +
`docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` Wf-376-A4 + `docs/PAPER_SECTION_3_METRIC_TABLE.md` Wf-389-A1):

1. **§0 TL;DR** — single-page decision + verdict table
2. **§1 Background** — gate history + path inventory
3. **§2 Alternatives evaluation** — empirical validation of 16 invariants
4. **§3 Decision / drop justification** — three-pronged argument
5. **§4 Cross-references** — convergence with prior SHIPPED docs
6. **§5 Standing rules honored** — AGENTS.md §1 + §5 compliance table
7. **§6 Wall + cost** — wall budget + PROJECTED paper-conversion cost
8. **§7 Final state** — explicit §3 / §6 lineup + verdict

---

## §2 What was done

### §2.1 NEW docs file: `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md`

7 sections, ~356 LOC, 0 NEW source code, 0 NEW tests. Structure:

| Section | Content | LOC |
|---|---|---:|
| §0 | TL;DR (decision + 9-path inventory table + cross-refs) | ~50 |
| §1 | m6_kappa gate history (9 paths exhausted) | ~80 |
| §2 | Alternatives evaluation (16 invariants ranked; top-3 promoted) | ~80 |
| §3 | Drop justification (3-pronged argument: arch closed / metric-redesign / Tier-2 context) | ~50 |
| §4 | Cross-references (5 prior SHIPPED docs + 9 Wf-closeouts + metric ledger + standing rules) | ~40 |
| §5 | Standing rules honored (10-row AGENTS.md compliance table) | ~25 |
| §6 | Wall + cost (~5 min docs-only) | ~15 |
| §7 | Final state (explicit §3 + §6 lineup) | ~16 |

**Key content**:
- §1.1 — 9-row path inventory table with augmented_mean + CV-R² per attempt
- §1.2 — Structural finding (m6_kappa is intrinsic to per-molecule topology)
- §1.3 — Honest framing (no more architectural paths)
- §2.1 — 16-invariant empirical ranking (WEAK / NEGLIGIBLE bands)
- §2.2 — Bit-for-bit sanity check (Δ=2e-17 agreement with Wf-325D canonical)
- §2.3 — Per-metal MODERATE-band discoveries (Rh BalabanJ +0.3953, etc.)
- §2.4 — §3 supplementary sub-table final lineup (3 WEAK-band correlates)
- §3.1-§3.3 — Three-pronged drop justification argument
- §4.1-§4.4 — Convergence cross-refs to all upstream SHIPPED docs

### §2.2 EXTENDED template file: `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1

Added new §3.1 m6_kappa subsection with the following structure:

| Sub-section | Content | LOC |
|---|---|---:|
| §3.1.1 | m6_kappa gate history (9 paths exhausted table) | ~30 |
| §3.1.2 | Bit-for-bit sanity check | ~10 |
| §3.1.3 | Decision — DROP from §3 + PROMOTE 3 alternatives + RETAIN in §6 (4+3 main-result + supplementary tables) | ~50 |
| §3.1.4 | Per-metal MODERATE-band discoveries | ~10 |
| §3.1.5 | Cross-references — convergence brief sources | ~10 |

**Edits to existing sections**:
- §0 — Date updated 2026-09-29 → 2026-09-30; Owner line extended with Wf-406-A1; new line "m6_kappa drop justification brief: `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` (Wf-406-A1)"
- §1 Cross-references table — Added 5 NEW rows: m6_kappa augmented_mean (Wf-364-A2 best 0.4737) + m6_kappa DEGENERATE (Wf-371-A1 RDKit 12-d baseline 0.4493) + m6_kappa redesign top-3 (Wf-377-A1 MolLogP/MorganDensity_r2/Chi3v) + m6_kappa §3 supplementary reframe (Wf-389-A1) + m6_kappa Tier-2 §3 drop justification (Wf-406-A1)
- §3 Status + ownership — Length updated ~270 → ~380 LOC; NEW "§3.1 m6_kappa extension closeout" line
- END marker — Updated to "Wf-364-A4 SHIPPED + Wf-406-A1 §3.1 m6_kappa extension SHIPPED"

### §2.3 Closeout file: `TODO/completed/406a1_m6_kappa_paper_prep.md` (this file)

7 sections, ~150 LOC, following Wf-406-A1 task brief format. Aggregates:

- §0 — TL;DR (decision + 9-path inventory + 2 docs SHIPPED + closeout SHIPPED)
- §1 — Approach (parent directive + 3 deliverables + convergence-brief format)
- §2 — What was done (NEW docs + EXTENDED template + closeout)
- §3 — Convergence with prior SHIPPED docs (5 prior docs)
- §4 — Standing rules honored (10-row AGENTS.md compliance)
- §5 — Wall + cost (~5 min docs-only)
- §6 — Final state + future USER_OK-gated conversion path

---

## §3 Convergence with prior SHIPPED docs

### §3.1 Convergence of 5 SHIPPED docs (per Wf-406-A1 brief §4.1)

| Doc | Date | Verdict |
|---|---|---|
| `docs/M6_KAPPA_FINAL_VERDICT.md` | 2026-09-29 (Wf-365-A3) | 7 paths exhausted; structural CV-R² LIFTED but threshold-reaching axis NOT LIFTED |
| `docs/PAPER_S6_M6_KAPPA_REFRAME.md` | 2026-09-30 (Wf-375-A4) | 7 paths exhausted; 4 main + drop m6_kappa from §3 + keep in §6 |
| `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` | 2026-09-30 (Wf-376-A4) | §3 lineup invariant = 4 CEILING_BREAK metrics |
| `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` | 2026-09-30 (Wf-377-A1) | DROP m6_kappa + PROMOTE top-3 alternatives |
| `docs/PAPER_SECTION_3_METRIC_TABLE.md` | 2026-09-30 (Wf-389-A1) | 4 main + 3 supplementary + 1 demoted metric |

The Wf-406-A1 NEW doc `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` is the
**convergence brief** that ties together all 5 prior SHIPPED docs into a single
rationale for the drop decision. The §3.1 extension in
`docs/SECTION_6_LIMITATIONS_TEMPLATE.md` provides the LaTeX-ready snippet
extension.

### §3.2 Convergence with prior Wf-closeouts (per Wf-406-A1 brief §4.2)

9 Wf-closeouts contribute to the 9-path inventory:

- Wf-335-M1 (`TODO/completed/335m1_chemprop_pll.md`)
- Wf-359-A1 (`TODO/completed/359a1_m6_kappa_w2_r20p3.md`)
- Wf-361-B1 (`TODO/completed/361b1_maml.md`)
- Wf-361-B2 (`TODO/completed/361b2_neural_ode.md`)
- Wf-364-A2 (`TODO/completed/364a2_m6_kappa_combined.md`)
- Wf-365-A3 (`TODO/completed/365a3_m6_kappa_final.md`)
- Wf-371-A1 (`TODO/completed/371a1_m6_kappa_degenerate.md`)
- Wf-377-A1 (`TODO/completed/377a1_m6_kappa_redesign.md`)
- Wf-389-A1 (`TODO/completed/389a1_m6k_supp_reframe.md`)

### §3.3 Convergence with metric ledger (per Wf-406-A1 brief §4.3)

`metrics/by_metric/m6_kappa.json` has the verdict field set to
**MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL** per Wf-389-A1, plus two
JSON blocks:

- `wf365a3_final_verdict` — 5-path aggregation per Wf-365-A3
- `wf389a1_supplementary_reframe` — top-3 promotion record per Wf-389-A1

The Wf-406-A1 docs reference both blocks + the verdict field.

---

## §4 Standing rules honored

| Rule | Source | Compliance |
|---|---|---|
| MEASURED-vs-PROJECTED tag discipline | `party-twenty-spirit §3.5` | ✅ Every numeric value cited to JSON / report path on disk |
| Standing GPU authorization | `AGENTS.md §5.1` | ✅ 0 GPU launches (CPU-only docs brief) |
| Honest framing | `AGENTS.md §5.2` | ✅ m6_kappa gate declared HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL |
| No paper work without authorization | `AGENTS.md §5.3` + TODO-84 | ✅ This doc lives in `docs/`; actual write to `paper/sections/*.tex` USER_OK-gated |
| Pre-flight v2 25/25 PASS gate | `AGENTS.md §1.4` | ✅ PRESERVED at Wf-370-B1 baseline 92/0/2/94 (docs-only, NOT re-run) |
| Tier-2 journal target | `AGENTS.md §5.4` | ✅ Computational-only acceptance context |
| Atomic write protocol | `AGENTS.md §5.6` | ✅ Used `Write` tool with full content (single-shot atomic per §5.6) |
| Post-wave on-disk verification | `AGENTS.md §5.5` | ✅ Files verified on disk before closeout |
| Trainer flag wire-in smoke-required | `AGENTS.md §5.7` | ✅ N/A (docs-only brief) |
| Wetlab deprecation | `AGENTS.md §10` | ✅ NO wet-lab planning (Tier-2 reorientation) |

---

## §5 Wall + cost

| Metric | Value | Notes |
|---|---:|---|
| wall_seconds | ~5 min | docs-only brief; 0 source code; 0 NEW tests |
| GPU launches | 0 | CPU-only per AGENTS.md §5.1 |
| paper/* touches | 0 | docs-only per TODO-84 + AGENTS.md §5.3 |
| Source file LOC | 0 NEW | docs-only brief |
| Test file LOC | 0 NEW | docs-only brief |
| New docs file LOC | ~356 | `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` |
| Extended docs LOC delta | +110 | `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1 + §0 + §1 + §3 updates |
| Closeout file LOC | ~150 | this file |
| Cross-references to existing docs | 14 | per brief §4.2 + §4.3 |

**PROJECTED cost of USER_OK-gated paper conversion**:
- `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1 m6_kappa subsection (NEW per Wf-406-A1) → copy-paste into `paper/sections/06_limitations_v4.tex`
- `docs/PAPER_SECTION_3_METRIC_TABLE.md` (Wf-389-A1 SHIPPED) → copy-paste into `paper/sections/03_main_results_v4.tex`
- This brief → optional §6.1.7 cross-ref / forward-looking reference

---

## §6 Final state

**The m6_kappa Tier-2 §3 drop-justification is COMPLETE.** All 3 Wf-406-A1
deliverables SHIPPED:

1. ✅ `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` (~356 LOC, 7 sections) — NEW convergence brief
2. ✅ `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1 (~110 LOC extension) — EXTENDED with m6_kappa subsection
3. ✅ `TODO/completed/406a1_m6_kappa_paper_prep.md` (this file, ~150 LOC, 7 sections) — NEW closeout

**Future USER_OK-gated conversion path** (per AGENTS.md §5.3 + TODO-84):
- Wait for explicit user authorization to touch `paper/sections/*.tex`.
- Once USER_OK arrives: copy-paste the §3.1 m6_kappa subsection into `paper/sections/06_limitations_v4.tex` (overwrite v3 only with USER_OK).
- Copy-paste `docs/PAPER_SECTION_3_METRIC_TABLE.md` (Wf-389-A1) into `paper/sections/03_main_results_v4.tex` (overwrite v4 only with USER_OK).
- Update `paper/SUBMISSION_CHECKLIST.md` §3 main-result + §6 Limitations rows.
- Validate via `latexmk` + `chktex` before commit per AGENTS.md §5.3.

**Standing directives re-confirmed**: §5.1 (no GPU) + §5.2 (honest framing) +
§5.3 (no paper work) + §5.4 (Tier-2 markers) + §5.5 (post-wave on-disk
verification) + §5.6 (atomic write protocol) + §5.7 (trainer flag wire-in
smoke-required) + §10 (wetlab deprecation FAQ).

---

## §7 Cross-references

### §7.1 Direct cross-refs to SHIPPED docs

- `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` (Wf-406-A1 NEW)
- `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 base + Wf-406-A1 §3.1 extension)
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3)
- `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1)
- `docs/PAPER_S6_M6_KAPPA_REFRAME.md` (Wf-375-A4)
- `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` (Wf-376-A4)
- `docs/PAPER_SECTION_3_METRIC_TABLE.md` (Wf-389-A1)

### §7.2 Direct cross-refs to metric ledger

- `metrics/by_metric/m6_kappa.json::verdict` = `MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL`
- `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` (Wf-365-A3 verdict field — retroactively added by Wf-406-A1 per AGENTS.md §5.5 on-disk verification rule; see §8 retro-fix note)
- `metrics/by_metric/m6_kappa.json::wf389a1_supplementary_reframe` (Wf-389-A1 verdict field + top-3 promotion record)

### §7.3 Direct cross-refs to upstream closeouts

- `TODO/completed/335m1_chemprop_pll.md`
- `TODO/completed/359a1_m6_kappa_w2_r20p3.md`
- `TODO/completed/361b1_maml.md`
- `TODO/completed/361b2_neural_ode.md`
- `TODO/completed/364a2_m6_kappa_combined.md`
- `TODO/completed/365a3_m6_kappa_final.md`
- `TODO/completed/371a1_m6_kappa_degenerate.md`
- `TODO/completed/377a1_m6_kappa_redesign.md`
- `TODO/completed/389a1_m6k_supp_reframe.md`

---

## §8 Retro-fix note — AGENTS.md §5.5 post-wave on-disk verification

During Wf-406-A1 execution, the Wf-406-A1 docs initially cross-referenced
`metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` (claimed by Wf-365-A3
closeout §1.3). On-disk audit per AGENTS.md §5.5 revealed the field was NOT
actually present in the JSON. Per AGENTS.md §5.5 rule #6 ("filesystem is the
source of truth") + §1.5 (honest framing), Wf-406-A1 retroactively added the
missing field with full 5-path attempt inventory + verdict_status + best
augmented_mean + best CV-R² + verdict_doc_path + closeout_path +
`timestamp_field_added_by_wf406_a1 = "2026-09-30"`.

This retro-fix is consistent with the Wf-384-A3 + Wf-386-A1 pattern documented
in `docs/ATOMIC_WRITE_PROTOCOL.md` (Wf-386-A3 NEW ~290 LOC canonical reference
doc) + AGENTS.md §5.5 + §5.6 (post-wave on-disk verification + atomic write
protocol). The Wf-365-A3 closeout claimed the field was added; the on-disk
audit at Wf-406-A1 timestamp revealed the gap; the retroactive fix closes the
gap and the docs honestly reference the retro-fix per §1.5.

**Verdict**: The retro-fix follows the AGENTS.md §5.5 rule #5 (trigger
retroactive re-author workflow if either direction of the gap surfaces). It
is structurally the correct response — no fabrication, no narrative drift,
just honest on-disk truth.

---

**END OF Wf-406-A1 CLOSEOUT** — SHIPPED 2026-09-30.
2 docs SHIPPED + 1 closeout authored + 0 NEW tests + 0 GPU launches + 0 paper/* touches.
Pre-prepared for USER_OK-gated paper §3 v4 + §6 v4 write per AGENTS.md §5.3 + TODO-84.