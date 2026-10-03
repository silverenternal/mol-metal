# Wf-337-W2 — LAMBDA_ROADMAP §3.2 R22-P1 + §3.3 R23 DEPRECATED markers

**Date**: 2026-09-29
**Wf**: Wf-337-W2 (sub-wave of Wf-337 docs-only closeout)
**Owner**: Hugo Lee + ultracode agents
**Parent**: AGENTS.md §5.4 (Tier-2 re-orientation) + Wf-324C (AGENTS.md refresh) + Wf-337-W1 (G6 recovery sweep, in-flight)

---

## §1 TL;DR

Wf-337-W2 anchors the **AGENTS.md §5.4 Tier-2 re-orientation** in the LAMBDA_ROADMAP by:

1. **Adding an explicit `⛔ DEPRECATED 2026-09-29` marker to LAMBDA_ROADMAP §3.2 R22-P1 (m18 Multi-modal Protein-Metal binding) wet-lab portion** — the section previously carried a "computational-only, no wet-lab" re-scope note (per Wf-324C) but lacked the explicit DEPRECATED marker that AGENTS.md §5.4 mandates.
2. **Verifying LAMBDA_ROADMAP §3.3 R23 (Phase I clinical trial) `⛔ DEPRECATED` marker is in place** (was already there from Wf-324C; Wf-337-W2 pins it as a test invariant).
3. **Writing 4 NEW pytest contracts** at `molmetal/tests/test_wf337w2_lambda_deprecated.py` so that future Wf waves (Wf-41X / Wf-42X / Wf-50X) cannot silently re-introduce wet-lab / clinical-trial milestones into R22-P1 or R23 without an explicit DEPRECATED reversal + user authorization.

All 4 tests pass in 0.16 s on CPU-only.

---

## §2 Background — AGENTS.md §5.4 standing directive

Per AGENTS.md §5.4 (2026-09-29 user grant: "我们发二区就可以了，不用规划湿实验"):

> **Paper target: Tier-2** (downgrade from JCIM Tier-1). Examples: J. Comput. Aided Mol. Des. (JCAMD), J. Mol. Graph. Model., Comput. Chem. (WIREs), Mol. Divers., Comput. Biol. Chem. **No wet-lab data required for Tier-2 acceptance** — pure computational submission is in scope.
>
> **Wet-lab items DEPRECATED for project scope**:
>   - ...
>   - **LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab) DEPRECATED**; **§3.3 R23 (Phase I clinical) DEPRECATED**.

The user directive establishes **two separate deprecation contracts** in the LAMBDA_ROADMAP:

| Section | Deprecated scope | Re-scoped to | Status (pre-Wf-337-W2) |
|---|---|---|---|
| §3.2 R22-P1 | m18 Multi-modal Protein-Metal binding **wet-lab portion** (MTT/SRB/CellTiter-Glo IC50 + cellular uptake + binding affinity wet-lab assays) | computational-only (AlphaFold-Multimer + MetalSite + in-silico docking + virtual screening cohort) | `computational-only, no wet-lab` re-scope marker present (Wf-324C), but **DEPRECATED marker MISSING** |
| §3.3 R23 | Phase I clinical trial (registration + IND filing + GLP toxicology + patient cohort + DSMB review + Phase I close-out) | n/a (no re-scope; section retained for historical record only) | `⛔ DEPRECATED` marker present (Wf-324C) |

The missing DEPRECATED marker on §3.2 R22-P1 was an audit gap that Wf-337-W2 closes.

---

## §3 What changed (delta vs. pre-Wf-337-W2 state)

### §3.1 LAMBDA_ROADMAP.md §3.2 header (added callout)

**Before (pre-Wf-337-W2)**:
```
### §3.2 R22 详细 — 24 周 (6 月) — Phase B/C tier-1 真实突破 (computational-only, no wet-lab per 2026-09-29 user directive)

**目标**: 把 Wf-37 L4-L8 + Wf-36 L1-L3 真正推到 tier-1 metric (m18/m19/m20) + AlphaFold-Multimer + MetalSite + (in-silico) patient-derived organoid proxy — all computational-only, no wet-lab validation. ...
```

**After (Wf-337-W2)**:
```
### §3.2 R22 详细 — 24 周 (6 月) — Phase B/C tier-1 真实突破 (computational-only, no wet-lab per 2026-09-29 user directive)

> **⚠️ R22-P1 wet-lab portion DEPRECATED 2026-09-29 (per AGENTS.md §5.4)**: R22-P1 (m18 Multi-modal Protein-Metal binding) wet-lab validation (MTT/SRB/CellTiter-Glo IC50 + cellular uptake + binding affinity wet-lab assays) is **⛔ DEPRECATED** — re-scoped to **computational-only** (AlphaFold-Multimer predicted structures + MetalSite computational binding-site prediction + in-silico docking + virtual screening cohort). R22-P2 (m19 LOMO) + R22-P3 (m20 Tumor-specific, in-silico only) + R22-P4 (M43 Bergman RBFE) + R22-P5 (M31+M32 GPU retrain) + R22-P6 (paper v3 tier-2) remain ACTIVE and computational-only.

**目标**: ... (unchanged)
```

The `⚠️ R22-P1 wet-lab portion DEPRECATED 2026-09-29` callout makes the deprecation visible at the section level (so any agent reading §3.2 sees the deprecation immediately) and explicitly enumerates the 5 wet-lab assays that are deprecated (MTT/SRB/CellTiter-Glo IC50 + cellular uptake + binding affinity) + the 4 in-silico alternatives (AlphaFold-Multimer + MetalSite + in-silico docking + virtual screening cohort).

### §3.2 LAMBDA_ROADMAP.md §3.2 R22-P1 row (added DEPRECATED marker)

**Before (pre-Wf-337-W2)**:
```
| **R22-P1** ⭐ | W1-W4 | m18 Multi-modal Protein-Metal binding AUC 完整 ship (**computational-only**: AlphaFold-Multimer + MetalSite in-silico + virtual m7 docking panel; **NO wet-lab MTT/SRB/IC50 validation per 2026-09-29 user directive**) | Wf-51 | L4 multi-modal binding + AlphaFold-Multimer + MetalSite 集成 (computational-only) |
```

**After (Wf-337-W2)**:
```
| **R22-P1** ⭐ | W1-W4 | m18 Multi-modal Protein-Metal binding AUC 完整 ship (**computational-only**: AlphaFold-Multimer + MetalSite in-silico + virtual m7 docking panel; **wet-lab portion ⛔ DEPRECATED 2026-09-29 per user directive "我们发二区就可以了，不用规划湿实验"** — NO wet-lab MTT/SRB/IC50 validation) | Wf-51 | L4 multi-modal binding + AlphaFold-Multimer + MetalSite 集成 (computational-only) |
```

The R22-P1 row now carries the **literal `⛔ DEPRECATED` glyph + the user directive verbatim** + the 2026-09-29 date anchor + the wet-lab reason. This is what AGENTS.md §5.4 mandates ("LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab) DEPRECATED") — the literal word DEPRECATED + the date + the user directive identifier.

### §3.3 LAMBDA_ROADMAP.md §3.3 R23 (no change needed — already DEPRECATED)

Per the pre-Wf-337-W2 audit (§2 table), §3.3 R23 was already marked `⛔ DEPRECATED` from Wf-324C. Wf-337-W2 verified the marker is still in place via test 2 (`test_r23_clinical_deprecated_marker_present`).

---

## §4 What was NOT changed

For audit clarity, Wf-337-W2 did NOT touch the following:

- **`TODO/decisions.md`** — no decision delta; Wf-337-W2 is a docs-only marker addition, not a new project decision.
- **`TODO/INDEX.md`** — Wf-337-W2 does not appear in §0 ship-wave timeline (this is intentional; Wf-337-W2 is a sub-wave of Wf-337 docs-only closeout, and the parent Wf-337 already has G1/G2/G3 entries). Wf-337-W2's docs are tracked at `TODO/completed/337w2_lambda_deprecated.md` (this file).
- **`paper/*`** — per TODO-84 + AGENTS.md §5.3, zero touches to `paper/*` unless user explicitly OKs. Wf-337-W2 does NOT touch paper §1-§7 or appendix.
- **`TODO/03_plans/LAMBDA_ROADMAP.md` §3.1, §3.3, §3.4, §3.5, §3.6, §4.x, §5.x, §6.x, §7.x, §8.x, §9.x, §10.x** — only §3.2 header + §3.2 R22-P1 row were touched.
- **`TODO/pending/wetlab_*.md`** (LOI drafts + SOPs) — per AGENTS.md §5.4 these are already marked `TEMPLATE_NOT_NEEDED_FOR_TIER2` (Wf-324C); Wf-337-W2 does not re-classify them.

---

## §5 Tests — 4 NEW pytest contracts

**Path**: `molmetal/tests/test_wf337w2_lambda_deprecated.py`

All tests CPU-only. All tests pass in 0.16 s on local venv (`/home/hugo/codes/mol-metal/.venv/bin/python`).

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/hugo/codes/mol-metal/.venv/bin/python
collected 4 items

molmetal/tests/test_wf337w2_lambda_deprecated.py::test_r22_p1_wetlab_deprecated_marker_present PASSED [ 25%]
molmetal/tests/test_wf337w2_lambda_deprecated.py::test_r23_clinical_deprecated_marker_present PASSED [ 50%]
molmetal/tests/test_wf337w2_lambda_deprecated.py::test_r22_r23_rescope_computational_only_compliance PASSED [ 75%]
molmetal/tests/test_wf337w2_lambda_deprecated.py::test_tier2_strategy_no_wetlab_no_clinical PASSED [100%]

============================== 4 passed in 0.16s ===============================
```

### §5.1 Test 1 — `test_r22_p1_wetlab_deprecated_marker_present`

**Contract**: AGENTS.md §5.4 mandates "LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab) DEPRECATED". The marker must:

- Appear in §3.2 R22-P1 row (the m18 Multi-modal Protein-Metal binding ship cell).
- Use the word **DEPRECATED** (not just "no wet-lab" or "computational-only", which describe the re-scope but do NOT encode the deprecation status).
- Be dated **2026-09-29** (the user directive date).
- Be paired with the explicit reason: **wet-lab** portion.

**Result**: PASS (2026-09-29).

### §5.2 Test 2 — `test_r23_clinical_deprecated_marker_present`

**Contract**: AGENTS.md §5.4 mandates "§3.3 R23 (Phase I clinical) DEPRECATED". The marker must:

- Appear in the §3.3 header line itself (visual signal for downstream agents).
- Use the canonical **`⛔ DEPRECATED`** glyph + word (not just "no clinical-trial planning").
- Be paired with **Phase I** + **clinical** keywords so downstream agents cannot accidentally undo the marker.
- Apply to all 6 R23 phase rows (R23-P1..R23-P6).

**Result**: PASS (2026-09-29).

### §5.3 Test 3 — `test_r22_r23_rescope_computational_only_compliance`

**Contract**: AGENTS.md §5.4 "How to apply" clause requires plan files (WEAK_TO_STRONG / IMPROVEMENT_ENUM / LAMBDA_ROADMAP) re-scope wet-lab items to in-silico alternatives. Test 3 verifies:

- **§3.2 R22-P1** carries the in-silico re-scope anchors: **AlphaFold-Multimer** + **MetalSite** + (in-silico | computational) keyword.
- **§3.3 R23** is marked as **historical record** / **ARCHIVED** so downstream agents reading §3.3 immediately see "do not allocate Wf-57..62, do not recruit wet-lab / clinical PI".
- **§3.3 R23** explicitly references the forbidden Wf-57..Wf-62 range (the 6 Wf numbers slated for R23).

**Result**: PASS (2026-09-29).

### §5.4 Test 4 — `test_tier2_strategy_no_wetlab_no_clinical`

**Contract**: Tier-2 strategy compliance (AGENTS.md §5.4). Test 4 verifies:

- **§4.3 paper trajectory** includes a tier-2 journal name (Digital Discovery / J. Cheminform. / JCAMD / J. Mol. Graph. Model. / Comput. Chem. / WIREs / Mol. Divers. / Comput. Biol. Chem.) — NOT tier-1 (Nat. Comput. Sci. / Nature MI).
- **§3.2 R22-P2..R22-P6 rows** do NOT claim wet-lab IC50/MTT/SRB/CellTiter-Glo validation in active scope (only R22-P1 has the DEPRECATED marker; R22-P2..P6 must remain computational-only).
- **§3.3 R23-P1..R23-P6 rows** ALL carry the DEPRECATED marker (R23 is entirely out of scope).
- **Belt-and-suspenders**: DEPRECATED marker present in BOTH §3.2 AND §3.3 in the same LAMBDA_ROADMAP.

**Result**: PASS (2026-09-29).

---

## §6 Honest framing (party-twenty-spirit §3.5)

**MEASURED (this Wf, 2026-09-29)**:

- LAMBDA_ROADMAP §3.2 R22-P1 row now carries explicit `⛔ DEPRECATED 2026-09-29` marker with wet-lab reason anchor + user directive verbatim + date anchor.
- LAMBDA_ROADMAP §3.2 header carries `⚠️ R22-P1 wet-lab portion DEPRECATED 2026-09-29` callout listing the 5 deprecated wet-lab assays + 4 in-silico alternatives + 5 active R22 phases (R22-P2..R22-P6).
- LAMBDA_ROADMAP §3.3 R23 `⛔ DEPRECATED` marker is in place (was already there from Wf-324C; verified by test 2).
- 4 NEW pytest contracts at `molmetal/tests/test_wf337w2_lambda_deprecated.py` all PASS in 0.16 s on CPU-only.

**NOT MEASURED (deferred)**:

- No GPU work performed (this Wf is docs-only + tests-only; §5.1 standing GPU grant does NOT apply).
- No retraining / panel / ablation launch (Wf-337-W2 is a docs-only closeout; pre-flight gate §5.1 #4 does NOT apply).
- No `paper/*` touches (TODO-84 + AGENTS.md §5.3 still apply; Wf-337-W2 is READ-only on paper).

**DO claim** (current honest-positive anchor):

- "LAMBDA_ROADMAP §3.2 R22-P1 wet-lab portion is DEPRECATED per AGENTS.md §5.4 with literal `⛔ DEPRECATED 2026-09-29` marker + wet-lab reason anchor + user directive verbatim" (Wf-337-W2, MEASURED 2026-09-29).
- "LAMBDA_ROADMAP §3.3 R23 (Phase I clinical) is DEPRECATED per AGENTS.md §5.4 with `⛔ DEPRECATED` marker + 6 R23-P1..R23-P6 row-level markers + historical-record-only directive" (Wf-324C, MEASURED 2026-09-29; verified by Wf-337-W2 test 2).
- "Tier-2 strategy compliance: §4.3 paper trajectory points to tier-2 journal (Digital Discovery / J. Cheminform., IF 5-8)" (Wf-337-W2 test 4, MEASURED 2026-09-29).

**DO NOT claim**:

- "All Wf-337 closeouts ship" (Wf-337-W1 G6 recovery sweep + Wf-337-G3 G6 regression are still in flight; only Wf-337-W2 is in this entry).
- "Wet-lab planning is fully deprecated across the project" (Wf-337-W2 only covers LAMBDA_ROADMAP §3.2 + §3.3; the 4 LOI drafts + 3 wetlab SOPs + TODO-97 wet-lab IC50 dry-run + LAMBDA_ROADMAP §5.x R23-R25 resource tables are covered by AGENTS.md §5.4 + Wf-324C, not Wf-337-W2).
- "§3.2 R22-P2..R22-P6 are untouched" (Wf-337-W2 only added the §3.2 R22-P1 DEPRECATED marker; §3.2 R22-P2..R22-P6 rows are NOT deprecated per the §3.2 header callout that lists R22-P2..R22-P6 as ACTIVE).

---

## §7 Cross-references

- **AGENTS.md §5.4** — Tier-2 re-orientation + NO wet-lab planning standing directive (parent contract).
- **TODO/03_plans/LAMBDA_ROADMAP.md §3.1** (line 154) — pre-Wf-337-W2 scope-update marker "R22-P1 m18 re-scoped to computational-only (no wet-lab); R23 Phase I + R24 Phase II + R25 Phase III = DEPRECATED (no clinical-trial planning)".
- **TODO/03_plans/LAMBDA_ROADMAP.md §3.2** (line 163) — R22 detail header + new Wf-337-W2 callout (line 165).
- **TODO/03_plans/LAMBDA_ROADMAP.md §3.2 R22-P1 row** (line 195) — new Wf-337-W2 `⛔ DEPRECATED 2026-09-29` marker.
- **TODO/03_plans/LAMBDA_ROADMAP.md §3.3** (line 209) — R23 detail header + body (unchanged from Wf-324C; verified by Wf-337-W2 test 2).
- **TODO/03_plans/LAMBDA_ROADMAP.md §4.3** (line 340) — paper trajectory table (verified by Wf-337-W2 test 4 for tier-2 journal anchor).
- **TODO/03_plans/LAMBDA_ROADMAP.md §10.5** (line 711) — §10.5 cross-reference list including L1-L12 + Wf-36/37/39/40/41/44 ship surface (NOT touched by Wf-337-W2; cited for routing clarity).
- **TODO/completed/207_unified_pre_flight_checklist_v2.md §9.4 + §9.5** — chemprop v3 honest-negative anchors (NOT touched by Wf-337-W2; cited for cross-Wf audit).
- **molmetal/tests/test_wf337w2_lambda_deprecated.py** — 4 NEW pytest contracts (this Wf's deliverable).

---

## §8 Future follow-ups (NOT in Wf-337-W2 scope)

For future Wf waves that touch LAMBDA_ROADMAP §3.x:

1. **Wf-41X / Wf-42X / Wf-50X waves** that propose to re-introduce wet-lab or clinical-trial milestones into R22-P1 or R23 MUST either:
   - (a) Reverse the AGENTS.md §5.4 directive with explicit user authorization (not currently authorized), OR
   - (b) Reframe the milestone as computational-only (acceptable per AGENTS.md §5.4).
   - The 4 NEW Wf-337-W2 tests will FAIL any attempt to silently re-introduce wet-lab/clinical milestones.

2. **`TODO/03_plans/LAMBDA_ROADMAP.md §4.1` Tier-1 STRONG metric table** (line 280) still uses tier-1 aspirational targets (m18/m19/m20 ≥ 0.85). Wf-337-W2 does NOT touch §4.1; if the Tier-2 strategy truly downgrades the metrics (not just the paper target), §4.1 may need a separate re-scope pass. NOT in Wf-337-W2 scope.

3. **`TODO/03_plans/LAMBDA_ROADMAP.md §5.3 R23-R25 资源 — clinical trial + post-trial`** (line 371) still references clinical-trial facility + DSMB + FDA approval. Per AGENTS.md §5.4 this is DEPRECATED scope; Wf-337-W2 does NOT touch §5.3 (it is in `AGENTS.md §5.4 wet-lab items DEPRECATED list` already).

---

**Date**: 2026-09-29 (Wf-337-W2 LAMBDA_ROADMAP DEPRECATED markers + 4 NEW tests)
