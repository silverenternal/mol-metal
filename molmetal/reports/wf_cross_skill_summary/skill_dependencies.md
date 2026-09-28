# Skill Dependency Graph — Mol-Metal 8 Skills

**Date**: 2026-09-26
**Companion to**: `cross_skill_table.md` (40 cells × 5 dims)
**Source**: `.claude/skills/*/SKILL.md` + `TODO/EXECUTION_ORDER.md` + `TODO/MECHANISM_METRIC_MAPPING.md`

---

## 1. ASCII dependency graph

```
              ┌─────────────────────────────────────┐
              │ S8 party-twenty-spirit              │  MANDATORY PRE-ACTION
              │ (5-dim checklist + ≤30-line output) │
              │ §3 / §11 R18 self-check             │
              └──────────────┬──────────────────────┘
                             │ ≤30-line mapping conclusion
                             ▼
              ┌─────────────────────────────────────┐
              │ S5 mol-metal-mechanism-mapping      │  HORIZONTAL DIAGNOSTIC
              │ (44 mechanisms × 20 metrics)        │  owns M22-M44 cross-ref
              │ §5/§6/§7 R18 cross-mapping          │
              └──────────────┬──────────────────────┘
                             │ failure-mode → ablation → §6
                             ▼
       ┌─────────────┬───────────────┬──────────────┐
       │             │               │              │
       ▼             ▼               ▼              ▼
┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐
│S1 lambda │ │S2 cfm-train  │ │S3 click  │ │S4 anticancer │
│(MLC β-NF)│ │(Lipman+EGNN+ │ │(5 click+ │ │(MetalCytoTxDB│
│KataGo+   │ │ YuelBond)    │ │ 101 MLE) │ │ +PlatinAI+   │
│ PLL+ASTSR│ │              │ │          │ │ NCI-60+kNN)  │
└────┬─────┘ └──────┬───────┘ └────┬─────┘ └──────┬───────┘
     │              │              │              │
     │  substrate   │  substrate   │  rules       │  data + scoring
     │  M1-M4,M8-   │  M5-M7,M12-  │  M10-M11     │  M9, M18-M21
     │  M11         │  M17         │              │
     └──────────────┴──────────────┴──────────────┘
                             │
                             │ code/test/data + MECHANISM tags
                             ▼
              ┌─────────────────────────────────────┐
              │ S6 paper-writing                    │  §1-§7 + appendix
              │ (MEASURED/PROJECTED/ARCHITECTURAL)  │  + traceability.csv
              └──────────────┬──────────────────────┘
                             │ §3 method extension + refs.bib 23 keys
                             ▼
              ┌─────────────────────────────────────┐
              │ S7 r18-math                         │  9 theorems + T10
              │ (T1-T10 → M22-M44)                  │  → 23 R18 ships
              └─────────────────────────────────────┘
```

---

## 2. Dependency edges (12 edges, all directional)

| # | From | To | Edge type | Trigger / anchor |
|---|---|---|---|---|
| E1 | **S8** | S5 | pre-action mandatory | EXECUTION_ORDER.md §5/§7 决策点 → S5 §3 matrix |
| E2 | **S5** | S1/S2/S3/S4 | diagnostic dispatch | metric regressed → owning skill module |
| E3 | **S1** | S2 | substrate → refiner | MLC β-NF → CFM atom cloud |
| E4 | **S2** | S3 | backbone → rules | YuelBond bond head → click SMARTS target |
| E5 | **S3** | S4 | rules → scored | click reaction → A2780/MCF7 score |
| E6 | **S4** | S5 | metric → mechanism | m1/m2/m17/m21 → M9, M18-M21 |
| E7 | **S5** | S6 | limitations → paper | §4 failure modes → paper §6 |
| E8 | **S7** | S6 | theorem → method | T1-T10 → §3.x paragraph |
| E9 | **S7** | S5 | mechanism → diagnostic | M22-M44 → §5/§6/§7 R18 |
| E10 | **S6** | S7 | cite demand → bib | paper needs → refs.bib 23 keys |
| E11 | **S1** | S7 | MLC owns paper-§3 math | M31/M32 (quantum + Hochschild) |
| E12 | **S4** | S7 | anticancer owns R18 math | M25/M26/M43 (Hilbert + ScoreRep + Bergman) |

---

## 3. Cluster view (4 clusters × 8 skills)

| Cluster | Skills | Role | Key artifact |
|---|---|---|---|
| **C1: Pre-action** | S8 | Mandatory decision-checklist before any TODO update | ≤30-line mapping conclusion + R18 §11 self-check |
| **C2: Diagnostic** | S5 | Horizontal mechanism × metric view | `TODO/MECHANISM_METRIC_MAPPING.md` §1-§7 |
| **C3: Technical execution** | S1 / S2 / S3 / S4 | Vertical module implementation per skill | MLC AST + CFM EGNN + click SMARTS + cytotoxicity D-MPNN |
| **C4: Paper + R18** | S6 / S7 | Final artifact drafting + R18 math upgrade | `paper/main.tex` + `math_skills/*` + 23 cite keys |

**Dependency rule**: C1 → C2 → C3 → C4. Never skip C1 (violates §8 hook #1). C2 mandatory before C3 edits touching multi-skill modules. C4 cannot ship without C3 stable + C2 diagnostic clean.

---

## 4. Per-skill dependency list (concrete file anchors)

### S1 — mol-metal-lambda-calculus
- **Depends on**: S8 (pre-action); S5 (§3 matrix for M1-M4, M8-M11 attribution).
- **Depended on by**: S2 (MLC substrate → CFM refiner); S3 (atoms-as-combinators); S4 (β-NF candidates → scoring); S7 (M31/M32 paper-§3 math).
- **Self-tests**: 89 + 20 + 11 + 69 + 17 = 206+ tests.

### S2 — mol-metal-cfm-training
- **Depends on**: S8; S5 (M5-M7, M12-M17 attribution); S1 (substrate).
- **Depended on by**: S3 (bond head → SMARTS); S4 (CFM-decoded → scored); S7 (W2.1/W2.2/W2.3 adapters wrap here).
- **Self-tests**: 18 (pitfall guards) + 30/30 decode probes + 14/14 evaluators.

### S3 — mol-metal-click-chemistry
- **Depends on**: S8; S5 (M10-M11); S1 (substrate); S2 (bond head target).
- **Depended on by**: S4 (rules → A2780/MCF7); S7 (M38/M36/M42).
- **Self-tests**: 101-rule contract + 25 + 25 + 33 = 184+ tests.

### S4 — mol-metal-anticancer
- **Depends on**: S8; S5 (M9, M18-M21); S1 (β-NF candidates); S2 (CFM-decoded); S3 (rules).
- **Depended on by**: S7 (M25/M26/M43 anchors).
- **Self-tests**: 17 tmQM + 14 evaluators + 3-seed m17/m21 + 50/50 credibility = 81+ tests.

### S5 — mol-metal-mechanism-mapping
- **Depends on**: S8 (mandatory pre-action); all of S1-S4 inputs (owning modules).
- **Depended on by**: S6 (§6 limitations); S7 (R18 cross-mapping); S1-S4 (diagnostic return path).
- **Self-tests**: 14 evaluators + 7 pitfall guards + 15 credibility = 36 tests.

### S6 — mol-metal-paper-writing
- **Depends on**: S8 (pre-action before any section ship); S5 (§6 limitations); S7 (refs.bib 23 keys).
- **Depended on by**: S7 (cite key consumer loop); external submission.
- **Self-tests**: n/a (writing helper; verifiable via `pdflatex 4-pass 0 unresolved` + `bibtex unresolved=0`).

### S7 — mol-metal-r18-math
- **Depends on**: S8 (pre-action); S5 (§5/§6/§7 R18 cross-mapping); S1 (paper-§3 math ownership); S4 (R18 scoring math).
- **Depended on by**: S6 (cite key consumer); S2 (adapters wrap here without modifying R17 close-out).
- **Self-tests**: 25/25 + 25/25 + 25/25 + 32/32 + 31/31 + 49/49 = 250+ tests (CPU-only Blackwell).

### S8 — party-twenty-spirit
- **Depends on**: nothing (root skill).
- **Depended on by**: ALL of S1-S7 (mandatory pre-action §8 hooks: keyword + decision-point + BLOCK close-out).
- **Self-tests**: n/a; tracked via §10 verification log (4 entries 2026-09-25).

---

## 5. Call ordering (强制 sequence)

```
[ANY stage-end signal] → [S8 first] → [S5 second] → [S1-S4 in any order, only if needed]
                       → [S6 + S7 last] → [paper §-section or TODO update]

Exception: S6 paper writing can iterate with S7 directly (cite key ↔ mechanism chain).
Exception: S5 can be skipped for trivial single-module edits (no cross-skill impact).
```

**Anti-pattern**: skipping S8 = 违规 per §8 hooks; calling S6 before S5 = §6 limitations orphan; calling S7 before S5 = mechanism without diagnostic.

---

## 6. Coupling strength (informational)

| Edge pair | Coupling | Why |
|---|---|---|
| S1 ↔ S2 | **STRONG** | MLC β-NF substrate + CFM refiner share atom-type + bond-type ontology; every S1 PR risks S2 regression |
| S2 ↔ S3 | STRONG | bond head learns from click rule SMARTS targets; every S3 rule change risks m13/m14 |
| S3 ↔ S4 | MEDIUM | click rules → cytotoxicity correlation; metric-level coupling, not module-level |
| S4 → S5 | STRONG | S4 owns m1/m2/m17/m18/m21 — every metric regression lands here |
| S5 → S6 | MEDIUM | §6 limitations sourced from §4 failure modes; one-way |
| S7 ↔ S2 | MEDIUM (R18) | W2.1/W2.2/W2.3 adapters wrap S2 without modifying it; isolated by design |
| S8 → all | MANDATORY | §8 hooks 1/2/3 enforce; ignore = 违规 |
| S1 ↔ S7 | WEAK | M31/M32 paper-only math; no metric impact |
| S4 ↔ S7 | MEDIUM (R18) | M25/M26/M43 affect m1/m4/m7; no module coupling |
| S6 ↔ S7 | STRONG (writing) | cite key ↔ mechanism table is the loop driver for §3/§4/§6 |

---

## 7. Failure modes if dependency violated

| Violation | Symptom | Recovery |
|---|---|---|
| Skip S8 | TODO update without 5-dim checklist; "远离 headline" 凑数 | Re-run S8 §3 + write §10 verification log entry |
| Skip S5 diagnostic | Mechanism change without metric prediction → silent regression | Run `pytest molmetal/tests/test_pitfall_guards.py` + §4 failure-mode probe |
| S1 PR without S2 review | Atom-vocab mismatch → 100% batch skip | Verify `bad_atoms` skip-optimizer + vocab_mask |
| S3 rule added without `pt_click_compat` | Matrix diverges → silent 0 valid products | Update `pt_click_compat.py` + run `can_apply` contract |
| S4 metric ship without S5 §3 mark | Orphan mark → unfalsifiable | Update `MECHANISM_METRIC_MAPPING.md` §2/§3/§4/§5 + cross-ref S6 |
| S6 §4 LEAD claim without tag | Reviewer flags MEASURED vs PROJECTED | Re-derive every claim; regen `paper/traceability.csv` |
| S7 mechanism without adapter | Math ships, no ML surface | Either ship adapter or down-tag to "math-only paper §3" |
| S8 §11 self-check missing 1/5 | R18 ship incomplete | Add the missing answer or refuse to ship |

---

**End of skill dependency graph.**