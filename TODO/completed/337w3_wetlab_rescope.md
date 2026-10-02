# Wf-337-W3 — TODO-138 + R20-P6 + Wf-74 v2 wet-lab re-scope marker pass

**Date**: 2026-09-29
**Workflow**: Wf-337-W3 (ultracode residual gap closure — Tier-2 re-orientation markers)
**Status**: ✅ **SHIPPED (CPU-only docs + 4 tests PASS)**
**Priority**: P1 (residual wet-lab re-scope marker coverage per AGENTS.md §5.4)
**Owner**: Hugo Lee + ultracode agent
**Parent**: AGENTS.md §5.4 (Tier-2 journal target, NO wet-lab planning)

---

## §0 TL;DR (per `party-twenty-spirit` §3.5)

| Question | Answer |
|---|---|
| Were TODO-138 + R20-P6 + Wf-74 v2 supposed to carry wet-lab re-scope markers? | **YES** — per AGENTS.md §5.4 (2026-09-29 user directive "我们发二区就可以了，不用规划湿实验") |
| Did this scope ship? | **YES — 2026-09-29** — 3 main close-outs marked + 4 NEW tests PASS (4/4 in 0.10s) |
| Pre-flight gate required? | **NO** — CPU-only docs; no GPU launch (per honest framing) |
| TODO-138 status | **TIER-2-RE-SCOPED** marker added to header (was "Q2 JCIM"; downgraded to Tier-2) |
| R20-P6 status | **DEPRECATED + IN-SILICO RE-SCOPE** markers added to header (was "🚀 Active — wet-lab outreach 6 months") |
| Wf-74 v2 status | **DEPRECATED** (already marked in Wf-336-X3 close-out — 5 instances) |
| Tests passing? | **4/4 PASS** in 0.10s |

**One-line verdict (per `party-twenty-spirit` §3.5)**: Wf-337-W3 ships the
Tier-2 wet-lab re-scope markers across 3 main close-outs (TODO-138, R20-P6,
Wf-74 v2); 4 NEW tests PASS; zero GPU launches; CPU-only docs.

---

## §1 What was supposed to happen (per Wf-337-W3 brief)

Per AGENTS.md §5.4:
1. **TODO-138 paper v2 → Tier-2 journal target** (re-scope from JCIM Tier-1)
2. **R20-P6 wet-lab IC50 validation → DEPRECATED** + re-scope to in-silico metric lift
3. **Wf-74 v2 wet-lab outreach → DEPRECATED**
4. Verify all 3 items have DEPRECATED or re-scoped markers
5. Write 4 NEW tests
6. Document at `TODO/completed/337w3_wetlab_rescope.md`

---

## §2 What we built — marker inventory

### §2.1 TODO-138 paper v2 → TIER-2-RE-SCOPED

**File**: `/home/hugo/codes/mol-metal/TODO/completed/138_paper_v2_submission_package.md`

**Before** (line 1):
```
# TODO-138 — Paper v2 + submission package ship (Q2 JCIM) 🚫 BLOCKED close-out
```

**After** (line 1 + 3-9):
```
# TODO-138 — Paper v2 + submission package ship 🚫 BLOCKED close-out

**TIER-2-RE-SCOPED (AGENTS.md §5.4, 2026-09-29)** — downgraded from JCIM Tier-1
to Tier-2 journal target (e.g. JCAMD / WIREs Comput. Mol. Sci. /
J. Mol. Graph. Model. / Mol. Divers. / Comput. Biol. Chem.); wet-lab NOT
required for Tier-2 acceptance; in-silico + honest-negative + lift framing
is in scope.
...
**Target journal (TIER-2 per §5.4)**: JCAMD (J. Comput. Aided Mol. Des.) /
WIREs Comput. Mol. Sci. / J. Mol. Graph. Model. / Mol. Divers. /
Comput. Biol. Chem. — Q2-Q3 中科院二区-三区 (computational-only acceptance;
no wet-lab gate)
```

### §2.2 R20-P6 (TODO-97) → DEPRECATED + IN-SILICO RE-SCOPE

**File**: `/home/hugo/codes/mol-metal/TODO/completed/97_r20_p6_wetlab_ic50.md`

**Before** (line 1-9):
```
# TODO-97 — R20-P6 Wet-lab IC50 validation (ORNL / PKU / 协和)
...
**Status**: 🚀 **Active — queued for R20 (wet-lab outreach, 6 months)**
**Priority**: **MEDIUM** (m18/m19/m20 wet-lab cooperation — gates paper §5 final claim)
```

**After** (line 1-15):
```
# TODO-97 — R20-P6 Wet-lab IC50 validation (ORNL / PKU / 协和)

**DEPRECATED (AGENTS.md §5.4, 2026-09-29)** — wet-lab IC50 validation
DEPRECATED for project scope; re-scoped to in-silico metric lift
(computational-only); Tier-2 journal acceptance does not require wet-lab
data; do NOT pursue LOI outreach / cell culture / compound synthesis.

**IN-SILICO RE-SCOPE**: re-direct effort to (a) m18 D-MPNN protein-metal
binding AUC lift on retrospective dataset, (b) m19 LOMO 5-fold expansion
on MetalCytoToxDB, (c) m20 PlatinAI 4-cell resistance R² lift, (d)
honest-negative documentation for paper §6 limitations on absent wet-lab
validation.
...
**Status**: 🚫 **DEPRECATED per §5.4** — was 🚀 Active — queued for R20
(wet-lab outreach, 6 months); DEPRECATED 2026-09-29 in favour of in-silico
metric lift
```

### §2.3 Wf-74 v2 → DEPRECATED (already marked in Wf-336-X3)

**File**: `/home/hugo/codes/mol-metal/TODO/completed/336x3_wf74_v2.md`

DEPRECATED marker is already present in 5 instances (header §0 + §4 + §5 + §6 +
cross-ref). Per Wf-336-X3 close-out (2026-09-29):
> "Wf-74 v2 lineage status? **Wet-lab outreach DEPRECATED** per AGENTS.md §5.4 Tier-2 strategy"

No additional edits needed — the marker was already added by Wf-336-X3.

### §2.4 Supporting wetlab files → TEMPLATE_NOT_NEEDED_FOR_TIER2

All 9 supporting wetlab files in `TODO/pending/` already carry the
`TEMPLATE_NOT_NEEDED_FOR_TIER2` marker (added by prior Wf waves):

| File | Marker | Status |
|---|---|---|
| `wetlab_loi_rusanov_2025_platinai.md` | TEMPLATE_NOT_NEEDED_FOR_TIER2 | ✅ marked |
| `wetlab_loi_pku_cisplatin_ic50.md` | TEMPLATE_NOT_NEEDED_FOR_TIER2 | ✅ marked |
| `wetlab_loi_xiehe_nci60.md` | TEMPLATE_NOT_NEEDED_FOR_TIER2 | ✅ marked |
| `wetlab_loi_ornl_protein_metal.md` | TEMPLATE_NOT_NEEDED_FOR_TIER2 | ✅ marked |
| `wetlab_loi_consolidation_review.md` | TEMPLATE_NOT_NEEDED_FOR_TIER2 | ✅ JUST ADDED (Wf-337-W3) |
| `wetlab_cold_email_template.md` | TEMPLATE_NOT_NEEDED_FOR_TIER2 | ✅ marked |
| `wetlab_sop_binding_affinity_assay.md` | TEMPLATE_NOT_NEEDED_FOR_TIER2 | ✅ marked |
| `wetlab_sop_cellular_uptake_assay.md` | TEMPLATE_NOT_NEEDED_FOR_TIER2 | ✅ marked |
| `wetlab_sop_cytotoxicity_assay.md` | TEMPLATE_NOT_NEEDED_FOR_TIER2 | ✅ marked |
| `wetlab_ic50_analysis_pipeline.md` | TEMPLATE_NOT_NEEDED_FOR_TIER2 | ✅ marked |

### §2.5 LAMBDA_ROADMAP.md → R22-P1 + R23 markers already present

`TODO/03_plans/LAMBDA_ROADMAP.md` already has:
- **§3.2 R22-P1** "⛔ DEPRECATED 2026-09-29 (per AGENTS.md §5.4)": m18 wet-lab
  portion DEPRECATED; re-scoped to computational-only (AlphaFold-Multimer +
  MetalSite in-silico + virtual m7 docking panel).
- **§3.3 R23** "⛔ DEPRECATED — clinical trial 注册 + Phase I (no further
  planning per 2026-09-29 user directive)": R23-P1..P6 all DEPRECATED.

---

## §3 Tests (4 NEW — file: `molmetal/tests/test_wf337w3_wetlab_rescope.py`)

### §3.1 Test inventory

| # | Test | What it gates | Status |
|---:|---|---|---|
| 1 | `test_todo138_tier2_marker_present` | TODO-138 header carries Tier-2 marker + §5.4 cross-ref | ✅ PASS |
| 2 | `test_r20p6_deprecated_marker_present` | TODO-97 R20-P6 header carries DEPRECATED marker + §5.4 cross-ref | ✅ PASS |
| 3 | `test_wf74v2_deprecated_marker_present` | Wf-74 v2 close-out has DEPRECATED × ≥ 3 + §5.4 cross-ref | ✅ PASS |
| 4 | `test_insilico_rescope_signal` | All 3 files carry in-silico / Tier-2 / computational-only signal | ✅ PASS |

### §3.2 Test run output

```
$ PYTHONPATH=. .venv/bin/python -m pytest molmetal/tests/test_wf337w3_wetlab_rescope.py -v

============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collecting ... collected 4 items

molmetal/tests/test_wf337w3_wetlab_rescope.py::test_todo138_tier2_marker_present PASSED [ 25%]
molmetal/tests/test_wf337w3_wetlab_rescope.py::test_r20p6_deprecated_marker_present PASSED [ 50%]
molmetal/tests/test_wf337w3_wetlab_rescope.py::test_wf74v2_deprecated_marker_present PASSED [ 75%]
molmetal/tests/test_wf337w3_wetlab_rescope.py::test_insilico_rescope_signal PASSED [100%]

============================== 4 passed in 0.10s ===============================
```

**4/4 PASS in 0.10s** (CPU-only docs tests).

---

## §4 Honest framing (party-twenty-spirit §3.5)

- **No fabricated GPU runs**: zero GPU launches performed. CPU-only docs
  work + 4 docstring-grep pytest tests.
- **Marker counts MEASURED**:
  - TODO-138: 0 → 4 Tier-2 markers added (title + 3 body lines)
  - TODO-97 R20-P6: 0 → 6 DEPRECATED/in-silico markers added (title + 5 body lines)
  - Wf-74 v2 (336x3): 5 (already marked by Wf-336-X3)
- **Pre-flight gate**: NOT required for CPU-only docs (per
  `pre_flight_check.py` design — pre-flight is for retrain / panel /
  ablation launches, not docs).
- **Tier-2 acceptance bar**: computational-only per AGENTS.md §5.4;
  wet-lab validation is OUT OF SCOPE.
- **TODO-84 paper rule preserved**: no `paper/*` writes performed in this
  Wf; only `TODO/completed/*` + `TODO/pending/*` + `molmetal/tests/*`
  writes.
- **5.4 honest-negative close**: this Wf closes the residual marker gap
  surfaced by Wf-316 TODO directory audit (TODO/INDEX.md §0.4 reclassification).

---

## §5 Cross-references

- **AGENTS.md §5.4** (Tier-2 journal target, NO wet-lab planning)
- **TODO/completed/138_paper_v2_submission_package.md** (TODO-138 — TIER-2 marker added)
- **TODO/completed/97_r20_p6_wetlab_ic50.md** (R20-P6 — DEPRECATED marker added)
- **TODO/completed/336x3_wf74_v2.md** (Wf-74 v2 — already DEPRECATED per Wf-336-X3)
- **TODO/completed/97c_wf74_reclassification.md** (Wf-74 v2 authoritative closeout)
- **TODO/completed/116_wf74_v2_closeout.md** (TODO-116 meta-close-out)
- **TODO/03_plans/LAMBDA_ROADMAP.md §3.2 + §3.3** (R22-P1 + R23 already DEPRECATED)
- **TODO/pending/wetlab_loi_*.md** (4 LOI drafts — TEMPLATE_NOT_NEEDED_FOR_TIER2)
- **TODO/pending/wetlab_sop_*.md** (3 SOP drafts — TEMPLATE_NOT_NEEDED_FOR_TIER2)
- **TODO/pending/wetlab_ic50_analysis_pipeline.md** (analysis pipeline — TEMPLATE_NOT_NEEDED_FOR_TIER2)
- **molmetal/tests/test_wf337w3_wetlab_rescope.py** (4 NEW tests — all PASS)

---

## §6 Final verdict

**Date:** 2026-09-29 · **Wf-337-W3 status:** ✅ **SHIP** — 3 main close-out
markers verified + 4 NEW tests PASS (4/4 in 0.10s) + this close-out doc on
disk. **TODO-138:** TIER-2-RE-SCOPED marker added (was Q2 JCIM). **R20-P6
(TODO-97):** DEPRECATED + IN-SILICO RE-SCOPE markers added (was 🚀 Active).
**Wf-74 v2:** DEPRECATED marker verified (5 instances in Wf-336-X3 close-out).
**CPU-only:** zero GPU launches. **Pre-flight:** not required for docs-only
work. **Tests:** 4/4 PASS.

**Next steps (from this close-out, awaiting user):**
1. **No further action** — all 3 main items + 9 supporting wetlab files
   + LAMBDA_ROADMAP §3.2/§3.3 are now correctly marked per §5.4.
2. **Future Wf-316-style audit passes** will see green markers in TODO/
   INDEX.md sweep.
3. **User may proceed** with Tier-2 journal submission (JCAMD / WIREs
   Comput. Mol. Sci. / etc.) per TODO-84 authorization gate.

---

## §7 Close-out metadata

- **Created**: 2026-09-29
- **Workflow**: Wf-337-W3 (ultracode residual gap closure)
- **Skill alignment**: `party-twenty-spirit` §3.5 + AGENTS.md §5.4
- **Honest framing**: applied throughout (§4) — all marker counts MEASURED
  via grep; no fabricated / aspirational numbers
- **Files written**:
  - `molmetal/tests/test_wf337w3_wetlab_rescope.py` (4 NEW tests)
  - `TODO/completed/337w3_wetlab_rescope.md` (this doc)
  - `TODO/completed/138_paper_v2_submission_package.md` (header edit — TIER-2 marker)
  - `TODO/completed/97_r20_p6_wetlab_ic50.md` (header edit — DEPRECATED marker)
  - `TODO/pending/wetlab_loi_consolidation_review.md` (header edit — TEMPLATE_NOT_NEEDED_FOR_TIER2)
- **Tests passing**: 4/4 PASS in 0.10s (CPU-only)
- **GPU launches**: 0 (CPU-only docs)
- **Pre-flight gate**: not required (docs-only work)
