# Wf-317G — Wet-lab SOP + IC50 pipeline Tier-2 marker (historical reference only)

**Date**: 2026-09-29
**Workflow**: Wf-317G (per user directive relayed through harness)
**Status**: ✅ SHIPPED — Tier-2 marker banner added to 4 wet-lab pending files
**Owner**: Hugo Lee (per user directive)
**Driver**: User-originated request: "我们发二区就可以了，不用规划湿实验"
**Parent directive**: `AGENTS.md` §5.4 (2026-09-29 snapshot)

---

## §0 Why this doc exists

Per `AGENTS.md` §5.4 (2026-09-29 user directive), the project re-orients from
Tier-1 (JCIM) to Tier-2 (JCAMD / J. Mol. Graph. Model. / Comput. Chem. WIREs /
Mol. Divers. / Comput. Biol. Chem.) journal target. Tier-2 acceptance bar is
**computational-only**; wet-lab validation is a 6–12 month external-dependency
that does NOT gate Tier-2 acceptance.

As a consequence, the 3 institutional-tailored wet-lab SOPs (cytotoxicity /
cellular uptake / binding affinity) and the IC50 analysis pipeline template
(`wetlab_ic50_analysis_pipeline.md`) are **DEPRECATED for project scope**.
They remain on disk as **historical reference** — not for execution.

This Wf-317G pass adds a clear **Tier-2 marker banner** to each file so that
future agents (and the human reviewer) immediately see that execution is
deferred, and the templates are preserved solely as paper §6 honest-limitations
or §7 future-work material.

---

## §1 Files updated

| # | Path | Banner added |
|---|---|---|
| 1 | `/home/hugo/codes/mol-metal/TODO/pending/wetlab_sop_cytotoxicity_assay.md` | ✅ Tier-2 marker inserted after line 7 (Owner line) |
| 2 | `/home/hugo/codes/mol-metal/TODO/pending/wetlab_sop_cellular_uptake_assay.md` | ✅ Tier-2 marker inserted after line 7 (Owner line) |
| 3 | `/home/hugo/codes/mol-metal/TODO/pending/wetlab_sop_binding_affinity_assay.md` | ✅ Tier-2 marker inserted after line 7 (Owner line) |
| 4 | `/home/hugo/codes/mol-metal/TODO/pending/wetlab_ic50_analysis_pipeline.md` | ✅ Tier-2 marker inserted after line 10 (Companion template line) |

**n_files_updated = 4.**

Banner text (verbatim, identical on all 4 files):

```
> **Status (2026-09-29): TEMPLATE_NOT_NEEDED_FOR_TIER2** — per AGENTS.md §5.4 user
> directive "我们发二区就可以了，不用规划湿实验", Tier-2 journal is sufficient and
> wet-lab validation is OUT OF SCOPE. This template is preserved as historical
> reference but NOT for execution.
```

---

## §2 Cross-reference to AGENTS.md §5.4

The Tier-2 marker aligns with `AGENTS.md` §5.4 (Tier-2 journal target + NO
wet-lab planning, 2026-09-29 user directive). Specifically, §5.4 declares:

> "3 wetlab SOPs (cytotoxicity / cellular uptake / binding affinity) → keep
> templates in `TODO/pending/` as reference; do NOT execute."

> "Wf-74 v2 wet-lab outreach → DEPRECATED."

The `wetlab_ic50_analysis_pipeline.md` template (Wf-74 Phase 3 analysis layer
doc) falls under the Wf-74 v2 deprecation.

---

## §3 Honest framing (party-twenty-spirit §3.5)

- **No wet-lab data collected.** All 4 files remain **TEMPLATE** state.
- **No LOI sent.** Per `TODO/pending/84_no_paper_task_without_authorization.md`
  and `AGENTS.md` §5.4, the 4 LOI drafts (Rusanov / PKU / 协和 / ORNL) are
  preserved as reference but NOT sent.
- **No paper §5 claim.** The pipeline template ships as historical-reference
  only; paper §5 cross-validation gate is OUT OF SCOPE for Tier-2.
- **Tier-2 metric lift is the in-scope deliverable.** Per `AGENTS.md` §5.4,
  every `metrics/by_metric/*.json` and `TODO/completed/*.md` is Tier-2-shippable
  as is.

---

## §4 Pre-flight verification

**Pre-flight gate NOT triggered for this Wf.** Per `AGENTS.md` §4, pre-flight
v2 25/25 PASS is required only for **retrain / panel / ablation launch**.
Wf-317G is a doc-marker edit (no GPU launch, no model retrain, no panel
re-emission); pre-flight is independent of the 4 wet-lab files updated here
(the check covers TODO-207 25/25 gate items: GPU env probe, anchor JSON,
metrics freshness, R/Wf ledger, honest-negative catalog, etc.).

**Pre-flight attempt logged for traceability:** the harness ran
`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py`
twice; both invocations entered long-running state (>2 min wall, expected —
pre-flight probes GPU env + 113 metrics files + R/Wf ledger). The harness
task was stopped to keep Wf-317G within its CPU-only doc-edit scope. No
FAIL signal observed in the captured window.

**Status:** Pre-flight was not required for this Wf per `AGENTS.md` §4
scope rule. Wf-317G ships CPU-only with the doc marker on 4 files.

---

## §5 Revision history

| Rev | Date | Author | Notes |
|---|---|---|---|
| 1.0 | 2026-09-29 | Wf-317G | Initial Tier-2 marker banner pass on 4 wet-lab pending files (3 SOPs + IC50 pipeline). Per user directive relayed through harness + `AGENTS.md` §5.4. |