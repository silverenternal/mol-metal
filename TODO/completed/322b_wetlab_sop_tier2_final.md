# Wf-322B — Wet-Lab SOP + IC50 Pipeline Tier-2 Final Markers

**Date**: 2026-09-29
**Workflow**: Wf-322B (Wf-317G partial-completion follow-up)
**Status**: SHIP — all 4 files verified with Tier-2 marker
**Owner**: Hugo Lee + ultracode agent (Wf-322B sub-agent)
**Parent**: Wf-317G partial close-out; AGENTS.md §5.4 user directive
   "我们发二区就可以了，不用规划湿实验"

---

## §1 Files verified (4/4)

| # | File | Tier-2 marker present? | Marker line(s) |
|---|---|:---:|---|
| 1 | `TODO/pending/wetlab_sop_cytotoxicity_assay.md` | YES | L9-12 |
| 2 | `TODO/pending/wetlab_sop_cellular_uptake_assay.md` | YES | L9-12 |
| 3 | `TODO/pending/wetlab_sop_binding_affinity_assay.md` | YES | L9-12 |
| 4 | `TODO/pending/wetlab_ic50_analysis_pipeline.md` | YES | L12-15 |

**Marker text (verbatim, identical across all 4)**:

> **Status (2026-09-29): TEMPLATE_NOT_NEEDED_FOR_TIER2** — per AGENTS.md §5.4 user
> directive "我们发二区就可以了，不用规划湿实验", Tier-2 journal is sufficient and
> wet-lab validation is OUT OF SCOPE. This template is preserved as historical
> reference but NOT for execution.

No edits required — all 4 files already carry the marker from the prior Wf-317G
partial pass. Wf-322B is a verification gate, not a write gate.

## §2 Pre-flight verification

**Script**: `molmetal/scripts/pre_flight_check.py` (TODO-207 v2 unified)

**Invocation**:

```
env PYTHONPATH=. python3 molmetal/scripts/pre_flight_check.py --pipeline frontier
```

**Result**: **23/23 PASS** in 23.2 sec — 0 FAIL, 0 SKIP. Front-end subset covers the
P3 frontier-math + AnchorGuard + 2026-09-29 follow-up gates per Wf-306C. The full
all-pipeline invocation (`pre_flight_check.py` without `--pipeline`) also exits 0
on this run, confirming the Tier-2 marker addition does not regress any gate.

**Exit code**: 0 (PASS).

**MEASURED evidence** (party-twenty-spirit §3.5):

```
{
  "timestamp": "2026-09-29T02:52:37+0000",
  "schema_version": "207.v1",
  "pipeline": "frontier",
  "summary": {
    "pass": 23, "fail": 0, "skip": 0, "total": 23,
    "total_wall_sec": 23.2116
  }
}
```

## §3 Cross-file consistency check

| Aspect | All 4 files consistent? |
|---|:---:|
| Marker location (top-of-doc, before §1) | YES |
| Date in marker (2026-09-29) | YES |
| Directive quoted (用户原话) | YES |
| `OUT OF SCOPE` declaration | YES (3 SOPs + 1 pipeline) |
| `historical reference` framing | YES |
| AGENTS.md §5.4 cross-ref | YES |

The 4 files together cover the complete wet-lab cooperation surface (cytotoxicity +
cellular uptake + binding affinity + IC50 analysis pipeline) for the 3 institutional
PIs (ORNL / PKU / 协和). All are now explicitly marked Tier-2-out-of-scope per
AGENTS.md §5.4 user directive, eliminating any residual ambiguity for the Tier-2
journal submission gate.

## §4 Honest framing (party-twenty-spirit §3.5)

- **No fabricated Tier-2 readiness.** The 4 files are TEMPLATES, not executed
  protocols — no wet-lab hands validation, no institutional QC, no K_d / pIC50 /
  ng/cell measured. The `TEMPLATE_NOT_NEEDED_FOR_TIER2` marker documents that
  wet-lab validation is **OUT OF SCOPE** for the Tier-2 journal submission path.
- **No paper §5 claim.** The wet-lab validation section is deferred per TODO-84
  paper rule + AGENTS.md §5.3 no-paper-work-without-authorization. This task does
  NOT touch `paper/*`.
- **Pipeline doc preserved as historical reference.** `wetlab_ic50_analysis_pipeline.md`
  ships the CPU-only analysis scaffold today; the P6.2 wet-lab IC50 gate is
  PROJECTED on Wf-58 timeline and remains blocked on PI outreach (NOT actively
  pursued per AGENTS.md §5.4).

## §5 Relation to Wf-322 wave

Wf-322 is a Tier-2 final-marker wave. The wave slots:

- `TODO/completed/322a_4_loi_tier2_final.md` — 4 LOI drafts (shipped)
- `TODO/completed/322b_wetlab_sop_tier2_final.md` — THIS doc (4 wet-lab SOPs)
- `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` — M19 LOMO close-out
- `TODO/completed/322g_todo130_g3_closure_30metal_closeout.md` — G3 closure 30-metal

Wf-322B (this doc) closes the wet-lab cooperation side; no further wet-lab SOP work
is queued for Tier-2 ship surface.

## §6 Next actions (deferred, NOT active)

1. **NO PI outreach.** ORNL / PKU / 协和 outreach is paused per AGENTS.md §5.4
   user directive. The 4 LOI drafts (`TODO/pending/wetlab_loi_*.md`) carry the
   `TEMPLATE_NOT_NEEDED_FOR_TIER2` marker from Wf-322A and remain on disk as
   historical reference.
2. **NO Wf-58 P6.2 wet-lab IC50 retrain.** The 6-month ORNL/PKU/协和 timeline is
   PROJECTED only; no GPU/CPU work is gated on it.
3. **NO paper §5 work.** TODO-84 active; `mol-metal-paper-writing` skill is
   READ-only on paper sections.

---

**Date**: 2026-09-29 · **Verdict**: SHIP — 4/4 files verified, pre-flight 23/23 PASS
(frontier pipeline) + exit code 0 (full pipeline). Tier-2 markers in place. No
wet-lab execution, no paper §5 touches. · **Co-Authored-By**: Claude Code
<noreply@anthropic.com>
