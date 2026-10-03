# Wf-322A — 4 LOI pending files: Tier-2 final marker verification

**Date:** 2026-09-29
**Workflow:** Wf-322A (Wf-317F follow-up)
**Status:** ✅ COMPLETE — all 4 LOI files verified Tier-2 final

---

## §1 Background

Per Wf-317F partial completion + AGENTS.md §5.4 user directive
**"我们发二区就可以了，不用规划湿实验"** (translation: "submitting to a
Tier-2 journal is sufficient — no need to plan wet-lab experiments"):

- Tier-2 = pure computational publication scope
- Wet-lab validation (R20-P6 §5) is OUT OF SCOPE for current ship
- LOIs are **deprecated, not deleted** — kept as historical reference
- TODO-84 still applies for `paper/*`; outreach LOIs themselves are NOT
  gated by TODO-84 (per `TODO/pending/84_no_paper_task_without_authorization.md`
  cross-ref "outreach is not gated")

---

## §2 Verification matrix (4 LOI files)

| # | File | Tier-2 marker present? | Line |
|---|---|---|---|
| 1 | `TODO/pending/wetlab_loi_rusanov_2025_platinai.md` | ✅ YES | lines 13-17 |
| 2 | `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` | ✅ YES | lines 12-16 |
| 3 | `TODO/pending/wetlab_loi_xiehe_nci60.md` | ✅ YES | lines 12-16 |
| 4 | `TODO/pending/wetlab_loi_ornl_protein_metal.md` | ✅ YES | lines 12-16 |

**Marker template (consistent across all 4 files):**

```
> **TEMPLATE_NOT_NEEDED_FOR_TIER2 (2026-09-29)**
>
> Per AGENTS.md §5.4 user directive "我们发二区就可以了，不用规划湿实验",
> Tier-2 journal is sufficient and wet-lab validation is OUT OF SCOPE.
> This LOI draft is preserved as historical reference but NOT for sending.
```

---

## §3 Pre-flight verification (TODO-207 v2 25/25 gate)

**Command run:**
```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/pre_flight_check.py
```

**Result:** exit code 0 (PASS).

**Tail of last report (background ID `b6ofzlq5h`):**

```
"name": "p3_ablation_runner_aggregate_csv_present",
"priority": "P3",
"pipeline": "wf319a",
"status": "PASS",
"evidence": "aggregate.csv OK: 12 rows (≥ 11); 14 columns;
             DESIGN_ONLY rows free of fabricated metrics"

"name": "p3_ablation_runner_registry_complete",
"priority": "P3",
"pipeline": "wf321e",
"status": "PASS",
"evidence": "ABLATION_TABLE ok: 11/11 §7 ablations mirrored
             (M3/M4/M7/M8/M10/M11/M12/M13/M14/M15/M21) + 1 BASELINE;
             all 12 entries carry the §7 contract fields"

"v2_doc_reference": "TODO/completed/207_unified_pre_flight_checklist_v2.md
                     (§3.1 chemprop, §4.2 CFM, §5 P2)"

[exited with code 0]
```

**Note:** Pre-flight gate is required before retrain / panel / ablation
launches (TODO-207). This Wf-322A task is documentation-only (no retrain),
so the pre-flight is run as a project-health sanity check, not as a launch
gate.

---

## §4 Honest framing (party-twenty-spirit §3.5)

- **4 LOI files are deprecated, NOT deleted** — kept on disk as historical
  reference of the institutional-outreach pipeline Wf-74 once drafted.
- **0 LOI sent** for all 4 files at this revision (per status headers).
- **TODO-84 paper rule** still applies to `paper/*` — the wet-lab
  validation section is **DEFERRED**, not removed.
- **AGENTS.md §5.4 user directive** is the authority for deprecating the
  wet-lab send; future agents who propose re-activating outreach must
  re-confirm with the user before resuming any send.
- **Tier-2 publication scope** = pure computational claims backed by
  Rusanov 2025 PlatinAI + Krasnov 2026 MetalCytoToxDB + tmQM public
  datasets. Pearson r ≥ 0.7 wet-lab-vs-in-silico honesty gate cannot be
  MEASURED today (synthetic-label pIC50 only); honest-negative framing
  per Wf-297B §6 still applies.

---

## §5 Cross-references

- `AGENTS.md` §5.4 — user directive "我们发二区就可以了，不用规划湿实验"
- `AGENTS.md` §5.3 — TODO-84 paper rule (still applies to `paper/*`)
- `TODO/pending/84_no_paper_task_without_authorization.md` — outreach
  is NOT gated by TODO-84
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` — pre-flight
  v2 25/25 PASS gate (TODO-207)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` — Wf-58 in-silico
  dry-run layer ship (computational scope)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` — R20-P6 wet-lab plan
  (deprecated scope; not deleted)

---

## §6 Ship wave record

| Date       | Workflow | Notes                                |
|------------|----------|-------------------------------------|
| 2026-09-29 | Wf-322A  | 4 LOI Tier-2 marker verification PASS |

---

**END OF Wf-322A** — Owner Hugo Lee; cross-ref AGENTS.md §5.4 + party-twenty-spirit §3.5; ship 2026-09-29.
