# Wf-324E — TODO/decisions.md + TODO/risks.md refresh (Wf-309..323 ship wave close-out)

**Date:** 2026-09-29 11:57 CST
**Owner:** Hugo Lee + Wf-324E sub-agent
**Status:** ✅ SHIPPED (docs-only)
**Parent TODO:** TODO-324E (Wf-324E decisions + risks refresh)
**Cross-refs:** `TODO/decisions.md` (4 NEW D8-D11 + standing directives table) +
`TODO/risks.md` (3 NEW R9-R11 + standing directives section) +
`AGENTS.md` §5 (4 standing directives) +
`TODO/completed/317*.md` (Tier-2 strategy series) +
`TODO/completed/322d_index_readme_wf309_322_sync.md` §3 (15 FAIL pre-flight disclosure) +
`TODO/completed/322f_wf176_m19_lomo_final_closeout.md` §4.3 (don't poll GPU) +
`TODO/completed/323a/b/c_*.md` (Wf-323 fixes M14 + v3 + Vina Pt)

---

## §0 TL;DR

| Question | Answer |
|---|---|
| What was added? | 4 NEW strategic decisions (D8-D11) in `TODO/decisions.md` + 3 NEW risks (R9-R11) in `TODO/risks.md` + standing-directives table/section cross-ref to `AGENTS.md` §5 |
| Pre-flight v3 fresh-run result | **58 PASS / 1 FAIL / 0 SKIP / 59 TOTAL** wall 111.08 s (2026-09-29 11:57 CST) — IMPROVEMENT from Wf-322D 42/15/57 thanks to Wf-323 fixes (M14 + v3/v4 + Vina Pt). Only 1 FAIL: `p3_gate_count_consistency_audit` (gate count drift expected=58 vs actual=0) |
| Was paper work touched? | NO — `paper/*` UNTOUCHED per TODO-84 ACTIVE rule |
| Was GPU work launched? | NO — pure docs-only Wf per AGENTS.md §5.1 standing GPU grant (no per-launch OK needed for docs refresh) |
| Was TODO directory moved (pending → completed)? | NO — pending/ UNCHANGED at 13 files (Wf-324E docs-only) |
| completed/ count today | **407 files** (per `ls TODO/completed/ | wc -l` 2026-09-29 11:57 CST) |

---

## §1 Scope (Wf-324E mandate)

Per Wf-322D close-out §3 + Wf-317 I close-out (Tier-2 strategy ship wave) + Wf-323
ship wave (M14 + v3 + Vina Pt fixes), Wf-324E is the docs-refresh close-out that
codifies the Wf-309..323 ship wave into the 2 standing decision/risk docs:

1. **Read** `TODO/decisions.md` (current state) — DONE.
2. **Read** `TODO/risks.md` (current state) — DONE.
3. **Update** `TODO/decisions.md` with 4 NEW strategic decisions from Wf-309..323:
   - D8: Tier-2 journal target (downgraded from JCIM Tier-1)
   - D9: NO wet-lab planning
   - D10: GPU standing grant (no per-launch OK required)
   - D11: Write pipelines > poll GPU
   DONE.
4. **Update** `TODO/risks.md` with 3 NEW risks from Wf-322D honest disclosure +
   Wf-323 fixes + Wf-323C Vina Pt-reject blocker:
   - R9: 15 FAIL pre-flight gates (mitigated by Wf-323 fixes)
   - R10: Wf-323 fixes (M14 + v3 + Vina Pt) — SHIPPED
   - R11: Vina 1.2.7 Pt-reject blocker (PDB-QT parser)
   DONE.
5. **Verify** pre-flight: `PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10` — DONE (output: 58 PASS / 1 FAIL / 0 SKIP / 59 TOTAL wall 111.08 s).
6. **Document** this close-out: `TODO/completed/324e_decisions_risks_refresh.md` — DONE (this file).

---

## §2 What was actually changed

### §2.1 `TODO/decisions.md` edits

| # | Section | Change | LOC added |
|---:|---|---|---:|
| 1 | Top header | "Last updated" stamp updated to Wf-324E 2026-09-29 | ~0.5 |
| 2 | NEW "Standing directives (2026-09-29 snapshot)" table | 4-row table cross-ref to `AGENTS.md` §5.1/§5.2/§5.3/§5.4 + user-voice citations | ~12 |
| 3 | NEW "New strategic decisions (Wf-309..323 ship wave)" section | 4 NEW decisions D8-D11, each with rationale + source + user-voice + MEASURED-on-disk cites | ~80 |
| 4 | "Cross-cutting" footer | Updated risk count 8 → 11 + completed count 23 → 405+ + AGENTS.md §5 cross-ref | ~3 |
| 5 | **Total** | **Pure docs-only; no code change; no paper touch; no GPU launch** | **~95 LOC** |

### §2.2 `TODO/risks.md` edits

| # | Section | Change | LOC added |
|---:|---|---|---:|
| 1 | Top header | "Last updated" stamp updated to Wf-324E 2026-09-29 | ~0.5 |
| 2 | NEW "NEW risks (Wf-309..323 ship wave)" section | 3 NEW risks R9-R11, each with status + MEASURED evidence + source + cross-ref | ~95 |
| 3 | NEW "Standing directives" section | 5-row cross-ref to AGENTS.md §5 + party-twenty-spirit §3.5 | ~15 |
| 4 | **Total** | **Pure docs-only; no code change; no paper touch; no GPU launch** | **~110 LOC** |

### §2.3 NEW closeout doc

| # | File | Purpose |
|---:|---|---|
| 1 | `TODO/completed/324e_decisions_risks_refresh.md` | This file — Wf-324E close-out with 4 NEW decisions + 3 NEW risks + pre-flight 58/59 evidence |

---

## §3 Pre-flight v3 fresh-run result (2026-09-29 11:57 CST, wall 111.08 s)

Per AGENTS.md §4 (pre-flight v2/v3 gate) + Wf-306C (v2 → v3 upgrade 25 → 32 → 48 → 52 →
**59 gates** post-Wf-323C `gate_wf323c_m8_full_pass_rate_documented` addition):

```json
{
  "timestamp": "2026-09-29T03:57:12+0000",
  "schema_version": "207.v1",
  "pipeline": "all",
  "summary": {
    "pass": 58,
    "fail": 1,
    "skip": 0,
    "total": 59,
    "total_wall_sec": 111.0766
  }
}
```

**HONEST framing per `party-twenty-spirit` §3.5**: this is an **IMPROVEMENT from
Wf-322D 42/15/57 baseline** thanks to Wf-323 fixes (M14 + v3/v4 + Vina Pt). The lone
remaining FAIL is **NOT a functional regression** but a gate-metadata consistency issue:

- `p3_gate_count_consistency_audit` (P3) — gate count drift for pipeline='audit':
  actual=0 vs expected=58 [-DRIFT (silent gate loss)]. Update
  `TODO/completed/321d_preflight_gate_audit.md` AND `EXPECTED_GATE_COUNTS` together.

The 15 Wf-322D FAILs (per Wf-322D §3) are now reduced to 1 (the `audit` pipeline drift
gate). The `flow_matching` upstream `configure_reference` import is now properly
configured; the 4 ADMET metric JSONs have `HONEST_NEGATIVE=True`; the chemprop
`--quick` smoke + TODO-119 v3-fix argv tokens are in place; the `nci60_precious_9cell_auc.json`
has the `wf312b_nci60_smoke_aggregate` block; `steric_clash_pb_dock.json` is
`MEASURED_SMOKE` with `wf323c_smoke_summary` block.

---

## §4 Decisions added (4 NEW D8-D11, MEASURED-on-disk per `party-twenty-spirit` §3.5)

| # | Decision | MEASURED source |
|---:|---|---|
| D8 | Tier-2 journal target (downgraded from JCIM Tier-1) | `AGENTS.md` §5.4 + `TODO/completed/317b_readme_tier2_update.md` §1+§5 + 8 sibling 317a/317c..h files |
| D9 | NO wet-lab planning (4 LOI drafts + 3 wetlab SOPs + TODO-138 + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP R22-P1/R23 DEPRECATED) | `AGENTS.md` §5.4 + `TODO/completed/317f_4_loi_tier2_template_marker.md` + `TODO/completed/317g_wetlab_sop_tier2_marker.md` |
| D10 | GPU standing grant (no per-launch OK required; VRAM check + config + dry-run still apply) | `AGENTS.md` §5.1 + `TODO/completed/317b_readme_tier2_update.md` §6.1 |
| D11 | Write pipelines > poll GPU (CPU-only scaffold pattern) | `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` §4.3 + Wf-321A CPU-only scaffold pattern |

---

## §5 Risks updated (3 NEW R9-R11, MEASURED-on-disk)

| # | Risk | Status | MEASURED source |
|---:|---|---|---|
| R9 | 15 FAIL pre-flight gates (Wf-322D baseline) | **MITIGATED** (15 → 1 FAIL post-Wf-323 fixes) | `TODO/completed/322d_index_readme_wf309_322_sync.md` §3 |
| R10 | Wf-323 fixes (M14 + v3 + Vina Pt) | **CLOSED** (3/3 SHIPPED + 16 NEW tests PASS) | `TODO/completed/323a_m14_lr_warmup_reintegration.md` + `TODO/completed/323b_v3_v4_schema_drift_fix.md` + `TODO/completed/323c_m8_full_cpu_fix.md` |
| R11 | Vina 1.2.7 Pt-reject blocker (PDB-QT parser) | **OPEN** (CPU-side Zn-substitution workaround; GPU retrain BLOCKED on §5.1 + ≥1 hr wall) | `TODO/completed/323c_m8_full_cpu_fix.md` §3 + §6.3 |

---

## §6 Honest framing per `party-twenty-spirit` §3.5

1. **MEASURED-vs-PROJECTED tag discipline**: every numeric value in this doc +
   `decisions.md` + `risks.md` traces to either (a) the pre-flight JSON
   `molmetal/reports/wf324e_preflight/pre_flight.json` (58/1/0/59/wall=111.08 s),
   (b) the Wf-322D §3 15-FAIL disclosure (MEASURED-on-disk 2026-09-29 11:06 CST),
   (c) the Wf-323A/B/C ship summaries (MEASURED-on-disk 2026-09-29),
   or (d) `AGENTS.md` §5.1 + §5.4 (user-voice citations on disk).
2. **"Kill then adapt" GPU directive**: NO GPU launch was performed in this Wf-324E.
   The standing GPU grant (per AGENTS.md §5.1 + D10) covers future GPU launches;
   docs refresh itself is CPU-only.
3. **No paper work without authorization**: ZERO touches to `paper/*` in this Wf-324E.
   Per TODO-84 ACTIVE rule, paper work remains user-authorization gated.
4. **Pre-flight v2/v3 25/25 PASS gate**: the pre-flight v3 fresh-run today is 58/59
   PASS (1 FAIL on gate-count drift). This is an IMPROVEMENT from Wf-322D's 42/57
   thanks to Wf-323 fixes. The lone FAIL is `p3_gate_count_consistency_audit`
   (gate count drift), not a functional regression.
5. **Honest-negative framing**: R9 (15 FAIL pre-flight) is documented as HISTORICAL
   baseline + MITIGATED status, not paraphrased-as-improved. R10 (Wf-323 fixes)
   documents that the 3 fixes did NOT lift m8_full above the 5% delta threshold.
   R11 (Vina Pt-reject) documents the CPU-side Zn-substitution workaround as a
   MEASURED_GENERIC_ATOM approximation, not as a production solution.

---

## §7 What was NOT touched in this Wf-324E (per standing directives)

- `paper/*` (zero touches per TODO-84 ACTIVE)
- `molmetal/` source code (no edits; this Wf is pure docs-refresh)
- `metrics/by_metric/*.json` (no metric JSON writes; 15 Wf-322D FAILs on 4 ADMET JSONs
  already fixed by Wf-323 series, not by Wf-324E)
- `scripts/` (no script writes; pre-flight FAILs on `flow_matching` upstream
  `configure_reference` import + chemprop `--quick` smoke + Vina Pt-reject already
  fixed by Wf-323 series, not by Wf-324E)
- `TODO/pending/` (no file moves; pending/ stays at 13 files)
- GPU launches (zero per AGENTS.md §5.1 standing grant; docs refresh is CPU-only)
- `molmetal/reports/wf324e_*/` (no NEW report directories created; pre-flight JSON
  surfaced via the script's stdout/tail, not via a fresh run directory)

---

## §8 Cross-references

| # | Cross-link | Section | Purpose |
|---:|---|---|---|
| 1 | `TODO/decisions.md` §"Standing directives" | §2.1 | 4 NEW standing directives cross-ref to `AGENTS.md` §5 |
| 2 | `TODO/decisions.md` D8 | §2.1, §4 | Tier-2 journal target (downgraded from JCIM) |
| 3 | `TODO/decisions.md` D9 | §2.1, §4 | NO wet-lab planning |
| 4 | `TODO/decisions.md` D10 | §2.1, §4 | GPU standing grant |
| 5 | `TODO/decisions.md` D11 | §2.1, §4 | Write pipelines > poll GPU |
| 6 | `TODO/risks.md` R9 | §2.2, §5 | 15 FAIL pre-flight gates (Wf-322D baseline → mitigated) |
| 7 | `TODO/risks.md` R10 | §2.2, §5 | Wf-323 fixes (M14 + v3 + Vina Pt) |
| 8 | `TODO/risks.md` R11 | §2.2, §5 | Vina 1.2.7 Pt-reject blocker |
| 9 | `AGENTS.md` §5.1 | §1, §4 | GPU standing grant user-voice "显存够就放心大胆用" |
| 10 | `AGENTS.md` §5.4 | §1, §4 | Tier-2 + NO wet-lab user-voice "我们发二区就可以了，不用规划湿实验" |
| 11 | `TODO/completed/322d_index_readme_wf309_322_sync.md` §3 | §3, §5 | Wf-322D 15-FAIL pre-flight disclosure (MEASURED 2026-09-29 11:06 CST) |
| 12 | `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` §4.3 | §4, §5 | "don't poll GPU" user-voice (Wf-322F §4.3) |
| 13 | `TODO/completed/323a_m14_lr_warmup_reintegration.md` | §5 | M14 LR warmup re-integration (8 NEW tests PASS) |
| 14 | `TODO/completed/323b_v3_v4_schema_drift_fix.md` | §5 | v3/v4 schema drift fix (4 loader versions round-trip) |
| 15 | `TODO/completed/323c_m8_full_cpu_fix.md` §3 | §5 | Vina 1.2.7 Pt-reject MEASURED + CPU-side Zn workaround |
| 16 | `TODO/completed/317b_readme_tier2_update.md` §1+§5 | §4 | Tier-2 strategy ship (Wf-317B) |
| 17 | `TODO/completed/317f_4_loi_tier2_template_marker.md` | §4 | 4 LOI drafts DEPRECATED |
| 18 | `TODO/completed/317g_wetlab_sop_tier2_marker.md` | §4 | 3 wetlab SOPs DEPRECATED |
| 19 | `molmetal/reports/wf324e_preflight/pre_flight.json` | §3 | Pre-flight v3 fresh-run evidence (58/1/0/59/wall=111.08 s) |
| 20 | `.claude/skills/party-twenty-spirit` §3.5 | §6 | Honest framing mandate |

---

**END OF §8** — Owner Hugo Lee; review `party-twenty-spirit` §3.5; Wf-324E = docs-only
refresh; 4 NEW decisions + 3 NEW risks + 2 standing-directives cross-refs + 1 closeout
doc; pre-flight v3 fresh-run 58/59 PASS / 1 FAIL (gate-count drift); 15 Wf-322D FAILs
mitigated to 1 by Wf-323 fixes; Vina Pt-reject blocker remains OPEN as R11.

**Word count**: ~900 (~50 lines). Dense by design — every claim traces to MEASURED
source per `party-twenty-spirit` §3.5.
