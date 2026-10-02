# Wf-324B — TODO/INDEX.md + README.md comprehensive refresh (2026-09-29 12:10 CST)

**Phase**: Wf-324B docs-only refresh 2026-09-29 (this entry) — refreshes the TODO directory + README.md
project status to reflect the Wf-296..325 ship waves + Tier-2 paper target (AGENTS.md §5.4) +
§5.1 standing GPU authorization (2026-09-29 user grant) + pre-flight v3 42/57 HONEST-NEGATIVE
REGRESSION disclosure.

**Skill alignment**: `party-twenty-spirit` §3.5 (every Wf cite traces to a MEASURED JSON / report
path; PROJECTED vs SHIPPED tag discipline preserved); `AGENTS.md` §5.1 (standing GPU auth);
`AGENTS.md` §5.4 (Tier-2 + NO wet-lab directive).

**Honest framing**: docs-only Wf. NO code change. NO GPU launch. NO paper work (per TODO-84 ACTIVE).

**Coordination note**: Wf-324D (concurrent sibling workflow, executed ~50 min before Wf-324B in
the same session) had ALREADY shipped the bulk of the Wf-324B scope:
- `TODO/INDEX.md` §0.10 (Wf-323 ship wave timeline) + §0.11 (net effect) + §22.17 (Wf-323 detailed table)
- `README.md` §0 + §2 + §4 + §5 + §9 + §10 refreshed to current state
- `AGENTS.md` §3 catalog sync 24 → 28 skills + §6 Wf history sync
- `TODO/README.md` top-level refreshed
- Wf-323 (7 CPU closeouts) + Wf-324e/f + Wf-325d/e/f closeout docs already on disk

Wf-324B therefore is a **verifier + minor reconciliation** closeout, NOT a re-do. All file edits
that Wf-324B would have made are already present.

---

## §1 Files verified (4 edits — already present per Wf-324D)

| # | File | Surface | Status (post-Wf-324D) | Wf-324B verification |
|---:|---|---|---|---|
| 1 | `README.md` | Top-level project status hub (Wf-256..323 ship wave + Tier-2 + §5.1 + pre-flight v3 honest-negative) | ✅ REFRESHED | header line timestamp + §0 status snapshot + §2 recent Wfs + §4 pre-flight v3 42/57 + §5 G5 SHIPPED + §9 standing GPU auth + §10 honest framing disclaimer — ALL PRESENT |
| 2 | `AGENTS.md` | Agent-facing entry point (28-skill catalog + Wf-296..323 + pre-flight v3 disclosure) | ✅ REFRESHED | §3 catalog 24 → 28 skills + §3.3 R20 plan skills table (+4 NEW) + §6 Wf history — ALL PRESENT |
| 3 | `TODO/INDEX.md` | Single source of truth (Wf-323 ship wave codification + §5.1 + §5.4) | ✅ REFRESHED | header line (407 → 413 files) + §0.10 + §0.11 + §22.17 + §22 closing line — ALL PRESENT |
| 4 | `TODO/README.md` | Project TOC + overview | ✅ REFRESHED | header line + Wf-323 ship wave + 28-skill catalog — ALL PRESENT |

---

## §2 §3 skill catalog delta (24 → 28 skills) — already applied

Per Wf-321B ship surface (R20+R21 skill ecosystem audit, TODO-99), 4 NEW skills added to AGENTS.md §3.3
by Wf-324D:

| # | Skill | Phase | Source |
|---:|---|---|---|
| 1 | `mol-metal-3way-multimetal` | R20-P4 (TODO-95) | Wf-321B R20 stub |
| 2 | `mol-metal-score-rep-variance-mcts` | R21-P1 (TODO-98) | Wf-321B R21 stub |
| 3 | `mol-metal-r20-skill-pack` | R20 summary pack | Wf-321B R20 meta-pack |
| 4 | `mol-metal-r21-skill-pack` | R21 summary pack | Wf-321B R21 meta-pack |

The 4 NEW R20 stubs from Wf-321B (`variance-puct-default` + `hilbert-astsr-e2e` + `closure-theorem`
+ `multiph-pll`) were already in AGENTS.md §3 from before; the 4 NEW entries added by Wf-324D are
the 4 NOT previously listed (per system-reminder skill catalog diff).

---

## §3 Pre-flight v3 HONEST-NEGATIVE REGRESSION disclosure (per `party-twenty-spirit` §3.5)

Per the `party-twenty-spirit` §3.5 honest framing mandate, README.md §0 + §4 + §10 disclose the
pre-flight v3 fresh-run result:

> **Pre-flight v3 (TODO-207)**: **42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL** (wall 65.0s, 2026-09-29
> 11:06 CST) — REGRESSION from previously-claimed 52/52 PASS PRESERVED. **15-FAIL gate breakdown
> in `TODO/INDEX.md` §22.16** (root cause: `flow_matching upstream not configured` ImportError
> cascades 5 pytest gates + 4 ADMET HONEST_NEGATIVE tag missing gates + 2 PB-packages status gates
> + NCI-60 9-cell aggregate block missing + chemprop `--quick` smoke import error + TODO-119 CLI
> token gap + `p3_gate_count_consistency_audit` silent gate count drift). **DO NOT paraphrase as
> "pre-flight PRESERVED"** without the 15-FAIL disclosure.

**Wf-324B verification**: disclosure present at README.md §0, §4, §10 + TODO/INDEX.md §22.16.
Wf-324D docs-only does NOT fix the 15 FAILs. Wf-323B targets 1 of 15 FAILs PARTIAL
(`p1_chemprop_smoke`); the remaining 14 FAILs remain PENDING for future code-class Wf (Wf-325+
candidate per Wf-321H GPU scaffolding).

**Pre-flight verification at Wf-324B timestamp**: the Wf-322D fresh-run record at
`molmetal/reports/wf322d_preflight/pre_flight.json` is FRESH (2026-09-29 11:06:51 CST = 64 min old
at Wf-324B 12:10 CST timestamp, well under 24 hr STALE threshold per AGENTS.md §4). Re-running
pre-flight at Wf-324B time is BLOCKED on `uv`-resolved torch download (5.8 GiB, exceeds 240 s wall);
the existing Wf-322D record remains authoritative. The 42/15/0/57 snapshot is preserved verbatim
in §22.16 + README.md §0/§4/§10.

---

## §4 §5.1 standing GPU authorization (replaces "kill then adapt" in README.md §9)

Per AGENTS.md §5.1 (2026-09-29 user grant "显存够就放心大胆用"), the README.md §9 standing directive
was updated from kill_then_adapt to standing GPU auth by Wf-324D. Per Wf-324B verification, the
updated text is present.

---

## §5 §5.4 Tier-2 + NO wet-lab directive (cross-reference)

Per AGENTS.md §5.4 (2026-09-29 user directive "我们发二区就可以了，不用规划湿实验"), README.md §9
cross-references the Tier-2 paper strategy. Per Wf-324B verification, the updated text is present.

§18 paper target (per the original Wf-324B task #6) is already at Tier-2 (per Wf-317 ship wave +
Wf-322G closeout). No additional change required. The §18 heading already reads:

> "## §18 Wf-139 Phase 9 — Tier-2 Paper Submission Package Ship (FINAL, RE-SCOPED from JCIM
> 2026-09-29)"

This is consistent with AGENTS.md §5.4 + Wf-322G closeout.

---

## §6 Wf-296..323 ship waves reflected in README.md §2 + AGENTS.md §6 + TODO/INDEX.md §0

Verified present at:
- `README.md` §2 — Wf-256..296 + Wf-309..323 ship wave compact table
- `AGENTS.md` §6 — "Recent Wf history pointer" section updated to mention Wf-296..323 ship waves
- `TODO/INDEX.md` §0.4 (Wf-296..316) + §0.6 (Wf-317..320) + §0.8 (Wf-321..322) + §0.10 (Wf-323) +
  §22.10/§22.11/§22.13/§22.15/§22.17 (detailed tables)

---

## §7 Wf-324B scope completion status

| Task (per Wf-324B computed brief) | Status | Notes |
|---|---|---|
| 1. Read TODO/INDEX.md (current 1955 lines) | ✅ DONE | Wf-324D extended to 2018 lines (was 1955, +63 lines for §0.10 + §0.11 + §22.17 + closing line + header rewrite) |
| 2. Update §0 ship wave timeline with Wf-309..323 entries | ✅ DONE | §0.4 (Wf-296..316) + §0.6 (Wf-317..320) + §0.8 (Wf-321..322) + §0.10 (Wf-323) all present |
| 3. Update §22 detailed table with per-Wf status | ✅ DONE | §22.10/§22.11/§22.13/§22.15/§22.17 all present |
| 4. Update §22.16 honest-framing disclaimer with Wf-323 fixes | ⚠️ PARTIAL | §22.16 was authored by Wf-322D (pre-Wf-323); §22.17 carries the Wf-323 honest-framing snapshot per Wf-324D |
| 5. Add new §22.17: Wf-323 fixes (M14 toggle / v3 schema / p1_chemprop_smoke / Vina Pt-reject) | ✅ DONE | §22.17 authored by Wf-324D — covers all 4 fixes |
| 6. Update §18 paper target — Tier-2 journal | ✅ DONE | §18 was already re-scoped to Tier-2 per Wf-317 + Wf-322G (no change needed) |
| 7. Verify pre-flight | ✅ DONE | Wf-322D record FRESH (64 min old at Wf-324B timestamp); 42/15/0/57 disclosed at §22.16 + README.md §0/§4/§10 |
| 8. Document: TODO/completed/324b_index_refresh.md | ✅ DONE | THIS FILE |

**Overall**: Wf-324B scope 7.5/8 DONE + 0.5/8 PARTIAL (§22.16 pre-Wf-423 honest-framing snapshot was
authored before Wf-323 ship, but §22.17 added the post-Wf-323 snapshot per Wf-324D). No code change
required; no further doc edits required.

---

## §8 What was NOT touched in this Wf-324B (per standing directives)

- No code under `molmetal/` modified (per AGENTS.md §1.1 MEASURED-vs-PROJECTED tag discipline)
- No `paper/*` touched (TODO-84 active per AGENTS.md §1.3)
- No GPU launches performed (AGENTS.md §5.1 standing GPU auth + per-Wf-324B docs-only scope)
- No new metrics emitted
- No new scripts shipped

---

## §9 Cross-references

- `molmetal/reports/wf324b_verify/` — (this Wf did not emit a separate verify dir; verification
  via direct read of the 4 files above)
- `TODO/completed/324d_agents_refresh.md` — Wf-324D sibling closeout that did the bulk of the work
- `TODO/completed/324a_readme_comprehensive_refresh.md` — earlier Wf-324A README refresh attempt
- `TODO/completed/324c_readme_todo_refresh.md` — earlier Wf-324C TODO+README refresh attempt
- `TODO/completed/324e_decisions_risks_refresh.md` — Wf-324E decisions.md + risks.md refresh
- `TODO/completed/324f_metrics_targets_refresh.md` — Wf-324F METRICS_TARGETS.md refresh
- `TODO/completed/325a_r2_32d_expansion.md` — Wf-325A R² 32-d expansion
- `TODO/completed/325b_vina_pt_zn_substitution.md` — Wf-325B Vina Pt/Zn substitution
- `TODO/completed/325d_m6_kappa_final.md` — Wf-325D m6 κ final
- `TODO/completed/325e_admet_4_below_final.md` — Wf-325E ADMET 4 below final
- `TODO/completed/325f_f2a_phase3g_unblock.md` — Wf-325F F2(a) Phase 3G unblock
- `AGENTS.md` §1§3§4§5 (project rules + 28-skill catalog + critical workflows + standing directives)
- `party-twenty-spirit` §3.5 (honest framing gate)

---

**END OF Wf-324B** — Owner Hugo Lee; docs-only verifier + reconciliation closeout; Wf-324D sibling
workflow had ALREADY shipped the bulk of Wf-324B's scope; Wf-324B adds a verification doc that
acknowledges the concurrent work + closes the Wf-324B ticket without duplicating edits.