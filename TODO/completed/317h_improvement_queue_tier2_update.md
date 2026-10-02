# Wf-317H — IMPROVEMENT_QUEUE Tier-2 journal re-scope

**Date**: 2026-09-29
**Owner**: Hugo Lee (per user explicit instruction 2026-09-29)
**Status**: ✅ SHIPPED — paper target re-scoped from "Q2 JCIM" → "Tier-2 journal"; wet-lab planning path explicitly closed
**Parent**: TODO-138 paper v2 submission package (Wf-317H; cross-ref Wf-317A `317a_index_md_tier2_update.md` for INDEX.md sibling update)

---

## §0 User directive (source of truth)

> 哦对了，我们发二区就可以了，不用规划湿实验
>
> ("Oh right, we'll just publish in Tier-2 journals, no need to plan wet-lab experiments")

Per AGENTS.md §0 (Wf-harness user-relay priority), the user message **wins** over the workflow-computed task text. The original Wf-317H brief asked for `JCIM → Tier-2 journal` swap, which is the surface update; the user message additionally mandates **wet-lab path closed** (no further wet-lab planning).

---

## §1 What changed (file-level)

| File | Lines changed | Edit |
|---|---:|---|
| `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` | 4 (row 5 of pending table; §2 "Then: 任务 #4"; §3 ledger row; §6 fallback) | Tier-2 journal re-scope |
| `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` | new §2 sub-block "RECENT SHIP WAVE (2026-09-29 cross-ref)" | Recent ship wave cross-ref per AGENTS.md §6 |

### §1.1 Edits in detail

1. **Row 5 of `### Still PENDING` table (line ~45)**
   - Before: `TODO-138 | Paper v2 submission package (Q2 JCIM) | P0 | paper builds clean + cover letter + figures + supp`
   - After:  `TODO-138 | Paper v2 submission package (Tier-2 journal — per user 2026-09-29) | P0 | paper builds clean + cover letter + figures + supp`

2. **§2 "⏳ Then: 任务 #4 — TODO-138 (Paper v2 submission package)"**
   - Before: `Verify: paper builds clean (0 unresolved refs) + JCIM submission checklist complete`
   - After:  `Verify: paper builds clean (0 unresolved refs) + Tier-2 journal submission checklist complete (per user 2026-09-29 明示 — no wet-lab planning required)`

3. **§3 Progress Ledger row "Wf-138 (TODO-138)"**
   - Before: `Paper v2 + JCIM submission package`
   - After:  `Paper v2 + Tier-2 journal submission package (user 2026-09-29 re-scope)`

4. **§6 Final Verdict fallback line**
   - Before: `Fallback: paper §4 + §6 ship current 8 真 SOTA + 8 honest-negative — Q2 ready`
   - After:  `Fallback: paper §4 + §6 ship current 8 真 SOTA + 8 honest-negative — Tier-2 journal ready (per user 2026-09-29, no wet-lab planning required)`

5. **§2 NEW sub-block "RECENT SHIP WAVE (2026-09-29 cross-ref)"**
   - Cross-references Wf-296 (SHIPPED, per AGENTS.md §6 + TODO/README.md §259) + Wf-300/301/303/304 (in flight)
   - Explicitly notes Wf-309..317 IDs from brief were NOT found on disk (verified via Bash `ls TODO/completed/` + grep on TODO/INDEX.md + TODO/README.md 2026-09-29); per `party-twenty-spirit` §3.5 honest framing, no fabrication

---

## §2 Wet-lab planning closed (per user 2026-09-29)

User instruction "不用规划湿实验" (no wet-lab planning) means:
- TODO-138 paper v2 ships **without** wet-lab validation section in §5
- TODO-97 (`TODO/pending/97_r20_p6_wetlab_ic50_dryrun.md` / `TODO/completed/97_r20_p6_wetlab_ic50.md`) status: **NOT in flight under Wf-317H scope** — neither promoted nor demoted by this wave
- The `pending/wetlab_sop_*.md` (3 files: binding_affinity / cellular_uptake / cytotoxicity) and `pending/wetlab_loi_*.md` (4 files: cold_email_template / consolidation_review / ornal / pku_cisplatin_ic50 / rusanov_2025_platinai / xiehe_nci60) — these files are **out of scope** for this Wf-317H ship and remain in `TODO/pending/` untouched. Per user directive, no new wet-lab planning work; the existing pending files are NOT auto-archived by this wave.

Per AGENTS.md §5.3 "No paper work without authorization" (TODO-84 active), this wave also did NOT touch `paper/*` — only the planning doc in `TODO/03_plans/`.

---

## §3 Pre-flight v2 (TODO-207) gate status

| Metric | Value | Source |
|---|---|---|
| Total gates | 48 | `pre_flight_check.py` JSON |
| PASS | 48 / 48 | same |
| FAIL | 0 | same |
| SKIP | 0 | same |
| Exit code | 0 | `b5wbgyi7b` background task (completed 2026-09-29) |
| Total wall | 305.98 sec (~5 min) | `total_wall_sec` field |
| STALE flag | NOT triggered | pre-flight report refreshed within 24 hr |

All P0/P1/P2/P3 gates PASS, including:
- `p0_config_schema_verify`, `p0_csv_hash_verify`, `p0_scaffold_zero_leak` — 3/3 P0
- `p1_chemprop_indices_fix`, ..., `p1_cfm_catherine_30metals` — 18/18 P1
- `p2_data_metalcytotox_exists`, `p2_data_metallo_drugs_exists`, `p2_tests_pass`, `p2_vina_installed` — 4/4 P2
- `p3_*` — 23/23 P3 (frontier math wire, anchor-guard adoption, harness auto-fallback, ADMET honest-negative tagging, NCI-60 9-cell per-cell aggregate, LOI send-ready, R20-P0 loaders, PoseBusters packages, SA/QED score metrics, etc.)

---

## §4 Why JCIM is Tier-2 (cross-ref for §1.1 edits)

JCIM (Journal of Chemical Information and Modeling, ACS) impact factor ~5-6, indexed in Web of Science as Q1-Q2 depending on year, but commonly categorized as **Tier-2** in chemistry-computation community for metallodrug / 3D-SBDD papers. With paper §4 headline currently a mix of 8 MEASURED + 8 honest-negative metrics (per AGENTS.md §6 / TODO/README.md §259 Wf-296 ship), Tier-2 is the realistic venue — higher-tier (Nature Comp Sci / JCISD / Nature Chem) would require additional wet-lab validation, which user has now closed the door on.

Alternative Tier-2 venues to consider at submission time (NOT in scope for this wave):
- **JCIM** (current) — closest fit for §3 method + §4 METRICS_TARGETS table
- **J. Cheminform.** (Springer Open Access) — better for §6 limitations-heavy honest-negative framing
- **Mol. Inform.** (Wiley) — fits §3 algorithm ship surface
- **J. Comput. Chem.** — alternative

---

## §5 Honest framing (per `party-twenty-spirit` §3.5)

- Tier-2 re-scope is **user-authoritative**, NOT a metric-claim downgrade. The §4 headline content (8 MEASURED + 8 honest-negative) is unchanged. What changes is the venue tier ceiling. Per AGENTS.md §1 rule 1, this is a MEASURED re-scope event, not a fabrication.
- Wf-309..317 ship-wave IDs from brief: **NOT verified on disk**. Per AGENTS.md §1 rule 1 (zero tolerance for fabricated numbers) + party-twenty-spirit §3.5 honest framing, the §1.5 [5] edit uses **actual** recent ship wave data (Wf-296 + 300/301/303/304 per AGENTS.md §6) rather than fabricating the brief's Wf-309-317 entries.
- Wet-lab path: closed by user directive; existing `TODO/pending/wetlab_*` files untouched. No new wet-lab planning work created by this wave.

---

## §6 Files touched (cross-ref)

| Path | Status |
|---|---|
| `/home/hugo/codes/mol-metal/TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` | MODIFIED (4 row edits + 1 new sub-block) |
| `/home/hugo/codes/mol-metal/TODO/completed/317h_improvement_queue_tier2_update.md` | NEW (this file) |
| `paper/*` | NOT TOUCHED (AGENTS.md §5.3 / TODO-84 active) |
| `TODO/pending/wetlab_*` (8 files) | NOT TOUCHED (out of scope per user) |
| `TODO/pending/97_r20_p6_wetlab_ic50*.md` | NOT TOUCHED (out of scope per user) |

---

## §7 Output (PLAIN TEXT)

n_rows_updated = **4** (row 5 pending table + §2 "Then: 任务 #4" + §3 ledger row + §6 fallback)
preflight_status = **PASS (48/48, exit=0, ~306s wall, NOT STALE)**

Per AGENTS.md §8 (Wf-291C / Wf-302C text-only brief compliance): plain-text output only; **NO** StructuredOutput calls.

---

**END OF Wf-317H ship report** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; last update 2026-09-29.