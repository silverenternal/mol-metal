# Wf-324D — README + AGENTS + TODO/INDEX refresh (2026-09-29 11:20 CST)

**Phase**: Wf-324D docs-only refresh 2026-09-29 (this entry) — refreshes `README.md` + `AGENTS.md` + `TODO/INDEX.md` + `TODO/README.md` to reflect the Wf-296..323 ship waves + Tier-2 re-orientation (AGENTS.md §5.4) + §5.1 standing GPU authorization + pre-flight v3 42/57 HONEST-NEGATIVE REGRESSION disclosure.

**Skill alignment**: `party-twenty-spirit` §3.5 (every Wf cite traces to a MEASURED JSON / report path; PROJECTED vs SHIPPED tag discipline preserved); `AGENTS.md` §5.1 (standing GPU authorization); `AGENTS.md` §5.4 (Tier-2 + NO wet-lab directive).

**Honest framing**: docs-only Wf. NO code change. NO GPU launch. NO paper work (per TODO-84 ACTIVE). 4 file edits only.

---

## §1 Files refreshed (4 edits)

| # | File | Surface | Status | Lines changed (approx) |
|---:|---|---|---|---|
| 1 | `README.md` | Top-level project status hub (Wf-256..323 ship wave + Tier-2 + §5.1 + pre-flight v3 honest-negative) | ✅ REFRESHED | header line + §0 status snapshot (8 rows) + §1 quick navigation (28 skills + Tier-2 line) + §2 recent Wfs (Wf-256..323 compact table) + §4 pre-flight v3 42/57 HONEST-NEGATIVE REGRESSION + §5 G5 SHIPPED via Wf-310A + G6 dry-run READY + §9 standing GPU authorization (replaces "kill then adapt") + §10 honest framing disclaimer |
| 2 | `AGENTS.md` | Agent-facing entry point (28-skill catalog + Wf-296..323 + pre-flight v3 disclosure) | ✅ REFRESHED | §3 header (24 → 28 skills) + §3.3 R20 plan skills table (+4 NEW: `mol-metal-3way-multimetal` + `mol-metal-score-rep-variance-mcts` + `mol-metal-r20-skill-pack` + `mol-metal-r21-skill-pack`) + §6 recent Wf history (Wf-296..323 + Wf-324D ship waves) |
| 3 | `TODO/INDEX.md` | Single source of truth (Wf-323 ship wave codification + §5.1 + §5.4) | ✅ REFRESHED | header line (407 → 413 files) + §0.10 Wf-323 ship-wave timeline + §0.11 net effect + §22.17 Wf-323 detailed table + §22 "END OF" closing line + §22.16 in-line correction (kill_then_adapt → standing GPU auth per §5.1) |
| 4 | `TODO/README.md` | Project TOC + overview (header line + Wf-324D refresh marker) | ✅ REFRESHED | header line (407 → 413 files + 28-skill catalog + Wf-323 ship wave) |

---

## §2 §3 skill catalog delta (24 → 28 skills)

Per Wf-321B ship surface (R20+R21 skill ecosystem audit, TODO-99), 4 NEW skills added to AGENTS.md §3.3:

| # | Skill | Phase | Source |
|---:|---|---|---|
| 1 | `mol-metal-3way-multimetal` | R20-P4 (TODO-95) | Wf-321B R20 stub |
| 2 | `mol-metal-score-rep-variance-mcts` | R21-P1 (TODO-98) | Wf-321B R21 stub |
| 3 | `mol-metal-r20-skill-pack` | R20 summary pack | Wf-321B R20 meta-pack (aggregates 4 R20 + 12 R19 → 20-skill R20 ship surface) |
| 4 | `mol-metal-r21-skill-pack` | R21 summary pack | Wf-321B R21 meta-pack (aggregates 4 R21 + 16 R17-R20 → 20-skill R21 ship surface) |

Note: the 4 NEW R20 stubs from Wf-321B (`variance-puct-default` + `hilbert-astsr-e2e` + `closure-theorem` + `multiph-pll`) were ALREADY in AGENTS.md §3 from before Wf-324D; the 4 NEW entries added by Wf-324D are the 4 NOT previously listed (per system-reminder skill catalog diff).

---

## §3 Pre-flight v3 HONEST-NEGATIVE REGRESSION disclosure (per `party-twenty-spirit` §3.5)

Per the `party-twenty-spirit` §3.5 honest framing mandate, README.md §0 + §4 + §10 now disclose the pre-flight v3 fresh-run result:

> **Pre-flight v3 (TODO-207)**: **42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL** (wall 65.0s, 2026-09-29 11:06 CST) — REGRESSION from previously-claimed 52/52 PASS PRESERVED. **15-FAIL gate breakdown in `TODO/INDEX.md` §22.16** (root cause: `flow_matching upstream not configured` ImportError cascades 5 pytest gates + 4 ADMET HONEST_NEGATIVE tag missing gates + 2 PB-packages status gates + NCI-60 9-cell aggregate block missing + chemprop `--quick` smoke import error + TODO-119 CLI token gap + `p3_gate_count_consistency_audit` silent gate count drift). **DO NOT paraphrase as "pre-flight PRESERVED"** without the 15-FAIL disclosure.

**Wf-324D scope**: docs-only — does NOT fix the 15 FAILs. Wf-323B targets 1 of 15 FAILs PARTIAL (`p1_chemprop_smoke`); the remaining 14 FAILs remain PENDING for future code-class Wf (Wf-325+ candidate).

---

## §4 §5.1 standing GPU authorization (replaces "kill then adapt" in README.md §9)

Per AGENTS.md §5.1 (2026-09-29 user grant), the README.md §9 standing directive is updated from:

> Per `mol-metal-gpu-execution` §0: **explicit fresh user OK required before any GPU launch > 1 hr wall.**

to:

> Per AGENTS.md §5.1: User has granted **GENERAL GPU authorization** — **"显存够就放心大胆用"** (use freely as long as VRAM is enough). No per-launch user OK required. Constraints: **VRAM budget check first** (`nvidia-smi`); **hyperparameters via config files**; **pre-flight PASS** still required; **dry-run defaults** still apply.

TODO/INDEX.md §22.16 in-line correction also added: kill_then_adapt → standing GPU authorization.

---

## §5 §5.4 Tier-2 + NO wet-lab directive (cross-reference)

Per AGENTS.md §5.4 (2026-09-29 user directive, "我们发二区就可以了，不用规划湿实验"), README.md §9 now cross-references the Tier-2 paper strategy:

> **Tier-2 paper strategy (AGENTS.md §5.4, 2026-09-29 user directive)**: User said **"我们发二区就可以了，不用规划湿实验"** — Tier-2 journal is sufficient; do NOT plan wet-lab experiments. Paper target = Tier-2 computational-only (JCAMD / J. Mol. Graph. Model. / WIREs Comput. Mol. Sci. / Mol. Divers. / Comput. Biol. Chem.). 4 LOI drafts + 3 wetlab SOPs + R20-P6 wet-lab IC50 + Wf-74 v2 wet-lab outreach **DEPRECATED**; in-silico alternatives re-scope the wet-lab effort.

---

## §6 Wf-296..323 ship waves reflected in README.md §2 + AGENTS.md §6 + TODO/INDEX.md §0

| Wf | Round | Surface | Status | Honest framing |
|---|---|---|---|---|
| **Wf-256..258** | R20 | CFM TODO gap audit + research + synthesize | ✅ SHIPPED | 5 CFM TODOs + 37.8 hr GPU total |
| **Wf-259..262** | R20 | Pre-flight v1 → v2 + 4 NEW scripts | ✅ SHIPPED | 16 conversion decisions; 4 NEW scripts wired |
| **Wf-263A..268** | R20 | chemprop v3 DrugOOD fix + §9 expansion + G5/G6 dry-run verify + paper §4 claim audit + collect/dashboard + LOMO 3-flag + CFM panel stub + cross-ref sync | ✅ SHIPPED | 0 aspirational; 5 minor §3.5 violations; pre-flight 25/25 PASS |
| **Wf-296** | R20 | Pipeline integration audit + integration (5 agents) | ✅ SHIPPED | 42 orphans found; 16 dead flags wired |
| **Wf-300/301/303/304** | R20 | Below-avg remediation + all-in-one fix + TODO audit + final closeout | ✅ SHIPPED (via Wf-309..317) | drugood BELOW + Vina Pt-stripped + PlatinAI 4-cell stratified + stale-read + harness detector + AnchorGuard consumer adoption + PlatinAI Selenium + TODO top 3 implement |
| **Wf-309..317** | R20 | Frontier-math wire-in (CPU) + actionable-now bundle + Part 4 deterministic edits + G5 retrain + LOI integration + §9 expansion 30→36 + R20 Phase 0 loaders + OOD RDKit fix + SA/QED wiring + NCI-60 scaffold fallback + TODO directory cleanup + **Tier-2 re-orientation** | ✅ SHIPPED | 11 Wf in single day; §9 honest framing 30→36 entries |
| **Wf-318..320** | R20 | Actionable-now closeout + Wf-315 closeout + Wf-313 closeout + final synthesis | ✅ SHIPPED | downstream lift dashboard surfaces 4/6 consumers as HONEST_NEGATIVE; Vina Phase 2 pipeline 6/6 pockets green |
| **Wf-321..322** | R20 | Wf-176 m19 LOMO CPU + R20/R21 skill ecosystem (8 NEW skills) + pre-flight gate audit + ablation runner integration + chemprop v2 reconciliation + TODO-119 P1.4 CPU + 4 LOI + 3 SOP + IMPROVEMENT_ENUM Tier-2 markers + INDEX.md §0 + §22 + README sync | ✅ SHIPPED (docs + CPU) | 8 NEW skill stubs; 16/16 stub tests PASS; pre-flight 21/21 PASS |
| **Wf-323** | R20 | M14 --lr-warmup-steps re-integration + v3/v4 schema drift fix + M8 full CPU + ablation runner integration + TODO-21 lambda CFM coupling + M4 SA/QED sub-criterion + TODO-134 Vina install attempt | ✅ SHIPPED (CPU) | 7 closeouts; pre-flight v3 fresh-run 42/57 PASS / 15 FAIL REGRESSION NOT fixed |
| **Wf-324D** | R20 | README.md + AGENTS.md + TODO/INDEX.md refresh (this entry) | ✅ SHIPPED (docs-only) | 28-skill catalog + Tier-2 + §5.1 + pre-flight v3 honest-negative disclosure |

---

## §7 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED today (2026-09-29 12:02 CST)**:
  - Wf-324D: 4 file edits SHIPPED (README.md + AGENTS.md + TODO/INDEX.md + TODO/README.md).
  - 8 INDEX.md entries added (1 new §0.10 + 1 new §0.11 + 1 new §22.17 + 1 §22.16 in-line correction + 1 AGENTS.md §3 catalog sync 24→28 + 1 AGENTS.md §6 Wf history sync + 1 §22 "END OF" closing line update + 1 README.md status snapshot refresh).
  - AGENTS.md §3 catalog updated to 28 skills (24 + 4 NEW per Wf-321B).
  - README.md refreshed to current state (Wf-296..323 ship wave + Tier-2 §5.4 + §5.1 standing GPU auth + pre-flight v3 honest-negative).
  - pending/ **13 files** UNCHANGED (Wf-323 + Wf-324D docs-only; no pending/ moves).
  - completed/ 407 → **413 files** (per `ls TODO/completed/ | wc -l` 2026-09-29 12:02 CST; +6 since Wf-322D = 7 Wf-323 closeouts + 1 Wf-324D aggregation + 5 concurrent Wf-324e/f + Wf-325d/e/f writes - retroactive baseline drift between Wf-322D write and now; concurrent writes during this Wf-324D session).

- **NOT MEASURED today (PENDING or honest-negative disclosure)**:
  - ⚠️ HONEST-NEGATIVE REGRESSION PRESERVED: pre-flight v3 fresh-run **42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL** (wall 65.0s, 2026-09-29 11:06 CST) — 15-FAIL breakdown per `TODO/INDEX.md` §22.16. Wf-323B targets 1 of 15 FAILs PARTIAL (`p1_chemprop_smoke`); the remaining 14 FAILs remain PENDING for future code-class Wf (Wf-325+ candidate).
  - Wf-323C PARTIAL_BLOCK 19/25 promotion to 25/25 — GPU 5×5×50 paper-grade sweep DEFERRED to R20-P1 (TODO-120) per §5.1 standing GPU auth.
  - Wf-321A + Wf-321H + Wf-322F GPU follow-ups (Wf-310E Wave-A / Wave-B / Wave-C + Wf-310E Wf-176 Phase 2): GPU launchable under §5.1 standing GPU authorization (no per-launch user OK); total GPU wall ~7-9 hr if all 3 waves launch.

---

## §8 Standing directives (per `../AGENTS.md` §5, 2026-09-29 snapshot)

- **§5.1 Standing GPU authorization (2026-09-29 user grant)**: "显存够就放心大胆用" — no per-launch user OK required; VRAM budget check + config-file hyperparameters + dry-run default still apply.
- **§5.4 Tier-2 journal target + NO wet-lab planning (2026-09-29 user directive)**: "我们发二区就可以了，不用规划湿实验" — paper target = Tier-2 computational-only; 4 LOI drafts + 3 wetlab SOPs + R20-P6 wet-lab IC50 + Wf-74 v2 wet-lab outreach DEPRECATED; in-silico alternatives re-scope the wet-lab effort.
- Honest framing mandate (per `party-twenty-spirit` §3.5): every numeric value MUST trace to JSON / report path on disk; PROJECTED vs MEASURED tag required.
- No paper work without authorization (TODO-84 active): zero touches to `paper/*` unless user explicitly OKs.
- Pre-flight v3 (TODO-207): 42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL (HONEST-NEGATIVE REGRESSION PRESERVED per `TODO/INDEX.md` §22.16); STALE flag (24 hr) blocks PASS.

---

**END OF Wf-324D** — Owner Hugo Lee; review `party-twenty-spirit` §3.5; last update 2026-09-29 12:02 CST.
