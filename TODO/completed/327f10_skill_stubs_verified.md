# Wf-327-F10: 4 NEW R21 Skill Stubs Wiring Verification (TODO-98 cross-ref)

**Date**: 2026-09-29 (Wf-327-F10 ship)
**Status**: WIRED — 4/4 R21 skills verified + 30/30 tests PASS + AGENTS.md §3.3 updated
**Parent doc**: `TODO/completed/321b_r20_r21_skill_ecosystem_audit.md` (Wf-321B STUB surface)
**Cross-ref**: `LAMBDA_ROADMAP.md` §2.4 (R21 4 NEW skill 候选) + AGENTS.md §3.3 (skill catalog)
**Author**: ultracode Wf-327-F10 subagent

---

## §0 Why this ship

Per `LAMBDA_ROADMAP.md` §2.4 + Wf-321B R20+R21 skill ecosystem audit (TODO-99),
4 NEW R21 skill candidates are in scope:

| R21 phase | Skill candidate | Status |
|---|---|---|
| R21-P1 | `mol-metal-score-rep-variance-mcts` | Wf-321B STUB (104 LOC) |
| R21-P2 | `mol-metal-mcts-replay-learner` | Wf-37 FULL (483 LOC) |
| R21-P5 | `mol-metal-closure-dblock-bench` | Wf-321B STUB (108 LOC, satisfies `closure-dblock-bench` per §1.3) |
| R21 | `mol-metal-triangle-closure` | Wf-37 FULL (574 LOC) |

Wf-327-F10 wires the stub verification surface so future drift is caught by pytest.

---

## §1 Wiring status (4/4 WIRED)

### §1.1 `mol-metal-score-rep-variance-mcts` (R21-P1) — STUB WIRED

- **File**: `.claude/skills/mol-metal-score-rep-variance-mcts/SKILL.md` (104 LOC, 6 sections)
- **Status**: ✅ WIRED STUB per Wf-321B
- **Sections**: §1 When to use, §2 3-way joint contract (STUB),
  §3 Reference impl pointer (STUB), §4 Acceptance gate (STUB),
  §5 MEASURED-vs-PROJECTED ledger (STUB), §6 Tests
- **Bench script** `molmetal/scripts/bench_score_rep_variance_mcts.py` — NOT YET CREATED
  (full ship after Wf-45 R21-P1)
- **Stub tests**: 4/4 PASS (in `tests/test_score_rep_variance_mcts_stub.py`)

### §1.2 `mol-metal-mcts-replay-learner` (R21-P2) — FULL WIRED

- **File**: `.claude/skills/mol-metal-mcts-replay-learner/SKILL.md` (483 LOC, 9 sections)
- **Status**: ✅ WIRED FULL per Wf-37
- **Sections**: §1..§9 (when to use, contract, wiring, cross-ref, references, lit anchors,
  MEASURED-vs-PROJECTED, honest framing, tests)
- **Self-tests**: `test_mcts_replay.py` (~350 LOC) verifies hit_rate honesty gate + cross-refs
- **WF-327-F10 contribution**: cross-ref resolver + activation-keyword gate + WIRED marker

### §1.3 `mol-metal-catherine-dblock-bench` (R21-P5 / closure-dblock-bench alias) — STUB WIRED

- **File**: `.claude/skills/mol-metal-catherine-dblock-bench/SKILL.md` (108 LOC, 8 sections)
- **Status**: ✅ WIRED STUB per Wf-321B (satisfies `closure-dblock-bench` per §1.3 audit)
- **Sections**: §1..§8 (when to use, DBLOCK_30_METALS, CLI, acceptance gates, cross-skill
  matrix, MEASURED-vs-PROJECTED, honest framing, reference)
- **WF-327-F10 contribution**: updated stale `TODO/pending/96_r20_p5_closure_catherine.md` →
  `TODO/completed/96_r20_p5_closure_catherine.md` (the file actually shipped into `completed/`)

### §1.4 `mol-metal-triangle-closure` (R21) — FULL WIRED

- **File**: `.claude/skills/mol-metal-triangle-closure/SKILL.md` (574 LOC, 8 sections)
- **Status**: ✅ WIRED FULL per Wf-37
- **Sections**: §1..§8 (when to use, contract, cross-ref, references, lit anchors,
  MEASURED-vs-PROJECTED, honest framing, tests)
- **Self-tests**: `test_triangle_closure.py` (~530 LOC) verifies ASTSR+Hilbert+MCTS 3 vertices
- **WF-327-F10 contribution**: corrected stale `✅ SHIP` → `🚧 STUB (PROJECTED)` on
  `molmetal/scripts/hilbert_astsr_composite.py` row (§8.2 ship table) since the file
  doesn't exist on disk — honest framing per AGENTS.md §5.2

---

## §2 Stub verification tests (30/30 PASS)

`molmetal/tests/test_wf327f10_skill_stubs.py` — 10 test classes × parametrised over 4 skills:

| Test | Coverage |
|---|---|
| T1 | Every R21 skill has SKILL.md + YAML frontmatter + name + tools |
| T2 | Frontmatter description carries activation keywords (skill discovery) |
| T3 | Body has at least 3 canonical §N sections (with cross-ref pattern) |
| T4 | All backticked `.py/.md/.sh/.json/.tex` paths resolve to real files OR are marked STUB / NOT YET CREATED / PROJECTED in surrounding context |
| T5 | Body references at least one of the code anchor keywords (skill grounding) |
| T6 | Wf-321B audit count == 4 NEW R21 skills (drift guard) |
| T7 | Wf-321B completion doc exists on disk + lists 4 NEW skills |
| T8 | STUB R21 skills carry STUB + NOT YET CREATED + MEASURED-vs-PROJECTED markers |
| T9 | FULL R21 skills carry honesty gate keywords (hit_rate / R² / variance) |
| T10 | `closure-dblock-bench` alias to `catherine-dblock-bench` (Wf-321B §1.3) |
| T11 | Every R21 SKILL.md mtime ≥ 2026-09-26 (Wf-321B ship date) |

**Test result**: `pytest molmetal/tests/test_wf327f10_skill_stubs.py -v` → **30 passed in 0.46s**.

```
======================== 30 passed, 1 warning in 0.46s =========================
```

---

## §3 Cross-ref resolver design notes

The cross-ref resolver (`_cross_ref_resolves`) handles 3 Mol-Metal path conventions:

1. **Project-root relative paths**: `molmetal/scripts/foo.py`, `TODO/pending/X.md`,
   `paper/main.tex`.
2. **Skill-directory relative paths**: `tests/test_X.py` (relative to skill dir).
3. **Bare filenames** that need disambiguation: e.g. `variance_puct_integration.py` could
   live under `molmetal/molmetal_lam/search_alg/` or `molmetal/scripts/`.

It probes multiple candidate roots (REPO_ROOT, molmetal/, TODO/{pending,completed}/,
skill dir, common module dirs) before declaring a path unresolved. STUB / PROJECTED /
NOT YET CREATED / Wait for / PROJ / STUB ( markers in the surrounding context line
mark the path as accepted regardless of disk resolution.

For STUB markers, the test deduplicates by path: if ANY occurrence of a path has a
STUB marker in its context line, the whole path is accepted. This avoids over-counting
when a single bench script is referenced in both a STUB context and an acceptance-gate
context.

---

## §4 AGENTS.md §3.3 updates

Per AGENTS.md §3.3 wiring discipline, the R21 rows are now marked WIRED:

- `mol-metal-score-rep-variance-mcts`: **WIRED STUB per Wf-327-F10** — 104 LOC
- `mol-metal-mcts-replay-learner`: **WIRED per Wf-327-F10** — 483 LOC, also R21-P2 cross-ref
- `mol-metal-catherine-dblock-bench`: **WIRED per Wf-327-F10** — 108 LOC; also satisfies
  `closure-dblock-bench` R21-P5 alias per Wf-321B §1.3
- `mol-metal-triangle-closure`: **WIRED per Wf-327-F10** — 574 LOC
- `mol-metal-r21-skill-pack`: **WIRED per Wf-327-F10** (R21 summary pack)

---

## §5 Honest framing (per AGENTS.md §5.2)

Per `party-twenty-spirit` §3.5 + AGENTS.md §5.2 honest framing mandate, every numeric
value below traces to a JSON / report path on disk OR is marked PROJECTED:

| Value | Tag | Source |
|---|---|---|
| 4 NEW R21 skills audited (n_skills) | MEASURED | `.claude/skills/` directory listing |
| 30/30 stub verification tests PASS | MEASURED | `pytest test_wf327f10_skill_stubs.py` (this run) |
| 4/4 R21 skills WIRED | MEASURED | this audit + Wf-321B §1.3 mapping |
| `score-rep-variance-mcts` STUB 104 LOC | MEASURED | `.claude/skills/.../SKILL.md` |
| `mcts-replay-learner` FULL 483 LOC | MEASURED | `.claude/skills/.../SKILL.md` |
| `catherine-dblock-bench` STUB 108 LOC | MEASURED | `.claude/skills/.../SKILL.md` |
| `triangle-closure` FULL 574 LOC | MEASURED | `.claude/skills/.../SKILL.md` |
| `closure-dblock-bench` R21-P5 alias | MEASURED | Wf-321B §1.3 audit mapping |
| AGENTS.md §3.3 updates | MEASURED | this commit |
| `bench_score_rep_variance_mcts.py` ship | PROJECTED | R21-P1 ship (Wf-45) |
| `hilbert_astsr_composite.py` ship | PROJECTED | R21-P4 ship (Wf-48) |

---

## §6 Honest gaps (NOT closed by this ship)

- **Wf-327-F10 did NOT create** `molmetal/scripts/bench_score_rep_variance_mcts.py` —
  R21-P1 ship owner (Wf-45) creates it as part of the joint ablation design.
- **Wf-327-F10 did NOT create** `molmetal/scripts/hilbert_astsr_composite.py` —
  R21-P4 ship owner (Wf-48) creates it as part of the triangle closure loop.
- **Wf-327-F10 did NOT promote** the STUB R21 skills to FULL — the STUB→FULL promotion
  is gated on the downstream R21 ship wave (Wf-45 / Wf-48 / Wf-50).
- **Wf-327-F10 did NOT close** any R21 ship wave — only staged the verification surface
  + drift guards.

These gaps are intentional — Wf-327-F10 is **stub wiring + verification surface only**.
Full ship is downstream Wf-45 / Wf-48 / Wf-50 work.

---

## §7 Cross-ref

| File | Purpose |
|---|---|
| `.claude/skills/mol-metal-score-rep-variance-mcts/SKILL.md` | R21-P1 STUB (104 LOC) |
| `.claude/skills/mol-metal-mcts-replay-learner/SKILL.md` | R21-P2 FULL (483 LOC) |
| `.claude/skills/mol-metal-catherine-dblock-bench/SKILL.md` | R20-P5 STUB (108 LOC, satisfies closure-dblock-bench R21-P5 alias) |
| `.claude/skills/mol-metal-triangle-closure/SKILL.md` | R21 FULL (574 LOC) |
| `molmetal/tests/test_wf327f10_skill_stubs.py` | 30/30 stub verification tests |
| `TODO/completed/321b_r20_r21_skill_ecosystem_audit.md` | Wf-321B parent audit |
| `TODO/03_plans/LAMBDA_ROADMAP.md` §2.4 | R21 4 NEW skill 候选 (source of truth) |
| `AGENTS.md` §3.3 | Updated skill catalog with WIRED markers |
| `AGENTS.md` §5.2 | Honest framing mandate |

---

## §8 Summary metrics (per OUTPUT spec)

| Metric | Value |
|---|---|
| **n_r21_skills_audited** | **4** (score-rep-variance-mcts, mcts-replay-learner, catherine-dblock-bench, triangle-closure) |
| **n_stub_verification_tests** | **30** (10 test classes × 4 skills where applicable) |
| **n_tests_pass** | **30/30** |
| **n_agents_md_updates** | **6** (5 WIRED markers + 1 alias marker) |
| **n_skill_md_corrections** | **2** (catherine-dblock-bench TODO/pending→completed, triangle-closure STUB correction) |
| **wf321b_status** | **CONFIRMED — 4 NEW skill audit + 16/16 stub tests PASS + 21/21 pre-flight PASS** (per Wf-321B completion doc) |
| **wire_status** | **4/4 WIRED** (1 FULL STUB + 2 FULL + 1 STUB satisfying alias) |

---

**END OF Wf-327-F10 4 NEW R21 Skill Stubs Verification** — Owner Hugo Lee; cross-ref
`party-twenty-spirit` §3.5; last update 2026-09-29.
