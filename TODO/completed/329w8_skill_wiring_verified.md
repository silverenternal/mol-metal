# Wf-329-W8: 2 NEW R20/R21 Skill Stub Wiring — VERIFIED

**Date**: 2026-09-29 (Wf-329-W8 ship)
**Status**: ✅ **SHIPPED** (2/2 NEW skills WIRED; **15/15 tests PASS**)
**Parent docs**:
- `TODO/completed/327f9_skill_stubs_verified.md` (Wf-327-F9 R20 sweep — 4/4)
- `TODO/completed/327f10_skill_stubs_verified.md` (Wf-327-F10 R21 sweep — 4/4)
- `TODO/completed/321b_r20_r21_skill_ecosystem_audit.md` (Wf-321B audit)
**Cross-ref**: AGENTS.md §3.3 (skill catalog updated to mark WIRED)

---

## §0 Why this workflow

Per `LAMBDA_ROADMAP.md` §1.5 (R20 4 NEW) + §2.4 (R21 4 NEW) + Wf-321B R20+R21 skill
ecosystem audit, 8 NEW R20+R21 skills are in scope. Wf-327-F9 (R20 sweep) wired
4 of them (`variance-puct-default` + `hilbert-astsr-e2e` + `multiph-pll` +
`closure-theorem`). Wf-327-F10 (R21 sweep) wired 4 of them
(`score-rep-variance-mcts` + `mcts-replay-learner` + `catherine-dblock-bench`
+ `triangle-closure`).

**Wf-329-W8 wires the 1 remaining skill** that was NOT covered by either
Wf-327 sweep:

| Skill | Why Wf-327 missed it |
|---|---|
| `mol-metal-closure-catherine` (R20-P5) | Different skill from `closure-theorem` (Wf-327-F9) and `catherine-dblock-bench` (Wf-327-F10); was a partial (92 LOC, no YAML frontmatter) at Wf-327 audit time |

The 2nd target (`mol-metal-triangle-closure`) was already WIRED per Wf-327-F10
(574 LOC) — Wf-329-W8 only re-verifies + adds drift guards.

---

## §1 Wf-327 audit cross-check

| Audit | Skill set | Status at Wf-327 ship | Wf-329-W8 outcome |
|---|---|---|---|
| Wf-327-F9 | `variance-puct-default`, `hilbert-astsr-e2e`, `multiph-pll`, `closure-theorem` | 4/4 WIRED | unchanged |
| Wf-327-F10 | `score-rep-variance-mcts`, `mcts-replay-learner`, `catherine-dblock-bench`, `triangle-closure` | 4/4 WIRED | `triangle-closure` re-verified |
| **Wf-329-W8** | **`mol-metal-closure-catherine`** | **NOT YET WIRED** (92 LOC, no YAML frontmatter) | **WIRED** (227 LOC, 8 sections, YAML frontmatter) |

---

## §2 Per-skill wiring status (Wf-329-W8)

### §2.1 `mol-metal-closure-catherine` (R20-P5, TODO-96) — STUB → FULL

**Before Wf-329-W8:**
- File: `.claude/skills/mol-metal-closure-catherine/SKILL.md` (92 LOC, 6 sections §0-§6)
- No YAML frontmatter (skill discovery broken)
- 6 sections: §0 Why, §1 Catherine properties, §2 Implementation, §3 Cross-domain OOD, §4 Pre-conditions, §5 Cross-refs, §6 Honest caveats

**After Wf-329-W8:**
- File: `.claude/skills/mol-metal-closure-catherine/SKILL.md` (**227 LOC, 8 sections §0-§8**)
- **YAML frontmatter added** (name: + description: + tools:) — skill discovery works
- 8 sections: §0-§6 (preserved) + **§7 Tests** + **§8 Cross-references** (expanded)
- All 8 canonical sections + YAML frontmatter + MEASURED-vs-PROJECTED ledger + cross-skill matrix
- Sister-skill references to `closure-theorem` (R20-P5) + `catherine-dblock-bench` (R20-P5 / R21-P5 alias) + `triangle-closure` (R21)
- Code anchors verified on disk: `bench_catherine_6_props_25metals.py` + `bench_closure_ood.py` + `catherine_properties.py` + `closure.py`

### §2.2 `mol-metal-triangle-closure` (R21, TODO-93 L8 cross-ref) — FULL → RE-VERIFIED

- File: `.claude/skills/mol-metal-triangle-closure/SKILL.md` (574 LOC, 8 sections §1-§8)
- YAML frontmatter present (already WIRED per Wf-327-F10)
- Wf-329-W8 update: added `Wf-327-F10` + `Wf-329-W8` WIRED acknowledgement to footer
- All 8 canonical sections + MEASURED-vs-PROJECTED ledger
- Code anchors verified on disk: `triangle_closure.py` + `bench_lambda_triangle_end_to_end.py` + `hilbert_primal_dual.py` + `lambda_astsr.py` + `variance_mcts_integration.py`

---

## §3 Test results (15/15 PASS)

`molmetal/tests/test_wf329w8_skill_wiring.py` — 9 test functions × parametrised over 2 skills = **15 tests**.

Run command (per task spec):
```bash
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/cades/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf329w8_skill_wiring.py -v
```

Actual run (mol-metal venv at `/home/hugo/codes/mol-metal/.venv/bin/python`):
```
============================= test session starts ==============================
collected 15 items

molmetal/tests/test_wf329w8_skill_wiring.py::test_skill_md_exists_with_frontmatter[mol-metal-closure-catherine] PASSED [  6%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_skill_md_exists_with_frontmatter[mol-metal-triangle-closure] PASSED [ 13%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_skill_md_has_eight_sections[mol-metal-closure-catherine] PASSED [ 20%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_skill_md_has_eight_sections[mol-metal-triangle-closure] PASSED [ 26%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_skill_cross_references_resolve[mol-metal-closure-catherine] PASSED [ 33%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_skill_cross_references_resolve[mol-metal-triangle-closure] PASSED [ 40%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_skill_activation_and_anchors[mol-metal-closure-catherine] PASSED [ 46%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_skill_activation_and_anchors[mol-metal-triangle-closure] PASSED [ 53%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_skill_md_above_stub_threshold[mol-metal-closure-catherine] PASSED [ 60%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_skill_md_above_stub_threshold[mol-metal-triangle-closure] PASSED [ 66%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_wf329w8_audit_both_wired PASSED [ 73%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_skill_has_measured_vs_projected_ledger[mol-metal-closure-catherine] PASSED [ 80%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_skill_has_measured_vs_projected_ledger[mol-metal-triangle-closure] PASSED [ 86%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_triangle_closure_wf327f10_acknowledged PASSED [ 93%]
molmetal/tests/test_wf329w8_skill_wiring.py::test_closure_catherine_wf329w8_wire_marker PASSED [100%]

============================== 15 passed in 0.38s ==============================
```

Per-test breakdown:

| # | Test | Parametrizations | PASS |
|---:|---|---:|---:|
| 1 | `test_skill_md_exists_with_frontmatter` | 2 | 2/2 |
| 2 | `test_skill_md_has_eight_sections` | 2 | 2/2 |
| 3 | `test_skill_cross_references_resolve` | 2 | 2/2 |
| 4 | `test_skill_activation_and_anchors` | 2 | 2/2 |
| 5 | `test_skill_md_above_stub_threshold` | 2 | 2/2 |
| 6 | `test_wf329w8_audit_both_wired` | 1 | 1/1 |
| 7 | `test_skill_has_measured_vs_projected_ledger` | 2 | 2/2 |
| 8 | `test_triangle_closure_wf327f10_acknowledged` | 1 | 1/1 |
| 9 | `test_closure_catherine_wf329w8_wire_marker` | 1 | 1/1 |
| **Total** | | **15** | **15/15 PASS** |

**Wall time**: 0.38 s (CPU-only doc work, no GPU/torch required for the test itself — but pytest does require torch at conftest time, hence the venv requirement).

---

## §4 AGENTS.md §3.3 update

`AGENTS.md §3.3` R20 plan skills table updated:

| Skill | GPU budget (before) | GPU budget (after Wf-329-W8) |
|---|---|---|
| `mol-metal-closure-catherine` | TBD | TBD (**WIRED per Wf-329-W8** — 227 LOC, 8 sections, YAML frontmatter added) |

`mol-metal-triangle-closure` was already marked WIRED per Wf-327-F10 in AGENTS.md §3.3
(line 112). Wf-329-W8 added only the SKILL.md footer acknowledgement, no AGENTS.md
update needed for that skill.

---

## §5 Cross-ref resolver design notes

The cross-ref resolver (`_path_resolves` + `_backticked_paths`) handles 5 Mol-Metal path conventions:

1. **Project-root relative paths**: `molmetal/scripts/foo.py`, `TODO/pending/X.md`,
   `paper/main.tex`.
2. **Skill-directory relative paths**: `references.md` (lives in skill dir).
3. **Bare filenames** that need disambiguation: `closure.py` could live under
   `molmetal/molmetal_lam/lam_chem/` or `molmetal/scripts/`.
4. **Project-relative nested paths**: `wf37_triangle/triangle_bench.json` →
   `molmetal/reports/wf37_triangle/triangle_bench.json`.
5. **STUB / PROJECTED / NOT YET CREATED markers**: lines that contain these markers
   are accepted regardless of disk resolution (e.g. PROJECTED Wf-54 output paths).

It probes multiple candidate roots (`_CANDIDATE_ROOTS` tuple — 14 candidate dirs
spanning `molmetal/scripts`, `molmetal/molmetal_lam/*`, `molmetal/math_skills`,
`molmetal/reports`, `TODO/*`, `paper/*`) before declaring a path unresolved.

---

## §6 Honest framing (per party-twenty-spirit §3.5)

| Value | Tag | Source |
|---|---|---|
| 2 NEW R20/R21 skill stubs target per Wf-329-W8 task spec | MEASURED | AGENTS.md §3.3 + Wf-321B audit |
| `mol-metal-closure-catherine` UPGRADED 92 → 227 LOC | MEASURED | `wc -l` output |
| `mol-metal-triangle-closure` was already WIRED per Wf-327-F10 | MEASURED | `TODO/completed/327f10_skill_stubs_verified.md` |
| 15/15 tests PASS | MEASURED | pytest output (this run) |
| 0.38 s wall time | MEASURED | pytest output |
| AGENTS.md §3.3 updated for closure-catherine | MEASURED | git diff |
| YAML frontmatter added to closure-catherine | MEASURED | filesystem read |
| 8 sections + MEASURED-vs-PROJECTED ledger for both skills | MEASURED | this audit |
| All 4 code anchors resolve on disk per skill | MEASURED | filesystem probe |
| Activation keywords present in description | MEASURED | frontmatter parse |
| `closure-catherine` 6/6 PASS on ≥ 25 metals | PROJECTED | R20-P5 Wf-54 (NOT this skill) |
| `triangle-closure` `mean_hilbert_r2 ≥ 0.85` on real pIC50 | PROJECTED | R21-P4 Wf-48 (NOT this skill) |

**Honest gap acknowledged**: Wf-329-W8 verifies the **SKILL.md wire status** only.
The downstream R20-P5 (TODO-96) + R21-P4 (Wf-48) ship surfaces (Catherine 25-metal
bench, triangle closure per-cell breakdown) are NOT closed by this workflow — they
remain Wf-54 + Wf-48 follow-ups.

---

## §7 What was NOT done (honest)

- **No GPU retrain was performed** — this is a CPU-only documentation verification workflow.
- **No paper §3 update was made** — TODO-84 still gates paper work without user authorization.
- **No new bench script was added** — both skills reference existing on-disk scripts:
  - closure-catherine → `bench_catherine_6_props_25metals.py` + `bench_closure_ood.py`
  - triangle-closure → `triangle_closure.py` + `bench_lambda_triangle_end_to_end.py`
- **No Wf-327 status was changed** — Wf-327-F9 + Wf-327-F10 ships are unchanged; Wf-329-W8
  only ADDED `closure-catherine` to the wire surface.

---

## §8 Cross-references

- `TODO/completed/321b_r20_r21_skill_ecosystem_audit.md` — Wf-321B parent audit
- `TODO/completed/327f9_skill_stubs_verified.md` — Wf-327-F9 R20 sweep (4/4 WIRED)
- `TODO/completed/327f10_skill_stubs_verified.md` — Wf-327-F10 R21 sweep (4/4 WIRED)
- `AGENTS.md` §3.3 — updated with WIRED marker for `closure-catherine`
- `AGENTS.md` §5.2 — Honest framing mandate
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.5 (R20) + §2.4 (R21) — NEW skill 候选
- `TODO/pending/96_r20_p5_closure_catherine.md` — R20-P5 plan
- `TODO/pending/98_wf44_closure_ood_phase2.md` — Wf-44 P2 in flight
- `TODO/pending/95_r20_p4_3way_100pocket.md` — R20-P4 (triangle upstream)
- `molmetal/tests/test_wf329w8_skill_wiring.py` — 15/15 wire verification tests
- `.claude/skills/mol-metal-closure-catherine/SKILL.md` (227 LOC, 8 sections, YAML frontmatter)
- `.claude/skills/mol-metal-triangle-closure/SKILL.md` (574 LOC, 8 sections, YAML frontmatter)

---

## §9 Summary metrics (per OUTPUT spec)

| Metric | Value |
|---|---|
| **wf327_status** | F9 (R20 sweep) 4/4 WIRED + F10 (R21 sweep) 4/4 WIRED — closure-catherine was NOT in either sweep (verified) |
| **per_skill_wired_status** | `closure-catherine` WIRED (Wf-329-W8, 92→227 LOC, 8 sections, YAML frontmatter added) / `triangle-closure` WIRED (Wf-327-F10, 574 LOC, re-verified 2026-09-29) |
| **n_tests_pass** | **15/15 PASS** in 0.38 s wall |
| **AGENTS_md_updated** | ✅ §3.3 table updated with `closure-catherine` WIRED marker + LOC count |
| **n_new_skills_created** | 0 (both skills already existed; Wf-329-W8 only upgraded `closure-catherine` to FULL format) |

---

**END OF Wf-329-W8 2 NEW R20/R21 Skill Wiring Verification** — Owner Hugo Lee + ultracode Wf-329-W8 subagent; cross-ref `party-twenty-spirit` §3.5; last update 2026-09-29.
