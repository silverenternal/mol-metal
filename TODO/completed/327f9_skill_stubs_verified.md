# Wf-327-F9: 4 NEW R20 Skill Stubs Wiring — VERIFIED

**Date**: 2026-09-29 (Wf-327-F9 ship)
**Status**: ✅ **SHIPPED** (4/4 R20 skills fully WIRED; 34/34 tests PASS)
**Parent doc**: `TODO/completed/321b_r20_r21_skill_ecosystem_audit.md` (Wf-321B ship wave)
**Cross-ref**: AGENTS.md §3.3 (skill catalog updated to mark WIRED)

---

## §0 Why this workflow

Per `LAMBDA_ROADMAP.md` §1.5 (R20 4 NEW skill candidates):
- **`mol-metal-variance-puct-default`** (R20-P1, TODO-92)
- **`mol-metal-hilbert-astsr-e2e`** (R20-P2, TODO-93)
- **`mol-metal-multiph-pll`** (R20-P3, TODO-94)
- **`mol-metal-closure-theorem`** (R20-P5, TODO-96)

Wf-321B audit (`TODO/completed/321b_r20_r21_skill_ecosystem_audit.md`) reported R20 coverage = 4/4 (100%) — all 4 NEW R20 skill candidates already had SKILL.md files at full content (Sep 26 for 3 + Sep 28 for variance-puct-default), with no STUB markers.

Wf-327-F9 **VERIFIED the wire status** + added a single regression test file (`molmetal/tests/test_wf327f9_skill_stubs.py`) that gates the 4-skill WIRED contract.

---

## §1 Per-skill wiring status

| Skill | Path | LOC | Wf-321B status | Wf-327-F9 status | Code anchors resolve |
|---|---|---:|---|---|---|
| `mol-metal-variance-puct-default` | `.claude/skills/mol-metal-variance-puct-default/SKILL.md` | **114** | ✅ COVERED | ✅ **WIRED** | ✅ 2/2 (variance_puct_integration.py + run_proof_search_variance_puct.py) |
| `mol-metal-hilbert-astsr-e2e` | `.claude/skills/mol-metal-hilbert-astsr-e2e/SKILL.md` | **526** | ✅ COVERED | ✅ **WIRED** | ✅ 2/2 (hilbert_primal_dual.py + lambda_astsr.py) |
| `mol-metal-multiph-pll` | `.claude/skills/mol-metal-multiph-pll/SKILL.md` | **518** | ✅ COVERED | ✅ **WIRED** | ✅ 2/2 (multi_parameter_ph.py + pll_signature.py) |
| `mol-metal-closure-theorem` | `.claude/skills/mol-metal-closure-theorem/SKILL.md` | **615** | ✅ COVERED | ✅ **WIRED** | ✅ 2/2 (closure.py + catherine_properties.py) |

**n_skills_wired = 4 / 4 (100%)** — all 4 NEW R20 skills fully wired (≥ 50 LOC threshold per STUB contract; all 4 far exceed it).

---

## §2 Test results (34/34 PASS)

`molmetal/tests/test_wf327f9_skill_stubs.py` — 10 test functions × parametrized over 4 skills = **34 tests** (counting parametrizations).

Run command (per task spec):
```bash
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/cades/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf327f9_skill_stubs.py -v
```

Actual run (mol-metal venv at `/home/hugo/codes/mol-metal/.venv/bin/pytest`):
```
============================== 34 passed in 0.25s ==============================
```

Per-test breakdown:

| # | Test | Parametrizations | PASS |
|---:|---|---:|---:|
| 1 | `test_skill_md_exists` | 4 | 4/4 |
| 2 | `test_skill_md_above_stub_threshold` | 4 | 4/4 |
| 3 | `test_skill_frontmatter_complete` | 4 | 4/4 |
| 4 | `test_skill_has_canonical_sections` | 4 | 4/4 |
| 5 | `test_skill_activation_keywords` | 4 | 4/4 |
| 6 | `test_skill_code_anchors_resolve` | 4 | 4/4 |
| 7 | `test_skill_todo_ref_resolves` | 4 | 4/4 |
| 8 | `test_skill_has_measured_vs_projected_ledger` | 4 | 4/4 |
| 9 | `test_all_4_r20_skills_wired` | 1 | 1/1 |
| 10 | `test_skill_ecosystem_audit_wf321b_status` | 1 | 1/1 |
| **Total** | | **34** | **34/34 PASS** |

**Wall time**: 0.25 s (CPU-only doc work, no GPU/torch required for the test itself — but pytest does require torch at conftest time, hence the venv requirement).

---

## §3 Wf-321B audit cross-ref

Wf-321B audit (`TODO/completed/321b_r20_r21_skill_ecosystem_audit.md`) reported:
- **R20 coverage = 4/4 (100%)** — all 4 NEW R20 skill candidates already had SKILL.md files.
- **n_existing_verified = 24** (full audit, all frontmatter + SKILL.md verified).
- **Wf-321B did NOT create any NEW STUB** for R20 §1.5 candidates — they were already fully wired.

Wf-327-F9 closed the verification loop:
1. Re-verified the 4 skill files exist + LOC counts (matches Wf-321B).
2. Added a regression test (34/34 PASS) to gate the contract going forward.
3. Updated `AGENTS.md §3.3` to flip the status from `STUB` to `WIRED`.

---

## §4 AGENTS.md §3.3 update

`AGENTS.md §3.3` R20 plan skills table updated:

| Skill | R20 phase | GPU budget (before) | GPU budget (after Wf-327-F9) |
|---|---|---|---|
| `mol-metal-hilbert-astsr-e2e` | R20-P2 (TODO-93) | TBD (NEW per Wf-321B) | TBD (NEW per Wf-321B; **WIRED per Wf-327-F9** — 526 LOC) |
| `mol-metal-multiph-pll` | R20-P3 (TODO-94) | TBD (NEW per Wf-321B) | TBD (NEW per Wf-321B; **WIRED per Wf-327-F9** — 518 LOC) |
| `mol-metal-variance-puct-default` | R20-P1 (TODO-92) | CPU-only first (NEW per Wf-321B) | CPU-only first (NEW per Wf-321B; **WIRED per Wf-327-F9** — 114 LOC) |
| `mol-metal-closure-theorem` | R20-P5 (TODO-96) | TBD (NEW per Wf-321B) | TBD (NEW per Wf-321B; **WIRED per Wf-327-F9** — 615 LOC) |

**Status flip**: 4 entries updated from `STUB` (implicit) → `WIRED per Wf-327-F9 (LOC count)`.

---

## §5 Honest framing (per party-twenty-spirit §3.5)

| Value | Tag | Source |
|---|---|---|
| 4 NEW R20 skill stubs were per Wf-321B claim | MEASURED | Wf-321B audit doc |
| 4/4 skills are WIRED (not STUB) | MEASURED | Wf-327-F9 LOC count + section count |
| 34/34 tests PASS | MEASURED | pytest output |
| 0.25 s wall time | MEASURED | pytest output |
| AGENTS.md §3.3 updated | MEASURED | git diff |
| `molmetal/tests/test_wf327f9_skill_stubs.py` written | MEASURED | filesystem listing |
| VariancePUCT default-on contract implemented | PROJECTED | R20-P1 Wf-50 follow-up (NOT this skill) |
| Hilbert-ASTSR R² ≥ 0.85 on PlatinAI 4-cell | PROJECTED | R20-P2 Wf-51 follow-up |
| Multi-PH + PLL R² ≥ 0.70 on PlatinAI 4-cell | PROJECTED | R20-P3 Wf-52 follow-up |
| Catherine 6/6 PASS on 25 metals × 1000 samples | PROJECTED | R20-P5 Wf-54 follow-up |

**Honest gap acknowledged**: Wf-327-F9 verifies the **SKILL.md wire status** only. The downstream R20-P1/P2/P3/P5 ship surfaces (variance_puct default-on contract, Hilbert-ASTSR retrain, Multi-PH + PLL feature extraction, Catherine 25-metal bench) are NOT closed by this workflow — they remain Wf-50 / Wf-51 / Wf-52 / Wf-54 follow-ups.

---

## §6 What was NOT done (honest)

- **No SKILL.md content was added** — all 4 SKILL.md files were already fully wired per Wf-321B (Sep 26 + Sep 28 ships).
- **No new module code was added** — the 4 skills reference existing modules (variance_puct_integration.py, hilbert_primal_dual.py, etc.) which are pre-existing R17-R19 ships.
- **No GPU retrain was performed** — this is a CPU-only documentation verification workflow.
- **No paper §3 update was made** — TODO-84 still gates paper work without user authorization.

---

## §7 Cross-references

- `TODO/completed/321b_r20_r21_skill_ecosystem_audit.md` — parent audit (Wf-321B)
- `AGENTS.md` §3.3 — updated with WIRED markers
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.5 — R20 4 NEW skill 候选 (source of truth)
- `TODO/pending/92_r20_p1_variance_puct_default.md` — R20-P1 plan
- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` — R20-P2 plan
- `TODO/pending/94_r20_p3_multiph_pll_feature.md` — R20-P3 plan
- `TODO/pending/96_r20_p5_closure_catherine.md` — R20-P5 plan
- `molmetal/tests/test_wf327f9_skill_stubs.py` — new regression test file
- `.claude/skills/mol-metal-variance-puct-default/SKILL.md` (114 LOC)
- `.claude/skills/mol-metal-hilbert-astsr-e2e/SKILL.md` (526 LOC)
- `.claude/skills/mol-metal-multiph-pll/SKILL.md` (518 LOC)
- `.claude/skills/mol-metal-closure-theorem/SKILL.md` (615 LOC)

---

## §8 Summary metrics (per OUTPUT spec)

| Metric | Value |
|---|---|
| **wf321b_status** | SHIPPED (per `TODO/completed/321b_r20_r21_skill_ecosystem_audit.md`); R20 coverage 4/4 |
| **per_skill_wired_status** | variance-puct-default WIRED (114 LOC) / hilbert-astsr-e2e WIRED (526 LOC) / multiph-pll WIRED (518 LOC) / closure-theorem WIRED (615 LOC) |
| **n_tests_pass** | **34/34 PASS** in 0.25 s wall |
| **AGENTS_md_updated** | ✅ §3.3 table updated with WIRED markers + LOC counts |
| **n_new_skills_created** | 0 (all 4 already existed per Wf-321B) |

---

**END OF Wf-327-F9 R20 Skill Wiring Verification** — Owner Hugo Lee + ultracode agents; cross-ref `party-twenty-spirit` §3.5; last update 2026-09-29.
