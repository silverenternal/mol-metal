# TODO-84 — No paper task without authorization (rule memo)

**Date**: 2026-09-26
**Status**: 🚫 **Active rule — applies to all workflows**
**Priority**: **CRITICAL** — user authorization gate
**Owner**: Hugo Lee (decision) + ultracode agents (execution)
**Type**: **Process rule** (not a feature task)

---

## §0. Why this memo exists

User (Hugo Lee) issued a **STOP rule on 2026-09-26 13:00 CST**:

> "所有 paper 类型的任务在我授权之前禁止执行。"

(All paper-type tasks are forbidden from execution without my authorization.)

This memo records the rule so that any future Claude / ultracode agent consults `TODO/` before doing paper work.

---

## §1. Rule scope

### §1.1 Forbidden without authorization

The following are **paper-type tasks** and require explicit user OK:

| Category | Examples | Files touched |
|---|---|---|
| **Paper sections** | §1 intro / §2 related work / §3 method / §3.4-3.7 SE3 SB etc. / §4 results / §4.5 honest-negative / §5 ablation / §6 limitations / §7 future work / abstract / cover letter | `paper/sections/*.tex` (except `*_v2.tex` and `*_v3.tex` historical versions which are read-only) |
| **Paper appendix** | math_theorems.tex / closure_theorem.tex | `paper/appendices/*.tex` (read-only historical) |
| **Paper supplementary** | S1-S8 + REPRODUCIBILITY / DATA_AVAILABILITY / CODE_AVAILABILITY / ETHICS | `paper/supplementary/*.{tex,md}` (read-only historical) |
| **Paper main.tex** | Any modification to `\\input{}` directives | `paper/main.tex` (read-only historical) |
| **Paper main.pdf build** | pdflatex / bibtex / make | `paper/main.pdf` rebuild |
| **Reviewer rebuttal** | Q1-Q20 response | any rebuttal file |
| **arXiv preprint** | Upload to arXiv | arxiv.org submission |
| **Journal submission** | Digital Discovery / JCIM / JMC submit | any journal portal |

### §1.2 Allowed without authorization (CODE-ONLY work)

The following are **NOT paper tasks** and may continue without paper-specific OK:

| Category | Examples | Files touched |
|---|---|---|
| **Train scripts** | `train_se3sb.py` / `train_transport.py` / `train_ru_5metal_se3sb.py` / etc. | `molmetal/scripts/*.py` |
| **Bench scripts** | `benchmark_*.py` / `verify_*.py` | `molmetal/scripts/*.py` |
| **Adapter modules** | `se3_sb_integration.py` / `hilbert_primal_dual.py` / etc. | `molmetal/{adapters,math_skills,scripts,tests}/` |
| **Skills** | NEW skill creation / updates | `.claude/skills/*/SKILL.md` |
| **TODO docs** | Plan / close-out / INDEX / EXECUTION_ORDER | `TODO/**/*.md` |
| **Test files** | pytest / verification | `molmetal/tests/test_*.py` |
| **Reports (JSON / MD, NOT paper)** | `molmetal/reports/wf_*/final.md` | `molmetal/reports/**` |
| **Bench data JSON** | metrics / ablation results | `molmetal/reports/wf_*/*.json` |

---

## §2. Authorization protocol

### §2.1 When to ask user

Before launching any wf that touches:
1. `paper/sections/*.tex` (excluding `*_v2.tex` / `*_v3.tex` historical)
2. `paper/appendices/*.tex`
3. `paper/supplementary/`
4. `paper/main.tex`
5. `paper/main.pdf` build
6. New `paper/` directory files
7. `R18_PLAN.md` § paper-related sections

**Agent MUST ask user first**: "Can I launch paper-type wf X?"

### §2.2 What user can authorize

User may grant:
- ✅ Full OK: launch wf + ship all paper artifacts
- ✅ Partial OK: launch wf + ship only specific paper sections
- ❌ Block: do not launch wf at all
- ⏸️ Defer: park wf until R19+ GPU retrain data ready

### §2.3 Default behavior

If uncertain → **ask user before doing paper work**.

If the wf has been previously launched and is in flight → **let it complete**, but DO NOT launch new paper sub-phases.

---

## §3. Historical context

Paper work that has already shipped (frozen, no further action):

| Wf | Date | Description | Status |
|---|---|---|---|
| Wf-3 | 2026-09-25 | §3.4-3.7 method + appendix math_theorems.tex | ✅ SHIP (read-only historical) |
| Wf-6 | 2026-09-25 | CFM M3-M5 + Frontier + paper §3.4 | ✅ SHIP |
| Wf-8 | 2026-09-25 | Reality Check paper §6/§4/abstract | ✅ SHIP |
| Wf-26 | 2026-09-25 | paper v2 supplementary S1-S8 (179 pages main.pdf) | ✅ SHIP |
| Wf-33 | 2026-09-26 | paper v3 templates (§4 Table 2 / §6 Limitations / §3.4-3.7 method) + main.tex update + main.pdf rebuild | ✅ SHIP (with \MEASURED{TBD Wf-NN} placeholders) |

**All previous paper wf status**: SHIP / FROZEN. **No follow-up work** without new user authorization.

---

## §4. Currently in-flight wf (still running, NOT paper-touching)

The following wf were launched before the rule and are **code / bench / skill work only**:

| Task ID | Description | Touches paper? | Allowed to continue? |
|---|---|:---:|:---:|
| `wkl1t7tun` Wf-29 | Phase A: 修 bug + smoke (CODE-only) | ❌ No | ✅ YES |
| `wz9khezge` Wf-30 | Phase B: Vina Dock GPU retrain (CODE + bench) | ❌ No | ✅ YES |
| `w32m82npo` Wf-31 | Phase C: m1/m2/m7 GPU retrain (CODE + bench) | ❌ No | ✅ YES |
| `w8zm3dsu1` Wf-32 | Phase D: m16/m19/m8 bench (CODE + bench) | ❌ No | ✅ YES |
| `wbl30aehe` Wf-35 | Refactor train scripts (CODE) | ❌ No | ✅ YES |
| `wgqtai7iu` Wf-36 | Lambda closure + MCTS (CODE) | ❌ No | ✅ YES |
| `woof7cpu8` Wf-37 | Lambda triangle (CODE + bench) | ❌ No | ✅ YES |

**Decision**: Continue all 7 in-flight wf (they don't touch paper).

**Note**: The bench JSON output of Wf-29~32 + Wf-36~37 **could later be used to fill paper §4 / §6 placeholders** — but the **writing-up** of those placeholders into paper §4 Table 2 / §6 Limitations v3 / etc. is **forbidden without new user OK**.

---

## §5. Future paper work authorization request template

When user is ready to authorize paper work, agent should ask:

> "User, can I [specific paper action]?
> Touches: [list files]
> Based on data: [list Wf reports that supply MEASURED values]
> Risk: [if paper §X has honest-negative findings, they will be visible to reviewer]"

User reply options:
- "Yes, launch paper X"
- "Yes, but only §X + §Y" (partial)
- "No, defer to R19+"
- "No, do not modify paper"

---

## §6. Cross-reference

- `TODO/03_plans/R19_PLAN.md` §13-18 — paper v3 timeline (deferred until user OK)
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4 — honest framing rules (always apply)
- `TODO/pending/82_metric_gap_filling_plan.md` §3.5 — TODO-82 honest framing
- `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 — 0 容忍 + 不掩盖 (always apply)
- `.claude/skills/mol-metal-paper-writing/SKILL.md` — paper §1-§7 + appendix writing helper (USE WITH AUTHORIZATION)

---

## §7. Final verdict

**Until user says otherwise**:
1. ✅ Code / bench / skill / test / TODO work: continue normally
2. 🚫 Paper section / appendix / supplementary / main.tex / main.pdf / rebuttal / submission: **DO NOT TOUCH** without explicit user OK
3. 🚫 New paper wf launches: **DO NOT LAUNCH** without explicit user OK

This rule is in effect from **2026-09-26 13:00 CST** indefinitely until user revokes.

**Date**: 2026-09-26 (created)
