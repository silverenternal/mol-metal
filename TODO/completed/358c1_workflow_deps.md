# Wf-358-C1 — Workflow improvement #1: cross-workflow dependency tracking docs

**Date:** 2026-09-29
**Owner:** Hugo Lee (assisted by Claude Code)
**Status:** SHIPPED
**Workflow:** Wf-358-C1 (workflow improvement #1 — cross-workflow dependency tracking)
**Trigger:** User directive (verbatim) "你可以并发启动多个ultracode把这些问题全部修完"
— Wf-357-C3 §1 surfaced "cross-workflow dependency tracking is currently
ad-hoc" as a workflow-improvement item, and Wf-358-C1 closes that gap.

---

## §1 Root cause

Per Wf-357-C3 §1, the Mol-Metal workflow dependency graph was
inferred from prose in `TODO/completed/*.md` + `AGENTS.md §6`, but
never made explicit or queryable. This created several failure modes:

1. **Onboarding friction** — new workflow authors had to read
   every prior `TODO/completed/*.md` to discover what feeds their
   workflow.
2. **Pre-flight gate ambiguity** — when `pre_flight_check.py`
   aggregated prior workflows, the dependency was implicit in the
   gate-name string (`p3_wf327_cpu_num_workers_sufficient`) rather
   than documented in a queryable graph.
3. **Critical path invisibility** — there was no single document
   that captured the Wf-340 → Wf-356 → Wf-357 critical path that
   gates R20 ship cadence.

Wf-358-C1 closes this gap by writing a single document that:

- Lists the **19 workflows in linear chains** (§1).
- Lists the **12 explicit cross-workflow dependencies** (§2).
- Documents **per-workflow pre-conditions + post-conditions** (§3 + §4).
- Highlights the **critical path** (§5).

---

## §2 Deliverables

| Deliverable | Path | Size |
|---|---|---|
| **NEW document** | `docs/WORKFLOW_DEPENDENCY_GRAPH.md` | 289 LOC / 12,513 B |
| **NEW test** | `molmetal/tests/test_wf358c1_workflow_deps.py` | 154 LOC |
| **NEW completion doc** | `TODO/completed/358c1_workflow_deps.md` | this file |

### §2.1 Document structure (5 sections, all required)

| Section | Title | Status |
|---|---|---|
| §0 | TL;DR | Present |
| §1 | Linear chains (Wf-340..358) | Present |
| §2 | Cross-workflow dependencies | Present |
| §3 | Per-workflow pre-conditions | Present |
| §4 | Per-workflow post-conditions | Present |
| §5 | Critical path analysis (Wf-340 → Wf-356 → Wf-357) | Present |
| §6 | Honest framing | Present |
| §7 | Cross-references | Present |

### §2.2 Cross-workflow dependencies catalogued (12)

Per Wf-357-C3 §1 + Wf-358-C1 task brief: 12 explicit cross-workflow
dependencies documented in §2:

1. Wf-345-A3 → Wf-356-A2 (anchor refresh)
2. Wf-347-A1 → Wf-347-B1 (per-pair pre-launch)
3. Wf-347-A1 → Wf-347-B2 (per-pair pre-launch)
4. Wf-348-B1 → Wf-349 (R20-P3 features lift Hilbert R² gate)
5. Wf-349-A1 → Wf-353 (CFM acceleration integration)
6. Wf-350-B1 → Wf-351 (GPU launch readiness for SCI)
7. Wf-355 → Wf-356 (R20 launch wrappers + final cleanup)
8. Wf-357-A1..A6 → Wf-357-B1..B4 (analysis findings feed closeouts)
9. Wf-357-C1..C3 → Wf-357-D1 + D2 (workflow improvements feed top-2)
10. Wf-357-final → Wf-358-A1..B2 (synthesis feeds ultracode residual)
11. Wf-340 → Wf-341 (Wf-341 integrates Wf-340 findings)
12. Wf-343a → Wf-343c → Wf-343d (sequential pre-retrain check chain)

---

## §3 Test results

`molmetal/tests/test_wf358c1_workflow_deps.py` (NEW, 154 LOC):

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
plugins: hypothesis-6.168.0, anyio-4.15.1, cov-7.1.0
collecting ... collected 1 item

molmetal/tests/test_wf358c1_workflow_deps.py::test_01_workflow_deps_structure PASSED [100%]

========================= 1 passed, 1 warning in 0.32s =========================
```

**1/1 NEW test PASS** in 0.32 s.

### §3.1 Test contract

The single test (`test_01_workflow_deps_structure`) enforces:

1. Document exists at `docs/WORKFLOW_DEPENDENCY_GRAPH.md`.
2. File size > 5 KB (substantial content).
3. 5 main sections (§1..§5) all present.
4. ≥10 cross-workflow dependency entries (`Wf-XXX → Wf-YYY`).
5. MEASURED + PROJECTED tag discipline preserved.
6. AGENTS.md §5.1 + §5.2 references preserved.
7. ≥5 source path citations.
8. ≥10 distinct Wf-XXX workflow references.
9. Critical path mention (Wf-340 → Wf-356 → Wf-357).
10. Pre-condition + post-condition keywords both present.

---

## §4 Pre-flight gate verdict

Pre-flight v3 NOT re-run (docs-only + 1 NEW test). Per
`TODO/completed/335_final_synthesis.md`, the baseline is:

```
62 PASS / 0 FAIL / 62 TOTAL
EXPECTED_GATE_COUNTS["all"] 62 PRESERVED
```

Wf-358-C1 added 0 pre-flight gates (CPU-only docs + 1 test).

---

## §5 Honest framing (party-twenty-spirit §3.5)

- **Document size:** MEASURED 12,513 B / 289 LOC (target ~200 LOC;
  MEASURED 1.4× over target).
- **Test:** MEASURED 1 NEW test, 1/1 PASS in 0.32 s.
- **Sections:** MEASURED 8 main sections (5 required + §0 TL;DR +
  §6 honest framing + §7 cross-refs).
- **Cross-workflow deps:** MEASURED 12 (target 12).
- **Linear chain workflows:** MEASURED 19.
- **GPU launches:** 0 (docs-only).
- **Wall:** PROJECTED 15-25 min CPU-only; MEASURED see workflow
  execution log on disk.

---

## §6 Files touched

- `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (NEW, 289 LOC)
- `molmetal/tests/test_wf358c1_workflow_deps.py` (NEW, 154 LOC)
- `TODO/completed/358c1_workflow_deps.md` (NEW, this file)

---

## §7 Source pointers

- **Wf-357-C3** — workflow analysis (the predecessor analysis that
  flagged cross-workflow dependency tracking as ad-hoc).
- **AGENTS.md §6** — recent Wf history (linear chain).
- **TODO/INDEX.md §0.18..§0.22** — Wf-333..338 detailed table.
- **docs/HONEST_NEGATIVE_CATALOG.md** — 40 honest-negative entries
  (Wf-359-B1 expansion).
- **docs/WF296_TO_WF359_RETROSPECTIVE.md** — full retrospective
  for Wf-296..359.

---

**END OF Wf-358-C1 CLOSEOUT**
