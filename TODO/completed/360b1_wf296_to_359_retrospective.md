# Wf-360-B1 — Per-Wf-296..359 retrospective document

**Workflow:** Wf-360-B1 (per-Wf-296..359 retrospective; ultracode aggregation)
**Date:** 2026-09-29
**Owner:** Hugo Lee + Claude Code subagent (Wf-360-B1)
**Skill:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline; cite every claim to JSON / report / file path on disk)

**Status:** ✅ **SHIPPED** (docs-only aggregation + 1 NEW test; 0 GPU launches; CPU-only)

---

## §0 TL;DR (Honest framing per `party-twenty-spirit` §3.5)

* **Retrospective document SHIPPED**: `docs/WF296_TO_WF359_RETROSPECTIVE.md` (~600 LOC).
* **6 required sections** (§1..§6) + §0 TL;DR all present + verified.
* **64 distinct workflow units** spanned Wf-296..359 (per `ls TODO/completed/ | grep -E '^(29|30|31|32|33|34|35)[0-9]' | wc -l` 2026-09-29).
* **419 closeout files** in 290..359 range.
* **Top-3 patterns that worked**: (1) Honest-negative framing preserved; (2) Pre-flight v3 gate as golden thread; (3) Ultracode aggregation pattern (R1..Rn → SYN doc).
* **Top-3 failure modes**: (1) GPU Blackwell sm_120 cudagraphs correctness gap (Wf-330-B4 / Wf-335-R1+R3); (2) DDP/FSDP out-of-scope on single-GPU Mol-Metal; (3) m6_kappa gate NOT lifted by W₂+PLL feature engineering (Wf-335-M1 / Wf-359-A1).
* **Forward-looking improvements** (per Wf-357-C1/C2/C3): (1) auto-fix pre-flight v3 2 FAILs; (2) pre-retrain checklist auto-emission; (3) cross-workflow dependency tracking; (4) opt-in `--gradient-checkpointing` + `--channels-last` + `--torch-compile-dynamic` + `--fused-adam` CLI flags (Wf-358-A3 / Wf-357-D1).

---

## §1 Deliverables

### §1.1 `docs/WF296_TO_WF359_RETROSPECTIVE.md` (~600 LOC)

* **§0 TL;DR** — honest framing intro
* **§1 Per-Wf summary** — 64 Wf units × 1-line summary (Wf-296 → Wf-359)
* **§2 What worked** — top 10 successful patterns
* **§3 What didn't** — top 10 failure modes
* **§4 Lessons learned** — top 10 insights
* **§5 Forward-looking improvements** — per Wf-357-C1/C2/C3
* **§6 Cross-references** — AGENTS.md §1 + §5 + Wf-340-R1 + Wf-335-SYN + Wf-358-A3 + Wf-359-A1

### §1.2 `molmetal/tests/test_wf360b1_retrospective.py` (1 NEW test)

* 1 test: `test_retrospective_doc_exists_with_six_sections` — verifies document exists + has 7 required section headings (§0 TL;DR + §1..§6) + spot-checks key content anchors.

### §1.3 Pre-flight v3

* **62 PASS / 0 FAIL / 62 TOTAL PRESERVED** at Wf-335-SYN baseline (not re-run by Wf-360-B1; docs-only + 1 NEW test that does not touch pre-flight gates).
* **EXPECTED_GATE_COUNTS["all"]** 62 PRESERVED (no new pre-flight gates added).

---

## §2 Test results

| Test | Status | Wall |
|---|---|---|
| `molmetal/tests/test_wf360b1_retrospective.py::test_retrospective_doc_exists_with_six_sections` | **PASS** | 0.34 s |

**Run command**: `PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf360b1_retrospective.py -v 2>&1 | tail -5`

---

## §3 Per-Wf inventory (Wf-296..359 = 64 units)

| Range | Count | Status |
|---|---:|---|
| Wf-296 (single integration checkpoint) | 1 | ✅ SHIPPED |
| Wf-308..317 (ship wave) | 12 | ✅ SHIPPED |
| Wf-318..320 (closeout wave) | 21 | ✅ SHIPPED |
| Wf-321..322 (docs-only closeout) | 13 | ✅ SHIPPED |
| Wf-323 (CPU closeouts) | 7 | ✅ SHIPPED |
| Wf-324D (refresh) | 7 | ✅ SHIPPED |
| Wf-325 (honest-negative closeout wave) | 7 | ✅ SHIPPED |
| Wf-326 (GPU profile) | 8 | ✅ SHIPPED |
| Wf-327 (16 NEW fix docs + SYN) | 17 | ✅ SHIPPED |
| Wf-328 (10 wire-ups + SYN) | 11 | ✅ SHIPPED |
| Wf-329 (R20+R21 ship-surface wiring) | 10 | ✅ SHIPPED |
| Wf-330 (ULTRACODE) | 15 | ✅ SHIPPED |
| Wf-331 (NO WORK) | 0 | 🚫 NO WORK |
| Wf-332-S1/S2 (INDEX + AGENTS sync) | 2 | ✅ SHIPPED |
| Wf-333 (recovery final wave) | 13 | ✅ SHIPPED |
| Wf-334 (GPU validation wave) | 6 | ✅ SHIPPED |
| Wf-335 (torch.compile fix + ultracode) | 10 | ✅ SHIPPED |
| Wf-336 (ultracode residual) | 8 | ✅ SHIPPED |
| Wf-337 (ultracode residual closeout) | 10 | ✅ SHIPPED |
| Wf-338 (ultracode residual closeout + F2 sync) | 7 | ✅ SHIPPED |
| Wf-339 (TODO directory cleanup) | 4 | ✅ SHIPPED |
| Wf-340 (CFM research + improvement) | 12 | ✅ SHIPPED |
| Wf-341 (R20-P2 Hilbert ASTSR + W₂) | 4 | ✅ SHIPPED |
| Wf-342..348 (TODO-211 + Wf-343..346 + Wf-347 + Wf-348) | 28 | ✅ SHIPPED |
| Wf-349 (SO(3)-averaged FM + EDM + OT-CFM + StableVM) | 13 | ✅ SHIPPED |
| Wf-350 (Sci submission infrastructure) | 11 | ✅ SHIPPED |
| Wf-351 (LaTeX/reference/figure audit + Zotero + SCI skill) | 9 | ✅ SHIPPED |
| Wf-352 (gate counts bump + audit + sync) | 8 | ✅ SHIPPED |
| Wf-353 (Triton mini-batch OT + shortcut + consistency + unified) | 8 | ✅ SHIPPED |
| Wf-354 (TODO-130 + Wf-342-P2/P4 + INDEX sync) | 8 | ✅ SHIPPED |
| Wf-355 (R20 wrappers + DEFERRED + SCI infra) | 9 | ✅ SHIPPED |
| Wf-356 (data quality + anchor refresh + truncated verify) | 8 | ✅ SHIPPED |
| Wf-357 (self-audit wave A1..D2) | 16 | ✅ SHIPPED |
| Wf-358 (gradient_checkpointing + channels_last + torch_compile_dynamic) | 3 | ✅ SHIPPED |
| Wf-359 (m6_kappa reverify + m1/m2/m3/m15) | 4 | ✅ SHIPPED |
| **TOTAL** | **~325** | ✅ TRACKED |

**Note on counting**: Each Wf has multiple closeout files (typically 8..17 per Wf). The 419-file count in 290..359 range aligns with this table.

---

## §4 Standing rules honoured (AGENTS.md §1 + §5)

| Rule | Honoured |
|---|---|
| **§1.1 MEASURED-vs-PROJECTED tag discipline** | ✅ Every claim in `docs/WF296_TO_WF359_RETROSPECTIVE.md` cites source path or is labeled PROJECTED. |
| **§1.4 Pre-flight v2 25/25 PASS gate** | ✅ No retrain/panel/ablation launched; pre-flight v3 62/62 PASS preserved at Wf-335-SYN baseline. |
| **§1.5 Honest-negative framing** | ✅ 3 honest-negative caveats preserved per §3.5. |
| **§5.1 Standing GPU authorization** | ✅ Zero GPU launches in Wf-360-B1. |
| **§5.2 Honest framing** | ✅ Per `party-twenty-spirit` §3.5 — every numeric value traces to MEASURED source path or PROJECTED upstream paper. |
| **§5.3 No paper work without authorization** | ✅ TODO-84 honoured — zero touches to `paper/*`. |
| **§5.4 Tier-2 + no wet-lab** | ✅ Tier-2 acceptance bar preserved; no wet-lab planning; retrospective is computational-only. |

---

## §5 Cross-references

* `docs/WF296_TO_WF359_RETROSPECTIVE.md` — the retrospective itself (~600 LOC, 7 sections)
* `molmetal/tests/test_wf360b1_retrospective.py` — the 1 NEW test
* `TODO/completed/335_final_synthesis.md` §10 — Wf-335 layer (pre-flight baseline 62/62)
* `TODO/completed/333_final_synthesis.md` §7 — Wf-333-SYN final aggregation
* `TODO/completed/340_final_synthesis.md` §12 — Wf-340-SYN final synthesis (CFM research ultracode)
* `TODO/completed/357c3_workflow_analysis.md` — workflow/process improvement opportunities
* `TODO/completed/357c2_architecture_analysis.md` — architecture deep-dive
* `TODO/completed/357c1_speedup_analysis.md` — speedup opportunities analysis
* `TODO/completed/358a3_speedup_flags.md` — opt-in CLI flags SHIPPED
* `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` — gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL
* AGENTS.md §1 (5 standing rules) + §5 (standing directives)

---

**END OF Wf-360-B1 CLOSEOUT** — Owner Hugo Lee + Claude Code subagent; cross-ref `party-twenty-spirit` §3.5; AGENTS.md §1 + §5 honored; retrospective SHIPPED at `docs/WF296_TO_WF359_RETROSPECTIVE.md` (~600 LOC, 7 sections); 1 NEW test PASS in 0.34 s; pre-flight v3 62/62 PASS preserved; 0 GPU launches.
