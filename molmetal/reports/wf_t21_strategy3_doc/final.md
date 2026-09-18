# WF-T21-Strategy3-Doc — final verdict

**Date:** 2026-09-17 (UTC)
**Workflow:** WF-T21-Strategy3-Doc — write Strategy 3 architecture design doc + flow diagram.
**Status:** **SHIPPED — documentation only (DESIGN ONLY).**
**Reference doc:** `molmetal/docs/architecture/lambda_cfm_cascaded.md`

---

## 0. TL;DR

* **Wrote** `molmetal/docs/architecture/lambda_cfm_cascaded.md` (~280 lines) covering: Overview, ASCII flow diagram, two-stage implementation roadmap, known blockers, honest framing, reference pseudo-code, SBDD paper lineage.
* **Updated** `TODO/pending/21_lambda_model_coupling.md` with the new doc reference (added at Strategy 3 header).
* **No code changes.** No GPU consumed. CPU-only documentation work as constrained.
* **Honest framing preserved:** doc explicitly marks Strategy 3 as DESIGN ONLY; Vina lift PROJECTED +1 to +2 kcal/mol; decode_ratio lift PROJECTED +0.10 to +0.30; cascades known SBDD technique (TargetDiff §4.3, FLOWr §3.4, Pocket2Mol §4.2) — Mol-Metal contribution is the typed-variable path constraint, not the cascade itself.

---

## 1. Deliverables

| File | Lines | Status |
|---|---:|---|
| `molmetal/docs/architecture/lambda_cfm_cascaded.md` | ~280 | NEW, DESIGN ONLY |
| `TODO/pending/21_lambda_model_coupling.md` | +4 (header note + link) | UPDATED |
| `molmetal/reports/wf_t21_strategy3_doc/final.md` | this report | NEW |

**Sections in the architecture doc:**

1. **Overview** — defines Lambda MCTS and CFM as independent generators; positions Strategy 3 as the pragmatic middle between Strategy 1 (reward shaping, 1 week) and Strategy 2 (joint training, 5-10 d).
2. **Architecture diagram** — ASCII flow with 4 boxes (Stage A: Lambda MCTS / Bridge: RDKit embed / Stage B: CFM refine / Selection), showing pocket.pdb + SMILES hint inputs flowing through to top-N refined candidates.
3. **Two-stage implementation roadmap** — Stage A (already shipped), Bridge (1-line addition), Stage B (2-3 d engineering, 7 sub-steps with effort + GPU costs), Stage C (1 d paper integration).
4. **Known blockers** — 4 blockers detailed: (4.1) `decode_ratio > 0` gate, (4.2) GPU flakiness, (4.3) cascade inference cost, (4.4) no feedback to Lambda.
5. **Honest framing** — 3 sub-sections: (5.1) refine is well-known SBDD technique with a table of 5 papers (TargetDiff, FLOWr, Pocket2Mol, DiffSBDD, DecompDiff) all using init→refine; (5.2) what is NOT measured; (5.3) when to start Strategy 3.
6. **Reference implementation** — ~30 lines of pseudo-code showing the `lambda_cfm_cascade` function with Stage A + Bridge + Stage B + Selection.
7. **References** — 8 cited papers and reports with file paths / arXiv IDs.

---

## 2. Source files referenced

| File | Purpose |
|---|---|
| `TODO/pending/21_lambda_model_coupling.md:81-110` | Strategy 3 spec source |
| `molmetal/molmetal_lam/search_alg/proof_search.py:3216` | Lambda MCTS `search` method (Stage A) |
| `molmetal/adapters/flow_matching_lipman/__init__.py:2298` | CFM `_generate_impl` (Stage B) |
| `molmetal/adapters/flow_matching_lipman/__init__.py:2457` | CFM `ODESolver.sample` entry point |
| `molmetal/molmetal_lam/lam_chem/decoder_rework.py:419-465` | Path B chem-aware soft bond prior |
| `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` | 5×5 Pt-click compat matrix |
| `molmetal/reports/wf_cfm_path_b_decoder_rework/final.md` | Path B decoder verdict (192/192 bond-bearing on synthetic clouds) |
| `molmetal/reports/wf_cfm_internal_review/diagnose.md` | 4 root causes of CFM decode failure |
| `molmetal/reports/wf_gpu_recovery_now/final.md` | 2026-09-15 GPU recovery + 5000-step CFM retrain (decode_ratio=0/192) |

---

## 3. Honest MEASURED vs PROJECTED

### MEASURED today

* **Lambda MCTS Stage A** already ships per Round-12 N=10×3 pilot (50.69 s wall, 5×1 with metal-seed cisplatin; honest singleton collapse documented per `wf_round12_lambda_pilot/final.md`).
* **RDKit EmbedMolecule + MMFFOptimizeMolecule** helper already exists in `r4_lambda_only_run.py:_embed_3d_for_rmsd` — the doc reuses this for the Bridge layer.
* **Path B chem-aware soft bond prior** lifts `decode_ratio` from 0/192 to 192/192 on synthetic CFM-style atom clouds per `wf_cfm_path_b_decoder_rework/final.md` — the doc references this as the bond-prior mask carrier for the typed-variable constraint.
* **5×5 Pt-click compat matrix** ships in `pt_click_compat.py` (17 tests pass per `wf_lambda_fix_full_path_v2/fix2_scaffold_aware.md`).
* **CFM internal review** identified 4 root causes (per `wf_cfm_internal_review/diagnose.md`); all 5 P0 fixes shipped per `wf_cfm_p0_fixes` 2026-09-15.
* **GPU recovery + 5000-step retrain** completed 2026-09-15 per `wf_gpu_recovery_now/final.md` (decode_ratio=0/192 at this budget).

### PROJECTED (NOT measured today)

* Vina lift +1 to +2 kcal/mol (TODO-21 line 92).
* decode_ratio lift +0.10 to +0.30 (TODO-21 line 92).
* Cascade wall-clock 4-6 min per pocket at K=5.
* Bond-prior mask effectiveness on real CFM output.
* 200-step ODE refine produces non-divergent coords (not measured on real CFM).

All PROJECTED values are flagged in the doc with `\PROJECTED{}`-equivalent markers and explicit "NOT MEASURED" framing.

---

## 4. Constraints honored

| Constraint | Status |
|---|---|
| CPU-only (no code changes) | HONORED — only Markdown files written; zero Python imports touched; no GPU consumed |
| Honest framing (future-work, not current impl) | HONORED — doc explicitly labelled "DESIGN ONLY" in 3 places (header, §6 code header, end); pseudo-code raises `NotImplementedError` |
| Cite known SBDD papers (TargetDiff, FLOWr, Pocket2Mol) | HONORED — 5-paper table in §5.1 with arXiv IDs / venues; doc references TargetDiff §4.3, FLOWr §3.4, Pocket2Mol §4.2 as required |
| Two-stage cascade described | HONORED — §2 ASCII diagram + §3 roadmap + §6 pseudo-code |
| Reference implementation (~30 lines) | HONORED — `lambda_cfm_cascade` function is 28 LOC excluding docstring + bond_prior_mask helper; total ~30 lines |

---

## 5. Follow-up workflow gates

Strategy 3 implementation is **NOT** kicked off by this doc — only the design is locked. Per TODO-21 line 7, the gating dependencies are:

1. **decode_ratio > 0** on real CFM output (currently 0/192 per `wf_cfm_frontier_research/final.md`)
2. **GPU stable** for 12+ hour runs (currently flaky per `wf_gpu_diag_fix`)
3. **Round-13 100×3 λ-only column ships** (so cascade has a baseline to compare against)

When those gates clear, Strategy 3 picks up at Stage B.3 in §3 of the doc and proceeds through Stage B.7 (Vina lift measurement) + Stage C (paper integration).

---

## 6. Schema metrics

```yaml
status: SHIPPED
agent_id: wf_t21_strategy3_doc
deliverables:
  architecture_doc:
    path: molmetal/docs/architecture/lambda_cfm_cascaded.md
    lines: ~280
    sections: 7
    ascii_flow: true
    pseudo_code_loc: ~30
  todo21_update:
    path: TODO/pending/21_lambda_model_coupling.md
    change: +4 lines (header note + link)
  verdict_report:
    path: molmetal/reports/wf_t21_strategy3_doc/final.md
    lines: this report
constraints:
  cpu_only: true
  code_changes: 0
  gpu_consumed: 0
  honest_framing: true
  sbdd_papers_cited: 5  # TargetDiff, FLOWr, Pocket2Mol, DiffSBDD, DecompDiff
gates_remaining:
  - decode_ratio > 0 on real CFM (currently 0/192)
  - GPU stable 12+ h (currently flaky)
  - Round-13 100x3 λ-only column shipped
recommendation: doc locks the design; defer implementation until all 3 gates clear
next_action: round-13 lambda-only sweep ships → re-evaluate gates → kick off Stage B.3
```

---

## 7. Files written

* `/home/hugo/codes/try_triton_on_rocm/molmetal/docs/architecture/lambda_cfm_cascaded.md` (NEW, ~280 lines)
* `/home/hugo/codes/try_triton_on_rocm/TODO/pending/21_lambda_model_coupling.md` (UPDATED, +4 lines)
* `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_t21_strategy3_doc/final.md` (NEW, this report)

---

**Verdict: SHIPPED — DESIGN ONLY. No measurements. No code. Documentation locked for post-gate implementation.**
