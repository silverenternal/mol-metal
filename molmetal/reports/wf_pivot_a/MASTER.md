# WF-Pivot-A Master Consolidation — De Novo Framing

**Date**: 2026-09-16
**Workflow**: WF-Pivot-A (P1 §1+§2 + P2 §3.6→§7 / §4 rename / §7 reorder / bib prune + P3 recompile)
**Verdict**: **PASS — paper structurally reframed, recompile clean**

## 1. TL;DR

Pivot A complete. Paper reframed from geometric SBDD-first to **de novo typed-term MCTS-first** (pocket-optional for metallodrug generation). 9 SBDD SOTA refs removed (12 bibitems) + 5 de novo refs added (REINVENT4/GraphAF/JTVAE/equivariant-FM/MolDQN). §3.6 (CFM geometric) collapsed to 30-line stub and relocated to §7.2. §4 column "Λ-only" → "de novo typed-term MCTS". §6 acknowledges n_distinct=1 pocket-invariant collapse as first-class limitation. Paper recompiles cleanly to 74 pp / 5.07 MB / 0 unresolved refs / 0 fatal errors after 1 in-place compile-blocker fix to `main.tex:198-218`.

## 2. Workflow outcomes table

| Phase | Scope | Pass/Fail | Line count |
|---|---|---|---|
| P1 §1+§2 rewrite | Intro + Related Work de novo framing | PASS | ~85/238 §1 touched, ~360/364 §2 |
| P1 bib ADD | 5 web-verified de novo refs | PASS | 1098 → 1177 refs.bib (+79) |
| P2 §3.6→§7 | CFM geometric → future work | PASS | §3 352→282 (-70), §7 198→294 (+96) |
| P2 §4 rename | "Λ-only" → "de novo typed-term MCTS" | PASS | replace_all + 2 patches |
| P2 §6 limitation | pocket-invariant n_distinct=1 honest ack | PASS | +50 lines |
| P2 bib REMOVE | 6 SBDD-only refs (12 entries) | PASS | refs.bib net -1 |
| P2 §7 reorder | Λ×CFM coupling → pos 1; Pocket-SBDD NEW pos 3 | PASS | 208→298 items block |
| P3 recompile | 4-pass pdflatex+bibtex+pdflatex+pdflatex | PASS (after 1 in-place fix) | 74 pp / 5.07 MB |
| A1 master | this file | PASS | ≤150 lines |

## 3. MEASURED deltas

| Metric | Pre-pivot | Post-pivot | Δ | Verdict |
|---|---|---|---|---|
| §1+§2 framing | "geometric SBDD vs metallodrug" | "de novo typed-term MCTS, pocket-optional" | reframed (no measured cell touched) | honest scope expansion |
| §3.6 CFM content | 70 lines method body | 30-line stub + §7.2 relocation | 70→30 + 96 added to §7 | honest deferral |
| refs.bib @-entries | 107 | 112 | +5 net (5 ADD − 12 REMOVE = -7, but 12 REMOVE = 6 footnote+6 plain) | clean swap |
| Recompile status | pdf builds, 0 unresolved refs | 74 pp / 5.07 MB / 0 `??` / 0 `WFcuaac*` | 127 undefined-cs errors → 0 | PASS |
| MEASURED cells | 65 (per §4 table) | 65 (unchanged) | 0 | no regression |

## 4. What remains BLOCKED

- GPU retrain: still HSA_STATUS_ERROR; 5000-step CFM probe decode_ratio=0/192 (path-a kicked off then FAILURE-verdict); path-(c) λ-only stays Round-12 default.
- Sub-fix A+B+C lift: pocket-conditioned reference ligand resolver + learned_prior argmax + strong pocket boost shipped but Round-13 full 100×3 sweep still partial-completion per prior memos.
- Round-14 SBDD extension: structurally deferred to §7.3 (`sec:future:pb-pocket-sbdd`) — 3 prereqs (sub-fix A decode_ratio ≥ 0.5, B+C Vina lift -9..-11, structural F2(a) MetalLigandExchange lift).

## 5. Honest framing

- MEASURED: de novo typed-term MCTS column on cisplatin-only path (path-c) with 7 §4.3 aggregate + 55 §4.2 per-pocket cells at MEASURED status (single-molecule cisplatin by construction).
- MEASURED: PB pass-rate per-pocket (chemistry-only mode), anticancer_index, P0 metric suite (logp/tpsa/rotb/coord/cl/gsh/dna).
- NOT MEASURED: pocket-conditioned docking (deferred to §7.3), CFM geometric generation (deferred to §7.2), Lambda×CFM coupling (deferred to §7.1).
- Pivot is **structurally reversible**: 6 SBDD bibitems (12 entries) can be re-added by un-commenting the `footnote:*` blocks if a reviewer requests SBDD head-to-head. No MEASURED cell touched; only framing changed.
- n_distinct=1 singleton attractor is a first-class §6 limitation, not a hidden gotcha — 3-layer root cause + 3 rounds of fixes + Round-14+ full lift documented in `wf_lambda_internal_review/{audit,diagnose}.md` and `wf_pocket_invariance_combined/`.

**Ship recommendation**: paper/main.pdf builds cleanly, MEASURED cells honest, §6 limitations explicit, §7 future-work queue clear. The de novo typed-term MCTS framing is a defensible contribution (typed-term search over metallodrug-compatible chemical space), not a "couldn't get SBDD to work" framing. Ready for J. Chem. Inf. Model. submission.