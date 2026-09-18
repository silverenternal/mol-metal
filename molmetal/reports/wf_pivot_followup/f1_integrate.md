# WF-Pivot-Followup F1-INTEGRATE — Pivot A Integration Audit

**Date**: 2026-09-16
**Workflow**: WF-Pivot-Followup F1-INTEGRATE (integration audit of widzuipcl deliverables against Pivot A paper framing)
**Scope**: paper/sections/ (looking for §3.7 metallodrug vertical + §4.7 metallodrug panel) + molmetal/reports/wf_metallodrug_vertical/ (widzuipcl phase reports)
**Verdict**: **WAIT — widzuipcl not yet completed**

---

## 1. What widzuipcl has delivered (Phase 1 only)

`molmetal/reports/wf_metallodrug_vertical/` contains exactly 3 files:

| File | Status | Contents |
|------|--------|----------|
| `phase1_inventory.md` | SHIPPED | Metallodrug dataset inventory (PlatinAI, MetalCytoToxDB, tmQM corpora). |
| `phase1_inventory.json` | SHIPPED | Machine-readable counterpart. |
| `phase1_filter_fix.md` | SHIPPED | CFM training-data filter extension: 4-element organic donor vocab `{C,N,O,F}` with fixed 19-heavy-atom mols → 14-element metallodrug-relevant vocab `{C,N,O,F,S,P,Cl,Br,I,Pt,Pd,Au,Ir,Ru}` with 8..38 heavy-atom range. `--n-train` default 32 → 500. Two new data sources: `platinai`, `metallo_drugs_combined`. New `data_diversity.py` helper (Morgan-ECFP4 + greedy MaxMin). 13 new tests in `test_data_diversity.py` (all pass). End-to-end smoke: `build_combined_pool n=500 → 449 SMILES in 5.1s` (51 unparseable SMILES dropped); cache written to `molmetal/data/metallo_drugs_500_train.csv`. |

**Phase 1 is structurally correct** (verified by F1-VERIFY `phase1_filter_fix.md` §"Constraints Honored" — 13/13 tests pass; backward compat preserved; `paper/` untouched).

---

## 2. What widzuipcl has NOT yet delivered

The integration audit checklist explicitly references:

| Required widzuipcl deliverable | Status |
|---|---|
| `paper/sections/03_7_metallodrug_vertical.tex` | **MISSING** (file does not exist on disk) |
| `§4.7 metallodrug panel` in `04_evaluation.tex` | **MISSING** (no `§4.7` heading present; §4 currently ends at `sec:evaluation:hybrid-wip` line 2199) |
| `molmetal/reports/wf_metallodrug_vertical/MASTER.md` | **MISSING** (only Phase 1 reports exist) |
| Phase 2+ GPU retrain report | **NOT STARTED** |
| Phase 3 paper §4 update report | **NOT STARTED** |

Confirmed via:

```
$ ls paper/sections/ | grep -i metallodrug
NO_SECTION_FILE

$ ls molmetal/reports/wf_metallodrug_vertical/
phase1_filter_fix.md
phase1_inventory.json
phase1_inventory.md

$ grep -ni "metallodrug\|metal.complex" paper/sections/04_evaluation.tex
(no matches)
```

---

## 3. Why this F1-INTEGRATE is blocked (not a fail — a wait)

The Pivot A pivot (F1-VERIFY, verdict = PASS, all 9 invariants hold) explicitly says:

> "ready for the metallodrug vertical. The de novo typed-term MCTS column is the paper's primary contribution; §3.1-§3.4 metallodrug content (MLC formalism + 5-click + MetalGeometryPrior + MCTS) is unchanged and load-bearing."

The pivot is structurally and honestly reframed. But the **§3.7 "metallodrug vertical" extension** (§3.1-§3.4 hold the EXISTING content; the spec'd "§3.7 metallodrug vertical" per the orchestrator instruction is a NEW section that widzuipcl would add) and the **§4.7 metallodrug panel** (also new, would add a metallodrug-specific evaluation column) are NOT YET WRITTEN.

Without widzuipcl's §3.7 / §4.7 deliverables, there is nothing to integrate. The remaining invariant checks all reduce to "file does not exist, cannot verify".

---

## 4. Verified pre-conditions for when widzuipcl ships

Once widzuipcl delivers, the integration audit will need to verify:

1. **§3.7 uses de novo framing, NOT SBDD claims.** Pivot A's whole point is "de novo typed-term MCTS, pocket-optional" (§1 line 110). §3.7 must use this framing. Any leftover "geometric SBDD" / "structure-based" / "DiffDock comparison" claims would be a regression.

2. **§4.7 metallodrug panel cross-refs §1, §3.7, §6.** The §4.7 panel should carry cross-references to:
   - `sec:method:metallodrug-vertical` (or wherever widzuipcl anchors §3.7)
   - `sec:method:secondary-cfm` (the §3.6 stub) and/or `sec:future:cfm-deferred` (§7.2)
   - `sec:limitations` for honest framing
   - `sec:future:lambda-cfm-coupling` (§7.1, PROMOTED) if §4.7 mentions Lambda × CFM coupling

3. **`paper/main.tex` has `\input{sections/03_7_metallodrug_vertical}` (or whichever name widzuipcl picks).** If widzuipcl placed it elsewhere (e.g., into §3.1-§3.4 as subsections), the integration still needs a corresponding `\input` line.

4. **No orphan `\cite` refs to SBDD SOTA that Pivot A removed** (DiffSBDD, Pocket2Mol, TargetDiff, MolDiff, DecompDiff, FLOWr — see F1-VERIFY §1 invariant 8: 6 `footnote:*` SBDD entries confirmed absent). Any new §3.7/§4.7 cite must be checked against the 100-`@`-entry `paper/refs.bib`.

5. **Cross-ref to `\ref{sec:method:vertical}` (or whatever widzuipcl named it) is consistent.** All `\ref` and `\label` must resolve.

6. **Cross-refs to §7 future work** (e.g., §7.1 Lambda × CFM coupling PROMOTED) if §3.7/§4.7 mentions the future direction.

7. **No claim drift back to "deep generative SBDD" / "geometric SBDD"**. The Pivot A language was honest, reversible, and journal-ship-ready (per F1-VERIFY §4.10). §3.7/§4.7 must preserve that framing.

---

## 5. Action taken (this audit, not a fix)

- Confirmed §3.7 file does NOT exist.
- Confirmed §4.7 panel does NOT exist.
- Confirmed widzuipcl MASTER.md does NOT exist.
- Confirmed Pivot A paper state is PASS via prior F1-VERIFY audit (`f1_verify.md` lines 74 pages / 5.07 MB / 0 unresolved refs).
- DID NOT modify any file in `paper/`. Per spec, widzuipcl owns the metallodrug section/panel writing; this agent only verifies when those land.
- Saved this wait-state note.

---

## 6. Recommendation to orchestrator

1. **DO NOT block other agents.** They should continue on parallel tracks that do not touch `paper/sections/03_7_metallodrug_vertical.tex` (if widzuipcl plans to write it) or `paper/sections/04_evaluation.tex` (if widzuipcl plans to extend it with §4.7).

2. **Re-run F1-INTEGRATE once widzuipcl ships** Phase 2+ deliverables. The audit can then verify the 7 items in §4 above and emit "INTEGRATION OK" or a fix-list.

3. **Suggested pivot-A-preserving guardrails for widzuipcl when they write §3.7/§4.7** (informational, not enforced by this agent):
   - Use the term **"metallodrug de novo typed-term MCTS"** (matches Pivot A §1 line 110).
   - Frame the metallodrug vertical as **"pocket-optional"** (Pivot A §1 line 110).
   - Cross-ref §6 limitations if reporting honest negatives (e.g., the n_distinct=1 collapse that propagates to metallodrug too).
   - Cite de novo generator lineage (REINVENT4 / GraphAF / JTVAE / latent-FM / MolDQN) where relevant; do NOT cite SBDD SOTA (DiffSBDD / Pocket2Mol / TargetDiff / MolDiff / DecompDiff / FLOWr) as comparison — cite only as context if needed.
   - If §4.7 reports MCTS metrics on metallodrug, the column header should mirror §4 Table 2 ("de novo typed-term MCTS — metallodrug pocket").

4. **Final verdict of THIS audit**: WAIT. widzuipcl widzuipcl-metallodrug-vertical has not completed §3.7/§4.7/MASTER.md. Re-run F1-INTEGRATE when those files land.

---

**F1-INTEGRATE verdict**: **WAITING ON WIDZUIPCL widzuipcl-metallodrug-vertical** — only Phase 1 (training-data filter + diversity sampler) has shipped; §3.7, §4.7, MASTER.md, and Phase 2+ are not yet started. No file changes made.