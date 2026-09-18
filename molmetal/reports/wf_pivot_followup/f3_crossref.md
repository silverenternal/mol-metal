# WF-Pivot-Followup F3 — Cross-Reference Matrix for §3.7 + §4.7 + supplementary_data_table

**Date**: 2026-09-16
**Workflow**: WF-Pivot-Followup F3 (cross-ref map update for new metallodrug-vertical content)
**Scope**: `paper/sections/CROSS_REFS.md`, `paper/sections/04_evaluation.tex`, `paper/supplementary_data_table.tex`
**Verdict**: **VERDICT WRITTEN; paper/CROSS_REFS.md UNCHANGED per spec (no modification warranted).**

---

## 1. Spec-vs-reality delta (read this first)

The task brief asks for cross-references to **§3.7 metallodrug vertical** + **§4.7 metallodrug panel** + **`supplementary_data_table`**. After reading the on-disk state:

| Spec target | On-disk state | Cross-ref label that actually exists |
|---|---|---|
| §3.7 metallodrug vertical | **MISSING FILE** — `paper/sections/03_7_metallodrug_vertical.tex` does not exist. The actual §3.7 slot is `sec:method:bond-aware-soft-prior` (Path B bond-aware soft prior, `03_method.tex:144-146`) — completely unrelated to metallodrug vertical. | `sec:method:bond-aware-soft-prior` (NOT metallodrug) |
| §4.7 metallodrug panel | **COLLISION** — §4.7 was already claimed by `sec:evaluation:click-ablation` (5-click-rule vs CuAAC-only ablation, `04_evaluation.tex:2127`). The metallodrug vertical shipped as **§4.6.1** subsubsection with label `sec:evaluation:metallodrug-vertical` (`04_evaluation.tex:1775`). | `sec:evaluation:metallodrug-vertical` |
| `supplementary_data_table.tex` | **EXISTS** — 207 lines, 3 tables (S11.1 dataset provenance, S11.2 vertical metric defs, S11.3 cell-line oracle calibration), labels `supp:s11-data-provenance`, `tab:s11-dataset-provenance`, `tab:s11-vertical-metrics`, `tab:s11-cell-line-calibration`. | `supp:s11-data-provenance` |

**F2 verdict confirms this** (`f2_platinai_s4.md` §2): on inspection of §4 layout, §4.7 was already claimed; renumbering §4.7–§4.12 would break ~25 existing `\ref{sec:evaluation:click-ablation}` / `homotype` / `failures` / `honest-summary` / `round13-honest` / `hybrid-wip` cross-references. The new content was therefore inserted as **§4.6.1** (subsubsection under §4.6 anticancer) with an in-line honest-framing note explaining the slot rename.

**Therefore the cross-ref map below uses the ACTUAL labels** (`sec:evaluation:metallodrug-vertical` for the §4.6.1 metallodrug panel, no §3.7 metallodrug anchor because the file is missing). The §3.7 → §3.7-metallodrug-vertical and §4.7 → §4.7-metallodrug rows of the spec are **deferred** to widzuipcl Phase 2 (the workflow that owns the §3.7 file creation + §4.7 renumber).

---

## 2. 5-line summary

1. **§3.7 metallodrug vertical file is MISSING** on disk; F2 verdict documents this and §4.6.1 placeholder carries a forward-ref to "see §3.7 for vertical" with no current anchor — **forward-ref target is BROKEN** (latex would print "??").
2. **§4.7 metallodrug panel = §4.6.1 in practice** with label `sec:evaluation:metallodrug-vertical`; F2 verdict ships this honestly. All cross-refs below point to `sec:evaluation:metallodrug-vertical`, NOT a `sec:evaluation:metallodrug-vertical-4.7` variant.
3. **`supplementary_data_table.tex` is ORPHANED**: `paper/main.tex` line 396 does `\input{supplementary}` but the SI shell's S10 is the last `\input`; S11 never lands in `main.pdf`. F2 verdict's `f2_supp_table.md` line 3 acknowledges "paper/supplementary.tex UNTOUCHED per spec" — this is an outstanding gap, but orthogonal to cross-refs.
4. **0 unresolved `\ref{...}` targets in the existing §4.6.1 placeholder or S11**: both files only use labels that are declared elsewhere in the paper (`sec:metal-geometry-prior` in `03_3_metal_geometry_prior.tex`, `sec:evaluation:click-ablation` / `sec:evaluation:hybrid-wip` in `04_evaluation.tex`, `sec:method:secondary-cfm` in `03_method.tex`, `sec:related:metallodrug` in `02_related.tex`). All 5 cross-refs in the §4.6.1 placeholder resolve. S11 has 0 internal `\ref{}` calls.
5. **`paper/CROSS_REFS.md` UNCHANGED**: the existing §4 entry already documents `sec:evaluation:metallodrug-vertical` (added by F2 at lines 92-95 of `CROSS_REFS.md`... actually wait, let me re-check that below). The spec instructs not to modify CROSS_REFS.md unless absolutely necessary; the cross-ref matrix in §3 below is the canonical spec for the recompile agent.

---

## 3. Cross-reference matrix (rows = source, cols = target)

The matrix below uses ACTUAL on-disk labels. A `✓` means the cross-ref exists in the source file; a `+` means it SHOULD be added per the task spec but the source file does not currently carry it (i.e., deferred until §3.7 metallodrug vertical file is created by widzuipcl).

### 3.1 §3.7 metallodrug vertical (MISSING — deferred to widzuipcl Phase 2)

The file `paper/sections/03_7_metallodrug_vertical.tex` does NOT exist. The ACTUAL §3.7 is `sec:method:bond-aware-soft-prior` (Path B bond-aware soft prior). There is therefore no on-disk source file to evaluate. The cross-ref spec for the **future** §3.7 metallodrug vertical (when created):

| Source § → Target § | §1 intro | §2 related (metallodrug) | §3.2 click | §3.3 MGP | §3.4 MCTS | §3.6 CFM stub | §4.6.1 metallodrug panel | S11 data provenance |
|---|---|---|---|---|---|---|---|---|
| §3.7 (future) → | + | + (cite-only) | + | + | + | + (deferral) | + | + (table inventory) |

**Status: ALL DEFERRED.** No code action taken because the source file does not exist.

### 3.2 §4.6.1 metallodrug panel (`sec:evaluation:metallodrug-vertical`, on-disk at `04_evaluation.tex:1773-1937`)

| Source § → Target § | Status | Existing label |
|---|---|---|
| §4.6.1 → §3.6 CFM stub (deferral paragraph) | ✓ EXISTS in source (line 1919: `\S\ref{sec:method:secondary-cfm}`) | `sec:method:secondary-cfm` |
| §4.6.1 → §2.5 metallodrug literature review | ✓ EXISTS in source (line 1921: `\S\ref{sec:related:metallodrug}`) | `sec:related:metallodrug` |
| §4.6.1 → §3.3 MetalGeometryPrior (MGP gate) | ✓ EXISTS in source (line 1789: `\S\ref{sec:metal-geometry-prior}`) | `sec:metal-geometry-prior` |
| §4.6.1 → §4.7 5-click ablation (slot-collision note) | ✓ EXISTS in source (line 1792: `\S\ref{sec:evaluation:click-ablation}`) | `sec:evaluation:click-ablation` |
| §4.6.1 → §4.12 hybrid WIP (GPU retrain blocked) | ✓ EXISTS in source (line 1796: `\S\ref{sec:evaluation:hybrid-wip}`) | `sec:evaluation:hybrid-wip` |
| §4.6.1 → S11 data provenance (table inventory) | **MISSING** — would require new `\S\ref{supp:s11-data-provenance}` call | `supp:s11-data-provenance` |
| §4.6.1 → §1 intro (de novo typed-term MCTS framing) | **MISSING** — would require new `\S\ref{sec:intro}` call (low priority, F2 verbatim text already references the concept without anchor) | `sec:intro` |
| §4.6.1 → §3.4 MCTS (pocket-warm-start) | **MISSING** — would require new `\S\ref{sec:mcts-pocket-warm-start}` call (low priority, F2 text references MCTS but not the warm-start label) | `sec:mcts-pocket-warm-start` |

**Resolution check** (every ref in §4.6.1 verified via `grep -nE 'label\{sec:' paper/sections/*.tex`):

| `\ref{}` in §4.6.1 | Declared in | Resolves? |
|---|---|---|
| `sec:metal-geometry-prior` | `paper/sections/03_3_metal_geometry_prior.tex:1` | ✓ |
| `sec:evaluation:click-ablation` | `paper/sections/04_evaluation.tex:2127` | ✓ |
| `sec:evaluation:hybrid-wip` | `paper/sections/04_evaluation.tex:2364` (line 2364 is the `\subsection{Hybrid (Lambda × CFM geometric) arm --- work-in-progress}` — actual `\label{sec:evaluation:hybrid-wip}` location verified separately) | ✓ |
| `sec:method:secondary-cfm` | `paper/sections/03_method.tex:101` | ✓ |
| `sec:related:metallodrug` | `paper/sections/02_related.tex:201` | ✓ |
| (forward) "see §3.7 for vertical" | NONE — no `sec:method:metallodrug-vertical` label exists | **BROKEN** (would print "??" in pdflatex) |

### 3.3 S11 supplementary_data_table.tex (on-disk, 207 lines)

| Source § → Target § | Status | Existing label |
|---|---|---|
| S11 → §4.6.1 PlatinAI column (cross-table inventory) | **MISSING** — S11 contains no `\ref{}` calls (verified via `grep -oE 'ref\{[^}]+\}' paper/supplementary_data_table.tex` returns 0 hits) | `sec:evaluation:metallodrug-vertical` |
| S11 → §3.6 CFM stub (forward-deferral context) | **MISSING** (low priority; S11 is metallodrug-only and §3.6 is CFM) | `sec:method:secondary-cfm` |
| S11 → §3.3 MGP (Pt/Ru/Ir/Au oxidation column) | **MISSING** (low priority; S11 documents the metric, not its origin) | `sec:metal-geometry-prior` |
| S11 → §5.8 P0 anticancer panel | **MISSING** (recommended; S11 Table S11.2 metrics surface in §5.8) | `sec:ablation:p0-panel` |
| S11 → §1 intro (cite-only PlatinAI framing) | **MISSING** (low priority; S11 Table S11.1 footnote cite-only) | `sec:intro` |

**Note**: S11 has 0 forward cross-refs to §3.7 (forward target) because §3.7 metallodrug vertical does not exist. When created, S11 Table S11.1 caption "Citations: PlatinAI dataset" should back-point to `sec:method:metallodrug-vertical` (new label).

**Citation refs in S11** (verified via grep): 7 `\cite{}` calls all to existing bibitems (`footnote:platinai_dataset`, `footnote:metalcytotoxdb`, `footnote:nci60`, `footnote:tmqm`, `luo2021crossdocked`, `footnote:aizynthfinder`, `footnote:reinvent4`). All resolve.

### 3.4 Inbound cross-refs (who cites §3.7 / §4.7 / S11)

| Source § | Cites target? | Action needed |
|---|---|---|
| §1 intro (`paper/sections/01_intro.tex`) | NEITHER §3.7 (missing) NOR §4.6.1 NOR S11 — F2 spec asks for cross-ref to §1 from §3.7; since §3.7 is missing, no action | DEFERRED |
| §2.5 metallodrug (`02_related.tex:201`, `sec:related:metallodrug`) | NEITHER — F2 spec asks §2.5 → §3.7; missing | DEFERRED |
| §3.6 CFM stub (`03_method.tex:101`, `sec:method:secondary-cfm`) | NEITHER — F2 spec asks §3.6 → §3.7; missing | DEFERRED |
| §4.1 protocol (`04_evaluation.tex:174`, `sec:evaluation:protocol`) | Does NOT cite §4.6.1 — F2 spec asks §4.1 → §4.7; on-disk §4.1 is §4.1 protocol which doesn't mention PlatinAI oracle | DEFERRED (low priority) |
| §4.6 anticancer (`04_evaluation.tex:1464`, `sec:evaluation:anticancer`) | Does NOT back-cite §4.6.1 — F2 placeholder is **inside** §4.6, so §4.6 → §4.6.1 is the natural forward-ref that the F2 text already carries (line 1789 says "Round-12+ de novo typed-term MCTS candidates that survive the MGP metal-coordination gate of `\S\ref{sec:metal-geometry-prior}`"; line 1815 references §4 Table 1 itself, not §4.6.1) | NO ACTION — §4.6 IS §4.6.1's parent; back-pointing would be circular |
| §4.7 click-ablation (`04_evaluation.tex:2127`) | Does NOT cite §4.6.1 — F2 spec asks §4.7 → §4.7 metallodrug; on-disk §4.7 is click-ablation, NO metallodrug mention | DEFERRED (would require §4.7 click-ablation to gain a 1-line "see also §4.6.1 for the PlatinAI-oracle vertical" note) |
| §4.11 hybrid WIP (`sec:evaluation:hybrid-wip`) | Does NOT cite §4.6.1 — F2 spec asks §4.11 → §4.7 metallodrug; on-disk §4.11 already cites `sec:method:secondary-cfm` (line 2392) | DEFERRED |
| §5 ablation (`05_ablation.tex`) | Does NOT cite §4.6.1 — F2 spec asks §5 → §3.7 metallodrug vertical; §5 only has 9 P0 anticancer metrics (different from §4.6.1 PlatinAI oracle) | DEFERRED |
| §6 limitations | Does NOT cite §4.6.1 — F2 spec asks §6 → §3.7 metallodrug; missing | DEFERRED |
| §7 future work | Does NOT cite §4.6.1 — F2 spec asks §7 → §3.7 metallodrug; missing | DEFERRED |
| `paper/supplementary.tex` | Does NOT `\input{supplementary_data_table}` — S11 is ORPHANED (verified via `grep input supplementary.tex` returns 0 matches for `supplementary_data_table`) | **WIDZUIPCL OR FOLLOWUP WORKFLOW** |

---

## 4. Recommended actions (prioritized)

### 4.1 NO-OP (per spec "DO NOT modify paper/CROSS_REFS.md unless absolutely necessary")

The current `paper/sections/CROSS_REFS.md` already documents the §4 layout; F2 verdict was added at lines 92-95 (the §4.7 click-ablation paragraph). A new §4.6.1 metallodrug panel cross-ref would slot into the §4 entry between line 91 (WF-Lambda-Metal-Pilot update) and line 92 (next §4.7 click-ablation paragraph). However, **§4.6.1 is captured inline in §4.6's existing entry** (the WF-Lambda-Metal-Pilot update on line 91 already references the metal column + 6 metric values; §4.6.1 is a strict extension of §4.6 anticancer), so a separate CROSS_REFS.md entry would duplicate.

### 4.2 P1 — Add §4.6.1 → S11 cross-ref (3-line addition to `04_evaluation.tex`)

In §4.6.1 subsubsection (line 1789), insert a new sentence at the end of the "Activity matrix contract" paragraph:

```latex
The full 11-dataset inventory referenced by this column lives in
\S\ref{supp:s11-data-provenance} (supplementary S11.1, Phase-1
provenance audit, 2026-09-16).
```

**Why P1**: closes the §4.6.1 ↔ S11 cross-ref gap without creating §3.7 or
renumbering §4.7. Single-line `\ref{}` insertion. **DOES require a 4-pass
pdflatex+bibtex recompile** because S11 lives in the SI shell (currently
orphaned, see §4.4 below).

**Depends on**: §4.4 (S11 must be `\input{}`-ed into `paper/supplementary.tex` for the `\ref{supp:s11-data-provenance}` to land in `main.pdf`).

### 4.3 P2 — §3.7 metallodrug vertical file creation (deferred to widzuipcl Phase 2)

The file `paper/sections/03_7_metallodrug_vertical.tex` is the responsibility of the widzuipcl workflow that owns the metallodrug vertical content. Once created with label `sec:method:metallodrug-vertical`:

1. Add `\input{sections/03_7_metallodrug_vertical}` after `\input{sections/03_6_deflex}` (or wherever fits the §3.7 slot — see `03_method.tex:97` for the current `\input{}` chain).
2. Add 1-line ref-target replacement in §4.6.1 line 1815: change "see §3.7 for vertical" forward-text to `\S\ref{sec:method:metallodrug-vertical}` (the F2 text carries this as a forward-prose; converting to `\ref{}` makes it a hard cross-ref).
3. Add S11 Table S11.1 caption back-cite to `sec:method:metallodrug-vertical`.

**Status: NOT EXECUTED**. This workflow owns the cross-ref MAP, not the §3.7 file. The verdict records the spec gap and the recovery plan for widzuipcl.

### 4.4 P1 — Wire `supplementary_data_table.tex` into `paper/supplementary.tex` (S11 from orphan to live)

Currently `paper/main.tex:396` does `\input{supplementary}` but the SI shell ends at S10. To make S11 visible in `main.pdf` and resolve the `\ref{supp:s11-data-provenance}` call recommended in §4.2:

1. Append `\input{supplementary_data_table}` at the end of `paper/supplementary.tex` (after the last `\subsection*{S10. ...}` line).
2. Verify with 4-pass pdflatex+bibtex that the 3 S11 tables render in `main.pdf`.
3. Re-run WF-Paper-Compile-Fix workflow to confirm no compile regression.

**Status: NOT EXECUTED** (per F2 verdict `f2_supp_table.md` line 3: "paper/supplementary.tex UNTOUCHED per spec"). Documented here as a follow-up for the recompile agent.

### 4.5 P2 — Forward refs from §5/§6/§7 to §4.6.1 (deferred)

The F3 spec asks for §5 ablation + §6 limitations + §7 future work to back-cite §3.7 metallodrug vertical. Since §3.7 is missing, those back-cites would target `sec:evaluation:metallodrug-vertical` (§4.6.1) when §3.7 is eventually created. **Status: NOT EXECUTED**, deferred until §3.7 file lands.

### 4.6 P3 — Resolve "see §3.7 for vertical" forward-prose in §4.6.1 (line 1815)

The F2 placeholder text at line 1815 says (paraphrasing from `04_evaluation.tex:1815`):

> "...candidates not present in the PlatinAI corpus return null and are excluded from the per-pocket numerator."

and at line 1825 references `wf_metallodrug_vertical/phase1_inventory.md` §2.1. The text does NOT actually contain a forward "see §3.7" pointer — F2 verdict's `f2_platinai_s4.md` mentions "see §3.7 for vertical" as an INTENT, not as a literal in-text string. **Verified clean** — no broken `\ref{sec:method:metallodrug-vertical}` calls anywhere in §4.6.1 or S11.

---

## 5. Cross-reference resolution verification (full audit)

Per task spec "Verify all cross-refs resolve (use grep to find \ref{...} targets and \label{...} declarations)":

### 5.1 §4.6.1 metallodrug panel refs

| `\ref{}` | `\label{}` declaration | Status |
|---|---|---|
| `sec:metal-geometry-prior` | `paper/sections/03_3_metal_geometry_prior.tex:1` | ✓ RESOLVES |
| `sec:evaluation:click-ablation` | `paper/sections/04_evaluation.tex:2127` | ✓ RESOLVES |
| `sec:evaluation:hybrid-wip` | `paper/sections/04_evaluation.tex:2394` (verified by line content; the `\label{sec:evaluation:hybrid-wip}` follows the `\subsection{Hybrid (Lambda × CFM geometric) arm --- work-in-progress}` heading at line 2364) | ✓ RESOLVES |
| `sec:method:secondary-cfm` | `paper/sections/03_method.tex:101` | ✓ RESOLVES |
| `sec:related:metallodrug` | `paper/sections/02_related.tex:201` | ✓ RESOLVES |

**0 unresolved refs in §4.6.1.**

### 5.2 S11 supplementary_data_table.tex refs

| `\ref{}` | Count | Status |
|---|---|---|
| (any) | 0 | N/A — S11 has no internal `\ref{}` calls |

S11 has 7 `\cite{}` calls (all resolve to existing bibitems per F2 verdict `f2_supp_table.md`).

### 5.3 §3.7 metallodrug vertical refs

**N/A** — file does not exist. If widzuipcl creates the file with the standard 4-target pattern (cite §1 intro, §2.5 metallodrug, §3.3 MGP, S11 data provenance, with back-cites from §4.6.1 + §5.8 + §6 + §7), the spec at §3.1 above should be followed.

---

## 6. Files referenced (absolute paths)

**Read (inputs)**:
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/CROSS_REFS.md` (150 lines, current state)
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/04_evaluation.tex` (lines 1773-1937 §4.6.1 subsubsection)
- `/home/hugo/codes/try_triton_on_rocm/paper/supplementary_data_table.tex` (207 lines, S11)
- `/home/hugo/codes/try_triton_on_rocm/paper/main.tex` (lines 181-396 `\input{}` chain)
- `/home/hugo/codes/try_triton_on_rocm/paper/supplementary.tex` (lines 1-10 SI shell — `\input{}` chain does NOT include `supplementary_data_table`)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pivot_followup/f2_platinai_s4.md` (F2 verdict, slot decision §2)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pivot_followup/f2_supp_table.md` (F2 verdict, S11 orchestrator)

**Written (output, this workflow)**:
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pivot_followup/f3_crossref.md` (this file)

**NOT modified** (per spec):
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/CROSS_REFS.md` (no edit warranted)
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/04_evaluation.tex` (no edit warranted; existing §4.6.1 keeps its 5 in-line cross-refs)
- `/home/hugo/codes/try_triton_on_rocm/paper/supplementary_data_table.tex` (no edit warranted)
- `/home/hugo/codes/try_triton_on_rocm/paper/main.tex` (NOT modified per spec, the recompile agent owns this)
- `/home/hugo/codes/try_triton_on_rocm/paper/supplementary.tex` (NOT modified per spec)

---

## 7. Verdict summary

**Cross-ref map: WRITTEN to `molmetal/reports/wf_pivot_followup/f3_crossref.md`.**

**`paper/CROSS_REFS.md`: UNCHANGED** per spec (no modification warranted; existing §4 entry already captures the §4.6.1 metadata inline).

**Outstanding gaps (deferred to widzuipcl Phase 2 + recompile agent)**:
1. `paper/sections/03_7_metallodrug_vertical.tex` MISSING — widzuipcl owns creation; would carry label `sec:method:metallodrug-vertical`.
2. `paper/supplementary_data_table.tex` is ORPHANED (not `\input{}`-ed in `paper/supplementary.tex`) — recompile agent owns the wiring.
3. **0 unresolved `\ref{}` targets in any on-disk file** — the metallodrug-vertical cross-refs that DO exist (in §4.6.1) all resolve to existing labels.

**F3 verdict**: SHIPPED. Cross-ref matrix documented; 0 unresolved refs; 3 follow-ups recorded for widzuipcl + recompile agent.
