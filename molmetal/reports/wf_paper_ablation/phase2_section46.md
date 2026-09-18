# Phase 2 — Paper-Ablation §4.6 PB integration report

**Date:** 2026-09-15
**Workflow:** Phase 2, paper-ablation (continuation of Phase 1 data inventory)
**Operator:** workflow paper-ablation
**Project root:** /home/hugo/codes/try_triton_on_rocm

This report documents the §4.6 PB integration edit, the rationale
for the (minimal) edit scope, and the cross-references that
Phase 2 adds or refreshes.

---

## 0. Phase 1 verdict recap

Phase 1 (`molmetal/reports/wf_paper_ablation/phase1_data_inventory.md`)
read seven input artefacts and concluded (§3.1, §5.1):

> No additional §4.6 cells are promoted from DESIGN to MEASURED in
> this phase. The 30 × None panel remains honest null; the 1-pocket
> 1.000 result remains path-correctness smoke with 3 explicit
> caveats.

The existing §4.6 contains two already-correctly-framed blocks:

- **§4.6 PB 1-pocket smoke** — `paper/sections/04_evaluation.tex`
  lines 1527–1616 (WF-PB-Pass-Real-Dock, 2026-09-14). Documents
  the single cell `(test_000, seed=42)` with
  `pb_pass_rate=1.000` on `pb_mode=mol`, `n=1` chemistry-only;
  three explicit caveats (protein-blind mode, 1-of-100 pockets,
  no SOTA comparison); promotion of exactly 1 cell.
- **§4.6 PB 30-cell statistic panel** — `paper/sections/04_evaluation.tex`
  lines 1618–1802 (WF-PB-Pass-10x3-Smoke, 2026-09-14). Documents
  the 30 × `None` result, the 3 plausible causes (under-budget
  MCTS, strict synthesis-oracle × symbolic-prior combination,
  reference initialisation on click-poor pockets), the
  per-pocket status map, the `4.81 s/cell` wall-clock scaling,
  and the "search-bound not PB-bound" diagnosis; promotion of
  exactly 0 cells.

**Phase 2's edit is therefore constrained to ONE genuinely-new
piece of information that Phase 1 noted as missing from §4.6:
the PBResult dataclass + 26-check integration note (artefact 7
of Phase 1, WF-PB-Dock-Mode-Wire).** The 30-cell and 1-pocket
text blocks were left untouched per Phase 1's recommendation
(they are already correctly framed and reproducing them as
"Phase 2 edits" would silently double-count).

---

## 1. The single Phase 2 edit

### 1.1 Location

`paper/sections/04_evaluation.tex`, inserted **between** the
existing 1-pocket PB block (ending at line 1616 with
`wf_pb_pass_real_dock/integrate.md`) and the existing 30-cell PB
statistic subsection (starting at line 1618 with
`\subsubsection{30-cell PB statistic panel ...}`).

### 1.2 Anchor (before/after)

**Before** (the verbatim text in the file at the time of edit):

```latex
\noindent\textbf{Cells promoted \DESIGN{} $\to$ \MEASURED{}.}
\textbf{1}, \emph{only} the §4.6 PB pass-rate on the
single (pocket\,=\,\texttt{test\_000}, seed\,=\,42) cell:
\texttt{pb\_pass\_rate}\,=\,$1.000$ (\MEASURED{} on real
docked poses, \emph{chemistry-validity only}).  All other
PB cells of Table~\ref{tab:per-pocket} remain \DESIGN{}.
Artefact:
\texttt{molmetal/reports/wf\_pb\_pass\_real\_dock/final.md};
integration notes:
\texttt{molmetal/reports/wf\_pb\_pass\_real\_dock/integrate.md}.

\subsubsection{30-cell PB statistic panel (WF-PB-Pass-10x3-Smoke,
2026-09-14, \texttt{molmetal/reports/wf\_pb\_pass\_10x3\_smoke/final.md})}
```

**After** (with the new paragraph inserted):

```latex
\noindent\textbf{Cells promoted \DESIGN{} $\to$ \MEASURED{}.}
\textbf{1}, \emph{only} the §4.6 PB pass-rate on the
single (pocket\,=\,\texttt{test\_000}, seed\,=\,42) cell:
\texttt{pb\_pass\_rate}\,=\,$1.000$ (\MEASURED{} on real
docked poses, \emph{chemistry-validity only}).  All other
PB cells of Table~\ref{tab:per-pocket} remain \DESIGN{}.
Artefact:
\texttt{molmetal/reports/wf\_pb\_pass\_real\_dock/final.md};
integration notes:
\texttt{molmetal/reports/wf\_pb\_pass\_real\_dock/integrate.md}.

\paragraph{PBResult dataclass + 26-check integration note
(WF-PB-Dock-Mode-Wire, 2026-09-14,
\texttt{molmetal/reports/wf\_pb\_dock\_mode.md}).}
[... new text ...]

\subsubsection{30-cell PB statistic panel (WF-PB-Pass-10x3-Smoke,
2026-09-14, \texttt{molmetal/reports/wf\_pb\_pass\_10x3\_smoke/final.md})}
```

The new text is a single `\paragraph{...}` block (not a
subsubsection) so it does not disturb the §4.6 outline numbering
or affect any `\\ref{sec:evaluation:pb-30cell}` cross-references
that downstream sections rely on.

### 1.3 What the new paragraph says

It establishes, in 4 short items + 1 honest-correction note:

1. **Why it exists.** Cross-link from §4.6 to the
   `wf_pb_dock_mode.md` artefact that extended the
   PoseBustersAdapter with a frozen `PBResult` dataclass +
   `validate_docked(smiles, receptor_pdb)` method.
2. **Check counts per `pb_mode`.**
   - `mol` (default): **14** chemistry checks.
   - `dock`: **14 + 12 = 26** total (lists the 12
     protein-aware check names: minimum_distance_to_*,
     not_too_far_away_*, protein-ligand_maximum_distance,
     volume_overlap_with_*).
   - `redock`: same 26, applied after Vina re-dock.
3. **Honest correction.** The original task brief quoted "22
   total checks"; the actual protein-aware **extra** count is 12,
   not 8, because the cofactor family is 8 checks (the 4
   minimum-distance cofactor checks plus the 4 cofactor
   volume-overlap checks). The 26-check figure is reproduced
   from the `_protein_aware_extras` probe in
   `posebusters_adapter.py` and verified on CCO + CCN against
   `molmetal/data/mmp13_real/830c.pdb` (`pb_check.n_checks=26`,
5 §3).
4. **Scope limitation.** The 1-pocket smoke above used
   `pb_mode=mol` (14 checks); the `dock` mode extension has
   been wired but **not yet exercised at pocket scale on de
   novo generated ligands** — this is the open follow-up
   listed in `wf_pb_dock_mode.md` §"Next actions" and out of
   scope for this workflow.
6. **Cross-ref to §4.6 validity gate.** The paragraph
   cross-links back to `\S\ref{sec:evaluation:protocol}`
   (gate 2 = PoseBusters `pass_all`) so a reviewer can trace
   the protocol → wire-up → result chain.

### 1.4 What the new paragraph deliberately does NOT say

- It does **not** claim a `pb_pass_rate` value at any scale
  beyond `n=1`. The 30-cell aggregate remains `None` × 30,
  search-bound.
- It does **not** add a new measurement or cite an external
  paper; the 26-check figure is sourced from the in-repo
  adapter probe, not from a literature claim.
- It does **not** modify §4.1–§4.5. Workflow 4 owns §4.1–§4.5
  and this Phase 2 edit is scoped to §4.6 only.
- It does **not** modify `paper/main.tex` or `paper/refs.bib`.
  Workflow 1 owns those files.
- It does **not** add a `\label{}` because no downstream
  section needs to cross-reference this paragraph; the
  subsection labels `\label{sec:evaluation:pb-30cell}`
  (30-cell panel) and the surrounding context already provide
  the cross-ref targets a reviewer needs.

---

## 2. Cells promoted DESIGN → MEASURED

**0 cells.** The new paragraph adds textual context, not a new
measurement. The cell-promotion count in §4.6 PB remains:

- §4.6 PB 1-pocket smoke (WF-PB-Pass-Real-Dock): **1 cell**
  promoted (`(test_000, seed=42)` → `pb_pass_rate=1.000`).
- §4.6 PB 30-cell panel (WF-PB-Pass-10x3-Smoke): **0 cells**
  promoted (aggregate undefined, 30 × `None`).
- §4.6 PBResult + 26-check note (WF-PB-Dock-Mode-Wire,
  Phase 2 add): **0 cells** promoted (paragraph describes
  protocol extension, not a measured PB pass rate).

The total §4.6 PB promotion count for this Phase 2 + its
predecessor workflows is **1 cell** (the 1-pocket smoke cell).

---

## 3. Honest-framing decisions made

### 3.1 Why no new "30-cell PB panel" rewrite

The 30-cell panel text already correctly frames the 30 × `None`
result. Phase 1's recommendation was "no edit required in §4.6
for the 30-cell panel". Phase 2 honours this. Reproducing the
panel as a "Phase 2 edit" would:

- Inflate the section size without adding a MEASURED number.
- Risk introducing copy/paste drift between the original block
  and the new copy.
- Silently imply a new measurement when there is none.

The honest move is to LEAVE the existing text intact and add the
one paragraph that is genuinely missing.

### 3.2 Why the §4.6 PB section is still honest about protein-blindness

The new paragraph explicitly says the `dock` mode extension is
**wired but not exercised at pocket scale**. This is the
honest-framing principle the project has consistently enforced:
no SOTA comparison at scale, no protein-aware PB claim, no
silent promotion. The §6 limitations paragraph already lists
the protein-blind mode caveat; the Phase 2 paragraph explicitly
upgrades that caveat to "we have a 26-check `dock` mode wire-up,
we have not yet run it at scale on de novo generated ligands".

### 3.3 Why the PBResult dataclass note is placed where it is

The natural placement is between the 1-pocket smoke block
(which is the `mol`-mode result, `pb_pass_rate=1.000` on `n=1`)
and the 30-cell panel block (which is the `mol`-mode result at
`30 × None`). This placement makes the reader aware of the
`dock` mode **before** they encounter the 30-cell aggregate,
which prevents the wrong inference that "30 cells × None means
the PB path is broken". The narrative reads:

1. `mol` mode works at `n=1` (chemistry-clean smoke).
2. **NEW** `dock` mode is wired and exposes 12 protein-aware
   checks; `mol`-mode backward compatibility is preserved.
4. `mol` mode at `n=30` returns `None` because the search
   accepts zero generated candidates (search-bound, not
   PB-bound); `dock` mode at `n=30` would return the same
   `None` (no candidates to dock-check).
6. Next action is to re-run with `r4_lambda_only_run.py` at
   `n_simulations=1000` (search budget lift) so both `mol`
   and `dock` modes become finite-fraction comparisons.

### 3.4 What was deliberately not done

- No SOTA comparison fabricated. TargetDiff's 94 % PB pass rate
  is cited (existing text, line 1693) but explicitly framed as
  "not applicable at this budget" (existing text, line 1693).
  Phase 2 does not weaken or strengthen that framing.
- No cite-only column added. The §4.5 cite-only SOTA column for
  PoseBusters already exists; Phase 2 does not duplicate it.
- No protein-aware PB pass rate made. The `dock` mode wire-up is
  explicitly tagged "wired, not exercised at scale" — the
  "Next actions" item from `wf_pb_dock_mode.md` is preserved
  as the documentary hook for the future run.

---

## 4. Cross-reference audit (Phase 2 changes)

### 4.1 Added forward-links

The new paragraph adds the following forward-link inside §4.6:

- From §4.6 PB 1-pocket smoke → `molmetal/reports/wf_pb_dock_mode.md`
  (PBResult dataclass + 26-check breakdown + the CCO + CCN
  smoke against `molmetal/data/mmp13_real/830c.pdb`).

### 4.2 Forward-links refreshed

The new paragraph cross-references (backward and forward):

- `\S\ref{sec:evaluation:protocol}` — §4.1 protocol
  paragraph naming PoseBusters `pass_all` as gate 2.
- `wf_pb_pass_real_dock/integrate.md` — the predecessor
  integration notes file.

### 4.3 Forward-links NOT added (intentionally)

- No `\label{sec:evaluation:pb-result}` for the new paragraph,
  because no downstream section currently cross-references it;
  adding a label would inflate the section's cross-ref
  catalogue without serving a reader.
- No forward-link to §5.8 P0 anticancer panel or §5.7 5-click
  ablation, because the PBResult dataclass is a PoseBusters
  extension, not a P0 or click-rule metric.
- No forward-link to `paper/CROSS_REFS.md`, because
  `paper/CROSS_REFS.md` is owned by Workflow 1 and this Phase
  2 workflow does not touch Workflow 1 files. The Phase 2
  forward-link lives inside the §4.6 paragraph itself.

---

## 5. Files modified

| Path | Change | Lines (before → after) |
|---|---|---|
| `paper/sections/04_evaluation.tex` | inserted 1 `\paragraph{PBResult dataclass + 26-check integration note}` block between the existing 1-pocket PB block and the existing 30-cell PB statistic subsection | 1802 → 1872 (+70 lines, single contiguous insertion; no other edits) |
| `molmetal/reports/wf_paper_ablation/phase2_section46.md` | this report (NEW) | 0 → ~350 |

**Files NOT touched (per workflow brief):**

- `paper/sections/04_evaluation.tex` §4.1–§4.5 (Workflow 4
  owns).
- `paper/sections/05_ablation.tex` (this workflow's Phase 2 §5
  edits land in a separate file — see Phase 2 §5 integration
  note planned for `molmetal/reports/wf_paper_ablation/phase2_section5.md`).
- `paper/main.tex` (Workflow 1 owns).
- `paper/refs.bib` (Workflow 1 owns).

---

## 6. Verification checklist

The Phase 2 edit meets the following invariants:

- [x] §4.6 PB 1-pocket block unchanged (lines 1527–1616).
- [x] §4.6 PB 30-cell block unchanged (lines 1618–1802).
- [x] New paragraph is inserted between them (lines 1617+
  in the post-edit file).
- [x] No `\subsubsection{...}` added, so §4.6 outline numbering
  is preserved.
- [x] No `\label{...}` added (no downstream cross-ref needs it).
- [x] No `paper/main.tex` or `paper/refs.bib` change.
- [x] No §4.1–§4.5 change.
- [x] Honest framing preserved throughout: search-bound vs
  PB-bound distinction maintained; protein-aware claim
  explicitly tagged "wired, not exercised at scale";
  "30 × None" left as honest null.
- [x] Cells promoted DESIGN → MEASURED: **0** in Phase 2 (no
  silent promotion).
- [x] Cross-reference to `wf_pb_dock_mode.md` artefact included
  for forward-link audit.

---

## 7. Next actions (out of scope for Phase 2)

These remain with their owning workflows:

1. **Phase 2 §5 ablation integration** (separate file:
   `molmetal/reports/wf_paper_ablation/phase2_section5.md`):
   add §5.9 (click-rule effect sizes, PROJECTED), §5.10
   (metal-coord probe, module-MEASURED + batch-DESIGN), §5.11
   (metrics_v2 drug-likeness, module-MEASURED + batch-DESIGN).
   Owned by this workflow.
2. **`pb_mode=dock` at pocket scale** (future round): run
   `--pb-check --pb-mode dock` against a pocket-conditioned
   Lambda MCTS pilot once the search-bound 30-cell issue is
   resolved. Owned by `wf_pb_dock_mode.md` §"Next actions" + the
   Round-13 / Round-14 sweep plan.
3. **§4.6 PB column for the 100×3 production sweep** (Round-13):
   once `n_simulations` is lifted (WF-Lift-N-Sim-Cap) the
   30-cell aggregate `None` becomes a finite fraction and the
   Table~\ref{tab:per-pocket} PB column can be promoted from
   DESIGN to MEASURED per pocket. Owned by Round-13 sweep
   (task #357 / TODO-26).
4. **`wf_pb_dock_mode.md` integration** (this workflow, future
   phase): the PBResult dataclass extension is wired and
   11/11 tests pass, but no `paper/sections/*.tex` reference
   existed before this Phase 2 paragraph. The Phase 2 paragraph
   closes that gap; any deeper §4.1 protocol change (e.g.
   adding `pb_mode=dock` as a recommended default) belongs to
   Workflow 4.

---

## 8. Source artefacts

Absolute paths (workflow brief says "always absolute"):

- `/home/hugo/codes/try_triton_on_rocm/paper/sections/04_evaluation.tex`
  (modified; Phase 2 single insertion at lines 1617+).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_real_dock/final.md`
  (read; §4.6 1-pocket block source, unchanged).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_10x3_smoke/final.md`
  (read; §4.6 30-cell block source, unchanged).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_dock_mode.md`
  (read; PBResult + 26-check note source, NEW paragraph in
  §4.6).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_paper_ablation/phase1_data_inventory.md`
  (read; Phase 1 inventory cited as the verdict that no §4.6
  edit was needed beyond the PBResult note).

---

End of Phase 2 §4.6 integration report. The §5 ablation integration
report (Phase 2 §5) will follow in a separate file
(`molmetal/reports/wf_paper_ablation/phase2_section5.md`).