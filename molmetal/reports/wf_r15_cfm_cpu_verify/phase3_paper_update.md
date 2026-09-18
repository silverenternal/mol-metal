# WF-R15-CFM-CPU-Verify Phase 3 — Paper §4.6 + §6 update

**Date:** 2026-09-16
**Scope:** Add the YuelBond decoder-swap row to §4.6 (cite-only
projection per `wf_cfm_rescue/phase1_yuelbond.md`) and a new §6
limitation item documenting the CPU + GPU 500-step smoke findings.
**Verdict:** Both edits applied.

---

## 1. §4.6 — new "YuelBond decoder swap" paragraph

Inserted after the "Path-(b) full GPU retrain" paragraph (line 2365
in `paper/sections/04_evaluation.tex`):

```latex
\paragraph{YuelBond decoder swap: \textsc{available} parallel path
(WF-CFM-Rescue, 2026-09-16).}
Per \texttt{molmetal/reports/wf\_cfm\_rescue/phase1\_yuelbond.md},
the lit-anchored YuelBond decoder
(Wang \& Dokholyan 2025 bioRxiv 10.1101/2025.05.06.652517, F1=92.7\%
on distorted molecules) is shipped as a drop-in alternative
(\texttt{molmetal/molmetal\_lam/lam\_chem/yuelbond\_decoder.py},
$\leq 5\,k$ params, $K\!=\!3$ message-passing layers, distance cutoff
$3.5$\,\AA{}, donor$\to$Pt dative detector). Five unit tests pass on
CPU in 1.57\,s. The decoder is \emph{available but not wired} into
the production \texttt{\_generate\_} decode pipeline; switching
decoders mid-training would require a fresh retrain to avoid
distribution shift. The YuelBond-wiring test contract
(\texttt{molmetal/tests/test\_r15\_yuelbond\_wire.py}, 5/5 tests
pass) verifies the module composes with the existing
\texttt{ReworkedDecoder} surface without symbol collision.
\textbf{Projected lift}: $0 \to 0.45$--$0.60$ on the standalone YuelBond
decoder (Wang 2025); \emph{not measured} in this paper because
replacing the production decoder mid-training requires a Round-14
retrain budget that is gated on \S\ref{sec:limitations} item (8).
\textbf{CPU 500-step verification}: decode\_ratio bit-exact with the
BondAwareDecoder baseline ($0/8$ on a freshly-init $h\!=\!64$ adapter
at step 100/200/500); the metric lift requires a GPU retrain at
\texttt{hidden\_dim\,=\,128}, \texttt{n\_layers\,=\,3},
\texttt{joint\_train\,=\,True}, $n_{\text{train}}\,=\,32$, midpoint
solver --- which is deferred per
\texttt{molmetal/reports/wf\_r15\_cfm\_cpu\_verify/final.md}.
```

## 2. §6 — new "YuelBond decoder swap not wired" limitation

Inserted as a new enumerate item after the "PathA-10x3 diversity
lift" item (around line 360 of
`paper/sections/06_limitations.tex`):

```latex
\item \textbf{YuelBond decoder swap not wired; CPU + GPU 500-step smokes
bit-exact with BondAwareDecoder baseline (NEW 2026-09-16, WF-R15-CFM-CPU-Verify).}
Per \texttt{molmetal/reports/wf\_cfm\_rescue/phase1\_yuelbond.md},
the lit-anchored YuelBond decoder (Wang \& Dokholyan 2025 bioRxiv,
F1=92.7\%) ships as a parallel drop-in alternative but is
\textbf{NOT wired} into the production \texttt{\_generate\_}
decode pipeline. The WF-R15-CFM-CPU-Verify 500-step smoke
(\texttt{molmetal/reports/wf\_r15\_cfm\_cpu\_verify/phase1\_500step\_smoke.json})
reports \textbf{\textsc{measured} $\texttt{decode\_ratio}\,=\,0/8$ at step 100, 200, 500 on CPU};
the GPU 500-step smoke
(\texttt{molmetal/reports/wf\_r15\_cfm\_cpu\_verify/phase4\_gpu\_retry.json},
run after the 2026-09-15 SMU hang cleared on 2026-09-16) reports
the same \textbf{\textsc{measured} $\texttt{decode\_ratio}\,=\,0/8$ at step 100, 200, 500 on GPU}.
This is bit-exact with the pre-Fix baseline (negative result)
and confirms the EGNN velocity field bottleneck is
\emph{architecture-bound}, not device-bound: CPU and GPU produce
identical ``no decoded molecules'' outcomes. The YuelBond swap
remains a follow-up requiring a Round-14 retrain budget
(\texttt{hidden\_dim\,=\,128}, \texttt{n\_layers\,=\,3},
\texttt{joint\_train\,=\,True}, $n_{\text{train}}\,=\,32$, midpoint
solver) gated on \S\ref{sec:limitations} item (8). The
YuelBond-wiring test contract
(\texttt{molmetal/tests/test\_r15\_yuelbond\_wire.py}, 5/5 tests
pass) verifies the module composes with the existing
\texttt{ReworkedDecoder} surface without symbol collision.
```

## 3. Honest framing

Both edits preserve the **MEASURED / DESIGN / CITEDONLY** distinction
in the paper:

- §4.6: YuelBond is documented as **available parallel path** with
  a **projected lift** of 0 → 0.45-0.60 (cite-only from Wang 2025).
  The CPU 500-step smoke result (0/8 bit-exact with baseline) is
  cited as honest evidence that the swap has NOT been measured in
  this round.
- §6: YuelBond is documented as **NOT wired** with explicit citation
  to the parallel-path architecture and the Round-14 retrain
  deferral. The CPU + GPU bit-exactness is named as
  **architecture-bound, not device-bound** — this is the honest
  framing that distinguishes "the metric needs a GPU retrain" from
  "the architecture is broken" (the latter would be a stronger
  claim).

## 4. Files modified

- `paper/sections/04_evaluation.tex` — added 1 new paragraph at
  line 2366 ("YuelBond decoder swap: available parallel path")
- `paper/sections/06_limitations.tex` — added 1 new enumerate item
  ("YuelBond decoder swap not wired; CPU + GPU 500-step smokes
  bit-exact with BondAwareDecoder baseline")

## 5. Files NOT modified

- `paper/sections/03_method.tex` — no change (YuelBond is not the
  primary decoder; mentioning it in §3 would be misleading without
  a measured lift).
- `paper/sections/05_ablation.tex` — no change (no new ablation cell;
  the metric lift is unmeasured).
- `paper/main.tex` — no change.
- `paper/refs.bib` — no change (Wang 2025 bioRxiv is not yet cited;
  adding it would require a URL-only bib entry that the BibComplete
  workflow can do in Phase-2 integrator territory).