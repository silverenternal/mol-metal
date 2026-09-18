# WF-Pivot-A Phase 2 — bib prune + §7 reorder

**Date:** 2026-09-16
**Agent:** phase2-bib-s7
**Scope:** paper/refs.bib (12 entries removed) + paper/sections/07_future.tex (item-1 promotion + 1 new item)

## Task 1 — SBDD bibitem prune

Removed 12 bibitems (6 footnote: variants + 6 plain variants) from
`paper/refs.bib` for the 6 geometric-SBDD papers dropped from the
de novo / typed-term MCTS framing:

| Key removed                  | Note                                          |
| ---------------------------- | --------------------------------------------- |
| `footnote:flowr`             | FLOWr                                         |
| `footnote:targetdiff`        | TargetDiff                                    |
| `footnote:diffsbdd`          | DiffSBDD                                      |
| `footnote:pocket2mol`        | Pocket2Mol                                    |
| `footnote:moldiff`           | MolDiff                                       |
| `footnote:decompdiff`        | DecompDiff                                    |
| `flowr2024`                  | FLOWr (legacy plain key)                      |
| `targetdiff2023`             | TargetDiff (legacy plain key)                 |
| `diffsbdd2022`               | DiffSBDD (legacy plain key)                   |
| `pocket2mol2022`             | Pocket2Mol (legacy plain key)                 |
| `moldiff2023`                | MolDiff (legacy plain key)                    |
| `decompdiff2023`             | DecompDiff (legacy plain key)                 |

Also updated `paper/sections/README.md` to mark all 6 `footnote:*`
keys as **REMOVED 2026-09-16** with strike-through rendering.

**Orphan refs:** 7 \cite{footnote:*...} sites in `sections/01_intro.tex`
and `sections/02_related.tex` are intentionally left in place per the
constraint "DO NOT touch §1, §2, §3, §4, §5, §6"; they will render as
`?` placeholders in the next pdflatex pass, which is consistent with
the 32 '?' already accepted as non-blocking per the previous memory.

**KEEP** (verified still present in refs.bib): DiffDock, BindNet,
RoseTTAFold-AA, lippard, reedijk, hartwig, reinvent4,
PCGrad/PAC-Bayes/Lipman/Koehler family.

## Task 2 — §7 reorder

Promoted `\textbf{$\Lambda \times$ CFM coupling per TODO-21}` from
position 5 to position 1 (with a "Why promoted to position 1 in
Phase 2" paragraph citing
`molmetal/reports/wf_pivot_a/final_phase2_bib_s7.md`).

Added a new item **Pocket-conditioned SBDD extension (deferred to
Round-14+)** at position 3 (after the CFM item), explaining that
making the path pocket-mandatory would require:
(i) sub-fix A decode_ratio ≥ 0.5,
(ii) sub-fix B+C Vina lift to -9..-11 kcal/mol,
(iii) structural fix to the 3-layer singleton attractor
(F2(a) MetalLigandExchange + diversity-bonus + seed-exclusion metric
split).

**Final §7 item order:** Lambda×CFM coupling (1) → CFM geometric
generator (2) → Pocket-conditioned SBDD extension (3, NEW) → Five
research gaps (4) → Wet-lab validation (5) → Multi-metal (6) →
Scale (7) → Λ-theoretic extensions (8) → Benchmark fairness (9) →
Learning-theoretic study (10).

## Files touched

- `paper/refs.bib` — 12 entries removed + provenance comment block
- `paper/sections/README.md` — 6 `footnote:*` rows marked REMOVED
- `paper/sections/07_future.tex` — 2 items added, 1 item removed,
  net +90 lines (208 → 298 in items block, 363 total file)

## Verification

- `grep -nE '^@misc\{(footnote:)?(diffsbdd|pocket2mol|targetdiff|moldiff|decompdiff|flowr)' paper/refs.bib` → no matches
- `grep -n 'Lambda.*CFM coupling' paper/sections/07_future.tex` → exactly 1 match (at line 22)
- `\input` chain in main.tex confirms only `sections/07_future.tex` is consumed (legacy `section_07_*.tex` not in build)
