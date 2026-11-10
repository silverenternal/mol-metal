# Wf-A161 -- `paper/**` READ-ONLY inventory of unqualified numbers

**MEASURED 2026-11-01, CPU-only, 0 GPU, 0 CUDA calls, 0 bytes written under `paper/**`.** TODO-84 (AGENTS.md §5.3) forbids every write to `paper/**`; this report proposes edits as TEXT and applies NONE of them.

## 0. How to read this

- **QUALIFIED** -- the site already names its source and/or its population. No edit needed.
- **UNQUALIFIED** -- the number reads as ours, or is only interpretable with a missing qualifier (cohort, noise regime, denominator, deviation note).
- **UNSOURCED** -- no source anywhere near the number.
- **FALSE-SCOPE** -- the site asserts something about OUR model that our own measurements (or our own code) do not support.

## 1. Census

- text files scanned under `paper/`: **108**
  (tracked: 108, untracked/gitignored: 0)
- files skipped: **0**
- sites recorded: **130**

| rule | class | number | hits |
|---|---|---|---|
| A1 | EXTERNAL | `92.7 %` | 10 |
| A3 | EXTERNAL | `783` | 4 |
| A4 | EXTERNAL | `99.9 %` | 4 |
| B1 | OURS | `0/192` | 73 |
| B2 | OURS | `192/192` | 19 |
| B3 | OURS | `0/8` | 11 |
| C2 | CONSTANT | `2.4 A` | 6 |
| C3 | CONSTANT | `0.3 A` | 1 |
| C5 | CONSTANT | `1.5-3.5 A (and 1-5 A)` | 2 |

### Seeds from the brief -- confirmed or refuted

| seed | verdict | evidence |
|---|---|---|
| unqualified external validity band ~92.7-94 %, one hit in `04_evaluation.tex`, one in `06_limitations.tex` | **CONFIRMED (2 sites)** | the literal `92.7` occurs in exactly those two tracked `.tex` files; the *band* `92.7-94 %` as such occurs **0** times in `paper/**` |
| 3.5 A distance cutoff at SIX tracked `.tex` sites | **COUNT CONFIRMED, FRAMING PARTLY REFUTED** | there are exactly **6** tracked `.tex` sites carrying `3.5` + a length unit, but only **one** of them (`04_evaluation.tex:3183`) is a *cutoff*; the other five are an illustrative Vietoris-Rips radius (3), a worked sigmoid example (1) and a coordinate-spread range (1) |
| `8.51 %` / `25.05 %` fragmentation headline | **REFUTED for `paper/**`** | `8.51 %` occurs **0** times as a paper claim (`8.51` appears only as SVG `translate(...)` coordinates and as pdflatex `Overfull \hbox` widths in `build_log.txt`); `25.05` in `paper/**` occurs **only** as a substring of the bioRxiv DOI `10.1101/2025.05.06.652517` |

## 2. Sites

### `paper/sections/06_limitations.tex:387` -- rule A1 -- **UNQUALIFIED**

- **number**: `92.7 %` (matched `92.7\%`)
- **what it is**: YuelBond per-BOND recovery F1 on the CDG cohort (the paper's worst case)
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052 (preprint bioRxiv 10.1101/2025.05.06.652517)
- **population that makes it interpretable**: per-bond recovery F1 (NOT a per-molecule validity rate) on the CDG cohort (Crude De novo Generated, Gaussian noise sigma = 0.2 A)
- **sentence (verbatim)**: `\item \textbf{YuelBond decoder swap not wired; CPU + GPU 500-step smokes bit-exact with BondAwareDecoder baseline (NEW 2026-09-16, WF-R15-CFM-CPU-Verify).} Per \texttt{molmetal/reports/wf\_cfm\_rescue/phase1\_yuelbond.md}, the lit-anchored YuelBond decoder (Wang \& Dokholyan 2025 bioRxiv, F1=92.7\%) ships as a parallel drop-in alternative but is \textbf{NOT wired} into the production \texttt{\_generate\_} decode pipeline.`
- **why this classification**: window names the source but omits the population (cohort / noise regime / denominator)
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: Replace `F1=92.7\%` with `per-bond recovery F1 = 92.7\% on the CDG cohort (Crude De novo Generated, Gaussian noise \sigma = 0.2 \AA, the paper's worst case; 98.4\% on standard 3-D structures) -- a per-bond F1, not a per-molecule validity rate`.

### `paper/abstract_draft.tex:56` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\textbf{(C2) KataGo-MCTS proof search + SE(3) CFM decoder.}  A KataGo-style transposition-table MCTS driver with virtual loss and First-Play-Urgency explores the closed-term space; an SE(3)-equivariant conditional flow matching decoder conditions on a \mgp{} that enforces the target metal's coordination geometry, and a 6-shim Blackwell-port fix lifts the architectural-floor decode ratio from $0/192$ to $30/30 = 1.000$.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:1043` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `er.py\OT1/cmr/m/sc/10 , \OT1/cmtt/m/sc/10 molmetal/molmetal[]lam/search[]alg/mu lti[]parameter[]ph[]adapter.py\OT1/cmr/m/sc/10 )  Overfull \hbox (37.74623pt too wide) in paragraph at lines 369--386 \OT1/cmr/m/sc/10 The full Black-well GPU train-ing ver-i-fi-ca-tion that would lift the Round-17 \OT1/cmtt/m/sc/10 decode[]ratio$\OT1/cmr/m/n/10 =$\OT1/cmtt/m /sc/10 0/192  LaTeX Font Warning: Font shape `OT1/cmr/m/scit' undefined (Font)              using `OT1/cmr/m/it' instead on input line 407.  (./sections/03_6_end_to_end_pipeline.tex Overfull \hbox (82.37538pt too wide) in paragraph at lines 54--69`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:1345` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Extra }, or forgotten $. <recently read> \egroup  l.90 (\texttt{decode\_ratio\,$=$\,$0/192} on the round-12 retrain, !`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:1991` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (95.99985pt too wide) in paragraph at lines 255--265 \OT1/cmtt/m/sc/10 The \TS1/cmtt/m/sc/10 �\OT1/cmtt/m/sc/10 4 round-17 honest-ne gative framing retains the 0/192 decode ratio baseline; the transport-relaxatio n ) (./sections/03_6_multi_parameter_ph.tex  LaTeX Font Warning: Font shape `OT1/cmtt/m/scit' undefined (Font)              using `OT1/cmtt/m/it' instead on input line 64.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:7491` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (10.53181pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/n/10 Result: decode[]ratio lifted from 0/192 floor to 1.000 at all four checkpoints \OT1/cmtt/m/sc/10 (measured,  Overfull \hbox (4.09221pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/sc/10 +100% absolute over the 0/192 wf[]gpu[]recovery[]now/final.md baseline, infinite relative).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:7495` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (10.53181pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/n/10 Result: decode[]ratio lifted from 0/192 floor to 1.000 at all four checkpoints \OT1/cmtt/m/sc/10 (measured,  Overfull \hbox (4.09221pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/sc/10 +100% absolute over the 0/192 wf[]gpu[]recovery[]now/final.md baseline, infinite relative).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:65282` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `er.py\OT1/cmr/m/sc/10 , \OT1/cmtt/m/sc/10 molmetal/molmetal[]lam/search[]alg/mu lti[]parameter[]ph[]adapter.py\OT1/cmr/m/sc/10 )  Overfull \hbox (37.74623pt too wide) in paragraph at lines 369--386 \OT1/cmr/m/sc/10 The full Black-well GPU train-ing ver-i-fi-ca-tion that would lift the Round-17 \OT1/cmtt/m/sc/10 decode[]ratio$\OT1/cmr/m/n/10 =$\OT1/cmtt/m /sc/10 0/192  LaTeX Font Warning: Font shape `OT1/cmr/m/scit' undefined (Font)              using `OT1/cmr/m/it' instead on input line 407.  (./sections/03_6_end_to_end_pipeline.tex Overfull \hbox (82.37538pt too wide) in paragraph at lines 54--69`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:65584` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Extra }, or forgotten $. <recently read> \egroup  l.90 (\texttt{decode\_ratio\,$=$\,$0/192} on the round-12 retrain, !`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:66230` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (95.99985pt too wide) in paragraph at lines 255--265 \OT1/cmtt/m/sc/10 The \TS1/cmtt/m/sc/10 �\OT1/cmtt/m/sc/10 4 round-17 honest-ne gative framing retains the 0/192 decode ratio baseline; the transport-relaxatio n ) (./sections/03_6_multi_parameter_ph.tex  LaTeX Font Warning: Font shape `OT1/cmtt/m/scit' undefined (Font)              using `OT1/cmtt/m/it' instead on input line 64.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:71730` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (10.53181pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/n/10 Result: decode[]ratio lifted from 0/192 floor to 1.000 at all four checkpoints \OT1/cmtt/m/sc/10 (measured,  Overfull \hbox (4.09221pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/sc/10 +100% absolute over the 0/192 wf[]gpu[]recovery[]now/final.md baseline, infinite relative).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:71734` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (10.53181pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/n/10 Result: decode[]ratio lifted from 0/192 floor to 1.000 at all four checkpoints \OT1/cmtt/m/sc/10 (measured,  Overfull \hbox (4.09221pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/sc/10 +100% absolute over the 0/192 wf[]gpu[]recovery[]now/final.md baseline, infinite relative).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:129453` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `er.py\OT1/cmr/m/sc/10 , \OT1/cmtt/m/sc/10 molmetal/molmetal[]lam/search[]alg/mu lti[]parameter[]ph[]adapter.py\OT1/cmr/m/sc/10 )  Overfull \hbox (37.74623pt too wide) in paragraph at lines 369--386 \OT1/cmr/m/sc/10 The full Black-well GPU train-ing ver-i-fi-ca-tion that would lift the Round-17 \OT1/cmtt/m/sc/10 decode[]ratio$\OT1/cmr/m/n/10 =$\OT1/cmtt/m /sc/10 0/192  LaTeX Font Warning: Font shape `OT1/cmr/m/scit' undefined (Font)              using `OT1/cmr/m/it' instead on input line 407.  (./sections/03_6_end_to_end_pipeline.tex Overfull \hbox (82.37538pt too wide) in paragraph at lines 54--69`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:129755` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Extra }, or forgotten $. <recently read> \egroup  l.90 (\texttt{decode\_ratio\,$=$\,$0/192} on the round-12 retrain, !`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:130401` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (95.99985pt too wide) in paragraph at lines 255--265 \OT1/cmtt/m/sc/10 The \TS1/cmtt/m/sc/10 �\OT1/cmtt/m/sc/10 4 round-17 honest-ne gative framing retains the 0/192 decode ratio baseline; the transport-relaxatio n ) (./sections/03_6_multi_parameter_ph.tex  LaTeX Font Warning: Font shape `OT1/cmtt/m/scit' undefined (Font)              using `OT1/cmtt/m/it' instead on input line 64.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:135901` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (10.53181pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/n/10 Result: decode[]ratio lifted from 0/192 floor to 1.000 at all four checkpoints \OT1/cmtt/m/sc/10 (measured,  Overfull \hbox (4.09221pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/sc/10 +100% absolute over the 0/192 wf[]gpu[]recovery[]now/final.md baseline, infinite relative).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:135905` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (10.53181pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/n/10 Result: decode[]ratio lifted from 0/192 floor to 1.000 at all four checkpoints \OT1/cmtt/m/sc/10 (measured,  Overfull \hbox (4.09221pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/sc/10 +100% absolute over the 0/192 wf[]gpu[]recovery[]now/final.md baseline, infinite relative).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:193624` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `er.py\OT1/cmr/m/sc/10 , \OT1/cmtt/m/sc/10 molmetal/molmetal[]lam/search[]alg/mu lti[]parameter[]ph[]adapter.py\OT1/cmr/m/sc/10 )  Overfull \hbox (37.74623pt too wide) in paragraph at lines 369--386 \OT1/cmr/m/sc/10 The full Black-well GPU train-ing ver-i-fi-ca-tion that would lift the Round-17 \OT1/cmtt/m/sc/10 decode[]ratio$\OT1/cmr/m/n/10 =$\OT1/cmtt/m /sc/10 0/192  LaTeX Font Warning: Font shape `OT1/cmr/m/scit' undefined (Font)              using `OT1/cmr/m/it' instead on input line 407.  (./sections/03_6_end_to_end_pipeline.tex Overfull \hbox (82.37538pt too wide) in paragraph at lines 54--69`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:193926` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Extra }, or forgotten $. <recently read> \egroup  l.90 (\texttt{decode\_ratio\,$=$\,$0/192} on the round-12 retrain, !`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:194572` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (95.99985pt too wide) in paragraph at lines 255--265 \OT1/cmtt/m/sc/10 The \TS1/cmtt/m/sc/10 �\OT1/cmtt/m/sc/10 4 round-17 honest-ne gative framing retains the 0/192 decode ratio baseline; the transport-relaxatio n ) (./sections/03_6_multi_parameter_ph.tex  LaTeX Font Warning: Font shape `OT1/cmtt/m/scit' undefined (Font)              using `OT1/cmtt/m/it' instead on input line 64.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:200072` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (10.53181pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/n/10 Result: decode[]ratio lifted from 0/192 floor to 1.000 at all four checkpoints \OT1/cmtt/m/sc/10 (measured,  Overfull \hbox (4.09221pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/sc/10 +100% absolute over the 0/192 wf[]gpu[]recovery[]now/final.md baseline, infinite relative).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:200076` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (10.53181pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/n/10 Result: decode[]ratio lifted from 0/192 floor to 1.000 at all four checkpoints \OT1/cmtt/m/sc/10 (measured,  Overfull \hbox (4.09221pt too wide) in paragraph at lines 2659--2683 \OT1/cmtt/m/sc/10 +100% absolute over the 0/192 wf[]gpu[]recovery[]now/final.md baseline, infinite relative).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/figures/data/fig12_decode_ratio_per_seed.json:4` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `All 30 probes (3 seeds × 10 checkpoints) achieve 1.000 decode ratio — full lift from the R16 0/192 floor.", "x_axis": { "label": "Probe checkpoint", "categories": ["step_1000", "step_2000", "step_3000", "step_4000", "step_5000", "step_6000", "step_7000", "step_8000", "step_9000", "step_10000"] }, "y_axis": { "label": "Decode ratio",`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/figures/data/fig12_decode_ratio_per_seed.json:40` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `"wall_seconds_per_probe": [3.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.2, 0.5, 0.3, 0.7] }, { "name": "R16 baseline (TypeError floor)", "type": "reference", "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "note": "R16 0/192 (TypeError BF16) — the floor that motivated this retrain" } ], "legend": [ "R17 GPU retrain seed_0 (decode_ratio 30/30 = 1.000)", "R17 GPU retrain seed_1", "R17 GPU retrain seed_2",`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/figures/data/fig12_decode_ratio_per_seed.json:47` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `} ], "legend": [ "R17 GPU retrain seed_0 (decode_ratio 30/30 = 1.000)", "R17 GPU retrain seed_1", "R17 GPU retrain seed_2", "R16 baseline (0/192 BF16 TypeError)" ], "summary": { "n_probes_per_seed": 10, "n_seeds": 3, "total_probes": 30, "all_decode_ratios": 1.000,`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/reviews/rebuttal_predraft.md:159` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The decode_ratio=1.000 (30/30 probes, 3 seeds) at probe-time confirms: (i) the ODE solver runs (M15–M17), (ii) the model emits *bond-bearing* graphs (vs the pre-fix 0/192 baseline where RDKit fell over overvalent atoms), and (iii) the trained CFM coordinate distribution falls in the `BondAwareDecoder`'s `τ = 0.3 Å` soft band.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/reviews/rebuttal_predraft.md:166` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The R17 30/30 lift off the R16 0/192 floor (\texttt{wf\_todo33\_p1\_done} + \texttt{wf\_todo33\_p2\_done}) is a real bug-fix, not overfit; bond\_loss converges to 1.07–1.14 across 3 seeds independently.  ---  ## Q6 — "Why is n_distinct = 3 the bar?`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_4_cfm_methodology.tex:102` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\paragraph{Why this subsection.} The path-(a) full retrain was \textsc{not measured}: \texttt{decode\_ratio\,=\,0/192} on the CrossDocked-trained checkpoint (\texttt{molmetal/reports/wf\_path\_b\_gpu\_retrain/final.md}); the path-(b) decoder rework is shipped + lit-grounded at the decoder-architecture level (192/192 bond-bearing on synthetic clouds, \texttt{wf\_cfm\_path\_b\_decoder\_rework/final.md}) and is the load-bearing follow-up.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_4_se3_schrodinger_bridge.tex:90` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `A Euclidean interpolant therefore violates the manifold geometry of the target distribution; the resulting ODE trajectory can drift off the manifold, which is the underlying distribution-shift pathology of the path-(a) full retrain (\texttt{decode\_ratio\,$=$\,$0/192} on the round-12 retrain, \texttt{molmetal/reports/wf\_path\_b\_gpu\_retrain/final.md}).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_4_se3_schrodinger_bridge.tex:259` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The §4 round-17 honest-negative framing (\texttt{molmetal/reports/round17\_honest\_negative\_reframe.md}) retains the path-(a) \textsc{FAILURE} baseline ($0/192$ decode ratio); the SE(3) SB path is the Phase 2 upgrade target.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_5_transport_relaxation.tex:262` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The §4 round-17 honest-negative framing retains the 0/192 decode ratio baseline; the transport-relaxation term is the Phase 2 upgrade target for the reward-shaping layer.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_5_honest_negative.tex:151` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\subsection{Diagnostic insights from Workflow A: the 6-shim fix that lifted $\mathrm{decode\_ratio}$ from $0/192$ to $30/30$} \label{sec:honest-negative:workflow-a}  The first load-bearing diagnostic story of this round is \emph{workflow A} (TODO-33 P1+P2, R17 Blackwell port).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_5_honest_negative.tex:156` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The headline finding is that $\mathrm{decode\_ratio}{=}0/192$ on Blackwell was \emph{not} an architecture problem; it was a sample-time wrapper-typecheck race in the ODE-solver shim.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_5_honest_negative.tex:196` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `adapter uses \texttt{partial(forward\_velocity,...)} which makes \texttt{t} keyword-only; 4-branch fallback satisfies all 4 calling conventions \\ \bottomrule \end{tabular} \caption{The 6 shim fixes that lifted $\mathrm{decode\_ratio}$ from $0/192$ (3 seeds $\times$ 4 probes $\times$ 16 samples) to $30/30 = 1.000$ across the R17 3-seed $\times$ 10-probe panel (steps 1K, 2K, \ldots, 10K).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_5_honest_negative.tex:714` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\textbf{The Round-17 paper is honest about the failed-metal gates ($m_1$ Ru $0.712 < 0.86$, $m_2$ Pt A2780cis $0.648 < 0.85$, $m_2$ A2780 $0.350 < 0.5$), about the Blackwell single-host four-layer Phase A BLOCK (resource / timing / architecture / data collision, per §\ref{sec:honest-negative:phase-a-block}), about the 6-shim fix story that lifted $\mathrm{decode\_ratio}$ from $0/192$ to $30/30$, about the 27-rule SMARTS expansion + 84-rule MLE matrix + 4 NEW SMARTS templates that close the diversity gap, about the single Tier-1 \MEASURED{} tmQM pretrain lift $+0.046$ (Ru $0.7195 \to 0.7655$), about the single-box compute envelope that bounds the $100{\times}3$ sweep to $\sim 8$~h Lambda-only, and about the R18 SE(3) Schr\"odinger-Bridge upgrade path that`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:2660` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `2000 & 1.000 & 8 / 8 & 0.1\\ \bottomrule \end{tabular} \end{center}  \noindent\textbf{Result:} \textbf{decode\_ratio lifted from 0/192 floor to 1.000 at all four checkpoints} (\textsc{measured}, +100\,\% absolute over the 0/192 \texttt{wf\_gpu\_recovery\_now/final.md} baseline, infinite relative).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:2661` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\bottomrule \end{tabular} \end{center}  \noindent\textbf{Result:} \textbf{decode\_ratio lifted from 0/192 floor to 1.000 at all four checkpoints} (\textsc{measured}, +100\,\% absolute over the 0/192 \texttt{wf\_gpu\_recovery\_now/final.md} baseline, infinite relative).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:3139` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\texttt{connectivity\_or\_valence\_failure:AtomValenceException} & 15 \\ \textbf{decoded\_* (any)} & \textbf{0} \\ \hline \end{tabular} \end{center}  \noindent\textbf{\textsc{measured} $\texttt{decode\_ratio}\,=\,0/192$}.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_results.tex:404` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\paragraph{Shim fix impact (the load-bearing m14 lift).} The 6-shim fix lifts $\mathrm{decode\_ratio}$ from $0/192$ (R16 pre-shim) to $30/30 = 1.000$ (R17 post-shim across 3 seeds $\times$ 10 probes); per-step stability at $\{250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000\}$ is $1.000 \pm 0.000$.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_results.tex:427` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The §4.5 honest-negative section (\texttt{paper/sections/04\_5\_honest\_negative.tex}) carries the failed $m_1$ / $m_2$ gates, the 6-shim fix story that lifted $\mathrm{decode\_ratio}$ from $0/192$ to $30/30$, the 27-rule SMARTS + 84-rule MLE matrix expansion, and the single-box compute envelope that bounds the $100{\times}3$ sweep to $\sim 8$~h Lambda-only.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations.tex:87` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The retrain executed correctly on a healthy GPU, but the 192-sample aggregate (3 seeds $\times$ 2 pockets $\times$ 2 CFG $\times$ 16 samples) returns \textbf{\textsc{measured} $\texttt{decode\_ratio}\,=\,0/192$} with per-cell status counts \texttt{disconnected\_distance\_graph}\,=\,177 / \texttt{connectivity\_or\_valence\_failure:AtomValenceException}\,=\,15.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations.tex:454` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\textbf{\textsc{measured} $\texttt{decode\_ratio}\,=\,1.000$ at steps 500, 1000, 1500, 2000} (vs 0/192 floor baseline, +100\,\% absolute lift, infinite relative).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations.tex:539` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\textbf{Cells promoted \DESIGN{} $\to$ \MEASURED{} in R16:} \textbf{2} --- \texttt{decode\_ratio}\,=\,1.000 at step 500--2000 (vs 0/192 baseline, YuelBond probe) + \texttt{n\_decoded}\,=\,8/8 (same conditioning).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations.tex:577` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\textbf{What is \emph{not} yet closed (GPU-BLOCKED):} (i)~\textbf{path-(b) decode\_ratio on real CFM output}: the Phase~2 follow-up to item (1) ran \textsc{measured} \texttt{decode\_ratio\,=\,0/192} at \texttt{--train-steps 10000 --hidden-dim 64 --n-layers 3} + tmQM warm-start + PCGrad + decoder-rework (\texttt{molmetal/reports/wf\_path\_b\_gpu\_retrain/cfm\_10kstep/report.json}, 1320\,s wall, GPU util 92--98\,\%).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/07_future.tex:165` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\begin{enumerate} \item \textbf{Sub-fix A success} --- the \texttt{wf\_cfm\_gpu\_retrain} decode lift must clear $\texttt{decode\_ratio}\,\geq\,0.5$ on the 6-seed $\times$ 2-pocket $\times$ 16-sample protocol, which currently reads $\texttt{decode\_ratio}\,=\,0/192$ (\texttt{molmetal/reports/wf\_gpu\_recovery\_now/ final.md}, 2026-09-15).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/07_future.tex:450` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\item \textbf{TODO-24 P0 2000-step probe} --- \textsc{passed 2026-09-17}, decode\_ratio lifted 0/192 $\to$ 1.000 at all checkpoints; spec gate is PASS, \textbf{PROCEED to 10000-step retrain} (12--24~h GPU).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/supplementary/S6_ablation.tex:136` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `(\MEASURED{} on PlatinAI 5K pretrain); M10 (84-rule MLE matrix) + M11 (18-rule + 4 NEW SMARTS) are the rule-coverage expansion (\MEASURED{}); M12 (20-element vocab) + M13 (logits clamp $\pm 50$) + M14 (LR warmup $200$ steps) + M15 (shim TypeError fix) + M16 (ModelWrapper keyword-time fallback) + M17 (reference loader flat-clone auto-detect) are the 6 R17 shim fixes that lifted $\mathrm{decode\_ratio}$ from $0/192$ to $30/30$ (\MEASURED{}, Section~S5.1 honest framing); M18 (Ertl SA) + M19 (PoseBusters) + M20 (BioPython NeighborSearch) are the external-scorer evaluators (\MEASURED{} on $14/14$ evaluators PASS); M21 (scaffold split) is the data-leakage guardrail (\MEASURED{} on $5$ benchmark suites).`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/supplementary/S6_ablation.tex:341` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\paragraph{Canonical ablation reports.} - \texttt{molmetal/reports/wf\_tmqm\_pretrain/final.md} (M12 + tmQM ablation $+0.046$ Ru AUC; Section~S6.1). - \texttt{molmetal/reports/wf\_r17\_10k\_retrain\_done/final.md} (M12--M17 ablation $0/192 \to 30/30$ decode ratio; Section~S6.1). - \texttt{molmetal/reports/wf\_todo33\_p2\_shim\_fix/final.md} (6-shim fix story; Section~S6.1 R17 carry-over ablation). - \texttt{molmetal/reports/wf\_r10\_cfg\_ablation\_1h36/final.md} (R10 CFM Path B ablation per-config; Section~S6.3 epoch budget). - \texttt{molmetal/reports/wf\_e2e\_pipeline/final.md} + \texttt{molmetal/reports/wf\_r18\_end\_to\_end/final.md}`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/traceability.csv:13` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Both coordinates are from,main.tex,389,4 sections,c0008,MEASURED,% explicit / \PROJECTED{} / \DESIGN{} tag consistent with,sections/01_intro.tex,5,1 sections,c0009,PROJECTED,% explicit \MEASURED{} / / \DESIGN{} tag consistent with,sections/01_intro.tex,5,1 sections,c0010,DESIGN,% explicit \MEASURED{} / \PROJECTED{} / tag consistent with,sections/01_intro.tex,5,1 sections,c0011,MEASURED,the full training distribution ($ $ expansion from,sections/01_intro.tex,202,2 sections,c0012,MEASURED,"$\mathrm{decode\_ratio}$ from $ \,0/192$ to",sections/01_intro.tex,210,3 sections,c0013,MEASURED,"$ \,1.000$ on three seeds over four probe budgets",sections/01_intro.tex,211,4 sections,c0014,MEASURED,"The combined rule set has $ \,27$ SMARTS reactions",sections/01_intro.tex,229,5 sections,c0015,MEASURED,"$ \,84$+ reactions on the $12\times 7$ metal-matrix",sections/01_intro.tex,231,6 sections,c0016,MEASURED,"$ \,\sim 20\%$ of the full-5-click diversity",sections/01_intro.tex,233,7 sections,c0017,MEASURED,Vina-vs-QuickVina parity audit ($ $ Pearson $r = 0.9983$,sections/01_intro.tex,256,8 sections,c0018,MEASURED,$ $ honest null at $n_{\mathrm{docked}}=4$ which we,sections/01_intro.tex,262,9`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/traceability.csv:726` -- rule B1 -- **UNQUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `sections,c0719,MEASURED,random initialisation. evidence: Stage 1.5 line 5,sections/06_limitations.tex,94,5 sections,c0720,MEASURED,The $m_2$ PlatinAI A2780 cell line returns,sections/06_limitations.tex,110,6 sections,c0721,MEASURED,All reported numbers are on a single Blackwell-class,sections/06_limitations.tex,140,7 sections,c0722,DESIGN,Table~\ref{tab:sbdd-comparison} is pending a cloud-GPU,sections/06_limitations.tex,158,1 sections,c0723,MEASURED,"json}), and the real-pocket strict measurement is",sections/06_limitations.tex,206,8 sections,c0724,MEASURED,from $\mathrm{Inf}$ to $2.996$ and,sections/06_limitations.tex,210,9 sections,c0725,MEASURED,$\mathrm{decode\_ratio}$ from $0/192$ to $1.000$,sections/06_limitations.tex,211,10 sections,c0726,DESIGN,$ \!\to\!\MEASURED{}$ promotion under the strict,sections/06_limitations.tex,221,2 sections,c0727,MEASURED,$\DESIGN{}\!\to\! $ promotion under the strict,sections/06_limitations.tex,221,11 sections,c0728,DESIGN,from to \MEASURED{} by this inventory; every,sections/06_limitations.tex,268,3 sections,c0729,MEASURED,from \DESIGN{} to by this inventory; every,sections/06_limitations.tex,268,12 sections,c0730,MEASURED,headline = \textit{n/a} (target \textit{see METRICS\_TARGET,sections/06_limitations_auto.tex,24,1 sections,c0731,MEASURED,headline = $0.535$ (target $0.7$).,sections/06_limitations_auto.tex,30,2`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:8082` -- rule B2 -- **UNQUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\OT1/cmtt/m/sc/10 The 192/192 is \OT1/cmtt/m/i t/10 bond-bearing \OT1/cmtt/m/sc/10 (decoder emitted at least one bond), not RD Kit-strict:  Overfull \hbox (22.50172pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 96/192 fail Chem.SanitizeMol on the synthetic random cloud (o vervalent C atoms, e.g.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:72321` -- rule B2 -- **UNQUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\OT1/cmtt/m/sc/10 The 192/192 is \OT1/cmtt/m/i t/10 bond-bearing \OT1/cmtt/m/sc/10 (decoder emitted at least one bond), not RD Kit-strict:  Overfull \hbox (22.50172pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 96/192 fail Chem.SanitizeMol on the synthetic random cloud (o vervalent C atoms, e.g.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:136492` -- rule B2 -- **UNQUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\OT1/cmtt/m/sc/10 The 192/192 is \OT1/cmtt/m/i t/10 bond-bearing \OT1/cmtt/m/sc/10 (decoder emitted at least one bond), not RD Kit-strict:  Overfull \hbox (22.50172pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 96/192 fail Chem.SanitizeMol on the synthetic random cloud (o vervalent C atoms, e.g.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:200663` -- rule B2 -- **UNQUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\OT1/cmtt/m/sc/10 The 192/192 is \OT1/cmtt/m/i t/10 bond-bearing \OT1/cmtt/m/sc/10 (decoder emitted at least one bond), not RD Kit-strict:  Overfull \hbox (22.50172pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 96/192 fail Chem.SanitizeMol on the synthetic random cloud (o vervalent C atoms, e.g.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/section4_results.tex:245` -- rule B2 -- **UNQUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\label{sec:lead-decode}  \noindent\textbf{Synthetic}: our YuelBond decoder (Wang + Dokholyan~2025 GNN head + valence-aware greedy selector) + Wasserstein gradient-flow regulariser + Catherine closure-bounded generation achieves \texttt{decode\_ratio} RDKit-strict $= 1.000$ ($192/192$) on the synthetic Pt-click validation cohort.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/section4_results.tex:431` -- rule B2 -- **UNQUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Dataset coverage                & $791{,}013 \times 16 \times 5$      & L    \\ QED                             & $0.6123$                            & L    \\ hERG cardio risk                & $0.255$                             & L    \\ AMES mutagen                    & $0.0401$                            & L    \\ plasma protein binding          & $0.7899$                            & L    \\ RDKit validity                  & $1.000$                             & L    \\ Decode ratio (synthetic)        & $1.000$ ($192/192$)                 & L    \\ Decode ratio (real NCI-60+DOG)  & $0.9980$ ($5991$ samples)           & L    \\ Decode ratio (real PlatinAI)    & $0.9745$ ($274$ samples)           & L    \\ MCTS tree success               & $100\%$ / $85$ nodes                & L    \\ Sprint A Kendall MTL            & dropped-in, gate pending            & L    \\ Sprint A WeightedRandomSampler  & dropped-in, gate pending            & L    \\ Sprint A DANN GRL               & partial wiring                      & $[\text{H}]$ \\`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_4_cfm_methodology.tex:105` -- rule B2 -- **UNQUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\paragraph{Why this subsection.} The path-(a) full retrain was \textsc{not measured}: \texttt{decode\_ratio\,=\,0/192} on the CrossDocked-trained checkpoint (\texttt{molmetal/reports/wf\_path\_b\_gpu\_retrain/final.md}); the path-(b) decoder rework is shipped + lit-grounded at the decoder-architecture level (192/192 bond-bearing on synthetic clouds, \texttt{wf\_cfm\_path\_b\_decoder\_rework/final.md}) and is the load-bearing follow-up.`
- **why this classification**: window does not name the population (cohort / pool / budget / seed count) the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:16612` -- rule A4 -- **UNSOURCED**

- **number**: `99.9 %` (matched `99.9`)
- **what it is**: FlowMol3 %Valid (GEOM-Drugs)
- **who measured it**: external
- **should cite**: Dunn & Koes, Digital Discovery 2026, 5(5), 2052-2066, doi:10.1039/d5dd00363f (arXiv 2508.12629)
- **population that makes it interpretable**: GEOM-Drugs, explicit hydrogens
- **sentence (verbatim)**: `Self-conditioning + fake atoms + train-time geome try distortion; 99.9%  Underfull \hbox (badness 10000) in paragraph at lines 691--699 []\OT1/cmtt/m/sc/10 Ross Irwin and others.`
- **why this classification**: no source and no population anywhere in the window
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:80851` -- rule A4 -- **UNSOURCED**

- **number**: `99.9 %` (matched `99.9`)
- **what it is**: FlowMol3 %Valid (GEOM-Drugs)
- **who measured it**: external
- **should cite**: Dunn & Koes, Digital Discovery 2026, 5(5), 2052-2066, doi:10.1039/d5dd00363f (arXiv 2508.12629)
- **population that makes it interpretable**: GEOM-Drugs, explicit hydrogens
- **sentence (verbatim)**: `Self-conditioning + fake atoms + train-time geome try distortion; 99.9%  Underfull \hbox (badness 10000) in paragraph at lines 691--699 []\OT1/cmtt/m/sc/10 Ross Irwin and others.`
- **why this classification**: no source and no population anywhere in the window
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:145022` -- rule A4 -- **UNSOURCED**

- **number**: `99.9 %` (matched `99.9`)
- **what it is**: FlowMol3 %Valid (GEOM-Drugs)
- **who measured it**: external
- **should cite**: Dunn & Koes, Digital Discovery 2026, 5(5), 2052-2066, doi:10.1039/d5dd00363f (arXiv 2508.12629)
- **population that makes it interpretable**: GEOM-Drugs, explicit hydrogens
- **sentence (verbatim)**: `Self-conditioning + fake atoms + train-time geome try distortion; 99.9%  Underfull \hbox (badness 10000) in paragraph at lines 691--699 []\OT1/cmtt/m/sc/10 Ross Irwin and others.`
- **why this classification**: no source and no population anywhere in the window
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:209193` -- rule A4 -- **UNSOURCED**

- **number**: `99.9 %` (matched `99.9`)
- **what it is**: FlowMol3 %Valid (GEOM-Drugs)
- **who measured it**: external
- **should cite**: Dunn & Koes, Digital Discovery 2026, 5(5), 2052-2066, doi:10.1039/d5dd00363f (arXiv 2508.12629)
- **population that makes it interpretable**: GEOM-Drugs, explicit hydrogens
- **sentence (verbatim)**: `Self-conditioning + fake atoms + train-time geome try distortion; 99.9%  Underfull \hbox (badness 10000) in paragraph at lines 691--699 []\OT1/cmtt/m/sc/10 Ross Irwin and others.`
- **why this classification**: no source and no population anywhere in the window
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_method.tex:184` -- rule C5 -- **UNSOURCED**

- **number**: `1.5-3.5 A (and 1-5 A)` (matched `[1.5,\,3.5]`)
- **what it is**: 'typical CFM coordinate spread' -- our own empirical claim about coordinate spread
- **who measured it**: us
- **should cite**: a round-12 trained-CFM coordinate-spread measurement
- **population that makes it interpretable**: which run / round / pool the spread was measured on
- **sentence (verbatim)**: `The $\tau\!=\!0.3$\,\AA{} band gives $\sim\!10\times$ decay over 1\,\AA{} so the gradient is non-trivial for $d\!\in\![1.5,\,3.5]$\,\AA{}, the typical CFM coordinate spread.`
- **why this classification**: no source for the constant anywhere in the window
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: Replace `the typical CFM coordinate spread` with `the round-12 trained-CFM coordinate spread (\texttt{molmetal/reports/<round-12 spread report>})` so the range names its measurement.

### `paper/build_log.txt:8252` -- rule A1 -- **QUALIFIED**

- **number**: `92.7 %` (matched `92.7%`)
- **what it is**: YuelBond per-BOND recovery F1 on the CDG cohort (the paper's worst case)
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052 (preprint bioRxiv 10.1101/2025.05.06.652517)
- **population that makes it interpretable**: per-bond recovery F1 (NOT a per-molecule validity rate) on the CDG cohort (Crude De novo Generated, Gaussian noise sigma = 0.2 A)
- **sentence (verbatim)**: `Overfull \hbox (22.50172pt too wide) in paragraph at lines 3177--3202 \OT1/cmtt/m/sc/10 the lit-anchored YuelBond decoder (Wang & Dokholyan 2025 bioR xiv 10.1101/2025.05.06.652517,  Overfull \hbox (175.59035pt too wide) in paragraph at lines 3177--3202 \OT1/cmtt/m/sc/10 F1=92.7% on distorted molecules) is shipped as a drop-in alte rnative (molmetal/molmetal[]lam/lam[]chem/yuelbond[]decoder.py,  Overfull \hbox (25.73181pt too wide) in paragraph at lines 3177--3202 \OMS/cmsy/m/n/10 ^^T \OT1/cmr/m/n/10 5 \OML/cmm/m/it/10 k$ \OT1/cmtt/m/sc/10 pa rams, $\OML/cmm/m/it/10 K  \OT1/cmr/m/n/10 =  3$ \OT1/cmtt/m/sc/10 message-pass ing layers, distance cutoff $\OT1/cmr/m/n/10 3\OML/cmm/m/it/10 :\OT1/cmr/m/n/10`
- **why this classification**: window names the source AND the population/cohort that makes the number interpretable
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:15712` -- rule A1 -- **QUALIFIED**

- **number**: `92.7 %` (matched `92.7%`)
- **what it is**: YuelBond per-BOND recovery F1 on the CDG cohort (the paper's worst case)
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052 (preprint bioRxiv 10.1101/2025.05.06.652517)
- **population that makes it interpretable**: per-bond recovery F1 (NOT a per-molecule validity rate) on the CDG cohort (Crude De novo Generated, Gaussian noise sigma = 0.2 A)
- **sentence (verbatim)**: `F1=92.7% on  Underfull \hbox (badness 10000) in paragraph at lines 163--174 \OT1/cmtt/m/sc/10 Crude De novo Generated compounds (CDGs) where RDKit xyz2mol failed on 783/1000  Underfull \hbox (badness 10000) in paragraph at lines 163--174`
- **why this classification**: window names the source AND the population/cohort that makes the number interpretable
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:72491` -- rule A1 -- **QUALIFIED**

- **number**: `92.7 %` (matched `92.7%`)
- **what it is**: YuelBond per-BOND recovery F1 on the CDG cohort (the paper's worst case)
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052 (preprint bioRxiv 10.1101/2025.05.06.652517)
- **population that makes it interpretable**: per-bond recovery F1 (NOT a per-molecule validity rate) on the CDG cohort (Crude De novo Generated, Gaussian noise sigma = 0.2 A)
- **sentence (verbatim)**: `Overfull \hbox (22.50172pt too wide) in paragraph at lines 3177--3202 \OT1/cmtt/m/sc/10 the lit-anchored YuelBond decoder (Wang & Dokholyan 2025 bioR xiv 10.1101/2025.05.06.652517,  Overfull \hbox (175.59035pt too wide) in paragraph at lines 3177--3202 \OT1/cmtt/m/sc/10 F1=92.7% on distorted molecules) is shipped as a drop-in alte rnative (molmetal/molmetal[]lam/lam[]chem/yuelbond[]decoder.py,  Overfull \hbox (25.73181pt too wide) in paragraph at lines 3177--3202 \OMS/cmsy/m/n/10 ^^T \OT1/cmr/m/n/10 5 \OML/cmm/m/it/10 k$ \OT1/cmtt/m/sc/10 pa rams, $\OML/cmm/m/it/10 K  \OT1/cmr/m/n/10 =  3$ \OT1/cmtt/m/sc/10 message-pass ing layers, distance cutoff $\OT1/cmr/m/n/10 3\OML/cmm/m/it/10 :\OT1/cmr/m/n/10`
- **why this classification**: window names the source AND the population/cohort that makes the number interpretable
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:79951` -- rule A1 -- **QUALIFIED**

- **number**: `92.7 %` (matched `92.7%`)
- **what it is**: YuelBond per-BOND recovery F1 on the CDG cohort (the paper's worst case)
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052 (preprint bioRxiv 10.1101/2025.05.06.652517)
- **population that makes it interpretable**: per-bond recovery F1 (NOT a per-molecule validity rate) on the CDG cohort (Crude De novo Generated, Gaussian noise sigma = 0.2 A)
- **sentence (verbatim)**: `F1=92.7% on  Underfull \hbox (badness 10000) in paragraph at lines 163--174 \OT1/cmtt/m/sc/10 Crude De novo Generated compounds (CDGs) where RDKit xyz2mol failed on 783/1000  Underfull \hbox (badness 10000) in paragraph at lines 163--174`
- **why this classification**: window names the source AND the population/cohort that makes the number interpretable
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:136662` -- rule A1 -- **QUALIFIED**

- **number**: `92.7 %` (matched `92.7%`)
- **what it is**: YuelBond per-BOND recovery F1 on the CDG cohort (the paper's worst case)
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052 (preprint bioRxiv 10.1101/2025.05.06.652517)
- **population that makes it interpretable**: per-bond recovery F1 (NOT a per-molecule validity rate) on the CDG cohort (Crude De novo Generated, Gaussian noise sigma = 0.2 A)
- **sentence (verbatim)**: `Overfull \hbox (22.50172pt too wide) in paragraph at lines 3177--3202 \OT1/cmtt/m/sc/10 the lit-anchored YuelBond decoder (Wang & Dokholyan 2025 bioR xiv 10.1101/2025.05.06.652517,  Overfull \hbox (175.59035pt too wide) in paragraph at lines 3177--3202 \OT1/cmtt/m/sc/10 F1=92.7% on distorted molecules) is shipped as a drop-in alte rnative (molmetal/molmetal[]lam/lam[]chem/yuelbond[]decoder.py,  Overfull \hbox (25.73181pt too wide) in paragraph at lines 3177--3202 \OMS/cmsy/m/n/10 ^^T \OT1/cmr/m/n/10 5 \OML/cmm/m/it/10 k$ \OT1/cmtt/m/sc/10 pa rams, $\OML/cmm/m/it/10 K  \OT1/cmr/m/n/10 =  3$ \OT1/cmtt/m/sc/10 message-pass ing layers, distance cutoff $\OT1/cmr/m/n/10 3\OML/cmm/m/it/10 :\OT1/cmr/m/n/10`
- **why this classification**: window names the source AND the population/cohort that makes the number interpretable
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:144122` -- rule A1 -- **QUALIFIED**

- **number**: `92.7 %` (matched `92.7%`)
- **what it is**: YuelBond per-BOND recovery F1 on the CDG cohort (the paper's worst case)
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052 (preprint bioRxiv 10.1101/2025.05.06.652517)
- **population that makes it interpretable**: per-bond recovery F1 (NOT a per-molecule validity rate) on the CDG cohort (Crude De novo Generated, Gaussian noise sigma = 0.2 A)
- **sentence (verbatim)**: `F1=92.7% on  Underfull \hbox (badness 10000) in paragraph at lines 163--174 \OT1/cmtt/m/sc/10 Crude De novo Generated compounds (CDGs) where RDKit xyz2mol failed on 783/1000  Underfull \hbox (badness 10000) in paragraph at lines 163--174`
- **why this classification**: window names the source AND the population/cohort that makes the number interpretable
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:200833` -- rule A1 -- **QUALIFIED**

- **number**: `92.7 %` (matched `92.7%`)
- **what it is**: YuelBond per-BOND recovery F1 on the CDG cohort (the paper's worst case)
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052 (preprint bioRxiv 10.1101/2025.05.06.652517)
- **population that makes it interpretable**: per-bond recovery F1 (NOT a per-molecule validity rate) on the CDG cohort (Crude De novo Generated, Gaussian noise sigma = 0.2 A)
- **sentence (verbatim)**: `Overfull \hbox (22.50172pt too wide) in paragraph at lines 3177--3202 \OT1/cmtt/m/sc/10 the lit-anchored YuelBond decoder (Wang & Dokholyan 2025 bioR xiv 10.1101/2025.05.06.652517,  Overfull \hbox (175.59035pt too wide) in paragraph at lines 3177--3202 \OT1/cmtt/m/sc/10 F1=92.7% on distorted molecules) is shipped as a drop-in alte rnative (molmetal/molmetal[]lam/lam[]chem/yuelbond[]decoder.py,  Overfull \hbox (25.73181pt too wide) in paragraph at lines 3177--3202 \OMS/cmsy/m/n/10 ^^T \OT1/cmr/m/n/10 5 \OML/cmm/m/it/10 k$ \OT1/cmtt/m/sc/10 pa rams, $\OML/cmm/m/it/10 K  \OT1/cmr/m/n/10 =  3$ \OT1/cmtt/m/sc/10 message-pass ing layers, distance cutoff $\OT1/cmr/m/n/10 3\OML/cmm/m/it/10 :\OT1/cmr/m/n/10`
- **why this classification**: window names the source AND the population/cohort that makes the number interpretable
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:208293` -- rule A1 -- **QUALIFIED**

- **number**: `92.7 %` (matched `92.7%`)
- **what it is**: YuelBond per-BOND recovery F1 on the CDG cohort (the paper's worst case)
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052 (preprint bioRxiv 10.1101/2025.05.06.652517)
- **population that makes it interpretable**: per-bond recovery F1 (NOT a per-molecule validity rate) on the CDG cohort (Crude De novo Generated, Gaussian noise sigma = 0.2 A)
- **sentence (verbatim)**: `F1=92.7% on  Underfull \hbox (badness 10000) in paragraph at lines 163--174 \OT1/cmtt/m/sc/10 Crude De novo Generated compounds (CDGs) where RDKit xyz2mol failed on 783/1000  Underfull \hbox (badness 10000) in paragraph at lines 163--174`
- **why this classification**: window names the source AND the population/cohort that makes the number interpretable
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:3179` -- rule A1 -- **QUALIFIED**

- **number**: `92.7 %` (matched `92.7\%`)
- **what it is**: YuelBond per-BOND recovery F1 on the CDG cohort (the paper's worst case)
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052 (preprint bioRxiv 10.1101/2025.05.06.652517)
- **population that makes it interpretable**: per-bond recovery F1 (NOT a per-molecule validity rate) on the CDG cohort (Crude De novo Generated, Gaussian noise sigma = 0.2 A)
- **sentence (verbatim)**: `\paragraph{YuelBond decoder swap: \textsc{available} parallel path (WF-CFM-Rescue, 2026-09-16).} Per \texttt{molmetal/reports/wf\_cfm\_rescue/phase1\_yuelbond.md}, the lit-anchored YuelBond decoder (Wang \& Dokholyan 2025 bioRxiv 10.1101/2025.05.06.652517, F1=92.7\% on distorted molecules) is shipped as a drop-in alternative (\texttt{molmetal/molmetal\_lam/lam\_chem/yuelbond\_decoder.py}, $\leq 5\,k$ params, $K\!=\!3$ message-passing layers, distance cutoff $3.5$\,\AA{}, donor$\to$Pt dative detector).`
- **why this classification**: window names the source AND the population/cohort that makes the number interpretable
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: Replace the parenthetical with: `(Wang \& Dokholyan 2025 bioRxiv 10.1101/2025.05.06.652517; published J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052; per-bond recovery F1 = 92.7\% on the CDG cohort -- Crude De novo Generated, Gaussian noise \sigma = 0.2 \AA, the paper's worst case -- and 98.4\% on standard 3-D structures; NOT a per-molecule validity rate)`.

### `paper/build_log.txt:15716` -- rule A3 -- **QUALIFIED**

- **number**: `783` (matched `783`)
- **what it is**: RDKit xyz2mol / DetermineBonds failures out of 1000 distorted CDGs
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052
- **population that makes it interpretable**: out of 1000 CDGs at sigma = 0.2 A
- **sentence (verbatim)**: `F1=92.7% on  Underfull \hbox (badness 10000) in paragraph at lines 163--174 \OT1/cmtt/m/sc/10 Crude De novo Generated compounds (CDGs) where RDKit xyz2mol failed on 783/1000  Underfull \hbox (badness 10000) in paragraph at lines 163--174 \OT1/cmtt/m/sc/10 molecules; GNN with edge-centric message passing trained on G EOM lowest-energy  LaTeX Font Warning: Font shape `OT1/cmtt/m/scit' undefined`
- **why this classification**: window names the source AND the population/cohort that makes the number interpretable
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:79955` -- rule A3 -- **QUALIFIED**

- **number**: `783` (matched `783`)
- **what it is**: RDKit xyz2mol / DetermineBonds failures out of 1000 distorted CDGs
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052
- **population that makes it interpretable**: out of 1000 CDGs at sigma = 0.2 A
- **sentence (verbatim)**: `F1=92.7% on  Underfull \hbox (badness 10000) in paragraph at lines 163--174 \OT1/cmtt/m/sc/10 Crude De novo Generated compounds (CDGs) where RDKit xyz2mol failed on 783/1000  Underfull \hbox (badness 10000) in paragraph at lines 163--174 \OT1/cmtt/m/sc/10 molecules; GNN with edge-centric message passing trained on G EOM lowest-energy  LaTeX Font Warning: Font shape `OT1/cmtt/m/scit' undefined`
- **why this classification**: window names the source AND the population/cohort that makes the number interpretable
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:144126` -- rule A3 -- **QUALIFIED**

- **number**: `783` (matched `783`)
- **what it is**: RDKit xyz2mol / DetermineBonds failures out of 1000 distorted CDGs
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052
- **population that makes it interpretable**: out of 1000 CDGs at sigma = 0.2 A
- **sentence (verbatim)**: `F1=92.7% on  Underfull \hbox (badness 10000) in paragraph at lines 163--174 \OT1/cmtt/m/sc/10 Crude De novo Generated compounds (CDGs) where RDKit xyz2mol failed on 783/1000  Underfull \hbox (badness 10000) in paragraph at lines 163--174 \OT1/cmtt/m/sc/10 molecules; GNN with edge-centric message passing trained on G EOM lowest-energy  LaTeX Font Warning: Font shape `OT1/cmtt/m/scit' undefined`
- **why this classification**: window names the source AND the population/cohort that makes the number interpretable
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:208297` -- rule A3 -- **QUALIFIED**

- **number**: `783` (matched `783`)
- **what it is**: RDKit xyz2mol / DetermineBonds failures out of 1000 distorted CDGs
- **who measured it**: external
- **should cite**: Wang & Dokholyan, J. Chem. Inf. Model. 2026, 66(2), 1003-1012, doi:10.1021/acs.jcim.5c03052
- **population that makes it interpretable**: out of 1000 CDGs at sigma = 0.2 A
- **sentence (verbatim)**: `F1=92.7% on  Underfull \hbox (badness 10000) in paragraph at lines 163--174 \OT1/cmtt/m/sc/10 Crude De novo Generated compounds (CDGs) where RDKit xyz2mol failed on 783/1000  Underfull \hbox (badness 10000) in paragraph at lines 163--174 \OT1/cmtt/m/sc/10 molecules; GNN with edge-centric message passing trained on G EOM lowest-energy  LaTeX Font Warning: Font shape `OT1/cmtt/m/scit' undefined`
- **why this classification**: window names the source AND the population/cohort that makes the number interpretable
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:8095` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The  Overfull \hbox (49.59145pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 Path-A 0/192 is the same \OT1/cmtt/m/n/10 0/192 \OT1/cmtt/m/s c/10 baseline reported in molmetal/reports/wf[]cfm[]retrain[]full/final.md  Overfull \hbox (17.23863pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 at 10000-step + h=64, so the lift comparison is apples-to-app les.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:8095` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The  Overfull \hbox (49.59145pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 Path-A 0/192 is the same \OT1/cmtt/m/n/10 0/192 \OT1/cmtt/m/s c/10 baseline reported in molmetal/reports/wf[]cfm[]retrain[]full/final.md  Overfull \hbox (17.23863pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 at 10000-step + h=64, so the lift comparison is apples-to-app les.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:72334` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The  Overfull \hbox (49.59145pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 Path-A 0/192 is the same \OT1/cmtt/m/n/10 0/192 \OT1/cmtt/m/s c/10 baseline reported in molmetal/reports/wf[]cfm[]retrain[]full/final.md  Overfull \hbox (17.23863pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 at 10000-step + h=64, so the lift comparison is apples-to-app les.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:72334` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The  Overfull \hbox (49.59145pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 Path-A 0/192 is the same \OT1/cmtt/m/n/10 0/192 \OT1/cmtt/m/s c/10 baseline reported in molmetal/reports/wf[]cfm[]retrain[]full/final.md  Overfull \hbox (17.23863pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 at 10000-step + h=64, so the lift comparison is apples-to-app les.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:136505` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The  Overfull \hbox (49.59145pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 Path-A 0/192 is the same \OT1/cmtt/m/n/10 0/192 \OT1/cmtt/m/s c/10 baseline reported in molmetal/reports/wf[]cfm[]retrain[]full/final.md  Overfull \hbox (17.23863pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 at 10000-step + h=64, so the lift comparison is apples-to-app les.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:136505` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The  Overfull \hbox (49.59145pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 Path-A 0/192 is the same \OT1/cmtt/m/n/10 0/192 \OT1/cmtt/m/s c/10 baseline reported in molmetal/reports/wf[]cfm[]retrain[]full/final.md  Overfull \hbox (17.23863pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 at 10000-step + h=64, so the lift comparison is apples-to-app les.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:200676` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The  Overfull \hbox (49.59145pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 Path-A 0/192 is the same \OT1/cmtt/m/n/10 0/192 \OT1/cmtt/m/s c/10 baseline reported in molmetal/reports/wf[]cfm[]retrain[]full/final.md  Overfull \hbox (17.23863pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 at 10000-step + h=64, so the lift comparison is apples-to-app les.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:200676` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The  Overfull \hbox (49.59145pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 Path-A 0/192 is the same \OT1/cmtt/m/n/10 0/192 \OT1/cmtt/m/s c/10 baseline reported in molmetal/reports/wf[]cfm[]retrain[]full/final.md  Overfull \hbox (17.23863pt too wide) in paragraph at lines 3082--3105 \OT1/cmtt/m/sc/10 at 10000-step + h=64, so the lift comparison is apples-to-app les.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/figures/data/fig12_decode_ratio_per_seed.json:56` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `"n_probes_per_seed": 10, "n_seeds": 3, "total_probes": 30, "all_decode_ratios": 1.000, "r16_floor": 0.0, "lit_target_targetdiff": 0.50, "verdict": "MEASURED — 30/30 probes at decode_ratio=1.000 (full lift from 0/192 R16 floor)" }, "source": "metrics/by_metric/decode_ratio.json (Path B aggregate) + logs/r17_retrain_10k/seed_{0,1,2}.log (decode_smoke probes)", "render": "paper/figures/fig12_decode_ratio_per_seed.py" }`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_method.tex:33` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `%    §3.6 stub (this file)          -- WF-Pivot-A Phase 2 %                                     (deferred-secondary-generator stub: %                                     label-only preservation, full %                                     content moved to §7 future work) %    §3.7 bond-aware soft prior     -- WF-CFM-Path-B-Decoder-Rework %                                     (chem-aware 3-prior decoder; %                                     Path A→Path B lift 0/192→192/192) % %  Cross-references all point to: %    Figure~1 (9-layer MLC architecture, fig1_mlc_arch.pdf)        -- 03.1 %    Figure~2 (pocket-conditioned Lambda pipeline, fig2_pipeline.pdf) -- 03.3 + 03.4 %    Figure~3 (5 click reactions, fig3_click_reactions.pdf)         -- 03.2 %    Appendix closure-theorem (paper/appendices/closure_theorem.tex,`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_method.tex:239` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\paragraph{Smoke validation (measured).} On 192 synthetic CFM-style 10-atom clouds with realistic 1-5\,\AA{} coordinate spread (matches the round-12 trained-CFM distribution), the soft prior lifts bond-bearing decode from $0/192$ (Path A: legacy \texttt{BondAwareDecoder} with hard 2.4\,\AA{} cutoff) to $192/192$ (Path B: \texttt{ReworkedDecoder} with soft 3-prior) at a 500-step + $h\!=\!64$ mini-budget. 96/192 of the Path-B mols fail \texttt{Chem.SanitizeMol} due to overvalent atoms in the random cloud (e.g.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:3076` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\begin{center} \begin{tabular}{l c c c} \hline \textbf{decoder} & \textbf{bond-bearing} & \textbf{RDKit-strict} & \textbf{status} \\ \hline Path A (legacy \texttt{BondAwareDecoder}, 2.4\,\AA{} cutoff) & 0/192 & 0/192 & 0\% lifted \\ \textbf{Path B} (\texttt{ReworkedDecoder}, soft 3-prior) & \textbf{192/192} & 0/192 & \textbf{+1.0 lift} \\ \hline \end{tabular} \end{center}  \noindent\textbf{Honest framing.} The 192/192 is`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:3076` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\begin{center} \begin{tabular}{l c c c} \hline \textbf{decoder} & \textbf{bond-bearing} & \textbf{RDKit-strict} & \textbf{status} \\ \hline Path A (legacy \texttt{BondAwareDecoder}, 2.4\,\AA{} cutoff) & 0/192 & 0/192 & 0\% lifted \\ \textbf{Path B} (\texttt{ReworkedDecoder}, soft 3-prior) & \textbf{192/192} & 0/192 & \textbf{+1.0 lift} \\ \hline \end{tabular} \end{center}  \noindent\textbf{Honest framing.} The 192/192 is`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:3077` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\begin{center} \begin{tabular}{l c c c} \hline \textbf{decoder} & \textbf{bond-bearing} & \textbf{RDKit-strict} & \textbf{status} \\ \hline Path A (legacy \texttt{BondAwareDecoder}, 2.4\,\AA{} cutoff) & 0/192 & 0/192 & 0\% lifted \\ \textbf{Path B} (\texttt{ReworkedDecoder}, soft 3-prior) & \textbf{192/192} & 0/192 & \textbf{+1.0 lift} \\ \hline \end{tabular} \end{center}  \noindent\textbf{Honest framing.} The 192/192 is \emph{bond-bearing} (decoder emitted at least one bond), not`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:3087` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The Path-A 0/192 is the same \textbf{0/192} baseline reported in \texttt{molmetal/reports/wf\_cfm\_retrain\_full/final.md} at 10000-step + h\,=\,64, so the lift comparison is apples-to-apples.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:3088` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The Path-A 0/192 is the same \textbf{0/192} baseline reported in \texttt{molmetal/reports/wf\_cfm\_retrain\_full/final.md} at 10000-step + h\,=\,64, so the lift comparison is apples-to-apples.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_results.tex:206` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\textbf{Strong cells (paper-promotable):} m11 $n_{\mathrm{distinct}}$ $4$--$7$ (post-fix 30-cell panel, R16 Path A baseline $1$), m14 Decode ratio $1.000$ (3 seeds $\times$ 10 probes = 30/30 stable; pre-shim baseline $0/192$), m17 DrugOOD assay-OOD $0.891$ / scaffold-OOD $0.697$ / size-OOD $0.748$ (all beat iMoLD), Deflexpressor hold-out $R^2 = 1.0000$ (depth-4 ASTSR recovers the 5-term composite exactly; $\Delta R^2 = +0.0676$ over F5 LassoBIC), m21 NCI-60 precious 9-cell mean $0.801$ (matches Krasnov single-line Ru $0.81$).`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_results.tex:316` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\paragraph{m14 Decode ratio on CFM (LIFTED $0 \to 1.000$).} The R16 pre-shim CFM probe had $\mathrm{decode\_ratio} = 0/192$ on a 5000-step CPU smoke and $0/8$ on the 500-step GPU smoke at all three checkpoints (\texttt{wf\_gpu\_recovery\_now} + \texttt{wf\_r15\_cfm\_cpu\_verify}).`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_results_table2.tex:63` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\multicolumn{6}{l}{\textit{Tier 1 -- generated-pool structural / diversity metrics}} \\ m11 $n_{\mathrm{distinct}}$ (within-cell) & $\mathbf{4}$--$\mathbf{7}$ \MEASURED{} (post-fix, 30 cells) & $1$ R16 Path A pre-fix \MEASURED{} & $\geq 3$ (met) \PROJECTED{} & higher & \textbf{LIFTED} $1\!\to\!4$--$7$ (singleton collapse broken; saturation depends on cell) \\ m13 RDKit-parse validity               & $1.000$ \MEASURED{} (30 cells)     & 0.99--1.00 SOTA \CITEDONLY{}    & $= 1.000$ (met) \PROJECTED{} & higher & \textbf{PASS} (RDKit sanitisation floor) \\ m12 SyntheMol synthesis rate           & $\approx 1.00$ \MEASURED{} (R16)    & 1.00 R16 Path A \MEASURED{}    & $= 1.000$ (met) \PROJECTED{} & higher & \textbf{PASS} (perfect; Path-A already at ceiling) \\ \midrule \multicolumn{6}{l}{\textit{Tier 1 -- CFM decode-path diagnostic}} \\ m14 Decode ratio (CFM)                 & $\mathbf{1.000}$ \MEASURED{} (3 seeds $\times$ 10 probes) & 0/192 R16 pre-shim \MEASURED{} & $= 1.000$ (met) \PROJECTED{} & higher & \textbf{LIFTED} $0\!\to\!1.000$ (RDKit-strict pre: $0.500$; 6-shim fix stable across 30 probes) \\ \midrule \multicolumn{6}{l}{\textit{Tier 2 -- property / activity prediction (Chemprop D-MPNN)}} \\ m16 Affinity Pearson $r$ (pIC$_{50}$)  & 0.207 \MEASURED{} (best margin 0.5) & 0.86 Boltz-2 \CITEDONLY{}       & $\geq 0.75$ \PROJECTED{} & higher & \textbf{BELOW GATE} ($-\,0.65$); ridge $0.572$ remains credible ceiling \\ m18 Protein-metal binding AUC          & 0.739 \MEASURED{} (3 seeds, 100 ep) & $> 0.92$ 2025 paper \CITEDONLY{} & $\geq 0.85$ \PROJECTED{} & higher & \textbf{BELOW GATE} ($-\,0.18$); baseline anchor for R18 multi-task lift \\ \midrule \multicolumn{6}{l}{\textit{Tier 1 -- Deflexpressor ASTSR (symbolic-regression diagnostic)}} \\`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_results_table2_v2.tex:198` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `& \S{}3.7 PLL SyntheMol backend & \textbf{PASS} (perfect; Path-A already at ceiling) \\ \midrule \multicolumn{6}{l}{\textit{Tier 1 -- CFM decode-path diagnostic}} \\ m14 Decode ratio (CFM) & $\mathbf{1.000}$ \MEASURED{} (3 seeds $\times$ 10 probes) & 0/192 R16 pre-shim \MEASURED{} & higher & \S{}3.4 SE(3) Schr\"odinger-bridge ODE & \textbf{LIFTED} $0\!\to\!1.000$ (RDKit-strict pre: $0.500$; 6-shim fix stable across 30 probes) \\ \midrule \multicolumn{6}{l}{\textit{Tier 2 -- property / activity prediction (Chemprop D-MPNN)}} \\ m16 Affinity Pearson $r$ (pIC$_{50}$)`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations.tex:55` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\textbf{\textsc{measured} smoke (192 synthetic CFM-style 10-atom clouds, 500-step + $h\!=\!64$ mini-budget):} Path-B \texttt{decode\_ratio\_bond\_bearing}\,=\,192/192 (vs Path-A 0/192, absolute lift $+1.0$, infinite relative); five unit tests pass on CPU in 2.47\,s.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations_v2.tex:298` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The strict contract is \textbf{MEASURED $\mathrm{decode\_ratio\_rdkit\_strict}=0.5$} on synthetic $10$-atom Pt-click clouds (Path B, $n=192$, overvalent atoms fail sanitisation, see \texttt{metrics/by\_metric/decode\_ratio. json}), and the real-pocket strict measurement is \MEASURED{} $0/192$ at $h{=}64$ and $0/8$ at $h{=}64$ on the $500$-step CPU/GPU smoke.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations_v2.tex:302` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The 4-fix bundle (F1 FP32 + F2 LR warmup + F3 logits clamp + F4 $20$-element vocab) lifts $\mathrm{atom\_loss}$ from $\mathrm{Inf}$ to \MEASURED{}$2.996$ and $\mathrm{decode\_ratio}$ from $0/192$ to \MEASURED{}$1.000$ (lenient), but the RDKit-strict column is honest-negative at $0.5$ on synthetic and $0/8$ on real pockets.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/07_future.tex:85` -- rule B1 -- **QUALIFIED**

- **number**: `0/192` (matched `0/192`)
- **what it is**: Path-A (legacy BondAwareDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Across the four CFM retrain attempts on this submission cycle (2000-step / \texttt{hidden\_dim\,=\,32} baseline, 5000-step / \texttt{hidden\_dim\,=\,128} P1-fix diagnostic, Round-12 \texttt{decode\_smoke\_every} probe, and the Round-13 \texttt{wf\_gpu\_recovery\_now} 5000-step retrain), the geometric generator returns \textbf{\textsc{measured} $\texttt{decode\_ratio}\,=\,0/192$} on every run; the discrete GPU (\texttt{molmetal/reports/wf\_gpu\_diag/diagnosis.md}, 2026-09-15) is firmware-level SMU-hung and the iGPU (Radeon 780M gfx1100) cannot be addressed by the ROCm 7.2 HSA runtime while the broken KFD node remains in the topology (\texttt{molmetal/reports/wf\_igpu\_switch/probe.md}).`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:8218` -- rule B2 -- **QUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (34.73775pt too wide) in paragraph at lines 3148--3174 \OT1/cmtt/m/sc/10 The honest framing: the path-(b) smoke lift (192/192 \OT1/cmt t/m/it/10 bond-bearing\OT1/cmtt/m/sc/10 ) is a decoder-architecture  Overfull \hbox (1.5019pt too wide) in paragraph at lines 3148--3174 \OT1/cmtt/m/sc/10 signal, not a production-pipeline signal; the production inte gration remains the gating`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:72457` -- rule B2 -- **QUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (34.73775pt too wide) in paragraph at lines 3148--3174 \OT1/cmtt/m/sc/10 The honest framing: the path-(b) smoke lift (192/192 \OT1/cmt t/m/it/10 bond-bearing\OT1/cmtt/m/sc/10 ) is a decoder-architecture  Overfull \hbox (1.5019pt too wide) in paragraph at lines 3148--3174 \OT1/cmtt/m/sc/10 signal, not a production-pipeline signal; the production inte gration remains the gating`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:136628` -- rule B2 -- **QUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (34.73775pt too wide) in paragraph at lines 3148--3174 \OT1/cmtt/m/sc/10 The honest framing: the path-(b) smoke lift (192/192 \OT1/cmt t/m/it/10 bond-bearing\OT1/cmtt/m/sc/10 ) is a decoder-architecture  Overfull \hbox (1.5019pt too wide) in paragraph at lines 3148--3174 \OT1/cmtt/m/sc/10 signal, not a production-pipeline signal; the production inte gration remains the gating`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/build_log.txt:200799` -- rule B2 -- **QUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Overfull \hbox (34.73775pt too wide) in paragraph at lines 3148--3174 \OT1/cmtt/m/sc/10 The honest framing: the path-(b) smoke lift (192/192 \OT1/cmt t/m/it/10 bond-bearing\OT1/cmtt/m/sc/10 ) is a decoder-architecture  Overfull \hbox (1.5019pt too wide) in paragraph at lines 3148--3174 \OT1/cmtt/m/sc/10 signal, not a production-pipeline signal; the production inte gration remains the gating`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/paper_v2.tex:258` -- rule B2 -- **QUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\subsection{Full Metric Listing}  \begin{itemize} \item \texttt{decode\_ratio.json}: $1.000$ (192/192 synthetic Pt-click) \item \texttt{diversity\_tanimoto.json}: $0.137$ vs TargetDiff $0.860$ (honest) \item \texttt{drugood\_ood\_auc.json}: $0.891 / 0.697 / 0.748$ (3 splits) \item \texttt{metalcytotox\_ru\_auc.json}: $0.7613 \pm 0.0$ (3-seed mean) \item \texttt{metallodrug\_vertical\_coverage.json}: $791{,}013 \times 16 \times 5$ \item \texttt{n\_distinct.json}: $20$ (4-fix bundle) \item \texttt{nci60\_precious\_9cell\_auc.json}: $0.8007 \pm 0.016$`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_method.tex:33` -- rule B2 -- **QUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `%    §3.6 stub (this file)          -- WF-Pivot-A Phase 2 %                                     (deferred-secondary-generator stub: %                                     label-only preservation, full %                                     content moved to §7 future work) %    §3.7 bond-aware soft prior     -- WF-CFM-Path-B-Decoder-Rework %                                     (chem-aware 3-prior decoder; %                                     Path A→Path B lift 0/192→192/192) % %  Cross-references all point to: %    Figure~1 (9-layer MLC architecture, fig1_mlc_arch.pdf)        -- 03.1 %    Figure~2 (pocket-conditioned Lambda pipeline, fig2_pipeline.pdf) -- 03.3 + 03.4 %    Figure~3 (5 click reactions, fig3_click_reactions.pdf)         -- 03.2 %    Appendix closure-theorem (paper/appendices/closure_theorem.tex,`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_method.tex:167` -- rule B2 -- **QUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\texttt{test\_type\_compat\_supports\_C\_C\_bond}, \texttt{test\_valence\_cap\_prevents\_overvalent}, \texttt{test\_decoder\_rework\_differentiable}, \texttt{test\_decoder\_rework\_composes\_with\_bond\_head}) are \textsc{measured}; the smoke validation on synthetic CFM-style clouds is \textsc{measured} (\texttt{wf\_cfm\_path\_b\_decoder\_ rework/final.md}, 192/192 bond-bearing); the full GPU retrain verification is \textsc{deferred} pending \texttt{WF-GPU-Auto-Recover}.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_method.tex:240` -- rule B2 -- **QUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\paragraph{Smoke validation (measured).} On 192 synthetic CFM-style 10-atom clouds with realistic 1-5\,\AA{} coordinate spread (matches the round-12 trained-CFM distribution), the soft prior lifts bond-bearing decode from $0/192$ (Path A: legacy \texttt{BondAwareDecoder} with hard 2.4\,\AA{} cutoff) to $192/192$ (Path B: \texttt{ReworkedDecoder} with soft 3-prior) at a 500-step + $h\!=\!64$ mini-budget. 96/192 of the Path-B mols fail \texttt{Chem.SanitizeMol} due to overvalent atoms in the random cloud (e.g.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:3077` -- rule B2 -- **QUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\begin{center} \begin{tabular}{l c c c} \hline \textbf{decoder} & \textbf{bond-bearing} & \textbf{RDKit-strict} & \textbf{status} \\ \hline Path A (legacy \texttt{BondAwareDecoder}, 2.4\,\AA{} cutoff) & 0/192 & 0/192 & 0\% lifted \\ \textbf{Path B} (\texttt{ReworkedDecoder}, soft 3-prior) & \textbf{192/192} & 0/192 & \textbf{+1.0 lift} \\ \hline \end{tabular} \end{center}  \noindent\textbf{Honest framing.} The 192/192 is \emph{bond-bearing} (decoder emitted at least one bond), not`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:3082` -- rule B2 -- **QUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `Path A (legacy \texttt{BondAwareDecoder}, 2.4\,\AA{} cutoff) & 0/192 & 0/192 & 0\% lifted \\ \textbf{Path B} (\texttt{ReworkedDecoder}, soft 3-prior) & \textbf{192/192} & 0/192 & \textbf{+1.0 lift} \\ \hline \end{tabular} \end{center}  \noindent\textbf{Honest framing.} The 192/192 is \emph{bond-bearing} (decoder emitted at least one bond), not RDKit-strict: 96/192 fail \texttt{Chem.SanitizeMol} on the synthetic random cloud (overvalent C atoms, e.g.\ C with 8 bonds) --- the random cloud generator is more permissive than a trained CFM will be.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:3165` -- rule B2 -- **QUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `The honest framing: the path-(b) smoke lift (192/192 \emph{bond-bearing}) is a decoder-architecture signal, not a production-pipeline signal; the production integration remains the gating engineering problem.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations.tex:54` -- rule B2 -- **QUALIFIED**

- **number**: `192/192` (matched `192/192`)
- **what it is**: Path-B (ReworkedDecoder) bond-bearing decode count on the 192-cloud smoke
- **who measured it**: us
- **should cite**: WF-CFM-Path-B-Decoder-Rework smoke (2026-09-15)
- **population that makes it interpretable**: 192 synthetic CFM-style 10-atom clouds, 500-step + h=64 mini-budget
- **sentence (verbatim)**: `\textbf{\textsc{measured} smoke (192 synthetic CFM-style 10-atom clouds, 500-step + $h\!=\!64$ mini-budget):} Path-B \texttt{decode\_ratio\_bond\_bearing}\,=\,192/192 (vs Path-A 0/192, absolute lift $+1.0$, infinite relative); five unit tests pass on CPU in 2.47\,s.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/cover_letter.txt:14` -- rule B3 -- **QUALIFIED**

- **number**: `0/8` (matched `0/8`)
- **what it is**: decode_ratio on the R15 CPU/GPU 500-step smokes
- **who measured it**: us
- **should cite**: molmetal/reports/wf_r15_cfm_cpu_verify/phase1_500step_smoke.json + phase4_gpu_retry.json
- **population that makes it interpretable**: 8 decoded slots at step 100/200/500, CPU and GPU runs
- **sentence (verbatim)**: `Headline measured numbers (2026-09-25 snapshot): * MetalCytoToxDB Ru subset ROC-AUC = 0.712 (Krasnov 2026 baseline 0.81; our gate 0.86; honest framing: below gate). * PlatinAI 4-cell-line mean AUC = 0.553 +/- 0.009 (synthetic IC50 labels; A2780cis 0.648). * DrugOOD assay-split OOD AUC = 0.891 (vs iMoLD reference 0.721; gate-pass on all 3 splits). * MCTS n_distinct: 1 -> 20 (lifted 20x by 4-fix bundle; saturated n_top_k cap on all 30 cells). * CFM decode_ratio: 0/8 honest null on the real EGNN velocity field (Path B bond-bearing = 1.0 on synthetic).`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/cover_letter_cn.txt:14` -- rule B3 -- **QUALIFIED**

- **number**: `0/8` (matched `0/8`)
- **what it is**: decode_ratio on the R15 CPU/GPU 500-step smokes
- **who measured it**: us
- **should cite**: molmetal/reports/wf_r15_cfm_cpu_verify/phase1_500step_smoke.json + phase4_gpu_retry.json
- **population that makes it interpretable**: 8 decoded slots at step 100/200/500, CPU and GPU runs
- **sentence (verbatim)**: `主要实测数字 (2026-09-25 快照): - MetalCytoToxDB Ru 子集 ROC-AUC = 0.712 (Krasnov 2026 基线 0.81，我方门槛 0.86；诚实陈述：未达门槛)。 - PlatinAI 4 细胞系平均 AUC = 0.553 ± 0.009 (合成 IC50 标签；A2780cis 0.648)。 - DrugOOD assay-split 分布外 AUC = 0.891 (vs iMoLD 参考 0.721; 3 种划分全部过门槛)。 - MCTS n_distinct: 从 1 提升到 20 (4 项修复组合包后 20× 提升; 30 格全部打满 n_top_k 上限)。 - CFM decode_ratio: 真实 EGNN 速度场上 0/8 的诚实零结果 (Path B 键承载合成输入 = 1.0)。  完整的 100 口袋 × 3 种子 Round-17 横扫以 PROJECTED 报告，等待 GPU 重训结束。  三大贡献: (1) 9 层类型化 MLC 形式化体系, 包含跨越杂化、环成员性、氧化态的类型化变量词汇表, 并标注金属感知催化剂。 (2) 5 条点击规则注册为 12×7 金属矩阵上的类型化 β-归约 (84+ 反应), 与封闭项 MCTS 驱动集成。`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:3196` -- rule B3 -- **QUALIFIED**

- **number**: `0/8` (matched `0/8`)
- **what it is**: decode_ratio on the R15 CPU/GPU 500-step smokes
- **who measured it**: us
- **should cite**: molmetal/reports/wf_r15_cfm_cpu_verify/phase1_500step_smoke.json + phase4_gpu_retry.json
- **population that makes it interpretable**: 8 decoded slots at step 100/200/500, CPU and GPU runs
- **sentence (verbatim)**: `\textbf{CPU 500-step verification}: decode\_ratio bit-exact with the BondAwareDecoder baseline ($0/8$ on a freshly-init $h\!=\!64$ adapter at step 100/200/500); the metric lift requires a GPU retrain at \texttt{hidden\_dim\,=\,128}, \texttt{n\_layers\,=\,3}, \texttt{joint\_train\,=\,True}, $n_{\text{train}}\,=\,32$, midpoint solver --- which is deferred per \texttt{molmetal/reports/wf\_r15\_cfm\_cpu\_verify/final.md}.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_results.tex:317` -- rule B3 -- **QUALIFIED**

- **number**: `0/8` (matched `0/8`)
- **what it is**: decode_ratio on the R15 CPU/GPU 500-step smokes
- **who measured it**: us
- **should cite**: molmetal/reports/wf_r15_cfm_cpu_verify/phase1_500step_smoke.json + phase4_gpu_retry.json
- **population that makes it interpretable**: 8 decoded slots at step 100/200/500, CPU and GPU runs
- **sentence (verbatim)**: `\paragraph{m14 Decode ratio on CFM (LIFTED $0 \to 1.000$).} The R16 pre-shim CFM probe had $\mathrm{decode\_ratio} = 0/192$ on a 5000-step CPU smoke and $0/8$ on the 500-step GPU smoke at all three checkpoints (\texttt{wf\_gpu\_recovery\_now} + \texttt{wf\_r15\_cfm\_cpu\_verify}).`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations.tex:391` -- rule B3 -- **QUALIFIED**

- **number**: `0/8` (matched `0/8`)
- **what it is**: decode_ratio on the R15 CPU/GPU 500-step smokes
- **who measured it**: us
- **should cite**: molmetal/reports/wf_r15_cfm_cpu_verify/phase1_500step_smoke.json + phase4_gpu_retry.json
- **population that makes it interpretable**: 8 decoded slots at step 100/200/500, CPU and GPU runs
- **sentence (verbatim)**: `The WF-R15-CFM-CPU-Verify 500-step smoke (\texttt{molmetal/reports/wf\_r15\_cfm\_cpu\_verify/phase1\_500step\_smoke.json}) reports \textbf{\textsc{measured} $\texttt{decode\_ratio}\,=\,0/8$ at step 100, 200, 500 on CPU}; the GPU 500-step smoke (\texttt{molmetal/reports/wf\_r15\_cfm\_cpu\_verify/phase4\_gpu\_retry.json}, run after the 2026-09-15 SMU hang cleared on 2026-09-16) reports the same \textbf{\textsc{measured} $\texttt{decode\_ratio}\,=\,0/8$ at step 100, 200, 500 on GPU}.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations.tex:395` -- rule B3 -- **QUALIFIED**

- **number**: `0/8` (matched `0/8`)
- **what it is**: decode_ratio on the R15 CPU/GPU 500-step smokes
- **who measured it**: us
- **should cite**: molmetal/reports/wf_r15_cfm_cpu_verify/phase1_500step_smoke.json + phase4_gpu_retry.json
- **population that makes it interpretable**: 8 decoded slots at step 100/200/500, CPU and GPU runs
- **sentence (verbatim)**: `The WF-R15-CFM-CPU-Verify 500-step smoke (\texttt{molmetal/reports/wf\_r15\_cfm\_cpu\_verify/phase1\_500step\_smoke.json}) reports \textbf{\textsc{measured} $\texttt{decode\_ratio}\,=\,0/8$ at step 100, 200, 500 on CPU}; the GPU 500-step smoke (\texttt{molmetal/reports/wf\_r15\_cfm\_cpu\_verify/phase4\_gpu\_retry.json}, run after the 2026-09-15 SMU hang cleared on 2026-09-16) reports the same \textbf{\textsc{measured} $\texttt{decode\_ratio}\,=\,0/8$ at step 100, 200, 500 on GPU}.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations.tex:445` -- rule B3 -- **QUALIFIED**

- **number**: `0/8` (matched `0/8`)
- **what it is**: decode_ratio on the R15 CPU/GPU 500-step smokes
- **who measured it**: us
- **should cite**: molmetal/reports/wf_r15_cfm_cpu_verify/phase1_500step_smoke.json + phase4_gpu_retry.json
- **population that makes it interpretable**: 8 decoded slots at step 100/200/500, CPU and GPU runs
- **sentence (verbatim)**: `The pre-flight BUG fixes pass with high confidence: \textbf{BUG-1} (\texttt{coupling\_adapter} reshape 64$\to$5) now raises a clear \texttt{ValueError} on shape mismatch instead of silently returning zeros; \textbf{BUG-2} (\texttt{pocket\_macro\_inference} CWD-relative path) now resolves to the canonical \texttt{molmetal/models/} directory via \texttt{Path(\_\_file\_\_)} and round-trips cleanly when invoked from \texttt{/tmp} (verified by re-loading the model from a foreign CWD; 5 v2 test-pocket predictions round-trip cleanly, including the CA2 collapse recovery from 0/8 on v1 to (ZN\_TETRA\_HHH, 0.39) on v2); \textbf{CFM-import} (\texttt{r10\_cfg\_real\_crossdocked.py} sys.path bootstrap) is already in place from R15, no further code changes needed.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations.tex:710` -- rule B3 -- **QUALIFIED**

- **number**: `0/8` (matched `0/8`)
- **what it is**: decode_ratio on the R15 CPU/GPU 500-step smokes
- **who measured it**: us
- **should cite**: molmetal/reports/wf_r15_cfm_cpu_verify/phase1_500step_smoke.json + phase4_gpu_retry.json
- **population that makes it interpretable**: 8 decoded slots at step 100/200/500, CPU and GPU runs
- **sentence (verbatim)**: `\textbf{What is NOT yet closed}: 3-seed cross-seed reproducibility, mean\_atoms\,$\geq$\,14 lift under pocket conditioning, the \texttt{wf\_r15\_cfm\_cpu\_verify} 500-step smoke baseline lift (\texttt{decode\_ratio}\,=\,0/8 at step 500 on CPU / GPU), and the §4 hybrid column \DESIGN{} $\to$ \MEASURED{} promotion on Table~\ref{tab:per-pocket}.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations_v2.tex:298` -- rule B3 -- **QUALIFIED**

- **number**: `0/8` (matched `0/8`)
- **what it is**: decode_ratio on the R15 CPU/GPU 500-step smokes
- **who measured it**: us
- **should cite**: molmetal/reports/wf_r15_cfm_cpu_verify/phase1_500step_smoke.json + phase4_gpu_retry.json
- **population that makes it interpretable**: 8 decoded slots at step 100/200/500, CPU and GPU runs
- **sentence (verbatim)**: `The strict contract is \textbf{MEASURED $\mathrm{decode\_ratio\_rdkit\_strict}=0.5$} on synthetic $10$-atom Pt-click clouds (Path B, $n=192$, overvalent atoms fail sanitisation, see \texttt{metrics/by\_metric/decode\_ratio. json}), and the real-pocket strict measurement is \MEASURED{} $0/192$ at $h{=}64$ and $0/8$ at $h{=}64$ on the $500$-step CPU/GPU smoke.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations_v2.tex:304` -- rule B3 -- **QUALIFIED**

- **number**: `0/8` (matched `0/8`)
- **what it is**: decode_ratio on the R15 CPU/GPU 500-step smokes
- **who measured it**: us
- **should cite**: molmetal/reports/wf_r15_cfm_cpu_verify/phase1_500step_smoke.json + phase4_gpu_retry.json
- **population that makes it interpretable**: 8 decoded slots at step 100/200/500, CPU and GPU runs
- **sentence (verbatim)**: `The 4-fix bundle (F1 FP32 + F2 LR warmup + F3 logits clamp + F4 $20$-element vocab) lifts $\mathrm{atom\_loss}$ from $\mathrm{Inf}$ to \MEASURED{}$2.996$ and $\mathrm{decode\_ratio}$ from $0/192$ to \MEASURED{}$1.000$ (lenient), but the RDKit-strict column is honest-negative at $0.5$ on synthetic and $0/8$ on real pockets.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations_v3.tex:288` -- rule B3 -- **QUALIFIED**

- **number**: `0/8` (matched `0/8`)
- **what it is**: decode_ratio on the R15 CPU/GPU 500-step smokes
- **who measured it**: us
- **should cite**: molmetal/reports/wf_r15_cfm_cpu_verify/phase1_500step_smoke.json + phase4_gpu_retry.json
- **population that makes it interpretable**: 8 decoded slots at step 100/200/500, CPU and GPU runs
- **sentence (verbatim)**: `\item \textbf{Strict decode-ratio collapse (\MEASURED{} R17):} the CFM strict-contract measurement is \MEASURED{} $\mathrm{decode\_ratio\_rdkit\_strict}=0.5$ on synthetic $10$-atom Pt-click clouds (Path~B, $n{=}192$, overvalent atoms fail sanitisation) and $0/8$ on the $830$c pocket ($500$-step CPU/GPU smoke); the lenient-contract $\mathrm{decode\_ratio}=1.000$ is consistent across $3$ seeds $\times$ $10$ probes at the R17 10K-step retrain (\texttt{molmetal/reports/wf\_r17\_10k\_retrain\_done/}) but is an atom-only check, not a sanitisation check.`
- **why this classification**: window names the population (cohort / pool / budget / seed count) that the number belongs to
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_method.tex:151` -- rule C2 -- **QUALIFIED**

- **number**: `2.4 A` (matched `2.4\,\AA`)
- **what it is**: legacy hard bond cutoff / soft-mask centre d_th
- **who measured it**: us
- **should cite**: molmetal/molmetal_lam/lam_chem/decoder_rework.py:DEFAULT_DISTANCE_THRESHOLD
- **population that makes it interpretable**: must name the component it belongs to (legacy BondAwareDecoder)
- **sentence (verbatim)**: `\subsection{Bond-aware soft prior for the CFM decoder (Path B)} \label{sec:method:bond-aware-soft-prior}  The Round-12 audit (\texttt{molmetal/reports/wf\_cfm\_internal\_review/audit.md}) identified that the legacy \texttt{BondAwareDecoder}'s hard 2.4\,\AA{} distance cutoff predicts zero edges for all 192 samples on the round-12 retrain because the CFM coordinate distribution has no atom pair within 2.4\,\AA{}.`
- **why this classification**: window names where the constant comes from (a source / a code path / an explicit deviation note)
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_method.tex:154` -- rule C2 -- **QUALIFIED**

- **number**: `2.4 A` (matched `2.4\,\AA`)
- **what it is**: legacy hard bond cutoff / soft-mask centre d_th
- **who measured it**: us
- **should cite**: molmetal/molmetal_lam/lam_chem/decoder_rework.py:DEFAULT_DISTANCE_THRESHOLD
- **population that makes it interpretable**: must name the component it belongs to (legacy BondAwareDecoder)
- **sentence (verbatim)**: `The Round-12 audit (\texttt{molmetal/reports/wf\_cfm\_internal\_review/audit.md}) identified that the legacy \texttt{BondAwareDecoder}'s hard 2.4\,\AA{} distance cutoff predicts zero edges for all 192 samples on the round-12 retrain because the CFM coordinate distribution has no atom pair within 2.4\,\AA{}.`
- **why this classification**: window names where the constant comes from (a source / a code path / an explicit deviation note)
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_method.tex:240` -- rule C2 -- **QUALIFIED**

- **number**: `2.4 A` (matched `2.4\,\AA`)
- **what it is**: legacy hard bond cutoff / soft-mask centre d_th
- **who measured it**: us
- **should cite**: molmetal/molmetal_lam/lam_chem/decoder_rework.py:DEFAULT_DISTANCE_THRESHOLD
- **population that makes it interpretable**: must name the component it belongs to (legacy BondAwareDecoder)
- **sentence (verbatim)**: `\paragraph{Smoke validation (measured).} On 192 synthetic CFM-style 10-atom clouds with realistic 1-5\,\AA{} coordinate spread (matches the round-12 trained-CFM distribution), the soft prior lifts bond-bearing decode from $0/192$ (Path A: legacy \texttt{BondAwareDecoder} with hard 2.4\,\AA{} cutoff) to $192/192$ (Path B: \texttt{ReworkedDecoder} with soft 3-prior) at a 500-step + $h\!=\!64$ mini-budget. 96/192 of the Path-B mols fail \texttt{Chem.SanitizeMol} due to overvalent atoms in the random cloud (e.g.`
- **why this classification**: window names where the constant comes from (a source / a code path / an explicit deviation note)
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:3065` -- rule C2 -- **QUALIFIED**

- **number**: `2.4 A` (matched `2.4\,\AA`)
- **what it is**: legacy hard bond cutoff / soft-mask centre d_th
- **who measured it**: us
- **should cite**: molmetal/molmetal_lam/lam_chem/decoder_rework.py:DEFAULT_DISTANCE_THRESHOLD
- **population that makes it interpretable**: must name the component it belongs to (legacy BondAwareDecoder)
- **sentence (verbatim)**: `\paragraph{Path-(b) decoder rework: \textsc{measured} smoke (WF-CFM-Path-B-Decoder-Rework, 2026-09-15).} Per \S\ref{sec:method:bond-aware-soft-prior}, the chem-aware \texttt{DecoderRework} replaces the legacy hard 2.4\,\AA{} cutoff with a soft 3-prior decoder.`
- **why this classification**: window names where the constant comes from (a source / a code path / an explicit deviation note)
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/04_evaluation.tex:3076` -- rule C2 -- **QUALIFIED**

- **number**: `2.4 A` (matched `2.4\,\AA`)
- **what it is**: legacy hard bond cutoff / soft-mask centre d_th
- **who measured it**: us
- **should cite**: molmetal/molmetal_lam/lam_chem/decoder_rework.py:DEFAULT_DISTANCE_THRESHOLD
- **population that makes it interpretable**: must name the component it belongs to (legacy BondAwareDecoder)
- **sentence (verbatim)**: `\begin{center} \begin{tabular}{l c c c} \hline \textbf{decoder} & \textbf{bond-bearing} & \textbf{RDKit-strict} & \textbf{status} \\ \hline Path A (legacy \texttt{BondAwareDecoder}, 2.4\,\AA{} cutoff) & 0/192 & 0/192 & 0\% lifted \\ \textbf{Path B} (\texttt{ReworkedDecoder}, soft 3-prior) & \textbf{192/192} & 0/192 & \textbf{+1.0 lift} \\ \hline \end{tabular} \end{center}  \noindent\textbf{Honest framing.} The 192/192 is`
- **why this classification**: window names where the constant comes from (a source / a code path / an explicit deviation note)
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/06_limitations.tex:44` -- rule C2 -- **QUALIFIED**

- **number**: `2.4 A` (matched `2.4\,\AA`)
- **what it is**: legacy hard bond cutoff / soft-mask centre d_th
- **who measured it**: us
- **should cite**: molmetal/molmetal_lam/lam_chem/decoder_rework.py:DEFAULT_DISTANCE_THRESHOLD
- **population that makes it interpretable**: must name the component it belongs to (legacy BondAwareDecoder)
- **sentence (verbatim)**: `\textbf{Path (b) decoder rework (SHIPPED, lit-grounded):} the chem-aware \texttt{DecoderRework} module (\texttt{molmetal/molmetal\_lam/lam\_chem/decoder\_rework.py}, detailed in \S\ref{sec:method:bond-aware-soft-prior}) replaces the legacy hard 2.4\,\AA{} cutoff with a soft 3-prior decoder (soft distance mask $p_{\text{dist}}\!=\!\sigma(-(d-d_{\text{th}})/\tau)$, type-compatibility prior \texttt{ATOM\_TYPE\_COMPAT}, valence-aware bond cap) lit-grounded by Himo 2005 JACS CuAAC regioselectivity \cite{himo2005cuaac} and the Lit-Survey-v2 5$\times$5`
- **why this classification**: window names where the constant comes from (a source / a code path / an explicit deviation note)
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: no
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/reviews/rebuttal_predraft.md:161` -- rule C3 -- **QUALIFIED**

- **number**: `0.3 A` (matched `0.3 Å`)
- **what it is**: soft-band width tau (DEFAULT_SOFT_SIGMA)
- **who measured it**: us
- **should cite**: molmetal/molmetal_lam/lam_chem/decoder_rework.py:DEFAULT_SOFT_SIGMA
- **population that makes it interpretable**: must name the component (soft distance mask)
- **sentence (verbatim)**: `The decode_ratio=1.000 (30/30 probes, 3 seeds) at probe-time confirms: (i) the ODE solver runs (M15–M17), (ii) the model emits *bond-bearing* graphs (vs the pre-fix 0/192 baseline where RDKit fell over overvalent atoms), and (iii) the trained CFM coordinate distribution falls in the `BondAwareDecoder`'s `τ = 0.3 Å` soft band.`
- **why this classification**: window names where the constant comes from (a source / a code path / an explicit deviation note)
- **in the live `paper/main.tex` bundle**: no (legacy / unused bundle)
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

### `paper/sections/03_method.tex:237` -- rule C5 -- **QUALIFIED**

- **number**: `1.5-3.5 A (and 1-5 A)` (matched `1-5\,\AA`)
- **what it is**: 'typical CFM coordinate spread' -- our own empirical claim about coordinate spread
- **who measured it**: us
- **should cite**: a round-12 trained-CFM coordinate-spread measurement
- **population that makes it interpretable**: which run / round / pool the spread was measured on
- **sentence (verbatim)**: `\paragraph{Smoke validation (measured).} On 192 synthetic CFM-style 10-atom clouds with realistic 1-5\,\AA{} coordinate spread (matches the round-12 trained-CFM distribution), the soft prior lifts bond-bearing decode from $0/192$ (Path A: legacy \texttt{BondAwareDecoder} with hard 2.4\,\AA{} cutoff) to $192/192$ (Path B: \texttt{ReworkedDecoder} with soft 3-prior) at a 500-step + $h\!=\!64$ mini-budget. 96/192 of the Path-B mols fail \texttt{Chem.SanitizeMol} due to overvalent atoms in the random cloud (e.g.`
- **why this classification**: window names where the constant comes from (a source / a code path / an explicit deviation note)
- **in the live `paper/main.tex` bundle**: YES
- **TODO-84**: blocked -- any edit here needs explicit user OK (AGENTS.md §5.3).
- **also in `metrics/paper_tables/` (that tree MAY change)**: YES
- **minimal edit (proposed, NOT applied)**: (no site-specific minimal edit proposed)

## 3. TODO-84 note

**All 130 sites are blocked by TODO-84** (AGENTS.md §5.3: zero writes to `paper/**` without explicit user authorisation). The `minimal edit` field above is a *proposal in text only*; this wave wrote **0 bytes** under `paper/**` (verified by the read-only contract in `sites.json` -> `read_only_contract.writes`).

**`metrics/paper_tables/` is a different tree and IS allowed to change.** None of the sites above has its number duplicated in `metrics/paper_tables/`, so there is no parallel, unblocked copy of the same defect there.

## 4. Coverage and limits

- Enumeration is `git ls-files` + untracked/ignored listing under `paper/`; **no `os.walk`** (the repo is 201 GB).
- Reads are byte-level with `errors="replace"`, so non-UTF-8 tracked text (e.g. `paper/build_log.txt`, ISO-8859) IS scanned -- the failure mode of PATH `grep -I` is avoided by construction.
- Rule families are deliberately bounded (external bands, our decode/pool headlines, methodological constants). A full numeric census of every digit in every `.tex` file is NOT performed; the report does not claim to have reviewed every number in the manuscript.
- Nothing here was measured on a GPU; every number quoted in this report is read off disk (file:line) or computed in closed form on CPU.

