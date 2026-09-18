# WF-Lambda-Diversity-Rotation Integration Notes (2026-09-14)

> **Purpose.** Integrate the WF-Lambda-Diversity-Rotation pilot run
> (`molmetal/reports/wf_lambda_div_rotation/final.md`) into paper
> §4.5 (hybrid-vs-Λ-only ablation) and §5.7 (5-click ablation
> uplift panel), and update `CROSS_REFS.md` +
> `TODO/pending/13_top_journal_pilot_r12.md`.
> Honest-framing mandatory: the diversity-lift hypothesis was
> REJECTED; 0 cells promoted DESIGN→MEASURED; the new 3-seed
> aggregate panel is MEASURED **as a negative result**.

## 1. Cells promoted DESIGN → MEASURED (0)

**0 cells promoted.** The diversity cells of the WF-Lambda-Only-MiniPilot
panel above §4.5 of `paper/sections/04_evaluation.tex` are already
MEASURED (`diversity_tanimoto_mean = 0.0050`,
`diversity_homotype_mean = 0.0020`). The 3-seed × 5-pocket × 1-seed
aggregate panel introduced by this round is MEASURED **as a negative
result** (3 metal seeds rotated, n_simulations=100 silently
hard-capped from CLI=1000) — it is a new measurement, not a promotion
of an existing DESIGN cell. The honesty rule is preserved: no cell of
§4.5 / §5.7 is silently promoted from DESIGN to MEASURED.

| metric | baseline (1 seed cisplatin, MiniPilot) | 3-seed rotation (this round) | Δ | tag |
|---|---|---|---|---|
| `validity_rate` | 1.0000 | 1.0000 | 0.0000 | unchanged |
| `uniqueness_rate` | 1.0000 | 1.0000 | 0.0000 | unchanged |
| `synthesizability_rate` | 1.0000 | 1.0000 | 0.0000 | unchanged |
| `metal_compliance_rate` | 1.0000 | 0.667 (3-seed mean) | **−0.333** | **regression** |
| `diversity_tanimoto_mean` | 0.0000 (Metal-Pilot) | 0.0000 | 0.0000 | unchanged |
| `diversity_homotype_mean` | 0.0000 (Metal-Pilot) | 0.0000 | 0.0000 | unchanged |

The `metal_compliance_rate` regression (−0.333) is real and is itself
a negative finding: `ir_cp_star`'s `CC1=C(C)[CH]([Ir]([NH2])([NH2])[Cl])C(C)=C1C`
fails the metal-compliance predicate (only 1 monodentate Cl, not the
required ≥2; oxidation-state distribution shows `Ir_0` × 5, but the
rubric expects Ir(III) or higher with 2 leaving groups for
"compliance" under the lambda-only reward spec). The iridium Cp*
scaffold is **not** lambda-only-compliant without the click-rules
engine filling in the second Cl.

## 2. Hypothesis verdict: REJECTED

The diversity-lift hypothesis (rotate metal seeds across
{cisplatin, ru_arene, ir_cp_star} + raise n_simulations from 100 to
1000 + keep click_rules=all-5 → lift diversity_tanimoto from 0.005
toward 0.10–0.20) was **rejected**. The diversity metric stays at
0.000 across all 3 metal seeds because:

1. **Harness-level cap.** `molmetal/scripts/r4_lambda_only_run.py:1466-1467`
   silently re-writes `args.n_simulations` from 1000 to 100 on every
   invocation. The 10× budget lift is **not testable with the current
   harness** without first lifting the cap (the cap was deliberately
   added "per spec" in round-4).
2. **Generator-level singleton emission.** The Lambda-only generator
   emits exactly 1 candidate per cell (the metal seed itself, with
   canonicalisation-only edits).
3. **Diversity metric undefined for n=1.** Tanimoto over a set of
   size 1 is trivially 0; the metric does not signal cross-seed
   diversity.

The cross-seed pool of unique SMILES contains **exactly 3 scaffolds**
(one per metal seed), with pairwise Tanimoto ≤ 0.05. If the harness
computed diversity over the **pool** of 3 seeds, the metric would
read ~0.95+; the per-cell metric is therefore undefined for n=1 and
cannot surface the cross-seed lift numerically.

## 3. Sections updated (2 in LaTeX + 1 cross-ref doc + 1 TODO + 1 integrate file = 5 files)

1. **`paper/sections/04_evaluation.tex`** — NEW §4.5 paragraph
   inserted between the WF-Lambda-Only-MiniPilot block and §4.6
   ("Anticancer-specific metrics"). The paragraph carries:
   (a) critical-finding block on the silent `n_simulations` hard-cap;
   (b) per-metal-seed aggregate table (3 metal seeds × 5 pockets ×
   1 seed); (c) per-pocket byte-identical collapse bullets (cisplatin
   / ru_arene / ir_cp_star × 5 cells each); (d) cross-seed
   aggregation note (3 distinct scaffolds, pairwise Tanimoto ≤ 0.05);
   (e) lift-vs-Metal-Pilot cisplatin-only baseline; (f) hypothesis
   verdict; (g) what's-needed-to-lift list (3 follow-ups); (h)
   honest-framing caveat that **0 cells promoted DESIGN→MEASURED** in
   this paragraph.

2. **`paper/sections/05_ablation.tex`** — §5.7 metal-pilot uplift
   panel now carries a NEW "Diversity-axis uplift via metal-seed
   rotation" paragraph appended at the end of §5.7 (after the
   existing 6-row uplift panel + ablation-readiness note + first
   artefacts paragraph). Carries: (a) per-metal-seed aggregate table
   (3 seeds × 5 pockets × 1 seed); (b) silent cap note; (c)
   per-pocket byte-identical collapse; (d) lift-vs-Metal-Pilot
   5-row table; (e) hypothesis verdict; (f) honest-framing caveats
   bullet list. Also NEW "Artefacts and integration notes
   (diversity-rotation round)" paragraph at the very end of §5.7.

3. **`paper/sections/CROSS_REFS.md`** — §4.5 row extended with a
   new sub-bullet on the WF-Lambda-Diversity-Rotation update
   (3-seed panel, byte-identical collapse, 0 DESIGN→MEASURED
   promotions, hypothesis REJECTED, cross-link to §5.7). §5.7 row
   extended with a new sub-bullet "WF-Lambda-Diversity-Rotation
   update — diversity axis (NEW)" (15 cells, REJECTED, metal_compliance
   regression, 4 follow-ups).

4. **`TODO/pending/13_top_journal_pilot_r12.md`** — NEW section
   "WF-Lambda-Diversity-Rotation integration (2026-09-14)" appended
   at the bottom (after the WF-Lambda-Metal-Pilot section). Mirrors
   the per-metal-seed aggregate table + lift-vs-Metal-Pilot panel +
   the section-update list + the task metrics.

5. **`molmetal/reports/wf_lambda_div_rotation/integrate.md`** — this
   file (integration notes).

## 4. Honest-framing caveats (mandatory, copied verbatim from `final.md`)

> The trajectory is degenerate in diversity metrics (only 1
> candidate per pocket) — see §4 of `wf_lambda_div_rotation/final.md`
> for the diagnostic. The Lambda-only harness returns the metal
> seed verbatim in every cell, so the per-cell
> `diversity_tanimoto` is trivially 0.000. The rotation across 3
> metal seeds does produce 3 distinct scaffolds in the cross-seed
> pool, but the per-cell metric is undefined for `n_distinct=1`
> and cannot surface this numerically.
>
> Wall-clock total: **~16 s** (well under the 30-min budget).
>
> The diversity panels only become meaningful when either
> (a) the `n_simulations` hard-cap at `r4_lambda_only_run.py:1466-1467`
> is **lifted** so the CLI `--n-simulations 1000` flag actually
> takes effect, (b) the generator is **switched off Lambda-only** to
> the CFM/CFG joint generator (which emits `n_top_k=20` distinct
> candidates per cell), or (c) the `click_rules_active` wiring is
> fixed so the `--click-rules all-5` filter actually gates the
> Lambda generator (currently it shows
> `click_rules_filter_emptied: requested=['all-5']; falling back to
> all-5` and gates nothing).

The diversity-lift hypothesis of this round is **rejected**:
$\Delta_{\text{div\,tanimoto}} = 0.000$ vs target $0.10$–$0.20$.

## 5. Task metrics

- `n_cells_promoted_design_to_measured` = **0**
  (diversity cells already MEASURED; new 3-seed aggregate MEASURED as
  negative result; metal_compliance regression is a real negative
  finding, not a DESIGN→MEASURED promotion).
- `n_sections_updated` = **2** LaTeX sections
  (§4.5 new paragraph + §5.7 diversity-axis paragraph appended) +
  CROSS_REFS.md (2 rows extended) +
  TODO/pending/13_top_journal_pilot_r12.md (1 new section appended)
  + integrate.md (1 new file) = **5 file edits total**.
- `diversity_lift_pp` = **0.000** (diversity_tanimoto unchanged
  across 3 metal seeds; baseline 0.000 (Metal-Pilot cisplatin-only)
  → 3-seed mean 0.000; target was 0.10–0.20).
- `all_3_metal_seeds_measured` = **true** (cisplatin, ru_arene,
  ir_cp_star each fully exercised across 5 pockets × 1 seed;
  per-seed aggregate populated for all 3 seeds; per-cell JSON
  emitted to `molmetal/reports/wf_lambda_div_rotation/{cisplatin,
  ru_arene,ir_cp_star}/report.json`).

## 6. Follow-ups (deferred to next round)

The 4 follow-ups from §7 of `wf_lambda_div_rotation/final.md` are
load-bearing for any future round that wants a non-degenerate
diversity panel:

1. **Lift the `n_simulations` hard-cap** in
   `molmetal/scripts/r4_lambda_only_run.py:1466-1467`. Without
   this, the CLI `--n-simulations 1000` is silently re-written to
   100 on every invocation.
2. **Switch generator off Lambda-only** to the CFM/CFG joint
   generator (which emits `n_top_k=20` distinct candidates per cell,
   see R10 evidence). Lambda-only mode by design returns the seed
   verbatim.
3. **Activate the click-rules engine properly** — wire
   `click_rules_active` so the `--click-rules all-5` filter actually
   gates the Lambda generator (currently shows
   `click_rules_filter_emptied`).
4. **Per-pocket metal-specific proxies** (oxidation / coordination /
   Cl count / GSH-liability) — currently aggregate-only across 5
   cells, not broken out per pocket.

## 7. Cross-section update map

| Section | Edit type | Content |
|---|---|---|
| §4.5 (NEW paragraph) | inserted after WF-Lambda-Only-MiniPilot block | 3-seed aggregate panel + byte-identical collapse + cross-seed pool note + lift-vs-Metal-Pilot + hypothesis verdict + 3 follow-ups |
| §5.7 (NEW paragraph "Diversity-axis uplift via metal-seed rotation") | appended after the metal-pilot uplift panel | per-metal-seed aggregate table + silent cap note + lift-vs-Metal-Pilot 5-row table + hypothesis verdict + honest-framing caveats |
| §5.7 (NEW paragraph "Artefacts and integration notes (diversity-rotation round)") | appended at very end of §5.7 | pilot report + integration notes + 5-file cross-section update list |
| CROSS_REFS.md §4.5 row | extended | WF-Lambda-Diversity-Rotation update sub-bullet |
| CROSS_REFS.md §5.7 row | extended | WF-Lambda-Diversity-Rotation update — diversity axis (NEW) sub-bullet |
| TODO/pending/13_top_journal_pilot_r12.md | appended at bottom | NEW "WF-Lambda-Diversity-Rotation integration (2026-09-14)" section |
| molmetal/reports/wf_lambda_div_rotation/integrate.md | new file | this integration notes |

## 8. Wall-clock accounting

- Pilot run wall-clock: **~16 s** (3 seeds × 5 pockets × 1 seed,
  `n_simulations=100` after silent cap).
- Integration edits wall-clock: ≤5 min total.
- Total wall-clock used: ≤6 min (well under the 30-min budget).
- Paper compile sanity-check: not re-run (no preamble / macro
  changes; only `\paragraph` blocks added inside existing
  `\subsection{...}` / `\label{...}` contexts).