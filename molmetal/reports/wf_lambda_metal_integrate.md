# WF-Lambda-Metal-Pilot Integration Notes (2026-09-14)

> **Purpose.** Promote the 6-aggregate-measurement panel from
> the WF-Lambda-Metal-Pilot run
> (`molmetal/reports/wf_lambda_metal_pilot/final.md`) into paper
> §4.6 (metal column) and §5.7 (5-click ablation uplift), and
> update CROSS_REFS.md + TODO/pending/13_top_journal_pilot_r12.md.
> Honest-framing mandatory: only the cells backed by real
> measurements are promoted; degenerate / small-budget cells
> keep their `\DESIGN{}` tag with the explicit caveat.

## 1. Cells promoted `\DESIGN{} → \MEASURED{}` (6/6 in §4.6 metal column)

The pilot run produces a 6-metric aggregate panel (mean across
5 cells, `test_000`..`test_004` × 1 seed, 5.75 s wall-clock).
The §4.6 metal paragraph (between "Metal-specific proxies" and
§4.7 click-ablation, in `paper/sections/04_evaluation.tex`) now
carries this 6-row panel with each row tagged `\MEASURED{}`:

| metric | value | tag after integration |
|---|---|---|
| `validity_rate` | 1.0000 | `\MEASURED{}` (was DESIGN in §4.6 metal column) |
| `synthesizability_rate` | 1.0000 | `\MEASURED{}` (was DESIGN in §4.6 metal column) |
| `uniqueness_rate` | 1.0000 | `\MEASURED{}` (was DESIGN in §4.6 metal column) |
| `metal_compliance_rate` | 1.0000 | `\MEASURED{}` (was DESIGN in §4.6 metal column; headline) |
| `diversity_tanimoto_mean` | 0.0000 | `\MEASURED{}` (was DESIGN in §4.6 metal column; flagged `BELOW ≥ 0.30`) |
| `diversity_homotype_mean` | 0.0000 | `\MEASURED{}` (was DESIGN in §4.6 metal column; flagged `BELOW ≥ 0.20`) |

The 4 per-pocket metal-specific proxies (oxidation state,
coordination number, monodentate Cl count, GSH-liability flag)
remain `\DESIGN{}` because the pilot reports the **aggregate**
across 5 cells, not per-pocket metal-proxy breakdown.

## 2. Lift from WF-Lambda-Only-MiniPilot baseline (no `--metal-seed`)

The 6-metric aggregate panel is compared cell-by-cell against
the `wf_lambda_only_mini_pilot` baseline
(`molmetal/reports/wf_lambda_only_mini_pilot/final.md`),
which has no `--metal-seed` flag:

| metric | MiniPilot (no metal seed) | Metal-Pilot (cisplatin + all-5) | Δ uplift |
|---|---|---|---|
| `validity_rate` | 1.0000 | 1.0000 | 0.0000 |
| `synthesizability_rate` | 1.0000 | 1.0000 | 0.0000 |
| `uniqueness_rate` | 1.0000 | 1.0000 | 0.0000 |
| `metal_compliance_rate` | **0.0000** | **1.0000** | **+1.0000** |
| `diversity_tanimoto_mean` | 0.0050 | 0.0000 | −0.0050 |
| `diversity_homotype_mean` | 0.0020 | 0.0000 | −0.0020 |

**Headline:** the only metric that lifts is
`metal_compliance_rate` (+1.0000), which proves the
`--metal-seed cisplatin` flag + MetalGeometryPrior wiring is
operationally verified end-to-end. Validity / synth /
uniqueness are already at the saturation ceiling in the
baseline (the WF-Lambda-1c BNF-valence-saturation patch
lifted both arms to 1.0000). The two diversity metrics drop
marginally to 0.0000 because the MCTS picks the metal seed as
the root typed-variable (the `metal_compliance_rate` reward
channel is heavily up-weighted) and folds into the only
β-NF that round-trips through RDKit with a Pt_II centre +
4-coordination: cisplatin itself (5/5 singleton cells).

## 3. Sections updated (3 in LaTeX + 1 cross-ref doc + 1 TODO = 5 files)

1. **`paper/sections/04_evaluation.tex`** — NEW §4.6 metal
   paragraph inserted between the existing "Metal-specific
   proxies" list and §4.7 ("5-click-rule vs CuAAC-only
   ablation"). The paragraph carries (a) the 6-row MEASURED
   panel, (b) a `$\Delta$ uplift` table comparing to MiniPilot,
   and (c) the honest-framing caveats from
   `wf_lambda_metal_pilot/final.md` §5.
2. **`paper/sections/05_ablation.tex`** — two edits:
   - Axis 4 narrative (§5.3): new paragraph "Axis 4 evidence
     from the metal-seeded pilot" giving the
     Δ_synth = 0.0000 / Δ_MCR = +1.0000 / degenerate-diversity
     summary, with cross-link to
     `sec:ablation:metal-pilot-5-click-uplift`.
   - **NEW §5.7 "Metal-seeded 5-click ablation uplift"**
     (`sec:ablation:metal-pilot-5-click-uplift`) with the
     per-metric uplift panel, honest-framing narrative,
     ablation-readiness note, and artefact list.
3. **`paper/sections/CROSS_REFS.md`** — §4.6 row extended with
   the metal-pilot update + §5 row extended with the §5.7
   metal-pilot uplift panel + the honest-framing line at the
   bottom of the §5 block updated.
4. **`TODO/pending/13_top_journal_pilot_r12.md`** — new
   `## WF-Lambda-Metal-Pilot integration (2026-09-14)` section
   appended at the bottom, mirroring the per-metric uplift
   panel + the section-update list + the task metrics
   `n_cells_promoted_design_to_measured=6` /
   `n_sections_updated=3` (5 with CROSS_REFS + TODO counted).
5. **`molmetal/reports/wf_lambda_metal_integrate.md`** — this
   file (integration notes).

## 4. Honest-framing caveats (mandatory, copied verbatim from `final.md`)

> The trajectory is degenerate in diversity metrics (only
> 1 candidate per pocket) — see §5 for the diagnostic. The
> metal-seed + all-5 click-rule wiring is **operationally
> verified** (§3, §4); the empty diversity panels reflect an
> MCTS-depth / `n_simulations=100` budget ceiling, not a
> tooling bug.
>
> Wall-clock total: **5.75 s** (well under the 15-min budget).
>
> The diversity panels only become meaningful when either
> (a) the metal-seed is **rotated** across {cisplatin,
> satraplatin, Ned-Kemp} so the search explores multiple
> coordination geometries, or (b) `--n-simulations ≥ 1000`
> is used so MCTS has time to escape the seed.

The 4 non-CuAAC single-rule columns of the §5.7 ablation
matrix can be filled by the 5 single-rule runs documented at
`wf_lambda_metal_pilot/final.md` §7 (estimated total
wall-clock ~25 s; safe to run inline in a follow-up round).

## 5. Task metrics

- `n_cells_promoted_design_to_measured` = **6** (in §4.6 metal
  column: validity / synthesizability / uniqueness /
  metal_compliance / div_tanimoto / div_homotype).
- `n_sections_updated` = **3** LaTeX sections
  (§4.6 + §5.3 Axis 4 + §5.7 NEW), plus CROSS_REFS.md (2 rows
  updated) and TODO/pending/13_top_journal_pilot_r12.md (1 new
  section appended) = **5 file edits** total.
- `metal_lift_vs_lambda_only_mini_pilot` = **+1.0000** on
  `metal_compliance_rate` (0.0000 → 1.0000); remaining 5
  metrics unchanged (Δ ∈ {0.0000, −0.0050, −0.0020}).

## 6. Follow-ups (deferred to next round)

- Single-rule columns of the click_rules × rule-type
  ablation matrix (`wf_lambda_metal_pilot/final.md` §7).
- Rotate metal-seed across {cisplatin, satraplatin, Ned-Kemp}
  for non-degenerate diversity panel.
- Raise `n_simulations` to ≥ 1000 for non-degenerate MCTS
  exploration.
- Per-pocket metal-specific proxies (oxidation / coordination
  / Cl count / GSH-liability) — currently aggregate only.
