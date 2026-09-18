# WF-PB-Pass-Real-Dock Integration Notes (2026-09-14)

> **Purpose.** Integrate the PB pass-rate smoke measurement from
> `molmetal/reports/wf_pb_pass_real_dock/final.md` into paper §4.6
> (`sec:evaluation:anticancer`) and update
> `TODO/pending/14_full_100pocket_paper_r13.md` to mark the PB
> pass-rate status as `\MEASURED{}` (single-pocket smoke only, with
> three honest-framing caveats). This run is a **path-correctness
> smoke**, not a 100 × 3 production sweep.

## 1. What this integration does

The `--pb-check + --physical-docking` path of
`molmetal/scripts/r4_c_full_sweep.py` (WF-Wire-PoseBusters, Round-7
wiring) was exercised end-to-end on a single CrossDocked2020 pocket
(`test_000`, seed=42) and produced real docked poses that ran through
`PoseBustersAdapter.validate_list` on real RDKit-canonical
chemistry. The §4.6 paper paragraph is updated to carry the
**single (pocket=``test_000``, seed=42) cell** with the explicit
honest framing that:

1. **PB pass rate = 1.000 (1/1)** is a path-correctness smoke
   figure, not a statistical sample;
2. **Protein-blind mode**: the PoseBusters adapter is invoked in
   its default `pb_mode="mol"` configuration (chemistry + geometry
   only; no clash-against-receptor assessment);
3. **1-of-100 pockets exercised**: cross-pocket variance (especially
   on metal-binding pockets where the §3.3 MetalGeometryPrior is
   engaged) is unknown.

The §4.6 paragraph follows the `wf_lambda_metal_pilot` integration
pattern (a single `\paragraph{}` block between "Metal-specific
proxies (per-pocket, pilot)" and §4.7 click-ablation).

## 2. Cells promoted `\DESIGN{} → \MEASURED{}`

| cell | value | tag after integration | honest-framing caveat |
|---|---|---|---|
| `pb_pass_rate` (single-pocket smoke, pocket=`test_000`, seed=42) | 1.000 | `\MEASURED{}` (was DESIGN in §4.6 PB column) | n=1, chemistry-only, no SOTA anchor |
| `n_pb_pass` (same cell) | 1 / 1 | `\MEASURED{}` (was DESIGN) | same caveats |
| `pb_status` (same cell) | completed | `\MEASURED{}` (was DESIGN) | adapter ran without exception |
| `vina_best_kcal_mol` (same cell) | −6.929 | `\MEASURED{}` (was DESIGN; CPU Vina because `--engine both` is GPU-incompatible) | re-run with `--engine vina --physical-engine quickvina2-gpu` to re-engage GPU |

**Total: 4 cells promoted** in the §4.6 PB paragraph, all on the
single smoke cell. All **other** PB columns of
Table~\ref{tab:per-pocket} and Table~\ref{tab:aggregate} remain
`\DESIGN{}` pending Round-13 100 × 3 sweep. The §3.2 per-pocket
metric gap table at
`TODO/pending/14_full_100pocket_paper_r13.md` now carries an
explicit **PB pass rate status (Round-14 / 2026-09-14): \MEASURED{}**
banner with the same three caveats copied verbatim.

## 3. Honest framing preserved

The §4.6 paragraph explicitly carries the four honest-framing
bullet points copied verbatim from `final.md` §5:

1. **PB pass rate = 1.000 is a path-correctness smoke**, not a
   production figure. The projected 30–50 % window in the task
   brief is **untested** at scale.
2. **Protein-blind mode**: `pb_mode="mol"` validates chemistry +
   geometry in isolation; protein-aware check requires
   `pb_mode="dock"` with receptor input (a ~30-line adapter
   extension, **not in scope** for this run).
3. **One of 100 pockets exercised**: single-pocket smoke;
   cross-pocket variance is unknown.
4. **No SOTA comparison**: the PoseBusters pass-rate is not
   compared head-to-head against TargetDiff / DiffDock /
   Pocket2Mol reported numbers. The cite-only SOTA row for PB
   remains `\CITEDONLY{}`.

The §6 limitations line for protein-blind PB mode is the
documentary hook for caveat (1); this integration does not
duplicate or contradict that line.

## 4. Promotion rule (§4.10) compliance

Per §4.10 reporting protocol rule~4 (promotion rule):
> *A `\DESIGN{}` cell can be promoted to `\MEASURED{}` only by
> running the protocol and re-tagging at the point of use; no
> `\DESIGN{}` cell is silently promoted by editing the LaTeX
> source.*

The promotion rule is satisfied because:

1. The protocol (`--pb-check + --physical-docking`) was run
   end-to-end on a real docked pose (artefact:
   `molmetal/reports/wf_pb_pass_real_dock/r4c.csv`,
   `r4c.json`, `r4c.md`, `r4c_logs/`, `r4c_poses/`).
2. The promotion is **explicit at the point of use** (each `\MEASURED{}`
   tag in the §4.6 paragraph carries the `(pocket=test_000, seed=42)`
   qualifier in the surrounding text).
3. **No Table~\ref{tab:per-pocket} or Table~\ref{tab:aggregate} data
   cell is silently promoted.** Only the §4.6 paragraph
   (which already had a `\DESIGN{}` PB column at the aggregate
   level) is updated.
4. **No SOTA cell is touched.** The cite-only SOTA column for
   PoseBusters remains `\CITEDONLY{}`.

## 5. Files modified

| Path | Modification |
|---|---|
| `paper/sections/04_evaluation.tex` | Added a new `\paragraph{}` "PoseBusters pass rate on real docked poses (WF-PB-Pass-Real-Dock, 2026-09-14, …)" after "Metal-specific proxies (per-pocket, pilot)" and before §4.7 click-ablation. 4 cells tagged `\MEASURED{}` on the single (pocket=`test_000`, seed=42) cell; 3 honest-framing caveats explicit; cite to `molmetal/reports/wf_pb_pass_real_dock/final.md`. |
| `TODO/pending/14_full_100pocket_paper_r13.md` | Added a `\MEASURED{}` banner for "PB pass rate" status in §3.2 per-pocket metric gap table; banner carries the 3 honest-framing caveats copied from `final.md` §5; full 100 × 3 PB column remains `\DESIGN{}` pending Round-13 sweep. |

## 6. Files NOT modified (intentional)

- `paper/sections/CROSS_REFS.md` — left untouched. The §4.6
  metal paragraph already has its `WF-Lambda-Metal-Pilot` cross-ref
  entry (line 74); a second §4.6 cross-ref entry for the PB
  smoke would duplicate the §4.6 entry and dilute the cross-ref
  map. The PB smoke is a **sub-feature of §4.6's "Validity"
  gate description** in §4.1 (line 120) rather than a new
  top-level cross-ref.
- `paper/sections/01_intro.tex` and `paper/sections/02_related.tex` —
  PB pass rate is a §4 evaluation cell, not a §1 or §2 claim.
- `paper/main.tex` and `paper/refs.bib` — no new citations; the
  PoseBusters paper (`buttenschoen2024posebusters`) is already
  cited at §4.1 line 123.

## 7. Metrics for the parent agent

```yaml
pb_pass_rate_measured: 1.000     # 1/1 on (pocket=test_000, seed=42)
pb_pass_rate_target_diff: "30-50 % projection - untested at scale"
                                # The brief's projection is NOT measured.
                                # Only the n=1 chemistry-validity smoke is.
gap_pp_remaining: 100.0         # 100 of 100 PB production cells still \DESIGN{}
                                # pending Round-13 100x3 sweep. The smoke
                                # is a path-correctness check, not a cell
                                # that closes the per-pocket PB column.
n_sections_updated: 2           # paper/sections/04_evaluation.tex +
                                # TODO/pending/14_full_100pocket_paper_r13.md
```

Honest-framing note: `gap_pp_remaining = 100.0` is **intentionally
reported as the production-cell gap** (100 of 100 per-pocket PB
cells still `\DESIGN{}`); the integration closes 4 §4.6 paragraph
cells on a single (pocket, seed) point estimate, not the 100-cell
production panel. No silent promotion of any `\DESIGN{}`
production cell.

## 8. Verification checklist

- [x] §4.6 paragraph references the artefact
      (`molmetal/reports/wf_pb_pass_real_dock/final.md`).
- [x] All 4 promoted cells are qualified at the point of use
      with the (pocket, seed) tuple.
- [x] Three honest-framing caveats are explicit (protein-blind,
      1-of-100, no SOTA).
- [x] TODO/pending/14 carries the `\MEASURED{}` banner with the
      3 caveats copied from `final.md` §5.
- [x] No Table~\ref{tab:per-pocket} or Table~\ref{tab:aggregate}
      data cell is silently promoted.
- [x] No cite-only SOTA cell is touched.
- [x] No new citation introduced; `buttenschoen2024posebusters`
      was already cited in §4.1.
- [x] §4.10 promotion rule satisfied (real run + explicit tag at
      point of use, no silent LaTeX-only promotion).