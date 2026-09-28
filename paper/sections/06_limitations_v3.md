# §6 Limitations v3 — Narrative Summary (2026-09-26)

**Status**: TEMPLATE awaiting Wf-30/31/32 GPU-retrain numbers per `TODO/pending/82_metric_gap_filling_plan.md` §2 Phases B/C/D.
**Target**: Digital Discovery (RSC) Q1 IF 8.5 OR J. Chem. Inf. Model. (ACS) Q1 IF 5.6.
**Style**: IMRaD, honest framing. **MEASURED vs PROJECTED strictly labelled** — every numeric anchor carries one of three tags:
- `\MEASURED{}` = real generated-candidate measurement on this stack, with the source-report path.
- `\PROJECTED{}` = structural-fix ceiling (R18 target), from `TODO/02_metrics/METRICS_TARGETS.md` §2.
- `\MEASURED{TBD Wf-NN}` = pending GPU-retrain anchor; the Wf-NN tag maps to `molmetal/reports/wf_NN_*.md`.

---

## v3 vs v2 structural diff

| Diff | v2 (Wf-22 ship) | v3 (Wf-33 ship) |
|---|---|---|
| Layout | 8-item (§6.1–§6.8) | 6-item (§6.1–§6.6) + 1 cross-cutting footer |
| Wf mapping | §7 future-work prose | 1-to-1 onto Wf-30/31/32 GPU-retrain items + `\MEASURED{TBD Wf-NN}` placeholder |
| m1/m2 framing | R17 5-seed + Wf-22 honest-neg | R17 anchor + Wf-31 PROJECTED ceiling |
| Phase A framing | "structurally unrunnable" | "5-seed shipped end-to-end (79 min) but did NOT lift AUC" |
| CI 30-min framing | implicit in Phase A item | NEW §6.3, distinguishes CI runner (30 min structural) vs local RTX 5090 (~8 h experimental) |
| Vina Dock m3 | implicit in decode-ratio-lenient §6.7 | NEW §6.4, $830$c single-pocket + Wf-30 10-pocket × 5-seed PROJECTED |
| Affinity m16 | implicit in §6.8 wet-lab footer | NEW §6.5, Pearson r 0.207 seed-fragile + Wf-32 PROJECTED |
| PB-Valid m8 | implicit in decode-ratio-lenient §6.7 | NEW §6.6, 0.846 single-pocket smoke + Wf-32 production PROJECTED |

## Six items (1-to-1 onto Wf-30/31/32)

| § | Item | R17 anchor (\MEASURED{}) | R18 ceiling (\PROJECTED{}) | Wf fix |
|---|---|---|---|---|
| 6.1 | Tier-1 AUC | m1 Ru 0.7123 (1 seed); m2 Pt A2780cis 0.648 (3 seeds); m2 Pt 3-cell mean 0.546 (5 seeds) | m1 ≥ 0.81; m2 ≥ 0.80 (Rusanov tier) | **Wf-31** |
| 6.2 | Phase A BLOCK | Ru 7-line init; Pt 79 min single-ultracode; combined lift did not materialise | Blackwell + RTX 5090 dual-GPU split, ~8 h per run | **Wf-31** (Wf-29 smoke first) |
| 6.3 | Blackwell CI 30-min | CI runner timeout (structural) vs local RTX 5090 (experimental, ~8 h) | split + checkpoint wrapper for CI per-step | **Wf-29** + **Wf-31** |
| 6.4 | Vina Dock m3 | 830c median −1.2885 kcal/mol (n=100); strict decode-ratio 0.5; mean_atoms=8.0 unconditional-mode collapse | 10 pockets × 5 candidates × 5 seeds = 250 dockings; ≤ −8.5 kcal/mol | **Wf-30** |
| 6.5 | Affinity m16 | Pearson r 0.207 (margin=0.5); seed-fragile 0.005/0.519/0.071; ridge 0.572 beats D-MPNN by +0.365 | Pearson r ≥ 0.50 on PlatinAI 4-cell + MetalCytoToxDB 5-metal | **Wf-32** |
| 6.6 | PB-Valid m8 | 0.846 (1-pocket click-tile smoke, 26 checks); 0/30 production panel (n_distinct=1 collapse) | ≥ 0.90 on 10 pockets × 5 candidates × 5 seeds | **Wf-32** (cascades from m3 Wf-30 upstream repair) |

## Cross-cutting footer (stable across Wf-30/31/32)

- Compute envelope: Blackwell + RTX 5090 single-host vs TransDiffSBDD 64×A100.
- Subset coverage: PlatinAI 4-cell/4078 rows (Cloudflare block); MetalCytoToxDB 5-metal/19135 rows.
- No wet-lab validation: in-silico only; deferred to §future:wetlab-validation.

## G1–G7 gap inventory cross-reference

The 7 gaps in `TODO/pending/82_metric_gap_filling_plan.md` §0 map to §6 items as:

| Gap | Metric | MEASURED | SOTA target | §6 item |
|---|---|---|---|---|
| **G1** | m3 Vina Dock median | −1.2885 | −9.37 (TransDiffSBDD) | §6.4 |
| **G2** | m1 Ru AUC | 0.7123 | 0.81 (Krasnov) | §6.1 |
| **G3** | m2 Pt A2780cis | 0.648 | 0.85 (Rusanov) | §6.1 |
| **G4** | m7 Diversity Tanimoto | 0.1065 | 0.65 | (m7 carries Wf-31 TBD in Table 2 v3, no §6.1–§6.6 dedicated item) |
| **G5** | m16 Affinity r | 0.207 | 0.86 (Boltz-2) | §6.5 |
| **G6** | m19 LOMO | 0.535 | 0.70 | (m19 carries Wf-31 TBD in Table 2 v3, no §6.1–§6.6 dedicated item) |
| **G7** | Vina Dock on 10 CrossDocked | 5/10 pockets MEASURED | 100 pockets × 3 seeds | §6.4 |

G4 (m7 Diversity) and G6 (m19 LOMO) are not standalone §6 items in v3 because their MEASURED/PROJECTED framing is concentrated in Table 2 v3 row updates (`\MEASURED{TBD Wf-31}` placeholders); the limitation prose for m7/m19 is captured indirectly via the §6.1 (Tier-1 AUC) and §6.4 (Vina Dock) cross-references — diversity and LOMO are dependent on the same Wf-31 GPU retrain.

## Honest-negative inventory (cross-reference)

The 7 gaps above are the v3 honest-negative chain. No cell of `04_results_table2_v3.tex` is silently promoted from `\DESIGN{}` to `\MEASURED{}`; every `\MEASURED{TBD Wf-NN}` placeholder carries its Wf-NN tag so the auditor can grep `TBD Wf` and see the exact pending work item. When Wf-30/31/32 ship, each placeholder is replaced by the actual MEASURED number + `(n_seeds × epochs)` provenance string, and the §6 item is rewritten with the measured-vs-projected verdict per `party-twenty-spirit` §3.5 (MEASURED < PROJECTED ⇒ honest-negative; MEASURED ≥ PROJECTED ⇒ honest-positive).

## Files

- **`paper/sections/06_limitations_v3.tex`** (NEW, ~430 lines) — v3 LaTeX source.
- **`paper/sections/06_limitations_v2.tex`** (Wf-22 ship, 410 lines, UNCHANGED, historical preservation).
- **`paper/sections/06_limitations.tex`** (Wf-8 ship, 286 lines, UNCHANGED, historical preservation).
- **`paper/sections/06_limitations_auto.tex`** (auto-generated, sibling).
- **`paper/sections/04_results_table2_v3.tex`** (paired v3 Table 2, 1-to-1 Wf mapping).
- **`TODO/pending/82_metric_gap_filling_plan.md`** (G1–G7 source-of-truth).

## Date

2026-09-26 (Wf-33 ship, v3 TEMPLATE). Next refresh: 2026-10-01 (Wf-30 ship) → 2026-10-03 (Wf-31 ship) → 2026-10-05 (Wf-32 ship).