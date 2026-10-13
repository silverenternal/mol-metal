# Click-Rule Effect-Size Study — Summary

**Protocol:** enable each click rule in isolation via 
`--click-rules <rule>`, fix `n_simulations` per cell, 
compute the mean of 4 metrics across the (pockets x seeds) grid. 
Cohen's d is the standard between-rule pooled-variance form 
(Cohen 1988, eq. 2.3.5).

## Config
- pockets=20, seeds=[42, 0, 1234], n_simulations=200, n_top_k=20, metal_seed=None
- crossdocked_root=None; path_resolve_sources={'default:/home/hugo/codes/pocket/data/crossdocked2020/pocket10/crossdocked_pocket10': 20}; pockets with a real reference_smiles = 20 / 20

## Panel (mean across cells)

| rule | diversity_tanimoto | validity_rate | synthesizability_rate | metal_compliance_rate | citation |
| --- | --- | --- | --- | --- | --- |
| CuAAC | 0.0000 | 0.9500 | 0.9500 | 0.0000 | Himo 2005 JACS (CuAAC regiochemistry) |
| SPAAC | 0.0009 | 0.9500 | 0.9500 | 0.0000 | Worrell 1984 Science (SPAAC cyclooctyne) |
| ThiolEne | 0.0009 | 0.9500 | 0.9500 | 0.0000 | Kolb 2001 Angew (click canon, thiol-ene) |
| Suzuki | 0.0000 | 0.9500 | 0.9500 | 0.0000 | Suzuki 2011 Angew (Pd cross-coupling) |
| AmideCoupling | 0.0084 | 0.9500 | 0.9500 | 0.0000 | Kolb 2001 Angew (click canon, amide-coupling) |

## Cohen's d (rule vs OTHER rules)

Positive d = rule increases the metric relative to the other rules' average. 
Magnitude convention (Cohen 1988): |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, >= 0.8 large.

| rule | diversity_tanimoto | validity_rate | synthesizability_rate | metal_compliance_rate |
| --- | --- | --- | --- | --- |
| CuAAC | -0.275 | 0.000 | 0.000 | NA |
| SPAAC | -0.153 | 0.000 | 0.000 | NA |
| ThiolEne | -0.149 | 0.000 | 0.000 | NA |
| Suzuki | -0.275 | 0.000 | 0.000 | NA |
| AmideCoupling | 0.903 | 0.000 | 0.000 | NA |

## Lit anchors

- **CuAAC** — Himo 2005 JACS (CuAAC regiochemistry)
- **SPAAC** — Worrell 1984 Science (SPAAC cyclooctyne)
- **ThiolEne** — Kolb 2001 Angew (click canon, thiol-ene)
- **Suzuki** — Suzuki 2011 Angew (Pd cross-coupling)
- **AmideCoupling** — Kolb 2001 Angew (click canon, amide-coupling)

## Honest framing

- Cohen's d is computed *between rules*, not between rule vs. a hand-picked baseline.  The baseline is the population of the OTHER four rules' samples.
- With 1 cell per rule (default `--pockets 1 --seeds 42`), d is computed against an n=4 baseline (other rules) vs. n=1 treatment (this rule).  The variance term collapses to a single sample so d is reported as `NA`.  Use `--pockets 5 --seeds 42 0 1234` for n=15 cells per rule and a meaningful d.
- Effect-size magnitudes are NOT a comment on which rule is 'best' — they only say which rules shift the metric panel *more* than the average of the others.
- The 4 metrics are Lambda-MCTS downstream outcomes. None of them is a per-reaction chemical yield; a run of this module cannot answer a yield question.

