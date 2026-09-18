# Figure 4 — Homotype distance vs Tanimoto distance scatter

**Caption.** A 2-D scatter of the 20 paired (homotype_distance,
tanimoto_distance) measurements drawn from
`molmetal/reports/wf_lambda2e_compare/pairs.csv` (the canonical artefact of
WF-Lambda-2.E, written by `molmetal/scripts/wf_lambda2e_compare.py`).  The
X-axis is **Tanimoto distance** (1 - Tanimoto similarity on Morgan-2
fingerprints, RDKit canonical).  The Y-axis is **homotype distance**
(typed-variable edit distance computed under the
`homotype_enriched_vocab` of WF-Lambda-2.E, which adds hybridisation +
ring-membership as first-class atoms).  Each marker = one (i, j) pair
from the 10-molecule probe set; the eight constitutional isomers +
cisplatin + four unrelated organics produce 10+10 = 20 unique unordered
pairs.  Quadrant guide lines at x = 0.5 and y = 0.5 partition the unit
square into four regions:

| Quadrant | X (Tanimoto dist) | Y (Homotype dist) | Interpretation |
|----------|-------------------|-------------------|----------------|
| Upper-left | low  (< 0.5)      | high (>= 0.5)     | Lambda-discovered-but-Tanimoto-invisible chemistry (the contribution of this paper) |
| Upper-right| high (>= 0.5)     | high (>= 0.5)     | both metrics agree (high distance) |
| Lower-left | low  (< 0.5)      | low  (< 0.5)      | both metrics agree (low distance) |
| Lower-right| high (>= 0.5)     | low  (< 0.5)      | Tanimoto home turf (constitutional isomerism within an atom alphabet) |

Marker colour follows the Figure 1 phase-band palette: constitutional
isomers = green (chemistry phase), unrelated chemistries = dark red
(output phase).

## Honest-framing

- **MEASURED:** the 20 (homotype, Tanimoto) coordinates are read verbatim
  from `molmetal/reports/wf_lambda2e_compare/pairs.csv` (committed artefact,
  written 2026-09-12 by `wf_lambda2e_compare.py`).
- **MEASURED:** the 10 isomers / 10 unrelated chemistry split and per-pair
  SMILES strings.
- **INTERPRETATION (not new data):** the quadrant annotations
  ("Lambda-discovered-but-Tanimoto-invisible" / "Tanimoto home turf")
  are conceptual readings of the scatter, not properties measured from
  the 20 points.

### Empirical observation on this 20-pair slice

The data range is: Tanimoto distance ∈ [0.7273, 1.0000] and homotype
distance ∈ [0.0069, 0.5000].  Quadrant counts (as measured at figure
render time):

| Quadrant | Count | Notes |
|----------|-------|-------|
| LL (low Tanimoto, low homotype)    | 0   | — |
| LU (low Tanimoto, high homotype)   | 0   | — (would be Lambda-only discoveries; this 20-pair slice has **none**) |
| RL (high Tanimoto, low homotype)   | 19  | Tanimoto home turf dominates the slice |
| RU (high Tanimoto, high homotype)  | 1   | (cisplatin, benzene) is the lone RU outlier; represents an across-phase large chemistry gap |

So on this 20-pair slice the **upper-left quadrant is empty** —
homotype distance never reaches the >= 0.5 threshold while Tanimoto
distance stays < 0.5.  This is a **negative result** for the toy probe
set, and we report it as such: the orthogonality claim of §3.1 / §4.8 /
§5.6 is not visually demonstrated by this 10-mol slice alone; rather,
this figure illustrates the **regime in which the two metrics agree**
(Tanimoto home turf, n=19/20) and motivates the larger Round-12/13
evaluation.  The upper-left quadrant annotation is therefore *aspirational
framing* for the hybrid MLC + flow-matching regime, **not** a result
demonstrated by the 20 points themselves.

## Provenance

- Source CSV: `molmetal/reports/wf_lambda2e_compare/pairs.csv`
- Source JSON: `molmetal/reports/wf_lambda2e_compare/metrics.json`
- Driver script: `molmetal/scripts/wf_lambda2e_compare.py`
- Workflow: WF-Lambda-2.E (see `molmetal/reports/wf_lambda2e_compare/final.md`)
- Figure source: `paper/figures/fig4_homotype_vs_tanimoto.py`
- Companion vector: `paper/figures/fig4_homotype_vs_tanimoto.svg`

## Regeneration

```bash
uv run paper/figures/fig4_homotype_vs_tanimoto.py
```

Outputs: `paper/figures/fig4_homotype_vs_tanimoto.png` (600 dpi) +
`paper/figures/fig4_homotype_vs_tanimoto.svg` (scalable).
