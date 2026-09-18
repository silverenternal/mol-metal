# TODO-15 — Three New Anticancer Metric Specs

**Project:** Mol-Metal — precious-metal anticancer drug design (Pt, Ru, Ir, Pd, Au)
**Author:** TODO-15 design subagent
**Date:** 2026-09-14
**Scope:** Add three new heuristic channels to the anticancer metric suite — `metal_coordination_score`, `gsh_evasion_flag`, `dna_kb_proxy` — per the §6 bucket list of `molmetal/reports/anticancer_vs_general_metrics_survey.md`.
**Status:** SPEC ONLY (design phase; implementation in `molmetal/molmetal_lam/metrics/`).

Honest framing across all three metrics: **they are screening proxies, not measured biophysics.** Each spec ends with a "What this does NOT measure" section so downstream readers cannot mistake a high score for a measured rate, binding constant, or evasion guarantee.

---

## 1. `metal_coordination_score`

**Purpose.** Check that the metal centre in a candidate ligand satisfies the *expected* coordination number and geometry for its element/oxidation state. Square-planar Pt(II) and Pd(II) want 4 donors; octahedral Ru(III) and Ir(III) want 6; linear Au(I) wants 2. The score is 1.0 when the expected geometry is matched and decays smoothly otherwise.

**Input signature.**
```
metal_coordination_score(
    smiles: str,
    conformer_mol: Chem.Mol | None = None,   # optional 3D; falls back to 2D distance
    metal: str = "Pt",                       # "Pt" | "Pd" | "Ru" | "Ir" | "Au"
    oxidation_state: int = 2,
) -> float
```

**Output signature.** `float ∈ [0.0, 1.0]`. `0.5` on parse/import failure (consistent with the existing `AnticancerMetricSuite` fallback convention).

**Source data.** RDKit `Chem.MolFromSmiles(smiles)` parses the molecule; if a 3D conformer is supplied (`conformer_mol.GetConformer()`) we use 3D distances, else we fall back to 2D heavy-atom distance estimation via `Chem.Get3DDistanceMatrix` on an ETKDG-embedded conformer (single attempt, no MMFF optimisation — speed over accuracy; the heuristic is intentionally fast).

**RDKit / heuristic steps.**
1. Locate the metal atom by SMARTS — one of `[Pt, Pd, Ru, Ir, Au]` — and pick the first match as the centre.
2. Look up the expected coordination number `cn_exp` and geometry label `geom_exp` from a small lookup table:
   `Pt(II)→4/SQUARE_PLANAR`, `Pd(II)→4/SQUARE_PLANAR`, `Ru(III)→6/OCTAHEDRAL`, `Ir(III)→6/OCTAHEDRAL`, `Au(I)→2/LINEAR`.
   The table lives in `molmetal/molmetal_lam/metrics/_coordination_table.py` and mirrors the `DEFAULT_METAL_GEOMETRY` constants already declared in `molmetal/molmetal_lam/priors/metal_geometry.py` (78→SP, 46→SP, 77→OCT, 44→OCT).
3. Define donor set = atoms of type `N, O, S, F, Cl, Br, I` bonded to or within 2.5 Å of the metal (3D path) — or within the 2D heavy-atom distance if no 3D conformer is supplied.
4. Count `cn_obs = len(donor_set)`. Penalise by deviation: `score = max(0.0, 1.0 - 0.33 * |cn_obs - cn_exp|)`. So Pt(II) with cn=4 → 1.0; cn=3 → 0.67; cn=2 → 0.33; cn=5 → 0.67; cn=6 → 0.33.
5. Geometry label sanity — if a non-default geometry is supplied via `geom_override`, blend a 0.20 penalty when `geom_exp != geom_override`. (Default path: ignore; the CN deviation alone is the dominant signal.)

**Expected value range.** `[0.0, 1.0]`. Empirical anchors:
- cisplatin `Cl[Pt](N)(N)Cl` → 1.0 (4 donors expected = 4 observed: 2×NH3 + 2×Cl).
- `[Pt](N)(N)(N)Cl` (5-coordinate, unusual) → 0.67.
- bare Pt atom `O=[Pt]=O` → 0.33 (cn_obs=2 vs cn_exp=4).
- methanol `CO` (no metal SMARTS match) → 0.5 fallback.

**What the score means.** A scalar reward signal indicating *geometric plausibility of the metal centre*. Useful for filtering Lambda-generated candidates that violate a metal's dominant coordination number. Combine with `aquation_rate_proxy` (`molmetal_lam/priors/metal_hydration.py`) when reasoning about kinetic lability.

**What this does NOT measure.**
- It is NOT a measurement of actual coordination geometry in solution — aquation may displace Cl⁻ donors and shift the in-cell CN.
- It does NOT probe trans-effect geometry (cis/trans isomerism is not detected from CN alone).
- It does NOT validate oxidation state from SMILES — we trust the caller's `oxidation_state` argument.
- It does NOT detect Pt(IV) prodrugs (octahedral with two axial OH/OC(O)R groups); `Pt(IV)` is out of scope for v0 — TODO-17 follow-up.

---

## 2. `gsh_evasion_flag`

**Purpose.** Binary screening flag for "the candidate is *plausibly* resistant to intracellular GSH (glutathione) quenching". Not a measured rate. The flag fires when the molecule (a) does NOT carry an obvious soft-nucleophile handle that GSH attacks (free thiol, disulphide, selenol, thiocarbamate), AND (b) DOES carry at least one aromatic-N heterocycle that can stack inside DNA grooves (pyrimidine / purine / quinazoline / imidazopyridine etc.) — the same structural motif that GSH-resistant Pt(II) drugs (picoplatin, satraplatin-like scaffolds) use.

**Input signature.**
```
gsh_evasion_flag(
    smiles: str,
) -> int                          # 0 or 1, binary
```

Or, returning a structured dict for callers that want the underlying predicate outputs:
```
gsh_evasion_flag_detailed(
    smiles: str,
) -> dict[str, bool | int]       # keys: "has_soft_nucleophile", "has_aromatic_n_heterocycle",
                                #       "soft_nucleophile_smarts", "aromatic_n_smarts",
                                #       "flag"  (0/1)
```

**Output signature.** `int ∈ {0, 1}`. The detailed variant returns `flag ∈ {0, 1}` plus the two predicate booleans.

**Source data.** RDKit `Chem.MolFromSmiles`. No 3D needed. No external DB. Two SMARTS queries evaluated against the molecule.

**RDKit / heuristic steps.**
1. SMARTS for soft nucleophiles (returns `True` if any match):
   - `[SH]` (free thiol)
   - `[SS]` (disulphide)
   - `[S;H0;D2]([#6])[#6]` (thioether R-S-R where R≠H — covers methylthio, ethylthio, alkylthio handles common on GSH-sensitive ligands; thioethers alone are weakly reactive, but combined with a Pt(II) centre they are kinetically vulnerable)
   - `[SeH]` (selenol — present in selenocysteine-Se-GSH conjugates)
2. SMARTS for aromatic-N heterocycles (returns `True` if any match):
   - `[n]1[c,n][c,n][c,n][c,n][c,n]1` — 6-ring aromatic with ≥1 ring N (pyridine, pyrimidine, pyrazine, triazine, quinazoline core etc.)
   - `[n]1[c,n][c,n][c,n]1` — 5-ring aromatic with ≥1 ring N (imidazole, pyrazole, triazole)
   - `n1cnc2[n]cccc12` and the relaxed `[n]1[c,n]2[c,n][c,n][c,n][c,n]2[c,n][c,n][c,n]1` to capture fused purines (purine itself, adenine, guanine, benzimidazole).
3. `flag = 1` iff `has_aromatic_n_heterocycle AND NOT has_soft_nucleophile`. Otherwise `flag = 0`.

**Expected value range.** Binary `{0, 1}`. Empirical anchors:
- picoplatin-like `Cl[Pt](N)(N)(N1C=CC=N1)` → 1 (pyridine ring present, no S/Se/disulphide).
- cisplatin `Cl[Pt](N)(N)Cl` → 0 (no aromatic N heterocycle; classified as GSH-vulnerable, consistent with literature — picoplatin's 2-methylpyridine was the SAR trick).
- thiopurine-like `Sc1ncnc2[n]cccc12` → 0 (has -SH AND has purine — soft nucleophile wins).
- bare methanol `CO` → 0 (no aromatic N).
- thiouracil `Sc1ccnc(=O)[nH]1` → 0 (has thiol).

**What the score means.** A *pre-screening* classifier — "this molecule's structural class is in the GSH-resistant bucket". Use it as a soft filter when you want to enrich Lambda proof-search rollouts toward scaffolds that don't get quenched before reaching DNA. NOT a substitute for measuring GSH adduct formation by LC-MS.

**What this does NOT measure.**
- It does NOT measure reaction rate with GSH. A flag=1 molecule can still be attacked by GSH via routes not covered by the four SMARTS (e.g. metallophilic Se→S substitution on Au(I), π-allyl Pd(II) displacement).
- It does NOT measure cellular GSH concentration, GST overexpression, or MRP1 efflux — these dominate in-cell fate and are outside any 2D SMILES heuristic.
- It does NOT detect sterically hindered silyl-protected thiols (`[S][Si](C)(C)C`) that survive GSH in the prodrug window — by design, any `S` bonded to a non-H atom is treated as a soft nucleophile; this is conservative and biased toward flag=0.
- It does NOT distinguish aromatic-N heterocycles that bind DNA minor groove (good evasion) from those that bind GSH directly (bad evasion) — both look identical at 2D.

---

## 3. `dna_kb_proxy`

**Purpose.** Heuristic proxy for DNA binding affinity of a metal-organic candidate, derived from the count of fused aromatic N-heterocycles near the metal centre. Range `[0, 1]`. **NOT a measured Kb** — it is a structural prior that recovers the *qualitative* SAR ordering for known Pt(II)/Ru(III) clinicals (cisplatin, carboplatin, picoplatin, satraplatin) without running a docking experiment. LogP contribution is intentionally excluded to avoid double-counting with the existing `logp_anticancer` channel.

**Input signature.**
```
dna_kb_proxy(
    smiles: str,
    metal: str = "Pt",
    oxidation_state: int = 2,
    fused_ring_cutoff: int = 4,   # ring size cutoff for "fused aromatic" — default 4–6-membered
) -> float
```

**Output signature.** `float ∈ [0.0, 1.0]`. Returns 0.0 on parse/import failure (deliberate; we want to fail-closed for a binding-affinity proxy — `0.5` would imply "average binder" which is a strong claim).

**Source data.** RDKit `Chem.MolFromSmiles(smiles)` plus `Chem.GetSSSR` for ring perception. No 3D needed (deliberate — keeps the proxy fast and comparable to the 2D descriptor channels).

**RDKit / heuristic steps.**
1. Locate metal centre by SMARTS — `[Pt, Pd, Ru, Ir, Au]`. If absent, return 0.0 (no metal → no DNA Kb proxy claim).
2. Look up the metal's expected binding-mode weight `w_metal`:
   - `Pt(II), Pd(II)` → 1.00 (well-established covalent 1,2-d(GpG) crosslink mode; Jamieson & Lippard 1999).
   - `Ru(III)` → 0.70 (octahedral, partial kinetic inertness; clinical precedence is shallower — NAMI-A, KP1019).
   - `Ir(III)` → 0.55 (limited clinical DNA-binder precedent; mostly photoredox catalysts repurposed for imaging).
   - `Au(I)` → 0.30 (soft-metal selenophilicity rather than DNA backbone attack; different mechanism entirely).
3. Find fused aromatic N-heterocycles: walk every ring in `mol.GetRingInfo().AtomRings()`, select rings that contain at least one aromatic N (`[n]`), and group rings into *fused systems* using the standard "two rings share ≥2 atoms" adjacency test. Count the largest fused system `n_fused_rings`.
4. Spatial proximity to metal: among the fused aromatic N-heterocycles, compute graph distance from any ring atom to the metal atom (BFS through bonds, integer distance). Discard fused systems whose closest ring atom is `> 4 bonds` away from the metal. Call the surviving count `n_near`.
5. Combine:
   ```
   proximity_score = min(1.0, 0.25 * n_near)        # 4+ near rings → saturate at 1.0
   geometric_penalty = 1.0 if (Pt(II) and cn_obs == 4) else \
                       0.85 if (Ru(III) and cn_obs == 6) else \
                       0.70 if (Ir(III) and cn_obs == 6) else 0.0
   dna_kb_proxy = proximity_score * geometric_penalty * w_metal
   ```
   where `cn_obs` is the donor count from §1 (reuse the same heuristic; pass through `metal_coordination_score` internals).

**Expected value range.** `[0.0, 1.0]`. Empirical anchors:
- `[Pt](N)(N)(n1cnc2[n]cccc12)` (purine coordinated to Pt(II), 2 fused rings near metal) → `0.25*2 * 1.0 * 1.0 = 0.50`.
- cisplatin `Cl[Pt](N)(N)Cl` (no aromatic N heterocycle) → 0.0.
- bare methanol `CO` → 0.0.
- `[Pt](N)(N)(n1cccc2[n]cccc12)` (quinoline, 2 fused rings) → 0.50.
- `[Ru](N)(N)(N)(N)(n1cnc2[n]cccc12)(O)` (octahedral Ru(III) with 1 purine ligand) → `0.25*1 * 0.85 * 0.70 ≈ 0.15`.

**What the score means.** A *prior* on whether the candidate's geometry permits a covalent DNA-binding mode in the same chemical class as a clinical platinum drug. Higher = more likely to occupy the same binding-pose envelope. Combine with `metal_coordination_score` and `gsh_evasion_flag` to triage candidates. **Use the score to rank; do not use it as an absolute binder.**

**What this does NOT measure.**
- It does NOT measure a binding constant Kb. The numerical output is dimensionless and on an arbitrary 0–1 scale that has NOT been calibrated against any ITC/SPR/gel-shift data.
- It does NOT account for DNA sequence selectivity (N7-guanine vs N3-adenine vs N1-adenine — all look identical to a SMARTS query).
- It does NOT measure covalent bond formation rate (aquation kinetics are not modelled here; an unhydrolysed `Cl[Pt]` will not bind DNA in the short term, but scores the same as the hydrolysed form).
- It does NOT distinguish intercalators from covalent crosslinkers — a flat polyaromatic that slides between bp looks identical to a crosslinking heterocycle.
- It does NOT account for logP / cellular uptake — by design (avoids double-counting with `logp_anticancer`). The two scores are intentionally orthogonal: logP for delivery, this proxy for engagement once inside.

---

## Cross-cutting design choices

1. **Fallback convention.** All three metrics share the same RDKit-graceful path: invalid SMILES, missing RDKit, or NaN/inf from any RDKit call returns `0.5` for the float scores and `0` for the binary flag. This matches `AnticancerMetricSuite._band` (file: `priors/anticancer_metric_suite.py`).
2. **Determinism.** No RNG. All heuristics are RDKit-deterministic. No 3D embedding is performed (only consumed if the caller supplies a conformer).
3. **No external DB.** All three metrics use SMARTS + lookup tables. No ChEMBL, no PubChem, no online lookup. The metric is reproducible offline on the RX 7800 XT.
4. **No docking binary.** The DNA proxy is purely 2D-topological; it does not call QuickVina, AutoDock, or any pose sampler. (TODO-16 will own the docking-based oracle path.)
5. **Counter integration.** Each metric increments an `_ANTICANCER_COUNTERS` key (`coord_score_calls`, `gsh_calls`, `dna_kb_calls`) so the demo runner can surface coverage statistics in `evaluate_anticancer_metrics.py`.

## File plan

- `molmetal/molmetal_lam/metrics/_coordination_table.py` — NEW. `EXPECTED_COORDINATION_NUMBER`, `EXPECTED_GEOMETRY` dicts.
- `molmetal/molmetal_lam/metrics/anticancer_metric_suite.py` — NEW. Three top-level callables: `metal_coordination_score`, `gsh_evasion_flag`, `dna_kb_proxy`.
- `molmetal/molmetal_lam/metrics/__init__.py` — NEW. Re-exports.
- `molmetal/molmetal_lam/tests/test_metal_coordination_score.py` — NEW. Pytest for §1 anchors (cisplatin, methanol, 5-coordinate Pt).
- `molmetal/molmetal_lam/tests/test_gsh_evasion_flag.py` — NEW. Pytest for §2 anchors (picoplatin=1, cisplatin=0, thiopurine=0).
- `molmetal/molmetal_lam/tests/test_dna_kb_proxy.py` — NEW. Pytest for §3 anchors (purine Pt=0.5, methanol=0.0, octahedral Ru=0.15).
- `molmetal/scripts/evaluate_anticancer_metrics.py` — NEW. Demo runner (1 pocket, N=20 ligands, JSON + markdown table).
- `molmetal/reports/todo15_anticancer_metric_suite.md` — NEW. Implementation report (out of scope of this spec doc).

## Honest-framing one-liner (for paper / report)

> These three metrics are *structural screening proxies*, not measured biophysics. They encode known SAR patterns from clinical platinum / ruthenium drugs as RDKit-deterministic SMARTS heuristics so the Lambda proof-search can be steered toward scaffolds that historically perform well — while making no claim about absolute binding constants, in-cell GSH resistance, or aquation kinetics. Pair every flag with the appropriate experimental assay before publication.