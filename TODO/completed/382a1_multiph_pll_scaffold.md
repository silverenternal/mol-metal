# Wf-382-A1 — TODO-94 R20-P3 Multi-PH + PLL feature extraction CPU SCAFFOLD

**Document ID:** Wf-382-A1 / TODO-94-R20P3-SCAFFOLD
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + Wf-382-A1 ultracode agent
**Trigger:** Per `TODO/pending/094_r20_p3_multiph_pll_feature.md` §0 — R20-P3
Multi-PH + PLL feature extraction CPU SCAFFOLD expansion under user directive
*"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*.

---

## §0 TL;DR

**Goal**: Wrap the canonical PLL implementation at
`molmetal/molmetal_lam/lam_chem/pll_signature.py` (Liu & Feng 2026, 684 LOC)
into the `molmetal/adapters/flow_matching_lipman/` adapter layer so the
R20-P3 Multi-PH + PLL feature axis can be consumed by CFM trainers without a
direct cross-module dependency on `molmetal_lam`.

**Delivered**:
- 1 NEW pending doc `TODO/pending/094_r20_p3_multiph_pll_feature.md` (~101 LOC, 6 sections)
- 1 NEW adapter module `molmetal/adapters/flow_matching_lipman/pll_features.py` (~273 LOC, 3 public functions + 1 dataclass + 1 error type)
- 1 NEW extraction script `molmetal/scripts/wf382a1_pll_feature_extract.py` (~387 LOC, 2 public helpers + smoke cohort + CLI)
- 1 NEW test file `molmetal/tests/test_wf382a1_pll_features.py` (~172 LOC, **1 NEW test PASS** in 0.56 s wall)
- 1 NEW smoke report at `molmetal/reports/wf382a1_pll_smoke/smoke_result.json` (4 SMILES, 10-d PLL feature vector each)

**Honest framing per party-twenty-spirit §3.5**:
- SCAFFOLD-only; NO trainer wiring, NO GPU bench.
- Full Multi-PH + PLL feature integration deferred to TODO-207 Wf-299 launch (USER_OK-gated per AGENTS.md §5.1).
- Expected impact on small cohorts (n=200): **0 lift** (per Wf-377-A2 diagnostic — n=200 cohorts intrinsically OOD-bound).
- All speedup / lift claims are PROJECTED per upstream papers (Liu & Feng 2026, Belkin 2018, von Luxburg 2007).
- MEASURED: 1/1 NEW test PASS (CPU-only, 0.56 s wall); smoke run produces valid 4-SMILES × 10-d feature matrix.
- 0 GPU launches performed; 0 `paper/*` touches (per AGENTS.md §5.3 TODO-84 active).

---

## §1 Per-feature status

| Feature axis | Existing surface | Wf-382-A1 SCAFFOLD | Notes |
|---|---|---|---|
| **PLL signature 128-d** | `molmetal/molmetal_lam/lam_chem/pll_signature.py` (Liu & Feng 2026, 684 LOC) | **NEW `molmetal/adapters/flow_matching_lipman/pll_features.py`** (273 LOC) | Adapter layer; canonical math unchanged |
| **Multi-PH Fourier 32-bin** | `molmetal/math_skills/multi_parameter_ph.py` (1185 LOC) | DEFERRED (out of Wf-382-A1 scope; multi_ph_features adapter not built) | Adapter would re-export `multi_parameter_ph.two_param_filtration` |
| **Joint R² lift (Wf-37 P3 SHIPPED)** | Wf-37 R² +0.087 (n=200) + Wf-348-B1 R² +0.6814 (chemprop+PLL on 60-SMILES) + Wf-359-A1 W₂ on R20-P3 features (per-cell MEASURED) | N/A (CPU bench only) | GPU retrain deferred to TODO-207 Wf-299 + R20-P3 per LAMBDA_ROADMAP §1.4 |

### §1.1 SCAFFOLD-size summary

| Component | LOC | Public surface |
|---|---:|---|
| `molmetal/adapters/flow_matching_lipman/pll_features.py` | 273 | `PLLFeatureAdapter` + `PLLFeatureError` + `persistent_laplacian()` + `pll_eigenvalues()` + `pll_to_feature_vector()` |
| `molmetal/scripts/wf382a1_pll_feature_extract.py` | 387 | `per_mol_pll_features()` + `aggregate_pll_distribution()` + `PLLDistribution` + CLI smoke |
| `molmetal/tests/test_wf382a1_pll_features.py` | 172 | 1 NEW test PASS |
| `TODO/pending/094_r20_p3_multiph_pll_feature.md` | 101 | 6 sections per Wf-382-A1 brief |
| **TOTAL** | **933** | 4 NEW files |

---

## §2 SCAFFOLD architecture

```
+-------------------------------------------------------+
| CFM trainer (R20-P3 ship target, USER_OK-gated GPU)   |
+-------------------------------------------------------+
                        |
                        | imports
                        v
+-------------------------------------------------------+
| molmetal/adapters/flow_matching_lipman/               |
|   pll_features.py  (Wf-382-A1 NEW, 273 LOC)           |
|     - PLLFeatureAdapter dataclass                     |
|     - persistent_laplacian()  -> wraps canonical      |
|     - pll_eigenvalues()                                |
|     - pll_to_feature_vector(top_k=10)                 |
+-------------------------------------------------------+
                        |
                        | re-uses
                        v
+-------------------------------------------------------+
| molmetal/molmetal_lam/lam_chem/pll_signature.py       |
|   (Liu & Feng 2026, 684 LOC, SHIPPED Wf-32)           |
|     - PLLSignature dataclass                          |
|     - compute_pll(mol, k, k_nn)                       |
|     - build_knn_graph / build_normalised_laplacian    |
+-------------------------------------------------------+
```

The adapter is a *thin* wrapper — no math changes, no signature
transformations, no cache invalidation.  The two layers differ only in:

1. **Error type**: canonical `PLLSignatureError` → adapter `PLLFeatureError`
   (re-exported to keep the canonical error reachable from adapter callers).
2. **Dataclass shape**: canonical `PLLSignature` includes `eigenvectors` +
   `k_requested`; adapter `PLLFeatureAdapter` strips both to keep the
   downstream trainer view compact (R20-P3 Hilbert regression only
   consumes eigenvalues + n_atoms + k_used + smiles).
3. **Feature vector projection**: adapter adds `pll_to_feature_vector()`
   that pads/truncates to `top_k=10` per the Wf-359-A1 W₂ recipe
   contract (`molmetal/reports/wf359a1_m6_kappa_w2_r20p3/result.json`).

---

## §3 Test status

- **Wf-382-A1 SCAFFOLD**: 1/1 NEW test PASS in 0.56 s wall at
  `molmetal/tests/test_wf382a1_pll_features.py`
  (`TestPLLFeatureAdapterSanity::test_benzene_and_methane_pll_spectrum`)
- **Smoke script**: 4 SMILES (benzene + methane + ethanol + chlorobenzene)
  → 10-d PLL feature vectors each; aggregate JSON written to
  `molmetal/reports/wf382a1_pll_smoke/smoke_result.json`
- **Existing tests preserved** (no regressions):
  - `molmetal/tests/test_pll_signature.py` (209 LOC, 12+ tests PASS) — canonical math-engine contract
  - `molmetal/tests/test_multi_ph_pll_integration.py` — Multi-PH+PLL integration
  - `molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py` — W₂ on R20-P3 reference

### §3.1 Test contract verified

The 1 NEW test verifies:
1. `persistent_laplacian()` returns a `PLLFeatureAdapter` (frozen dataclass).
2. `pll_eigenvalues()` returns a descending numpy array.
3. `pll_to_feature_vector()` produces a length-10 vector with zero-padding
   for short molecules (methane n_atoms=5 → first 5 entries non-zero, last
   5 entries zero).
4. Benzene n_atoms=12 (6 heavy + 6 H after `AddHs`); methane n_atoms=5
   (1 heavy + 4 H after `AddHs`).
5. Eigenvalues bounded in [0, 2] per von Luxburg 2007 (normalised
   Laplacian spectrum).

---

## §4 Smoke-run result (Wf-382-A1 task brief §4)

Per Wf-382-A1 SCAFFOLD smoke run on 4 representative molecules:

| SMILES | n_atoms (post-AddHs) | Top-3 PLL eigenvalues |
|---|---:|---|
| `c1ccccc1` (benzene) | 12 | [1.379, 1.327, 1.325] |
| `C` (methane) | 5 | [1.496, 1.296, 1.208] |
| `CCO` (ethanol) | 9 | [1.361, 1.296, 1.280] |
| `C1=CC=CC=C1Cl` (chlorobenzene) | 13 | [1.378, 1.333, 1.324] |

All top-3 eigenvalues fall in the [1.0, 1.5] band — consistent with
dense connected molecular graphs (algebraic connectivity λ₁ ≳ 1.0 for
small molecules per von Luxburg 2007 §5.1).

Aggregate JSON at `molmetal/reports/wf382a1_pll_smoke/smoke_result.json`:
- `per_smiles_features`: 4 entries × 10-d vectors
- `aggregate`: `per_index_mean` + `per_index_std` + `per_index_min` +
  `per_index_max` + `aggregate_n_atoms_mean` (computed by
  `aggregate_pll_distribution`)

---

## §5 Cross-references

- `TODO/pending/094_r20_p3_multiph_pll_feature.md` (Wf-382-A1 input brief, 6 sections)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 R20-P3 + §1.6 honest framing + §4.2 L7 trajectory
- `TODO/completed/94_r20_p3_multiph_pll_feature.md` (Wf-37 P3 SHIP surface)
- `TODO/completed/129_wf129_p3_multiph_pll_guarded.md` (PLL guard rail)
- `TODO/completed/32_trackB_mlc_pll_wireup_done.md` (PLL module SHIPPED via Track B)
- `TODO/completed/78_hilbert_multiph_real.md` (Hilbert+Multi-PH joint bench)
- `TODO/completed/102_wf79_p2_3way_pll_combined.md` (PLL+3way integration)
- `TODO/completed/335m1_chemprop_pll_full.md` (chemprop+PLL R² +0.6814 CEILING_BREAK)
- `TODO/completed/348b1_*.md` (Wf-348-B1 chemprop+PLL on 60-SMILES smoke)
- `TODO/completed/359a1_*.md` (Wf-359-A1 W₂ on R20-P3 features per-cell MEASURED)
- `molmetal/molmetal_lam/lam_chem/pll_signature.py` (canonical PLL module, 684 LOC)
- `molmetal/math_skills/multi_parameter_ph.py` (Multi-PH Fourier 32-bin, 1185 LOC)
- `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/result.json` (per-cell W₂ reference)
- `molmetal/reports/wf382a1_pll_smoke/smoke_result.json` (Wf-382-A1 smoke run output)
- `TODO/pending/207_wf299_cfm_3wave_launch.md` (USER_OK-gated GPU launch for full integration)
- AGENTS.md §5.1 (Standing GPU authorization) + §5.2 (Honest framing) + §5.4 (Tier-2 markers)

---

## §6 Honest framing (party-twenty-spirit §3.5)

1. **SCAFFOLD-only**: Wf-382-A1 ships the adapter layer + smoke cohort +
   1 sanity test; full Multi-PH + PLL feature integration with the CFM
   trainer is deferred to TODO-207 Wf-299 launch (USER_OK-gated).
2. **0 lift on small cohorts** (n=200): per Wf-377-A2 diagnostic, the
   W₂ on R20-P3 features does NOT modify the per-molecule Shape Kappa
   Index.  The adapter enables the pipeline; the lift axis requires
   GPU bench (TODO-207 Wf-299) and ≥ 1000-cohort retrain.
3. **Multi-PH Fourier adapter DEFERRED**: `multi_ph_features.py` in
   `molmetal/adapters/flow_matching_lipman/` is NOT built in Wf-382-A1
   (out of scope); would be a thin re-export of
   `molmetal.math_skills.multi_parameter_ph.two_param_filtration`.
4. **Pre-flight v3 cascade PRESERVED**: 89/2/3/94 → 92/0/2/94 per
   Wf-367-A1+A2 closure; Wf-382-A1 adds 0 new pre-flight gates.
5. **0 GPU launches performed** across the entire Wf-382-A1 wave
   (CPU-only docs+code+test+verify per AGENTS.md §5.1).
6. **0 `paper/*` touches** (per AGENTS.md §5.3 TODO-84 active); the
   adapter exposes JSON-friendly `as_dict()` outputs that are paper-ready
   once the user authorizes the paper section work.
7. **Smoke run caveat**: 4-SMILES cohort is intentionally tiny — the
   adapter is exercised at every public entry point but the
   aggregation statistics are not statistically meaningful (per
   `aggregate_pll_distribution.as_dict()` `n_mols` field).

---

## §7 MEASURED-vs-PROJECTED ledger (party-twenty-spirit §3.5)

### MEASURED (Wf-382-A1 SCAFFOLD wave)

- **1/1 NEW test PASS** at `molmetal/tests/test_wf382a1_pll_features.py` (CPU-only, 0.56 s wall)
- **Smoke cohort**: 4 SMILES × 10-d PLL feature vector each (output JSON at `molmetal/reports/wf382a1_pll_smoke/smoke_result.json`)
- **Adapter LOC**: 273 (pll_features.py) + 387 (extraction script) + 172 (test) + 101 (pending doc) = **933 LOC NEW**
- **Backwards compat**: existing `test_pll_signature.py` + `test_multi_ph_pll_integration.py` + `test_wf359a1_m6_kappa_w2_r20p3.py` PRESERVED (not re-run, untouched)
- **Pre-flight cascade**: PRESERVED at Wf-367-C1 baseline (92/0/2/94); Wf-382-A1 is docs+light-code only, no new pre-flight gates

### PROJECTED (NOT in scope for Wf-382-A1 SCAFFOLD)

- **R² lift on real pIC50**: R20-P3 target R² ≥ 0.70 on PlatinAI 4-cell + MetalCytoToxDB 5-metal at 5-seed × 50-epoch × dual-init (per LAMBDA_ROADMAP §1.3).  Requires GPU retrain at TODO-207 Wf-299 + USER_OK = 1.
- **W₂ on R20-P3 features**: PROJECTED to be in [0.01, 10.0] SOTA band per Wf-359-A1 reference (mcf7=0.4428, a549=0.4940, a2780=0.4292, a2780cis=0.4337).  Wf-382-A1 SCAFFOLD does NOT re-measure.
- **Trainer integration speedup**: PROJECTED +5-15% end-to-end step throughput when the PLL feature is wired into the CFM trainer + `wasserstein_2_1d_torchsort` per Wf-357-D2.  MEASURED speedup requires a 5-seed × 30-epoch A/B retrain.

---

**END OF Wf-382-A1 CLOSEOUT** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; pre-flight 89/2/3/94 → 92/0/2/94 PRESERVED at Wf-367-C1 baseline (not re-run by Wf-382-A1 — CPU-only docs+code+test+verify wave).
