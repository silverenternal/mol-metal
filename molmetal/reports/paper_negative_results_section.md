# Negative Results — What Did Not Work and What We Learned

*Paper §X (Discussion). Drop-in section. ~650 words.*

---

We document four findings where our pipeline, baselines, or eval protocol produced weaker-than-expected results, and what each tells us about the broader molecular-ML and SBDD methodology.

**(1) Pocket2Mol head-to-head on 1h36 is indistinguishable from a SMILES fallback (n=100).** Pocket2Mol `mean_vina=-5.95`, `success=0.189`; Lambda `mean_vina=-5.92`, `success=0.194`. The Pocket2Mol numbers come from a `smarts_fallback` adapter because `pretrained_Pocket2Mol.pt` is not shipped — our policy forbids pulling third-party pretrained weights. *Interpretation:* a 0.02 kcal/mol gap is not a real result; it is the SMILES fallback talking. Any reported SBDD baseline run without the original checkpoint is, in practice, comparing against a drug-like SMILES pool, not against the published model. Lesson: ship the checkpoint or do not claim the comparison.

**(2) Counter-ion features are a previously-unreported spurious-correlation channel (ΔAUC ≈ −0.076 on temporal split).** Adding 5 binary counter-ion indicators (PF6⁻, OTf⁻, Cl⁻, BF4⁻, NO3⁻) to a Morgan-ECFP4 / XGBoost baseline on MetalCytoToxDB Ru costs nothing on random splits (ΔAUC ≈ −0.003) but costs 0.076 AUC on a 2024-cutoff temporal split. Manual inspection shows OTf⁻ and PF6⁻ grow monotonically with publication era while Cl⁻ declines — a *publication-era drift*, not a chemistry mechanism. *Interpretation:* modular chemistry features (ligand, metal, counter-ion, co-ligand) create high-order conditional dependencies invisible to scaffold splitting. Lesson: for OOD evaluation, *omit* counter-ion features; report a seen-fraction metric; random-split AUC alone is misleading.

**(3) D-MPNN is *more* vulnerable to chemistry drift than XGBoost.** On the Ru temporal split, attentive D-MPNN drops to AUC ≈ 0.51, vs XGBoost at 0.64. D-MPNN is less leaky (Ru 0.79 → 0.66 dedup gap) but more sensitive to scaffold/temporal drift. *Interpretation:* GNNs smooth over fingerprint-level leakage but over-fit the latent chemistry manifold, so the *prospective* generalisation gap is larger than for fixed fingerprints. Lesson: report *both* leak-free and temporal-split AUC; never pick the higher number.

**(4) PoseBusters pass-rate was 0 % on chemically valid molecules.** The PoseBusters adapter treated integer count columns (`num_h_added`, `number_clashes`) as boolean checks, so a healthy count of 0 was reported as "failed". Switching 3D optimisation to MMFF94 (with UFF fallback) and restricting `pass_rate` to bool-dtype columns brought 25/25 hand-drawn CuAAC/SPAAC triazole products and click tiles to pass-rate = 1.000. *Interpretation:* we almost reported a synthetic-zero PoseBusters number that would have implied our click-chemistry candidates were non-drug-like. Lesson: any third-party validity/QC adapter needs a *gold-positive* sanity check before being trusted — silent dtype coercions are silent killers.

**General lessons.** Across (1)–(4): *without the original pretrained weights, an SBDD baseline is not a baseline.* *Modular featurisation can produce spurious correlations that survive scaffold splitting.* *GNN smoothing hides fingerprint leakage but inflates drift.* *Validity adapters must be sanity-checked on gold positives.* The honest protocol for this paper is therefore: published numbers cited, not reproduced (rule: no third-party weights); temporal split + no-counter-ion + seen-fraction reported jointly; PoseBusters guarded by gold-positive tests; per-axis `NOT REPORTED` flags rather than fabricated values. Negative results tighten the empirical perimeter of the paper more than another 0.05 Vina improvement would.

---

*Word count: ~630. References to: `pocket2mol_vs_lambda_1h36.md`, `f4_paper_leakage_section.md` §5.3.2–5.3.5, `mmff94_fix.md`, `h5_lambda_honest_framing.md` §2.*
