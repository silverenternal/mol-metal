# Ethics Statement — Mol-Metal (R17 / R18 close-out)

**Date:** 2026-09-26 (Wf-26 Phase 4 ship)
**Scope:** Ethical considerations for the work reported in the manuscript
and supplementary items S1–S8.

---

## 1. Human and animal subjects

This work involves **no human subjects** and **no animal testing**. All
datasets are molecular / chemical (PlatinAI 5K, MetalCytoToxDB 5-metal
subset, tmQM 108K, CrossDocked2020 pocket list, ChEMBL 33 cytotoxicity
endpoints — see `DATA_AVAILABILITY.md`). No biological tissue, clinical
samples, patient records, or personal data are used at any stage.

## 2. AI safety and automated decision-making

The system described here is **pure computational / in-silico**:
- All model components are generative chemistry models operating on
  tokenised SMILES / molecular graphs and 3-D coordinates.
- No component makes a clinical, diagnostic, prescribing, or any other
  decision affecting a natural person.
- The pipeline has no autonomous deployment loop; every output is a
  ranked candidate list that a human chemist reviews before any
  downstream use.
- No model output is acted upon without expert-in-the-loop review.

## 3. Dual-use considerations

Generative drug-design models are inherently **dual-use**: the same
generative capacity that proposes candidate metallodrugs could, in
principle, be redirected to harmful molecules. We address this as
follows:
- The paper **reports a method**, not drug candidates. We do not
  enumerate, characterise, or advocate specific generated molecules
  as deployable therapeutics.
- The released benchmark corpus contains only established literature
  compounds (PlatinAI, tmQM, MetalCytoToxDB) and a small curated
  subset for evaluation — not generated hits.
- The training data contains no chemical-warfare, biological-toxin,
  or controlled-substance molecules to our knowledge; the upstream
  corpora are chemistry-curated and assay-derived.
- The shipped model weights are released under MIT; downstream users
  are responsible for their own dual-use review (consistent with
  precedent set by RxnFlow, SyntheMol, and DiffDock releases).

## 4. Broader impact

Positive: lowers the entry cost for *in-silico* exploration of
metallo-drug space, particularly for Pt / Ru / Ir / Au / Pd
chemotypes that are under-represented in generic generative pipelines.
Negative (mitigated): the method is bounded to known metal–ligand
chemistries defined by the typed β-NF + click-rule library; we do not
extrapolate beyond the curated reaction set.

## 5. Contact

Ethics-related questions go to the corresponding author via the
address on the manuscript's title page.
