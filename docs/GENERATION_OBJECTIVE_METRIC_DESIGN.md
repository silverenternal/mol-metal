# Generation Objective — Metric Design

> **Status (2026-10-10):** design proposal, awaiting user sign-off. Nothing here
> is implemented. Every comparison-work claim is a **search-derived lead, not a
> paper that has been read in full** — see §4 before citing any of it externally.

## 0. Why this document exists

The CFM generator has no generation objective. Measured on 2026-10-10 (BLK-C12):
**8 of the 12 registered loss terms are exact zeros**, and `bond_ce` is ~95% of
the live loss by magnitude. The optimiser is therefore never asked to generate
well — it is asked to predict bonds, and the sampler inherits that.

A clean single-variable A/B (BLK-C11) then showed the consequence:

| | `use_fm_objective: true` | `false` |
|---|---|---|
| `gen_validity` @ ep29 | 0.0625 | **0.0000** |
| `decode_ratio` @ ep29 | 0.5833 | 0.2500 |
| `bond_f1` @ ep29 | 0.6703 | **0.7836** |

Both arms collapse at the same epoch. Flow matching roughly doubles residual
generation capability and costs 0.113 bond F1. **It is a mitigation, not a
cure**, because the cause is the loss composition, not the missing FM term.

So: what does "an actually effective noble-metal anticancer drug" mean as a
number we can put in a loss (or a sampler)?

## 1. What the comparison work actually optimises

Four works, retrieved 2026-10-10.

### 1.1 PlatinAI — Rusanov et al., ChemRxiv 2025-pp32k (under Nat. Biotech. revision)

- **Scope:** Pt only. 4,078 curated complexes / 18,995 IC₅₀ values.
- **Method:** metal-aware deep learning (KANO / metal-aware graph attention),
  fragment assembly, 214,373 screened.
- **Main metric:** predicted anticancer **activity** — ROC-AUC, precision,
  recall, F1, across MCF-7, A549, A2780 and **A2780ᶜⁱˢ (resistant)**.
- **Validation:** prospective — ~20 compounds synthesised and tested, ~72%
  agreement with prediction.
- **Output:** a bis-NHC Pt(II) complex that beats cisplatin in A2780ᶜⁱˢ via a
  **non-DNA-adduct mechanism**.
- **Stated future work:** *"predicting geometries"* — coordination geometry is
  explicitly **not** handled.

### 1.2 MetalCytoToxDB — Krasnov et al., ChemRxiv 2025-1nqvm → J. Med. Chem. 2026

- **Scope:** Ru/Ir/Rh/Re/Os. 26,500 IC₅₀, 7,050 complexes, 754 cell lines,
  1,921 papers.
- **Method:** composition-only features, classical ML.
- **Main metric:** binary cytotoxicity **ROC-AUC** (Ru 0.81, Ir 0.73) plus a
  >2× hit-rate improvement over random sampling.
- **Notable:** the authors explicitly *"explore the possibility of multi-metal
  predictions"*.
- **Limits:** not generative, **no leave-one-metal-out**, **no Pt**.

### 1.3 Biginelli THPM–Ru — Simović et al., J. Inorg. Biochem. 2025

- **Scope:** CReM generative model → 28 derivatives → Ru(II)-*p*-cymene.
- **Main metrics:** binding energy < −31.3 kJ/mol; docking vs Caspase 3/8;
  **selectivity index up to 19** vs normal cells; Kb/Ksv against DNA and HSA;
  MTT cytotoxicity.

### 1.4 JT-VAE — Strandgaard et al., J. Chem. Au. 2025, 5(5) 2294–2308

- **Scope:** tmQMg-L ligand library; conditional generation on HOMO–LUMO gap,
  metal-centre charge, solubility, steric bulk.
- **Caveat:** the generated complexes are homoleptic **iridium**. Platinum does
  not appear in the output set.

**Common thread: the main metric is always "predicted activity in a named cell
line."** Nobody makes cross-metal generalisation a *generation* objective.

## 2. Three verified gaps

| # | Gap | Evidence | Blocked? |
|---|---|---|---|
| **A** | No formal **leave-one-metal-out** protocol as a named methodology | Search synthesis: this framing "does not appear as a named methodology in any single paper". The pieces are distributed — MetalCytoToxDB does cross-metal prediction without LOMO/transfer/Pt; Duarte & Bonnet's transfer learning is organic→Ru only; Harper et al. (JCP 2022) has the transfer machinery but for materials properties. | no |
| **B** | No generative model for **binuclear / bimetallic Pt–Ru** prodrugs | Pt–Ru hybrids are made by one-pot self-assembly (Chem. Sci. 2022, 13, 12460) or manual conjugate design; the ML side is property prediction only. | **yes** — essentially no training data for binuclear complexes, plus Pt(IV) octahedral / axial-ligand / oxidation-state bookkeeping |
| **C** | No **geometry-constrained** generative model for the Pt octahedral coordination sphere | PlatinAI names geometry prediction as future work; no 2024–25 paper combines a generative model with explicit octahedral coordination-sphere generation for Pt. | partly |

## 3. Proposed metric design

**PRIMARY — cross-metal generalisable generation.**
Generate candidates, and require predicted activity to hold on metals the model
never saw. Report leave-one-metal-out AUC and retained hit-rate.

Chosen because it is (a) clinically no narrower than single-metal activity —
the field's own leading database paper already flags multi-metal as the open
question; (b) absent as a named *generation* objective; (c) the project's
existing strength, since LOMO across 5–6 metals already runs; (d) not
data-blocked, unlike gap B.

**SECONDARY — predicted anticancer activity.**
pIC₅₀ / ROC-AUC in a **named** cell line, with the resistant line (A2780ᶜⁱˢ) as
the discriminating case, matching PlatinAI's setup. Report prospective-style
agreement if any synthesis is ever done.

**Ordering:** the primary is the differentiator and is unrepresented in the
comparison set. The secondary is what a reviewer will check against, so it is
reported even though everyone already does it.

## 4. Honest boundary on novelty

> Absence from search is **not** proof of absence in the literature.

Gap A rests on a search-synthesis statement, not an exhaustive Scopus/Scholar
scan. Gaps B and C have harder evidence — PlatinAI writes its own future-work
line. Before claiming novelty in a submission, run a proper systematic scan.

Nothing in this document has been read in full. Treat every entry as a lead.

## 5. Cross-references

- BLK-C11 (the clean A/B that motivated this), BLK-C12 (loss composition),
  BLK-C13 (freeze removes the gradient path), BLK-C15 (execution steps),
  BLK-D2 (metal absent from the featurisation), BLK-C5 (the
  gradient-detached-constant trap this objective must not repeat).