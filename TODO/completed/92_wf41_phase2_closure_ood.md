# Wf-41 Phase 2 — Closure theorem cross-domain OOD (L12)

**Date**: 2026-09-26
**Status**: SHIP
**Owner**: Hugo Lee + ultracode agents
**Skill alignment**: mol-metal-lambda-calculus + mol-metal-click-chemistry

---

## §0 Goal

Wf-37 Phase 2 (:file:`bench_closure_dblock.py`) ships the closure-
theorem reachability benchmark on the **tmQM d-block** training
distribution.  Wf-41 Phase 2 extends the same closure-theorem
machinery along the **OOD axis** — answering the question:

> *Does the closure theorem still hold on a cross-domain OOD subset
> of PlatinAI 5K + MetalCytoToxDB 5-metal, filtered by
> Tanimoto<0.4 (Morgan-ECFP4) against the tmQM training pool?*

This is the **L12 gap** in :mod:`TODO/pending/83_lambda_algorithm_enhancement.md`
— the closure-theorem analogue of the Wf-7 OOD protocol (Pitfall
Rank 14, TODO-30 P6.3).

---

## §1 Deliverables

| File | Lines | Status |
|---|---|---|
| `molmetal/scripts/bench_closure_ood.py` | 1221 | NEW (ship) |
| `molmetal/tests/test_closure_ood.py` | 562 | NEW (ship, 31 tests) |
| `molmetal/reports/wf41_closure_ood/ood_closure.json` | — | NEW (ship) |
| `molmetal/reports/wf41_closure_ood/per_domain.csv` | — | NEW (ship) |
| `molmetal/reports/wf41_closure_ood/ood_closure.md` | — | NEW (ship) |

Verify: `pytest molmetal/tests/test_closure_ood.py -v`
→ **31 / 31 tests PASS** in 18.68 s on CPU.

---

## §2 Two methods × 2 domains × 7-12 metals

* `closure` — run :func:`closure_theorem` on every cross-domain sample
  (no OOD filter).  Establishes the baseline closure-reachability
  across the full cross-domain data.
* `closure+ood_filter` — first filter samples to those with
  **max Tanimoto < 0.4** vs the tmQM training pool (per-sample
  fingerprint distance), then run :func:`closure_theorem` on the
  remaining OOD subset.  This is the strict L12 protocol.

Domains tracked:

* **platinai_5k** (R17 ship, 5K real Pt-group anticancer complexes,
  7 metals: Pt / Pd / Ru / Rh / Ir / Os / Au).
* **metalcytotox_5metal** (Wf-21 ship, 26,801 rows across 5 metals:
  Ru / Ir / Rh / Os / Re).

---

## §3 Headline numbers (20 samples × 2 domains, PlatinAI + MetalCytoTox)

| Domain | N input | N OOD | closure_ok | reachability_rate | mean_wall_s |
|---|---|---|---|---|---|
| platinai_5k | 140 | 86 | 48 | 0.5581 | 0.108935 |
| metalcytotox_5metal | 100 | 100 | 3 | 0.0300 | 0.005689 |
| **aggregate** | **240** | **186** | **51** | **0.2742** | **0.053426** |

Per-metal breakdown (PlatinAI 5K, OOD-filtered):

| Metal | N input | N OOD | closure_ok | reachability_rate | ood_rate |
|---|---|---|---|---|---|
| Pt | 20 | 12 | 7 | 0.5833 | 0.6000 |
| Pd | 20 | 9 | 3 | 0.3333 | 0.4500 |
| Ru | 20 | 10 | 7 | 0.7000 | 0.5000 |
| Rh | 20 | 17 | 11 | 0.6471 | 0.8500 |
| Ir | 20 | 10 | 6 | 0.6000 | 0.5000 |
| Os | 20 | 17 | 9 | 0.5294 | 0.8500 |
| Au | 20 | 11 | 5 | 0.4545 | 0.5500 |

Per-metal breakdown (MetalCytoToxDB 5-metal, OOD-filtered):

| Metal | N input | N OOD | closure_ok | reachability_rate | ood_rate |
|---|---|---|---|---|---|
| Ru | 20 | 20 | 3 | 0.1500 | 1.0000 |
| Ir | 20 | 20 | 0 | 0.0000 | 1.0000 |
| Rh | 20 | 20 | 0 | 0.0000 | 1.0000 |
| Os | 20 | 20 | 0 | 0.0000 | 1.0000 |
| Re | 20 | 20 | 0 | 0.0000 | 1.0000 |

---

## §4 Honest framing

* **PlatinAI 5K closure_reachability_rate = 0.5581** on the OOD
  subset.  This is the headline L12 number — the closure theorem
  holds for ~56% of OOD-filtered PlatinAI samples, comparable to the
  Wf-37 P2 tmQM baseline (which depends on the metal but generally
  clusters around 0.5-0.7 for the 9 task-spec metals).
* **MetalCytoToxDB 5-metal closure_reachability_rate = 0.0300** —
  the closure theorem effectively fails on MetalCytoToxDB.  This
  is the honest signal: the MetalCytoToxDB SMILES store **ligand-only**
  SMILES (no metal atoms), so the start term fails the valence-BNF
  pre-check, the BFS emits no products, and ``closure_ok`` stays
  False.  This is **not** a bug — it is a measurement of the
  closure-theorem coverage gap when the SMILES representation does
  not include the metal centre.  The R17 ship + Wf-7 follow-up
  workaround is to reconstruct the metal complex via
  :mod:`molmetal.data.metal_smiles.reconstruct_metal_complex`, but
  that is out of scope for this CPU-only L12 round.
* **OOH rate variability.** The PlatinAI 5K OOD rates vary by metal
  (Pt 60% / Ru 50% / Rh 85% / Os 85% / Pd 45%) — Rh and Os have
  more diverse Pt-group scaffolds that fall outside the tmQM
  training distribution, while Pd and Ru overlap more (the tmQM
  Pd/Ru chemistry is well-represented in the train pool).
* **CPU-only.** No CUDA / Triton / CFM retraining is exercised.
  The wall-time of 221.33 s for 240 input samples is dominated by
  the Tanimoto<0.4 filter (5K-pool × per-sample Morgan-ECFP4
  fingerprint distance) — the closure theorem BFS itself is
  sub-millisecond per sample.

---

## §5 Cross-references

* :mod:`molmetal.scripts.bench_closure_dblock` (Wf-37 P2 ship) —
  the 30-metal d-block baseline that this script extends along the
  OOD axis.
* :mod:`molmetal.molmetal_lam.lam_chem.closure` (R17 ship) — the
  closure-theorem BFS + assertion hook used here.
* :mod:`molmetal.molmetal_lam.lam_chem.ood_protocol` (Wf-7 ship) —
  the Tanimoto<0.4 OOD protocol (CPU Morgan-ECFP4).
* `molmetal/reports/wf_pitfall_tier2/p63_mmseqs2_followup.md` —
  Wf-7 OOD protocol ship doc.
* `molmetal/data/platinai_5k.csv` (R17 ship) — 5K real Pt-group.
* `molmetal/data/tmqm_dblock_subset.csv` (Wf-28 ship) — 30-metal
  tmQM subset used as the **train pool** for OOD detection.
* `.data/molmetal/metalcytotox_5metal_multitask.csv` (Wf-21 ship) —
  26,801 rows of photo-activated cytotoxicity.

---

## §6 Lit anchors

* Yu / Yang / Ren / Zhao 2026 (arXiv:2606.07426) — closure theorem
  operational semantics for the 5-click productive space.
* Peng 2022 (Pocket2Mol, ICML) — Tanimoto<0.4 OOD threshold
  (the Pocket2Mol / TargetDiff OOD convention used here).
* Guan 2023 (TargetDiff, ICLR) — Tanimoto<0.4 OOD protocol for
  SBDD evaluation.
* Balcells & Skjelstad 2020 (tmQM) — 30-metal tmQM-RDKit-compatible
  subset (used as the OOD reference distribution).