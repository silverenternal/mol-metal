# WF-CFM-Rescue Phase 1 — YuelBond decoder swap

**Date:** 2026-09-16
**Scope:** Replace the ``decode_distance_graph`` covalent-radius heuristic
with a learned GNN edge predictor (Wang & Dokholyan 2025 YuelBond
bioRxiv 10.1101/2025.05.06.652517) that maps (positions, atomic_numbers)
to per-pair bond logits.
**Status:** Phase 1 SHIPPED — module + tests + report. The decoder is
available as a drop-in alternative to the existing BondOrderHead pipeline
but the existing pipeline (used in production) is the wired default.

---

## 1. Files

- **NEW**: `molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py` (~290 LOC)
  - `YuelBondDecoderHead` — small message-passing GNN (atom encoder
    + K=3 message-passing layers + pair head → 5-class bond logits)
  - `YuelBondDecoder` — wrapper that emits `PairFeature` + `YuelBondResult`
  - `YuelBondResult` — dataclass container with pair_features, logits,
    edge_index, decode_succeeded, error
  - Vit: Z one-hot via `_YUELBOND_Z_TABLE` (matches round-10 atom vocab)
  - Vit: distance cutoff 3.5 Å (excludes non-bonded pairs from E) —
    **DECLARED DEVIATION**: the published paper's candidate-edge cutoff is
    **3 Å** ("any pair of atoms within 3 A is connected by an edge"); 3.5 Å is
    this repo's own value, deliberately UNCHANGED (changing it would
    invalidate every trained checkpoint and every measured decode metric, and
    re-validation needs a GPU run this CPU-only chain may not do).  Every
    number produced under 3.5 Å must be labelled as such, not as the paper's
    3 Å protocol.
  - Vit: dative candidate flag for (N/O/S) ↔ (Pt) pairs (any ordering)

- **NEW**: `molmetal/tests/test_cfm_rescue_yuelbond.py` (5 tests)

> **⚠️ DEVIATION DISCLOSED (2026-10-31, Wf-A148).** The bullet above
> previously read "  - Vit: distance cutoff 3.5 Å (excludes non-bonded pairs
> from E)" and carried no deviation flag, so a reader could take 3.5 Å for the
> paper's protocol.  The published paper's candidate-edge cutoff is **3 Å**
> ("any pair of atoms within 3 A is connected by an edge", at four independent
> sites in the published text) — so 3.5 Å is this repo's own DECLARED
> DEVIATION, and it is deliberately UNCHANGED: changing it would invalidate
> every trained checkpoint and every measured decode metric, and re-validation
> needs a GPU run that this CPU-only chain is forbidden to perform.  The
> constant's definition sites are
> `molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py`
> (`YuelBondDecoderHead.cutoff = 3.5`, `RBF_MAX_DISTANCE = 3.5` — the RBF
> expansion-range top, NOT the candidate-edge cutoff),
> `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`
> (`DEFAULT_EDGE_CUTOFF = 3.5`) and
> `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py`
> (`distance_cutoff = 3.5`).  A DIFFERENT constant —
> `molmetal/adapters/flow_matching_lipman/__init__.py::bond_candidate_cutoff`
> = 2.4 — is a reference site, not a definition site.  Superseded wording
> preserved verbatim per AGENTS.md §5.5 rule #6.

## 2. Architecture

```
YuelBondDecoderHead (≤5k params)
    ├─ AtomEncoder: Z one-hot (13) -> Linear -> H
    ├─ K=3 Message-passing iterations:
    │   for k in range(K):
    │       msg_i = MLP([h_i, h_j, d_ij, is_short, is_long, is_dative])
    │       h_i += scatter_add(msg_i, dim=0)
    └─ PairHead: Linear(2H+3, H) -> ReLU -> Linear(H, 5)

YuelBondDecoder
    └─ featurise(cloud) -> YuelBondResult
          - logits: (E, 5)
          - pair_features: edge_index (2,E), distance, z_i, z_j,
            angle_to_metal=0, is_dative_candidate
          - decode_succeeded: bool
```

## 3. Lit anchor

Wang, J.; Dokholyan, N. V. (2026). *Multimodal Bond Reconstruction toward
Generative Molecular Design.*  J. Chem. Inf. Model. 66(2), 1003-1012.
DOI 10.1021/acs.jcim.5c03052  (preprint: bioRxiv 10.1101/2025.05.06.652517,
2025-05-11).
- F1=92.7 % on the CDG set (Gaussian noise sigma = 0.2 A) — the paper's
  worst case; F1 = 98.4 % on standard 3-D structures
- **WHAT THE PAPER MEASURED** (published: Wang, J.; Dokholyan, N. V.
  (2026). *Multimodal Bond Reconstruction toward Generative Molecular
  Design.*  J. Chem. Inf. Model. 66(2), 1003-1012,
  doi 10.1021/acs.jcim.5c03052; preprint: bioRxiv 10.1101/2025.05.06.652517,
  2025-05-11): RDKit's `DetermineBonds` — which internally leverages
  `xyz2mol` — processed only 217 of 1,000 CDGs (=> 783 failures), and
  YuelBond scored **92.7 % F1 on that same CDG set** (Gaussian noise
  sigma = 0.2 A, the paper's worst case).  The paper attributes that integer
  split to RDKit itself and prints no percentage for it.  The baseline the
  paper benchmarks is `DetermineBonds`; `DetermineConnectivity` is the
  connectivity-only step that `DetermineBonds` calls internally before
  assigning bond orders — the two are not independent siblings (MEASURED
  locally: every bond `DetermineConnectivity` emits is SINGLE, while
  `DetermineBonds` emits SINGLE/DOUBLE/AROMATIC).
- **WHAT THIS REPO HAS (not the paper's result).** The YuelBond decoder
  module exists here — `molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py`
  (`YuelBondDecoderHead` / `YuelBondDecoder`), 5 unit tests in
  `molmetal/tests/test_cfm_rescue_yuelbond.py` — but it is **not** the
  production decode path
  (`molmetal/molmetal_lam/lam_chem/cfm_decoder.py::CFMDecoder` defaults to
  `use_yuelbond=False`; its YuelBond branch lazily builds a fresh,
  UNTRAINED head), and this repo holds **no trained YuelBond checkpoint**:
  every checkpoint path recorded in `logs/yuelbond*/seed_*/final.json`
  (`.../ckpt.pt`) is missing from disk, the only CFM checkpoint files under
  `molmetal/checkpoints/` are the two `*_smoke_stub.pt` stubs, and the
  recorded multiseed decode probes report `decode_ratio = 0.0` with error
  `TypeError: 'model' must be a VelocityNet; got ModelWrapper.`
  **The recovery of the 783 RDKit failures is therefore NOT MEASURED in this
  repo — it is PROJECTED**, and it is an **UPPER** bound on what our
  decoder could achieve, not a lower one: the paper's F1 FALLS as
  distortion grows and 92.7 % was measured at sigma = 0.2 A, whereas our
  CFM's generated-geometry distortion is UNCONTROLLED — no sigma-controlled
  CDG benchmark exists in this repo (the only CDG references in the code
  are docstrings/comments quoting the paper's F1).
- Projected lift: 0% → 0.45-0.60 decode (standalone, no retrain)

> **⚠️ MEASURED-vs-PROJECTED SPLIT (2026-10-31, Wf-A148).** The bullet above
> previously read "- RDKit's `DetermineBonds` — which internally leverages
> `xyz2mol` — processed only 217 of 1,000 CDGs (=> 783 failures) → YuelBond
> recovers.  The paper attributes that integer split to RDKit itself and
> prints no percentage for it.  The baseline the paper benchmarks is
> `DetermineBonds`; `DetermineConnectivity` is the connectivity-only sibling
> and assigns no bond orders (MEASURED locally: every bond it emits is
> SINGLE)."  The clause "→ YuelBond recovers" was UNQUALIFIED inside a repo
> report whose own YuelBond module is not the production decode path and has
> no trained checkpoint in this repo — a MEASURED-vs-PROJECTED violation
> (AGENTS.md §5.2).  The claim is now split: (a) WHAT THE PAPER MEASURED — the
> 217/1,000 integer split and the 92.7 % F1 on the CDG set at sigma = 0.2 A
> — attributed to the paper (published: J. Chem. Inf. Model. 2026, 66(2),
> 1003-1012, doi 10.1021/acs.jcim.5c03052; preprint: bioRxiv
> 10.1101/2025.05.06.652517, 2025-05-11); and (b) WHAT THIS REPO HAS — the
> module exists and is unit-tested, it is not the production decode path, no
> trained YuelBond checkpoint exists on disk, so the recovery of the 783 RDKit
> failures is NOT MEASURED here and the lift is PROJECTED — in agreement with
> the pre-existing "Projected lift: 0% → 0.45-0.60 decode (standalone, no
> retrain)" line.  (The "sibling" wording inside the quoted bullet is
> separately corrected in the next note.)  Superseded wording preserved
> verbatim per AGENTS.md §5.5 rule #6.
>
> **⚠️ RELATION CORRECTED (2026-10-31, Wf-A148).** The superseded bullet above
> described `DetermineConnectivity` as "the connectivity-only sibling" of
> `DetermineBonds`.  They are not independent siblings: `determineBonds` calls
> `determineConnectivity` internally before assigning bond orders
> — `Code/GraphMol/DetermineBonds/DetermineBonds.cpp`, the call at L543 in
> rdkit Release_2026_03_6 (identical on master; the same call is at L418 in
> Release_2022_09_1, so this is long-standing behaviour, not a recent
> refactor).  The Python `DetermineBonds` binding transitively executes the
> same `determineConnectivity` function that the Python `DetermineConnectivity`
> binding executes.  MEASURED locally on rdkit 2026.03.6 (aspirin + Hs,
> ETKDGv3, seed 42): both entry points emit 21 bonds and the bond SETS are
> identical (symmetric difference = empty); `DetermineBonds` assigns
> SINGLE/DOUBLE/AROMATIC bond orders while `DetermineConnectivity` leaves
> every bond SINGLE.  Superseded wording preserved verbatim per AGENTS.md
> §5.5 rule #6.

> **⚠️ CITATION CORRECTED (2026-10-31, Wf-A143).** The lit anchor above
> previously read "Wang, S., Dokholyan, P. (2025). *YuelBond: a graph neural
> network for recovering the molecular bond network from distorted 3D
> structures.* bioRxiv 10.1101/2025.05.06.652517." and attributed the
> 783/1000 failure count to RDKit's `DetermineConnectivity`.  That title
> matches neither the preprint nor the published paper, the author
> initials were wrong, and `DetermineConnectivity` is the wrong RDKit entry
> point for a bond-**order** baseline.  The historical text is preserved
> verbatim in `docs/HONEST_NEGATIVE_CATALOG.md` §6.257.
> **⚠️ ATTRIBUTION CORRECTED (2026-10-31, Wf-A146).** The bullet above
> previously read "- RDKit's `xyz2mol` / `DetermineBonds` processed only
> 217 of 1,000 CDGs (=> 783 failures) → YuelBond recovers."  The paper
> attributes the 217/783 split to **RDKit** itself ("Out of 1,000 test
> molecules with noisy coordinates, RDKit was able to process only 217,
> while it failed on the remaining 783 molecules"), prints that integer
> split rather than a percentage, and names `xyz2mol` only as the program
> `DetermineBonds` "internally … leverages".  Superseded wording
> preserved verbatim per AGENTS.md §5.5 rule #6; see
> `docs/HONEST_NEGATIVE_CATALOG.md` §6.259.

Secondary anchor: Liu, Z., et al. (2025). *NExT-Mol: 1D SELFIES + 3D
conformer decoupling for molecular generation.*  ICLR 2025
(arXiv:2502.12638). MoLlama 1.8B-pretrain shows learned bond topology
outperforms covalent-radius heuristics on OOD scaffolds.

## 4. Tests (5/5 pass)

```
$ uv run pytest molmetal/tests/test_cfm_rescue_yuelbond.py --tb=short
molmetal/tests/test_cfm_rescue_yuelbond.py ..... [100%]
5 passed in 1.57s
```

Coverage:
- `test_yuelbond_head_forward_returns_correct_shape` — (E, 5) logits + edge count
- `test_yuelbond_decoder_featurise_pairs_match_edge_count` — PairFeature consistency
- `test_yuelbond_decoder_handles_empty_cloud` — N=0 and N=1 don't crash
- `test_yuelbond_decoder_distance_cutoff_filters_long_pairs` — 10 Å pairs excluded
- `test_yuelbond_decoder_dative_flag_for_N_Pt_pair` — (Z_i, Z_j) donor↔Pt detection

## 5. Honest framing

This module is **available** as a drop-in replacement but is **not wired**
into the production `_generate_impl` decode path. The existing
`BondAwareDecoder` (which uses `BondOrderHead` + bond-pattern mask) is
the production decoder; switching to YuelBond mid-training would
require a fresh retrain to avoid distribution shift.

**Why ship the module now?**
1. The lit anchor (Wang 2025) makes the GNN decoder a documented
   alternative; future work can A/B test YuelBond vs BondOrderHead.
2. The implementation is CPU-friendly (≤5k params, K=3 layers) so it
   can be embedded into decode smokes without GPU cost.
3. The 5 unit tests verify the contract; regression guard for future
   refactors that might want to swap decoders.

## 6. Next step

A/B test integration in `_generate_impl` (gated by `--decoder=yuelbond`
CLI flag), with `--decoder=bondorder` (current default) preserved for
backward compat. This requires a GPU retrain; for now, the module
ships as a parallel path.
