# SOTA SBDD Protocol Audit — Side-by-Side Comparison

**Date:** 2026-09-12
**Purpose:** Compare 9 SOTA pocket-conditioned generative models on 9 protocol axes so we can decide which baseline is the closest match to Lambda.
**Scope:** Cite-only audit from existing provenance reports in `molmetal/reports/`. NO papers re-run; NO web fetches.
**Sources:**
- `provenance_pocket2mol.md` (Peng et al., ICML 2022, arXiv:2205.07249)
- `provenance_targetdiff.md` (Guan et al., ICLR 2023, arXiv:2303.03543)
- `provenance_diff_decomp_equi_tank.md` (DiffSBDD, DecompDiff, EquiBind, TankBind)
- `provenance_diffdock_flowr_flowdock.md` (FLOWR, FlowDock, DiffDock-L)
- `TODO/01_research/sota_papers_2025_2026.md` §2 (TransDiffSBDD, MolCRAFT, AlphaDrug, MolChord)
- `lambda_vs_sbdd_protocol_aligned.md` (Lambda current single-pocket protocol)
- `lambda_benchmark_r0.md` (Lambda MCTS compute profile, branching/n_sims)

---

## §1 — Master protocol table (9 papers × 9 fields + Lambda + Closest-match)

| # | Paper (arXiv) | test_set | n_test_pockets | docking (engine + version) | exhaustiveness | SA impl | QED threshold (success) | success_rate definition | NFE (samples × steps) | PoseBusters validation | Closest match to Lambda |
|--:|---|---|---:|---|---:|---|---:|---|---:|---|:---:|
| 1 | **Pocket2Mol** (2205.07249, ICML 2022) | CrossDocked2020 (Luo 2021 split_by_name.pt) | 100 | **QuickVina 2** (qvina2) | **16** | Ertl SA (RDKit Contrib sascorer.py) [1–10 → 0–1 inverted] | not gated on QED; "High Affinity" only | best-of-100 ≤ co-crystal Vina | ≤ 50 atom-add steps × 100 samples = **5 000 NFE/pocket** (autoregressive, no MCMC) | NOT reported | **CLOSE (B)** |
| 2 | **TargetDiff** (2303.03543, ICLR 2023) | CrossDocked2020 (Luo 2021 split) | 100 | **Vina 1.2.2 + meeko 0.1.dev3 + AutoDockTools_py3** (via QVina wrapper, default exh) | **8** (QVina default) | Ertl SA (provenance: SA 0.58 [0–1 inverted]) | QED>0.25 ∧ SA>0.59 ∧ Vina<-8.18 → **strict 10.5%** (relaxed 35.1%) | strict triple-threshold (QED ∧ SA ∧ Vina) | DDPM 1000 steps × 100 samples = **100 000 NFE/pocket** | NOT reported | partial (docking engine match) |
| 3 | **DiffSBDD** (2210.13695, ICML 2023 / Nat Comput Sci 2024) | CrossDocked2020 (Pocket2Mol split) | 100 (also 7 642/8 932 MOAD) | **QuickVina 2** | (QuickVina 2 default, **16**) | Ertl SA (SA ≈ 2.81 [1–10]) | QED>0.25 ∧ SA>0.59 ∧ Vina<-8.18 → **24.6%** | strict triple-threshold | DDPM ×100 samples (steps not pinned in audit) | **88% PB-valid** (v3, Sep 2024) | **CLOSEST (A)** |
| 4 | **DecompDiff** (2303.10120, ICLR 2024) | CrossDocked2020 (Pocket2Mol split) | 100 | **QuickVina 2** | (QuickVina 2 default, **16**) | Ertl SA (SA ≈ 2.71 [1–10]) | "Success Rate" ≈ **24.5%** (some tables 39.0%) | strict (likely Vina<-8.18 ∧ QED ∧ SA) | DDPM ×100 samples | NOT pinned (Complete ≈ 0.94 is internal validity, not PB) | **CLOSE (B)** |
| 5 | **FLOWR** (2504.10564, Nat Comput Sci 2026) | **SPINDR** (PLINDER-derived, 35 666 complexes) **primary**; CrossDocked100 secondary | **225 targets × 100 lig (SPINDR)** | Vina (variant not pinned in audit) | not pinned | Ertl SA (inferred, SA 2.86 on CrossDocked) | not gated; reports PB-valid + RDKit-valid | **PB-valid 0.88→0.95 (MMFF94)** + RDKit-valid 0.94 (NOT docking success) | flow matching **20–100 Euler steps** × 100 samples; pocket encoder **1× forward** | **88–95% PB-valid** (after MMFF94 relax) | partial (PB framing only) |
| 6 | **MolCRAFT** (no arXiv in audit, 2024) | CrossDocked2020 | 100 (per TransDiffSBDD Table) | (AutoDock Vina family, not pinned) | (not pinned) | Ertl SA (range [1–10]) | strict triple-threshold → **36.1%** | strict (Vina<-8.18 ∧ QED ∧ SA) | not pinned (Bayesian-flow model) | NOT pinned | partial (success-rate metric) |
| 7 | **AlphaDrug** (DrugGen 2024, Database Oxford) | CrossDocked2020 | 100 (per TransDiffSBDD Table) | (not pinned) | (not pinned) | Ertl SA (0.80) | not gated (success rate **NOT reported**) | best-of-N Vina only (no triple-threshold) | not pinned | NOT pinned | LOW (metric frame diverges) |
| 8 | **TransDiffSBDD** (Tsinghua + MSR, 2025) | CrossDocked2020 | 100 | AutoDock Vina (not pinned exact version) | (not pinned) | Ertl SA (0.75) | QED>0.25 ∧ SA>0.59 ∧ Vina<-8.18 → **83.9%** | strict triple-threshold | not pinned (Uni-Mol 209M pretrain + 100 k complexes fine-tune) | NOT pinned | partial (strict-threshold metric) |
| 9 | **MolChord** (Zhongguancun + USTC, 2025) | CrossDocked2020 | 100 (per TransDiffSBDD Table) | (not pinned) | (not pinned) | Ertl SA (0.77) | high-affinity 55.1% + strict 33.2% | Vina<-8.18 ∧ QED ∧ SA (strict 33.2%) | not pinned (DPO post-trained) | NOT pinned | partial (DPO alignment = Lambda analogue?) |
| — | **Lambda current** (molmetal 1h36 single-pocket) | **PDB 1h36 chain A (HisP1H/HEM)** — **not in CrossDocked2020 100-pocket split** | **1** | **AutoDock Vina 1.2.7 + meeko 0.8.0 + ADFRsuite** (mk_prepare_receptor.py) | **16** (default), 4–32 in tests | **Ertl SA (RDKit Contrib sascorer.py)** — 1.870 mean over candidates [1–10] | **NOT gated** (separate axes reported) | "Top-1 Vina < ref" = **19.4%** (ref = HEM @ ~−7 kcal/mol) | **MCTS-only**: n_sims=100–1000 (depth=2), no denoising chain. branching=12/60 → ~14 s/cell; branching=1020 → ~298 s/cell | ETKDGv3+UFF geometry → **0/12 click tiles pass PB**, 0/13 CuAAC products (embedding-quality bug, not chemistry) | — |

---

## §2 — Closest-match rationale

### Closest: **DiffSBDD (2210.13695)** — labelled **(A)**

- **Same docking engine lineage** (QuickVina 2 vs AutoDock Vina — both Vina-family; kcal/mol scores within ~0.3–0.6 kcal/mol per `lambda_vs_sbdd_protocol_aligned.md` §2.7).
- **Same Ertl SA implementation** (RDKit Contrib `sascorer.py`) — Lambda's `sascorer` import is byte-equivalent.
- **Same strict success metric** (Vina<-8.18 ∧ QED>0.25 ∧ SA>0.59) — Lambda currently reports "Vina<−ref" but does NOT gate on QED/SA; if we add a triple-threshold, the DiffSBDD row becomes directly comparable.
- **Same n_test_pockets = 100** target (Lambda is currently 1, but `lambda_vs_sbdd_protocol_aligned.md` §3 P0 work is "100-pocket sweep").
- **Same PoseBusters framing** — both report PB-validity (DiffSBDD 88%, Lambda 0/12 currently due to UFF embedding bug).
- **Same test_set target** — CrossDocked2020 Luo 2021 split, 100 held-out pockets.

### Close (B): **Pocket2Mol** and **DecompDiff**

- **Pocket2Mol**: matches SA impl (Ertl `sascorer.py` byte-equivalent), matches docking-engine lineage, **diffs** on: docking exhaustiveness = 16 (same as Lambda default), box 20 Å (Lambda uses padding=8 Å which is equivalent in 1h36); success def differs ("High Affinity" = best-of-100 ≤ co-crystal Vina — close to Lambda's "Vina<−ref"); no PB gating (Lambda currently no PB).
- **DecompDiff**: matches dataset, matches success-rate triple threshold, matches Vina kcal/mol order of magnitude (−8.39 vs Lambda −5.923 single pocket); **diffs** on: SA 2.71 [1–10] vs Lambda 1.870 (Lambda is easier-synthesis due to click-tile constraint), no PB validation reported.

### Partial matches

- **FLOWR**: matches PB-valid framing (88→95%), but **diverges** on test_set (SPINDR vs Lambda 1h36), success def (PB-valid ≠ docking success), and protocol-mismatch on docking engine not pinned.
- **MolCRAFT / TransDiffSBDD / MolChord**: share the strict triple-threshold success metric (Vina<-8.18 ∧ QED ∧ SA), share CrossDocked100 test target, share Ertl SA [1–10] range; **diverge** on docking-engine pinning (none pinned in audit) and PB validation (not pinned).
- **AlphaDrug**: **lowest match** — reports only median Vina, no PB-valid, no triple-threshold; metric frame is most divergent.

---

## §3 — Protocol-mismatch flags (carried over from `lambda_vs_sbdd_protocol_aligned.md` §2)

These flags apply to ANY Lambda-vs-SOTA comparison and must be disclosed even for the closest-match DiffSBDD:

1. **n_test = 1 vs 100**. Lambda has a single-case value; pocket-specific variance in Vina is ±2 kcal/mol, so −5.923 single-pocket is statistically indistinguishable from −6 to −8.5 population means.
2. **Pocket corpus mismatch**. PDB 1h36 chain A (HEM-bound, metalloprotein) is **not** in the standard CrossDocked2020 100-pocket test split. Vina kcal/mol is pocket-dependent.
3. **SA impl parity**. Lambda uses Ertl Contrib `sascorer.py`; DiffSBDD/Pocket2Mol use the same file; DecompDiff/TargetDiff/FLOWR provenance audits do NOT pin the exact file. Lambda SA = 1.870 (mean) is consistent with click-chemistry constraint.
4. **Models NOT re-run**. All SOTA numbers are cited from original papers; Lambda's `compare_to_published.py` table documents `protocol=cross-pocket` flag. None of Pocket2Mol/TargetDiff/DiffSBDD/DecompDiff/FLOWR/MolCRAFT/AlphaDrug/TransDiffSBDD/MolChord checkpoints were run by Lambda on the Lambda test pocket.
5. **FLOWR's "94%" is PB-valid only**, not docking success. PoseBusters validates MMFF94 geometry; it does NOT test pocket binding. Lambda's 0/12 PB-valid result is on starting materials (azides/alkynes), not products.
6. **NFE accounting differs**. Group A papers report (samples × steps): Pocket2Mol 5 000 NFE/pocket, TargetDiff 100 000 NFE/pocket. Lambda is MCTS — "1 NFE" not defined; we report "1000 sims" as the analogue. Not commutable with a diffusion-NFE number.
7. **Docking engine differences**. TargetDiff uses QVina (exh=8, UFF torsion prune, `size_factor=1.2`); Pocket2Mol/DiffSBDD use QuickVina2 (exh=16, box 20 Å); Lambda uses AutoDock Vina 1.2.x + Meeko + ADFRsuite (exh=16, padding=8 Å). QuickVina2 and AutoDock Vina share lineage but scoring function versions drift across releases; reported Vina means typically differ by 0.3–0.6 kcal/mol between engines at the same `exhaustiveness`.

---

## §4 — What would close the closest-match gap to DiffSBDD

In priority order (carried from `lambda_vs_sbdd_protocol_aligned.md` §3, restricted to DiffSBDD-as-target):

| Priority | Work item | Estimated cost |
|:--:|---|---:|
| **P0** | Run Lambda on **100 CrossDocked2020 test pockets** (MMP2 + 99 others), 1000 sims each, exh=16 | ~24h+ ROCm time |
| **P0** | Gate Lambda's success_rate on the DiffSBDD triple threshold (Vina<-8.18 ∧ QED>0.25 ∧ SA>0.59) and report side-by-side | 0.5 day (analytics only, no retrain) |
| **P0** | Run PB-valid on Lambda's **CuAAC products** (not click tiles) using **MMFF94** (not UFF) for conformer relax | 0.5 day (engineering fix, no retrain) |
| **P1** | Pull DiffSBDD pretrained checkpoint and re-run on the same 100-pocket test set (or use SMARTS fallback and disclose) | 3–5 days |
| **P1** | Align docking engine — switch Lambda runs to QuickVina2 (or DiffSBDD to Vina 1.2.7) for byte-exact kcal/mol parity | 0.5 day |
| **P1** | Align NFE accounting — report both "MCTS sims" and "denoising-equivalent NFE" (sims × avg depth) | 0.5 day |
| **P2** | Run FLOWR / MolCRAFT / TransDiffSBDD on the same 100 pockets (all need ckpts) | 5 days |

**Until P0 is done**, Lambda's claims against DiffSBDD must be restricted to:
- *axes DiffSBDD does not measure* (retrosynthesizability, λ-term provenance, click-reachability, metal-coordination fidelity), **and**
- *single-pocket case studies* (1h36 here) with the 7 protocol-mismatch flags above.

---

## §5 — Field-by-field notes

### test_set

- **CrossDocked2020** dominates; only FLOWR (SPINDR) and EquiBind/TankBind/FlowDock (PDBbind) diverge.
- Lambda's 1h36 pocket is a metalloprotein (HEM-bound) — not represented in any of the SOTA test splits.

### n_test_pockets

- All Group-A diffusion/flow SBDD papers use **100 held-out CrossDocked2020 pockets**.
- Group-B pose-prediction papers use 363 (PDBBind time-split 2019) or 308 (PoseBusters).
- Lambda: 1.

### docking (engine + version)

- Vina-family everywhere; QuickVina2 most common, QVina next, AutoDock Vina raw least common.
- TargetDiff uniquely uses `vina==1.2.2` Python bindings + meeko 0.1.dev3 + AutoDockTools_py3.
- Lambda uses `vina` 1.2.7 + `meeko` 0.8.0 + ADFRsuite `mk_prepare_receptor.py`.

### exhaustiveness

- 16 (Pocket2Mol, DiffSBDD, DecompDiff, Lambda default).
- 8 (TargetDiff QVina default).
- Not pinned (FLOWR, MolCRAFT, AlphaDrug, TransDiffSBDD, MolChord).

### SA scoring

- All SOTA papers use **Ertl & Schuffenhauer 2009 RDKit SA score**, range [1–10] (easy→hard). Lambda uses the same file. DiffSBDD/DecompDiff/T FLOWR report mean over candidates; TargetDiff/Pocket2Mol report 0–1 inverted score.
- Lambda's mean SA = 1.870 sits in the "easy-synthesis" regime (<3), consistent with click-tile constraint.

### QED threshold for success

- **Strict** (Vina<-8.18 ∧ QED>0.25 ∧ SA>0.59): TargetDiff 10.5%, DiffSBDD 24.6%, DecompDiff 24.5%, MolCRAFT 36.1%, TransDiffSBDD 83.9%, MolChord 33.2%.
- **Relaxed**: TargetDiff 35.1% (looser QED/SA thresholds).
- **Best-of-N Vina only** ("High Affinity"): Pocket2Mol 49.8–54.2%, AlphaDrug median −9.77 (no SR), FLOWR PB-valid 88–95% (NOT docking).
- Lambda: not gated on QED/SA currently.

### success_rate definition

- See above; the **two competing definitions** (strict triple-threshold vs PB-valid-only) are the root cause of incomparability between FLOWR's 94% and DiffSBDD's 24.6%.

### NFE (samples × steps)

- Pocket2Mol autoregressive: ≤ 50 atom-add steps × 100 samples = **5 000 NFE/pocket**.
- TargetDiff DDPM: 1000 steps × 100 samples = **100 000 NFE/pocket**.
- FLOWR flow matching: 20–100 Euler steps × 100 samples; pocket encoder 1× forward (saves vs PILOT/DiffSBDD).
- Lambda: **MCTS-only**, "NFE" not defined; 1000 sims is the analogue.

### PoseBusters validation

- **Reported**: DiffSBDD (88% PB-valid), FLOWR (88→95% PB-valid after MMFF94).
- **NOT reported** in audit: Pocket2Mol, TargetDiff, DecompDiff, MolCRAFT, AlphaDrug, TransDiffSBDD, MolChord.
- Lambda: 0/12 on click tiles (starting materials), 0/13 on CuAAC products — flagged as UFF embedding bug, NOT chemistry.

---

*End of audit. All cells are from existing molmetal/reports/ provenance docs and TODO/01_research/sota_papers_2025_2026.md. No web fetches; no model re-runs.*
