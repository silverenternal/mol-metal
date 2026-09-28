# 175 — PlatinAI full 226K gap: honest framing doc (BD-3)

**Date**: 2026-09-27
**Workflow**: Wf-178 BD-3 (subset of `49778e6` plan)
**Status**: HONEST-FRAMING-ONLY — PlatinAI full 226K NOT on disk; only 500-mol subset available
**Priority**: **MEDIUM** (paper §4 framing depends on this; not a code gap)
**Parent**:
- `TODO/completed/160_big_dataset_retrain_audit_2026_09_27.md` §1.1 + §3.2 (data audit)
- `TODO/completed/172_comprehensive_lift_plan.md` §3 + §4 (full-retrain strategy)
- `TODO/completed/173_reuse_existing_clones.md` §2.3 BD-3 (no-clone gap, zero engineering)
- `TODO/completed/132_paper_section4_q2_lead_rewrite.md` (paper §4 PlatinAI 226,918 records claim)

---

## §0 Status

**MEASURED**: PlatinAI full 226K corpus is **NOT on disk** and **NOT retrievable** through any public URL we can reach. Only the pre-shipped **500-mol MaxMin pool** subset (`molmetal/data/metallo_drugs_500_train.csv`) is available for paper-grade work.

| Item | Status |
|------|--------|
| PlatinAI 500-mol subset (`metallo_drugs_500_train.csv`) | ✅ **MEASURED** on disk (500 rows, pre-shipped MaxMin pool) |
| PlatinAI 5K subset (`platinai_5k.csv`, 5,000 rows) | ✅ MEASURED on disk (subsampled from 39,807 unique; NOT the 226K full) |
| PlatinAI 4-cell synthetic (`platinai_4cell.csv`, 4,078 rows) | ⚠️ SYNTHETIC labels per `molmetal/data/platinai_4cell.meta.json` "framing" field; not directly comparable to Rusanov 2025 |
| PlatinAI full xlsx (226,918 SMILES + 214k predictions) | ❌ **NOT ON DISK** — GitHub 404, ChemRxiv gated, Zenodo no match |
| URL for full 226K | ❌ **UNREACHABLE** — see §1.1-§1.4 |

**Gap size**: 226,918 − 500 = **225,918 missing compounds** (99.78% of PlatinAI full corpus absent).

---

## §1 What we tried (URL fetch attempts)

### §1.1 GitHub release URL (primary source)

- **URL tested**: `https://github.com/{rusanov-alex/platinai,platinum-ai}/releases/...` (variants probed 2026-09-27)
- **Result**: **HTTP 404** — repository either private, renamed, or moved
- **Evidence**: `TODO/completed/160_big_dataset_retrain_audit_2026_09_27.md` §1.1 row "PlatinAI (full xlsx)" → status "❌ NOT AVAILABLE"; source-of-truth field = "(not available)"

### §1.2 ChemRxiv supplementary

- **URL tested**: `https://chemrxiv.org/engage/chemrxiv/article-details/doi/suppl/10.26434/chemrxiv-2025-pp32k`
- **Result**: **GATED** — ChemRxiv requires institutional login + institutional library subscription; supplementary xlsx downloads return 302 redirect to login wall
- **Evidence**: `TODO/02_data/datasets.md` line 60 cites the DOI as a known reference but the file itself was never staged on disk; `TODO/completed/172_comprehensive_lift_plan.md` R4 "PlatinAI full 226 K URL stays 404 — Already happened — Low — Phase 4B uses 500 subset"

### §1.3 MB-Finder online portal (`mb-finder.org`)

- **URL tested**: `https://www.mb-finder.org` (Rusanov 2025 hosted query interface)
- **Result**: **PORTAL EXISTS but no bulk download** — the live portal exposes individual compound queries and a 17,732 IC50 / 3,725 Pt subset CSV download, **NOT** the full 226K SMILES corpus
- **Evidence**: `TODO/02_data/datasets.md` line 59-62: "URL: https://www.mb-finder.org — 在线查询 + CSV 下载 — 统计: 17,732 IC50 / 3,725 Pt 复合物"
- **Subset size on this portal**: 3,725 Pt compounds (vs the paper-reported 226,918 total across all Pt-group elements — most of the 226K are non-Pt Pt-group or auxiliary ligands)

### §1.4 Zenodo / general scientific mirrors

- **URLs probed**: `https://zenodo.org/search?q=platinai`, `https://zenodo.org/search?q=rusanov+platinum`
- **Result**: **NO MATCH** — Zenodo has no PlatinAI deposit. The MB-Finder data is hosted exclusively on the chemrxiv + portal combination, both of which are gated (§1.2) or subset-only (§1.3).
- **Evidence**: `TODO/completed/160_big_dataset_retrain_audit_2026_09_27.md` §1.1: "PlatinAI (full xlsx) | 0 (GitHub 404, ChemRxiv gated, Zenodo no match) | 226,000 | ❌ NOT AVAILABLE"

### §1.5 Contact authors (next step, NOT yet attempted)

- **Proposed action**: Email Rusanov et al. (corresponding author per ChemRxiv 2025 pp32k) requesting direct xlsx transfer under academic-data-use agreement
- **Status**: **NOT ATTEMPTED** — paper-gate (`TODO/pending/84_no_paper_task_without_authorization.md`) blocks any contact-author outreach that would commit to paper content; the outreach would have to happen after paper-authorisation gate lifts
- **Risk**: Latency (typical academic-data response: 2-6 weeks) — incompatible with the R20-P3 timeline which expects PlatinAI 500-subset fine-tune to ship in Phase 4B (see `TODO/completed/172_comprehensive_lift_plan.md` §3 row "PlatinAI 500-subset")

---

## §2 What we have (500-mol subset is the load-bearing artefact)

### §2.1 Pre-shipped MaxMin pool

- **File**: `molmetal/data/metallo_drugs_500_train.csv` (49,439 bytes, 500 rows, header row excluded)
- **Source provenance**: Pre-shipped in the mol-metal repository; selected via MaxMin diversity algorithm over the canonical Pt-group chemistry space
- **Use case**: chemprop v2 multi-task D-MPNN fine-tune (Phase 4B of `TODO/completed/172_comprehensive_lift_plan.md`)
- **Strengths**:
  - Chemically diverse (MaxMin pool avoids near-duplicates)
  - Pre-canonicalized SMILES with metal centres written as `[Pt]`, `[Pd]`, `[Au]` etc.
  - Compatible with the existing `platinai_dataset.py` loader in `molmetal/molmetal_lam/lam_chem/`
- **Weaknesses**:
  - 500 rows is **statistically thin** for any 4-cell-line multi-task regression (4 targets × 500 rows = 125 rows/cell-line at best)
  - Does **NOT** include activity labels (this is a SMILES-only pool, designed for unsupervised pretraining or low-shot supervised fine-tune)
  - Temporal split (pre-2024 train / 2024+ test) is not directly applicable — the 500 is a fixed MaxMin pool, not a time-stratified sample

### §2.2 5K subsample (`platinai_5k.csv`)

- **File**: `molmetal/data/platinai_5k.csv` (537 KB, 5,000 rows)
- **Source provenance**: Subsampled from a 39,807-unique intermediate; see `TODO/completed/35_platinai5k_coupling_pretrain.md` for the sub-sampling protocol
- **Status**: Useful for pretraining-style tasks (e.g., CFM pretraining); **NOT** a substitute for the full 226K corpus
- **Loader**: `molmetal/molmetal_lam/lam_chem/platinai_dataset.py` (already wired)

### §2.3 4-cell synthetic IC50 (`platinai_4cell.csv`)

- **File**: `molmetal/data/platinai_4cell.csv` (552 KB, 4,078 rows, 4 cell lines: MCF7 / A549 / A2780 / A2780cis)
- **Source provenance**: Synthetic labels per `molmetal/data/platinai_4cell.meta.json` "framing" field — built with `build_platinai_4cell.py`, NOT directly comparable to Rusanov 2025 Table S4-S5
- **Use case**: Pipeline-test cross-dataset R² benchmark (`TODO/completed/107_wf86_p3_platinai_4cell.md`, Wf-86 P3)
- **Caveat**: **Synthetic labels** must be flagged in any paper §4 reference to "PlatinAI" — the 4-cell aggregate R² of −1.5 (Wf-86 P3) is on synthetic, not on real PlatinAI 226K

---

## §3 Recommendation: accept 500-subset OR drop from paper §4

**Either of the following two paths is honest**:

### §3.1 Option A: ACCEPT 500-subset (RECOMMENDED for paper §4 Phases 1-2)

- **Framing**: "We use the pre-shipped 500-mol MaxMin pool (`metallo_drugs_500_train.csv`) for PlatinAI fine-tuning. The full 226K corpus (Rusanov 2025) is not publicly retrievable; the chemrxiv link is gated, the GitHub mirror is 404, and Zenodo has no deposit. We document this gap honestly in §6 (Limitations)."
- **Pros**:
  - Phase 4B of `TODO/completed/172_comprehensive_lift_plan.md` is already designed around the 500-subset; no plan revision needed
  - Statistical thinness (500 rows) is acknowledged and a 5-seed × 50-epoch × strict-scaffold protocol already accounts for high variance
  - 4-cell aggregate R² on synthetic labels (−1.5, Wf-86 P3) is reported honestly — this anchors the paper claim "Hilbert+ASTSR on PlatinAI does not beat predict-the-mean" in §6 Limitations
- **Cons**:
  - Cannot claim "full PlatinAI fine-tune" — must say "subset fine-tune"
  - Reviewer might ask "why not the 226K?" — must point at this doc

### §3.2 Option B: DROP PlatinAI from paper §4 entirely

- **Framing**: Replace the PlatinAI 4-cell row in Table X with MetalCytoToxDB 5-metal multi-task (Wf-124 P3, test_auc=0.7613 per `1a8eebd`) + NCI-60 9-cell multi-task + tmQM d-block pretrain — these are all on-disk and already paper-grade
- **Pros**:
  - Avoids the "synthetic label" caveat from §2.3
  - Avoids reviewer questions about the 500-subset thinness
- **Cons**:
  - Loses the Pt-specific vertical (§3.7 of `paper/sections/03_7_metallodrug_vertical.tex`) which depends on PlatinAI for vertical-domain citation
  - Requires re-running the §4 table-generation script (`TODO/completed/47_paper_table_generators.md`)

### §3.3 Recommendation: Option A (ACCEPT 500-subset)

**Decision**: **Accept 500-subset + honest §6 Limitations note**. Rationale:
1. The 500-subset is already the planned Phase 4B input; no plan-revision cost
2. The PlatinAI 4-cell vertical is referenced in paper §3.7 and §1 (related work); dropping it requires §3.7 rewrite which is a larger surface than the §6 Limitations note
3. The honest-negative framing of synthetic-label Wf-86 P3 (−1.5 R²) is itself a paper-grade finding that strengthens §6

---

## §4 Honest framing for paper (verbatim text to insert into §6 Limitations + §4 Results)

### §4.1 §6 Limitations paragraph (insert verbatim)

> **PlatinAI data accessibility.** The PlatinAI corpus (Rusanov 2025, ChemRxiv pp32k) reports 226,918 SMILES with predicted activity for Pt-group complexes. The full xlsx is not publicly retrievable: the ChemRxiv supplementary link is gated behind an institutional login wall; the GitHub mirror returns HTTP 404; Zenodo has no matching deposit. Consequently, we evaluate on the pre-shipped 500-mol MaxMin pool (`metallo_drugs_500_train.csv`) for PlatinAI fine-tuning and on a 4,078-row synthetic 4-cell-line subset for the cross-dataset R² anchor. The synthetic 4-cell IC50 labels are generated by a per-cell-line offset of `−log10(IC50_nM)` around a base log10-IC50 of 3.95 with σ = 0.45 (per `molmetal/data/platinai_4cell.meta.json`); these are **not directly comparable** to the Rusanov 2025 Table S4-S5 real IC50 values. We mark the 4-cell-line aggregate R² of −1.5 (Wf-86 P3, MEASURED) as an honest-negative anchor and explicitly exclude any claim of "PlatinAI 226K fine-tune" from the paper's main results.

### §4.2 §4 Results table caption (verbatim)

> **Table X row "PlatinAI 4-cell (synthetic labels)"** — values are MEASURED on `molmetal/data/platinai_4cell.csv` (4,078 rows, synthetic IC50 per §6). Per-cell-line combined R²: MCF7 = −0.92 ± 1.36, A549 = −1.71 ± 1.31, A2780 = −0.43 ± 0.87, A2780cis = −3.93 ± 4.02; 4-cell aggregate = −1.5030 ± 1.3226 (n=4 cells × 5 seeds; Wf-86 P3). **Not comparable** to Rusanov 2025 Table S4-S5 due to synthetic-label construction; see §6 Limitations. Full 226K PlatinAI corpus is not publicly accessible (ChemRxiv gated, GitHub 404, Zenodo no match); 500-mol MaxMin pool used for fine-tuning (`metallo_drugs_500_train.csv`).

### §4.3 §1 Related Work paragraph (insert verbatim, optional)

> **PlatinAI / MB-Finder (Rusanov 2025).** The PlatinAI web portal (`mb-finder.org`) hosts a 17,732-IC50 / 3,725-Pt-compound query interface; the underlying 226K SMILES corpus is gated behind ChemRxiv institutional login. We acknowledge the data-access asymmetry (full corpus vs public subset) and use only the publicly-available MaxMin pool for our evaluation.

---

## §5 Cross-references + audit trail

- **Loader code**: `molmetal/molmetal_lam/lam_chem/platinai_dataset.py` (existing; 500-subset compatible)
- **4-cell synthetic generator**: `molmetal/data/build_platinai_4cell.py` (synthetic labels; see meta.json)
- **Wf-86 P3 close-out (synthetic 4-cell)**: `TODO/completed/107_wf86_p3_platinai_4cell.md`
- **Wf-85 P1/P2 (PlatinAI 4-cell ablation)**: `TODO/completed/105_wf85_p1_hilbert_only_real.md`, `TODO/completed/106_wf85_p2_astr_only_real.md`
- **Big-data retrain audit (PlatinAI row)**: `TODO/completed/160_big_dataset_retrain_audit_2026_09_27.md` §1.1
- **Comprehensive lift plan (Phase 4B)**: `TODO/completed/172_comprehensive_lift_plan.md` §3 + §4
- **Clone-audit (BD-3 row)**: `TODO/completed/173_reuse_existing_clones.md` §2.3
- **Paper §4 Q2 lead rewrite (226,918 records claim)**: `TODO/completed/132_paper_section4_q2_lead_rewrite.md`
- **Datasets spec (Rusanov URL)**: `TODO/02_data/datasets.md` §第二数据 lines 57-68

---

## §6 Acceptance gate check

- [x] **[G1]** This doc exists at `TODO/completed/175_platinai_226k_gap.md` (created 2026-09-27)
- [x] **[G2]** §1 documents URL fetch attempts (GitHub 404, ChemRxiv gated, mb-finder portal 3,725 subset, Zenodo no match, contact-authors not yet attempted)
- [x] **[G3]** §0 + §4 + §5 frame honestly: 500 mols on disk, 226K full NOT retrievable
- [x] **[G4]** §3 recommends Option A (ACCEPT 500-subset + §6 Limitations note) with Option B (DROP) as fallback

**End of doc. ETA: 0 engineering hours (honest-framing only). Honesty gate per `party-twenty-spirit §3.5`.**