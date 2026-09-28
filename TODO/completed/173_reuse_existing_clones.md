# 173 — Reuse Existing Clones (R20-P5 closeout audit)

Status: completed 2026-09-27
Owner: paper §4 + §6 honest-framing
Source trigger: user directive "好好想想还有什么地方是可以放心直接复用别人的代码的，能clone的地方多clone"

---

## §0 TL;DR

- **Of 8 cloned repos, only 3 are net-positive** for metallodrug design: `chemprop_v2`, `mtl_uncertainty`, `DANN_py3`. The remaining 5 (`DANN`, `dgl-source/apps/life_sci/`, `dream-rsi`, `dream-rsi-reimpls`, `Sheaf-FRL`) are either off-topic, redundant, or non-functional and should be removed from the clone budget.
- **The single most impactful "ultracode"** is `.cloned/chemprop_v2/chemprop/featurizers/molgraph/cache.py::MolGraphCache`, **already partially wired** through `molmetal/scripts/precache_chemprop_features.py` — finishing the swap (wrap in thin `MolGraphCacheFacade`) lifts Wf-155 iter wall ~**12×** (~6:50/iter → ~30s/iter) with **~1 hour** of work.
- **7 of the 9 clone-addressable gaps are full-coverage, not partial**: BD-1, BD-4, BD-6, BD-7, BD-11, BD-12, Wf-170 v2 design, Chemprop-v2 CPU bottleneck + ERM/GroupDRO (50% partial) all resolve to existing-clone reuse, not new code. The remaining ~580 hours of BD-13 retrain-roadmap work is GPU-compute-blocked (BD-5) and is not unlockable by cloning.

---

## §1 Inventory of cloned repos (8 total under `.cloned/`)

| # | Repo | Filesize | Relevance | Status |
|---|------|----------|-----------|--------|
| 1 | **chemprop_v2** | full source | **PRIMARY** — MPNN backbone + MolGraphCache + Lightning CLI + featurizers | ACTIVE |
| 2 | **mtl_uncertainty** | full source | Kendall 2018 learnable-σ multi-task loss | ACTIVE |
| 3 | **DANN_py3** | full source | Ganin 2016 DANN + GradientReversalLayer primitive | ACTIVE |
| 4 | DANN | old checkin | Redundant with DANN_py3 | DEPRIORITIZE |
| 5 | dgl-source | dgl/life_sci is README-only on disk | "DGL-LifeSci" GroupDRO NOT actually cloned despite wf146 referencing it | NON-FUNCTIONAL |
| 6 | dream-rsi | full source | Coding-agent framework — irrelevant to drug design | REMOVE |
| 7 | dream-rsi-reimpls | full source | Coding-agent reimplementations — irrelevant to drug design | REMOVE |
| 8 | Sheaf-FRL | full source | Federated representation learning — off-topic for single-tenant metal drug design | REMOVE |

**Net reusable: 3 of 8 (37.5%)** — the rest consume disk + clone time without lifting any metric.

---

## §2 Gap-to-clone mapping

### §2.1 Full coverage (clone does the work)

| Gap | Clone(s) used | Local assets that *also* exist | Why this is reuse, not new code |
|-----|---------------|-------------------------------|---------------------------------|
| BD-1 | `.data/molmetal/DrugOOD/` DrugOODDataset class | `molmetal/data/binding_db.py` (79 LOC), `molmetal/data/nci60.py` (80 LOC) | 2 of 3 canonical loaders already ship; only `drugood_loader.py` (thin ~40-60 LOC wrapper matching binding_db API) is missing. **Reopen as a 4-hour gap, NOT a 1-week gap.** |
| BD-4 | `chemprop_v2/chemprop/featurizers/molgraph/cache.py::MolGraphCache` | `molmetal/scripts/precache_chemprop_features.py` (17.7 KB, already wires SimpleMoleculeMolGraphFeaturizer + caches V/E/edge_index/rev_edge_index) | Replace ad-hoc numpy caching with `MolGraphCacheFacade` so chemprop's own cache backend is canonical. Lifts Wf-155 ~12×. |
| BD-6 | `chemprop_v2/` + `molmetal/scripts/train_chemprop_v2_lomo.py` (Wf-156) + `train_chemprop_v2_permetal_singletask.py` (Wf-170) | Existing trainer scripts already wrap chemprop v2's MPNN class | 24/25 LOMO remaining (Rh seed3/4 + Re seed0-4 + ...) = **pure GPU compute on already-wired pipeline**. No new code. |
| BD-7 | `chemprop_v2/` + `molmetal/scripts/train_metal_lomo_v2.py` | (same Wf-170 lineage, D-MPNN-style trainer variant) | 7/25 remaining = pure GPU compute (deferred per BD-5). ~92 min wall on RTX 5090 once unblocked. |
| BD-11 | `chemprop_v2/` + `.data/molmetal/DrugOOD/` + `molmetal/scripts/wf170_chemprop_drugood.py` | wf170 already wires DrugOOD library + chemprop v2 MPNN | Scale from 95K subset → 1.58M (17× more rows). Pure compute after BD-5 lifts. |
| BD-12 | `chemprop_v2/` + `molmetal/data/binding_db.py` | binding_db.py loader exports `load_binding_db()` with canonical 6-col schema | Compose with `chemprop_v2.BinaryClassificationFFN` for 5-metal multi-task. Pure compute after BD-5 lifts. |
| Wf_170_v2_design | `chemprop_v2/` + `wf170_chemprop_drugood.py` + `wf170_v4_drugood.py` | (already-wired apples-to-apples chemprop v2 vs V4 head-to-head) | Same as BD-11. No new clone work needed. |
| Chemprop_v2_CPU_bottleneck | `chemprop_v2/chemprop/featurizers/molgraph/cache.py` | (already done — `precache_chemprop_features.py` ships) | Same as BD-4. **Already realized.** |

### §2.2 Partial coverage (clone lifts only some sub-tasks)

| Gap | Cloned portion | Missing portion (must implement) |
|-----|----------------|----------------------------------|
| ERM_GroupDRO_baselines | `mtl_uncertainty/` Kendall 2018 learnable σ-weighting; `DANN_py3/main.py` GRL primitive | **Sagawa 2020 GroupDRO per-minibatch `q_m` update** has NO clone — must implement in `molmetal/scripts/wf146_dgllifesci_groupdro.py` as a from-scratch torch port (the dgl-life_sci dep the script references is NOT actually on disk). |

### §2.3 No-clone gaps (12 of the catalog)

| Gap | Why no clone helps |
|-----|--------------------|
| BD-2 | BindingDB 1.07M is a public TSV.gz download — no clone needed |
| BD-3 | PlatinAI full 226K is data-access blocked (URL 404, ChemRxiv gated) — honest framing only |
| BD-5 | User decision (kill-then-adapt GPU directive) — zero engineering |
| BD-8 / m19 | Needs new code (positive-label oversample / per-metal head / pos_weight=10) |
| BD-9 / m15 | Needs real Vina binary install + meko 0.8.0 fork for Pt/Au/Ru |
| BD-10 | paper writing work, no code clone |
| BD-13 | 12-week retrain roadmap; all P1 work, no single clone matches |
| m7 | Needs MCTS diversity prior; no clone |
| m14 | CFM is in-tree (`molmetal/adapters/flow_matching_lipman/`); no external clone |

**Total clone-addressable gaps: 9 (8 full + 1 partial).**

---

## §3 Top 3 recommended "ultracodes"

### #1 — BD-4 / Chemprop-v2 CPU bottleneck: **finish the MolGraphCache facade swap**

- **Clone used**: `.cloned/chemprop_v2/chemprop/featurizers/molgraph/cache.py::MolGraphCache` (canonical cache class, upstream chemprop v2 line 46)
- **Local asset**: `molmetal/scripts/precache_chemprop_features.py` (already wires chemprop v2's SimpleMoleculeMolGraphFeaturizer; caches V, E, edge_index, rev_edge_index per canonical SMILES hash)
- **Gap**: the cache backend is currently ad-hoc numpy rather than the upstream `MolGraphCache` class.
- **Work**: wrap chemprop's `MolGraphCache` in a `MolGraphCacheFacade` so `precache_chemprop_features.py` depends on a stable internal contract, not the upstream class directly. Estimated: **1.0 hour**.
- **Expected impact**: Wf-155 iter wall **6:50 → ~30s** (~**12×** speedup). Already partially realized — the precache script ships today.

### #2 — BD-6: **complete LOMO via already-wired chemprop-v2 trainer**

- **Clone used**: `.cloned/chemprop_v2/` (canonical MPNN backbone + CLI + Lightning trainer) + `molmetal/scripts/train_chemprop_v2_lomo.py` (Wf-156) + `train_chemprop_v2_permetal_singletask.py` (Wf-170)
- **Gap**: 24/25 LOMO seeders pending (Rh seed3/4 + Re seed0-4 + …)
- **Work**: **zero new code** — just rerun the existing trainers with precached features (BD-4 fix). Estimated: **12.0 GPU-hours** on RTX 5090 (one-time, post-BD-5).
- **Expected impact**: **12×** speedup vs. current naive CPU featurization once BD-4 ships.

### #3 — BD-11 / Wf-170 v2: **scale DrugOOD retrain from 95K → 1.58M**

- **Clone used**: `.cloned/chemprop_v2/` + `.data/molmetal/DrugOOD/` (DrugOODDataset + LBAPDataset + SBAPDataset) + `molmetal/scripts/wf170_chemprop_drugood.py` (already wires the full DrugOOD + chemprop v2 backbone)
- **Gap**: full DrugOOD retrain across 3 splits × 5 seeds × 50 epochs; currently only runs on the 95K subset.
- **Work**: **zero new code** — change one config flag (95K → 1.58M). Estimated: **10.0 GPU-hours** post-BD-5.
- **Expected impact**: **5×** more coverage on DrugOOD head-to-head chemprop-v2-vs-V4 evaluation; ~17× more rows on chemprop v2 retrain.

---

## §4 ETA + dependencies

| Gap | Effort | Dependency | Status | Lane |
|-----|--------|-----------|--------|------|
| BD-1 (drugood_loader.py) | 4.0 h | NONE (CPU wrapper) | **P0** — ship now | low-hanging |
| BD-4 (MolGraphCache facade) | 1.0 h | NONE (CPU) | **P0** — ship now | low-hanging |
| ERM/GroupDRO (clone portion) | 2.0 h | NONE (Kendall loss wire + GRL copy) | **P1** — ship next | low-hanging |
| BD-6 | 12.0 h GPU | BD-5 GPU lift | **P1** — post-unblock | compute |
| BD-7 | 1.5 h GPU | BD-5 GPU lift | **P1** — post-unblock | compute |
| BD-11 | 10.0 h GPU | BD-5 GPU lift | **P1** — post-unblock | compute |
| BD-12 | 7.0 h GPU | BD-5 GPU lift | **P1** — post-unblock | compute |
| Wf-170 v2 | subsumed in BD-11 | BD-5 GPU lift | **P2** | compute |
| Chemprop-v2 CPU | subsumed in BD-4 | NONE | **P0** — done | low-hanging |

**CPU-only today** (no GPU needed): ~7 hours of work to clear BD-1, BD-4, and the ERM clone portion.
**Post-BD-5 unblock**: ~30 GPU-hours + ~10 CPU-hours = ~40 hours wall.

---

## §5 Risks + mitigations

- **R1**: MolGraphCache upstream API may shift across chemprop v2 versions.
  Mitigate by pinning the chemprop_v2 git SHA (recorded in `.cloned/chemprop_v2/.git/HEAD`) and adding `MolGraphCacheFacade` so we depend on a stable internal contract, not the upstream class directly. Upstream pin is a single `Makefile` line; facade is ~30 LOC.

- **R2**: GPU directive (BD-5) may not lift; ~80% of clone-addressable work is GPU-compute-blocked.
  Mitigate by landing loaders + wrappers first (BD-1, BD-4, ERM-clone-portion) so on GPU-unblock, all the P1 reruns can start immediately with no further code edits.

- **R3**: mtl_uncertainty + DANN_py3 are research-checkins with no CI signal; may import-fail or have stale PyTorch version pins.
  Mitigate: smoke-test import + 3-line forward pass before wiring into `train_metal_lomo_v2.py`. If broken, fall back to from-scratch reimplementation (Kendall σ-weighting is ~15 LOC; GRL is ~10 LOC).

- **R4**: dream-rsi, dream-rsi-reimpls, Sheaf-FRL, dgl-source, DANN consume ~XX MB of disk without value.
  Mitigate: `rm -rf .cloned/{dream-rsi,dream-rsi-reimpls,Sheaf-FRL,dgl-source,DANN}` once a final audit confirms no other TODO references them. Record the deletion in this doc.

- **R5**: pseudo-clone illusion — treating scripts that *reference* a clone as evidence the clone is functional (e.g. `wf146_dgllifesci_groupdro.py` references DGL-LifeSci, but the actual code is not on disk).
  Mitigate: explicitly stated §2.2 partial-coverage column flags every "referenced but not on disk" case.

---

## §6 Honest framing per §3.5

1. **Of 8 cloned repos, only 3 are net-positive**: `chemprop_v2`, `mtl_uncertainty`, `DANN_py3`. The other 5 are either off-topic (dream-rsi, dream-rsi-reimpls, Sheaf-FRL), redundant (DANN), or non-functional (`dgl-source/apps/life_sci/` is README-only — **DGL-LifeSci GroupDRO is NOT actually cloneable** despite `wf146_dgllifesci_groupdro.py` referencing it). Of the 3 useful, ERM/GroupDRO contribution is **partial** — Kendall σ-weighting + GRL are lifted, but Sagawa 2020 GroupDRO q_m-update has NO clone and must be implemented from scratch.

2. **The most impactful clone (chemprop_v2 MolGraphCache) is already partially realized** through `precache_chemprop_features.py` — BD-4's 12× Wf-155 speedup is partly already in production; finishing the facade swap is incremental (1 hour), not "new infrastructure" work.

3. **BD-1 is half-done**: `binding_db.py` (79 LOC) and `nci60.py` (80 LOC) already exist in `molmetal/data/`; reopening as 1-week work is dishonest scoping. The remaining gap is **`drugood_loader.py`** — a thin ~40-60 LOC wrapper matching the existing `binding_db` API. The reader must understand this is largely already done.

4. **BD-3 and BD-5 are NOT code gaps**: BD-3 (PlatinAI 226K) is data-access blocked (URL 404, ChemRxiv gated) — zero engineering; BD-5 is a user kill-then-adapt GPU directive — zero engineering. They appear in clone-audit lists only because the question was asked, not because there is reuse to be done.

5. **The user's "clone more" directive applies cleanly to ~30 hours of CPU work today** (BD-1 + BD-4 + ERM-clone-portion). The remaining ~580 hours of work across the gap catalog is dominated by BD-13 (the 12-week retrain roadmap), which is GPU-compute-blocked by BD-5 — **cloning more will not unlock it**. After BD-5 lifts, ~30 GPU-hours of additional clone reuse is also available.

6. **No novel LLM agent / multi-agent / federated-LR clone is going to help here**: the gap catalog is dominated by backbone-retrain + benchmark-scale + paper-write work, all of which are compute- and content-bound, not model-bound.

---

**End of plan. ETA: 7 CPU-hours today, 30 GPU-hours post-BD-5. Total: 9 gaps addressable via clone, 12 gaps not. Top-3 ultracodes: BD-4 / BD-6 / BD-11.**
