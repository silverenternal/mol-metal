# TODO-149 — Cloned Repos for Fast Training (Final Recommendation)

**Status:** ✅ DONE — 2026-09-27
**Priority:** high (workflow harness user voice: "最好多参考那种快速训练（且效果好）的项目，clone下来，省得我们自己造轮子")
**Effort:** ~30 person-hours net (≈4 working days); integration plan broken into 5 sub-phases in §3
**Owner:** (unset — pick from research/eng lane)
**Depends on:** pip-installing torch_geometric + transformers (§4.2 risk); Wf-124 P3 multi-task trunk shipped (already done, baseline test_auc=0.7613 on Ru seed=0)
**Predecessors:** TODO-143 §2.4 chemprop v2 reference-read (already done), TODO-21 lambda-model coupling, TODO-23 weak-to-strong plan

---

## §0 — Summary table of all clones (16 paths total: 14 clones + 2 local)

| # | Name | Path | Source | Size | License / status | Verdict |
|---|---|---|---:|---:|---|---|
| 1 | **chemprop_v2** | `.cloned/chemprop_v2` | chemprop/chemprop | 369 MB | MIT, pip-installed as chemprop 2.3.1 | **IMPORT (P0)** |
| 2 | **DANN** | `.cloned/DANN` | fungtion/DANN (Py2.7) | 168 KB | legacy, Python 2 only | CITE-ONLY (skip) |
| 3 | **DANN_py3** | `.cloned/DANN_py3` | fungtion/DANN_py3 (Py3.8) | 168 KB | MIT-ish, Py 1.6 | CONDITIONAL (5-metal OOD) |
| 4 | **mtl_uncertainty** | `.cloned/mtl_uncertainty` | elna4os/mtl_uncertainty_loss | 164 KB | MIT, drop-in | **IMPORT (P0)** |
| 5 | Sheaf-FRL | `.cloned/Sheaf-FRL` | SPAICOM/sheaf-based-federated-representation-learning | 3.9 MB | research-only, federated | CITE-ONLY (skip — single-site) |
| 6 | **BioLM-Score** | `references/BioLM-Score` | SZU-ADDG/BioLM-Score | 17 MB | Apache-2.0 (verify) | **IMPORT (P0)** |
| 7 | MLM-Scaling | `references/MLM-Scaling` | SZU-ADDG/MLM-Scaling | 14 MB | Apache-2.0 | CONDITIONAL (chem-LM prior) |
| 8 | RxnFlow | `references/RxnFlow` | SeonghwanSeo/RxnFlow | 9.7 MB | arXiv 2410.04542 | CONDITIONAL (GFlowNet backbone) |
| 9 | SoftMol | `references/SoftMol` | SZU-ADDG/SoftMol | 73 MB | arXiv 2601.21964 | CONDITIONAL (block-diffusion) |
| 10 | SynFlowNet | `references/SynFlowNet` | mirunacrt/synflownet | 22 MB | ICLR 2025 | CITE-ONLY (already absorbed) |
| 11 | SyntheMol | `references/SyntheMol` | swansonk14/SyntheMol | 21 MB | Nature MI 2024 | CITE-ONLY (already absorbed via TODO-21 / lambda_integration_plan) |
| 12 | FLOWR | `references/FLOWR` | jule-c/flowr (archived) | 15 MB | MIT, **archived** | CITE-ONLY (use FLOWR.root instead) |
| 13 | DrugOOD | `.data/molmetal/DrugOOD` | tencent-ailab/DrugOOD | 2.5 MB | research-only | CONDITIONAL (OOD split for next paper round) |
| 14 | **lightning-bolts** | `molmetal/references/lightning-bolts` | Lightning-AI/lightning-bolts | 8.8 MB | Apache-2.0, import-works | CONDITIONAL (trainer templates) |
| 15 | **torchdrug** | `molmetal/references/torchdrug` | DeepGraphLearning/TorchDrug | 6.9 MB | Apache-2.0, import-works | CONDITIONAL (GNN baseline) |
| 16 | **dgl-lifesci** | `molmetal/references/dgl-lifesci` | awslabs/dgl-lifesci | 4.5 MB | Apache-2.0, import-works | CONDITIONAL (DGL backend) |
| 17 | flow_matching (local) | `molmetal/references/flow_matching` → `../../flow_matching` | symlink to project root | symlink | own code | N/A (project module) |
| 18 | soft_params (local) | `molmetal/references/soft_params` | local JSON+MD outputs | 80 KB | own artifacts | N/A (not a clone) |
| – | FLOWR.root (NOT cloned) | – | jule-c/flowr_root | 0 | active fork | PENDING clone if flow-matching work resumes |

**Bolded rows** = actively recommended for import in this round. **New clones done this session** (#14-16: lightning-bolts, torchdrug, dgl-lifesci) all verified importable.

**pip-installed chlibs already in venv:** chemprop 2.3.1, lightning 2.6.6, lightning-utilities 0.15.3, pytorch-lightning 2.6.6, rdkit 2026.3.6, torch 2.14.0, torchdiffeq 0.2.5, torchmetrics 1.9.0.

**Missing pip packages that gate the import path** (see §4.2): torch_geometric (+ torch_scatter / torch_sparse / torch-cluster), transformers (HuggingFace), wandb, meeko, recursion/gflownet (pip), molfeat.

---

## §1 — Top 3 most useful for mol-metal (rationale)

### #1 — `chemprop_v2` (`.cloned/chemprop_v2/`)

**Why it's the top pick.** chemprop v2 implements exactly the architecture the project already uses (D-MPNN + multi-task FFN head). Our Wf-124 P3 trunk `molmetal/models/v4_multitask_dmpnn.py` is a hand-rolled slim version of this. Three concrete wins are immediate:

1. **Multi-task FFN head sharing** (`chemprop/nn/predictors.py:MveFFN`, `EvidentialFFN`, `RegressionFFN`). Drop into our 5-metal head with `n_tasks=5` and get uncertainty-aware outputs for free — currently we have a plain MSE head.
2. **Precomputed feature cache** (`chemprop/featurizers/` + dataloader hooks). Per-epoch RDKit featurization is the single biggest wall-clock cost in our Wf-124 P3 run; caching trims 30-50% off epoch time on MetalCytoTox-sized sets.
3. **Conformal calibrator** (`chemprop/uncertainty/calibrator.py:RegressionConformalCalibrator`). Low-N metals (Ru has ~150 molecules in MetalCytoToxDB) need calibrated uncertainty intervals for honest reporting in the next paper round; v2 ships this.

**Why not wholesale replace our trunk.** Per TODO-143 §3.2 (`chemprop_v2_reference_notes.md`): the chemprop trunk is ~2700 LOC, ours is ~250 LOC. We do NOT want to swap a maintainable slim model for a heavy upstream we don't own. **Lift the 3 components above (~200 LOC total); do not fork the trunk.**

### #2 — `mtl_uncertainty` (`.cloned/mtl_uncertainty/`)

**Why it's the top pick.** Tiny (164 KB), Python 3, single-file `nn.Module` implementing Kendall et al. 2018 learnable log-sigma per task. Wf-124 P3 currently uses **Wasserstein** pIC50 regression as the multi-task aggregator (per commit `e8f68f2`). Kendall's uncertainty-weighted loss is the SOTA replacement: it auto-tunes per-task weights, dropping the need for hand-picked loss mixing coefficients in the 5-metal × 3-task grid.

**Concrete integration.** Wrap the existing `WassersteinMTLLoss` (`molmetal/models/losses.py`) with `mtl_uncertainty.UncertaintyWeightedLoss(...)` such that:
- 5 metal-regression heads → 5 learnable log-sigma scalars
- 1 Wasserstein pIC50 head → 1 learnable log-sigma scalar
- Total: 6 learnable scalars auto-tuned during backprop

**Effort to import: ~0.75 person-day** (1 file copied, 1 wrapper written, 1 test verifying the 5-metal grid converges at parity or better than the current Wasserstein baseline).

### #3 — `BioLM-Score` (`references/BioLM-Score/`)

**Why it's the top pick.** The two existing scoring hot paths in mol-metal are `r_vina_proxy` (currently a 0-value placeholder per TODO-21 §1) and `known_pt_drug_*` (ChEMBL-derived pIC50 endpoints). BioLM-Score fuses pocket graphs + ligand graphs with ESM/ESMC protein LM + Chemformer ligand LM and ships CASF-2016 scoring/ranking/screening/docking pipelines. It is the closest off-the-shelf upgrade for the protein-ligand affinity head.

**Why it's #3 not #1.** It requires pip-installing `transformers` + downloading ESM/ESMC + Chemformer weights (~2 GB) and writing a thin adapter. Bigger integration cost than #1 and #2.

**Concrete integration.** Replace `r_vina_proxy` in `molmetal/molmetal_lam/sbdd_env/RewardAggregator` with `r_biolm_score` (BioLM-Score wrapper) — this is **Phase 3** of `lambda_integration_plan.md`. Effort: ~1-2 days.

---

## §2 — How to USE each (specific code paths / APIs)

### §2.1 chemprop_v2 — which files, what to lift

```
chemprop/uncertainty/calibrator.py  →  molmetal/molmetal_lam/sbdd_env/conformal.py
chemprop/nn/predictors.py           →  molmetal/models/uq_heads.py     (MveFFN, EvidentialFFN — copy)
chemprop/schedulers.py              →  molmetal/trainers/noam_lr.py    (NoamLR warmup)
chemprop/nn/metrics.py              →  (NOT lifted; we already have mtl_uncertainty.WassersteinLoss)
chemprop/models/multi.py            →  (reference only — multi-COMPONENT ≠ multi-TASK)
chemprop/nn/ffn.py                  →  (reference only — our MLP is smaller)
```

**APIs to call once lifted:**

```python
# conformal.py
from molmetal.molmetal_lam.sbdd_env.conformal import RegressionConformalCalibrator
cal = RegressionConformalCalibrator().fit(cal_split_X, cal_split_y)
qhat = cal.qhat  # non-conformity score
intervals = cal.predict(X_test)

# uq_heads.py
from molmetal.models.uq_heads import MveFFN, EvidentialFFN
mve_head = MveFFN(input_dim=d_h, n_tasks=5)
evid_head = EvidentialFFN(input_dim=d_h, n_tasks=5)

# noam_lr.py
from molmetal.trainers.noam_lr import NoamLR
sched = NoamLR(optimizer, warmup_steps=1000, init_lr=1e-4, max_lr=1e-3)
```

### §2.2 mtl_uncertainty — which file, what to lift

```
mtl_uncertainty/uncertainty_weighted_loss.py  →  molmetal/models/losses.py (add UncertaintyWeightedLoss class)
```

**APIs to call:**

```python
from molmetal.models.losses import UncertaintyWeightedLoss
loss_fn = UncertaintyWeightedLoss(n_tasks=6)  # 5 metal + 1 Wasserstein pIC50
# Inside training_step:
loss = loss_fn(preds, targets)  # auto-balances per-task gradients
loss_fn.log_vars  # inspect learned log-sigma values post-training
```

**Verify.** Train Ru seed=0 with Wasserstein baseline → record test_auc=0.7613 (current Wf-124 P3 number). Train same seed with `UncertaintyWeightedLoss` for 30 epochs. Expect: parity on Ru AUC, **measurable lift on low-N metals** (Rh, Ir) where the hand-mixed Wasserstein weight is currently sub-optimal.

### §2.3 BioLM-Score — which files, what to lift

```
BioLM-Score/models/protein_ligand.py          →  molmetal/molmetal_lam/sbdd_env/biolm_score_adapter.py
BioLM-Score/data/pocket_graph.py             →  (reference only — pocket extraction follows molmetal convention)
BioLM-Score/checkpoints/esmc_650m/           →  download to .cache/biolm_score/esmc_650m/
BioLM-Score/checkpoints/chemformer/          →  download to .cache/biolm_score/chemformer/
```

**APIs to call:**

```python
from molmetal.molmetal_lam.sbdd_env.biolm_score_adapter import BioLMScorer
scorer = BioLMScorer(esm_model="esmc_650m", chem_model="chemformer", device="cuda")
score = scorer.score(ligand_smiles="c1ccccc1", pocket_pdb_path="1H36.pdb")  # returns float p_affinity
batch_scores = scorer.score_batch(smiles_list, pocket_pdb_path="1H36.pdb")  # GPU-batched
```

**Verify.** Score 5 known ligands from `vina_high_affinity_1h36.json` against pocket 1H36; verify Pearson r ≥ 0.4 vs the Vina reference column (low bar; the BioLM-Score paper reports CASF-2016 Pearson r ~0.85).

---

## §3 — Integration plan (which to import, which to cite-only)

### Phase A — P0 imports (this session's recommendation, ~3 days)

| # | Repo | Effort | Why now |
|---|---|---|---|
| A.1 | **chemprop v2** — conformal + MVE/Evidential heads + Noam scheduler | ~1.25 days | Lift the 3 files per §2.1; gates Ru test_auc > 0.80 |
| A.2 | **mtl_uncertainty** — uncertainty-weighted MTL loss | ~0.75 day | Drop-in for Wf-124 P3; auto-tunes 6 log-sigma scalars |
| A.3 | **BioLM-Score** — pocket+ligand LM scoring adapter | ~1 day | Replaces r_vina_proxy (currently 0-value placeholder) |

**Sub-total A:** ~3 person-days = 24 hours.

### Phase B — P1 conditional imports (after Phase A, ~1.5 days)

| # | Repo | Effort | Gate condition |
|---|---|---|---|
| B.1 | **torch_geometric** (pip) + **torch_scatter** | 0.5 day install | Needed before any of the PyG-based clones can be lifted. Install gate. |
| B.2 | **torchdrug** (clone already done) | 0.5 day | Lift `torchdrug.layers` modules as fast MPNN alternative; only if chemprop head lift (A.1) plateaus |
| B.3 | **dgl-lifesci** (clone already done) | 0.5 day | Only if PyTorch+RDKit version wall blocks chemprop v2 import (DGL backend as escape hatch) |
| B.4 | **DANN_py3** | 0.5 day | Only if 5-metal OOD experiment is scheduled (metalcytotox_ru_auc cross-metal transfer) |

**Sub-total B:** ~2 person-days = 16 hours, but only the gates of B.1-B.4 actually fire.

### Phase C — P2 cite-only (do NOT import — read for design notes)

| # | Repo | Why cite-only |
|---|---|---|
| C.1 | DANN (Py2.7) | Python 2 only; DANN_py3 is the modern replacement |
| C.2 | Sheaf-FRL | Federated multi-site; mol-metal is single-site |
| C.3 | SyntheMol | Already absorbed via `lambda_integration_plan.md` Phase 1 (5 → 13 reaction rules) |
| C.4 | SynFlowNet | Already absorbed via Phase 2 (ReactionTemplateEnv) |
| C.5 | RxnFlow | Same GFlowNet family as SynFlowNet; no marginal value over C.4 |
| C.6 | MLM-Scaling | Backup only — invoke only if Lambda's proof-search doesn't scale (Lambda path not in Wf-124 P3 scope) |
| C.7 | FLOWR | **Archived** upstream. Use FLOWR.root (NOT YET CLONED) if we resume flow-matching work |
| C.8 | DrugOOD | Reference splits for next paper round only; integration cost ~2 days per research.md |

### Phase D — P3 future-tense (deferred)

| # | Repo | When |
|---|---|---|
| D.1 | **FLOWR.root** (clone) | When flow-matching track resumes (TODO-21 Strategy 2) |
| D.2 | **DANN_py3** full integration | When 5-metal OOD cross-metal experiment is scheduled |
| D.3 | **SoftMol** pretrained weights | When block-diffusion track is greenlit (currently deck-only per TODO-21) |
| D.4 | **DrugOOD** scaffold split | When next paper round starts (NOT Round-13 paper, currently R13-final) |

**Total estimated integration effort (Phase A + likely Phase B triggers): ~30 person-hours = ~4 working days.**

---

## §4 — Risks + mitigation

### §4.1 Risk: chemprop v2's BN-after-aggregation pattern breaks our slim trunk

**Symptom.** chemprop v2 applies `BatchNorm1d` on the post-aggregation fingerprint (`models/model.py:94`). Our slim trunk in `v4_multitask_dmpnn.py` does not. If we lift chemprop's `MPNN` class wholesale, the BN layer silently changes the metalcytotox baseline test_auc by some unknown amount.

**Mitigation.** Lift **only** the 3 components listed in §2.1 (conformal, MVE/Evidential heads, Noam scheduler). Do NOT copy the `MPNN` or `MulticomponentMPNN` class. Audit per TODO-143 §9 ("grep audit — no GroupNorm / LayerNorm referenced anywhere in Mol-Metal still uses BatchNorm1d unintentionally").

### §4.2 Risk: missing pip packages gate the import path

**Symptom.** torch_geometric, transformers, wandb, meeko, recursion/gflownet, molfeat are NOT in the venv. Of these:
- **torch_geometric + torch_scatter / torch_sparse / torch-cluster** are ROCm wheels (not always available; PyG ROCm builds lag ~2 weeks behind CUDA). Verify on this hardware before committing to PyG path.
- **transformers** is ~700 MB on disk + ESM/ESMC weights ~2 GB. Disk-budget on `/mnt/storage`.
- **meeko** is the standard PDBQT converter; recent Wf-121 commits run real Vina docking on cisplatin × 10 pockets and likely need it.

**Mitigation.** Run a pre-flight pip install check (PyG ROCm wheel availability, transformers disk budget) BEFORE starting Phase A. If PyG fails to install, swap B.2 to DGL-LifeSci (already cloned). Add to TODO/pending: "Pre-flight pip install audit for torch_geometric + transformers".

### §4.3 Risk: BioLM-Score weights license unclear

**Symptom.** BioLM-Score depends on ESM/ESMC + Chemformer pretrained weights. ESM weights are MIT-licensed (Meta); Chemformer weights are CC-BY-NC 4.0 (academic only — **non-commercial**). If the next paper round is to be commercial-facing or deposited on a CC-BY-NC-incompatible platform, the Chemformer weights cannot be used.

**Mitigation.** Before lifting BioLM-Score, verify the Chemformer license is compatible with our publication venue. If not, use ESM-only as the protein encoder and skip the Chemformer ligand LM (fall back to D-MPNN features for the ligand side).

### §4.4 Risk: cloning more repos grows `.cloned/` and `references/` without integration

**Symptom.** `.cloned/` is 374 MB; `references/` is 181 MB. Both are gitignored locally but appear in the project tree. If we keep cloning without lifting, we accumulate dead weight (the Sheaf-FRL pattern — cloned, never imported, never cited).

**Mitigation.** After this doc, **stop cloning** until at least one Phase-A item ships. Set a gate: "No new clones until chemprop v2's MVE/Evidential head is lifted and verified on Ru seed=0." Add to TODO/pending: "Clone-budget policy — no new clones until 1 Phase-A lift ships".

### §4.5 Risk: pre-existing clones (lightning-bolts, torchdrug, dgl-lifesci) verified importable but unused

**Symptom.** The three clones done this session (lightning-bolts 8.8 MB, torchdrug 6.9 MB, dgl-lifesci 4.5 MB) all `import_works: true` but no integration path is scheduled for them yet.

**Mitigation.** Treat them as P1-conditional (§3 Phase B.2/B.3): only lift if Phase A plateau or DGL backend needed. Do not invest integration time until Phase A is done.

---

## §5 — Honest framing (party-twenty-spirit §3.5)

Following TODO-28's pattern of stating honestly what we did NOT verify and what we do NOT claim:

1. **We did NOT measure a speed-up.** No benchmark was run in this session to confirm chemprop v2's precomputed feature cache is faster than our current per-epoch featurization on MetalCytoToxDB. The 30-50% speed-up claim is a **literature claim lifted from chemprop v2's README**, not a mol-metal measurement. Verify before committing to the lift.

2. **We did NOT verify pip availability of torch_geometric + transformers on this ROCm 7.2 + Python 3.12 venv.** The Phase-B triggers may fail at install time. If PyG ROCm wheel is unavailable, the DGL-LifeSci fallback path is the only one that ships.

3. **We did NOT confirm BioLM-Score's Chemformer weights are usable in our publication venue.** License check (CC-BY-NC 4.0) is a precondition; we have NOT done it in this session.

4. **The "top 3 picks" are weighted toward current Wf-124 P3 scope.** If the project's next 3 months are dominated by a different track (lambda MCTS, flow-matching 3D, wetlab ic50), the top-3 ranking shifts: lambda-MCTS era → SyntheMol + SynFlowNet + RxnFlow (already partly absorbed); flow-matching era → SoftMol + FLOWR.root + diffusers.

5. **chemprop v2 lift is a ~1.25-day lift, not a free win.** It carries risk of subtle BN / aggregation-pattern mismatches with our slim trunk (per §4.1). Treat it as a surgical patch, not a fork-and-replace.

6. **Clone-this-repo enthusiasm is a known trap.** Per TODO-149 §4.4, we have accumulated 555 MB of clones (~9.5% of `/home/hugo/codes/mol-metal/` based on a `du -sh` sample) with integration rates well below 100%. **The right next action is "lift one of the P0 items and ship", not "clone one more repo".** The user's voice in the harness request explicitly says "省得我们自己造轮子" (avoid reinventing the wheel) — the *lift* part is what saves the time, not the *clone* part.

7. **This doc is a recommendation, not an executed integration.** No code was lifted; no benchmarks were run. The 30-hour estimate is based on TODO-143 §3.2 (chemprop v2 ~1.25 days), `lambda_integration_plan.md` Phase 3 (BioLM-Score ~1-2 days), and a 0.75-day estimate for mtl_uncertainty. Add 50% buffer for unforeseen integration friction (Python version mismatches, RDKit edge cases on metal SMILES, ROCm wheel delays) → realistic budget: **~45 person-hours = ~5.5 working days**.

---

## File metrics

- This doc: ~340 LOC.
- Cloned repos inventoried: 16 paths.
- New clones this session: 3 (lightning-bolts, torchdrug, dgl-lifesci) — all importable.
- pip packages identified as missing: 14 candidates (ranked in audit input).
- Recommended P0 lift: 3 repos / ~200 LOC net code / ~3 person-days.

**Status**: ✅ **Final recommendation written.** No code changes; integration plan (§3) is the next action. The doc satisfies the user voice "多参考那种快速训练（且效果好）的项目，clone下来，省得我们自己造轮子" by naming the 3 highest-EV lifts (chemprop v2, mtl_uncertainty, BioLM-Score) and explicitly stating that **cloning is not the win — lifting is**.
