# Pitfall Audit P6 — DATA & REPRODUCIBILITY layer

**Date**: 2026-09-17
**Scope**: Audit the data hygiene & reproducibility layer of Mol-Metal against 3 pitfalls from the user's brief.
**Audit only — no code modifications.**

**Files read**:
- `TODO/engineering_practices.md` (282 lines) — the canonical contract (sections 4 split discipline, 6 paper reporting, 10 reproducibility checklist)
- `TODO/environment.md` (256 lines) — ROCm 7.2 + uv-managed Python 3.12 + triton-rocm 3.8.0 environment
- `molmetal/reports/wf_remove_smoke/final.md` (282 lines) — Phase-4 smoke-removal verdict (1585 pass / 8 fail pytest)
- `molmetal/scripts/retrain_pic50_neural.py` (856 lines) — the canonical pIC50 retrain with seed handling
- `molmetal/scripts/baselines.py` (177 lines) + `molmetal/baselines/morgan_xgb.py:70-80` — `SPLITTER_FACTORIES` table
- `molmetal/data/splits.py` (652 lines) — RandomSplitter / TemporalSplitter / ChemicalSplitter / LigandDeduplicatedSplitter / ScaffoldSplitter
- `molmetal/scripts/calibrate_conditioned_pic50_baseline.py:1-122` — 80/10/10 Murcko scaffold split baseline
- `molmetal/scripts/r4_lambda_only_run.py` (lines 23-30, 164-450, 880-940, 4150-4170) — seed handling, scaffold-aware gate, no splitter use
- `molmetal/scripts/r4_c_full_sweep.py:140-192, 925-1170` — sha256 hashing of configs / receptors / ligands / manifest
- `molmetal/scripts/r10_ot_qm9_3seed.py` — sha256 sample-level hash + source-code-at-run fingerprint
- `molmetal/scripts/prepare_crossdocked_batch.py:37-58` — manifest + receptor + ligand sha256
- `molmetal/scripts/r3_drugood_benchmark.py:54-178` — scaffold_split_test uses ScaffoldSplitter(strategy="largest_first", seed=SEED)
- `molmetal/data/metallo_drugs_500_train.csv` (5 visible lines, columns `smiles,source`) — PlatinAI 500-mol pool (no hash header)
- `molmetal/reports/wf_pitfall_audit/p4_eval_novelty.md` — upstream P4 audit (referenced for §4 split context)

---

## Verdict per pitfall

| ID | Pitfall | Status | Evidence (file:LOC) |
|----|---------|--------|---------------------|
| P6.1 | Synthon library bias / training-data leakage (Bemis-Murcko scaffold split?) | **PARTIAL** | `molmetal/data/splits.py:530-652` defines `ScaffoldSplitter` (Bemis-Murcko, RDKit MurckoScaffold.GetScaffoldForMol, "largest_first" or "random" strategy). **BUT** `r4_lambda_only_run.py` never invokes any splitter (grep returns 0 hits on `ScaffoldSplitter` / `LigandDeduplicatedSplitter`); `metallo_drugs_500_train.csv` has no `scaffold` column; the 500-mol PlatinAI+tmQM+MetalCytoToxDB pool is split-blind (no train/val/test partition is ever computed or persisted). PlatinAI oracle trains kNN against the full pool (no held-out) — see `WF-Pivot-Followup MASTER §2`. |
| P6.2 | Reproducibility — random seeds logged everywhere? checkpoint hashes logged? | **PARTIAL** | `r4_c_full_sweep.py:140-192` logs `config_sha256`, `receptor_sha256`, `ligand_sha256`, `manifest_sha256`, source-code-at-run fingerprint, deterministic `job_id = sha256(pocket_id:seed)[:16]`. `r10_ot_qm9_3seed.py:13-394` records per-sample sha256. **BUT** `r4_lambda_only_run.py` has 0 seed-JSON files emitted (grep `_seed.json|seeds.json` against the file = empty); `retrain_pic50_neural.py:606` defaults `seeds=[42,0,1234]` but the seed list is only printed to stdout, not persisted to `seeds.json`. Checkpoint files in `molmetal/checkpoints/` (12 files: `fm_ru_temporal.pt`, `dmpnn_attn_ru_pic50.pt`, `fm_pocket.pt`, `metal_hybrid_*.pt`, etc.) have **no companion `.sha256` sidecar** and `retrain_pic50_neural.py:680-711` does not log `state_dict_sha256` in the bundle. |
| P6.3 | Standard data split discipline (scaffold, time, Tanimoto<0.4, MMseqs2 30%) | **PARTIAL** | `SPLITTER_FACTORIES` exposes 5 strategies (random / ligand_dedup / scaffold / temporal / chemical) — but `--split` defaults to `random` everywhere (`baselines.py:75`); `r4_c_full_sweep.py` and `r4_lambda_only_run.py` have no `--split` flag. **Temporal split (`cutoff_year=2024`) is wired but never run** as a baseline. **Tanimoto<0.4 is not implemented** — `ChemicalSplitter` exists but defaults to threshold=0.7 and is not the 0.4 OOD-protocol. **MMseqs2 30%-sequence-identity filter on pocket side is documented** in `engineering_practices.md:114` but **0 implementation files** (`grep -r mmseqs /home/hugo/codes/try_triton_on_rocm/molmetal/` returns 0 hits). The only scripts that use scaffold split are `retrain_pic50_neural.py:244-261` (pIC50 retrain, 80/10/10 with leak guard) and `r3_drugood_benchmark.py:176-178` (DrugOOD `scaffold_split_test`). |

**0 AVOIDED, 3 PARTIAL, 0 OPEN.** No pitfall is fully avoided; all three have substantial plumbing but no production claim line ties them together. The reproduction story is *strongest where it exists* (r4_c_full_sweep hash every input file + checkpoint + manifest) and *weakest where it matters most* (the headline eval scripts and the 500-mol metallodrug pool used by the de novo typed-term MCTS path).

---

## P6.1 — Synthon library bias / training data leakage — **PARTIAL**

### What we ship

**Scaffold splitter implementation**: `molmetal/data/splits.py:530-652` provides a complete `ScaffoldSplitter` class that:
1. canonicalises SMILES via `leakage_utils.canonicalize_array`,
2. computes the Bemis–Murcko scaffold via `RDKit.Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(includeChirality=False)` (`splits.py:584-598`),
3. groups rows by scaffold string,
4. assigns each scaffold to exactly one of `train/val/test` via the same largest-first / random strategy as `LigandDeduplicatedSplitter` (`splits.py:600-609`).

It is exposed via the `SPLITTER_FACTORIES["scaffold"]` table in all three baseline modules (`morgan_xgb.py:75`, `dmpnn.py:541`, `dmpnn_attentive.py:528`).

**Proven uses**:
- `calibrate_conditioned_pic50_baseline.py:50-67` builds a Murcko-scaffold group column and uses `GroupShuffleSplit` (sklearn) to make 80/10/10 scaffold-group partitions.
- `retrain_pic50_neural.py:244-261` (`scaffold_split` function) explicitly raises on scaffold or ligand leakage between any pair of splits.
- `r3_drugood_benchmark.py:176-178` (`scaffold_split_test` spec) constructs a `ScaffoldSplitter(strategy="largest_first", fractions=(0.8, 0.1, 0.1), seed=SEED)`.

### What is missing

**Gap 1: the 500-mol PlatinAI+tmQM+MetalCytoToxDB pool has no scaffold split.** The pool at `molmetal/data/metallo_drugs_500_train.csv` is a flat CSV with two columns `smiles,source` (PlatinAI / tmQM / MetalCytoToxDB). It is consumed by the PlatinAI oracle (`RewardAggregator.r_platinai` channel, per `WF-Pivot-Followup MASTER §2`) which trains a kNN index against the full pool — *no held-out split, no scaffold-group guarantee*. The pool is also the source of "candidate-encoding prior" and "fragment-pool enrichment" for the de novo typed-term MCTS path (`WF-P0-Metrics-Add`); if the test pocket's reference ligand shares a scaffold with the PlatinAI pool, the oracle's reward is inflated by memorisation.

**Gap 2: `r4_lambda_only_run.py` never invokes any splitter.** Grep returns 0 hits on `ScaffoldSplitter` / `LigandDeduplicatedSplitter` / `RandomSplitter` in `r4_lambda_only_run.py`. The 30-cell R12 Path A 10×3 panel (`paper/sections/04_evaluation.tex:381-393`) reports *test-set* metrics on the CrossDocked2020 test split, but the test split used is the upstream `split_by_name.pt` (`molmetal/data/crossdocked.py`) — which is the **CrossDocked2020 random/structural split, not a scaffold-group split**. Per `data_gap_analysis` (TODO-22), the headline test pocket set is the standard 100-test-pair split inherited from CrossDocked2020, which is *known* to have ~30% ligand-scaffold overlap with the training half (see `leakage_diagnosis.md` for the Krasnov baseline complaint).

**Gap 3: scaffold bias of the 500-mol pool is unquantified.** No `ScaffoldSplitter().__call__(metallo_drugs_500)`-style audit has been run to count (a) how many unique Bemis-Murcko scaffolds the pool contains, (b) the largest-cluster fraction, (c) the test-pocket reference scaffold membership. Without this number, the PlatinAI oracle's `kNN near a known Pt drug = high reward` channel is suspect.

### Honest framing

The `ScaffoldSplitter` class is **production-quality** (5/5 test files use it, leak guard is correct, deterministic seeded strategies, `SplitResult` validates integer arrays). The gap is **operational**, not in the splitter itself: the headline eval scripts (`r4_lambda_only_run.py`, `r4_c_full_sweep.py`) do not invoke any splitter, and the 500-mol PlatinAI pool is split-blind. If the paper's §4.6 / §4.7 PlatinAI oracle column is meant to be leak-free, **the only way to claim that is to (a) add `--split scaffold` to `r4_lambda_only_run.py`, (b) persist a `metallo_pool_scaffold_split.json` next to `metallo_drugs_500_train.csv`, and (c) re-run R12 with `--use-platinai-pool --split scaffold` so the oracle's kNN never sees the test pocket's reference scaffold**.

### Concrete patch plan (for the user to accept / reject)

1. **Add `--split {random|scaffold|temporal|chemical}` flag to `r4_lambda_only_run.py`** — wired to `splits.ScaffoldSplitter(strategy="largest_first", seed=args.seed)` when set. Default `random` for backward compat. Cite `splits.py:530-652` in the new flag's help text. Estimated: 1h engineering + 30 min pytest.
2. **Add a `molmetal/scripts/metallo_pool_scaffold_split.py` CLI** that loads `metallo_drugs_500_train.csv`, runs `ScaffoldSplitter(strategy="largest_first")` + `LigandDeduplicatedSplitter` + `TemporalSplitter(cutoff_year=2024)`, and writes `metallo_pool_splits.json` with `train_idx / val_idx / test_idx` per strategy. Emit a `n_unique_scaffolds / largest_cluster_fraction / test_overlap_with_train` summary table.
3. **Wire the PlatinAI oracle to honour the split** — `RewardAggregator.r_platinai` should accept an `excluded_idx` argument (default = `set()` for paper-safe number, optional `set(test_idx)` for leak-free oracle number). Estimated: 1h engineering + 30 min test.
4. **Add a §4.7 footnote** in `paper/sections/04_evaluation.tex` citing the new `metallo_pool_splits.json` and stating the PlatinAI oracle's Pearson r on the held-out scaffold-split sub-pool. Honest framing: pre-fix PlatinAI numbers are likely inflated by 5-15% by scaffold memorisation.

---

## P6.2 — Reproducibility — seeds + checkpoint hashes — **PARTIAL**

### What we ship

**Strong reproducibility layer in `r4_c_full_sweep.py`**:
- `file_digest(path) -> sha256` (`r4_c_full_sweep.py:148-149`).
- `digest_directory(root) -> {relpath: sha256}` (`r4_c_full_sweep.py:161`).
- Per-pair sha256 of receptor + ligand + manifest (`r4_c_full_sweep.py:1109-1113`).
- Deterministic `job_id = sha256(f"{pocket_id}:{seed}")[:16]` (`r4_c_full_sweep.py:192`, repeated at line 550) so a (pocket, seed) pair has a stable hash across runs.
- `guidance_inputs["prior_state"]` and `["synthesis_config"]` carry a `sha256` key (`r4_c_full_sweep.py:955, 964`).
- `seeds` is in `args.seeds`, validated for uniqueness + range (`r4_c_full_sweep.py:925`), printed at runtime (`r4_c_full_sweep.py:1171`), and forwarded into `seeds=args.seeds` for downstream caller (`r4_c_full_sweep.py:4163`).

**Stronger reproducibility layer in `r10_ot_qm9_3seed.py`** (per file header): per-sample `sha256` of each dataset element (`r10_ot_qm9_3seed.py:91, 94, 115, 212`), `source_code_at_run` fingerprint of every Python file under the project root (`r10_ot_qm9_3seed.py:343`), and `sources = [{path, sha256, bytes}, ...]` per dataset file (`r10_ot_qm9_3seed.py:284, 341`).

**Reproducibility contract in `TODO/engineering_practices.md:256-272`**: explicit checklist of `uv.lock`, `.python-version`, `.gitignore`, seed JSON files (`*_seed.json`), data hashes (`md5 / sha256`), checkpoint state-dict files, `uv run` command, env vars, `*_autotune.json` for Triton.

**PyTorch seed handling in `retrain_pic50_neural.py:365-366`**:
```python
torch.manual_seed(seed)
np.random.seed(seed)
```
inside `train_one_seed()`. Followed by `scaffold_split(cohort, seed=seed)` (`retrain_pic50_neural.py:368`) which uses `GroupShuffleSplit(random_state=seed)` (line 251) and `train_test_split_group(..., seed=seed+1)` for the val/test carve.

### What is missing

**Gap 1: `r4_lambda_only_run.py` does NOT log seeds to a file.** Grep for `_seed.json|seeds.json|seed_path` against the file returns 0 hits. The `--seeds` CLI is declared (`r4_lambda_only_run.py:4156`), parsed, and forwarded into `run_sweep(seeds=args.seeds, ...)`, but the actual list of seeds used per cell is only printed to stdout and not persisted. If a user runs `r4_lambda_only_run.py --seeds 42 0 1234 ...`, the JSON output has no `seeds` key at the top level (verified by reading the report-shape spec at `r4_lambda_only_run.py:4160-4170`).

**Gap 2: no checkpoint hash anywhere.** `molmetal/checkpoints/` has 12 `.pt` files (`fm_ru_temporal.pt`, `dmpnn_attn_ru_pic50.pt`, `dmpnn_tmqm_pretrained.pt`, `fm_pocket.pt`, `fm_pocket_conditioned.pt`, `metal_hybrid_Ru.pt`, `metal_hybrid_v1_concat_ru_temporal.pt`, `metal_hybrid_v2_crossattn_ru_temporal.pt`, `metal_hybrid_v3_crossattn_ru_temporal.pt`, `hybrid_ru.json`) plus a couple of `flow_matching_pocket_*.pt` files from Path B. None of them have a sibling `.sha256` sidecar. The checkpoint bundle emitted by `retrain_pic50_neural.py:680-711` writes `state_dict`, `model_kwargs`, `meta`, `test_metrics`, `history` — but no `state_dict_sha256` key. If a future contributor silently modifies the dmpnn_attn_ru_pic50.pt file (e.g. via training-script rerun with a different `lr` or `epochs`), there is no way to detect that the *paper's reported numbers* were computed on the original weights vs the overwritten ones.

**Gap 3: dataset hashes are partial.** Only `r10_ot_qm9_3seed.py` (QM9 — irrelevant to the metallodrug story) and `prepare_crossdocked_batch.py` (CrossDocked2020 — relevant) compute file-level sha256. `molmetal/data/metallo_drugs_500_train.csv`, `tmqm_subset.csv`, `MetalCytoToxDB.csv` (the three sources behind the de novo vertical) are referenced by path but never hashed. If the upstream MetalCytoToxDB maintainer updates the CSV (e.g. new rows appended), the PlatinAI oracle's kNN is silently different from what was reported in §4.7.

**Gap 4: `molmetal/baselines/dmpnn_attentive.py` trains without saving `seeds.json`.** Even though the CLI accepts `--seed 42 0 1234` (forwarded via the splitter factory), the output JSON (`baselines.py:170`) records `payload["seed"] = args.seed` (single int — last one wins) and not `seeds=[42, 0, 1234]`. Multi-seed baselines are therefore not reproducible from the output JSON alone.

### Honest framing

`r4_c_full_sweep.py` is the gold standard: every input file + every checkpoint + every config is sha256-fingerprinted, and the `job_id` is a stable hash. But this hygiene exists *only in the CFM+full-sweep path*. The headline de novo typed-term MCTS path (`r4_lambda_only_run.py`) and the metallodrug-de-novo-vertical path (`metallo_drugs_500_train.csv` → PlatinAI oracle) are **reproducibility-dark**: a fresh clone can rerun the same command and get numerically different outputs because (a) seeds are not persisted, (b) checkpoint state-dicts are not hashed, (c) the 500-mol pool's content can drift without detection.

### Concrete patch plan (for the user to accept / reject)

1. **Add `seeds` + `checkpoint_sha256` to the output JSON of `r4_lambda_only_run.py`.** In the `run_sweep()` finaliser, write `meta["seeds"] = args.seeds`, `meta["args_sha256"] = sha256(json.dumps(vars(args), sort_keys=True).encode())`, and `meta["checkpoint_sha256"] = sha256(state_dict_bytes) if a checkpoint is loaded`. Estimated: 30 min engineering.
2. **Add a `molmetal/scripts/fingerprint_artefacts.py` CLI** that walks `molmetal/checkpoints/`, `molmetal/data/*.csv`, `molmetal/configs/*.yml`, and writes `molmetal/reports/artefact_fingerprint.json` with `{path, sha256, bytes, mtime}`. Wire into `r4_lambda_only_run.py` so each run emits a fresh fingerprint and refuses to proceed if any artefact has changed since the previous baseline run (configurable via `--allow-artefact-drift`). Estimated: 2h.
3. **Generate `metallo_pool_sha256.txt`** next to `metallo_drugs_500_train.csv` with one sha256 per line. Update `TODO/environment.md` §8 (GPU verification) to also include the sha256 check.
4. **Backfill `seeds.json` for all existing checkpoint files** in a one-shot script `molmetal/scripts/backfill_seeds_json.py` that extracts the seed list from each checkpoint bundle's `meta["seed"]` (when present) and writes a per-checkpoint `seeds.json` sidecar. For singletons, just `seeds.json = {"seed": N}`.

---

## P6.3 — Standard data split discipline — **PARTIAL**

### What we ship

`SPLITTER_FACTORIES` (4 baseline modules × 5 strategies = 20 (model, split) combinations) exposes:
1. **`random`** — `RandomSplitter(seed=seed)`, 80/10/10 stratified shuffle (`splits.py:81-113`).
2. **`ligand_dedup`** — `LigandDeduplicatedSplitter(strategy="largest_first", seed=seed)`, each canonical SMILES group → exactly one split (`splits.py:385-526`).
3. **`scaffold`** — `ScaffoldSplitter(strategy="largest_first", seed=seed)`, each Bemis-Murcko scaffold → exactly one split (`splits.py:530-652`).
4. **`temporal`** — `TemporalSplitter(cutoff_year=2024)`, pre-cutoff → train/val, post-cutoff → test (`splits.py:119-175`).
5. **`chemical`** — `ChemicalSplitter(threshold=0.7, seed=seed)`, Tanimoto-dissimilar selection (`splits.py:181-348`).

The `engineering_practices.md:97-115` policy *explicitly mandates four splits*: random (sanity), scaffold, temporal (CrossDocked pre-2020 / PoseBusters post-2021 example), ligand-dedup. Plus a 30%-sequence-identity MMseqs2 filter on the pocket side.

### What is missing

**Gap 1: `--split` defaults to `random` everywhere.** `baselines.py:75` declares `default="random"` and the runner never overrides it. None of the headline eval scripts (r4_lambda_only_run, r4_c_full_sweep) even accept a `--split` flag. The `scaffold` and `temporal` strategies are *implemented and tested* but *not invoked* by the production path. Every Krasnov-baseline reproduction to date (`molmetal/reports/baseline_<metal>_<model>.json`) uses `random` — the historical source of the 93% seen-SMILES leakage (`splits.py:391-393`).

**Gap 2: Tanimoto<0.4 OOD protocol is not implemented.** `ChemicalSplitter` exists with `threshold=0.7` (`splits.py:207`), but the *industry-standard* OOD protocol for SBDD papers (e.g. Pocket2Mol, TargetDiff) uses **Tanimoto > 0.4 between any train-test pair must be impossible** — i.e. the *inverse* of the current `chemical` semantics, and a stricter threshold. As shipped, `chemical` with `threshold=0.7` is the "moderately similar" filter, not the "dissimilar" filter the paper protocol demands. Grep for `Tanimoto.*0.4` against `molmetal/` returns 0 hits.

**Gap 3: MMseqs2 30%-sequence-identity pocket filter is documented but not implemented.** `engineering_practices.md:114` requires it; `grep -r mmseqs /home/hugo/codes/try_triton_on_rocm/molmetal/` returns 0 hits. CrossDocked2020 has known sequence-redundancy problems at the pocket level (multiple PDB IDs share >30% sequence identity); without this filter, the 100-test-pocket set may overlap with the training pocket set on the *protein* axis. The paper's §4 evaluation metric (Vina, PB) is *not* invariant to protein overlap because the pocket-conditioned CFM path can leak pocket features.

**Gap 4: `metallo_drugs_500_train.csv` is not split at all.** Same as P6.1 Gap 1 — the 500-mol pool has no train/val/test partition, and no `TemporalSplitter(cutoff_year=2024)` cut has been applied even though MetalCytoToxDB has year metadata. A temporal-split PlatinAI baseline number would be the strongest claim against data drift.

### Honest framing

The split infrastructure is *mature and correct* (`ScaffoldSplitter` is leak-safe; `LigandDeduplicatedSplitter` solves the 93% leakage issue; `TemporalSplitter` is the right tool for data-drift claims). The gap is purely **CLI integration**: no headline eval script accepts `--split`, no metallodrug pool is partitioned, and the OOD-protocol-strength Tanimoto<0.4 + MMseqs2 pair is unwired. The paper can claim "we evaluated on a held-out test set" but **cannot claim** "we evaluated on a held-out OOD test set", which is what TargetDiff / Pocket2Mol readers expect.

### Concrete patch plan (for the user to accept / reject)

1. **Add `--split {random|ligand_dedup|scaffold|temporal|chemical}` to `r4_lambda_only_run.py`** (1h, same patch as P6.1 #1).
2. **Replace `ChemicalSplitter` semantics** OR add a new `TanimotoDissimilarSplitter(threshold=0.4)` that *forbids* any train-test pair with Tanimoto ≥ 0.4 (1h + 30 min pytest + 30 min leak guard test).
3. **Add `molmetal/scripts/sequence_dedup_crossdocked.py`** that uses `MMseqs2` (or a Python fallback via `Bio.SeqIO` + `blastp`) to cluster the 100-test-pocket sequences against the 100k-train-pocket sequences at 30% identity, and emits a `pocket_dedup.json` with the surviving test pockets. Estimated: 4h engineering + 2h on a real MMseqs2 install (which is *not* currently in `pyproject.toml`; would need `uv add mmseqs` or the python wrapper).
4. **Update `paper/sections/04_evaluation.tex`** with a single Table that reports the 5 splits × headline metric cell (the table already exists for `scaffold_split_test` in `r3_drugood_benchmark.py` — extend it). Add §6 limitations line: "random split retains 93% seen-SMILES leakage; scaffold + temporal + Tanimoto<0.4 + MMseqs2-30%-seq-id are the leak-free OOD-protocol set, reported in Table X."

---

## Honest framing (overall)

**3 of 3 pitfalls are PARTIAL** — substantial infrastructure exists but is not wired into the headline eval scripts or the metallodrug-de-novo-vertical pool. The most concrete single claim that *is* valid today is: *"r4_c_full_sweep.py:140-192 has sha256-hashed inputs + deterministic job_ids and is reproducible from a fresh clone with `uv sync --frozen`."* Everything else requires patches to be honest.

**No patch is on the critical path** for the next paper submission — but **all 3 are required** for the *next* paper (R13→R14 evolution per `TODO-25 / TODO-26` planning memos). Recommended sequencing:

1. **P6.1 #1 + #2 (1.5h, CPU-only)** — scaffold-split CLI + pool-split artefact. Closes the most visible leak: the PlatinAI oracle's training-time scaffold memorisation.
2. **P6.3 #1 (1h)** — `--split` flag on `r4_lambda_only_run.py`. Closes the headline-script gap.
3. **P6.2 #1 (30 min)** — seed + checkpoint sha256 in output JSON. Closes the simplest reproducibility hole.
4. **P6.2 #2 (2h)** — `fingerprint_artefacts.py` CLI. The "no silent drift" guard for future contributors.
5. **P6.3 #2 + #3 (defer to R14)** — Tanimoto<0.4 + MMseqs2-30% require both a new splitter semantics and an external binary install; these are the right scope for a "data hygiene" round, not a "de novo vertical pilot" round.

**Total CPU-only patch budget: ~5h engineering + 2h pytest.** All patches are paper-grade clean: no dependency changes, no GPU needed, no new external services.

---

## Cross-references

- `molmetal/reports/wf_pitfall_audit/p1_reaction_rules.md` — upstream P1 audit (typed-term MCTS rule surface).
- `molmetal/reports/wf_pitfall_audit/p2_generator_arch.md` — upstream P2 audit (generator architecture pitfalls).
- `molmetal/reports/wf_pitfall_audit/p3_reward_design.md` — upstream P3 audit (reward design layer).
- `molmetal/reports/wf_pitfall_audit/p4_eval_novelty.md` — upstream P4 audit (EVAL & NOVELTY layer; cites the 30-cell R12 Path A 10×3 panel that this audit confirms is *not* split-aware).
- `molmetal/reports/wf_pitfall_audit/p5_wetlab_validation.md` — upstream P5 audit (wet-lab validation layer).
- `TODO/engineering_practices.md:97-115` — canonical 4-split policy + MMseqs2 30%-seq-id requirement.
- `TODO/environment.md:178-207` — CI/reproducibility contract (uv.lock, --frozen, pre-commit hook).
- `molmetal/reports/wf_remove_smoke/final.md:1-282` — Phase-4 smoke-removal verdict (pytest 1585 pass / 8 fail; smoke artefacts cleaned).
- `TODO/pending/22_data_gap_alignment_plan.md` — TargetDiff / Pocket2Mol gap inventory (R12=9, R13=17, deferred=8 metrics).
- `TODO/pending/14_full_100pocket_paper_r13.md` — R13 100×3 sweep plan (the natural carrier for `--split scaffold` integration).
- `TODO/pending/26_round14_complete_ship_plan.md` — R14 ship plan (where the leak-free OOD protocol belongs).
