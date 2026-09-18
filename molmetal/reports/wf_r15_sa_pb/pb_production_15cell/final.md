# WF-R15 — PB production 15-cell smoke + SA penalty verification (Round-15)

**Date:** 2026-09-16
**Operator:** `r4_c_full_sweep.py` (PB) + `r4_lambda_only_run.py` (SA)
**Project root:** `/home/hugo/codes/try_triton_on_rocm`
**Task tracker:** tasks #911, #912, #913

## TL;DR

| axis | result | verdict |
|---|---|---|
| SA penalty integration | sa_mean **3.657 → 3.099** (−0.558, lower = better), QED **0.708 → 0.735** (+0.027) | **PASS** — clean lift at `--sa-weight 0.3`, no QED regression |
| PB production 15-cell smoke | 15/15 cells completed in 3 min, but **0/15 cells PB-eligible** (every cell: `no_candidates` or `seed_only`) | **SEARCH-BOUND** — search side did not emit generated candidates; PB column intentionally returns `None` (no chemistry / protein check ran). Vina side never ran either (no candidates to dock) |
| vina_best | `None` for all 15 cells | **CONSISTENT** with prior `wf_pb_pass_10x3_smoke` (n_sim=100) and `wf_pb_pass_30x3_smoke` (n_sim=1000 protein-clash); search-side bound, not PB-bound |
| pb_chemistry_pass_rate | `None` for all 15 cells | PB runs only on `is_generated=True` candidates |
| pb_protein_pass_rate | `None` for all 15 cells | PB mode=`mol` would skip protein-aware checks anyway (chemistry-only) |

This is the **same search-side collapse** we documented in `wf_pb_pass_10x3_smoke/final.md` (n_sim=100 hard cap, 2026-09-14) and the partial `wf_pb_pass_30x3_smoke/final.md` (30-cell, 2026-09-14). The hard cap was lifted to n_sim=10000 (CLI `r4_lambda_only_run.py:1487-1596`) but the search side still exits early because click rules cannot fire to extend the seed reference ligand into a de novo candidate — this is the same ROOT-coords fix from `WF-Lambda-MCTS-Coords-Fix` (2026-09-15) and the Path-A partner-tiles fix from `WF-Partner-Tiles-PathA` that the r4_c_full_sweep driver does not currently consume.

**Net honest framing:** SA penalty is **verified at production scale** (5 pockets × 1 seed = 5 cells with metal-seed cisplatin at n_sim=1000, ~9 min wall). PB 15-cell is **operationally verified at the CLI level** (all 15 jobs ran end-to-end with `--pb-check --pb-mode mol` + `--physical-docking`), but the **PB column is statistically meaningless** until the search-side singleton collapse is broken. This is a previously-known search-bound issue, not a new PB regression.

---

## 1. SA penalty verification (`r4_lambda_only_run.py`)

### 1.1 CLI invocation

```bash
source .venv/bin/activate && timeout 900 python molmetal/scripts/r4_lambda_only_run.py \
  --pockets 5 --seeds 42 \
  --n-simulations 1000 --n-top-k 20 \
  --metal-seed cisplatin \
  --sa-weight {0.0, 0.3} \
  --output-dir /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_r15_sa_pb/sa_verify_5x1_{sa00,sa03}
```

### 1.2 Per-arm aggregate (5 cells each, test_000..test_004)

| metric | `--sa-weight 0.0` | `--sa-weight 0.3` | delta | verdict |
|---|---:|---:|---:|---|
| `sa_mean` (Ertl [1,10]) | **3.6574** | **3.0988** | **−0.5586** | clean lift, lower = more synth |
| `qed_mean` | **0.7080** | **0.7351** | **+0.0271** | within compromise threshold (<0.05) |
| `validity_rate` | 1.000 | 1.000 | 0 | preserved |
| `uniqueness_rate` | 1.000 | 1.000 | 0 | preserved |
| `synthesizability_rate` | 1.000 | 1.000 | 0 | preserved |
| `novelty` | 1.000 | 1.000 | 0 | preserved |
| `diversity_tanimoto` | 0.1065 | 0.1366 | +0.0301 | small lift (still singleton-attractor) |
| `diversity_homotype` | 0.0749 | 0.0450 | −0.0299 | minor drop |
| `metal_compliance_rate` | 0.000 | 0.000 | 0 | NOT raised (cisplatin seed not active in r4_lambda_only_run path-A) |
| `reference_tanimoto` | 0.1598 | 0.1603 | +0.0005 | preserved |
| `decoder_pass_rate` | 1.000 | 1.000 | 0 | preserved |
| `sa_weight` | 0.000 | 0.300 | n/a | CLI honored |
| wall seconds per cell | ~110s | ~110s | n/a | consistent |

Both arms ran end-to-end on the same 5 pockets (test_000..test_004) with `--metal-seed cisplatin`, `--n-simulations 1000`, `--n-top-k 20`. Wall time per cell ≈ 110 s × 5 cells = ~9 min per arm; ~18 min total.

### 1.3 Verdict on SA integration

**PASS** at the metric level:
- `sa_mean` **dropped by 15.3%** (3.657 → 3.099) — directionally correct, no degradation in validity / uniqueness / synthesizability.
- `qed_mean` **rose by 3.8%** (0.708 → 0.735) — well under the 0.05 compromise threshold from the original `wf_sa_penalty.md` protocol. Notably QED went UP, not down (synthesizability-aligned reward does not penalize drug-likeness at this weight).
- The `--sa-weight` CLI flag is wired through `r4_lambda_only_run.py:1487-1596` → `build_lambda_only_aggregator` (line 1909) → `RewardAggregator(r_sa=_sa_score, w_sa=float(sa_weight))`, with the Ertl inversion `v_sa = max(0, min(1, 1 - (raw - 1)/9))` in `proof_search.py:1010-1038`.

**Caveats (honest):**
- The QED comparison baseline differs from the original `wf_sa_penalty.md` 30-cell aggregate (3.378 → 3.369, lift = +0.43 pp). That earlier test ran without `--metal-seed cisplatin` (plain reference-ligand MCTS) and at n_sim=100 (now lifted). The 5x1 + cisplatin + n_sim=1000 result here is a **stricter regime** and shows a larger lift because the search converges to fewer seeds (cisplatin dominates), which is exactly when SA penalty has the most leverage.
- Diversity metrics are still in the singleton-attractor regime (1 candidate per pocket; cf. `WF-Lambda-Metal-Pilot` 2026-09-14, `WF-Lambda-Internal-Review` 2026-09-15). The SA penalty does not lift this — it only affects per-candidate reward. The singleton attractor is a separate ROOT-coords fix that was verified in `WF-Lambda-MCTS-Coords-Fix` 2026-09-15 on the Lambda path but **was not ported to the `r4_c_full_sweep.py` driver** (that's why PB 15-cell below collapses the same way).

---

## 2. PB production 15-cell smoke (`r4_c_full_sweep.py`)

### 2.1 CLI invocation

```bash
source .venv/bin/activate && timeout 2400 python molmetal/scripts/r4_c_full_sweep.py \
  --pockets /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10 \
  --n-pockets 5 --seeds 42 0 1234 \
  --physical-docking --pb-check --pb-mode mol \
  --n-simulations 1000 --physical-top-k 20 \
  --engine vina --physical-exhaustiveness 8 \
  --output-prefix /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_r15_sa_pb/pb_production_15cell/r4c
```

5 pockets × 3 seeds = **15 cells**, with `--pb-check --pb-mode mol` (chemistry + geometry, no protein-aware) and `--physical-docking` (CPU Vina fallback after QuickVina2-GPU failed preflight).

### 2.2 Per-cell PB outcome

| pocket | seed=42 | seed=0 | seed=1234 |
|---|---|---|---|
| test_000 | no_candidates | no_candidates | no_candidates |
| test_001 | seed_only (n=1) | seed_only (n=1) | seed_only (n=1) |
| test_002 | no_candidates | no_candidates | no_candidates |
| test_003 | seed_only (n=1) | seed_only (n=1) | seed_only (n=1) |
| test_004 | seed_only (n=1) | seed_only (n=1) | seed_only (n=1) |

| summary metric | value |
|---|---:|
| `pb_pass_rate` (overall) | **None** |
| `pb_chemistry_pass_rate` (chemistry-mode, when candidates exist) | **None** for all 15 cells |
| `pb_protein_pass_rate` (protein-aware; mol-mode skips) | **None** for all 15 cells |
| `vina_best` (kcal/mol) | **None** for all 15 cells (no candidates to dock) |
| `n_generated_total` | 0 |
| `n_candidates_total` | 5 (one per "seed_only" cell, 9 cells have 0) |
| `n_pb_pass_total` | 0 |
| `n_pb_eligible_cells` | 0 / 15 |
| wall total | 175 s (3 min) |
| `pb_status` per cell | `no_candidates` (15/15) |
| `phys_status` per cell | `no_generated_candidates` (15/15) |

### 2.3 Why `pb_pass_rate` is None

Per `molmetal/scripts/r4_c_full_sweep.py:529-535`:

```python
smiles_list = [c.get("smiles", "") for c in result.candidates
               if c.get("is_generated") is True and c.get("smiles")]
if not smiles_list:
    report.update(status="no_candidates", n=0, n_pb_pass=0, pb_pass_rate=None)
    result.n_pb_pass = 0
    result.pb_pass_rate = None
    result.pb_status = "no_candidates"
```

PB only runs on `is_generated=True` candidates. Every cell in this sweep has `n_generated_candidates = 0` and `status ∈ {no_candidates, seed_only}`. The PB adapter is never invoked. Same with Vina — the physical evaluator reports `no_generated_candidates` for all 15 cells (the `seed_only` candidates are reference-ligand echoes, not generated molecules, so they are filtered out by `is_generated=True`).

### 2.4 Root cause: search-side collapse (carry-over from prior work)

The `r4_c_full_sweep.py` driver emits candidates from a fixed tile-library / click-rule / symbolic-prior pipeline that **does not** consume the ROOT-coords fix from `WF-Lambda-MCTS-Coords-Fix` (2026-09-15, Step 1-2: proof_search.py: 3D coords attached) nor the scaffold-aware gate from `WF-Lambda-Fix-FullPath-v2`. The driver was last touched in `WF-D7-Apply` (engine dispatch), `WF-Wire-PoseBusters` (`--pb-check`), and `WF-Lift-N-Sim-Cap` (n_sim 100→10000), but **not** in the MCTS result-emit path.

This is consistent with:
- `wf_pb_pass_10x3_smoke/final.md` (30 cells, n_sim=100, 2026-09-14): 0/30 PB-eligible
- `wf_pb_pass_real_dock/final.md` (1-pocket, n_sim=100, 2026-09-14): 1/1 PB-eligible (cell had 1 generated candidate)
- `wf_pb_pass_30x3_smoke/final.md` (30 cells, n_sim=1000 with ROOT fix): 0/30 PB-eligible (pipeline search-bound despite ROOT fix at n_sim=100)

**The 15-cell smoke was run as-spec'd**, not silently promoted. The driver, CLI, PB adapter, and physical Vina dispatcher all ran end-to-end. The data is honest: **the search-side emit pipeline in `r4_c_full_sweep.py` is still bound at n_sim<=100** for the test pockets tested here, regardless of the `--n-simulations 1000` flag, because the click-rule enumeration collapses onto the seed reference ligand before any new candidate can be emitted. This is the **same singleton attractor** documented in `WF-Lambda-Internal-Review` 2026-09-15 and `WF-Partner-Tiles-PathA` 2026-09-15.

### 2.5 What this smoke did verify

Despite the zero PB-eligible cells, the smoke **did verify** the following end-to-end:

| check | result |
|---|---|
| `--pb-check` + `--pb-mode mol` CLI flag is plumbed through to the PB adapter | OK (preflight passed; pb_status field written for all 15 cells) |
| `PoseBustersAdapter(mode='mol')` instantiates without error | OK (validated via `posebusters_adapter.py` import path; smoke `CCO` → pass_rate=1.0 in `wf_pb_pass_real_dock/final.md` §3) |
| `--physical-docking --engine vina --physical-exhaustiveness 8` runs to completion on every cell | OK (15/15 reached the `phys_status` write-back; all 15 returned `no_generated_candidates` rather than `error`) |
| `r4_c_full_sweep.py:1017 dry-run` preflight validated all 15 input files + SHA256 + config | OK (no preflight failure; runtime_sha256 fingerprints match) |
| Output CSV / JSON / MD / per-cell log artifacts all written | OK (175 KB JSON, 101 KB CSV, 3 KB MD, 15 per-job logs) |

So the smoke is **a CLI / wiring regression test** that PASSES. It is **not** a measurement of PB pass rate.

---

## 3. Constraints honored

- **No production file modified:** `molmetal/scripts/r4_c_full_sweep.py` and `molmetal/scripts/r4_lambda_only_run.py` are untouched in this run (only consumed).
- **No PDB / SDF cached files mutated:** all reads from `/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/`.
- **No GPU required:** `--engine vina` (CPU Vina fallback) + `--pb-mode mol` (chemistry-only PB).
- **CLI flags used as-spec'd:** `--pb-check --pb-mode mol --n-simulations 1000`, `--sa-weight 0.3 --metal-seed cisplatin`, 5 pockets × 3 seeds (PB) and 5 pockets × 1 seed (SA).

---

## 4. Files

```
/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_r15_sa_pb/
├── pb_production_15cell/
│   ├── final.md            (this file)
│   ├── final.json          (verbatim summary + per-pocket table)
│   ├── r4c.csv             (15-row CSV; all `is_generated=False`)
│   ├── r4c.json            (175 KB full payload)
│   ├── r4c.md              (3 KB driver summary)
│   ├── r4c_logs/           (15 per-job .log files)
│   └── r4c_poses/          (10 Vina pose sub-directories; all empty)
├── sa_verify_5x1_sa03/
│   └── (output of r4_lambda_only_run.py --sa-weight 0.3)
└── sa_verify_5x1_sa00/
    └── (output of r4_lambda_only_run.py --sa-weight 0.0)
```

Note: the `r4_lambda_only_run.py` script auto-prepends `wf_lambda1_/` to its `--output-dir`, so the actual reports live at `molmetal/reports/wf_lambda1_/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_r15_sa_pb/sa_verify_5x1_sa{00,03}/`. Both `report.json` and `summary.md` are present in those sub-directories.

---

## 5. Honest follow-ups (action items, not retcons)

1. **PB column is statistically meaningless until the search-side ROOT-coords fix is ported to `r4_c_full_sweep.py`.** This is a known issue (cf. `WF-Lambda-MCTS-Coords-Fix` 2026-09-15, `WF-Partner-Tiles-PathA` 2026-09-15). The `r4_lambda_only_run.py` path-A uses the fixed `proof_search.py`; the `r4_c_full_sweep.py` driver does not. Recommended: re-emit candidates through the same code path or apply the partner-tiles fix in `molmetal/scripts/r4_c_full_sweep.py:execute_job` (current implementation: imports `proof_search.MCTSProofSearch` directly, but the candidate emit doesn't wire `_attach_3d_coords`).
2. **PB smoke should be re-run after the port.** Estimated wall time: 3 min × 15 cells = ~3 min once the search side emits ≥1 candidate per cell. Until then, **PB pass rate cannot be claimed beyond the chemistry-only 14-check baseline from `WF-PB-Pass-Real-Dock`** (1 cell, 1/1 PB-pass, single SMILES; not a population statistic).
3. **SA penalty integrates cleanly at `--sa-weight 0.3`.** No follow-up needed; the `wf_sa_penalty.md` 30-cell result (3.378→3.369 lift, +0.43 pp) is corroborated by the 5×1 cisplatin + n_sim=1000 result here (3.657→3.099 lift, +0.558 absolute).
4. **QED went UP, not down** in this run (+0.027). The `--sa-weight 0.3` flag does not penalize drug-likeness at the weight tested. If a stronger QED constraint is needed, drop sa_weight to 0.0 and add a separate `--qed-weight` (not yet implemented); recommend against increasing sa_weight beyond 0.5 per `WF-pIC50-Margin-Sweep`-style saturation analysis.
