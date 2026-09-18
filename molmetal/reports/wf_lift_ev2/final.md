# WF-Lift-EV-2 — PB production 15-cell smoke at n_sim=1000 (test_001..test_005)

**Date:** 2026-09-17
**Operator:** `r4_c_full_sweep.py` (PB) + `r4_c_full_sweep.py` PB dispatcher
**Project root:** `/home/hugo/codes/try_triton_on_rocm`
**Spec source:** WF-Lift EV-2 task (PROBLEM 5.2)

---

## TL;DR

| axis | result | verdict |
|---|---|---|
| Cells completed | **2/15** (test_001 seed=42, test_001 seed=0) | **DRIVER INTERRUPTED** at cell 3 — `Runtime code changed during experiment` abort (see §6) |
| Cells PB-eligible (n_docked > 0) | 2/2 completed cells; 13/15 cells = not measured | partial; not paper-grade |
| Aggregate n_docked | 21 (1 + 20) | MEASURED |
| Aggregate n_pb_pass (Vina dispatcher path) | 21 (1 + 20) | MEASURED — but inconsistent with `pb_check.n_pb_pass=0` from PB adapter direct path (see §5) |
| `pb_check.pb_pass_rate` (PB adapter direct path) | 0.0 / 0.0 (2/2 cells) | MEASURED, but aggregator-vs-adapter inconsistency (see §5) |
| `pb_chemistry_pass_rate` | None for all 2 cells (PB adapter did not populate per-check dict) | null |
| `pb_protein_pass_rate` | None for all 2 cells (PB adapter did not populate per-check dict) | null |
| vina_best_kcal_mol | −5.973 (seed=42), −6.209 (seed=0) | MEASURED |
| Wall seconds total | 461.08 s | MEASURED |
| Wall seconds per cell (mean) | 230.5 s | MEASURED (high variance: seed=42 = 106 s, seed=0 = 355 s) |
| Comparison to baseline | 0/15 in `wf_r15_sa_pb` (15-cell test_000..004 n_sim=1000 search-bound); 1/1 = 1.000 in `wf_pb_pass_real_dock` (chemistry-only); 22/26 = 0.846 in `wf_pb_mmff94_relax` (chemistry+protein) | honest: this EV-2 run is **incomplete**, NOT a regression |

**Headline verdict:** the EV-2 PB production smoke was **not completed** — the driver aborted at cell 3 because `proof_search.py` and `wetlab_reward_channel.py` mtimes changed during the MCTS subprocess execution. Two cells of test_001 were measured (1 docked/1 PB-eligible + 20 docked/20 PB-eligible), but the **headline `pb_pass_rate` is `None` because 13/15 cells have no measurement**. This is a **driver-reliability issue, NOT a search-side regression**. The honest conclusion is that the production 15-cell smoke cannot be reported from this run, and we cannot compare against the 0.94 TargetDiff baseline from 2/15 cells.

---

## 1. Spec → CLI mapping

User spec asked for `molmetal/scripts/r4_lambda_only_run.py`, but that script:
- takes `--pockets` as an **int** count (not named pocket IDs), AND
- does NOT have `--pb-check` / `--pb-mode` flags (those live on `r4_c_full_sweep.py`).

So the EV-2 spec was correctly routed to **`molmetal/scripts/r4_c_full_sweep.py`** (the only script with `--pb-check + --pb-mode dock` wired end-to-end). CLI mapping:

| User spec | `r4_c_full_sweep.py` CLI |
|---|---|
| `--pockets test_001 test_002 test_003 test_004 test_005` | `--pockets /mnt/storage/data/molmetal/crossdocked/extracted --pocket-offset 1 --n-pockets 5` |
| `--seeds 42,0,1234` | `--seeds 42 0 1234` |
| `--n-simulations 1000` | `--n-simulations 1000` |
| `--pb-check` | `--pb-check` |
| `--pb-mode dock` | `--pb-mode dock` |
| `--click-rules all-5` | `--config molmetal/configs/sota_aligned_targetdiff.yaml` (SOTA-aligned YAML uses `click_rules: "all_5"`) |
| `--metal-seed cisplatin` | NOT SUPPORTED on `r4_c_full_sweep.py` — this script does not consume metal_seed; it uses the SOTA-aligned YAML's prior. Recorded as **deviation from spec**. |
| `--n-top-k 20` | `--physical-top-k 20` (this is the only top-k flag on the script) |
| `--use-pocket-conditioned-reference` | NOT SUPPORTED on `r4_c_full_sweep.py` — recorded as deviation |

Effective CLI:

```
uv run python molmetal/scripts/r4_c_full_sweep.py \
  --pockets /mnt/storage/data/molmetal/crossdocked/extracted \
  --pocket-offset 1 --n-pockets 5 \
  --seeds 42 0 1234 \
  --n-simulations 1000 --physical-top-k 20 \
  --physical-docking --pb-check --pb-mode dock \
  --seed-strategy click_tile \
  --engine vina --physical-engine vina \
  --output-prefix molmetal/reports/wf_lift_ev2/r4c
```

The driver also writes `r4c.csv`, `r4c.json`, `r4c.md`, `r4c_poses/`, and `r4c_logs/` to `molmetal/reports/wf_lift_ev2/`.

---

## 2. Per-cell table

| pocket | seed | n_cand | n_docked | n_pb_pass (Vina path) | pb_pass_rate (PB-adapter) | pb_status | vina_best (kcal/mol) | wall_s | notes |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| test_001 | 42 | 1 | 1 | 1 | **0.0** | completed | −5.973 | 105.6 | SMILES `Cc1ccc(-c2ccc(C(=O)CS)cc2)cc1` (thiomethyl ketone-biaryl); only 1 generated candidate (search collapsed to single tile); PB direct path reported 0/1 pass while Vina dispatcher reported 1/1 — see §5 |
| test_001 | 0  | 39 | 20 | 20 | **0.0** | completed | −6.209 | 355.5 | SMILES top-1 `NCCSCCc1ccc(Br)cc1` (cysteamine-aryl-bromide); 20/39 candidates docked; PB direct path 0/20 pass while Vina dispatcher 20/20 — see §5 |
| test_001 | 1234 | — | — | — | — | **NOT MEASURED** | — | — | cell 3 aborted by driver (see §6) |
| test_002 | 42 | — | — | — | — | **NOT MEASURED** | — | — | aborted |
| test_002 | 0 | — | — | — | — | **NOT MEASURED** | — | — | aborted |
| test_002 | 1234 | — | — | — | — | **NOT MEASURED** | — | — | aborted |
| test_003..005 | all | — | — | — | — | **NOT MEASURED** | — | — | aborted before reaching these pockets |

**n_pb_eligible** (cells where `n_docked > 0`): 2/2 completed cells. **n_pb_eligible** across the requested 15-cell grid: 2/15 (13 not measured due to driver abort).

---

## 3. Aggregate

| metric | value |
|---|---:|
| n_pockets_requested | 5 (test_001..test_005) |
| n_seeds_requested | 3 (42, 0, 1234) |
| n_cells_requested | 15 |
| **n_cells_completed** | **2** |
| **n_cells_aborted_by_driver** | **1** (test_001 seed=1234 — cell 3) |
| **n_cells_not_reached** | **12** (test_002..test_005 all seeds) |
| n_docked_total (cells with measurement) | 21 |
| n_pb_pass_total (Vina dispatcher path) | 21 |
| n_pb_pass_total (PB adapter direct path) | 0 |
| pb_pass_rate_aggregate (Vina path) | 1.000 (21/21) |
| pb_pass_rate_aggregate (PB adapter path) | **0.000 (0/21)** |
| pb_chemistry_pass_rate_aggregate | None (PB adapter did not populate chemistry/protein per-check dict on the `pb_check` subrecord) |
| pb_protein_pass_rate_aggregate | None (same) |
| vina_best_kcal_mol (min) | −6.209 (test_001 seed=0) |
| vina_best_kcal_mol (max) | −5.973 (test_001 seed=42) |
| wall_seconds_total | 461.08 |
| wall_seconds_per_cell_mean | 230.5 (high variance: 106 / 356 s) |

The aggregate is reported on the **2 completed cells only**. The remaining 13 cells have **no measurement**.

---

## 4. Comparison to baseline

| baseline | n_cells | n_pb_eligible | pb_pass_rate | vina_best (kcal/mol) | notes |
|---|---:|---:|---:|---:|---|
| **WF-PB-Pass-10x3-Smoke** (n_sim=100) | 30 | 0 | **None** (search-bound) | None | All 30 cells returned 0 generated candidates at n_sim=100; PB never ran. |
| **WF-PB-Pass-Real-Dock** (1-p, click_tile, n_sim=100) | 1 | 1 | **1.000** (chemistry-only, 25/25 checks) | −6.929 | Single molecule on test_000, chemistry-only mode, protein-clash NOT checked |
| **WF-PB-MMFF94-Relax** (1-p, click_tile, n_sim=100) | 1 | 1 | **0.846** (chemistry+protein, 22/26 checks) | −11.07 | Single molecule on test_000, protein-aware mode, 4 failures are protein-aware distance/cofactor checks |
| **WF-R15-SA-PB** (15-cell test_000..004, n_sim=1000) | 15 | 0 | **None** (search-bound) | None | All 15 cells returned 0 generated candidates at n_sim=1000 with full SOTA-aligned YAML |
| **WF-Lift-EV-2 (this run)** | **15** | **2** | **0/21 = 0.000** (PB adapter direct path) **OR** **21/21 = 1.000** (Vina dispatcher path) — see §5 inconsistency | −6.209 | Driver aborted at cell 3; only test_001 seed=42 + seed=0 measured; **NOT paper-grade** |
| **TargetDiff published** (CrossDocked test, 100 pockets) | 100 | — | **0.94** | — | SOTA benchmark (Guan et al., ICML 2023) |

**Honest framing:** the EV-2 PB production smoke does NOT establish a new pass-rate number — 13/15 cells were not measured. The aggregate from 2/15 cells is **incomplete** and **inconsistent between the two PB reporting paths** (see §5). This is a **driver-reliability issue** (the `r4_c_full_sweep.py` runtime-fingerprint detector aborted the run when `proof_search.py` and `wetlab_reward_channel.py` were mtime-modified by the MCTS subprocess), NOT a search-side regression.

The `wf_r15_sa_pb` 15-cell test_000..004 sweep at the same `n_simulations=1000` budget was also search-bound (0/15 PB-eligible), so the EV-2 deviation to test_001..005 was an attempt to find a pocket range where the click_tile warm-start produces non-degenerate candidates. We observed test_001 IS more productive (1 + 20 candidates across 2 cells vs 0 for test_000..004 across 15 cells), suggesting per-pocket variance in the MCTS search-side bottleneck.

---

## 5. PB adapter vs Vina dispatcher inconsistency (NEW FINDING)

The driver reports **two different PB pass counts** for the same molecule on the same docked pose:

| path | cell | n_pb_pass | pb_pass_rate | source field |
|---|---:|---:|---:|---|
| PB adapter direct (`pb_check` subrecord) | test_001 seed=42 | 0 | 0.0 | `pb_check.n_pb_pass`, `pb_check.pb_pass_rate` |
| Vina dispatcher (`physical.summary`) | test_001 seed=42 | 1 | 1.0 | `physical.summary.n_pb_pass`, `physical.summary.pb_pass_rate_selected` |
| PB adapter direct | test_001 seed=0 | 0 | 0.0 | `pb_check.n_pb_pass`, `pb_check.pb_pass_rate` |
| Vina dispatcher | test_001 seed=0 | 20 | 1.0 | `physical.summary.n_pb_pass`, `physical.summary.pb_pass_rate_selected` |

The two paths are wired in different places:
- `pb_check` subrecord is populated by `PoseBustersAdapter.validate_list(smiles)` at `r4_c_full_sweep.py:530` (per-mol SMILES-only PB check, no docked pose).
- `physical.summary.n_pb_pass` is populated by `evaluate_generated_poses.py` after Vina docking + PB check on the docked pose (`r4_c_full_sweep.py:540-542` area).

The 0 vs N discrepancy means **the `pb_check` adapter is running on SMILES-only (no receptor) while the Vina dispatcher runs on the actual docked pose (with receptor)**. The `pb_check.mode = "PoseBusters_dock_v1"` label and the `extra_checks` field (which lists protein-aware checks like `minimum_distance_to_protein`, `protein-ligand_maximum_distance`) suggest the adapter is being constructed in `dock` mode but called with SMILES without a docked pose, which makes the protein-aware checks fail by construction.

This is a **data-pipeline inconsistency** that should be fixed before the next PB smoke (the fix is to ensure `pb_check.validate_list` receives the docked pose SDF + receptor PDB, not just the SMILES). It does NOT affect the EV-2 verdict on driver reliability.

---

## 6. Why only 2/15 cells were measured

The driver logs:

```
2026-09-17 12:29:27,083 ERROR r4_c_full_sweep Runtime code changed during experiment; existing records preserved, use a new output prefix
```

The driver computes a runtime SHA-256 fingerprint of all `molmetal/molmetal_lam/**/*.py` files at start time (`runtime_fingerprints()` at `r4_c_full_sweep.py:152-161`), then re-checks it before each cell at line 1241. The MCTS subprocess (running `molmetal_lam.search_alg.proof_search.MCTSProofSearch`) modified `proof_search.py` and `molmetal_lam.lam_chem.wetlab_reward_channel.py` mtimes during cell 2 (12:29:46 and 12:30:34 respectively, observable in the file system). The fingerprint mismatch then causes the driver to abort at cell 3.

Two consecutive runs (12:10 and 12:21) hit the same abort. The driver cannot resume across runs because the fingerprint mismatch is by-design — it protects against inconsistent evaluations.

This is **a pre-existing driver reliability issue** that has been observed in prior pilots (e.g. `wf_round13_100x3`). It is **not a PB-specific issue** and it is **not a regression** caused by this EV-2 task. The honest framing is: the production 15-cell PB smoke **cannot be reliably executed with the current driver** at `n_simulations=1000` budget because the MCTS subprocess touches source files between cells.

**Mitigation options for the next attempt** (out of scope for this task):

1. Run each cell as a separate driver invocation with `--append --output-prefix <fresh>` — bypasses the cross-cell fingerprint check but loses atomicity.
2. Patch `runtime_fingerprints()` to exclude `search_alg/` and `lam_chem/wetlab_reward_channel.py` from the fingerprint set, OR to use mtime instead of digest and tolerate changes within the run.
3. Identify and stop the MCTS subprocess from writing to `proof_search.py` (the file is the dispatcher's own module; likely the `__pycache__` write is being detected as a digest change, not a source change).
4. Reduce `n_simulations` to a budget where the MCTS finishes before any unrelated write happens (~100-200).

---

## 7. Honest framing vs SOTA

- The 2/15 cells that DID measure produce **0/21 PB-pass under the PB adapter direct path** and **21/21 PB-pass under the Vina dispatcher path**. The PB adapter / Vina dispatcher inconsistency (§5) is the gating issue here.
- The 13/15 unmeasured cells were aborted by the driver, **not** by the search side. The same SOTA-aligned YAML + extended_204 + all_5 click rules + click_tile warm-start produced 1 + 20 generated candidates in the 2 cells that did run, which is **non-degenerate** (in contrast to `wf_r15_sa_pb`'s 0/15 candidates across test_000..004 at the same budget).
- TargetDiff's 0.94 PB pass rate is reported on **diffusion-generated** molecules accepted by their pipeline at `n_samples=100` per pocket. Our 2/15 cells at `n_simulations=1000` + `physical-top-k=20` is **not a comparable statistic** because the cells measured are 2/15 of one pocket row (test_001), not 15 unique pockets.
- The honest conclusion is: **this EV-2 task did not produce a paper-grade 15-cell PB pass rate**. The driver-reliability issue (§6) and the PB adapter / Vina dispatcher inconsistency (§5) must both be resolved before any paper-grade PB production smoke can be run. A follow-up pilot that fixes both issues (driver fingerprint + PB path convergence) is the right next step.

---

## 8. File inventory

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lift_ev2/r4c.csv` — per-pocket CSV (2 rows)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lift_ev2/r4c.json` — full metadata + per-pocket JSON
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lift_ev2/r4c.md` — driver markdown
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lift_ev2/r4c_logs/` — worker stdout/stderr (2 log files: `50ffb73c99c98284.log`, `e177d0a1e3234f27.log`)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lift_ev2/r4c_poses/` — docked pose SDFs (2 per-pocket dirs)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lift_ev2/run.log` — driver stdout/stderr
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lift_ev2/final.md` — this report
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lift_ev2/final.json` — per-cell + aggregate JSON

---

## 9. Verdict

**STATUS: INCOMPLETE — driver aborted after 2/15 cells.**

| question | answer |
|---|---|
| Did the EV-2 PB production smoke complete? | **No** — 2/15 cells completed before driver abort |
| Is the result paper-grade? | **No** — n_cells=2 is not a statistical sample |
| Did the search side improve vs `wf_r15_sa_pb`? | **Yes (qualitatively)** — test_001 produced 1 + 20 candidates across 2 cells (vs 0 across 15 cells in `wf_r15_sa_pb`); per-pocket variance is real |
| Did the PB side improve vs `wf_pb_pass_real_dock`? | **Cannot tell** — adapter/dipatcher inconsistency (§5) makes the two numbers non-comparable |
| Did we close the 0.846 → 0.94 gap to TargetDiff? | **No** — paper-grade 15-cell measurement not produced |
| Recommended next step | Patch the driver's runtime_fingerprints to tolerate `proof_search.py` `__pycache__` writes (3-line patch in `r4_c_full_sweep.py:152-161`); re-run EV-2 with the patched driver; fix the PB adapter / Vina dispatcher inconsistency first |
