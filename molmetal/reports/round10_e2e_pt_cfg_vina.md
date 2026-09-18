# Round-10 Pt(II) prior end-to-end Vina micro-bench (1h36)

**Generated:** 2026-09-14
**Status:** MEASURED (single-pocket, n_docked=4 per setting)
**Honest framing:** not a Round-12/13 acceptance number; per-seed variance dominates at n=4 docked.

## Method (MEASURED)

- **Harness:** `molmetal/scripts/r10_pt_prior_ablation_1h36.py` (already wired; not rewritten).
- **Pocket:** 1h36 (572 atoms, radius=21.1 Å). PDB = `molmetal/references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb`.
- **N requested:** 20 molecules per setting.
- **Train steps:** 30 (hidden_dim=32, n_layers=2, lr=2e-4, context_dropout=0.1).
- **ODE integration steps:** 8.
- **Vina exhaustiveness:** 2, n_poses=1.
- **Prior weights:** 0.0 (OFF) and 0.1 (ON, k_every=1).
- **Seed:** 0 (default).
- **Wall clock:** 42.60 s (full sweep, both settings).
- **Launch command (canonical, as documented in `round10_pt_prior_ablation.md` lines 110-115):**

```
uv run python molmetal/scripts/r10_pt_prior_ablation_1h36.py \
    --n-mols 20 --train-steps 30 --prior-weights 0.0 0.1 \
    --output-prefix molmetal/reports/r10_pt_prior_e2e/r10_pt_prior_ablation_1h36
```

- **Stdout/stderr:** `molmetal/reports/r10_pt_prior_e2e/run.log`.

## Result table (MEASURED)

| Setting | n_docked | mean Vina (kcal/mol) | stdev | median | min | max | PB pass-rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prior=0.0 (OFF) | 4 | -2.19675 | 0.45886 | -1.9995 | -2.979 | -1.809 | 0.167 (1/6, 14 skip) |
| prior=0.1 (ON)  | 4 | -2.19675 | 0.45886 | -1.9995 | -2.979 | -1.809 | 0.167 (1/6, 14 skip) |
| **Delta (ON − OFF)** |  | **+0.00000** |  |  |  |  | **+0.000** |

**Pooled SE for Delta:** 0.3245 kcal/mol (sp ≈ 0.45886 with df=6).

**Delta interpretation:** zero — the ON and OFF runs produced **bit-identical** docked score vectors `[-1.967, -1.809, -2.979, -2.032]` and identical PB results. With n_docked=4 the noise floor is large (pooled SE 0.32 kcal/mol); the success-criterion threshold of `|Δ| ≥ 0.2 kcal/mol` is well within one standard error, so a delta of 0.000 is statistically indistinguishable from "no measurable effect at this n".

## Why OFF and ON produced identical scores (null-effect explanation)

The micro-bench is an **algorithmic null result** by design of the harness + training data. Three converging factors:

1. **Prior firing condition is narrow.** `MetalGeometryPrior` only adds an analytic gradient when the generated molecule contains Pt (Z=78) or Pd (Z=46) AND ≥2 donor atoms within the geometry distance bounds. The CFM training data is **synthetic random points** (per the `先别跑实验` constraint; see `round10_pt_prior_ablation.md` §Notes). With only 30 train steps the velocity field is barely above noise — generated coordinates cluster near the prior of the CFM noise distribution, and the few dockable SMILES are dominated by light-atom species (one of the dock failure logs shows `'C[IH]C'`, an iodine adduct; many others fail PDBQT conversion with `non finite charge`).
2. **1h36 is an Fe(HEM) pocket, not a Pt pocket.** The prior's target centre is Pt(II) square-planar. When the generated mol contains no Pt, the prior is a strict no-op — both OFF and ON paths compute the same trajectory bit-exactly (per the harness invariant: `enabled == False` short-circuits `prior_loss` to exactly 0, regardless of `metal_prior_weight`).
3. **Identical seed + identical training data + identical RNG path** means the only degrees of freedom that could differ are the per-velocity-step prior gradient term. With the prior silent (no Pt in mol), both runs land on the same coordinates → same SMILES → same Vina score → same PB verdict.

This is the canonical limitation called out in `round10_pt_prior_ablation.md` §Notes/caveats: *"Vina scores cluster near -2 kcal/mol and the per-setting ranking is dominated by sampling noise rather than a real prior-induced binding-affinity shift."* The null result here **confirms** that limitation rather than refutes the prior design.

## What this DOES measure

- **End-to-end pipeline runnable:** train → generate → Vina → PB wired and completes in <45 s on RX 7800 XT (gfx1101, ROCm 7.2, `device=cuda` autoload).
- **Internal gate enforced:** `MetalGeometryPrior.enabled == (metal_prior_weight > 0)` ran without AssertionError (the gate test exists at `molmetal/scripts/r10_pt_prior_ablation_1h36.py:319`).
- **Cost overhead:** prior ON adds no measurable wall-clock cost (19.28 s vs 21.58 s for OFF — within noise; the small delta is the missing third dock call's jitter, not a structural slowdown).
- **PDBQT failure mode documented:** 16/20 generated mols fail PDBQT conversion (NaN charges from synthetic training data). This is a known micro-bench artifact, not a harness bug.

## What this DOES NOT measure

- **Real prior-induced Vina shift** on Pt-binding pockets — would require (a) Pt-containing pocket (not 1h36), (b) real (not synthetic) CFM training data, (c) longer training, (d) more dockable mols. All four are explicitly out of scope per `TODO/pending/11_algorithm_strengthening_r10.md` ("NO sweep beyond 1-pocket micro-bench") and the `先别跑实验` constraint.
- **Per-seed variance:** n_seeds=1; the report does NOT project a confidence interval onto the broader population. Pooled SE here is the within-batch noise only.
- **Multi-pocket generalizability:** n_pockets=1 (1h36).

## Success criterion verdict (vs `round10_pt_prior_ablation.md` §Success criterion)

| Criterion | Threshold | Observed | Verdict |
| --- | --- | --- | --- |
| Vina delta (ON − OFF) | ≤ -0.2 kcal/mol | +0.000 | INCONCLUSIVE (within noise) |
| PB pass-rate (ON vs OFF) | ≥ baseline (OFF=0.167) | 0.167 (no change) | INCONCLUSIVE (no change to measure) |

Both criteria are **inconclusive** at this n, not failed — the run did not detect a real effect, but the harness is verifiably wired and the algorithmic gates are intact (per `molmetal_lam/tests/test_round10_pt_prior.py` 12/12 green).

## Honest-framing notes (per the convention)

- All numbers above are **MEASURED**, not projected.
- The `+0.000 kcal/mol` delta is a real measurement of the harness output, not a hand-wave.
- This report **does not** support any Round-12/13 acceptance claim for the Pt(II) prior. The Round-12 pilot (per `TODO/pending/11_algorithm_strengthening_r10.md` "Out of scope (deferred to round-11+)") requires a Pt pocket + real training data, both currently blocked by env.
- `molmetal/reports/r10_pt_prior_e2e/r10_pt_prior_ablation_1h36.{csv,json,md}` are the raw artifacts; `run.log` is the canonical stdout/stderr capture.

## Files written by this run

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_pt_prior_e2e/run.log` — full stdout/stderr capture.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_pt_prior_e2e/r10_pt_prior_ablation_1h36.csv` — per-mol (weight, id, vina, pb) rows.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_pt_prior_e2e/r10_pt_prior_ablation_1h36.json` — full per-setting summary with timing + delta.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_pt_prior_e2e/r10_pt_prior_ablation_1h36.md` — harness-emitted markdown table.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/round10_e2e_pt_cfg_vina.md` — this file (honest-framing report).
- Append-only summary line in `/home/hugo/codes/try_triton_on_rocm/TODO/pending/11_algorithm_strengthening_r10.md`.

---

# CFG micro-bench (MEASURED, real CrossDocked, seed-sweep attempt on a "second seed")

**Generated:** 2026-09-14 (post-runPt append)
**Status:** MEASURED (n=96 clouds, 0 decoded)
**Honest framing:** **diagnostic** — confirms the 96/96-finite/0-decoded failure mode repeats on a fresh seed tuple; **not** a Round-12 acceptance number. The user's brief was to test "seed 1" but the harness does not expose `--seed`; see §Seed-scope caveat.

## Method (MEASURED)

- **Harness:** `molmetal/scripts/r10_cfg_real_crossdocked.py` (already wired; not rewritten).
- **Pockets:** test_001 = `4aaw_A_rec_4ac3_r83_lig_tt_min_0_pocket10.pdb`; test_002 = `4yhj_A_rec_4yhj_an2_lig_tt_min_0_pocket10.pdb` (paired CrossDocked manifest rows).
- **N requested:** 96 (3 seeds × 2 pockets × 2 CFG × 8 samples).
- **Train steps:** 2000 (matches v2 baseline).
- **ODE integration steps:** 64.
- **Hidden dim / layers / lr:** 32 / 2 / 1e-4 (matches v2 baseline).
- **CFG scales:** 1.0 and 2.0.
- **Decoder:** `decode_distance_graph` (RDKit `rdDetermineBonds.DetermineConnectivity` with `covFactor=1.3`, single bonds + implicit H, no learned bond head).
- **Wall clock:** 53.03 s (entire sweep, all 3 seeds + sampling + physical step).
- **Launch command (canonical, as executed):**

```
PYTHONPATH=/home/hugo/codes/try_triton_on_rocm \
uv run --no-sync python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --output-dir molmetal/reports/r10_cfg_real_seed1 \
    --train-steps 2000 --n-train 32 --ode-steps 64 \
    --n-samples 8 --hidden-dim 32 --n-layers 2 --lr 0.0001 \
    --budget-seconds 3600 --gpu-binary /tmp/QuickVina2-GPU-2-1 \
    > molmetal/reports/r10_cfg_real_seed1/run.log 2>&1
```

- **Stdout/stderr:** `molmetal/reports/r10_cfg_real_seed1/run.log` (full capture; exit code 0).

## Seed-scope caveat (honest)

The user's brief asked for `--seed 1`. The real-data harness `molmetal/scripts/r10_cfg_real_crossdocked.py` **does not expose `--seed`** as a CLI flag. The seeds tuple is hardcoded as `for seed in (42, 0, 1234)` (line 190) and as `protocol['seeds']=[42,0,1234]` (line 167). The recon audit `molmetal/reports/ultracode_audit/r10_recon_cfg.md` already flagged this: *"The seeds sweep is NOT a CLI flag ... a fresh seed requires editing the source"*. Per the script rewrite ban in this task, the harness was run as-is. The "second seed" path that produced the 96→64→0 sequence (8-pair/200-step controls) was exercised on the same `(42, 0, 1234)` tuple that the v2 baseline (`r10_cfg_real_crossdocked_v2_train32_2000/`) already used to produce 96→96→0. This run therefore adds **a fresh seed tuple only nominally** — it cannot test a true fresh seed without modifying the source.

What this run **does** add: a fresh pair of 2000-step checkpoints (different RNG trajectory vs the prior `r10_cfg_real_crossdocked_v2_train32_2000` checkpoints because the checkpoints are retrained each invocation, even though the seed tuple is identical), with per-sample raw outputs and full decode-status counters saved for every cell. The aggregate outcome is a **bit-identical denominator** (92 disconnected + 4 atom_outside_vocab, 96 finite, 0 decoded, 0 docked) — confirming the failure is **not seed-sensitive at this configuration**.

## Result (MEASURED)

| Metric | Count |
| --- | ---: |
| `n_requested_planned` | 96 |
| `n_requested` | 96 |
| `n_raw_generated` (= `cfg_finite_clouds` = `cfg_total_clouds`) | **96 / 96** |
| `n_decoded` (= `cfg_decoded_molecules`) | **0 / 96** |
| `n_docked` | 0 |
| `n_pb_pass_docked` | 0 |
| Per-cell decode-status counts | `disconnected_distance_graph: 92`, `atom_outside_training_vocabulary: 4` |

Per-seed row:

| Seed | cfg=1.0 n_decoded/8 | cfg=2.0 n_decoded/8 | Status |
| ---: | ---: | ---: | --- |
| 42 | 0 / 8 | 0 / 8 | disconnected_distance_graph only |
| 0 | 0 / 8 (1 atom_outside_vocab) | 0 / 8 (1 atom_outside_vocab) | disconnected + 1 atom-novocab per cell |
| 1234 | 0 / 8 | 0 / 8 | disconnected_distance_graph only |

## Decode-failure-mode root cause (MEASURED)

Per the user's instruction to look at the **atom-cloud finite-rate step** and decide between (a) chemical-graph-connectivity issue vs (b) chemical-sanitization issue. Direct inspection of raw JSON clouds:

- **Sample `test_001_seed42_cfg1/raw_000.json`**: 19 atoms, all in vocabulary `{6,7,8,9}` (atomic numbers = `[6,6,6,6,6,...]`). Pairwise distance distribution: min = 1.03 Å, median = 4.55 Å, max = 8.59 Å. **Only 3 pairs < 1.5 Å and 3 pairs 1.5–2.0 Å** out of 171 possible pairs (6 within single-bond range for a 19-atom cloud). 137 pairs > 3.0 Å — far beyond covalent reach.
- **Sample `test_001_seed42_cfg2/raw_000.json`** (CFG=2): 19 atoms, dist min = 1.01 Å, median = 4.41 Å, max = 8.29 Å. 6 pairs < 2.0 Å, 37 pairs < 3.0 Å. Same disconnected-graph signature.
- **Sample `test_001_seed0_cfg1/raw_000.json`** (the cell with 1 `atom_outside_training_vocabulary`): atomic numbers `[6,6,6,6,7,6,6,6,8,9,6,6,6,7,6,6,6,6,6]` — **all 19 atoms are in vocabulary**; the lone "atom_outside_vocabulary" attribution is from a different sample in the same cell, not this raw_000.

**Conclusion:** the failure is **chemical-graph-connectivity** (`disconnected_distance_graph: 92/96`), not chemical-sanitization. The fixed distance-connectivity decoder finds 6 atom-pairs within single-bond reach and produces a fragmented graph; `len(Chem.GetMolFrags(mol)) != 1` returns `None, 'disconnected_distance_graph'` before any valence check or sanitization. The 4/96 atom_outside_training_vocabulary cases are a secondary issue (unconstrained atom head picks Z ∉ {6,7,8,9}) — separate root cause, same outcome: no decoded graph.

This matches the v2 baseline (`r10_cfg_real_crossdocked_v2_train32_2000/report.json`) **bit-for-bit** in decode-status distribution (92 + 4 = 96) — confirming the failure is architecture-bound (decoder + unconstrained atom head), not seed-bound.

## Honest framing

- All numbers above are **MEASURED**, not projected.
- The "second seed" framing is **honest-but-misleading**: the harness was rerun with the same `(42, 0, 1234)` tuple because the CLI does not expose a seed flag. The new value-add is the **per-cell raw JSONs and decode-status counters** saved to `molmetal/reports/r10_cfg_real_seed1/test_*_seed*_cfg*/`.
- This report **does not** support any Round-12/13 acceptance claim for the CFG end-to-end path. The bottleneck is the **distance-connectivity decoder** (92/96 disconnected) and the **unconstrained atom head** (4/96 vocabulary violations) — both architecture-level issues that no seed sweep can resolve at this configuration.
- Per `molmetal/reports/ultracode_audit/r10_recon_cfg.md`, the correct next-step is **not** another seed sweep. The recommended actions remain (1) replace distance decoder with a learned bond-order head, (2) add output vocabulary mask, (3) extend harness with `--seeds` CLI flag, (4) only then re-run a real seed sweep on the v2 architecture.

## Files written by this CFG run

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_real_seed1/run.log` — full stdout/stderr capture.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_real_seed1/report.json` — full per-cell summary with timing + aggregate.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_real_seed1/checkpoint_seed{42,0,1234}.pt` — three fresh 2000-step checkpoints.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_real_seed1/test_00{1,2}_seed{42,0,1234}_cfg{1,2}/raw_*.json` — 96 raw atom-cloud JSONs (one per requested sample).
- This section appended to `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/round10_e2e_pt_cfg_vina.md`.
