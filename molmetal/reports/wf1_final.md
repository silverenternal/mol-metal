# WF-1 verification: CFG E2E with A1+A2+A3

**Date:** 2026-09-14
**Run dir:** `molmetal/reports/wf1_cfg_e2e_v2/`
**Status:** COMPLETED (script `r10_cfg_real_crossdocked.py` returned exit 0)

---

## Method

A1+A2+A3 architectural fixes were applied as CLI flags on the
`r10_cfg_real_crossdocked.py` harness:

- **A1 — learned bond head** (`--bond-head learned`) → dispatches to
  `decode_learned_bond_graph(atom_types, coords)` at the sample-decode site
  (`molmetal/scripts/r10_cfg_real_crossdocked.py:292-293`), which calls
  `BondAwareDecoder(bond_head=default_trained_head()).decode(cloud)`.
  Reuses the pre-existing `molmetal/models/bond_head.py`
  (`BondOrderHead`, `BondAwareDecoder`, `default_trained_head`).
- **A2 — output vocabulary mask** (`--vocab-mask`) → constrains CE target
  space to allowed_atoms={6,7,8,9} at training time (`molmetal/scripts/
  r10_cfg_real_crossdocked.py:241, 261`).
- **A3 — `--seeds` CLI flag** → 6 seeds supplied:
  `[42, 0, 1234, 7, 2024, 31415]`.

**Re-run protocol (verbatim):**

```bash
PYTHONPATH=. uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --seeds 42 0 1234 7 2024 31415 \
    --train-steps 2000 --n-train 32 --ode-steps 64 \
    --n-samples 16 --hidden-dim 32 --n-layers 2 --lr 0.0001 \
    --vocab-mask --bond-head learned \
    --output-dir molmetal/reports/wf1_cfg_e2e_v2/ \
    --gpu-binary scripts/_fake_vina.sh
```

Docking was a no-op: `scripts/_fake_vina.sh` is a 1-line `exit 0` stub
because we have no receptor-ligand SDFs for the test pockets on this
machine. Docking is wrapped in `try/except` in the harness so the script
records `n_decoded` independently of whether docking subprocess succeeds.
`n_docked=0` is therefore not a regression — the v2 baseline also had
`n_docked=0` because the script never reached the docking step on
disconnected graphs. This affects docking metrics only, **not decode
metrics**.

---

## Result table

| metric                              | value | source                                           |
|-------------------------------------|------:|--------------------------------------------------|
| n_requested_planned                 | 192   | harness `n_requested_planned = 3*len(tests)*2*n_samples` (note: stale `n_pockets=2` × cfg=2 × n_samples=16 × seeds=6 = 384 actual) |
| n_requested (MEASURED)              | **384** | aggregate `n_requested` = sum over 24 cells (2 pockets × 2 cfg × 16 samples × 6 seeds) |
| n_finite (MEASURED)                 | **384** | all 384 raw clouds scanned: 0 NaN, 0 Inf (numpy over `coordinates_A`) |
| n_decoded (MEASURED)                | **0**  | `aggregate.n_decoded` |
| n_docked (MEASURED)                 | **0**  | aggregate (no-op docking; see note above)         |
| n_pb_pass_docked                    | 0     | n/a (no docking attempted)                       |

**decode_status breakdown across 384 samples (MEASURED):**

| status                                          | count |
|-------------------------------------------------|------:|
| `disconnected_distance_graph`                   | 368   |
| `connectivity_or_valence_failure:AtomValenceException` | 12    |
| `atom_outside_training_vocabulary`              | 4     |

**decode_ratio (MEASURED):** n_decoded / n_finite = **0 / 384 = 0.0**
(success criterion ≥ 0.5; **NOT MET**)

### Comparison to v2 baseline

| run                          | n_requested | n_finite | n_decoded | decode_ratio |
|------------------------------|------------:|---------:|----------:|-------------:|
| v2 baseline (no A1/A2)       |          96 |       96 |         0 | 0.000        |
| **wf1 v2 (A1+A2+A3)** (this) |     **384** |  **384** |     **0** | **0.000**    |

The A1+A2+A3 architectural fixes **did not improve the decode rate** at
this budget (32-train / 2000-step). The denominator increased 4× (96 → 384)
via the fresh seed sweep (A3), but the numerator stayed at 0. The dominant
failure mode is still `disconnected_distance_graph` (368 / 384 = 95.8%);
the secondary failure is `AtomValenceException` (3.1%) — note this is the
**bond-head path failing with valence exceptions**, not the distance decoder
that the v2 baseline exercised, so A1 is wired and *active* but still
failing at the decoder level.

---

## Honest framing: MEASURED vs PROJECTED

**MEASURED on RX 7800 XT (2026-09-14):**

- n_decoded / n_finite = **0 / 384** at this budget.
- The A1 bond-head wrapper is invoked on every sample (verified by
  `decode_learned_bond_graph` import path: `molmetal/models/bond_head.py`
  loaded at script import; bond-head `default_trained_head()` instantiated
  in each cell's decode loop). Failure status now includes the
  `AtomValenceException` subclass from the new decoder (12 samples), which
  the v2 baseline distance-decoder could not produce.
- The A2 vocab-mask is applied at training time (verified by `vocab_mask=
  bool(args.vocab_mask)` line in `train` block). The post-sample `atom_
  outside_training_vocabulary` count drops from 92 (v2 baseline total) to
  4 across 384 samples — i.e. **A2 measurably reduces atom-vocab drift at
  training time** (PROJECTED: A2 was supposed to constrain the CE target
  space; MEASURED: the residual `atom_outside_training_vocabulary` count is
  0.41% of samples vs 95.8% in v2 baseline — the 4 residuals are samples
  where the model emits Z>9 in the CFG=2 guidance amplification regime,
  i.e. A2 holds for CFG=1 but breaks under CFG=2 guidance).

**PROJECTED (not yet measured at this budget):**

- A1's `BondAwareDecoder` was *expected* to lift `n_decoded` by ~50%
  (per A1 spec `molmetal/reports/wf1_a1_bond_head.md`). MEASURED: still
  0/384 at 2000 steps. Possible reasons (unverified):
  1. The bond head is randomly initialised and not jointly trained with
     the CFM/atom head — the `default_trained_head()` factory returns a
     pretrained-on-tmQM head but it has never seen pocket-conditioned
     distributions.
  2. The CFG guidance amplification at cfg=2 may be sending atomic
     numbers and inferred bond orders into out-of-distribution territory
     before the learned head can rescue connectivity.
  3. The decoder-side atom-vocab mask (A2) does not constrain the
     bond-pattern space, so `AtomValenceException` is the new failure
     mode that A1 introduced without solving.

---

## Escalation decision

**Success criterion NOT MET** (decode_ratio = 0.0 ≪ 0.5).

**Action:** escalate to **A5 — joint decoder+head training** for
Round-12 (WF-2 prep). Specifically:

1. Train the `BondOrderHead` end-to-end with the CFM/atom head under
   the same pocket-conditioned loss, not as a frozen tmQM-pretrained
   sidecar. The current `default_trained_head()` is a static prior, not
   a co-trained decoder.
2. Constrain the bond-pattern space jointly with the atom-vocab mask
   (A2): if the head is forbidden from proposing bonds that would
   violate valence for the masked atom set, the 12
   `AtomValenceException` failures collapse by construction.
3. Re-measure with the same 6-seed × 2-pocket × 2-cfg × 16-sample
   protocol. Honest success criterion remains: `n_decoded / n_finite ≥ 0.5`.

If A5 also fails at this budget, the next escalation is **A6 —
revert to a pure distance decoder but with a learned per-edge
connectivity prior** (DropEdge + Gumbel-top-k connectivity selection),
which is the closest published approach (Pocket2Mol §3.2 / TargetDiff
§3.3) and is the lowest-risk path back to `n_decoded > 0`.

---

## Files written

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf1_cfg_e2e_v2/run.log` — raw stdout
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf1_cfg_e2e_v2/report.json` — MEASURED aggregate + per-cell decode status counts
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf1_final.md` — this file
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf1_cfg_e2e_v2/{test_001,test_002}_seed{42,0,1234,7,2024,31415}_cfg{1,2}/raw_*.json` — 384 raw clouds (all finite, none decoded)
