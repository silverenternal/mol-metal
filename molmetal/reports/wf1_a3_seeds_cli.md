# WF-1 A3 — `--seeds` CLI flag for the CFG E2E harness

**Generated:** 2026-09-14
**Status:** MEASURED (CLI plumbing) + MEASURED (smoke run, 1 seed / 4 samples / 200 train-steps)
**Scope:** Extend `molmetal/scripts/r10_cfg_real_crossdocked.py` with `--seeds`, `--vocab-mask`, and `--bond-head` CLI flags so seed sweeps and the A1/A2 architectural fixes are CLI-driven and no longer require source edits. No edits to the existing decoder / harness internals beyond the new CLI surface, the new `decode_learned_bond_graph` wrapper, and the decoder-dispatch line at the sample-decode site.

---

## CLI surface diff

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--seeds` | `int` nargs="+" | `[42, 0, 1234]` | Was hardcoded at lines 167 (protocol payload) and 190 (loop iterator). Now wired into `report['protocol']['seeds']` via `list(args.seeds)` and into `for seed in args.seeds` at the loop. |
| `--vocab-mask` / `--no-vocab-mask` | `BooleanOptionalAction` | `True` | Wired into `LipmanFlowMatchingAdapter(vocab_mask=bool(args.vocab_mask))`. Default `True` matches the A2 round-10 spec; `--no-vocab-mask` recovers the legacy full-support sampling softmax. |
| `--bond-head` | `choice ["distance", "learned"]` | `"distance"` | Wired into the per-sample decode call: `args.bond_head == "learned"` calls `decode_learned_bond_graph(...)`, else `decode_distance_graph(...)`. The legacy "distance" path is bit-exact unchanged (default). |

`--help` MEASURED output (top of file shows new flags):

```
--seeds SEEDS [SEEDS ...]   Seed sweep; one checkpoint per seed. Default [42, 0, 1234].
--vocab-mask, --no-vocab-mask  Restrict the sampling atom-head softmax to {1,6,7,8,9,15,16,17,34,35,53,78} (default True).
--bond-head {distance,learned}  Decoder: distance connectivity (default, legacy) or learned bond-order head (A1).
```

`report['protocol']` carries the wired values into the JSON aggregate (`seeds`, `vocab_mask`, `decoder`) so downstream consumers can introspect which configuration was actually executed.

## New code path — `decode_learned_bond_graph`

`molmetal/scripts/r10_cfg_real_crossdocked.py` now exports a second decoder wrapper `decode_learned_bond_graph(atom_types, coords, allowed_atoms=(6,7,8,9)) -> (mol, status_str)` that:

1. Reuses the same vocabulary / finite-coords early-rejection path as `decode_distance_graph` (so `atom_outside_training_vocabulary`, `nonfinite_coordinates`, `disconnected_distance_graph`, `radical_graph`, `connectivity_or_valence_failure:<ExcName>` are reused — the cell-level `decode_status_counts` schema is unchanged).
2. Lazily imports `molmetal.models.bond_head.{AtomCloud, BondAwareDecoder, BondOrderHead, default_trained_head}` so the harness stays import-cheap when `--bond-head=distance` (the default).
3. Builds a fresh `default_trained_head(n_epochs=0, seed=0)` per sample (the synthetic-trained checkpoint used by the A1 tests; full tmQM re-train would slot in here).
4. Runs `BondAwareDecoder.decode(cloud)`, then re-validates fragment count / radical count / sanitisation / atom-count match before returning `mol, "decoded_learned_bond_graph"`.

Import failure / init failure → categorical status `learned_decoder_unavailable:<ExcName>` or `learned_decoder_init_failure:<ExcName>` so the harness can count them like any other decode failure.

## Smoke test — MEASURED

**Command (exact, from the task spec):**

```
PYTHONPATH=. uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --seeds 42 --n-samples 4 --train-steps 200 \
    --vocab-mask --bond-head learned \
    --output-dir /tmp/wf1_smoke --gpu-binary /tmp/fake_vina_binary
```

(Added `PYTHONPATH=.` because the harness is invoked from the project root and `molmetal` is a top-level package; `--gpu-binary` is required by argparse even though the smoke never reaches docking — the per-cell try/except in the harness catches docking failures and lets the report complete.)

**Observed stdout (last 8 lines, MEASURED):**

```
trained real-data seed=42, checkpoint=f614db0d0441
test_001 seed=42 CFG=1.0 decoded=0/4 completed
test_001 seed=42 CFG=2.0 decoded=0/4 completed
test_002 seed=42 CFG=1.0 decoded=0/4 completed
test_002 seed=42 CFG=2.0 decoded=0/4 completed
{"n_requested_planned": 48, "n_requested": 16, "n_raw_generated": 16, "n_decoded": 0, "n_docked": 0, "n_pb_pass_docked": 0}
```

**`report['protocol']` (MEASURED):**

```
vocab_mask: True
seeds: [42]
decoder: learned bond-order head (A1, BondOrderHead + BondAwareDecoder)
```

**`decode_status_counts` per cell (MEASURED, 4 cells × 4 samples each):**

| Pocket | Seed | CFG | counts | n_decoded |
|---|---|---|---:|---:|
| test_001 | 42 | 1.0 | `{'disconnected_distance_graph': 4}` | 0 |
| test_001 | 42 | 2.0 | `{'disconnected_distance_graph': 4}` | 0 |
| test_002 | 42 | 1.0 | `{'disconnected_distance_graph': 4}` | 0 |
| test_002 | 42 | 2.0 | `{'disconnected_distance_graph': 4}` | 0 |

**Smoke metrics:**

```
smoke_exit_code = 0
smoke_n_decoded = 0
```

## Honest framing

**MEASURED (CLI plumbing, this audit):**
- `--seeds`, `--vocab-mask`, `--bond-head` flags are exposed, parsed, wired into the adapter constructor, the protocol payload, and the decoder-dispatch site.
- `--help` output reflects the new surface exactly as specified.
- Smoke run completes `status='completed'` (exit 0) with the protocol payload correctly recording `seeds=[42]`, `vocab_mask=True`, `decoder="learned bond-order head ..."`.

**MEASURED (decode outcome, this audit):**
- `n_decoded = 0` across all 4 cells (16 raw generated samples).
- All 16 samples decode to `disconnected_distance_graph` (i.e. `len(Chem.GetMolFrags(mol)) != 1` after the bond-head emits bonds). No `atom_outside_training_vocabulary`, no `learned_decoder_unavailable:*`, no `learned_decoder_init_failure:*` — the head is wired and runs; the issue is that 19-atom clouds emitted by the v0-velocity-field after 200 train-steps on 4 real-data ligands do not have enough atom pairs inside the bond-head's `[1.0, 2.4]` Å cut that form a single connected component after `Chem.SanitizeMol`.

**Honest interpretation:**
- The A3 CLI extension is **complete and verified**. Seed sweeps no longer require editing the source.
- The A1+A2 architectural fix is **NOT yet sufficient** to recover `n_decoded > 0` at the smoke configuration (1 seed / 4 samples / 200 train-steps / `--bond-head learned --vocab-mask`). The task brief hypothesised that this combination *should* yield `n_decoded > 0` "otherwise A1+A2 are broken"; the smoke result confirms A1+A2 are *not yet* sufficient at the 200-step budget. The predicted root cause is the velocity-field's 19-atom cloud geometry (median pairwise distance 4.4-4.6 Å, far beyond covalent reach — see `wf1_recon_decoder.md` and `model_real_cfg_prior_spatial_repairs_20260913.md`): even with the right vocabulary and a working bond head, the underlying coordinate generator cannot produce atom clouds dense enough for the decoder to recover connectivity. This is the v0 architecture bottleneck that WF-5 (Round-13 100-pocket sweep) and WF-2 (Pt-pocket + real CFM data redesign) are designed to address, not A3 itself.
- A1+A2 *may* recover `n_decoded > 0` at the full v2 32-pair / 2000-step training budget (the existing `r10_cfg_real_crossdocked_v2_train32_2000/` checkpoint) — that is PROJECTED and unverified by this smoke test; the smoke only exercises the small `--train-steps 200` budget specified in the brief.

**No regressions:** the legacy `--bond-head distance` (default) path is untouched; the `decode_distance_graph` function is the same as before with one extra entry-point on its right (the new `decode_learned_bond_graph` wrapper). The protocol payload gains two new keys (`vocab_mask`, `decoder` becomes conditional); downstream consumers that ignore unknown keys are unaffected.

## File paths (this audit's outputs)

- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py` — extended in-place with `--seeds`, `--vocab-mask`, `--bond-head` flags + `decode_learned_bond_graph` wrapper + decoder-dispatch line.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf1_a3_seeds_cli.md` — this file (output).
- `/tmp/wf1_smoke/report.json` — MEASURED smoke run report (4 cells, `status='completed'`, `n_decoded=0`).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py` — pre-existing A1 module (NOT modified by this audit; `BondOrderHead`, `BondAwareDecoder`, `default_trained_head` consumed by `decode_learned_bond_graph`).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_bond_head.py` — pre-existing A1 test suite (NOT modified by this audit).

## Repro recipe (verbatim)

```bash
PYTHONPATH=. uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --seeds 42 --n-samples 4 --train-steps 200 \
    --vocab-mask --bond-head learned \
    --output-dir /tmp/wf1_smoke --gpu-binary /tmp/fake_vina_binary
```

Observed: `EXIT_CODE=0`, `n_decoded=0`, 4 cells each with `decode_status_counts={'disconnected_distance_graph': 4}`. Wall-clock on RX 7800 XT: ~110 s for the full 1-seed smoke (training 200 steps + sampling 16 raw clouds + per-sample decode attempt + per-cell docking-failure catch).
