# Lambda × CFM coupling — Integration test log + benchmark summary

**Workflow:** Lambda × CFM coupling audit (TODO-21 follow-up, R17 ultracode dispatch)
**Phase:** R17 ultracode dispatch — extended integration tests + benchmark
**Date:** 2026-09-26
**Owner:** R17 ultracode dispatch
**Inputs:** `molmetal/molmetal_lam/lam_chem/coupling_joint_inference.py` (shipped),
`molmetal/tests/test_lambda_cfm_joint_inference.py` (25 unit tests, shipped),
`molmetal/scripts/verify_lambda_cfm_coupling.py` (smoke, shipped),
`molmetal/data/platinai_5k.csv` (5000 samples, shipped).

---

## §1 — Files added

| File | LOC | Purpose |
|---|---:|---|
| `molmetal/tests/test_lambda_cfm_integration.py` | ~ 470 | 15 integration tests — wires the joint pipeline to LambdaASTSR (Strategy 1), `coupling_adapter.CouplingAdapter` (9-d → 64-d legacy), Deflexpressor R², real PlatinAI 5K CSV, real PlatinAI 5K checkpoint, CascadedRefiner (Strategy 3). CPU-only. |
| `molmetal/scripts/benchmark_lambda_cfm_coupling.py` | ~ 165 | CPU benchmark on `--n-samples` PlatinAI SMILES; writes `benchmark.json` (mean ± std wall-time, peak memory, mean n_atoms, n_decoded_smiles, bridge_source, alpha, n_steps, pocket_size, timestamp_utc). |
| `molmetal/reports/wf_lambda_cfm_coupling_audit/integration_test_log.md` | this file | Audit log + honest framing. |

---

## §2 — Integration test inventory (15 tests)

The 15 integration tests are listed below with the honest framing of
what each test *proves* and what it does **not** prove.  All tests are
CPU-only and run on a single core.  The suite completes in < 60 s on
the project's reference host.

| # | Test | Asserts | Module wired |
|---:|---|---|---|
| 1 | `test_integration_with_lambda_astsr` | Pipeline tags outputs with `use_lambda_astsr`; the env-var gate is honored; the flag is metadata-only (numerical output identical on/off) | Strategy 1 (LambdaASTSR) |
| 2 | `test_integration_with_lambda_astsr_channel_returns_finite` | `make_lambda_astsr_reward_channel(astsr=...)` returns a finite float on a 5-column measured-reward row | Strategy 1 |
| 3 | `test_integration_with_coupling_adapter` | `CouplingAdapter.embed_features` is dimension-compatible with the 32-d → 64-d PlatinAI 5K bridge (both live in the 64-d space) | Legacy adapter |
| 4 | `test_integration_coupling_adapter_and_joint_pipeline_share_embed_dim` | `EMBED_DIM == 64` in both adapters (the joint pipeline contract) | Legacy adapter |
| 5 | `test_integration_with_deflexpressor` | `LambdaASTSR.fit + predict` is finite and bounded (R² > −10, guards against catastrophic collapse) on a linear synthetic target | Deflexpressor (R17 L-R17-B3) |
| 6 | `test_integration_deflexpressor_channel_after_pipeline` | ASTSR reward channel accepts the 5-column measured-reward row dict (not the 32-d lambda features) | Deflexpressor wiring |
| 7 | `test_full_pipeline_smoke` | `smoke_test_full_pipeline` returns `True` in < 30 s on the real PlatinAI 5K checkpoint | End-to-end CPU smoke |
| 8 | `test_pipeline_deterministic` | Two sequential `run_full_pipeline` calls return bit-for-bit identical output (λ-features, λ-embed, CFM, z0, decoded SMILES) | Determinism |
| 9 | `test_pipeline_batch` | `embed_lambda_features(X)` for `X.shape == (N, 32)` returns `(N, 64)`; each row is independently L2-normalised | Batching |
| 10 | `test_pipeline_real_platinai_csv` | Full pipeline on the first 5 real PlatinAI 5K SMILES produces finite, L2-normalised output with non-empty decoded SMILES | Real CSV |
| 11 | `test_pipeline_with_real_pretrain_checkpoint` | The canonical PlatinAI 5K checkpoint is loaded; `bridge.embed_raw(32-d)` returns finite, L2-normalised 64-d vector | Real checkpoint |
| 12 | `test_pipeline_handles_metallo_smiles` | Pt/Pd/Au metallo SMILES produce finite pipeline output (the `metal_indicator` slot is exercised) | Metallo SMILES |
| 13 | `test_pipeline_top_k_decode_consistent` | `top_k=1` returns the closest tile; `top_k=3` has `top_k=1` as its prefix | Decode |
| 14 | `test_cascaded_refiner_after_pipeline` | `CascadedRefiner.refine(lambda_embed, cfm_output)` produces a finite, L2-normalised 64-d output | Strategy 3 (cascade) |
| 15 | `test_pipeline_memory_bounded` | 50 sequential `run_full_pipeline` calls stay within a 50 MiB peak delta (tracemalloc) — guards against a numpy/torch leak | Memory |
| 16 | `test_pipeline_smiles_round_trip_preserves_features` | `lambda_features_from_smiles(smi)` is bit-for-bit idempotent across calls | Determinism |
| 17 | `test_pipeline_writes_summary_json` | The pipeline output dict serialises to JSON (numpy scalars coerced) | JSON export |
| 18 | `test_pipeline_alpha_extremes` | `alpha=0` reduces the CFM target to pocket; `alpha=1` reduces it to the bridge-embedded lambda vector; the two endpoints are distinct | Alpha convex-weight contract |

(Note: the brief asked for "15 tests"; the table enumerates the 18
distinct test functions shipped.  Tests 1–15 are the headline list
written to the test file; tests 16–18 are three additional
defensive checks added during the build to nail the determinism +
JSON export + alpha contract.  The actual count in
`test_lambda_cfm_integration.py` is 18 — all in scope of the
"extended smoke + integration tests" deliverable.)

---

## §3 — Verify gates (commands + expected exit codes)

```
# Run the 15 (18) integration tests.
uv run --no-sync python -m pytest molmetal/tests/test_lambda_cfm_integration.py -v 2>&1 | tail -30

# Run the CPU benchmark on the first 100 PlatinAI 5K samples.
uv run --no-sync python molmetal/scripts/benchmark_lambda_cfm_coupling.py --n-samples 100

# The 25-shipped unit tests still pass.
uv run --no-sync python -m pytest molmetal/tests/test_lambda_cfm_joint_inference.py -v 2>&1 | tail -10

# The shipped smoke verify still returns 0.
uv run --no-sync python molmetal/scripts/verify_lambda_cfm_coupling.py --smoke --n-iter 100
```

All four commands exit 0 on the project's reference host.

---

## §4 — Benchmark JSON schema (output)

`benchmark.json` is written to
`molmetal/reports/wf_lambda_cfm_coupling_audit/benchmark.json`:

```json
{
  "n_samples": 100,
  "mean_wall_time_s": 0.0123,
  "std_wall_time_s":  0.0014,
  "peak_memory_mb":   3.21,
  "mean_n_atoms":     32.0,
  "n_decoded_smiles": 100,
  "bridge_source":    "stub | <path to .npz>",
  "alpha":            0.5,
  "n_steps":          50,
  "pocket_size":      8,
  "timestamp_utc":    "2026-09-26T..."
}
```

* `mean_wall_time_s` is the per-sample wall-clock (mean over `n_samples`).
* `peak_memory_mb` is the Python `tracemalloc` peak delta across the run.
* `mean_n_atoms` is the average size of the 32-d λ-feature vector (the
  proxy "atom count" in this CPU skeleton — there are no 3-D coordinates).
* `bridge_source` is `"stub"` when no PlatinAI 5K checkpoint is on disk,
  else the absolute path of the loaded `.npz`.

---

## §5 — Honest framing — what this **does** and does **not** prove

**This audit proves:**

1. The Lambda × CFM joint inference skeleton (CPU only) is end-to-end
   **wired and verifiable** on real PlatinAI 5K data — 18 integration
   tests pass and the benchmark emits a JSON summary in < 5 s.
2. The 32-d → 64-d PlatinAI 5K bridge and the 9-d → 64-d legacy
   coupling adapter share `EMBED_DIM = 64` (the joint pipeline contract
   is dimension-compatible across adapters).
3. The LambdaASTSR Deflexpressor reward channel (Strategy 1) is
   composable with the joint pipeline at the row-dict level (the 5-d
   measured-reward row dict, *not* the 32-d lambda features).
4. The CascadedRefiner (Strategy 3) composes correctly on the
   pipeline output: bridge-embedded lambda → CFM-ODF → refine →
   64-d L2-normalised output.
5. The pipeline is **bit-for-bit deterministic** and **memory-bounded**
   across many sequential runs (no leak).

**This audit does NOT prove:**

1. **Real end-to-end GPU retraining** of the Lambda × CFM coupling.
   The "CFM ODE" integrated here is a **closed-form analytic
   linear-path CFM** (Lipman et al. 2023, eq. 13), not a learned
   equivariant velocity field.  The bridge is an embedding, not a
   velocity field.  Real joint training is deferred to the
   `wf_cfm_gpu_retrain` workflow.
2. **3-D atom-count / structure recovery**.  The benchmark's
   `mean_n_atoms = 32.0` is the λ-feature dimension, **not** the
   number of 3-D atoms emitted by the skeleton (the skeleton does
   not emit 3-D coordinates).
3. **Real PlatinAI bioactivity prediction**.  The PlatinAI 5K CSV
   has NaN placeholders for IC50 / selectivity; the bench measures
   integration throughput, not bioactivity.
4. **LambdaASTSR beating F5 in production.** The R² integration test
   only guards against catastrophic collapse (R² > −10).  The real
   gate lives in `tests/test_lambda_astsr.py::test_30_cell_holdout_*`.

---

## §6 — Deferred to post-GPU-retrain

* **GPU joint training of the bridge + CFM velocity field** as one
  end-to-end module (TODO-21 Strategy 2).  Blocked by
  `wf_cfm_gpu_retrain` — see
  `molmetal/reports/wf_cfm_gpu_retrain/final.md`.
* **Decoder rework to consume the joint CFM output.**  Today the
  decoder uses the 9-d → 64-d legacy `CouplingAdapter`; the
  post-GPU-retrain rebuild will consume the 32-d → 64-d bridge
  output directly.  See
  `molmetal/reports/wf_cfm_path_b_decoder_rework/final.md`.
* **3-D coordinate emission** from the joint pipeline (currently
  the skeleton emits only the 32-d λ-feature + 64-d bridge + 64-d
  CFM terminal + tile-SMILES decode).

---

## §7 — How to reproduce

```
# 1. Run the 18 integration tests (CPU only, ~ 30-60 s).
uv run --no-sync python -m pytest molmetal/tests/test_lambda_cfm_integration.py -v

# 2. Run the benchmark (CPU only, ~ 5 s for 100 samples).
uv run --no-sync python molmetal/scripts/benchmark_lambda_cfm_coupling.py --n-samples 100

# 3. Re-run the shipped 25 unit tests for regression coverage.
uv run --no-sync python -m pytest molmetal/tests/test_lambda_cfm_joint_inference.py -v

# 4. Run the shipped verify smoke for a 30-s end-to-end gate.
uv run --no-sync python molmetal/scripts/verify_lambda_cfm_coupling.py --smoke
```

All four commands exit 0 on the reference host.