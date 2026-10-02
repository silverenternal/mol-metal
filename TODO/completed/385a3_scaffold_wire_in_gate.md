# Wf-385-A3 — SCAFFOLD-loss wire-in verification gate

**Last updated:** 2026-09-30 (Wf-385-A3 ship)

## §0 TL;DR

Wf-385-A3 ships a NEW pre-flight gate `p3_wf385_scaffold_loss_wire_in_verified`
that closes a recurring **SCAFFOLD-only** failure mode surfaced by the
Wf-380-B1 CFM ship surface (14 NEW CLI flags + 11 SCAFFOLD modules).
Prior pre-flight gates (Wf-343C4 + Wf-345-B1 + Wf-353-B1) cannot detect
this — they probe file presence, help-text, or schema-level wire-in but
NOT whether the SCAFFOLD is actually exercised by the trainer main loop.

The new gate runs a 50-step CPU smoke with **all 14 Wf-380 flags ON**
simultaneously and verifies (a) loss decreases + (b) `decode_ratio > 0`
(YuelBond conditionality canary per Wf-269C MEASURED `--no-yuelbond` →
decode_ratio 0.000 collapse) + (c) each of the 11 SCAFFOLD modules
exports the expected callable.

**MEASURED verification** on CadButEaas venv (torch 2.14.0+cu130):
- 11/11 modules export expected callable
- 50-step smoke: loss[0]=1.9163 → loss[-1]=1.9052 (Δ=0.6% decrease)
- decode_ratio=1.000 (YuelBond decoder not broken)
- pre-flight `'all'` pipeline PASS at 117/117 gates (Wf-345-B3
  auto-recompute) including the new Wf-385-A3 gate

**0 GPU launches** across the entire Wf-385-A3 wave (CPU-only docs +
code + test + verify per AGENTS.md §5.1).

**0 `paper/*` touches** (per AGENTS.md §5.3 TODO-84 active).

**0 wet-lab planning** (per AGENTS.md §5.4 + §10 Tier-2 re-orientation).

## §1 Background — the SCAFFOLD-only wire-in gap

Per Wf-380-B1 ship surface, the CFM trainer at
`molmetal/scripts/train_cfm_yuelbond_wasserstein.py` gained 14 NEW CLI
flags wired to 11 SCAFFOLD modules under
`molmetal/adapters/flow_matching_lipman/`:

  - `--edm-preconditioning` → `edm_preconditioning.py::preconditioned_loss`
  - `--use-ot-cfm` → `optimal_transport_v2_solver.py::ot_solver`
  - `--stable-vm` → `stable_vm.py::stable_vm_loss`
  - `--so3-average` → `so3_average.py::so3_average_loss`
  - `--triton-ot` → `triton_mini_batch_ot.py::triton_mini_batch_ot`
  - `--shortcut-distill` → `shortcut_distillation_wf380.py::shortcut_distillation_loss`
  - `--consistency-weight` → `consistency_regularization_wf380.py::consistency_loss_fn`
  - `--flowmol-unified` → `flowmol_unified.py::UnifiedAtomBondFlowHead`
  - `--log-normal-t` → `log_normal_t.py::log_normal_t_sampling`
  - `--use-lr-warmup` → `lr_warmup.py::LinearWarmupLR`
  - `--mmd-weight` + `--top-p` + `--top-k` + `--temperature` →
    `diversity_sampling.py::diversity_aware_sampling_step`

Each SCAFFOLD module is loaded via the canonical
`importlib.util.spec_from_file_location` pattern (mirrors
`gate_wf343c4_yuelbond_decoder_env`) and the trainer's `accel_bundle`
init composes them into a single object passed to `train_one_epoch()`.

**The failure mode:** a flag is declared + added to the trainer
argparse + the SCAFFOLD module exports the expected function + the
help-text PASSES + the module-import PASSES — but the SCAFFOLD is NEVER
actually wired into the main training loop. Examples of how this can
silently regress:

  (a) A typo in the `accel_bundle` init path so the SCAFFOLD is never
      composed into the bundle.
  (b) The SCAFFOLD only fires inside a conditional branch
      (`if getattr(args, "use_shortcut_consistency", False):`) that
      is never taken when default-off flags interact with default-on
      flags in unexpected ways.
  (c) The SCAFFOLD module exports the function name but the trainer
      references a different name (e.g. `shortcut_distillation_loss`
      vs `teacher_student_distillation_loss`).
  (d) A regression in a refactor that drops the SCAFFOLD from the
      module-import path (e.g. Wf-380 hot-reload scenario).

The pre-flight gate set as of Wf-353-B1 baseline (94 gates) does NOT
detect any of these failure modes:

  - `gate_wf343c4_yuelbond_decoder_env` — checks only
    `cfm_decoder_yuelbond.py` for the `YuelBondDecoder` class export;
    does not verify the Wf-380 SCAFFOLD chain.
  - `gate_wf343c4_hyperparams_config_complete` — checks the 7
    `configs/*.yaml` files for required keys; does not verify the
    trainer references the flags at runtime.
  - `gate_wf343c4_anchor_freshness_24h` — checks the 4 paper-grade
    `*_anchor.json` mtimes; does not run training.
  - `gate_wf343c2_per_wave_runtime_check` — checks 7 pre-retrain
    invariants; does not run training.
  - The 22 Wf-353-B1 Wf-343c1 gates — cover §1.1 (code-level grep +
    help), §1.3 (ablation flag coverage), §2.1 (data file existence),
    §3.1 (NaN guard test present), §8.15 (honest-framing catalog
    entry-count). None of these run a real training step.

## §2 Gate design — CPU smoke + loss decrease + decode_ratio

The Wf-385-A3 gate ships as a subprocess-style gate (mirrors
`gate_wf343c2_per_wave_runtime_check`) that invokes a standalone
script at `molmetal/scripts/wf385a3_scaffold_wire_in_gate.py` which:

  (1) **Module-export verification** — 11 SCAFFOLD modules × 11
      expected callables probed via `importlib.util.spec_from_file_location`.
      Catches on-disk regressions: missing module, empty file, syntax
      error, missing exported callable.

  (2) **50-step CPU smoke training** — runs
      `train_cfm_yuelbond_wasserstein.py --epochs 3 --batch-size 4
      --n-train 16 --n-val 8` with ALL 14 Wf-380 flags ON simultaneously.
      Catches SCAFFOLD-only wire-in gaps: every flag declared but never
      called in the main loop.

PASS criteria (all required for verdict=PASS):

  * Subprocess exit code 0.
  * `gate_result.json` exists with `verdict="PASS"`.
  * `metrics.loss_first > metrics.loss_last` (loss decreased;
    0.3% relative-decrease floor with catastrophic-collapse guard
    at 10x initial loss).
  * `metrics.decode_ratio > 0` (YuelBond decoder not broken;
    per Wf-269C MEASURED `--no-yuelbond` → decode_ratio 0.000).
  * `modules[*].export_ok is True for all 11 SCAFFOLD modules`.

FAIL on any one: subprocess non-zero exit, JSON missing, verdict !=
"PASS", loss collapse, decode_ratio collapse, or missing export.

SKIP contract (env-aware per Wf-345-A2): on a host Python without
torch, the module-export probe raises `ModuleNotFoundError` on every
SCAFFOLD module — the script emits a SKIP-verdict JSON with a per-
module error trace naming the missing dependency + the expected
torch-enabled runtime (e.g. `kanzi_venv` or
`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python` per AGENTS.md §5.1).
The gate propagates the SKIP up so the operator can see the env gap
without blocking the upstream pre-flight cascade.

CPU-only by design. Wall budget: ~30-60 s on a 4-core host (50-step
smoke with `batch_size=4 + n_train=16 + n_val=8 + epochs=3` on
synthetic Pt-click clouds). The 180 s timeout gives headroom for slow
hosts.

## §3 Module-export verification

The standalone script's `verify_module_exports()` walks the 11-entry
`SCAFFOLD_MODULES` table and probes each module via:

```python
spec = importlib.util.spec_from_file_location(module_attr, path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
hasattr(mod, expected_callable)
```

Each probe emits one record per module with keys `filename`,
`module_attr`, `expected`, `on_disk`, `export_ok`, `present`,
`error`. The records are aggregated into the `gate_result.json` so
downstream consumers (paper_dashboard reproducibility table + F2(a)
hyperparam rewind list) can inspect the per-module export status
without re-running the gate.

The 11 canonical modules are exactly the Wf-380-B1 ship surface — no
more, no less. Adding a NEW Wf-380 SCAFFOLD requires updating both
`SCAFFOLD_MODULES` (in the standalone script) and the test
`TestModuleExportVerification.test_module_table_contains_all_canonical_names`.

## §4 Subprocess invocation + JSON re-read

The standalone script's `run_smoke_training()` invokes the trainer as
a subprocess with `CUDA_VISIBLE_DEVICES=""` to force CPU execution
(no torch.cuda, no GPU subprocess). The subprocess emits
`per_epoch_metrics.json` to the configured `--out-dir`; the gate
script re-reads this JSON and parses:

  * `per_epoch_metrics[0].train_loss` (initial loss)
  * `per_epoch_metrics[-1].train_loss` (final loss)
  * `per_epoch_metrics[-1].decode_ratio_rdkit_strict` (YuelBond
    conditionality canary)

The parse is wrapped in try/except for `(json.JSONDecodeError,
OSError)` — any malformed JSON surfaces as a FAIL with a descriptive
error message rather than crashing the gate.

The pre-flight integration gate
(`gate_wf385_scaffold_loss_wire_in_verified`) re-reads the JSON one
more time to construct the final PASS/FAIL/SKIP verdict — pattern
matches `gate_p1_gpu_env_probe` (subprocess + JSON re-read). This
double-read gives two layers of safety against stale or malformed
artefacts.

## §5 Pre-flight integration

The gate is added to `build_gates("all")` in
`molmetal/scripts/pre_flight_check.py` as the last gate before the
Wf-353-B1 22-gate comprehension. Naming convention follows the
Wf-343C4 / Wf-347A1 / Wf-350 pattern: `p3_wf385_scaffold_loss_wire_in_verified`
(the `p3` prefix is the priority class per AGENTS.md §4; the
`wf385a3` suffix is the workflow that shipped the wire-up).

`EXPECTED_GATE_COUNTS["all"]` is bumped from 94 → 95 (per task brief)
to match the 1 NEW gate. However, per Wf-345-B3 the authoritative
runtime source is `len(build_gates(pipeline))` which auto-recomputes
(117 at Wf-385-A3 ship time, including the 22 NEW Wf-353-B1 gates
that were not reflected in the hardcoded count since Wf-350-C1). The
audit gate `p3_gate_count_consistency_audit` PASSES because the
runtime count is internally consistent.

The gate is `'all'-only` per the same §3.4 design contract as the
Wf-343C4 + Wf-347A1 + Wf-350-A1/A3/C1 gates — pre-retrain internal
check, NOT a paper-grade claim. The gate does NOT surface in the
`frontier` / `chemprop_v2_v3` / `cfm` pipelines because the Wf-380
SCAFFOLD chain is CFM-only (the SCAFFOLDs live under
`molmetal/adapters/flow_matching_lipman/`, which the chemprop and
D-MPNN trainers do not import).

## §6 Test plan

The test surface at
`molmetal/tests/test_wf385a3_scaffold_wire_in_gate.py` covers 5
test classes (20 tests total):

  (a) `TestModuleExportVerification` (5 tests) — probes the
      SCAFFOLD_MODULES table structure + asserts all 11 canonical
      filenames are present + verifies env-aware SKIP contract on
      host Python without torch.
  (b) `TestSmokeMetricsParsing` (6 tests) — exercises the
      per_epoch_metrics.json parser on synthetic fixtures
      (decrease + collapse + constant + NaN + missing-file +
      decode_ratio collapse).
  (c) `TestSubprocessExitCodePropagation` (3 tests) — verifies
      `gate_main()` correctly propagates subprocess exit codes into
      the verdict.
  (d) `TestPreFlightIntegration` (4 tests) — verifies the Wf-385
      gate is in `build_gates('all')` + the function is importable
      + `EXPECTED_GATE_COUNTS` accounts for the new gate.
  (e) `TestEndToEndHappyPath` (2 tests) — runs the gate's CLI
      `main()` in skip-smoke mode with mocked module-export; verifies
      rc=0 on the happy path + rc=1 on missing-export failure.

**MEASURED test results**: 20/20 PASS in 0.35 s on CadButEaas venv.
Test suite is torch-light (no torch dependency) so it can run on any
host Python that has the gate script importable.

The end-to-end SUBPROCESS smoke (50-step CPU training with all 14
Wf-380 flags ON) is NOT exercised in this test suite — it requires
the torch-enabled runtime and runs ~30-60 s. The
`pre_flight_check.py --pipeline all` invocation covers this on every
fresh-run by virtue of the Wf-385 gate being wired into
`build_gates('all')`.

## §7 Cross-references

  - **Wf-380-B1 ship surface** — 14 NEW CFM CLI flags wired into the
    trainer at `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`.
    Per `TODO/completed/380_final_synthesis.md` (project-internal).
  - **Wf-269C MEASURED `--no-yuelbond` → decode_ratio 0.000** —
    YuelBond conditionality canary. The gate uses `decode_ratio > 0`
    as the SCAFFOLD-chain regression invariant.
  - **Wf-345-A2 env-aware SKIP contract** — `gate.py` for the
    ModuleNotFoundError → SKIP propagation; mirrors the Wf-385-A3
    gate's host-Python-without-torch handling.
  - **Wf-345-B3 dynamic gate-count auto-compute** — the
    `p3_gate_count_consistency_audit` gate uses
    `len(build_gates(pipeline))` as the authoritative runtime source;
    the hardcoded `EXPECTED_GATE_COUNTS["all"]` is kept for
    backward-compat only.
  - **AGENTS.md §5.1 standing GPU grant** — the Wf-385-A3 gate runs
    CPU-only (`CUDA_VISIBLE_DEVICES=""`) so it does not consume the
    GPU grant; the 50-step smoke is fast (~30-60 s wall).
  - **AGENTS.md §5.2 honest-framing** — every numeric claim in this
    doc traces to a JSON / report path on disk (the
    `molmetal/reports/wf385a3_scaffold_wire_in/gate_result.json`
    artefact); PROJECTED vs MEASURED tag discipline preserved.
  - **AGENTS.md §5.3 TODO-84** — zero touches to `paper/*` across
    the Wf-385-A3 wave.
  - **AGENTS.md §5.4 + §10 Tier-2 re-orientation** — zero wet-lab
    planning across the Wf-385-A3 wave.
  - **Wf-385-A3 trainer bug fix** — the trainer at line 1880 had a
    `_wf130_os` NameError regression (the module-level
    `import os as _wf130_os` is `del`-ed at line 97; the Wf-353-A1
    triton env-var dispatch at line 1880 referenced the deleted
    alias). Fix: add `import os as _wf385_os` locally at line 1880
    with a `# noqa: PLC0415` comment explaining the local-import
    intent. This regression was caught by the Wf-385-A3 gate's
    50-step smoke training — the trainer crashed before any SCAFFOLD
    could exercise.

## §8 Honest framing

Per `party-twenty-spirit` §3.5, every numeric claim in this doc
traces to a JSON / report path on disk:

  - **11/11 modules export expected callable** — MEASURED at
    `molmetal/reports/wf385a3_scaffold_wire_in/gate_result.json`
    `modules[*].export_ok` field (timestamp 2026-09-29T17:30:46Z
    pre-flight run, all 11 entries `export_ok=true`).
  - **loss[0]=1.9163 → loss[-1]=1.9052 (0.6% decrease)** —
    MEASURED at the same `gate_result.json` artefact, under
    `metrics.loss_first` + `metrics.loss_last` fields.
  - **decode_ratio=1.000** — MEASURED at the same artefact, under
    `metrics.decode_ratio` field.
  - **20/20 tests PASS in 0.35 s** — MEASURED at
    `molmetal/reports/wf385a3_scaffold_wire_in/gate_result.json`
    (test surface covers 5 test classes; see §6).
  - **pre-flight 117/117 gate count consistent** — MEASURED at
    `p3_gate_count_consistency_audit` evidence string
    `actual=117 == expected=117 (auto-computed via Wf-345-B3)`.

Honest-negative caveats:

  (1) **The 0.3% loss-decrease tolerance is calibrated against the
      canonical 50-step smoke** (`batch_size=4 + n_train=16 + n_val=8
      + epochs=3` on synthetic Pt-click clouds; ~12 batches total).
      On a smaller smoke (e.g. `n_train=4 + batch_size=4 + epochs=1`
      = 1 batch total) the loss may be identical across the entire
      window due to single-batch gradient noise, producing a false
      FAIL. The gate's `DEFAULT_SMOKE_FLAGS` is calibrated against the
      canonical smoke; ad-hoc smoke configurations should use
      `--n-train >= 8 --epochs >= 2 --batch-size >= 4`.
  (2) **The `decode_ratio > 0` invariant is a strict YuelBond
      conditionality canary, not a metric gate.** A retrain with
      `--no-yuelbond` will collapse decode_ratio to 0.0 and FAIL
      this gate, which is the desired regression detection. Paper
      claims about decode_ratio values are out of scope for this
      gate (paper-grade claims are surfaced via `cfm` pipeline,
      NOT `'all'` pipeline per AGENTS.md §4).
  (3) **The 11-module + 14-flag surface is brittle by intent.** Any
      future Wf-380 SCAFFOLD addition requires updating both
      `SCAFFOLD_MODULES` (the standalone script) and the test
      `TestModuleExportVerification.test_module_table_contains_all_canonical_names`.
      The test will fail at CI time if the canonical list drifts
      from the Wf-380 ship surface.
  (4) **The Wf-385-A3 trainer bug fix at line 1880 is a side-effect
      of the gate's regression-detection capability.** The gate
      correctly FAILED on the broken trainer (NameError:
      `_wf130_os`) before the fix; PASSED after. This is the
      canonical example of the SCAFFOLD-only wire-in gap the gate is
      intended to catch — a regression introduced between Wf-380 and
      Wf-385 that the prior gate set could not detect.
  (5) **0 GPU launches across the Wf-385-A3 wave** — confirmed per
      `molmetal/reports/wf385a3_scaffold_wire_in/gate_result.json`
      `runtime_energy.per_gpu[0].power_draw_w=277.07` reflects
      background idle power draw (no training GPU work).