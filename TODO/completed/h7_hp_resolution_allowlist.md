# H7 closeout — hyperparam resolution allow-list (HOLE 7)

**Wave:** Wf-A13 / HOLE-7 (pre_flight_check.py hyperparam-resolution allow-list)
**Date:** 2026-10-03
**Status:** SHIPPED to working tree (filesystem-verified; `ls -la` evidence in §7). **No commit** (lane rule).
**Owned files:** `molmetal/scripts/pre_flight_check.py`; `molmetal/tests/test_wfa13_hp_resolution_allowlist.py` (NEW); `molmetal/tests/test_wf345a1_hyperparam_aliases.py` (1 stale assertion fixed); this file.
**Did NOT touch:** `scripts/_validate_metrics.py` (user deferred), `.pre-commit-config.yaml` / pre-commit hook (other lane), `paper/*`, wet-lab items, any HOLE-9 file, `TODO/completed/321d_preflight_gate_audit.md` (H5 lane).

## §1 What changed (all inside `pre_flight_check.py`)

1. **NEW `HP_RESOLUTION_EXEMPT_KEYS`** (L4816, 18 exact dotted keys). Each carries a citation naming its non-trainer consumer (`scripts/start.sh::parse_yaml` L147-300 + `TRAINER_CMD` L366 appends real CLI flags): `model.hidden_dim`, `model.n_layers`, `train.batch_size`, `train.lr`, `train.epochs`, `flow.n_steps`, `flow.method`, `data.dataset`, `protocol.epochs`, `protocol.weight_decay`, `protocol.max_norm`, `protocol.scheduler.*` (type/warmup_epochs/min_lr), `output.ckpt_dir`, `output.checkpoint_interval`, `output.resume_from`, `output.events_jsonl_path`.
2. **NEW `HP_RESOLUTION_EXEMPT_PREFIXES`** (L4844, 20 prefixes): `eval.`, `data.split.`, `data.cohort_filter.`, `data.n_rows_expected`, `data.expected_columns_subset`, `data.target_index`, `protocol.full_protocol.`, `protocol.enumerated_runs`, `protocol.n_runs_total`, `protocol.total_runs`, `protocol.wave_decomposition`, `output.per_pocket_dir`, `output.per_seed_dir`, `output.per_config_dir`, `output.per_metal_dir`, `output.wasserstein_decay_dir`, `output.comparison_30vs50`, `output.ablation_report`, `output.orchestrator`, `output.metrics_json` — spec / provenance namespaces, each grep-verified to have NO executable consumer.
3. **NEW `HP_RESOLUTION_NEVER_EXEMPT`** (L4871): `eval.target_decode_ratio` names a REAL CFM trainer dest — stays RESIDUAL so dead wiring can never be exempted.
4. **NEW `_classify_unresolved_key`** (L4874-4895): returns `(bucket, why)`; bucket ∈ reader / spec / residual; prefix match is dot-boundary correct (`p = prefix.rstrip(".")`; `k == p or k.startswith(p + ".")`) so `data.split_extra` is NOT swallowed by `data.split.`.
5. **`_resolve_one_config_via_loader`** (L4897) now returns `(List[{"key","bucket","why"}], Optional[str])`; the `loader` bucket REUSES the canonical loader's own predicate `wfA2_config_loader._registered_non_trainer` (guarded `callable(...)`, so a loader without it degrades to residual, never to exemption).
6. **Gate body** (L5218-5324): per-YAML report; residual>0 → stderr WARNING with the FULL residual list + `loader=/reader=/spec=` counts (evidence shows first 6); residual==0 → INFO. Evidence suffix is verbatim `" | Wf-A13 RESOLUTION REPORT (non-blocking — blocking switch OFF; totals: N loader-exempt + N reader-exempt + N spec-exempt + N RESIDUAL; full residual lists on stderr): …"`. Strict branch (L5298-5310) FAILs ONLY on `n_residual or required_unresolved`, and states the exempt keys "are NOT counted".
7. **Docstring** (L5012) records the MEASURED split: 528 unresolved keys on the 9 active configs = 306 loader-exempt + 27 reader-exempt + 134 spec-exempt + **61 RESIDUAL** (31 distinct residual keys).

## §2 MEASURED residual debt on the healthy tree (2026-10-03, 9 configs)

`cfm_full_retrain` 18 (wrong `entry` = `metallo_drug_smoke_retrain.py` — out of scope, see §5), `cfm_per_innovation_ablation_p1.4` 9, `cfm_real_decode_lift_p1.3` 9, `platinai_pt_4cell_5seed_50ep_v1` 9, `nci60_9cell_5seed_50ep_v1` 8, `default` 3, `wf124_p3_remaining_metals_g4` 2, `wf124_p3_50epoch_full_sweep_g5` 2, `wf124_p3_per_innovation_ablation_g6` 1. Sum 61. Top keys: `output.events_jsonl` ×9 (dead boolean; the trainer's real switch is `--emit-json-events`), `model.task` ×4, `model.atom_vocab_size` ×3, `model.cfm_decoder` ×3, `flow.ot_solver` ×3, `compute.*` (torch_compile/cudnn_benchmark/channels_last/tf32) ×2 configs, `train./protocol.ema_decay` + `edge_cutoff`, `data.num_workers` / `pin_memory`, `protocol.gradient_accumulation_steps`.

## §3 FAIL proof (strict switch CAN block)

`PREFLIGHT_HYPERPARAMS_RESOLUTION_STRICT=1` + fixture `/tmp/hole7/residual.yaml` (1 documented-exempt reader key + 1 loader key + 1 spec key + `train.definitely_not_a_dest`):

```
PROOF1 status: FAIL
PROOF1 evidence: STRICT RESOLUTION MODE (PREFLIGHT_HYPERPARAMS_RESOLUTION_STRICT=1): 1 RESIDUAL key(s) resolve to no trainer argparse dest (documented loader/reader/spec exemptions are NOT counted; 3 exempt key(s) ignored) + 0 REQUIRED-HYPERPARAM unresolved. … | Wf-A13 RESOLUTION REPORT (… totals: 1 loader-exempt + 1 reader-exempt + 1 spec-exempt + 1 RESIDUAL …)
```

The same fixture with the switch OFF → **PASS** (non-blocking default preserved).

## §4 PASS proofs (two)

1. **NEW reachable strict-ON state**: `STRICT=1` + fixture `/tmp/hole7/exempt_only.yaml` (only reader- + spec-exempt keys) → `PASS` with `totals: … + 0 RESIDUAL`. Pre-Wf-A13, strict mode FAILed this fixture on its 3 unresolved exempt keys — i.e. the switch now has a reachable ON state.
2. **Healthy tree**: default mode, 9 live `configs/*.yaml` → `PASS`, evidence `totals: 306 loader-exempt + 27 reader-exempt + 134 spec-exempt + 61 RESIDUAL`. `REQUIRED-HYPERPARAM UNRESOLVED: default.yaml:model.n_layers` is reported (D-MPNN dest is `depth`) and would block strict mode — deliberately kept visible.

## §5 Deliberate NON-fixes (with reasons)

1. **Strict switch NOT flipped.** `HYPERPARAMS_RESOLUTION_STRICT_DEFAULT` stays `False` — residual is 61, not 0; flipping now would block on the first launch gate.
2. **`cfm_full_retrain.yaml` entry fix** (18 of 61 residuals; the YAML names the wrong entry trainer) — outside this file's ownership; config-side lane.
3. **`output.events_jsonl` kept RESIDUAL** (×9) — it IS dead wiring shaped like a boolean; exempting it would hide exactly the class the strict mode exists to catch.
4. **`scripts/_validate_metrics.py` untouched** (user deferred `--strict`), pre-commit hook / `.pre-commit-config.yaml` untouched (other lane), `paper/*` untouched (§5.3), no wet-lab (§5.4/§10), no commit.

## §6 Tests

- NEW `molmetal/tests/test_wfa13_hp_resolution_allowlist.py` (230 lines, 4 tests): classifier unit contract incl. never-exempt override + dot-boundary near-miss; strict PASS on exempt-only fixture; strict FAIL naming the residual key + relaxed PASS on the same fixture; default-mode PASS on the 9 live configs with bucket totals. **4/4 PASS** (0.58 s).
- `test_wf345a1_hyperparam_aliases.py`: fixed 1 stale assertion — `"default.yaml" not in evidence.split("alias-resolved")[-1]` tripped on the Wf-A12/A13 resolution report (which names default.yaml by design). Scoped to the alias-resolutions dict segment. Staleness PROVED pre-existing using a HEAD copy (`/tmp/hole7/pre_flight_check_HEAD.py`): tail contained `default.yaml` at HEAD too.
- Combined run: `test_wfa13 + test_wf343c4_integration + test_wf345a1 + test_wf346a1 + test_pre_flight_check + test_pre_flight_p0_gates` → **28/28 PASS** (3m35s). `test_wf327f11_gate_count_consistency` → 3/3 PASS (gate COUNT unchanged — internal-only change). H5-lane markers (`csv_specs` L978, `gate_nci60_split_no_leak` L1177) verified intact in the merged file.

Honest framing: fixture numbers are synthetic (tmp_path / /tmp); §2/§4 live numbers are MEASURED from the on-disk `configs/` tree on 2026-10-03. No GPU used.

## §7 Filesystem evidence (§5.5 rule #2 — `ls -la` + `wc -l`)

```
-rw------- 1 hugo hugo 453402 Oct  3 11:12 molmetal/scripts/pre_flight_check.py
-rw-r--r-- 1 hugo hugo  16429 Oct  3 11:13 molmetal/tests/test_wf345a1_hyperparam_aliases.py
-rw-r--r-- 1 hugo hugo   9142 Oct  3 11:13 molmetal/tests/test_wfa13_hp_resolution_allowlist.py
 10335 molmetal/scripts/pre_flight_check.py
   230 molmetal/tests/test_wfa13_hp_resolution_allowlist.py
   389 molmetal/tests/test_wf345a1_hyperparam_aliases.py
```

## §8 Write-protocol note (§5.6)

This closeout was written with `molmetal/scripts/wf386a3_atomic_write.py::write_closeout` (atomic temp-file + `os.replace`; `safe_overwrite_check` + `detect_truncation` run around it) — the §5.6 protocol's future-closeout path, exercised here on its first HOLE-lane closeout.
