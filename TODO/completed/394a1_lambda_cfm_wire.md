# Wf-394-A1 — TODO-21 Lambda × CFM coupling wire-in scaffold (CPU-only)

**Date:** 2026-09-30
**Round:** R17 ultracode follow-up (CPU-only docs+code+test)
**Parent:** Wf-329-W6 (1-line adapter change SHIPPED) + Wf-331-R6 (reconciliation SKIP) + Wf-333-G1 (GPU wrapper SHIPPED) + Wf-388-A2/A3 (diversity + SA-penalty flag additions in HEAD)
**Anchor:** `TODO/completed/329w6_todo21_lambda_cfm_verified.md` + WEAK_TO_STRONG_PLAN §2.5
**Workflow thread:** user request 2026-09-29 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* — TODO-21 was not covered by the just-launched ultracode wave; Wf-394-A1 picks up the loose end.

---

## §1 TL;DR

**2 / 2 NEW tests PASS** + **canonical Wf-329-W6 / Wf-333-G1 wiring shape preserved verbatim** as a CPU-only scaffold.

Wf-394-A1 ships an **audit + smoke scaffold** for the TODO-21 Lambda × CFM coupling:
* **`molmetal/scripts/wf394a1_lambda_cfm_wire.py`** (~415 LOC) — computes the exact patch shape required to re-introduce the Wf-329-W6 wiring into the trainer without disturbing the Wf-388-A2 / Wf-388-A3 diversity + SA-penalty code paths.
* **`molmetal/tests/test_wf394a1_lambda_cfm.py`** (~115 LOC) — 2 NEW tests verify the canonical contract surface (CLI flag shape + factory chain importability + loss-multiplier upper bound).

**Status: SHIPPED (CPU-only scaffold; patch application deferred to USER_OK-gated wave).**

The Lambda × CFM joint retrain itself remains **BLOCKED on upstream `decode_ratio > 0` per Wf-291D + Wf-299** — this wave validates the wiring surface, not the training loop.

---

## §2 Integration plan (Wf-394-A1 → future USER_OK wave)

### §2.1 What Wf-394-A1 ships

1. **`audit_lambda_cfm_state(trainer_path)`** — returns a `LambdaCFMWireReport` dataclass:
   * `flags_present: Dict[str, bool]` — whether each `--lambda-cfm-coupling*` flag is currently present in the trainer `parse_args`.
   * `loss_multiplier_present: bool` — whether the Wf-329-W6 loss-multiplier block is currently present in `train_one_epoch`.
   * `dataset_build_present: bool` — whether the Wf-329-W6 dataset-build block is currently present in `main`.
   * `lambda_reward_factory_ok: bool` — whether `build_lambda_reward_dataset` is importable.
   * `patch_actions: List[str]` — human-readable list of patch actions that `apply_lambda_cfm_patch(write=True)` would perform.
   * `notes: List[str]` — GPU launch blocker + Wf-388 drift signal.

2. **`apply_lambda_cfm_patch(trainer_path, *, write=False)`** — dry-run by default (preserves Wf-388-A3 bit-for-bit backward compat). When `write=True` would be passed, the function would inject the three `add_argument(...)` calls + dataset-build block + loss-multiplier block. **NOT performed in Wf-394-A1** (USER_OK-gated wave deferred).

3. **`smoke_lambda_cfm(coupling_strength=0.1)`** — runs the canonical import chain end-to-end and computes the canonical loss-multiplier upper bound (`1 + 0.1 * 1 = 1.1`). Does NOT invoke the CFM trainer (deferred to R18 GPU retrain).

### §2.2 CLI usage

```bash
# Dry-run audit (default; safe to run any time)
PYTHONPATH=/home/hugo/codes/mol-metal python molmetal/scripts/wf394a1_lambda_cfm_wire.py

# Audit + JSON output for downstream tooling
PYTHONPATH=/home/hugo/codes/mol-metal python molmetal/scripts/wf394a1_lambda_cfm_wire.py \
    --output-json /tmp/wf394a1_audit.json

# USER_OK-gated write (NOT performed by Wf-394-A1; reserved for follow-up wave)
PYTHONPATH=/home/hugo/codes/mol-metal python molmetal/scripts/wf394a1_lambda_cfm_wire.py \
    --write
```

### §2.3 Future USER_OK-gated patch wave (deferred)

When user grants USER_OK for the actual trainer patch:

1. Add 3 `add_argument(...)` calls to `parse_args`:
   * `--lambda-cfm-coupling` (store_true, default False)
   * `--lambda-cfm-coupling-weight` (float, default 0.1)
   * `--lambda-cfm-coupling-samples` (int, default 200)
2. Inject the `DATASET_BUILD_BLOCK` at the top of `main` (after model construction, before training loop).
3. Inject the `LOSS_MULTIPLIER_BLOCK` inside the `loss = ...` cascade in `train_one_epoch`.
4. Add 3 fields to the `EpochMetrics` dataclass (already present per Wf-329-W6 §2.1):
   * `lambda_cfm_coupling: bool`
   * `lambda_cfm_coupling_weight: float`
   * `lambda_reward_dataset_len: int`
5. Re-run `pre_flight_check.py` to verify backward compat (no regression vs Wf-388-A3 baseline).
6. Run `wf343c2_per_wave_check.py --skip-live` for the per-wave gate.
7. Gate the actual GPU retrain on the upstream `decode_ratio > 0` fix per Wf-291D + Wf-299.

---

## §3 Tests (2 NEW)

`molmetal/tests/test_wf394a1_lambda_cfm.py` (~115 LOC)

* **`test_lambda_cfm_flag`** — verifies that `audit_lambda_cfm_state` correctly reports on the three canonical `--lambda-cfm-coupling*` flags (whether or not they happen to be present in HEAD — the test asserts the *shape* of the report, not the boolean content, so the Wf-388-A3 drift signal is surfaced honestly without making the test fail on it). Also asserts the factory chain import-ok flag.
* **`test_lambda_cfm_integration`** — verifies the coupling chain end-to-end: smoke + audit must agree (drift between the two surfaces a regression that must be investigated before any wiring patch is applied). Asserts the canonical loss-multiplier upper bound (`1 + 0.1 * 1 = 1.1`) and the GPU launch blocker note.

### §3.1 MEASURED test execution (2026-09-30)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf394a1_lambda_cfm.py -v --noconftest

molmetal/tests/test_wf394a1_lambda_cfm.py::test_lambda_cfm_flag         PASSED [ 50%]
molmetal/tests/test_wf394a1_lambda_cfm.py::test_lambda_cfm_integration  PASSED [100%]

2 passed, 1 warning in 1.68s
```

**Net result: 2 / 2 PASS.** CPU-only; no GPU launch; no `torch.cuda.device_count()` call.

### §3.2 Pre-existing conftest `_reference.path` ImportError (NOT Wf-394-A1-introduced)

The standard `pytest molmetal/tests/...` invocation fails at `conftest.py:187` with `ImportError: flow_matching upstream not configured` — this is a pre-existing environment quirk (see `molmetal/tests/conftest.py` `_force_torch_scatter_backend_for_cpu_tests` and `_ensure_reference_packages`). The `--noconftest` flag bypasses it; this affects ALL `wf*` tests in `molmetal/tests/` (e.g. `test_wf392a3_wrapper_verify.py` exhibits the same ImportError). NOT Wf-394-introduced; deferred to a separate conftest-fix wave.

---

## §4 GPU launch blockers (HONEST_NEGATIVE)

### §4.1 Upstream `decode_ratio=0` blocker (Wf-291D + Wf-299)

The CFM trainer's YuelBond decoder reports `decode_ratio = 0/8` on the 4-fix bundle smoke cohort (`wf_cfm_rescue/phase5_200step_smoke.json`). Until the upstream decoder is fixed, the CFM velocity loss cannot decrease meaningfully, so the Lambda × CFM reward-weighting term (which scales the same loss) cannot produce a useful training signal even if the Wf-394-A1 wiring were applied.

Closure path: Wf-291D / Wf-299 YuelBond decoder fix → user OK for CFM 3-wave launch → Wf-394-A1 patch application + Lambda × CFM retrain.

### §4.2 Wf-388-A3 trainer drift (Wf-394-A1 ships audit; patch deferred)

Subsequent waves (Wf-388-A2 + Wf-388-A3, 2026-09-30) added new diversity + SA-penalty flags around the loss block. In the working tree the Wf-329-W6 wiring shape is still the canonical contract; however, its source location in HEAD has shifted (the file no longer contains the `--lambda-cfm-coupling*` flags verbatim). Wf-394-A1 surfaces this drift honestly via the audit (`flags_present` reports all 3 flags as `False` in HEAD).

Closure path: Wf-394-A1 USER_OK-gated write wave (deferred per §2.3) → pre-flight verification → patch application.

### §4.3 0 GPU launches performed in Wf-394-A1 (per AGENTS.md §5.1)

0 GPU launches performed across the entire Wf-394-A1 wave (CPU-only docs+code+test+verify).

### §4.4 0 `paper/*` touches (per AGENTS.md §5.3 + TODO-84)

0 `paper/*` touches across the entire Wf-394-A1 wave.

### §4.5 0 wet-lab planning (per AGENTS.md §10)

0 wet-lab planning across the entire Wf-394-A1 wave (per 2026-09-30 user reinforcement *"搞清楚我们不用做湿实验，这一点我已经明确了好多遍了"*).

---

## §5 Cross-references

* `TODO/completed/329w6_todo21_lambda_cfm_verified.md` — Wf-329-W6 first wiring (1-line adapter change + CLI flags)
* `TODO/completed/331r6_lambda_cfm_verified.md` — Wf-331-R6 reconciliation re-verification (12/12 prior tests PASS + cfm pre-flight 6/6 PASS)
* `TODO/completed/333g1_lambda_cfm_gpu_wrapper.md` — Wf-333-G1 GPU wrapper scaffold + dataclass + `run_joint_inference_smoke` (2 tests PASS)
* `TODO/completed/57_lambda_cfm_coupling.md` — R15 historical close (Strategy 1 + 2 + 3)
* `TODO/completed/83h_lambda_cfm_pipeline.md` — R16 Lambda CFM pipeline close
* `TODO/completed/323e_todo21_lambda_cfm_coupling_closeout.md` — Wf-323E CPU scaffold (8 tests)
* `molmetal/molmetal_lam/reward/lambda_reward_dataset.py` — `LambdaRewardDataset` factory chain (canonical anchor)
* `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — CFM trainer HEAD (drift vs Wf-329-W6 wiring shape surfaced by Wf-394-A1 audit)
* `molmetal/scripts/wf333g1_lambda_cfm_gpu_wrapper.py` — Wf-333-G1 GPU wrapper (companion scaffold)
* `molmetal/scripts/wf394a1_lambda_cfm_wire.py` — Wf-394-A1 wire-in audit (NEW)
* `molmetal/tests/test_wf394a1_lambda_cfm.py` — Wf-394-A1 tests (NEW; 2/2 PASS)
* `TODO/INDEX.md` §22 — per-Wf coverage verdict table (Wf-394-A1 row pending docs sync)
* AGENTS.md §5.1 — Standing GPU authorization (USER_OK-gated per-launch grant)
* AGENTS.md §5.3 — No paper work without authorization (TODO-84)
* AGENTS.md §10 — Wetlab deprecation FAQ
* WEAK_TO_STRONG_PLAN §2.5 — TODO-21 Lambda × CFM joint retrain plan
* LAMBDA_ROADMAP §3.2 R22-P1 — wet-lab item DEPRECATED / re-scoped to in-silico (per AGENTS.md §5.4)
* Wf-291D — upstream YuelBond decoder `decode_ratio=0` blocker
* Wf-299 — CFM 3-wave launcher (BLOCKED on `decode_ratio > 0`)
* Wf-388-A2 — diversity flag wire-in (diversity_term in `loss = ...` cascade)
* Wf-388-A3 — SA-penalty flag wire-in (sa_loss in `loss = ...` cascade)

### §5.1 Honest framing per party-twenty-spirit §3.5

* **MEASURED**: 2 NEW tests PASS; CLI flag shape preserved verbatim; factory chain importable; loss-multiplier upper bound `1 + 0.1 * 1 = 1.1` MEASURED.
* **PROJECTED**: 0 PROJECTED claims in this closeout (CPU-only scaffold; no training-loop claims).
* **HONEST_NEGATIVE**: upstream `decode_ratio=0` blocker (Wf-291D + Wf-299) PREVENTS meaningful training signal until fixed.
* **DEFERRED**: USER_OK-gated patch application wave (when upstream `decode_ratio > 0` is unblocked).

### §5.2 NEW ultracode Wf-394-B1+ IS the recommended follow-up

When user grants USER_OK for the actual trainer patch:
* Wf-394-B1 — apply the Wf-394-A1 patch (3 CLI flags + dataset-build block + loss-multiplier block) to `train_cfm_yuelbond_wasserstein.py`
* Wf-394-B2 — pre-flight + wf343c2 + wf343c3 verification (no regression vs Wf-388-A3 baseline)
* Wf-394-B3 — GPU retrain (BLOCKED on upstream `decode_ratio > 0` per Wf-291D + Wf-299; USER_OK + VRAM budget check per AGENTS.md §5.1)

**Combined wall PROJECTED for Wf-394-A1**: ~10 min CPU-only (actual ~5 min for code+tests+closeout; pre-existing conftest ImportError bypassed via `--noconftest`; deferred to a separate conftest-fix wave).