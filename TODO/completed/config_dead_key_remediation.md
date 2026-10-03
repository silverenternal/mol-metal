# Wf-A11 — Config dead-key remediation (YAML keys that never reached the trainer)

**Date:** 2026-10-03. **Trigger:** user request "你启动ultracode（可以是多个），把刚才累计发现的这么多问题全部都解决了" (task B1 of the ultracode run). **Scope:** `molmetal/scripts/wfA2_config_loader.py` + 9 YAML configs. `pre_flight_check.py` and `scripts/_validate_metrics.py` were NOT touched (another workstream owns them; per task brief).

**Honest framing (§5.2):** every number below is MEASURED with the command cited. Where a value CHANGES for future runs, old value / new value / why are all recorded. Nothing was silently overwritten.

---

## §1 The durable fix (loader) — `/home/hugo/codes/mol-metal/molmetal/scripts/wfA2_config_loader.py`

Before: a YAML leaf with no argparse dest hit a bare `continue` (`# YAML key has no argparse counterpart. Forward-compat: skip silently.`). The config DECLARED one value; the trainer RAN the argparse default.

Now (215 added lines):

1. **Every skip is REPORTED.** Collected per call, attached to the returned namespace as `args.yaml_unresolved_keys`, exposed via `last_skipped_keys()`, and printed to stderr one line per key:
   `[wfA2_config_loader] WARNING: YAML key 'model.hidden_dim' (value=300) skipped — no argparse dest`
   Suppress per-key lines with `WFA2_CONFIG_QUIET=1` (summary line stays). Default remains NON-FATAL so existing launches do not break.
2. **Strict switch.** `WFA2_CONFIG_STRICT=1` raises `UnresolvedConfigKey` listing every unresolved key. Exempt (by documented design): keys in `_NON_TRAINER_SECTIONS` (entry/directives/honest_framing/cross_references/waves/ablation_configs/sweep_pairs/runs/pending_pairs) and leaf keys in `_NON_TRAINER_KEYS` with a NAMED, verified consumer (`n_seeds`, `gpu`, `base_dir`, `log_master`, `training_summary`, `events_jsonl_path` → `scripts/start.sh` L239-261; dry_run/user_ok_* → run-control).
3. **Alias table** `_LEAF_ALIASES` (unambiguous schema-name → real dest): `hidden_dim→hidden`, `grad_clip_norm→max_norm`, `warmup_steps→lr_warmup_steps`, `warmup_epochs→lr_warmup_epochs`, `min_lr→lr_min_lr`, `cpu_workers→cpu_num_workers`. An alias NEVER overrides a YAML key that names the dest directly (two-pass resolution); a suppressed alias is itself reported (strict-fatal) — never silent.
4. **`train` added to `_LEAF_FALLBACK_SECTIONS`** so `configs/default.yaml`'s `train.{batch_size,lr,epochs,...}` block resolves instead of being dropped whole.

Tests: `/home/hugo/codes/mol-metal/molmetal/tests/test_config_yaml.py` — updated the old `..._silently_skipped` test to assert reporting, added 3 Wf-A11 tests (alias resolves; direct key beats alias + reported; strict raises). **MEASURED: `20 passed, 1 skipped`** (`.venv/bin/python -m pytest molmetal/tests/test_config_yaml.py -q`); the skip is the pre-existing "M17 canonical config not on disk" test.

## §2 Instance remediation table (declared value → used until now → fix)

"Used until now" = what the trainer actually ran when launched with only `--config <yaml>` (the `entry.command_template` path).

| Key | Files | Declared / used until now | Fix (a/b/c) | New behaviour |
|---|---|---|---|---|
| `model.hidden_dim` | g5/g6/g4/platinai | 300 / parser default `None` → model built at `DMPNN_HIDDEN=128` | **(a)** alias `hidden_dim→hidden` | 300 reaches `args.hidden` (see §3) |
| `model.backbone` | 8 configs | `chemprop_v2_v3`/`_mt` / never reached (no `--backbone` in either trainer — verified by flag scan) | **(c)** deleted + comment | gone; comment records why |
| `protocol.bf16`, `compute.bf16`, `compute.amp` | 8 configs | true/false / never reached. **`bf16` occurs 0 times in both trainers** (`grep -c bf16` = 0/0; no autocast/GradScaler either) — the "only CFM trainer has Wf-431-B1 --bf16" claim in the platinai comment was FALSE | **(c)** deleted + comment | gone; no runtime effect (nothing to enable/disable) |
| `protocol.grad_clip_norm` | 9 configs | 5.0 (D-MPNN), 1.0 (CFM) / never reached; `--max-norm` (default 5.0 in BOTH trainers, L2306-2308 D-MPNN / L1969-1971 CFM) was the operative key and is declared in every file | **(a)** alias kept for future `grad_clip_norm`-only configs + **(c)** the redundant/conflicting keys deleted | `max_norm` unchanged (5.0 everywhere). CFM `1.0`-vs-`5.0` intent flagged AMBIGUOUS in-file; 1.0 never ran and still does not — authors must reconcile |
| `protocol.warmup_steps` | cfm p1.4 / p1.3 (50); default.yaml (200) | never reached | CFM: **(a)** alias → `--lr-warmup-steps` (default 100); default.yaml: **(c)** deleted (D-MPNN has no steps dest) | CFM config value now set, but `--use-lr-warmup` default False ⇒ no numerical effect yet (comment says so) |
| `protocol.scheduler.warmup_epochs` | all | never reached; both trainers default 5 | **(a)** alias → `--lr-warmup-epochs` | 5 applied = default ⇒ no numeric change |
| `protocol.scheduler.min_lr` | all | never reached | **(a)** alias → `--lr-min-lr` | 1e-6 now applied (cosine floor) |
| `compute.cpu_workers` | D-MPNN: g4/g5/g6 (4), platinai/nci60 (8) | never reached (`--cpu-num-workers` default 0) | **(a)** alias → `cpu_num_workers` | value now applied (0→4/8) for `--config`-only launches |
| `compute.cpu_workers` | CFM: p1.4/p1.3/full | never reached; CFM trainer has NO workers dest | **(c)** deleted + comment | gone |
| `protocol.n_seeds` | all | no argparse dest | **(a)**-class: KEPT, registered in `_NON_TRAINER_KEYS` with consumer `scripts/start.sh` L259 (`N_SEEDS`) | unchanged; loader now reports it WITH its consumer instead of silence |
| `protocol.seeds` (list) | CFM p1.4/p1.3 | never reached (CFM dest is singular `--seed`, enumerated by `entry.command_template`) | **(c)** deleted + comment | unchanged runtime; `n_seeds` retained for start.sh |
| `protocol.seeds` (list) | D-MPNN configs | **LIVE** via `--seeds` (leaf fallback) | none | unchanged |
| `train.*` block | default.yaml (+ cfm_full_retrain) | whole block dropped (`train` not in fallback sections) | **(a)** `train` added to `_LEAF_FALLBACK_SECTIONS` | batch_size/lr/epochs etc. now applied as declared for future runs (default.yaml: 32/1e-4/30) |
| `dry_run` + top-level `user_ok_required` | g5/g6/g4/p1.4/p1.3 | never reached; **no consumer anywhere** (repo grep, non-bak) | **(c)** deleted + comment | launch mode remains start.sh flags + `USER_OK` env (§5.1) |
| `seed: 42` | default.yaml | never reached (D-MPNN dest is `--seeds`) | **(c)** deleted + comment | gone |
| `data.csv`/`data.dataset`/`cohort_filter.*`/`split.*` | CFM configs | not a CFM-trainer input (CFM reads cached clouds); retained as launcher/schema/pre-flight metadata | **NOT remediated** — classified metadata, now VISIBLE in the report | unchanged; flagged for triage |

## §3 hidden_dim evidence — REAL parse path (required verification)

Command: `.venv/bin/python /tmp/wfa11_evidence.py` — imports `molmetal.scripts.train_dmpnn_multitask_wasserstein.parse_args` and calls `parse_args(["--config", <cfg>])` (the real path; the namespace below is post-`apply_yaml_overrides`).

```
BEFORE (aliases disabled, = pre-Wf-A11 behaviour):
  wf124_p3_50epoch_full_sweep_g5.yaml          hidden=None
  wf124_p3_per_innovation_ablation_g6.yaml     hidden=None
  wf124_p3_remaining_metals_g4.yaml            hidden=None
  platinai_pt_4cell_5seed_50ep_v1.yaml         hidden=None
AFTER (current loader):
  wf124_p3_50epoch_full_sweep_g5.yaml          hidden=300 max_norm=5.0 cpu_num_workers=4 lr_warmup_epochs=5 lr_min_lr=1e-06 epochs=50 batch=64 lr=0.001 skipped=84
  wf124_p3_per_innovation_ablation_g6.yaml     hidden=300 max_norm=5.0 cpu_num_workers=4 lr_warmup_epochs=5 lr_min_lr=1e-06 epochs=50 batch=64 lr=0.001 skipped=95
  wf124_p3_remaining_metals_g4.yaml            hidden=300 max_norm=5.0 cpu_num_workers=4 lr_warmup_epochs=5 lr_min_lr=1e-06 epochs=30 batch=64 lr=0.001 skipped=47
  platinai_pt_4cell_5seed_50ep_v1.yaml         hidden=300 max_norm=5.0 cpu_num_workers=8 lr_warmup_epochs=5 lr_min_lr=1e-06 epochs=50 batch=128 lr=0.002 skipped=43
```

`hidden=None` is the pre-fix used value: the model builder resolves `hidden=None → DEFAULT_HIDDEN` (`train_dmpnn_multitask_wasserstein.py` L1505; `DEFAULT_HIDDEN = MT_HIDDEN`, L162; `MT_HIDDEN = DMPNN_HIDDEN`, `molmetal/baselines/dmpnn_multitask.py:78`; `test_wf251` asserts MT_HIDDEN == 128). All four configs now deliver **hidden=300**.

CFM spot-check (same script, `train_cfm_yuelbond_wasserstein.parse_args`): `cfm_per_innovation_ablation_p1.4.yaml` → `hidden_dim=128 lr_warmup_steps=50 use_lr_warmup=False max_norm=5.0 lr_min_lr=1e-06 epochs=3 batch_size=16 lr=0.0001` (aliases wired; grad clip still 5.0 as before).

## §4 Was any SHIPPED metric/experiment produced under the OLD (wrong) value?

- **g4** (`results/wf124_p3_remaining_metals_g4/`): SHIPPED runs exist (Sep 28-29, pre-YAML-unification path). `grep -o 'hidden[^,}]*' training_summary_v2.json` → `"hidden": 300`. Produced **under 300** (CLI), not under 128. `A/Ru_seed1.log` L3 shows `epochs=30` for those runs.
- **g5**: no real runs — only `DRY_RUN.log` + `plan.json`. **g6**: `master.log` Sep 30 shows recovery enumeration `PENDING: 60`; no metrics.
- **platinai**: newest run (`results/platinai_pt_4cell_5seed_50ep/master.log`, Oct 3 07:39) aborted pre-training (`FileNotFoundError: MetalCytoToxDB not found at /mnt/storage/data/molmetal/MetalCytoToxDB.csv`) ⇒ no metric under either value.
- **CFM configs**: existing `results/cfm_per_innovation_ablation_p1.4/` metrics ran with max_norm=5.0 before and after (unchanged); `warmup_steps` newly reaches the namespace but `use_lr_warmup=False` ⇒ no observable change.
- **default.yaml / cfm_full_retrain**: no launches on the affected keys (see adjacent findings).

**Net: no SHIPPED metric is invalidated by this change.** The forward-looking behaviour change (§2 rows marked "(a)") affects FUTURE `--config`-only launches: hidden 128→300 (4 configs), cpu_num_workers 0→4/8 (5 configs), lr_min_lr→1e-6, CFM lr_warmup_steps=50 (inactive), default.yaml train block live (32/1e-4/30).

## §5 Known remaining debt (now VISIBLE, not silent)

Measured skipped-key counts after remediation (`last_skipped_keys()`): g5 **84**, g6 **95**, g4 **47**, platinai **43**, nci60 **48**, cfm p1.4 **85**. Classes: `data.split.*` / `eval.*` / `flow.*` / `output.per_*_dir` / `model.atom_vocab_size` etc. — schema/documentation or non-trainer metadata; NOT remediated in this wave. `WFA2_CONFIG_STRICT=1` currently fails on these (intended visibility; the gate workstream can extend `_NON_TRAINER_KEYS` with verified consumers or drive config cleanup).

**Adjacent findings (flagged, NOT fixed — out of task scope):**
1. `scripts/launch_wf124_p3_g5.sh` L336 passes `"${COMMON_ARGS[@]}"` incl. `--num-workers 8` + `--no-cuda-graphs`, but the D-MPNN trainer `--help` (measured) lists NEITHER ⇒ any real launch through that launcher aborts at argparse. No real G5/G6 runs exist (DRY_RUN only), consistent.
2. `configs/cfm_full_retrain.yaml` `entry.command_template` invokes `metallo_drug_smoke_retrain.py --config ...`; that script rejects `--config` (`results/cfm_full_retrain_v2/master.log`: `error: unrecognized arguments: --config ... --max-norm 5.0`) ⇒ start.sh launch fails immediately.

## §6 Files changed (§5.5 filesystem evidence)

`ls -la` (Oct 3 09:29-09:31, authoring time):

```
20216  molmetal/scripts/wfA2_config_loader.py
 7393  configs/wf124_p3_50epoch_full_sweep_g5.yaml
 7383  configs/wf124_p3_per_innovation_ablation_g6.yaml
 5561  configs/wf124_p3_remaining_metals_g4.yaml
20471  configs/training/platinai_pt_4cell_5seed_50ep_v1.yaml
13790  configs/training/nci60_9cell_5seed_50ep_v1.yaml
 7715  configs/cfm_per_innovation_ablation_p1.4.yaml
 5162  configs/cfm_real_decode_lift_p1.3.yaml
 6510  configs/cfm_full_retrain.yaml
 1988  configs/default.yaml
```
plus `molmetal/tests/test_config_yaml.py` (+3 tests, 1 updated) and this closeout.

Repro commands: `.venv/bin/python /tmp/wfa11_evidence.py` (before/after), `.venv/bin/python /tmp/wfa11_config_fix.py` (deterministic edits, abort-on-mismatch), `.venv/bin/python -m pytest molmetal/tests/test_config_yaml.py -q`.
