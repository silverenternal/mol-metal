# Closeout — `--config` flag wire-in in `metallo_drug_smoke_retrain.py`

**Date:** 2026-10-03. **Lane:** `molmetal/scripts/metallo_drug_smoke_retrain.py`
(+ this closeout). No other repo files modified; nothing committed.

## 1. Defect (MEASURED)

`molmetal/scripts/metallo_drug_smoke_retrain.py` declared `--config` (Wf-A8c,
2026-10-03) with help text promising "applies overrides via wfA2_config_loader",
but `args.config` was never read and the module never imported the loader — the
flag was parsed and discarded while its help text documented behaviour that did
not exist (AGENTS.md §5.7 FLAG_ONLY class). `results/cfm_full_retrain_v2/master.log`
(2026-10-03 07:40) records the corroborating launch failure:
`error: unrecognized arguments: --config configs/cfm_full_retrain.yaml ... --max-norm 5.0`.

## 2. Decision: option (a) — WIRE IT

Chosen because the loader integration is the same proven pattern already used by
`train_dmpnn_multitask_wasserstein.py` (`apply_yaml_overrides(p, args, args.config)`),
it is non-fatal by default on unknown YAML keys (skip + warn; strict only via
`WFA2_CONFIG_STRICT=1`), and the only obstacle was mechanical: the loader
re-parses with `parser.parse_args([])` to capture defaults, which `SystemExit`s
on argparse-required arguments — `--out-dir` was the single `required=True` arg,
so it is now `default=None` with the required contract enforced manually right
after the YAML overrides (same error message as before). The no-`--config` path
returns from the loader unchanged (`if not config_path: return args`), so
CLI-only invocations are bit-for-bit unchanged (verified: no config banner,
defaults intact, same early-exit behaviour).

## 3. Diff shape (35 insertions, 4 deletions in one file)

1. `--config` help text corrected to the now-true behaviour + wire-in comment.
2. `--out-dir` `required=True` -> `default=None` (+ explanatory note).
3. After `parse_args`: `apply_yaml_overrides(parser, args, args.config)` +
   one-line banner (`config: ... N key(s) skipped`) + manual `--out-dir` error.
4. Robustness fix in `_summary()`: empty loss curve now returns the same key set
   (adds `delta_first_to_last: None`); previously the final.md writer raised
   `KeyError: 'delta_first_to_last'` on a 0-step run, masking the real error.

## 4. §5.7 smoke (50 steps)

- Invocation: `.venv/bin/python` + `--config <yaml>` + `--out-dir`; the YAML
  supplied `steps: 50 / batch_size: 8 / n_train: 64 / decode_smoke_every: 25 /
  seed: 0` (CLI omitted those flags, so a pass also proves YAML->args).
- **CPU clause NOT met — structural:** the trainer's training path calls
  `triton_kernels/equivariant_ops.py` (`scatter_sum`/`aggregate_vectors`), which
  hard-raises `RuntimeError` on any CPU tensor with no fallback, so a CPU
  50-step smoke cannot execute (measured: `ERROR at step 1: Triton kernel
  received CPU tensor(s)`). The smoke was therefore run on GPU 0
  (`CUDA_VISIBLE_DEVICES=0`) under the standing §5.1 GPU grant after a VRAM
  check (2 MiB / 97887 MiB used, 0% util at launch).
- Runtime workaround (disclosed): a PRE-EXISTING, unrelated adapter import
  breakage on committed HEAD makes `LipmanFlowMatchingAdapter.setup()` raise
  `ModuleNotFoundError: ..._reference.path` (see §5b). The smoke driver
  (`/tmp/wf_st_wrapper.py`, not in the repo) appended the `_reference` shim dir
  to `_reference.__path__` at runtime — the apparent 1-line intended fix —
  before running the trainer unchanged.

RESULTS (measured, `/tmp/wf_st_smoke_verdict.json` + `/tmp/wf_st_smoke_out/final.json`):

| # | §5.7 invariant | Result | Numbers |
|---|---|---|---|
| 1 | parse_args accepts `--config` | PASS | rc=0; no argparse error; YAML applied — banner `training 50 steps` / `n_train=64` with those flags absent from the CLI; `csv` from a bogus YAML path reproduced in a pre-check |
| 2 | trainer imports + initialises with the flag | PASS | `building LipmanFlowMatchingAdapter` / `resolved device: cuda`; no traceback |
| 3 | `decode_ratio > 0` | PASS | 1.000 at step 25 and 1.000 at step 50 (16/16 decoded, mean_atoms 8.0, disconnected 0); `decode_summary.max_decode_ratio = 1.0` |
| 4 | `loss[0] > loss[49]` | **FAIL** | total first=inf last=inf; atom first=inf last=inf (min 1.235); bond first=inf last=inf (min 1.035); total min 17.631; cfm 17.740 -> 12.196 (min 6.758) |

50/50 param steps ran (0.69 s/step, 34.4 s wall; 38.1 s incl. startup). The four-deep invariant set is 3/4: invariant 4 fails because the atom/bond loss components spike to `inf` at some steps (finite minima mid-run), so `inf > inf` is false — the composite curve is not monotone non-increasing at the 50-step horizon.

CONTROL (attribution): the same 50-step run with the same values passed entirely
on the CLI (no `--config`) reproduced the *same* curves — cfm 17.739914 -> 12.196362
(min 6.758395), atom min 1.2353307, bond min 1.0354308, total min 17.631254 (all
endpoints inf, identical to the `--config` run to ~7 significant digits; decode max
1.0). So invariant 4's failure is a property of the trainer's loss dynamics at this
configuration, NOT of the `--config` plumbing — and the two paths are behaviourally
equivalent at equal settings.

§5.7 verdict: **NOT PASS (3/4) — wire-in not claimable as SHIPPED**. The accurate
label is UNVERIFIED-BY-§5.7, not FLAG_ONLY: FLAG_ONLY means parsed-and-discarded,
which is exactly what this fix removed; the flag is demonstrably read and applied
(invariant 1 + the YAML pre-check + the CLI control).

## 5. Findings outside this lane (NOT fixed; report only)

### 5a. `scripts/start.sh` launch path rejects 6 of its own flags for this trainer

`scripts/start.sh:366` appends `--events-jsonl-path --lr-scheduler
--lr-warmup-epochs --lr-min-lr --weight-decay --max-norm` to every entry
command. `metallo_drug_smoke_retrain.py`'s parser accepts none of those 6
(re-measured today with the current parser; `--checkpoint-interval` +
`--resume-from` are accepted after Wf-A8c). So the YAML-driven start.sh launch
of this trainer stays broken regardless of the `--config` wire-in — that needs
a decision (per-trainer flag filtering in start.sh, or adding the flags to the
smoke trainer). The same six flags do exist on the D-MPNN trainer
(`--events-jsonl-path` verified in its parse_args), so start.sh's generic block
is D-MPNN-shaped.

### 5b. PRE-EXISTING adapter import breakage on committed HEAD

`molmetal/adapters/flow_matching_lipman/_reference/__init__.py` sets
`__path__ = [str(configure_reference())]` which resolves to the upstream
clone's flat `flow_matching/` dir; the on-disk shim subpackages
(`_reference/path/`, `_reference/solver/`, `_reference/utils/`) are therefore
unreachable and every `LipmanFlowMatchingAdapter.setup()` raises
`ModuleNotFoundError: ..._reference.path` (reproduced in isolation; last
touched by the initial release commit — long-standing, not a working-tree
regression; likely surfaced when the local `molmetal/references/flow_matching`
clone moved to upstream's post-restructure flat layout). Reproduced:

    .venv/bin/python -c "from molmetal.adapters.flow_matching_lipman._reference.path import AffineProbPath"

Fix belongs to the adapter owner; candidate 1-liner:
`_reference.__path__` should also include the shim dir
(`os.path.dirname(__file__)`), which is what made the smoke runnable.

### 5c. Trainer verdict banner mis-reports for non-1000-step runs

`metallo_drug_smoke_retrain.py:590-594` derives its verdict from
`decode_summary["decode_ratio_at_1000"]`; on a shorter `--steps` run that key is
`None`, so the banner printed "STILL 0 — coordinate-quality wall persists" while
the measured max decode_ratio was 1.0. Cosmetic/pre-existing; not fixed here.

## 6. Honest framing

- The wire-in ran end-to-end (parse + import + init + 50/50 training steps +
  decode + final.json) **with two disclosed deviations**: GPU instead of CPU
  (§5.7's CPU clause is structurally impossible for this trainer), and the
  runtime shim-dir patch for the pre-existing §5b breakage.
- §5.7 gate result: **3/4 — NOT PASS**. Invariants 1-3 PASS (incl. the
  decode_ratio canary at 1.000); invariant 4 FAILS on `inf` endpoints in the
  pre-existing atom/bond loss. No success claim is made for the wire-in; the
  `--config` plumbing itself is verified by invariant 1 (YAML values provably
  reached args) plus the no-op control (CLI path unchanged).
- On the unmodified tree, the raw launch still dies at `setup()` (§5b) before
  any flag behaviour can be observed — so "smoke passed" must always be quoted
  together with that caveat.
- No test added; evidence is the §5.7 run + final.json artefacts under
  `/tmp/wf_st_smoke_out/`.

## 7. On-disk evidence (ls -la, current disk state)

```
-rw-r--r-- 1 hugo hugo 30860 Oct  3 09:54 molmetal/scripts/metallo_drug_smoke_retrain.py
-rw-r--r-- 1 hugo hugo  3028 Oct  3 09:55 /tmp/wf_st_smoke_out_cli/final.json
-rw-r--r-- 1 hugo hugo  3019 Oct  3 09:54 /tmp/wf_st_smoke_out/final.json
-rw-r--r-- 1 hugo hugo  1309 Oct  3 09:54 /tmp/wf_st_smoke_verdict.json
```
-rw------- 1 hugo hugo 9221 Oct  3 09:56 TODO/completed/config_flag_wirein_metallo_smoke.md
```
(permissions 0600 as written by `write_closeout`; cosmetic, not re-chmodded)

`git diff --stat` on the lane file: 1 file changed, 35 insertions(+), 4 deletions(-).
Nothing committed (working tree only). `TODO/INDEX.md` not touched (outside lane).
