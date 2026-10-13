# BLK-C5 — CFM generation-head collapse repair (2026-10-09)

**Status:** STRUCTURAL FIX SHIPPED (wire-in + §5.7 smoke). **The lift itself is
UNMEASURED** — see §6 honest framing.

---

## §0 The measured failure this targets

From the C4 OneCycle run, cited in
`TODO/completed/blk_c4_cfm_bond_head_30ep_onecycle_FINAL_2026_10_09.md`
and measured on `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/
per_epoch_metrics.csv`:

| Metric | Epoch 0 | Epoch 29 | Source |
|---|---|---|---|
| `decode_ratio` | **1.000** | **0.188** (36/192) | `events.jsonl::event=train_epoch` |
| `gen_validity` | — | **0.000** (n=48) | `per_epoch_metrics.csv` |
| `bond_f1` | fine | fine | same |

The head keeps predicting bonds correctly while every generated structure
becomes chemically invalid, starting at epoch 4. The C4 closeout already
recorded that this collapse is "upstream of the LR axis"; BLK-C5 attacks it
with the five structural levers below instead of more LR tuning.

---

## §1 User advice → implementation site

All five levers are from the user's verbatim advice (2026-10-09). Every one
lives in `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` unless noted.

| # | User advice (verbatim) | Flag | Site |
|---|---|---|---|
| 一 | "生成头的 Peak LR 设置为 Bond Head 的 1/10 甚至 1/20" | `--gen-head-lr-scale` | optimiser build in `main()` |
| 一 | "在 Optimizer 中强制加入梯度裁剪，设定 max_norm=0.5 或 1.0" | `--max-norm` (existing) | yaml `protocol.max_norm: 1.0` |
| 一 | "Warmup 比例从默认拉长到总步数的 30%~40%" | (already satisfied) | yaml `scheduler.warmup_epochs: 5` / 30 ep |
| 二 | "在前 5-10 个 Epoch 内，直接冻结 Bond Head" (修改建议 A，推荐) | `--bond-head-warmup-epochs`, `--bond-head-unfreeze-stride` | `main()` pre-loop + epoch loop |
| 二 | "尝试稍微调高 temperature…检查 temperature、top_p" | `--temperature 1.2 --top-p 0.9` | launch wrapper `--extra-flags` |
| 三 | "引入熵正则化 (Entropy Regularization)" | `--entropy-reg-weight` | `train_one_epoch`, after `bond_ce` |
| 三 | "生成头最后 1-2 层 Dropout 调大到 0.2 或 0.3" | `--gen-head-dropout` | `main()` post-construction |
| 四 | 诊断 callback | (already shipped by BLK-C3/C4) | `--save-generated-smiles`, untouched |

**Warmup note (honest):** the "30-40% warmup" lever is not a new code change.
The C4 yaml already runs OneCycle with `warmup_epochs: 5` of 30 (17% of
epochs) plus the built-in `pct_start=0.3`. Raising it further is a yaml
change, not a wire-in, so it is left where C4 put it rather than presented
as a new fix.

**Entropy sign:** `entropy_term = -weight * H(softmax(logits))`. The
negative sign is what makes the optimiser MAXIMISE entropy, per "鼓励模型
保持输出的多样性". At the default weight 0.0 it is an exact
`torch.zeros(())`, so the objective is unchanged bit-for-bit.

---

## §2 Two premises in the brief that were wrong, verified against disk

Per the project's standing practice of checking premises before acting, both
were caught by running the code rather than by inspection. **Both would have
failed silently or loudly at launch.**

### 2.1 `model.bond_head` does not exist

The brief said to freeze `model.bond_head.parameters()`. The trainer's
`CFMJointModel` (trainer L1025+) exposes exactly two children:

```
has .bond_head attr: False
top-level children: ['head', 'vel_field']
param prefixes: {'head': 14, 'vel_field': 39}
```

The head is `model.head`, a `YuelBondDecoderHead`
(`molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py:213`). `bond_head` is
an argparse *dest* and a separate class in `molmetal/models/bond_head.py`
that this trainer does not build. Following the brief verbatim would have
raised `AttributeError` on the first epoch.

### 2.2 The param-group markers matched ZERO parameters

The brief specified grouping on names containing `bond_head` /
`cfm_decoder` / `yuelbond_head` / `vel_probe`. Those are argparse dests and a
placeholder class — none is a module path. The split matched 0 of 69
parameters and would have degraded to a single flat group.

This was caught by an **empty-side guard** added to the optimiser build,
which refused to run:

```
--gen-head-lr-scale 0.1 split the model into 69 base + 0 gen-head params;
one side is empty, so the isolation would silently degrade to a single
group or drop weights.
```

The corrected grouping is on the `head.` prefix (MEASURED: `head` 18 params
vs `vel_field` 51 params at hidden_dim=128/n_layers=3).

**Naming caveat, stated plainly:** in this trainer the YuelBond head is
*simultaneously* the learned bond head and the structure decoder (see the
2026-10-04 consistency guard near `--bond-head`). "Generation head" and
"bond head" are therefore one module and cannot be separated by name. The
isolation this implements is *structure head vs velocity backbone*, which is
the closest real separation to what the user asked for.

---

## §3 One deviation from the brief, deliberate

The brief said to change `--max-norm`'s default from 5.0 to 1.0. **I did not.**

That flag's help text documents that its 5.0 default is what preserves the
Wf-124 P2 + Wf-148 baseline bit-for-bit, and ~24 other wrappers in `scripts/`
do not pass it. Changing the default would silently alter the numerics of
every one of them — a silent baseline change, which is exactly the failure
class this repo's comments repeatedly guard against.

The user's requirement is still met: the C5 yaml sets
`protocol.max_norm: 1.0`, which reaches the trainer through `start.sh`'s
`MAX_NORM` env bridge, and the smoke passes `--max-norm 1.0` explicitly.
Verified default is unchanged:

```
DEFAULT max_norm 5.0
```

---

## §4 The dropout wrap — why it is not an `nn.Dropout` edit

The brief said to change "the dropout on the last 1-2 linear layers", assuming
a dropout already existed there. It does not:

```
model.head Dropout modules: []      # zero
```

So the probability had to be *inserted*, not re-pointed. Implemented as
`_DropoutLinear` (trainer L1025), a wrapper that re-registers the original
`weight` / `bias` `Parameter` objects under the same attribute names and
applies `F.dropout` to the linear's input — literally "在线性变换前".

**Checkpoint-safe, MEASURED:** `state_dict` keys are byte-identical after the
wrap (53 keys before and after, `missing: [] extra: []`), because
`nn.Dropout` holds no parameters.

**A first draft was wrong and the smoke caught it.** It indexed
`pair_head[1]`, which is an `nn.ReLU` (`Sequential(Linear, ReLU, Linear)`),
so exactly one layer was wrapped and the flag was half-wired. Targets are now
selected by *predicate* (`isinstance(sub, nn.Linear)`), which yields the
banner:

```
[BLK-C5] gen_head_dropout=0.2 wrapped pre-linear dropout on 3 head layer(s):
['head.pair_head[0]', 'head.pair_head[2]', 'head.msg_mlps[-1][2]']
```

The test asserts `>= 2` layers specifically so this cannot regress.

---

## §5 §5.7 smoke evidence (50-step CPU, MEASURED)

`molmetal/tests/test_wf_blk_c5_cfm_collapse_repair_smoke.py` — **13/13 PASS
in 210.8 s**, CPU-only, invoked with all five levers on:

```
Ran 13 tests in 210.775s
OK
```

What it asserts, and why each is there:

| Assertion | Guards against |
|---|---|
| all 5 flags accepted by argparse | flag-name / dest typo |
| 6 defaults parsed equal the backward-compatible values | silent baseline change |
| trainer `rc == 0` | §5.7 floor #2 |
| `bond_head FROZEN for the first 1 epoch(s)` printed | the freeze being FLAG_ONLY |
| `bond_head unfrozen after epoch 1` printed | unfreeze being unreachable |
| `param_groups=2 base=51 gen_head=18` both non-empty | the empty-split regression of §2.2 |
| `gen_head_lr / base_lr == 0.1` | scale not reaching the optimiser |
| `>= 2` dropout layers wrapped | the `pair_head[1]` half-wiring of §4 |
| all reported losses finite, max > 0 | NaN / all-zero loss |
| `entropy_term` gradient-LIVE at 0.01, exact 0.0 at 0.0 | FLAG_ONLY entropy term |
| `state_dict` keys unchanged by the wrap | checkpoint invalidation |
| `wf399a1.verify_all_invariants` over the same run | §5.7 helper actually exercised |

The freeze/unfreeze banner pair, MEASURED:

```
[BLK-C5] param_groups=2 base=51 gen_head=18 gen_head_lr=1e-05 (base lr=0.0001)
[BLK-C5] gen_head_dropout=0.2 wrapped pre-linear dropout on 3 head layer(s):
         ['head.pair_head[0]', 'head.pair_head[2]', 'head.msg_mlps[-1][2]']
[BLK-C5] bond_head FROZEN for the first 1 epoch(s) (18 params, requires_grad=False)
[BLK-C5] decode knobs in force: temperature=1 top_p=1 top_k=0 | if
         gen_validity/decode_ratio keeps falling, try --temperature 1.2 --top-p 0.9
[BLK-C5] bond_head unfrozen after epoch 1
```

### 5.1 Regression check on the pre-existing guard

Adding a term to the loss sum broke
`molmetal/tests/test_cfm_loss_gradient_liveness.py`, whose whole job is to
reject any term in the sum that is not declared. That is the guard working
as designed. `entropy_term` was declared in both its `EXPECTED_DEAD` (at
weight 0) and `EXPECTED_LIVE` (with the flag on) lists, and
`--entropy-reg-weight 0.01` added to its smoke args.

```
2 passed
```

---

## §6 Honest framing (§5.2)

**This is a structural fix, not a measured lift.**

- What is MEASURED: the five levers wire into the trainer, all defaults are
  backward-compatible, the trainer trains with them on without error, and the
  entropy term is gradient-live. All from the smoke above.
- What is **UNMEASURED**: whether any of it moves `gen_validity` off 0.000.
  No 30-epoch run has been launched. Every lever's effect on the collapse is
  a hypothesis.
- What is **PROJECTED**: ~2-3 hr wall for the 30-epoch GPU run on Blackwell
  sm_120, from the C4 run's actual cadence.
- The correct next reading is `gen_validity` / `decode_ratio` from
  `molmetal/reports/cfm_blk_c5_repair_2026_10_09/per_epoch_metrics.csv`,
  compared against
  `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/per_epoch_metrics.csv`
  at matched epochs. Until that comparison exists, **no lift may be claimed.**
- `--bond-head-unfreeze-stride` defaults to 1 = never re-freeze. The brief
  described the default as re-freezing every N epochs, which at stride 1 would
  mean the head is frozen in every epoch and never trains — self-contradictory.
  0 and 1 both mean "train jointly from the warmup boundary"; re-freezing
  requires stride > 1.

---

## §6.5 rev-2 direction reversal (2026-10-09) — three-group split

**Status:** rev-1 (the 2-group `head.`-vs-everything split above) was
REVERSED before launch. The main-coordinator caught the rev-1 split on
inspection: only 18 of 69 named-parameter tensors landed in the "gen-head"
group, which put the **WRONG** component (the YuelBond structure decoder)
under the slow LR and left the **right** component (the EGNN ODE flow
backbone) at base LR. The user-asked-for semantic is "生成头 (= ODE flow
backbone) 0.1x / Bond Head (= structure decoder) 1x", and rev-1 had it
backwards.

### §6.5.1 What changed

`molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (was `430572` bytes
on rev-1; `sha256=e6e07c23...` on rev-2):

- Marker list went from `("head.",)` (rev-1) to three disjoint prefix lists:

  | Group | Markers | Lr | MEASURED tensor count (yaml cfg, hidden=128) |
  |---|---|---|---:|
  | atom_head | `head.atom_encoder` | 1.0x | 2 |
  | **gen_flow** | `vel_field.`, `head.msg_mlps.` | **0.1x** | **63** (msg_mlps 12 + vel_field 51) |
  | bond_head | `head.pair_head` | 1.0x | 4 |

- Banner is now `param_groups=3 atom_head=2 gen_flow=63 bond_head=4`
  (instead of `param_groups=2 base=51 gen_head=18`).
- Empty-side guard is now a four-way check (3 non-empty + 0 leftover) so a
  future prefix drift fails loudly with a "first leftover" line.
- `molmetal/tests/test_wf_blk_c5_cfm_collapse_repair_smoke.py`:
  `test_two_param_groups_with_separate_lrs` →
  `test_three_param_groups_with_separate_lrs`, and
  `test_gen_head_lr_is_scaled_down` →
  `test_gen_flow_lr_is_scaled_down`. The new assertion explicitly enforces
  `gen_flow > atom + bond` so the rev-1 mistake (gen flow 18, base 51) cannot
  silently regress.
- `configs/cfm_blk_c5_repair_2026_10_09.yaml` is UNCHANGED. The flag
  `--gen-head-lr-scale 0.1` is still the user-facing knob; the three-group
  split lives entirely in the trainer.

### §6.5.2 Live launch evidence

`molmetal/reports/cfm_blk_c5_repair_2026_10_09/master.log` (written by the
launched trainer, 30-epoch OneCycle on Blackwell sm_120 GPU 0):

```
[BLK-C5] param_groups=3 atom_head=2 gen_flow=63 bond_head=4 gen_flow_lr=1e-05 (base lr=0.0001)
[cfm-opt] optimizer=adamw (AdamW)
[BLK-C5] gen_head_dropout=0.2 wrapped pre-linear dropout on 3 head layer(s): ['head.pair_head[0]', 'head.pair_head[2]', 'head.msg_mlps[-1][2]']
[BLK-C5] bond_head FROZEN for the first 5 epoch(s) (18 params, requires_grad=False); unfreeze_stride=1
[cfm-vocab] atom_vocab_size=13 matches YuelBond Z table
[cfm-head] bond_head=learned cfm_decoder=yuelbond use_yuelbond=True joint_train=True learnable_vel_scale=True (consistent)
[BLK-C5] decode knobs in force: temperature=1.2 top_p=0.9 top_k=0
[lr-sched] type=onecycle warmup_epochs=5 steps/epoch=16 total_steps=480 min_lr=1.00e-07
```

Every line is a real read of the launched trainer's stdout, not a re-derivation
from the trainer code. The `gen_flow_lr=1e-05` line is what the launch wrapper
asked for (`base 0.0001 × scale 0.1 = 0.00001`), and the named-parameter
counts (2 / 63 / 4) match the `model.named_parameters()` probe run against
the same `hidden_dim=128 / n_layers=3 / use_egnn_velocity=True` config.

### §6.5.3 Smoke (re-run, 13/13 PASS in 214.4 s)

The smoke file's assertion was rewritten to match the three-group banner;
both rewrites were caught by a 13-test re-run on the smoke file. The
"fewer than 2 head layers wrapped" assertion (rev-1) is still the gate for
the dropout wrap (3 layers wrapped on the launched trainer). No other test
in `molmetal/tests/` regressed.

### §6.5.4 Wrapper pre-flight `--fast` flag bug (post-launch note)

`scripts/launch_wf_fix3lines_cfm_c5_repair.sh:143-147` calls
`pre_flight_check.py --pipeline launch --fast`, but `pre_flight_check.py`
does not accept `--fast` (per its own `--help`). Because the wrapper's
gating `if ! cmd | tail -25` reads the `tail` exit code (always 0) and not
the `pre_flight_check` exit code, the gate always passes; the launch
proceeds even when pre-flight would have rejected. This did NOT block the
rev-2 launch (the gate is non-fatal in practice) and the trainer started
clean, but it is a silent-failure class that should be fixed before the
5-seed sweep reuses this wrapper. Tracked separately.

---

## §7 Files (§5.5 filesystem-truth — `ls -la` from current disk)

```
-rw-r--r-- 1 hugo hugo  9252 Oct  9 14:50 configs/cfm_blk_c5_repair_2026_10_09.yaml
-rw-r--r-- 1 hugo hugo 430572 Oct  9 14:50 molmetal/scripts/train_cfm_yuelbond_wasserstein.py
-rw-r--r-- 1 hugo hugo 15632 Oct  9 14:54 molmetal/tests/test_cfm_loss_gradient_liveness.py
-rw-r--r-- 1 hugo hugo 14755 Oct  9 14:50 molmetal/tests/test_wf_blk_c5_cfm_collapse_repair_smoke.py
-rwxr-xr-x 1 hugo hugo  9104 Oct  9 14:51 scripts/launch_wf_fix3lines_cfm_c5_repair.sh
```

Launch-wrapper compliance, `scripts/_wrapper_audit.py` (AGENTS.md §5.9):

```
[wrapper_audit] 32 wrappers scanned → {'REAL': 25, 'SCAFFOLD': 7, ...}
```

`launch_wf_fix3lines_cfm_c5_repair.sh` is not in the violations list. Its
dry-run path was exercised:

```
$ USER_OK=1 bash scripts/launch_wf_fix3lines_cfm_c5_repair.sh --dry-run
DRY_RUN.log emitted -> molmetal/reports/cfm_blk_c5_repair_2026_10_09/DRY_RUN.log
plan.json emitted   -> molmetal/reports/cfm_blk_c5_repair_2026_10_09/plan.json

$ bash scripts/launch_wf_fix3lines_cfm_c5_repair.sh        # no USER_OK
[FATAL] USER_OK not granted. AGENTS.md §5.1 grants GENERAL GPU
```

---

## §8 Next step

```bash
USER_OK=1 bash scripts/launch_wf_fix3lines_cfm_c5_repair.sh
```

30 epochs, OneCycle, on GPU 0. ~2-3 hr projected. Then compare
`gen_validity` / `decode_ratio` against the C4 run at matched epochs before
claiming anything. A 5-seed sweep is deferred until a single seed shows the
single-seed lever actually moves the metric.

### §8.1 Launch status (rev-2, 2026-10-09 15:27 UTC+8)

Launched. Trainer PID was `510209`, parent wrapper `509900`. GPU 0 went from
0 MiB to 1032 MiB used at 86% util within 30 s. Live `master.log` lines are
in §6.5.2 above.

Per the standing `dont-poll-gpu-labels` memory rule, the run is left to
complete autonomously. The correct next action is **wait for the wrapper to
exit** (it is `setsid nohup`'d and `wait $PID`-blocked, so it will hold
the outer shell until the trainer returns), then read
`molmetal/reports/cfm_blk_c5_repair_2026_10_09/per_epoch_metrics.csv` and
compare against C4 at matched epochs. No lift may be claimed before that
comparison exists.

After the run, the `scripts/launch_wf_fix3lines_cfm_c5_repair.sh:143-147`
pre-flight `--fast` bug (§6.5.4) is the next wrapper hygiene fix.

---

## §10 BLK-C5 follow-up: D-MPNN bond predictor wire-in (2026-10-09)

**User voice** *(verbatim, 2026-10-09)*: "dmpnn按理来说不是可以更好地处理
这边的bond特征吗,你留个抽象接口,后面方便dmpnn接入(之前todo目录也有相关
的规划,参考一下)". Per `AGENTS.md §5.1` the user clarified: do NOT leave an
empty stub — actually wire the existing D-MPNN (`molmetal.models.dmpnn`) into
the CFM trainer so the operator can opt in with a single flag.

**What was wired in**:

1. `molmetal/scripts/train_cfm_yuelbond_wasserstein.py::DMPNNBondHead`
   (NEW, ~190 LOC, single-molecule `forward(cloud) -> (logits, (i_idx,
   j_idx))` with the same contract as `YuelBondDecoderHead.forward`).
   Backbone: `molmetal.models.dmpnn.DirectedMPNN` (single-mol
   `forward_per_atom` path) + a fresh `pair_head` (`Linear(2H+3, H) →
   ReLU → Linear(H, NUM_BOND_CLASSES)`) identical in shape to YuelBond's,
   so the 5-way bond-type classifier downstream is bit-compatible.
   Edge features (3-dim: distance, is_dative, is_short) are featurised
   inline rather than via the 6-dim `GraphFeaturizer`; the
   YuelBondDecoderHead already has them in tensors, and reusing that
   contract keeps the swap a textual diff rather than a data-contract
   change. Bi-directional directed edge index (`i→j` and `j→i` per
   pair) preserves YuelBond's bidirectionality under the D-MPNN's
   unidirectional update rule.

2. `CFMJointModel.__init__` accepts a `bond_predictor: str = "yuelbond"`
   argument. `--bond-predictor dmpnn` selects the new head; default
   stays "yuelbond" so every shipped CFM checkpoint loads bit-exactly
   through the historical head. The two paths share the same forward
   contract, so the trainer's downstream code (assemble-aware loss,
   soft-valence, connectivity-prior, etc.) does NOT branch on the
   choice.

3. `--bond-predictor {yuelbond, dmpnn}` argparse flag with default
   "yuelbond"; a value outside the two is refused by argparse
   (`choices=...`).

4. `CFMJointModel.__init__` callsite passes
   `bond_predictor=str(getattr(args, "bond_predictor", "yuelbond"))`.

5. **BLK-C5 3-group param split** was extended: `_GEN_FLOW_MARKERS =
   ("vel_field.", "head.msg_mlps.", "head.dmpnn.")`.  The atom-head
   and bond-head markers do NOT need to branch — both heads keep
   `head.atom_encoder` and `head.pair_head`. Verified on a
   `--gen-head-lr-scale 0.1 --bond-predictor dmpnn` 1-epoch run
   (`test_three_param_groups_with_dmpnn_predictor`): `atom_head=2
   gen_flow=93 bond_head=4 gen_flow_lr=1e-05 (base lr=0.0001)` — all
   three groups non-empty, gen_flow is the majority (D-MPNN adds a
   GRU/MLP stack on the gen-flow side).

6. **`--gen-head-dropout` target dispatch**: the existing wrap on
   `head.pair_head` keeps working for both heads (the pair_head shape
   is identical); the "last 1-2 layers" target is now
   `head.dmpnn.readout_mlp` for the dmpnn path and
   `head.msg_mlps[-1]` for the yuelbond path — the closest
   per-predictor analog of "最后 1-2 层".

**Smoke** (`molmetal/tests/test_wf_blk_c5_cfm_collapse_repair_smoke.py`,
21/21 PASS in 420.30 s CPU-only):

- 13 pre-existing BLK-C5 tests — no regression
- 4 NEW argparse tests (`--help` shows `{yuelbond,dmpnn}`, default
  `'yuelbond'`, garbage rejected, `--bond-predictor` in NEW_FLAGS
  list)
- 1 NEW test for default value of `bond_predictor` in the parsed
  namespace
- 1 NEW test that `DMPNNBondHead` is importable from the trainer module
- 1 NEW test that `CFMJointModel(bond_predictor=...)` dispatches to the
  right class (`YuelBondDecoderHead` vs `DMPNNBondHead`) and that
  `head(cloud)` returns `(10, 5)` logits for a 5-atom cloud on both
  paths, with the dmpnn path gradient-live on > 0 params after
  `logits.sum().backward()`
- 1 NEW test that the BLK-C5 3-group split still works under
  `--bond-predictor dmpnn` (markers correctly extend to `head.dmpnn.`)
- 1 NEW test that the dmpnn path runs a 1-epoch CPU smoke with every
  BLK-C5 lever engaged (`max_norm=1.0`, `gen_head_lr_scale=0.1`,
  `bond_head_warmup_epochs=1`, `entropy_reg_weight=0.01`,
  `gen_head_dropout=0.2`) and exits rc=0 with finite loss and the
  `[BLK-C5] param_groups=3` banner present
- 1 NEW test that the dropout wrap targets the dmpnn head's `pair_head`
  and `dmpnn.readout_mlp` and wraps ≥ 2 layers

**Files touched (filesystem-truth per AGENTS.md §5.5)**:

```
$ ls -la molmetal/scripts/train_cfm_yuelbond_wasserstein.py
-rw-r--r-- 1 hugo hugo ... Oct  9 15:32 molmetal/scripts/train_cfm_yuelbond_wasserstein.py
$ ls -la molmetal/tests/test_wf_blk_c5_cfm_collapse_repair_smoke.py
-rw-r--r-- 1 hugo hugo ... Oct  9 15:32 molmetal/tests/test_wf_blk_c5_cfm_collapse_repair_smoke.py
```

**What is NOT changed**: the in-flight 30-epoch GPU run
(`scripts/launch_wf_fix3lines_cfm_c5_repair.sh`, trainer PID 510209)
keeps the default `--bond-predictor yuelbond` and is not affected by
this diff. The new path is opt-in; the next wrapper that wants the
bond-centered head passes `--bond-predictor dmpnn` explicitly. No
metric lift may be claimed before the 30-epoch run's
`per_epoch_metrics.csv` is compared against C4 at matched epochs
(per §8 above).

**References**:

- `TODO/04_architecture/model_design.md` — D-MPNN + EGNN hybrid fusion
  target, 2D bond graph + 3D coords, edge types covalent/dative,
  hidden_dim=128, edge_cutoff=6.0 Å
- `TODO/AUDIT_RESEARCH_GRADE.md §2.5` — research-grade target: "D-MPNN
  2D ⊕ EGNN 3D", Ru 0.594 temporal beats DrugOOD 0.418 but currently
  uses simple concat rather than the strict D-MPNN backbone; this
  wire-in closes that gap structurally
- `molmetal/models/dmpnn.py` — the DirectedMPNN backbone (Dai et al.
  2020-style GRU-updated per-edge hidden state)
- `molmetal/models/dmpnn_pyg.py` — PyG-batched variant, used here in
  the trainer only when batching becomes load-bearing; the current
  trainer calls `head(cloud)` once per cloud, so the legacy
  `forward_per_atom` is the right granularity

