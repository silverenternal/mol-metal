# Wf-285B — Kendall UW + DANN GRL Wiring Report

**Date:** 2026-09-29
**Branch:** master
**Author:** Hugo Lee
**Status:** COMPLETE — 3/3 wiring tests PASS, smoke run OK (both ON and OFF paths).

---

## 1. Problem (Wf-283B honest gap)

Wf-283B added the `--no-kendall` / `--no-dann` CLI flags to
`molmetal/scripts/train_dmpnn_multitask_wasserstein.py` and the
`test_no_kendall_no_dann_flags.py` suite pinned the argparse contract.
**But the flags were NOT wired into the actual loss / model / train-loop
code.**  Concretely:

* `MultiTaskLoss(alpha=alpha)` was hardcoded — `args.kendall` was ignored
  (the manual alpha path always ran, even when `--no-kendall` was set).
* `MultiTaskWassersteinModel(...)` had no `use_dann` parameter and no
  `DomainAdversarialHead` sub-module — `args.dann` was a no-op.
* `train_one_epoch_joint(...)` never invoked `ReverseLayerF` or computed
  a domain loss.

Result: toggling `--no-kendall --no-dann` was a parse-only contract. The
per-innovation ablation study (TODO-143 §2.1 + §2.2 + Wf-251-F3 ablation
grid) was blocked because the lever had no effect on the trainer.

Wf-285B closes the gap by passing the toggles all the way through.

## 2. What got wired

### 2.1 `MultiTaskWassersteinModel.__init__` (file
`/home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_multitask_wasserstein.py`)

Added three new constructor parameters:

```python
def __init__(self, ..., use_dann: bool = False,
             num_domains: int = 4, domain_head_dropout: float = 0.0):
```

When `use_dann=True`, the model instantiates a
`molmetal.models.dmpnn.DomainAdversarialHead(in_dim=2*hidden, hidden_dim=hidden,
num_domains=num_domains)` and registers it as `self.domain_head`.  When
`use_dann=False`, `self.domain_head = None` and the parameter count is
identical to the Wf-131 P1 baseline.

### 2.2 `MultiTaskLoss` instantiation (same file)

`criterion = MultiTaskLoss(alpha=alpha, use_kendall_loss=use_kendall)` —
the existing Kendall UW head (lazy-built in
`molmetal.baselines.dmpnn_multitask.MultiTaskLoss._get_kendall`) is now
triggered when `use_kendall=True` and bypassed when `use_kendall=False`.

### 2.3 `train_one_epoch_joint` — DANN GRL block

Added three optional kwargs:
`use_dann: bool = False`, `domain_id: int = 0`,
`grl_alpha: float = 0.1`, `lambda_grl: float = 0.1`.  When `use_dann=True`
AND `model.domain_head is not None`, the per-batch loop:

1. Runs the encoder via `model.encode_pool(...)` to get the (B, 2H)
   pooled embedding.
2. Passes the pooled embedding through `ReverseLayerF.apply(...)`
   (forward = identity, backward = `-grl_alpha * dL/dh`).
3. Computes `domain_logits = model.domain_head(rev)` and a cross-entropy
   loss against a constant per-batch `domain_id` label.
4. Adds `lambda_grl * domain_loss` to the total before `backward()`.

Same block applied to both the DataLoader path (`num_workers > 0`) and
the manual-loop path (`num_workers == 0`).  When `use_dann=False` the
block is bypassed and the Wf-131 P1 bit-for-bit behaviour is preserved.

### 2.4 `train_one_metal_one_seed` — forwarding layer

Added five new parameters (`use_kendall`, `use_dann`, `domain_id`,
`grl_alpha`, `lambda_grl`, `num_domains`) and threads them through
`MultiTaskWassersteinModel(...)`, `MultiTaskLoss(...)`, and
`train_one_epoch_joint(...)`.  Default values preserve the Wf-131 P1
shipped behaviour (Kendall ON, DANN ON).

### 2.5 `main()` — CLI wire-in

The dispatch to `train_one_metal_one_seed(...)` now forwards
`use_kendall=bool(args.kendall)`,
`use_dann=bool(args.dann)`,
`domain_id=int(getattr(args, "domain_id", 0))`, etc.  Four new optional
CLI flags (`--domain-id`, `--lambda-grl`, `--grl-alpha`, `--num-domains`)
are added with sensible defaults so callers can vary DANN hyperparameters
without editing source.

## 3. Tests added

File: `molmetal/tests/test_kendall_dann_wiring.py` (3 tests)

| Test                       | Contract                                                                  |
|----------------------------|---------------------------------------------------------------------------|
| `test_kendall_on`          | `MultiTaskLoss(use_kendall_loss=True).forward(...)` triggers the Kendall UW sub-module, total loss is differentiable, no NaN/Inf. |
| `test_kendall_off`         | `MultiTaskLoss(use_kendall_loss=False).forward(...)` reproduces `alpha*BCE + (1-alpha)*MSE` exactly; Kendall UW head is NOT instantiated. |
| `test_dann_on_off`         | `MultiTaskWassersteinModel(use_dann=False)` has `domain_head=None` and a baseline parameter count; `use_dann=True` registers a `DomainAdversarialHead` and strictly increases parameter count. |

## 4. Smoke verification (1 epoch, Ru, seed=0, CPU)

Two runs:

### 4.1 `--no-kendall --no-dann` (manual alpha + no DANN)

```
[Ru seed=0] epoch 1/1 loss=2.3035 cls=0.6074 reg=3.9996 w=1.1845
            val_auc=0.6857 val_pearson=0.017 (65.2s)
[Ru seed=0] DONE best_epoch=1 val_auc=0.6857 test_auc=0.5849
            test_pearson=0.204 (65.2s, featurise=0.4s)
```

### 4.2 Defaults (Kendall ON + DANN ON)

```
[Ru seed=0] epoch 1/1 loss=2.3814 cls=0.6348 reg=4.1280 w=1.2197
            val_auc=0.6797 val_pearson=0.030 (108.0s)
[Ru seed=0] DONE best_epoch=1 val_auc=0.6797 test_auc=0.6012
            test_pearson=0.203 (108.0s, featurise=0.7s)
```

Both paths run end-to-end without crash.  The DANN ON path is
~1.65× slower (108s vs 65s) due to the extra GRL + domain-head forward
+ backward pass per mini-batch — expected given the extra `Linear`
layers and gradient reversal.  Test AUC difference is within the
1-epoch noise floor (DANN ON 0.6012 vs OFF 0.5849, no statistical
significance at n=1 epoch).

## 5. Final verdict: PASS

* 3/3 new wiring tests pass (`molmetal/tests/test_kendall_dann_wiring.py`)
* 8/8 existing CLI-flag tests still pass
  (`molmetal/tests/test_no_kendall_no_dann_flags.py`)
* Smoke run completes in both ON and OFF paths without import, shape,
  NaN, or backward-graph errors.

## 6. Files touched

* `/home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
  — wired `use_kendall`, `use_dann`, `domain_id`, `grl_alpha`,
  `lambda_grl`, `num_domains` through the entire stack; added DANN
  GRL block to `train_one_epoch_joint` (both DataLoader and manual
  paths); added four new optional CLI flags with defaults.
* `/home/hugo/codes/mol-metal/molmetal/tests/test_kendall_dann_wiring.py`
  — new file with 3 tests (`test_kendall_on`, `test_kendall_off`,
  `test_dann_on_off`).

## 7. Follow-ups (out-of-scope for Wf-285B)

* Per-cell-line `domain_id` labelling (TODO-143 §2.2 mentions the
  four-domain mode where `domain_id` varies per-batch based on the
  row's data source — MetalCytoToxDB / DrugOOD / NCI-60 / BindingDB).
  Today's driver passes a constant `domain_id`; the per-row labelling
  needs a column in the 5-metal CSV.
* Multi-domain training pool (mix MetalCytoToxDB + DrugOOD rows in a
  single batch with their respective `domain_id`).  Today the driver
  only loads MetalCytoToxDB.
* Paper §6 ablation table entry for `use_dann × use_kendall ∈ {0,1}²`
  (4 cells).  Today's wiring gives us the levers; the 4-cell grid run
  is left for a follow-up workflow (Wf-289 / Wf-291 candidate).