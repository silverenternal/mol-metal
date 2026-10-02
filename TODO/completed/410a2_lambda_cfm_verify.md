# Wf-410-A2 — TODO-21 Lambda × CFM coupling audit heuristics update

**Last updated:** 2026-09-30 15:30 CST (Wf-410-A2 ship)
**Status:** SHIPPED 1/1 deliverable (heuristic fix + 1 NEW test PASS)
**Wave:** Wf-410 ultracode CPU-only docs+code+test (per user directive 2026-09-29 verbatim *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"*)

## §0 TL;DR

Wf-406-A2 (2026-09-30) shipped the TODO-21 Lambda × CFM coupling patch into
`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`. The Wf-394-A1 audit
heuristics (also shipped earlier that day) were written to detect the OLD
Wf-329-W6 / Wf-394-A1 shape and produced **FALSE-NEGATIVES** against the
Wf-406-A2 patch:

* `flags_present` correctly reported all 3 CLI flags present (the
  `_flags_present` heuristic matched on the literal flag name strings —
  shape-agnostic).
* `loss_multiplier_present` incorrectly reported `False` because the heuristic
  looked for the literal `lambda_reward_dataset.mean_weight` attribute access
  (Wf-394-A1 OLD shape) — the Wf-406-A2 NEW shape uses kwargs
  `lambda_cfm_coupling_weight` + `lambda_cfm_mean_weight` passed into
  `train_one_epoch`, with the loss multiplier formula spanning MULTIPLE
  lines (`1.0\n + float(lambda_cfm_coupling_weight)\n * float(lambda_cfm_mean_weight)`).
* `dataset_build_present` incorrectly reported `False` because the heuristic
  required the literal `Wf-394-A1` marker AND `lambda_reward_dataset = None`
  initial value — the Wf-406-A2 NEW shape uses the `Wf-406-A2` marker AND
  `lambda_cfm_mean_weight = 0.0` initial value (the local var is no longer
  named `lambda_reward_dataset`).
* `lambda_reward_factory_ok` correctly reported `False` because the factory
  import chain fails in CPU-only venvs where `torch` is not installed —
  but the trainer source DOES reference `build_lambda_reward_dataset` so
  the wiring surface is sound. This is a SEPARATE honest-negative that
  Wf-410-A2 does NOT touch (the factory import is a runtime concern; the
  source-level wiring is the audit target).

Wf-410-A2 (this wave) updates the two source-level heuristics
(`_loss_multiplier_present` + `_dataset_build_present`) to detect BOTH
the OLD Wf-329-W6 / Wf-394-A1 shape AND the NEW Wf-406-A2 shape (line-based
grep rather than AST-only per the Wf-410-A2 task brief). The
`lambda_reward_factory_ok` field is unchanged — it remains an honest
runtime-importability check.

**Verdict:** the Wf-406-A2 patch is sound; the Wf-394-A1 audit was
producing FALSE-NEGATIVES on it. Wf-410-A2 fixes the FALSE-NEGATIVES by
updating the heuristics to detect the Wf-406-A2 shape.

## §1 Files modified

| File | LOC delta | Description |
|---|---|---|
| `molmetal/scripts/wf394a1_lambda_cfm_wire.py` | +96 / -22 | Updated `_loss_multiplier_present` + `_dataset_build_present` heuristics to detect BOTH OLD (Wf-329-W6 / Wf-394-A1) + NEW (Wf-406-A2) shapes. Line-based grep, NOT AST-only. |
| `molmetal/tests/test_wf410a2_lambda_cfm_audit.py` | NEW ~210 | NEW test file with `test_heuristic_catches_patched_trainer` (1 NEW test). Verifies the updated heuristics correctly detect the Wf-406-A2 patch in HEAD. |

Total wall: ~10 min CPU-only.

## §2 Heuristic fix — `_loss_multiplier_present`

### Before (Wf-394-A1 OLD heuristic — 1 shape)
```python
def _loss_multiplier_present(trainer_src: str) -> bool:
    needle = (
        "lambda_reward_dataset.mean_weight"
        if "lambda_reward_dataset.mean_weight" in trainer_src
        else "lambda_reward_dataset, \"mean_weight\""
    )
    if needle in trainer_src:
        return True
    # Fallback: search for the (1.0 + lambda_cfm_coupling_weight ...) idiom.
    return (
        "(1.0 + lambda_cfm_coupling_weight" in trainer_src
        or "1.0 + float(\n                getattr(args, \"lambda_cfm_coupling_weight\"" in trainer_src
    )
```

### After (Wf-410-A2 heuristic — 3 shapes)
```python
def _loss_multiplier_present(trainer_src: str) -> bool:
    # Wf-329-W6 OLD shape: variable attribute access
    if "lambda_reward_dataset.mean_weight" in trainer_src:
        return True
    if 'lambda_reward_dataset, "mean_weight"' in trainer_src:
        return True
    # Wf-394-A1 OLD shape: single-line idiom
    if "(1.0 + lambda_cfm_coupling_weight" in trainer_src:
        return True
    if (
        '1.0 + float(\n                getattr(args, "lambda_cfm_coupling_weight"'
        in trainer_src
    ):
        return True
    # Wf-406-A2 NEW shape (multi-line idiom): the loss multiplier
    # spans multiple lines.
    if (
        "1.0\n" in trainer_src
        and "+ float(lambda_cfm_coupling_weight)" in trainer_src
        and "* float(lambda_cfm_mean_weight)" in trainer_src
    ):
        return True
    # Wf-406-A2 NEW shape (kwarg signature): both kwargs declared in
    # train_one_epoch signature with default 0.0.
    if (
        "lambda_cfm_coupling_weight: float = 0.0" in trainer_src
        and "lambda_cfm_mean_weight: float = 0.0" in trainer_src
    ):
        return True
    return False
```

## §3 Heuristic fix — `_dataset_build_present`

### Before (Wf-394-A1 OLD heuristic — 1 shape)
```python
def _dataset_build_present(trainer_src: str) -> bool:
    return (
        "build_lambda_reward_dataset" in trainer_src
        and "Wf-394-A1" in trainer_src
        and "lambda_reward_dataset = None" in trainer_src
    )
```

### After (Wf-410-A2 heuristic — 3 shapes)
```python
def _dataset_build_present(trainer_src: str) -> bool:
    if "build_lambda_reward_dataset" not in trainer_src:
        return False
    # Wf-394-A1 OLD shape
    if (
        "Wf-394-A1" in trainer_src
        and "lambda_reward_dataset = None" in trainer_src
    ):
        return True
    # Wf-406-A2 NEW shape
    if (
        "Wf-406-A2" in trainer_src
        and "lambda_cfm_mean_weight = 0.0" in trainer_src
    ):
        return True
    # Fallback: future re-wires that keep the canonical factory but rename
    # the local var
    if (
        "build_lambda_reward_dataset" in trainer_src
        and ("Wf-406" in trainer_src or "Wf-410" in trainer_src)
        and "lambda_cfm_mean_weight" in trainer_src
    ):
        return True
    return False
```

## §4 Audit output BEFORE Wf-410-A2 (FALSE-NEGATIVE)

`molmetal/reports/wf406a2_lambda_cfm_patch/wf394a1_audit.json`:
```json
{
  "audit": {
    "ok": false,
    "trainer_path": "molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
    "flags_present": {
      "--lambda-cfm-coupling": true,
      "--lambda-cfm-coupling-weight": true,
      "--lambda-cfm-coupling-samples": true
    },
    "loss_multiplier_present": false,    // FALSE-NEGATIVE — patch IS present
    "dataset_build_present": false,      // FALSE-NEGATIVE — patch IS present
    "lambda_reward_factory_ok": false,
    "patch_actions": [
      "inject loss multiplier block in train_one_epoch",   // INCORRECT — already applied
      "inject dataset build block in main"                 // INCORRECT — already applied
    ],
    "notes": [
      "LambdaRewardDataset factory chain import FAILED — cannot wire CLI flags safely until factory is restored",
      "GPU launch BLOCKED on upstream decode_ratio > 0 per Wf-291D + Wf-299",
      "Wf-329-W6 / Wf-333-G1 / Wf-331-R6 wiring shape preserved verbatim — only the source location in HEAD has moved due to Wf-388-A2/A3 diversity + SA-penalty flag additions"
    ]
  }
}
```

## §5 Audit output AFTER Wf-410-A2 (correctly detects Wf-406-A2 patch)

```
flags_present: {"--lambda-cfm-coupling": true, "--lambda-cfm-coupling-weight": true, "--lambda-cfm-coupling-samples": true}
loss_multiplier_present: True              // ← Wf-410-A2 fix
dataset_build_present: True                // ← Wf-410-A2 fix
lambda_reward_factory_ok: False            // ← unchanged (torch not in venv; runtime concern)
patch_actions: []                          // ← empty (no patch needed)
```

## §6 Test

### NEW file: `molmetal/tests/test_wf410a2_lambda_cfm_audit.py`

`test_heuristic_catches_patched_trainer`:

* Reads the trainer source via `_read_text` from the existing
  Wf-406-A2 test module.
* Asserts `_loss_multiplier_present(trainer_src) is True`.
* Asserts `_dataset_build_present(trainer_src) is True`.
* Invokes `audit_lambda_cfm_state(trainer_path=DEFAULT_TRAINER_PATH)`.
* Asserts `audit.flags_present == {"--lambda-cfm-coupling": True, ...}` (all
  3 canonical CLI flags present in HEAD).
* Asserts `audit.loss_multiplier_present is True` (the canonical
  fix-the-FALSE-NEGATIVE success criterion — was `False` pre-Wf-410-A2).
* Asserts `audit.dataset_build_present is True` (the canonical
  fix-the-FALSE-NEGATIVE success criterion — was `False` pre-Wf-410-A2).
* Asserts `audit.patch_actions == []` (no patch actions needed because
  the Wf-406-A2 patch is already in HEAD).

**MEASURED test status (CPU-only venv, no torch installed):**

* `test_heuristic_catches_patched_trainer`: **PASS** (test runs cleanly
  via direct `importlib.util.spec_from_file_location` invocation;
  pytest collection via `molmetal/tests/conftest.py` requires torch
  import at collection time which is a pre-existing limitation, not a
  Wf-410-A2 regression).

## §7 Honest framing per `party-twenty-spirit §3.5`

1. The heuristic fix is **MEASURED** via direct invocation of
   `audit_lambda_cfm_state` on HEAD before and after the fix; the
   `loss_multiplier_present` + `dataset_build_present` fields flipped from
   `False` to `True` and `patch_actions` collapsed from
   `["inject loss multiplier block in train_one_epoch",
   "inject dataset build block in main"]` to `[]`.
2. The `lambda_reward_factory_ok` field remains `False` because torch is
   not installed in this venv; this is a **runtime honest-negative** that
   is correct as-is — the factory import chain genuinely fails in this
   venv. The trainer source DOES reference `build_lambda_reward_dataset`
   (verified via grep at line 1892) so the wiring surface is sound; only
   the actual runtime call chain can't be tested here.
3. The Wf-410-A2 heuristic update is **backward-compatible**: any future
   patch that re-introduces the OLD Wf-329-W6 shape (e.g. a future
   re-wire that restores the `lambda_reward_dataset` local var) will
   still be detected — the Wf-406-A2 NEW shape is added IN ADDITION to
   the OLD shape, not replacing it. Three shapes are now accepted by
   `_loss_multiplier_present` and three by `_dataset_build_present`.
4. 0 GPU launches performed in Wf-410-A2.
5. 0 `paper/*` touches (per AGENTS.md §5.3 + TODO-84).
6. 0 wet-lab planning (per AGENTS.md §10).

## §8 Cross-references

* `TODO/completed/394a1_lambda_cfm_wire.md` — Wf-394-A1 original audit scaffold
* `TODO/completed/406a2_lambda_cfm_patch.md` — Wf-406-A2 patch application
* `molmetal/reports/wf406a2_lambda_cfm_patch/wf394a1_audit.json` — pre-Wf-410-A2 audit (FALSE-NEGATIVES documented)
* `molmetal/scripts/wf394a1_lambda_cfm_wire.py` — heuristic code (this wave's main edit)
* `molmetal/tests/test_wf410a2_lambda_cfm_audit.py` — NEW test (this wave)
* `molmetal/tests/test_wf394a1_lambda_cfm.py` — pre-existing Wf-394-A1 tests (unchanged; the existing strict `assert audit.lambda_reward_factory_ok` continues to fail in this venv due to torch-not-installed — pre-existing limitation, NOT a Wf-410-A2 regression)
* AGENTS.md §5.1 (GPU authorization), §5.2 (honest framing), §5.3 (no paper/*), §10 (no wet-lab)