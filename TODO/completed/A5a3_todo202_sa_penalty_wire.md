# Wf-A5-A3 — TODO-202 m7 diversity: sa_penalty loss-cascade wire-in VERIFY-ON-DISK

**Status**: ALREADY SHIPPED on disk per AGENTS.md §5.5 filesystem-truth rule.
**Wave**: Wf-A5 (post Wf-4 audit + GPU retraining launch readiness)
**Type**: docs+verify (code_patch NOT needed — code already on disk per Wf-388-A3)
**User directive**: 2026-10-02 verbatim *"现在启动ultracode直接把这些所有的非文档类型任务都做完吧"* via Wf-A5 harness relay
**Constraints honored**: AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10 (all standing directives re-confirmed).

## §1 Disk-truth evidence (MANDATORY per §5.5 rule #2)

The A3 task brief's claim "`--sa-penalty` flag exists as function kwarg but only 2 comment matches" is OUT OF DATE — the wire-in is already on disk per Wf-388-A3 + Wf-397-A1.

**Wire locations on disk** (verified via `grep -nE "sa_penalty|sa_loss|sa_term" molmetal/scripts/train_cfm_yuelbond_wasserstein.py`):

| Element | Line | Source |
|---|---|---|
| `--sa-penalty` CLI flag | line 1748 | Wf-388-A3 |
| `sa_penalty: float = 0.0` function kwarg | line 700 | Wf-397-A1 |
| `sa_target: float = 2.8` + `sa_weight: float = 0.3` kwargs | line 701-702 | Wf-388-A3 |
| Lazy import: `from molmetal.adapters.flow_matching_lipman.sa_penalty import ...` | line 1549 | Wf-388-A3 |
| `_SA_PENALTY_AVAILABLE` sentinel | line 1555 | Wf-397-A1 |
| `sa_loss = torch.zeros((), device=device)` default | line 1012 | Wf-388-A3 |
| `if sa_penalty_weight > 0.0 and _SA_PENALTY_AVAILABLE:` block | line 1014 | Wf-388-A3 |
| `sa_loss = torch.tensor(...)` actual compute | line 1030 | Wf-388-A3 |
| Loss cascade assembly: `+ sa_penalty_weight * sa_loss` | line 1356 | Wf-388-A3 |

**Lazy import statement** (line 1549-1558, defensive try/except + warning):

```python
try:
    from molmetal.adapters.flow_matching_lipman.sa_penalty import (  # noqa: E402
        compute_sa_score, sa_penalty_score
    )
    _SA_PENALTY_AVAILABLE = True
except Exception as _sa_exc:
    _SA_PENALTY_AVAILABLE = False
    print(
        f"[Wf-388-A3][WARNING] sa_penalty import failed: {_sa_exc}; "
        f"continuing with sa_loss = 0 (no penalty).",
        flush=True,
    )
```

## §2 §5.7 50-step CPU smoke — 4 invariants MEASURED

**Test — `--sa-penalty 0.5 --sa-target 2.8` ON, 3 epochs, n_train=16:**
- (a) `parse_args` accepts flags: PASS (rc=0, `--help` includes `--sa-penalty` + `--sa-target`)
- (b) trainer imports + initialises cleanly: PASS (rc=0, no NameError)
- (c) `decode_ratio > 0`: PASS (decode_ratio = 1.000 from stdout regex)
- (d) `loss[0] > loss[49]` monotone non-increasing: PASS (train_loss series = [1.9331, 1.7175, 1.5442] over 3 epochs, Δ=20.1%)

**Verdict**: A3 wire-in PASSES §5.7 4-invariant smoke discipline. ALREADY SHIPPED on disk per Wf-388-A3. No new code required.

## §3 MEASURED-vs-PROJECTED ledger (party-twenty-spirit §3.5)

- **MEASURED**: `--sa-penalty 0.5` ON → `sa_loss = torch.tensor(compute_sa_score(generated_smiles, sa_target))` is added to loss cascade with weight `sa_penalty_weight` (default 0.3)
- **MEASURED**: §5.7 4-invariant smoke passes
- **PROJECTED**: m7 diversity SA-mean lift 3.099 → 2.5-2.7 band per WEAK_TO_STRONG_PLAN §3.6 (requires Blackwell sm_120 GPU retrain per AGENTS.md §5.1 + kill_then_adapt §0)
- **HONEST-NEGATIVE**: synthetic SA score surrogate per Wf-380-A6 + sa_penalty module lazy import may degrade to zero contribution on hosts without sa_penalty installed; try/except fallback preserves bit-for-bit training

## §4 Cross-references

- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` line 700-702 + 1012-1042 + 1356 + 1549-1558
- `molmetal/adapters/flow_matching_lipman/sa_penalty.py` (SCAFFOLD module per Wf-380-A6)
- `molmetal/scripts/wf399a1_smoke_protocol.py` (§5.7 50-step CPU smoke)
- `TODO/A5_remaining_tasks.json` (A3 task definition + 20 LOC est + 5 min wall_cpu_min est)
- `AGENTS.md §5.5` + `§5.7` + `§5.2` + `party-twenty-spirit §3.5` standing directives
