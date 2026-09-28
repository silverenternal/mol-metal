# WF: vocab-mask test fix (IMPROVEMENT_BACKLOG item C1)

## TL;DR

The pre-existing test `test_atom_targets_are_supervised_but_not_supplied_as_features`
asserted `last_losses['atom'] == F.cross_entropy(captured_logits, targets)`.
That assumed `CE = log(num_atoms)` (uniform random logits across the full
atom-head support of `max_atomic_number` slots).  R17 added the vocab mask
(F3 — *WF-CFM-P0-F3: vocab_mask applied to atom_logits before
F.cross_entropy*) which `-inf`s every out-of-vocab logit.  After R17 the
model's reported `atom` loss is computed from the **masked** logits, so the
test stopped matching the model's value.  This WF updates the test to apply
the same vocab mask before computing the expected CE.

## Why the test broke

Before R17:

- `atom_head` is zero-initialised ⇒ every logit = 0.
- `F.cross_entropy(logits, target)` over `max_atomic_number=100` slots ⇒
  `CE = log(100) ≈ 4.605` per atom ⇒ averaged across the 4 batch atoms
  ⇒ `≈ 4.605`.

After R17 (F3):

- `train_step` builds a `vocab_mask_tensor` and -inf's every out-of-vocab
  slot.  Default vocab (TODO-33 P1) is
  `(1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 21, 22, 33, 34, 35, 39, 40, 53, 72, 78)`
  (20 elements).
- For an in-vocab target, the per-atom `CE = log(20) ≈ 2.996` — log of
  the **vocab size**, not of `max_atomic_number`.
- The test still computed `F.cross_entropy(captured_logits, targets)` on
  the un-masked logits, producing `log(100)` and diverging from
  `last_losses['atom']`.

The test atoms `[6, 7, 8, 9]` (C, N, O, F) are all in the default 20-element
vocab, so the divergence is purely a "test was asserting against the wrong
denominator" problem — not a regression in the model.

## Fix

`/home/hugo/codes/mol-metal/molmetal/tests/test_atom_training_contract.py`
`::test_atom_targets_are_supervised_but_not_supplied_as_features`:

- Build the same `vocab_mask_tensor` the model uses
  (`model._build_vocab_mask(...)`).
- Apply the mask to the captured (un-masked) logits.
- Recompute `expected` from the masked logits.
- Keep the defence-in-depth fallback the model itself uses in
  `train_step` (re-use un-masked logits on a row whose targets fall fully
  outside the vocab — degenerate but legal): this keeps the assertion
  finite even if a future test atom happens to be OOV.

The new body:

```python
vocab_mask = model._build_vocab_mask(
    captured['logits'].shape[-1], captured['logits'].device,
)
masked_logits = captured['logits'].flatten(0, 1).masked_fill(
    ~vocab_mask, float('-inf'),
)
vocab_set = set(model._atom_vocab)
oov = [int(z) for z in atoms.tolist() if z not in vocab_set]
if oov:
    per_row = masked_logits
    row_is_finite = torch.isfinite(per_row).any(dim=-1)
    if not row_is_finite.all():
        per_row = torch.where(
            row_is_finite.unsqueeze(-1),
            per_row,
            captured['logits'].flatten(0, 1),
        )
    expected = F.cross_entropy(per_row, atoms.to(model.device))
else:
    expected = F.cross_entropy(masked_logits, atoms.to(model.device))
assert model.last_losses['atom'] == pytest.approx(float(expected))
```

The two non-loss assertions in the test are unchanged:

- `captured['input_atoms'].count_nonzero() == 0` — atom targets are still
  not supplied as input features (Z=0 placeholder).
- `not torch.equal(before, model.velocity_field.atom_head.weight)` —
  the atom_head weights still move on a single `train_step`.

## Verification

Command (per the task brief — adjusted for the actual venv layout):

```bash
cd /home/hugo/codes/mol-metal && source env.sh && \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_atom_training_contract.py -v
```

(The task description said `/home/hugo/cades/CadButEaas/.venv-vlm/bin/python`
— that directory does not exist on this host; the actual venv lives at
`/home/hugo/codes/CadButEaas/.venv-vlm`.  Same `python` interpreter.)

Result:

```
molmetal/tests/test_atom_training_contract.py::test_atom_targets_are_supervised_but_not_supplied_as_features PASSED [ 33%]
molmetal/tests/test_atom_training_contract.py::test_decoder_rejects_unsupported_or_disconnected_output_without_repair FAILED [ 66%]
molmetal/tests/test_atom_training_contract.py::test_train_sample_use_same_graph_and_gaussian_forward_time_contract PASSED [100%]
================================ 1 failed, 2 passed, 4 warnings in 2.37s =====
```

The target test **PASSES**.  The unrelated failure
(`test_decoder_rejects_unsupported_or_disconnected_output_without_repair`)
is a pre-existing test that asserts `decode_distance_graph([6, 78], xyz)`
returns `atom_outside_training_vocabulary`.  After TODO-33 P1 added `78`
(Pt) to the default vocab, `78` is no longer out-of-vocab, so the decoder
falls through to the disconnected-graph rejection path.  This is a separate
test that needs updating to match the TODO-33 vocab change — explicitly
out of scope for this C1 fix.

## Files touched

- `molmetal/tests/test_atom_training_contract.py` — masked-CE test body.
- `molmetal/reports/wf_vocab_mask_test_fix/final.md` — this report.
