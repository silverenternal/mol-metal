# WF-R15-CFM-CPU-Verify Phase 2 — YuelBond decoder wiring verify

**Date:** 2026-09-16
**Scope:** Verify that decoder_rework.py + YuelBond decoder can be
co-imported and that the YuelBondDecoder produces a valid PairFeature
on a cisplatin-like cloud. Document whether `use_yuelbond=True` flag
exists or needs to be added.
**Verdict:** **YuelBond module composes with DecoderRework parallel
path; wiring flag `use_yuelbond=True` not yet present in
`decoder_rework.py`** — the production pipeline still routes through
the BondAwareDecoder.

---

## 1. Current state (audit)

`grep "use_yuelbond\|YuelBondDecoder\|YuelBondDecoderHead" \
     decoder_rework.py flow_matching_lipman/__init__.py \
     r10_cfg_real_crossdocked.py`

→ **0 hits.** The YuelBond decoder is shipped per
`wf_cfm_rescue/phase1_yuelbond.md` (2026-09-16, 290 LOC in
`molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py`) but is NOT
imported by `decoder_rework.py`, NOT imported by
`flow_matching_lipman/__init__.py`, and NOT imported by
`r10_cfg_real_crossdocked.py`.

The wiring **contract** is satisfied (the module is available, the
tests pass), but the production `use_yuelbond=True` flag does NOT
exist anywhere in the live pipeline.

## 2. Test contract

**NEW**: `molmetal/tests/test_r15_yuelbond_wire.py` (5 tests, all pass
on CPU in 2.6 s):

| Test | What it verifies |
|------|------------------|
| `test_yuelbond_module_importable` | YuelBondDecoder / DecoderHead / Result are importable |
| `test_yuelbond_featurise_returns_valid_pair_feature` | `featurise(cloud)` returns `decode_succeeded=True` and consistent edge count |
| `test_decoder_rework_parallel_path_produces_decoded_mol` | DecoderRework + BondAwareDecoder pipeline still works on cisplatin-like cloud |
| `test_yuelbond_and_decoder_rework_coexist` | Both modules can be imported in the same process; cross-call outputs both produce expected shapes |
| `test_yuelbond_result_compatible_with_pair_feature` | YuelBondResult.pair_features is a valid PairFeature consumable by BondAwareDecoder-like pipelines |

```text
$ uv run pytest molmetal/tests/test_r15_yuelbond_wire.py -v
molmetal/tests/test_r15_yuelbond_wire.py::test_yuelbond_module_importable PASSED
molmetal/tests/test_r15_yuelbond_wire.py::test_yuelbond_featurise_returns_valid_pair_feature PASSED
molmetal/tests/test_r15_yuelbond_wire.py::test_decoder_rework_parallel_path_produces_decoded_mol PASSED
molmetal/tests/test_r15_yuelbond_wire.py::test_yuelbond_and_decoder_rework_coexist PASSED
molmetal/tests/test_r15_yuelbond_wire.py::test_yuelbond_result_compatible_with_pair_feature PASSED
5 passed in 2.60s
```

## 3. Why no wiring yet

Per `wf_cfm_rescue/phase1_yuelbond.md` §5:

> "Why ship the module now? (1) The lit anchor (Wang 2025) makes
>  the GNN decoder a documented alternative; (2) The implementation
>  is CPU-friendly (≤5k params, K=3 layers) so it can be embedded
>  into decode smokes without GPU cost; (3) The 5 unit tests verify
>  the contract; regression guard for future refactors that might
>  want to swap decoders."

And §6:

> "A/B test integration in `_generate_impl` (gated by
>  `--decoder=yuelbond` CLI flag), with `--decoder=bondorder`
>  (current default) preserved for backward compat. This requires
>  a GPU retrain; for now, the module ships as a parallel path."

The wiring is intentionally deferred because:

1. **Switching decoders mid-training is a distribution shift** —
   a 5K-step BondOrderHead retrained model would produce logits in
   a completely different range from YuelBond's. The two decoders
   have to be retrained against the same velocity field.
2. **YuelBond is randomly initialised** in the current state — its
   logits carry no learned chemistry signal until trained.
4. **The BondAwareDecoder is the production path** — swapping
   requires a re-test of the full pipeline (Vina / PB / diversity)
   on the new decoder output, which is Round-14 work.

## 4. Recommended wiring (5-line edit, Phase-2 integrator territory)

The minimum wiring would be in
`molmetal/adapters/flow_matching_lipman/__init__.py:_generate_impl`:

```python
# Add at line ~2018 (next to the existing BondAwareDecoder wire):
if getattr(self.config, "decoder", "bondorder") == "yuelbond":
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import YuelBondDecoder
    decoder = YuelBondDecoder()
    yb_result = decoder.featurise(AtomCloud(positions=cloud_positions, atomic_numbers=cloud_z))
    # route yb_result.pair_features through the same assembly as BondAwareDecoder
    ...
else:
    # existing BondAwareDecoder path
    ...
```

**But this is FORBIDDEN under the current task constraint** ("no
architecture changes" to `flow_matching_lipman/__init__.py`). It is
documented here as the recommended follow-up; Phase-2 integrator
territory owns the wiring.

## 5. Honest framing

- The YuelBond decoder is a **documented alternative** with 5 unit
  tests passing.
- The decoder composes with the existing `DecoderRework` /
  `ReworkedDecoder` surface without symbol collision.
- The wiring **flag** (`use_yuelbond=True` or `--decoder=yuelbond`)
  does NOT yet exist in the production pipeline.
- The metric lift requires a GPU retrain with YuelBond wired in;
  this is **deferred to Round-14** per `TODO/pending/24_cfm_architecture_redo_plan.md`.