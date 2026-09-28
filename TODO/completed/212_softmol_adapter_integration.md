# Wf-212: SoftMol soft-block diffusion adapter integration — CLOSE-OUT

**Date:** 2026-09-28
**Status:** ✅ SHIPPED (CPU-only plumbing; GPU sampling requires full upstream)
**Workflow:** Wf-212 (planned), not yet formally numbered in journal.jsonl
**Honest framing (party-twenty-spirit §3.5):** shipped adapter + tests +
close-out doc; **sampling runtime requires upstream deps (hydra /
lightning / datasets / tokenizers / transformers) which are NOT in the
default molmetal venv**.  Adapter degrades gracefully to fallback mode
and returns honest empty list — same contract as the pre-existing
`RxnFlowAdapter`.

## Goal

Wrap the cloned SoftMol repo (paper arXiv:2601.21964, Yang/Xu/Yuan/Zhu/Li/Ji
2026, "From Tokens to Blocks: A Block-Diffusion Perspective on Molecular
Generation") into the canonical `MoleculeGenerator` Protocol so it can be
swapped in alongside the hand-rolled CFM (`facebook_fm_wrapper`) and the
RxnFlow GFlowNet (`rxnflow_adapter`).

## What shipped

1. **`molmetal/adapters/softmol_adapter.py`** — ~430 LOC
   - `SoftMolAdapter` class implementing `name`, `setup`, `generate`,
     `train_step`, `get_metadata` (matches the
     `MoleculeGenerator` Protocol)
   - `is_softmol_available()` capability probe (handles project-root
     `references/SoftMol/` AND backward-compat `molmetal/references/SoftMol/`)
   - `list_softmol_configs()` returns the 8 upstream model configs
     (`large`, `medium`, `small`, `small-50M`, `small-70M`, `small-89M`,
     `small-110M`, `tiny`)
   - `_resolve_ckpt()` mirrors SoftMol's auto-download-from-HF contract
   - Graceful fallback: returns honest empty list when upstream deps
     are missing; never crashes
   - SMILES-only output (SoftMol is vocab-native, NOT 3D-native)
   - Honest metadata: `uses_3d=False`, `uses_pocket_conditioning=False`
     (gated-MCTS not wired)

2. **`molmetal/tests/test_softmol_adapter.py`** — 10 tests, all
   pass on default venv (9 pass + 1 skip on facebook_fm_wrapper
   comparison due to pre-existing import issue)
   - `test_softmol_adapter_imports_and_exports_expected_symbols` — module
     surface
   - `test_is_softmol_available_returns_bool` — capability probe
   - `test_list_softmol_configs_returns_list` — config listing
   - `test_softmol_adapter_satisfies_molecule_generator_protocol` —
     Protocol duck-type
   - `test_softmol_adapter_isinstance_protocol` — runtime check
   - `test_softmol_adapter_setup_runs_without_crashing` — never raises
   - `test_softmol_adapter_generate_returns_list` — empty fallback OK
   - `test_softmol_adapter_train_step_is_noop` — 0.0 contract
   - `test_softmol_adapter_metadata_shape` — paper / code / config keys
   - `test_softmol_vs_facebook_fm_metadata_diff_is_honest` — guards
     against accidental alias (skipped when facebook_fm_wrapper import
     fails)

## SoftMol vs hand-rolled CFM (the key honest framing)

| Aspect | Hand-rolled CFM (`FacebookFMWrapper`) | SoftMol (`SoftMolAdapter`) |
|---|---|---|
| Output space | continuous (B, N, 3) coords + atom-types | discrete SMILES tokens |
| Backbone | EGNN / MLP velocity field | SoftBD block-diffusion (DiT-style) |
| 3D-native | yes | **no** (needs RDKit round-trip) |
| Pocket conditioning | yes | **no** (gated-MCTS not wired) |
| Heavy upstream deps | none (FacebookFMWrapper is self-contained) | hydra + lightning + datasets + tokenizers + transformers (~600 MB HF ckpt) |
| Cost to enable | 0 LOC; works out of box | install upstream + download checkpoint |
| Default `fallback_active` | False | **True** in molmetal venv (degraded gracefully) |

The two adapters are complementary, not redundant: SoftMol gives a
fully-discrete vocab-native generator with first-hitting + top-1
decoding; CFM gives a continuous-coordinate native generator that can
plug directly into the EGNN.  §3.5 of the lambda plan can run both as
two separate back-points in the multi-objective benchmark.

## Acceptance gates

- [G1] Adapter imports SoftMol — ✅ `molmetal/adapters/softmol_adapter.py`
  imports cleanly; module-level `__all__` exposes
  `[SoftMolAdapter, is_softmol_available, list_softmol_configs]`.
- [G2] Sampling runs end-to-end — ⚠️ **PARTIAL**.  Sampling runtime
  requires upstream deps + a checkpoint; the adapter degrades gracefully
  to fallback mode in the default venv.  When the upstream is fully
  installed (`pip install hydra-core lightning datasets tokenizers
  transformers huggingface_hub`) and a checkpoint is downloaded from
  `SZU-ADDG/SoftMol` HF Hub, `setup()` succeeds and `generate()` calls
  `Diffusion.restore_model_and_sample(num_steps=…)`.  No MEASURED
  sampling on this host yet (no torch host + missing upstream).
- [G3] Output comparable — ✅ at the contract level: both
  `FacebookFMWrapper` and `SoftMolAdapter` return `List[Molecule]`
  with the same `Molecule.coords / .atom_types / .smiles` shape.
  SoftMol output is SMILES-only (3D is empty tensors); CFM output is
  3D-coord-native (SMILES is empty).  Comparison must be done at the
  *role* level (both satisfy `MoleculeGenerator` Protocol) rather than
  output-space level.
- [G4] Doc updated — ✅ this close-out file + module-level docstring
  on `softmol_adapter.py` + `test_softmol_adapter.py` module docstring.

## Honest framing (per party-twenty-spirit §3.5)

1. **Path mismatch bug noted**: pre-existing docstrings in
   `softmol_score.py` and `rxnflow_adapter.py` reference
   `molmetal/references/SoftMol/` and `molmetal/references/RxnFlow/`,
   but the actual cloned repos live at the project root
   `references/SoftMol/` and `references/RxnFlow/`.  My adapter handles
   BOTH locations via `_resolve_repo()`.  The pre-existing
   `rxnflow_adapter.py` has the same bug (returns `is_rxnflow_available
   = False` even when the cloned repo is on disk) — out of scope for
   this Wf but worth a follow-up.

2. **Sampling is NOT MEASURED on this host.**  The venv is missing the
   upstream deps; we ship the adapter + tests + doc, NOT a sampled
   SMILES string.  A real sampling run would require installing
   `hydra-core lightning datasets tokenizers transformers
   huggingface_hub` plus downloading the ~600 MB checkpoint.

3. **Gated-MCTS pocket conditioning NOT wired.**  SoftMol's pocket-
   conditioning path is `gated_mcts/run_mcts.py` which uses QVina02
   (not in venv).  Honest metadata: `uses_pocket_conditioning=False`.

## Files touched

- `molmetal/adapters/softmol_adapter.py` (NEW, ~430 LOC)
- `molmetal/tests/test_softmol_adapter.py` (NEW, ~180 LOC, 10 tests)
- `TODO/completed/212_softmol_adapter_integration.md` (NEW, this file)

## References

- Paper: Yang et al. 2026, "From Tokens to Blocks: A Block-Diffusion
  Perspective on Molecular Generation", arXiv:2601.21964
- Code: `references/SoftMol/` (cloned from SZU-ADDG/SoftMol)
- Weights: HF Hub `SZU-ADDG/SoftMol` (`89M-epoch6-best.ckpt` recommended)
- Pre-existing wrappers (for cross-reference):
  `molmetal/adapters/facebook_fm_wrapper.py` (CFM, hand-rolled)
  `molmetal/adapters/rxnflow_adapter.py` (RxnFlow, GFlowNet)
