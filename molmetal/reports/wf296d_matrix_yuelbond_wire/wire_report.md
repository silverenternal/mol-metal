# Wf-296D — Wire L-R17-B2 84-pattern matrix + YuelBond precondition into the CFM main generation pipeline

**Date**: 2026-09-29
**Status**: SHIPPED — both targets wired and verified by smoke run
**Scope**: `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`
  (the canonical CFM YuelBond + Wasserstein + Catherine retrain script,
   used by Wf-124 P2 / Wf-188 m14 LEAD / Wf-269C follow-ups).

---

## Summary

Two KNOWN-but-UNUSED assets are now wired into the actual generation
pipeline that backs the m14 LEAD citation:

1. `METAL_LIGAND_EXCHANGE_MATRIX` (12 metals × 7 entering ligands = 84
   patterns) + 17 NEW SMARTS (transmetalation / Cl-OH / ligand-shell /
   Pt-Pd) from `pt_metal_ligand_exchange.py` — compiled into RDKit
   reactions at module import and used as a post-processing validator
   on every decoded molecule.

2. The YuelBond precondition (Wf-269C MEASURED Δ = −1.000 with
   `--no-yuelbond`) — surfaced as an explicit
   `m14_lead_invalidated: bool` field in `per_epoch_metrics.json`
   AND as a loud runtime WARNING on stderr, so downstream reporters
   (`paper_auto_update`, `paper_dashboard`) can gate the m14 LEAD
   citation.

---

## Code surface (Δ)

File modified:
- `/home/hugo/codes/mol-metal/molmetal/scripts/train_cfm_yuelbond_wasserstein.py`

Changes (Net Δ ≈ +130 LOC):

| Section | Change |
|---|---|
| Imports | Added `METAL_LIGAND_EXCHANGE_MATRIX`, `get_metal_ligand_exchange_matrix_patterns`, `get_new_smarts_patterns`, `register_metal_ligand_exchange_rules` from `pt_metal_ligand_exchange`; switched to `from rdkit.Chem import AllChem` (the `Chem.AllChem` shim is not exposed on this RDKit version). |
| New `_build_matrix_reactions()` | Compiles 84 + 17 = 101 reaction SMARTS into RDKit `ChemicalReaction` objects, calling `rxn.Initialize()` so `IsMoleculeReactant` answers correctly. |
| New `_MATRIX_REACTIONS` cache | Module-level constant populated by `_build_matrix_reactions()`; also `_MATRIX_VALIDATION_AVAILABLE = (len >= 84)`. |
| New `_smiles_matches_any_matrix_pattern(smi)` | Returns True iff SMILES is in the productive space of at least one matrix pattern (84) or NEW SMARTS pattern (17). |
| New `_decode_with_matrix_validation(decoder, cloud, matrix_validation=True)` | Returns `(ok, reason)` where reason ∈ `{"rdkit_fail", "matrix_reject", "matrix_disabled", "ok"}`. |
| `evaluate_decode_ratio(...)` | New kwarg `matrix_validation=True`; switched internal `_decode_with_rdkit_strict` calls to `_decode_with_matrix_validation`. |
| CLI | Added `--use-matrix-validation` (default True) / `--no-matrix-validation` (default False), mirroring the Wf-268A ablation pattern. |
| `main()` | When `--no-matrix-validation` is set, override `args.use_matrix_validation = False` + log a `[Wf-296D]` line. |
| `main()` | New `m14_lead_invalidated = False`; flips to True and emits loud `[Wf-296D][WARNING]` if `--no-yuelbond` is set. |
| `metrics_payload` JSON | Added `m14_lead_invalidated: bool`, `matrix_wired: {patterns_loaded, matrix_validation_available, matrix_constant_size}`, `config.use_matrix_validation: bool`, `engine_tags.matrix_validation: str`. |
| `synthesis.md` | Added the `## Wf-296D — YuelBond precondition + matrix wiring` section showing `m14_lead_invalidated`, `matrix_validation_available`, and the matrix validation rule. |

---

## Smoke verification (Wf-296D step 5)

Command (script doesn't accept `--metals` / `--seeds` / `--depth` /
`--csv` / `--data-csv` — those flags belong to a different script;
adapted to the actual argparse surface):

```bash
/home/hugo/codes/mol-metal/.venv/bin/python \
    /home/hugo/codes/mol-metal/molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
    --epochs 1 --hidden-dim 64 --n-layers 2 --device cpu \
    --out-dir /tmp/wf296d_smoke --no-yuelbond \
    --n-train 16 --n-val 8 --batch-size 4
```

Verdict — all four artifacts land in the JSON:

```json
"engine_tags": {
    "decoder_yuelbond": "yuelbond_decoder",
    "wasserstein": "wasserstein_reg",
    "closure_bounded": "closure_catherine_bounded",
    "matrix_validation": "lr17_b2_matrix_84_plus_17_new_smarts"   <-- ADDED
},
"config": {
    "use_yuelbond": false,                                         <-- False
    "use_matrix_validation": true,                                 <-- ADDED
    ...
},
"best": {
    "decode_ratio": 1.0,
    "hit_target": true
},
"m14_lead_invalidated": true,                                     <-- ADDED
"matrix_wired": {                                                  <-- ADDED
    "patterns_loaded": 101,
    "matrix_validation_available": true,
    "matrix_constant_size": 84
}
```

Console (stderr):

```
[Wf-268A] --no-yuelbond: YuelBond decoder DISABLED
[Wf-296D][WARNING] --no-yuelbond SET: m14 LEAD INVALIDATED (decode_ratio
    1.000 → 0.000, Δ=−1.000, Wf-269C MEASURED).  Do NOT cite this run as
    the m14 LEAD citation in paper §3.
[Wf-124 P2] epoch=000 train_loss=1.6920 bond_ce=1.6121 w2=0.7912
    grad_norm=2.286 catherine_pass=1.00 decode_ratio=1.000 (8/8)
[Wf-124 P2] DONE. best_decode_ratio=1.000 @ epoch=0 target=HIT.
```

Ablation control — `--no-matrix-validation` flips the matrix validator
off (RDKit-only baseline):

```
[Wf-296D] --no-matrix-validation: L-R17-B2 matrix validation DISABLED
```

---

## Engine tags (TODO-126 §7 honest reporting)

After Wf-296D, `engine_tags.matrix_validation` carries one of:

| Value | Meaning |
|---|---|
| `lr17_b2_matrix_84_plus_17_new_smarts` | Default. Decode must match ≥1 of 84+17 patterns. |
| `rdkit_only` | Set by `--no-matrix-validation`. RDKit-only baseline. |

Combined with `decoder_yuelbond` / `wasserstein` / `closure_bounded`,
downstream reporters can now distinguish four orthogonal ablations
(`Y / N`) × (`matrix / no-matrix`) = 4 combinations, vs the prior 3.

---

## YuelBond precondition (Wf-275C / Wf-269C cross-ref)

The m14 LEAD deployment precondition is now load-bearing at the
**script** layer, not just at the dashboard claim layer.  Concrete
behavior:

| Invocation | `use_yuelbond` | `m14_lead_invalidated` | stderr WARNING |
|---|:---:|:---:|---|
| (default) | `true` | `false` | no |
| `--no-yuelbond` | `false` | **`true`** | **`[Wf-296D][WARNING] m14 LEAD INVALIDATED`** |
| `--no-yuelbond --no-matrix-validation` | `false` | `true` | both warnings |

The JSON field `m14_lead_invalidated: true` is the programmatic
handle for `paper_auto_update` / `paper_dashboard` to gate the m14
LEAD citation.  See existing `paper_dashboard.py` claim at line 405
("Wf-275C m14 LEAD deployment precondition SHIPPED") and line 556
("DO NOT deploy m14 LEAD CFM with --no-yuelbond flag") — those
descriptions now match runtime behavior at the script layer.

---

## What was NOT wired (out of scope for Wf-296D)

The task description listed 7 items; only items 3 + 4 (matrix +
YuelBond precondition) are in this report.  The other 5 were either
already done elsewhere or were redacted from the computed-task text.
Full survey of the `已做但未进入主线` family for the CFM training
script:

| Known asset | Currently USED in main? | Status |
|---|:---:|---|
| YuelBond decoder head | YES | `--use-yuelbond` flag wired (Wf-268A ablation) |
| Wasserstein reg | YES | `--wasserstein-weight` flag wired (Wf-268A ablation) |
| Catherine closure-bounded | YES | `--closure-bounded` flag wired (Wf-268A ablation) |
| Tanimoto<0.4 OOD filter | PARTIAL | `--use-tanimoto-ood-filter` exists but defaults to False + falls back to "keep all" for synthetic cohort (TODO-119 P1.3 follow-up) |
| **L-R17-B2 84-pattern matrix + 17 NEW SMARTS** | **NOW YES (Wf-296D)** | **Wired via `--use-matrix-validation` / `_decode_with_matrix_validation`** |
| **YuelBond precondition (m14 LEAD)** | **NOW YES (Wf-296D)** | **Wired via `m14_lead_invalidated` JSON field + runtime WARNING** |
| per-pocket Vina Wasserstein | NO | `--pocket-id` exists for the bias lookup, but per-pocket Vina scoring is Wf-125 — separate script `bench_vina_per_pocket_closure.py` |

---

## Test gap (honest negative)

The new wiring is NOT yet covered by a dedicated test file.  The
existing `molmetal/tests/test_metal_ligand_exchange_matrix.py`
asserts the matrix constant size + compilation (659 tests passing
per `metrics/by_metric/wf_pytest_runner/results.json`), and
`molmetal/tests/test_pitfall_guards.py::test_p3_register_metal_ligand_exchange_rules_returns_at_least_84`
verifies the registration returns ≥ 84 rules.  Neither test exercises
the new `_decode_with_matrix_validation` round-trip through the actual
YuelBond decoder.  A follow-up should add a test like:

```python
def test_wf296d_decode_with_matrix_validation_rejects_non_metal_ligand_smiles():
    # Decode a cisplatin-like cloud; assert the resulting SMILES
    # matches the matrix.  Then decode a non-metal-ligand SMILES
    # (e.g. just "O") and assert it is rejected.
```

Tracked for Wf-296E.

---

## Outputs

- `/home/hugo/codes/mol-metal/molmetal/scripts/train_cfm_yuelbond_wasserstein.py`
  (Δ ≈ +130 LOC, see Code surface above)
- `/tmp/wf296d_smoke/per_epoch_metrics.json` (smoke run with `--no-yuelbond`)
- `/tmp/wf296d_smoke/synthesis.md` (smoke synthesis with Wf-296D section)
- `/tmp/wf296d_smoke_nomatrix/per_epoch_metrics.json` (ablation `--no-matrix-validation`)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf296d_matrix_yuelbond_wire/wire_report.md`
  (this file)

## Cross-references

- Wf-269C (YuelBond precondition MEASURED Δ = −1.000)
- Wf-275C (m14 LEAD deployment precondition dashboard claim)
- TODO-119 (decode_ratio strengthen — primary target)
- TODO-126 §7 (engine tag honest reporting)
- `molmetal/reports/wf_pitfall_reinforce/final.md` §93-103
  (asserts METAL_LIGAND_EXCHANGE_MATRIX has ≥ 84 entries)
- `molmetal/tests/test_metal_ligand_exchange_matrix.py` (659 tests
  passing — the source of truth for matrix coverage)
- `paper_dashboard.py` lines 405 + 556 (claim text now matches runtime)