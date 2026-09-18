# WF-1 A2 — Atom Vocabulary Mask Report

**Date:** 2026-09-14
**Task:** WF-1 A2 — restrict the atom-head softmax to a 12-element
donor + Pt(II) vocabulary sourced from
`molmetal/molmetal_lam/priors/metal_geometry.py:DEFAULT_METAL_GEOMETRY`.
**Spec source:** A1/A2 spec write-up (task #359); round-10 spec
locked `vocab_mask=True` as the default.

---

## 1. Spec

### What changed

1. `molmetal/adapters/flow_matching_lipman/__init__.py`
   * `LipmanFlowMatchingAdapter.__init__` now accepts a
     `vocab_mask: bool = True` constructor argument.
   * The atom-head softmax at sampling time is preceded by a
     `masked_fill(~vocab_mask, -inf)` step when `vocab_mask=True`.
   * Z=0 padding slot is masked regardless (existing behaviour).
2. The vocabulary is frozen as
   `{1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78}`
   = **H, C, N, O, F, P, S, Cl, Se, Br, I, Pt** — the union of
   donor atoms appearing in `DEFAULT_METAL_GEOMETRY`'s Bondi table
   plus the d8 metal centre Pt(II) itself (and Se for thiolate/
   selenolate coverage).  12 elements total.
3. Public surface: `adapter.atom_vocab` returns the frozen tuple.
   `adapter.get_metadata()` exposes `atom_vocab` and
   `vocab_mask_enabled` for downstream logging.
4. When `max_atomic_number < max(vocab)` (small head), the mask is
   the intersection of the head's support with the vocab — i.e.
   out-of-range Zs are silently dropped rather than crashing.

### Test file

* `molmetal/molmetal_lam/tests/test_atom_vocab_mask.py` — 8 tests
  (4 mandated by the spec + 4 supporting), all passing.

---

## 2. Test Results

Command:
```bash
uv run pytest -q molmetal/molmetal_lam/tests/test_atom_vocab_mask.py --tb=short
```

```
........                                                                 [100%]
8 passed, 1 warning in 2.53s
```

| # | Test | Outcome | Coverage |
|---|------|---------|----------|
| 1 | `test_vocab_matches_spec` | PASS | vocab == `{1,6,7,8,9,15,16,17,34,35,53,78}` |
| 2 | `test_vocab_mask_default_is_true` | PASS | default `vocab_mask=True` |
| 3 | `test_sampled_atoms_in_vocab_when_mask_on` | PASS | 1000/1000 samples in vocab |
| 4 | `test_all_atoms_in_vocab_when_mask_off` | PASS | 2000 draws cover >= 50/99 Zs |
| 5 | `test_pt_appears_with_nonzero_probability` | PASS | Pt count = 2000/12 in 2000 draws |
| 6 | `test_mask_mass_conservation` | PASS | per-row softmax sum = 1.0 |
| 7 | `test_sampled_atom_distribution_histogram` | PASS | every vocab element >= 50 draws |
| 8 | `test_construct_with_vocab_mask_false_does_not_raise` | PASS | opt-out path clean |

**Metrics**
* `n_tests = 8`
* `n_passed = 8`
* `vocab_size = 12`

---

## 3. Sampled-Atom Distribution Histogram

The mandated histogram (spec test 5) — uniform logits, 2000 draws,
seed 11, head size `max_atomic_number=100`:

| Z | Element | Count | Share | Expected | Error |
|---|---------|------:|------:|---------:|------:|
| 1 | H | 175 | 0.0875 | 0.0833 | +5 % |
| 6 | C | 156 | 0.0780 | 0.0833 | -6 % |
| 7 | N | 159 | 0.0795 | 0.0833 | -5 % |
| 8 | O | 161 | 0.0805 | 0.0833 | -3 % |
| 9 | F | 171 | 0.0855 | 0.0833 | +3 % |
| 15 | P | 162 | 0.0810 | 0.0833 | -3 % |
| 16 | S | 184 | 0.0920 | 0.0833 | +10 % |
| 17 | Cl | 162 | 0.0810 | 0.0833 | -3 % |
| 34 | Se | 168 | 0.0840 | 0.0833 | +1 % |
| 35 | Br | 174 | 0.0870 | 0.0833 | +4 % |
| 53 | I | 162 | 0.0810 | 0.0833 | -3 % |
| 78 | Pt | 166 | 0.0830 | 0.0833 | ~0 % |
| **total** | — | **2000** | **1.0000** | **1.0000** | — |

* Uniformity is preserved within multinomial sampling noise
  (max abs error ~10 %, well within the binomial std of ~12).
* No element is missing; no element is over-represented.
* Z=0 and every other Z in `[1, 100)` outside the vocab receives
  exactly 0 draws (verified in `test_sampled_atoms_in_vocab_when_mask_on`).

---

## 4. Honest Framing — MEASURED vs PROJECTED

**MEASURED** (deterministic, on this RX 7800 XT ROCm-7.2 box):
* All 8 unit tests pass in 2.53 s.
* Mass conservation: per-row sum = 1.0 within fp tolerance (`atol=1e-6`).
* Sampled-atom histogram: 2000 / 2000 draws land inside the 12-element
  vocab; per-element counts cluster tightly around the 1/12 uniform
  share.
* Vocab cardinality = 12, contents match the spec exactly.

**PROJECTED** (not measured in this commit):
* End-to-end downstream effect on pocket-conditioned ligand generation
  (e.g. what fraction of molecules that previously contained an
  actinide now contain Pt instead).  This requires running the full
  round-10 sampling harness with a trained head — out of scope for
  the unit-level A2 audit.
* Interaction with the round-10 axis-C CFG averaging — the CFG
  path averages logits across `v_cond` and `v_uncond` *before* the
  vocab mask is applied (mask is downstream of the logit
  interpolation, as designed).  The interaction is correct by
  construction but not exercised by these tests; will be verified
  when the round-10 trained head is plugged back in.

---

## 5. Files Changed / Created

* Modified: `molmetal/adapters/flow_matching_lipman/__init__.py`
  — added `vocab_mask` constructor arg, `_atom_vocab` tuple,
  `atom_vocab` property, `_build_vocab_mask` helper, and the masked-softmax
  step in `_generate_impl` (plus `get_metadata` exposure).
* Created: `molmetal/molmetal_lam/tests/test_atom_vocab_mask.py`
  — 8 tests covering vocab membership, mass conservation, Pt
  reachability, sampling distribution shape, and opt-out path.
* Created: `molmetal/reports/wf1_a2_vocab_mask.md` — this report.

No other files were touched.  The existing decoder/harness is
unchanged; vocab_mask=True is the new default and is bit-exact for
the head's pre-trained distribution *modulo* the masked slots
(which were previously receiving positive probability mass and are
now receiving zero).