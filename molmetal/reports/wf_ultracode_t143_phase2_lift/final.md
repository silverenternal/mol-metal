# wf_ultracode_t143_phase2_lift — Final report

**Phase**: TODO-143 §5.2 Phase 2 ultracode lift (CPU-only Multiflow → Mol-Metal adapt)
**Date**: 2026-09-27
**Status**: ✅ DONE — 33/33 new tests pass, no regressions on the 32 previously-shipped TODO-143 Phase 1 tests (65/65 total)

## Context

User instruction 2026-09-27:

> "你启动训练干什么，先kill然后启动ultracode按照todo目录里的规划做改进和适配啊"

Translation: *"Why did you start training? Kill it first, then start ultracode and follow the TODO plans for improvements and adaptations."*

Pre-existing state:
- GPU-0 (RTX PRO 6000 Blackwell 96GB) and GPU-1 (RTX 5090) both at 0% utilisation
- Wf-131 P3 training job (PID 649993) had already been killed before this session started — no training process was running when this session began
- TODO-143 Phase 1 (4 drop-in mechanisms: GroupNorm + Kendall MTL + WeightedRandomSampler + DANN GRL) was already 100% shipped with 32/32 tests passing

## What this workflow shipped

Two new CPU-only modules adapted from the local multiflow checkout (P0-236L) per TODO-143 §5.2:

### Module 1 — `molmetal/molmetal_lam/sbdd_env/sheaf_section.py`

A verbatim shape-port of `multiflow/pocket_core/contact_sheaf.py`, re-interpreting
the four-section vocabulary from *pocket-ligand contact graphs* to
*cross-molecule similarity*:

| Multiflow `ContactSection` field | Mol-Metal `MoleculeSection` field |
|----------------------------------|-----------------------------------|
| `ligand_atoms: frozenset[int]`    | `atoms: frozenset[int]`            |
| `pocket_patches: frozenset[str]`  | `fragments: frozenset[str]`        |
| `contact_edges: frozenset[tuple]` | `bonds: frozenset[tuple]`          |

Public API (verbatim Multiflow shape, semantically re-interpreted for Mol-Metal):

- `MoleculeSection(name, atoms, fragments, bonds, weight=1.0)` — frozen dataclass
- `section_from_payload(payload)` — inverse of `to_payload`
- `overlap_residual(left, right)` — Jaccard-disagreement on atoms / fragments / bonds
- `molecule_atlas_payload(sections, *, atlas_id, residual_threshold=0.35)` — cross-mol atlas gluing diagnostic

The overlap-residual formula is **byte-for-byte** the Multiflow formula:
`max(bond_disagreement, 0.5·atom_disagreement, 0.5·fragment_disagreement)`,
so the upstream `failed_overlap` threshold stays comparable across pocket
and molecule atlases (see `party-twenty-spirit` §3.5 honesty note in
the module docstring — every number traces back to Multiflow or to a
local JSON path).

### Module 2 — `molmetal/molmetal_lam/sbdd_env/cross_domain_uncertainty.py`

A verbatim port of
`multiflow/scripts/build_uncertainty_calibration_applicability_panel.py`
(443 LOC) with a **5th `metal_cytotox` domain** added to the original 4
(binding / admet / pose / postprocess). The 5-domain vocabulary lets
Mol-Metal report per-cell-line, per-metal pIC50 calibration alongside
the canonical docking / ADMET / pose / postprocess axes.

Mol-Metal-specific extensions:
1. **5th domain** `metal_cytotox` — aliases for `pic50_probability`, `pIC50`,
   `active`, `kendall_sigma`, `metal_applicability`, `metal_ood`,
   `scaffold_ood`, `cell_line_ood`. None of these exist in the original
   Multiflow script.
2. **Admet label aliases** — added per-endpoint labels (`herg_label`,
   `ames_label`, `cyp_label`, `cyp_max_label`, `dili_label`,
   `clintox_label`) to pair with the per-endpoint probability aliases
   that Multiflow already had.
3. **Output schema** — `ROW_FIELDS` adds `metal`, `cell_line`,
   `data_source` so downstream paper §4 cross-domain reporting can
   join on per-metal / per-cell-line / per-source provenance.
4. **CLI** — `build_arg_parser` + `main(argv)` mirror Multiflow's CLI
   exactly; defaults `bin_count=10`, `low_confidence_margin=0.10`.

Honest framing (party-twenty-spirit §3.5):
- Fail-closed `broad_claim_allowed` flag is preserved verbatim from
  Multiflow (any blocker → False).
- Per-domain ECE / per-row blocker counts are reported verbatim.
- Module-level docstring documents the Multiflow provenance for every
  copied function.

## Tests — 33/33 new pass

### `molmetal/tests/test_sheaf_section.py` — 16 tests

| Group | Tests |
|-------|-------|
| MoleculeSection dataclass | 4 (creation, default weight, frozen-mutation rejection, JSON-safe payload) |
| Payload round-trip | 2 (round-trip, section_id alias for legacy keys) |
| `overlap_residual` | 7 (identical-zero, disjoint atoms, disjoint fragments, Jaccard bonds, partial atoms, all keys, similarity-invert) |
| `molecule_atlas_payload` | 3 (empty, identical glues, disjoint no-overlap) |

### `molmetal/tests/test_cross_domain_uncertainty.py` — 17 tests

| Group | Tests |
|-------|-------|
| `DOMAIN_SPECS` vocabulary | 3 (5 Mol-Metal domains, metal_cytotox unique aliases, binding verbatim Multiflow) |
| Panel builder | 5 (minimal_json, all-5-clean, OOD-blocks-broad-claim, metal_cytotox rows, empty-input-fail-closed) |
| Calibration ECE | 1 (range check on perfectly-separated bin population) |
| Score→probability helper | 4 (Vina -7 → 0.5, more-negative-better, unit-interval passthrough, None handling) |
| Schema constants | 2 (Mol-Metal columns in ROW_FIELDS, BIN_FIELDS canonical) |
| CLI | 2 (parser defaults, end-to-end main invocation) |

Total: **65 tests pass** for the combined TODO-143 Phase 1 (32, pre-existing) + Phase 2 (33, new) drop-in & lift surface. No regressions.

## Honest reporting

- No training was started by this session (PID 649993 was already dead before this session began).
- No GPU work was done — both modules are pure CPU-only (RDKit + stdlib).
- The 5th `metal_cytotox` domain is a Mol-Metal **extension** of Multiflow's 4-domain vocabulary — it does NOT exist upstream and is documented in the module docstring + this report.
- Wf-128 L10 3-way R² orphan (per `TODO/INDEX.md` §17.9.2) is NOT addressed here — that is a separate workflow that needs to be created from scratch.
- Wf-74 v2 close-out (the only P0 not shipped per `party-twenty-spirit` §3.5 violation) is NOT addressed here — that is a documentation task separate from this adaptation work.

## What this workflow does NOT cover

Per honest framing (`party-twenty-spirit` §3.5):

- **Real-data demo run**: no candidate CSV/JSON was provided to `build_cross_domain_panel` in this session; the module is fully tested but not run against real cross-domain candidate dumps. That requires Mol-Metal's 5-domain evaluation pipeline to produce candidate files first.
- **Paper §4 integration**: per TODO-141 §3, the JCIM paper ship requires Phase 4 to rewrite §4 lead narrative based on Phase 1+2+3 outputs; this lift is a prerequisite but does not on its own lift any paper metric.
- **Sheaf-FRL Phase 3** (TODO-143 §6, 2-3 wk, optional high-risk): NOT attempted in this session.

## Cross-references

- `TODO/pending/143_clone_adapt_cross_domain_training.md` — §5 Phase 2 source plan
- `TODO/pending/144_q2_paper_ship_priority_roadmap.md` — Sprint A1 reuse of this work
- `TODO/pending/141_q2_jcim_completion_sprint.md` — §3 Phase 2 prerequisite
- `/home/hugo/codes/multiflow/pocket_core/contact_sheaf.py` — upstream reference for `sheaf_section.py`
- `/home/hugo/codes/multiflow/scripts/build_uncertainty_calibration_applicability_panel.py` — upstream reference for `cross_domain_uncertainty.py`
- `.claude/skills/party-twenty-spirit` — §3.5 honesty discipline applied throughout

## Final verdict

**✅ DONE** — 2 modules + 33 new tests ship; 65/65 total TODO-143 Phase 1+2 surface passes; no training was started, no GPU was used, no Multiflow code was modified-only (read-only reference).