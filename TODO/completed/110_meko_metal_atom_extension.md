# TODO-110 — Meeko metal atom extension (Pt/Ru/Ir/Os/Rh/Pd/Au/Re)

**Date**: 2026-09-27
**Status**: 🚀 **Active — unblocks metallodrug Vina dock on PlatinAI 4-cell + MetalCytoToxDB 5-metal**
**Priority**: **P0 CRITICAL** (paper-blocking — without this, all metallodrug m3 Vina dock stays at -9.6574 stub floor)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Agent 1 meko research report (2026-09-27)
**Skill alignment**: `mol-metal-cfm-training` + `mol-metal-click-chemistry`

---

## §0 Why this plan exists

Agent 1 research (2026-09-27) confirmed meeko 0.8.0 **does not support Pt/Ru/Ir/Os/Rh/Pd/Au/Re** in any of:
- `meeko/data/params/ad4_types.json` (only Mg/Ca/Mn/Fe/Zn)
- `meeko/data/params/vina_params.json` (same 5 metals)
- `meeko/data/params/metal_vdw.toml` (Fe/Cu/Ni/Zn/Co/Mg/Mn/Hg/Ca — missing 8 d-block Pt-group metals)

**Failure point**: `meeko/writer.py:481-499` `_is_molsetup_ok` rejects `setup.atom_type is None` — when no SMARTS matches Pt, atom_type stays None, writer refuses to emit PDBQT.

This is THE blocker for:
- m3 Vina dock on metallodrugs (currently stuck at stub-floor -9.6574)
- Wf-30 10-pocket sweep with cisplatin/oxaliplatin
- Wf-58 wet-lab top-20 candidates (cisplatin-class)
- Any paper §3.4 binding-affinity claim for metallodrugs

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU-only (no GPU runs) |
| Wall-time | 4-8 hours (small code changes + tests) |
| Output | `molmetal/molmetal_lam/sbdd_env/metal_pdbqt_overrides.py` + patched `vina_adapter.py` |
| Tests | ~15 new tests (per-metal atom-type rule + fallback chain) |

## §2 Solution Path (Path D — hybrid, recommended)

**Two complementary changes:**

1. **Path A**: `MoleculePreparation(add_atom_types=[...])` — provide SMARTS rules for Pt/Ru/Ir/Os/Rh/Pd/Au/Re → type as generic `"M"` / `"M1"` atoms. Vina accepts unknown atom types (logs warning, falls back to generic VDW). ~30 LOC.

2. **Path C**: When meeko still fails (e.g., complex coordination), raise `MetalPDBQTUnavailable` and fall back to `boltz_affinity` (D-MPNN pIC50, MEASURED, SMILES-only). ~10 LOC + REGISTRY entry.

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** Define SMARTS rules for d-block Pt-group | 2 hr | `metal_pdbqt_overrides.py` (NEW, ~80 LOC) | pytest: each rule matches Pt/Ru/Ir/etc. SMILES |
| **P1.2** Patch `_ligand_to_pdbqt` in `vina_adapter.py:293-323` | 1 hr | patched adapter | pytest: cisplatin SMILES → valid PDBQT |
| **P1.3** Add `MetalPDBQTUnavailable` exception + boltz fallback | 2 hr | `score_any.py` + REGISTRY entry | pytest: metallodrug SMILES falls back to boltz_affinity |
| **P1.4** Run real Vina on cisplatin + 10 pockets | 2 hr | `molmetal/reports/wf_real_metal_vina/cisplatin_per_pocket.json` | MEASURED real kcal/mol |

## §4 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:
- **MEASURED**: per-metal SMARTS rule PASS + per-pocket cisplatin real kcal/mol (post-fix)
- **PROJECTED**: cisplatin dock score < -5 kcal/mol on most pockets (realistic given ~221 nM IC50)

## §5 Honest framing + scoring-related concerns

**This IS a scoring adapter bug** — meeko + Vina pipeline fails silently for metal complexes. Path A adds graceful degradation (generic `M` atom type), Path C adds boltz fallback. Neither silently fabricates: both return `status="MEASURED_FALLBACK"` or `"MEASURED_GENERIC_ATOM"` so downstream consumers know the score is not full-fidelity Vina kcal/mol.

The D-MPNN pIC50 fallback (Path C) **replaces kcal/mol with pIC50** (different unit). The score_any registry must record this unit change so downstream consumers don't compare across scales.

## §6 Cross-references

- Agent 1 meko research report (2026-09-27)
- TODO-92 R20-P1 VariancePUCT (also scoring-adjacent)
- `molmetal/molmetal_lam/sbdd_env/vina_adapter.py:293-323` (`_ligand_to_pdbqt`)
- `meeko/data/params/ad4_types.json`, `vina_params.json`, `metal_vdw.toml` (the gap)
- `molmetal/scoring/score_any.py:288-317` (`_w_vina_score`)
- `molmetal/scoring/REGISTRY.md` + `ROUTING_POLICY.md`
- LAMBDA_ROADMAP §4.2 L1/L6 columns (Closure on metallodrugs)
- TODO-84 paper rule (paper §4 Vina tables not blocked)

## 2026-09-27 update

-9.6574 was the legacy stub-floor artifact; Wf-121 P3 ship real Vina path now active per TODO-113 honest reframe.
