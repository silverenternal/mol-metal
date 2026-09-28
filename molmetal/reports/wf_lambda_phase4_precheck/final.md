# WF-Lambda Phase-4 Precheck — 18-reaction rule registry wiring
Date: 2026-09-25. Verifier: claude-code.

## Step-by-step verdict
1. `list_registered_rules` (pt_click_compat) — PASS. Returns
   ('CuAAC','SPAAC','SPC','DielsAlder','ThiolEne'). Added in precheck.
2. `build_synthemol_rules()` — PASS. 18 SyntheMol rules (13 legacy
   + 5 from 5→18 extension). Task asks ≥13; 18 OK.
3. `REACTION_RULES` — PASS. 27 entries: 9 click/metal-coord (CuAAC,
   SPAAC, SPC, DielsAlder, ThiolEne, Suzuki, AmideCoupling,
   MetalLigandExchange, AquaExchange) + 18 SyntheMol wrappers.
4. Canonical-reactant apply for 5 click rules — PASS. Each rule
   produces ≥1 RDKit-sanitisable product on a canonical educt pair.
5. `SynFlowNetEnv.forward_smoke_test()` — PASS.
   `OK | rules=27 | forward=ok | backward=ok | synflownet=missing`.

## Findings
- Registry fully wired. 13 SyntheMol → 18 (tetrazole,
  benzimidazole, thiazole, oxadiazole, hantzsch-pyridine added);
  hantzsch-pyridine skipped on 2-reactant canonical pair (MCR)
  per `_SKIP_CANONICAL`.
- Gap filled: `list_registered_rules` was missing in pt_click_compat;
  added with docstring linking to `CLICK_RULE_NAMES` and
  `lam_chem.rules.list_reactions`.
- `synflownet=missing` expected (torch_geometric absent; SMARTS-only
  mode is the supported path on the shared venv).

## Files touched
- `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` (added
  `list_registered_rules`).

## Headline
"18" target → today's registry is **27** (5 legacy click + 2 R10
click + 2 metal-coord + 18 SyntheMol). All checks pass.