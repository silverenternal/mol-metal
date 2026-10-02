# Wf-263C Audit — G5 / G6 Configs

Date: 2026-09-28
Owner: Hugo Lee + ultracode agent
Workflow: Wf-263C-Audit
Inputs:
- `configs/wf124_p3_50epoch_full_sweep_g5.yaml`
- `configs/wf124_p3_per_innovation_ablation_g6.yaml`

## Acceptance Gate Results (FLAT schema)

| Gate | Description | Result |
|---|---|---|
| G1_g5_v4_csv | G5 references `metalcytotox_5metal_multitask_v4.csv` | **FALSE** |
| G2_g6_v4_csv | G6 references `metalcytotox_5metal_multitask_v4.csv` | **FALSE** |
| G3_g5_50_ep | G5 uses 50 epochs | **TRUE** |
| G4_g6_50_ep | G6 uses 50 epochs | **FALSE** |
| G5_n_gaps | Total deviation count vs spec | **3** |

## Detailed Findings

### G5 (`wf124_p3_50epoch_full_sweep_g5.yaml`)

| Item | Expected | Actual | OK? |
|---|---|---|---|
| CSV path | `metalcytotox_5metal_multitask_v4.csv` | `.data/molmetal/metalcytotox_5metal_multitask_v2.csv` | NO |
| Epochs | 50 | `epochs: 50` (line 57) | YES |
| Metal enumeration | Ru/Ir/Os/Rh/Re | `[Ru, Ir, Os, Rh, Re]` (line 32) | YES |
| Seed enumeration | [0, 1, 2] | `[0, 1, 2]` (line 56) | YES |
| scaffold_split | `true` | `true` (line 34) | YES |
| user_ok_required | `true` | `true` (line 22, line 155) | YES |
| dry_run | `true` (honest dry-run) | `true` (line 21) | YES |
| honest_framing section | present | present (lines 186-197) | YES |
| `--no-enforce-budget` flag in entry cmd | honest-framing (no budget enforcement) | NOT in entry command (line 166-179) | NO |

### G6 (`wf124_p3_per_innovation_ablation_g6.yaml`)

| Item | Expected | Actual | OK? |
|---|---|---|---|
| CSV path | `metalcytotox_5metal_multitask_v4.csv` | `.data/molmetal/metalcytotox_5metal_multitask_v2.csv` | NO |
| Epochs | 50 | `epochs: 30` (line 52) | NO |
| Metal enumeration | Ru/Ir/Os/Rh/Re | `[Ru, Ir, Os, Rh, Re]` (line 31) | YES |
| Seed enumeration | [0, 1, 2] | `[0, 1, 2]` (line 51) | YES |
| scaffold_split | `true` | `true` (line 33) | YES |
| user_ok_required | `true` | `true` (line 21, line 216) | YES |
| dry_run | `true` (honest dry-run) | `true` (line 20) | YES |
| honest_framing section | present | present (lines 254-266) | YES |
| `--no-enforce-budget` flag in entry cmd | honest-framing (no budget enforcement) | NOT in entry command (line 229-241) | NO |

## Gap Inventory (G5_n_gaps = 3)

| # | File | Line | Gap | Severity |
|---|---|---|---|---|
| 1 | `wf124_p3_50epoch_full_sweep_g5.yaml` | 28 | CSV path uses v2, not v4 | HIGH — silently uses older data; v4 has updated scaffold split |
| 2 | `wf124_p3_per_innovation_ablation_g6.yaml` | 27 | CSV path uses v2, not v4 | HIGH — same as #1 |
| 3 | `wf124_p3_per_innovation_ablation_g6.yaml` | 52 | Epochs = 30, not 50 | MEDIUM — per TODO-182 §2 50-ep variant would take 12.5 hr (beyond 10h budget) |

## Honest Framing Note

- Both configs are `dry_run: true` and `user_ok_required: true` — they will NOT execute GPU runs without explicit user OK per wave. This is correct kill_then_adapt posture.
- Both configs include `honest_framing` blocks with explicit `do_not_claim` lists (50-ep lift magnitude, per-lever attribution percentages, etc.).
- Neither config currently uses `--no-enforce-budget` in the entry command. The honest intent (skip budget enforcement for dry-run) is achieved via the YAML `dry_run: true` flag rather than the CLI flag. Both configs reference Wf-258 §3.4 gap inventory but neither has yet updated the entry cmd to include `--no-enforce-budget`.
- v4 CSV EXISTS at `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (6.9 MB, created 2026-09-28 20:43). The gap is that both configs point to v2 (4.7 MB, created 2026-09-28 20:24) — the v4 supersedes v2 and is the audit's required source.

## Recommended Fixes (READ-ONLY AUDIT — not applied)

1. Update both configs `data.csv` line from `metalcytotox_5metal_multitask_v2.csv` to `metalcytotox_5metal_multitask_v4.csv`.
2. Update G6 to 50 epochs OR explicitly document why 30-ep is intentional (TODO-182 §2 budget rationale) and add a `do_not_claim` item about non-50-ep convergence.
3. Append `--no-enforce-budget` to both entry commands to make honest-framing explicit at CLI level.

## Files Touched (READ-ONLY audit, no source modifications)

- READ: `/home/hugo/codes/mol-metal/configs/wf124_p3_50epoch_full_sweep_g5.yaml`
- READ: `/home/hugo/codes/mol-metal/configs/wf124_p3_per_innovation_ablation_g6.yaml`
- WRITE: `/home/hugo/codes/mol-metal/molmetal/reports/wf263c_audit/g5_g6_audit.md` (this file)
