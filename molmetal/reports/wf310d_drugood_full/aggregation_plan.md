# Wf-310D PART 1 — DrugOOD DANN full aggregation plan

Date: 2026-09-29
Wf: Wf-310D (Wf-301A full aggregation)
Owner: Hugo Lee + ultracode agent
Status: aggregation template ready; awaits Phase-A full run output.

## Purpose

Aggregate the Wf-301A full 5-seed × 50-epoch GPU DrugOOD DANN run output
(launched via `launch_wf306a_drugood_vina_gpu.sh --phase A`) into a
paper-grade aggregate + claim cards, with honest MEASURED-vs-PROJECTED
framing.

## Inputs

| Path | Owner | Notes |
|------|-------|-------|
| `molmetal/reports/wf301a_drugood_dann/seed<N>.json` (N=0..4) | Wf-301A full | 5-seed × 50-epoch |
| `molmetal/reports/wf301a_drugood_dann/aggregate.json` | Wf-301A smoke | cross-comparison checkpoint |

## Outputs

| Path | Format | Purpose |
|------|--------|---------|
| `molmetal/reports/wf310d_drugood_full/aggregate.json` | JSON | paper-grade aggregate |
| `molmetal/reports/wf310d_drugood_full/claim_cards.md` | Markdown | 4-card claim sheet |

## Pass-gate verdict convention (per Wf-301A)

| Verdict | Condition | Meaning |
|---------|-----------|---------|
| PASS_WF185 | `test_mean >= 0.6707` | matches Wf-185 v2 baseline (cross-substrate) |
| PASS_FIELD_AVG | `test_mean in [0.55, 0.6707)` | field-avg band; partial recovery |
| PASS_FIELD_AVG_ONLY | `test_mean in [0.55, 0.65]` | field-avg band only |
| HONEST_NEGATIVE | `test_mean < 0.55` | scaffold-bin DANN does NOT rescue Wf-253-D |

## Honest-negative framing (per Wf-301A honest-negative)

1. **Cross-substrate Δ vs Wf-185 v2**: Wf-185 v2 is on
   `lbap_refined_ic50_scaffold` (different substrate from DrugOOD scaffold).
   The Δ is **directional** only; do not interpret as apples-to-apples.
2. **Full vs smoke Δ**: mixes epoch-budget (1 → 50) with seed-noise (3 → 5).
   Use only as a within-DANN consistency check.
3. **Top-15 + OOV scaffold binning**: may not generalise beyond the most-
   frequent Bemis-Murcko scaffolds.  Long-tail scaffolds collapse to OOV
   bin, weakening the adversarial signal.

## Aggregation template

```bash
# 1. After Wf-306A Phase-A completes:
python molmetal/scripts/wf310d_drugood_dann_full_aggregate.py \
    --per-seed-dir molmetal/reports/wf301a_drugood_dann \
    --smoke-aggregate molmetal/reports/wf301a_drugood_dann/aggregate.json \
    --out-dir molmetal/reports/wf310d_drugood_full \
    --n-seeds 5

# 2. Inspect:
cat molmetal/reports/wf310d_drugood_full/aggregate.json | python -m json.tool | head -60
cat molmetal/reports/wf310d_drugood_full/claim_cards.md
```

## Pass-gate verdict example

```text
TEST AUC = 0.8123 ± 0.0142
vs Wf-185 v2 = 0.6707 ± 0.0042
vs Wf-301A smoke = 0.8024
verdict=PASS_FIELD_AVG -- DANN full 0.8123 in field-avg band; Wf-185 not matched (cross-substrate).
```

## Wall budget (PROJECTED per Wf-306A)

- Per-seed: ~40 min on RTX 5090 (50 epoch × 95k rows × chemprop v2 ECFP4 + DANN).
- 5 seeds: ~3-4 hr GPU-1 (per Wf-306A Phase-A budget).
- Aggregate post-processing: <1 min CPU.

## See also

- `molmetal/scripts/wf310d_vina_full_aggregate.py` (PART 2 template)
- `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` (Phase-A launcher)
- `molmetal/scripts/wf301a_drugood_dann.py` (per-seed trainer)
- TODO/pending/130_wf130_drugood_dann_followup.md (Wf-301A task spec)
