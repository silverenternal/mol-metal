# Wf-310D PART 2 — Vina pt-strip full aggregation plan

Date: 2026-09-29
Wf: Wf-310D (Wf-301B full aggregation)
Owner: Hugo Lee + ultracode agent
Status: aggregation template ready; awaits Phase-B full run output.

## Purpose

Aggregate the Wf-301B full 100-mol × 4-pocket GPU Vina pt-strip run output
(launched via `launch_wf306a_drugood_vina_gpu.sh --phase B`) into a
paper-grade aggregate + claim cards, with honest MEASURED-vs-PROJECTED
framing.

## Inputs

| Path | Owner | Notes |
|------|-------|-------|
| `molmetal/reports/wf306a_drugood_vina_launch/phase_b/panel.json` | Wf-306A Phase-B | full panel |
| `molmetal/reports/wf301b_vina_pt_strip/vina.json` | Wf-301B smoke | cross-comparison checkpoint |

## Outputs

| Path | Format | Purpose |
|------|--------|---------|
| `molmetal/reports/wf310d_vina_full/aggregate.json` | JSON | paper-grade aggregate |
| `molmetal/reports/wf310d_vina_full/claim_cards.md` | Markdown | 5-card claim sheet |

## Pass-gate verdict convention (per Wf-301B honest-negative)

| Verdict | Condition | Meaning |
|---------|-----------|---------|
| PASS_FIELD_AVG | grand mean ∈ [-9.5, -7.5] kcal/mol AND n_docked == n_total | full band match + complete docking |
| HONEST_NEGATIVE_BAND | n_docked == n_total BUT grand mean ∉ band | pt-strip enables docking; binding affinity axis is a follow-up |
| HONEST_NEGATIVE_DOCK | n_docked < n_total | pt-strip recovery not fully realised |

## Honest-negative framing (per Wf-301B honest-negative)

1. **Field-avg band -9.5 to -7.5 kcal/mol**: standard medicinal-chemistry
   expected binding range.  Per Wf-301B smoke mean = -4.60 kcal/mol,
   ABOVE the band; recovery axis is *being able to dock at all*, not
   absolute binding affinity.
2. **Top-1 -10.299 vs -9.5**: top-ranked compound exceeds the band even
   when grand mean is outside it.  Always report top-1 separately.
3. **Grand mean -5.564**: full panel may produce grand means BELOW the
   band; do not collapse to single-number headline.
4. **Vina is CPU-only**: `cuda:1` is unused during Phase-B; Phase-A + B
   cannot run concurrently for time-budget accounting.

## Aggregation template

```bash
# 1. After Wf-306A Phase-B completes:
python molmetal/scripts/wf310d_vina_full_aggregate.py \
    --panel molmetal/reports/wf306a_drugood_vina_launch/phase_b/panel.json \
    --out-dir molmetal/reports/wf310d_vina_full

# 2. Inspect:
cat molmetal/reports/wf310d_vina_full/aggregate.json | python -m json.tool | head -60
cat molmetal/reports/wf310d_vina_full/claim_cards.md
```

## Per-pocket summary example

```text
| pocket | n_docked | mean_kcal_mol | std | best_kcal_mol | in_band |
|--------|----------|---------------|-----|---------------|---------|
| 1h36   | 100      | -7.812        | 1.234 | -10.299     | yes     |
| 4rlu   | 100      | -7.345        | 0.987 | -9.456      | yes     |
| 3eml   |  98      | -6.123        | 1.456 | -8.789      | no      |
| 1iep   | 100      | -8.012        | 1.012 | -10.456     | yes     |
```

## Wall budget (PROJECTED per Wf-306A)

- Per-pocket: ~25 min (100 mols × ~15 s/mol on 1 pocket).
- 4 pockets: ~1.5 hr CPU.
- Receptor pre-dock: ~15 min total.
- Total: ~1-2 hr (per Wf-306A Phase-B budget).

## See also

- `molmetal/scripts/wf310d_drugood_dann_full_aggregate.py` (PART 1 template)
- `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` (Phase-B launcher)
- `molmetal/scripts/wf305d_vina_pt_strip.py` (per-mol trainer)
- TODO/pending/130_wf130_drugood_dann_followup.md (Wf-301B task spec)
