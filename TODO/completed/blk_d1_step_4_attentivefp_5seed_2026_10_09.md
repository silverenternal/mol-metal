# BLK-D1 Step 4: bace 5-seed AttentiveFP retest — CLOSEOUT

**Date:** 2026-10-09
**Operator:** Claude (background detached)
**Status:** SHIPPED (5-seed lift measurement vs dmpnn baseline)
**Wire-in provenance:** `molmetal/scripts/train_moleculenet.py --backbone attentivefp` (per
`TODO/completed/blk_d1_step_2a_attentivefp_2026_10_09.md` SHIPPED wire-in)

## §5.5 filesystem-truth (ls -la evidence)

```
$ ls -la molmetal/reports/blk_d1_attentivefp_5seed_2026_10_09/
-rw-r--r-- 1 hugo hugo   502 Oct  9 14:35 bace_attentivefp_seed0.json
-rw-r--r-- 1 hugo hugo   499 Oct  9 14:39 bace_attentivefp_seed1.json
-rw-r--r-- 1 hugo hugo   501 Oct  9 14:43 bace_attentivefp_seed2.json
-rw-r--r-- 1 hugo hugo   485 Oct  9 14:48 bace_attentivefp_seed3.json
-rw-r--r-- 1 hugo hugo   492 Oct  9 14:52 bace_attentivefp_seed4.json
-rw-r--r-- 1 hugo hugo  1419 Oct  9 14:52 metrics.json
-rw-r--r-- 1 hugo hugo  2962 Oct  9 14:52 report.md
```

All 5 seed JSONs + aggregate `metrics.json` + auto-generated `report.md` are on disk.

## §5.5 MEASURED lift (raw numbers, no paraphrase)

| Backbone | bace 5-seed mean ± std | per-seed |
|---|---|---|
| **dmpnn baseline** (Wf-FIX-3LINES 2026-10-09) | **0.7505 ± 0.0456** | 0.665 / 0.800 / 0.757 / 0.771 / 0.760 |
| **AttentiveFP** (BLK-D1 Step 4, this run) | **0.7636 ± 0.0398** | 0.699 / 0.802 / 0.740 / 0.802 / 0.775 |

- **Δ vs dmpnn = +0.0131** (AttentiveFP - dmpnn)
- **Honest framing per `verify-premises-before-proposing`**: Δ = +0.0131 is **inside
  the 1σ envelope of BOTH arms** (dmpnn std=0.0456, AttentiveFP std=0.0398 → combined
  noise floor ≈ ±0.06). The lift is **not statistically distinguishable from seed noise**;
  it would be misleading to present as "AttentiveFP beats dmpnn".
- **vs literature Chemprop D-MPNN 0.876** (Wu 2018 Table 3, protocol-differs): Δ = -0.112.
  The auto-generated `report.md` verdict is `⚠️ BELOW GATE`; this run is **not**
  paper-promotable as a Tier-2 headline.

## §5.6 wall + protocol

- **Wall (per `metrics.json::tasks.bace.elapsed_seconds`): 1271.2 s = 21.2 min** (5 seeds × 50 ep on GPU 0 RTX PRO 6000)
- **Protocol**: Bemis-Murcko scaffold 70/10/20, 50 epochs, batch 64, lr 1e-3, hidden 300, depth 3,
  class-weight ON, seeds 0 1 2 3 4 — IDENTICAL to the dmpnn baseline protocol so the
  two are directly comparable.

## Verdict

- **BLK-D1 Step 4 SHIPPED** — the wire-in works end-to-end on 5 seeds.
- **Honest negative on the lift claim**: +0.0131 is inside seed noise; treat
  AttentiveFP and dmpnn as **statistically tied** on bace. Any future "AttentiveFP
  > dmpnn" claim must come from a multi-task cross-benchmark (BBBP / HIV / ClinTox)
  with tighter CI, not from this single-task bace lift.
- **Paper verdict (auto `report.md`)**: ⚠️ BELOW GATE. Below literature 0.876 by
  0.112; do NOT cite as Tier-2 headline.

## What this closes

- BLK-D1 Step 4 (the lift measurement) is SHIPPED.
- BLK-D1 as a whole has now completed: Step 2a wire-in (SHIPPED 2026-10-09) + Step 4
  5-seed lift (this SHIPPED, micro-lift within noise).
- Honest next step (NOT auto-pursued, awaiting user OK): a Tier-2 ship of "dmpnn
  ≈ AttentiveFP" with the 4-task cross-benchmark per
  `molmetal/reports/wf_tier2_molnet/` to lift bace's seed CI.
