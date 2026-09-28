# TODO-112 — D-MPNN ckpt seed integrity fix (Rh seed0=seed1 identity bug)

**Date**: 2026-09-27
**Status**: ✅ **SHIPPED via Wf-120 Phase 1+2** — see `TODO/completed/112_dmpnn_seed_integrity_closeout.md` (25 ckpts on disk, SHA256 diversity verified, per_metal_summary regenerated)
**Priority**: **P0 CRITICAL** (silent data integrity issue — per-metal paper claims would be wrong)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Agent 3 scoring issues research (2026-09-27)
**Skill alignment**: `mol-metal-cfm-training`

---

## §0 Why this plan exists

Agent 3 audit found a **silent data integrity bug** in `metrics/by_metric/dmpnn_ckpt_status.json`:

**Rh metal**:
- `dmpnn_pic50_rh_seed0.pt`: 758733 bytes, mtime `2026-09-26T13:00:00`, test_pearson=0.277
- `dmpnn_pic50_rh_seed1.pt`: **758733 bytes** (same), mtime `2026-09-26T13:00:00` (same), test_pearson=0.277 (same)
- File sizes **identical**, mtimes **identical**, metrics **identical**

**Diagnosis**: Rh seed1 ckpt is almost certainly a copy of Rh seed0. The training script for Rh seed1 either never ran independently OR the file was overwritten with the seed0 file.

**Secondary bug**: `per_metal_summary.Rh` shows `test_pearson_mean: -0.067, test_pearson_std: 0.344` but per-checkpoint entries show (0.277, 0.277). The summary is **stale** — was not regenerated when per-checkpoint entries were updated.

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU + GPU-1 RTX 5090 |
| Wall-time | 4 hr CPU + 1 day GPU |
| Output | retrained `dmpnn_pic50_rh_seed1.pt` + SHA256-audited `dmpnn_ckpt_status.json` + 5-seed expansion (Ru/Ir/Os/Rh/Re) |
| Tests | ~10 new tests (SHA256 audit + per-metal summary regen) |

## §2 Solution path

1. **Retrain Rh seed1** from scratch with explicit `seed=1` in CUDA RNG state.
2. **Verify SHA256**: `sha256(dmpnn_pic50_rh_seed0.pt) ≠ sha256(dmpnn_pic50_rh_seed1.pt)` — both files must hash differently.
3. **Add SHA256 to JSON**: extend `dmpnn_ckpt_status.json` schema to include `sha256` field per checkpoint.
5. **Regenerate `per_metal_summary`** from the per-checkpoint entries (not from stale cached values).
6. **Expand to 5 seeds**: retrain Rh (and other low-data metals) with seeds 0-4 for paper-grade mean±std.
7. **Add training-script guard**: in `train_dmpnn_pic50_real.py`, refuse to overwrite an existing ckpt with `seed=N` if its SHA256 matches the previous run's `seed=N` ckpt — log loud error.

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** Retrain Rh seed1 from scratch | 1 hr GPU | `dmpnn_pic50_rh_seed1.pt` (NEW) | SHA256 differs from seed0 |
| **P1.2** Add SHA256 to JSON schema | 2 hr CPU | patched `dmpnn_ckpt_status.json` | JSON has `sha256` per ckpt |
| **P1.3** 5-seed expansion for all 5 metals | 1 day GPU-1 | `dmpnn_pic50_{metal}_seed{0..4}.pt` (25 NEW ckpts) | per-metal mean±std stable across 5 seeds |
| **P1.4** Regenerate `per_metal_summary` | 30 min CPU | `dmpnn_ckpt_status.json` updated | summary values match per-checkpoint entries |
| **P1.5** Training-script SHA256 overwrite guard | 1 hr CPU | patched `train_dmpnn_pic50_real.py` + 6 new tests | pytest: simulated overwrite raises loud error |

## §4 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:
- **MEASURED**: SHA256 audit PASS + per-metal 5-seed mean±std
- **PROJECTED**: per-metal 5-seed std < 0.15 (vs current 2-seed std up to 0.34)

## §5 Honest framing

This bug exists because the training script allowed Rh seed1 ckpt to be overwritten with the seed0 file. The downstream JSON was not validated for SHA256 diversity. **The `per_metal_summary` is fabricated/stale** — its values don't match the per-checkpoint entries. Until this is fixed, any paper citation of per-metal Rh pIC50 would be wrong.

**Scoring-related concern**: This is a **scoring/training-pipeline bug**, not a fabrication bug — but it IS silent corruption of the canonical metric file. Must be fixed before any paper §4 per-metal pIC50 table.

## §6 Cross-references

- Agent 3 scoring issues report (2026-09-27) — Issue 4a + 4b
- TODO/completed/89c_wf47_wf49_round2.md (Wf-71 close-out — D-MPNN round 2)
- TODO/completed/107_combined_mcts_variance.md (Wf-82 close-out)
- `metrics/by_metric/dmpnn_ckpt_status.json` (the buggy file)
- `molmetal/checkpoints/dmpnn_pic50_{ru,ir,os,rh,re}_seed{0,1}.pt` (10 ckpts, 1 likely duplicate)
- TODO-84 paper rule (paper §4 per-metal tables not blocked)
- LAMBDA_ROADMAP §4.2 m22 (D-MPNN pIC50) column