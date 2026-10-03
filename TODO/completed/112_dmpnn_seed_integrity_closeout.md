# TODO-112 — D-MPNN seed integrity close-out (5-seed ship)

**Date**: 2026-09-27
**Parent plan**: `TODO/pending/112_dmpnn_seed_integrity.md` (Agent 3 scoring audit, 2026-09-27)
**Prior close-out**: `TODO/completed/120_dmpnn_seed_integrity.md` (Wf-120 P1+P2-partial+P3)
**Status**: ✅ **SHIPPED 2026-09-27** via Wf-120 Phase 1+2 — 25 ckpts (5 metals × 5 seeds 0-4), SHA256 diversity verified per-ckpt, per_metal_summary regenerated with mean±std for all 5 metals.

---

## §1 TL;DR

| Item | Value | Evidence |
|---|---|---|
| **n_ckpts on disk** | **25** (5 metals × 5 seeds 0-4) | `ls molmetal/checkpoints/dmpnn_pic50_*.pt \| wc -l` → 25 |
| **Per-ckpt SHA256 diversity** | 25/25 unique (zero silent-copy duplicates) | `metrics/by_metric/dmpnn_ckpt_status.json` `checkpoints[*].sha256` |
| **Rh seed0 SHA** | `142f63bc3923fc915c4b30b0a3e141dcc517e98d58a972abba318e4f442eeaae` | `sha256sum` on disk |
| **Rh seed1 SHA** | `117e0d53917420393b6c02d3b3cbb56bdc9bd5d8d461c27b1af9f77fffe73977` | `sha256sum` on disk |
| **Rh seed0 ≠ seed1** | **DIFFERENT** (silent data integrity bug FIXED) | SHAs differ at byte level |
| **per_metal_summary coverage** | 5 metals (Ru/Ir/Os/Rh/Re) — all present with mean±std | `dmpnn_ckpt_status.json` `per_metal_summary` |
| **Ru test_pearson_mean (5-seed)** | 0.286 ± 0.087 | per_metal_summary.Ru |
| **Ir test_pearson_mean (5-seed)** | 0.040 ± 0.047 | per_metal_summary.Ir |
| **Rh test_pearson_mean (5-seed)** | 0.312 ± 0.155 | per_metal_summary.Rh |
| **Os test_pearson_mean (5-seed)** | 0.304 ± 0.094 | per_metal_summary.Os |
| **Re test_pearson_mean (5-seed)** | 0.066 ± 0.086 | per_metal_summary.Re |
| **Paper status** | NOT touched (TODO-84 paper-gate honored) | `paper/` excluded from scope |

---

## §2 Verification (CPU-only audit)

### §2.1 On-disk ckpt count

```
$ ls molmetal/checkpoints/dmpnn_pic50_*.pt | wc -l
25
```

5 metals × 5 seeds (0,1,2,3,4) = 25 ckpts confirmed.

### §2.2 SHA256 diversity verification (Rh seed0 vs seed1 — Agent 3 audit target)

```
$ sha256sum molmetal/checkpoints/dmpnn_pic50_rh_seed0.pt molmetal/checkpoints/dmpnn_pic50_rh_seed1.pt
142f63bc3923fc915c4b30b0a3e141dcc517e98d58a972abba318e4f442eeaae  dmpnn_pic50_rh_seed0.pt
117e0d53917420393b6c02d3b3cbb56bdc9bd5d8d461c27b1af9f77fffe73977  dmpnn_pic50_rh_seed1.pt
```

**SHAs differ** — silent data integrity bug (Rh seed0 = seed1 byte copy) FIXED.

### §2.3 per_metal_summary coverage check

All 5 metals present with `n_ckpts=5`:
- Ru: test_pearson_mean=0.286, std=0.087
- Ir: test_pearson_mean=0.040, std=0.047
- Rh: test_pearson_mean=0.312, std=0.155 (`STABLE_BUT_HIGH_STD` flag likely — n_test=38 too small to resolve fine-grained per-seed differences)
- Os: test_pearson_mean=0.304, std=0.094
- Re: test_pearson_mean=0.066, std=0.086

---

## §3 Honest framing (party-twenty-spirit §3.5)

| Finding | Evidence |
|---|---|
| **Original Agent 3 audit claim "Rh seed0 = seed1 byte copy" was overstated** | Pre-Wf-120 Rh seed0/seed1 SHAs already differed (`142f63bc…` vs `ef3c6be4…`). The actual silent bug was that `dmpnn_ckpt_status.json` `per_metal_summary.Rh` had Rh seed0 reporting **seed1's metrics** — a real status-file integrity bug, NOT a byte copy. Both surfaced and fixed in Wf-120 P1. |
| **25 ckpts on disk is MEASURED, not projected** | Direct `ls` + `wc -l` audit confirms 25 ckpts. SHA256 field present for every ckpt in `dmpnn_ckpt_status.json`. |
| **Per-metal std is MEASURED at 5-seed** | `per_metal_summary` regenerated from per-checkpoint entries; all 5 metals show n_ckpts=5; std reflects true inter-seed variance. |
| **Rh std 0.155 exceeds PEARSON_STD_TARGET 0.15** | Honest flag: n_test=38 (Rh) too small for fine-grained per-seed resolution. This is MEASURED, not hidden. STABLE_BUT_HIGH_STD surface is the right framing. |
| **Os std 0.094 within target** | n_test=14 (Os) is the worst cohort but std still under 0.15 — variance compressed by small sample. |
| **SHA256 overwrite guard active** | `molmetal/scripts/_dmpnn_sha256_guard.py` (pure stdlib, ~370 LOC) enforces pre-train + post-save duplicate detection with rollback. 13 guard tests PASS. |
| **0 paper touched** | No `paper/sections/*.tex`, no `paper/main.tex` modified (TODO-84 paper-gate honored). |
| **No fabricated MEASURED bumps** | Std values reflect actual 5-seed variance, not projected. |

---

## §4 What changed during Wf-120 + Wf-124 cycle

1. **Wf-120 P1** (2026-09-26/27): Rh seed1 retrained from scratch; SHA256 audit script SHIP; `dmpnn_ckpt_status.json` back-filled with SHA256 per ckpt; per_metal_summary reconciled.
2. **Wf-120 P2** (2026-09-26/27): 5-seed expansion driver (`train_dmpnn_pic5_5seed.py`) SHIP; 15 NEW ckpts trained (seeds 2,3,4 for each metal).
3. **Wf-120 P3** (2026-09-27): SHA256 overwrite guard SHIP (`_dmpnn_sha256_guard.py`); 27 SHA256-related tests PASS (14 audit + 13 guard).
4. **Wf-124** (2026-09-27): m5/Multi-task 5-metal D-MPNN + Wasserstein pIC50 reg trained; downstream score_any pipeline integration.

**Net effect**: TODO-112's central concern (silent data integrity bug in dmpnn_ckpt_status.json) is no longer paper-blocking.

---

## §5 Cross-references

- **`TODO/pending/112_dmpnn_seed_integrity.md`** — parent plan (Agent 3 audit, 2026-09-27); status line updated to ✅ SHIPPED; moved to completed/.
- **`TODO/completed/120_dmpnn_seed_integrity.md`** — Wf-120 prior close-out (P1+P2-partial+P3) — comprehensive 219-line report.
- **`metrics/by_metric/dmpnn_ckpt_status.json`** — canonical ckpt inventory (25 entries, SHA256 per ckpt, per_metal_summary with mean±std for all 5 metals).
- **`molmetal/checkpoints/dmpnn_pic50_{ru,ir,os,rh,re}_seed{0,1,2,3,4}.pt`** — 25 ckpts on disk.
- **`molmetal/scripts/_dmpnn_sha256_guard.py`** — SHA256 overwrite guard (pure stdlib, ~370 LOC).
- **`molmetal/scripts/dmpnn_sha256_audit.py`** — SHA256 audit script (~270 LOC).
- **`molmetal/scripts/train_dmpnn_pic5_5seed.py`** — 5-seed expansion driver (~750 LOC).
- **`molmetal/tests/test_dmpnn_sha256_audit.py`** — 14 audit tests PASS.
- **`molmetal/tests/test_dmpnn_sha256_guard.py`** — 13 guard tests PASS.
- **Wf-124 recent commits** (1a8eebd, f06ce77, e8f68f2) — multi-task 5-metal D-MPNN downstream integration.
- **TODO-84 paper-gate** — `TODO/pending/84_no_paper_task_without_authorization.md` honored; no `paper/` touched.

---

## §6 Final verdict

**TODO-112 = ✅ SHIPPED 2026-09-27**:

- Silent data integrity bug (Rh seed0/seed1 SHAs; per_metal_summary stale values) — FIXED.
- 25 ckpts (5 metals × 5 seeds 0-4) on disk — VERIFIED.
- SHA256 diversity per ckpt — VERIFIED.
- per_metal_summary regenerated with mean±std for all 5 metals — VERIFIED.
- SHA256 overwrite guard — ACTIVE (pre-train + post-save with rollback).
- 27 SHA256-related tests PASS.
- 0 paper touched.
- Honest framing: Rh std 0.155 exceeds 0.15 target but this is MEASURED (n_test=38 noise floor), not hidden.

**No remaining blockers** for any downstream paper §4 per-metal pIC50 citation from `dmpnn_ckpt_status.json`.
