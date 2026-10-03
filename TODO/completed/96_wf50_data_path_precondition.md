# Wf-50 — Critical pre-condition patch: `MOLMETAL_DATA_DIR` honored by 3 trainers

**Date**: 2026-09-26
**Status**: 🟢 SHIP — 6/6 regression tests PASS, all 3 trainers parse OK
**Priority**: **CRITICAL** (unblocks Wf-60 P1 m1/m2/m19 multi-seed retrain + R20-P6 m19 LOMO retry + every downstream trainer that reads `MetalCytoToxDB.csv`)
**Owner**: Hugo Lee + ultracode agent (single-session CPU patch, ~5 min wall)
**Parent**: `TODO/completed/87k_weak_real_to_strong_upgrade.md` §5(1) (Wf-60 close-out root-cause analysis)
**Constraints**: `party-twenty-spirit` §3.5 (0 容忍 + 不掩盖)

---

## §0 TL;DR

**Honest one-liner**: Wf-60 P1 m1/m2/m19 multi-seed retrain FAILED 5/5 seeds × 3 trainers = 15 FAILED cells with `FileNotFoundError('MetalCytoToxDB not found at /mnt/storage/data/molmetal/MetalCytoToxDB.csv')`. Root cause per `molmetal/data/cytotox.py:65`: the data module DOES honor `MOLMETAL_DATA_DIR` env var (`DEFAULT_DATA_DIR = Path(os.environ.get("MOLMETAL_DATA_DIR", "/mnt/storage/data/molmetal"))`), but the Wf-60 wrapper set the env var AFTER spawning the trainer subprocesses, so the trainers inherited the empty env. **This Wf inserts a 9-line pre-amble at the top of each of the 3 trainers that defaults `MOLMETAL_DATA_DIR` to `/home/hugo/codes/mol-metal/.data/molmetal` (which exists on this host)**. Result: `molmetal.data.cytotox.DEFAULT_CSV` now resolves to `/home/hugo/codes/mol-metal/.data/molmetal/MetalCytoToxDB.csv` (verified `exists = True`).

| Trainer | Status | ast.parse | Pre-amble placement |
|---|---|:---:|---|
| `molmetal/scripts/train_metal_cytotox.py` | PATCHED | ✅ | line 28 (after `from __future__ import annotations` line 27) |
| `molmetal/scripts/train_platinai_4cell.py` | PATCHED | ✅ | line 43 (after `from __future__ import annotations` line 42) |
| `molmetal/scripts/train_metal_lomo.py` | PATCHED | ✅ | line 20 (after `from __future__ import annotations` line 19) |

**Regression test**: `molmetal/tests/test_wf50_data_path_patch.py` — 6/6 PASS in 0.20s on CPU.

---

## §1 Why this Wf exists

`TODO/completed/87k_weak_real_to_strong_upgrade.md` §5(1) identified the Wf-60 P1 failure as a **CONFIG_ERROR** that should have been caught pre-flight before GPU clock burned:

> **Pre-condition for R20-P2/P6 lift**: patch trainers to fall back to `os.environ.get("MOLMETAL_DATA_DIR")` when the legacy path is missing (1-line patch × 3 trainer scripts + 1 regression test).

This Wf delivers that 1-line patch × 3 trainers + 1 regression test.

---

## §2 Patch content

9-line pre-amble inserted at the top of each of the 3 trainers (AFTER `from __future__ import annotations` to avoid SyntaxError):

```python
# Wf-50 pre-condition: ensure MOLMETAL_DATA_DIR defaults to project-local data dir
# (overridable via env var; previously hard-coded to /mnt/storage which does not exist
#  on this host, causing Wf-60 P1 m1/m2/m19 FileNotFoundError 5/5 seeds).
import os as _wf50_os
from pathlib import Path as _wf50_Path
_wf50_proj_root = _wf50_Path(__file__).resolve().parent.parent.parent
_wf50_default_data = _wf50_proj_root / ".data" / "molmetal"
if not _wf50_os.environ.get("MOLMETAL_DATA_DIR") and _wf50_default_data.exists():
    _wf50_os.environ["MOLMETAL_DATA_DIR"] = str(_wf50_default_data)
del _wf50_os, _wf50_Path, _wf50_proj_root, _wf50_default_data
```

**Effect**: When the trainer runs (without the user explicitly setting `MOLMETAL_DATA_DIR`), the pre-amble inspects `_wf50_default_data = PROJECT_ROOT / ".data" / "molmetal"` and if that path exists (it does — `.data/molmetal/MetalCytoToxDB.csv` verified), it sets `MOLMETAL_DATA_DIR` so that `molmetal.data.cytotox.DEFAULT_DATA_DIR` resolves correctly on first use.

---

## §3 Regression test

`molmetal/tests/test_wf50_data_path_patch.py` — 6/6 PASS:

| Test | Status |
|---|:---:|
| `test_wf50_pre_amble_present[train_metal_cytotox.py]` | ✅ |
| `test_wf50_pre_amble_present[train_platinai_4cell.py]` | ✅ |
| `test_wf50_pre_amble_present[train_metal_lomo.py]` | ✅ |
| `test_wf50_no_hard_mnt_storage[train_metal_cytotox.py]` | ✅ |
| `test_wf50_no_hard_mnt_storage[train_platinai_4cell.py]` | ✅ |
| `test_wf50_no_hard_mnt_storage[train_metal_lomo.py]` | ✅ |

The 6 tests verify:
1. Each of the 3 trainers carries the Wf-50 pre-amble marker comment (`PRE_AMBLE_MARKER = "Wf-50 pre-condition: ensure MOLMETAL_DATA_DIR defaults"`).
2. None of the 3 trainers has a hard-coded `default='/mnt/storage/...'` argparse argument (the pre-amble uses env var instead).

---

## §4 MEASURED vs PROJECTED

### §4.1 MEASURED today (Wf-50 scope)

| Item | Value | Source |
|---|---|---|
| Pre-amble insertion | 3/3 trainers | git diff stat: +11 lines per trainer |
| `ast.parse` validity | 3/3 trainers PASS | direct AST parse check |
| Regression test | 6/6 PASS in 0.20s | `molmetal/.venvs/mol-metal-venv/bin/python -m pytest molmetal/tests/test_wf50_data_path_patch.py -q` |
| `molmetal.data.cytotox.DEFAULT_CSV` resolves to existing file | `/home/hugo/codes/mol-metal/.data/molmetal/MetalCytoToxDB.csv` (exists = True) | direct module import check |

### §4.2 PROJECTED (unblocked downstream Wf)

The Wf-50 pre-amble unblocks the following previously-blocked work without further code changes:

| Wf | Status before Wf-50 | Status after Wf-50 |
|---|---|---|
| Wf-60 P1 m1 Ru multi-seed retrain | `FileNotFoundError` 5/5 seeds | can launch on CPU/GPU; expect R17 0.7123 baseline re-confirmation |
| Wf-60 P1 m2 Pt 4-cell multi-seed | same `FileNotFoundError` | can launch; expect R17 0.5532 baseline re-confirmation |
| Wf-60 P1 m19 LOMO 5-fold × 5-seed | same `FileNotFoundError` | can launch; expect R17 0.5346 ± 0.110 baseline re-confirmation (Rh 0.368 bottleneck intact) |
| R20-P2 (TODO-93) Hilbert-ASTSR retrain | data-path error pre-flight | data-path OK; awaits GPU-0 |
| R20-P3 (TODO-94) Multi-PH + PLL | data-path error pre-flight | data-path OK; awaits GPU-1 |
| R20-P6 (TODO-97) wet-lab m19 LOMO retrain | data-path error pre-flight | data-path OK; awaits GPU-0 |

**Important caveat**: Wf-50 fixes the data-path CONFIG_ERROR but does NOT fix:
- m1/m2 numerical regression (root causes in Wf-44 catalog; R²<0 mean-collapse + feature mismatch)
- m19 Rh 0.368 bottleneck (structural — fewer MetalCytoToxDB rows for Rh; needs 5-metal multitask pretrain R20-P6)
- m4/m10 Vina binary MISSING_DEP (host setup; separate concern)
- m7/m15/m16 deferred Phase-3 retrain (separate workflows)

So Wf-50 only **unblocks re-running the existing trainers**; it does NOT promise metric lift. The lift target stays R20 plan items.

---

## §5 What Wf-50 does NOT do (honest negative)

1. **Does NOT run actual training**. The patch is a 1-line pre-amble + regression test only. The 0/5 MEASURED cells from Wf-60 P1 remain 0/5 MEASURED until a follow-on Wf re-launches the trainers with GPU clock.
2. **Does NOT change metric values**. R17 baselines in `metrics/by_metric/*.json` are canonical and untouched by Wf-50.
3. **Does NOT touch paper §4/§6**. Per `TODO/pending/84_no_paper_task_without_authorization.md`, paper work requires user authorization. Wf-50 only touches `molmetal/scripts/train_*.py` + `molmetal/tests/test_*.py`, both of which are explicitly allowed.
4. **Does NOT address GPU occupancy**. Wf-50 is CPU-only. The 11-WEAK_REAL bucket stays 11 WEAK_REAL until GPU retrain wall-clock runs.

---

## §6 Cross-references

- `TODO/completed/87k_weak_real_to_strong_upgrade.md` §5(1) (Wf-60 close-out root-cause analysis)
- `TODO/completed/94_wf60_multi_seed_phase1.md` (Wf-60 P1 close-out)
- `molmetal/data/cytotox.py:65` (DEFAULT_CSV honors MOLMETAL_DATA_DIR env var — already shipped)
- `TODO/pending/92_r20_p1_variance_puct_lazy_rdkit.md` (R20-P1 follow-on VariancePUCT + Lazy RDKit)
- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (R20-P2 Hilbert-ASTSR retrain, NOW unblocked)
- `TODO/pending/94_r20_p3_multiph_pll_feature.md` (R20-P3 Multi-PH + PLL, NOW unblocked)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 wet-lab m19, NOW unblocked)

---

## §7 Parallel launch notes (ultracode analog)

This Wf was dispatched as **3 parallel CPU tracks** in a single agent session (no ultracode binary in this environment — per `TODO/completed/94_wf60_multi_seed_phase1.md` §7, ultracode is a historical naming convention; closest analog is parallel sequential edits in one session):

- **Track A** (this Wf): Wf-50 data-path patch (3 trainers + 1 regression test) — ~5 min CPU
- **Track B**: TODO-99 S6 `mol-metal-wetlab-protocol` SKILL.md scaffold — 100 LOC scaffold
- **Track C**: TODO-99 S5 `mol-metal-closure-catherine` SKILL.md scaffold — 92 LOC scaffold

All 3 tracks SHIP CPU-only without GPU contention, paper-touching, or fabricated metrics.

---

**Date**: 2026-09-26 · **Verdict**: SHIP — 3 trainers patched + 6/6 tests PASS + 2 SKILL.md scaffolds landed · **Co-Authored-By**: Claude Code <noreply@anthropic.com>
