# Wf-120 Phase 1 — D-MPNN Rh seed1 retrain + SHA256 audit

**Date**: 2026-09-27
**Status**: PASS — Rh seed1 retrained from scratch; all 10 D-MPNN ckpts SHA-unique; status JSON reconciled.
**Owner**: Claude (MiniMax-M3) for Hugo Lee
**Parent**: TODO/pending/112_dmpnn_seed_integrity.md

---

## §1 What was done

1. **Backup**: pre-retrain copies of both Rh ckpts are at
   `molmetal/checkpoints/_backup_wf120_p1/*.pt.before_retrain`.
2. **Retrain**: Rh seed1 from scratch on GPU-1 RTX 5090, 30 epochs, alpha=1.0
   (pure pIC50 regression). Test metrics (n=38):
   - **test_pearson** = 0.276
   - **test_rmse** = 3.204
   - **elapsed** = 8.8 s training + ~1.5 s featurise
3. **SHA256 audit** (`molmetal/scripts/dmpnn_sha256_audit.py`):
   - All 10 D-MPNN ckpts have **unique** SHAs (no duplicates).
   - `dmpnn_ckpt_status.json` back-filled with `sha256` for all 10 ckpts.
   - `per_metal_summary` regenerated from per-checkpoint entries
     (was stale before; Rh `test_pearson_mean` corrected).
4. **SHA256 overwrite guard** added to
   `molmetal/scripts/train_dmpnn_pic50_real.py` via
   `_guard_ckpt_overwrite()`. Fires `RuntimeError` when a re-train would
   produce a ckpt whose SHA256 already matches another (metal, seed) sibling.
5. **Tests** (`molmetal/tests/test_dmpnn_sha256_audit.py`, 14 tests, all PASS):
   - SHA256 helpers, parse_filename, discover_dmpnn_ckpts
   - audit() PASS/FAIL paths (unique vs duplicate SHAs)
   - JSON reconciliation (mismatch + missing-SHA)
   - End-to-end smoke run on the real ckpt dir
   - Training-script SHA256 overwrite guard + seed-setting coverage

## §2 Honest framing

* The pre-existing `dmpnn_pic50_rh_seed1.pt` on disk had **different**
  bytes / SHA256 / model weights than `seed0.pt` — so it was **not** a
  literal byte-copy of seed0 (the plan's central claim was overstated).
  However, the `dmpnn_ckpt_status.json` had Rh seed0 reporting
  seed1's metrics (`test_pearson=0.277, test_rmse=3.229`), which is the
  real silent-data-integrity bug. The new audit script + JSON
  reconciliation both surface and fix this.
* After retrain: Rh seed1 SHA = `117e0d53…`, seed0 SHA = `142f63bc…`.
  Both unique; `diversity_check.differ = True`.
* The retrained seed1 still gives `test_pearson ≈ 0.276` — same value
  as the pre-retrain run. This is the honest signal: with n=38 test
  rows and 60 train rows, Rh pIC50 R² is too noisy to distinguish
  different training runs at the Pearson level. Paper-grade reporting
  requires the **5-seed expansion** (Wf-120 P1.3) which is **out of
  scope** for this P1 deliverable.

## §3 Outputs

| Path | Purpose |
|------|---------|
| `molmetal/checkpoints/dmpnn_pic50_rh_seed1.pt` | Retrained ckpt (NEW SHA `117e0d53…`) |
| `molmetal/checkpoints/_backup_wf120_p1/*.before_retrain` | Pre-retrain backup (2 files) |
| `molmetal/scripts/dmpnn_sha256_audit.py` | SHA256 audit script (~270 LOC) |
| `molmetal/scripts/train_dmpnn_pic50_real.py` | Patched with `_guard_ckpt_overwrite()` + CUDA seed |
| `molmetal/tests/test_dmpnn_sha256_audit.py` | 14 new tests (all PASS) |
| `molmetal/reports/wf120_p1_rh_retrain/train_log.json` | Retrain training log + SHA256 + mtime |
| `molmetal/reports/wf120_p1_rh_retrain/test_metrics.json` | Test pIC50 R² / RMSE + diversity_check |
| `molmetal/reports/wf120_p1_rh_retrain/training_summary.json` | Per-metal aggregate |
| `molmetal/reports/wf120_p1_rh_retrain/sha256_audit.json` | Audit report (status=PASS) |
| `molmetal/reports/wf120_p1_rh_retrain/rh/per_seed_metrics_seed1.json` | Per-seed metrics |
| `metrics/by_metric/dmpnn_ckpt_status.json` | Back-filled SHA256 + corrected `per_metal_summary` |

## §4 Verification

```bash
# SHA256 diversity (10 unique SHAs):
$ sha256sum molmetal/checkpoints/dmpnn_pic50_{ru,ir,rh,os,re}_seed{0,1}.pt | wc -l
10
$ sha256sum ... | awk '{print $1}' | sort -u | wc -l
10

# Audit exits 0:
$ molmetal/.venvs/mol-metal-venv/bin/python molmetal/scripts/dmpnn_sha256_audit.py
... "status": "PASS"   # exit 0

# All 14 new tests pass:
$ molmetal/.venvs/mol-metal-venv/bin/python -m pytest molmetal/tests/test_dmpnn_sha256_audit.py
... 14 passed

# Existing D-MPNN tests still pass (with MOLMETAL_DATA_DIR set):
$ MOLMETAL_DATA_DIR=.data/molmetal molmetal/.venvs/mol-metal-venv/bin/python -m pytest \
    molmetal/tests/test_dmpnn.py molmetal/tests/test_dmpnn_ckpt_diagnostic.py
... 16 passed
```

## §5 Known limitations

* Only Rh seed1 was retrained in P1. The other 9 D-MPNN ckpts were
  verified to have unique SHAs but their on-disk test_pearson values
  were **back-filled** from the Wf-84 per_seed_metrics JSONs (the
  status JSON was the stale source). The 5-seed expansion (P1.3) is
  the next step for paper-grade reporting.
* The guard only fires on **byte-identical** SHAs across siblings; it
  does not detect "same initial weights, same training trajectory,
  slightly different RNG noise" — that would require comparing the
  actual tensor values, which is a stronger check left for a future
  audit pass.