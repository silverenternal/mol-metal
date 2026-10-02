# Wf-284B: cfm_wasserstein_loss.py Recovery Report

_Generated: 2026-09-29 (Wf-284B follow-up to Wf-282D)_

## 1. File status

- `git log -- molmetal/molmetal_lam/lam_chem/cfm_wasserstein_loss.py` → **empty** (file NEVER in git history)
- `find ... -name 'cfm_wasserstein_loss*'` → **no matches**
- `find ... -name 'wasserstein_loss.py'` → **1 match**: `molmetal/molmetal_lam/reward/wasserstein_loss.py` (32138 bytes, mtime 2026-09-28 17:48)

The originally-reported "missing" path was a **stale gate definition**, not a
missing file.  The Wasserstein loss module is implemented under
`molmetal/molmetal_lam/reward/` (not `lam_chem/`), and is consumed by
`train_cfm_yuelbond_wasserstein.py` line 262 (per the inline comment at
`molmetal/scripts/pre_flight_check.py:555-558`).

## 2. Decision rationale

Wf-260 (2026-09-28) already corrected the gate path inside
`molmetal/scripts/pre_flight_check.py:558`:

```python
cfm_wass_loss = MOLMETAL_ROOT / "molmetal_lam" / "reward" / "wasserstein_loss.py"
```

This fix is verified in `molmetal/reports/wf263c_verify/pre_flight.json`,
where `p1_cfm_wasserstein_loss_exists` reports `PASS` with evidence
`"file exists: .../molmetal_lam/reward/wasserstein_loss.py"`.

No restore from git is possible (file never existed there).  No gate edit is
needed (already fixed).  No new module needs to be written.

## 3. Action taken

- Verified gate path fix (`pre_flight_check.py:555-558` comment matches code).
- Re-ran `per_wave_checklist.py --wave Wf-282` against the up-to-date
  `wf263c_verify/pre_flight.json` (rather than the stale `wf259_run` JSON).
- Wrote this report.

## 4. Per-wave verdict (Wf-282)

```
preflight pass count: 25
preflight fail count: 0
P0 BLOCKER gates  : 3 / 3  PASS
P1 VERIFY gates   : 16 / 16 PASS  ← p1_cfm_wasserstein_loss_exists PASS
P2 DOCUMENTED     : 6 / 6  PASS
wave budget       : 10.0 h (9.5 h remaining after abort penalties)
```

`p1_cfm_wasserstein_loss_exists` evidence:
`file exists: /home/hugo/codes/mol-metal/molmetal/molmetal_lam/reward/wasserstein_loss.py`

## 5. Conclusion

Wf-282D's "MISSING" finding was a **false positive** caused by the script
pointing at `lam_chem/cfm_wasserstein_loss.py` instead of the actual
implementation at `reward/wasserstein_loss.py`.  Wf-260 already corrected
this.  Wf-282 is **GREEN: 25/25 PASS**, ready for proceed.

No further action required.
