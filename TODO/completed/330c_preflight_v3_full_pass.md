# Wf-330-C: Pre-flight v3 — 62/62 PASS (the Wf-327 gate false positive)

**Status:** SHIPPED (CPU-only)
**Date:** 2026-09-29

---

## Root cause of the lone FAIL

Pre-flight v3 fresh-run reported 1 FAIL:

```
p3_wf327_cpu_num_workers_sufficient: FAIL
  evidence: missing launch script(s): ['/home/hugo/codes/mol-metal/molmetal/scripts/launch_wf124_p3_g4.sh', ...]
```

The evidence path `molmetal/scripts/launch_*.sh` did NOT match reality: the launch
scripts live at `/home/hugo/codes/mol-metal/scripts/launch_*.sh` (REPO_ROOT/scripts),
NOT at `molmetal/scripts/launch_*.sh` (MOLMETAL_ROOT/scripts).

Two issues:

1. **System Python env mismatch**: The pre-flight was run with `python` (system
   /usr/bin/python = Python 3.14.7) which doesn't have `torch`. The Catherine
   property verifier imports torch at module-load time, so the import failed and
   the bench fell through to the `except` branch returning `n_pass=0`. The gate
   evidence then showed `molmetal/scripts/launch_*.sh` paths because the bench
   attempted the fallback path, but those paths actually resolve via REPO_ROOT
   (`/home/hugo/codes/mol-metal/scripts/`) when running from the correct env.
2. **Verdict false-positive**: The gate reported FAIL even though the launch
   scripts all exist and use `--cpu-num-workers 8`. Direct gate invocation with
   the proper venv-python passes 6/6.

## Fix

Use `CadButEaas/.venv-vlm/bin/python` (Python 3.12.13 + torch 2.14.0+cu130) for
all pre-flight runs that import `molmetal.math_skills.catherine_properties`
or any other torch-required module. The pre-flight `import torch` is implicit
through `catherine_properties`; without torch the bench verdict is fabricated.

## Verification

```
$ PYTHONPATH=. /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/pre_flight_check.py --pipeline all

SUMMARY: {'PASS': 62, 'FAIL': 0, 'SKIP': 0}
```

All 62 pre-flight v3 gates PASS. No FAIL gates.

## R9 risk closure

This Wf-330-C delivery, combined with Wf-323 + Wf-325 ship waves, closes
`TODO/risks.md` R9 (15 FAIL pre-flight gates). The historical baseline of
42 PASS / 15 FAIL / 57 TOTAL is now upgraded to 62 PASS / 0 FAIL / 62 TOTAL.

## Pre-flight gate count

Pre-flight v3 expanded 52 → 57 → **62 gates** in this ship wave. The 5 NEW
gates added by Wf-321D + Wf-321E + Wf-321H + Wf-322E + Wf-322G (per
`TODO/completed/321d_preflight_gate_audit.md`) all PASS with venv-vlm.

## Cross-references

- `TODO/risks.md` R9 — closed (15 FAIL → 0 FAIL)
- `TODO/completed/322d_index_readme_wf309_322_sync.md` — historical baseline
- `TODO/completed/323a_m14_lr_warmup_reintegration.md` — first 5 FAILs closed
- `TODO/completed/323c_m8_full_cpu_fix.md` — 3 ADMET/PB FAILs closed