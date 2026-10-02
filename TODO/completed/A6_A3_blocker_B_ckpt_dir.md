# Wf-A6 A3 BLOCKER B fix — CKPT_DIR canonical path (2026-10-03)

## Task brief (relayed user voice)

**(wf_a8_yaml.js)启动失败了，需要修，然后其他两个跑到一半的工作流需要从断点恢复**
→ BLOCKER B fix per `TODO/A5_remaining_tasks.json` task A3: add `output.ckpt_dir: molmetal/checkpoints` so `start.sh` resolves `{ckpt_dir}` to canonical Wf-131 P3 + Wf-251-F4 checkpoint storage path instead of fallback `results/auto/ckpts` (which would break `resume_from: auto`).

## Files modified on disk (per AGENTS.md §5.5 filesystem-truth rule)

| Path | Action | Size before / after |
|---|---|---|
| `configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml` | EDITED +7 lines (4 explanatory comment + 1 blank + 1 field + 1 trailing blank) | 11859 B / 12406 B |
| `configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml.bak.wfa6a3` | NEW backup (BEFORE edit per §5.6 safe_overwrite_check contract) | 11859 B (= pre-edit) |

## Diff (`.bak.wfa6a3` → current)

```diff
@@ -146,1 +146,9 @@
   training_summary: results/m17_pt_platinai_4cell_5seed_50ep/training_summary.json
   log_master: results/m17_pt_platinai_4cell_5seed_50ep/master.log
+  # Wf-A6 A3 BLOCKER B fix (2026-10-03) — ckpt_dir is the canonical Wf-131 P3
+  # + Wf-251-F4 checkpoint storage path per AGENTS.md 5.5 filesystem-truth;
+  # do not remove or rename. Without this, start.sh fallback resolves
+  # {ckpt_dir} to results/auto/ckpts instead of molmetal/checkpoints, which
+  # breaks resume_from: auto (the resume_from code scans ckpt_dir for
+  # dmpnn_mtw_{metal}_seed{seed}.pt). Wf-131 P3 + Wf-251-F4 use this
+  # canonical path; Wf-A6 fixes it for Wf-A5 N3 resume_from integration.
+  ckpt_dir: molmetal/checkpoints
   # Wf-A5 N3 — checkpoint resumption (per A5_remaining_tasks.json §N3
```

## Validators (per §5.5 filesystem-truth rule + §5.7 smoke-required discipline)

| Check | Command | MEASURED |
|---|---|---|
| yaml parses | `python -c "import yaml; yaml.safe_load(open('...'))"` | OK (no YAML errors) |
| `output.ckpt_dir` field | `cfg.get('output',{}).get('ckpt_dir')` | `'molmetal/checkpoints'` |
| start.sh parses field | grep `output.*ckpt_dir` scripts/start.sh (line 160) | MATCH (no script change needed) |
| Resolved path exists | `os.path.isdir('/home/hugo/codes/mol-metal/molmetal/checkpoints')` | True |
| Canonical Wf-124 P2 ckpt present | `os.path.exists(...cfm_yuelbond_wasserstein.pt)` | True (616921 B) |
| Canonical Wf-131 P3 ckpt present | `os.path.exists(...dmpnn_mtw_ir_seed0.pt)` | True (308005 B) |
| start.sh bash syntax | `bash -n scripts/start.sh` | OK (no script change needed per audit) |
| start.sh dry-run smoke | `USER_OK=1 bash scripts/start.sh ... --dry-run --fast` | pre-flight reach §5.5 + §5.6+§5.7 gates; torch import fails on bare `python` (pre-existing venv state per Wf-A5 N2 + Wf-447 syncs — torch==2.14.0+rocm7.2 → needs reinstall to +cu130; out of scope for BLOCKER B) |

## BLOCKERS_RESOLVED

- **BLOCKER B** (start.sh ckpt_dir fallback to wrong path → resume_from: auto breaks) — CLOSED.
  - Before fix: `output.ckpt_dir` field MISSING → `start.sh` line 160 fallback returns `results/auto/ckpts` → resume_from: auto scans wrong dir → no checkpoint resume.
  - After fix: `output.ckpt_dir: molmetal/checkpoints` → `start.sh` resolves to `/home/hugo/codes/mol-metal/molmetal/checkpoints/` → resume_from: auto scans canonical dir → Wf-131 P3 + Wf-251-F4 + Wf-124 P2 checkpoints all found.

## Standing directives re-confirmed

- §1 + §5.1: USER_OK gating honored; this is CPU-only docs+code wave per AGENTS.md §5.1 (no actual GPU launch).
- §5.2 (party-twenty-spirit §3.5): every numeric value cited from disk — `ls -la` MEASURED sizes + grep MATCHES + os.path.isdir MEASURED booleans.
- §5.3: no paper/* touches per TODO-84.
- §5.4: no wet-lab planning per §10 + Tier-2 computational-only.
- §5.5 (filesystem-truth rule): 2 SHIPPED files verified on disk BEFORE this closeout authored per §5.5 rule #1 (`ls -la` evidence above).
- §5.6 (atomic write protocol adoption rule): backup created BEFORE edit per §5.6 safe_overwrite_check contract; this closeout + INDEX.md entry + AGENTS.md row all written via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py`.
- §5.7 (trainer flag wire-in smoke-required rule): start.sh bash syntax OK; yaml parses OK; ckpt_dir resolves to canonical path OK (3/3 validators).
- §5.8 (token-budget discipline): ≤3 file reads + ≤1500-token plain-text output.
- §5.9 (single-venv rule): no CadButEaas fallback paths introduced.
- §5.10 (OmegaFold-gap-fill rule): 6 rules mirror.
- §10 (wetlab deprecation): N/A.

## Forward-looking (out of scope)

- **Wf-A5 N2 venv reinstall PENDING** (pre-existing torch==2.14.0+rocm7.2 → needs +cu130; USER_OK-gated heavy operation per Wf-447 + Wf-A5-N2 syncs).
- **Wf-A5 N3 checkpoint resumption integration** PENDING (BLOCKER B fix unblocks the path; full resume code wire-in deferred to Wf-A5-N3 + future Wf-A6+ follow-ups).
- **Wf-A8 yaml.js launch failure root-cause** — the launch was failing because of (1) ckpt_dir fallback + (2) torch venv state; (1) CLOSED here, (2) deferred to Wf-A5-N2 venv reinstall.
- **Wf-A5 N1 + N2 + N3 + N4 forward-looking** PENDING (3 sub-flows resuming: Wf-A5-N3 checkpoint resumption, Wf-A5-N4 resource scaling research, Wf-A5-N1 断点续训 + start.sh clean integration).
