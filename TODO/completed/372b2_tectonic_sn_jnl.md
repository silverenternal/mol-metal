# Wf-372-B2: tectonic + sn-jnl end-to-end test compile

**Date:** 2026-09-30
**Status:** SHIPPED
**USER_OK:** User standing grant 2026-09-29 verbatim "Zotero + tectonic install可以做，其实其他重新训练和论文以外的事情我都给你授权" — install + tooling work is authorized; only retrain + paper/* are off-limits.

## §0 TL;DR

End-to-end `tectonic` compile of the `sn-jnl` template now succeeds — produces a valid 7.4 KiB PDF. Required two minimal defensive patches to `.tools/latex/sn-jnl.cls` (with backup at `.tools/latex/sn-jnl.cls.bak.wf372b2`). 1 NEW test PASS in 0.96 s at `molmetal/tests/test_wf372b2_tectonic_sn_jnl.py`. READ-only on `paper/*` per TODO-84 + AGENTS.md §5.3.

## §1 Patches applied to `sn-jnl.cls`

Both patches are **defensive no-ops when the upstream packages are loaded** — they only kick in when the older Springer template assumptions about LaTeX kernel versions don't hold.

### §1.1 Patch 1 — `allowdisplaybreaks` guarded

**File offset (original):** 53346
**Pattern:**
```latex
\AtBeginDocument{\allowdisplaybreaks}%
```
**Patched to:**
```latex
\AtBeginDocument{\providecommand{\allowdisplaybreaks}{}\allowdisplaybreaks}%
```

**Why:** `allowdisplaybreaks` is provided by `mathtools` (loaded transitively via `amsmath`). The newer LaTeX kernels in tectonic's bundle no longer pre-load `amsmath` into `article` derivatives. Wrapping in `\providecommand{...}{}` makes the command a no-op if absent instead of an undefined-control-sequence error at `\begin{document}`.

### §1.2 Patch 2 — `headerps@out` no-op

**File offset (after patch 1):** 53346 + len(patch1)
**Added immediately after patch 1:**
```latex
\providecommand{\headerps@out}{}%
```

**Why:** The newer LaTeX kernel registers a `shipout/firstpage` hook (used by `breakurl` + `hyperref` combination) that tries to invoke `\headerps@out`. The Springer `sn-jnl.cls` only sets up `\ps@headings` pagestyle and never defines `\headerps@out`. Defining it as a no-op silences the hook without affecting the actual page style.

## §2 End-to-end compile evidence

**Test file:** `molmetal/tests/test_wf372b2_tectonic_sn_jnl.py`
**Compile command:** `tectonic --keep-logs <tmp>/wf372b2_smoke.tex` (in tmp dir with `sn-jnl.cls` copied in)
**Source:**
```latex
\documentclass[springer]{sn-jnl}
\begin{document}
Test article.
\end{document}
```

**Compile result:**
```
note: Writing `wf372b2_smoke.pdf` (7.408203125 KiB)
```

**PDF verification:**
- File: `/tmp/.../wf372b2_smoke.pdf`
- Size: 7.4 KiB
- Format: `PDF document, version 1.5 (zip deflate encoded)` per `file(1)`
- Magic bytes: `%PDF-` PASS

## §3 Test results

| Test | Wall | Verdict |
|---|---:|---|
| `molmetal/tests/test_wf372b2_tectonic_sn_jnl.py::test_tectonic_sn_jnl_compile_produces_valid_pdf` | 0.96 s | **PASS** |

Pre-flight v3 not re-run by this sub-flow (CPU-only doc+test+verify; NOT Wf-372-introduced regression); preserved at Wf-370-B1 baseline **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL**.

## §4 Files changed

| Path | Change |
|---|---|
| `.tools/latex/sn-jnl.cls` | 2-line patch: `allowdisplaybreaks` `\providecommand` wrap + `\providecommand{\headerps@out}{}` |
| `.tools/latex/sn-jnl.cls.bak.wf372b2` | Original 55,331 B pre-patch backup (NEW) |
| `molmetal/tests/test_wf372b2_tectonic_sn_jnl.py` | NEW 1 test PASS in 0.96 s |

## §5 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED vs PROJECTED:** Compile SUCCESS is MEASURED (PDF on disk, magic bytes verified, test PASS). Speedup claims are N/A (this is a correctness sub-flow).
- **2 patches to upstream `sn-jnl.cls`:** documented with backup. Both are no-op-defensive; original Springer template behavior is preserved when `mathtools` + legacy hook macros are present.
- **0 paper/* touches:** READ-only on paper per TODO-84 + AGENTS.md §5.3. `paper/*` (existing `paper/main.tex` etc.) untouched.
- **0 retrain launches:** no GPU work per user voice "其他重新训练...以外的事情我都给你授权".
- **`sn-jnl.cls` is LPPL 1.3c maintained** (per file header); patches are defensive guards against newer LaTeX kernel hook management. Original `.bak.wf372b2` preserved for rollback if needed.

## §6 Status

**SHIPPED.** `sn-jnl` template + `tectonic` end-to-end pipeline is READY for USER_OK-gated paper write per Wf-368-B1 §7.2 batch 3. No follow-up required by this sub-flow.