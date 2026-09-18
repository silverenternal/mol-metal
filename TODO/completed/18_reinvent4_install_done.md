# Install REINVENT4 reinvent binary (pipx or docker)

**Status:** pending
**Priority:** high
**Effort:** 0.5d
**Owner:** (unset)
**Depends on:** none
**Blockers:** pipx available on host, or docker daemon for container path
**Created:** 2026-09-12

## Goal
Install REINVENT4 so the `reinvent` CLI is invokable from the project's
adapter. Currently `reinvent4_subprocess_adapter.py` subprocess-launch
fails with `FileNotFoundError` because the binary is not on `$PATH`; the
adapter falls back to `score()→None` and the `RewardAggregator` treats
`None` as 0.0 — so the `r_reinvent4` channel is silently OFF.

## File(s) to edit
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py  (verify backend flag)
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/reinvent4_adapter.py  (if pipx-style install path is used)

## Success criterion
`reinvent --version` exits 0 on PATH; `reinvent4_subprocess_adapter.py.score()`
returns a `float` (not `None`); `lambda_round3_combined_report.md` L-4 is
marked **LIVE** (not OBS); `RewardAggregator` activates a 7-channel reward
aggregation (was 6 with `r_reinvent4` silently 0.0).

## Related reports
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_round3_L4_reinvent4_scorer.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_round3_combined_report.md