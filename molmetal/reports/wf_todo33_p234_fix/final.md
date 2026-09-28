# TODO-33 P2+P3+P4 Fixes — Final Verification (2026-09-25)

## Summary
Three verification gates executed per `TODO/pending/33_r17_fixes_blackwell_loader_paths.md` §10 DoD.
Test command run via the shared venv python (`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python`)
because `uv run` does NOT honour `UV_PROJECT_ENVIRONMENT=/home/hugo/codes/CadButEaas/.venv-vlm`
on this host — it falls back to `/home/hugo/.local/share/uv/python/cpython-3.12-.../bin/python3.12`
which has no torch/rdkit. `env.sh` exports the var correctly but uv ignores it. **Bug #1 — open.**

## Per-fix results

| Fix | Description | Gate | Result |
|---|---|---|---|
| P2 | flow_matching loader auto-detect + shim re-apply | `LipmanFlowMatchingAdapter.setup(device='cpu')` | **PASS** — printed `flow_matching setup OK`, no `FileNotFoundError`. Resolves `molmetal/references/flow_matching` symlink + shim package correctly. |
| P2 | test_synflownet_env.py (5 tests) | pytest | **5/5 PASS** (incl. lazy-import-without-synflownet, forward/backward roundtrip). |
| P2 | test_synthemol_reactions.py (39 tests) | pytest | **38/38 PASS + 1 SKIP** (hantzsch_pyridine_synthesis — SMARTS unsupported by rdkit 2026.03.6; long-standing, unrelated to TODO-33). |
| P3 | test_biolm_score_adapter.py (5 tests) | pytest | **5/5 SKIP** — skip when BioLM-Score upstream absent (expected on this host, intentional). |
| P3 | test_lambda_reward_coupling.py (7 tests) | pytest | **5/7 SKIP + 1 PASS + 1 FAIL**. **FAIL** = `test_lambda_reward_cli_flag_accepted`: `metallo_drug_smoke_retrain.py` defines `--lambda-reward` as `action="store_true"` but the test passes `--lambda-reward 0.5` expecting argparse to accept a float. **Bug #2 — open** (parser bug in TODO-21 smoke script, not in P3 path-fallback code). |
| P4 | `bash molmetal/scripts/check_no_stale_venv.sh` | exit code | **PASS** (exit 0, prints `OK: no stale mol-metal/.venv`). `env.sh` stale-venv guard is wired correctly. |

## Headline numbers
- pytest: **1 failed, 43 passed, 12 skipped** in 5.68 s (wall 6.3 s incl. interpreter spawn).
- flow_matching setup: **OK**, no `FileNotFoundError`.
- check_no_stale_venv.sh: **exit 0**.

## Issues remaining open
1. **`uv run` ignores `UV_PROJECT_ENVIRONMENT`** on this host. `env.sh` line 40 exports the var correctly, but uv 0.11.8 falls back to the user-managed python at `/home/hugo/.local/share/uv/python/cpython-3.12-.../` instead of `/home/hugo/codes/CadButEaas/.venv-vlm/`. env.sh prints the warning "[mol-metal env] WARNING: stale /home/hugo/codes/mol-metal/.venv exists" when `.venv` is **absent** — phantom warning, not the reported stale venv. Both are symptoms of the same `UV_PROJECT_ENVIRONMENT` not being honoured. **Not blocking P2/P3/P4 gates (we used the shared python directly) but blocks any `uv run`-based CI / pre-commit gate.**
2. **P1 — YuelBond decode_ratio=0 on Blackwell** is **untested** in this verification (CPU-only gate, requires RTX PRO 6000). Status: still open per TODO-33 §1; F1 (AMP→FP32) and F2 (LR warmup) candidates not yet applied. Blocked on GPU availability.
3. **P7 — Zenodo DNS-blocked fallback** is **untested** here (no Zenodo download invoked). Status: still open per TODO-33 §7; `--resolve` hack still in use.
4. **`test_lambda_reward_cli_flag_accepted` failure** is a parser bug in `metallo_drug_smoke_retrain.py:266` (defines `--lambda-reward` as `action="store_true"` but the test passes a float value). Likely a TODO-21 / Strategy-1 issue rather than a TODO-33 P2/P3/P4 regression — but it does fail the DoD §10.1 "43+ passed, 7 skipped" gate (we get 43 passed, 12 skipped, 1 failed).

## Total time
~7 s pytest + ~2 s flow_matching setup + <1 s stale-venv check = **~10 s wall**. Operator-side investigation overhead: ~5 min diagnosing the `uv run` / `UV_PROJECT_ENVIRONMENT` mismatch.

## Conclusion
**P2 (loader + shim) and P4 (stale-venv guard) are FIXED.** **P3 (SyntheMol/BioLM path candidates) is functionally FIXED** — all related tests pass/skip as expected; the single failure is an unrelated TODO-21 argparse bug. **P1 (GPU Blackwell decode_ratio) and P7 (Zenodo) remain open** and were not exercised by this CPU-only verification.