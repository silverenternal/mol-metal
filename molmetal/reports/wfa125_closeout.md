# Wf-A125 — Filesystem-Truth Evidence (2026-10-31)

**Wave**: Wf-A125 (CPU-only, **0 GPU**) · **Generated**: 2026-10-31
**Purpose**: AGENTS.md §5.5 filesystem truth — "SHIPPED" requires the file to exist on disk, and every claim must be backed by an `ls -la` / `sha256sum` that was actually run on 2026-10-31.

All paths are relative to the repository root `/home/hugo/codes/mol-metal`. Sizes are bytes; mtimes are UTC. The **git state** column is load-bearing: it is what separates "works on this machine" from "ships".

---


## 1. MODIFIED — 9 wave files + 3 close-out files

| path | size | mtime (UTC) | git state |
|---|---|---|---|
| `molmetal/tests/conftest.py` | 29,268 B | 2026-10-31 01:49:20 | tracked |
| `molmetal/scripts/pre_flight_check.py` | 646,402 B | 2026-10-31 01:56:43 | tracked |
| `molmetal/scripts/registry.py` | 97,237 B | 2026-10-31 01:50:31 | tracked |
| `molmetal/scripts/run_ledger.py` | 35,582 B | 2026-10-31 01:54:05 | tracked |
| `molmetal/scripts/wf_cfm_run_census.py` | 153,401 B | 2026-10-31 01:53:24 | tracked |
| `metrics/by_metric/wf32_pb_valid.json` | 4,175 B | 2026-10-31 01:51:18 | tracked |
| `metrics/by_metric/cfm_run_census.json` | 551,977 B | 2026-10-31 01:53:28 | tracked |
| `registry/metrics.yaml` | 676,885 B | 2026-10-31 02:31:32 | tracked |
| `molmetal/reports/run_ledger.jsonl` | 203,678 B | 2026-10-31 01:54:13 | tracked |
| `docs/HONEST_NEGATIVE_CATALOG.md` | 1,642,381 B | 2026-10-31 03:02:21 | tracked |
| `AGENTS.md` | 26,782 B | 2026-10-31 02:58:32 | tracked |
| `TODO/INDEX.md` | 203,007 B | 2026-10-31 03:04:08 | tracked |

## 2. NEW — 6 wave files + 2 close-out files

| path | size | mtime (UTC) | git state |
|---|---|---|---|
| `conftest.py` | 8,137 B | 2026-10-31 01:50:45 | **GIT-IGNORED** |
| `molmetal/tests/test_wfa125_root_conftest_gpu_lock.py` | 8,829 B | 2026-10-31 01:50:07 | **untracked (`??`)** |
| `molmetal/tests/test_wfa125_gpu_contract.py` | 8,957 B | 2026-10-31 01:50:20 | **untracked (`??`)** |
| `molmetal/tests/test_wfa125_published_tree_guard.py` | 15,657 B | 2026-10-31 01:57:36 | **untracked (`??`)** |
| `molmetal/tests/test_wfa125_preflight_warn_contract.py` | 12,095 B | 2026-10-31 02:00:24 | **untracked (`??`)** |
| `molmetal/tests/test_wfa125_registry_sha_refresh.py` | 14,057 B | 2026-10-31 02:02:12 | **untracked (`??`)** |
| `TODO/completed/wfa125_guard_and_gate_residuals.md` | 26,217 B | 2026-10-31 03:05:41 | **untracked (`??`)** |
| `molmetal/reports/wfa125_closeout.md` (this file) | 8,325         B | *(self)* | **untracked (`??`)** |

> **NONE** of the wave's five new test files is tracked, and the repo-root `conftest.py` is **git-ignored** — so G1 / G9 / G10 are measurable on this machine and **do not ship** to a fresh clone. See catalog §6.224, §6.232, §6.233.


## 3. sha256 of every NEW file

| path | bytes | sha256 |
|---|---|---|
| `conftest.py` | 8,137 | `b2fb1f66cfb5cd064f5e4137499d8d0a4ac047745cc7f02520d2a24d2bc0f66e` |
| `molmetal/tests/test_wfa125_root_conftest_gpu_lock.py` | 8,829 | `f2a1c6cfc4e3e7dea050a59606dbf12905666fd3f63ae0a4758b52d7c0edd8bc` |
| `molmetal/tests/test_wfa125_gpu_contract.py` | 8,957 | `9cf8bbab5a2a5128832ddbc263f9c983f17bee860c7f1dc3e8bd6c79a165caef` |
| `molmetal/tests/test_wfa125_published_tree_guard.py` | 15,657 | `3e6ecb5de7aa48459fdcd04f9d2b6b3c70135f06829c332f82d4b56cee510c50` |
| `molmetal/tests/test_wfa125_preflight_warn_contract.py` | 12,095 | `c0f27de499a5c6dc7a081cbfb29740fda61c1b5c6a0318dec6ef2977c7196a0c` |
| `molmetal/tests/test_wfa125_registry_sha_refresh.py` | 14,057 | `8be2bc53696aa819a2036690c3bbb4bfa5068e6f78d3cb11fe94b6bdc1d7fa5b` |
| `TODO/completed/wfa125_guard_and_gate_residuals.md` | 26,217 | `962c8bc41be9677943bd68dda77de41efcf761e93b2ba0789beeb58982b25eff` |
| `molmetal/tests/PUBLISHED_WRITE_DECLARATIONS.json` (**NOT** changed by the wave — byte-identical) | 1,432 | `53ba26a6466128d741be1eb5609a4609b30876bedb6142b157904394690e0de1` |

## 4. sha256 of the MODIFIED wave files (for the record)

| path | bytes | sha256 |
|---|---|---|
| `molmetal/tests/conftest.py` | 29,268 | `6cbfdeaa04f45e1d20dd151d178cd651c5c0554f451dfea43cba2844fd32799c` |
| `molmetal/scripts/pre_flight_check.py` | 646,402 | `40f4699a9c4c717615f4f5235e61691723ff4339da2747b6cde4aa673d80d945` |
| `molmetal/scripts/registry.py` | 97,237 | `daacdac92a05b647b591184caeef825cb2c8ae310d42f01c290a4362072a5c0d` |
| `molmetal/scripts/run_ledger.py` | 35,582 | `d51c5a6c955b6ea7db84faa453da7fec9f6f913fc734fc97c7a36020321d9ede` |
| `molmetal/scripts/wf_cfm_run_census.py` | 153,401 | `3b68bb1b5be9e8f081fe3304f503c02a2b430d4e862be7c4eb963f64858ecb8a` |
| `metrics/by_metric/wf32_pb_valid.json` | 4,175 | `d6eec79ee513e8427939b79096fd462b62b70d2aa648a04cf0e4338f951b6127` |
| `metrics/by_metric/cfm_run_census.json` | 551,977 | `e85c529e49e034f121e2a1a16878878239cb986139a23b0836bdc463292e4cf9` |
| `registry/metrics.yaml` | 676,885 | `f39d901fccdb37672404ad2dfe825bca16067b3c16f35a1ef558e960a7bfd517` |
| `molmetal/reports/run_ledger.jsonl` | 203,678 | `de2272dd125fe184bc52ac17e020346d9351cf5a15e7ae7f2fa87167c82d0809` |
| `docs/HONEST_NEGATIVE_CATALOG.md` | 1,642,381 | `365b13a601f27104bf98f958d4e9e327652fed0c21f008d2d821e80044ebc0d2` |
| `AGENTS.md` | 26,782 | `d1bdc5829297914c2df18f92b7f8be0a762bbaaf20fc94b4878290ff823b8ccd` |
| `TODO/INDEX.md` | 203,007 | `99ee1eded96a235b75a1e59c026f567a8d00a9b83e748b597d4ffd2f2b27ca74` |

## 5. `ls -la` — raw

```
-rw-r--r-- 1 hugo hugo   26782 2026-10-31 10:58:32.836867598 +0800 AGENTS.md
-rw-r--r-- 1 hugo hugo    8137 2026-10-31 09:50:45.094531626 +0800 conftest.py
-rw-r--r-- 1 hugo hugo 1642381 2026-10-31 11:02:21.692478876 +0800 docs/HONEST_NEGATIVE_CATALOG.md
-rw-r--r-- 1 hugo hugo  551977 2026-10-31 09:53:28.038283613 +0800 metrics/by_metric/cfm_run_census.json
-rw-r--r-- 1 hugo hugo    4175 2026-10-31 09:51:18.522045543 +0800 metrics/by_metric/wf32_pb_valid.json
-rw-r--r-- 1 hugo hugo  203678 2026-10-31 09:54:13.643757052 +0800 molmetal/reports/run_ledger.jsonl
-rw-r--r-- 1 hugo hugo  646402 2026-10-31 09:56:43.657100156 +0800 molmetal/scripts/pre_flight_check.py
-rw-r--r-- 1 hugo hugo   97237 2026-10-31 09:50:31.950329915 +0800 molmetal/scripts/registry.py
-rw-r--r-- 1 hugo hugo   35582 2026-10-31 09:54:05.389628615 +0800 molmetal/scripts/run_ledger.py
-rw-r--r-- 1 hugo hugo  153401 2026-10-31 09:53:24.991000827 +0800 molmetal/scripts/wf_cfm_run_census.py
-rw-r--r-- 1 hugo hugo   29268 2026-10-31 09:49:20.404235929 +0800 molmetal/tests/conftest.py
-rw-r--r-- 1 hugo hugo    8957 2026-10-31 09:50:20.274150915 +0800 molmetal/tests/test_wfa125_gpu_contract.py
-rw-r--r-- 1 hugo hugo   12095 2026-10-31 10:00:24.581574869 +0800 molmetal/tests/test_wfa125_preflight_warn_contract.py
-rw-r--r-- 1 hugo hugo   15657 2026-10-31 09:57:36.458928361 +0800 molmetal/tests/test_wfa125_published_tree_guard.py
-rw-r--r-- 1 hugo hugo   14057 2026-10-31 10:02:12.627281831 +0800 molmetal/tests/test_wfa125_registry_sha_refresh.py
-rw-r--r-- 1 hugo hugo    8829 2026-10-31 09:50:07.671957916 +0800 molmetal/tests/test_wfa125_root_conftest_gpu_lock.py
-rw-r--r-- 1 hugo hugo  676885 2026-10-31 10:31:32.622519003 +0800 registry/metrics.yaml
-rw-r--r-- 1 hugo hugo   26217 2026-10-31 11:05:41.738378974 +0800 TODO/completed/wfa125_guard_and_gate_residuals.md
-rw-r--r-- 1 hugo hugo  203007 2026-10-31 11:04:08.655371686 +0800 TODO/INDEX.md
```

## 6. Self-measurement note

This report (`molmetal/reports/wfa125_closeout.md`) records its own **size** above (measured after the write, via a fixed-width placeholder so the byte count could not go stale). It does **not** record its own **sha256**: a file cannot contain its own digest. The post-write measurement is quoted in the wave close-out's final output instead.


## 7. Verification commands (run 2026-10-31, CPU-only)

```
grep -c "^## §6\." docs/HONEST_NEGATIVE_CATALOG.md
grep -oE "^## §6\.[0-9]+" docs/HONEST_NEGATIVE_CATALOG.md | grep -oE "[0-9]+$" | sort -n | tail -1
git diff --numstat docs/HONEST_NEGATIVE_CATALOG.md
git status --porcelain -- paper/
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -i mol-metal
```

---

**0 GPU.** Every python / pytest invocation for this wave was prefixed `CUDA_VISIBLE_DEVICES=""`. No training, sampling, or sweep was launched; no watchdog was started.
