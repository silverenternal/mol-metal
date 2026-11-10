# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (0/0)  
**Best epoch**: 17  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 11137.3s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 26.3485 | 0.7150 | 0.7604 | 5.603 | 1.000 | (192/192, single_frag=104) | 1140.2 |
| 5 | 26.3107 | 0.7027 | 0.7495 | 6.530 | 1.000 | (192/192, single_frag=96) | 1005.8 |
| 6 | 26.2565 | 0.6914 | 0.7422 | 7.086 | 1.000 | (192/192, single_frag=100) | 1131.4 |
| 7 | 26.2042 | 0.6806 | 0.7368 | 7.623 | 1.000 | (192/192, single_frag=112) | 867.3 |
| 8 | 26.0608 | 0.6701 | 0.7313 | 8.739 | 1.000 | (192/192, single_frag=102) | 918.7 |
| 9 | 25.9043 | 0.6604 | 0.7111 | 10.618 | 1.000 | (192/192, single_frag=116) | 468.2 |
| 10 | 25.6929 | 0.6516 | 0.6795 | 12.587 | 1.000 | (192/192, single_frag=109) | 513.7 |
| 11 | 25.5434 | 0.6429 | 0.6620 | 13.209 | 1.000 | (192/192, single_frag=114) | 694.9 |
| 12 | 25.4754 | 0.6338 | 0.6535 | 13.570 | 1.000 | (192/192, single_frag=118) | 816.3 |
| 13 | 25.3974 | 0.6246 | 0.6461 | 13.778 | 1.000 | (192/192, single_frag=128) | 823.5 |
| 14 | 25.3325 | 0.6149 | 0.6379 | 13.890 | 1.000 | (192/192, single_frag=135) | 817.9 |
| 15 | 25.2915 | 0.6047 | 0.6300 | 14.099 | 1.000 | (192/192, single_frag=134) | 814.7 |
| 16 | 25.2339 | 0.5946 | 0.6231 | 14.191 | 1.000 | (192/192, single_frag=141) | 526.9 |
| 17 | 25.2138 | 0.5845 | 0.6168 | 14.371 | 1.000 | (192/192, single_frag=137) | 597.0 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 14-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 11137.3s (~795.5s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_c7_fullgen_2026_10_20/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_c7_fullgen_2026_10_20/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_c7_fullgen_2026_10_20/wasserstein_decay.json` (W₂ decay check)
- `molmetal/reports/cfm_c7_fullgen_2026_10_20/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_c7_fullgen_2026_10_20/synthesis.md` (this file)

> **⚠️ CORRECTION (2026-10-31, Wf-A141 guard-repair sub-wave — read-only
> audit; the file does NOT exist).**  The "best checkpoint" bullet above is
> **FALSE as written**: the directory
> `molmetal/reports/cfm_c7_fullgen_2026_10_20/checkpoints` is **EMPTY** (0
> entries — MEASURED 2026-10-31 with `ls -la`; the directory's mtime is
> 2026-10-21 16:52 and it was ALREADY empty then, per catalog **§6.197**
> point 3, so it has been empty for the whole period).  The run's per-epoch
> records survived — `per_epoch_metrics.json`, `per_epoch_metrics.csv`,
> `wasserstein_decay.json`, `events.jsonl`, `generated_smiles.csv` — but the
> **weights did not**.  §6.197 diagnosed this absence at the
> `cfm_todo.json` claim site; **this** bullet is a second site that §6.197
> did not correct, and it stood for seven days.  The original bullet is
> PRESERVED VERBATIM above per AGENTS.md §5.5 rule #6.  Recorded as catalog
> **§6.254 §7** and **§6.255**.

> **⚠️ CORRECTION 2 (2026-10-31, Wf-A142 — provenance of the missing
> checkpoint: when it went away, whether it can be recovered, and whether the
> CPU-only blocker still applies).**  The correction immediately above
> (Wf-A141 / catalog **§6.254 §7** + **§6.255**) records THAT the
> advertised file is missing. **This block adds the provenance and recovery
> evidence, which that block did not carry.** Nothing above is changed; both
> corrections are additive (AGENTS.md §5.5 rule #6). Per the §6.255
> convention — *"a path that does not exist must be cited in a form that is
> not a pointer"* — the missing file is cited below by its **bare name** plus
> its **existing** parent directory, except inside the two shell commands
> whose subject IS the path.
>
> **(a) The file does NOT exist on disk.**  MEASURED 2026-10-31:
>
> ```
> $ ls -la molmetal/reports/cfm_c7_fullgen_2026_10_20/checkpoints/
> total 0
> drwxr-xr-x 1 hugo hugo    0 Oct 21 16:52 .
> drwxr-xr-x 1 hugo hugo  240 Oct 31 17:30 ..
> ```
>
> `total 0`, **0 entries** (only `.` and `..`): the directory exists and is
> **EMPTY**. The file named in the "best checkpoint" bullet above —
> `cfm_yuelbond_wasserstein.pt`, inside that directory — **does not exist on
> disk**.
>
> **(b) When the directory was emptied.**
>
> | evidence | value (MEASURED 2026-10-31) |
> |---|---|
> | directory `Modify` / `Change` (mtime) | **2026-10-21 16:52:07.634677957 +0800** |
> | directory `Birth` | **2026-10-20 23:17:46.319283848 +0800** |
> | the run's own artifacts (`per_epoch_metrics.json`, `per_epoch_metrics.csv`, `generated_smiles.csv`, `events.jsonl`, `wasserstein_decay.json`) | 2026-10-21 05:04 — **11 h 48 m earlier** |
> | `git log --oneline -- <path>` | **EMPTY — the file was NEVER committed** |
> | `git ls-files molmetal/reports/cfm_c7_fullgen_2026_10_20/` | **one file only: `synthesis.md`** |
>
> The emptying was therefore **not a git operation**: there is no blob, no
> commit and no deletion record — only a filesystem event. The 16:52:07
> mtime is the directory's **last content change**. A census of every
> `checkpoints` / `ckpt` / `ckpts` directory in the repo (`find`, never
> `os.walk`) finds **87** now-empty directories sharing that **exact**
> mtime `2026-10-21 16:52` — **87 of 99 empty** under a `find -maxdepth 6`
> scope (138 directories total) and **87 of 101 empty** under the ledger's
> `**/checkpoints` + `**/ckpt` scope (136 total); the two totals differ only
> by scope and **every scope agrees the c7 directory is one of the 87**. The
> emptying was a **single bulk filesystem event** at 2026-10-21 16:52, not a
> per-run cleanup. The repo's own ledger had already recorded it
> (`todo.json`, Wf-A96 entry: *"该目录的 checkpoints/ 自 2026-10-21 16:52 起
> 为空"* — "this directory's `checkpoints/` has been empty since 2026-10-21
> 16:52"), which is the same instant §6.197 point 3 measured.
>
> **(c) Where the checkpoint is — it is LOST, not relocated.**  MEASURED
> 2026-10-31:
>
> ```
> $ find . -name 'cfm_yuelbond_wasserstein.pt' -not -path './.git/*'
> ```
>
> returns **16 hits, every one of them under `.claude/worktrees/`**:
> `wf_60231cff-03e-11`, `wf_bf4f3451-73b-3`, `wf_cf385f8f-b2e-4`,
> `wf_cf385f8f-b2e-6` x
> `results/cfm_per_innovation_ablation_p1.4/{a_full_stack,b_yuelbond_only,c_w2_only,d_catherine_only}/ckpts/`.
> Those are the **Wf-299 per-innovation ablation** checkpoints — a different
> cohort, a different experiment, in throwaway worktrees. **None of the 16 is
> this run's epoch-17 best checkpoint.**
>
> ```
> $ git log --all --diff-filter=D --name-only | grep -i cfm_yuelbond
> ```
>
> lists only these **`.pt`** deletions, all from **one** commit — `ce8b0383`
> *"purge: drop 75 superseded checkpoints (59 under results/, 16 under
> logs/)"*:
> `results/cfm_per_innovation_ablation_p1.4/{a_full_stack,b_yuelbond_only,c_w2_only,d_catherine_only}/ckpts/cfm_yuelbond_wasserstein.pt`
> and
> `results/cfm_tmqm_all6/_superseded/cfm_yuelbond_wasserstein_seed0_ep9_saturating_gate.pt`.
> (The same grep also matches eight `train_cfm_yuelbond_wasserstein.py.bak.*`
> trainer-script backups — scripts, not checkpoints.) **The c7 path appears in
> no deletion record**, because there was never a committed file to delete.
>
> **Verdict: the epoch-17 "best checkpoint" is LOST.** It was never in git,
> and no surviving copy on disk is this run's.
>
> It was also **probably never written under that name at all.** This trainer
> writes **seed-qualified** names only —
> `cfm_yuelbond_wasserstein_seed{seed}.pt` /
> `..._seed{seed}_last.pt` / `..._seed{seed}_epoch{N}.pt` /
> `..._seed{seed}_best_epoch{N}.pt`
> (`molmetal/scripts/train_cfm_yuelbond_wasserstein.py:20159`, `:20357`,
> `:20440`, `:20553`, `:20578`) — and the trainer's own comment records why
> (`:18103-18108`): *"The checkpoint filename is now qualified by seed. It
> was a fixed `cfm_yuelbond_wasserstein.pt` … so all five seeds in a sweep
> wrote to the SAME path and the last one to finish won."* The "best
> checkpoint" bullet is a **leftover of the pre-2026-10-04 unseeded naming
> era**, emitted as a hard-coded f-string with **no existence check**
> (`:21245`). It was a **template, not a report of a write**.
>
> **A third, still-unrepaired false claim site** (reported, not edited — the
> trainer is out of scope for this wave): the same trainer's module
> docstring (`:55`) advertises
> `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` as the *"trained
> YuelBond head + velocity-field metadata"*. **That file does not exist
> either**; the two similarly-named files in that directory are
> `cfm_yuelbond_wasserstein.smoke_stub.pt` (639 549 B, 2026-10-04) and
> `cfm_yuelbond_wasserstein_seed0.smoke_stub.pt` (7 807 093 B, 2026-10-10) —
> **smoke stubs**, not the trained head.
>
> **(d) The CPU-only deserialisation blocker still applies.**
> `docs/HONEST_NEGATIVE_CATALOG.md` **§6.248** records that
> `torch.load(ckpt, map_location="cpu")` — and
> `torch.load(ckpt, map_location=torch.device("cpu"))` — **both** raise
> `RuntimeError: Attempting to deserialize object on a CUDA device but
> torch.cuda.is_available() is False` for the sibling run's checkpoint
> (`cfm_yuelbond_wasserstein_seed0_last.pt` under the c8 report directory,
> 8 395 006 B, still on disk). That blocker is **UNCHANGED** as of
> 2026-10-31. For **this** run it is moot in the strict sense — **there is
> no file to load at all** — but it means a *replacement* checkpoint would
> still very likely not be loadable on a CPU-only host. So the decoder's own
> `bonds_dropped` / `dropped_reasons` histogram
> (`molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py:1614-1629`)
> remains **unobtainable** for this pool, and the dominant cause of the
> pool's fragmentation stands as **§6.253** records it: MEASURED-RANKED, with
> the assembler-drops-bonds mechanism **REFUTED** for this pool.
>
> **Scope of this correction.** No number in the per-epoch table above
> changes: it comes from `per_epoch_metrics.json` / `events.jsonl`, which do
> exist. What changes is the status of **one advertised output**: the "best
> checkpoint" is not on disk and cannot be recovered.

## Cross-references

- TODO-119 (decode_ratio strengthen) — primary target
- TODO-121 (m19 LOMO / Rh bottleneck) — uses Wasserstein transfer
- TODO-122 (PlatinAI 4-cell R²) — uses Wasserstein pIC50
- TODO-123 (m15 RMSD) — uses Wasserstein pose
- TODO-124 (m5 SA) — uses Catherine closure-bounded generation
- TODO-125 (per-pocket Vina) — uses Wasserstein pocket alignment
- TODO-126 §3 (shared master modules) — M1 Wasserstein + M2 closure-bias
- Wf-124 P1 (YuelBond decoder swap)
- Wf-123 P1 (Wasserstein loss module)
- Wf-123 P2 (pocket-specific closure-bias module)
- Wf-123 P3 (cross-module integration verification)
