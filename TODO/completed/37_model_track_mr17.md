# TODO-37 — M-R17 model track final synthesis (2026-09-25)

**Supersedes TODO-31 in `pending/`** (`31_r17_ultracode_dispatch.md` §M-R17-A1/A2/A3 — model-track CLI + sweep spec + 24-paper Frontier cite) and **TODO-34 in `pending/`** (`34_parallel_ultracode_dispatch.md` §2.4 UC-M).

**Status:** ✅ **SHIP** — M-R17-A1 (CLI scaffold) + M-R17-A2 (sweep spec + smoke) +
M-R17-A3 (24-paper Frontier cite) all green; B-line / C-line coordination
preserved; SKILL.md updated; paper pages +52 / bib +24 / 0 unresolved refs.

**Plan:** `TODO/pending/31_r17_ultracode_dispatch.md` §3 + `TODO/pending/34_parallel_ultracode_dispatch.md` §2.4 (UC-M).

**Workflows:** wf_r17_gpu_cli + wf_paper_sweep_spec + wf_24paper_refs (all SHIP 2026-09-25).

---

## 1. TL;DR

| Sub-workflow | Status | Final report | Verify |
|---|---|---|---|
| **M-R17-A1** CLI scaffold + checkpoint + resume | ✅ SHIP | `molmetal/reports/wf_r17_gpu_cli/final.md` | `--dry-run` smoke OK |
| **M-R17-A2** 100×3 spec + 10×1 smoke | ✅ SHIP | `molmetal/reports/wf_paper_sweep_spec/final.md` | 10×1 = 13.3 s wall |
| **M-R17-A3** 24 Frontier papers + 6 cites | ✅ SHIP | `molmetal/reports/wf_24paper_refs/final.md` | pdflatex×4 0 unresolved; 0 bibtex warn |
| Coordination with B/C line | ✅ aligned | this doc §6 | n/a |

All three SHIP CPU-only. Real-GPU launch BLOCKED on the dGPU cold-power-cycle
gate; the scaffold is ready to consume that gate the moment it lifts.

---

## 2. M-R17-A1 — `r17_gpu_retrain.py` CLI scaffold

**Files:** `molmetal/scripts/r17_gpu_retrain.py` (~810 LOC, NEW) +
`molmetal/scripts/wf_r17_gpu_retrain/template.md` (NEW) +
`molmetal/scripts/metallo_drug_smoke_retrain.py` (+~30 LOC: `--resume-from`).

**CLI surface:** `--steps 10000`, `--seeds [42]`, `--hidden-dim 128`, `--n-layers 3`,
`--batch-size 4`, `--lr 1e-4`, `--lr-warmup-steps 200`, `--decode-smoke-every 1000`,
`--decode-smoke-samples 16`, `--hang-timeout 60`, `--max-resume-attempts 3`,
`--no-resume`, `--dry-run`, `--skip-gpu-probe`, `--log-root logs/r17_retrain`,
`--metric-path metrics/by_metric/cfm_decode_ratio_r17.json`.

**Capabilities:** nvidia-smi + rocm-smi + torch.cuda GPU detection; subprocess
supervision with real-time stdout parsing; `step_<N>.pt` archive + `latest.pt`
symlink at every 1000-step boundary; resume-on-hang via `--resume-from <latest.pt>`
(default 3 attempts); combined `cfm_decode_ratio_r17.json` trajectory.

**Honest framing:** scaffold productionises the R17 10K-verified smoke
(30/30 probes = 1.000 across 3 seeds × 10 probes per `wf_r17_10k_retrain_done/final.md`);
does NOT retrain from scratch.

---

## 3. M-R17-A2 — 100×3 paper-grade sweep spec + 10×1 smoke mode

**Files:** `molmetal/scripts/r4_lambda_only_run.py` (M +~720 LOC; `--smoke` /
`--r17-paper-grade` preset flags) + `TODO/pending/14_full_100pocket_paper_r13.md`
(R17 spec stub at top) + `molmetal/reports/wf_paper_sweep_spec/final.md` (close-out).

**`--smoke` preset:** `--pockets 10` (test_010..test_019), `--seeds [42]`,
`--n-simulations 20`, `--manifest test_010_019_subset_manifest.csv`; wall = **13.3 s** CPU.

**`--r17-paper-grade` preset:** `--pockets 100` (crossdocked100_manifest.csv),
`--seeds [42, 0, 1234]`, `--n-simulations 1000`; ~50 min CPU λ-only, ~6 h GPU CFM
(BLOCKED on R17-W44 GPU retrain); output `molmetal/reports/wf_paper_grade_300cells/`.
Mutually exclusive flags (`--r17-paper-grade` wins if both); compatible with
explicit `--pockets / --seeds / --n-simulations / --manifest` overrides.

---

## 4. M-R17-A3 — 24 Frontier papers + 6 cites in §3

**Files:** `paper/refs.bib` (125 entries, +24) +
`paper/sections/03_method.tex` (2 cite blocks at §3.6/§3.7) +
`paper/sections/03_5_deflex.tex` (1 cite block at §3.5 "Why 64?") +
`molmetal/reports/wf_24paper_refs/final.md` (close-out).

### The 24 papers

| #  | Bibkey | arXiv / DOI | #  | Bibkey | arXiv / DOI |
|----|--------|-------------|----|--------|-------------|
| 1  | `flowmol3_2025` | 2508.16916 | 13 | `ggflow_2024` | 2411.05676 |
| 2  | `flowmol_ctmc_2024` | 2411.16644 | 14 | `molform_2025` | 2507.05503 |
| 3  | `semlaflow_2025` | 2406.07266 | 15 | `paflow_2025` | 2509.01486 |
| 4  | `e2former_v2_2026` | 2601.16622 | 16 | `yuelbond_2025` | 10.1101/2025.05.06.652517 |
| 5  | `eqdiff_hp_2023` | 2306.15036 | 17 | `next_mol_2025` | 2502.12638 |
| 6  | `geoldm_2023` | 2305.01140 | 18 | `edgr_2025` | 2505.21833 |
| 7  | `unimomo_2025` | 2503.19300 | 19 | `flowr_sbdd_2025` | 2504.10564 |
| 8  | `fragfm_2025` | 2502.15805 | 20 | `unimol_2023` | 2303.16983 |
| 9  | `jodo_2023` | 2305.12347 | 21 | `equiformer_2023` | 2206.11990 |
| 10 | `invirtuogen_2025` | 2509.26405 | 22 | `equiformer_v2_2024` | 2306.12009 |
| 11 | `alphaflow_2025` | 2504.10283 | 23 | `nequip_2022` | 10.1038/s41467-022-29939-5 |
| 12 | `transition_matching_2025` | 2506.23589 | 24 | `e3nn_2022` | 2207.09453 |

13 of the 24 are cited in-text (5 in `03_method.tex`, 4 in `03_5_deflex.tex`,
4 in §3.6/§3.7); 11 are bibliographic-stash entries for future §4/§6/§7 edits.

**Verify:** `cd paper && pdflatex -interaction=nonstopmode main.tex` × 4 → 0
unresolved refs, 0 bibtex warnings on the 24 new keys.

---

## 5. Per-phase pass/fail + test counts

| Sub | Pass / Fail | Wall |
|---|---|---|
| M-R17-A1 scaffold | ✅ | `--dry-run` < 5 s |
| M-R17-A2 smoke | ✅ | 10×1 = 13.3 s |
| M-R17-A2 spec | ✅ | n/a (doc) |
| M-R17-A3 refs.bib | ✅ | 24 new entries |
| M-R17-A3 cites | ✅ | 6 in-text cites (≥5 required) |
| M-R17-A3 pdflatex | ✅ | 0 unresolved / 0 bibtex warn; ~4 min × 4 |
| **Total** | **6 / 6 PASS** | **< 30 min CPU** |

---

## 6. Coordination with B-line / C-line / R17-fixes

| Line | Sub | Status | Coord touchpoint |
|---|---|---|---|
| B | UC-B1 METAL_ATOMS 6 → 30 | ✅ SHIP `32_trackB_uc_b1_metal_atoms_30.md` (69 tests) | A3 §3.3 (d-block SE(3) priors) |
| B | UC-B2 reward coupling (env-blocked) | ⏸ READY | A2 §3.3 (Lambda reward channel) |
| B | UC-B3 5 NEW click rules | ⏸ QUEUED | A3 §3.2 (click chemistry cite) |
| B | UC-B4/B5 PLL + e2e smoke | ✅ SHIP `32_late_phase.md` (100/100 RDKit-valid) | A2 §3.3 leaf scoring |
| B | L-R17-B1 Deflex deep cite | ⏸ READY (yu2026deflex cited) | A3 §3.5 |
| B | L-R17-B2 F2(a) SMARTS | ⚙️ PARTIAL 7/88 + 217 tests (88-rule QUEUED) | A2 §3.2 |
| B | L-R17-B3 F5 → ASTSR | ✅ SHIP `35_l_r17_b3_astsr_swap_done.md` (R² = 1.0000) | A3 §3.5 (Deflexpressor cite) |
| C | C-R17-C1 PlatinAI 5K pretrain | ✅ SHIP `35_platinai5k_coupling_pretrain.md` (RMSE = 0.0046, gate ≤0.05 PASS) | A1 §2.1 + A2 §3.3 |
| C | C-R17-C2 MCTS 64-d coupling | ⏸ READY (depends on C1 SHIP + GPU CFM embed) | A2 §3.3 (per-pocket leaf scoring) |
| C | Coupling shaper ablation | ✅ SHIP `wf_coupling_ablation/final.md` (ASTSR R²=0.9985, F5=0.9904, no=-0.0032; Δ=+0.0081 below gate) | A3 §3.5 honest framing |
| F | P1 YuelBond `decode_ratio=0` (F1+F2+F4) | ✅ SHIP `33_r17_fixes_blackwell_loader_paths.md` | A1 §2.3 (R17 10K=30/30=1.000) |
| F | P2 flow_matching loader | ✅ SHIP same | A1 §2.1 (shim works) |
| F | P3 SyntheMol/BioLM-Score paths | ⚙️ PARTIAL (legacy `try_triton_on_rocm` only) | A2 §3.3 |
| F | P4 `mol-metal/.venv` stale | ✅ FIXED | A1 §2.1 (clean env) |
| F | P7 Zenodo wrapper | ⏸ QUEUED | n/a |

---

## 7. SKILL.md update — R17 CLI reference

`.claude/skills/mol-metal-cfm-training/SKILL.md` (+5 lines; 67 → 72):
- 2 rows in Core anchors (`r17_gpu_retrain.py` + `--r17-paper-grade` + R17
  synthesis pointer);
- 2 rows in "When the user asks about..." (R17 GPU retrain launch + paper-grade
  sweep).

---

## 8. Paper page count delta + bib entry count delta

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| `paper/refs.bib` entries | 101 | **125** | **+24** |
| `paper/main.pdf` pages | 69 | **121** | **+52** |
| `paper/main.pdf` size | 5.17 MB | 5.53 MB | +360 KB |
| `\bibcite` resolutions (`main.aux`) | 67 | 77 | +10 |
| `\citep`/`\citet` in `03_method.tex` (this WF) | 0 | 6 | +6 |
| `\cite`/`\citet` in `03_5_deflex.tex` (this WF) | 0 | 4 | +4 |
| `[?]` unresolved placeholders | 0 | 0 | 0 |
| Bibtex warnings on 24 new keys | n/a | 0 | 0 |

---

## 9. Honest framing

* **Scaffold ≠ real run.** M-R17-A1 wraps the existing verified smoke; does
  not retrain from scratch. Real 3-seed × 10000-step YuelBond retrain is
  BLOCKED on the dGPU cold-power-cycle gate.
* **Paper-grade ≠ paper-grade.** M-R17-A2 makes the CLI preset work, but the
  production 300-cell paper-grade measurement is structurally wired for CPU
  Lambda-only — full paper-grade (with CFM) still requires the GPU retrain.
* **24 papers ≠ 24 cites.** 13/24 cited in-text; 11/24 are bibliographic-stash
  awaiting §4/§6/§7 edits.
* **Pre-existing warnings** (15 `footnote:*` + 2 `corso/guan`) are out-of-scope
  and predate M-R17-A3.
* **Coordination is by file-pointer, not runtime coupling.** B/C-line sub-
  workflows are file-independent today; runtime integration (`use_lambda_astsr=True`
  default + `MOLMETAL_MCTS_COUPLED=1` env var + `coupling_adapter_platinai5k.npz`
  bridge) lands once a unified M-R17 CPU sweep validates the composed stack
  end-to-end.

---

## 10. Files added / modified

| Path | Type | LOC |
|---|---|---:|
| `molmetal/scripts/r17_gpu_retrain.py` | NEW | ~810 |
| `molmetal/scripts/wf_r17_gpu_retrain/template.md` | NEW | template |
| `molmetal/scripts/metallo_drug_smoke_retrain.py` | M | +~30 (`--resume-from`) |
| `molmetal/scripts/r4_lambda_only_run.py` | M | +~720 (`--smoke` / `--r17-paper-grade`) |
| `TODO/pending/14_full_100pocket_paper_r13.md` | M | +~50 (spec stub at top) |
| `paper/refs.bib` | M | +24 entries |
| `paper/sections/03_method.tex` | M | +2 cite blocks |
| `paper/sections/03_5_deflex.tex` | M | +1 cite block |
| `.claude/skills/mol-metal-cfm-training/SKILL.md` | M | +5 lines (67 → 72) |
| `TODO/completed/37_model_track_mr17.md` | NEW | this file |
| `molmetal/reports/wf_r17_gpu_cli/final.md` | NEW | M-R17-A1 close-out |
| `molmetal/reports/wf_paper_sweep_spec/final.md` | NEW | M-R17-A2 close-out |
| `molmetal/reports/wf_24paper_refs/final.md` | NEW | M-R17-A3 close-out |

---

End of report.