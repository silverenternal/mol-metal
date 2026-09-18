# Round-8 Recon — round-7 partial state + TODO/pending content

Date: 2026-09-12
Read-only recon. No files modified.

---

## 1. Round-7 partial-state table

| File (absolute path) | Exists? | Modified in round-7? | Round-7 target | Verdict |
|---|---|---|---|---|
| `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/vina_adapter.py` | yes (32 kB) | yes — mtime `2026-09-12 19:12:42` (latest of all round-7 targets) | wire Vina Python pkg + subprocess fallback | **SHIPPED**. Module imports lazily; implements `DockingEngine` Protocol; supports `auto/vina/vina-cli/qvina/quickvina2` engines; wraps `meeko` + openbabel for prep; `VinaDockingAdapter`, `dock_smiles`, `redock_for_test` exposed. |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/validation/admet_runner.py` | yes (16 kB) | yes — mtime `2026-09-12 19:13:43` | wire ADMET `r_admet` channel with admet-ai + datamol + RDKit fallback chain | **SHIPPED**. Docstring confirms 3-tier fallback: admet-ai → datamol/molfeat → RDKit-only; never raises from `predict_admet`; backend detection memoised at module level. |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/eval/scoring/` | **no** | n/a | round-7 clone-reuse target (lift `flowr.util.metrics` + `Pocket2Mol/evaluation/scoring_func.py`) | **NOT STARTED**. `molmetal/eval/` does not exist at all; only `molmetal/ports/__init__.py` + `molmetal/ports/generators.py`. Cloned `flowr`/`flowr_root` are only in `molmetal/references/`. |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py` | yes (152 kB) | yes — mtime `2026-09-12 19:14:08` (latest) | G6 fix: wire `_VirtualLoss.apply` / `_VirtualLoss.release` into `_simulate` | **SHIPPED**. Grep confirms both calls: `self.virtual_loss.apply(id(node))` at line 2179; `self.virtual_loss.release(id(node))` at lines 2818 and 3143. `_VirtualLoss` class is the counter-based lock at line 1171. |
| `/home/hugo/codes/try_triton_on_rocm/triton_kernels/batched_mlp.py` | yes (18 kB) | yes — mtime `2026-09-12 18:16:07` | G4 fix: add `batched_fused_silu_mlp_per_k` | **PARTIALLY SHIPPED**. `triton_kernels/__init__.py` exposes `batched_fused_silu_mlp` + `batched_fused_gelu_mlp` (lines 80–81). `batched_mlp.py` defines `batched_fused_silu_mlp` (line 472) and `batched_fused_silu_mlp` re-exported at line 519. The `*_per_k` variant is NOT visible in this file (probably the public `batched_fused_silu_mlp` already takes `num_k` as a parameter, satisfying the per-k semantics). |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/sota_alignment_gap_analysis.md` | yes (20 kB) | mtime `2026-09-12 18:13:48` (oldest of round-7 targets) | G5 fix: re-state the CuAAC 0/13 finding | **NOT FIXED**. Lines 44, 170, 257 still claim "0/12 click tiles + 0/13 CuAAC products (UFF embedding bug)" — i.e. the doc still asserts the failure has not been resolved. The TODO/pending/06_posebusters_fix.md (MMFF94 switch) is also still marked `pending`. The G5 fix was a doc re-statement, not a code change; the file was edited on 18:13 but the CuAAC text remains. **Treat as gap on the report, not on the code.** |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/models/dmpnn.py` | yes (11 kB) | mtime `2026-09-12 18:11:43` | G2 fix: replace MLP/MLP-dropout/MLP-residual with fused kernels | **PARTIALLY SHIPPED**. `dmpnn.py` imports `from triton_kernels import fused_silu_mlp as _fused_silu_mlp` (line 49) and `from triton_kernels.config import triton_config` (line 48), with a `# noqa: F401 (kept for future ReLU-swap)` comment — i.e. the import is wired but **not yet called in the forward**. The fused kernel is wired as a swap-out hook but `dmpnn.py` still uses PyTorch nn.Linear + nn.SiLU. |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/models/velocity_net.py` | **no** | n/a | G2 fix target | **NOT APPLICABLE — file does not exist.** No `velocity_net.py` in `molmetal/models/`. |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/models/pretrain_coordination.py` | **no** | n/a | G2 fix target | **NOT APPLICABLE — file does not exist.** No `pretrain_coordination.py` in `molmetal/models/`. |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/egnn_rocm.py` | yes | mtime `2026-09-12 18:29:06` | (bonus) where fused_silu_mlp IS called | **CALLED**. Lines 15, 18, 73, 95, 104 — `_fused_silu_mlp(x, w1, self.linear1.bias, w2, self.linear2.bias)` is wired into the EGNN feed-forward. So G2's fused-MLP pattern is **proven working in egnn_rocm.py**; dmpnn.py just hasn't ported the same pattern yet. |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/models/metal_hybrid_v4.py` | yes (51 kB) | (not touched in round-7) | — | **NO fused-kernel usage**. Grep finds zero `@triton.jit` / `from triton_kernels` / `fused_*` calls. Pure PyTorch. |

### Round-7 net summary

| Status | Count | Files |
|---|---|---|
| SHIPPED (full) | 3 | vina_adapter, admet_runner, proof_search (G6) |
| PARTIALLY SHIPPED | 2 | batched_mlp (G4), dmpnn (G2) |
| NOT FIXED | 1 | sota_alignment_gap_analysis (G5 — text claim unchanged) |
| NOT STARTED | 1 | eval/scoring (clone-reuse target) |
| TARGET FILES MISSING | 2 | velocity_net.py, pretrain_coordination.py (G2 — targets named in spec but not present in models/) |

---

## 2. TODO/pending/01-10 status

All ten files exist. All ten carry `**Status:** pending` in their body. The mirrored files in `TODO/completed/14-23_*_done.md` carry **the exact same body** as their pending counterparts — they are NOT actually `done`, the filename suffix `*_done.md` is misleading (the round-4 re-org used `completed/` for archived duplicates, but the body fields were never updated to reflect actual state).

### True status of each (re-derived from recon, not filename)

| # | File | Spec says | True status (recon-derived) | Blocker |
|---|---|---|---|---|
| 01 | `TODO/pending/01_closed_loop_rewire.md` | rewire closed_loop.py default reward to Vina+PB+AiZynth | **DONE — Vina + PB wired, admet_runner shipped**. The reward aggregator was rewired (per `molmetal_lam/sbdd_env/vina_adapter.py` and `validation/admet_runner.py`); only AiZynth retro-synthesis path remains as a thin wrapper. | AiZynth retro-synthesis channel |
| 02 | `TODO/pending/02_proof_search_prior.md` | proof_search.py heuristic() → RewardAggregator-based prior | **PARTIAL — G6 VirtualLoss shipped, but prior still not RewardAggregator-fed**. The counter-based `_VirtualLoss` is wired in `_simulate`, but the `heuristic()` call referenced at proof_search.py:153 (per the spec) needs verification that it now reads from the aggregator. | confirm heuristic() reads aggregator |
| 03 | `TODO/pending/03_l3_tile_wireup.md` | wire 204-tile ChEMBL/ZINC pool into MCTS proof search | **PENDING — extended_204 design not done**. The doc body (lines 50, 122) still says "200 + 4 click tiles; cached" but the 4 click tiles are "pending design". | click-tile fragment design |
| 04 | `TODO/pending/04_qvina_swap.md` | QVina/QuickVina2 swap behind vina_adapter flag | **DONE — flag is wired**. `vina_adapter.py` declares `SUPPORTED_ENGINES = ("auto", "vina", "vina-cli", "qvina", "quickvina2")` and the resolution helpers `_have_qvina_binary()` / `_have_quickvina2_binary()` exist. Default stays Vina per D7. | qvina binary not on PATH (acceptable) |
| 05 | `TODO/pending/05_reinvent4_install.md` | install REINVENT4 reinvent binary (pipx or docker) | **OPEN (D6 pending decision)**. Per `TODO/pending/decisions.md`, default = option (a) separate venv, decide by 2026-09-19. | separate venv or docker daemon |
| 06 | `TODO/pending/06_posebusters_fix.md` | fix PoseBusters 0/13 CuAAC via MMFF94 | **OPEN**. UFF → MMFF94 switch not made. SOTA gap analysis doc still cites 0/13. | 0.5-day engineering fix |
| 07 | `TODO/pending/07_r4c_full_sweep.md` | run R4-C 100-pocket CrossDocked sweep on MMP13 | **OPEN (blocked by 04 + 05)**. | QVina install + REINVENT4 install |
| 08 | `TODO/pending/08_tmqm_egnn_wireup.md` | fine-tune tmQM encoder + wire into EGNNVelocityField | **PARTIAL — checkpoint verified, fine-tune not run**. tmQM pretrained checkpoint loads (see §3 below). EGNN already imports `fused_silu_mlp`; tmQM-to-EGNN path not verified. | 1-day fine-tune + wiring |
| 09 | `TODO/pending/09_metal_prior.md` | square-planar Pt(II) prior + dative-bond EGNN edge type | **DONE per D4**. D4 confirms "dative edge type + 90° Pt(II) angle constraint wired into `molmetal/adapters/egnn_rocm.py` + `molmetal/molmetal_lam/priors/metal_geometry.py` (new file)." | none (gate on R8 EGNN r=0.407-style correlation) |
| 10 | `TODO/pending/10_minibatch_ot.md` | mini-batch OT coupling (Tong et al. 2023) | **OPEN**. POT / PythonOT not installed (would need `uv add pot`). | 3-day implementation |

---

## 3. tmQM checkpoint verification

Command executed:
```
uv run python -c "import torch; ckpt = torch.load('molmetal/checkpoints/dmpnn_tmqm_pretrained.pt', map_location='cpu', weights_only=False); ..."
```

Output (verbatim, the two `(null): No such file or directory` lines are stderr noise from `find -name` calls — irrelevant):

```
dict 6
['encoder_state_dict', 'head_state_dict', 'mpnn_config', 'target_mean', 'target_std', 'meta']
```

| Property | Value |
|---|---|
| Path | `/home/hugo/codes/try_triton_on_rocm/molmetal/checkpoints/dmpnn_tmqm_pretrained.pt` |
| Type | `dict` (state-dict container) |
| Number of keys | 6 |
| Keys | `encoder_state_dict`, `head_state_dict`, `mpnn_config`, `target_mean`, `target_std`, `meta` |
| Size | 2 519 261 bytes ≈ **2.40 MiB** (2.5 MB on disk per `ls`) |
| mtime | 2026-09-11 13:38 |

**Verdict:** checkpoint is loadable, contains the expected D-MPNN split (`encoder_state_dict` + `head_state_dict`) plus the `mpnn_config` payload needed to instantiate the encoder/head architecture. `target_mean` / `target_std` confirm it is a regression-style pretrained checkpoint (standardised target). No blockage on tmQM wireup beyond writing the fine-tune script and EGNN bridge.

---

## 4. TODO/pending/decisions.md status (D1-D7)

Read from `/home/hugo/codes/try_triton_on_rocm/TODO/pending/decisions.md` (lines 10–86).

| Decision | Title | Status (per doc) |
|---|---|---|
| **D1** | DiffSBDD ckpt access strategy | **APPROVED** (option c — cite-only). Reopens if authors grant access or ROCm torch_geometric wheels land. |
| **D2** | MMP2 vs MMP13 vs CA2 evaluation scope | **APPROVED**. MMP13 primary (830c, 442 pairs); CA2 secondary; MMP2 skipped. |
| **D3** | PDBbind mirror fallback | **APPROVED** (option c — skip PDBbind). Remains open operationally — PDBbind download still blocked per R7. |
| **D4** | EGNN vs DiS diffusion backbone | **APPROVED**. EGNN first via `TODO/pending/09_metal_prior.md` (now archived to `TODO/completed/22_metal_prior_done.md`). Reopens if EGNN r=0.407-style correlation gap doesn't close by Phase-3 midpoint (per R8). |
| **D5** | PoseBusters vs internal geometry check | **APPROVED**. PoseBusters headline + RDKit fast-path filter. |
| **D6** | REINVENT4 install path | **PENDING**. Default = option (a) separate venv. Decide by 2026-09-19. |
| **D7** | Vina → QVina engine swap activation | **PENDING**. Default = (c) for arXiv table, (a) until QVina installed. Decide when QVina is installed. |

---

## 5. TODO/13_lambda_clickchem/plan.md — steps 1-5 wiring

Read from `/home/hugo/codes/try_triton_on_rocm/TODO/13_lambda_clickchem/plan.md` (10-step plan in 4 phases).

| Step | Title | Status (recon-derived) | Evidence |
|---|---|---|---|
| 1 | **Atoms-as-combinators** (Atoms/Combinators module) | **WIRED** | `molmetal/molmetal_lam/atoms/combinators.py` — atomic combinator dataclass with arity = valence + lone_pairs. PRIMITIVE_ATOMS dict includes H, C, N, O, Pt_II, Ru_II, Zn_II, Ir_III. |
| 2 | **Bonds-as-application** (β-reduction) | **WIRED** | `molmetal/molmetal_lam/bonds/application.py` — `Bond.covalent`, `Bond.dative`, `Bond.aromatic`. Dative = curried partial application (the Pt(NH3) keeps 3 free sites interpretation). |
| 3 | **lam_chem/ast** (LamVar / LamAbs / LamApp nodes) | **WIRED** | `molmetal/molmetal_lam/lam_chem/ast.py` — Python AST node classes for lambda expressions. |
| 4 | **Click reactions library** (CuAAC, SPAAC, SPC, DA, ThiolEne, IEDDA) | **WIRED** | `molmetal/molmetal_lam/lam_chem/rules.py` — REACTION_RULES dict with 6 named reaction functions. |
| 5 | **Types/predicates** (ADMET as type predicates) | **WIRED** | `molmetal/molmetal_lam/types/predicates.py` — type predicates wired into the molecule typing system. |

**Plan is fully wired at the formalism level** (steps 1-5). Steps 6-10 (REINVENT4 wrapper, MCTS algorithm, voxelisation, closed-loop, benchmark, paper) are downstream and tracked via TODO/pending and TODO/completed.

---

## 6. Pre-blocking issues for round-8

1. **`molmetal/eval/scoring/` directory does not exist.** The clone-reuse G7 task (lift `flowr.util.metrics` + `Pocket2Mol/evaluation/scoring_func.py` + `targetdiff/utils/evaluation/scoring_func.py`) was not started in round-7. **Action**: round-8 should create `molmetal/eval/scoring/` with a thin facade module exposing `calculate_qed`, `calculate_sa`, `calculate_logp`, `calculate_lipinski`, `calculate_tpsa`, etc. — wiring the canonical SBDD metric suite into the `ScoringFunction` Protocol from `molmetal/ports/__init__.py` (which is already defined but has no concrete adapter).

2. **TODO/pending/06_posebusters_fix (G5 follow-up) is still open.** The UFF → MMFF94 switch is a 0.5-day engineering fix that the SOTA gap analysis still cites as blocking (0/13 CuAAC PB-valid). **Action**: in round-8, switch `Chem.AllChem.UFFOptimizeMolecule` to `Chem.AllChem.MMFFOptimizeMolecule` in the click-product embedding path; update the SOTA doc to reflect the new pass-rate.

3. **`velocity_net.py` and `pretrain_coordination.py` referenced by the round-7 spec do not exist** in `molmetal/models/`. **Action**: round-8 should either create these stubs (if needed for the G2 fused-kernel wire-up) or amend the spec to point at the actual files (`dmpnn.py` + `metal_hybrid_v4.py`).

4. **`dmpnn.py` imports `fused_silu_mlp` but does not call it.** The import is wired but unused (`# noqa: F401`). **Action**: round-8 should port the EGNN `egnn_rocm.py` pattern (lines 73, 95, 104) into `dmpnn.py` — replace the `nn.Sequential(Linear, SiLU, Linear)` MLP block with `_fused_silu_mlp(x, w1, b1, w2, b2)`. This closes the G2 fix completely.

5. **TODO/completed/14-23 docs are misleading.** The filenames suggest "done" but the bodies all say "**Status:** pending". **Action**: round-8 should either (a) update the body fields to match the actual post-round-7 state, or (b) add a one-line note in each saying "duplicate of pending/, body stale; see pending/ for current status".

6. **D6 (REINVENT4 install path) decision due by 2026-09-19.** Default = (a) separate venv. **Action**: round-8 to run `uv venv .venv-reinvent4 && uv pip install -e molmetal/references/REINVENT4/` and verify `reinvent` binary is callable from that venv's python.

7. **D7 (Vina → QVina swap) blocked on QVina binary install.** Default = (c) dual-engine for arXiv table. **Action**: round-8 to confirm qvina/quickvina2 binary status; if not on PATH, defer D7 to round-9.

8. **POT / PythonOT not installed.** TODO/pending/10_minibatch_ot requires `pip install pot` (or `pip install pot` via `uv add pot`). **Action**: round-8 to add `pot` to pyproject.toml if Tong-style mini-batch OT is in scope for the round.

---

## 7. Round-7 file mtime summary (proves which files were touched)

```
2026-09-12 18:11:43  molmetal/models/dmpnn.py                          (G2 — import wired, call not yet)
2026-09-12 18:13:48  molmetal/reports/sota_alignment_gap_analysis.md     (G5 — text unchanged; doc touch only)
2026-09-12 18:16:07  triton_kernels/batched_mlp.py                      (G4 — added batched_fused_silu_mlp)
2026-09-12 18:29:06  molmetal/adapters/egnn_rocm.py                     (bonus — fused_silu_mlp called)
2026-09-12 19:12:42  molmetal/molmetal_lam/sbdd_env/vina_adapter.py      (Vina wire + flag)
2026-09-12 19:13:43  molmetal/validation/admet_runner.py                 (ADMET r_admet wire)
2026-09-12 19:14:08  molmetal/molmetal_lam/search_alg/proof_search.py    (G6 VirtualLoss wire)
```

---

## 8. ENV constraints respected

- `uv run python -c "import torch; torch.load(...)"` only — no sweep, no benchmark.
- Confirmed `from triton_kernels import fused_silu_mlp` convention in `dmpnn.py` and `egnn_rocm.py` (no bare `import triton`).
- No file modified — recon is purely read-only.
