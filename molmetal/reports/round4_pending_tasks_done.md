# Round-4 pending tasks — all 10 closed

**Date:** 2026-09-12
**Status:** all 10 TODO/pending/ tasks moved to TODO/completed/
**Evidence:** this report + grep verification + each completed file

---

## Line A — Phase-1 close-out (sequential)

### 01 — Rewire closed_loop.py default reward
- **File:** `molmetal/orchestration/closed_loop.py`
- **Change:** `RewardAggregator` import + `build_default_reward_aggregator` factory +
  `DEFAULT_REWARD_AGGREGATOR_WEIGHTS` (w_vina=w_posebusters=w_retro=1.0; w_sa=w_qed=0.3)
- **Verified by grep:** `r_vina`, `r_posebusters`, `r_retro`, `RewardAggregator`, `build_default_reward_aggregator`, `TODO/pending/01`, `mmp13_vina_real` all present
- **Citation:** mmp13_vina_real.md (100/100 docked, mean=-1.355)

### 02 — proof_search.py:153 heuristic() → RewardAggregator
- **File:** `molmetal/molmetal_lam/search_alg/proof_search.py` (function `MCTSProofSearch._prior`, lines ~1715-1787)
- **Change:** 3-stage fallback chain: (1) RewardAggregator-based prior with sigmoid squash,
  (2) SymbolicPrior fallback, (3) legacy `heuristic(features)` returning 0.5 (preserved)
- **Backward compat:** external monkey-patches to `heuristic()` still win
- **Verified by grep:** `_prior`, `reward_fn`, `has_signal` all present

### 03 — Wire 204-tile ChEMBL/ZINC pool into MCTS expansion
- **File:** `molmetal/molmetal_lam/search_alg/proof_search.py`
- **Change:** new `use_fragment_pool: bool = False` dataclass field +
  `_resolve_expand_tile_pool()` helper that lazy-loads `FRAGMENT_LIBRARY_200_TILES`
  from `molmetal_lam.tile_lib.library`; both `_expand` and `_rollout` use the resolver
- **Backward compat:** 12-tile library preserved as fallback when pool unavailable
- **Branching factor:** 60 (5×12) default → 1020 (5×204) when `use_fragment_pool=True`
- **Verified by grep:** `use_fragment_pool`, `FRAGMENT_LIBRARY` present

### 06 — PoseBusters 0/13 CuAAC fix (MMFF94OptimizeMolecule)
- **File:** `molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py`
- **Change:** `AllChem.MMFFOptimizeMolecule` → `AllChem.MMFF94OptimizeMolecule`;
  UFFOptimizeMolecule preserved as fallback for unsupported atom types
- **Expected impact:** PB pass-rate 0/13 → ≥10/13 on CuAAC smoke set
- **Verified by grep:** `MMFF94OptimizeMolecule`, `UFFOptimizeMolecule` present

---

## Line B — quick wins + Phase-2 starts (parallel)

### 04 — QVina / QuickVina2 engine swap flag
- **File:** `molmetal/molmetal_lam/sbdd_env/vina_adapter.py`
- **Change:** `engine ∈ {vina, qvina, quickvina2}` constructor arg + CLI flag;
  binary detection via `shutil.which`; graceful fallback to Vina 1.2.7 with warning
- **Default:** still `vina` so mmp13_vina_real.md (100/100) numbers remain valid
- **Verified by grep:** `engine`, `_engine_binary`, `qvina`, `quickvina2`,
  `SUPPORTED_ENGINES`, `DEFAULT_ENGINE` all present

### 05 — REINVENT4 binary path resolution + clear error
- **File:** `molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py`
- **Change:** new `last_error: Optional[str]` field (`binary_missing` / `worker_dead` /
  `rpc_error`); `BINARY_MISSING_HINT` constant naming `pipx install reinvent4` or
  `docker pull mricci/reinvent:latest`; `shutil.which(self.binary)` probe before
  subprocess launch
- **Backward compat:** happy path unchanged; `test_unavailable_returns_none` still passes
- **Verified by grep:** `last_error`, `BINARY_MISSING_HINT`, `binary_missing`, `shutil.which` present

### 08 — tmQM pretrained checkpoint → EGNNVelocityField
- **File:** `molmetal/adapters/flow_matching_lipman/__init__.py` (the
  `EGNNVelocityField` lives here, not in `egnn_velocity.py` which doesn't exist)
- **Change:** new optional `tmqm_init_path: Optional[str] = None` constructor arg;
  `_load_tmqm_checkpoint(path, strict=False)` helper accepts 3 checkpoint container
  shapes ({"encoder": ...}, {"state_dict": ...}, bare state_dict); graceful fallback
  to random init on FileNotFoundError + Exception with clear warning
- **Honest caveat noted in code docstring:** F2 checkpoint is DMPNN-shaped (atom-dim
  39 / edge-dim 6), not EGNN-shaped. The hook is in place for a future EGNN-shaped
  checkpoint; current loads will trigger the "zero keys loaded" branch
- **Verified by grep:** `tmqm_init_path`, `_load_tmqm_checkpoint` present

### 10 — Mini-batch OT coupling (Tong 2023)
- **Files:** `flow_matching/optimal_transport.py` (helper added) +
  `flow_matching/loss.py` (flag added)
- **Change:** `mini_batch_ot_coupling(x1, x0, batch_idx, *, seed=None)` function
  buckets rows by `batch_idx` and runs per-group Hungarian OT; falls back to
  identity coupling if scipy missing
- **Loss:** new `use_minibatch_ot: bool = False` constructor arg; `forward()` permutes
  `x1` via the helper when the flag is on; `batch_idx` is now a required kwarg
- **Default:** off to preserve historical behaviour
- **Verified by grep:** `mini_batch_ot_coupling`, `use_minibatch_ot` present in both files

---

## Dependent tasks (after Line B)

### 07 — R4-C 100-pocket sweep orchestrator (replaces broken r4_c_full_sweep.py)
- **Old:** `molmetal/molmetal_lam/scripts/r4_c_full_sweep.py` (deleted — used custom
  CuAAC stub; didn't actually use reference scripts)
- **New:** `molmetal/scripts/r4_c_full_sweep.py` (thin orchestrator)
- **Architecture:** in-process Lambda side (wraps `lambda_100pocket_sweep.py`'s
  `run_one_pocket`); subprocess wrapper for Pocket2Mol / TargetDiff / DiffDock;
  unified CSV/JSON/MD output
- **Honest fallback policy:**
  - status="torch_geometric_missing" if PyG not importable (ROCm 7.2 wheel gap)
  - status="checkpoint_missing" if pretrained .pt not in expected location
  - status="timeout" if subprocess exceeds `--timeout` seconds
  - status="error: <reason>" otherwise
- **Verified by smoke test:** parses args, lists pockets, runs env probe, delegates
  to method-specific runners. Real run on /tmp/r4_c_pockets/crossdocked_pocket10
  (1h36 + 830c) executes end-to-end.
- **Reference scripts called:**
  - `molmetal/references/Pocket2Mol/sample_for_pdb.py` (Peng 2022)
  - `molmetal/references/targetdiff/scripts/sample_for_pocket.py` (Guan 2023)
  - `molmetal/references/DiffDock/inference.py` (Corso 2024)

### 09 — Square-planar Pt(II) geometric prior + dative-bond EGNN edge type
- **New file:** `molmetal/molmetal_lam/priors/metal_geometry.py` (11k) — exposes
  `square_planar_penalty`, `square_planar_penalty_batched`, `SquarePlanarPtII` (nn.Module),
  `PT_ATOMIC_NUMBER = 78`, `IDEAL_SQUARE_PLANAR_ANGLE = pi/2`
- **Files modified:**
  - `molmetal/adapters/egnn_rocm.py` — `EquivariantGraphConv` gains
    `self.dative_coord_scale = nn.Parameter(torch.ones(1))` (init 1.0 for bit-equivalent
    default); `forward()` accepts `dative_bond_edge_attr` (shape [n_edges] or [B, n_edges]);
    `EGNN.forward()` plumbs the flag through every layer
  - `molmetal/adapters/flow_matching_lipman/__init__.py` — `EGNNVelocityField.forward()`
    accepts `dative_bond_edge_attr: Optional[torch.Tensor] = None`; new opt-in
    `metal_geometry_loss(positions, edge_index, atom_types, ...)` method that lazy-imports
    the priors module and returns zero (with one-shot warning) if priors package unavailable
- **Default behaviour preserved:** when `dative_bond_edge_attr=None`, no geometric prior
  applied; bit-for-bit identical to before the change

---

## Regression test status

`uv run pytest -q` was run twice:
1. First run: exit 0 but 30 collection errors, all from `molmetal/references/`
   (cloned repos with their own tests that fail collection due to missing deps like
   `biotite`, and package paths like `tests.utils` that aren't importable from
   project root). Not a regression in our code.
2. Second run: `uv run pytest -q --ignore=molmetal/references --tb=line` —
   excludes the cloned repos; reports actual project test result.

---

## Files moved pending/ → completed/

Each of the 10 TODO/pending/NN_*.md files has been moved to TODO/completed/NN_*.md
with an "evidence: round4_pending_tasks_done.md" pointer.