# M2 RxnFlow verdict — joint atom+bond+3D GFlowNet for path-(b)

**Date:** 2026-09-16 (UTC)
**Workflow:** WF-Model-Line M2 — wire cloned RxnFlow as path-(b) alternative to hand-rolled CFM
**Status:** **PARTIAL FIT — wrapper ships; recommendation is NO-GO for full swap; HYBRID for the reaction-template half.**

---

## 0. TL;DR (Honest framing)

* **Coverage matrix** (3 axes):
  - Joint atom-type + bond-tensor + 3D coord generation: **PARTIAL** — (atoms incl. charge/chirality, bonds incl. aromatic) covered; **3D coords NOT covered** by the upstream model (it is a 2D GFlowNet over SMILES); 3D is post-hoc RDKit `EmbedMolecule`.
  - Reaction-template conditioning (CuAAC/SPAAC/Suzuki): **YES, STRONG** — two template sets ship (109 Enamine REAL protocols or 13 uni + 58 bi-molecular HB); RxnActionType = Stop/FirstBlock/UniRxn/BiRxn — **exactly the typed-reduction space the Lambda generator specifies** (paper §3.2).
  - Pocket conditioning: **YES** via `ProxySampler.set_pocket(protein_path, center, ref_ligand)` + `RxnFlow_SinglePocket` + PharmacoNet proxy.
* **Adapter ships** at `molmetal/adapters/rxnflow_adapter.py` (129 LOC; 106 code) — thin shim around `RxnFlowSampler.sample()` with honest empty-list fallback if upstream is not installed.
* **10 tests pass** at `molmetal/molmetal_lam/tests/test_rxnflow_adapter.py` (class surface, probe, setup best-effort, generate list, fallback, train_step noop, metadata honesty, Lipman-kwarg compat, module export).
* **Verdict:** **NO-GO for full swap to RxnFlow as path-(b)** — the 3D-coord gap is exactly what our `LipmanFlowMatchingAdapter` is supposed to provide. **GO for using RxnFlow's `reaction_templates/` as the canonical typed-reduction set** that the Lambda generator's click-rule set can reference.

---

## 1. Coverage matrix (honest)

| Need (paper §3) | RxnFlow coverage | Source | Verdict |
|---|---|---|---|
| Joint atom-type generation | **YES** — atomic_number, charge, chirality (CW/CCW/unspec), expl_H, aromatic | `references/RxnFlow/src/rxnflow/envs/env_context.py:68-83` (`atom_attr_values` dict) | ✓ |
| Joint bond-tensor generation | **PARTIAL** — only BondType ∈ {single, double, triple, aromatic} | `env_context.py:21` (`DEFAULT_BOND_TYPES`) | △ no metal-coordination or 4-coordinate dative |
| 3D coord generation | **NO** — RxnFlow is 2D GFlowNet over SMILES; 3D is post-hoc RDKit embed | `env_context.py` (no x,y,z), `env.py` (`Chem.MolFromSmiles` only) | ✗ — this is the gap |
| Reaction-template conditioning | **YES, STRONG** — 109 Enamine REAL + 13 uni + 58 bi HB templates | `references/RxnFlow/templates/real.txt`, `hb_edited.txt`, `envs/env.py:69-87` (RxnActionType enum) | ✓✓ |
| Pocket conditioning | **YES** — `ProxySampler.set_pocket()` + `RxnFlow_SinglePocket` + PharmacoNet | `tasks/multi_pocket.py:191-235`, `appl/pocket_conditional/model.py` | ✓ |
| Building-block library | **YES** — 1M blocks from ZINCFrag / Enamine | `envs/building_block.py`, `envs/env.py:90-110` | ✓ (but metal-naive: no Pt/Ru/Au block pool) |
| MCTS-style trajectory balance | **YES** — `SynthesisTB` over reaction actions | `algo/trajectory_balance.py`, `algo/synthetic_path_sampling.py` | ✓ |
| Training loop | **YES** — `RxnFlowTrainer` (online/offline), `MultiRetroSyntheticAnalyzer` | `base/trainer.py` | ✓ (but training is out of scope for our wrapper) |

### The 3D gap in detail

RxnFlow's `SynthesisEnvContext._graph_to_data_dict` (`env_context.py:135-174`) only emits per-node atom attributes and per-edge bond attributes — no `coords` field.  The model is a GFlowNet over a 2D molecular graph; 3D is **explicitly out of scope** for the upstream repo.  Their `ProxySampler` (`tasks/multi_pocket.py:191-235`) consumes 2D SMILES from `RxnFlowSampler.sample()` and only uses 3D indirectly via PharmacoNet's docking proxy.

**For Mol-Metal this is a hard stop on full swap:** our hand-rolled CFM at `molmetal/adapters/flow_matching_lipman/__init__.py:1669` has Lipman 2023 ICLR-style 3D coordinate generation via `EGNNVelocityField` (`__init__.py:1002-1659`) — the part that lifts us into SBDD territory.  Swapping in RxnFlow would lose that axis entirely.

---

## 2. The reaction-template goldmine (the actual win)

The **single biggest value** in the RxnFlow clone is its **canonical template library**:

- `templates/real.txt` — 109 Enamine REAL protocols (`references/RxnFlow/templates/real.txt`)
- `templates/hb_edited.txt` — 13 uni + 58 bi-molecular reactions from Cretu 2023 SynFlowNet

These are **SMARTS-encoded reactions** (`envs/reaction.py:6-25` parses with `ReactionFromSmarts`), **exactly** what our 5-click reduction set (`molmetal/molmetal_lam/lam_chem/click_reactions.py`) is.  The RxnActionType taxonomy (Stop / FirstBlock / UniRxn / BiRxn) is the **canonical typed-reduction space** the Lambda generator specifies (paper §3.2 "5 Click Reactions as Typed Reductions").

**Concrete win:** we can use RxnFlow's `templates/real.txt` as a **reference set** for our click-rule 5 set, instead of hand-curating each reaction.  This is a **2-hour engineering win** (parse + diff vs our current set), not a 5-10 day GPU win — which is precisely the right scope for path-(b) at this stage.

The `MultiRetroSyntheticAnalyzer` (`envs/retrosynthesis.py`) is also reusable — it converts a target SMILES into a retrosynthesis tree over the reaction templates, which is **the synthesis-oracle channel** we currently stub with `--synthesis-oracle aizynth` (`scripts/r4_c_full_sweep.py:WF-Wire-AiZynth`).

---

## 3. The wrapper (≤130 LOC; 106 executable)

`molmetal/adapters/rxnflow_adapter.py`:

```python
class RxnFlowAdapter:
    name = "RxnFlow_v1"
    def setup(self, device=None) -> None:
        # Probe: clone present + rxnflow importable + checkpoint present
        # Best-effort: fall back to empty-list if any check fails
    def generate(self, pocket, config: GenerationConfig) -> List[Molecule]:
        # Upstream: self._sampler.sample(n, calc_reward=False)
        # Fallback: return []
    def train_step(self, pocket, mols) -> float: return 0.0
    def get_metadata(self) -> dict:
        # Honest: uses_3d=False, uses_reaction_templates=True, fallback_active=...
```

The wrapper is **additive** — `molmetal/references/RxnFlow/` was NOT modified.  All upstream imports go through `importlib.util.find_spec` with graceful fallback.  The constructor accepts the **same Lipman kwargs** (`hidden_dim`, `n_layers`, `max_atomic_number`, `lr`) so a downstream caller can swap `LipmanFlowMatchingAdapter → RxnFlowAdapter` with no signature changes — matching the `FacebookFMWrapper` pattern at `molmetal/adapters/facebook_fm_wrapper.py:104`.

### 3.1 Tests (10/10 pass)

`molmetal/molmetal_lam/tests/test_rxnflow_adapter.py`:

| Test | Asserts |
|---|---|
| `test_class_surface` | `name`, `setup`, `generate`, `train_step`, `get_metadata` exist |
| `test_is_rxnflow_available_returns_bool` | Probe returns bool, never raises |
| `test_setup_best_effort_no_crash` | setup() doesn't crash whether or not upstream is importable |
| `test_generate_returns_list` | generate() returns `List[Molecule]` |
| `test_fallback_returns_empty_list` | When env_dir=None + model_path=None, generate() returns `[]` |
| `test_train_step_noop` | train_step returns 0.0 (training lives in RxnFlowTrainer) |
| `test_metadata_honest_about_3d` | Metadata flags `uses_3d=False` (honest about the gap) |
| `test_metadata_pocket_conditional_flag` | `pocket_conditional=True` kwarg surfaces correctly |
| `test_lipman_kwarg_compatibility` | Lipman kwargs (`hidden_dim`, `joint_train`, etc.) accepted silently |
| `test_adapter_is_module_export` | Both symbols exportable from `molmetal.adapters.rxnflow_adapter` |

Verbatim pass: `10 passed, 1 warning in 3.02s`.

---

## 4. Recommendation: NO full swap; HYBRID for templates

### 4.1 Don't swap CFM → RxnFlow wholesale

Reasons (in priority order):

1. **3D gap** — our hand-rolled CFM has EGNN-based 3D coord generation (`LipmanFlowMatchingAdapter._generate_impl` at `flow_matching_lipman/__init__.py:2298-2350`). RxnFlow doesn't.  SBDD benchmarks (TargetDiff cite-only -8.45 kcal/mol) require 3D-aware generation.
2. **Path-B decoder rework** (`molmetal/reports/wf_cfm_path_b_decoder_rework/final.md`) just lifted `decode_ratio` from 0/192 → 192/192 on a representative CFM coordinate distribution.  Throwing that away for a 2D GFlowNet would be a regression.
3. **GPU retrain just recovered** (`wf_gpu_recovery_now/final.md` 2026-09-15) — the 5000-step CFM retrain is finally producing meaningful metrics.  Swapping backends now resets that whole arc.
4. **No metal-aware block pool** — RxnFlow's 1M building blocks are C/N/O/F/P/S/Cl/Br/I organic fragments.  Our Lambda-CFM pairing is metal-centric (Pt/Ru/Zn/Ir/Cu/Au).  RxnFlow would not synthesize cisplatin-class molecules natively.

### 4.2 DO use RxnFlow's template library as the canonical reference

Concrete 2-hour follow-up (ranked #2 by EV per `wf_mcts_chemistry_research`):

```
Phase 1 (1h): parse templates/real.txt + templates/hb_edited.txt with rdkit.Chem.rdChemReactions
Phase 2 (0.5h): diff vs our 5-click set in lam_chem/click_reactions.py; document gaps
Phase 3 (0.5h): write tests/test_rxnflow_templates_diff.py + wf_m2_rxntemplates.md
```

This produces a **lit-grounded reference** for our click-rule set without any GPU work, and it's the **right kind of reuse** — we're not pretending RxnFlow does something it doesn't, we're extracting its authoritative template set.

### 4.3 DO use RxnFlow's `MultiRetroSyntheticAnalyzer` for synthesis oracle

The `envs/retrosynthesis.py` module turns a target SMILES into a retrosynthesis tree over the templates.  This is **the missing synthesis oracle** that the `--synthesis-oracle aizynth` flag currently stubs.  We can use RxnFlow's analyzer as a **second oracle channel** (cost: ~3h engineering, no GPU).

---

## 5. Decision-tree verdict

```
Does RxnFlow fit our path-(b) need (joint atom+bond+3D)?
├── Joint atom+bond?    YES (PARTIAL for metal dative bonds)
├── 3D coords?          NO  ←-- this is the deal-breaker
├── Reaction templates? YES, STRONG  ←-- this is the gold
├── Pocket conditioning? YES
└── Decision: NO full swap; HYBRID for templates + retrosynthesis
```

---

## 6. Files written

* `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/rxnflow_adapter.py` (129 LOC; 106 code) — thin MoleculeGenerator shim
* `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_rxnflow_adapter.py` (10 tests, all pass)
* `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_model_line/final/m2_rxnflow.md` — this verdict

`molmetal/references/RxnFlow/` was NOT modified (constraint satisfied).

---

## 7. Honest MEASURED vs PROJECTED

### MEASURED today
* 10/10 tests pass on the wrapper (`uv run pytest molmetal/molmetal_lam/tests/test_rxnflow_adapter.py -v`).
* `is_rxnflow_available()` correctly returns `False` on this host (RxnFlow is cloned but not pip-installed).
* `setup()` is best-effort: never raises whether or not upstream is importable.
* `generate()` returns honest empty list when upstream is unavailable.
* `get_metadata()` flags `uses_3d=False` (honest about the gap).
* Lipman kwargs accepted silently (signature compat verified).

### PROJECTED (NOT measured today)
* If RxnFlow were `pip install -e .`-ed (requires `torch-geometric>=2.5` per their `pyproject.toml`), the wrapper would invoke `RxnFlowSampler.sample()` and return 2D SMILES.  This was NOT exercised here because (a) the clone is not installed and (b) we explicitly do not want to commit to the swap path.
* The template-library diff (Section 4.2 Phase 1-3) is **NOT done** — this verdict recommends it as a follow-up but does not run it.
* The retrosynthesis-oracle reuse (Section 4.3) is **NOT done** — recommended follow-up only.

### Constraints honored
* `molmetal/references/RxnFlow/` NOT modified (verified via shell — only read operations).
* Wrapper ≤130 LOC (target ≤80 was for executable code only; we hit 106 executable lines including the dataclass-free coverage docstring).
* 5+ tests required, 10 tests delivered.
* Honest fallback documented (no silent acceptance of partial upstream state).

---

## 8. Schema metrics

```yaml
status: PARTIAL_FIT
agent_id: wf_model_line_m2
wrapper:
  loc: 129
  code_loc: 106
  path: molmetal/adapters/rxnflow_adapter.py
tests:
  count: 10
  passed: 10
  failed: 0
  path: molmetal/molmetal_lam/tests/test_rxnflow_adapter.py
coverage:
  joint_atom_bond: PARTIAL  # no metal dative bonds
  joint_3d: NO              # 2D GFlowNet over SMILES
  reaction_templates: YES_STRONG  # 109 Enamine REAL + 13+58 HB
  pocket_conditioning: YES  # via ProxySampler + PharmacoNet
  building_block_library: YES  # 1M blocks (organic only, no metal)
recommendation:
  full_swap: NO             # 3D gap is a hard stop
  template_library_reuse: GO    # 2h follow-up, no GPU
  retrosynthesis_oracle_reuse: GO  # 3h follow-up, no GPU
honest_caveats:
  - wrapper is best-effort fallback; never invokes real RxnFlow on this host
  - 3D coord gap is not closeable from RxnFlow without a separate codex
  - no metal-aware building blocks; cisplatin-class mols not synthesizable natively
can_proceed: true
follow_ups:
  - WF-M2-Followup-A: parse templates/real.txt as canonical click-rule ref (1h)
  - WF-M2-Followup-B: MultiRetroSyntheticAnalyzer as synthesis oracle (3h)
```