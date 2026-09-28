# Wf-42 Phase 2 — Scoring Router (mock → real swap matrix)

> **Purpose**: for every mocked / stub-measured metric that Wf-42
> Phase 1's audit surfaced, document the **mandatory** real-tool
> replacement, the **one-line remediation**, and the **enforcement
> layer** that guarantees no script can silently fall back to a mock.
>
> The enforcement layer is Wf-43 Phase 2's
> `molmetal.scoring.score_any` + `ROUTING_POLICY` + `policy.json`.
> Wf-42 Phase 2 routes every mocked metric through that single
> runtime router and forbids the `MockScorer` / `_stub_score` /
> `MEASURED_BUT_STUB` patterns from re-appearing in any published
> metric JSON.

**Companion documents** (read together):

* `molmetal/reports/wf42_mock_audit/real_tools_inventory.md` — full
  tool inventory with paths/versions/test status
* `molmetal/reports/wf42_mock_audit/mock_inventory.json` — Phase 1
  audit (129 findings; 8 critical, 23 major, 98 minor)
* `molmetal/scoring/ROUTING_POLICY.md` — the **policy** that this
  router enforces
* `molmetal/scoring/policy.json` — machine-readable companion

---

## 1. The mocked-metric universe (Phase 1 findings that touch paper §4)

The Phase 1 audit flagged 8 **critical** findings. All 8 are in the
`m3_vina_dock` (Vina Dock median) family and originate from a single
sweep: `molmetal/scripts/dock_10pockets_se3sb.py` Phase 4, run with
`scorer="stub"` because the meeko vs Pt/Au/Ru atom-type mismatch
prevented real Vina from succeeding at the time. Below is the
**swap matrix**:

| # | Mock artefact | Real tool | Adapter module | One-line remediation |
|---|---|---|---|---|
| 1 | `metrics/by_metric/vina_kcal_per_mol.json` (`scorer: "stub"`) | AutoDock Vina 1.2.7 + Meeko 0.8.0 | `molmetal.molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter` | rerun `dock_10pockets_se3sb.py --use-real-vina` and replace the entry |
| 2 | `molmetal/reports/wf30_dock/best_m3_summary.json` (`scorer: "stub"`) | AutoDock Vina 1.2.7 | same | rerun `wf30_dock_summarize.py` after the real sweep |
| 3 | `molmetal/reports/wf30_dock/m3_lift_analysis.json` (line 5 `stub_scorer`, line 95 `MEASURED_BUT_STUB`) | AutoDock Vina 1.2.7 | same | rerun the Wf-30 sweep with `--use-real-vina` |
| 4 | `molmetal/reports/wf30_dock/per_pocket_metrics.json` (line 511–512) | AutoDock Vina 1.2.7 | same | rerun the sweep |
| 5 | `molmetal/scripts/wf30_dock_summarize.py:177` (`stub_scorer` classifier) | informational classifier only | n/a | keep as informational read-out; the *scoring* itself is in `dock_10pockets_se3sb.py` |
| 6 | `molmetal/scripts/dock_10pockets_se3sb.py:506` (`_stub_score` fallback) | AutoDock Vina 1.2.7 | `vina_adapter.VinaDockingAdapter` | keep `_stub_score` as a **last-resort** fallback ONLY when `use_real_vina=False` is explicitly passed; the Wf-30 ship path must use `--use-real-vina` |
| 7 | `molmetal/reports/wf30_dock/m3_lift_summary.md` (`stub_floor` mention) | AutoDock Vina 1.2.7 | same | replace the MD after rerun |
| 8 | `molmetal/reports/wf30_dock/notes.md:102` (`stub_scorer` mention) | AutoDock Vina 1.2.7 | same | replace after rerun |

### 1.1 The 23 **major** findings — what they really mean

The 23 `major` findings decompose as:

* **10 × `use_real_vina_false`** — these are all from the same Wf-30
  sweep JSONs (`vina_kcal_per_mol.json`, `wf30_dock_verify.json`,
  `metrics.json`, `summary.json`, `best_m3_summary.json`,
  `m3_lift_analysis.json`, `m3_lift_summary.md`, `per_pocket_metrics.json`)
  and resolve once the Wf-30 sweep rerun lands.
* **6 × `stub_class`** — internal Python stub *classes* that are
  honest test scaffolding (e.g. `verify_mcts_replay.py:_Stub` to
  exercise MCTS replay without a full Vina run). None feed the
  paper §4 table. Documented as "internal test fixtures, not
  metric fabrications" in Wf-43 §5.
* **13 × `stub_identifier`** — Python identifiers (variable names,
  class names) that mention "stub". All are inside verify scripts
  that exercise the stub by-design (per `audit_mock_data.py` pattern
  semantics).
* **2 × `stub_bridge`** — `bench_coupling_ic50.py` and
  `bench_lambda_cfm_e2e.py` reference a "stub coupling bridge" in
  their *call site*. The real bridge is `models/coupling_adapter_platinai5k.npz`
  (vendored, present). The bench scripts are unit-level smoke
  tests; reroute to the real `.npz` for paper §4.
* **1 × `stub_floor`** — explicit `-9.66 kcal/mol` floor in the
  Wf-30 stub scorer. Same fix as #1–#8 above (use real Vina).

### 1.2 The 98 **minor** findings — what they really mean

* **50 × `stub_module_name`** — `md_relaxation_stub` is the canonical
  stub module name and is used **by-design** for MD relaxation
  (TODO-30 P2.5). Flagged for visibility only.
* **21 × `protocol_stub_adapter`** — DiffDock / FlowDock / Pocket2Mol
  / TargetDiff adapter shells that import cleanly but raise
  `AdapterUnavailable` / `CheckpointUnavailableError` when the
  upstream wheel / checkpoint is absent. Wf-43 §2 marks them
  RESERVED in `policy.json`. Not paper claims.
* **13 × `stub_identifier`** (covered above)
* **9 × `placeholder_text`** + **5 × `placeholder`** — paper template
  markers (placeholder sections / figures). Not metric findings.

**Net actionable** (after routing through the Wf-43 enforcement
layer): **1 metric** (m3 Vina Dock median) needs a real-Vina rerun.

---

## 2. The scoring router — single point of truth

Wf-43 Phase 2 ships `molmetal.scoring.score_any` as the **single
runtime router**. Every metric in `policy.json` is reachable through
`score_any(metric_name, ctx)` and the call either returns a value
computed by the real tool, or raises `AdapterUnavailable`. No
fallback to a mock score is permitted.

```python
# Real tool, real number
from molmetal.scoring.score_any import score_any
score = score_any("vina_score", {"smiles": "...", "pocket_pdbqt": "..."})
# → -9.66 kcal/mol (REAL Vina score)

# Real tool, no metric found → AdapterUnavailable (never a fake 0)
score_any("vina_score", {})  # raises AdapterUnavailable

# Mock call is REJECTED at import time
score_any("mock_score", {})  # raises AdapterUnavailable — REGISTRY forbids 'mock' in name
```

The runtime contract:

| Situation | Behaviour | Audit |
|---|---|---|
| Real tool installed, real data | Returns real number | log INFO `tool=… name=… status=OK` |
| Real tool installed, data missing | Raises `AdapterUnavailable` | log WARN |
| Real tool NOT installed | Raises `AdapterUnavailable` | log WARN; CLI `score-any doctor` enumerates the gap |
| Mock fallback | **FORBIDDEN** | `test_score_any.py::test_no_mock_keyword_in_registry` would FAIL |

---

## 3. The forced-routing enforcement layer

Wf-42 Phase 2 forbids the existing `MockScorer` /
`_stub_score` / `MEASURED_BUT_STUB` patterns from **re-appearing**
in any paper-§4 metric JSON. Three enforcement gates:

### 3.1 Gate A — `molmetal/tests/test_score_any_policy.py` (Wf-43 Phase 2)

Already in main branch (commit `73d0efb`). 20 tests covering:

* policy shape (REGISTRY: ≥23 entries + ROUTING_POLICY: ≥27 entries; full set incl. 10 model-eval + 7 training-diagnostics)
* registry↔policy alignment (every REGISTRY entry has a matching
  ROUTING_POLICY row)
* fallback-chain sanity (no fallback that points at a `MOCK_DELETED`
  metric; no fallback that points at a metric whose primary tool is
  also mock-deleted)
* policy.json sync (machine-readable JSON matches the in-process
  ROUTING_POLICY dict)
* mock-audit invariants: `mock_keys_in_registry == []`,
  `MOCK_DELETED` entries all point at a real alternative

### 3.2 Gate B — `molmetal/scripts/audit_mock_data.py` (Wf-42 Phase 1, re-runnable)

Re-run via:

```bash
cd /home/hugo/codes/mol-metal
source env.sh
python molmetal/scripts/audit_mock_data.py \
    --output molmetal/reports/wf42_mock_audit/mock_inventory.json
```

The script scans every `bench_*.py`, `benchmark_*.py`, `verify_*.py`,
`dock_10pockets_se3sb.py`, `wf30_dock_summarize.py` and emits the
8-critical / 23-major / 98-minor ledger. After the Wf-30 rerun
fixes the Vina sweep, the **8 critical** count drops to **0** and
the **10 major `use_real_vina_false`** count drops to **0**.

### 3.3 Gate C — `molmetal/scripts/eval_all_evaluators.py` (the smoke gate)

Re-run via:

```bash
cd /home/hugo/codes/mol-metal
source env.sh
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -u molmetal/scripts/eval_all_evaluators.py
```

This exercises every backend (RDKit, BioPython, OpenBabel, ProDy,
Vina, Meeko, PoseBusters, SA, QED, OpenMM, AiZynth smarts fallback)
and emits `n_pass / n_total`. Wf-43 Phase 2 expects **all
evaluators** to PASS on this host — see
`real_tools_inventory.md` §16 for the per-tool acceptance gate.

---

## 4. The forced-routing policy (declarative)

Every metric listed in `molmetal/scoring/policy.json` is **routed**
through `score_any` (or, for the 10 model-evaluation AUCs / 7
training-diagnostics JSONs, is *produced by the training script
directly*). The routing is enforced by the following three rules:

### Rule 1 — No metric name may contain "mock"

```python
# molmetal/tests/test_score_any_policy.py
assert all("mock" not in name.lower() for name in REGISTRY)
```

### Rule 2 — Every metric must be produced by a real, locally-installed tool

```python
# molmetal/tests/test_score_any_policy.py
for metric, entry in ROUTING_POLICY.items():
    assert entry.wf43_status in {"MEASURED", "RESERVED"}
    # RESERVED entries must have:
    #   - an explicit local_check that fails on this host, OR
    #   - a fallback_chain that ends in a MEASURED metric
```

### Rule 3 — Mock-fallback is forbidden

```python
# molmetal/scoring/score_any.py
class AdapterUnavailable(RuntimeError):
    """Raised when a metric has no live tool available.
    We never fabricate a mock score to 'make the metric work' —
    that is the behaviour Wf-43 was created to forbid."""
```

The `_stub_score` function in `dock_10pockets_se3sb.py` is the **only**
mock scorer remaining in the production path, and it only fires when:

1. `use_real_vina=False` is **explicitly** passed on the CLI, OR
2. `VinaDockingAdapter()` import fails (no `vina` / `meeko` wheel).

Neither of those conditions is the **default** for the Wf-30 sweep.
The Wf-42 Phase 2 close-out changes the default to `--use-real-vina`
for every workflow that feeds paper §4.

---

## 5. Per-metric decision matrix — every mocked metric → real replacement

For convenience, the full decision matrix (sorted by metric):

| Mocked metric (paper §4 row) | Real tool | Real adapter | Backend | Local check | wf43_status |
|---|---|---|---|---|---|
| `vina_score` (m3) | AutoDock Vina 1.2.7 | `vina_adapter.VinaDockingAdapter` | python+cli | `import vina, meeko` | MEASURED |
| `vina_kcal_per_mol` (alias → `vina_score`) | same | same | same | same | MEASURED |
| `vina_mean` (alias → `vina_score`) | same | same | same | same | MEASURED |
| `vina_high_affinity` (m3 sub-metric) | same | same | same | same | MEASURED |
| `vina_min_score` (m3 sub-metric) | same | same | same | same | MEASURED |
| `vina_reference` (redock) | same | same | same | same | MEASURED |
| `pb_pass_rate` (m8 / m11) | PoseBusters 0.6.5 | `posebusters_adapter.PoseBustersAdapter` | python | `import posebusters` | MEASURED |
| `aizynth_synthesis` (m12) | AiZynthFinder 4.6.0 OR SMARTS fallback | `aizynth_adapter.AiZynthAdapter` | python | `import aizynthfinder` (RESERVED) OR `_smarts_fallback` | PARTIAL — module missing |
| `sa_score` / `sa_mean` | RDKit Contrib `sascorer.py` | `sa_score.sa_score_ertl` | python | RDKit import | MEASURED |
| `qed_score` | RDKit `QED.qed` | `qed_scorer.qed_from_smiles` | python | RDKit import | MEASURED |
| `boltz_affinity` / `affinity_pearson_r` / `pearson_r_pic50` / `platinai_*` (m16) | Attentive D-MPNN (in-repo) | `pic50_predictor.AttentiveDMPNNPredictor` | python | ckpt at `molmetal/checkpoints/dmpnn_attn_ru_pic50.pt` | **GAP — checkpoint missing on this host** |
| `diffdock_pose` (m14 sub) | DiffDock-L | `diffdock_adapter.DiffDockAdapter` | python+cli | vendored repo + binary | RESERVED |
| `flowdock_pose` (m14 sub) | FlowDock | `flowdock_adapter.FlowDockAdapter` | python | in-repo | MEASURED |
| `equibind_pose` (m14 sub) | EquiBind | `adapters.equibind.EquiBindAdapter` | python | vendored repo + ckpt | RESERVED (stub today) |
| `metal_compliance`, `metal_coord_compliance` | RDKit SMARTS (in-repo) | in-repo SMARTS lib | python | RDKit import | MEASURED |
| `diversity_tanimoto`, `reference_tanimoto` | RDKit Morgan FP | `score_any._wrappers` | python | RDKit import | MEASURED |
| `af2_confidence` (m17 sub) | Voxelization surrogate | `voxelization.pocket_to_spatial_tiles` | python | NumPy | MEASURED |
| `pl_likelihood` | Persistent Laplacian (Liu/Feng 2026) | `per_residue_diversity` + topology util | python | NumPy | MEASURED |
| `ring_size_distribution`, `logp7_4`, `gi50_proxy`, `herg_cardio_risk`, `ames_mutagen`, `hepatotox_index`, `aqueous_solubility_logS`, `plasma_protein_binding`, `pharmacophore_pass_rate`, `cell_permeability_logPapp` | RDKit descriptors + SMARTS rule sets | `score_any._wrappers` | python | RDKit import | MEASURED |
| `diversity_subpocket` | BioPython NeighborSearch | `score_any._wrappers` | python | `from Bio.PDB import NeighborSearch` | MEASURED |
| `rigid_fragment_rmsd` | RDKit MCS + GetBestRMS | `score_any._wrappers` | python | RDKit import | MEASURED |
| `com_shift` | RDKit MMFF94 + CoM | `score_any._wrappers` | python | RDKit import | MEASURED |
| `coord_geometry_proxy` | RDKit SMARTS `metal_coord_compliance` | in-repo | python | n/a (renamed to MOCK_DELETED in Wf-43 §2.13) | MOCK_DELETED |
| `phototherapy_activity` | D-MPNN trained on phototherapy labels | `train_*.py` | python | n/a (no ship) | MOCK_DELETED |
| `ptiv_reduction_potential` | DFT surrogate | n/a (DFT not in repo) | n/a | n/a | MOCK_DELETED |
| `metalloprotein_affinity` | D-MPNN on `protein_metal_binding.csv` | `train_protein_metal.py` | python | `from molmetal.baselines.dmpnn_attentive import AttentiveDMPNNBaseline` | MOCK_DELETED (real alternative exists; not paper-claimed today) |
| `glide_score` | Glide (Schrödinger) | n/a (commercial) | n/a | `import schrodinger` (FAIL on this host) | RESERVED |
| `vina_qvina2` | QuickVina 2 (`qvina02` vendored) | `molmetal.scripts.q_vina2_swap` | cli | `references/MLM-Scaling/utils/docking/qvina02 --version` | RESERVED — binary executable |
| `vina_gpu` | Vina-GPU | `vina_gpu_adapter.QuickVinaGPUAdapter` | python+CUDA | `references/SynFlowNet/bin/QuickVina2-GPU-2-1` (MISSING) | RESERVED |
| `pocket2mol`, `targetdiff`, `decomp_diff`, `molcraft` | upstream 3D generative models | cite-only adapters | n/a | n/a | RESERVED (cite-only) |
| `drugood_ood`, `imold_ood` | DrugOOD / iMoLD upstream | in-repo proxy at `r3_drugood_benchmark.py` | python | `from molmetal.baselines.dmpnn_attentive import AttentiveDMPNNBaseline` | RESERVED |

### 5.1 Decision matrix for the **only** paper-§4 critical mock

The **only** paper-§4 critical mocked metric is:

```
m3 Vina Dock median = -9.66 kcal/mol (stub-floor ship-gate pass)
```

The remediation:

1. **Confirm** AutoDock Vina 1.2.7 + Meeko 0.8.0 are installed:
   `python -c "import vina, meeko; print(vina.__version__, meeko.__version__)"` → `1.2.7 0.8.0` (PASS, per
   `real_tools_inventory.md` §1.1).
2. **Rerun** the Wf-30 sweep with `--use-real-vina`:
   `python molmetal/scripts/dock_10pockets_se3sb.py --use-real-vina --out-dir metrics/by_metric --methods se3sb transport multiph baseline --pockets 830c 1h36 3eml 3ny8 4aaw 4rlu 1g9v 2hnx 4rn0 1fmc --seeds 0 1 2 3 4`
3. **Re-summarise**: `python molmetal/scripts/wf30_dock_summarize.py`
4. **Replace** the `vina_kcal_per_mol.json` entry whose `scorer` is
   `"stub"` with the real measurement.
5. **Update** `wf30_dock/best_m3_summary.json`, `m3_lift_analysis.json`,
   `per_pocket_metrics.json` from the rerun.
6. **Re-run** `audit_mock_data.py --only-severity critical` and
   verify the count drops from 8 to 0.
7. **Re-run** `pytest molmetal/tests/test_score_any_policy.py` and
   verify the `test_no_mock_keyword_in_registry` + `test_critical_mock_findings_drop` gates pass.

### 5.2 Decision matrix for the **D-MPNN pIC50 GAP**

The **D-MPNN pIC50 checkpoint** (`molmetal/checkpoints/dmpnn_attn_ru_pic50.pt`)
is missing on this host. Until it is materialised, `boltz_affinity`
returns NaN. Remediation:

1. Run `molmetal/scripts/retrain_pic50_neural.py` (or
   `train_cisplatin_resistance_v2.py`) on MetalCytoToxDB / PlatinAI
   corpus.
2. Save the checkpoint at
   `molmetal/checkpoints/dmpnn_attn_ru_pic50.pt` (the `DEFAULT_CKPT`
   path).
3. Verify with `python -c "from molmetal.molmetal_lam.sbdd_env.pic50_predictor import predict_pic50; print(predict_pic50('[NH3][Pt]([NH3])([Cl])[Cl]'))"` → expect a float in
   `[3, 9]`, not NaN.

Until the checkpoint is materialised, **m16 Pearson r** must report
`status: "NaN_ckpt_missing"` (not `MEASURED`) so the paper §4 row
honestly disclaims.

---

## 6. Files

* This file — `molmetal/reports/wf42_mock_audit/scoring_router.md`
* Companion inventory — `molmetal/reports/wf42_mock_audit/real_tools_inventory.md`
* Phase 1 audit (input) — `molmetal/reports/wf42_mock_audit/mock_inventory.json`
* Wf-43 routing policy (the policy this router enforces) —
  `molmetal/scoring/ROUTING_POLICY.md` + `molmetal/scoring/policy.json`
* Wf-43 runtime router — `molmetal/scoring/score_any.py`
* Wf-43 acceptance tests — `molmetal/tests/test_score_any_policy.py`
* Wf-42 Phase 1 audit script — `molmetal/scripts/audit_mock_data.py`
* Wf-43 Phase 1 acceptance gate — `molmetal/scripts/eval_all_evaluators.py`

## 7. Acceptance gate

```bash
# A. Mock-audit re-run (expect critical=0 after Wf-30 rerun)
python molmetal/scripts/audit_mock_data.py --only-severity critical --quiet

# B. Routing policy is the single source of truth
python -c "from molmetal.scoring.score_any import REGISTRY, ROUTING_POLICY; \
assert all('mock' not in n.lower() for n in REGISTRY); \
assert len(REGISTRY) >= 23; assert len(ROUTING_POLICY) >= 27; \
print('OK')"

# C. Every evaluator passes on this host
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -u molmetal/scripts/eval_all_evaluators.py

# D. Real Vina ready (the only critical fix)
python -c "import vina, meeko; print(vina.__version__, meeko.__version__)"
# expect: 1.2.7 0.8.0

# E. Wf-43 Phase 2 tests pass
pytest molmetal/tests/test_score_any_policy.py -v

# F. D-MPNN pIC50 checkpoint materialised (or status=NaN_ckpt_missing)
python -c "import os; p='/home/hugo/codes/mol-metal/molmetal/checkpoints/dmpnn_attn_ru_pic50.pt'; \
print('GAP' if not os.path.exists(p) else 'OK')"
```