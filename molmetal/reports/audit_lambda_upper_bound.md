# Audit: Lambda Upper Bound — Distance-to-Theory Report

*Date: 2026-09-11*
*Scope: `molmetal/molmetal_lam/` (8 layers) vs MLC theory memo §1-§9 + cloned references under `molmetal/references/`.*
*Companion to `h5_lambda_honest_framing.md`, `paper_outline.md`.*

---

## §1 — Current Lambda implementation vs theory: the 5 biggest gaps

The 8 MLC layers (atoms / bonds / molecules / reactions / types / binding / proof-search / pipeline) are *implemented*, but **theory sits at an upper bound we are nowhere near**. Five concrete gaps:

### Gap 1 — Heuristic prior is a constant (0.5), not a learned symbolic function

`search_alg/proof_search.py:106` `heuristic(features) → 0.5` and `closed_loop.py:331` fits PySR each iteration but **discards the tree**. MLC §9 prescribes a *closed-loop* in which every iteration's prior `P(a|s)` biases the next tree's PUCT selection, not just a paper equation. Today the heuristic affects *nothing* in the search — `_prior` returns a constant regardless of `features`, so the PUCT term collapses to `c_puct * 0.5 * sqrt(N) / (1+N_child)` and the search behaves like vanilla UCB with *no learned guidance*. PySR is exercised but never integrated into the live search loop.

### Gap 2 — Tile library is 12 hand-curated SMILES, not a ChEMBL / ZINC / tmQM-scale pool

`tile_lib/click_tiles.py:158` hardcodes 4 + 4 + 4 SMILES. The theory memo §3/§9 prescribes 10k–100k tiles drawn from ChEMBL reactive handles (and, for metalloproteinases, the **tmQM** 108k TMC corpus). With only 12 tiles the MCTS expansion factor is bounded above by `|rules| × 12 = ~72`, so the proof search **cannot discover novel chemistry** — it only permutes the 12 standard handles. This is the dominant ceiling.

### Gap 3 — Reward signal `scorer` is a stub, not a true multi-property predictor

`pipeline/closed_loop.py:153` declares `scorer: REINVENT4Scorer` with `batch_score(smiles_list)`, but the constructor accepts `Any` — there is no enforced Protocol contract. In practice the closed loop scores candidates with whatever callable it is handed (often `RandomForest` from `pipeline/extract_features.py`). MLC §7 prescribes scoring as `(qed, binding, sas, novelty)` from a REINVENT4-style multi-property head. Without a real binding predictor (DiffDock/EquiBind/TargetDiff output) the search **optimises a proxy**, not the proof obligation `∃ M: M : BindingType(site)`.

### Gap 4 — No learning across iterations (stateless MCTS rebuild)

`closed_loop.py:27-29` is explicit: *"the loop is intentionally stateless across iterations: it does not backpropagate rewards into the MCTS tree (the tree is rebuilt every call from the seed tile)"*. Theory §8 prescribes a *persistent* tree where `_prior` improves as `(features → reward)` data accumulates — i.e. **the prior IS the function approximator** (AlphaZero-style). Today's loop refits a heuristic but the next iteration's MCTS ignores it. The closed loop is open.

### Gap 5 — Binding & 3D are post-hoc, not in the search objective

`sbdd_env/vina_adapter.py`, `posebusters_adapter.py`, `aizynth_adapter.py` all exist and produce paper-grade numbers *outside* the MCTS loop. They are **post-hoc filters** on the top-K. The proof obligation `M : BindingType(site)` is type-checked at the leaf (`_binds_target` → `typecheck(state, self.binding_site)`) but `typecheck` delegates to **RDKit-only fingerprint matching**, not to a learned pose predictor. DiffDock/EquiBind/TankBind/FlowDock outputs are never threaded back into the search; they are only invoked in the validation report. MLC §6 prescribes *higher-order* types whose inhabitation check requires a 3D pose prediction head.

> **Bonus gap (acknowledged)**: the `Pipeline` layer is single-seed, single-pocket, single-iteration; there is no multi-pocket batched design or population-based training as in Pocket2Mol/TargetDiff.

---

## §2 — Per-cloned-project plug plan

For each repo: **(1)** which MLC layer it enhances, **(2)** plug mechanism, **(3)** Protocol class + 5–10 line adapter sketch.

### 2.1 Pocket2Mol → `molecules` layer (generator port)

1. **Layer**: `molecules` — de novo 3D ligand generator (replaces / augments MCTS expansion).
2. **Plug**: **direct python import** — `from references.Pocket2Mol.sample import sample_iter`. Pocket2Mol is pure PyG and exposes a `Sampler.sample(pocket, n_samples)` entry point.
3. **Adapter sketch**:
```python
from typing import List, Protocol
from molmetal.domain import Pocket, Molecule
from molmetal.ports import MoleculeGenerator, GenerationConfig

class Pocket2MolAdapter(MoleculeGenerator):
    def __init__(self, ckpt="references/Pocket2Mol/ckpt/pre-trained/pocket2mol.pt"):
        import sys; sys.path.insert(0, "references/Pocket2Mol")
        from sample import Sampler
        self.sampler = Sampler(ckpt)
    @property
    def name(self) -> str: return "Pocket2Mol_v1"
    def setup(self, device="cuda"): self.sampler.load(device=device)
    def generate(self, pocket: Pocket, cfg: GenerationConfig) -> List[Molecule]:
        return self.sampler.sample(pocket.pdb_id, n_samples=cfg.n_samples)
    def get_metadata(self) -> dict: return {"engine": "Pocket2Mol (Peng 2022)"}
```
**Use case**: pocket-conditioned generator. Used to **seed** the MCTS root or to propose initial SMILES that the proof-search then re-derives a λ-term for.

### 2.2 targetdiff → `molecules` + `binding` layers

1. **Layers**: `molecules` (denoising generator) and `binding` (companion affinity predictor).
2. **Plug**: **weight checkpoint load** — `targetdiff/pretrained_models/...pt`. Direct import of `models.molopt_score_model` for the architecture, then load weights.
3. **Adapter sketch**:
```python
class TargetDiffAdapter(MoleculeGenerator):
    def __init__(self, ckpt="references/targetdiff/pretrained_models/..."):
        import sys; sys.path.insert(0, "references/targetdiff")
        from models.molopt_score_model import ScorePosNet3D
        self.net = ScorePosNet3D.load(ckpt)
    def generate(self, pocket: Pocket, cfg: GenerationConfig) -> List[Molecule]:
        return self.net.sample(pocket, n_samples=cfg.n_samples,
                                sampling_mode=cfg.mode)  # ['ddpm','ddim','torsion']
    def get_metadata(self) -> dict: return {"engine": "TargetDiff (Guan 2023)"}
```
**Use case**: state-of-the-art equivariant diffusion, optional **torsion-aware** sampling. Stronger generator than Pocket2Mol on CrossDocked2020.

### 2.3 DiffDock → `binding` layer (docking port)

1. **Layer**: `binding` — pose prediction (replaces RDKit-only `typecheck`).
2. **Plug**: **weight checkpoint load** — DiffDock ships `workdir/v1.1/score_model_4.pt`. Direct import of `models.aa_model.AA_Model`.
3. **Adapter sketch**:
```python
class DiffDockAdapter(DockingEngine):
    def __init__(self, ckpt="references/DiffDock/.../score_model_4.pt"):
        import sys; sys.path.insert(0, "references/DiffDock")
        from models.aa_model import AA_Model
        self.model = AA_Model.load(ckpt)
    def dock(self, molecule: Molecule, pocket: Pocket, cfg: DockingConfig) -> List[Complex]:
        return self.model.sample_diffusion_ligand(molecule, pocket,
                                                   n_poses=cfg.n_poses,
                                                   batch_size=cfg.batch_size)
    def get_metadata(self) -> dict: return {"engine": "DiffDock-L (Corso 2024)"}
```
**Use case**: real pose prediction replaces the `typecheck` inhabit check in `_binds_target`. Plugs as **the** binding type-check oracle.

### 2.4 REINVENT4 → `pipeline` layer (RL scorer for MCTS rollout)

1. **Layer**: `pipeline` (closed-loop scorer) — directly replaces the `scorer` callable in `LamClickDesignLoop`.
2. **Plug**: **subprocess RPC** — REINVENT4 ships a CLI (`python -m reinvent …`) plus a Python API. Run a scoring subprocess per iteration; scores stream back over a JSON-line socket. Alternative: direct import of `reinvent.runmodes.scorer`.
3. **Adapter sketch**:
```python
class REINVENT4Scorer:
    def __init__(self, config="references/REINVENT4/configs/score.yml"):
        import subprocess, json, socket
        self.proc = subprocess.Popen(
            ["python", "-m", "reinvent", "--scoring", config,
             "--jsonl-out", "/tmp/scores.jsonl"], stdout=subprocess.PIPE)
    def batch_score(self, smiles_list: List[str]) -> List[float]:
        # pipe smiles via stdin, read JSONL until len(smiles_list) returned
        for s in smiles_list: self.proc.stdin.write(s.encode() + b"\n")
        self.proc.stdin.flush()
        return [float(json.loads(l)["total_score"])
                for l in iter(self.proc.stdout.readline, b"")]
```
**Use case**: real multi-property scoring (qed + binding + sas + novelty) feeds back into MCTS `_rollout`'s `base = scorer(state)` line. This is the **highest-impact** plug.

### 2.5 flow_matching (Lipman CFM) → `molecules` + `binding` (custom training)

1. **Layers**: `molecules` (loss/objective) and `binding` (path/solver) — *not* a single model but a **library** that supports training our own equivariant flow head.
2. **Plug**: **direct python import** — `from flow_matching.path import AffineProbPath`, `from flow_matching.solver import ODESolver`.
3. **Adapter sketch**:
```python
class LipmanCFMHeuristic:
    """Trainable CFM head that replaces the constant heuristic() stub."""
    def __init__(self, dim_features=8, dim_cond=128):
        from flow_matching.path import AffineProbPath
        from flow_matching.solver import ODESolver
        self.path = AffineProbPath()
        self.solver = ODESolver(velocity_model=self._build_net(dim_features, dim_cond))
    def fit(self, X_feats, y_rewards):
        # x_0 ~ N(0,I), x_1 = normalised reward
        self._train_step(X_feats, y_rewards)
    def predict(self, X_feats) -> float:
        x1 = self.solver.sample(x_init=torch.randn_like(X_feats), t_grid=torch.linspace(0,1,50))
        return float(x1.mean())
```
**Use case**: replaces the PySR `HeuristicRegressor` stub with a **continuous, differentiable** value prior over feature space. Once fitted, plumbed into `_prior` so PUCT has real `P(a)`.

### 2.6 PySR → `search_alg` / `pipeline` (symbolic prior)

1. **Layer**: `search_alg` (heuristic) — already wired in `lam_chem/pysr_wrapper.py`, but currently a *post-hoc* fit.
2. **Plug**: **direct python import** (already present in `pysr_wrapper.py`) + **subprocess RPC** for Julia backend.
3. **Adapter sketch**:
```python
class PySRPrior:
    """Symbolic prior spliced directly into MCTSProofSearch._prior."""
    def __init__(self):
        from pysr import PySRRegressor
        self.reg = PySRRegressor(niterations=64, binary_operators=["+","*","/","-"],
                                 unary_operators=["square","sqrt","exp","log"])
    def fit(self, X_feats, y_rewards): self.reg.fit(X_feats, y_rewards)
    def __call__(self, features) -> float:
        import numpy as np
        return float(self.reg.predict(np.asarray(features).reshape(1,-1))[0])
    def equation(self) -> str: return str(self.reg.sympy())
```
**Use case**: produces an *interpretable* equation of the form `score ≈ a·MW + b·logP + c·TPSA + d` that becomes the **documented deliverable** of the closed-loop paper section. Already integrated at the wrapper level; needs lifting into the live `_prior` of MCTS (see Gap 4).

### 2.7 FlowDock → `binding` layer (joint flow for pose + affinity)

1. **Layer**: `binding` — geometric flow matching for protein-ligand docking + affinity regression in a single model.
2. **Plug**: **weight checkpoint load** — `flowdock/pretrained_models/...pt`.
3. **Adapter sketch**:
```python
class FlowDockAdapter(DockingEngine):
    def __init__(self, ckpt="references/FlowDock/checkpoints/flowdock.pt"):
        import sys; sys.path.insert(0, "references/FlowDock")
        from flowdock.models.flowdock_fm_module import FlowDockFM
        self.model = FlowDockFM.load_from_checkpoint(ckpt)
    def dock(self, molecule: Molecule, pocket: Pocket, cfg: DockingConfig) -> List[Complex]:
        # returns poses + per-pose affinity (single forward pass)
        return self.model(molecule, pocket, n_poses=cfg.n_poses)
    def predict_affinity(self, complex: Complex) -> float:
        return self.model.affinity(complex)
    def get_metadata(self) -> dict: return {"engine": "FlowDock (ISMB 2025)"}
```
**Use case**: replaces DiffDock with a **flow-matching** alternative that also regresses binding affinity — closes the gap between docking (`binding`) and scoring (`pipeline`).

### 2.8 FLOWR → `molecules` layer (interaction-aware generator)

1. **Layer**: `molecules` — pocket-conditioned generator with interaction-fragment priors.
2. **Plug**: **weight checkpoint load** — FLOWR ships pretrained equivariant flow heads.
3. **Adapter sketch**:
```python
class FLOWRAdapter(MoleculeGenerator):
    def __init__(self, ckpt="references/FLOWR/pretrained/flowr.pt"):
        import sys; sys.path.insert(0, "references/FLOWR")
        from flowr.models.fm_pocket import PocketFlowMatching
        self.model = PocketFlowMatching.load(ckpt)
    def generate(self, pocket: Pocket, cfg: GenerationConfig) -> List[Molecule]:
        return self.model.sample(pocket, n_samples=cfg.n_samples,
                                 interaction_prior=cfg.interactions)
    def get_metadata(self) -> dict: return {"engine": "FLOWR (2025)"}
```
**Use case**: structure-aware generator that respects **protein-ligand interaction constraints** (H-bond donor/acceptor, hydrophobic anchors). Strongest published 2025 baseline.

### 2.9 EquiBind → `binding` layer (regression-style docking)

1. **Layer**: `binding` — fast SE(3)-equivariant *regression* docking (no diffusion sampling).
2. **Plug**: **weight checkpoint load** + **direct python import** of `models.equibind.EquiBind`.
3. **Adapter sketch**:
```python
class EquiBindAdapter(DockingEngine):
    def __init__(self, ckpt="references/EquiBind/runs/.../best.ckpt"):
        import sys; sys.path.insert(0, "references/EquiBind")
        from models.equibind import EquiBind
        self.model = EquiBind.load_from_checkpoint(ckpt)
    def dock(self, molecule: Molecule, pocket: Pocket, cfg: DockingConfig) -> List[Complex]:
        return self.model(molecule, pocket, n_poses=cfg.n_poses)  # direct, no MCMC
    def get_metadata(self) -> dict: return {"engine": "EquiBind (Stärk 2022)"}
```
**Use case**: low-latency pose prediction (no diffusion chain). Useful as a **fast filter** before the slow DiffDock call inside MCTS rollout.

### 2.10 TankBind → `binding` layer (trigonometry-aware, regression)

1. **Layer**: `binding` — `tankbind/model.py` provides both pose prediction and affinity regression.
2. **Plug**: **direct python import** + weight load (`tankbind/saved_models/...pt`).
3. **Adapter sketch**:
```python
class TankBindAdapter(DockingEngine):
    def __init__(self, ckpt="references/TankBind/saved_models/tankbind.pt"):
        import sys; sys.path.insert(0, "references/TankBind")
        from tankbind.model import TankBindModel
        self.model = TankBindModel(); self.model.load_state_dict(torch.load(ckpt))
    def dock(self, molecule: Molecule, pocket: Pocket, cfg: DockingConfig) -> List[Complex]:
        coords, affinity = self.model.predict(molecule, pocket)
        return [Complex(pocket=pocket, molecule=molecule,
                        coords=coords, affinity=affinity)]
    def get_metadata(self) -> dict: return {"engine": "TankBind (Lu 2022)"}
```
**Use case**: alternative regression-style binder; useful for **affinity calibration** alongside DiffDock's pose-only output.

### 2.11 tmQM → `atoms` / `bonds` layer (pretraining data + bond-order priors)

1. **Layer**: `atoms` (metal centres) and `bonds` (Wiberg bond orders) — pretrains the dative-bond combinators.
2. **Plug**: **direct python import** of the `tmQM` data module — read `tmQM_X1.xyz.gz` + `tmQM_y.csv` and build a torch geometric dataset.
3. **Adapter sketch**:
```python
class TmQMPretrainAdapter:
    """Loads 108k transition-metal complexes for dative-bond pretraining."""
    def __init__(self, root="references/tmQM/tmQM"):
        import pandas as pd, gzip
        self.df = pd.read_csv(f"{root}/tmQM_y.csv")
        self.geom = self._lazy_load_xyz(f"{root}/tmQM_X1.xyz.gz")
    def __iter__(self):
        for _, row in self.df.iterrows():
            atoms, coords = self.geom[row["CSD_code"]]
            yield {"smiles": row["SMILES"], "atoms": atoms, "coords": coords,
                   "wiberg": row["Wiberg_BM"], "metal": row["metal"]}
```
**Use case**: pretrains the `Bond.dative` combinator on **real Wiberg bond orders** from tmQM's DFT single-points. Closes the metalloprotein (MMP13/MMP9) accuracy gap (see `f3_metalloprotein_coverage.md`).

---

## §3 — Recommended strengthening priority

| Rank | Project(s) | Why | P-class |
|:---:|:---|:---|:---:|
| 1 | **REINVENT4Scorer** (§2.4) | Closes the biggest gap (Gap 3): real multi-property reward in the loop. Without it the search optimises a proxy. | **P0** |
| 2 | **DiffDock** (§2.3) or **FlowDock** (§2.7) | Closes Gap 5 — turns `_binds_target` from RDKit fingerprint into a real pose prediction. | **P0** |
| 3 | **PySR / LipmanCFMHeuristic** (§2.5, §2.6) | Closes Gaps 1 + 4 — the prior finally becomes informative, and is *actually consulted* by `_prior`. | **P0** |
| 4 | **tmQM** (§2.11) | Closes the metalloprotein blind spot (f3 audit); ~108k complexes for dative-bond combinator pretraining. | **P1** |
| 5 | **Pocket2Mol** + **targetdiff** (§2.1, §2.2) | Augments the `molecules` layer with a learned generator; lets MCTS propose starting states from a real pocket-conditioned prior. | **P1** |
| 6 | **FLOWR** + **flow_matching** (§2.8, §2.5) | The 2025 SOTA — interaction-aware generation + continuous flow prior. | **P1** |
| 7 | **EquiBind** + **TankBind** (§2.9, §2.10) | Fast regression-style binding oracles for the MCTS rollout (cheaper than DiffDock per sample). | **P2** |

**P0 ships within one sprint** — three files: `molmetal_lam/sbdd_env/reinvent4_adapter.py`, `molmetal_lam/sbdd_env/diffdock_adapter.py`, plus lifting the PySR output into `MCTSProofSearch._prior`. **P1 is the metalloprotein/MMP13 paper-grade push**. **P2 is regression-docking fallback** when DiffDock is unavailable (CUDA missing).

---

## §4 — One-sentence per-layer summary

| Layer | Status | One-line takeaway |
|:---|:---:|:---|
| **atoms** | Layer 1 — basic primitives + dative combinators (L1) exist; **tmQM pretraining** would close the metal-coverage gap (Gap → §2.11). | Combinators are structurally complete but lack DFT-calibrated bond-order priors from tmQM. |
| **bonds** | Layer 2 — covalent/dative/coordination bond dataclasses implemented; **Bond.dative** needs tmQM-trained weights to be chemically realistic on MMP-active sites. | Bond algebra is sound; physical parameters are RDKit defaults, not learned. |
| **molecules** | Layer 3 — `MoleculeClosedTerm` works; **no learned generator** means MCTS can only permute 12 tiles (Gap 2). Plug Pocket2Mol/TargetDiff/FLOWR as `MoleculeGenerator` ports. | Closed-term algebra is the strongest layer — needs a generator port to escape the 12-tile ceiling. |
| **reactions** | Layer 4 — CuAAC/SPAAC/SPC/Diels-Alder/ThiolEne rules implemented; **no learned rule proposal**, all reactions are hand-curated. | Reaction set is exhaustive for click chemistry but never augmented from data. |
| **types** | Layer 5 — `TypePredicate` (Lipinski/Veber/Egan) implemented; **ADMET predicates are RDKit-only**, not learned. | Lipinski-style predicates are correct but coarse; a free-energy-aware refinement would tighten the type signature. |
| **binding** | Layer 6 — `typecheck(state, BindingSite)` uses RDKit fingerprint matching; **no DiffDock/EquiBind pose predictor** (Gap 5). Plug §2.3/§2.7/§2.9/§2.10. | The binding type-check is the weakest link — it must be replaced by a learned pose oracle. |
| **search_alg** (proof-search) | Layer 8 — MCTS is implemented (PUCT + α-conflation); **heuristic is a constant**, rollout is uniform, tree is rebuilt per call (Gaps 1, 4). Plug PySR/LipmanCFM (P0). | The MCTS algorithm is structurally correct but uninformed — the prior is the next biggest leverage point. |
| **pipeline** (closed-loop) | Layer 9 — `LamClickDesignLoop.run` works and produces a paper equation; **the loop is stateless** (Gap 4) and the heuristic fit does not feed back into MCTS. Plug REINVENT4Scorer + persist the tree (P0). | The loop is the right shape but open — it must backpropagate into a persistent tree to reach the MLC §9 upper bound. |

---

*End of audit. All cloned references listed in `molmetal/reports/h4_paper_grade_comparison.md` and protocol shapes in `molmetal/ports/__init__.py`.*

---

### SOTA comparison (strict-protocol)

For the strict-protocol placement of every reference repo cited in §2 above (Pocket2Mol, TargetDiff, DiffDock, REINVENT4, flow_matching, PySR, FlowDock, FLOWR, EquiBind, TankBind, tmQM) against Lambda, see `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` §1. The Group A (CrossDocked100 / Vina) and Group B (PDBBind / RMSD) rows there give each reference's reported mean (Pocket2Mol −7.07 Vina @ n=100, TargetDiff −8.45 @ n=100, DiffDock-L 43.0% RMSD<2 Å @ n=363, FlowDock 51% @ n=308, etc.) with arXiv IDs from the four provenance audits. Until the P0 work (100-pocket Lambda run, 100k MCTS sims on ROCm) is complete, none of the §2 plug plans above can be evaluated on a paper-grade directly-comparable basis; the master table documents the gap and the §3 priority ranking of this audit is the next-best proxy.
