# WF-Lambda-2 Audit — Homotype Diversity Surface in Lambda

**Auditor**: WF-Lambda-2.A
**Date**: 2026-09-14
**Goal**: Define `homotype_diversity(mol_set)` — a Lambda-native diversity metric that uses *typed-variable hits* (per-atom constructor symbol counts in the BNF) + *beta-reduction path length* (= depth in the MCTS search tree) to measure chemical diversity, independent of SE(3) distance or atom-level Morgan Tanimoto. This is the third first-class algorithmic asset Lambda brings to the paper (after `alpha_equivalence` used in deduplication and `beta-NF` used in synthesizability).

Honest-framing: every claim below is marked **MEASURED** (read directly from a code path) vs **PROJECTED** (inferred / requires a new accessor).

---

## 1. Lambda-term AST — what typed-variable hits look like

**File**: `molmetal/molmetal_lam/lam_chem/ast.py`

The Molecular Lambda Calculus (MLC) AST is a tiny 3-node dataclass hierarchy:

- `LamVar(name: str, domain: str = "real")` — a *variable* / leaf. `name` is the typed-variable identifier (e.g. `"T"`, `"cat"`, but in our chemistry usage the *constructor symbol* is `Atom.symbol` — see §5). `domain` is a type-tag (`"real"`, `"molar"`, `"kelvin"`) — not enforced structurally, just metadata. **MEASURED** at `ast.py:78-92`.
- `LamAbs(var: LamVar, body: LamNode)` — `λx.M`. `var` is the binder; `body` is the scope. **MEASURED** at `ast.py:120-129`.
- `LamApp(func: LamNode, arg: LamNode)` — `(M N)`. `func` and `arg` are the two children. **MEASURED** at `ast.py:184-191`.

Methods on each node: `to_string()`, `free_vars()`, `subst(x, n)`, `beta_reduce()`, `to_normal_form(max_depth)`. **MEASURED**.

Important scoping semantics:
- `LamVar.free_vars()` returns `frozenset({self.name})` — a free occurrence. **MEASURED** at `ast.py:98-100`.
- `LamAbs.free_vars()` returns `self.body.free_vars() - {self.var.name}` — scope-shadowing is structural. **MEASURED** at `ast.py:135-137`.
- `LamApp.beta_reduce()` implements **leftmost normal-order** reduction: `(λx.M) N → M[x := N]`. **MEASURED** at `ast.py:209-227`.
- `LamApp.to_normal_form(max_depth=256)` loops `beta_reduce()` until fixed-point. **MEASURED** at `ast.py:229-242`.

**Caveat (MEASURED)**: this AST is *purely generic*. It is not the AST that the MLC search actually manipulates at runtime. The runtime representation is `MoleculeClosedTerm` (a graph with atoms + bonds + ledger, not a `LamApp` tree). The relationship between `MoleculeClosedTerm` and `LamApp` is:

> *A molecule IS a closed lambda-term in beta-normal form.* (verbatim from `lam_chem/__init__.py`)

So "typed-variable hits" in MLC means: **the multiset of `Atom.symbol` values present in the BNF** (each atom occurrence = one typed-variable hit). The `LamVar.name` here is the constructor symbol (e.g. `Pt`, `Ru_II`, `C`, `N`), not a lambda-binder name. The reduction path (a sequence of β-reductions `term_0 → term_1 → … → term_BNF`) is realised at runtime as `MoleculeClosedTerm.reduce_once()` calls; the BNF predicate is `check_beta_normal_form_for_rdkit_term` (§2).

**PROJECTED**: `homotype_diversity` should consume `MoleculeClosedTerm.atoms` (a `List[Atom]`), where each `Atom` has `.symbol` (the typed-variable constructor symbol) and `.valence`, `.lone_pairs`, `.arity` (the combinator properties). This is the same surface that `diversity_alpha` (r4) and `metal_geometry_prior_bonus` already use. **MEASURED at `atoms/combinators.py:66-106`** (Atom dataclass).

---

## 2. Beta-normal-form path — what the new check returns

**File**: `molmetal/molmetal_lam/lam_chem/well_formedness.py`

The new check `check_beta_normal_form_for_rdkit_term(term)` lives at **`well_formedness.py:306-353`**. Per-atom, it reads `term.valence_used[i]` (an RDKit-populated dict keyed by atom index) and cross-checks against `term.ledger.site(atom).used_sites`. An atom is valence-saturated iff `used >= atom.valence` (NOT `atom.arity` — lone pairs don't keep the term open). **MEASURED**.

The `term.valence_used` dict is the per-atom saturation field; for SMILES-parsed terms it is populated by `MoleculeClosedTerm.from_rdkit` to be RDKit's `atom.GetTotalValence()` which already counts implicit Hs. **MEASURED at `molecules/closed_term.py:114-120`** (the `valence_used: Dict[int, int]` dataclass field) and at `molecules/closed_term.py:565-605` (`from_rdkit` populates it from RDKit).

**Beta-reduction path** (sequence from initial term to BNF):

1. `MoleculeClosedTerm.has_redex() -> bool` — **MEASURED at `molecules/closed_term.py:318`**. Returns `True` iff the term has at least one reducible application (= an unbeta-reduced click redex awaiting firing).
2. `MoleculeClosedTerm.reduce_once() -> MoleculeClosedTerm` — **MEASURED at `molecules/closed_term.py:333`**. Returns the next one-step reduction (consumes a free site via `AtomSite.consume()`); or returns the term unchanged if `not has_redex()`.
3. Iterating `reduce_once()` until `not has_redex()` is the chemistry realisation of `LamApp.to_normal_form(max_depth)`. **PROJECTED** (no `_to_normal_form` helper is exposed on `MoleculeClosedTerm` today, but the loop is one-liner-trivial and the existing `from_rdkit`/`reduce_once` is fully exercised by tests).

**Path-length metric**: `len(reduce_once_sequence) = total_click_steps_to_close`. For an N-click-chemistry trajectory this is at most `N`. **PROJECTED** that this becomes the `beta_reduction_depth` per-node datum.

**Per-rule click-rule data**: each `_MCTSNode.rule_name` carries the reaction rule that produced it (e.g. `"CuAAC"`, `"SPAAC"`, `"ThiolEne"`). **MEASURED at `proof_search.py:1556-1561`** — see §3.

---

## 3. `_MCTSNode` — per-node data already exposed

**File**: `molmetal/molmetal_lam/search_alg/proof_search.py`

`_MCTSNode` dataclass at **`proof_search.py:1521-1561`** stores:

```python
state: MoleculeClosedTerm            # the closed lambda-term at this node
parent: Optional[_MCTSNode]          # back-pointer (== previous BNF)
N: int = 0                           # visit count
W: float = 0.0                       # total value
P: float = 0.5                       # prior probability
children: List[_MCTSNode]            # one β-reduction step successors
rule_name: Optional[str] = None      # reaction rule that produced this node
tile: Optional[MoleculeClosedTerm] = None  # the partner used in the reaction
virtual_loss: VirtualLoss = field(...)
expansion_complete: bool = False
```

**Per-node data needed for `homotype_diversity`**, mapped to the audit checklist:

| Checklist item | Field / accessor | Status |
| --- | --- | --- |
| (a) typed-variable hit count | `node.state.atoms` → multiset of `Atom.symbol` (e.g. `{"C": 5, "N": 1, "Pt_II": 1}`) | **MEASURED** at `proof_search.py:1550` + `atoms/combinators.py:66-95`. Aggregate over the leaf-set = the Σ C/N/O/Pt hit counts. |
| (b) beta-reduction depth | `len(path_to_root)` = tree-depth of node | **MEASURED** at `proof_search.py:2389-2420` (`path: List[_MCTSNode]` is collected in `_simulate`) and `proof_search.py:2345` (`self.tree_diversity = len(leaves_by_smi) / max(1, self._count_nodes(root))` already walks the depth structure). Also `_rollout_depth_hist` populated at `proof_search.py:2446`. |
| (c) click-rule application count by `rule_name` | `node.rule_name` (one `str` per node) | **MEASURED** at `proof_search.py:1556` + backpropagated along path. `Counter(rule_name)` across a leaf-set is exactly the per-click-rule click count. |

**Already-collected per-run diagnostics that `homotype_diversity` can re-use**:
- `self.tree_diversity: float` — **MEASURED** at `proof_search.py:2345`. Fraction of unique leaves over total nodes; this is **α-equivalence diversity**, not typed-variable diversity (but it is one of the three first-class Lambda assets — used by `alpha_equivalence_uniqueness_score` in `r4_lambda_only_run.py:176-198`).
- `self.rollout_depth_hist: Dict[int, int]` — **MEASURED** at `proof_search.py:2446`. Histogram of β-chain length per simulation. Could be averaged per pocket as the existing β-reduction-depth summary statistic.
- `accumulated_leaf_pairs: List[Tuple[List[float], float]]` — **MEASURED** at `proof_search.py:1966-1968`. Per-leaf feature vector + score.
- `accumulated_leaf_pairs[i]` uses `_default_feature_extractor` → `[n_atoms, n_bonds, n_free_sites, n_heavy]` (4 floats). **MEASURED** at `proof_search.py:347-364`. For typed-variable hits the extractor would need to grow by N symbol dimensions or be replaced by `homotype_signature(state)` returning the per-symbol hit dict.

**Reducer-side diagnostic (external)**: `diagnose_lambda_diversity.py:64-67` already records `reduction_counts[(rule_name, "calls"/"nonfires"/"product_instances")]` per rule via an observing subclass. **MEASURED**. This is the cleanest pre-existing baseline for (c).

---

## 4. Existing diversity metrics — Tanimoto-based baseline

**Two Tanimoto-based diversity baselines exist today**:

### 4a. `molecular_diversity(smiles_list, device='auto')` — RDKit Morgan FP + GPU Tanimoto
**File**: `molmetal/validation/gpu_molecular_metrics.py:9-53`.

- Builds Morgan radius-2, 2048-bit fingerprints on CPU.
- Computes pairwise binary Tanimoto on `torch.float32` tensors on the requested device (CPU or CUDA).
- Returns `mean_pairwise_tanimoto` and `diversity = 1 - mean`. **MEASURED**.
- Used by `r10_cfg_real_crossdocked.py:379` and `summarize_physical_sweep.py:62,74`.
- **Honest framing**: requires CUDA for the GPU path; on RX 7800 XT (gfx1101, ROCm 7.2) the torch binary Tanimoto is **PROJECTED** to work (ROCm + `cuda` PyTorch API) but the existing paths were tested on NVIDIA. The CPU fallback is **MEASURED** at `gpu_molecular_metrics.py:38-50`.

### 4b. `diversity_alpha(states)` — character-set-difference proxy (existing Lambda channel)
**File**: `molmetal/scripts/r4_lambda_only_run.py:437-475`.

```python
def diversity_alpha(states):
    # mean pairwise |sym(smi_a) Δ sym(smi_b)| / max(len_a, len_b)
    # set_i = set(smi_i)   # set of characters in canonical SMILES
    # sym_diff = |set_i Δ set_j|
    # returns sym_diff / max(len_i, len_j)
```

This is the closest existing "Lambda-native" diversity baseline, but it is *very crude*: it operates on the raw character set of the canonical SMILES string, so `"CC"` and `"CO"` give `sym_diff = 2 / 2 = 1.0` while `"c1ccccc1"` and `"C1CCCCC1"` give `sym_diff = 6 / 7 ≈ 0.857` (they share `"1"` and `"c"`). It is not typed-variable-aware. **MEASURED** at `r4_lambda_only_run.py:437-475`.

### 4c. `alpha_equivalence_uniqueness_score(states)` — distinct β-NF count
**File**: `molmetal/scripts/r4_lambda_only_run.py:176-198`.

`1 / (1 + (n_distinct_canonical_smi - 1))` — saturates to 1.0 when all unique, 0.5 when 2 distinct, 0.33 when 3, etc. **MEASURED**.

### 4d. `tree_diversity` — α-classes / total nodes ratio
**File**: `molmetal/proof_search.py:2345`. `len(leaves_by_smi) / max(1, _count_nodes(root))`. **MEASURED**.

### 4e. `n_unique` (candidates-only)
Trivial set-cardinality over the top-K candidates' canonical SMILES — `lambda_100pocket_sweep.py` and `r4_lambda_only_run.py` both surface it as a per-cell column.

**Comparison baseline for `homotype_diversity`**: `molecular_diversity(smis, device='cuda:0')` is the canonical Morgan-Tanimoto reference. The Lambda-native metric should be benchmarked alongside it on the same `mol_set` and shown to be **uncorrelated** with Tanimoto diversity (or at least only partially correlated) — that is the contribution.

---

## 5. Confirmation: typed-variable names are the constructor symbol

**MEASURED at `atoms/combinators.py:66-95`**: `Atom(symbol: str, atomic_num: int, valence: int, lone_pairs: int, geometry: str)` where `symbol` is the *constructor* / typed-variable name (`"C"`, `"N"`, `"Pt_II"`, `"Ru_II"`, `"Au_III"`, `"Ir_III"`, `"Cu_II"`, `"Zn_II"`). The standard chemistry rules at `atoms/combinators.py:165-173` and the metal combinator library at `atoms/combinators.py:195-225` define the typed-variable vocabulary.

**MEASURED at `molecules/closed_term.py:477-498`** (`_value_signature`): the per-term signature used internally is `tuple(sorted(a.symbol for a in self.atoms))` — i.e. the sorted multiset of typed-variable symbols. This is already the canonical "typed-variable hit" representation in MLC. **`homotype_diversity` can directly reuse `MoleculeClosedTerm._value_signature()`** as the per-molecule typed-variable histogram (or recompute the symbol multiset itself — `_value_signature` is private API but conceptually exactly what we need).

**NOT used today**: there is no existing helper that takes the multiset-of-symbols view and turns it into a diversity score. `homotype_diversity(mol_set)` would:

1. For each `state` in `mol_set`, compute `state._value_signature()` (or `Counter(a.symbol for a in state.atoms)`).
2. Compute pairwise **multiset-Jaccard distance** (or symmetric multiset difference normalised by total symbol count) over the multiset representation — this is the *typed-variable-hit* dimension.
3. Compute pairwise **β-reduction-depth distance** (the depth of the leaf in the MCTS tree, or `len(reduce_once_sequence)` to BNF for terms without search-tree context) — this is the *reduction-path* dimension.
4. Return the mean pairwise combined distance, or report them as two orthogonal axes (typed-variable-Jaccard vs β-depth-L1) for the paper figure.

**Honest framing**: steps 1 and 4 are MEASURED on existing code. Step 2 (multiset-Jaccard) is **PROJECTED** — no helper exists; needs to be added to `homotype_diversity` (likely `molmetal_lam/metrics/homotype.py` or `molmetal/scripts/r4_lambda_only_run.py`). Step 3 (β-depth) is MEASURED at `proof_search.py:2446` for search-tree terms but **PROJECTED** for terms coming from outside an MCTS run (need to compute `reduce_once` loop length manually).

---

## 6. Implementation plan (PROJECTED — to be approved before coding)

### File changes
1. **NEW** `molmetal/molmetal_lam/metrics/homotype.py` — the `homotype_diversity(mol_set, *, depth_lookup=None)` function. Uses `state._value_signature()` for the typed-variable hit multiset, computes multiset-Jaccard pairwise distance and (optionally) L1 β-depth distance.
2. **PATCH** `molmetal/scripts/r4_lambda_only_run.py` — add `homotype_diversity` to the `CellResult` dataclass (line 519) and to the per-cell run (around line 711).
3. **PATCH** `molmetal/molmetal_lam/search_alg/proof_search.py` — surface the per-leaf `Counter(rule_name)` and per-leaf `_value_signature()` into a new attribute on `MCTSProofSearch` (e.g. `self.typed_variable_hits: Dict[str, Counter[str]] = {}` populated at `_collect_leaves`).
4. **TESTS** `molmetal/tests/test_homotype_diversity.py` — ≥6 tests:
   - empty set → 0.0
   - identical molecules → 0.0
   - all distinct → 1.0 (typed-variable Jaccard)
   - cis-platin vs trans-platin → low (shared {Pt, NH3, Cl})
   - cisplatin vs doxorubicin → high (very different typed-variable histograms)
   - β-depth: same molecule reached at depth 2 vs depth 5 → β-depth component ≥ 0
   - independence from Tanimoto: pick a mol_set where Tanimoto diversity is low but typed-variable hit diversity is high (or vice versa) → assert correlation < 0.5.

### Paper framing
`homotype_diversity` is *not* a replacement for Morgan-Tanimoto diversity; it is a **Lambda-native** orthogonal axis. Paper §3 (Lambda as First-Class Generator) gains a third subsection: *homotype diversity* (typed-variable hits + β-reduction path) alongside *α-equivalence uniqueness* and *β-NF synthesizability*. The figure shows scatter of Tanimoto vs homotype per pocket — diversity along the Lambda axis is information not captured by atom-fingerprint similarity.

---

## 7. File:line index (the load-bearing references)

- AST node definitions: `lam_chem/ast.py:78-92` (LamVar), `lam_chem/ast.py:120-129` (LamAbs), `lam_chem/ast.py:184-191` (LamApp).
- `beta_reduce` leftmost normal-order: `lam_chem/ast.py:209-227`.
- `to_normal_form` looping: `lam_chem/ast.py:229-242`.
- New valence-aware BNF check: `lam_chem/well_formedness.py:306-353`.
- Per-atom saturation dict: `molecules/closed_term.py:114-120` (`valence_used`).
- `has_redex`: `molecules/closed_term.py:318`.
- `reduce_once`: `molecules/closed_term.py:333`.
- `is_closed` per-atom valence check: `molecules/closed_term.py:157-208`.
- `_value_signature` (sorted-symbol multiset): `molecules/closed_term.py:477-498`.
- `Atom` dataclass (the typed-variable constructor): `atoms/combinators.py:66-95`.
- Standard chemistry typed-variables: `atoms/combinators.py:165-173`.
- Metal combinator typed-variables (Pt_II, Ru_II, …): `atoms/combinators.py:195-225`.
- `_MCTSNode` dataclass: `proof_search.py:1521-1561`.
- `_MCTSNode.rule_name` and `tile`: `proof_search.py:1556-1557`.
- `_MCTSNode.expansion_complete`: `proof_search.py:1561`.
- `_simulate` collecting `path`: `proof_search.py:2389-2446`.
- `_rollout_depth_hist` (β-chain length histogram): `proof_search.py:2446`.
- `tree_diversity` (α-class ratio): `proof_search.py:2345`.
- `accumulated_leaf_pairs`: `proof_search.py:1966-1968`.
- `_default_feature_extractor` (4-float baseline): `proof_search.py:347-364`.
- `diversity_alpha` (character-set-difference baseline): `scripts/r4_lambda_only_run.py:437-475`.
- `alpha_equivalence_uniqueness_score` (BNF-class baseline): `scripts/r4_lambda_only_run.py:176-198`.
- `metal_geometry_prior_bonus` (typed-variable-aware prior): `scripts/r4_lambda_only_run.py:222-276`.
- `molecular_diversity` (Morgan-Tanimoto baseline, GPU): `validation/gpu_molecular_metrics.py:9-53`.
- `diagnose_lambda_diversity` observing subclass (pre-existing β-reduction telemetry): `scripts/diagnose_lambda_diversity.py:63-78`.
- `ReactionRule.reduce` and `self.name` (rule_name source): `reactions/beta_reductions.py:145-356` and concrete rules at `:545+`.

---

## 8. Verdict

**STATUS**: `homotype_diversity` is **PROJECTED** — the underlying data (typed-variable symbols, β-reduction depth, per-rule counts) is **MEASURED** at the leaf and node level, but no function currently fuses them into a Lambda-native diversity score. Two straightforward steps are needed:

1. A new `homotype_diversity(mol_set)` function (~40 lines, lives in `molmetal_lam/metrics/homotype.py`).
2. A surface patch on `MCTSProofSearch` to populate `self.typed_variable_hits` per leaf at `_collect_leaves`.

The MEASURED surface (typed-variable multiset via `_value_signature`, β-depth via `_rollout_depth_hist`, per-rule counts via `_MCTSNode.rule_name`) is already complete; the PROJECTED glue is a one-evening patch + tests. This is the third first-class Lambda asset for the paper, independent of Tanimoto and SE(3) distance.
