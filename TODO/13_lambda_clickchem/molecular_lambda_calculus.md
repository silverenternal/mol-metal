# Molecular Lambda Calculus — 分子设计的计算基础

## 为什么 Lambda 演算在更上游?

我之前的 plan 把 Lambda 演算降级为"产率预测器"——这只是下游的标量函数 (y = f(T, [cat], t))。这是降维:把化学的丰富性压成单标量。

真正的深度是:**化学本身就是 Lambda 演算**。每一种化学概念都有直接的 λ-演算对应:

| 化学概念 | λ-演算对应 | 为什么不是降级 |
|---|---|---|
| 原子 | primitive combinator (S, K, B, C 或 Church 编码) | 不是黑盒 — 原子类型本身是 type |
| 共价键 | β-reduction (function application) | 键就是 application 的执行 |
| 配位键 (dative) | curried partial application | 部分应用 = 未饱和的配位点 |
| 芳香性 | η-conversion class | 等价类 = 同一 aromatic system |
| 立体化学 | distinct β-normal forms | 不同的 NF = 不同的手性 |
| 分子 | closed λ-term | 分子 = β-NF |
| 同分异构体 | α-equivalent but non-β-equivalent terms | 重命名不改变结构 |
| 反应 | β-reduction step | 每一步反应 = 一次归约 |
| 合成 (正向) | β-reduction sequence | 合成路径 = 归约序列 |
| 逆合成 | β-EXPANSION | 逆归约 = 拆解 |
| 结合 (蛋白 + 配体) | β-reduction + type-check | 复合物 = 归约 + 类型匹配 |
| ADMET | type predicate | "好分子" = 满足 type 的 term |
| 药物活性 | constructive proof | 药物 = "该靶点可结合"的构造证明 |
| 等变性 (SE(3)) | α-conversion invariance | 重命名坐标不改变分子 |
| GNN 表达能力 | η-equivalence class | GNN 拟合 η-equivalence |

**核心论断** (论文 Thesis Statement):
> "De novo drug design is constructive proof search in the Molecular
>  Lambda Calculus. Every generated molecule is a witness (proof term)
>  for the proposition 'this molecule binds target T with acceptable
>  ADMET'. Lambda-calculus gives us free interpretability (the term IS
>  the synthesis path) and synthetic accessibility (the term IS the
>  retrograhic tree)."

---

## 1. Atom 层: Atoms-as-Combinators

每个原子类型 (H, C, N, O, ..., Pt, Ru, Ir) 是一个**原始组合子**(primitive combinator)。我们用 SKI 演算 + 扩展:

```
基础组合子 (SKI 演算):
  I x = x                    (identity — 没有配位的 atom)
  K x y = x                  (constant — H, halide)
  S x y z = (x z)(y z)      (composition — C, N, O)

金属原子作为 4/5/6-arity 配位函数:
  Pt_II ≡ λa.λb.λc.λd. complex(a,b,c,d)     # square-planar
  Ru_II ≡ λa.λb.λc.λd.λe.λf. complex(a..f)   # octahedral
  Zn_II ≡ λa.λb.λc.λd. complex(a,b,c,d)      # tetrahedral

每个配体也是一个 λ-term,所以 metal + 配体的复合物 = 部分应用:
  Cisplatin = Pt_II(NH3)(NH3)(Cl)(Cl)
            = ((Pt_II NH3) NH3) Cl Cl      # left-associated partial application
```

**关键**: metal 是 **curried function with fixed arity**。任何缺少参数的 metal 就是一个 partial application (在反应中等价于"未饱和的配位点")。

```python
@dataclass(frozen=True)
class Atom:
    """An atom is a λ-combinator with a fixed arity (= valence + lone pairs)."""
    symbol: str                # 'H', 'C', 'Pt', 'Ru', ...
    atomic_num: int
    valence: int               # how many bonds it accepts (= arity)
    lone_pairs: int            # additional electron-pair sites
    is_metal: bool = False
    geometry: str = ""         # 'sp3', 'sp2', 'square_planar', ...
    
    @property
    def arity(self) -> int:
        """The arity of this atom as a λ-combinator."""
        return self.valence + self.lone_pairs
    
    def is_saturated(self, current_bonds: int) -> bool:
        """A saturated atom is in β-normal form."""
        return current_bonds == self.arity
    
    def can_bond_with(self, other: 'Atom') -> bool:
        """Two atoms can β-reduce (= bond) iff both have free arity."""
        return (not self.is_saturated(0)  # pseudo-check
                and not other.is_saturated(0))
```

---

## 2. Bond 层: Bonds-as-Application

**化学键 = λ-term 的 application**。

| 键类型 | λ-演算对应 | 例子 |
|---|---|---|
| 单键 | direct application | C—H = (C H) |
| 双键 | nested application | C=O = ((C O) O)  (双键 = 两次"partial double bond order") |
| 三键 | triple application | C≡N = (((C N) N) N) |
| 芳香键 | η-equivalence | benzene 的 6 个 C—C 都是 η-conversions |
| **配位键 (dative)** | **curried partial application** | Pt—NH₃ = (Pt NH₃) **保留 3 个未应用位** |
| 氢键 | type-checked application | H-bond = 两个 saturated atom 的 partial charge interaction |
| 范德华 | free variables shared | vdW = unbound variables in scope |

**关键洞察**: 配位键 = **curried application**。Pt 与 NH₃ 形成 dative bond 不是 bond,而是 Pt(NH₃) —— 一个 4-arity 函数的 partial application,保留了 3 个未应用的"配位位点"。

```python
class Bond:
    """A bond is a β-reduction (= function application) between two atoms."""
    
    def apply(self, atom_a: Atom, atom_b: Atom) -> 'Molecule':
        """β-reduction: applying atom_a to atom_b (or vice versa)
        
        Returns the reduced term (= molecule with one fewer free site).
        """
        new_a = atom_a.with_one_less_free_site()
        new_b = atom_b.with_one_less_free_site()
        return Molecule([new_a, new_b], bonds=[(a.id, b.id, self.order)])
    
    @classmethod
    def dative(cls, metal: Atom, ligand: Atom) -> 'Molecule':
        """Dative bond = curried application. Metal retains free sites."""
        # Pt(NH3) — Pt now has 3 free sites, NH3 is fully saturated
        new_metal = metal.with_one_less_free_site()
        saturated_ligand = ligand.saturate()  # ligand is now a value, not a function
        return Molecule([new_metal, saturated_ligand], 
                       bonds=[(metal.id, ligand.id, 'dative')],
                       metal_free_sites=new_metal.arity - new_metal.used_sites)
```

---

## 3. Molecule 层: Molecules-as-Closed-Terms

一个分子就是一个 **closed λ-term** (无自由变量),即 **β-normal form** (NF)。

**关键性质**:
- 同一分子的不同 Lewis 结构 = **α-equivalent** (重命名变量) — 不改变分子
- 同分异构体 = **β-NF 相同但 derivation 不同** — 不同合成路径产生同一 NF
- 立体异构 (R/S, E/Z) = **不同的 β-NF** (即使拓扑结构相同)

```python
@dataclass(frozen=True)
class Molecule:
    """A molecule is a closed λ-term in β-normal form."""
    atoms: List[Atom]                # atom stack
    bonds: List[Bond]                # applications applied (β-reduction history)
    
    @property
    def is_closed(self) -> bool:
        """Closed term = no free variables = no unsatisfied atoms."""
        return all(a.is_saturated() for a in self.atoms)
    
    @property
    def is_beta_normal_form(self) -> bool:
        """β-NF: no more reducible applications."""
        return self.is_closed and not self.has_redex()
    
    def alpha_equivalent(self, other: 'Molecule') -> bool:
        """α-equivalence: same topology, different atom labeling."""
        return self.canonical_smiles() == other.canonical_smiles()
    
    def has_redex(self) -> bool:
        """A redex = (λx.M) N, a reducible application."""
        # In our chemistry: an unsaturated atom with a free lone pair next
        # to an H atom (acidic) is a redex waiting to fire.
        for a in self.atoms:
            if not a.is_saturated() and a.has_acidic_neighbor(self):
                return True
        return False
    
    def reduce_once(self) -> 'Molecule':
        """One β-reduction step = one chemical transformation."""
        # Find a redex and reduce it (= perform the reaction)
        ...
```

---

## 4. Reaction 层: Reactions-as-β-Reduction

**Every chemical reaction is exactly one β-reduction step.**

传统反应分类:
- Substitution (SN2): 一个基团被另一个替换 — 就是一个 redex 的 β-reduction
- Addition: 两个分子合并 — 多个 redex 同时 fire
- Elimination: 一个分子分裂 — redex 在不同方向 fire
- Rearrangement: 原子重排 — α-conversion

```python
class Reaction:
    """A chemical reaction IS a β-reduction rule."""
    
    def reduce(self, mol: Molecule) -> Molecule:
        """Apply the redex pattern = perform the reaction."""
        ...

# Concrete reactions as β-reduction rules:

def SN2(nucleophile: Atom, substrate: Atom, leaving_group: Atom) -> Molecule:
    """SN2 substitution = one β-reduction with inversion.
    
    Pattern:    Nu: + R-LG  →  Nu-R + :LG
    
    λ-calculus:  (λx.R-x) (Nu LG)  →  (λx.R-Nu) LG
                          ↑ redex here
    
    (Walden inversion = the new bond forms on the opposite face)
    """
    new_substrate = substrate.replace_leaving_group(nucleophile)
    return Molecule([new_substrate, nucleophile, leaving_group],
                   bonds=[(new_substrate.id, nucleophile.id, 'single')])

def CuAAC(azide_tile: 'Tile', alkyne_tile: 'Tile') -> 'Tile':
    """CuAAC click reaction = β-reduction in the tile algebra.
    
    R-N3 + R'-C≡CH --[Cu(I)]--> 1,4-disubstituted triazole
    
    As λ-calculus:
      triazole = λ(R-N3). λ(R'-C≡CH). 
                 combine(
                   attach(R-N3, "azide_site"),
                   attach(R'-C≡CH, "alkyne_site")
                 )
    """
    ...

def cycloaddition(diene: 'Tile', dienophile: 'Tile', ring_size: int) -> 'Tile':
    """[4+2] Diels-Alder cycloaddition.
    
    In λ-calculus: a 4-arity function meeting a 2-arity function 
    produces a 6-ring (= saturated cycle).
    """
    ...
```

**关键洞察**: 我们不需要"分类"反应 — **任何 β-reduction 规则都是合法的反应**。这给了我们**自动的反应发现**: 在化学相容性约束下搜索所有可能的 β-reduction。

---

## 5. Synthesis 层: Forward & Retro Synthesis

**正向合成** = **β-reduction sequence** (从原料到产物)。
**逆合成** = **β-expansion** (从产物到原料)。

```python
def synthesize(target: Molecule, starting_materials: List[Tile]) -> List[Reaction]:
    """Forward synthesis = β-reduction sequence.
    
    Input: target Molecule (closed λ-term in NF)
    Output: sequence of reactions (β-reductions) that produce it from
            starting_materials (also closed terms, smaller NF).
    """
    # Search for a derivation: starting_materials ⤓ target
    return mcts_search_for_derivation(target, starting_materials)

def retrosynthesize(target: Molecule) -> List['ReactionPattern']:
    """Retrosynthesis = β-expansion.
    
    For each redex in the target's β-expansion space, generate
    a candidate precursor + the reaction that forms the bond.
    """
    # Apply all β-expansions to target until we reach "atomic" precursors
    expansions = []
    for redex_pattern in all_expansion_patterns():
        for precursor in target.beta_expand(redex_pattern):
            expansions.append((redex_pattern, precursor))
    return expansions
```

---

## 6. Binding 层: Protein + Ligand = β-Reduction + Type Check

**配体-蛋白结合** 是一个**类型检查的 β-reduction**:
- Ligand 是 closed λ-term
- Pocket 是 higher-order type (有 binding site 的形状约束)
- Binding = ligand 应用 pocket 的 binding-site type → 形成 complex
- 亲和力 = 该 application 在能量 landscape 上的稳定性 (= β-NF 的"可达性")

```python
class BindingType:
    """A binding site IS a higher-order type.
    
    E.g. MMP2 active site: "Zn-coordination" + "hydrophobic pocket" + "backbone H-bond"
    """
    def __init__(self, name: str, constraints: List['Constraint']):
        self.name = name
        self.constraints = constraints
    
    def typecheck(self, ligand: Molecule) -> 'TypeCheckResult':
        """Type-check ligand against this binding site.
        
        Returns: success (with affinity), or failure (with reason).
        """
        for c in self.constraints:
            if not c.satisfied_by(ligand):
                return TypeCheckResult(success=False, violated=c)
        return TypeCheckResult(success=True, affinity=predict_affinity(ligand, self))

def dock(ligand: Molecule, pocket: 'BindingType') -> 'Complex':
    """Dock = β-reduce ligand against pocket's type constraints."""
    typecheck = pocket.typecheck(ligand)
    if not typecheck.success:
        return Complex(ligand, pocket, success=False, reason=typecheck.violated)
    return Complex(ligand, pocket, success=True, 
                   affinity=typecheck.affinity,
                   pose=find_optimal_pose(ligand, pocket))
```

---

## 7. ADMET 层: Properties-as-Type-Predicates

每个 ADMET 性质是一个 **type predicate**:
- 满足 QED > 0.5 → type-inhabited
- 违反 Lipinski → type-error
- **Lipinski 五规则 = type signature**: 任何分子若违反就是 "ill-typed"

```python
class TypePredicate:
    """A property constraint = a type predicate."""
    def __init__(self, name: str, predicate_fn):
        self.name = name
        self.predicate = predicate_fn
    
    def __call__(self, mol: Molecule) -> bool:
        return self.predicate(mol)

# Standard predicates
LIPINSKI = TypePredicate("Lipinski", lambda m: 
    m.mw <= 500 and m.logp <= 5 and m.hbd <= 5 and m.hba <= 10)

VEBER = TypePredicate("Veber", lambda m:
    m.tpsa <= 140 and m.rotatable_bonds <= 10)

QED_ACCEPTABLE = TypePredicate("QED_acceptable", lambda m: m.qed >= 0.5)

# Well-typed molecule = inhabits all type predicates
def well_typed(mol: Molecule, predicates: List[TypePredicate]) -> bool:
    return all(p(mol) for p in predicates)
```

**Curry-Howard 同构**:
- 类型 = 命题 (proposition)
- 项 = 证明 (proof)
- inhabitation = 可满足性

**应用**: 寻找满足所有 ADMET 类型的分子 = 寻找 type-correct 的证明 = 自动 ADMET 过滤的 type checker

---

## 8. Drug Discovery Layer: Drug-Design-as-Proof-Search

**药物设计的核心命题**:
> "存在一个分子 M,使得 (M satisfies all ADMET types) ∧ (M binds target T)"

**λ-演算中的对应**:
- 寻找满足所有类型的证明 term (即分子)
- **Proof search** = 搜索满足所有 type 的 term
- MCTS 就是 proof search 的算法

```python
def drug_design_proof_search(
    target: BindingType,
    admet_predicates: List[TypePredicate],
    tile_library: List[Tile],
    max_iterations: int = 1000
) -> List[Molecule]:
    """Find all molecules that:
    - Inhabit all ADMET type predicates (well-typed)
    - Type-check against the binding site (bind target)
    - Are constructible from the tile library (synthesizable)
    """
    well_typed_molecules = []
    for it in range(max_iterations):
        # 1. Generate candidate via MCTS (proof search)
        candidate = mcts_proof_search_step(target, admet_predicates, tile_library)
        
        # 2. Type-check
        if not well_typed(candidate, admet_predicates):
            continue
        if not target.typecheck(candidate).success:
            continue
        
        # 3. Synthesize (verify the proof is constructive)
        synthesis_path = synthesize(candidate, tile_library)
        if synthesis_path is None:
            continue
        
        well_typed_molecules.append((candidate, synthesis_path))
    
    return well_typed_molecules
```

**每个生成的分子都是一个 proof term**。论文的杀手锏是: 我们可以**展示这个 proof term 本身** —— 它就是分子的构造路径。

---

## 9. Equivariance 层: GNN = η-Conversion Learner

**为什么图神经网络 (GNN) 在分子上工作得很好?** 因为 GNN 拟合的是 **η-conversion class**:
- η-conversion: `λx.f x ≡ f` (如果 f 是函数, λx.f x 和 f 行为相同)
- GNN 的 permutation-invariant aggregation 拟合的就是这种"重命名不变性"
- **GNN 学习的是 η-equivalence 的特征化**(feature of the equivalence class)

```python
class EGNN(nn.Module):
    """EGNN is morally learning a η-conversion-invariant function.
    
    The message passing step:
        m_ij = MLP(h_i || h_j || ||x_i - x_j||²)   # pairwise interaction
        h_i^{l+1} = h_i^l + Σ_j m_ij               # sum = permutation-invariant
    
    The η-conversion is the aggregation:
        The output h_i depends only on the multiset of h_j's, not their order.
        So h_i = λ(σ permutation). η-reduce_to_canonical_form
    """
    ...

class FlowMatching:
    """Flow matching for molecules = learning the β-NF from noise.
    
    Forward (β-reduction):
        x_t = (1-t) * x_0 + t * x_1
    
    This is exactly the affine-linear path from Lipman 2023 §4.8,
    which is the SIMPLEST β-reduction path between two λ-terms.
    """
    ...
```

**关键**: Mol-Metal 的 EGNN + Flow Matching **就是 λ-演算的 η-conversion + β-reduction 算子**。EGNN 把分子投影到 η-equivalence class,FM 把噪声投影到 β-NF。

这就是为什么两轨融合如此自然 — **它们是同一计算 (λ-演算) 的两个不同算子**!

---

## 10. Type Theory: Mol-Metal's EGNN = Type Inference

**Mol-Metal 的 EGNN IC50 预测器** 在 λ-演算框架下就是 **type inference**:
- 给定分子 (λ-term)
- 输出该分子在 BindingType 上的 inhabitation score (= IC50)
- 这就是**类型推断**:给定 term,推断它 inhabits 哪些类型

```python
class MolMetalIC50Predictor:
    """EGNN-based IC50 predictor = type inference.
    
    IC50(M) = "how strongly does M inhabit BindingType(T)?"
    
    In λ-calculus: type inference assigns the "deepest" inhabitation score.
    EGNN approximates this via η-invariant features.
    """
    def predict(self, molecule: Molecule, target: BindingType) -> float:
        # Step 1: η-canonicalize (GNN aggregation)
        canonical_features = egnn_encode(molecule)
        
        # Step 2: Type inhabitation score
        binding_score = self.binding_head(canonical_features)
        
        # Step 3: Convert to pIC50
        return 9.0 - binding_score  # higher pIC50 = lower binding_score
```

---

## 11. 完整的理论框架

### 11.1 语法 (Syntax)

```
Atoms (Combinators):
  A ::= S | K | I | Pt_II | Ru_II | C | N | O | H | ...     (primitive)

Molecules (λ-terms):
  M ::= A | M M | λx.M                                      (currying)

Reactions (β-reduction rules):
  R ::= (λx.M) N → M[x:=N]                                  (β-reduction)
    | M →_α M'                                              (α-conversion, structural isomerism)
    | M →_η M'                                              (η-conversion, functional equivalence)

Types (ADMET / binding):
  T ::= QED | Lipinski | Veber | Binding_MMP2 | ...
```

### 11.2 类型规则

```
Well-typed atom: Γ ⊢ A : Arity(A)
β-reduction:    Γ, x:T ⊢ M : σ    Γ ⊢ N : τ    T = τ
                ─────────────────────────────────────
                          Γ ⊢ (λx:τ.M) N : σ

ADMET well-formed:
  Γ ⊢ M : QED        Γ ⊢ M : Lipinski     ...
  ─────────────────────────────────────────
       Γ ⊢ M : DrugCandidate

Binding inhabitation:
  Γ ⊢ M : σ     σ ≤ BindingType(T)
  ──────────────────────────────────
       Γ ⊢ M : "binds T"
```

### 11.3 Synthesis = Proof Search

> **Curry-Howard 同构 (扩展版)**:
> 在化学里, "可合成的药物" = "满足所有 ADMET 类型 AND 能 inhabit binding 类型 的项"

---

## 12. 总结:为什么这个深度足够 SCI 一区

1. **形式化基础** (Foundational): 第一次把化学定义为 λ-演算,把药物设计定义为构造性证明搜索。
2. **统一框架** (Unification): 12_flow_matching 的 FM + EGNN 和 13_lambda_clickchem 的 click + MCTS 是**同一 λ-演算的不同算子** (β-reduction vs η-conversion)。
3. **可解释性免费** (Free interpretability): 每个生成的分子 = 一个证明 term,可以直接展示推导过程。
4. **可合成性免费** (Free synthesizability): 每个证明 = 一个 β-reduction sequence = 合成路径。
5. **新范式** (New paradigm): "Algebraic Drug Design" —— 把整个药物设计重写为类型论 + 项重写。

**论文题目候选**:
- "Molecular Lambda Calculus: A Computational Foundation for Interpretable, Synthesizable Drug Design"
- "Drug Design as Proof Search: A Type-Theoretic Framework for De Novo Molecular Generation"
- "Algebraic Drug Design via Type Theory and Lambda-Calculus-Driven Click Chemistry"

---

## 立即可做 (修订版)

```bash
# 1. 重新组织 molmetal_lam/ 结构, 把 Lambda 上推到上游
mkdir -p molmetal/molmetal_lam/{atoms,bonds,molecules,reactions,types,binding,synthesis}

# 2. 第一版: atoms-as-combinators
cat > molmetal/molmetal_lam/atoms/combinators.py << 'EOF'
"""Atoms as primitive combinators of the Molecular Lambda Calculus."""
class Atom:
    def __init__(self, symbol, atomic_num, valence, lone_pairs=0, geometry=""):
        self.symbol = symbol
        self.arity = valence + lone_pairs  # how many bonds it accepts
        ...
EOF

# 3. 第一版: bonds-as-application
cat > molmetal/molmetal_lam/bonds/application.py << 'EOF'
"""A chemical bond IS a function application (= β-reduction step)."""
def bond(atom_a, atom_b, bond_type='single'):
    """β-reduce atom_a(atom_b) -> new term."""
    ...
EOF

# 4. 第一版: molecule = closed λ-term in β-NF
cat > molmetal/molmetal_lam/molecules/closed_term.py << 'EOF'
"""A molecule IS a closed λ-term in β-normal form."""
class Molecule:
    def is_closed(self): ...   # all atoms saturated
    def is_beta_nf(self): ...  # no more reducible applications
    def reduce_once(self): ... # one β-reduction = one reaction
EOF
```

## 待你回答 (深化后的 3 个问题)

| # | 问题 | 选项 |
|---|---|---|
| 1 | **第一步聚焦** (atoms-as-combinators) | (a) 8 个标准原子 (H/C/N/O/F/P/S/Cl) + 5 金属 (Pt/Ru/Ir/Zn/Cu) 第一批 (b) 全部周期表 (c) 只 H/C/N/O 第一批 |
| 2 | **类型系统优先级** | (a) **Lipinski + Veber + QED** 第一批 (b) ADMET 全套 (c) Lipinski + Binding 各 1 个 |
| 3 | **第一个合成证明** | (a) **cisplatin** 作为 Pt(II) square-planar + 4 个 NH3/Cl 配体的构造证明 (b) imatinib (kinase 经典) (c) 任何 5-原子简单分子 |
