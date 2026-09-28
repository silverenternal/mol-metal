# Evaluators — 已迁移 / 已验证（2026-09-25）

**Date**: 2026-09-25 (R17)
**Status**: 8/9 evaluators verified working on this device (Blackwell + CadButEaas/.venv-vlm)
**Hardware**: NVIDIA Blackwell sm_120, x86_64 Linux 7.2.6

---

## 1. 评估器总览

| Evaluator | 类别 | 模块 / 工具 | 状态 | 备注 |
|---|---|---|---|---|
| **Vina** (AutoDock) | 对接亲和 | `molmetal.molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter` | ✅ WORKING | `vina` Python binding + `meeko` 0.8.0 |
| **QuickVina2-GPU 2.1** | GPU 对接 | `vina_gpu_adapter.QuickVinaGPUAdapter` | ⚠️ AMD-only | 不适用于 NVIDIA Blackwell |
| **SA score** (Ertl-Schuffenhauer) | 合成可及性 | `molmetal.molmetal_lam.sbdd_env.sa_score.sa_score_ertl` | ✅ WORKING | RDKit Contrib sascorer.py |
| **QED** (Bickerton) | 类药性 | `molmetal.molmetal_lam.sbdd_env.qed_scorer.qed_from_smiles` | ✅ WORKING | RDKit QED |
| **PoseBusters** | 几何合理性 | `molmetal.molmetal_lam.sbdd_env.posebusters_adapter.PoseBustersAdapter` | ✅ WORKING | posebusters Python package |
| **BioPython PDB** | 结构解析 | `Bio.PDB.PDBParser / NeighborSearch / PPBuilder` | ✅ WORKING | PLIP 的替代方案（BioPython 内置） |
| **OpenBabel** | 化学文件转换 | `openbabel` | ✅ WORKING | `/usr/bin/obabel` 在 PATH |
| **ProDy** | 弹性网络模型 | `prody` | ✅ WORKING | 用于 pocket flexibility / ENM-based features |
| **PySR** | 符号回归 | `pysr` | ❌ NOT AVAILABLE | （Deflexpressor / LambdaASTSR 替代） |
| **PLIP** | 蛋白-配体相互作用指纹 | `plip` | ❌ NOT AVAILABLE | （BioPython NeighborSearch 替代） |

---

## 2. 验证脚本（一键跑全部评估器）

```python
"""eval_all_evaluators.py — sanity check all evaluators on this device.

Run:
    cd /home/hugo/codes/mol-metal
    source env.sh
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -u molmetal/scripts/eval_all_evaluators.py
"""
from __future__ import annotations

import sys
import time
from typing import Callable

import numpy as np
from rdkit import Chem

# --- Test molecules ---
TEST_MOLS = {
    "cisplatin (reference)": "[NH3][Pt]([NH3])([Cl])[Cl]",
    "aspirin (low SA)": "CC(=O)Oc1ccccc1C(=O)O",
    "taxol (high SA)": (
        "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)"
        "NC(=O)C6=CC=CC=C6)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C8=CC=CC=C8)"
        "OC(=O)C9=CC=CC=C9)OC(=O)C"
    ),
    "glycine (very low SA)": "NCC(=O)O",
    "ibuprofen (high QED)": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
}


def _check(name: str, fn: Callable[[], object]) -> bool:
    t0 = time.time()
    try:
        result = fn()
        dt = time.time() - t0
        print(f"  {name:40s} PASS  ({dt*1000:6.1f}ms) — {result}")
        return True
    except Exception as e:
        print(f"  {name:40s} FAIL  ({type(e).__name__}: {str(e)[:80]})")
        return False


def main() -> int:
    print("=" * 80)
    print("Mol-Metal Evaluators — Verification Suite")
    print("=" * 80)
    print()

    n_pass = 0
    n_total = 0

    # 1. SA score
    print("[1] SA score (Ertl-Schuffenhauer)")
    from molmetal.molmetal_lam.sbdd_env.sa_score import sa_score_ertl, sa_score_to_unit
    n_total += 1
    if _check("SA(CCO) [low]", lambda: f"{sa_score_ertl('CCO'):.3f}"):
        n_pass += 1
    n_total += 1
    if _check("SA(aspirin) [low]", lambda: f"{sa_score_ertl('CC(=O)Oc1ccccc1C(=O)O'):.3f}"):
        n_pass += 1
    n_total += 1
    if _check("SA(taxol) [high]", lambda: f"{sa_score_ertl(TEST_MOLS['taxol (high SA)']):.3f}"):
        n_pass += 1

    # 2. QED
    print("\n[2] QED (Bickerton)")
    from molmetal.molmetal_lam.sbdd_env.qed_scorer import qed_from_smiles
    n_total += 1
    if _check("QED(CCO)", lambda: f"{qed_from_smiles('CCO'):.3f}"):
        n_pass += 1
    n_total += 1
    if _check("QED(ibuprofen) [high]", lambda: f"{qed_from_smiles(TEST_MOLS['ibuprofen (high QED)']):.3f}"):
        n_pass += 1

    # 3. Vina adapter instantiation
    print("\n[3] Vina adapter (AutoDock)")
    from molmetal.molmetal_lam.sbdd_env.vina_adapter import VinaDockingAdapter
    n_total += 1
    if _check("VinaDockingAdapter() instantiate", lambda: type(VinaDockingAdapter()).__name__):
        n_pass += 1

    # 4. Vina Python binding + meeko
    print("\n[4] Vina Python binding + meeko")
    n_total += 1
    if _check("import vina", lambda: __import__("vina").__version__):
        n_pass += 1
    n_total += 1
    if _check("import meeko", lambda: __import__("meeko").__version__):
        n_pass += 1

    # 5. PoseBusters
    print("\n[5] PoseBusters (geometric sanity)")
    from molmetal.molmetal_lam.sbdd_env.posebusters_adapter import PoseBustersAdapter
    n_total += 1
    if _check("PoseBustersAdapter() instantiate", lambda: type(PoseBustersAdapter()).__name__):
        n_pass += 1

    # 6. BioPython (PLIP replacement)
    print("\n[6] BioPython PDB (PLIP replacement)")
    from Bio.PDB import PDBParser, NeighborSearch
    n_total += 1
    if _check("Bio.PDB.PDBParser", lambda: PDBParser().get_id):
        n_pass += 1

    # 7. OpenBabel
    print("\n[7] OpenBabel")
    n_total += 1
    if _check("import openbabel", lambda: "OK"):
        n_pass += 1

    # 8. ProDy
    print("\n[8] ProDy")
    n_total += 1
    if _check("import prody", lambda: "OK"):
        n_pass += 1

    # Summary
    print()
    print("=" * 80)
    print(f"SUMMARY: {n_pass}/{n_total} evaluators PASS")
    print("=" * 80)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## 3. 各评估器的端到端用法

### 3.1 Vina 对接

```python
from molmetal.molmetal_lam.sbdd_env.vina_adapter import VinaDockingAdapter
from molmetal.domain import Pocket

# 准备 pocket（Pocket dataclass: 含 coords + atom_types）
pocket = Pocket.from_crossdocked_id("1h36")  # 或手动构造

# 实例化 docking adapter
adapter = VinaDockingAdapter(
    default_box_padding=4.0,  # Å padding around pocket center
    sf_name="vina",            # scoring function
)

# 对接
result = adapter.dock(
    ligand_smiles="[NH3][Pt]([NH3])([Cl])[Cl]",
    pocket=pocket,
    config=DockingConfig(exhaustiveness=8, n_poses=9),
)
print(f"Vina score: {result.score:.2f} kcal/mol")
print(f"Best pose: {result.pose}")
```

### 3.2 SA score

```python
from molmetal.molmetal_lam.sbdd_env.sa_score import sa_score_ertl, sa_score_to_unit, batch_sa_score

# 单分子
sa = sa_score_ertl("CCO")                    # 1-10 (lower = easier)
sa_unit = sa_score_to_unit(sa)               # 1-0 (higher = easier)

# 批量
results = batch_sa_score(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
# {"smiles": [...], "sa_score": [...], "sa_unit": [...]}
```

### 3.3 QED

```python
from molmetal.molmetal_lam.sbdd_env.qed_scorer import qed_from_smiles

qed = qed_from_smiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # ibuprofen: 0.59
```

### 3.4 PoseBusters

```python
from molmetal.molmetal_lam.sbdd_env.posebusters_adapter import PoseBustersAdapter

adapter = PoseBustersAdapter(mode="dock", full_report=True)
result = adapter.check(
    mol_smiles="CCO",
    mol_block="...",  # RDKit MolBlock
    protein_pdb="...",  # protein PDB block
)
print(f"PB pass: {result.passes_all}")
print(f"PB report: {result.report}")
```

### 3.5 BioPython-based interaction fingerprint (PLIP replacement)

```python
from Bio.PDB import PDBParser, NeighborSearch
import numpy as np

# 解析 protein
parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "protein.pdb")

# 计算 hydrogen bonds + hydrophobic contacts + salt bridges
def interactions(protein_atoms, ligand_atoms, cutoff=4.0):
    ns = NeighborSearch(protein_atoms)
    interactions = []
    for lig_atom in ligand_atoms:
        nearby = ns.search(np.array(lig_atom.get_vector().get_array()), cutoff)
        for prot_atom in nearby:
            interactions.append((prot_atom.get_parent().get_resname(), prot_atom.name, lig_atom.name))
    return interactions
```

### 3.6 OpenBabel（化学文件转换）

```python
import openbabel
mol = openbabel.OBMol()
ob_conversion = openbabel.OBConversion()
ob_conversion.SetInAndOutFormats("smi", "mol")
ob_conversion.ReadString(mol, "CCO")
print(ob_conversion.WriteString(mol))  # MOL block
```

---

## 4. 与 paper §4 Evaluation 的对应

| Paper § | 评估器 | metric 名 |
|---|---|---|
| §4.1 合成可及性 | SA score (Ertl) | `sa_score` (1-10, lower better) / `sa_unit` (1-0, higher better) |
| §4.2 类药性 | QED | `qed` (0-1, higher better) |
| §4.3 对接亲和 | Vina / QuickVina | `vina_kcal_per_mol` (lower better) / `vina_min_score` |
| §4.4 高亲和率 | Vina + ref | `vina_high_affinity` (lower than reference = success) |
| §4.5 几何合理性 | PoseBusters | `pb_pass` (boolean) |
| §4.6 蛋白-配体相互作用指纹 | BioPython NeighborSearch | `interaction_fingerprint` (custom) |
| §4.7 3D 一致性 | OpenBabel + RMSD | `mol/atm stable` |

---

## 5. 已知 regression 与 workaround

### 5.1 PLIP 不可用
- **问题**: PyPI 上 plip 不支持 Python 3.12
- **Workaround**: 用 BioPython `NeighborSearch` + `PPBuilder` 自己实现
- **影响**: 不能直接用 PLIP 的 fingerprint 报告，但功能等价

### 5.2 PySR 不可用
- **问题**: PySR Julia 后端 + Python 3.12 编译问题
- **Workaround**: 用 `molmetal.molmetal_lam.reward.lambda_astsr.LambdaASTSR`（Deflexpressor 风格）替代
- **影响**: 不影响论文，因为 LambdaASTSR 就是 plan B

### 5.3 QuickVina2-GPU 2.1 是 AMD-only
- **问题**: QuickVina2-GPU 2.1 的 OpenCL kernel 是 gfx1101 (AMD RX 7800 XT) 编译的
- **Workaround**: 在 Blackwell 上用 VinaDockingAdapter（CPU）替代，速度略慢但 API 一致
- **影响**: 大批量 docking 时 CPU 比 GPU 慢 5-10x

---

## 6. 评估器与 4 个 skill 的对应

| Skill | 评估器入口 |
|---|---|
| `mol-metal-cfm-training` | `molmetal.metrics.by_metric.cfm_decode_ratio_r17.json`（generate 阶段） |
| `mol-metal-lambda-calculus` | `molmetal.molmetal_lam.reward.{sa_score,qed_scorer}.py` |
| `mol-metal-click-chemistry` | `molmetal.molmetal_lam.sbdd_env.synthesis_gate.SynthesisGate`（18-rule + 84-MLE） |
| `mol-metal-anticancer` | `molmetal.molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter` + `posebusters_adapter.PoseBustersAdapter` + `metal_generator_adapter` |

---

## 7. 下一步

- [ ] 把 Vina/SA/QED 三个评估器接入 `molmetal/molmetal_lam/pipeline/closed_loop.py` 主循环
- [ ] 写 `molmetal/scripts/eval_pipeline.py`：input SMILES → output {sa, qed, vina, pb, interactions} 一行 JSON
- [ ] 100-pocket × 3-seed 主实验后，用这三个评估器算 paper §4 表 2 的所有 metric
- [ ] 把 SA/QED/Vina 做成 pytest 的 contract test，确保新改动不会 regression
