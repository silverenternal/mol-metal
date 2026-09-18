# Clone-Repo Abstract-Layer Adapters

Status: implemented (Phase 0 stubs)
Date: 2026-09-11
Scope: four adapters wired into `molmetal/molmetal_lam/sbdd_env/`.

## Why stubs

The cloned repos under `molmetal/references/` (FlowDock, FLOWR,
REINVENT4) all depend on heavy optional stacks:

- `torch_cluster` + `torch_scatter` CUDA wheels (FlowDock, FLOWR)
- proprietary REINVENT4 plugins (QSAR, docking, shape)
- pybind11-built C++/CUDA kernels (EquiBind, DiffDock sphere sampler)

ROCm wheels for these rarely exist and rebuilding from source is out
of scope. Each adapter therefore exposes a **Protocol-shaped stub**:
duck-typed conformance with `molmetal.ports`, lazy discovery of the
cloned repo on disk, and graceful fallback when the heavy deps are
absent.

## Adapters

| Module | Class | Port | Reference | Returns |
|---|---|---|---|---|
| `flowdock_adapter.py` | `FlowDockReferenceAdapter` | `DockingEngine` | arXiv:2403.04777 (Morehead & Chen 2024) | `List[Complex]` (n_poses) |
| `flowr_adapter.py` | `FLOWRReferenceAdapter` | `MoleculeGenerator` | arXiv:2404.02819 (Alvira et al. 2024) | `List[Molecule]` (n_samples) |
| `pybind_adapter.py` | `Pybind11StubAdapter` | `PybindBindingProtocol` | pybind11 (Jakob et al. 2017, doi:10.5281/zenodo.1042867) | passthrough `Any` |
| `reinvent4_adapter.py` | `REINVENT4Adapter` | `ScoringFunction` | He et al. REINVENT4 (ChemRxiv 2024) + Olivecrona J. Cheminform. 2017 | `List[ScoredCandidate]` |

## Public surface per adapter

### FlowDockReferenceAdapter
- `name` (property) → `"FlowDock_reference_stub_v1"`
- `setup(device)` — probes repo path, sets `_loaded`
- `is_stub()` — `True` iff no real FlowDock import
- `dock(molecule, pocket, config: DockingConfig) -> List[Complex]`
- `get_metadata() -> dict` — includes arXiv id
- module-level: `is_flowdock_available(repo_path)`

### FLOWRReferenceAdapter
- `name`, `setup`, `is_stub`
- `generate(pocket, config: GenerationConfig) -> List[Molecule]`
- `train_step(pocket, mols) -> float` (returns 0.0)
- `get_metadata()` — includes arXiv id
- module-level: `is_flowr_available(repo_path)`

### Pybind11StubAdapter
- attrs: `name`, `version`, `module_path`, `is_loaded`, `call_log`
- `setup(device)`, `forward(*args, **kwargs) -> Any`
- `get_metadata() -> dict`
- module-level: `is_binding_loaded(path)`
- companion: `PybindBindingProtocol` (documents surface)

### REINVENT4Adapter
- `name`, `setup` (builds RDKit fallback scorer + lists TOMLs)
- `is_stub()` — always True
- `score(candidates) -> List[ScoredCandidate]` (sorted, ranked 0..N-1)
- `get_metadata()` — includes `toml_presets`
- module-level: `discover_reinvent4_tomls(repo_path)`, `is_reinvent4_available(repo_path)`

## Tests

File: `molmetal/tests/test_clone_integration_adapters.py` — 6 tests, all
passing in ~2s on CPU-only venv (no ROCm kernels needed):

```
test_flowdock_adapter_import_and_protocol       PASSED
test_flowr_adapter_import_and_protocol           PASSED
test_pybind_stub_import_and_protocol             PASSED
test_reinvent4_adapter_import_and_protocol       PASSED
test_all_adapters_export_expected_symbols        PASSED
test_adapters_instantiate_without_external_state PASSED
============================== 6 passed in 1.99s ===============================
```

Each test asserts:

1. module import + `__all__` symbol presence
2. duck-typed conformance with the relevant port (required attrs)
3. method invocation returns the documented shape (`Complex`,
   `Molecule`, `ScoredCandidate`)

## Out of scope

- Real FlowDock / FLOWR weights (not downloaded per policy).
- Real REINVENT4 RL loop (RDKit fallback only).
- pybind11 `.so` linking (stubs only).

Each adapter has a single, named, citable reference — no anonymous
plagiarised code paths.

### Plug-in adapters 2026-09-11

四个 adapter 以 Protocol stub 形式接入 `molmetal/molmetal_lam/sbdd_env/`,
duck-typed 遵从 `molmetal.ports`,运行时延迟探测 `references/` 下克隆仓,
重依赖缺失时优雅退化。

| Adapter | Port | 接入方式 | 当前状态 |
|---|---|---|---|
| FlowDockReferenceAdapter | `DockingEngine` | Protocol stub: `setup/is_stub/dock/get_metadata`,模块级 `is_flowdock_available(repo_path)`;返回 `List[Complex]`(n_poses) | implemented,stub;6/6 tests pass |
| FLOWRReferenceAdapter | `MoleculeGenerator` | Protocol stub: `generate/train_step/get_metadata`,模块级 `is_flowr_available(repo_path)`;返回 `List[Molecule]` | implemented,stub;6/6 tests pass |
| Pybind11StubAdapter | `PybindBindingProtocol` | Protocol stub: `forward(*args, **kwargs) -> Any`,`call_log` 记录调用;返回 passthrough | implemented,stub;6/6 tests pass |
| REINVENT4Adapter | `ScoringFunction` | Protocol stub: `score -> List[ScoredCandidate]`(ranked 0..N-1),`discover_reinvent4_tomls` 列 TOML 预设;RDKit fallback | implemented,stub;6/6 tests pass |

注:文件 `/home/hugo/codes/try_triton_on_rocm/TODO/12_flow_matching/integration_with_molmol.md`
不存在(仅有 `integration_with_molflow.md`),故本节直接追加至此 report。
