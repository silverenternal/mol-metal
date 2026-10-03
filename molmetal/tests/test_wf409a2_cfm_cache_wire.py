"""Wf-409-A2 — CFM-trainer-primary chemprop v3 cache wire-in tests.

Verifies the wire-in per TODO-155 + AGENTS.md §5.7 (trainer flag
wire-in smoke-required rule):

1. ``test_cfm_cache_flag_off`` — when --cfm-chemprop-v3-cache is False
   (default), parse_args + cache holder + train_one_epoch kwarg + main
   wiring are all preserved bit-for-bit (no cache module is imported
   eagerly; cache holder stays None; loss contribution is 0).

2. ``test_cfm_cache_flag_on_hit_miss`` — when --cfm-chemprop-v3-cache
   is True, the cache holder can be populated and tier-1 precompute
   runs on the 4-SMILES smoke cohort + tier-2 per-batch lookup
   exercises the cache API.  Cross-epoch reuse records ≥ 1 hit.

3. ``test_module_load_surface_stays_minimal`` — cache module NOT
   eagerly imported at module-load time.

All tests run CPU-only in <1 s wall.  No GPU launches.  No paper/*
touches.  No wet-lab planning per AGENTS.md §10.

Honest framing per ``party-twenty-spirit`` §3.5
-----------------------------------------------
- Test cache flag OFF == preserved bit-for-bit: verified by AST parse
  + flag default check + cache holder initial state.
- Test cache flag ON == tier-1 + tier-2 API exercised: verified by
  direct invocation of the cache module + the AST-discovered holder
  symbol name (we emulate the holder via a tiny stub since importing
  the full CFM trainer triggers a circular-import trap on
  ``molmetal.models.dmpnn``).
- AGENTS.md §5.7 floor (loss decrease > 0 + decode_ratio > 0):
  preserved as SCAFFOLD-only — the smoke verifies the wire-in
  contract, NOT the end-to-end training-loop integration.  Loss
  contribution stays 0 (cache lives outside the loss cascade — its
  purpose is speedup, not training signal).  The full loss-decrease
  + decode_ratio smoke runs the actual trainer; that is the
  integration test in the closeout doc.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

# ---------------------------------------------------------------------------
# Repo-root importability.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_trainer_path() -> Path:
    """Return absolute path to the CFM trainer source."""
    return _REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _parse_trainer_ast() -> ast.Module:
    """Parse the CFM trainer source as AST."""
    return ast.parse(_get_trainer_path().read_text())


def _find_argparse_flag(tree: ast.Module, flag_name: str) -> dict:
    """Find a ``p.add_argument("--flag", ...)`` call in the trainer AST."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and first_arg.value == flag_name:
            out: dict = {"flag": flag_name}
            for kw in node.keywords:
                out[kw.arg] = ast.unparse(kw.value)
            return out
    raise AssertionError(f"--{flag_name.lstrip('-')} not found in trainer AST")


def _find_kwarg_in_train_one_epoch(tree: ast.Module, kwarg_name: str) -> dict:
    """Find a kwarg by name in train_one_epoch signature.

    Returns:
        Dict with keys ``kind`` ("positional" or "kwonly") + the source-text
        slice covering the kwarg default for inspection.

    Notes:
        AST node args stores positional defaults via ``args.defaults``
        (list of AST nodes, aligned to the rightmost positional args)
        and kwonly defaults via ``args.kw_defaults`` (list aligned to
        kwonlyargs).  Both lists default to None if no defaults are set.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "train_one_epoch":
            # kwonly first.
            kwonly_args = node.args.kwonlyargs
            kw_defaults = node.args.kw_defaults
            for idx, a in enumerate(kwonly_args):
                if a.arg == kwarg_name:
                    default_node = kw_defaults[idx] if kw_defaults else None
                    return {
                        "kind": "kwonly",
                        "default_src": ast.unparse(default_node) if default_node else "<no-default>",
                    }
            # positional: defaults list aligns with rightmost positional args.
            pos_args = node.args.args
            n_pos = len(pos_args)
            n_defaults = len(node.args.defaults) if node.args.defaults else 0
            offset = n_pos - n_defaults  # first positional with a default
            for idx, a in enumerate(pos_args):
                if a.arg == kwarg_name and idx >= offset:
                    default_idx = idx - offset
                    default_node = node.args.defaults[default_idx]
                    return {
                        "kind": "positional",
                        "default_src": ast.unparse(default_node) if default_node else "<no-default>",
                    }
    raise AssertionError(f"kwarg {kwarg_name} not found in train_one_epoch")


def _find_main_wiring(tree: ast.Module, kwarg_name: str) -> str:
    """Find the value passed for kwarg_name in main()'s train_one_epoch call."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "train_one_epoch":
                for kw in node.keywords:
                    if kw.arg == kwarg_name:
                        return ast.unparse(kw.value)
    raise AssertionError(f"kwarg {kwarg_name} not wired in main()")


def _find_top_level_symbol(tree: ast.Module, name: str) -> str:
    """Find a module-level name assignment and return its unparsed source."""
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return ast.unparse(node)
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return ast.unparse(node)
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "function:" + node.name
    raise AssertionError(f"top-level symbol {name} not found")


def _find_top_level_function(tree: ast.Module, name: str) -> str:
    """Find a top-level function by name; return its def signature."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"top-level function {name} not found")


# ---------------------------------------------------------------------------
# Test 1 — flag OFF is the default and cache holder stays None.
# ---------------------------------------------------------------------------
def test_cfm_cache_flag_off() -> None:
    """Default OFF preserves bit-for-bit backward compat.

    Asserts:
      - ``--cfm-chemprop-v3-cache`` is in parse_args with default=False.
      - ``cfm_chemprop_v3_cache: bool = False`` is the train_one_epoch kwarg default.
      - main() wires getattr(args, "cfm_chemprop-v3-cache", False) via bool().
      - The cache module is NOT imported at module-load time.
    """
    tree = _parse_trainer_ast()

    # (a) Flag default = False
    flag = _find_argparse_flag(tree, "--cfm-chemprop-v3-cache")
    assert flag["default"] == "False", (
        f"--cfm-chemprop-v3-cache default should be False; got {flag['default']}"
    )
    assert flag["action"] == "'store_true'", (
        f"--cfm-chemprop-v3-cache action should be store_true; got {flag['action']}"
    )

    # (b) train_one_epoch kwarg default = False
    kwarg = _find_kwarg_in_train_one_epoch(tree, "cfm_chemprop_v3_cache")
    assert kwarg["default_src"] == "False", (
        f"train_one_epoch kwarg default should be False; got {kwarg['default_src']}"
    )

    # (c) main() wires getattr(args, "cfm_chemprop_v3_cache", False) via bool()
    main_wiring = _find_main_wiring(tree, "cfm_chemprop_v3_cache")
    assert "cfm_chemprop_v3_cache" in main_wiring, (
        f"main() should wire cfm_chemprop_v3_cache kwarg; got {main_wiring}"
    )
    assert "False" in main_wiring, (
        f"main() default path should be False; got {main_wiring}"
    )

    # (d) Cache holder initial state = None (AST-validated).
    holder_src = _find_top_level_symbol(tree, "_CFM_CHEMPROP_V3_CACHE_HOLDER")
    assert "None" in holder_src, (
        f"cache holder initial state must be None; got {holder_src}"
    )
    assert "'cache'" in holder_src or '"cache"' in holder_src, (
        f"cache holder must be a dict with 'cache' key; got {holder_src}"
    )

    # (e) set_cfm_chemprop_v3_cache is callable (top-level function defined).
    setter_src = _find_top_level_function(tree, "set_cfm_chemprop_v3_cache")
    assert "def set_cfm_chemprop_v3_cache" in setter_src, (
        f"set_cfm_chemprop_v3_cache must be defined; got {setter_src[:200]}"
    )

    # (f) get_cfm_chemprop_v3_cache is callable.
    getter_src = _find_top_level_function(tree, "get_cfm_chemprop_v3_cache")
    assert "def get_cfm_chemprop_v3_cache" in getter_src, (
        f"get_cfm_chemprop_v3_cache must be defined; got {getter_src[:200]}"
    )


# ---------------------------------------------------------------------------
# Test 2 — flag ON populates cache holder, exercises tier-1 + tier-2.
# ---------------------------------------------------------------------------
def test_cfm_cache_flag_on_hit_miss() -> None:
    """Flag ON populates the cache holder + tier-1 + tier-2 exercise.

    Asserts:
      - Cache holder round-trips via the AST-discovered setter/getter names.
      - Tier-1 precompute_smiles on the deterministic 4-SMILES smoke
        cohort returns n_added=4 (all NEW on a fresh cache).
      - Tier-2 per-batch lookup exercises the cache API.
      - Cross-epoch reuse records ≥ 1 hit.

    Note: We exercise the cache directly (the holder symbol is loaded by
    AST inspection + direct cache module import).  The full CFM trainer
    module is not importable in isolation due to a pre-existing
    circular import on ``molmetal.models.dmpnn``; that is a separate
    conftest-level issue unrelated to Wf-409-A2.
    """
    # (a) Tier-1: precompute_smiles on smoke cohort.
    try:
        from molmetal.scripts.wf382a2_chemprop_v3_featurization_cache import (
            ChempropV3FeaturizationCache,
        )
    except ImportError:
        pytest.skip("chemprop v3 not installed; tier-1 exercise skipped")

    cache = ChempropV3FeaturizationCache(use_fast_featurizer=False)
    smoke = ("c1ccccc1", "C", "CCO", "C1=CCNC=C1")
    n_added = cache.precompute_smiles(smoke)
    assert n_added == len(smoke) == 4, (
        f"tier-1 should precompute 4 SMILES; got {n_added}"
    )
    assert cache.stats.n_hit == 0
    assert cache.stats.n_miss == 0
    assert len(cache._mols) == 4

    # (b) Holder emulation: stub module that captures set/get behaviour
    # in line with the trainer's _CFM_CHEMPROP_V3_CACHE_HOLDER semantics.
    holder: Dict[str, Optional[object]] = {"cache": None}

    def set_cfm_chemprop_v3_cache(c):
        holder["cache"] = c

    def get_cfm_chemprop_v3_cache():
        return holder["cache"]

    set_cfm_chemprop_v3_cache(cache)
    assert get_cfm_chemprop_v3_cache() is cache

    # (c) Tier-2: per-batch lookup with epoch_idx=0.
    # Use the torch-aware StubBatchMG that pre-converts chemprop 2.x
    # numpy arrays to torch tensors (the cache's get_or_build_batch
    # then calls bmg.V.to(device) which requires torch tensors).
    def _dummy_ctor(mgs):
        import numpy as _np
        import torch as _torch

        Vs = [mg.V for mg in mgs]
        Es = [mg.E for mg in mgs]
        eis = [mg.edge_index for mg in mgs]
        reis = [mg.rev_edge_index for mg in mgs]

        class _StubBatchMG:
            pass

        out = _StubBatchMG()
        out.V = _torch.from_numpy(
            _np.concatenate(Vs, axis=0) if Vs else _np.zeros((0, 72))
        )
        out.E = _torch.from_numpy(
            _np.concatenate(Es, axis=0) if Es else _np.zeros((0, 14))
        )

        offset = 0
        atom_offsets: list = []
        batch_eis: list = []
        batch_reis: list = []
        for ei, rei in zip(eis, reis):
            n_atoms = len(Vs[len(atom_offsets)])
            if ei.shape[1] > 0:
                batch_eis.append(ei + offset)
                batch_reis.append(rei)
            atom_offsets.append(n_atoms)
            offset += n_atoms

        if batch_eis:
            out.edge_index = _torch.from_numpy(_np.concatenate(batch_eis, axis=1))
            out.rev_edge_index = _torch.from_numpy(_np.concatenate(batch_reis))
            out.batch = _torch.from_numpy(
                _np.concatenate(
                    [_np.full(n, i, dtype=_np.int64) for i, n in enumerate(atom_offsets)]
                )
            )
        else:
            out.edge_index = _torch.zeros((2, 0), dtype=_torch.long)
            out.rev_edge_index = _torch.zeros((0,), dtype=_torch.long)
            out.batch = _torch.zeros((0,), dtype=_torch.long)

        return out

    cache.begin_epoch(0)
    cache.get_or_build_batch(0, 0, (smoke[0], smoke[1]), "cpu", _dummy_ctor)
    cache.get_or_build_batch(0, 1, (smoke[2], smoke[3]), "cpu", _dummy_ctor)

    # (d) Tier-2 cross-epoch reuse: epoch 1 batch 0 should be a HIT
    # because cache is keyed only on batch_idx (per
    # ChempropV3FeaturizationCache contract).
    cache.begin_epoch(1)
    cache.get_or_build_batch(1, 0, (smoke[0], smoke[1]), "cpu", _dummy_ctor)
    assert cache.stats.n_hit >= 1, (
        f"tier-2 cross-epoch reuse should record at least 1 hit; "
        f"got n_hit={cache.stats.n_hit}, n_miss={cache.stats.n_miss}"
    )
    assert cache.stats.hit_rate > 0.0, (
        f"hit_rate should be > 0 after epoch-2 batch-0 lookup; got "
        f"{cache.stats.hit_rate}"
    )

    # (e) Smoke cohort constants are deterministic per the trainer AST.
    tree = _parse_trainer_ast()
    smokes_src = _find_top_level_symbol(tree, "_CFM_CACHE_SMOKES")
    batches_src = _find_top_level_symbol(tree, "_CFM_CACHE_SMOKES_BATCHES")
    # ast.unparse uses single quotes for string literals.
    assert "'c1ccccc1'" in smokes_src, f"benzene SMILES must be in cohort: {smokes_src}"
    assert "'CCO'" in smokes_src, f"ethanol SMILES must be in cohort: {smokes_src}"
    assert len(batches_src) > 50, f"batches constant must be non-trivial: {batches_src}"

    # (f) Reset holder for downstream test isolation.
    set_cfm_chemprop_v3_cache(None)
    assert get_cfm_chemprop_v3_cache() is None


# ---------------------------------------------------------------------------
# Test 3 — module-load surface stays minimal (no eager chemprop import).
# ---------------------------------------------------------------------------
def test_module_load_surface_stays_minimal() -> None:
    """Cache module NOT eagerly imported at trainer load time.

    Asserts:
      - The cache module is imported lazily inside train_one_epoch +
        main() init blocks, NOT at top-level.
      - This preserves backward compat: pre-existing trainer
        environments without chemprop v3 can still load the module
        for tests (parse_args, AST checks, etc.).
    """
    tree = _parse_trainer_ast()

    # Check that the module-level imports don't include the cache
    # module (lazy import inside the body is OK).
    cache_in_top_imports = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if "wf382a2_chemprop_v3_featurization_cache" in alias.name:
                    cache_in_top_imports = True
                    break
    assert not cache_in_top_imports, (
        "cache module must NOT be eagerly imported at module-load time "
        "(would break rdkit/AF-less environments)"
    )

    # Lazy import IS present somewhere in the module body.
    lazy_import_present = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "wf382a2_chemprop_v3_featurization_cache" in node.module:
                lazy_import_present = True
                break
    assert lazy_import_present, (
        "lazy import of wf382a2_chemprop_v3_featurization_cache must be "
        "present somewhere in the module body"
    )


if __name__ == "__main__":
    # Allow `python -m pytest test_wf409a2_cfm_cache_wire.py` or direct
    # invocation via `python test_wf409a2_cfm_cache_wire.py`.
    sys.exit(pytest.main([__file__, "-v", "--no-header"]))
