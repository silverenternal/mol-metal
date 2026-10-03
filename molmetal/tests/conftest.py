"""Pytest conftest for the Wf-235 wire-in tests.

Pre-loads the local ``_reference`` packages (the ODESolver shim) into
``sys.modules`` so that ``molmetal.adapters.flow_matching_lipman`` can
be imported even when the upstream Facebook ``flow_matching`` clone is
unavailable in the test environment.

This conftest is a pure environment shim — it does not modify any
production behaviour.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _preload_init(init_path: Path, mod_name: str, search_locations: list) -> None:
    """Pre-load a package init file into ``sys.modules`` with explicit path.

    Returns early if the package is already in ``sys.modules`` or the
    file is missing; otherwise it loads ``init_path`` as ``mod_name`` and
    stores it under ``sys.modules[mod_name]`` so subsequent Python
    imports find the package without touching the filesystem.
    """
    if not init_path.is_file():
        return
    if mod_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        mod_name, str(init_path),
        submodule_search_locations=search_locations,
    )
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)


def _ensure_reference_packages() -> None:
    """Pre-load the local ``_reference`` shim packages into ``sys.modules``.

    The trick: we monkey-patch ``_reference_loader.configure_reference``
    so that the ``_reference/__init__.py`` call doesn't trip the
    upstream Facebook-clone auto-detect in environments where the
    clone is missing.  We don't actually need the upstream package —
    the local shim packages (path/, solver/, utils/) handle everything
    locally.

    No-op when the upstream Facebook flow_matching clone is already
    available.
    """
    base = PROJECT_ROOT / "molmetal" / "adapters" / "flow_matching_lipman"

    loader_name = "molmetal.adapters.flow_matching_lipman._reference_loader"
    loader_path = base / "_reference_loader.py"
    if loader_path.is_file() and loader_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(loader_name, str(loader_path))
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[loader_name] = mod
            spec.loader.exec_module(mod)
            # Capture the real function BEFORE replacing it: the patched
            # version calls back into it, and reading it off `mod` afterwards
            # would recurse into the patch itself.
            _real_configure_reference = mod.configure_reference
            # Monkey-patch configure_reference so the auto-detect check is
            # bypassed in environments without the Facebook clone.  We
            # return a synthetic root pointing at our local _reference
            # package — the downstream code path doesn't actually import
            # from the upstream clone (the ODESolver shim handles
            # everything locally).
            def _patched_configure(reference_path=None):
                # Wf-415-A3 fix: also set the module's _reference_root global
                # so the `_reference_root is None` guard in
                # `_reference/path/__init__.py` and `_reference/solver/__init__.py`
                # is bypassed. Previously the conftest only REPLACED
                # `configure_reference` with a return-only stub, leaving
                # `_reference_root` unset and triggering ImportError on
                # any test that imports the ODESolver shim.
                # Point `_reference_root` at the shim directory
                # `<base>/_reference/` where the `path`/`solver`/`utils` shim
                # subpackages live (so `_ReferenceFinder.find_spec` can resolve
                # `_reference.path` → `<base>/_reference/path/__init__.py`).
                # The auto-bootstrap code inside `_reference/path/__init__.py`
                # separately walks `<base>/_reference/flow_matching/` (which
                # is a symlink to `<repo>/molmetal/references/flow_matching/
                # flow_matching/`) to locate the upstream modules.
                # ALSO register the `_ReferenceFinder` in `sys.meta_path` —
                # the original `configure_reference` does this on first call,
                # but since we replace the function with a stub the Finder
                # never gets registered, so `_reference.path` etc. fail to
                # resolve via the standard import machinery.
                # 2026-10-04: this stub used to ALWAYS return and never
                # raise, which silently disabled a production guard --
                # `configure_reference` raises RuntimeError when asked to
                # switch the reference clone mid-process, and
                # test_reference_location_change_rejected_without_changing_public_imports
                # exists precisely to check that it does.  That test failed
                # with this conftest and PASSED under --noconftest, i.e. the
                # harness, not the code, was the thing under test.
                #
                # Try the real function first; `base / "_reference"` contains
                # a `flow_matching` symlink to the upstream clone, so the real
                # path resolves and the guard stays live.  Only fall back to
                # the stub when the clone genuinely is not there, which is the
                # single case this shim was written for.
                try:
                    return _real_configure_reference(reference_path)
                except FileNotFoundError:
                    pass
                mod._reference_root = base / "_reference"
                if mod._finder not in sys.meta_path:
                    sys.meta_path.insert(0, mod._finder)
                return base / "_reference"
            mod.configure_reference = _patched_configure

    # Pre-load _reference, _reference/path, _reference/path/scheduler,
    # _reference/solver, _reference/utils.  The path shim is the one that
    # fails without this preload because the upstream Facebook clone
    # is not available in this conftest's test environment.
    _preload_init(
        base / "_reference" / "__init__.py",
        "molmetal.adapters.flow_matching_lipman._reference",
        [str(base / "_reference")],
    )
    _preload_init(
        base / "_reference" / "path" / "__init__.py",
        "molmetal.adapters.flow_matching_lipman._reference.path",
        [str(base / "_reference" / "path")],
    )
    _preload_init(
        base / "_reference" / "path" / "scheduler" / "__init__.py",
        "molmetal.adapters.flow_matching_lipman._reference.path.scheduler",
        [str(base / "_reference" / "path" / "scheduler")],
    )
    _preload_init(
        base / "_reference" / "solver" / "__init__.py",
        "molmetal.adapters.flow_matching_lipman._reference.solver",
        [str(base / "_reference" / "solver")],
    )
    _preload_init(
        base / "_reference" / "utils" / "__init__.py",
        "molmetal.adapters.flow_matching_lipman._reference.utils",
        [str(base / "_reference" / "utils")],
    )


def _patch_triton_autotuner_for_constexpr_keys() -> None:
    """Monkey-patch :class:`triton.runtime.autotuner.Autotuner` so that
    ``key=`` / ``reset_to_zero=`` entries that reference ``tl.constexpr``
    parameters (which are stripped from ``arg_names`` at autotuner
    construction time in Triton >=2.2) are silently dropped instead of
    raising ``ValueError``.

    Without this patch, importing any module that pulls in
    :mod:`triton_kernels.fused_softmax` (which uses ``key=["N"]`` on a
    constexpr param) crashes the test collection with a stack trace
    pointing at ``autotuner.py:39``.  The patch is a no-op when Triton
    is unavailable (i.e. never imported) and when the keys happen to
    already be valid ``arg_names``.

    Added per Wf-328-W7 — keeps the new ``test_wf328w7_dmpnn_pyg.py``
    module importable without fixing the upstream Triton issue (out of
    scope for this Wf).
    """
    try:
        import triton  # noqa: F401
        import triton.runtime.autotuner as _at_mod
    except ImportError:
        return

    if getattr(_at_mod.Autotuner.__init__, "_wf328w7_patched", False):
        return
    _orig = _at_mod.Autotuner.__init__
    import inspect as _inspect
    _orig_params = set(_inspect.signature(_orig).parameters.keys())

    def _patched_init(self, *args, **kwargs):
        """Wf-327-F6 / Wf-328-W7: filter ``key``/``reset_to_zero`` entries
        that reference ``tl.constexpr`` params (stripped from ``arg_names``
        at autotuner construction time in Triton >=2.2). Adapts to both
        the legacy 6-positional signature and Triton's full modern kwargs
        (``pre_hook``, ``post_hook``, ``use_cuda_graph``, etc.) by
        dropping any kwarg the installed Triton version does not accept.
        """
        # positional layout in every supported Triton version:
        #   (fn, arg_names, configs, key, reset_to_zero, prune_configs_by, warmup, rep)
        if len(args) >= 5:
            fn, arg_names, configs, key, reset_to_zero = args[:5]
            if key is not None:
                key = [k for k in key if k in arg_names]
            if reset_to_zero is not None:
                reset_to_zero = [k for k in reset_to_zero if k in arg_names]
            args = (fn, arg_names, configs, key, reset_to_zero) + args[5:]
        kwargs = {k: v for k, v in kwargs.items() if k in _orig_params}
        return _orig(self, *args, **kwargs)

    _patched_init._wf328w7_patched = True
    _at_mod.Autotuner.__init__ = _patched_init


def _force_torch_scatter_backend_for_cpu_tests() -> None:
    """Force the scatter backend to ``'torch'`` when running on CPU-only
    test environments (the Triton aggregate kernel cannot accept CPU
    pointers).  Idempotent and a no-op on GPU hosts.

    The legacy ``molmetal.models.dmpnn.DirectedMPNN.forward`` loop
    calls :func:`scatter_sum_legacy` per-molecule; on CPU we need to
    fall back to the torch path.  Doing it here in the conftest keeps
    the test module itself free of env-conditional scaffolding.

    Added per Wf-328-W7.
    """
    import torch

    if torch.cuda.is_available():
        return
    try:
        from models._scatter import set_scatter_backend  # type: ignore
    except Exception:
        return
    set_scatter_backend("torch")


_patch_triton_autotuner_for_constexpr_keys()
_force_torch_scatter_backend_for_cpu_tests()
_ensure_reference_packages()