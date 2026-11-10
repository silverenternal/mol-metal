"""Pytest conftest for the Wf-235 wire-in tests.

Pre-loads the local ``_reference`` packages (the ODESolver shim) into
``sys.modules`` so that ``molmetal.adapters.flow_matching_lipman`` can
be imported even when the upstream Facebook ``flow_matching`` clone is
unavailable in the test environment.

This conftest is a pure environment shim — it does not modify any
production behaviour.

THE NO-GPU CONTRACT (Wf-A123 FIX 5)
----------------------------------
At MODULE IMPORT TIME this conftest blanks ``CUDA_VISIBLE_DEVICES`` unless
the caller opted in with ``MOLMETAL_TEST_ALLOW_GPU=1``.  The standing user
directive for this repository is 禁止使用gpu: a test run must NOT silently
take VRAM.  MEASURED 2026-10-31: a full ``pytest molmetal/tests`` run held
1444 MiB (the pytest process itself — some test allocates CUDA IN-PROCESS)
plus 2218 MiB (a smoke subprocess) of GPU 0 VRAM, in breach of that
directive.

WHY IMPORT TIME AND NOT A FIXTURE: ``torch`` may already be imported by the
time any fixture runs, and the environment variable has to be in place
BEFORE anything can initialise a CUDA context.  ``torch.cuda.is_available()``
consults ``CUDA_VISIBLE_DEVICES`` at CALL time, so every test that
self-selects ``"cuda" if torch.cuda.is_available() else "cpu"`` now resolves
to ``cpu``.

GPU TESTING IS STILL AVAILABLE, EXPLICITLY: run with
``MOLMETAL_TEST_ALLOW_GPU=1 pytest ...`` and ``CUDA_VISIBLE_DEVICES`` is left
exactly as the caller set it.  GPU-REQUIRED tests skip cleanly when CUDA is
unavailable (a GPU test that silently runs on CPU is a false green).

The published-tree write guard installed by the Wf-A123 Implement agent
lives at the END of this file and is untouched by the block above.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest


# --- Wf-A123 FIX 5: THE NO-GPU CONTRACT, ENFORCED AT IMPORT TIME ----------
# See the module docstring.  This MUST stay at module level (top-level
# code), NOT inside a fixture: the variable has to be set before any test
# module (or this conftest's own imports) can initialise a CUDA context.
MOLMETAL_TEST_ALLOW_GPU_ENV = "MOLMETAL_TEST_ALLOW_GPU"
if os.environ.get(MOLMETAL_TEST_ALLOW_GPU_ENV) != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# Per env.sh §1b — point the data loaders at the local repo CSV so
# CPU-only test hosts (no /mnt/storage mount) can read MetalCytoToxDB
# without setting MOLMETAL_DATA_DIR manually.  This fixes the recurring
# FileNotFoundError at MetalCytotoxDataset.from_csv that 4 dmpnn tests
# were tripping on.
_DEFAULT_DATA_DIR = PROJECT_ROOT / ".data" / "molmetal"
if not os.environ.get("MOLMETAL_DATA_DIR") and _DEFAULT_DATA_DIR.is_dir():
    os.environ["MOLMETAL_DATA_DIR"] = str(_DEFAULT_DATA_DIR)


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
# Wf-A269 wave-28 (2026-11-06) — BOTH module-level torch hooks are SKIPPED
# when torch is not importable.  They used to run unconditionally and both
# reach `import torch` ⇒ the conftest raised ModuleNotFoundError at IMPORT
# time in any torch-less environment, which kills the whole pytest session
# before a single test is collected.
#
# MEASURED 2026-11-06 (pytest-only venv, CI's own install line): the conftest
# died at `_force_torch_scatter_backend_for_cpu_tests` (its own `import
# torch`, line 269), and once that was guarded it died one step later at
# `_ensure_reference_packages` → `_reference/path/__init__.py` →
# `molmetal/references/flow_matching/flow_matching/interpolation.py:19`
# (`import torch`).  BOTH are torch-mandatory, so BOTH are skipped here.
#
# WHY THIS MATTERS: CI job 4d (`.github/workflows/ci.yml`) installs pytest
# ONLY — its own comment says "this job needs pytest and nothing else" — and
# its pytest step therefore died at conftest import ⇒ a CI JOB THAT CAN NEVER
# PASS (the §6.269.17 class, fourth costume).
#
# THE SAME DEFECT WAS NAMED AS THE CLOSURE PATH BY AT LEAST FIVE EARLIER
# WAVES AND NEVER EXECUTED: TODO/completed/420a3_wave_a_launch.md:122
# ("guard `conftest.py:197` with `try: import torch; except ImportError:
# pytest.skip(...)` (~5 LOC CPU-only)"), 394a1:99, 394a3:235, 356b1:154,
# 384a1:82, 414a3:193 (all "deferred to a separate conftest-fix wave").
#
# WHY A SKIP, NOT A `pytest.skip`: skipping the whole session would defeat
# exactly the torch-less CI jobs that need it (the guard modules under test
# import NO torch).  `pytest.skip` is also wrong at CONFTEST IMPORT time —
# no test item exists yet.
#
# THE HOOKS' PURPOSE IS UNAFFECTED: with torch absent there is no scatter to
# redirect and no `_reference` package to preload, so skipping them is the
# no-op they were always meant to be.
#
# LINE-NUMBER NOTE (honest, §6.241): the wave-18 test docstring
# (`molmetal/tests/test_wfa269_pitfall_runner_interpreter.py:20`) quotes the
# 2026-11-04 traceback at `conftest.py:269` — that position is UNCHANGED by
# this edit (the guard sits below it).  `TODO/INDEX.md:1686` cites the
# `_published_tree_guard` fixture at `conftest.py:479-480` — that citation
# was ALREADY STALE before this edit (HEAD carries no
# `_published_tree_guard` at all; the working tree had it at ~687) and is
# re-quoted by this wave.
if importlib.util.find_spec("torch") is not None:
    _force_torch_scatter_backend_for_cpu_tests()
    _ensure_reference_packages()


# ---------------------------------------------------------------------------
# Wf-A123 — published-tree write guard
# ---------------------------------------------------------------------------
# RULE: a test run must NEVER mutate ``metrics/by_metric/`` or
# ``metrics/paper_tables/``.  Those trees are published artifacts; they
# change only when a script deliberately regenerates them, and that change
# must be visible in ``git status``.
#
# Why this exists: ``molmetal/tests/test_promote_metrics.py`` used to
# DELETE nine real files under ``metrics/by_metric/`` and re-promote older
# content over them, so a plain ``pytest`` run silently rewrote published
# artifacts (MEASURED 2026-10-31: the 8 committed targets changed on disk
# and ``hilbert_4cell_r2.json``'s ``aggregate_combined_r2_mean`` went from
# -5.662524090882017 to -6.712634861296495).
#
# MECHANISM: session-start snapshot of sha256 over the published trees,
# session-end recompute, diff.  A mutation is allowed ONLY when the test
# module that ran during the session declared that exact path in
# ``PUBLISHED_WRITE_DECLARATIONS.json``.  This mirrors AGENTS.md §5.11's
# ``DECLARED_INERT_PATHS``: a key either reaches a consumer, has a
# verified non-consumer, or is DECLARED with documented intent.
#
# The guard is read-only (it never writes) and cheap (sha256 only).
# ---------------------------------------------------------------------------

PUBLISHED_TREE_ROOTS: Tuple[Path, ...] = (
    Path("metrics") / "by_metric",
    Path("metrics") / "paper_tables",
    Path("molmetal") / "metrics" / "by_metric",
)

# ---------------------------------------------------------------------------
# Wf-A125 G2 — the CURATED FILE list.
#
# WHY A FILE LIST AND NOT A TREE ROOT: ``molmetal/reports/`` is NOT a
# published tree.  MEASURED 2026-10-31: 486 files under ``molmetal/scripts/``
# mention ``molmetal/reports`` or ``"reports"``, so a guard root over that
# tree would fire on essentially every test that runs any script.  The two
# SSOTs that live outside the three tree roots are therefore guarded as
# EXPLICIT FILES with an enumerated authorised-writer set:
#
#   * ``registry/{metrics,datasets,protocols}.yaml`` — the registry SSOT.  Its
#     ``source_json_sha256`` is the currency of ``registry.py check``; a
#     write-then-restore to identical bytes (which is exactly what
#     ``molmetal/tests/test_registry_generate_import_path.py`` does) is
#     invisible to a sha-only diff, so the file must be in the guarded set.
#   * ``molmetal/reports/run_ledger.jsonl`` — the append-only provenance
#     SSOT.  It has NO authorised writer: no test may append to the real
#     ledger (every ledger test uses a tmp ledger).
#
# The list is CLOSED: a path that is not named here is not guarded, and a
# path that is named here is guarded for EVERY kind of mutation.
# ---------------------------------------------------------------------------
PUBLISHED_FILE_PATHS: Tuple[Dict[str, Any], ...] = (
    {
        "path": "registry/metrics.yaml",
        "authorised_writers": (
            "molmetal/tests/test_registry_generate_import_path.py",
        ),
    },
    {
        "path": "registry/datasets.yaml",
        "authorised_writers": (
            "molmetal/tests/test_registry_generate_import_path.py",
        ),
    },
    {
        "path": "registry/protocols.yaml",
        "authorised_writers": (
            "molmetal/tests/test_registry_generate_import_path.py",
        ),
    },
    {
        "path": "molmetal/reports/run_ledger.jsonl",
        "authorised_writers": (),
    },
)

DECLARATIONS_PATH = (
    Path(__file__).resolve().parent / "PUBLISHED_WRITE_DECLARATIONS.json"
)

# Repo-relative paths of the test modules that executed in this session.
_ACTIVE_TEST_MODULES: set = set()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stat_entry(path: Path) -> Dict[str, Any]:
    """sha256 + the stat quadruple of one published file.

    Wf-A125 G3 — the sha alone is NOT enough.  A write-then-restore to the
    SAME bytes leaves the sha unchanged, so the guard could not see it.  The
    stat quadruple (mtime_ns / size / ino) is what makes that mutation
    visible: ``atomic_write_file`` uses ``os.replace``, which installs a NEW
    inode, and even an in-place rewrite moves the mtime.
    """
    st = path.stat()
    return {
        "sha256": _sha256_file(path),
        "mtime_ns": st.st_mtime_ns,
        "size": st.st_size,
        "ino": st.st_ino,
    }


def snapshot_published_trees(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    """Stat quadruple of every file under the published trees + curated files.

    Keys are repo-relative POSIX paths so a failure message is
    copy-pasteable into ``git diff``.  Read-only by construction.
    """
    snapshot: Dict[str, Dict[str, Any]] = {}
    for rel_root in PUBLISHED_TREE_ROOTS:
        root = repo_root / rel_root
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(repo_root).as_posix()
            snapshot[rel] = _stat_entry(path)
    for spec in PUBLISHED_FILE_PATHS:
        rel = str(spec["path"])
        path = repo_root / rel
        if not path.is_file():
            continue
        snapshot[rel] = _stat_entry(path)
    return snapshot


def diff_published_trees(
    before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """``relpath -> {"before": entry|None, "after": entry|None, "kind": ...}``.

    FOUR kinds (Wf-A125 G3):

    * ``created``  — the file did not exist before.
    * ``deleted``  — the file does not exist any more.
    * ``modified`` — the sha256 changed.
    * ``touched``  — the sha256 is IDENTICAL but the stat quadruple moved.
      This is the write-then-restore case: the bytes are back, the file was
      still rewritten.  A sha-only diff cannot see it.
    """
    diff: Dict[str, Dict[str, Any]] = {}
    for rel in set(before) | set(after):
        b = before.get(rel)
        a = after.get(rel)
        if b is None and a is None:
            continue
        if b is None:
            kind = "created"
        elif a is None:
            kind = "deleted"
        elif b.get("sha256") != a.get("sha256"):
            kind = "modified"
        elif (
            b.get("mtime_ns") != a.get("mtime_ns")
            or b.get("size") != a.get("size")
            or b.get("ino") != a.get("ino")
        ):
            kind = "touched"
        else:
            continue
        diff[rel] = {"before": b, "after": a, "kind": kind}
    return diff


def load_declarations(
    path: Optional[Path] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """Read the declaration file -> ``{module: [{"path": ..., "reason": ...}]}``.

    Raises ``ValueError`` on a malformed entry — a declaration file that
    cannot be parsed must not silently degrade into "no declarations".
    """
    p = Path(path) if path is not None else DECLARATIONS_PATH
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{p} must be a JSON object mapping module -> list")
    out: Dict[str, List[Dict[str, str]]] = {}
    for module, entries in raw.items():
        if module.startswith("_"):  # reserved for future metadata keys
            continue
        if not isinstance(entries, list):
            raise ValueError(f"{p}: entry for {module!r} must be a list")
        normalised: List[Dict[str, str]] = []
        for entry in entries:
            if not isinstance(entry, dict) or "path" not in entry:
                raise ValueError(
                    f"{p}: every entry for {module!r} needs a 'path' key"
                )
            normalised.append(
                {"path": str(entry["path"]), "reason": str(entry.get("reason", ""))}
            )
        out[module] = normalised
    return out


def is_test_module_key(key: str) -> bool:
    """Is this declaration key a TEST MODULE (i.e. can it AUTHORISE)?

    Wf-A125 G4 — the declaration file mixes two kinds of statement and the
    loader used to treat them identically:

    * an **AUTHORISATION** — "the test module ``X`` may write published path
      ``P``".  Only a test module can hold one, because only a test module
      runs inside the guarded session.
    * a **CAPABILITY RECORD** — "the generator script ``X`` writes published
      path ``P`` by default".  It documents a capability; it NEVER authorises
      anything.

    MEASURED 2026-10-31: all three shipped keys name SCRIPTS
    (``molmetal/scripts/*.py``), so all three were permanently inert — the
    file *claimed* to record generator capabilities and the loader *silently*
    treated them as test authorisations.  The inverse fragility was worse: if
    the matcher were ever loosened to "any key in the file", all three would
    start authorising writes into ``metrics/by_metric/``.

    The contract is now OPERATIVE: a key authorises iff it names a ``*.py``
    under a ``tests/`` directory.
    """
    p = PurePosixPath(str(key).replace("\\", "/"))
    if p.suffix != ".py":
        return False
    return any(part == "tests" for part in p.parts)


def split_declarations(
    declarations: Dict[str, List[Dict[str, str]]],
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]]]:
    """``(authorisations, capability_records)`` — the G4 classification.

    An AUTHORISATION is a key that names a test module; a CAPABILITY RECORD
    is every other key.  Only the authorisations may appear in the guard's
    allowed set.
    """
    auth: Dict[str, List[Dict[str, str]]] = {}
    cap: Dict[str, List[Dict[str, str]]] = {}
    for module, entries in declarations.items():
        if is_test_module_key(module):
            auth[module] = entries
        else:
            cap[module] = entries
    return auth, cap


def load_capability_records(
    path: Optional[Path] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """The declaration entries that are CAPABILITY RECORDS, not authorisations.

    Wf-A125 G4 — these were permanently inert before.  They are now LOADED,
    COUNTED and REPORTED so that "this script can write this published path"
    is a visible statement rather than a silently-ignored key.
    """
    _auth, cap = split_declarations(load_declarations(path))
    return cap


def find_undeclared_writes(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
    declarations: Dict[str, List[Dict[str, str]]],
    active_test_modules: Iterable[str] = (),
    published_file_paths: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Optional[str]]]:
    """Return every published mutation that no active test module declared.

    A mutation is authorised only by a **test module** that executed in
    this session and that declared the exact published path.  Entries
    keyed by a non-test module (a generator script) are capability
    records: they document which scripts can write into the published
    tree, but they do not authorise a test to do so — per Wf-A123 no test
    module legitimately regenerates a published file.

    Wf-A125 G4 — the test-module check is now a CONTRACT
    (:func:`is_test_module_key`), not an accident of the key set.  A
    capability record can never authorise, even if it is passed as the
    active module.

    Wf-A125 G2 — the curated FILES carry their own authorised-writer set;
    a curated file is authorised only when one of its named writers ran.

    Wf-A125 G3 — a mutation of kind ``touched`` (identical bytes, moved
    stat quadruple) is a violation like any other.
    """
    active = set(active_test_modules)
    authorisations, _capability_records = split_declarations(declarations)
    allowed = {
        entry["path"]
        for module, entries in authorisations.items()
        if module in active
        for entry in entries
    }
    specs = (
        PUBLISHED_FILE_PATHS
        if published_file_paths is None
        else tuple(published_file_paths)
    )
    for spec in specs:
        rel = str(spec.get("path", ""))
        if not rel or rel in allowed:
            continue
        writers = set(spec.get("authorised_writers") or ())
        if writers & active:
            allowed.add(rel)
    violations: List[Dict[str, Optional[str]]] = []
    for rel, entry in sorted(diff_published_trees(before, after).items()):
        if rel in allowed:
            continue
        b = entry.get("before")
        a = entry.get("after")
        violations.append(
            {
                "path": rel,
                "kind": entry["kind"],
                "before_sha": b.get("sha256") if b else None,
                "after_sha": a.get("sha256") if a else None,
            }
        )
    return violations


def format_violations(
    violations: List[Dict[str, Optional[str]]],
    active_test_modules: Iterable[str] = (),
    declarations: Optional[Dict[str, List[Dict[str, str]]]] = None,
) -> str:
    """Human-readable guard failure message (names every offending path)."""
    lines = [
        "PUBLISHED TREE WRITE GUARD (Wf-A123): a test run modified published "
        "metric artifacts.",
        "",
    ]
    for v in violations:
        lines.append(f"  - {v['path']}  ({v['kind']})")
        lines.append(f"      before sha256: {v['before_sha']}")
        lines.append(f"      after  sha256: {v['after_sha']}")
        if v["kind"] == "touched":
            lines.append(
                "      NOTE: the bytes are identical — the file was rewritten "
                "and restored.  A sha-only diff cannot see this."
            )
    if declarations is not None:
        _auth, cap = split_declarations(declarations)
        lines.append("")
        lines.append(
            f"Declaration file: {len(_auth)} test authorisation(s), "
            f"{len(cap)} generator capability record(s) (the capability "
            "records never authorise a write)."
        )
    lines.append("")
    lines.append("Test modules that ran in this session:")
    for module in sorted(set(active_test_modules)):
        lines.append(f"  - {module}")
    lines.append("")
    lines.append(
        "Published artifacts change only when a script deliberately regenerates "
        "them, and that change must be visible in git status.  A test module "
        "that must write a published path has to declare it in "
        "molmetal/tests/PUBLISHED_WRITE_DECLARATIONS.json with a one-line "
        "reason."
    )
    return "\n".join(lines)


@pytest.fixture(autouse=True)
def _published_guard_record_module(request: pytest.FixtureRequest):
    """Record which test module executed (used by the session guard)."""
    node = getattr(request, "node", None)
    path = getattr(node, "path", None)
    if path is None:
        path = getattr(node, "fspath", None)
    if path is not None:
        try:
            rel = Path(str(path)).resolve().relative_to(PROJECT_ROOT)
        except (ValueError, OSError):
            rel = Path(str(path))
        _ACTIVE_TEST_MODULES.add(rel.as_posix())
    yield


@pytest.fixture(scope="session", autouse=True)
def _published_tree_guard():
    """Fail the session if a test mutated the published metric trees."""
    before = snapshot_published_trees(PROJECT_ROOT)
    if not before:
        # No published tree on this host — nothing to guard.
        yield None
        return
    yield None
    after = snapshot_published_trees(PROJECT_ROOT)
    declarations = load_declarations()
    violations = find_undeclared_writes(
        before, after, declarations, _ACTIVE_TEST_MODULES
    )
    if violations:
        pytest.fail(
            format_violations(violations, _ACTIVE_TEST_MODULES, declarations),
            pytrace=False,
        )