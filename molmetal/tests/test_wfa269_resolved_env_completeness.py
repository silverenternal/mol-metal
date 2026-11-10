"""Wf-A269 frontier sweep — the provenance environment capture is COMPLETE.

THE DEFECT (found by read-only review on 2026-11-04, no GPU run).

``molmetal/scripts/wfA2_config_loader.py::resolved_environment`` records the
environment a run was executed with, so a wrapper's registration-time
environment and the trainer's exec-time environment can be compared.  Before
this repair the capture was two rules:

    RESOLVED_ENV_EXACT = (6 names)          # an exact list
    name.startswith(("MOLMETAL_", "WFA2_")) # a prefix rule

MEASURED by an AST census over the trainer population (the transitive
repo-local import closure of every ``molmetal/scripts/train_*.py``; 209
modules, 28 distinct ambient env names), THIRTEEN ambient reads matched
NEITHER rule — so the provenance record attested to an environment it could
not reproduce:

    CHECK_MATH_PRECONDITIONS, CROSSDOCKED_2020_ROOT, CUBLAS_WORKSPACE_CONFIG,
    DRUGOOD_IC50_CSV_DIR, DRUGOOD_IC50_TARGET_ROOT, FUSED_RELU_MLP,
    LAMBDA_ASTSR_ENABLED, TRITON_MINIBATCH_OT, TRITON_RELU_MLP,
    TRITON_USE_FUSED, TRITON_WASSERSTEIN_SLICED, WFA78_DEBUG, WFA_STEP_TIMING

THE REPAIR.

* The loader records all thirteen (the PRE-REPAIR six-name tuple is preserved
  VERBATIM in a dated comment, per AGENTS.md §5.5 rule #6).
* The prefix rule moved out of an inline tuple into the named constant
  ``RESOLVED_ENV_PREFIXES`` so the coverage rule can be CHECKED from outside
  the module instead of being RESTATED by a test.
* THIS FILE re-derives the census from the source and asserts the invariant:
  **every ambient env read in the trainer population is recorded.**

THE SCANNER HAS THREE BLIND SPOTS, AND EVERY ONE OF THEM IS A REAL DEFECT
THAT WAS FOUND AND CLOSED WHILE BUILDING THIS FILE (each is pinned below):

  1. ALIASES — the repo writes ``import os as _os`` (and ``_wf130_os``,
     ``_wf50_os``, ...), so a scanner that matches the literal ``os.environ``
     sees none of them.  That hid ``FUSED_RELU_MLP``.
  2. DYNAMIC KEYS — the repo's ``_env_flag(name, default)`` helpers read the
     env NAME as an ARGUMENT, so a scanner that only matches a literal
     ``os.environ.get("X")`` sees none of the Triton dispatch gates.
  3. RELATIVE IMPORTS — inside a package the repo writes ``from .x import
     y``; a scanner that follows only absolute ``molmetal.*`` imports stops
     at the package boundary.

A scanner with a blind spot is the §6.269.17 DEAD-GATE class in a new
costume, so each blind spot is pinned by a test that would FAIL if the
scanner regressed to the naive form.

THE COVERAGE ORACLE IS THE PRODUCTION FUNCTION.

``resolved_environment`` is called with a one-name environment, so the
question "is this name recorded?" is answered by the SAME code that writes
the record — not by a restatement of the rule.  A test that restates the
rule can silently diverge from the implementation; this one cannot.

NO GPU, NO PAPER, NO TRAINING: this file only parses source and calls one
pure function.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"


# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------
def _os_aliases(tree: ast.AST) -> Set[str]:
    """Every local name bound to the `os` module (BLIND SPOT 1)."""
    names = {"os"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    names.add(alias.asname or "os")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "os":
                for alias in node.names:
                    names.add(alias.asname or alias.name)
    return names


def _is_environ(node: ast.AST, aliases: Set[str]) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases
    )


def _is_dynamic_env_call(node: ast.AST, aliases: Set[str]) -> bool:
    """A call that reads the env with a NON-LITERAL key (BLIND SPOT 2)."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    attr, recv = node.func.attr, node.func.value
    if attr in ("get", "getenv") and _is_environ(recv, aliases):
        return not (node.args and isinstance(node.args[0], ast.Constant))
    if attr == "getenv" and isinstance(recv, ast.Name) and recv.id in aliases:
        return not (node.args and isinstance(node.args[0], ast.Constant))
    if attr == "setdefault" and _is_environ(recv, aliases):
        return not (node.args and isinstance(node.args[0], ast.Constant))
    return False


def env_reads(path: Path) -> Dict[str, List[int]]:
    """The ambient env READS in one module -> {name: [line, ...]}.

    A READ is one of

      * ``<alias>.environ.get(K)`` / ``.getenv(K)`` / ``<alias>.getenv(K)``
      * ``<alias>.environ[K]``  (a Subscript in a Load context)
      * ``<alias>.environ.setdefault(K, v)`` — reads the ambient value when
        the key is already present, and the AMBIENT value wins over ``v``
      * a call to a LOCAL DYNAMIC READER (``_env_flag("K", "0")``) with a
        literal first argument

    A WRITE (``<alias>.environ[K] = v``, ``update``, ``pop``) is NOT an
    ambient read.
    """
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src, str(path))
    aliases = _os_aliases(tree)
    out: Dict[str, List[int]] = {}

    def add(name, line: int) -> None:
        if isinstance(name, str) and name:
            out.setdefault(name, []).append(line)

    # Pass 1 — the DYNAMIC READERS: a function whose body reads the env with
    # a non-literal key.  Every literal call site of such a function is a
    # read of that literal.
    dynamic_readers: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if _is_dynamic_env_call(sub, aliases):
                    dynamic_readers.add(node.name)
                    break

    def _called_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr, recv = node.func.attr, node.func.value
            is_environ_get = attr in ("get", "getenv") and _is_environ(recv, aliases)
            is_module_getenv = (
                attr == "getenv" and isinstance(recv, ast.Name) and recv.id in aliases
            )
            if is_environ_get or is_module_getenv:
                if node.args and isinstance(node.args[0], ast.Constant):
                    add(node.args[0].value, node.lineno)
                continue
            if attr == "setdefault" and _is_environ(recv, aliases):
                if node.args and isinstance(node.args[0], ast.Constant):
                    add(node.args[0].value, node.lineno)
                continue
        if (
            isinstance(node, ast.Subscript)
            and _is_environ(node.value, aliases)
            and isinstance(node.ctx, ast.Load)
        ):
            sl = node.slice
            if isinstance(sl, ast.Index):  # pragma: no cover - py<3.9
                sl = sl.value  # type: ignore[attr-defined]
            if isinstance(sl, ast.Constant):
                add(sl.value, node.lineno)
            continue
        if isinstance(node, ast.Call) and node.args:
            fname = _called_name(node.func)
            if fname in dynamic_readers and isinstance(node.args[0], ast.Constant):
                add(node.args[0].value, node.lineno)
    return out


def _dotted_of(path: Path) -> str:
    """`.../molmetal/adapters/x.py` -> `molmetal.adapters.x`.

    A fixture OUTSIDE the repo (the mutation proofs) has no repo-relative
    dotted name, so its own directory is used as the package root.
    """
    try:
        parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    except ValueError:
        parts = [path.parent.name, path.stem]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_path(dotted: str) -> Path | None:
    parts = dotted.split(".")
    cand = REPO_ROOT.joinpath(*parts).with_suffix(".py")
    if cand.is_file():
        return cand
    pkg = REPO_ROOT.joinpath(*parts) / "__init__.py"
    if pkg.is_file():
        return pkg
    return None


def _local_imports(path: Path) -> Set[str]:
    """The dotted names of the repo-local modules a file imports.

    Handles ABSOLUTE (``import molmetal.x``, ``from molmetal.x import y``)
    and RELATIVE (``from .x import y``) forms (BLIND SPOT 3).
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), str(path))
    here = _dotted_of(path)
    pkg_parts = here.split(".") if path.name == "__init__.py" else here.split(".")[:-1]
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                tail = node.module.split(".") if node.module else []
                base = ".".join(base_parts + tail)
                if base:
                    out.add(base)
                    for alias in node.names:
                        out.add(f"{base}.{alias.name}")
            else:
                base = node.module or ""
                if base.startswith("molmetal"):
                    out.add(base)
                    for alias in node.names:
                        out.add(f"{base}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("molmetal"):
                    out.add(alias.name)
    return out


def trainer_population() -> Set[Path]:
    """The transitive repo-local import closure of every `train_*.py`."""
    trainers = sorted(SCRIPTS_DIR.glob("train_*.py"))
    seen: Set[Path] = set()
    queue: List[Path] = list(trainers)
    while queue:
        cur = queue.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for dotted in _local_imports(cur):
            mp = _module_path(dotted)
            if mp is not None and mp not in seen:
                queue.append(mp)
    return seen


def _site(path: Path, line: int) -> str:
    """`path:line`, repo-relative when the file is inside the repo."""
    try:
        rel = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()
    return f"{rel}:{line}"


def census_reads(population: Iterable[Path]) -> Dict[str, List[str]]:
    """{env name: [site, ...]} over a population of modules."""
    out: Dict[str, List[str]] = {}
    for path in sorted(population):
        for name, lines in env_reads(path).items():
            out.setdefault(name, []).append(_site(path, lines[0]))
    return out


def unrecorded(reads: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """The reads the loader does NOT record.

    THE COVERAGE ORACLE IS THE PRODUCTION FUNCTION: ``resolved_environment``
    is called with a one-name environment, so "is this name recorded?" is
    answered by the code that WRITES the record.
    """
    from molmetal.scripts import wfA2_config_loader as loader

    out: Dict[str, List[str]] = {}
    for name, sites in reads.items():
        recorded = loader.resolved_environment({name: "1"})
        if name not in recorded:
            out[name] = sites
    return out


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------
class TestTheCensusIsNotVacuous:
    """The scanner must actually SEE something, or the invariant is theatre."""

    def test_the_population_is_not_empty(self) -> None:
        population = trainer_population()
        assert population, "the trainer population is empty — the census is theatre"
        trainers = [p for p in population if p.name.startswith("train_")]
        assert len(trainers) >= 40, (
            f"the trainer population lost its trainers: {len(trainers)}"
        )

    def test_the_scanner_finds_a_known_read(self) -> None:
        """A read the population certainly contains must be found."""
        reads = census_reads(trainer_population())
        assert "CUDA_VISIBLE_DEVICES" in reads, (
            "the scanner found no ambient read at all — it is broken, not clean"
        )

    def test_the_population_reaches_the_shared_argument_builder(self) -> None:
        """The closure must not stop at the trainer entry points."""
        population = trainer_population()
        assert SCRIPTS_DIR / "_common_train_args.py" in population, (
            "the shared argument builder is not in the population — its "
            "ambient reads are invisible to the census"
        )


class TestEveryAmbientReadIsRecorded:
    """THE INVARIANT: no ambient read escapes the record."""

    def test_every_ambient_env_read_in_the_trainer_population_is_recorded(self) -> None:
        reads = census_reads(trainer_population())
        assert reads, "the census found nothing — the invariant would be vacuous"
        missing = unrecorded(reads)
        assert not missing, (
            "these ambient env reads are NOT recorded by "
            f"resolved_environment: {sorted(missing)}.  Each one is read by "
            "the trainer process, so the provenance record cannot reproduce "
            "the run's environment.  Add the name to RESOLVED_ENV_EXACT (or "
            "give it the MOLMETAL_/WFA2_ prefix).  Sites:\n"
            + "\n".join(f"  {n}: {', '.join(s[:3])}" for n, s in sorted(missing.items()))
        )

    def test_the_recorded_names_are_carried_by_the_production_function(self) -> None:
        """The constant is WIRED, not merely declared.

        PRE-REPAIR this could pass with the constant present but the loop
        not reading it — the record would be empty while the source looked
        repaired.
        """
        from molmetal.scripts import wfA2_config_loader as loader

        for name in loader.RESOLVED_ENV_EXACT:
            recorded = loader.resolved_environment({name: "1"})
            assert name in recorded, (
                f"{name!r} is in RESOLVED_ENV_EXACT but resolved_environment "
                f"does not carry it — the constant is not wired"
            )

    def test_the_prefix_rule_is_named_and_applied(self) -> None:
        """The prefix rule is a NAMED constant, so it is checkable."""
        from molmetal.scripts import wfA2_config_loader as loader

        assert loader.RESOLVED_ENV_PREFIXES, "the prefix rule is empty"
        for prefix in loader.RESOLVED_ENV_PREFIXES:
            recorded = loader.resolved_environment({f"{prefix}PROBE": "1"})
            assert f"{prefix}PROBE" in recorded, (
                f"the prefix {prefix!r} is declared but not applied"
            )


class TestTheCompletenessCheckCanFail:
    """MUTATION PROOFS — a check that cannot fail is the defect we hunt."""

    def test_removing_a_name_from_the_recorded_set_is_detected(self) -> None:
        """M1 — drop one name from the recorded set; the check must notice."""
        from molmetal.scripts import wfA2_config_loader as loader

        reads = census_reads(trainer_population())
        victim = "CUBLAS_WORKSPACE_CONFIG"
        assert victim in reads, "the M1 victim is not read by the population"

        original = loader.RESOLVED_ENV_EXACT
        loader.RESOLVED_ENV_EXACT = tuple(n for n in original if n != victim)
        try:
            missing = unrecorded(reads)
        finally:
            loader.RESOLVED_ENV_EXACT = original
        assert victim in missing, (
            f"removing {victim!r} from RESOLVED_ENV_EXACT was NOT detected — "
            "the completeness check cannot fail"
        )

    def test_a_module_that_reads_an_unrecorded_name_is_detected(self, tmp_path: Path) -> None:
        """M2 — a synthetic module reading an unrecorded name must be flagged."""
        synthetic = tmp_path / "synthetic_trainer_env_reader.py"
        synthetic.write_text(
            "import os as _o\n"
            "FLAG = _o.environ.get('WFA_SYNTHETIC_ESCAPE', '0')\n",
            encoding="utf-8",
        )
        reads = census_reads([synthetic])
        assert "WFA_SYNTHETIC_ESCAPE" in reads, (
            "the scanner did not find a literal aliased read"
        )
        missing = unrecorded(reads)
        assert "WFA_SYNTHETIC_ESCAPE" in missing, (
            "a synthetic unrecorded read was not flagged — the completeness "
            "check cannot fail"
        )

    def test_the_scanner_sees_a_dynamic_key_helper(self, tmp_path: Path) -> None:
        """M3 — BLIND SPOT 2: the name arrives as an ARGUMENT."""
        synthetic = tmp_path / "synthetic_dynamic_env_reader.py"
        synthetic.write_text(
            "import os\n"
            "def _env_flag(name: str, default: str = '0') -> bool:\n"
            "    return os.environ.get(name, default) != '0'\n"
            "FLAG = _env_flag('WFA_SYNTHETIC_DYNAMIC', '0')\n",
            encoding="utf-8",
        )
        reads = census_reads([synthetic])
        assert "WFA_SYNTHETIC_DYNAMIC" in reads, (
            "the scanner does not see a DYNAMIC-key read — the repo's "
            "_env_flag(name, default) family is invisible to it"
        )

    def test_the_scanner_sees_an_aliased_os_import(self, tmp_path: Path) -> None:
        """M4 — BLIND SPOT 1: `import os as _os`."""
        synthetic = tmp_path / "synthetic_aliased_env_reader.py"
        synthetic.write_text(
            "import os as _os\n"
            "FLAG = _os.environ.get('WFA_SYNTHETIC_ALIAS', '0')\n",
            encoding="utf-8",
        )
        reads = census_reads([synthetic])
        assert "WFA_SYNTHETIC_ALIAS" in reads, (
            "the scanner does not see an ALIASED os import"
        )

    def test_the_scanner_sees_a_setdefault_read(self, tmp_path: Path) -> None:
        """M5 — `setdefault` READS the ambient value; the ambient wins."""
        synthetic = tmp_path / "synthetic_setdefault_env_reader.py"
        synthetic.write_text(
            "import os\n"
            "os.environ.setdefault('WFA_SYNTHETIC_SETDEFAULT', '1')\n",
            encoding="utf-8",
        )
        reads = census_reads([synthetic])
        assert "WFA_SYNTHETIC_SETDEFAULT" in reads, (
            "the scanner does not see a setdefault read — an ambient value "
            "wins over the default, so it IS an outcome-relevant read"
        )

    def test_the_scanner_does_not_count_a_write_as_a_read(self, tmp_path: Path) -> None:
        """M6 — a pure WRITE is not an ambient read."""
        synthetic = tmp_path / "synthetic_env_writer.py"
        synthetic.write_text(
            "import os\n"
            "os.environ['WFA_SYNTHETIC_WRITE'] = '1'\n",
            encoding="utf-8",
        )
        reads = census_reads([synthetic])
        assert "WFA_SYNTHETIC_WRITE" not in reads, (
            "a pure environment WRITE was counted as an ambient read"
        )

    def test_the_closure_follows_a_relative_import(self, tmp_path: Path) -> None:
        """M7 — BLIND SPOT 3: `from .x import y` inside a package."""
        pkg = tmp_path / "wfa_synthetic_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "inner.py").write_text(
            "import os as _o\n"
            "FLAG = _o.environ.get('WFA_SYNTHETIC_RELATIVE', '0')\n",
            encoding="utf-8",
        )
        (pkg / "outer.py").write_text(
            "from .inner import FLAG\n",
            encoding="utf-8",
        )
        names = _local_imports(pkg / "outer.py")
        assert "wfa_synthetic_pkg.inner" in names, (
            f"the relative import was not resolved: {names}"
        )
        reads = census_reads([pkg / "inner.py"])
        assert "WFA_SYNTHETIC_RELATIVE" in reads, (
            "the scanner does not see a read inside a relatively-imported module"
        )


if __name__ == "__main__":  # pragma: no cover - manual smoke surface
    raise SystemExit(pytest.main([__file__, "-q"]))
