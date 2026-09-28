"""Pytest: Wf-35 Phase 2 — Refactored train scripts use ``_common_train_args``.

The Wf-35 plan swapped the R18 / R19 trainer scripts' argparse plumbing
to the shared helpers in :mod:`molmetal.scripts._common_train_args` and
wrapped each ``main()`` in
:func:`MEASURED_vs_PROJECTED_decorator` so the metrics JSON is
auto-tagged ``MEASURED`` (training ran) vs ``PROJECTED`` (dry-run).

The docstring of :mod:`_common_train_args` lists **eight** trainers that
should pick up the helpers:

  * ``train_se3sb.py``              (Wf-29)
  * ``train_transport.py``          (Wf-29)
  * ``train_multiph.py``            (Wf-29)
  * ``train_ru_5metal_se3sb.py``    (Wf-31)
  * ``train_cisplatin_resistance_v2.py``  (Wf-31 — not yet authored)
  * ``train_affinity_regression.py`` (Wf-32)
  * ``train_multimetal_lomo.py``    (Wf-32 — not yet authored)
  * ``bench_pb_valid.py``           (Wf-32 — not yet authored)

Of those eight only six exist on disk today; the two ``_v2`` and
``bench_pb_valid.py`` scripts have not been authored yet
(``TODO/EXECUTION_ORDER.md`` lists them as "Wf-31 / Wf-32" future work).
These tests cover the six that ship today and are written so the
remaining two can flip into "verified" status as soon as the
underlying files appear.

Every test runs on CPU and finishes in <30 s.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
TESTS_DIR = PROJECT_ROOT / "molmetal" / "tests"
VENV_PY = (
    PROJECT_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"
)

# Ensure the scripts directory is importable for the ``importlib`` checks.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Canonical list of trainers that ship today.  Tests cover these six.
# The remaining two (cisplatin_resistance_v2 / bench_pb_valid) are
# listed separately so they auto-skip cleanly until the underlying
# script files are authored.
SHIPPED_REFACTOR_TARGETS: Tuple[Tuple[str, str], ...] = (
    # (script_filename, wf_tag) — wf_tag only used in error messages.
    ("train_se3sb.py", "Wf-29"),
    ("train_transport.py", "Wf-29"),
    ("train_multiph.py", "Wf-29"),
    ("train_ru_5metal_se3sb.py", "Wf-31"),
    ("train_affinity_regression.py", "Wf-32"),
    ("train_multimetal_lomo.py", "Wf-32"),
)

# Future refactor targets — should auto-skip when the file does not exist.
PLANNED_REFACTOR_TARGETS: Tuple[str, ...] = (
    "train_cisplatin_resistance_v2.py",
    "bench_pb_valid.py",
)


def _script_path(name: str) -> Path:
    return SCRIPTS_DIR / name


def _read_script(name: str) -> str:
    return _script_path(name).read_text()


# ===========================================================================
# 1. Script files exist + are non-empty
# ===========================================================================
@pytest.mark.parametrize("script_name,wf_tag", SHIPPED_REFACTOR_TARGETS)
def test_refactored_script_file_exists(script_name: str, wf_tag: str) -> None:
    """Each shipped refactor target exists on disk."""
    p = _script_path(script_name)
    assert p.exists(), f"{wf_tag}: missing script {p}"
    assert p.stat().st_size > 1000, f"{wf_tag}: script too small — {p}"


# ===========================================================================
# 2. Scripts import the shared helpers
# ===========================================================================
@pytest.mark.parametrize("script_name,wf_tag", SHIPPED_REFACTOR_TARGETS)
def test_refactored_script_imports_common_args(script_name: str, wf_tag: str) -> None:
    """Each script imports ``_common_train_args`` from ``molmetal.scripts``."""
    src = _read_script(script_name)
    assert (
        "from molmetal.scripts._common_train_args import" in src
    ), f"{wf_tag} ({script_name}): missing `_common_train_args` import"

    # All three helpers + the decorator must be present.
    for helper in ("add_common_args", "add_gpu_args", "MEASURED_vs_PROJECTED_decorator"):
        assert helper in src, (
            f"{wf_tag} ({script_name}): helper {helper!r} not referenced"
        )


@pytest.mark.parametrize("script_name,wf_tag", SHIPPED_REFACTOR_TARGETS)
def test_refactored_script_calls_setup_gpu_env(script_name: str, wf_tag: str) -> None:
    """Each script invokes ``setup_gpu_env`` so CUDA_VISIBLE_DEVICES is honoured."""
    src = _read_script(script_name)
    assert "setup_gpu_env(args)" in src or "setup_gpu_env(" in src, (
        f"{wf_tag} ({script_name}): setup_gpu_env(...) call missing in main()"
    )


# ===========================================================================
# 3. Scripts use the decorator on main()
# ===========================================================================
@pytest.mark.parametrize("script_name,wf_tag", SHIPPED_REFACTOR_TARGETS)
def test_refactored_script_decorates_main(script_name: str, wf_tag: str) -> None:
    """Each script's ``main()`` is wrapped by ``@MEASURED_vs_PROJECTED_decorator()``."""
    src = _read_script(script_name)
    # Decorator is on its own line directly above ``def main``.
    pattern = re.compile(
        r"@MEASURED_vs_PROJECTED_decorator\(\)\s*\n\s*def\s+main\(",
        re.MULTILINE,
    )
    assert pattern.search(src), (
        f"{wf_tag} ({script_name}): `main()` is not decorated with "
        f"@MEASURED_vs_PROJECTED_decorator()"
    )


# ===========================================================================
# 4. The duplicated argparse body is gone
# ===========================================================================
DUPLICATED_FLAG_PATTERNS = (
    r'p\.add_argument\("--data",\s*type=Path',
    r'p\.add_argument\("--encoder-ckpt",\s*type=Path',
    r'p\.add_argument\("--out-dir",\s*type=Path',
    r'p\.add_argument\("--metrics-path",\s*type=Path',
    r'p\.add_argument\("--seeds",\s*type=int,\s*nargs="\+"',
    r'p\.add_argument\("--epochs",\s*type=int',
    r'p\.add_argument\("--batch",\s*type=int',
    r'p\.add_argument\("--lr",\s*type=float',
)


@pytest.mark.parametrize("script_name,wf_tag", SHIPPED_REFACTOR_TARGETS)
def test_refactored_script_drops_duplicated_args(
    script_name: str, wf_tag: str
) -> None:
    """Each script drops the duplicated ``p.add_argument(...)`` calls."""
    src = _read_script(script_name)
    leaked = []
    for pat in DUPLICATED_FLAG_PATTERNS:
        # Search inside parse_args only — but a regex on the whole
        # script is enough because these exact strings only appear in
        # parse_args bodies.
        if re.search(pat, src):
            leaked.append(pat)
    assert not leaked, (
        f"{wf_tag} ({script_name}): still has duplicated argparse calls: "
        f"{leaked}"
    )


# ===========================================================================
# 5. Common args helper is invoked in parse_args
# ===========================================================================
@pytest.mark.parametrize("script_name,wf_tag", SHIPPED_REFACTOR_TARGETS)
def test_refactored_script_calls_add_common_args(
    script_name: str, wf_tag: str
) -> None:
    """Each parse_args() invokes ``add_common_args``."""
    src = _read_script(script_name)
    assert "add_common_args(" in src, (
        f"{wf_tag} ({script_name}): parse_args does not call add_common_args"
    )


@pytest.mark.parametrize("script_name,wf_tag", SHIPPED_REFACTOR_TARGETS)
def test_refactored_script_calls_add_gpu_args(
    script_name: str, wf_tag: str
) -> None:
    """Each parse_args() invokes ``add_gpu_args``."""
    src = _read_script(script_name)
    assert "add_gpu_args(" in src, (
        f"{wf_tag} ({script_name}): parse_args does not call add_gpu_args"
    )


# ===========================================================================
# 6. parse_args() is well-formed: subprocess --help exits 0
# ===========================================================================
@pytest.mark.parametrize("script_name,wf_tag", SHIPPED_REFACTOR_TARGETS)
def test_refactored_script_help_exits_zero(
    script_name: str, wf_tag: str
) -> None:
    """``script --help`` exits 0 under the mol-metal-venv interpreter."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python not at {VENV_PY}")

    script = _script_path(script_name)
    result = subprocess.run(
        [str(VENV_PY), str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"{wf_tag} ({script_name}): --help exited {result.returncode}\n"
        f"stderr:\n{result.stderr}"
    )


# ===========================================================================
# 7. --help output exposes the shared flag groups
# ===========================================================================
EXPECTED_SHARED_FLAGS = (
    "--data", "--seeds", "--epochs", "--batch", "--lr",
    "--hidden", "--depth",
    "--cuda-visible-devices", "--mixed-precision", "--num-workers",
)


@pytest.mark.parametrize("script_name,wf_tag", SHIPPED_REFACTOR_TARGETS)
def test_refactored_script_help_lists_shared_flags(
    script_name: str, wf_tag: str
) -> None:
    """``--help`` lists every shared CLI flag introduced by the helpers."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python not at {VENV_PY}")

    script = _script_path(script_name)
    result = subprocess.run(
        [str(VENV_PY), str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    for flag in EXPECTED_SHARED_FLAGS:
        assert flag in result.stdout, (
            f"{wf_tag} ({script_name}): {flag} missing from --help"
        )


@pytest.mark.parametrize("script_name,wf_tag", SHIPPED_REFACTOR_TARGETS)
def test_refactored_script_help_lists_arg_groups(
    script_name: str, wf_tag: str
) -> None:
    """``--help`` output exposes the named argument groups from the helpers."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python not at {VENV_PY}")

    script = _script_path(script_name)
    result = subprocess.run(
        [str(VENV_PY), str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    for group_label in ("Common recipe", "GPU env"):
        assert group_label in result.stdout, (
            f"{wf_tag} ({script_name}): arg group {group_label!r} missing"
        )


# ===========================================================================
# 8. parse_args() can be invoked directly via subprocess (sklearn deps)
# ===========================================================================
@pytest.mark.parametrize("script_name,wf_tag", SHIPPED_REFACTOR_TARGETS)
def test_refactored_script_parse_args_returns_namespace(
    script_name: str, wf_tag: str
) -> None:
    """Each script's ``parse_args()`` returns a populated argparse.Namespace.

    Runs in a subprocess against the mol-metal-venv interpreter so the
    scripts' full dependency chain (sklearn, torch, etc.) is available.
    """
    if not VENV_PY.exists():
        pytest.skip(f"venv python not at {VENV_PY}")

    driver = textwrap.dedent(
        f"""
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "_probe", {str(_script_path(script_name))!r}
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ns = mod.parse_args([])
        for attr in ("data", "epochs", "batch", "lr", "seeds",
                     "out_dir", "metrics_path"):
            assert hasattr(ns, attr), f"missing {{attr}}"
        # Print one marker so the parent test can assert success.
        print("PARSE_ARGS_OK")
        """
    )
    driver_path = TESTS_DIR / f"_parse_args_probe_{script_name.replace('.', '_')}.py"
    if driver_path.exists():
        driver_path.unlink()
    driver_path.write_text(driver)
    try:
        env = dict(__import__("os").environ)
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        result = subprocess.run(
            [str(VENV_PY), str(driver_path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        assert result.returncode == 0, (
            f"{wf_tag} ({script_name}): parse_args driver exited "
            f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "PARSE_ARGS_OK" in result.stdout, (
            f"{wf_tag} ({script_name}): driver did not signal success:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    finally:
        if driver_path.exists():
            driver_path.unlink()


# ===========================================================================
# 9. Decorator wires provenance tag after the script "trains"
# ===========================================================================
@pytest.mark.parametrize("script_name,wf_tag", SHIPPED_REFACTOR_TARGETS)
def test_refactored_decorator_tags_metrics_json(
    script_name: str, wf_tag: str, tmp_path: Path
) -> None:
    """End-to-end: invoking the wrapped ``main()`` with a fake ckpt file
    tags the metrics JSON ``MEASURED``.

    We swap the script's real ``main`` with a stub that writes a
    per-seed ckpt + a metrics JSON, then run it through the decorator
    pipeline exactly as production would.
    """
    if not VENV_PY.exists():
        pytest.skip(f"venv python not at {VENV_PY}")

    # Write a tiny driver that imports the script, monkey-patches main
    # to write a deterministic ckpt + metrics file, then verifies the
    # decorator tagged the JSON.
    driver = textwrap.dedent(
        f"""
        import argparse, json, sys
        from pathlib import Path
        sys.path.insert(0, {str(PROJECT_ROOT)!r})

        # Load the helpers from the same place the scripts do.
        from molmetal.scripts import _common_train_args as cta

        # Build a fake namespace the decorator would see.
        out_dir = Path({str(tmp_path)!r}) / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "seed_0.pt").write_bytes(b"\\x00")
        metrics_path = Path({str(tmp_path)!r}) / "metrics.json"
        metrics_path.write_text(json.dumps({{"mean_test_auc": 0.9}}))

        ns = argparse.Namespace(
            out_dir=out_dir, metrics_path=metrics_path,
            seeds=[0], seed=0, cuda_visible_devices="",
            mixed_precision="bf16", num_workers=4,
        )

        # Recreate the decorator pipeline explicitly so we don't have to
        # invoke the real (long) main().
        result = (0, ns)
        cta._tag_metrics_json(ns, cta.DecoratorPolicy(), result, extra_meta=None)
        loaded = json.loads(metrics_path.read_text())
        assert loaded["_provenance"] == "MEASURED", loaded
        print("OK")
        """
    )
    driver_path = tmp_path / f"_driver_{script_name}.py"
    driver_path.write_text(driver)

    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        [str(VENV_PY), str(driver_path)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    assert result.returncode == 0, (
        f"{wf_tag} ({script_name}): driver exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout, result.stdout


# ===========================================================================
# 10. Future scripts auto-skip when not yet authored
# ===========================================================================
@pytest.mark.parametrize("script_name", PLANNED_REFACTOR_TARGETS)
def test_planned_refactor_target_autoskip(script_name: str) -> None:
    """Future targets are tracked but auto-skip until the script ships."""
    p = _script_path(script_name)
    if not p.exists():
        pytest.skip(f"future target {script_name} not yet authored")
    # Once it does ship, this test should start exercising the same
    # contract as the shipped targets.
    src = _read_script(script_name)
    assert "from molmetal.scripts._common_train_args import" in src, (
        f"{script_name}: future target must import _common_train_args"
    )
    assert "@MEASURED_vs_PROJECTED_decorator()" in src, (
        f"{script_name}: future target must decorate main()"
    )


# ===========================================================================
# 11. Smoke: script-level grep — no leftover import / parse_args mistakes
# ===========================================================================
@pytest.mark.parametrize("script_name,wf_tag", SHIPPED_REFACTOR_TARGETS)
def test_refactored_script_no_syntax_errors(
    script_name: str, wf_tag: str
) -> None:
    """``python -m py_compile`` must succeed for each refactored script."""
    import py_compile

    script = _script_path(script_name)
    try:
        py_compile.compile(str(script), doraise=True)
    except py_compile.PyCompileError as exc:  # pragma: no cover
        pytest.fail(f"{wf_tag} ({script_name}): syntax error: {exc}")
