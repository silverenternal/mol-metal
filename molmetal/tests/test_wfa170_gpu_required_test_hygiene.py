# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wf-A170 — GPU-REQUIRED test hygiene: the Wf-350-C1 env-probe tests.

THE DEFECT THIS GUARDS AGAINST
------------------------------
``molmetal/tests/test_wf350c1_gpu_env_probe.py`` was RED in every
CPU-locked session (``CUDA_VISIBLE_DEVICES=""`` — the repo's standing
CPU lock, applied at IMPORT time by BOTH ``/home/hugo/codes/mol-metal/
conftest.py`` and ``molmetal/tests/conftest.py``).  None of the 3
failures had anything to do with what the tests guard: the gate
``gate_p1_gpu_env_probe`` correctly returns **SKIP** in such a session
(Wf-A128, 2026-10-31) and the tests asserted PASS.

The repair must NOT be the lazy one.  A blanket
``pytest.mark.skipif`` over the module would throw away the coverage
that IS available on CPU and would make the tests vacuous.  The RIGHT
repair keeps the assertion honest in BOTH sessions:

  * GPU visible (``CUDA_VISIBLE_DEVICES`` non-empty AND torch sees a
    GPU): the ORIGINAL assertions hold — the gate PASSes, the fixture
    JSON exists and parses, the registration invariants hold.
  * GPU masked (the CPU-locked session): the gate returns **SKIP**
    (NOT a false PASS), the SKIP evidence is NON-EMPTY and names the
    session/host facts, and everything checkable on CPU still passes.

WHAT THIS FILE PROVES (the negative control)
--------------------------------------------
1. **The SKIP is REAL, NOT VACUOUS.**  Under a blank
   ``CUDA_VISIBLE_DEVICES`` the gate returns ``SKIP`` — never
   ``PASS`` — and the SKIP evidence is non-empty and names the session
   fact plus the measured host facts.  Without this, the repaired
   tests' "SKIP" branch could be asserting nothing.
2. **THE CPU-CHECKABLE COVERAGE WAS NOT THROWN AWAY.**  Parsed
   statically (AST), every one of the 3 repaired test functions still
   carries AT LEAST ONE ``assert`` OUTSIDE the GPU-conditional branch
   (so it runs unconditionally in a CPU-locked session), AND at least
   one ``assert`` INSIDE the GPU-conditional branch (so the GPU half is
   still asserted when a GPU IS visible — a branch that asserts
   nothing would be vacuous the other way round).
3. **THE LAZY ROUTE WAS NOT TAKEN.**  Neither the module nor any of
   the 3 test functions carries a ``pytest.mark.skipif`` — a blanket
   skip would silently delete the CPU coverage.
4. **THE REPAIRED FILE IS GREEN UNDER THE CPU LOCK.**  A subprocess
   pytest run of the repaired file with ``CUDA_VISIBLE_DEVICES=""``
   must report 3 passed.

CPU-ONLY BY CONSTRUCTION
-------------------------
This file imports no torch, allocates no GPU memory, and runs its one
subprocess (the end-to-end pytest re-run) with
``CUDA_VISIBLE_DEVICES=""`` set explicitly.  The gate itself spawns
only ``nvidia-smi`` (a read-only host query, unaffected by the CUDA
lock) + a ``torch`` probe subprocess whose CUDA half is masked by the
lock.

Cross-references
----------------
* ``molmetal/tests/test_wf350c1_gpu_env_probe.py`` — the repaired file.
* ``molmetal/scripts/pre_flight_check.py`` — ``gate_p1_gpu_env_probe``
  (SKIP branch) + ``EXPECTED_GATE_COUNTS`` + ``build_gates``.
* ``/home/hugo/codes/mol-metal/conftest.py`` +
  ``molmetal/tests/conftest.py`` — the two-level CPU lock.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The repaired file under guard.
TARGET_PATH = REPO_ROOT / "molmetal" / "tests" / "test_wf350c1_gpu_env_probe.py"
PREFLIGHT_SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"

#: The 3 test functions of the repaired file.
TARGET_TESTS = (
    "test_gate_wf350c1_gpu_env_probe_pass_on_host",
    "test_gate_wf350c1_gpu_env_probe_json_schema_validity",
    "test_gate_wf350c1_gpu_env_probe_registered_in_build_gates",
)

#: The name of the session-state predicate the repaired file branches on.
GPU_BRANCH_NAME = "_session_gpu_masked"

#: The tokens a SKIP evidence must carry (the session fact + the
#: measured host facts).  Mirrors the gate's SKIP branch wording.
SKIP_EVIDENCE_TOKENS = (
    "CUDA_VISIBLE_DEVICES",
    "SESSION",
    "nvidia_smi_rc=",
    "gpu_count=",
    "gpu_names=",
    "vram_free_gb=",
    "torch_probe=",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_preflight_module():
    """Import ``pre_flight_check.py`` via importlib (same pattern as the
    repaired file and the other pre-flight integration tests)."""
    spec = importlib.util.spec_from_file_location(
        "pre_flight_check_wfa170_under_test",
        PREFLIGHT_SCRIPT,
    )
    assert (
        spec is not None and spec.loader is not None
    ), f"pre_flight_check.py has no import spec ({PREFLIGHT_SCRIPT})"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pre_flight_check_wfa170_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _target_tree() -> ast.Module:
    """Parse the repaired file into an AST."""
    assert TARGET_PATH.is_file(), f"repaired file not on disk: {TARGET_PATH}"
    return ast.parse(TARGET_PATH.read_text(encoding="utf-8"), filename=str(TARGET_PATH))


def _is_gpu_branch(node: ast.If) -> bool:
    """True when this ``if`` is the GPU-session branch of the repair."""
    for sub in ast.walk(node.test):
        if isinstance(sub, ast.Name) and sub.id == GPU_BRANCH_NAME:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr == GPU_BRANCH_NAME:
            return True
    return False


def _count_asserts(statements) -> int:
    """Number of ``assert`` statements reachable from ``statements``."""
    return sum(
        1
        for stmt in statements
        for node in ast.walk(stmt)
        if isinstance(node, ast.Assert)
    )


def _split_asserts(func: ast.FunctionDef) -> tuple[int, int]:
    """Return ``(unconditional_asserts, gpu_conditional_asserts)``.

    An assert inside an ``if _session_gpu_masked(): ... else: ...``
    branch (either half) is CONDITIONAL — it only runs in one of the
    two sessions.  Every other assert in the function body is
    UNCONDITIONAL: it runs in BOTH sessions.
    """
    unconditional = 0
    conditional = 0
    for stmt in func.body:
        if isinstance(stmt, ast.If) and _is_gpu_branch(stmt):
            conditional += _count_asserts(stmt.body)
            conditional += _count_asserts(stmt.orelse)
        else:
            unconditional += _count_asserts([stmt])
    return unconditional, conditional


def _target_functions(tree: ast.Module) -> dict:
    """Map the 3 target test names to their FunctionDef nodes."""
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = node
    return found


# ---------------------------------------------------------------------------
# 1 — NEGATIVE CONTROL: the SKIP is real, loud, and not a false PASS
# ---------------------------------------------------------------------------


def test_wfa170_the_gate_skips_under_a_blank_cuda_visible_devices(
    monkeypatch,
    tmp_path,
) -> None:
    """NEGATIVE CONTROL — under a blank ``CUDA_VISIBLE_DEVICES`` the gate
    returns **SKIP**, never PASS, with a NON-EMPTY evidence string.

    This is what makes the repaired tests' GPU-masked branch non-vacuous:
    if the gate ever returned PASS here, the repaired tests would be
    asserting a false green and this control would go RED.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    preflight = _load_preflight_module()
    fixture = tmp_path / "wfa170_negative_control_env.json"

    result = preflight.gate_p1_gpu_env_probe(
        "p1_gpu_env_probe_wfa170_negative_control",
        "P3",
        "wfa170",
        fixture,
    )

    assert result.status == "SKIP", (
        f"the gate must SKIP in a CPU-locked session (CUDA_VISIBLE_DEVICES "
        f"is blank); got {result.status}: {result.evidence}"
    )
    assert result.status != "PASS", (
        "the gate reported PASS in a CPU-locked session — a false green"
    )

    evidence = result.evidence
    assert evidence and evidence.strip(), (
        "SKIP evidence is empty; a silent SKIP is indistinguishable from a "
        "false green"
    )
    for token in SKIP_EVIDENCE_TOKENS:
        assert token in evidence, (
            f"SKIP evidence does not name {token!r}; a SKIP that omits the "
            f"session fact or the measured host facts is not auditable. "
            f"Evidence: {evidence}"
        )

    # The CPU-checkable half is REAL: the probe artefact is still
    # written, still parses, and still carries the canonical schema.
    assert fixture.is_file(), f"probe artefact not written at {fixture}"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), (
        f"probe artefact is not a JSON object; got {type(payload).__name__}"
    )
    for key in ("gpu_count", "gpu_names", "torch_probe"):
        assert key in payload, (
            f"probe artefact missing top-level key {key!r}; got keys: "
            f"{sorted(payload.keys())}"
        )
    assert isinstance(payload["gpu_names"], list), (
        f"probe artefact['gpu_names'] is not a list; got "
        f"{type(payload['gpu_names']).__name__}"
    )
    assert isinstance(payload["torch_probe"], dict), (
        f"probe artefact['torch_probe'] is not a dict; got "
        f"{type(payload['torch_probe']).__name__}"
    )
    assert f"gpu_count={payload['gpu_count']}" in evidence, (
        f"the SKIP evidence does not carry the measured gpu_count="
        f"{payload['gpu_count']} read back from the artefact; the SKIP must "
        f"report the host facts it measured. Evidence: {evidence}"
    )


# ---------------------------------------------------------------------------
# 2 — the CPU-checkable coverage was NOT thrown away
# ---------------------------------------------------------------------------


def test_wfa170_the_repaired_tests_keep_unconditional_cpu_assertions() -> None:
    """Every repaired test function still asserts on CPU, unconditionally.

    Parsed statically: each of the 3 test functions must carry AT LEAST
    ONE ``assert`` outside the GPU-conditional branch (it runs in the
    CPU-locked session) AND AT LEAST ONE ``assert`` inside it (the GPU
    half is still asserted when a GPU is visible).
    """
    tree = _target_tree()
    functions = _target_functions(tree)

    for name in TARGET_TESTS:
        assert name in functions, (
            f"{name} is missing from the repaired file {TARGET_PATH}; the "
            f"functions present: {sorted(functions)}"
        )

    # The session predicate the repair branches on must exist.
    assert GPU_BRANCH_NAME in functions, (
        f"the repaired file does not define {GPU_BRANCH_NAME}(); the "
        f"two-sided session contract cannot be expressed without it"
    )

    for name in TARGET_TESTS:
        func = functions[name]
        unconditional, conditional = _split_asserts(func)
        assert unconditional >= 1, (
            f"{name} has NO unconditional assert: every assert in it sits "
            f"inside the GPU-conditional branch, so the test asserts "
            f"nothing in a CPU-locked session — the lazy repair"
        )
        assert conditional >= 1, (
            f"{name} has NO assert inside the GPU-conditional branch: the "
            f"GPU-visible half of the contract is not asserted at all"
        )
        # The branch must exist in the function (not merely the asserts).
        branches = [
            stmt
            for stmt in func.body
            if isinstance(stmt, ast.If) and _is_gpu_branch(stmt)
        ]
        assert branches, (
            f"{name} does not branch on {GPU_BRANCH_NAME}(); the repaired "
            f"file no longer distinguishes the two sessions"
        )


# ---------------------------------------------------------------------------
# 3 — the lazy route (a blanket skip) was NOT taken
# ---------------------------------------------------------------------------


def test_wfa170_the_repaired_tests_are_not_blanket_skipped() -> None:
    """No ``pytest.mark.skipif`` on the module or on the 3 test functions.

    A blanket skip would delete the CPU-checkable coverage, which is
    exactly what this wave forbids.
    """
    tree = _target_tree()

    module_level = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            if "pytestmark" in names and "skipif" in ast.unparse(stmt.value):
                module_level.append(ast.unparse(stmt))
    assert not module_level, (
        f"the repaired file blanket-skips itself via pytestmark: "
        f"{module_level}"
    )

    functions = _target_functions(tree)
    skipped = []
    for name in TARGET_TESTS:
        for dec in functions[name].decorator_list:
            if "skipif" in ast.unparse(dec):
                skipped.append((name, ast.unparse(dec)))
    assert not skipped, (
        f"the repaired test functions carry a blanket skipif decorator: "
        f"{skipped}"
    )


# ---------------------------------------------------------------------------
# 4 — END-TO-END: the repaired file is GREEN under the CPU lock
# ---------------------------------------------------------------------------


def test_wfa170_the_repaired_test_file_passes_under_the_cpu_lock() -> None:
    """The repaired file must report 3 passed with ``CUDA_VISIBLE_DEVICES=""``.

    This is the end-to-end statement of the repair: the 3 tests that
    were RED for a reason unrelated to what they guard are GREEN
    because the SKIP contract is now asserted honestly.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MOLMETAL_TEST_ALLOW_GPU"] = "0"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "molmetal/tests/test_wf350c1_gpu_env_probe.py",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "-q",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, (
        f"the repaired file is RED under the CPU lock (rc={proc.returncode}). "
        f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )
    assert "3 passed" in proc.stdout, (
        f"expected '3 passed' for the repaired file, got:\n"
        f"{proc.stdout[-3000:]}"
    )
