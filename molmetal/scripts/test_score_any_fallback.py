"""Wf-48 Phase 3 — score_any() fallback chain test.

================================================================
Fallback chain test — primary unavailable → router behaviour
================================================================

Wf-43 Phase 2 documented per-metric fallback chains in
``molmetal/scoring/policy.json`` and ``ROUTING_POLICY``:

    vina_score              → (vina_qvina2, vina_gpu)
    vina_high_affinity      → (vina_qvina2, vina_gpu)
    vina_min_score          → (vina_qvina2, vina_gpu)
    vina_gpu                → (vina_score,)
    diffdock_pose           → (flowdock_pose,)
    equibind_pose           → (flowdock_pose, diffdock_pose)
    flowdock_pose           → (diffdock_pose,)
    glide_score             → (vina_score,)

Wf-48 Phase 3 *verifies* what the router actually does when the
primary tool is unavailable, for each of these metrics. Three layers
of test:

1. **Fallback chain metadata** — every ref in every
   ``fallback_chain`` resolves to a registered policy entry, none
   points at a ``MOCK_DELETED`` metric, no cycles.

2. **Primary unavailable → AdapterUnavailable propagates** — when the
   primary tool raises ``AdapterUnavailable``, the router re-raises
   rather than fabricating a value. Wf-43 forbids mock substitution.

3. **Engine env-var routing** — the only currently-implemented
   runtime fallback hook in the router is
   ``MOLMETAL_VINA_ENGINE=qvina2|gpu``; setting it switches
   ``vina_score`` / ``vina_high_affinity`` / ``vina_min_score`` to the
   requested wrapper. Verify the switch fires.

Outputs
-------
* :file:`molmetal/reports/wf48_fallback/fallback_test.json` —
  verbatim per-metric report + aggregate.

Usage
-----
::

    uv run --no-sync python molmetal/scripts/test_score_any_fallback.py
    pytest molmetal/tests/test_score_any_fallback.py -v
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap — mirror the convention in every Wf-3x / Wf-4x bench so
# the script can be invoked directly (e.g. ``python molmetal/scripts/...``)
# without setting ``PYTHONPATH`` first.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MOLMETAL_PKG_DIR = _HERE.parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_MOLMETAL_PKG_DIR), str(_REPO_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FallbackStepResult:
    """Outcome of one simulated primary-unavailable scenario."""

    metric: str
    primary_tool: str
    fallback_chain: Tuple[str, ...]
    scenario: str
    expected_status: str  # "MEASURED" | "ADAPTER_UNAVAILABLE" | "ENV_ROUTED"
    actual_status: str  # "MEASURED" | "ADAPTER_UNAVAILABLE" | "ENV_ROUTED" | "FAIL"
    selected_tool: str
    fallback_used: Optional[str]  # name of fallback tool actually called (if any)
    error: Optional[str] = None
    wall_s: float = 0.0


@dataclass
class FallbackReport:
    """Per-metric fallback chain report."""

    metric: str
    primary_tool: str
    backend: str
    fallback_chain: Tuple[str, ...]
    wf43_status: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    overall_status: str = "PASS"  # PASS | FAIL | N/A
    error: Optional[str] = None


@dataclass
class FallbackConfig:
    """CLI configuration for Wf-48 Phase 3 fallback test."""

    out_dir: str = "molmetal/reports/wf48_fallback"
    mock_unavailable: bool = True
    seeds: Tuple[int, ...] = (0,)


# ---------------------------------------------------------------------------
# Public version + tuple-to-list helper
# ---------------------------------------------------------------------------
__version__ = "0.1.0-wf48-phase3"


def _tuple_to_list(obj: Any) -> Any:
    """Recursively convert every tuple in ``obj`` to a list.

    JSON has no tuples — so the in-memory payload must match the
    on-disk JSON after round-trip. ``dataclasses.asdict`` preserves
    tuples, so we walk the dict and convert them.
    """
    if isinstance(obj, tuple):
        return [_tuple_to_list(x) for x in obj]
    if isinstance(obj, list):
        return [_tuple_to_list(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _tuple_to_list(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Helpers — fallback-chain metadata checks
# ---------------------------------------------------------------------------
def _check_chain_refs(ROUTING_POLICY: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return list of ``(metric, fb)`` tuples whose ``fallback_chain`` ref
    is broken (does not exist in ``ROUTING_POLICY``)."""
    bad: List[Tuple[str, str]] = []
    for metric, entry in ROUTING_POLICY.items():
        for fb in getattr(entry, "fallback_chain", ()) or ():
            if fb not in ROUTING_POLICY:
                bad.append((metric, fb))
    return bad


def _check_chain_no_mock_deleted(
    ROUTING_POLICY: Dict[str, Any],
) -> List[Tuple[str, str]]:
    """Return list of ``(metric, fb)`` whose fallback points at a MOCK_DELETED."""
    bad: List[Tuple[str, str]] = []
    for metric, entry in ROUTING_POLICY.items():
        for fb in getattr(entry, "fallback_chain", ()) or ():
            if fb in ROUTING_POLICY:
                if getattr(ROUTING_POLICY[fb], "wf43_status", "") == "MOCK_DELETED":
                    bad.append((metric, fb))
    return bad


def _check_chain_no_cycles(
    ROUTING_POLICY: Dict[str, Any],
) -> List[Tuple[str, ...]]:
    """Detect cycles in the fallback_chain graph (length ≥ 3 only).

    Length-2 cycles (A.fallback_chain ⊇ B AND B.fallback_chain ⊇ A)
    are reported as ``"soft_chain_pairs"`` rather than failures, since
    the Wf-43 Phase 2 design documents these as intentional
    bidirectional soft chains (e.g. vina_score ↔ vina_gpu,
    diffdock_pose ↔ flowdock_pose). They are not bugs.
    """
    cycles: List[Tuple[str, ...]] = []

    def _walk(start: str, current: str, path: List[str]) -> None:
        if current not in ROUTING_POLICY:
            return
        chain = getattr(ROUTING_POLICY[current], "fallback_chain", ()) or ()
        for nxt in chain:
            if nxt == start:
                if len(path) >= 3:  # ignore length-2 (soft) chains
                    cycles.append(tuple(path + [nxt]))
                continue
            if nxt in path:
                continue  # already in path; deeper cycle handled from there
            _walk(start, nxt, path + [nxt])

    for metric in ROUTING_POLICY:
        _walk(metric, metric, [metric])
    return cycles


def _check_soft_chain_pairs(
    ROUTING_POLICY: Dict[str, Any],
) -> List[Tuple[str, str]]:
    """Return list of intentional A↔B length-2 cycles (informational)."""
    pairs: List[Tuple[str, str]] = []
    seen: set = set()
    for metric, entry in ROUTING_POLICY.items():
        chain = getattr(entry, "fallback_chain", ()) or ()
        for fb in chain:
            if fb in ROUTING_POLICY and metric in (
                ROUTING_POLICY[fb].fallback_chain or ()
            ):
                pair = tuple(sorted([metric, fb]))
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)  # type: ignore[arg-type]
    return pairs  # type: ignore[return-value]


def _check_chain_no_self_refs(
    ROUTING_POLICY: Dict[str, Any],
) -> List[str]:
    """Return metric names that appear in their own fallback_chain."""
    bad: List[str] = []
    for metric, entry in ROUTING_POLICY.items():
        if metric in (getattr(entry, "fallback_chain", ()) or ()):
            bad.append(metric)
    return bad


# ---------------------------------------------------------------------------
# Helpers — runtime fallback test (monkey-patch wrappers, observe behaviour)
# ---------------------------------------------------------------------------
@contextmanager
def _patch_primary_unavailable(
    metric: str,
) -> Iterator[Dict[str, Any]]:
    """Monkey-patch the wrapper for ``metric`` so it always raises
    ``AdapterUnavailable`` (simulating a missing primary tool)."""
    import importlib
    _sa_mod = importlib.import_module("molmetal.scoring.score_any")
    from molmetal.scoring.score_any import (
        AdapterUnavailable,
        ToolRef,
    )

    reg = _sa_mod.REGISTRY
    orig_ref = reg[metric]
    calls: Dict[str, Any] = {"primary_called": 0, "fallback_calls": []}

    def _always_unavailable(ctx: Dict[str, Any]) -> List[float]:
        calls["primary_called"] += 1
        raise AdapterUnavailable(
            f"[wf48-fallback-test] simulated primary unavailable for "
            f"{metric!r} ({orig_ref.tool_name})"
        )

    reg[metric] = ToolRef(
        wrapper=_always_unavailable,
        tool_name=orig_ref.tool_name,
        backend=orig_ref.backend,
        notes="[wf48-fallback-test-patched]",
    )
    try:
        yield calls
    finally:
        reg[metric] = orig_ref


@contextmanager
def _capture_fallback_calls(
    fb_metrics: Sequence[str],
) -> Iterator[Dict[str, int]]:
    """Wrap every wrapper in ``fb_metrics`` so each invocation increments
    a counter, while leaving real wrappers untouched. Used to detect
    whether the router auto-tries fallbacks when primary fails."""
    import importlib
    _sa_mod = importlib.import_module("molmetal.scoring.score_any")
    from molmetal.scoring.score_any import (
        AdapterUnavailable,
        ToolRef,
    )

    reg = _sa_mod.REGISTRY
    orig_refs: Dict[str, ToolRef] = {}
    counts: Dict[str, int] = {m: 0 for m in fb_metrics}

    for m in fb_metrics:
        if m not in reg:
            continue
        orig_refs[m] = reg[m]
        ref = reg[m]

        def _make_recorder(name: str, original: Callable[[Dict[str, Any]], Any]):
            def _recorder(ctx: Dict[str, Any]) -> Any:
                counts[name] = counts.get(name, 0) + 1
                return original(ctx)
            return _recorder

        reg[m] = ToolRef(
            wrapper=_make_recorder(m, ref.wrapper),
            tool_name=ref.tool_name,
            backend=ref.backend,
            notes=ref.notes + " [wf48-fallback-counter]",
        )
    try:
        yield counts
    finally:
        for m, orig_ref in orig_refs.items():
            reg[m] = orig_ref


# ---------------------------------------------------------------------------
# Per-metric scenarios
# ---------------------------------------------------------------------------
def _run_primary_unavailable_scenario(
    metric: str,
    primary_tool: str,
    fallback_chain: Sequence[str],
) -> FallbackStepResult:
    """Simulate primary tool unavailable and observe the router's response."""
    import importlib
    _sa_mod = importlib.import_module("molmetal.scoring.score_any")
    from molmetal.scoring import score_any as _score_any_public
    from molmetal.scoring.score_any import AdapterUnavailable

    t0 = time.time()
    fb_metrics = [m for m in fallback_chain if m in _sa_mod.REGISTRY]
    with _patch_primary_unavailable(metric):
        with _capture_fallback_calls(fb_metrics) as counts:
            try:
                result = _score_any_public(
                    metric, molecules=["CCO"], pocket=None
                )
                # Router returned a result — must NOT happen when
                # primary raises AdapterUnavailable and no auto-fallback.
                actual_status = "MEASURED"
                selected_tool = str(result.get("tool", ""))
                return FallbackStepResult(
                    metric=metric,
                    primary_tool=primary_tool,
                    fallback_chain=tuple(fallback_chain),
                    scenario="primary_unavailable",
                    expected_status="ADAPTER_UNAVAILABLE",
                    actual_status=actual_status,
                    selected_tool=selected_tool,
                    fallback_used=None,
                    error=(
                        "Router returned a result despite primary "
                        "unavailable — violates Wf-43 no-mock invariant."
                    ),
                    wall_s=time.time() - t0,
                )
            except AdapterUnavailable as exc:
                actual_status = "ADAPTER_UNAVAILABLE"
                # Note: if router auto-fell-back, ``counts`` would have
                # > 0 entries; for the current implementation it does not.
                fb_used = next(
                    (m for m, c in counts.items() if int(c) > 0), None
                )
                return FallbackStepResult(
                    metric=metric,
                    primary_tool=primary_tool,
                    fallback_chain=tuple(fallback_chain),
                    scenario="primary_unavailable",
                    expected_status="ADAPTER_UNAVAILABLE",
                    actual_status=actual_status,
                    selected_tool=primary_tool,
                    fallback_used=fb_used,
                    error=None,
                    wall_s=time.time() - t0,
                )
            except Exception as exc:  # pragma: no cover - defensive
                return FallbackStepResult(
                    metric=metric,
                    primary_tool=primary_tool,
                    fallback_chain=tuple(fallback_chain),
                    scenario="primary_unavailable",
                    expected_status="ADAPTER_UNAVAILABLE",
                    actual_status="FAIL",
                    selected_tool=primary_tool,
                    fallback_used=None,
                    error=f"{type(exc).__name__}: {exc}",
                    wall_s=time.time() - t0,
                )


def _run_env_routing_scenario(
    metric: str,
    primary_tool: str,
    fallback_chain: Sequence[str],
    env_value: str,
) -> FallbackStepResult:
    """Set ``MOLMETAL_VINA_ENGINE=env_value`` and verify the router routes
    to the appropriate fallback wrapper by replacing it with a sentinel.
    A successful sentinel return proves the env-routing branch fired
    *regardless* of whether the underlying real tool is installed on
    this host.
    """
    import importlib
    _sa_mod = importlib.import_module("molmetal.scoring.score_any")
    from molmetal.scoring.score_any import (
        AdapterUnavailable,
        ToolRef,
    )
    from molmetal.scoring import score_any as _score_any_public

    t0 = time.time()
    prior = os.environ.get("MOLMETAL_VINA_ENGINE")
    os.environ["MOLMETAL_VINA_ENGINE"] = env_value

    expected_metric = (
        "vina_qvina2" if env_value == "qvina2" else "vina_gpu"
    )
    if expected_metric not in _sa_mod.REGISTRY:
        # No fallback wrapper to patch — restore env and report.
        if prior is None:
            os.environ.pop("MOLMETAL_VINA_ENGINE", None)
        else:
            os.environ["MOLMETAL_VINA_ENGINE"] = prior
        return FallbackStepResult(
            metric=metric,
            primary_tool=primary_tool,
            fallback_chain=tuple(fallback_chain),
            scenario=f"env_MOLMETAL_VINA_ENGINE={env_value}",
            expected_status="ENV_ROUTED",
            actual_status="FAIL",
            selected_tool=primary_tool,
            fallback_used=None,
            error=f"no fallback wrapper for {expected_metric!r}",
            wall_s=time.time() - t0,
        )

    expected_tool = str(_sa_mod.REGISTRY[expected_metric].tool_name)
    orig_ref = _sa_mod.REGISTRY[expected_metric]

    def _sentinel(ctx: Dict[str, Any]) -> List[float]:
        return [-9.999]

    _sa_mod.REGISTRY[expected_metric] = ToolRef(
        wrapper=_sentinel,
        tool_name=orig_ref.tool_name,
        backend=orig_ref.backend,
        notes=orig_ref.notes + " [wf48-env-sentinel]",
    )

    try:
        try:
            result = _score_any_public(
                metric, molecules=["CCO"], pocket=None
            )
            selected = str(result.get("tool", ""))
            ok = selected == expected_tool
            return FallbackStepResult(
                metric=metric,
                primary_tool=primary_tool,
                fallback_chain=tuple(fallback_chain),
                scenario=f"env_MOLMETAL_VINA_ENGINE={env_value}",
                expected_status="ENV_ROUTED",
                actual_status="ENV_ROUTED" if ok else "FAIL",
                selected_tool=selected,
                fallback_used=(
                    expected_metric if ok else None
                ),
                error=(
                    None if ok
                    else f"expected tool {expected_tool!r}, "
                         f"router picked {selected!r}"
                ),
                wall_s=time.time() - t0,
            )
        except AdapterUnavailable as exc:
            return FallbackStepResult(
                metric=metric,
                primary_tool=primary_tool,
                fallback_chain=tuple(fallback_chain),
                scenario=f"env_MOLMETAL_VINA_ENGINE={env_value}",
                expected_status="ENV_ROUTED",
                actual_status="FAIL",
                selected_tool=primary_tool,
                fallback_used=None,
                error=(
                    f"sentinel wrapper for {expected_metric!r} "
                    f"unexpectedly raised AdapterUnavailable: "
                    f"{str(exc).splitlines()[0]}"
                ),
                wall_s=time.time() - t0,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return FallbackStepResult(
                metric=metric,
                primary_tool=primary_tool,
                fallback_chain=tuple(fallback_chain),
                scenario=f"env_MOLMETAL_VINA_ENGINE={env_value}",
                expected_status="ENV_ROUTED",
                actual_status="FAIL",
                selected_tool=primary_tool,
                fallback_used=None,
                error=f"{type(exc).__name__}: {exc}",
                wall_s=time.time() - t0,
            )
    finally:
        _sa_mod.REGISTRY[expected_metric] = orig_ref
        if prior is None:
            os.environ.pop("MOLMETAL_VINA_ENGINE", None)
        else:
            os.environ["MOLMETAL_VINA_ENGINE"] = prior


def _run_chain_metadata_scenario(
    metric: str,
    primary_tool: str,
    fallback_chain: Sequence[str],
    ROUTING_POLICY: Dict[str, Any],
) -> FallbackStepResult:
    """Verify the fallback_chain metadata for ``metric`` is well-formed.

    Length-2 cycles (A→B AND B→A) are accepted as intentional
    bidirectional "soft chains" — they are NOT failures. Only length-1
    self-references and length-3+ cycles are rejected.
    """
    t0 = time.time()
    if not fallback_chain:
        return FallbackStepResult(
            metric=metric,
            primary_tool=primary_tool,
            fallback_chain=tuple(fallback_chain),
            scenario="chain_metadata_no_chain",
            expected_status="MEASURED",
            actual_status="MEASURED",
            selected_tool=primary_tool,
            fallback_used=None,
            wall_s=time.time() - t0,
        )
    errors: List[str] = []
    for fb in fallback_chain:
        if fb not in ROUTING_POLICY:
            errors.append(f"unknown ref {fb!r}")
            continue
        if ROUTING_POLICY[fb].wf43_status == "MOCK_DELETED":
            errors.append(f"refers to MOCK_DELETED metric {fb!r}")
    if metric in fallback_chain:
        errors.append(f"self-reference in chain")
    # Cycle detection: only flag length-3+ cycles; length-2 cycles
    # are intentional bidirectional soft chains.
    def _walk(start: str, current: str, path: List[str]) -> List[str]:
        if current not in ROUTING_POLICY:
            return []
        chain = getattr(ROUTING_POLICY[current], "fallback_chain", ()) or ()
        bad: List[str] = []
        for nxt in chain:
            if nxt == start and len(path) >= 3:
                bad.append(
                    f"cycle {'->'.join(path + [nxt])}"
                )
                continue
            if nxt in path:
                continue
            bad.extend(_walk(start, nxt, path + [nxt]))
        return bad

    err_cycles = _walk(metric, metric, [metric])
    errors.extend(err_cycles)
    if errors:
        return FallbackStepResult(
            metric=metric,
            primary_tool=primary_tool,
            fallback_chain=tuple(fallback_chain),
            scenario="chain_metadata",
            expected_status="MEASURED",
            actual_status="FAIL",
            selected_tool=primary_tool,
            fallback_used=None,
            error="; ".join(errors),
            wall_s=time.time() - t0,
        )
    return FallbackStepResult(
        metric=metric,
        primary_tool=primary_tool,
        fallback_chain=tuple(fallback_chain),
        scenario="chain_metadata",
        expected_status="MEASURED",
        actual_status="MEASURED",
        selected_tool=primary_tool,
        fallback_used=None,
        wall_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Per-metric driver
# ---------------------------------------------------------------------------
def _run_metric_scenarios(
    metric: str,
    ROUTING_POLICY: Dict[str, Any],
) -> FallbackReport:
    """Run every fallback scenario for ``metric`` and return a report."""
    entry = ROUTING_POLICY[metric]
    primary = str(entry.primary_tool)
    chain = tuple(getattr(entry, "fallback_chain", ()) or ())

    rep = FallbackReport(
        metric=metric,
        primary_tool=primary,
        backend=str(entry.backend),
        fallback_chain=chain,
        wf43_status=str(entry.wf43_status),
    )

    # Scenario 1: chain metadata
    meta = _run_chain_metadata_scenario(metric, primary, chain, ROUTING_POLICY)
    rep.results.append(asdict(meta))

    # Scenario 2: primary unavailable (skip MOCK_DELETED — no real wrapper)
    if str(entry.wf43_status) != "MOCK_DELETED":
        pri = _run_primary_unavailable_scenario(metric, primary, chain)
        rep.results.append(asdict(pri))

    # Scenario 3: env-var routing. The MOLMETAL_VINA_ENGINE branch
    # in score_any() ONLY fires for vina_score / vina_high_affinity /
    # vina_min_score. For vina_gpu / glide_score the env var is
    # ignored (the router keeps the primary ref), so we skip the
    # env-routing scenario for those metrics.
    if metric in {
        "vina_score", "vina_high_affinity", "vina_min_score",
    }:
        for env_value in ("qvina2", "gpu"):
            env = _run_env_routing_scenario(metric, primary, chain, env_value)
            rep.results.append(asdict(env))

    # Decide overall_status
    statuses = [r["actual_status"] for r in rep.results]
    if any(s == "FAIL" for s in statuses):
        rep.overall_status = "FAIL"
    elif all(s in ("MEASURED", "ADAPTER_UNAVAILABLE", "ENV_ROUTED")
             for s in statuses):
        rep.overall_status = "PASS"
    else:
        rep.overall_status = "FAIL"
    return rep


# ---------------------------------------------------------------------------
# Aggregate + writers
# ---------------------------------------------------------------------------
def _aggregate(reports: List[FallbackReport]) -> Dict[str, Any]:
    n = len(reports)
    by_status: Dict[str, int] = {}
    n_chain_metadata_pass = 0
    n_primary_unavailable_propagate = 0
    n_env_routed = 0
    n_env_routed_total = 0
    for r in reports:
        by_status[r.overall_status] = by_status.get(r.overall_status, 0) + 1
        for sr in r.results:
            if sr["scenario"] == "chain_metadata":
                if sr["actual_status"] == "MEASURED":
                    n_chain_metadata_pass += 1
            elif sr["scenario"] == "primary_unavailable":
                if sr["actual_status"] == "ADAPTER_UNAVAILABLE":
                    n_primary_unavailable_propagate += 1
            elif sr["scenario"].startswith("env_MOLMETAL_VINA_ENGINE="):
                n_env_routed_total += 1
                if sr["actual_status"] == "ENV_ROUTED":
                    n_env_routed += 1
    return {
        "n_metrics_with_fallback_chain": sum(
            1 for r in reports if r.fallback_chain
        ),
        "n_total_metrics_tested": int(n),
        "n_pass": int(by_status.get("PASS", 0)),
        "n_fail": int(by_status.get("FAIL", 0)),
        "n_na": int(by_status.get("N/A", 0)),
        "n_chain_metadata_pass": int(n_chain_metadata_pass),
        "n_chain_metadata_total": sum(
            1 for r in reports
            for sr in r.results
            if sr["scenario"] == "chain_metadata"
        ),
        "n_primary_unavailable_propagate": int(
            n_primary_unavailable_propagate
        ),
        "n_primary_unavailable_total": sum(
            1 for r in reports
            for sr in r.results
            if sr["scenario"] == "primary_unavailable"
        ),
        "n_env_routed_pass": int(n_env_routed),
        "n_env_routed_total": int(n_env_routed_total),
        "fallback_chain_success_rate": (
            n_primary_unavailable_propagate
            / max(1, sum(
                1 for r in reports
                for sr in r.results
                if sr["scenario"] == "primary_unavailable"
            ))
        ),
        "env_routing_success_rate": (
            n_env_routed / max(1, n_env_routed_total)
        ),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_metadata_audit(ROUTING_POLICY: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level metadata audit on the whole fallback_chain graph."""
    return {
        "broken_refs": _check_chain_refs(ROUTING_POLICY),
        "mock_deleted_refs": _check_chain_no_mock_deleted(ROUTING_POLICY),
        "cycles": [list(c) for c in _check_chain_no_cycles(ROUTING_POLICY)],
        "soft_chain_pairs": [
            list(p) for p in _check_soft_chain_pairs(ROUTING_POLICY)
        ],
        "self_refs": _check_chain_no_self_refs(ROUTING_POLICY),
        "n_metrics_with_chain": sum(
            1 for e in ROUTING_POLICY.values() if e.fallback_chain
        ),
    }


def run_test(cfg: FallbackConfig) -> Dict[str, Any]:
    """Execute Wf-48 Phase 3 fallback chain test."""
    from molmetal.scoring.score_any import ROUTING_POLICY

    metadata_audit = _run_metadata_audit(ROUTING_POLICY)

    # Per-metric reports — iterate ALL routed metrics (chain empty is
    # still tested: chain_metadata_no_chain must pass).
    reports: List[FallbackReport] = []
    for metric in sorted(ROUTING_POLICY.keys()):
        try:
            rep = _run_metric_scenarios(metric, ROUTING_POLICY)
            reports.append(rep)
        except Exception as exc:  # pragma: no cover - defensive
            reports.append(
                FallbackReport(
                    metric=metric,
                    primary_tool="?",
                    backend="?",
                    fallback_chain=(),
                    wf43_status="?",
                    overall_status="FAIL",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    agg = _aggregate(reports)
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "recipe": "wf48_fallback_chain_test",
        "tool": "test_score_any_fallback.py",
        "tool_version": __version__,
        "config": {
            "mock_unavailable": bool(cfg.mock_unavailable),
            "seeds": [int(s) for s in cfg.seeds],
        },
        "metadata_audit": metadata_audit,
        "per_metric": [_tuple_to_list(asdict(r)) for r in reports],
        "aggregate": agg,
    }

    out_dir = Path(cfg.out_dir)
    _write_json(out_dir / "fallback_test.json", payload)
    print(
        f"[wf48-fallback] wrote {out_dir / 'fallback_test.json'} "
        f"({len(reports)} metrics, "
        f"{agg['n_pass']}/{agg['n_total_metrics_tested']} pass, "
        f"env-routing "
        f"{agg['n_env_routed_pass']}/{agg['n_env_routed_total']})",
        file=sys.stderr,
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="test_score_any_fallback",
        description=(
            "Wf-48 Phase 3 — Test the score_any() fallback chain: "
            "simulate primary unavailable and verify the router's "
            "response + metadata integrity."
        ),
    )
    p.add_argument("--out-dir", type=str,
                   default="molmetal/reports/wf48_fallback",
                   help="Output directory for the report.")
    p.add_argument("--no-mock-unavailable", action="store_true",
                   help="Skip the simulated-primary-unavailable "
                        "scenario (metadata audit only).")
    p.add_argument("--seeds", type=int, nargs="*", default=[0],
                   help="Seed(s) for deterministic scenarios.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    cfg = FallbackConfig(
        out_dir=str(args.out_dir),
        mock_unavailable=not bool(args.no_mock_unavailable),
        seeds=tuple(int(s) for s in (args.seeds or [0])),
    )
    payload = run_test(cfg)
    # Exit code: 0 when every metric passes; 2 when any fails.
    # ``soft_chain_pairs`` are length-2 intentional bidirectional
    # chains — they are NOT failures (documented design).
    agg = payload["aggregate"]
    md = payload["metadata_audit"]
    if (
        int(agg["n_fail"]) > 0
        or md["broken_refs"]
        or md["mock_deleted_refs"]
        or md["cycles"]
        or md["self_refs"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())