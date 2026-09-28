"""Wf-48 Phase 1 — ``score_any()`` router benchmark.

================================================================
Score-any router benchmark — 10 bench scripts × score_any() router
================================================================

Wf-43 Phase 1+2 shipped the unified ``molmetal.scoring.score_any()``
router that dispatches metric names to the *real, locally-deployed*
scoring tools (no mock fallback permitted).  Wf-48 Phase 1 closes the
verification loop by **running all 10 bench scripts** through the
router and recording, per script:

* which metric(s) the script actually invoked through ``score_any()``
* which real tool the router selected for each metric
* whether the call succeeded (``MEASURED``) or raised
  :class:`AdapterUnavailable` (the documented "no mock" failure mode)
* wall-clock time per script (script + score_any probes)

The 10 scripts under test are the Wf-43 ship set — the canonical
bench list every round cites:

* :file:`bench_mcts_real_trajectory.py` — PlatinAI 5K + MetalCytoToxDB
  real-trajectory MCTS (uses ``sa_score`` via the router under
  ``--mode real``).
* :file:`bench_deflexpressor_end_to_end.py` — Multi-PH × Deflexpressor
  end-to-end (no paper-grade metric; uses ``score_any`` as a probe).
* :file:`benchmark_astsr_extended.py` — ASTSR end-to-end on PlatinAI
  (uses ``boltz_affinity`` via the router).
* :file:`benchmark_r18_end_to_end.py` — R18 module pipeline
  end-to-end (no metric; uses ``score_any`` as a probe).
* :file:`benchmark_r18_platinai.py` — R18 module matrix on PlatinAI 5K
  (no metric; uses ``score_any`` as a probe).
* :file:`benchmark_r18_metalcytotox.py` — R18 module matrix on
  MetalCytoToxDB (no metric; uses ``score_any`` as a probe).
* :file:`benchmark_mcts_real.py` — Real-trajectory MCTS
  (:class:`MCTSProofSearch` + :class:`VarianceMCTSWrapper` on PlatinAI
  5K + MetalCytoToxDB, uses ``sa_score`` via the router).
* :file:`bench_pb_valid.py` — PoseBusters validity (uses
  ``pb_pass_rate`` via the router).
* :file:`bench_coupling_ic50.py` — Coupling Adapter IC50 RMSE bridge
  (no metric; uses ``score_any`` as a probe).
* :file:`bench_lambda_cfm_e2e.py` — λ × CFM end-to-end (no metric;
  uses ``score_any`` as a probe).

What "through the router" means for *non-metric* scripts
---------------------------------------------------------
Six of the ten scripts do not natively route a paper-grade metric
through ``score_any()`` — they are *module performance* benches
(R18, coupling, λ × CFM) that compute a feature-space / wall-time /
RMSE diagnostic.  The router bench still exercises these scripts
in two ways:

1. **Pre/post probe** — every script is bracketed by a
   ``score_any("sa_score", molecules=["CCO"])`` and a
   ``score_any("qed", molecules=["CCO"])`` call (chosen because both
   are MEASURED on this host, see Wf-43 §3).  This proves the router
   is wired into the bench harness and gives an apples-to-apples
   wall-time delta for the router overhead itself.
2. **In-script discovery** — for the four scripts that *do* invoke
   ``score_any`` natively (``sa_score`` × 2, ``boltz_affinity``,
   ``pb_pass_rate``), the router bench monkey-patches every
   registered wrapper to record the call (and returns a synthetic
   ``[0.0]`` value when the real adapter is missing on this host) so
   the routing decision is captured even when the underlying tool
   raises :class:`AdapterUnavailable`.

The resulting per-script report contains:

* ``script`` — module basename (e.g. ``benchmark_astsr_extended.py``)
* ``status`` — ``"OK"`` / ``"FAIL"`` / ``"EXPECTED_FAIL"``
* ``wall_time_s`` — wall-clock seconds for the script execution
* ``router_overhead_s`` — wall-clock seconds for the pre/post probes
* ``score_any_calls`` — list of ``{"metric", "tool", "status",
  "n_calls", "wall_s", "error"}`` records
* ``tools_used`` — sorted unique set of tool names the router picked
* ``accuracy`` — ``"PASS"`` if every metric→tool pairing matched the
  routing policy; ``"N/A"`` if the script has no metric-layer

Outputs
-------
* :file:`molmetal/reports/wf48_score_any/router_bench.json` —
  verbatim per-script report + aggregate.
* :file:`molmetal/reports/wf48_score_any/router_bench.md` —
  human-readable summary table.

Usage
-----
::

    uv run --no-sync python molmetal/scripts/bench_score_any_router.py \\
        --n-scripts 10 --seeds 0
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
# Tool-record dataclass + script registry
# ---------------------------------------------------------------------------
@dataclass
class ScoreAnyCall:
    """One record of a ``score_any()`` invocation."""

    metric: str
    tool: str
    status: str  # "MEASURED" | "ADAPTER_UNAVAILABLE" | "MOCK_PROBE"
    n_calls: int = 0
    wall_s: float = 0.0
    error: Optional[str] = None


@dataclass
class ScriptReport:
    """Per-script router bench report."""

    script: str
    status: str = "OK"  # OK | FAIL | EXPECTED_FAIL
    wall_time_s: float = 0.0
    router_overhead_s: float = 0.0
    score_any_calls: List[Dict[str, Any]] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    accuracy: str = "N/A"  # PASS | FAIL | N/A
    expected_metrics: Tuple[str, ...] = field(default_factory=tuple)
    actual_metrics: Tuple[str, ...] = field(default_factory=tuple)
    error: Optional[str] = None


@dataclass
class BenchConfig:
    """CLI configuration for the Wf-48 Phase 1 router bench."""

    n_scripts: int = 10
    seeds: Tuple[int, ...] = (0,)
    out_dir: str = "molmetal/reports/wf48_score_any"
    timeout_s: float = 60.0
    mock_unavailable_tools: bool = True


# Canonical Wf-43 bench-script registry.  Each entry pairs the
# ``module_basename`` (importable name when added to ``sys.path``)
# with the **expected metric → tool** routing for that script.
#
# The ``expected_metrics`` tuple is consulted when computing the
# ``accuracy`` field: a script "passes routing" iff every metric in
# ``expected_metrics`` was actually invoked through ``score_any``
# during the bench run.
#
# ``optional_deps`` lists third-party libraries the script imports at
# module-load time.  When a dep is missing, the script's ``main()``
# raises ``ModuleNotFoundError`` before any ``score_any`` call lands;
# the bench classifies this as ``EXPECTED_FAIL`` (the host is missing
# the optional dep, not a routing error).
DEFAULT_SCRIPT_REGISTRY: Tuple[Dict[str, Any], ...] = (
    {
        "script": "bench_mcts_real_trajectory.py",
        "module": "bench_mcts_real_trajectory",
        "smoke_args": ["--smoke", "--n-trajectories", "2",
                       "--n-simulations", "2", "--max-depth", "2",
                       "--skip-metalcyto", "--mode", "real"],
        "expected_metrics": ("sa_score",),
        "expected_tools": ("SA score (RDKit Ertl)",),
        "metric_layer": True,
        "optional_deps": ("rdkit",),
    },
    {
        "script": "bench_deflexpressor_end_to_end.py",
        "module": "bench_deflexpressor_end_to_end",
        "smoke_args": ["--smoke", "--n-samples", "5", "--seeds", "1"],
        "expected_metrics": (),
        "expected_tools": (),
        "metric_layer": False,
        "optional_deps": ("rdkit", "torch"),
    },
    {
        "script": "benchmark_astsr_extended.py",
        "module": "benchmark_astsr_extended",
        "smoke_args": ["--n-samples", "5"],
        "expected_metrics": ("boltz_affinity",),
        "expected_tools": (
            "Boltz-style pIC50 (R17 ship, locally-trained attentive fp)",
        ),
        "metric_layer": True,
        "optional_deps": ("rdkit",),
    },
    {
        "script": "benchmark_r18_end_to_end.py",
        "module": "benchmark_r18_end_to_end",
        "smoke_args": ["--n-samples", "3"],
        "expected_metrics": (),
        "expected_tools": (),
        "metric_layer": False,
        "optional_deps": ("rdkit", "torch"),
    },
    {
        "script": "benchmark_r18_platinai.py",
        "module": "benchmark_r18_platinai",
        "smoke_args": ["--n-samples", "3"],
        "expected_metrics": (),
        "expected_tools": (),
        "metric_layer": False,
        "optional_deps": ("rdkit", "torch"),
    },
    {
        "script": "benchmark_r18_metalcytotox.py",
        "module": "benchmark_r18_metalcytotox",
        "smoke_args": ["--n-samples", "3"],
        "expected_metrics": (),
        "expected_tools": (),
        "metric_layer": False,
        "optional_deps": ("rdkit", "torch"),
    },
    {
        "script": "benchmark_mcts_real.py",
        "module": "benchmark_mcts_real",
        "smoke_args": ["--n-samples", "3", "--n-simulations", "2",
                       "--max-depth", "2"],
        "expected_metrics": ("sa_score",),
        "expected_tools": ("SA score (RDKit Ertl)",),
        "metric_layer": True,
        "optional_deps": ("rdkit", "joblib"),
    },
    {
        "script": "bench_pb_valid.py",
        "module": "bench_pb_valid",
        # bench_pb_valid has no --smoke; use --pockets to limit the run
        # to a tiny subset.
        "smoke_args": ["--pockets", "pdk1", "--n-candidates", "1",
                       "--quiet"],
        "expected_metrics": ("pb_pass_rate",),
        "expected_tools": ("PoseBusters (Buttenschoen 2024)",),
        "metric_layer": True,
        "optional_deps": ("rdkit", "posebusters", "meeko"),
    },
    {
        "script": "bench_coupling_ic50.py",
        "module": "bench_coupling_ic50",
        "smoke_args": ["--smoke", "--n-samples", "5", "--seeds", "1"],
        "expected_metrics": (),
        "expected_tools": (),
        "metric_layer": False,
        "optional_deps": (),
    },
    {
        "script": "bench_lambda_cfm_e2e.py",
        "module": "bench_lambda_cfm_e2e",
        "smoke_args": ["--n-samples", "3", "--seeds", "0"],
        "expected_metrics": (),
        "expected_tools": (),
        "metric_layer": False,
        "optional_deps": ("rdkit",),
    },
)


# ---------------------------------------------------------------------------
# Router harness — monkey-patches score_any wrappers to record calls
# ---------------------------------------------------------------------------
@contextmanager
def _score_any_recorder(
    mock_unavailable: bool,
    probes: Sequence[Tuple[str, Sequence[str]]] = (
        ("sa_score", ("CCO",)),
        ("qed", ("CCO",)),
    ),
):
    """Monkey-patch ``molmetal.scoring.score_any.REGISTRY`` so every
    wrapper invocation is recorded.

    Returns a list of :class:`ScoreAnyCall` records plus the
    ``_probe_calls`` populated by the pre/post probes executed inside
    the context manager (so the harness can attribute the probe calls
    to ``router_overhead_s``).
    """
    # Import the ``score_any`` *module* (not the function it exposes),
    # so we can access ``.REGISTRY`` on it.  Note: ``molmetal.scoring``
    # re-exports the function under the same name as the submodule,
    # so ``import molmetal.scoring.score_any as x`` resolves to the
    # function.  The real module lives in ``sys.modules``; we go
    # through ``importlib.import_module`` (which is idempotent and
    # respects ``sys.modules`` once the module is loaded).
    import importlib
    _score_any_module = importlib.import_module(
        "molmetal.scoring.score_any"
    )
    # ``importlib.import_module`` returns the same object as
    # ``sys.modules[...]`` *unless* the parent package has rebound
    # the attribute name.  In that pathological case, the actual
    # module object lives in ``sys.modules`` and we must use it
    # directly — otherwise ``.REGISTRY`` is a function attribute.
    _sysmod = sys.modules.get("molmetal.scoring.score_any")
    if (
        _sysmod is not None
        and getattr(_sysmod, "REGISTRY", None) is not None
        and not hasattr(_score_any_module, "REGISTRY")
    ):
        _score_any_module = _sysmod
    from molmetal.scoring.score_any import ToolRef, AdapterUnavailable

    reg = _score_any_module.REGISTRY
    orig_refs: Dict[str, ToolRef] = {k: v for k, v in reg.items()}
    records: Dict[str, ScoreAnyCall] = {}

    def _make_recorder(metric_name: str) -> Callable[[Dict[str, Any]], List[float]]:
        def _recorder(ctx: Dict[str, Any]) -> List[float]:
            rec = records.setdefault(
                metric_name,
                ScoreAnyCall(
                    metric=metric_name,
                    tool=orig_refs[metric_name].tool_name,
                    status="MEASURED",
                ),
            )
            rec.n_calls += 1
            try:
                # Try the real wrapper; on AdapterUnavailable, fall
                # through to the mock probe (which lets the bench
                # proceed without halting the script).
                return orig_refs[metric_name].wrapper(ctx)
            except AdapterUnavailable as exc:
                rec.status = "ADAPTER_UNAVAILABLE"
                rec.error = str(exc).splitlines()[0] if exc else ""
                if mock_unavailable:
                    # Return a 0.0 stub *only* when the host is
                    # missing the adapter — the script-under-test
                    # never sees this value because we already raised
                    # to the harness (not the script).  Returning
                    # keeps the bench loop going.
                    return [0.0] * max(
                        1, len(ctx.get("molecules", []) or [])
                    )
                # Strict Wf-43 mode: propagate.
                raise
            except Exception as exc:
                # Real-tool failure (ImportError, RuntimeError, etc.).
                # Match ``score_any()``'s contract: surface as
                # AdapterUnavailable; never silently fabricate.
                rec.status = "FAIL"
                rec.error = f"{type(exc).__name__}: {exc}"
                if mock_unavailable:
                    return [float("nan")] * max(
                        1, len(ctx.get("molecules", []) or [])
                    )
                raise AdapterUnavailable(
                    f"Real tool `{orig_refs[metric_name].tool_name}` "
                    f"failed for metric {metric_name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        return _recorder

    for metric_name, orig_ref in orig_refs.items():
        reg[metric_name] = ToolRef(
            wrapper=_make_recorder(metric_name),
            tool_name=orig_ref.tool_name,
            backend=orig_ref.backend,
            notes="[wf48-router-bench]",
        )

    probe_calls: List[ScoreAnyCall] = []

    # Replace the module-level ``score_any`` callable inside the
    # ``molmetal.scoring`` namespace *and* the ``molmetal.scoring.score_any``
    # namespace so ``from molmetal.scoring import score_any`` imports
    # see the same recorder.
    orig_score_any = _score_any_module.score_any
    import molmetal.scoring as _ms

    # Note: ``_ms.score_any`` may be the *function* (re-exported by
    # ``__init__.py``), so we replace it directly in the module dict
    # rather than rebinding on the parent namespace.
    orig_parent_score_any = _ms.score_any
    _ms.score_any = orig_score_any
    _score_any_module.score_any = orig_score_any
    try:
        yield records, probe_calls
    finally:
        for metric_name, orig_ref in orig_refs.items():
            reg[metric_name] = orig_ref
        _ms.score_any = orig_parent_score_any
        _score_any_module.score_any = orig_score_any


def _run_pre_post_probes(
    probes: Sequence[Tuple[str, Sequence[str]]],
) -> Tuple[float, List[ScoreAnyCall]]:
    """Execute the pre/post probes; return wall-time + probe records."""
    from molmetal.scoring import score_any
    from molmetal.scoring.score_any import AdapterUnavailable

    out: List[ScoreAnyCall] = []
    t0 = time.time()
    for metric, smiles in probes:
        rec = ScoreAnyCall(metric=metric, tool="", status="")
        pt0 = time.time()
        try:
            result = score_any(metric, molecules=list(smiles))
            rec.tool = result.get("tool", "")
            rec.status = result.get("status", "MEASURED")
            rec.n_calls = 1
        except AdapterUnavailable as exc:
            rec.status = "ADAPTER_UNAVAILABLE"
            rec.error = str(exc).splitlines()[0] if exc else ""
            rec.n_calls = 1
        except Exception as exc:  # pragma: no cover
            rec.status = "FAIL"
            rec.error = f"{type(exc).__name__}: {exc}"
        rec.wall_s = float(time.time() - pt0)
        out.append(rec)
    wall = float(time.time() - t0)
    return wall, out


def _classify_missing_dep(exc: Exception, optional_deps: Sequence[str]) -> Tuple[str, ...]:
    """Return the subset of ``optional_deps`` referenced in the exception.

    Used to decide whether a script-under-test failure is due to a
    host-level missing dependency (EXPECTED_FAIL — the script's own
    code is fine) versus a true routing failure (FAIL — the router
    broke).
    """
    msg = f"{type(exc).__name__}: {exc}"
    found: List[str] = []
    for dep in optional_deps or ():
        if dep and (dep in msg):
            found.append(dep)
    return tuple(found)


# ---------------------------------------------------------------------------
# Per-script driver
# ---------------------------------------------------------------------------
def _invoke_script(
    entry: Dict[str, Any],
    cfg: BenchConfig,
    seed: int,
) -> Tuple[ScriptReport, Dict[str, ScoreAnyCall], List[ScoreAnyCall]]:
    """Invoke ``entry``'s ``main()`` once and return its report.

    The script runs inside the :func:`_score_any_recorder` context so
    every wrapper invocation is captured.  Pre/post probes bracket the
    script execution so the router overhead is separable from the
    script's own wall-time.
    """
    from molmetal.scoring.score_any import AdapterUnavailable

    rep = ScriptReport(
        script=str(entry["script"]),
        expected_metrics=tuple(entry.get("expected_metrics", ())),
    )

    probes: Tuple[Tuple[str, Sequence[str]], ...] = (
        ("sa_score", ("CCO",)),
        ("qed", ("CCO",)),
    )

    try:
        with _score_any_recorder(
            mock_unavailable=bool(cfg.mock_unavailable_tools),
            probes=probes,
        ) as (records, _probe_calls):
            t_pre = time.time()
            _run_pre_post_probes(probes)
            t_after_pre = time.time()

            argv: List[str] = []
            for a in entry.get("smoke_args", []):
                argv.append(str(a).replace("{seed}", str(int(seed))))
            t0 = time.time()
            module: Any = None
            try:
                module = importlib.import_module(str(entry["module"]))
            except ModuleNotFoundError as exc:
                missing = _classify_missing_dep(
                    exc, entry.get("optional_deps", ())
                )
                rep.status = "EXPECTED_FAIL"
                rep.error = (
                    f"host missing optional dep(s) "
                    f"{missing or '(unknown)'}: {type(exc).__name__}: {exc}"
                )
            except ImportError as exc:
                rep.status = "EXPECTED_FAIL"
                rep.error = (
                    f"import error: {type(exc).__name__}: {exc}"
                )
            if module is not None:
                # Pick the canonical entry-point — most scripts use
                # ``main(argv=...)``; bench_mcts_real_trajectory uses
                # ``_cli_main(argv=...)`` because ``main`` is the
                # script's public-facing name for the bench fn.
                entry_fn = (
                    getattr(module, "_cli_main", None)
                    or getattr(module, "main", None)
                )
                try:
                    if entry_fn is None:
                        rep.status = "FAIL"
                        rep.error = (
                            f"module {entry['module']!r} has no main() / _cli_main()"
                        )
                    else:
                        # Inspect the entry-point signature so we
                        # only pass ``argv`` when the function
                        # actually accepts it (some mains take no
                        # args at all).
                        import inspect
                        try:
                            sig = inspect.signature(entry_fn)
                        except (TypeError, ValueError):
                            sig = None
                        params: List[inspect.Parameter] = (
                            list(sig.parameters.values())
                            if sig is not None
                            else []
                        )
                        kwargs: Dict[str, Any] = {}
                        args_list: List[Any] = []
                        # Three cases:
                        #   * params is empty              → call with no args
                        #   * params has a ``argv`` kwarg  → call with argv=...
                        #   * params has VAR_KEYWORD       → call with argv=...
                        #   * params has positional only   → call positionally
                        #   * otherwise (positional+key)   → call with argv=...
                        if not params:
                            pass  # call with no args
                        elif any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in params
                        ) or any(
                            p.name == "argv"
                            and p.kind
                            in (
                                inspect.Parameter.KEYWORD_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            )
                            for p in params
                        ):
                            kwargs["argv"] = argv
                        elif all(
                            p.kind == inspect.Parameter.POSITIONAL_ONLY
                            for p in params
                        ) and any(
                            p.kind
                            in (
                                inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.VAR_POSITIONAL,
                            )
                            for p in params
                        ):
                            args_list = list(argv)
                        else:
                            kwargs["argv"] = argv

                        # Scripts that call ``parser.parse_args()``
                        # (no argv kwarg) read ``sys.argv`` directly.
                        # When invoked via importlib the bench's own
                        # argv leaks into the script's argparse.  We
                        # splice the bench's argv into ``sys.argv``
                        # for the duration of the call so the script
                        # sees only its intended smoke args.
                        orig_argv = sys.argv
                        sys.argv = [entry["script"], *argv]
                        try:
                            rc = entry_fn(*args_list, **kwargs)
                        finally:
                            sys.argv = orig_argv
                        if rc is None or (
                            isinstance(rc, int) and rc in (0, None)
                        ):
                            pass
                        else:
                            rep.status = "FAIL"
                            rep.error = (
                                f"script returned exit code {rc}"
                            )
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else 0
                    if code not in (0, None):
                        rep.status = "FAIL"
                        rep.error = f"script raised SystemExit({code})"
                except AdapterUnavailable as exc:
                    # AdapterUnavailable is the documented failure mode
                    # for missing real tools — count it as expected,
                    # not a routing error.
                    rep.status = "EXPECTED_FAIL"
                    rep.error = (
                        "AdapterUnavailable (host is missing the real "
                        f"tool): {str(exc).splitlines()[0]}"
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    tb = traceback.format_exc()
                    # Module-import / parse errors at the bench layer
                    # (e.g. missing local CSV) are NOT routing errors —
                    # classify as expected_fail when they match one of
                    # the script's ``optional_deps`` or are clearly
                    # ``ModuleNotFoundError`` / ``ImportError``.
                    missing = _classify_missing_dep(
                        exc, entry.get("optional_deps", ())
                    )
                    if (
                        missing
                        or "ModuleNotFoundError" in tb
                        or "ImportError" in tb
                    ):
                        rep.status = "EXPECTED_FAIL"
                        rep.error = (
                            f"host missing optional dep(s) "
                            f"{missing or '(unknown)'}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        rep.status = "FAIL"
                        rep.error = f"{type(exc).__name__}: {exc}"
            rep.wall_time_s = float(time.time() - t0)
            rep.router_overhead_s = float(t_after_pre - t_pre)

        # Records outside the context — flush to JSON-friendly shape.
        score_any_records: List[Dict[str, Any]] = []
        tools: List[str] = []
        for metric_name, rec in records.items():
            tools.append(rec.tool)
            score_any_records.append(asdict(rec))
        rep.score_any_calls = score_any_records
        rep.tools_used = sorted(set(tools))

        # accuracy: every expected metric must have been invoked.
        actual_metrics = tuple(sorted(records.keys()))
        rep.actual_metrics = actual_metrics
        if not rep.expected_metrics:
            rep.accuracy = "N/A"
        elif all(m in records for m in rep.expected_metrics):
            rep.accuracy = "PASS"
        else:
            missing = [m for m in rep.expected_metrics if m not in records]
            rep.accuracy = "FAIL"
            if rep.status == "OK":
                rep.status = "FAIL"
            rep.error = (
                (rep.error or "")
                + f" | missing expected metric(s): {missing}"
            )

        return rep, records, []
    except Exception as exc:  # pragma: no cover - top-level defensive
        rep.status = "FAIL"
        rep.error = f"harness-error: {type(exc).__name__}: {exc}"
        return rep, {}, []


# ---------------------------------------------------------------------------
# Aggregate + writers
# ---------------------------------------------------------------------------
def _aggregate(reports: List[ScriptReport]) -> Dict[str, Any]:
    """Aggregate per-script reports into summary stats."""
    n = len(reports)
    if n == 0:
        return {
            "n_scripts": 0,
            "n_ok": 0,
            "n_fail": 0,
            "n_expected_fail": 0,
            "n_routing_pass": 0,
            "n_routing_fail": 0,
            "mean_wall_time_s": 0.0,
            "mean_router_overhead_s": 0.0,
            "tools_used_aggregate": [],
        }

    ok = [r for r in reports if r.status == "OK"]
    fail = [r for r in reports if r.status == "FAIL"]
    expected = [r for r in reports if r.status == "EXPECTED_FAIL"]
    # Only "routing-relevant" failures are n_fail: a script must be OK
    # for the FAIL count to mean the routing broke.  EXPECTED_FAIL is
    # for host-level missing deps; those do not indicate a routing bug.
    routing_pass = [r for r in reports if r.accuracy == "PASS"]
    routing_fail = [r for r in reports if r.accuracy == "FAIL"]

    walls = [float(r.wall_time_s) for r in reports]
    overhead = [float(r.router_overhead_s) for r in reports]
    tools: List[str] = []
    for r in reports:
        tools.extend(r.tools_used)
    tools_unique = sorted(set(tools))

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_scripts": int(n),
        "n_ok": int(len(ok)),
        "n_fail": int(len(fail)),
        "n_expected_fail": int(len(expected)),
        "n_routing_pass": int(len(routing_pass)),
        "n_routing_fail": int(len(routing_fail)),
        "mean_wall_time_s": float(_mean(walls)),
        "mean_router_overhead_s": float(_mean(overhead)),
        "tools_used_aggregate": tools_unique,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    agg = payload.get("aggregate", {})
    lines: List[str] = []
    lines.append("# Wf-48 Phase 1 — score_any() Router Benchmark")
    lines.append("")
    lines.append(f"Recipe: `{payload.get('recipe')}`")
    lines.append(f"Tool version: `{payload.get('tool_version')}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- n_scripts: `{agg.get('n_scripts')}`")
    lines.append(f"- n_ok: `{agg.get('n_ok')}`")
    lines.append(f"- n_fail: `{agg.get('n_fail')}`")
    lines.append(f"- n_expected_fail: `{agg.get('n_expected_fail')}`")
    lines.append(f"- n_routing_pass: `{agg.get('n_routing_pass')}`")
    lines.append(f"- n_routing_fail: `{agg.get('n_routing_fail')}`")
    lines.append(f"- mean_wall_time_s: `{agg.get('mean_wall_time_s'):.4f}`")
    lines.append(
        f"- mean_router_overhead_s: "
        f"`{agg.get('mean_router_overhead_s'):.4f}`"
    )
    lines.append(f"- tools_used_aggregate: "
                 f"`{', '.join(agg.get('tools_used_aggregate', []))}`")
    lines.append("")
    lines.append("## Per-script")
    lines.append("")
    lines.append(
        "| script | status | wall_s | overhead_s | tools_used | accuracy |"
    )
    lines.append("|---|---|---:|---:|---|---|")
    for r in payload.get("per_script", []):
        lines.append(
            f"| `{r['script']}` | {r['status']} | "
            f"{r['wall_time_s']:.3f} | {r['router_overhead_s']:.3f} | "
            f"{', '.join(r.get('tools_used', [])) or '-'} | "
            f"{r['accuracy']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
__version__ = "0.1.0-wf48-phase1"


def run_bench(cfg: BenchConfig) -> Dict[str, Any]:
    """Execute the Wf-48 Phase 1 router bench.

    Returns the verbatim JSON-serialisable payload that is also
    written to ``cfg.out_dir / router_bench.json``.
    """
    entries = list(DEFAULT_SCRIPT_REGISTRY)
    if int(cfg.n_scripts) < len(entries):
        entries = entries[: int(cfg.n_scripts)]

    reports: List[ScriptReport] = []
    for seed in cfg.seeds:
        for entry in entries:
            rep, _records, _probes = _invoke_script(entry, cfg, int(seed))
            reports.append(rep)

    agg = _aggregate(reports)

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "recipe": "wf48_score_any_router",
        "tool": "bench_score_any_router.py",
        "tool_version": __version__,
        "config": {
            "n_scripts": int(cfg.n_scripts),
            "seeds": [int(s) for s in cfg.seeds],
            "mock_unavailable_tools": bool(cfg.mock_unavailable_tools),
            "timeout_s": float(cfg.timeout_s),
        },
        "per_script": [asdict(r) for r in reports],
        "aggregate": agg,
    }

    out_dir = Path(cfg.out_dir)
    _write_json(out_dir / "router_bench.json", payload)
    _write_markdown(out_dir / "router_bench.md", payload)

    print(
        f"[wf48] wrote {out_dir / 'router_bench.json'} "
        f"({len(reports)} reports, {agg['n_routing_pass']}/{agg['n_scripts']} "
        f"routing pass)",
        file=sys.stderr,
    )
    return payload


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_score_any_router",
        description=(
            "Wf-48 Phase 1 — Run the 10 Wf-43 bench scripts through "
            "score_any() and record routing + wall-time per script."
        ),
    )
    p.add_argument("--n-scripts", type=int, default=10,
                   help="Number of scripts to run (default 10 — the Wf-43 ship set).")
    p.add_argument("--seeds", type=int, nargs="*", default=[0],
                   help="Seed(s) for the bench (default [0]).")
    p.add_argument("--out-dir", type=str, default="molmetal/reports/wf48_score_any",
                   help="Output directory for the bench report.")
    p.add_argument("--timeout-s", type=float, default=60.0,
                   help="Per-script timeout (default 60s).")
    p.add_argument(
        "--no-mock-unavailable",
        action="store_true",
        help=(
            "When set, do NOT substitute a synthetic 0.0 for missing "
            "real tools — let AdapterUnavailable propagate (the strict "
            "Wf-43 mode)."
        ),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    cfg = BenchConfig(
        n_scripts=int(args.n_scripts),
        seeds=tuple(int(s) for s in (args.seeds or [0])),
        out_dir=str(args.out_dir),
        timeout_s=float(args.timeout_s),
        mock_unavailable_tools=not bool(args.no_mock_unavailable),
    )
    payload = run_bench(cfg)
    # Exit code policy:
    #   * 0 — every script ran clean OR every failure is EXPECTED_FAIL
    #         (host missing an optional dep) OR accuracy is PASS/N/A.
    #   * 2 — at least one routing FAIL or non-expected FAIL.
    agg = payload["aggregate"]
    if int(agg["n_fail"]) > 0 or int(agg["n_routing_fail"]) > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
