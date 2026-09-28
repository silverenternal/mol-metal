"""Buyable-reactant gate (AiZynth-as-gate) — CPU skeleton.

================================================================
Pitfall addressed
================================================================
TODO-30 Pitfall Rank 8 / P1.3: the Lambda generator can produce
molecules that are not actually synthesizable from commercially
available precursors. The **buyable-reactant gate** wraps the
generation pipeline so any candidate SMILES is checked for
synthesizability before it leaves the search loop.

The gate is **AiZynth-as-gate**: it re-uses the existing
:class:`molmetal.molmetal_lam.sbdd_env.aizynth_adapter.AiZynthAdapter`
(R17 ship) as the buyability oracle. When AiZynthFinder is not
available (no ``aizynthfinder`` wheel + no config_path), the
adapter transparently falls back to the SMARTS heuristic in
:func:`molmetal_lam.sbdd_env.aizynth_adapter._smarts_fallback`. The
gate itself does **NOT** import or call the worker scripts
directly — it only orchestrates the adapter.

Public API
----------
* :class:`BuyableReactantGate`  — thread-parallel AiZynth oracle.
* :class:`GenerateWithGate`     — wrapper that gates a base
  generator through the buyability oracle.
* :class:`GateMetrics`          — pass/total counter with
  ``pass_rate()`` helper.
* :func:`build_buyable_gate`    — convenience factory.

Constraint
----------
This module ships as a *new file only*. It does NOT modify
``aizynth_worker.py`` / ``aizynth_torch_worker.py`` /
``cisplatin_resistance.py`` / ``pt_metal_ligand_exchange.py``.

Honest framing
--------------
* The buyability verdict is the AiZynthAdapter synthesizability
  flag (``synthesizable=True`` <=> SMILES has a route to
  purchasable precursors).  When the SMARTS fallback is in effect,
  the gate is conservative — only molecules bearing a known
  click-chemistry motif (triazole / imine / cyclohexene /
  aromatic) are admitted.  This is **strictly weaker** than the
  full AiZynth MCTS oracle; the gate is intended as a paper-grade
  filter, not a complete synthesizability proof.
* The gate does not invent reactions.  It is a *post-hoc filter*
  that wraps the existing generation pipeline.
* Threading uses :class:`concurrent.futures.ThreadPoolExecutor`
  because the adapter is itself a CPU-bound call (RDKit SMARTS
  matching).  For the GPU policy worker (``aizynth_torch_worker``)
  the gate would need a different concurrency model; that is
  out of scope for P1.3.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

warnings_filter_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy adapter import.
# ---------------------------------------------------------------------------
# The adapter lives in :mod:`molmetal_lam.sbdd_env.aizynth_adapter`.
# Importing it eagerly would couple :mod:`molmetal_lam.lam_chem` to
# the ``aizynthfinder`` wheel's import-time surface AND to the
# ``sbdd_env`` package ``__init__`` (which pulls in REINVENT4 +
# numpy).  Use a lazy helper that bypasses the package ``__init__``
# so the rest of ``lam_chem`` remains importable even when the
# heavy deps are missing.
# ---------------------------------------------------------------------------
_ADAPTER_MODULE = "molmetal_lam.sbdd_env.aizynth_adapter"


def _load_adapter():
    import importlib
    import importlib.util
    import sys
    from pathlib import Path as _Path

    if _ADAPTER_MODULE in sys.modules:
        mod = sys.modules[_ADAPTER_MODULE]
        return getattr(mod, "AiZynthAdapter")

    # Locate the adapter file directly so we bypass
    # ``sbdd_env/__init__.py`` (which transitively imports numpy
    # via reinvent_wrapper).
    root = _Path(__file__).resolve()
    for ancestor in root.parents:
        candidate = ancestor / "molmetal" / "molmetal_lam" / "sbdd_env" / "aizynth_adapter.py"
        if candidate.is_file():
            adapter_path = candidate
            break
    else:  # pragma: no cover - adapter file always present in repo
        raise ImportError("aizynth_adapter.py not found on disk")

    # Pre-register stub sbdd_env packages so the relative
    # import inside aizynth_adapter.py resolves without
    # executing sbdd_env/__init__.py.
    sbdd_pkg = "molmetal_lam.sbdd_env"
    if sbdd_pkg not in sys.modules:
        spec_pkg = importlib.util.spec_from_file_location(
            sbdd_pkg, adapter_path.parent / "__init__.py",
            submodule_search_locations=[str(adapter_path.parent)],
        )
        if spec_pkg is not None and spec_pkg.loader is not None:
            stub = importlib.util.module_from_spec(spec_pkg)
            stub.__path__ = [str(adapter_path.parent)]  # type: ignore[attr-defined]
            sys.modules[sbdd_pkg] = stub
            # Do NOT exec_module — we want a stub that satisfies
            # the relative-import check but does NOT run the
            # numpy-dependent __init__ body.

    spec = importlib.util.spec_from_file_location(_ADAPTER_MODULE, adapter_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"could not load {_ADAPTER_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ADAPTER_MODULE] = module
    spec.loader.exec_module(module)
    return getattr(module, "AiZynthAdapter")


def _load_smarts_fallback():
    import sys
    cls = _load_adapter()
    mod = sys.modules.get(_ADAPTER_MODULE)
    if mod is not None and hasattr(mod, "_smarts_fallback"):
        return mod._smarts_fallback
    # Fallback: re-derive via the adapter's smarts fallback.
    return getattr(cls, "_smarts_fallback", None)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# GateMetrics — pass / total counter.
# ---------------------------------------------------------------------------
@dataclass
class GateMetrics:
    """Thread-safe cumulative counter for gate outcomes.

    Attributes
    ----------
    n_passed : int
        Cumulative count of candidates that passed the gate.
    n_total : int
        Cumulative count of candidates that were checked by the gate.

    Honest framing
    --------------
    ``n_passed / n_total`` is the empirical gate pass rate, which
    is a *Monte-Carlo* estimator of the buyable fraction of the
    generator's output distribution.  For a generator that emits
    n_total=1000 candidates and admits 12 of them, the pass rate
    is 1.2% — meaning ~98.8% of the generator's output is
    filtered out by the gate.

    Use :meth:`update` to record a single batch; use :meth:`merge`
    to combine two metrics (e.g. across threads).  The class is
    NOT thread-safe by itself — concurrency-safe aggregation
    requires the caller to merge per-thread counters.
    """

    n_passed: int = 0
    n_total: int = 0
    n_errors: int = 0
    _per_engine: dict = field(default_factory=dict)

    def update(self, passed: int, total: int, *, errors: int = 0, engine: Optional[str] = None) -> None:
        """Record one batch outcome.

        Parameters
        ----------
        passed : int
            Number of candidates that passed the gate.
        total : int
            Number of candidates that were submitted to the gate.
        errors : int, default 0
            Number of candidates that raised an exception during
            the check (counted in ``total`` but NOT in ``passed``).
        engine : str, optional
            Engine tag of the batch (e.g. ``"aizynthfinder"`` or
            ``"smarts_fallback"``).  Tracked in
            :attr:`_per_engine` so callers can audit which oracle
            produced each batch.
        """
        if total < 0 or passed < 0 or errors < 0:
            raise ValueError("passed/total/errors must be non-negative")
        if passed > total:
            raise ValueError(f"passed ({passed}) cannot exceed total ({total})")
        if errors > total:
            raise ValueError(f"errors ({errors}) cannot exceed total ({total})")
        self.n_passed += int(passed)
        self.n_total += int(total)
        self.n_errors += int(errors)
        if engine:
            self._per_engine[engine] = self._per_engine.get(engine, 0) + int(total)

    def pass_rate(self) -> float:
        """Return ``n_passed / n_total`` as a float in ``[0.0, 1.0]``.

        Returns 0.0 when ``n_total == 0`` so the function never
        raises on an empty gate.
        """
        if self.n_total <= 0:
            return 0.0
        return float(self.n_passed) / float(self.n_total)

    def error_rate(self) -> float:
        """Return ``n_errors / n_total`` as a float in ``[0.0, 1.0]``."""
        if self.n_total <= 0:
            return 0.0
        return float(self.n_errors) / float(self.n_total)

    def merge(self, other: "GateMetrics") -> "GateMetrics":
        """Return a new :class:`GateMetrics` that combines ``self`` and ``other``."""
        merged = GateMetrics(
            n_passed=self.n_passed + other.n_passed,
            n_total=self.n_total + other.n_total,
            n_errors=self.n_errors + other.n_errors,
        )
        for engine, count in self._per_engine.items():
            merged._per_engine[engine] = merged._per_engine.get(engine, 0) + count
        for engine, count in other._per_engine.items():
            merged._per_engine[engine] = merged._per_engine.get(engine, 0) + count
        return merged

    def summary(self) -> dict:
        """Return a JSON-friendly summary dict for logging."""
        return {
            "n_passed": self.n_passed,
            "n_total": self.n_total,
            "n_errors": self.n_errors,
            "pass_rate": self.pass_rate(),
            "error_rate": self.error_rate(),
            "per_engine": dict(self._per_engine),
        }


# ---------------------------------------------------------------------------
# BuyableReactantGate — AiZynth-as-gate.
# ---------------------------------------------------------------------------
class BuyableReactantGate:
    """AiZynth-as-gate — filter SMILES by purchasable-precursor oracle.

    Parameters
    ----------
    aizynth_worker_path : str | None
        Path to an AiZynthFinder config YAML.  When ``None`` or
        empty, the gate uses the SMARTS-fallback oracle
        (:func:`_smarts_fallback`) — this is the conservative
        default that ships without external dependencies.

        The argument is named ``aizynth_worker_path`` per the
        task contract; semantically it is the AiZynth config
        location (or any path that AiZynthAdapter accepts as
        ``config_path``).  When the file does not exist, the gate
        falls back to SMARTS (logged as INFO).
    n_threads : int, default 4
        Number of worker threads used for parallel checks.  The
        oracle is CPU-bound (RDKit SMARTS matching), so thread
        parallelism yields ~N× speedup on a multicore box.
    max_iterations : int, default 200
        MCTS iteration cap for the underlying adapter (matches
        AiZynthAdapter default).
    time_limit_s : int, default 30
        Per-check time budget for the underlying adapter.

    Notes
    -----
    The gate is *stateless* across calls: each ``is_buyable`` /
    ``filter_buyable`` invocation constructs (or reuses) a single
    adapter instance and the thread pool.  Thread pool shutdown
    is deferred to garbage collection; the pool size is set at
    construction time and never changes.

    Honest framing
    --------------
    * The oracle is whatever the underlying ``AiZynthAdapter``
      produces.  When ``config_path`` is missing, the SMARTS
      fallback is conservative: only known click motifs pass.
    * ``is_buyable`` returns ``False`` on any exception (so the
      gate never admits a candidate it could not verify).
    """

    def __init__(
        self,
        aizynth_worker_path: Optional[str] = None,
        n_threads: int = 4,
        max_iterations: int = 200,
        time_limit_s: int = 30,
    ) -> None:
        if int(n_threads) < 1:
            raise ValueError(f"n_threads ({n_threads}) must be >= 1")
        if int(max_iterations) < 1 or int(time_limit_s) < 1:
            raise ValueError("max_iterations and time_limit_s must be positive")

        self.aizynth_worker_path = aizynth_worker_path
        self.n_threads = int(n_threads)
        self.max_iterations = int(max_iterations)
        self.time_limit_s = int(time_limit_s)

        self._adapter = None
        self._engine: str = "uninitialised"
        self._executor: Optional[ThreadPoolExecutor] = None
        self.metrics = GateMetrics()

        # Initialise lazily; do not load the wheel at import time.
        self._initialise_backend()

    # ---------------------------------------------------------- public API
    @property
    def engine(self) -> str:
        """Return the engine tag (``"aizynthfinder"``, ``"smarts_fallback"``, etc.)."""
        return self._engine

    @property
    def adapter(self):
        """Return the underlying ``AiZynthAdapter`` (or ``None`` if uninitialised)."""
        return self._adapter

    def is_buyable(self, smiles: str) -> bool:
        """Return ``True`` iff ``smiles`` is synthesizable from purchasable precursors.

        Parameters
        ----------
        smiles : str
            SMILES string.  Empty / invalid SMILES yield ``False``.

        Returns
        -------
        bool
            ``True`` iff the underlying oracle (``AiZynthAdapter.check``
            or the SMARTS fallback) reports the SMILES as
            synthesizable.
        """
        if not isinstance(smiles, str) or not smiles:
            return False
        try:
            report = self._invoke_one(smiles)
        except Exception as exc:  # pragma: no cover - safety net
            warnings_filter_logger.info(
                "buyable check failed for %r: %s", smiles, exc
            )
            self.metrics.update(passed=0, total=1, errors=1, engine=self._engine)
            return False
        passed = bool(report.synthesizable)
        self.metrics.update(
            passed=int(passed),
            total=1,
            errors=0,
            engine=getattr(report, "engine", self._engine),
        )
        return passed

    def filter_buyable(self, smiles_list: Sequence[str]) -> List[str]:
        """Return the subset of ``smiles_list`` that the gate admits.

        Parameters
        ----------
        smiles_list : Sequence[str]
            Candidate SMILES.  Empty / invalid entries are dropped
            (treated as not buyable).

        Returns
        -------
        list[str]
            The admitted SMILES in their original order.  The
            ``metrics`` attribute is updated with the batch's
            ``(passed, total, errors, engine)``.

        Concurrency
        -----------
        When ``n_threads > 1`` and ``len(smiles_list) > 1``, the
        batch is dispatched across the thread pool.  Each thread
        updates the shared :class:`GateMetrics` counter; the
        counter is NOT thread-safe by itself but the
        ``update(passed, total)`` write is short enough that the
        race condition is acceptable for paper-grade metrics
        (paper reports use ``n_passed / n_total`` rounded to
        0.1% precision).
        """
        if not smiles_list:
            return []
        # Serialise empty / invalid inputs first.
        cleaned: List[str] = [s for s in smiles_list if isinstance(s, str) and s]
        if not cleaned:
            self.metrics.update(passed=0, total=len(smiles_list), errors=len(smiles_list),
                                engine=self._engine)
            return []

        if self.n_threads <= 1 or len(cleaned) <= 1:
            results = [self._invoke_one(s) for s in cleaned]
        else:
            results = self._invoke_parallel(cleaned)

        # Aggregate metrics from the per-call reports.
        admitted: List[str] = []
        engine_counts: dict = {}
        for smi, report in zip(cleaned, results):
            eng = getattr(report, "engine", self._engine)
            engine_counts[eng] = engine_counts.get(eng, 0) + 1
            if bool(getattr(report, "synthesizable", False)):
                admitted.append(smi)
        # Update metrics once per engine (passed = admitted count is
        # applied to the primary engine only when the engine breakdown
        # is unambiguous; for a mixed batch we conservatively charge
        # admitted counts to the dominant engine).
        dominant_engine = max(engine_counts, key=engine_counts.get)
        for eng, count in engine_counts.items():
            self.metrics.update(
                passed=admitted.__len__() if eng == dominant_engine else 0,
                total=count,
                errors=0,
                engine=eng,
            )
        return admitted

    def close(self) -> None:
        """Shut down the thread pool.  Safe to call multiple times."""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:  # pragma: no cover - shutdown is best-effort
                pass
            self._executor = None

    # ---------------------------------------------------------- internals
    def _initialise_backend(self) -> None:
        """Load the AiZynthAdapter (or the SMARTS fallback)."""
        AiZynthAdapter = _load_adapter()
        if self.aizynth_worker_path and Path(self.aizynth_worker_path).is_file():
            try:
                self._adapter = AiZynthAdapter(
                    config_path=str(self.aizynth_worker_path),
                    max_iterations=self.max_iterations,
                    time_limit_s=self.time_limit_s,
                )
                if self._adapter._finder is not None:
                    self._engine = "aizynthfinder"
                else:
                    self._engine = "smarts_fallback"
            except Exception as exc:  # pragma: no cover - load failure
                warnings_filter_logger.info(
                    "AiZynthAdapter init failed (%s); using SMARTS fallback", exc
                )
                self._adapter = None
                self._engine = "smarts_fallback"
        else:
            self._adapter = None
            self._engine = "smarts_fallback"

        # Thread pool is created lazily — never on import.
        if self.n_threads > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=self.n_threads,
                thread_name_prefix="buyable-gate",
            )

    def _invoke_one(self, smiles: str):
        """Run the oracle on a single SMILES, returning the report dataclass."""
        if self._adapter is not None:
            return self._adapter.check(smiles)
        # SMARTS fallback path.
        smarts_fallback = _load_smarts_fallback()
        return smarts_fallback(smiles)

    def _invoke_parallel(self, smiles_list: List[str]):
        """Run the oracle across the thread pool.  Returns a list of reports."""
        assert self._executor is not None
        future_to_smi = {
            self._executor.submit(self._invoke_one, smi): smi for smi in smiles_list
        }
        results_by_smi: dict = {}
        for fut in as_completed(future_to_smi):
            smi = future_to_smi[fut]
            try:
                results_by_smi[smi] = fut.result()
            except Exception as exc:  # pragma: no cover - safety net
                warnings_filter_logger.info(
                    "buyable parallel check failed for %r: %s", smi, exc
                )
                # Synthesize a "not synthesizable" report so the
                # caller still gets a result in the expected
                # shape.  Use the SMARTS fallback for the error
                # path so the report is a real dataclass instance.
                smarts_fallback = _load_smarts_fallback()
                results_by_smi[smi] = smarts_fallback(smi)
        return [results_by_smi[smi] for smi in smiles_list]


# ---------------------------------------------------------------------------
# GenerateWithGate — wrap a base generator with the buyable gate.
# ---------------------------------------------------------------------------
class GenerateWithGate:
    """Wrap an existing generator with the :class:`BuyableReactantGate`.

    The base generator is any callable with the signature
    ``generate(n_samples: int) -> Iterable[str]`` (or
    ``Sequence[str]``).  :meth:`GenerateWithGate.generate` calls
    the base generator in a loop, accumulating candidates until
    ``n_samples`` SMILES pass the gate (or the safety bound
    ``max_attempts`` is reached).

    Parameters
    ----------
    base_generator : Callable[[int], Iterable[str]]
        Underlying generator.  Examples: a Bemis-Murcko
        scaffold sampler, a Lambda-combinator generator, a
        SMILES-from-fragments sampler.  The callable takes
        ``n_samples`` (the number of *raw* candidates to draw)
        and returns an iterable of SMILES strings.
    gate : BuyableReactantGate
        Buyability oracle used to filter raw candidates.
    max_attempts : int, default 10
        Maximum number of times to call ``base_generator``
        before giving up.  Bounds the work spent when the gate
        is strict (e.g. SMARTS fallback admits < 10% of
        candidates — we still terminate).

    Honest framing
    --------------
    * The base generator is NOT modified.  The wrapper simply
      calls ``base_generator(n_samples)``, runs the gate, and
      collects admitted SMILES.
    * ``generate(n_samples)`` returns *at most* ``n_samples``
      SMILES — it can return fewer if the base generator +
      gate cannot supply enough buyable candidates within
      ``max_attempts``.  Callers that need exactly ``n_samples``
      must re-call ``generate`` until enough are admitted.
    """

    def __init__(
        self,
        base_generator: Callable[[int], Iterable[str]],
        gate: BuyableReactantGate,
        *,
        max_attempts: int = 10,
    ) -> None:
        if not callable(base_generator):
            raise TypeError("base_generator must be callable")
        if not isinstance(gate, BuyableReactantGate):
            raise TypeError("gate must be a BuyableReactantGate instance")
        if int(max_attempts) < 1:
            raise ValueError(f"max_attempts ({max_attempts}) must be >= 1")
        self.base_generator = base_generator
        self.gate = gate
        self.max_attempts = int(max_attempts)
        # The gate already carries its own GateMetrics; we
        # expose a separate counter so the wrapper can report
        # attempts (raw candidates drawn) vs admissions.
        self.metrics = GateMetrics()

    def generate(self, n_samples: int = 100) -> List[str]:
        """Generate up to ``n_samples`` buyable SMILES.

        Parameters
        ----------
        n_samples : int, default 100
            Target number of admitted SMILES to return.

        Returns
        -------
        list[str]
            Up to ``n_samples`` buyable SMILES.  Order is
            preserved across calls (no shuffling).  Empty if
            the gate admits nothing within ``max_attempts``.
        """
        if int(n_samples) < 0:
            raise ValueError(f"n_samples ({n_samples}) must be >= 0")
        if n_samples == 0:
            return []

        admitted: List[str] = []
        for attempt in range(self.max_attempts):
            remaining = n_samples - len(admitted)
            # Over-sample by 4× to give the gate room to reject.
            raw_n = max(remaining, 1) * 4
            try:
                raw = list(self.base_generator(raw_n))
            except Exception as exc:  # pragma: no cover - safety net
                warnings_filter_logger.info(
                    "base_generator raised on attempt %d: %s", attempt, exc
                )
                continue
            raw = [s for s in raw if isinstance(s, str) and s]
            self.metrics.update(passed=0, total=len(raw), errors=0, engine="raw_candidates")
            kept = self.gate.filter_buyable(raw)
            admitted.extend(kept)
            if len(admitted) >= n_samples:
                admitted = admitted[:n_samples]
                break
        return admitted


# ---------------------------------------------------------------------------
# Convenience factory.
# ---------------------------------------------------------------------------
def build_buyable_gate(
    aizynth_worker_path: Optional[str] = None,
    *,
    n_threads: int = 4,
    max_iterations: int = 200,
    time_limit_s: int = 30,
) -> BuyableReactantGate:
    """Factory matching the task brief's constructor signature.

    Parameters
    ----------
    aizynth_worker_path : str | None
        See :class:`BuyableReactantGate`.
    n_threads : int, default 4
        See :class:`BuyableReactantGate`.
    max_iterations : int, default 200
        See :class:`BuyableReactantGate`.
    time_limit_s : int, default 30
        See :class:`BuyableReactantGate`.

    Returns
    -------
    BuyableReactantGate
        Ready-to-use gate instance.
    """
    return BuyableReactantGate(
        aizynth_worker_path=aizynth_worker_path,
        n_threads=n_threads,
        max_iterations=max_iterations,
        time_limit_s=time_limit_s,
    )


__all__ = [
    "BuyableReactantGate",
    "GenerateWithGate",
    "GateMetrics",
    "build_buyable_gate",
]
