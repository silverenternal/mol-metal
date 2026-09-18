"""REINVENT4 subprocess-RPC adapter — multi-property scorer (L-4).

================================================================
What this module is
================================================================
This module implements :class:`REINVENT4Adapter`, a subprocess-RPC
scorer that drives REINVENT4 as an external binary (the upstream
``reinvent`` CLI) via JSON-line protocol on stdin/stdout and returns
a four-tuple ``(qed, sa, binding, novelty)`` per molecule, all in
``[0, 1]``.

If REINVENT4 is unavailable (no binary on PATH and no
``molmetal/references/REINVENT4`` checkout installed as a Python
package), the adapter flips to ``self.available = False`` and
:meth:`score` returns ``None`` for every batch.  The downstream
:class:`RewardAggregator` (see :mod:`molmetal_lam.search_alg.proof_search`)
falls back to a graceful ``0.0`` in that case so the search loop
never crashes.

================================================================
Why subprocess-RPC
================================================================
REINVENT4 ships heavy plugins (``reinvent_plugins`` — QSAR, docking,
shape, RAscore) that depend on proprietary models, Java runtimes,
and large deep-learning checkpoints.  Importing them inline slows
the unit-test path and binds the search to a particular REINVENT4
version.  Subprocess-RPC isolates the dependency:

* the test suite never imports the heavy REINVENT4 plugins;
* the Java + GPU + RL loop run in a child process that we can
  restart on heartbeat failure;
* a single :class:`REINVENT4Adapter` can be reused across many
  MCTS simulations (no fork-per-call cost).

The wire format is line-delimited JSON over stdin/stdout::

    {"op": "score", "smiles": ["CCO", "c1ccccc1"]}
    <- {"results": [{"qed": 0.5, "sa": 0.7, "binding": 0.4, "novelty": 0.9}, ...]}

================================================================
Public API
----------
* :class:`ScoreResult` — dataclass with the four [0,1] scores.
* :class:`REINVENT4Adapter` — four-component proxy subprocess-RPC
  adapter with health-check heartbeat and graceful fallback.
* :class:`REINVENT4MultipropertyAdapter` — WF-Extra-2 subprocess-RPC
  adapter that drives REINVENT4's native multi-property objective
  (sum of weighted components — logP / ring count / QED / …) via the
  ``reinvent4_multiproperty_jsonl_worker.py`` bridge.  Mode is fixed
  to ``"multiproperty"``; see :meth:`from_config` for the JSON
  config format consumed by the AMD ROCm environment.
* :class:`ScoreAggregator` — weighted-sum aggregator (defaults
  ``0.3 / 0.3 / 0.3 / 0.1``).
* :func:`is_reinvent4_binary_available` — runtime probe.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ScoreResult dataclass
# ---------------------------------------------------------------------------
@dataclass
class ScoreResult:
    """Per-molecule multi-property score returned by REINVENT4.

    All four components are normalised to ``[0, 1]``; ``None`` means
    the component could not be evaluated (RDKit parse failure, scorer
    timeout, etc.).
    """

    qed: Optional[float] = None
    sa: Optional[float] = None
    binding: Optional[float] = None
    novelty: Optional[float] = None

    def is_complete(self) -> bool:
        """True iff every component is a finite float in [0, 1]."""
        for v in (self.qed, self.sa, self.binding, self.novelty):
            if v is None:
                return False
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return False
            if not (0.0 <= fv <= 1.0):
                return False
        return True

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "qed": self.qed,
            "sa": self.sa,
            "binding": self.binding,
            "novelty": self.novelty,
        }


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------
def is_reinvent4_binary_available(binary_name: str = "reinvent") -> bool:
    """Return True iff the REINVENT4 CLI binary is on PATH.

    Also accepts the upstream entry-point name (``reinvent.py``)
    for environments that install REINVENT4 from source.
    """
    configured = os.environ.get("REINVENT4_BIN")
    candidates = [configured] if configured else []
    candidates.append(binary_name)
    # The isolated environment is intentionally outside the project venv:
    # REINVENT4 pins a CUDA torch version incompatible with ROCm torch.
    candidates.append("/mnt/storage/envs/reinvent4/bin/reinvent")
    if any(c and (os.path.isfile(c) and os.access(c, os.X_OK) or shutil.which(c)) for c in candidates):
        return True
    for alt in ("reinvent4", "reinvent.py"):
        if shutil.which(alt):
            return True
    # Fallback: the cloned repo at ``molmetal/references/REINVENT4`` ships
    # ``reinvent/Reinvent.py``.  We treat that as "available" so that the
    # subprocess-RPC layer can be exercised in environments that ship
    # the source tree but not the installed entry-point.
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(
        os.path.join(here, "..", "..", "references", "REINVENT4", "reinvent", "Reinvent.py")
    )
    return os.path.isfile(candidate)


def _is_jsonl_worker(binary: str) -> bool:
    """Return whether *binary* is an adapter-compatible JSONL worker.

    The standard REINVENT4 entry point is a TOML RL CLI, so treating its
    mere presence as an RPC worker causes a blocking ``readline()`` call.
    Explicitly named bridge workers are the only automatically trusted form.
    """
    return "jsonl_worker" in os.path.basename(binary).lower()


# ---------------------------------------------------------------------------
# REINVENT4 subprocess-RPC adapter
# ---------------------------------------------------------------------------
@dataclass
class REINVENT4Adapter:
    """Subprocess-RPC adapter for the REINVENT4 multi-property scorer.

    Parameters
    ----------
    binary : str, default "reinvent"
        Executable name or absolute path to the REINVENT4 CLI.  We
        spawn it once and reuse the child process for all ``score``
        calls (a "long-lived" RPC worker).
    scoring_config : str, optional
        Path to a REINVENT4 scoring TOML.  When ``None`` we use the
        bundled ``stage1_scoring.toml`` from the reference repo (if
        present) and otherwise the adapter is still available but
        every component returns ``None``.
    heartbeat_interval : float, default 30.0
        Seconds between background health-check probes.  When the
        worker subprocess dies we set ``self.dead = True`` and every
        subsequent :meth:`score` returns ``None``.
    timeout : float, default 60.0
        Per-call wall-clock timeout for a batch RPC roundtrip.

    Attributes
    ----------
    available : bool
        Set in :meth:`__post_init__`; ``False`` if the binary probe
        fails, in which case every :meth:`score` call returns ``None``.
    dead : bool
        Flipped to ``True`` by the heartbeat thread when the worker
        subprocess is no longer responsive.
    """

    binary: str = "reinvent"
    scoring_config: Optional[str] = None
    heartbeat_interval: float = 30.0
    timeout: float = 60.0

    #: Set after the constructor probe.
    available: bool = False
    #: Flipped to ``True`` if the heartbeat thread detects worker death.
    dead: bool = False
    #: Structured reason why a ``score()`` call would return ``None``.
    #: One of ``"binary_missing"``, ``"worker_dead"``, ``"rpc_error"``,
    #: or ``None`` when the adapter is healthy.  Callers can introspect
    #: this to distinguish "user forgot to install reinvent4" from
    #: transient subprocess crashes.
    last_error: Optional[str] = None
    #: Lazily-created worker subprocess (kept open across calls).
    _proc: Optional[subprocess.Popen] = None
    _heartbeat_thread: Optional[threading.Thread] = None
    _heartbeat_stop: bool = False
    _lock: threading.Lock = None  # type: ignore[assignment]

    #: Install-instruction string surfaced when the REINVENT4 binary is
    #: missing.  Kept as a module-level constant so tests can assert on
    #: the exact recommendation without re-deriving the wording.
    BINARY_MISSING_HINT = (
        "Install REINVENT4 via `pipx install reinvent4` "
        "(or `docker pull mricci/reinvent:latest`) "
        "and ensure the `reinvent` CLI is on PATH."
    )

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        if self.binary == "reinvent":
            configured = os.environ.get("REINVENT4_BIN")
            if configured and os.path.isfile(configured):
                self.binary = configured
            elif os.path.isfile("/mnt/storage/envs/reinvent4/bin/reinvent"):
                self.binary = "/mnt/storage/envs/reinvent4/bin/reinvent"
            # The upstream CLI is TOML/RL-oriented, while this adapter speaks
            # JSONL. Prefer the isolated RDKit-only compatibility worker for
            # the default protocol and leave REINVENT4_BIN available when a
            # learned plugin worker is explicitly configured.
            worker = os.path.join(os.path.dirname(__file__), "reinvent4_jsonl_worker.py")
            if not configured and os.path.isfile(worker):
                self.binary = worker
        self.available = is_reinvent4_binary_available(self.binary)
        if self.available and not _is_jsonl_worker(self.binary):
            # A normal REINVENT4 CLI is not the JSON-lines protocol consumed
            # by this adapter. Keep it discoverable, but refuse to launch it
            # as a long-lived RPC process. Users can point REINVENT4_BIN at
            # the isolated worker (or an equivalent wrapper) instead.
            self.available = False
            self.last_error = "cli_protocol_mismatch"
            logger.warning(
                "REINVENT4Adapter: '%s' is a standard TOML CLI, not a JSONL "
                "worker; use reinvent4_jsonl_worker.py or a compatible bridge.",
                self.binary,
            )
            return
        if not self.available:
            # Explicit check: ``shutil.which``-style probe failed, so the
            # user has no installed reinvent4 binary.  Surface a clear
            # WARNING with install instructions rather than silently
            # swallowing the subsequent FileNotFoundError.
            if shutil.which(self.binary) is None:
                # Only emit the "binary_missing" reason when the probe
                # genuinely finds no CLI on PATH — the reference-checkout
                # fallback may still produce a usable adapter.
                self.last_error = "binary_missing"
                logger.warning(
                    "REINVENT4Adapter: binary '%s' not found on PATH. "
                    "All score() calls will return None. %s",
                    self.binary,
                    self.BINARY_MISSING_HINT,
                )
            else:
                logger.warning(
                    "REINVENT4Adapter: binary '%s' present but adapter "
                    "still unavailable (reference checkout missing?). "
                    "All score() calls will return None.",
                    self.binary,
                )
            return
        # Resolve scoring config if the caller did not supply one.
        if self.scoring_config is None:
            self.scoring_config = self._discover_default_config()
        # Best-effort launch the long-lived worker subprocess.  Failures
        # here flip ``self.available`` back to False so callers degrade
        # cleanly without ever raising.
        try:
            self._start_worker()
        except FileNotFoundError as exc:
            # The subprocess.Popen call raised FileNotFoundError because
            # the reinvent binary is not on PATH.  Log a clear warning
            # with install instructions and surface a structured reason
            # instead of swallowing silently.
            self.last_error = "binary_missing"
            self.available = False
            logger.warning(
                "REINVENT4Adapter: failed to spawn worker for '%s' "
                "(FileNotFoundError: %s). All score() calls will return "
                "None. %s",
                self.binary, exc, self.BINARY_MISSING_HINT,
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("REINVENT4Adapter: failed to start worker (%s); falling back.", exc)
            self.last_error = "worker_dead"
            self.available = False
            return
        # Heartbeat thread keeps the worker alive.
        self._heartbeat_stop = False
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="REINVENT4-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _discover_default_config(self) -> Optional[str]:
        """Return the path to a usable scoring TOML, or ``None``."""
        here = os.path.dirname(os.path.abspath(__file__))
        ref_root = os.path.normpath(
            os.path.join(here, "..", "..", "references", "REINVENT4", "configs")
        )
        for name in ("stage1_scoring.toml", "scoring.toml", "scoring_components_example.toml"):
            p = os.path.join(ref_root, name)
            if os.path.isfile(p):
                return p
        return None

    def _start_worker(self) -> None:
        """Spawn the long-lived ``reinvent`` RPC worker subprocess."""
        cmd = [self.binary]
        if self.scoring_config and os.path.isfile(self.scoring_config):
            # REINVENT4 supports ``--scoring-config <toml>``.  When the
            # binary is the Python entry-point (``reinvent/Reinvent.py``)
            # we wrap it in ``python -m`` style invocation.
            if self.binary.endswith(".py"):
                cmd = [sys.executable, self.binary, "--scoring-config", self.scoring_config]
            else:
                cmd = [self.binary, "--scoring-config", self.scoring_config]
        if self.binary.endswith("reinvent4_jsonl_worker.py"):
            cmd = [sys.executable, self.binary]
        # Use line-buffered text mode; the wire protocol is JSON-Lines.
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )

    def _heartbeat_loop(self) -> None:
        """Background thread: poll the worker every ``heartbeat_interval``s."""
        while not self._heartbeat_stop:
            time.sleep(self.heartbeat_interval)
            if self._proc is None:
                self.dead = True
                break
            poll = self._proc.poll()
            if poll is not None:
                logger.warning(
                    "REINVENT4Adapter: worker subprocess exited (rc=%s).", poll,
                )
                self.dead = True
                break

    def close(self) -> None:
        """Stop the heartbeat thread and terminate the worker subprocess."""
        self._heartbeat_stop = True
        proc = self._proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except Exception:  # pragma: no cover
                pass
            self._proc = None
        self.dead = True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self, smiles_batch: List[str]) -> List[Optional[ScoreResult]]:
        """Score a batch of SMILES strings.

        Returns a list of the same length as ``smiles_batch``.  When
        the adapter is unavailable or dead, returns ``[None] * len(smiles_batch)``.
        Any per-element error (RPC failure, malformed JSON, etc.)
        degrades the corresponding element to ``None``.
        """
        if not self.available or self.dead:
            # Surface *why* every entry will be None.  ``binary_missing``
            # (set in __post_init__) is the dominant case for a fresh
            # install; ``worker_dead`` indicates the heartbeat killed
            # the subprocess.  We do not overwrite an existing reason
            # so a stale "binary_missing" is preserved across worker
            # restarts.
            if self.last_error is None:
                self.last_error = "worker_dead"
            return [None] * len(smiles_batch)
        if not smiles_batch:
            return []
        request: Dict[str, Any] = {"op": "score", "smiles": list(smiles_batch)}
        try:
            with self._lock:
                raw = self._rpc_roundtrip(request)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("REINVENT4Adapter.score RPC failed: %s", exc)
            self.dead = True
            self.last_error = "rpc_error"
            return [None] * len(smiles_batch)
        if raw is None:
            return [None] * len(smiles_batch)
        results = raw.get("results", [])
        out: List[Optional[ScoreResult]] = []
        for i in range(len(smiles_batch)):
            if i >= len(results):
                out.append(None)
                continue
            r = results[i]
            try:
                out.append(ScoreResult(
                    qed=_clip01(r.get("qed")),
                    sa=_clip01(r.get("sa")),
                    binding=_clip01(r.get("binding")),
                    novelty=_clip01(r.get("novelty")),
                ))
            except Exception:
                out.append(None)
        return out

    def _rpc_roundtrip(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send one JSON request, read one JSON response."""
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            self.dead = True
            return None
        if proc.poll() is not None:
            self.dead = True
            return None
        try:
            line = json.dumps(request) + "\n"
            proc.stdin.write(line)
            proc.stdin.flush()
        except Exception as exc:  # pragma: no cover
            logger.warning("REINVENT4Adapter: stdin write failed (%s)", exc)
            self.dead = True
            return None
        try:
            resp_line = proc.stdout.readline()
        except Exception as exc:  # pragma: no cover
            logger.warning("REINVENT4Adapter: stdout read failed (%s)", exc)
            self.dead = True
            return None
        if not resp_line:
            self.dead = True
            return None
        try:
            return json.loads(resp_line)
        except json.JSONDecodeError as exc:
            logger.warning("REINVENT4Adapter: malformed RPC response (%s)", exc)
            self.dead = True
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clip01(v: Any) -> Optional[float]:
    """Return ``v`` clipped to ``[0, 1]`` or ``None`` if non-numeric."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not (f == f):  # NaN check
        return None
    return max(0.0, min(1.0, f))


# ---------------------------------------------------------------------------
# ScoreAggregator — weighted sum of (qed, sa, binding, novelty)
# ---------------------------------------------------------------------------
@dataclass
class ScoreAggregator:
    """Combine (qed, sa, binding, novelty) into a single float in [0, 1].

    Default weights ``0.3 / 0.3 / 0.3 / 0.1`` (qed, sa, binding, novelty).
    Missing components (None) are treated as 0.0 so a partially-evaluable
    molecule still contributes what we know.
    """

    w_qed: float = 0.3
    w_sa: float = 0.3
    w_binding: float = 0.3
    w_novelty: float = 0.1

    def aggregate(self, score: Optional[ScoreResult]) -> float:
        """Return the weighted sum; ``None`` score → 0.0."""
        if score is None:
            return 0.0

        def _comp(v: Optional[float], w: float) -> float:
            if v is None:
                return 0.0
            try:
                f = float(v)
            except (TypeError, ValueError):
                return 0.0
            return w * max(0.0, min(1.0, f))

        return float(
            _comp(score.qed, self.w_qed)
            + _comp(score.sa, self.w_sa)
            + _comp(score.binding, self.w_binding)
            + _comp(score.novelty, self.w_novelty)
        )

    def total_weight(self) -> float:
        """Sum of the four weights — for diagnostics."""
        return float(self.w_qed + self.w_sa + self.w_binding + self.w_novelty)


# ---------------------------------------------------------------------------
# REINVENT4 multiproperty subprocess-RPC adapter (WF-Extra-2)
# ---------------------------------------------------------------------------
@dataclass
class REINVENT4MultipropertyAdapter:
    """Subprocess-RPC adapter for the REINVENT4 native multi-property scorer.

    This is the WF-Extra-2 sibling of :class:`REINVENT4Adapter`.  Instead
    of returning a four-tuple of QED / SA / binding / novelty components
    (the four-component proxy), this adapter drives REINVENT4's *native*
    multi-property objective (sum of weighted components — logP, ring
    count, QED, custom matching, etc.) and returns a single float in
    ``[0, 1]`` (the weighted aggregate produced by REINVENT4 itself).

    Wire format is line-delimited JSON over stdin/stdout — same contract
    as :class:`REINVENT4Adapter` but with a different op / payload::

        {"op": "score", "smiles": ["CCO"], "components": {"logp": 0.4, "qed": 0.4, "ring_count": 0.2}}
        <- {"results": [{"multiproperty_score": 0.85, "ok": true}, ...]}

    Failure modes are returned per-element so a single bad SMILES does
    not poison the batch: ``{"multiproperty_score": null, "ok": false,
    "reason": "..."}``.

    Parameters
    ----------
    mode : str, default "multiproperty"
        Identifies the wire protocol.  ``"multiproperty"`` is the only
        value implemented here; the constructor refuses anything else
        so a misconfiguration surfaces immediately rather than
        silently switching protocols.
    worker_python : str, default "/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/python"
        Absolute path to the Python interpreter that runs the worker
        subprocess.  The interpreter lives in the isolated REINVENT4
        ROCm project; the upstream torch in this venv is CUDA-built and
        intentionally NOT importable from the search project.
    worker_script : str, default
        "<this_module_dir>/reinvent4_multiproperty_jsonl_worker.py"
        Path to the JSON-lines RPC worker that talks to the upstream
        ``reinvent`` CLI.
    components : dict[str, float], default {"logp": 0.4, "ring_count": 0.2, "qed": 0.4}
        Per-component weights for REINVENT4's multi-property objective.
        See :mod:`reinvent4_multiproperty_jsonl_worker` for the full
        vocabulary.  Unknown keys are silently ignored by the worker.
    device : str, default "cuda:0"
        Logical device string.  Currently informational only — the
        worker itself runs on CPU (REINVENT4's built-in components
        don't need a GPU).
    timeout : float, default 60.0
        Per-call wall-clock timeout for one batch RPC roundtrip.

    Attributes
    ----------
    available : bool
        ``True`` iff the worker script exists on disk and we can spawn
        ``worker_python``.  When ``False``, every :meth:`score` returns
        ``0.0`` so the search loop degrades gracefully.
    last_error : str or None
        Structured reason for the most recent unavailability — one of
        ``"config_invalid"``, ``"worker_missing"``, ``"python_missing"``,
        ``"worker_spawn_failed"``, ``"rpc_error"``, or ``None``.
    """

    mode: str = "multiproperty"
    worker_python: str = "/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/python"
    worker_script: str = ""  # filled in __post_init__ to the default below
    components: Dict[str, float] = None  # type: ignore[assignment]
    device: str = "cuda:0"
    timeout: float = 60.0

    #: Set after the constructor probe.
    available: bool = False
    #: Structured failure reason; ``None`` when the adapter is healthy.
    last_error: Optional[str] = None
    #: Lazily-created worker subprocess (kept open across calls).
    _proc: Optional[subprocess.Popen] = None
    _lock: threading.Lock = None  # type: ignore[assignment]

    #: Default worker script path (relative to this module).
    DEFAULT_WORKER_SCRIPT = "reinvent4_multiproperty_jsonl_worker.py"
    #: Default component weights — the WF-Extra-2 task spec.
    DEFAULT_COMPONENTS: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode.strip().lower() != "multiproperty":
            # We refuse to silently default to a different protocol.  A
            # caller that passes ``mode=prior_nll`` should be using
            # :class:`reinvent_prior_adapter.REINVENT4PriorAdapter`
            # instead.
            self.available = False
            self.last_error = "config_invalid"
            logger.warning(
                "REINVENT4MultipropertyAdapter: unsupported mode=%r "
                "(only 'multiproperty' is implemented here).",
                self.mode,
            )
            return
        if not isinstance(self.components, dict) or not self.components:
            # Fill in the default component weights lazily so the
            # dataclass default-factory contract is honoured even if
            # the caller mutates DEFAULT_COMPONENTS at import time.
            self.components = {
                "logp": 0.4,
                "ring_count": 0.2,
                "qed": 0.4,
            }
        # Resolve worker_script if the caller left it blank.
        if not self.worker_script:
            self.worker_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.DEFAULT_WORKER_SCRIPT,
            )
        # Coerce component weights to floats (skip non-numeric).
        clean: Dict[str, float] = {}
        for k, v in self.components.items():
            try:
                w = float(v)
            except (TypeError, ValueError):
                continue
            if w <= 0.0:
                continue
            clean[str(k)] = w
        self.components = clean or {"qed": 1.0}
        self._lock = threading.Lock()
        if not os.path.isfile(self.worker_script):
            self.available = False
            self.last_error = "worker_missing"
            logger.warning(
                "REINVENT4MultipropertyAdapter: worker_script not found at %s",
                self.worker_script,
            )
            return
        if not os.path.isfile(self.worker_python):
            self.available = False
            self.last_error = "python_missing"
            logger.warning(
                "REINVENT4MultipropertyAdapter: worker_python not found at %s",
                self.worker_python,
            )
            return
        # Best-effort launch the long-lived worker subprocess.
        try:
            self._start_worker()
        except FileNotFoundError as exc:
            self.last_error = "python_missing"
            self.available = False
            logger.warning(
                "REINVENT4MultipropertyAdapter: failed to spawn %s (%s)",
                self.worker_python, exc,
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            self.last_error = "worker_spawn_failed"
            self.available = False
            logger.warning(
                "REINVENT4MultipropertyAdapter: failed to start worker (%s)",
                exc,
            )
            return
        self.available = True

    def _start_worker(self) -> None:
        """Spawn the long-lived multiproperty JSON-lines worker."""
        cmd = [self.worker_python, self.worker_script]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )

    def close(self) -> None:
        """Terminate the worker subprocess and release its file handles."""
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except Exception:  # pragma: no cover
                pass

    def score(self, smiles_or_smiles_batch) -> List[float]:
        """Score a single SMILES or a batch.  Returns 0.0 on any failure.

        Accepts both a single string and an iterable of strings.  The
        return type is always ``list[float]`` so callers can rely on
        the contract: same length as the input, ``0.0`` for any element
        that could not be evaluated (RDKit parse failure, subprocess
        crash, timeout, ``ok=false`` from the worker, etc.).
        """
        if isinstance(smiles_or_smiles_batch, str):
            batch = [smiles_or_smiles_batch]
        else:
            batch = list(smiles_or_smiles_batch)
        if not self.available or self._proc is None:
            return [0.0] * len(batch)
        if not batch:
            return []
        # The worker contract is one SMILES per request — it does NOT
        # accept a list.  We loop over the batch and send one RPC per
        # element so the result is always the same length as the input.
        out: List[float] = []
        for smiles in batch:
            request: Dict[str, Any] = {
                "op": "score",
                "smiles": smiles,
                "components": dict(self.components),
                "timeout": float(self.timeout),
            }
            try:
                with self._lock:
                    resp = self._rpc_roundtrip(request)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("REINVENT4MultipropertyAdapter.score RPC failed: %s", exc)
                self.last_error = "rpc_error"
                out.append(0.0)
                continue
            if resp is None:
                out.append(0.0)
                continue
            r = resp or {}
            if not r.get("ok"):
                # Surface the reason in last_error so a future call can
                # log "why are we getting 0.0?" without re-deriving the
                # cause.  Preserve the older reason if a more specific
                # one is already set.
                reason = r.get("reason") or "worker_error"
                if self.last_error is None:
                    self.last_error = str(reason)
                out.append(0.0)
                continue
            score = r.get("multiproperty_score")
            try:
                fv = float(score)
            except (TypeError, ValueError):
                out.append(0.0)
                continue
            # Clip to [0, 1] defensively — the worker should already
            # clip but a bad config could produce values outside.
            if fv != fv:  # NaN
                out.append(0.0)
                continue
            out.append(float(max(0.0, min(1.0, fv))))
        return out

    def _rpc_roundtrip(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            self.last_error = "rpc_error"
            return None
        if proc.poll() is not None:
            self.last_error = "rpc_error"
            return None
        try:
            line = json.dumps(request, allow_nan=False) + "\n"
            proc.stdin.write(line)
            proc.stdin.flush()
        except Exception as exc:  # pragma: no cover
            logger.warning("REINVENT4MultipropertyAdapter: stdin write failed (%s)", exc)
            self.last_error = "rpc_error"
            return None
        try:
            resp_line = proc.stdout.readline()
        except Exception as exc:  # pragma: no cover
            logger.warning("REINVENT4MultipropertyAdapter: stdout read failed (%s)", exc)
            self.last_error = "rpc_error"
            return None
        if not resp_line:
            self.last_error = "rpc_error"
            return None
        try:
            return json.loads(resp_line)
        except json.JSONDecodeError as exc:
            logger.warning("REINVENT4MultipropertyAdapter: malformed RPC response (%s)", exc)
            self.last_error = "rpc_error"
            return None

    @classmethod
    def from_config(cls, path: str) -> "REINVENT4MultipropertyAdapter":
        """Build an adapter from a JSON config file.

        The config must declare ``"mode": "multiproperty"``.  Unknown
        keys are silently dropped.  ``components`` is optional — if
        missing or empty, the default weights
        ``{"logp": 0.4, "ring_count": 0.2, "qed": 0.4}`` are used.
        """
        with open(path) as fh:
            config = json.load(fh)
        if not isinstance(config, dict):
            raise ValueError("reinvent_multiproperty config must be a JSON object")
        if config.get("mode", "").lower() != "multiproperty":
            raise ValueError(
                f"Expected mode='multiproperty' in {path}, got {config.get('mode')!r}"
            )
        # Resolve the worker_script relative to the config directory
        # when it is not absolute — keeps the config portable across
        # checkouts.
        worker_script = config.get("worker_script", "")
        if worker_script and not os.path.isabs(worker_script):
            base = os.path.dirname(os.path.abspath(path))
            candidate = os.path.normpath(os.path.join(base, worker_script))
            if os.path.isfile(candidate):
                worker_script = candidate
        return cls(
            mode=config.get("mode", "multiproperty"),
            worker_python=config.get("worker_python", cls.worker_python),
            worker_script=worker_script,
            components=config.get("components") or None,
            device=config.get("device", "cuda:0"),
            timeout=float(config.get("timeout", 60.0)),
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest() -> None:  # pragma: no cover
    agg = ScoreAggregator()
    perfect = ScoreResult(qed=1.0, sa=1.0, binding=1.0, novelty=1.0)
    assert abs(agg.aggregate(perfect) - agg.total_weight()) < 1e-9, (
        f"perfect molecule should aggregate to total_weight, "
        f"got {agg.aggregate(perfect)} vs {agg.total_weight()}"
    )


__all__ = [
    "ScoreResult",
    "REINVENT4Adapter",
    "REINVENT4MultipropertyAdapter",
    "ScoreAggregator",
    "is_reinvent4_binary_available",
]
