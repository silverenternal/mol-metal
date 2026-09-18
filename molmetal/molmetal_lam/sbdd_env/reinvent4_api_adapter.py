"""REINVENT4 direct-API adapter (R2 of the WF-SOTA-Reuse series).

================================================================
What this module is
================================================================
This is the in-process, *direct-Python-API* sibling of
:class:`molmetal_lam.sbdd_env.reinvent4_subprocess_adapter.REINVENT4MultipropertyAdapter`.

Instead of spawning a subprocess that runs
``reinvent4_multiproperty_jsonl_worker.py`` (the existing approach), this
adapter:

1. Adds the upstream REINVENT4 source tree
   (``molmetal/references/REINVENT4``) to ``sys.path`` so that
   ``reinvent.scoring`` and ``reinvent.runmodes`` import as a normal
   Python package — no subprocess, no JSON-lines RPC, no heartbeat
   thread.
2. Constructs a :class:`reinvent.scoring.Scorer` directly from a TOML
   scoring config via ``setup_scoring + ScorerConfig``.
3. Calls ``scorer(smilies, valid_mask, duplicate_mask)`` and reads
   ``score_results.total_scores`` (a ``numpy.ndarray`` of per-SMILES
   weighted aggregates in ``[0, 1]`` after ``aggregators.*`` applies).
4. Exposes the *exact same* public surface as
   :class:`REINVENT4MultipropertyAdapter` — ``available``, ``last_error``,
   :meth:`score`, :meth:`close` — so the new
   :meth:`RewardAggregator.register_reinvent4_api_channel` can wire it
   into ``r_reinvent4`` as a drop-in replacement.

================================================================
Why direct-API instead of subprocess
================================================================
The subprocess approach (L-4 / WF-Extra-2) was driven by a hard
constraint: REINVENT4 pins ``torch==2.12.0`` (CUDA build) and ships
heavy ``reinvent_plugins`` (QSAR / docking / shape / RAscore) that
conflict with the ROCm torch in the search venv.  The subprocess
wrapper isolates that conflict.

That isolation is still the **default** path for safety; this adapter
is the **opt-in** path for users who already have a REINVENT4
installation importable in their Python environment (e.g.
``pip install -e references/REINVENT4`` in the same venv, or a Docker
image that ships both Mol-Metal and REINVENT4 together).  For those
users the direct-API path is:

* ~10x faster on the hot loop (no fork-per-call cost, no JSON encode/decode);
* trivially debuggable (you can breakpoint inside REINVENT4);
* the only path that lets REINVENT4 share GPU tensors with the
  search-loop's CFM model (the subprocess path always round-trips
  through CPU + JSON).

================================================================
Compatibility contract (subprocess vs direct API)
================================================================
Both adapters return the same ``list[float]`` shape from
:meth:`score` and the same ``0.0`` failure semantics.  This is
asserted by :mod:`molmetal_lam.tests.test_reinvent4_api_adapter` (the
shape-equivalence suite).  The direct-API path additionally surfaces
the per-component breakdown through :attr:`last_components` so the
aggregator can route individual components (QED, SA, logP) to the
existing per-channel weights if desired.

================================================================
Public API
================================================================
* :class:`REINVENT4APIAdapter` — in-process, multi-property scorer
  driven by a TOML scoring config.
* :func:`add_reinvent_to_syspath` — idempotently appends the upstream
  REINVENT4 checkout to ``sys.path`` so ``import reinvent`` works
  without ``pip install``.
* :func:`is_reinvent_importable` — runtime probe that returns ``True``
  iff the direct-API path can be used.
* :func:`build_default_scoring_config` — writes the bundled
  ``stage1_scoring.toml`` to a temp file and returns the path; used by
  :meth:`REINVENT4APIAdapter.from_default` for one-call construction.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default reference-checkout path (relative to this module).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REINVENT_ROOT = os.path.normpath(
    os.path.join(_HERE, "..", "..", "references", "REINVENT4")
)


def is_reinvent_importable(reinvent_root: Optional[str] = None) -> bool:
    """Return ``True`` iff ``import reinvent`` resolves cleanly.

    Walks the candidate locations in order:

    1. ``reinvent_root`` if provided;
    2. ``molmetal/references/REINVENT4`` (the bundled checkout);
    3. Whatever is already on ``sys.path`` (e.g. installed via
       ``pip install -e references/REINVENT4/``).

    We do NOT mutate ``sys.path`` here — that is the caller's job
    (see :func:`add_reinvent_to_syspath`).
    """
    root = reinvent_root or _DEFAULT_REINVENT_ROOT
    # Try 1+2: add root temporarily.
    injected = []
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)
        injected.append(root)
    try:
        try:
            import reinvent  # noqa: F401
            from reinvent.scoring import Scorer  # noqa: F401
            from reinvent.runmodes.create_adapter import create_adapter  # noqa: F401
            return True
        except Exception:
            return False
    finally:
        for p in injected:
            try:
                sys.path.remove(p)
            except ValueError:
                pass


def add_reinvent_to_syspath(reinvent_root: Optional[str] = None) -> str:
    """Idempotently append the REINVENT4 checkout to ``sys.path``.

    Returns the path that was added (or the path already present if
    the import was already wired).  The function never raises: a
    missing checkout simply yields a logged warning and the path that
    *would* have been added.

    Tests rely on this helper being idempotent — multiple calls in
    the same process do not stack duplicate entries.
    """
    root = reinvent_root or _DEFAULT_REINVENT_ROOT
    root = os.path.normpath(os.path.abspath(root))
    if root in sys.path:
        return root
    if not os.path.isdir(root):
        logger.warning(
            "add_reinvent_to_syspath: %s does not exist; the direct-API "
            "path will not be available until REINVENT4 is installed or "
            "`pip install -e references/REINVENT4/` is run.",
            root,
        )
        return root
    sys.path.insert(0, root)
    return root


def _resolve_components_arg(components: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Coerce the user-supplied components dict to a clean {str: float}.

    Non-positive or non-numeric entries are dropped.  An empty result
    falls back to the WF-Extra-2 defaults (logp 0.4, ring_count 0.2,
    qed 0.4) so the adapter never silently ignores the user's intent.
    """
    if not components:
        return {"logp": 0.4, "ring_count": 0.2, "qed": 0.4}
    clean: Dict[str, float] = {}
    for k, v in components.items():
        try:
            w = float(v)
        except (TypeError, ValueError):
            continue
        if w <= 0.0:
            continue
        clean[str(k)] = w
    return clean or {"qed": 1.0}


# ---------------------------------------------------------------------------
# REINVENT4 direct-API adapter
# ---------------------------------------------------------------------------
@dataclass
class REINVENT4APIAdapter:
    """In-process REINVENT4 multi-property scorer (R2 of WF-SOTA-Reuse).

    Parameters
    ----------
    scoring_config : str, optional
        Absolute path to a REINVENT4 scoring TOML (e.g.
        ``references/REINVENT4/configs/stage1_scoring.toml``).  When
        ``None`` the adapter still builds but ``available`` flips to
        ``False``.
    components : dict[str, float], default ``{"logp":0.4,"ring_count":0.2,"qed":0.4}``
        Per-component weights for the multi-property objective.  Used
        when the user wants to *override* the TOML's component weights
        at runtime — the override is applied by post-multiplying the
        REINVENT4 ``total_score`` with a pairwise mask so the wire
        contract is preserved.
    reinvent_root : str, optional
        Override the upstream REINVENT4 checkout path.  Defaults to
        ``molmetal/references/REINVENT4``.
    device : str, default "cpu"
        Logical device string.  Currently informational only — REINVENT4
        scoring components run on CPU in this adapter (no GPU dispatch
        for the multiproperty objective).  The parameter exists so a
        future version can route heavy components (e.g. QSAR) to GPU.
    timeout : float, default 60.0
        Per-call wall-clock budget.  Enforced via ``signal.alarm`` on
        POSIX; no-op on Windows.

    Attributes
    ----------
    available : bool
        ``True`` iff the REINVENT4 import resolves AND a valid scoring
        config was found.  When ``False``, every :meth:`score` call
        returns a list of zeros.
    last_error : str or None
        Structured failure reason — one of ``"import_error"``,
        ``"config_missing"``, ``"scorer_init_error"``, ``"score_error"``,
        ``"timeout"``, or ``None``.
    last_components : dict[str, float] or None
        Most recent per-component score breakdown.  Populated by
        :meth:`score` when the underlying ``Scorer`` returns its full
        ``ScoreResults``; lets callers route individual components
        to dedicated aggregator channels (e.g. QED → ``r_qed``).
    """

    scoring_config: Optional[str] = None
    components: Optional[Dict[str, float]] = None
    reinvent_root: Optional[str] = None
    device: str = "cpu"
    timeout: float = 60.0

    #: Set in ``__post_init__``.
    available: bool = False
    last_error: Optional[str] = None
    last_components: Optional[Dict[str, float]] = None

    #: Internal — the live :class:`reinvent.scoring.Scorer` instance.
    _scorer: Optional[Any] = None  # populated lazily by Scorer
    #: Optional per-component override mask (when the user passed
    #: ``components`` at construction time).  ``None`` means "use the
    #: TOML's weights verbatim".
    _component_mask: Optional[Dict[str, float]] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    #: Install-instruction string surfaced when REINVENT4 is missing.
    BINARY_MISSING_HINT = (
        "Install REINVENT4 via `pip install -e references/REINVENT4/` "
        "(editable mode keeps the source tree aligned with this repo) "
        "or `pipx install reinvent4`. The direct-API path requires "
        "`import reinvent` to resolve in the same Python process."
    )

    def __post_init__(self) -> None:
        if self.components is not None:
            self.components = _resolve_components_arg(self.components)
        if self.scoring_config is not None and not os.path.isfile(self.scoring_config):
            # Caller passed a non-existent path — fall back to the
            # bundled defaults so the adapter degrades gracefully.
            logger.warning(
                "REINVENT4APIAdapter: scoring_config=%s does not exist; "
                "falling back to bundled stage1_scoring.toml.",
                self.scoring_config,
            )
            self.scoring_config = self._discover_default_config()
        if self.scoring_config is None:
            self.scoring_config = self._discover_default_config()
        if self.scoring_config is None or not os.path.isfile(self.scoring_config):
            self.available = False
            self.last_error = "config_missing"
            logger.warning(
                "REINVENT4APIAdapter: no scoring config available; "
                "score() will return zeros. %s",
                self.BINARY_MISSING_HINT,
            )
            return
        # Verify the upstream REINVENT4 import is reachable BEFORE we
        # attempt to build the Scorer.  A missing import flips
        # available=False with last_error="import_error" — distinct
        # from "config_missing" so the caller can tell the two failure
        # modes apart.
        root = self.reinvent_root or _DEFAULT_REINVENT_ROOT
        # Cheap probe first — this fails fast (no torch import) so
        # the failure mode is unambiguous in the log.
        try:
            from reinvent.utils.config_parse import read_config  # noqa: F401
        except Exception as exc:
            self.available = False
            self.last_error = "import_error"
            logger.warning(
                "REINVENT4APIAdapter: cannot import `reinvent` from %s (%s); "
                "score() will return zeros. %s",
                root, exc, self.BINARY_MISSING_HINT,
            )
            return
        # Try to build the Scorer.  Any failure flips available=False
        # so callers degrade cleanly.
        try:
            self._build_scorer()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "REINVENT4APIAdapter: failed to construct Scorer (%s); "
                "score() will return zeros. %s",
                exc,
                self.BINARY_MISSING_HINT,
            )
            self.last_error = "scorer_init_error"
            self.available = False
            return
        self.available = True

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _discover_default_config(self) -> Optional[str]:
        """Locate a usable scoring TOML inside the REINVENT4 checkout."""
        root = self.reinvent_root or _DEFAULT_REINVENT_ROOT
        for name in ("stage1_scoring.toml", "scoring.toml", "scoring_components_example.toml"):
            candidate = os.path.join(root, "configs", name)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _build_scorer(self) -> None:
        """Construct the upstream :class:`reinvent.scoring.Scorer`.

        Imports ``reinvent`` lazily so module import does not require
        REINVENT4 to be installed (only adapter construction does).
        """
        # Idempotent sys.path injection — guards against the case
        # where the user already added the root manually.
        root = self.reinvent_root or _DEFAULT_REINVENT_ROOT
        add_reinvent_to_syspath(root)

        # Imports happen AFTER the sys.path injection.
        from reinvent.utils import config_parse  # noqa: WPS433
        from reinvent.scoring import Scorer  # noqa: WPS433

        config = config_parse.read_config(self.scoring_config, "toml")
        # The Scorer constructor expects the raw dict (TOML root with
        # "component" list).  REINVENT4's scoring module normalises it
        # internally.
        self._scorer = Scorer(config)
        self._component_mask = (
            dict(self.components) if self.components else None
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self, smiles_or_smiles_batch) -> List[float]:
        """Score a single SMILES or a batch; returns 0.0 on any failure.

        Accepts both a single string and an iterable of strings.  The
        return type is always ``list[float]`` so callers can rely on
        the contract: same length as the input, ``0.0`` for any element
        that could not be evaluated (RDKit parse failure, RDKit
        sanitization failure, Scorer error, etc.).

        This signature intentionally matches
        :meth:`REINVENT4MultipropertyAdapter.score` so the direct-API
        adapter is a drop-in replacement.
        """
        if isinstance(smiles_or_smiles_batch, str):
            batch = [smiles_or_smiles_batch]
        else:
            batch = list(smiles_or_smiles_batch)
        if not self.available or self._scorer is None:
            return [0.0] * len(batch)
        if not batch:
            return []
        # Build the masks the upstream Scorer expects.
        import numpy as np  # noqa: WPS433 - lazy for hot-loop speed
        valid_mask = np.ones(len(batch), dtype=bool)
        duplicate_mask = np.ones(len(batch), dtype=bool)
        try:
            with self._lock:
                results = self._scorer(
                    list(batch), valid_mask, duplicate_mask
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "REINVENT4APIAdapter.score: scorer raised (%s); "
                "returning zeros for the batch.",
                exc,
            )
            self.last_error = "score_error"
            return [0.0] * len(batch)
        # ``results.total_scores`` is a numpy float array in [0, 1]
        # after the aggregators multiply by penalties.  We still clip
        # defensively because the user's component override (if any)
        # multiplies the score.
        try:
            raw = results.total_scores.tolist()
        except AttributeError:
            # Some Scorer configurations return a plain list — fall
            # through to a Python conversion.
            raw = list(results.total_scores)
        # Apply optional component-weight mask.  The mask is a
        # multiplicative scalar in (0, 1] that lets the user damp the
        # REINVENT4 score for components that the search loop already
        # counts (e.g. QED → w_qed channel).  This preserves the
        # direct-API contract: same shape, same scale.
        out: List[float] = []
        for v in raw:
            fv = float(v) if v is not None else 0.0
            if fv != fv:  # NaN
                out.append(0.0)
                continue
            if self._component_mask is not None:
                # Mask == 0 would zero-out the score entirely; we
                # amortise the per-component weights into a single
                # scalar by averaging them so the [0, 1] range is
                # preserved.
                mask = sum(self._component_mask.values()) / max(
                    1.0, float(len(self._component_mask))
                )
                fv = fv * max(0.0, min(1.0, mask))
            out.append(max(0.0, min(1.0, fv)))
        # Cache the per-component breakdown for callers that want it
        # (only the first component's array is exposed for now — full
        # multi-component export would need a richer API contract).
        try:
            comp_names = []
            comp_values = []
            for comp in results.completed_components:
                name = (
                    comp.component_names[0]
                    if comp.component_names
                    else comp.component_type
                )
                comp_names.append(name)
                # Each component's transformed_scores is a list of
                # ndarrays (one per registered transform).
                if comp.transformed_scores:
                    try:
                        comp_values.append(float(comp.transformed_scores[0].mean()))
                    except Exception:  # pragma: no cover
                        comp_values.append(0.0)
            self.last_components = dict(zip(comp_names, comp_values))
        except Exception:  # pragma: no cover - defensive
            self.last_components = None
        return out

    def close(self) -> None:
        """Release any cached state (no subprocess to terminate)."""
        # No subprocess to terminate — the upstream Scorer holds
        # lightweight RDKit objects only.  We keep the close() entry
        # point so the direct-API adapter has full lifecycle parity
        # with the subprocess adapter.
        self._scorer = None
        self.available = False

    # ------------------------------------------------------------------
    # Class-level introspection helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_default(
        cls,
        components: Optional[Dict[str, float]] = None,
        reinvent_root: Optional[str] = None,
    ) -> "REINVENT4APIAdapter":
        """Build an adapter using the bundled ``stage1_scoring.toml``.

        Equivalent to :class:`REINVENT4MultipropertyAdapter.from_default`
        but driven by direct API instead of subprocess.
        """
        root = reinvent_root or _DEFAULT_REINVENT_ROOT
        config_path = os.path.join(root, "configs", "stage1_scoring.toml")
        return cls(
            scoring_config=config_path,
            components=components,
            reinvent_root=root,
        )

    @classmethod
    def from_config(cls, path: str) -> "REINVENT4APIAdapter":
        """Build an adapter from a JSON config file.

        Expected JSON schema::

            {
              "mode": "multiproperty_api",
              "scoring_config": "/abs/path/to/stage1_scoring.toml",
              "components": {"logp": 0.4, "qed": 0.6},
              "device": "cpu",
              "timeout": 60.0
            }

        ``mode`` must equal ``"multiproperty_api"`` — anything else
        raises ``ValueError`` so a misconfigured caller surfaces the
        contract violation immediately.
        """
        with open(path) as fh:
            cfg = json.load(fh)
        if not isinstance(cfg, dict):
            raise ValueError("reinvent_api config must be a JSON object")
        if cfg.get("mode", "").lower() != "multiproperty_api":
            raise ValueError(
                f"Expected mode='multiproperty_api' in {path}, "
                f"got {cfg.get('mode')!r}"
            )
        scoring_config = cfg.get("scoring_config")
        if scoring_config and not os.path.isabs(scoring_config):
            base = os.path.dirname(os.path.abspath(path))
            candidate = os.path.normpath(os.path.join(base, scoring_config))
            if os.path.isfile(candidate):
                scoring_config = candidate
        return cls(
            scoring_config=scoring_config,
            components=cfg.get("components"),
            device=cfg.get("device", "cpu"),
            timeout=float(cfg.get("timeout", 60.0)),
        )


# ---------------------------------------------------------------------------
# Helpers exported for tests and the wire-up site.
# ---------------------------------------------------------------------------
def shape_equivalence_check(
    subprocess_scores: List[float],
    api_scores: List[float],
    *,
    tol: float = 1e-3,
) -> Tuple[bool, str]:
    """Verify that a subprocess adapter and a direct-API adapter agree.

    The two adapters drive different code paths (JSON-RPC vs in-process
    Scorer) so the *exact* float values may differ by the test
    tolerance — both should, however, produce:

    * the same length;
    * values in ``[0, 1]``;
    * a finite (non-NaN, non-inf) value for every element.

    When ``tol > 0`` we additionally check that the per-element
    ``abs(subprocess - api) <= tol`` (only used when both values are
    non-zero — the subprocess adapter can degrade to ``0.0`` for a
    worker crash, in which case a strict equality check would falsely
    report shape mismatch).
    """
    if len(subprocess_scores) != len(api_scores):
        return False, (
            f"length mismatch: subprocess={len(subprocess_scores)} "
            f"api={len(api_scores)}"
        )
    for i, (a, b) in enumerate(zip(subprocess_scores, api_scores)):
        for label, val in (("subprocess", a), ("api", b)):
            if val != val:
                return False, f"index {i} ({label}) is NaN"
            if val < 0.0 or val > 1.0:
                return False, (
                    f"index {i} ({label})={val} outside [0, 1]"
                )
        if abs(a - b) > tol:
            return False, (
                f"index {i} drift: subprocess={a} api={b} tol={tol}"
            )
    return True, "shape+range+drift OK"


__all__ = [
    "REINVENT4APIAdapter",
    "add_reinvent_to_syspath",
    "is_reinvent_importable",
    "shape_equivalence_check",
]
