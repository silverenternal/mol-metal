"""pybind11-style adapter stubs.

================================================================
What this replaces
================================================================
Several cloned SBDD repos ship C++/CUDA extensions bound through
``pybind11`` (e.g. EquiBind's geodesic-distance kernel, DiffDock's
custom sphere-sampler, FLOWR's geometric primitives). In our abstract
layer we don't link those extensions — ROCm wheels for ``pybind11``
extensions are rarely pre-built and rebuilding from source is slow
and out of scope for Phase 0.

This module exposes Protocol stubs that **look like** pybind11
extension modules:

* :class:`PybindBindingProtocol` — interface contract for a binding
* :class:`Pybind11StubAdapter`    — a no-op stub implementation
* :func:`is_binding_loaded`       — runtime probe

A future PR can replace the stub with a real ``import`` of the
compiled extension behind the same Protocol so downstream code does
not change.

Reference
---------
pybind11 — Wenzel Jakob et al., 2017.
https://github.com/pybind/pybind11
doi:10.5281/zenodo.1042867

Public API
----------
* :class:`PybindBindingProtocol`
* :class:`Pybind11StubAdapter`
* :func:`is_binding_loaded`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol stub — mirror of a pybind11 module
# ---------------------------------------------------------------------------
class PybindBindingProtocol:
    """Protocol describing the surface a pybind11-bound C++/CUDA module exposes.

    We use a plain class (not ``typing.Protocol``) because pybind11
    modules are concrete at import time; the class only documents the
    expected attribute set, it is *not* an abstract base.
    """

    name: str                       # e.g. "equibind_geodesic_kernel"
    version: str                    # pybind11 module version string
    is_loaded: bool                 # True iff the .so was actually loaded

    def setup(self, device: str = "cuda") -> None:
        """Optional one-shot init — equivalent to module import side-effects."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """The main numerical entry point — passes through to the C++ function."""

    def get_metadata(self) -> dict:
        """Reproducibility info — name, version, repo, paper."""


# ---------------------------------------------------------------------------
# Concrete stub
# ---------------------------------------------------------------------------
@dataclass
class Pybind11StubAdapter:
    """A no-op stub satisfying :class:`PybindBindingProtocol`.

    Records calls so test code can verify the protocol was hit, but
    does not link to any real ``.so``.
    """

    name: str = "pybind11_stub"
    version: str = "0.0.0-stub"
    module_path: str = "<unset>"
    call_log: List[str] = field(default_factory=list)

    @property
    def is_loaded(self) -> bool:
        """A stub is never truly loaded."""
        return False

    def setup(self, device: str = "cuda") -> None:
        self.call_log.append(f"setup(device={device})")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.call_log.append(f"forward(args={len(args)}, kwargs={list(kwargs)})")
        return None

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "module_path": self.module_path,
            "is_loaded": self.is_loaded,
            "stub": True,
        }


# ---------------------------------------------------------------------------
# Runtime probe
# ---------------------------------------------------------------------------
def is_binding_loaded(module_path: str) -> bool:
    """Best-effort runtime probe — checks that the ``.so`` file exists.

    We deliberately do NOT import it because:

    1. ROCm wheels for ``torch_cluster`` / ``torch_scatter`` are rare.
    2. Importing a pybind11 module may eagerly initialise CUDA contexts.
    3. Smoke tests must not depend on ROCm kernel availability.
    """
    import os

    return os.path.isfile(module_path)


__all__ = [
    "PybindBindingProtocol",
    "Pybind11StubAdapter",
    "is_binding_loaded",
]
