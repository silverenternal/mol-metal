"""3D-coordinate transforms for MolFlow-Triton.

This module provides a tiny torchvision-compatible API for 3D geometry
augmentation.  These transforms are *SE(3)-equivariant* operations on
``(N, 3)`` coordinate tensors: by randomizing rotation + translation,
the model never gets to rely on absolute pose of the molecule, which
is a cheap and well-known regularizer for EGNN-style networks
(see :class:`models.velocity_net.VelocityNet`).

Design choices
--------------

* The transforms work on plain ``torch.Tensor`` of shape ``(N, 3)``.
  They do not mutate the input; they return a new tensor.
* The rotation matrix is built from a QR decomposition of a
  Gaussian matrix.  This is the standard trick for sampling a uniform
  random rotation and only needs ``torch``, no ``scipy``.
* The random rotation / translation are deterministic functions of
  the ``torch.Generator`` passed in.  None of them have hidden state;
  pass a generator to make tests reproducible.
* :class:`Compose` mirrors :class:`torchvision.transforms.Compose` -
  it takes a list of callables and applies them in order, passing the
  output of one as the input of the next.

If a transform returns a tuple / dict / dataclass instead of a plain
tensor, :class:`Compose` will look for a ``coords`` attribute (or key
/ field) before falling back to the raw value.  That makes the same
:class:`Compose` work for both flat ``(N, 3)`` inputs and
:class:`data.mol_dataset.MoleculeSample` objects.
"""

from __future__ import annotations

from dataclasses import is_dataclass, fields
from typing import Any, Callable, Sequence

import torch

__all__ = [
    "random_rotation_3d",
    "random_translation_3d",
    "Compose",
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _coords_of(sample: Any) -> torch.Tensor | None:
    """Return the ``coords`` attribute / item of ``sample`` if present.

    Used by :class:`Compose` to support :class:`MoleculeSample`-like
    objects alongside raw ``(N, 3)`` tensors.
    """
    if isinstance(sample, torch.Tensor):
        return sample
    if isinstance(sample, dict):
        coords = sample.get("coords")
        return coords if isinstance(coords, torch.Tensor) else None
    if is_dataclass(sample):
        for f in fields(sample):
            if f.name == "coords":
                val = getattr(sample, f.name)
                if isinstance(val, torch.Tensor):
                    return val
        return None
    if hasattr(sample, "coords"):
        val = sample.coords
        if isinstance(val, torch.Tensor):
            return val
    return None


def _set_coords(sample: Any, new_coords: torch.Tensor) -> Any:
    """Return ``sample`` with its ``coords`` field replaced.

    Mirrors :func:`_coords_of`.  Falls back to ``new_coords`` itself
    for raw tensor inputs (which :class:`Compose` then forwards
    unmodified - the caller will receive the new tensor directly).
    """
    if isinstance(sample, torch.Tensor):
        return new_coords
    if isinstance(sample, dict):
        new = dict(sample)
        new["coords"] = new_coords
        return new
    if is_dataclass(sample):
        # Build a *new* instance rather than mutating so callers can
        # safely use these transforms in transforms pipelines.
        from dataclasses import replace
        return replace(sample, coords=new_coords)
    if hasattr(sample, "coords"):
        sample.coords = new_coords  # type: ignore[attr-defined]
        return sample
    return new_coords


# ----------------------------------------------------------------------
# Random rotation
# ----------------------------------------------------------------------
def random_rotation_3d(
    coords: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a uniform random rotation and apply it to ``coords``.

    Parameters
    ----------
    coords:
        ``(N, 3)`` coordinate tensor.  Any number of leading batch
        dimensions is supported (e.g. ``(B, N, 3)``), but the last two
        dims must be ``(N, 3)``.
    generator:
        Optional ``torch.Generator``.  Pass a seeded generator to get
        reproducible rotations.

    Returns
    -------
    A new tensor of the same shape and dtype as ``coords`` with the
    rotation applied.
    """
    if coords.shape[-1] != 3:
        raise ValueError(
            f"random_rotation_3d expects last dim = 3, got {coords.shape}"
        )
    device = coords.device
    dtype = coords.dtype

    # QR-based random orthogonal matrix: draw N(0,1), QR-decompose,
    # fix sign of Q so det(Q) = +1.  This is the standard recipe for
    # sampling from the Haar measure on SO(3).
    randn = torch.randn(3, 3, device=device, dtype=dtype, generator=generator)
    q, r = torch.linalg.qr(randn)
    # diag(R) -> +/-1; flip Q columns whose R entry is negative to keep
    # Q in SO(3) (determinant +1) and not just O(3).
    sign = torch.sign(torch.diagonal(r))
    # det(sign) can flip the determinant back; multiply one extra
    # column by det(sign) so the final matrix has det = +1.
    sign = sign * torch.sign(sign.prod())
    q = q * sign.unsqueeze(0)

    # Apply: x' = x @ Q^T  (Q is (3,3), coords is (..., N, 3))
    return coords @ q.transpose(-1, -2)


# ----------------------------------------------------------------------
# Random translation
# ----------------------------------------------------------------------
def random_translation_3d(
    coords: torch.Tensor,
    *,
    max_translation: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply a uniform random translation in ``[-max, max]`` per axis.

    Useful on top of :func:`random_rotation_3d` to make the model
    invariant to the absolute location of the molecule (e.g. when the
    3D coords come from SDF files where the origin is arbitrary).
    """
    if coords.shape[-1] != 3:
        raise ValueError(
            f"random_translation_3d expects last dim = 3, got {coords.shape}"
        )
    device = coords.device
    dtype = coords.dtype

    translation = (
        torch.rand(3, device=device, dtype=dtype, generator=generator) * 2.0 - 1.0
    ) * max_translation
    # Broadcast over leading dims.
    return coords + translation


# ----------------------------------------------------------------------
# Compose (torchvision-style)
# ----------------------------------------------------------------------
class Compose:
    """Sequentially apply a list of transforms, torchvision-style.

    Example
    -------
    >>> transform = Compose([
    ...     lambda c: random_rotation_3d(c, generator=g),
    ...     lambda c: random_translation_3d(c, max_translation=0.5, generator=g),
    ... ])
    >>> out = transform(coords)

    If a transform returns a :class:`MoleculeSample` / dict / dataclass
    with a ``coords`` field, :class:`Compose` will thread that field
    through the pipeline and re-pack the result before returning.
    Raw ``(N, 3)`` tensors pass through unchanged.
    """

    def __init__(self, transforms: Sequence[Callable[[Any], Any]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Any) -> Any:
        current = sample
        for t in self.transforms:
            coords = _coords_of(current)
            if coords is None:
                # Not a 3D-aware sample; let the transform handle it
                # directly (e.g. a numpy -> tensor cast).
                current = t(current)
                continue
            new_coords = t(coords)
            current = _set_coords(current, new_coords)
        return current

    def __repr__(self) -> str:
        body = ",\n    ".join(repr(t) for t in self.transforms)
        return f"Compose([\n    {body}\n])"