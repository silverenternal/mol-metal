"""Regression guard — Triton autotune keys must not be per-call dynamic sizes.

WHY THIS EXISTS (2026-10-20, FIX-C9).  ``_aggregate_vectors_kernel`` was
decorated with ``@triton.autotune(..., key=["n_edges", "feat_dim"])``.  The
kernel launches ``grid=(n_edges,)`` — one program per edge — and its body has
no loop over edges, so ``n_edges`` appears only in the guard
``eid >= n_edges``.  The per-program instruction stream is therefore
independent of ``n_edges``, and the optimal ``(num_warps, num_stages)`` cannot
depend on it either.

In the CFM training loop ``n_edges`` changes on every batch, and the eval path
runs one cloud at a time so it changes on every call.  Triton caches autotune
results per key tuple, so this re-benchmarked all 9 configs almost every call.

MEASURED (cProfile, 128-cloud / 2-step CFM run, before the fix): 576 module
forwards produced 981 autotune ``_bench`` calls, 109.4 s inside
``triton.testing.do_bench`` and 40.1 s inside ``torch._C._cuda_synchronize``,
out of 126.9 s of ``train_one_epoch``.

The test is static (reads the decorator's resolved ``keys`` attribute), so it
needs neither CUDA nor a compiled kernel.
"""

from __future__ import annotations

import pytest

pytest.importorskip("triton", reason="triton is not installed in this venv")


def _keys_of(module_path: str, attr: str):
    import importlib

    mod = importlib.import_module(module_path)
    fn = getattr(mod, attr)
    keys = getattr(fn, "keys", None)
    if keys is None:
        pytest.skip(f"{module_path}.{attr} is not wrapped by triton.autotune")
    return list(keys)


def test_aggregate_vectors_autotune_key_is_feat_dim_only():
    """``n_edges`` sets the grid, not the code — it must not be a key."""
    keys = _keys_of("triton_kernels.equivariant_ops", "_aggregate_vectors_kernel")
    assert keys == ["feat_dim"], (
        "aggregate_vectors autotunes on "
        f"{keys}; keying on n_edges re-benchmarks every config whenever the "
        "edge count changes, which is every batch and every eval cloud"
    )


def test_aggregate_vectors_kernel_body_does_not_loop_over_edges():
    """The premise of the key above, asserted rather than assumed.

    If someone later rewrites the kernel to loop over ``n_edges`` inside one
    program (a persistent-kernel shape), ``n_edges`` becomes a legitimate
    tuning axis and the key test above must be revisited.
    """
    import pathlib

    from triton_kernels import equivariant_ops

    src = pathlib.Path(equivariant_ops.__file__).read_text()
    start = src.index("def _aggregate_vectors_kernel(")
    end = src.index("@triton.jit", start)
    body = src[start:end]
    assert "range(n_edges" not in body, (
        "kernel now loops over n_edges; n_edges is a real tuning axis again"
    )
    assert "tl.program_id(axis=0)" in body
