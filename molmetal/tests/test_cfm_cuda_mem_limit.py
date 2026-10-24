"""MOLMETAL_CUDA_MEM_LIMIT_MIB — the per-process allocator cap.

Why this exists: the card is shared with a co-tenant, so "the job used N GB" has
to be enforceable rather than observed after the fact.  The cap is applied
through ``torch.cuda.set_per_process_memory_fraction``, which makes the caching
allocator raise OOM instead of growing past the fraction.

These tests never touch a GPU: they assert the off paths (unset, non-numeric,
CPU device) and that a bad value is reported rather than raised, because a
failed cap must not kill a run that would otherwise train.
"""
import pytest
import torch

from molmetal.scripts.train_cfm_yuelbond_wasserstein import _apply_cuda_mem_limit


def test_unset_is_a_no_op(monkeypatch, capsys):
    monkeypatch.delenv("MOLMETAL_CUDA_MEM_LIMIT_MIB", raising=False)
    _apply_cuda_mem_limit(torch.device("cuda"))
    assert capsys.readouterr().out == ""


def test_zero_is_a_no_op(monkeypatch, capsys):
    monkeypatch.setenv("MOLMETAL_CUDA_MEM_LIMIT_MIB", "0")
    _apply_cuda_mem_limit(torch.device("cuda"))
    assert capsys.readouterr().out == ""


def test_non_numeric_is_reported_not_raised(monkeypatch, capsys):
    monkeypatch.setenv("MOLMETAL_CUDA_MEM_LIMIT_MIB", "twenty")
    _apply_cuda_mem_limit(torch.device("cuda"))
    out = capsys.readouterr().out
    assert "not a number" in out


def test_cpu_device_is_reported_not_raised(monkeypatch, capsys):
    monkeypatch.setenv("MOLMETAL_CUDA_MEM_LIMIT_MIB", "20480")
    _apply_cuda_mem_limit(torch.device("cpu"))
    out = capsys.readouterr().out
    assert "ignored" in out


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="needs a CUDA device"
)
def test_cuda_cap_is_applied(monkeypatch, capsys):
    monkeypatch.setenv("MOLMETAL_CUDA_MEM_LIMIT_MIB", "1024")
    _apply_cuda_mem_limit(torch.device("cuda"))
    out = capsys.readouterr().out
    assert "allocator capped at 1024 MiB" in out
    # restore so the rest of the session is not capped at 1 GiB
    torch.cuda.set_per_process_memory_fraction(1.0, torch.cuda.current_device())
