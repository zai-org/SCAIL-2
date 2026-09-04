import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "wan" / "modules" / "attention.py"
SPEC = importlib.util.spec_from_file_location("scail_attention", MODULE_PATH)
attention = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(attention)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_uniform_lengths_use_dense_backend(monkeypatch):
    calls = {"dense": 0, "varlen": 0}

    def fake_dense(q, k, v, **kwargs):
        calls["dense"] += 1
        return q

    def fake_varlen(q, k, v, **kwargs):
        calls["varlen"] += 1
        return q

    monkeypatch.setattr(attention.flash_attn, "flash_attn_func", fake_dense)
    monkeypatch.setattr(
        attention.flash_attn, "flash_attn_varlen_func", fake_varlen
    )

    q = torch.randn(2, 32, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    output = attention.flash_attention(q, k, v, version=2)

    assert output.shape == q.shape
    assert calls == {"dense": 1, "varlen": 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_explicit_lengths_keep_varlen_backend(monkeypatch):
    calls = {"dense": 0, "varlen": 0}

    def fake_dense(q, k, v, **kwargs):
        calls["dense"] += 1
        return q

    def fake_varlen(q, k, v, **kwargs):
        calls["varlen"] += 1
        return q

    monkeypatch.setattr(attention.flash_attn, "flash_attn_func", fake_dense)
    monkeypatch.setattr(
        attention.flash_attn, "flash_attn_varlen_func", fake_varlen
    )

    q = torch.randn(2, 32, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lengths = torch.full((2,), 32, device="cuda", dtype=torch.int32)

    output = attention.flash_attention(
        q, k, v, q_lens=lengths, k_lens=lengths, version=2
    )

    assert output.shape == q.shape
    assert calls == {"dense": 0, "varlen": 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("returns_tuple", [False, True])
def test_uniform_lengths_use_fa3_dense_backend(monkeypatch, returns_tuple):
    calls = {"dense": 0, "varlen": 0}

    def fake_dense(q, k, v, **kwargs):
        calls["dense"] += 1
        return (q, None) if returns_tuple else q

    def fake_varlen(q, k, v, **kwargs):
        calls["varlen"] += 1
        return q, None

    monkeypatch.setattr(attention, "FLASH_ATTN_3_AVAILABLE", True)
    monkeypatch.setattr(
        attention,
        "flash_attn_interface",
        SimpleNamespace(
            flash_attn_func=fake_dense,
            flash_attn_varlen_func=fake_varlen,
        ),
        raising=False,
    )

    q = torch.randn(1, 32, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    output = attention.flash_attention(q, k, v, version=3)

    assert output.shape == q.shape
    assert calls == {"dense": 1, "varlen": 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("q_heads,kv_heads", [(8, 8), (8, 2), (8, 1)])
@pytest.mark.parametrize("q_len,k_len", [(64, 64), (32, 48)])
def test_dense_matches_full_length_varlen(
    dtype, causal, batch, q_heads, kv_heads, q_len, k_len
):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    q = torch.randn(
        batch,
        q_len,
        q_heads,
        64,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        batch,
        k_len,
        kv_heads,
        64,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    v = torch.randn(
        batch,
        k_len,
        kv_heads,
        64,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    q_lens = torch.full(
        (batch,), q_len, device="cuda", dtype=torch.int32
    )
    k_lens = torch.full(
        (batch,), k_len, device="cuda", dtype=torch.int32
    )

    dense = attention.flash_attention(q, k, v, causal=causal, version=2)
    varlen = attention.flash_attention(
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
        causal=causal,
        version=2,
    )

    assert dense.shape == varlen.shape == q.shape
    assert dense.dtype == varlen.dtype == dtype
    torch.testing.assert_close(dense, varlen, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dense_preserves_original_float32_dtype():
    q = torch.randn(1, 32, 8, 64, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float32)
    v = torch.randn_like(k)

    output = attention.flash_attention(
        q, k, v, dtype=torch.bfloat16, version=2
    )

    assert output.dtype == torch.float32
