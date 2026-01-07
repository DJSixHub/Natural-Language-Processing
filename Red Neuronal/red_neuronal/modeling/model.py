"""
Consolidated Transformer architecture module
Includes RMSNorm, RoPE, SwiGLU, and Decoder-only Transformer
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 1408  # SwiGLU style
    rope_base: float = 10_000.0
    dropout: float = 0.0


# === RMSNorm ===

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


# === RoPE (Rotary Position Embedding) ===

def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10_000.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RoPE cos/sin cache"""
    assert head_dim % 2 == 0
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to query or key tensors"""
    t = x.size(2)
    cos_t = cos[:t].unsqueeze(0).unsqueeze(0)
    sin_t = sin[:t].unsqueeze(0).unsqueeze(0)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x * cos_t + x_rot * sin_t


# === SwiGLU FFN ===

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))


# === Causal Self-Attention with RoPE ===

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.mask: torch.Tensor
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).bool(),
            persistent=False,
        )

        self.cos_cache: torch.Tensor | None = None
        self.sin_cache: torch.Tensor | None = None

    def _ensure_rope(self, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self.cos_cache is None
            or self.cos_cache.device != device
            or self.cos_cache.dtype != dtype
        ):
            cos, sin = build_rope_cache(
                self.cfg.max_seq_len,
                self.head_dim,
                base=self.cfg.rope_base,
                device=device,
                dtype=dtype,
            )
            self.cos_cache = cos
            self.sin_cache = sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        self._ensure_rope(x.device, x.dtype)

        cos = self.cos_cache
        sin = self.sin_cache
        assert cos is not None and sin is not None

        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=-1)

        q = q.view(b, t, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.cfg.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.mask[:t, :t]
        att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y)


# === Transformer Block ===

class Block(nn.Module):
    """Transformer decoder block with pre-norm"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# === Full Decoder-only Transformer LM ===

class DecoderOnlyTransformerLM(nn.Module):
    """Decoder-only Transformer Language Model"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.vocab_size, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.tok_emb(input_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
