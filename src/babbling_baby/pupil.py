"""The pupil: a small byte-level decoder-only transformer with
random initialisation. Predicts the next byte given a context."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


VOCAB_SIZE = 256  # one entry per byte value


@dataclass
class PupilConfig:
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    context_len: int = 128
    ffn_mult: int = 4


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(dim, n_heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        h = self.attn_norm(x)
        x = x + self.attn(h, h, h, mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Pupil(nn.Module):
    def __init__(self, config: PupilConfig | None = None) -> None:
        super().__init__()
        self.config = config or PupilConfig()
        cfg = self.config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, cfg.dim)
        self.position_embedding = nn.Embedding(cfg.context_len, cfg.dim)
        self.blocks = [
            TransformerBlock(cfg.dim, cfg.n_heads, cfg.ffn_mult)
            for _ in range(cfg.n_layers)
        ]
        self.out_norm = nn.LayerNorm(cfg.dim)
        self.out_proj = nn.Linear(cfg.dim, VOCAB_SIZE)

    def __call__(self, tokens: mx.array) -> mx.array:
        # tokens: (batch, seq_len)
        seq_len = tokens.shape[1]
        positions = mx.arange(seq_len)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        for block in self.blocks:
            x = block(x, mask)
        x = self.out_norm(x)
        return self.out_proj(x)  # (batch, seq_len, vocab_size)


def sample_emissions(
    pupil: Pupil,
    n_bytes: int,
    context: bytes = b"",
    temperature: float = 1.0,
) -> bytes:
    """Sample n_bytes from the pupil, optionally conditioned on a
    context (the partner's last word, typically)."""
    cfg = pupil.config
    # Seed the sequence with the context bytes (truncated to fit).
    seq = list(context[-(cfg.context_len - 1) :])
    out: list[int] = []
    for _ in range(n_bytes):
        window = seq[-cfg.context_len :] or [0]
        tokens = mx.array([window], dtype=mx.int32)
        logits = pupil(tokens)
        last_logits = logits[0, -1] / max(temperature, 1e-6)
        probs = mx.softmax(last_logits)
        token = int(mx.random.categorical(mx.log(probs + 1e-9)).item())
        out.append(token)
        seq.append(token)
    return bytes(out)
