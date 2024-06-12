import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn
except ImportError:
    warnings.warn("Missing FlashAttention Install", category=Warning)

from xminigrid.core.constants import NUM_TILES, NUM_COLORS


def get_alibi_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


# WARN: flash attention does not have cpu kernel implementation. You can use torch implementation on cpu, but remember
# that results from flash and torch will be different for the same input.
# example: https://github.com/Dao-AILab/flash-attention/issues/383
class FlashAliBiCausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0, normalize_qk=False):
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer(
            "alibi_slopes",
            torch.as_tensor(get_alibi_slopes(num_heads)),
            persistent=False,
        )
        if normalize_qk:
            self.q_norm = nn.LayerNorm(hidden_dim // num_heads)
            self.k_norm = nn.LayerNorm(hidden_dim // num_heads)

        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.normalize_qk = normalize_qk

    def forward(self, x, k_cache=None, v_cache=None, cache_seqlens=None):
        B, L, D = x.size()
        # (batch_size, seq_len, 3, num_heads, head_dim)
        qkv = self.in_proj(x).reshape(B, L, 3, self.num_heads, D // self.num_heads)

        # normalizing q,k, see: https://arxiv.org/abs/2302.05442
        if self.normalize_qk:
            q, k, v = qkv.unbind(2)
            q_norm, k_norm = self.q_norm(q), self.k_norm(k)
            qkv = torch.stack([q_norm, k_norm, v], dim=2).to(qkv.dtype)

        # (batch_size, seq_len, num_heads, head_dim)
        if k_cache is None or v_cache is None or cache_seqlens is None:
            out = flash_attn.flash_attn_qkvpacked_func(
                qkv=qkv,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                alibi_slopes=self.alibi_slopes.to(torch.float32).to(qkv.device),
            )
        else:
            assert not self.training
            q, k, v = qkv.unbind(2)
            out = flash_attn.flash_attn_with_kvcache(
                q=q,
                k=k,
                v=v,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                causal=True,
                alibi_slopes=self.alibi_slopes.to(torch.float32).to(qkv.device),
            )
        # (batch_size, seq_len, hidden_dim)
        out = self.out_proj(out.reshape(B, L, D))
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        normalize_qk: bool = False,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = FlashAliBiCausalSelfAttention(
            hidden_dim, num_heads, attention_dropout, normalize_qk=normalize_qk
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(residual_dropout),
        )
        self.pre_norm = pre_norm

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(self, x, k_cache=None, v_cache=None, cache_seqlens=None):
        if self.pre_norm:
            attention_out = self.attention(
                self.norm1(x),
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
            )
            x = x + self.drop(attention_out)
            x = x + self.mlp(self.norm2(x))
        else:
            attention_out = self.attention(
                x, k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens
            )
            x = self.norm1(x + self.drop(attention_out))
            x = self.norm2(x + self.mlp(x))

        return x


# WARN: these modules are just an examples of attention implementation from scratch
# they are only for educational purposes here!
def get_alibi_relative_positions(seq_len):
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return (x - y).to(torch.float)


class EmbeddingEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 2) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        self.entity_emb = nn.Embedding(NUM_TILES + 1, embedding_dim)
        self.color_emb = nn.Embedding(NUM_COLORS, embedding_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img_emb = torch.cat(
            [self.entity_emb(img[..., 0]), self.color_emb(img[..., 1])], dim=-1
        )
        img_emb.swapaxes_(2, 3).swapaxes_(1, 2)
        # we want to have [bs * seq_len, emb_size * 2, 5, 5]

        return img_emb


class ObservationEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 16, features_dim: int = 64) -> None:
        super().__init__()

        self.embeding_dim = embedding_dim
        self.features_dim = features_dim
        self.transform = EmbeddingEncoder(embedding_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(2 * embedding_dim, 32, (2, 2), padding="valid"),
            nn.ReLU(),
            nn.Conv2d(32, 32, (2, 2), padding="valid"),
            nn.ReLU(),
            nn.Conv2d(32, 32, (2, 2), padding="valid"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, features_dim),
        )

    def forward(self, img: torch.Tensor, cast_to=torch.float32) -> torch.Tensor:
        # img: shape [batch_size, seq_len, 5, 5, 2] or [batch_size, seq_len, 2, 5, 5]
        batch_size, seq_len = img.shape[0], img.shape[1]

        if img.shape != (batch_size, seq_len, 5, 5, 2):
            img.swapaxes_(2, 3).swapaxes_(3, 4)

        assert img.shape == (batch_size, seq_len, 5, 5, 2)

        img_transformed = self.transform(img.flatten(0, 1))
        out = self.encoder(img_transformed.to(cast_to))
        return out.reshape(batch_size, seq_len, -1)
