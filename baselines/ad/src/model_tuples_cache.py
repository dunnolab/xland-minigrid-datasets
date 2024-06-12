import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn import TransformerBlock


class KVCache:
    def __init__(
        self,
        batch_size,
        max_seq_len,
        num_layers,
        num_heads,
        head_dim,
        device,
        dtype,
    ):
        # we assume that all layers and all samples in the batch are updating their cache simultaneously
        # and have equal sequence length, i.e. during evaluation on the vector environment
        self.cache_shape = (num_layers, batch_size, max_seq_len, num_heads, head_dim)
        self.k_cache = torch.full(
            self.cache_shape, fill_value=torch.nan, dtype=dtype, device=device
        ).detach()
        self.v_cache = torch.full(
            self.cache_shape, fill_value=torch.nan, dtype=dtype, device=device
        ).detach()
        self.cache_seqlens = 0

    def __len__(self):
        return self.k_cache.shape[0]

    def __getitem__(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx], self.cache_seqlens

    def reset(self):
        self.cache_seqlens = 0

    def update(self):
        self.cache_seqlens = self.cache_seqlens + 1
        if self.cache_seqlens == self.cache_shape[2]:
            self.k_cache = torch.roll(self.k_cache, -1, dims=2)
            self.v_cache = torch.roll(self.v_cache, -1, dims=2)
            self.cache_seqlens = self.cache_seqlens - 1
            assert self.cache_seqlens >= 0, "negative cache sequence length"
            # for debug purposes
            self.k_cache[:, :, -1] = torch.nan
            self.v_cache[:, :, -1] = torch.nan


class Transformer(nn.Module):
    def __init__(
        self,
        seq_len: int = 40,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        attention_dropout: float = 0.5,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.1,
        normalize_qk: bool = False,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb2hid = nn.Linear(embedding_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    normalize_qk=normalize_qk,
                    pre_norm=pre_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        # taken from the nanoGPT, may be not optimal
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def init_cache(self, batch_size, dtype, device):
        cache = KVCache(
            batch_size=batch_size,
            max_seq_len=self.seq_len,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.hidden_dim // self.num_heads,
            device=device,
            dtype=dtype,
        )
        return cache

    def forward(self, sequence, cache: KVCache = None):
        _cache = cache or [(None, None, None) for _ in range(self.num_layers)]

        # [batch_size, seq_len, hidden_dim]
        sequence = self.emb2hid(sequence)

        out = self.emb_drop(sequence)
        for i, block in enumerate(self.blocks):
            out = block(out, *_cache[i])

        if cache is not None:
            cache.update()

        return out, cache
