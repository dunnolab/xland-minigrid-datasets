from typing import Tuple, Optional, Union

import torch
from torch import nn

from src.nn import ObservationEncoder
from src.model_tuples_cache import Transformer, KVCache


class XMiniGridAD(nn.Module):
    def __init__(
        self,
        num_actions: int,
        seq_len: int = 200,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        attention_dropout: float = 0.5,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.1,
        normalize_qk: bool = False,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.seq_len = seq_len

        self.obs_encoder = ObservationEncoder(
            embedding_dim=embedding_dim, features_dim=embedding_dim
        )
        self.action_encoder = nn.Embedding(num_actions, embedding_dim)

        self.transformer = Transformer(
            seq_len=seq_len,
            embedding_dim=2 * embedding_dim + 1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
            embedding_dropout=embedding_dropout,
            normalize_qk=normalize_qk,
            pre_norm=pre_norm,
        )
        self.action_head = nn.Linear(hidden_dim, num_actions)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        # if isinstance(module, nn.Conv2d):
        #     gain = nn.init.calculate_gain("relu")
        #     nn.init.orthogonal_(module.weight.data, gain)
        #     if hasattr(module.bias, "data"):
        #         module.bias.data.fill_(0.0)

    def init_cache(self, batch_size, dtype, device) -> KVCache:
        return self.transformer.init_cache(batch_size, dtype, device)

    def forward(
        self,
        observations: torch.Tensor,  # [batch_size, seq_len, 5, 5, 2] or [batch_size, seq_len, 2, 5, 5]
        prev_actions: torch.Tensor,  # [batch_size, seq_len]
        prev_rewards: torch.Tensor,  # [batch_size, seq_len]
        cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        obs_emb = self.obs_encoder(observations, cast_to=torch.float16)
        action_emb = self.action_encoder(prev_actions).to(torch.float16)
        reward_emb = prev_rewards.unsqueeze(-1).to(torch.float16)

        # [batch_size, seq_len, emb_dim * 3]
        sequence = torch.concatenate([action_emb, reward_emb, obs_emb], dim=-1)
        out, cache = self.transformer(sequence, cache=cache)

        # [batch_size, seq_len, num_actions]
        out = self.action_head(out)
        return out, cache
