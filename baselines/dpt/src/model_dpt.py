import torch
from torch import nn
from torch.nn import functional as F
from src.nn import TransformerBlock, ObservationEncoder


class XMiniGridDPT(nn.Module):
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
            pre_norm: bool = True
            ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        self.seq_len = seq_len

        self.emb_dropout = nn.Dropout(embedding_dropout)

        self.obs_encoder = ObservationEncoder(
            embedding_dim=embedding_dim,
            features_dim=embedding_dim
        )
        
        self.embed_transition = nn.Linear(
            2 * embedding_dim + num_actions + 1, # [state, next_state, action, reward]
            hidden_dim
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    normalize_qk=normalize_qk,
                    pre_norm=pre_norm,
                    with_alibi=False
                )
                for _ in range(num_layers)
            ]
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
        if isinstance(module, nn.Conv2d):
            gain = nn.init.calculate_gain("relu")
            nn.init.orthogonal_(module.weight.data, gain)
            if hasattr(module.bias, "data"):
                module.bias.data.fill_(0.0)
    
    def forward(self,
                query_observations: torch.Tensor, # [batch_size, 5, 5, 2] or [batch_size, 2, 5, 5]
                context_observations: torch.Tensor, # [batch_size, seq_len, 5, 5, 2] or [batch_size, seq_len, 2, 5, 5]
                context_actions: torch.Tensor, # [batch_size, seq_len]
                context_next_observations: torch.Tensor, # [batch_size, seq_len, 5, 5, 2] or [batch_size, seq_len, 2, 5, 5]
                context_rewards: torch.Tensor, # [batch_size, seq_len]
                ) -> torch.Tensor:
        batch_size, seq_len = context_rewards.shape[0], context_rewards.shape[1]
        
        # [batch_size, 1, embedding_dim]
        query_obs_emb = self.obs_encoder(query_observations.unsqueeze(1))
        
        # [batch_size, seq_len, embedding_dim]
        context_obs_emb = self.obs_encoder(context_observations)
        # [batch_size, seq_len, embedding_dim]
        context_next_obs_emb = self.obs_encoder(context_next_observations)
            
        context_actions_emb = F.one_hot(context_actions, num_classes=self.num_actions)

        zeros = torch.zeros(batch_size, 1, device=context_observations.device, dtype=context_observations.dtype)
        
        # [batch_size, seq_len + 1, embedding_dim]
        observation_seq = torch.cat([query_obs_emb, context_obs_emb], dim=1)

        action_seq = torch.cat(
            [zeros.to(context_actions.dtype).unsqueeze(-1).repeat_interleave(self.num_actions, dim=-1),
             context_actions_emb],
            dim=1
        )
        next_observation_seq = torch.cat(
            [zeros.unsqueeze(-1).repeat_interleave(self.embedding_dim, dim=-1),
             context_next_obs_emb],
            dim=1
        )

        # [batch_size, seq_len + 1]
        reward_seq = torch.cat(
            [zeros.to(context_rewards.dtype),
             context_rewards],
            dim=1
        ).unsqueeze(-1)

        sequence = torch.cat(
            [observation_seq, action_seq, next_observation_seq, reward_seq], dim=-1
        )  # [batch_size, seq_len + 1, 2 * state_embedding_dim + num_actions + 1]
        sequence = self.embed_transition(sequence)

        out = self.emb_dropout(sequence)
        for block in self.blocks:
            out = block(out)
        
        # [batch_size, seq_len + 1, num_actions]
        head = self.action_head(out)

        if not self.training:
            return head[:, -1, :]
        # return head[:, 1:, :]
        
        # [batch_size, seq_len + 1, num_actions]
        return head