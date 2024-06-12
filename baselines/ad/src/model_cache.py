import torch
import torch.nn as nn

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
        assert max_seq_len % 3 == 0, "cache should be divisible by 3: (s, a, r)"
        self.cache_shape = (num_layers, batch_size, max_seq_len, num_heads, head_dim)
        self.k_cache = torch.full(self.cache_shape, fill_value=torch.nan, dtype=dtype, device=device).detach()
        self.v_cache = torch.full(self.cache_shape, fill_value=torch.nan, dtype=dtype, device=device).detach()
        self.cache_seqlens = 0

    def __len__(self):
        return self.k_cache.shape[0]

    def __getitem__(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx], self.cache_seqlens

    def reset(self):
        self.cache_seqlens = 0

    def update(self):
        # on each inference step we add 3 steps to the cache for (prev_a, prev_r, s)
        self.cache_seqlens = self.cache_seqlens + 3
        if self.cache_seqlens == self.cache_shape[2]:
            self.k_cache = torch.roll(self.k_cache, -3, dims=2)
            self.v_cache = torch.roll(self.v_cache, -3, dims=2)
            self.cache_seqlens = self.cache_seqlens - 3
            assert self.cache_seqlens >= 0, "negative cache sequence length"
            # for debug purposes
            self.k_cache[:, :, -3:] = torch.nan
            self.v_cache[:, :, -3:] = torch.nan


class ADVanilla(nn.Module):
    def __init__(
            self,
            num_states: int,
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
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)

        self.state_emb = nn.Embedding(num_states, embedding_dim)
        self.action_emb = nn.Embedding(num_actions, embedding_dim)
        self.reward_emb = nn.Linear(1, embedding_dim)

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
        self.action_head = nn.Linear(hidden_dim, num_actions)

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_states = num_states
        self.num_actions = num_actions

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

    def init_cache(self, batch_size, dtype, device):
        cache = KVCache(
            batch_size=batch_size,
            max_seq_len=3 * self.seq_len,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.hidden_dim // self.num_heads,
            device=device,
            dtype=dtype,
        )
        return cache

    def forward(self, states, actions, rewards):
        # [batch_size, seq_len]
        assert states.shape[1] == actions.shape[1] == rewards.shape[1]
        batch_size, seq_len = states.shape[0], states.shape[1]

        assert states.ndim == 2 and actions.ndim == 2 and rewards.ndim == 2
        state_emb = self.state_emb(states)
        act_emb = self.action_emb(actions)
        rew_emb = self.reward_emb(rewards.unsqueeze(-1)).squeeze(-1)

        assert state_emb.shape == act_emb.shape == rew_emb.shape
        # [batch_size, 3 * seq_len, emb_dim], (s_0, a_0, r_0, s_1, a_1, r_1, ...)
        sequence = (
            torch.stack([state_emb, act_emb, rew_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        sequence = self.emb2hid(sequence)

        out = self.emb_drop(sequence)
        for block in self.blocks:
            out = block(out)

        # [batch_size, seq_len, num_actions]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 0::3])

        return out

    def step(self, prev_action, prev_reward, state, cache: KVCache):
        assert not self.training
        # during inference, we have the following process:
        # s_0 -> a_0, r_0, s_1 -> a_1, r_1, s_2 == (s_0, a_0, r_0, s_1, a_1, r_1, s_2, ...)
        # which is equivalent to the sequence used during training,
        # but it is convenient to use prev_a, prev_r, s to simplify correct cache management
        assert state.ndim == 2 and prev_action.ndim == 2 and prev_reward.ndim == 2
        assert state.shape[1] == prev_action.shape[1] == prev_reward.shape[1] == 1

        state_emb = self.state_emb(state)
        act_emb = self.action_emb(prev_action)
        rew_emb = self.reward_emb(prev_reward.unsqueeze(-1)).squeeze(-1)

        # [batch_size, seq_len==3, emb_dim], (prev_a, prev_r, s)
        sequence = torch.concatenate([act_emb, rew_emb, state_emb], dim=1)
        assert sequence.shape[1] == 3
        sequence = self.emb2hid(sequence)

        out = self.emb_drop(sequence)
        for i, block in enumerate(self.blocks):
            out = block(out, *cache[i])

        # [batch_size, 1, num_actions]
        out = self.action_head(out[:, 0::3])

        # if exceeded training seq-len, roll buffer 3 steps
        # back to free up space for the next (prev_a, prev_r, s)
        cache.update()

        return out, cache