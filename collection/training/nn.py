# Model adapted from XLand-MiniGrid baselines:
# https://github.com/corl-team/xland-minigrid
import math
from typing import Optional, TypedDict

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import glorot_normal, orthogonal, zeros_init
from flax.typing import Dtype
from utils import round_to_multiple
from xminigrid.core.constants import NUM_COLORS, NUM_TILES


# NB: tried to reset hidden state on done, worked better without it!
class GRU(nn.Module):
    hidden_dim: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, xs, init_state):
        seq_len, input_dim = xs.shape
        # this init might not be optimal, for example bias for reset gate should be -1 (for now ok)
        Wi = self.param(
            "Wi", glorot_normal(in_axis=1, out_axis=0, dtype=self.param_dtype), (self.hidden_dim * 3, input_dim)
        )
        Wh = self.param("Wh", orthogonal(column_axis=0, dtype=self.param_dtype), (self.hidden_dim * 3, self.hidden_dim))
        bi = self.param("bi", zeros_init(), (self.hidden_dim * 3,))
        bn = self.param("bn", zeros_init(), (self.hidden_dim,))

        def _step_fn(h, x):
            igates = jnp.split(Wi @ x + bi, 3)
            hgates = jnp.split(Wh @ h, 3)

            reset = nn.sigmoid(igates[0] + hgates[0])
            update = nn.sigmoid(igates[1] + hgates[1])
            new = nn.tanh(igates[2] + reset * (hgates[2] + bn))
            next_h = (1 - update) * new + update * h

            return next_h, next_h

        xs, init_state, Wi, Wh, bi, bn = promote_dtype(xs, init_state, Wi, Wh, bi, bn, dtype=self.dtype)
        last_state, all_states = jax.lax.scan(_step_fn, init=init_state, xs=xs)
        return all_states, last_state


class RNNModel(nn.Module):
    hidden_dim: int
    num_layers: int
    dtype: Optional[Dtype] = None

    @nn.compact
    def __call__(self, xs, init_state):
        # xs: [seq_len, input_dim]
        # init_state: [num_layers, hidden_dim]
        outs, states = [], []
        for layer in range(self.num_layers):
            xs, state = GRU(hidden_dim=self.hidden_dim, dtype=self.dtype)(xs, init_state[layer])
            outs.append(xs)
            states.append(state)

        # sum outputs from all layers, kinda like in ResNet
        return jnp.array(outs).sum(0), jnp.array(states)


BatchedRNNModel = flax.linen.vmap(
    RNNModel, variable_axes={"params": None}, split_rngs={"params": False}, axis_name="batch"
)


class RulesAndGoalsEncoder(nn.Module):
    emb_dim: float = 8
    hidden_dim: float = 256
    dropout: float = 0.0
    dtype: Optional[Dtype] = None

    @nn.compact
    def __call__(self, goal, rules, training: bool):
        B, S, *_ = goal.shape
        # slows down everything tremendously in bf16!!!!
        # goal_encoder = nn.Embed(15, self.emb_dim, dtype=self.dtype)
        # rules_encoder = nn.Embed(15, self.emb_dim, dtype=self.dtype)
        goal_encoder = nn.Embed(round_to_multiple(15, denom=64), self.emb_dim)
        rules_encoder = nn.Embed(round_to_multiple(15, denom=64), self.emb_dim)
        head = nn.Dense(self.hidden_dim, dtype=self.dtype)

        goal_emb = goal_encoder(goal).reshape(B, S, -1)
        rules_emb = rules_encoder(rules).reshape(B, S, -1)
        both_emb = jnp.concatenate([goal_emb, rules_emb], axis=-1)
        both_emb = head(both_emb)
        both_emb = nn.Dropout(rate=self.dropout, deterministic=not training)(both_emb)

        return both_emb


class EmbeddingEncoder(nn.Module):
    emb_dim: int = 2
    dtype: Optional[Dtype] = None

    @nn.compact
    def __call__(self, img):
        # slows down everything tremendously in bf16!!!!
        # entity_emb = nn.Embed(NUM_TILES, self.emb_dim, dtype=self.dtype)
        # color_emb = nn.Embed(NUM_COLORS, self.emb_dim, dtype=self.dtype)
        entity_emb = nn.Embed(round_to_multiple(NUM_TILES, denom=64), self.emb_dim)
        color_emb = nn.Embed(round_to_multiple(NUM_COLORS, denom=64), self.emb_dim)

        # [..., channels]
        img_emb = jnp.concatenate(
            [
                entity_emb(img[..., 0]),
                color_emb(img[..., 1]),
            ],
            axis=-1,
        )
        return img_emb


class ObservationEncoder(nn.Module):
    emb_dim: int = 16
    dtype: Optional[Dtype] = None

    @nn.compact
    def __call__(self, obs):
        B, S = obs.shape[:2]
        img_encoder = nn.Sequential(
            [
                EmbeddingEncoder(self.emb_dim, dtype=self.dtype),
                nn.Conv(
                    16,
                    (2, 2),
                    padding="VALID",
                    kernel_init=orthogonal(math.sqrt(2)),
                    dtype=self.dtype,
                ),
                nn.relu,
                nn.Conv(
                    32,
                    (2, 2),
                    padding="VALID",
                    kernel_init=orthogonal(math.sqrt(2)),
                    dtype=self.dtype,
                ),
                nn.relu,
                nn.Conv(
                    64,
                    (2, 2),
                    padding="VALID",
                    kernel_init=orthogonal(math.sqrt(2)),
                    dtype=self.dtype,
                ),
                nn.relu,
            ]
        )
        out = img_encoder(obs).reshape(B, S, -1)
        return out


class ActorCriticInput(TypedDict):
    observation: jax.Array
    prev_action: jax.Array
    goal_encoding: jax.Array
    rule_encoding: jax.Array


class ActorCriticRNN(nn.Module):
    num_actions: int
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    context_emb_dim: int = 16
    context_hidden_dim: int = 64
    context_dropout: float = 0.0
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 1
    head_hidden_dim: int = 64
    dtype: Optional[Dtype] = None

    @nn.compact
    def __call__(
        self, inputs: ActorCriticInput, hidden: jax.Array, training: bool
    ) -> tuple[distrax.Categorical, jax.Array, jax.Array]:
        # just in case...
        inputs = jax.tree_util.tree_map(lambda x: x.astype(jnp.int32), inputs)

        obs_encoder = ObservationEncoder(self.obs_emb_dim, dtype=self.dtype)
        act_encoder = nn.Embed(round_to_multiple(self.num_actions, denom=64), self.action_emb_dim)
        context_encoder = RulesAndGoalsEncoder(
            self.context_emb_dim, self.context_hidden_dim, self.context_dropout, dtype=self.dtype
        )

        rnn_core = BatchedRNNModel(self.rnn_hidden_dim, self.rnn_num_layers, dtype=self.dtype)
        actor = nn.Sequential(
            [
                nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2), dtype=self.dtype),
                nn.tanh,
                nn.Dense(self.num_actions, kernel_init=orthogonal(0.01), dtype=self.dtype),
            ]
        )
        critic = nn.Sequential(
            [
                nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2), dtype=self.dtype),
                nn.tanh,
                nn.Dense(1, kernel_init=orthogonal(1.0), dtype=self.dtype),
            ]
        )

        # [batch_size, seq_len, ...]
        obs_emb = obs_encoder(inputs["observation"])
        act_emb = act_encoder(inputs["prev_action"])
        context_emb = context_encoder(inputs["goal_encoding"], inputs["rule_encoding"], training=training)

        # TODO: add multiplicative gating?
        # [batch_size, seq_len, hidden_dim + act_emb_dim + context_emb_dim]
        out = jnp.concatenate([obs_emb, act_emb, context_emb], axis=-1)
        # core networks
        out, new_hidden = rnn_core(out, hidden)

        # concat context again (to propagate better)
        out = jnp.concatenate([out, context_emb], axis=-1)

        # casting to full precision for the loss, as softmax/log_softmax
        # (inside Categorical) is not stable in bf16
        logits = actor(out).astype(jnp.float32)
        
        dist = distrax.Categorical(logits=logits)
        values = critic(out).squeeze(axis=-1).astype(jnp.float32)

        return dist, values, new_hidden

    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, self.rnn_num_layers, self.rnn_hidden_dim), dtype=self.dtype)
