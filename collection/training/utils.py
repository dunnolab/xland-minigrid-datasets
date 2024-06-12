# utilities for PPO training and evaluation
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import struct
from flax.training.train_state import TrainState
from xminigrid.core.constants import NUM_COLORS, NUM_TILES
from xminigrid.environment import Environment, EnvParams


# Training stuff
class Transition(struct.PyTreeNode):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    # for rnn policy
    prev_action: jax.Array
    # for context
    goal_encoding: jax.Array
    rule_encoding: jax.Array


def calculate_gae(
    transitions: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = transition.reward + gamma * next_value * (1 - transition.done) - transition.value
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


def ppo_update_networks(
    rng: jax.Array,
    train_state: TrainState,
    transitions: Transition,
    init_hstate: jax.Array,
    advantages: jax.Array,
    targets: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _loss_fn(params):
        # RERUN NETWORK
        dist, value, _ = train_state.apply_fn(
            params,
            {
                # [batch_size, seq_len, ...]
                "observation": transitions.obs,
                "prev_action": transitions.prev_action,
                "goal_encoding": transitions.goal_encoding,
                "rule_encoding": transitions.rule_encoding,
            },
            init_hstate,
            training=True,
            rngs={"dropout": rng},
        )
        log_prob = dist.log_prob(transitions.action)

        # # CALCULATE VALUE LOSS
        value_loss = jnp.square(value - targets).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
        entropy = dist.entropy().mean()

        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
    }
    return train_state, update_info


# for evaluation (evaluate for N consecutive episodes, sum rewards)
# N=1 single task, N>1 for meta-RL
class RolloutStats(struct.PyTreeNode):
    reward: jax.Array = jnp.asarray(0.0)
    length: jax.Array = jnp.asarray(0)
    episodes: jax.Array = jnp.asarray(0)


def rollout(
    rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: jax.Array,
    dropout_enabled: bool = False,
    num_consecutive_episodes: int = 1,
) -> RolloutStats:
    def _cond_fn(carry):
        rng, stats, timestep, prev_action, hstate = carry
        return jnp.less(stats.episodes, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, timestep, prev_action, hstate = carry

        rng, _rng, _rng_drop = jax.random.split(rng, num=3)
        dist, _, hstate = train_state.apply_fn(
            train_state.params,
            {
                "observation": timestep.observation[None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "goal_encoding": timestep.state.goal_encoding[None, None, ...],
                "rule_encoding": timestep.state.rule_encoding[None, None, ...],
            },
            hstate,
            training=dropout_enabled,
            rngs={"dropout": _rng_drop},
        )
        action = dist.sample(seed=_rng).squeeze()
        timestep = env.step(env_params, timestep, action)

        stats = stats.replace(
            reward=stats.reward + timestep.reward,
            length=stats.length + 1,
            episodes=stats.episodes + timestep.last(),
        )
        carry = (rng, stats, timestep, action, hstate)
        return carry

    timestep = env.reset(env_params, rng)
    prev_action = jnp.asarray(0)
    init_carry = (rng, RolloutStats(), timestep, prev_action, init_hstate)

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[1]


def save_checkpoint(path, checkpoint):
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    orbax_checkpointer.save(path, checkpoint)


def load_checkpoint(path):
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    checkpoint = orbax_checkpointer.restore(path)
    return checkpoint


def round_to_multiple(n, denom=64):
    return n + abs((n % denom) - denom)


def compress_obs(obs):
    return obs[..., 0] * NUM_COLORS + obs[..., 1]


def decompress_obs(obs):
    if isinstance(obs, jax.Array):
        return jnp.stack(jnp.divmod(obs, NUM_COLORS), axis=-1)
    elif isinstance(obs, np.ndarray):
        return np.stack(np.divmod(obs, NUM_COLORS), axis=-1)
    else:
        raise RuntimeError("obs should be numpy or jax array.")
