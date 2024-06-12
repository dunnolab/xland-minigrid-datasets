# Adapted from XLand-MiniGrid baselines, source:
# https://github.com/corl-team/xland-minigrid
import gzip
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

import h5py
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import pyrallis
import wandb
import xminigrid
from flax.training.train_state import TrainState
from huggingface_hub import HfApi
from nn import ActorCriticRNN
from utils import (
    Transition,
    calculate_gae,
    compress_obs,
    decompress_obs,
    load_checkpoint,
    ppo_update_networks,
    rollout,
    save_checkpoint,
)
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import Environment, EnvParams
from xminigrid.wrappers import GymAutoResetWrapper


@dataclass
class TrainConfig:
    project: str = "xminigrid-datasets"
    group: str = "default"
    name: str = "ppo-rnn"
    # setup
    pretrain_multitask: bool = False
    env_id: str = "XLand-MiniGrid-R1-9x9"
    benchmark_id: str = "trivial-1m"
    ruleset_id: int = 0
    # agent
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    context_emb_dim: int = 16
    context_hidden_dim: int = 64
    context_dropout: float = 0.0
    rnn_hidden_dim: int = 512
    rnn_num_layers: int = 1
    head_hidden_dim: int = 512
    # training
    num_envs: int = 16384
    num_steps: int = 256
    update_epochs: int = 1
    num_minibatches: int = 16
    total_timesteps: int = 1_000_000
    lr: float = 0.001
    decay_lr: bool = True
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_episodes: int = 1024
    seed: int = 42
    eval_seed: int = 42
    pretrained_checkpoint_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    wandb_logging: bool = True
    use_bf16: bool = False
    # data
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_num_histories: int = 1
    upload_to_hf_repo: Optional[str] = None

    def __post_init__(self):
        assert 1 <= self.dataset_num_histories <= self.num_envs
        if self.dataset_path is not None and self.dataset_name is None:
            self.dataset_name = f"{self.benchmark_id}-{self.ruleset_id}.hdf5"

        if self.checkpoint_path is not None:
            self.checkpoint_path = f"{self.checkpoint_path}-{self.env_id}-seed{self.seed}-{str(uuid.uuid4())}"

        self.num_updates = self.total_timesteps // (self.num_steps * self.num_envs)
        self.name = f"{self.name}-{self.env_id}-{self.benchmark_id}-{self.ruleset_id}"


def make_states(config: TrainConfig):
    print(f"Num updates: {config.num_updates}")

    # for learning rate scheduling
    def linear_schedule(count):
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac

    # setup state
    rng = jax.random.PRNGKey(config.seed)
    # setup environment
    env, env_params = xminigrid.make(config.env_id)
    env = GymAutoResetWrapper(env)

    benchmark = xminigrid.load_benchmark(config.benchmark_id)
    if config.pretrain_multitask:
        print(
            "Multi-task pre-training is enabled. New ruleset ids will be sampled for training. "
            "Choosen single config.ruleset_id will be used for evaluation only."
        )
        rng, _rng = jax.random.split(rng)
        ruleset_ids = jax.random.choice(_rng, benchmark.num_rulesets(), shape=(config.num_envs,), replace=False)
        env_params = env_params.replace(ruleset=jax.vmap(benchmark.get_ruleset)(ruleset_ids))
    else:
        env_params = env_params.replace(ruleset=benchmark.get_ruleset(config.ruleset_id))

    rng, _rng = jax.random.split(rng)
    network = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        obs_emb_dim=config.obs_emb_dim,
        action_emb_dim=config.action_emb_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        context_emb_dim=config.context_emb_dim,
        context_hidden_dim=config.context_hidden_dim,
        context_dropout=config.context_dropout,
        dtype=jnp.bfloat16 if config.use_bf16 else None,
    )

    # [batch_size, seq_len, ...]
    init_obs = {
        "observation": jnp.zeros((1, 1, *env.observation_shape(env_params))),
        "prev_action": jnp.zeros((1, 1), dtype=jnp.int32),
        "goal_encoding": benchmark.get_ruleset(ruleset_id=0).goal[None, None],
        "rule_encoding": benchmark.get_ruleset(ruleset_id=0).rules[None, None],
    }
    init_hstate = network.initialize_carry(batch_size=config.num_envs)

    # init network params
    network_params = network.init(_rng, init_obs, init_hstate[0][None], training=False)
    if config.pretrained_checkpoint_path is not None:
        checkpoint = load_checkpoint(os.path.abspath(config.pretrained_checkpoint_path))
        network_params = checkpoint["params"]
        print("Loaded pre-trained checkpoint as initial params!")
        print(f"Checkpoint Config: {checkpoint['config']}")

    print("Number of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(network_params)))

    learning_rate = linear_schedule if config.decay_lr else config.lr
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate, eps=1e-8),  # eps=1e-5
    )
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    # dataset setup
    if config.dataset_path is not None:
        assert not config.pretrain_multitask
        os.makedirs(config.dataset_path, exist_ok=True)

        transitions_per_env = config.num_steps * config.num_updates
        path = os.path.join(config.dataset_path, config.dataset_name)
        with h5py.File(path, "w") as f:
            f.attrs["env-id"] = config.env_id
            f.attrs["benchmark-id"] = config.benchmark_id
            f.attrs["ruleset-id"] = config.ruleset_id

            obs_shape = env.observation_shape(env_params)[:2]
            f.create_dataset(
                "states", shape=(config.dataset_num_histories, transitions_per_env, *obs_shape), dtype=np.uint8
            )
            f.create_dataset(
                "actions",
                shape=(
                    config.dataset_num_histories,
                    transitions_per_env,
                ),
                dtype=np.uint8,
            )
            f.create_dataset(
                "rewards",
                shape=(
                    config.dataset_num_histories,
                    transitions_per_env,
                ),
                dtype=np.float16,
            )
            f.create_dataset(
                "dones",
                shape=(
                    config.dataset_num_histories,
                    transitions_per_env,
                ),
                dtype=np.bool_,
            )
            # for DPT
            f.create_dataset(
                "expert_actions",
                shape=(
                    config.dataset_num_histories,
                    transitions_per_env,
                ),
                dtype=jnp.uint8,
            )

    return rng, benchmark, env, env_params, init_hstate, train_state


def make_train(
    benchmark: Benchmark,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: jax.Array,
    config: TrainConfig,
):
    def train(rng: jax.Array):
        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                rng, train_state, prev_timestep, prev_action, prev_hstate = runner_state

                # SELECT ACTION
                rng, _rng, _rnd_drop = jax.random.split(rng, num=3)
                dist, value, hstate = train_state.apply_fn(
                    train_state.params,
                    {
                        # [batch_size, seq_len=1, ...]
                        "observation": prev_timestep.observation[:, None],
                        "prev_action": prev_action[:, None],
                        "goal_encoding": prev_timestep.state.goal_encoding[:, None],
                        "rule_encoding": prev_timestep.state.rule_encoding[:, None],
                    },
                    prev_hstate,
                    training=True,
                    rngs={"dropout": _rnd_drop},
                )
                action, log_prob = dist.sample_and_log_prob(seed=_rng)
                # squeeze seq_len where possible
                action, value, log_prob = action.squeeze(1), value.squeeze(1), log_prob.squeeze(1)

                # STEP ENV
                if config.pretrain_multitask:
                    timestep = jax.vmap(env.step, in_axes=(0, 0, 0))(env_params, prev_timestep, action)
                else:
                    timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(env_params, prev_timestep, action)

                transition = Transition(
                    done=timestep.last(),
                    action=action,
                    value=value,
                    reward=timestep.reward,
                    log_prob=log_prob,
                    obs=prev_timestep.observation,
                    prev_action=prev_action,
                    goal_encoding=timestep.state.goal_encoding,
                    rule_encoding=timestep.state.rule_encoding,
                )
                runner_state = (rng, train_state, timestep, action, hstate)
                return runner_state, transition

            initial_hstate = runner_state[-1]
            # transitions: [seq_len, batch_size, ...]
            runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            # saving to the dataset (from the first env)
            def _saving_callback(transitions, update_idx):
                offset = update_idx * config.num_steps

                path = os.path.join(config.dataset_path, config.dataset_name)
                with h5py.File(path, "a") as f:
                    compressed_obs = compress_obs(transitions.obs)
                    f["states"][:, offset : offset + config.num_steps] = np.asarray(compressed_obs)
                    f["actions"][:, offset : offset + config.num_steps] = np.asarray(transitions.action)
                    f["rewards"][:, offset : offset + config.num_steps] = np.asarray(transitions.reward)
                    f["dones"][:, offset : offset + config.num_steps] = np.asarray(transitions.done)

            if config.dataset_path is not None:
                jax.experimental.io_callback(
                    _saving_callback,
                    None,
                    jtu.tree_map(lambda t: t.swapaxes(0, 1)[: config.dataset_num_histories], transitions),
                    update_idx,
                    ordered=True,
                )

            # CALCULATE ADVANTAGE
            rng, train_state, timestep, prev_action, hstate = runner_state
            rng, _rng_drop = jax.random.split(rng)
            # calculate value of the last step for bootstrapping
            _, last_val, _ = train_state.apply_fn(
                train_state.params,
                {
                    "observation": timestep.observation[:, None],
                    "prev_action": prev_action[:, None],
                    "goal_encoding": timestep.state.goal_encoding[:, None],
                    "rule_encoding": timestep.state.rule_encoding[:, None],
                },
                hstate,
                training=True,
                rngs={"dropout": _rng_drop},
            )
            advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    rng, init_hstate, transitions, advantages, targets = batch_info
                    new_train_state, update_info = ppo_update_networks(
                        rng=rng,
                        train_state=train_state,
                        transitions=transitions,
                        init_hstate=init_hstate.squeeze(1),
                        advantages=advantages,
                        targets=targets,
                        clip_eps=config.clip_eps,
                        vf_coef=config.vf_coef,
                        ent_coef=config.ent_coef,
                    )
                    return new_train_state, update_info

                rng, train_state, init_hstate, transitions, advantages, targets = update_state

                # MINIBATCHES PREPARATION
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config.num_envs)
                # [seq_len, batch_size, ...]
                batch = (init_hstate, transitions, advantages, targets)
                # [batch_size, seq_len, ...], as our model assumes
                batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                # [num_minibatches, minibatch_size, ...]
                minibatches = jtu.tree_map(
                    lambda x: jnp.reshape(x, (config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
                )
                _update_rngs = jax.random.split(_rng, num=config.num_minibatches)
                minibatches = (_update_rngs,) + minibatches

                train_state, update_info = jax.lax.scan(_update_minbatch, train_state, minibatches)

                update_state = (rng, train_state, init_hstate, transitions, advantages, targets)
                return update_state, update_info

            # [seq_len, batch_size, num_layers, hidden_dim]
            init_hstate = initial_hstate[None, :]
            update_state = (rng, train_state, init_hstate, transitions, advantages, targets)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)

            # averaging over minibatches then over epochs
            loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

            rng, train_state = update_state[:2]

            # EVALUATE AGENT
            eval_hstate = initial_hstate[0][None]
            eval_rng = jax.random.split(jax.random.PRNGKey(config.eval_seed), num=config.eval_episodes)

            eval_env_params = env_params.replace(ruleset=benchmark.get_ruleset(config.ruleset_id))
            # vmap only on rngs
            eval_stats = jax.vmap(rollout, in_axes=(0, None, None, None, None, None, None))(
                eval_rng,
                env,
                eval_env_params,
                train_state,
                eval_hstate,
                False,
                1,
            )
            loss_info.update(
                {
                    "eval/returns": eval_stats.reward.mean(0),
                    "eval/lengths": eval_stats.length.mean(0),
                    "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
                }
            )

            # evaluation with dropout enabled to get the training returns estimate
            if config.context_dropout > 0.0:
                eval_stats_dropout = jax.vmap(rollout, in_axes=(0, None, None, None, None, None, None))(
                    eval_rng,
                    env,
                    eval_env_params,
                    train_state,
                    eval_hstate,
                    True,
                    1,
                )
                loss_info.update(
                    {
                        "eval_dropout/returns": eval_stats_dropout.reward.mean(0),
                        "eval_dropout/lengths": eval_stats_dropout.length.mean(0),
                    }
                )

            # evaluation on multiple tasks
            if config.pretrain_multitask:
                eval_ruleset_rng = jax.random.PRNGKey(config.eval_seed)
                ruleset_ids = jax.random.choice(
                    eval_ruleset_rng, benchmark.num_rulesets(), shape=(config.eval_episodes,), replace=False
                )
                eval_env_params = env_params.replace(ruleset=jax.vmap(benchmark.get_ruleset)(ruleset_ids))

                # vmap on rngs env env_params
                eval_stats_multitask = jax.vmap(rollout, in_axes=(0, None, 0, None, None, None, None))(
                    eval_rng,
                    env,
                    eval_env_params,
                    train_state,
                    eval_hstate,
                    False,
                    1,
                )
                loss_info.update(
                    {
                        "eval_multitask/returns": eval_stats_multitask.reward.mean(0),
                        "eval_multitask/lengths": eval_stats_multitask.length.mean(0),
                        "eval_multitask/returns_20p": jnp.percentile(eval_stats_multitask.reward, q=20),
                    }
                )

            if config.wandb_logging:

                def _callback(info, update_idx):
                    info.update({"transitions": int(update_idx) * (config.num_envs * config.num_steps)})
                    wandb.log(info)

                jax.debug.callback(_callback, loss_info, update_idx)

            runner_state = (rng, train_state, timestep, prev_action, hstate)
            return runner_state, loss_info

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)

        if config.pretrain_multitask:
            timestep = jax.vmap(env.reset, in_axes=(0, 0))(env_params, reset_rng)
        else:
            timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)
        prev_action = jnp.zeros(config.num_envs, dtype=jnp.int32)

        runner_state = (rng, train_state, timestep, prev_action, init_hstate)
        runner_state, loss_info = jax.lax.scan(
            _update_step,
            runner_state,
            jnp.arange(config.num_updates),
            config.num_updates,
        )
        return {"runner_state": runner_state, "loss_info": loss_info}

    return train


@pyrallis.wrap()
def train(config: TrainConfig):
    if config.wandb_logging:
        # logging to wandb
        run = wandb.init(
            project=config.project,
            group=config.group,
            name=config.name,
            config=asdict(config),
            save_code=True,
        )

    rng, benchmark, env, env_params, init_hstate, train_state = make_states(config)
    print("Compiling...")
    t = time.time()
    train_fn = make_train(benchmark, env, env_params, train_state, init_hstate, config)
    train_fn = jax.jit(train_fn).lower(rng).compile()
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    print("Training...")
    t = time.time()
    train_info = jax.block_until_ready(train_fn(rng))
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s")

    loss_info = train_info["loss_info"]
    print("Final return: ", float(loss_info["eval/returns"][-1]))

    if config.wandb_logging:
        run.summary["training_time"] = elapsed_time
        run.summary["steps_per_second"] = config.total_timesteps / elapsed_time
        run.summary["final_return"] = float(loss_info["eval/returns"][-1])

    if config.checkpoint_path is not None:
        print(f"Saving checkpoint to: {config.checkpoint_path}")
        checkpoint = {"config": asdict(config), "params": train_info["runner_state"][1].params}
        save_checkpoint(os.path.abspath(config.checkpoint_path), checkpoint)

    if config.dataset_path is not None:
        t = time.time()
        # relabeling expert actions for DPT training
        print("Relabeling actions with the latest policy...")

        final_params = train_info["runner_state"][1].params
        apply_fn = jax.jit(train_info["runner_state"][1].apply_fn, static_argnames=["training"])

        ruleset = benchmark.get_ruleset(ruleset_id=config.ruleset_id)
        goal_encoding = jnp.repeat(ruleset.goal[None], config.dataset_num_histories, axis=0)
        rule_encoding = jnp.repeat(ruleset.rules[None], config.dataset_num_histories, axis=0)
        hstate = init_hstate[: config.dataset_num_histories]
        prev_action = jnp.zeros((config.dataset_num_histories,))

        path = os.path.join(config.dataset_path, config.dataset_name)
        with h5py.File(path, "a") as df:
            for i in range(df["rewards"].shape[1]):
                compressed_obs = jnp.array(df["states"][:, i])
                observation = decompress_obs(compressed_obs)

                rng, _rng_drop = jax.random.split(rng)
                dist, _, hstate = apply_fn(
                    final_params,
                    {
                        "observation": observation[:, None],
                        "prev_action": prev_action[:, None],
                        "goal_encoding": goal_encoding[:, None],
                        "rule_encoding": rule_encoding[:, None],
                    },
                    hstate,
                    training=True,
                    rngs={"dropout": _rng_drop},
                )
                action = dist.mode()
                df["expert_actions"][:, i] = np.asarray(action.squeeze())
                prev_action = jnp.array(df["actions"][:, i])

        elapsed_time = time.time() - t
        print(f"Done in {elapsed_time:.2f}s")

        # compress hdf5 file
        print("Compressing dataset...")
        src = os.path.join(config.dataset_path, config.dataset_name)
        trg = os.path.join(config.dataset_path, f"{config.dataset_name}.gz")
        with open(src, "rb") as f_in:
            with gzip.open(trg, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(src)

        # uploading dataset to the HF repo if needed (and deleting local copy)
        if config.upload_to_hf_repo is not None:
            print("Uploading dataset to the HF...")
            api = HfApi()
            repo_url = api.create_repo(
                repo_id=config.upload_to_hf_repo,
                repo_type="dataset",
                private=True,
                exist_ok=True,
            )
            # Upload the file to the repository
            api.upload_file(
                path_or_fileobj=trg,
                path_in_repo=f"{config.dataset_name}.gz",
                repo_id=config.upload_to_hf_repo,
                repo_type="dataset",
            )
            print(f"Dataset uploaded successfully to {repo_url}. Will delete local copy.")
            os.remove(trg)

    if config.wandb_logging:
        run.finish()


if __name__ == "__main__":
    train()
