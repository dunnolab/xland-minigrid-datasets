import itertools
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import deepspeed
import jax
import numpy as np
import pyrallis
import torch
import xminigrid
from jax import numpy as jnp
from torch.nn import functional as F  # noqa
from xminigrid.core.constants import NUM_ACTIONS
from xminigrid.environment import EnvParams
from xminigrid.wrappers import GymAutoResetWrapper

from src.utils.data import XMiniGridADataset
# from src.model_tuples_cache import
from src.xland_ad import XMiniGridAD


@dataclass
class TrainConfig:
    # wandb params
    project: str = "xminigrid-datasets"
    group: str = "test"
    name: str = "ad-deepspeed-k2d"
    # model params
    embedding_dim: int = 64
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    seq_len: int = 4096
    attention_dropout: float = 0.5
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.3
    normalize_qk: bool = False
    pre_norm: bool = True
    # training params
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    clip_grad: Optional[float] = 1.0
    subsample: int = 1
    update_epochs: int = 1
    num_workers: int = 0
    label_smoothing: float = 0.0
    # evaluation params
    eval_every: int = 25_000
    eval_episodes: int = 200
    train_rulesets: int = 128
    eval_rulesets: int = 128
    # deepspeed
    train_micro_batch_size_per_gpu: int = 128
    adam_w_mode: bool = False  # whether to use Adam or AdamW
    local_rank: int = 0  # deepspeed needs it
    zero_stage: int = 2  # stage of ZeRO
    uuid: Optional[str] = None
    # general params
    learning_histories_path: str = "../data/trivial-taster-small.hdf5"
    checkpoints_path: Optional[str] = None
    train_seed: int = 42
    eval_seed: int = 42
    total_updates: int = 0
    # eval
    cfg_path: str = ""


def configure_ds(config):
    ds_config = {
        "train_micro_batch_size_per_gpu": config.train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": config.zero_stage,
            "stage3_param_persistence_threshold": 1e5,
            "zero_quantized_weights": False,
            "stage3_max_live_parameters": 3e7,
            "stage3_prefetch_bucket_size": 3e7,
            "memory_efficient_linear": False,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": config.learning_rate,
                "betas": list(config.betas),
                "weight_decay": config.weight_decay,
                "adam_w_mode": config.adam_w_mode,
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": config.total_updates,
                "warmup_num_steps": 500,
            },
        },
        # "bf16": {"enabled": True},
        "fp16": {"enabled": True},
        "gradient_clipping": 1.0,
        "steps_per_print": 500,
        "local_rank": config.local_rank,
    }

    return ds_config


@torch.no_grad()
def evaluate_in_context_with_cache(
    step_fn,
    reset_fn,
    env_params: EnvParams,
    reset_rng: jax.Array,
    model,
    ruleset_ids: np.ndarray,
    eval_episodes: int,
    rank: int = 0,
):
    num_envs = len(ruleset_ids)
    kv_cache = model.init_cache(
        batch_size=num_envs, dtype=torch.float16, device=model.device
    )

    num_episodes = np.zeros(num_envs)
    returns = np.zeros(num_envs)
    # num_episodes = jnp.zeros(num_envs)
    # returns = jnp.zeros(num_envs)

    eval_info = defaultdict(list)
    # pbar = tqdm.tqdm(total=eval_episodes)

    # timestep = jax.jit(jax.vmap(env.reset, in_axes=(0, 0)))(env_params, reset_rng)
    with jax.default_device(jax.devices("cuda")[rank]):
        timestep = jax.block_until_ready(reset_fn(env_params, reset_rng))
        prev_action, prev_reward = jnp.zeros(num_envs), jnp.zeros(num_envs)

    for step in itertools.count(start=1):
        # fill last s-a-r tuple
        state = torch.from_dlpack(timestep.observation).long()[:, None]
        prev_action = torch.from_dlpack(prev_action).long()[:, None]
        prev_reward = torch.from_dlpack(prev_reward).float()[:, None]

        # predict next_action
        # [num_envs, seq_len, num_actions] -> [num_envs, num_actions]
        logits, kv_cache = model(
            observations=state,
            prev_actions=prev_action,
            prev_rewards=prev_reward,
            cache=kv_cache,
        )
        logits = logits[:, -1]
        dist = torch.distributions.Categorical(logits=logits)
        # action = dist.sample()
        action = dist.mode
        action_jnp = jnp.from_dlpack(action)

        # query the worlds
        with jax.default_device(jax.devices("cuda")[rank]):
            timestep = jax.block_until_ready(step_fn(env_params, timestep, action_jnp))

        done = np.asarray(timestep.last())
        num_episodes += done.astype(int)
        returns += np.asarray(timestep.reward)

        # relabel for the next step
        prev_action = action_jnp
        prev_reward = timestep.reward

        # log returns if done
        for i, d in enumerate(done):
            if d and num_episodes[i] <= eval_episodes:
                eval_info[ruleset_ids[i]].append(returns[i])
                # reset return for this goal
                returns[i] = 0.0
                # returns = returns.at[i].set(0.0)
                # pbar.update(-pbar.n)
                # pbar.update(min(num_episodes))

        # check that all goals are done
        if jnp.all(num_episodes > eval_episodes):
            break

    return eval_info


@pyrallis.wrap()
def main(config: TrainConfig):
    os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = str(config.local_rank)
    deepspeed.init_distributed()
    savepath = f"./evaluation/"

    if config.local_rank == 0:
        os.makedirs(savepath, exist_ok=True)

    dataset = XMiniGridADataset(
        config.learning_histories_path,
        seq_len=config.seq_len,
    )

    print(f"Dataset Length: {len(dataset)}")

    with jax.default_device(jax.devices("cuda")[config.local_rank]):
        benchmark_id, env_id, train_rulesets = dataset.trajectories_metadata
        key = jax.random.PRNGKey(config.eval_seed)

        benchmark = xminigrid.load_benchmark(benchmark_id)
        all_rulesets = np.array(range(benchmark.num_rulesets()))
        eval_rulesets = np.setdiff1d(all_rulesets, train_rulesets)
        eval_indexes = np.random.randint(
            low=0, high=len(eval_rulesets), size=config.eval_rulesets
        )
        eval_rulesets = eval_rulesets[eval_indexes]

        all_rulesets = eval_rulesets
        num_envs = len(all_rulesets)

        env, env_params = xminigrid.make(env_id)
        env = GymAutoResetWrapper(env)
        rng, reset_rng = jax.random.split(key)

        reset_fn = jax.jit(jax.vmap(env.reset, in_axes=(0, 0)))
        step_fn = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

        # separate on ranks
    per_gpu = config.eval_rulesets // int(os.getenv("WORLD_SIZE", 1))
    idx_rank = np.arange(per_gpu * config.local_rank, per_gpu * (config.local_rank + 1))
    rulesets_per_gpu = eval_rulesets[idx_rank]
    env_params_per_gpu = env_params.replace(
        ruleset=jax.vmap(benchmark.get_ruleset)(rulesets_per_gpu)
    )
    reset_rng_per_gpu = jax.random.split(reset_rng, num_envs)[idx_rank]

    print(idx_rank)

    ds_config = configure_ds(config)

    num_actions = NUM_ACTIONS
    model = XMiniGridAD(
        num_actions=num_actions,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        seq_len=config.seq_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        normalize_qk=config.normalize_qk,
        pre_norm=config.pre_norm,
    )

    deepspeed.comm.barrier()
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    load_path, _ = model_engine.load_checkpoint(
        load_dir=config.checkpoints_path,
        tag="last",
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
        load_module_only=True,
    )

    assert load_path is not None, "Checkpoint loading has failed!"

    model_engine.eval()
    eval_info = evaluate_in_context_with_cache(
        step_fn=step_fn,
        reset_fn=reset_fn,
        env_params=env_params_per_gpu,
        reset_rng=reset_rng_per_gpu,
        model=model_engine,
        ruleset_ids=rulesets_per_gpu,
        eval_episodes=config.eval_episodes,
        rank=config.local_rank,
    )

    with open(os.path.join(savepath, f"eval_info_{config.local_rank}.pkl"), "wb") as f:
        pickle.dump(eval_info, f)


if __name__ == "__main__":
    main()
