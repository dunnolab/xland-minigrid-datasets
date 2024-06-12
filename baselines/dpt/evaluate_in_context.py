import pickle
from typing import Dict, List
import itertools
import os
# import uuid
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import deepspeed
import numpy as np
import pyrallis
import torch
from torch.nn import functional as F  # noqa
import xminigrid
import xminigrid.types

from src.model_dpt import XMiniGridDPT
from src.utils.data import XMiniGridDPTDataset
from src.utils.misc import set_seed, Timeit

import jax
from jax import numpy as jnp
from xminigrid.core.constants import NUM_ACTIONS
from xminigrid.benchmarks import Benchmark
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.environment import EnvParams


@dataclass
class TrainConfig:
    # wandb params
    project: str = "xminigrid-datasets"
    group: str = "trivial-dpt"
    name: str = "dpt-trivial"
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
    samples_per_task: int = 10_000
    update_epochs: int = 1
    num_workers: int = 0
    label_smoothing: float = 0.0
    with_prior: bool = True
    # evaluation params
    eval_every: int = 25_000
    eval_episodes: int = 200
    train_rulesets: int = 128
    eval_rulesets: int = 128
    # deepspeed
    train_micro_batch_size_per_gpu: int = 128
    adam_w_mode: bool = False  # whether to use Adam or AdamW
    local_rank: int = -1  # deepspeed needs it
    zero_stage: int = 2  # stage of ZeRO
    # general params
    learning_histories_path: str = "../data/trivial-taster-small.hdf5"
    train_seed: int = 42
    data_seed: int = 0
    eval_seed: int = 42
    total_updates: int = 0
    # path to checkpoints folder
    checkpoints_path: Optional[str] = None


@torch.no_grad()
def multi_episode_in_context(step_fn,
                             reset_fn,
                             env_params: EnvParams,
                            reset_rng: jax.Array,
                            model,
                            ruleset_ids: np.ndarray,
                            eval_episodes: int,
                            rank: int = 0) -> Dict[int, List[float]]:
    # during evaluation DPT takes as a buffer only a fixed number of episodes
    # so we stick to this rule and create the context with a number of samples that is
    # less than during the training to avoid OOD & OOM issues
    episode_max_steps = env_params.max_steps
    # assert model.seq_len % episode_max_steps == 0
    context_rollouts = model.seq_len // episode_max_steps
    num_envs = len(ruleset_ids)

    context_obs = torch.zeros(
        (num_envs, context_rollouts, episode_max_steps, 5, 5, 2), dtype=torch.long, device=model.device
    )
    context_actions = torch.zeros(
        (num_envs, context_rollouts, episode_max_steps), dtype=torch.long, device=model.device
    )
    context_next_obs = torch.zeros(
        (num_envs, context_rollouts, episode_max_steps, 5, 5, 2), dtype=torch.long, device=model.device
    )
    context_rewards = torch.zeros(
        (num_envs, context_rollouts, episode_max_steps), dtype=torch.float16, device=model.device
    )
    num_episodes = np.zeros(num_envs)
    returns = np.zeros(num_envs)
    eval_info = defaultdict(list)
    # pbar = tqdm(total=num_envs * eval_episodes, position=1)

    obs = torch.zeros(
        (num_envs, episode_max_steps, 5, 5, 2), dtype=torch.long, device=model.device
    )
    actions = torch.zeros(
        (num_envs, episode_max_steps), dtype=torch.long, device=model.device
    )
    next_obs = torch.zeros(
        (num_envs, episode_max_steps, 5, 5, 2), dtype=torch.long, device=model.device
    )
    rewards = torch.zeros(
        (num_envs, episode_max_steps), dtype=torch.float16, device=model.device
    )
    
    # timestep = jax.block_until_ready(reset_fn(env_params, reset_rng))
    # timestep = jax.block_until_ready(reset_fn(env_params, reset_rng))
    with jax.default_device(jax.devices("cuda")[rank]):
        timestep = jax.block_until_ready(reset_fn(env_params, reset_rng))
        prev_action, prev_reward = jnp.zeros(num_envs), jnp.zeros(num_envs)
    
    for step in itertools.count(start=1):
        # get query_obs
        query_obs = torch.from_dlpack(timestep.observation).long()

        obs = obs.roll(-1, dims=(1,))
        actions = actions.roll(-1, dims=1)
        next_obs = next_obs.roll(-1, dims=(1,))
        rewards = rewards.roll(-1, dims=1)

        with torch.cuda.amp.autocast():
            logits = model(query_obs,
                           context_obs.reshape(num_envs, -1, 5, 5, 2)[:, -step:],
                           context_actions.reshape(num_envs, -1)[:, -step:],
                           context_next_obs.reshape(num_envs, -1, 5, 5, 2)[:, -step:],
                           context_rewards.reshape(num_envs, -1)[:, -step:])
        dist = torch.distributions.Categorical(logits=logits)
        # action = dist.sample()
        action = dist.mode
        action_jnp = jnp.from_dlpack(action)

        # set current observation
        obs[:, -1] = query_obs

        # query the worlds
        # timestep = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))(env_params, timestep, action_jnp)
        # timestep = jax.block_until_ready(step_fn(env_params, timestep, action_jnp))
        with jax.default_device(jax.devices("cuda")[rank]):
            timestep = jax.block_until_ready(step_fn(env_params, timestep, action_jnp))

        actions[:, -1] = action
        next_obs[:, -1] = torch.from_dlpack(timestep.observation).long()
        rewards[:, -1] = torch.from_dlpack(timestep.reward).to(torch.float16)#.float()

        done = np.asarray(timestep.last())
        num_episodes += done.astype(int)
        returns += np.asarray(timestep.reward)

        # log returns if done and update contextual buffers
        for i, d in enumerate(done):
            # update buffers
            if d:
                context_obs = context_obs.roll(-1, dims=(1, ))
                context_actions = context_actions.roll(-1, dims=(1, ))
                context_next_obs = context_next_obs.roll(-1, dims=(1, ))
                context_rewards = context_rewards.roll(-1, dims=(1, ))

                context_obs[i, -1, :] = obs[i, :]
                context_actions[i, -1, :] = actions[i, :]
                context_next_obs[i, -1, :] = next_obs[i, :]
                context_rewards[i, -1, :] = rewards[i, :]

            if d and num_episodes[i] <= eval_episodes:
                eval_info[ruleset_ids[i]].append(returns[i])
                # reset return for this goal
                returns[i] = 0.0
                # returns = returns.at[i].set(0.0)

                # pbar.update(1)

        # check that all goals are done
        if jnp.all(num_episodes > eval_episodes):
            break
    
    return eval_info


def split_info_debug(eval_info, train_tasks, test_tasks):
    eval_info_train = defaultdict(list)
    eval_info_test = defaultdict(list)

    if not isinstance(train_tasks, list):
        train_tasks = train_tasks.tolist()
    if not isinstance(test_tasks, list):
        test_tasks = test_tasks.tolist()

    for i, (k, v) in enumerate(eval_info.items()):
        current_task = k

        if not isinstance(current_task, int):
            current_task = int(current_task)

        if not isinstance(v, list):
            v = v.tolist()

        if current_task in train_tasks:
            eval_info_train[k] = v
        if current_task in test_tasks:
            eval_info_test[k] = v

    return eval_info_train, eval_info_test


def configure_ds(config: TrainConfig):
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


@pyrallis.wrap()
def main(config: TrainConfig):
    set_seed(config.train_seed)
    dict_config = asdict(config)
    dict_config["mlc_job"] = os.getenv("PLATFORM_JOB_NAME")
    # config.local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
    # config.local_rank = int(os.environ['local_rank'])
    # print(f"MY RANK IS {config.local_rank}")
    deepspeed.init_distributed()
    savepath = f"./evaluation/"

    if config.local_rank == 0:
        os.makedirs(savepath, exist_ok=True)

    dataset = XMiniGridDPTDataset(
        config.learning_histories_path,
        seq_len=config.seq_len,
        samples_per_task=config.samples_per_task,
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

    del dataset

    print(idx_rank)

    num_actions = NUM_ACTIONS
    seq_len = config.seq_len

    # deepspeed setup
    # deepspeed.init_distributed(dist_backend='gloo')
    ds_config = configure_ds(config)

    # init ZeRO
    with (
        deepspeed.zero.Init(dtype=torch.float16)
        if (config.zero_stage == 3)
        else nullcontext()
    ):
        model = XMiniGridDPT(
            num_actions=num_actions,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            seq_len=seq_len,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            attention_dropout=config.attention_dropout,
            residual_dropout=config.residual_dropout,
            embedding_dropout=config.embedding_dropout,
            normalize_qk=config.normalize_qk,
            pre_norm=config.pre_norm,
        )
    

    print("MODEL INIT COMPLETE")

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

    print("##### MY DEVICE", model_engine.device)


    # evaluation itself
        
    eval_info = multi_episode_in_context(
            step_fn=step_fn,
            reset_fn=reset_fn,
            env_params=env_params_per_gpu,
            reset_rng=reset_rng_per_gpu,
            model=model_engine,
            ruleset_ids=rulesets_per_gpu,
            eval_episodes=config.eval_episodes,
            rank=config.local_rank
    )

    with open(os.path.join(savepath, f"eval_info_{config.local_rank}.pkl"), "wb") as f:
        pickle.dump(eval_info, f)


if __name__ == "__main__":
    main()