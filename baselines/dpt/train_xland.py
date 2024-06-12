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
import wandb
from torch.nn import functional as F  # noqa
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import xminigrid
import xminigrid.types
import tqdm

from src.model_dpt import XMiniGridDPT
from src.utils.data import XMiniGridDPTDataset
from src.utils.misc import set_seed, Timeit

import jax
from jax import numpy as jnp
from xminigrid.core.constants import NUM_ACTIONS
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
    checkpoints_path: Optional[str] = None
    train_seed: int = 42
    data_seed: int = 0
    eval_seed: int = 42

    def __post_init__(self):
        assert (
            self.hidden_dim / self.num_heads
        ) % 8 == 0, "head dim should be multiple of 8 for flash attn"

        # self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
        self.name = f"{self.name}-{self.seq_len}-{self.train_seed}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


@torch.no_grad()
def multi_episode_in_context(step_fn,
                             reset_fn,
                             env_params: EnvParams,
                            reset_rng: jax.Array,
                            model,
                            ruleset_ids: np.ndarray,
                            eval_episodes: int) -> Dict[int, List[float]]:
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
    timestep = jax.block_until_ready(reset_fn(env_params, reset_rng))
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
def train(config: TrainConfig):
    set_seed(config.train_seed)
    dict_config = asdict(config)
    dict_config["mlc_job"] = os.getenv("PLATFORM_JOB_NAME")
    # config.local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
    # config.local_rank = int(os.environ['local_rank'])
    # print(f"MY RANK IS {config.local_rank}")
    deepspeed.init_distributed()

    dataset = XMiniGridDPTDataset(
        config.learning_histories_path,
        seq_len=config.seq_len,
        samples_per_task=config.samples_per_task,
    )

    print(f"Dataset Length: {len(dataset)}")

    train_sampler = DistributedSampler(
        dataset,
        num_replicas=int(os.getenv("WORLD_SIZE")),
        rank=int(config.local_rank if config.local_rank != -1 else 0),
        shuffle=True,  # this is important, deepspeed does not allow to control it
        seed=0,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.train_micro_batch_size_per_gpu,
        sampler=train_sampler,
        num_workers=config.num_workers,
    )

    config.total_updates = len(dataloader) * config.update_epochs

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

    # save config to the checkpoint
    if config.local_rank == 0:
        if config.checkpoints_path is not None:
            print(f"Checkpoints path: {config.checkpoints_path}")
            os.makedirs(config.checkpoints_path, exist_ok=True)
            with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
                pyrallis.dump(config, f)

    deepspeed.comm.barrier()
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    model_engine.train()

    print("##### MY DEVICE", model_engine.device)

    if config.local_rank == 0:
        wandb.init(
            project=config.project,
            group=config.group,
            entity="dunnolab",
            name=config.name,
            config=dict_config,
        )

    global_step = 0
    for epoch in range(config.update_epochs):
        for batch in dataloader:
            query_states, states, actions, next_states, rewards, target_actions = [b.to(model_engine.device) for b in batch]

            query_states = query_states.to(torch.long)
            states = states.to(torch.long)
            actions = actions.to(torch.long)
            next_states = next_states.to(torch.long)
            rewards = rewards.to(torch.float16)
            target_actions = target_actions.to(torch.long)

            target_actions = target_actions.squeeze(-1)
            if config.with_prior:
                target_actions = F.one_hot(target_actions, num_classes=num_actions).unsqueeze(1).repeat(1, seq_len + 1, 1).float()
            else:
                target_actions = F.one_hot(target_actions, num_classes=num_actions).unsqueeze(1).repeat(1, seq_len, 1).float()

            with Timeit() as timer:
                predicted_actions = model_engine(
                    query_states,
                    states,
                    actions,
                    next_states,
                    rewards
                )
                if not config.with_prior:
                    predicted_actions = predicted_actions[:, 1:, :]
                
                loss = F.cross_entropy(
                    input=predicted_actions.flatten(0, 1).to(torch.float32),
                    target=target_actions.flatten(0, 1),
                    label_smoothing=config.label_smoothing,
                )

                model_engine.backward(loss)
                model_engine.step()

            with torch.no_grad():
                a = torch.argmax(predicted_actions.flatten(0, 1), dim=-1)
                t = torch.argmax(target_actions.flatten(0, 1), dim=-1)
                accuracy = torch.sum(a == t) / a.shape[0]

                if config.local_rank == 0:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "accuracy": accuracy,
                            "lr": model_engine.get_lr()[0],
                            "udpate-time": timer.elapsed_time_gpu,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

            global_step += 1

    if config.checkpoints_path is not None:
        model_engine.save_checkpoint(config.checkpoints_path, tag="last")


if __name__ == "__main__":
    train()