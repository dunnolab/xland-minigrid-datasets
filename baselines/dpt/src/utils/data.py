from typing import List, Dict, Any, Tuple
import random

import numpy as np
from torch.utils.data import Dataset
import h5py

import xminigrid
from xminigrid.core.constants import NUM_TILES, NUM_COLORS


class XMiniGridDPTDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 seq_len: int = 4096,
                 samples_per_task: int = 10_000) -> None:
        super().__init__()

        self.data_file = None
        self.seq_len = seq_len
        self.benchmark_id = None
        self.env_id = None
        self.max_len = None

        with h5py.File(data_path, "r") as df:
            self.num_tasks = len(df.keys())
            self.data_path = data_path
            self.indexes = []
            self.ruleset_ids = []
            
            self.benchmark_id = df["0"].attrs["benchmark-id"]
            self.env_id = df["0"].attrs["env-id"]
            self.max_len = df["0"]["states"][0].shape[0] - seq_len
            learning_history_len = self.max_len + seq_len
            num_learning_histories = df["0"]["states"].shape[0]

            for task_id in df.keys():
                
                self.ruleset_ids.append(df[task_id].attrs["ruleset-id"])

                for sample in range(samples_per_task):
                    query_learning_history = random.randint(0, num_learning_histories - 1)
                    # query_max_len = df[task_id]["states"][query_learning_history].shape[0] - seq_len
                    # query_idx = random.randint(0, query_max_len - 1)
                    query_idx = random.randint(0, self.max_len - 1)

                    context_learning_history = random.randint(0, num_learning_histories - 1)
                    context_indexes = np.random.randint(0, learning_history_len - 2, size=seq_len)

                    self.indexes.append((task_id, query_learning_history, query_idx, context_learning_history, context_indexes))
    
    def open_hdf5(self):
        self.data_file = h5py.File(self.data_path, "r")
    
    @staticmethod
    def get_episode_max_steps(env_id: str) -> int:
        env, env_params = xminigrid.make(env_id)
        return env_params.max_steps
    
    @property
    def trajectories_metadata(self) -> Tuple[str, str, List[np.int64]]:
        return self.benchmark_id, self.env_id, self.ruleset_ids
    
    @staticmethod
    def decompress_obs(obs: np.ndarray) -> np.ndarray:
        return np.stack(np.divmod(obs, NUM_COLORS), axis=-1)
    
    def __len__(self) -> int:
        return len(self.indexes)
    
    def __getitem__(self, index):
        if self.data_file is None:
            self.open_hdf5()
        
        task_id, query_learning_history, query_idx, context_learning_history, context_indexes = self.indexes[index]

        query_states = self.decompress_obs(self.data_file[task_id]["states"][query_learning_history][query_idx])
        states = self.decompress_obs(self.data_file[task_id]["states"][context_learning_history][context_indexes])
        next_states = self.decompress_obs(self.data_file[task_id]["states"][context_learning_history][context_indexes + 1])
        actions = self.data_file[task_id]["actions"][context_learning_history][context_indexes]
        target_actions = self.data_file[task_id]["expert_actions"][query_learning_history][query_idx]
        rewards = self.data_file[task_id]["rewards"][context_learning_history][context_indexes]

        return query_states, states, actions, next_states, rewards, target_actions
