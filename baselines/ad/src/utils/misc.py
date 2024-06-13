import itertools
import os
import time
import torch
import random
import numpy as np


class Timeit:
    def __enter__(self):
        if torch.cuda.is_available():
            self.start_gpu = torch.cuda.Event(enable_timing=True)
            self.end_gpu = torch.cuda.Event(enable_timing=True)
            self.start_gpu.record()
        self.start_cpu = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if torch.cuda.is_available():
            self.end_gpu.record()
            torch.cuda.synchronize()
            self.elapsed_time_gpu = self.start_gpu.elapsed_time(self.end_gpu) / 1000
        else:
            self.elapsed_time_gpu = -1.0
        self.elapsed_time_cpu = time.time() - self.start_cpu


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def filter_params(model, no_decay_name_list, weight_decay, learning_rate, head_width):
    optimizer_grouped_parameters = []
    final_optimizer_settings = {}

    for n, p in model.named_parameters():
        group_parameters = {}
        if p.requires_grad:
            if any(ndnl in n for ndnl in no_decay_name_list):
                group_parameters["weight_decay"] = 0.0
            else:
                group_parameters["weight_decay"] = weight_decay

            # Define learning rate for specific types of params

            is_embed = "embed" in n
            if "embed" in n or any(ndnl in n for ndnl in no_decay_name_list):
                group_parameters["lr"] = learning_rate * (3.3 if is_embed else 1.0)
            else:
                group_parameters["lr"] = learning_rate * (1 / head_width)

            group_parameters["params"] = [p]
            final_optimizer_settings[n] = {
                "lr": group_parameters["lr"],
                "wd": group_parameters["weight_decay"],
            }
            optimizer_grouped_parameters.append(group_parameters)

    return optimizer_grouped_parameters
