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
    torch.backends.cudnn.benchmark = True
