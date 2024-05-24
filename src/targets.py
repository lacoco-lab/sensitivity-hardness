import torch
import numpy as np
import logging

from typing import Tuple, Optional

from src.utils2 import get_output_loss_acc


class Target:
    def __init__(self, negative_value: int = 0):
        self.negative_value = negative_value

        get_output_loss_acc.negative_value = negative_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def postprocess(self, bool_result: torch.Tensor) -> torch.Tensor:
        if self.negative_value == -1:
            return 2 * bool_result - 1
        elif self.negative_value == 0:
            return bool_result
        else:
            raise ValueError(f"Unknown negative value: {self.negative_value}")


class Parity(Target):
    def __init__(self, negative_value: int = 0):
        super().__init__(negative_value=negative_value)

    # extended = True stands for the case when we have a scratchpad and consider the output on all prefixes
    def __call__(self, x: torch.Tensor, extended=False) -> torch.Tensor:
        if not extended:
            bool_result = torch.sum(x, dim=-1) % 2
        else:
            bool_result = x.cumsum(dim=-1) % 2

        return self.postprocess(bool_result)
    

class Majority(Target):
    def __call__(self, x: torch.Tensor, extended=False) -> torch.Tensor:
        bool_result = (torch.sum(x, dim=-1) > torch.sum((1 - x), dim=-1)).int()
        return self.postprocess(bool_result)


class First(Target):
    def __call__(self, x: torch.Tensor, extended=False) -> torch.Tensor:
        bool_result = x[:, 0]
        return self.postprocess(bool_result)


class Mean(Target):
    def __call__(self, x: torch.Tensor, extended=False) -> torch.Tensor:
        bool_result = x.float().mean(dim=-1)
        return self.postprocess(bool_result)
        

# random function on bistrings of a given length, initialized with a seed. required for a generalization experiment
class RandomFunc(Target):
    def __init__(self, seed: int, length: int, device: str):
        super().__init__()
        
        rng = np.random.default_rng(seed=seed)
        self.samples = torch.IntTensor(rng.choice(2, size=2 ** length)).to(device)
        self.mask = 2 ** torch.arange(length - 1, -1, -1).to(device)

        logging.info(f"Initialized random function with seed {seed} and length {length}, first 10 samples: {self.samples[:10]}")

    def __call__(self, x: torch.Tensor, extended=False) -> torch.Tensor:
        x_transformed = (x * self.mask).sum(dim=-1).long()
        bool_result = self.samples[x_transformed]
        return self.postprocess(bool_result)
