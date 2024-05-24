import torch
import numpy as np
import random
import logging
import os
import itertools

from src.model import TorchModel, TorchModelWithScratchpad
from src.targets import Target


def wandb_prefix():
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")

    if entity is None or project is None:
        raise ValueError("WANDB_ENTITY and WANDB_PROJECT must be set")

    return f"{entity}/{project}"


def set_random_seed(seed, verbose=True):
    if verbose:
        logging.info(f"Setting random seed to {seed}")

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_loss(criterion, output, y):
    return criterion(output, y.float())


# this class is only used for the generalization experiment.
# it generates a set of samples from {0, 1}^n and returns random batches from this set
class RestrictedSampler:
    def __init__(self, length: int, max_samples: int, seed: int = 42):
        self.length = length

        rng = np.random.default_rng(seed=seed)
        samples = torch.IntTensor(rng.choice(2 ** length, size=max_samples, replace=False))
        mask = 2 ** torch.arange(self.length - 1, -1, -1)
        self.samples = samples.unsqueeze(-1).bitwise_and(mask).ne(0).int()

        logging.info(f"Created restricted sampler: random seed {seed}, length {length}, max_samples {max_samples}, first 10 samples: {self.samples[:10]}")
    
    def __call__(self, batch_size: int) -> torch.Tensor:
        idxs = np.random.choice(len(self.samples), size=batch_size, replace=True)
        return self.samples[idxs]


def get_parities_for_sensitivity(length: int):
    N = length
    indices = list(range(2**N))
    bitstrings = []
    # Generate all bitstrings of length N
    for i in range(2**N):
        bitstrings.append([2*int(q)-1 for q in bin(i)[2:].zfill(N)])

    # generate all subsets of {1,...,N}
    subsets = []
    for i in range(0,N+1):
        subsets += list(itertools.combinations(range(N), i))
    # print(subsets)

    # all parities
    parities = []
    for s in subsets:
        parities.append(torch.FloatTensor([(-1)**sum([1 if bitstrings[i][j] > 0 else 0 for j in s]) for i in range(2**N)]))

    degrees = torch.FloatTensor([len(s) for s in subsets])
    parities = torch.stack(parities, dim=0)
    return parities, degrees


# computation of the average sensitivity of a function
def fast_average_sensitivity(guess: torch.Tensor, parities: torch.Tensor, degrees: torch.Tensor, negative_value: int):
    if negative_value == 0:
        guess = guess * 2 - 1
    return ((parities * guess.unsqueeze(0)).mean(dim=1).pow(2) * degrees).sum()
