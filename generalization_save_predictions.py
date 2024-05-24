import torch
import wandb
import numpy as np

from os import mkdir, listdir
from tqdm.notebook import tqdm
from collections import defaultdict

from src.utils import RestrictedSampler
from src.targets import RandomFunc
from src.utils import get_parities_for_sensitivity, fast_average_sensitivity, wandb_prefix, set_random_seed


def sweep_random_and_256(api):
    SWEEP_1_ID = input("Enter ID of generalization_first_step_1 sweep: ")
    LENGTH = 10
    N_SAMPLES = 256

    mkdir(f"saved_functions/256_random")

    set_random_seed(0)

    sweep = api.sweep(f"{wandb_prefix()}/{SWEEP_1_ID}")
    parities, degrees = get_parities_for_sensitivity(LENGTH)

    learned_s = []
    random_s = []

    for run in tqdm(sweep.runs):
        if run.state != "finished":
            continue

        f = RandomFunc(run.config["basic"]["target_seed"], LENGTH, "cpu")
        sampler = RestrictedSampler(LENGTH, N_SAMPLES, run.config["basic"]["random_state"])

        saved = np.load(run.config["basic"]["directory"] + "/predictions.npy")

        mask = 2 ** torch.arange(LENGTH - 1, -1, -1)
        xs = (sampler.samples * mask).sum(-1).long()

        f_results = f.samples[xs]

        saved_results = torch.IntTensor(saved[xs])

        if not (f_results == saved_results).all().item():
            print("Failed", run.id)
            print(run.config["basic"]["random_state"], run.config["basic"]["target_seed"], run.summary["train/accuracy"], (f_results == saved_results).float().mean().item())
            continue

        learned_s.append(fast_average_sensitivity(torch.IntTensor(saved), parities, degrees, 0).item())

        other_function = RandomFunc(np.random.randint(0, 10000), LENGTH, "cpu") 
        other_function.samples[xs] = f_results
        random_s.append(fast_average_sensitivity(other_function.samples, parities, degrees, 0).item())

        np.save(f"saved_functions/{SWEEP}/{run.id}_{run.config['basic']['random_state']}_{run.config['basic']['target_seed']}_learned.npy", saved)
        np.save(f"saved_functions/{SWEEP}/{run.id}_{run.config['basic']['random_state']}_{run.config['basic']['target_seed']}_random.npy", other_function.samples.numpy())

        print(f"Created {len(listdir(f'saved_functions/256_random'))} files")


def sweep_128_and_512(api):
    SWEEP_2_ID = "Enter ID of generalization_first_step_2 sweep: "
    LENGTH = 10

    mkdir(f"saved_functions/128_512")

    set_random_seed(0)

    sweep = api.sweep(f"{wandb_prefix()}/{SWEEP_2_ID}")
    parities, degrees = get_parities_for_sensitivity(LENGTH)

    sensitivity = defaultdict(list)

    for run in tqdm(sweep.runs):
        f = RandomFunc(run.config["basic"]["target_seed"], LENGTH, "cpu")
        n_samples = run.config["basic"]["restricted_size"]
        sampler = RestrictedSampler(LENGTH, n_samples, run.config["basic"]["random_state"])

        saved = np.load(run.config["basic"]["directory"] + "/predictions.npy")

        mask = 2 ** torch.arange(LENGTH - 1, -1, -1)
        xs = (sampler.samples * mask).sum(-1).long()

        f_results = f.samples[xs]

        saved_results = torch.IntTensor(saved[xs])

        if not (f_results == saved_results).all().item():
            print("Failed", run.id)
            print(run.config["basic"]["random_state"], run.config["basic"]["target_seed"], run.summary["train/accuracy"], (f_results == saved_results).float().mean().item())
            continue

        s = fast_average_sensitivity(torch.IntTensor(saved), parities, degrees, 0).item()
        sensitivity[n_samples].append(s)

        filename = f"{run.id}_{run.config['basic']['random_state']}_{run.config['basic']['target_seed']}_samples-{n_samples}_learned.npy"
        
        np.save(f"saved_functions/128_512/{filename}", saved)

    print(f"Created {len(listdir(f'saved_functions/128_512'))} files")


def main():
    wandb.login()
    api = wandb.Api()

    sweep_random_and_256(api)
    sweep_128_and_512(api)
