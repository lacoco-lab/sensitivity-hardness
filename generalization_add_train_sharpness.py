import wandb
import torch
import numpy as np
import os

from itertools import chain
from tqdm import tqdm, trange

from src.model import TorchModel
from src.measurements import hessian_norm_proxy
from src.training import sample_batch
from src.targets import RandomFunc
from src.utils import RestrictedSampler, wandb_prefix


def update_run(run):
    model = TorchModel(**run.config["model"])
    model.load_state_dict(torch.load(run.config["basic"]["directory"] + "/model.pt"))
    model = model.to("cuda")
    model.eval()

    target = RandomFunc(0, 10, "cuda")
    dir_path = run.config["basic"]["target_dir"]
    filename = os.listdir(dir_path)[run.config["basic"]["target_file_idx"]]
    samples = np.load(os.path.join(dir_path, filename))
    target.samples = torch.IntTensor(samples).to("cuda")
    loss = torch.nn.MSELoss()

    type = run.config["parent_function/type"]
    if type == "random":
        return
    if "parent_function/n_samples" in run.config:
        n_samples = run.config["parent_function/n_samples"]
    else:
        n_samples = 256

    sampler = RestrictedSampler(10, n_samples, int(run.config["parent_function/random_state"]))

    x, y = sampler.samples.cuda(), target(sampler.samples.cuda())
    sharpness = hessian_norm_proxy(
        x, y, model, loss,
        n_repeats=100, random_state=42,
        epsilon=0.02
    )

    run.summary[f"sharpness_train_set"] = sharpness
    run.summary.update()


def main():
    api = wandb.Api()
    wandb.login()

    sweep1 = api.sweep(f"{wandb_prefix()}/{input('Enter the ID of generalization_second_step_1 sweep')}")
    sweep2 = api.sweep(f"{wandb_prefix()}/{input('Enter the ID of generalization_second_step_2 sweep')}")
    runs = list(chain(sweep1.runs, sweep2.runs))

    for i, run in tqdm(enumerate(runs)):
        if run.summary["eval/loss"] != 'NaN' and run.summary["eval/loss"] <= 0.001:
            update_run(run)


if __name__ == "__main__":
    main()
