import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from tqdm import tqdm

from src.utils import get_parities_for_sensitivity, fast_average_sensitivity, wandb_prefix
from src.plotting.style import set_style


LENGTH = 10


def plot_generalization_experiment(api: wandb.Api, save_path: str, multiply_by_4: bool):
    GENERALIZATION_SWEEP_256 = input("Enter the ID of generalization_second_step_1 sweep: ")
    GENERALIZATION_SWEEP_ALL = input("Enter the ID of generalization_second_step_2 sweep: ")

    set_style()
    
    parities, degrees = get_parities_for_sensitivity(LENGTH)

    os.makedirs(os.path.join(save_path, "generalization"), exist_ok=True)

    sweep_256 = api.sweep(f"{wandb_prefix()}/{GENERALIZATION_SWEEP_256}")

    data_256 = pd.DataFrame(columns=["type", "sensitivity", "sharpness", "train_sharpness"], index=range(len(sweep_256.runs)))

    for i, run in tqdm(enumerate(sweep_256.runs)):
        if run.summary["eval/loss"] >= 0.001:
            print(f"Failed: {run.name, run.id}")
            continue

        fname = os.listdir(run.config["basic"]["target_dir"])[run.config["basic"]["target_file_idx"]]
        samples = np.load(os.path.join(run.config["basic"]["target_dir"], fname))
        sensitivity = fast_average_sensitivity(torch.tensor(samples), parities, degrees, 0).item()
        sharpness = run.summary["measurements/hessian_norm_proxy"]
        type = run.config["parent_function/type"]

        if type == "learned":
            train_sharpness = run.summary["sharpness_train_set"]
        else:
            train_sharpness = sharpness

        data_256.loc[i] = [type, sensitivity, sharpness, train_sharpness]

    sweep_all = api.sweep(f"{wandb_prefix()}/{GENERALIZATION_SWEEP_ALL}")

    data_other = pd.DataFrame(columns=["type", "sensitivity", "sharpness", "train_sharpness", "n_samples"], index=range(len(sweep_all.runs)))

    for i, run in tqdm(enumerate(sweep_all.runs)):
        if run.state != "finished":
            print(f"Running: {run.name, run.id}")
            continue

        if run.summary["eval/loss"] >= 0.001:
            print(f"Failed: {run.name, run.id}")
            continue

        fname = os.listdir(run.config["basic"]["target_dir"])[run.config["basic"]["target_file_idx"]]
        samples = np.load(os.path.join(run.config["basic"]["target_dir"], fname))
        sensitivity = fast_average_sensitivity(torch.tensor(samples), parities, degrees, 0).item()
        sharpness = run.summary["measurements/hessian_norm_proxy"]
        type = run.config["parent_function/type"]
        n_samples = run.config["parent_function/n_samples"]
    
        train_sharpness = run.summary["sharpness_train_set"]

        data_other.loc[i] = [type, sensitivity, sharpness, train_sharpness, n_samples]

    data_256["n_samples"] = 256
    data = pd.concat([data_256, data_other])
    data.dropna(inplace=True)

    if multiply_by_4:
        data["sharpness"] *= 4

    data_good = data[data["n_samples"] != 1024]
    data_good["style"] = data_good["n_samples"].apply(lambda x: f"Train size: {x} samples")
    data_good.loc[data_good["type"] == "random", "style"] = "Random function"

    styles = ["Train size: 128 samples", "Train size: 256 samples", "Train size: 512 samples", "Random function"]

    g = sns.jointplot(data=data_good, x="sensitivity", y="sharpness", hue="style", kind="scatter", hue_order=styles, height=3.5)
    g.ax_joint.cla()

    for style in styles:
        df = data_good[data_good["style"] == style]
        g.ax_joint.scatter(
            df["sensitivity"], 
            df["sharpness"], 
            label=style,
            marker="s" if style == "Random function" else "D",
            s=3,
            alpha=0.5
        )

    plt.xlabel("Average Sensitivity")
    plt.ylabel("Sharpness")

    g.ax_marg_x.grid(False)
    g.ax_marg_y.grid(False)

    plt.legend()

    plt.savefig(os.path.join(save_path, "generalization/generalization.pdf"), bbox_inches="tight")
    plt.clf()

    styles = ["Train size: 128 samples", "Train size: 256 samples", "Train size: 512 samples"]

    data_good = data_good[data_good["style"] != "Random function"]

    g = sns.jointplot(data=data_good, x="sharpness", y="train_sharpness", hue="style", kind="scatter", hue_order=styles, height=3.5)
    g.ax_joint.cla()

    for style in styles:
        df = data_good[data_good["style"] == style]
        g.ax_joint.scatter(
            df["sharpness"], 
            df["train_sharpness"], 
            label=style,
            marker="s" if style == "Random function" else "D",
            s=3,
            alpha=0.5
        )

    plt.xlabel("Sharpness")
    plt.ylabel("Train Sharpness")

    g.ax_marg_x.grid(False)
    g.ax_marg_y.grid(False)

    plt.legend()

    plt.savefig(os.path.join(save_path, "generalization/train_sharpness.pdf"), bbox_inches="tight")
    plt.clf()
