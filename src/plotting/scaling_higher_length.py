import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os
import numpy as np

from collections import defaultdict
from tqdm import tqdm

from src.plotting.style import set_style
from src.utils import wandb_prefix


def plot_scaling_higher_length_experiment(api: wandb.Api, save_path: str, multiply_by_4: bool,
                                          xlim=None, ylim=None):
    SCALING_HIGHER_LENGTH_SWEEP = input("Enter the ID of length_scaling_2 sweep: ")
    set_style()

    sweep = api.sweep(f"{wandb_prefix()}/{SCALING_HIGHER_LENGTH_SWEEP}")

    sharpness = defaultdict(list)
    length = defaultdict(list)
    blowup = defaultdict(list)
    param_norm = defaultdict(list)

    n_bad_runs = 0

    for run in tqdm(sweep.runs):
        if run.state != "finished" or len(dict(run.summary)) == 0 or run.summary["eval/loss"] >= 0.001:
            n_bad_runs += 1
            continue

        target = run.config["basic"]["target"]["_target_"].split(".")[-1]

        sharpness[target].append(run.summary["measurements/hessian_norm_proxy"])
        length[target].append(run.config["basic"]["max_len"])
        blowup[target].append(run.summary["measurements/blowup_encoder_maxs_prod"])
        param_norm[target].append(run.summary["measurements/param_norm"])

    if multiply_by_4:
        for target in sharpness.keys():
            sharpness[target] = [x * 4 for x in sharpness[target]]

    TARGETS = ["Majority", "First", "Mean"]

    for i, target in enumerate(TARGETS):
        sns.lineplot(x=length[target], y=sharpness[target], label=target, errorbar=("sd", 2), color=sns.color_palette()[i + 1])
        sns.scatterplot(x=length[target], y=sharpness[target], s=15, alpha=0.5, color=sns.color_palette()[i + 1])
        plt.legend(loc="upper left")
        plt.xlabel("Sequence Length")
        plt.ylabel("Sharpness")
        plt.savefig(os.path.join(save_path, f"scaling/sharpness_{target}_higher_len.pdf"), bbox_inches="tight")
        plt.clf()

    for i, target in enumerate(TARGETS):
        sns.lineplot(x=length[target], y=blowup[target], label=target, errorbar=("sd", 2), color=sns.color_palette()[i + 1])
        sns.scatterplot(x=length[target], y=blowup[target], s=15, alpha=0.5, color=sns.color_palette()[i + 1])
        plt.legend(loc="upper left")
        plt.xlabel("Sequence Length")
        plt.ylabel("LayerNorm Blowup")
        plt.savefig(os.path.join(save_path, f"scaling/blowup_{target}_higher_len.pdf"), bbox_inches="tight")
        plt.clf()

    for i, target in enumerate(TARGETS):
        sns.lineplot(x=length[target], y=param_norm[target], label=target, errorbar=("sd", 2), color=sns.color_palette()[i + 1])
        sns.scatterplot(x=length[target], y=param_norm[target], s=15, alpha=0.5, color=sns.color_palette()[i + 1])
        plt.legend(loc="upper left")
        plt.xlabel("Sequence Length")
        plt.ylabel("Parameter Norm (L2)")
        plt.savefig(os.path.join(save_path, f"scaling/param_norm_{target}_higher_len.pdf"), bbox_inches="tight")
        plt.clf()

    # now sharpness with aligned axis
        
    for i, target in enumerate(TARGETS):
        sns.lineplot(x=length[target], y=sharpness[target], label=target, errorbar=("sd", 2), color=sns.color_palette()[i + 1])
        sns.scatterplot(x=length[target], y=sharpness[target], s=15, alpha=0.5, color=sns.color_palette()[i + 1])
        plt.legend(loc="upper left")
        plt.xlabel("Sequence Length")
        plt.ylabel("Sharpness")
        plt.ylim(ylim)
        plt.savefig(os.path.join(save_path, f"scaling/sharpness_{target}_higher_len_aligned_axis.pdf"), bbox_inches="tight")
        plt.clf()
