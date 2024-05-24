import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os
import numpy as np
from scipy.stats import linregress

from collections import defaultdict
from tqdm import tqdm
from matplotlib.ticker import FormatStrFormatter

from src.plotting.style import set_style
from src.utils import wandb_prefix


def plot_scaling_experiment(api: wandb.Api, save_path: str, multiply_by_4: bool):
    SCALING_SWEEP = input("Enter the ID of length_scaling_1 sweep: ")

    set_style()

    sweep = api.sweep(f"{wandb_prefix()}/{SCALING_SWEEP}")

    sharpness = defaultdict(list)
    length = defaultdict(list)
    blowup = defaultdict(list)
    param_norm = defaultdict(list)

    n_bad_runs = 0

    for run in tqdm(sweep.runs):
        if run.summary["eval/loss"] >= 0.001:
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

    print(f"Discarded {n_bad_runs} runs")

    print(f"Scaling experiment: getting slopes:")
    for target in ["Parity", "Majority", "First", "Mean"]:
        x = np.array(length[target])
        y = np.array(sharpness[target])

        linreg_res = linregress(x, y)
        print(f"{target:10}: slope   {linreg_res.slope:12.8f}    corr   {linreg_res.rvalue:5.2f}    p-value   {linreg_res.pvalue:.2}")
    
    print("\n\n")

    TARGETS = ["Parity", "Majority", "First", "Mean"]

    os.makedirs(os.path.join(save_path, "scaling"), exist_ok=True)

    for i, target in enumerate(["Parity", "Majority"]):
        sns.lineplot(x=length[target], y=sharpness[target], label=target, errorbar=("sd", 2), color=sns.color_palette()[i])
        sns.scatterplot(x=length[target], y=sharpness[target], s=15, alpha=0.5, color=sns.color_palette()[i])
        plt.legend(loc="upper left")
        plt.xlabel("Sequence Length")
        plt.ylabel("Sharpness")

    plt.savefig(os.path.join(save_path, f"scaling/sharpness_parity_and_majority.pdf"), bbox_inches="tight")
    plt.clf()

    for i, target in enumerate(TARGETS):
        sns.lineplot(x=length[target], y=sharpness[target], label=target, errorbar=("sd", 2), color=sns.color_palette()[i])
        sns.scatterplot(x=length[target], y=sharpness[target], s=15, alpha=0.5, color=sns.color_palette()[i])
        plt.legend(loc="upper left")
        plt.xlabel("Sequence Length")
        plt.ylabel("Sharpness")
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        plt.savefig(os.path.join(save_path, f"scaling/sharpness_{target}.pdf"), bbox_inches="tight")
        plt.clf()

    for i, target in enumerate(TARGETS):
        sns.lineplot(x=length[target], y=blowup[target], label=target, errorbar=("sd", 2), color=sns.color_palette()[i])
        sns.scatterplot(x=length[target], y=blowup[target], s=15, alpha=0.5, color=sns.color_palette()[i])
        plt.legend(loc="upper left")
        plt.xlabel("Sequence Length")
        plt.ylabel("LayerNorm Blowup")
        plt.savefig(os.path.join(save_path, f"scaling/blowup_{target}.pdf"), bbox_inches="tight")
        plt.clf()

    for i, target in enumerate(TARGETS):
        sns.lineplot(x=length[target], y=param_norm[target], label=target, errorbar=("sd", 2), color=sns.color_palette()[i])
        sns.scatterplot(x=length[target], y=param_norm[target], s=15, alpha=0.5, color=sns.color_palette()[i])
        plt.legend(loc="upper left")
        plt.xlabel("Sequence Length")
        plt.ylabel("Parameter Norm (L2)")
        plt.savefig(os.path.join(save_path, f"scaling/param_norm_{target}.pdf"), bbox_inches="tight")
        plt.clf()

    # now sharpness with aligned axis
        
    for i, target in enumerate(TARGETS):
        sns.lineplot(x=length[target], y=sharpness[target], label=target, errorbar=("sd", 2), color=sns.color_palette()[i])
        sns.scatterplot(x=length[target], y=sharpness[target], s=15, alpha=0.5, color=sns.color_palette()[i])

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    ylim = (0, ylim[1])
    plt.clf()

    for i, target in enumerate(TARGETS):
        sns.lineplot(x=length[target], y=sharpness[target], label=target, errorbar=("sd", 2), color=sns.color_palette()[i])
        sns.scatterplot(x=length[target], y=sharpness[target], s=15, alpha=0.5, color=sns.color_palette()[i])
        plt.legend(loc="upper left")
        plt.xlabel("Sequence Length")
        plt.ylabel("Sharpness")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        plt.savefig(os.path.join(save_path, f"scaling/sharpness_{target}_aligned_axis.pdf"), bbox_inches="tight")
        plt.clf()

    return xlim, ylim
