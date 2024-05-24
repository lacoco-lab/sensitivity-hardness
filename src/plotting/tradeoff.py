import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os
import numpy as np
import pandas as pd
import matplotlib as mpl

from matplotlib import ticker
from collections import defaultdict
from tqdm import tqdm
from itertools import chain

from src.plotting.style import set_style
from src.utils import wandb_prefix


def check_run(config, summary):
    return summary["eval/loss"] < 0.001


def plot_tradeoff_main_experiment(api: wandb.Api, save_path: str, multiply_by_4: bool):
    TRADEOFF_MAIN_SWEEPS = [
        input("Enter the ID of weight_norm_blowup_tradeoff_1 sweep: "),
        input("Enter the ID of weight_norm_blowup_tradeoff_2 sweep: "),
    ]
    set_style()

    sweep_runs = [api.sweep(f"{wandb_prefix()}/{sweep}").runs for sweep in TRADEOFF_MAIN_SWEEPS]

    lengths = []
    blowups = []
    param_norms = []
    targets = []

    for run in tqdm(chain(*sweep_runs)):
        if run.state != "finished" or not check_run(run.config, run.summary):
            continue

        lengths.append(run.config["basic"]["max_len"])
        blowups.append(run.summary["measurements/blowup_encoder_maxs_prod"])
        param_norms.append(run.summary["measurements/param_norm"])
        targets.append(run.config["basic.target._target_"].split(".")[-1])

    metrics = pd.DataFrame({
        "length": lengths,
        "blowup": blowups,
        "param_norm": param_norms,
        "target": targets
    })

    for i, target in enumerate(["Parity", "Majority", "First", "Mean"]):
        plt.figure()
        ax = sns.scatterplot(data=metrics[metrics["target"] == target], x="blowup", y="param_norm", 
                            marker="D", s=15, hue="length", palette="plasma", alpha=0.5)
        plt.xscale("log")

        norm = mpl.colors.Normalize(vmin=metrics["length"].min(), vmax=metrics["length"].max())
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])

        # Add the color bar to the plot
        cbar = plt.colorbar(sm, label='Sequence Length')
        cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend_.remove()

        ax.set_xlabel("LayerNorm Blowup")
        ax.set_ylabel("Parameter Norm (L2)")

        plt.savefig(os.path.join(save_path, f"tradeoff/{target}_different_lengths.pdf"), bbox_inches="tight")

        plt.clf()

    # now same but with shared axis
        
    for i, target in enumerate(["Parity", "Majority", "First", "Mean"]):
        sns.scatterplot(data=metrics[metrics["target"] == target], x="blowup", y="param_norm", 
                        marker="D", s=15, hue="length", palette="plasma", alpha=0.5)
        plt.xscale("log")

    xlim = plt.xlim()
    ylim = plt.ylim()

    plt.clf()

    for i, target in enumerate(["Parity", "Majority", "First", "Mean"]):
        plt.figure()
        ax = sns.scatterplot(data=metrics[metrics["target"] == target], x="blowup", y="param_norm", 
                            marker="D", s=15, hue="length", palette="plasma", alpha=0.5)
        plt.xscale("log")
        plt.xlim(xlim)
        plt.ylim(ylim)

        norm = mpl.colors.Normalize(vmin=metrics["length"].min(), vmax=metrics["length"].max())
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])

        # Add the color bar to the plot
        cbar = plt.colorbar(sm, label='Sequence Length')
        cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend_.remove()

        ax.set_xlabel("LayerNorm Blowup")
        ax.set_ylabel("Parameter Norm (L2)")

        plt.savefig(os.path.join(save_path, f"tradeoff/{target}_different_lengths_aligned_axis.pdf"), bbox_inches="tight")

        plt.clf()


def plot_tradeoff_4_experiment(api: wandb.Api, save_path: str, multiply_by_4: bool):
    TRADEOFF_4_SWEEPS = [
        input("Enter the ID of weight_norm_blowup_tradeoff_3 sweep: "),
        input("Enter the ID of weight_norm_blowup_tradeoff_4 sweep: "),
    ]
    set_style()

    os.makedirs(os.path.join(save_path, "tradeoff"), exist_ok=True)

    sweep_runs = [api.sweep(f"{wandb_prefix()}/{sweep}").runs for sweep in TRADEOFF_4_SWEEPS]

    lengths = []
    blowups = []
    param_norms = []
    targets = []

    for run in tqdm(chain(*sweep_runs)):
        if run.state != "finished" or not check_run(run.config, run.summary):
            continue

        lengths.append(run.config["basic"]["max_len"])
        blowups.append(run.summary["measurements/blowup_encoder_maxs_prod"])
        param_norms.append(run.summary["measurements/param_norm"])
        targets.append(run.config["basic.target._target_"].split(".")[-1])

    metrics = pd.DataFrame({
        "length": lengths,
        "blowup": blowups,
        "param_norm": param_norms,
        "target": targets
    })

    f, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(6, 5))

    colormap = sns.color_palette()

    for j, length in enumerate([5, 10, 15, 20]):
        for i, target in enumerate(["Parity", "Majority", "First", "Mean"]):
            sns.scatterplot(data=metrics[(metrics["length"] == length) & (metrics["target"] == target)], x="blowup", y="param_norm", label=target,
                            ax=ax[j // 2, j % 2], color=colormap[i], marker="D", s=20, alpha=0.6)

        ax[j // 2, j % 2].set_xscale("log")
        ax[j // 2, j % 2].set_title(f"Sequence Length={length}")

        ax[j // 2, j % 2].set_xlabel("LayerNorm Blowup")
        ax[j // 2, j % 2].set_ylabel("Parameter Norm (L2)")

    plt.tight_layout()

    plt.savefig(os.path.join(save_path, "tradeoff/4_lengths.pdf"), bbox_inches="tight")
    plt.clf()
