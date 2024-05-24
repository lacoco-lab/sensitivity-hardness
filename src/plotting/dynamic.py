import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

from src.plotting.style import set_style
from src.utils import wandb_prefix


def plot_dynamic_main_experiment(api: wandb.Api, save_path: str, multiply_by_4: bool):
    DYNAMIC_MAIN_SWEEP = input("Enter the ID of dynamic_metrics_1 sweep: ")

    set_style()

    os.makedirs(os.path.join(save_path, "dynamic"), exist_ok=True)

    sweep = api.sweep(f"{wandb_prefix()}/{DYNAMIC_MAIN_SWEEP}")

    hists = defaultdict(list)

    n_bad_runs = 0

    for run in tqdm(sweep.runs):
        if run.state != "finished" or run.summary["eval/loss"] >= 0.001:
            n_bad_runs += 1
            continue

        wd = run.config["training"]["optimizer"]["weight_decay"]

        metrics = run.history().loc[:, [
            "dynamic_measurements/batch_0/loss",
            "dynamic_measurements/batch_0/hessian_norm_proxy",
            "dynamic_measurements/batch_0/accuracy",
            "dynamic_measurements/batch_0/blowup_encoder_maxs_prod",
            "dynamic_measurements/squared_param_norm",
            "_step"
        ]].dropna()

        metrics.columns = ["loss", "sharpness", "accuracy", "blowup", "param_norm", "step"]
        metrics["param_norm"] = metrics["param_norm"] ** 0.5

        if multiply_by_4:
            metrics["sharpness"] *= 4

        hists[wd].append(metrics)
        

    print(f"Discarded {n_bad_runs} runs")

    for wd in hists.keys():
        for metrics in hists[wd]:
            convergence_step = metrics[metrics["loss"] < 0.01].iloc[0]["step"]
            metrics["steps_after_convergence"] = metrics["step"] - convergence_step

    colors = sns.color_palette()

    f, ax = plt.subplots(figsize=(3.5, 1.5))

    for metric in hists[0.1]:
        sns.lineplot(data=metric, x="steps_after_convergence", y="loss", alpha=0.05, color=colors[0], ax=ax)

    merged_dataset = pd.concat(hists[0.1])
    max_count = merged_dataset.groupby("steps_after_convergence").count().max()["loss"]
    bool_idx = merged_dataset.groupby("steps_after_convergence").count()["loss"] >= max_count / 4
    good_steps = bool_idx[bool_idx].index
    merged_dataset = merged_dataset.set_index("steps_after_convergence").loc[good_steps].reset_index()

    lns1 = sns.lineplot(data=merged_dataset, x="steps_after_convergence", y="loss", color=colors[0], ax=ax, errorbar=None, linewidth=1.5,
                        label="Loss")

    ax2 = plt.twinx()


    for metric in hists[0.1]:
        sns.lineplot(data=metric, x="steps_after_convergence", y="sharpness", alpha=0.05, color=colors[1], ax=ax2)

    merged_dataset = pd.concat(hists[0.1])
    max_count = merged_dataset.groupby("steps_after_convergence").count().max()["sharpness"]
    bool_idx = merged_dataset.groupby("steps_after_convergence").count()["sharpness"] >= max_count / 4
    good_steps = bool_idx[bool_idx].index
    merged_dataset = merged_dataset.set_index("steps_after_convergence").loc[good_steps].reset_index()

    lns2 = sns.lineplot(data=merged_dataset, x="steps_after_convergence", y="sharpness", color=colors[1], ax=ax2, errorbar=None, linewidth=1.5,
                        label="Sharpness")

    ax.set_ylabel("Loss")
    ax2.set_ylabel("Sharpness")
    ax.set_xlabel("Steps after convergence")

    ax.set_xlim(-5000, 15000)

    labels = ["Loss", "Sharpness"]

    ax2.legend(lns1.lines[-1:] + lns2.lines[-1:], labels, loc="upper right");

    plt.savefig(os.path.join(save_path, f"dynamic/small.pdf"), bbox_inches="tight")
    plt.clf()

    left_borders = [-800, -900, -1100, -1200]

    for i_wd, wd in enumerate([0.0, 0.1, 0.2, 0.3]):

        print(f"Weight decay: {wd}")

        fig, ax = plt.subplots(4, 1, figsize=(3.5, 5), sharex=True)

        keys = ["loss", "sharpness", "blowup", "param_norm"]

        for i_key, key in enumerate(keys):
            for metric in hists[wd]:
                sns.lineplot(data=metric, x="steps_after_convergence", y=key, alpha=0.1, color=colors[i_key], ax=ax[i_key])

            merged_dataset = pd.concat(hists[wd])
            max_count = merged_dataset.groupby("steps_after_convergence").count().max()["loss"]
            bool_idx = merged_dataset.groupby("steps_after_convergence").count()["loss"] >= max_count / 4
            good_steps = bool_idx[bool_idx].index
            merged_dataset = merged_dataset.set_index("steps_after_convergence").loc[good_steps].reset_index()

            ax[i_key].axvline(x=left_borders[i_wd], color="red", linestyle="--")
            ax[i_key].axvline(x=0, color="red", linestyle="--")

            sns.lineplot(data=merged_dataset, x="steps_after_convergence", y=key, color=colors[i_key], ax=ax[i_key], errorbar=None, linewidth=1.5)

        ax[2].set_yscale("log")

        ax[0].set_ylabel("Loss")
        ax[1].set_ylabel("Sharpness")
        ax[2].set_ylabel("LN Blowup")
        ax[3].set_ylabel("Weight Norm (L2)")
        plt.xlabel("Steps after convergence")

        fig.align_ylabels()

        plt.savefig(os.path.join(save_path, f"dynamic/main_{wd}.pdf"), bbox_inches="tight")
        plt.clf()

    colors = sns.color_palette()

    hists2 = deepcopy(hists)

    for wd in hists2.keys():

        plt.figure(figsize=(3.5, 2))

        for i, metric in enumerate(hists2[wd]):
            hists2[wd][i] = metric[(metric["steps_after_convergence"] > 3000) & (metric["steps_after_convergence"] < 10000)]

        print(f"Weight decay: {wd}")

        for metric in hists2[wd]:
            sns.lineplot(data=metric, x="param_norm", y="blowup", alpha=0.1, color=colors[4])

        merged_dataset = pd.concat(hists2[wd])
        merged_dataset = merged_dataset.groupby("steps_after_convergence").mean().reset_index()

        sns.lineplot(data=merged_dataset, x="param_norm", y="blowup", color=colors[0], linewidth=2)

        plt.yscale("log")

        plt.xlabel("Weight Norm (L2)")
        plt.ylabel("LN Blowup")

        plt.savefig(os.path.join(save_path, f"dynamic/tradeoff_{wd}.pdf"), bbox_inches="tight")
        plt.clf()


def plot_dynamic_100k_experiment(api: wandb.Api, save_path: str, multiply_by_4: bool):
    DYNAMIC_100K_SWEEP = input("Enter the ID of dynamic_metrics_2 sweep: ")

    set_style()

    sweep = api.sweep(f"{wandb_prefix()}/{DYNAMIC_100K_SWEEP}")

    hists = defaultdict(list)

    n_bad_runs = 0

    for run in tqdm(sweep.runs):
        if run.state != "finished" or run.summary["eval/loss"] >= 0.001:
            n_bad_runs += 1
            continue

        wd = run.config["training"]["optimizer"]["weight_decay"]

        metrics = run.history().loc[:, [
            "dynamic_measurements/batch_0/loss",
            "dynamic_measurements/batch_0/hessian_norm_proxy",
            "dynamic_measurements/batch_0/accuracy",
            "dynamic_measurements/batch_0/blowup_encoder_maxs_prod",
            "dynamic_measurements/squared_param_norm",
            "_step"
        ]].dropna()

        metrics.columns = ["loss", "sharpness", "accuracy", "blowup", "param_norm", "step"]
        metrics["param_norm"] = metrics["param_norm"] ** 0.5

        if multiply_by_4:
            metrics["sharpness"] *= 4

        hists[wd].append(metrics)

    print(f"Discarded {n_bad_runs} runs")

    for wd in hists.keys():
        for metrics in hists[wd]:
            convergence_step = metrics[metrics["loss"] < 0.01].iloc[0]["step"]
            metrics["steps_after_convergence"] = metrics["step"] - convergence_step

    colors = sns.color_palette()

    left_borders = [-800, -900]

    for i_wd, wd in enumerate([0.0, 0.1]):

        print(f"Weight decay: {wd}")

        fig, ax = plt.subplots(4, 1, figsize=(3.5, 5), sharex=True)

        keys = ["loss", "sharpness", "blowup", "param_norm"]

        for i_key, key in enumerate(keys):
            for metric in hists[wd]:
                sns.lineplot(data=metric, x="steps_after_convergence", y=key, alpha=0.1, color=colors[i_key], ax=ax[i_key])

            merged_dataset = pd.concat(hists[wd])
            max_count = merged_dataset.groupby("steps_after_convergence").count().max()["loss"]
            bool_idx = merged_dataset.groupby("steps_after_convergence").count()["loss"] >= max_count / 4
            good_steps = bool_idx[bool_idx].index
            merged_dataset = merged_dataset.set_index("steps_after_convergence").loc[good_steps].reset_index()

            # ax[i_key].axvline(x=left_borders[i_wd], color="red", linestyle="--")
            # ax[i_key].axvline(x=0, color="red", linestyle="--")

            sns.lineplot(data=merged_dataset, x="steps_after_convergence", y=key, color=colors[i_key], ax=ax[i_key], errorbar=None, linewidth=1.5)

        ax[2].set_yscale("log")

        ax[0].set_ylabel("Loss")
        ax[1].set_ylabel("Sharpness")
        ax[2].set_ylabel("LN Blowup")
        ax[3].set_ylabel("Weight Norm (L2)")
        plt.xlabel("Steps after convergence")

        fig.align_ylabels()

        plt.savefig(os.path.join(save_path, f"dynamic/100k_main_{wd}.pdf"), bbox_inches="tight")
        plt.clf()

    # now with shared axis
        
    fig, ax = plt.subplots(4, 1, figsize=(3.5, 5), sharex=True)
    for i_wd, wd in enumerate([0.0, 0.1]):
        keys = ["loss", "sharpness", "blowup", "param_norm"]

        for i_key, key in enumerate(keys):
            for metric in hists[wd]:
                sns.lineplot(data=metric, x="steps_after_convergence", y=key, alpha=0.1, color=colors[i_key], ax=ax[i_key])

            merged_dataset = pd.concat(hists[wd])
            max_count = merged_dataset.groupby("steps_after_convergence").count().max()["loss"]
            bool_idx = merged_dataset.groupby("steps_after_convergence").count()["loss"] >= max_count / 4
            good_steps = bool_idx[bool_idx].index
            merged_dataset = merged_dataset.set_index("steps_after_convergence").loc[good_steps].reset_index()

            sns.lineplot(data=merged_dataset, x="steps_after_convergence", y=key, color=colors[i_key], ax=ax[i_key], errorbar=None, linewidth=1.5)

    ax[2].set_yscale("log")

    xlims = [ax[i].get_xlim() for i in range(4)]
    ylims = [ax[i].get_ylim() for i in range(4)]

    plt.clf()

    for i_wd, wd in enumerate([0.0, 0.1]):

        print(f"Weight decay: {wd}")

        fig, ax = plt.subplots(4, 1, figsize=(3.5, 5), sharex=True)

        keys = ["loss", "sharpness", "blowup", "param_norm"]

        for i_key, key in enumerate(keys):
            for metric in hists[wd]:
                sns.lineplot(data=metric, x="steps_after_convergence", y=key, alpha=0.1, color=colors[i_key], ax=ax[i_key])

            merged_dataset = pd.concat(hists[wd])
            max_count = merged_dataset.groupby("steps_after_convergence").count().max()["loss"]
            bool_idx = merged_dataset.groupby("steps_after_convergence").count()["loss"] >= max_count / 4
            good_steps = bool_idx[bool_idx].index
            merged_dataset = merged_dataset.set_index("steps_after_convergence").loc[good_steps].reset_index()

            # ax[i_key].axvline(x=left_borders[i_wd], color="red", linestyle="--")
            # ax[i_key].axvline(x=0, color="red", linestyle="--")

            sns.lineplot(data=merged_dataset, x="steps_after_convergence", y=key, color=colors[i_key], ax=ax[i_key], errorbar=None, linewidth=1.5)

        ax[2].set_yscale("log")

        ax[0].set_ylabel("Loss")
        ax[1].set_ylabel("Sharpness")
        ax[2].set_ylabel("LN Blowup")
        ax[3].set_ylabel("Weight Norm (L2)")
        plt.xlabel("Steps after convergence")

        for i in range(4):
            ax[i].set_xlim(xlims[i])
            ax[i].set_ylim(ylims[i])

        fig.align_ylabels()

        plt.savefig(os.path.join(save_path, f"dynamic/100k_main_{wd}_aligned_axis.pdf"), bbox_inches="tight")
        plt.clf()
