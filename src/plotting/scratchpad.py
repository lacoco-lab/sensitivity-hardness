import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from tqdm import tqdm

from src.plotting.style import set_style
from src.utils import wandb_prefix


def plot_scratchpad_experiment(api: wandb.Api, save_path: str, multiply_by_4: bool):
    SCRATCHPAD_SWEEP = input("Enter the ID of scratchpad sweep: ")
    set_style()

    os.makedirs(os.path.join(save_path, "scratchpad"), exist_ok=True)

    sweep = api.sweep(f"{wandb_prefix()}/{SCRATCHPAD_SWEEP}")

    sharpness = []
    length = []
    blowup_encoder = []
    blowup_decoder = []
    param_norm = []

    n_bad_runs = 0

    for run in tqdm(sweep.runs):
        if run.state != "finished" or run.summary["measurements/autoregressive_accuracy"] <= 0.999:
            n_bad_runs += 1
            continue

        sharpness.append(run.summary["measurements/hessian_norm_proxy"])
        length.append(run.config["basic"]["max_len"])
        blowup_encoder.append(run.summary["measurements/blowup_encoder_maxs_prod"])
        blowup_decoder.append(run.summary["measurements/blowup_decoder_maxs_prod"])
        param_norm.append(run.summary["measurements/param_norm"])

    if multiply_by_4:
        sharpness = [x * 4 for x in sharpness]

    print(f"Discarded {n_bad_runs} runs")

    sns.lineplot(x=length, y=sharpness, errorbar=("sd", 2))
    sns.scatterplot(x=length, y=sharpness, s=15, alpha=0.5)
    plt.xlabel("Sequence Length")
    plt.ylabel("Sharpness")

    plt.savefig(os.path.join(save_path, "scratchpad/sharpness.pdf"), bbox_inches="tight")
    plt.clf()

    sns.lineplot(x=length, y=blowup_encoder, errorbar=("sd", 2))
    sns.scatterplot(x=length, y=blowup_encoder, s=15, alpha=0.5)
    plt.xlabel("Sequence Length")
    plt.ylabel("LayerNorm Blowup in Encoder")

    plt.savefig(os.path.join(save_path, "scratchpad/blowup_encoder.pdf"), bbox_inches="tight")
    plt.clf()

    sns.lineplot(x=length, y=blowup_decoder, errorbar=("sd", 2))
    sns.scatterplot(x=length, y=blowup_decoder, s=15, alpha=0.5)
    plt.xlabel("Sequence Length")
    plt.ylabel("LayerNorm Blowup in Decoder")

    plt.savefig(os.path.join(save_path, "scratchpad/blowup_decoder.pdf"), bbox_inches="tight")
    plt.clf()
        
    sns.lineplot(x=length, y=param_norm, errorbar=("sd", 2))
    sns.scatterplot(x=length, y=param_norm, s=15, alpha=0.5)
    plt.xlabel("Sequence Length")
    plt.ylabel("Parameter Norm (L2)")

    plt.savefig(os.path.join(save_path, "scratchpad/param_norm.pdf"), bbox_inches="tight")
    plt.clf()
