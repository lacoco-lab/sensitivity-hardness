import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import logging
import argparse

from src.plotting import *


logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)s %(asctime)s - %(message)s', 
                    datefmt='%d-%m-%Y %H:%M')


def parse_args():
    parser = argparse.ArgumentParser(description='Process some experiments.')
    parser.add_argument('--experiment', choices=['scaling', 'dynamic', 'tradeoff', 'generalization', 'scratchpad'])
    parser.add_argument('--save_path', type=str, default="images/", required=False)
    return parser.parse_args()


def main():
    args = parse_args()

    api = wandb.Api()
    wandb.login()

    if args.experiment == 'scaling':
        xlim, ylim = plot_scaling_experiment(api, args.save_path, multiply_by_4=True)
        plot_scaling_higher_length_experiment(api, args.save_path, multiply_by_4=True, xlim=xlim, ylim=ylim)
    elif args.experiment == 'dynamic':
        plot_dynamic_main_experiment(api, args.save_path, multiply_by_4=True)
        plot_dynamic_100k_experiment(api, args.save_path, multiply_by_4=True)
    elif args.experiment == 'tradeoff':
        plot_tradeoff_4_experiment(api, args.save_path, multiply_by_4=True)
        plot_tradeoff_main_experiment(api, args.save_path, multiply_by_4=True)
    elif args.experiment == 'generalization':
        plot_generalization_experiment(api, args.save_path, multiply_by_4=True)
    elif args.experiment == 'scratchpad':
        plot_scratchpad_experiment(api, args.save_path, multiply_by_4=True)


if __name__ == "__main__":
    main()
