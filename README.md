# Why are Sensitive Functions Hard for Transformers?

This is the official repository of the [ACL 2024 paper](https://arxiv.org/abs/2402.09963) "Why are Sensitive Functions Hard for Transformers?".

# Usage

To run the code, you need Python 3.8. Installing required packages can be done by running `pip install -r requirements.txt`.

To train the models analogous to what we used in the paper and track their metrics, simply run `python train.py`. We use [Hydra](https://hydra.cc/docs/intro/) for parameter configuration. With it, you can change training or model parameters in the command line, e.g.

```
python train.py model.num_layers=12 training.batch_size=512
```

All parameters with their descriptions are listed in [conf/train_config.yaml](conf/train_config.yaml).

To use Weights & Biases (turned on by default), first configure W&B credentials:
```
export WANDB_ENTITY=your_username
export WANDB_PROJECT=your_project
wandb login
```

If something doesn't work, feel free to leave a GitHub issue or contact us by email!

# Replicating experiments

All our experiments are logged using Weights & Biases sweeps due to their easy configuration and scalability. The directory [sweeps/](sweeps/) contains all the configuration files. Please refer to the [Weights & Biases documentation](https://docs.wandb.ai) for details.

### Scaling experiment (Figures 1, 5, 6, 7, and 8 in the paper)
Run
```
wandb sweep sweeps/length_scaling_1.yaml
wandb sweep sweeps/length_scaling_2.yaml

wandb agent your_username/your_project/sweep_id_1
wandb agent your_username/your_project/sweep_id_2

python generate_plots.py --experiment scaling
```

Then, the visualizations will be saved in the [images/](images/) folder.

The first two commands will output sweep ids that are used in starting agents and generating plots.

### Tradeoff experiment (Figures 2, 9, and 10 in the paper)

Same as scaling, but use the sweeps [weight_norm_blowup_tradeoff_1.yaml](sweeps/weight_norm_blowup_tradeoff_1.yaml), [weight_norm_blowup_tradeoff_2.yaml](sweeps/weight_norm_blowup_tradeoff_2.yaml), [weight_norm_blowup_tradeoff_3.yaml](sweeps/weight_norm_blowup_tradeoff_3.yaml), and [weight_norm_blowup_tradeoff_4.yaml](sweeps/weight_norm_blowup_tradeoff_4.yaml). Also use the argument `--experiment tradeoff` for `generate_plots.py`.

### Dynamic experiment (Figures 3, 12, 13, and 14 in the paper)

Same as scaling, but use the sweeps [dynamic_metrics_1.yaml](sweeps/dynamic_metrics_1.yaml) and [dynamic_metrics_2.yaml](sweeps/dynamic_metrics_2.yaml). Also use the argument `--experiment dynamic` for `generate_plots.py`.

### Scratchpad experiment (Figure 11 in the paper)

Same as scaling, but use the sweep [scratchpad.yaml](sweeps/scratchpad.yaml). Also use the argument `--experiment scratchpad` for `generate_plots.py`.

### Generalization experiment (Figures 4 and 15 in the paper)

1. Run the sweeps [generalization_first_step_1.yaml](sweeps/generalization_first_step_1.yaml) and [generalization_first_step_2.yaml](sweeps/generalization_first_step_2.yaml).
2. Run `python generalization_save_predictions.py`.
3. Run the sweeps [generalization_second_step_1.yaml](sweeps/generalization_second_step_1.yaml) and [generalization_second_step_2.yaml](sweeps/generalization_second_step_2.yaml).
4. Run `python generalization_add_train_sharpness.py`.
5. Run `python generate_plots.py --experiment generalization`.

# Citation

```
@article{Hahn2024WhyAS,
  title={Why are Sensitive Functions Hard for Transformers?},
  author={Michael Hahn and Mark Rofin},
  journal={ArXiv},
  year={2024},
  volume={abs/2402.09963}
}
```