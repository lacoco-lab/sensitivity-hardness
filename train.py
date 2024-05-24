from omegaconf import DictConfig, OmegaConf
from tqdm import trange

import torch
import hydra
import logging
import wandb
import os

import numpy as np

from src.utils import set_random_seed, RestrictedSampler
from src.model import TorchModel, TorchModelWithScratchpad
from src.training import train, sample_batch

import src.targets as targets
import src.measurements as ms

from collections import defaultdict


logging.basicConfig(level=logging.DEBUG, 
                    format='%(levelname)s %(asctime)s - %(message)s', 
                    datefmt='%d-%m-%Y %H:%M')


# instantiate objects from config
def instantiate_objects(cfg, model: torch.nn.Module):
    cfg["training"]["optimizer"]["params"] = model.parameters()
    cfg["training"]["optimizer"] = hydra.utils.instantiate(cfg["training"]["optimizer"])

    if cfg["basic"]["target"]["_target_"] == "src.targets.RandomFunc":
        cfg["basic"]["target"]["length"] = cfg["basic"]["max_len"]
        cfg["basic"]["target"]["device"] = cfg["basic"]["device"]

        if cfg["basic"]["target_seed"] is None and cfg["basic"]["target_dir"] is None:
            raise ValueError("RandomFunc target requires seed or directory to be specified")
        elif cfg["basic"]["target_seed"] is not None and cfg["basic"]["target_dir"] is not None:
            raise ValueError("RandomFunc target requires only one of seed or directory to be specified")
        elif cfg["basic"]["target_seed"] is not None:
            cfg["basic"]["target"]["seed"] = cfg["basic"]["target_seed"]
            logging.info(f"Using RandomFunc target with seed {cfg['basic']['target_seed']}")
        else:
            cfg["basic"]["target"]["seed"] = 0

    cfg["basic"]["target"] = hydra.utils.instantiate(cfg["basic"]["target"])

    # load a saved function from file (required for generalization experiment)
    if isinstance(cfg["basic"]["target"], targets.RandomFunc):
        if cfg["basic"]["target_dir"] is not None:
            dir_path = os.path.join("../../..", cfg["basic"]["target_dir"])
            filename = os.listdir(dir_path)[cfg["basic"]["target_file_idx"]]
            samples = np.load(os.path.join(dir_path, filename))
            cfg["basic"]["target"].samples = torch.IntTensor(samples).to(cfg["basic"]["device"])
            logging.info(f"Loaded samples from {filename}!!")

            if "_samples-" in filename:
                run_id, random_state, target_seed, n_samples, t = filename.split("_")
                wandb.config["parent_function/n_samples"] = int(n_samples.replace("samples-", ""))
            else:
                run_id, random_state, target_seed, t = filename.split("_")
            wandb.config["parent_function/run_id"] = run_id
            wandb.config["parent_function/random_state"] = int(random_state)
            wandb.config["parent_function/target_seed"] = int(target_seed)
            wandb.config["parent_function/type"] = t.split(".")[0]
        else:
            torch.save(cfg["basic"]["target"].samples.cpu(), "target.pt")

    cfg["training"]["target"] = cfg["basic"]["target"]
    cfg["training"]["criterion"] = hydra.utils.instantiate(cfg["training"]["criterion"])

    if cfg["basic"]["restricted_size"] is not None:
        cfg["training"]["restricted_sampler"] = RestrictedSampler(
            length=cfg["basic"]["max_len"], max_samples=cfg["basic"]["restricted_size"], 
            seed=cfg["basic"]["random_state"]
        )

        torch.save(cfg["training"]["restricted_sampler"].samples.cpu(), "restricted_sampler.pt")

    del cfg["training"]["lr"]

@hydra.main(version_base=None, config_path="./conf", config_name="train_config")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    cfg["basic"]["directory"] = os.getcwd()

    if cfg["basic"]["use_wandb"]:
        wandb.init(
            name=cfg["basic"]["wandb_run"],
            config=cfg
        )
    else:
        logging.info("WandB disabled")

    logging.info(f"Starting run with config: {cfg}")
    set_random_seed(cfg["basic"]["random_state"])

    logging.info("Instantiating everything")

    if cfg["basic"]["scratchpad"]:
        model = TorchModelWithScratchpad(**cfg["model"]).to(cfg["basic"]["device"])
    else:
        model = TorchModel(**cfg["model"]).to(cfg["basic"]["device"])

    instantiate_objects(cfg, model)

    measurement_conf_for_training = dict(
        dynamic_measurement_interval = cfg["measurements"]["dynamic_interval"] if cfg["measurements"]["dynamic_run"] else -1,
        dynamic_measurement_n_batches = cfg["measurements"]["dynamic_n_batches"],
        random_seed = cfg["basic"]["random_state"],
        hessian_norm_proxy_args = cfg["measurements"]["hessian_norm_proxy"],
    )

    model, _ = train(model=model, **cfg["training"], **measurement_conf_for_training,
                     extended=cfg["basic"]["scratchpad"])

    if cfg["basic"]["save_model"]:
        torch.save(model.state_dict(), "model.pt")

    if cfg["basic"]["use_wandb"]:
        if cfg["basic"]["save_model"]:
            wandb.save("model.pt")

    # measuring all the metrics after training
    if cfg["measurements"]["run"]:
        logging.info("Registering hooks")
        hooks = ms.register_hooks(model)

        metrics = defaultdict(float)

        for _ in trange(cfg["measurements"]["num_steps"]):
            x, y = sample_batch(target=cfg["training"]["target"], batch_size=cfg["training"]["batch_size"], min_len=cfg["basic"]["max_len"],
                                max_len=cfg["basic"]["max_len"], device=cfg["basic"]["device"],
                                extended=isinstance(model, TorchModelWithScratchpad))
            
            batch_metrics, details = ms.inspect_one_batch(x, y, model, cfg["training"]["criterion"], cfg["training"]["target"],
                                                            hessian_norm_proxy_args=cfg["measurements"]["hessian_norm_proxy"])

            for k, v in batch_metrics.items():
                metrics[f"measurements/{k}"] += v / cfg["measurements"]["num_steps"]

        metrics["measurements/squared_param_norm"] = ms.squared_param_norm(model)
        metrics["measurements/param_norm"] = np.sqrt(metrics["measurements/squared_param_norm"])

        metrics2 = metrics.copy()

        for key, val in metrics2.items():
            key = key.replace("measurements/", "")
            if key.startswith("blowup"):
                metrics[f"measurements/param_norm_x_{key}"] = val * metrics["measurements/param_norm"]

        logging.info(f"Metrics: {metrics}")
        if cfg["basic"]["use_wandb"]:
            wandb.log(metrics)

        logging.info("Deleting hooks")
        ms.delete_hooks(hooks)

    if cfg["basic"]["use_wandb"]:
        wandb.finish()

    # saving prediction on all possible inputs, relevant for the generalization experiment
    if cfg["basic"]["save_prediction"]:
        xs = torch.arange(2 ** cfg["basic"]["max_len"])
        mask = 2 ** torch.arange(cfg["basic"]["max_len"] - 1, -1, -1)
        xs = xs.unsqueeze(-1).bitwise_and(mask).ne(0).int()

        predictions = []

        for i in trange(0, len(xs), cfg["training"]["batch_size"]):
            x = xs[i:i + cfg["training"]["batch_size"]].to(cfg["basic"]["device"])
            predictions.append((model(x) > 0.5).int().cpu().numpy())

        predictions = np.concatenate(predictions)
        np.save("predictions.npy", predictions)
        logging.info(f"Predictions saved to {os.getcwd()}/predictions.npy")

    if cfg["basic"]["save_model"]:
        logging.info(f"Model path: {os.getcwd()}/model.pt")

if __name__ == "__main__":
    main()
