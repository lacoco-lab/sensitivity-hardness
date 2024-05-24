import torch
import logging
import wandb
import tqdm
import random
import numpy as np

import src.targets as targets
import src.measurements as ms

from src.utils import set_random_seed, compute_loss, RestrictedSampler
from src.utils2 import get_output_loss_acc

from src.model import TorchModel
from typing import Tuple, Optional, Dict, Union


# samples batch from a target function
# restricted_sampler is needed only for the generalization experiment
# if it is not None, the samples will be generated only from a predefined set
def sample_batch(
    target: targets.Target, batch_size: int = 32, max_len: int = 1024,
    min_len: int = 16, device: torch.device = torch.device("cpu"), extended=False, restricted_sampler=None
) -> Tuple[torch.Tensor, torch.Tensor]:

    seq_len = random.randint(min_len, max_len)

    if restricted_sampler is not None:
        x = restricted_sampler(batch_size).to(device)
        y = target(x, extended=extended)
        return x, y

    x = torch.randint(low=0, high=2, size=(batch_size, seq_len), device=device)
    y = target(x, extended=extended)
    return x, y
        

# evaluates the model on a target function
def evaluate(
    model: TorchModel,
    target: targets.Target,
    criterion: torch.nn.modules.loss._Loss,
    restricted_sampler: Optional[RestrictedSampler] = None,
    max_len: int = 1024,
    min_len: int = 16,
    batch_size: int = 32,
    num_steps: int = 100,
    device: torch.device = torch.device("cpu"),
    extended=False
):
    model.eval()

    logging.info(f"Evaluating model on {device}")

    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for _ in tqdm.trange(num_steps):
            x, y = sample_batch(target=target, batch_size=batch_size, max_len=max_len, min_len=min_len,
                                device=device, extended=extended, restricted_sampler=restricted_sampler)

            output, loss, batch_accuracy = get_output_loss_acc(model, x, y, criterion)
            total_loss += loss.item()
            total_accuracy += batch_accuracy.item()

    return total_loss / num_steps, total_accuracy / num_steps


# the main function for training the model
def train(
    model: TorchModel,
    target: targets.Target,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    restricted_sampler: Optional[RestrictedSampler] = None,
    max_len: int = 1024,
    min_len: int = 16,
    batch_size: int = 32,
    num_steps: int = 10000,
    eval_steps: int = 100,
    log_interval: int = 100,
    device: torch.device = torch.device("cpu"),
    dynamic_measurement_interval: int = -1,
    dynamic_measurement_n_batches: int = 10,
    random_seed: int = 0,
    hessian_norm_proxy_args: Dict[str, Union[int, float]] = {},
    extended=False
) -> Tuple[TorchModel, bool]:
    model.train()

    if wandb.run:
        wandb.watch(model, log_freq=log_interval)

    logging.info(f"Training model on {device}")

    last_loss = 1e10
    logging_loss = logging_acc = 0

    measurement_batches = []
    hooks = None

    # generating batches to perform measurements on during training
    if dynamic_measurement_interval != -1:
        logging.info(f"Dynamic measurement is turned on")
        for _ in range(dynamic_measurement_n_batches):
            measurement_batches.append(
                sample_batch(target=target, batch_size=batch_size, max_len=max_len, min_len=min_len, device=device,
                             extended=extended)
            )
        hooks = ms.register_hooks(model)

    # training loop
    for step in tqdm.trange(num_steps):
        x, y = sample_batch(
            target=target, batch_size=batch_size, max_len=max_len, 
            min_len=min_len, device=device, extended=extended, restricted_sampler=restricted_sampler
        )

        output, loss, batch_accuracy = get_output_loss_acc(model, x, y, criterion)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        logging_loss += loss.item() / log_interval
        logging_acc += batch_accuracy.item() / log_interval
        last_loss = loss.item()

        if (step + 1) % log_interval == 0:
            if wandb.run:
                wandb.log({"train/loss": logging_loss, "train/accuracy": logging_acc}, step=step)
            logging.info(f"Step {step}: loss {logging_loss:10.3f}, accuracy {logging_acc:.2f}")

            logging_loss = logging_acc = 0

        # every dynamic_measurement_interval steps we track metrics of interest on pre-saved batches
        if dynamic_measurement_interval != -1 and (step + 1) % dynamic_measurement_interval == 0:
            
            rng = torch.Generator(device=device)
            rng.manual_seed(random_seed)
            hessian_norm_proxy_args["rng"] = rng

            for i, (x, y) in enumerate(measurement_batches):
                batch_metrics, details = ms.inspect_one_batch(
                    x, y, model, criterion, target,
                    hessian_norm_proxy_args=hessian_norm_proxy_args
                )

                for k, v in batch_metrics.items():
                    if wandb.run:
                        wandb.log({f"dynamic_measurements/batch_{i}/{k}": v}, step=step)
                    else:
                        raise ValueError("WandB is disabled but dynamic measurement is turned on")

                if wandb.run:
                    wandb.log({
                        f"dynamic_measurements/squared_param_norm": ms.squared_param_norm(model)
                    }, step=step)

    if hooks is not None:
        ms.delete_hooks(hooks)

    # final evaluation
    eval_loss, eval_accuracy = evaluate(
        model=model, target=target, criterion=criterion, max_len=max_len, min_len=min_len,
        batch_size=batch_size, num_steps=eval_steps, device=device,
        extended=extended, restricted_sampler=restricted_sampler
    )

    has_converged = int(eval_accuracy > 0.99)

    if wandb.run:
        wandb.log({"eval/loss": eval_loss, "eval/accuracy": eval_accuracy, "eval/has_converged": has_converged})

    logging.info(f"Eval loss {eval_loss:5.3f}, eval accuracy {eval_accuracy:5.3f}, has converged: {has_converged}")

    return model, has_converged
