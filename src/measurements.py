import torch
import logging

from torch.nn.utils.stateless import functional_call

from typing import Dict, Tuple, Union
from collections import defaultdict

from src.model import TorchModel, TorchModelWithScratchpad
from src.targets import Target
from src.utils import set_random_seed, compute_loss
from src.utils2 import get_output_loss_acc


HOOK_BUFFER = {}


def get_input(name):
    def hook(model, input, output):
        HOOK_BUFFER[name] = input[0].detach()
    return hook


# the hooks on the Transformer LayerNorm for tracking LN blowup
def register_hooks(model: Union[TorchModel, TorchModelWithScratchpad]):
    if isinstance(model, TorchModel):
        return [
            model.encoders.layers[idx].norm1.register_forward_hook(get_input(f'encoders.layers.{idx}.norm1'))
            for idx in range(len(model.encoders.layers))
        ] + [
            model.encoders.layers[idx].norm2.register_forward_hook(get_input(f'encoders.layers.{idx}.norm2'))
            for idx in range(len(model.encoders.layers))
        ]
    elif isinstance(model, TorchModelWithScratchpad):
        return [
            model.model.encoder.layers[idx].norm1.register_forward_hook(get_input(f'model.encoder.layers.{idx}.norm1'))
            for idx in range(len(model.model.encoder.layers))
        ] + [
            model.model.encoder.layers[idx].norm2.register_forward_hook(get_input(f'model.encoder.layers.{idx}.norm2'))
            for idx in range(len(model.model.encoder.layers))
        ] + [
            model.model.decoder.layers[idx].norm1.register_forward_hook(get_input(f'model.decoder.layers.{idx}.norm1'))
            for idx in range(len(model.model.decoder.layers))
        ] + [
            model.model.decoder.layers[idx].norm2.register_forward_hook(get_input(f'model.decoder.layers.{idx}.norm2'))
            for idx in range(len(model.model.decoder.layers))
        ]


def delete_hooks(hooks):
    for hook in hooks:
        hook.remove()


@torch.inference_mode()
def hessian_norm_proxy(x: torch.Tensor, y: torch.Tensor, model: TorchModel, 
        criterion: torch.nn.modules.loss._Loss, epsilon: float = 1e-3, n_repeats: int = 10, random_state: int = 0,
        rng=None, rho: float = None) -> torch.Tensor:
    """This function takes a batch and measures sharpness of loss landscape on this batch

        epsilon (float, optional): standard deviation of noise. Defaults to 1e-3.
        n_repeats (int, optional): number of noise additions. Defaults to 10.
        rng (_type_, optional): torch random number generator. Defaults to None.
        rho (float, optional): epsilon * sqrt(number of parameters). Is needed for consistently adding noise in experiments with varying number of parameteres. Defaults to None.
    """

    if epsilon is not None and rho is not None:
        raise ValueError("You can't specify both epsilon and rho")
    
    if rho is not None:
        n_params = sum([p.numel() for n, p in model.named_parameters() if "positional" not in n])
        epsilon = rho / (n_params ** 0.5)
    
    if rng is None:
        rng = torch.Generator(device=x.device)

    _, loss, _ = get_output_loss_acc(model, x, y, criterion)

    avg_diff = - loss.item()

    from tqdm import trange
    for _ in range(n_repeats):

        noised_params = {
            name: param + torch.randn(param.size(), generator=rng, device=param.device) * epsilon
            for name, param in model.state_dict().items()
            if "positional_encoding" not in name
        }

        if isinstance(model, TorchModel):
            output = functional_call(model, noised_params, x)
        elif isinstance(model, TorchModelWithScratchpad):
            output = functional_call(
                model, noised_params, 
                (x, torch.cat([torch.ones((y.size(0), 1), device=y.device, dtype=int) * 2, y], dim=1))
            )[:, :-1]
        
        avg_diff += compute_loss(criterion, output, y).item() / n_repeats

    return avg_diff


def squared_param_norm(model: torch.nn.Module) -> float:
    return sum([
        param.norm().item() ** 2
        for name, param in model.named_parameters()
        if "positional_encoding" not in name
    ])


# takes one batch and inputs and returns metrics
def inspect_one_batch(
        x: torch.Tensor, y: torch.Tensor, model: TorchModel, 
        criterion: torch.nn.modules.loss._Loss,
        target: Target, hessian_norm_proxy_args: Dict[str, Union[int, float]]={}
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

    output, loss, accuracy = get_output_loss_acc(model, x, y, criterion)

    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
    }

    n_layers = len(model.encoders.layers) if isinstance(model, TorchModel) else len(model.model.encoder.layers)

    prods = defaultdict(lambda: torch.ones(x.size(0), device=x.device))
    sums = defaultdict(lambda: torch.zeros(x.size(0), device=x.device))

    # measuring LN blowup
    for layer_idx in range(n_layers):
        for norm in [1, 2]:
            if isinstance(model, TorchModel):
                hook_name = f'!s.layers.{layer_idx}.norm{norm}'
                parts = ["encoder"]
            elif isinstance(model, TorchModelWithScratchpad):
                hook_name = f'model.!.layers.{layer_idx}.norm{norm}'
                parts = ["encoder", "decoder"]
            
            for part in parts:
                token_input = HOOK_BUFFER[hook_name.replace("!", part)]
                inv_std = (1 / token_input.std(dim=2))
                if layer_idx == n_layers - 1:
                    inv_std = inv_std[:, -1:]

                metrics[f'inv_std_{part}_{layer_idx}_ln{norm}_mean'] = inv_std.mean().item()
                metrics[f'inv_std_{part}_{layer_idx}_ln{norm}_max'] = inv_std.max(dim=1).values.mean().item()

                prods[f"blowup_{part}_means_prod"] *= inv_std.mean(dim=1)
                prods[f"blowup_{part}_maxs_prod"] *= inv_std.max(dim=1).values

                sums[f"blowup_{part}_means_sum"] += inv_std.mean(dim=1)
                sums[f"blowup_{part}_maxs_sum"] += inv_std.max(dim=1).values

    for agg in [prods, sums]:
        for key, value in agg.items():
            metrics[key] = value.mean().item()

    # sharpness
    metrics["hessian_norm_proxy"] = hessian_norm_proxy(x, y, model, criterion, **hessian_norm_proxy_args)

    # accuracy of a model with scratchpad
    if isinstance(model, TorchModelWithScratchpad):
        y_pred = model.autoregressive_answer(x, verbose=False)
        metrics["autoregressive_accuracy"] = (y == y_pred).float().mean().item()

    return metrics, None
