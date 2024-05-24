import torch

from src.model import TorchModel, TorchModelWithScratchpad


def get_output_loss_acc(model, x, y, criterion):
    if isinstance(model, TorchModel):
        output = model(x)
    elif isinstance(model, TorchModelWithScratchpad):
        # add the start of sequence token
        y_prefixed = torch.cat([torch.ones((y.size(0), 1), device=y.device, dtype=int) * 2, y], dim=1)
        output = model(x, y_prefixed)[:, :-1]
    else:
        raise ValueError(f"Unknown model type {type(model)}")

    loss = criterion(output, y.float())

    threshold = (get_output_loss_acc.negative_value + 1) / 2

    if get_output_loss_acc.negative_value == -1:
        batch_accuracy = ((output > threshold) * 2 - 1 == y).float().mean()
    else:
        batch_accuracy = ((output > threshold) == y).float().mean()

    return output, loss, batch_accuracy