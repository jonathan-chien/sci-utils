from __future__ import annotations
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loss import LossFn
    from .environment import BatchEnvironment

    import torch


@contextmanager
def temporarily_eval(model: torch.nn.Module):
    """ 
    Temporarily place model in eval mode for evaluation on validation set during training.
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()

def evaluate(
    model: torch.nn.Module, 
    batch_env: BatchEnvironment,
    loss_fn: LossFn, 
):
    batch_env.y_hat, batch_env.activations = model(batch_env.x)
    batch_env.loss = loss_fn(batch_env, dict(model.named_parameters()))

    return batch_env