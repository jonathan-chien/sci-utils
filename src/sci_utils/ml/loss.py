from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .environment import BatchEnvironment


class LossFn(Protocol):
    def __call__(self, batch_env: BatchEnvironment, params: dict[str, torch.nn.Parameter] | None) -> torch.Tensor: ...


@dataclass
class LinearCombinationLoss:
    loss_terms: Sequence[LossFn]
    weights: Sequence[float]

    def __post_init__(self):
        if len(self.loss_terms) != len(self.weights):
            raise ValueError(
                "`loss_terms` and `weights` must be sequences of the same "
                f"length, but got lengths {len(self.loss_terms)} and {len(self.weights)}, respectively."
            )

    def __call__(self, batch_env: BatchEnvironment, params: dict[str, torch.nn.Parameter]):
        loss = 0
        for weight, term in zip(self.weights, self.loss_terms):
            loss += weight * term(batch_env=batch_env, params=params)
        return loss


def mse_loss(batch_env: BatchEnvironment, params: dict[str, torch.nn.Parameter] | None = None):
    # XXX: Forgetting to reshape tensors to match is a dangerous failure mode
    # because it's silent, causing no excpetions to be raised but preventing
    # learning from happening. Could deal with this by creating a base LossFn
    # class instead of a Protocol or by reshaping in evaluate.
    return torch.mean((batch_env.y_hat.reshape_as(batch_env.y) - batch_env.y)**2)
