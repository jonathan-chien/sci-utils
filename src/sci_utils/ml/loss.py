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
    return torch.mean((batch_env.y_hat - batch_env.y)**2)
