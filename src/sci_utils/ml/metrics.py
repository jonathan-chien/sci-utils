from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Protocol, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .environment import BatchEnvironment


class Metric(Protocol):
    def __call__(batch_env: BatchEnvironment) -> object: ... # Possibly -> dict

class Accuracy:
    """ 
    """
    def __self__(self):
        pass

    def __call__(self, batch_env: BatchEnvironment):
        return self.compute_accuracy(batch_env.y_hat, batch_env.y)

    @staticmethod
    def compute_accuracy(y_hat: torch.Tensor, y: torch.Tensor):
        """ 
        Core logic is in static method so that it can be called by other classes as well to aid composition.
        """
        # Replace with validation logic from tensor module.
        if len(y_hat) != len(y):
            raise ValueError(
                f"`y_hat` and `y` must be the same length but got lengths {len(y_hat)} and {len(y)}, respectively."
            )
        num_elem = len(y)
        correct_ind = torch.where(y_hat.reshape_as(y) == y)[0]
        incorrect_ind = torch.arange(
            num_elem,
            device=correct_ind.device
        )[~torch.isin(torch.arange(num_elem, device=correct_ind.device), correct_ind)]
        num_correct = len(correct_ind)
        num_incorrect = len(incorrect_ind)
        accuracy = num_correct / num_elem

        out = {}
        out['num_elem'] = num_elem
        out['correct_ind'] = correct_ind
        out['incorrect_ind'] = incorrect_ind
        out['num_correct'] = num_correct
        out['num_incorrect'] = num_incorrect
        out['value'] = accuracy

        return out


@dataclass
class ThresholdAccuracy:
    """ 
    """
    threshold: float = 0.

    def __call__(self, batch_env: BatchEnvironment):
        """ 
        Cannot be static method because requires threshold value bound to class instance.
        """
        y_hat_thresh = self.apply_threshold(
            batch_env.y_hat, 
            threshold=self.threshold,
            return_dtype=batch_env.y_hat.dtype
        )
        y_thresh = self.apply_threshold(
            batch_env.y, 
            threshold=self.threshold,
            return_dtype=batch_env.y.dtype
        )
        return Accuracy.compute_accuracy(y_hat_thresh, y_thresh)
    
    @staticmethod
    def apply_threshold(x, threshold, return_dtype: torch.dtype | None = None):
            y = x > threshold
            return y if return_dtype is None else y.to(return_dtype)


# DEFAULT_METRICS: Metric = MappingProxyType({'threshold_accuracy': (ThresholdAccuracy(threshold=0.))})
