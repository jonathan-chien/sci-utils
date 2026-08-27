from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class BatchEnvironment:
    """ 
    Environmental variables during forward pass. This has the primary purpose
    of capturing a broad range of variables that a loss function and
    performance function might want in a central documented place, preventing
    the call signatures of these functions from becoming heterogeneous (if
    variable between functions) or long (if all items are passed manually), and
    unstable in either case. A secondary function is that a logger object can
    thus filter the fields of this BatchEnvironment object using a user provided
    list and then log directly from this BatchEnvironment object.
    """
    epoch_idx: int
    batch_idx: int
    batch_size: int
    x: torch.Tensor | None
    y: torch.Tensor | None
    y_hat: torch.Tensor | None = None
    activations: dict[str, torch.Tensor] | None = None
    loss: torch.Tensor | None = None
    metrics: dict[str, object] = None
    