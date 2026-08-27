from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class BackwardRule(Protocol):
    def __call__(self, loss: torch.Tensor, params: dict[str, torch.nn.Parameter]) -> None: ...


@dataclass
class BackwardI:
    zero_grad: bool = True

    def __call__(self, loss: torch.Tensor, params: dict[str, torch.nn.Parameter]):
        if self.zero_grad:
            for p in params.values():
                p.grad = None
        loss.backward()

        return None
        

class UpdateRule(Protocol):
    def __call__(self, params: dict[str, torch.nn.Parameter]) -> None: ...


@dataclass
class GradientSystemI:
    lr: float = 1e-3
    
    def __call__(self, params: dict[str, torch.nn.Parameter]):
        with torch.no_grad():
            for p in params.values():
                if p.grad is not None:
                    p -= self.lr * p.grad
                    