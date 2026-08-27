from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Protocol, TYPE_CHECKING

from ..nested import shallow_asdict

if TYPE_CHECKING:
    from .logging import Logger
    from .environment import BatchEnvironment
    from .metrics import Metric

    import torch


@dataclass
class CallbackBase:
    every_n_epochs: int

    def _should_run(self, epoch_idx: int):
        return epoch_idx % self.every_n_epochs == 0

    
# ---------------------------- Batch Callbacks ------------------------------ #
class BatchCallback(Protocol):
    """ 
    These callback functions have access to a BatchEnvironment that contains
    dynamic, per batch values, including epoch idx and batch idx.
    """
    def __call__(self, batch_env: BatchEnvironment) -> object | None: ...


@dataclass
class LogBatch(CallbackBase):
    """ 
    Should match BatchCallback Protocol.
    """
    logger: Logger
    items_to_log: list[str]

    def __call__(self, batch_env: BatchEnvironment):
        """ 
        TODO: Will store directories and use path traversal to support dotted paths in items_to_log.
        """
        if self._should_run(batch_env.epoch_idx):
            batch_log = shallow_asdict(batch_env)
            self.logger.log_batch(
                epoch_idx=batch_env.epoch_idx,
                batch_idx=batch_env.batch_idx,
                batch_size=batch_env.batch_size,
                **{key: val for key, val in batch_log.items() if key in self.items_to_log}
            )

        return None


@dataclass
class ComputeMetrics(CallbackBase):
    """ 
    Should match BatchCallback Protocol.
    """
    metrics: dict[str, Metric]

    def __call__(self, batch_env: BatchEnvironment):
        batch_env.metrics = {
            name: metric(batch_env) for name, metric in self.metrics
        }

        return None


# ---------------------------- Epoch Callbacks ------------------------------ #
class EpochCallback(Protocol):
    """ 
    These callback functions should be callable dataclasses accepting as an
    argument only epoch index, which is a dynamically changing variable. In
    general, other objects such as loggers and models should be closed over
    inside the callback dataclass.
    """
    def __call__(self, epoch_idx: int) -> object | None: ...


@dataclass
class ReduceBatches(CallbackBase):
    """ 
    Should match EpochCallback Protocol.
    """
    logger: Logger
    reduce_batches_for: list[str]
    
    def __call__(self, epoch_idx):
        if self._should_run(epoch_idx):
            # batch_sizes = self.logger.get_all_entries(
            #     dotted_path='batch_size', 
            #     level='batch', 
            #     epoch_idx=epoch_idx
            # )
            # mean_values = {
            #     dotted_path: self.logger.compute_weighted_sum(dotted_path=dotted_path, weights=batch_sizes)
            #     for dotted_path in self.reduce_batches_for
            # }
            # self.logger.log_epoch(**mean_values)
            self.logger.reduce_batches(
                epoch_idx=epoch_idx, 
                reduce_batches_for=self.reduce_batches_for
            )

        return None


@dataclass
class LogParams(CallbackBase):
    """ 
    Should match EpochCallback Protocol.
    """
    model: torch.nn.Module
    logger: Logger

    def __call__(self, epoch_idx):
        if self._should_run(epoch_idx):
            params = dict(self.model.named_parameters())
            params = {name: p.detach().clone() for name, p in params.items()}
            self.logger.log_epoch(epoch_idx=epoch_idx, params=params)


@dataclass
class EarlyStoppingCallback(CallbackBase):
    """ 
    Should match EpochCallback Protocol.
    """
    pass


@dataclass
class CheckpointMonitorCallback(CallbackBase):
    """ 
    Should match EpochCallback Protocol.
    """
    pass
