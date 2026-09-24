import copy
import json
from pathlib import Path
import torch
import warnings

from .. import recursion
from .. import tensor as tensor_
from ..nested import traverse_dotted_path


class Logger:
    """ 
    Utility class for logging per batch/epoch results during training or 
    testing. Can log an arbitrary number of items per batch/epoch.
    """
    def __init__(
            self, 
            log_name: str, 
            log_dir: str | None = None,
            verbose_batch: bool = False,
            verbose_epoch: bool = True,
            print_flush_epoch: bool = False, 
            print_flush_batch: bool = False
    ):
        self.log_name = log_name
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if self.log_dir is not None: 
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose_batch = verbose_batch
        self.verbose_epoch = verbose_epoch
        self.print_flush_epoch = print_flush_epoch
        self.print_flush_batch = print_flush_batch
        
        self.batch_logs = {}
        self.epoch_logs = {}

    def log_batch(self, *, epoch_idx: int, batch_idx: int, batch_size: int | None = None, suppress_print=False, **kwargs):
        """ 
        Add to existing batch entry or create new entry if none exists.
        """
        entry = {
            key : val.detach() if isinstance(val, torch.Tensor) else val 
            for key, val in kwargs.items()
        }
        
        if batch_size is not None:
            entry.update({'batch_size': batch_size})
            
        if (epoch_idx, batch_idx) in self.batch_logs:
            self.batch_logs[(epoch_idx, batch_idx)].update(entry)
        else:
            self.batch_logs[(epoch_idx, batch_idx)] = entry
        if self.verbose_batch and not suppress_print:
            for key, value in entry.items():
                print(
                    f"{self.log_name} {key} for epoch {epoch_idx}, batch {batch_idx}: {value}.", 
                    flush=self.print_flush_batch
                )

    def log_epoch(self, *, epoch_idx: int, suppress_print=False, **kwargs):
        """ 
        Add to existing epoch entry or create entry if none exists.
        """
        entry = {
            key : val.detach() if isinstance(val, torch.Tensor) else val 
            for key, val in kwargs.items()
        } 
        if epoch_idx in self.epoch_logs:
            self.epoch_logs[epoch_idx].update(entry)
        else:
            self.epoch_logs[epoch_idx] = entry
        if self.verbose_epoch and not suppress_print:
            for key, value in entry.items():
                print(
                    f"{self.log_name} {key} for epoch {epoch_idx}: {value}.", 
                    flush=self.print_flush_epoch
                )

    def to_cpu(self, level: str):
        """ 
        All tensors should have been detached at the time they were logged.
        Here, they can be periodically moved to the CPU.
        """
        if level == 'batch':
            target = self.batch_logs
        elif level == 'epoch':
            target = self.epoch_logs
        else:
            raise ValueError(
                f"Unrecognized value {level} for `level`. Must be in ('batch', 'epoch')."
            )
        recursion.recursive(
            target,
            branch_conditionals=(
                recursion.dataclass_branch,
                recursion.dict_branch,
                recursion.list_branch,
                recursion.tuple_branch
            ),
            leaf_fns=(
                tensor_.move_to_device('cpu'),
            )
        )

    def _get_source(self, level: str):
        """
        Internal method to dispatch to either epoch_logs or batch logs in
        retrieval methods.
        """
        if level not in ['batch', 'epoch']:
            raise ValueError(
                f"Unrecognized value {level} for `level`. Must be 'batch' or 'epoch'."
            )
                        
        return self.epoch_logs if level == 'epoch' else self.batch_logs

    def get_entry(self, dotted_path: str, level: str, epoch_idx: int, batch_idx: int | None = None):
        """ 
        Retrieve arbitrarily nested value from log for specific epoch or from
        log for specific epoch and batch.
        """
        source = self._get_source(level=level)
        if level == 'batch' and batch_idx is None:
            raise ValueError(
                f"`level` was passed in as 'batch', but no `batch_idx` was specified."
            )
        log = source[(epoch_idx, batch_idx)] if level == 'batch' else source[epoch_idx]
        return traverse_dotted_path(log, dotted_path)
        # if level == 'batch':
        #     if batch_idx is None:
        #         raise ValueError(
        #             f"`level` was passed in as 'batch', but no `batch_idx` was specified."
        #         )
        #     try:
        #         return copy.deepcopy(self.batch_logs[(epoch_idx, batch_idx)])
        #     except KeyError:
        #         raise IndexError(
        #             f"No entry found for epoch {epoch_idx}, batch {batch_idx}."
        #         )
        # elif level == 'epoch':
        #     try:
        #         return copy.deepcopy(self.epoch_logs[epoch_idx])
        #     except KeyError:
        #         raise IndexError(
        #             f"No entry found for epoch {epoch_idx}."
        #         )
        # else:
        #     raise ValueError(
        #         f"Unrecognized value {level} for `level`. Must be in ['batch', 'epoch']."
        #     )

    def get_all_entries(self, dotted_path: str, level: str, epoch_idx: int | None = None, return_tensor: bool = False):
        """ 
        Retrieve arbitrarily nested values and concatenate across epochs or
        across batches, either within a single epoch or across all epochs.
        """
        # if level not in ['batch', 'epoch']:
        #     raise ValueError(
        #         f"Unrecognized value {level} for `level`. Must be 'batch' or 'epoch'."
        #     )
        
        # source = self.epoch_logs if level == 'epoch' else self.batch_logs
        source = self._get_source(level=level)

        # If epoch index was provided for level='batch', retrieve only values 
        # from that epoch, else return all batches from all epochs.
        if level == 'batch' and epoch_idx is not None:
            source = {
                (i_epoch, i_batch): entry
                for (i_epoch, i_batch), entry in source.items()
                if i_epoch == epoch_idx
            }

        # 2026/08/31: Before adding dotted path traversal, this step just
        # required retrieving a value from the source dict with a key. Now that
        # more complicated path traversal logic is handled by
        # traverse_dotted_path, it may be more informative to just allow that
        # function's exceptions to be passed through.
        values = [traverse_dotted_path(entry, dotted_path) for entry in source.values()]
        # try:
        #     values = [traverse_dotted_path(entry, dotted_path) for entry in source.values()]
        # except KeyError:
        #     raise KeyError(
        #         f"The key '{dotted_path}' is missing from one or more {level} entries."
        #     )
        
        return torch.tensor(values) if return_tensor else values

    def compute_weighted_sum(self, dotted_path: str, level: str, weights: torch.Tensor, epoch_idx: int | None = None):
        """ 
        """
        values = self.get_all_entries(dotted_path=dotted_path, level=level, epoch_idx=epoch_idx, return_tensor=True)
        tensor_.validate_tensor(weights, 1)
        return torch.sum(values * weights)

    def reduce_batches(self, epoch_idx: int, reduce_batches_for: list[str]):
        batch_sizes = self.get_all_entries(dotted_path='batch_size', level='batch', epoch_idx=epoch_idx, return_tensor=True)
        total_num_obs = torch.sum(batch_sizes)
        weights = batch_sizes / total_num_obs
        mean_values = {
            dotted_path.replace('.', ':'): self.compute_weighted_sum( # Need to replace . with : otherwise what should be a key will be treated as a dotted path
                dotted_path=dotted_path, 
                level='batch', 
                weights=weights, 
                epoch_idx=epoch_idx
            )
            for dotted_path in reduce_batches_for
        }
        self.log_epoch(epoch_idx=epoch_idx, **mean_values)

    def save(self, log_dir: str | None = None):
        """ 
        """
        # TODO: User is currently responsible for calling serialization method
        # before attempting to save. Could add try except block for more
        # explicit exception handling.

        if log_dir is not None:
            log_dir_to_use = log_dir
        elif self.log_dir is not None:
            log_dir_to_use = self.log_dir
        else:
            raise RuntimeError(
                "Attempting to save logged values but log_dir was passed in as "
                "None at the time of object instantiation, and no log_dir was "
                "provided as an argument to the `save` method."
            )

        batch_path = log_dir_to_use / f'{self.log_name}_batch_log.jsonl'
        epoch_path = log_dir_to_use / f'{self.log_name}_epoch_log.jsonl'

        with open(batch_path, 'w', newline='\n') as f:
            for (epoch_idx, batch_idx), entry in self.batch_logs.items():
                entry = {**entry, 'epoch_idx': epoch_idx, 'batch_idx': batch_idx}
                f.write(json.dumps(entry, sort_keys=True) + '\n')

        with open(epoch_path, 'w', newline='\n') as f:
            for epoch_idx, entry in self.epoch_logs.items():
                entry = {**entry, 'epoch_idx': epoch_idx}
                f.write(json.dumps(entry, sort_keys=True) + '\n')

    def reset(self, warn=True):
        """ 
        """
        if warn:
            warnings.warn(
                f"Resetting logs for {self.log_name}. Any current content will be overwritten."
            )
        self.batch_logs = {}
        self.epoch_logs = {}
