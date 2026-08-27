import copy
import json
from pathlib import Path
import torch
import warnings

from .. import recursion as recursion

class Logger:
    """ 
    Utility class for logging per batch/epoch results during training or 
    testing. Can log an arbitrary number of items per batch/epoch.

    TODO: Add method to concatenate across epochs/batches, supporting dotted paths for nested dictionaries
    """
    def __init__(
            self, 
            log_name: str, 
            log_dir: str | None = None, # Added default None sentinel 2026/07/15
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
        # 2025/08/16: Calling this method used to create new entry; changing
        # this to create new entry if none exists and update existing one if
        # one does. Also on 2025/08/16, changing to only detaching tensor input
        # (no .item(), which assumes that input is scalar) and deferring cpu
        # transfer to every n log inputs.
        entry = {
            key : val.detach() if isinstance(val, torch.Tensor) else val 
            for key, val in kwargs.items()
        }
        # 2026/08/21 add batch_size as arg.
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
        TODO: Use recursive conversion to cpu/detach/item to support logging of nested dictionaries
        x.cpu().item() if isinstance(x, torch.Tensor) and x.numel() == 1 else x
        """
        # 2025/08/16: Calling this method used to create new entry; changing
        # this to create new entry if none exists and update existing one if
        # one does. Also on 2025/08/16, changing to only detaching tensor input
        # (no item(), which assumes that input is scalar) and deferring cpu
        # transfer to every n log inputs.
        entry = {
            key : val.detach() if isinstance(val, torch.Tensor) else val 
            for key, val in kwargs.items()
        } # This was changed in the 2026 notebook version as well to not require tensor_utils
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

    def to_cpu(self):
        # Use recursion module to move all tensors in logger to cpu.
        pass

    def get_entry(self, level: str, epoch_idx: int, batch_idx: int | None = None):
        """ 
        """
        if level == 'batch':
            if batch_idx is None:
                raise ValueError(
                    f"`level` was passed in as 'batch', but no `batch_idx` was specified."
                )
            try:
                return copy.deepcopy(self.batch_logs[(epoch_idx, batch_idx)])
            except KeyError:
                raise IndexError(
                    f"No entry found for epoch {epoch_idx}, batch {batch_idx}."
                )
        elif level == 'epoch':
            try:
                return copy.deepcopy(self.epoch_logs[epoch_idx])
            except KeyError:
                raise IndexError(
                    f"No entry found for epoch {epoch_idx}."
                )
        else:
            raise ValueError(
                f"Unrecognized value {level} for `level`. Must be in ['batch', 'epoch']."
            )
        
    def get_all_entries(self, key: str, level: str, epoch_idx: int | None = None, return_tensor: bool = False):
        """ 
        On 2026/08/03, made conversion to tensor optional (default False).
        """
        if level not in ['batch', 'epoch']:
            raise ValueError(
                f"Unrecognized value {level} for `level`. Must be 'batch' or 'epoch'."
            )
        
        source = self.epoch_logs if level == 'epoch' else self.batch_logs

        # If epoch index was provided for level='batch', retrieve only values 
        # from that epoch, else return all batches from all epochs.
        if level == 'batch' and epoch_idx is not None:
            source = {
                (i_epoch, i_batch): entry
                for (i_epoch, i_batch), entry in source.items()
                if i_epoch == epoch_idx
            }

        try:
            # TODO: Should probably add dotted path traversal here.
            values = [entry[key] for entry in source.values()]
        except KeyError:
            raise KeyError(
                f"The key '{key}' is missing from one or more {level} entries."
            )
        
        return torch.tensor(values) if return_tensor else values

    def compute_weighted_sum(self, key: str, level: str, weights: torch.Tensor, epoch_idx: int | None = None):
        """ 
        """
        values = self.get_all_entries(key=key, level=level, epoch_idx=epoch_idx, return_tensor=True)
        # tensor_utils.validate_tensor(weights, 1) # TODO: add this back when integrating this into standalone utils package
        return torch.sum(values * weights)

    def reduce_batches(self, epoch_idx: int, reduce_batches_for: list[str]):
        batch_sizes = self.get_all_entries(key='batch_size', level='batch', epoch_idx=epoch_idx, return_tensor=True)
        total_num_obs = torch.sum(batch_sizes)
        weights = batch_sizes / total_num_obs
        mean_values = {
            item_name: self.compute_weighted_sum(
                key=item_name, 
                level='batch', 
                weights=weights, 
                epoch_idx=epoch_idx
            )
            for item_name in reduce_batches_for
        }
        # 2026/08/17: this assumes that calling this method when an entry already exists for that epoch will add to the existing entry rather than overwrite it.
        self.log_epoch(epoch_idx=epoch_idx, **mean_values)

    
    def save(self, log_dir: str | None = None): # Added log_dir arg here 2026/07/15
        """ 
        """
        # TODO: User is currently responsible for calling serialization method
        # before attempting to save. Could add try except block for more explicit exception handling.

        # Added 2026/07/15.
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

        
# class Logger:
#     """ 
#     Utility class for logging per batch/epoch results during training or 
#     testing. Can log an arbitrary number of items per batch/epoch.
#     """
#     def __init__(
#             self, 
#             log_dir: str, 
#             log_name: str, 
#             verbose_batch: bool = False,
#             verbose_epoch: bool = True,
#             print_flush_epoch: bool = False, 
#             print_flush_batch: bool = False
#     ):
#         self.log_dir = Path(log_dir)
#         self.log_dir.mkdir(parents=True, exist_ok=True)
#         self.log_name = log_name
#         self.batch_logs = {}
#         self.epoch_logs = {}
#         self.verbose_batch = verbose_batch
#         self.verbose_epoch = verbose_epoch
#         self.print_flush_epoch = print_flush_epoch
#         self.print_flush_batch = print_flush_batch

#     def log_batch(self, *, epoch_idx: int, batch_idx: int, suppress_print=False, **kwargs):
#         entry = {key : tensor_utils.tensor_to_cpu_python_scalar(val) for key, val in kwargs.items()}
#         self.batch_logs[(epoch_idx, batch_idx)] = entry
#         if self.verbose_batch and not suppress_print:
#             for key, value in entry.items():
#                 print(
#                     f"{self.log_name} {key} for epoch {epoch_idx}, batch {batch_idx}: {value}.", 
#                     flush=self.print_flush_batch
#                 )

#     def log_epoch(self, *, epoch_idx: int, suppress_print=False, **kwargs):
#         entry = {key : tensor_utils.tensor_to_cpu_python_scalar(val) for key, val in kwargs.items()}
#         self.epoch_logs[epoch_idx] = entry
#         if self.verbose_epoch and not suppress_print:
#             for key, value in entry.items():
#                 print(
#                     f"{self.log_name} {key} for epoch {epoch_idx}: {value}.", 
#                     flush=self.print_flush_epoch
#                 )

#     def get_entry(self, level: str, epoch_idx: int, batch_idx: Optional[int] = None):
#         """ 
#         """
#         if level == 'batch':
#             if batch_idx is None:
#                 raise ValueError(
#                     f"`level` was passed in as 'batch', but no `batch_idx` was specified."
#                 )
#             try:
#                 return copy.deepcopy(self.batch_logs[(epoch_idx, batch_idx)])
#             except KeyError:
#                 raise IndexError(
#                     f"No entry found for epoch {epoch_idx}, batch {batch_idx}."
#                 )
#         elif level == 'epoch':
#             try:
#                 return copy.deepcopy(self.epoch_logs[epoch_idx])
#             except KeyError:
#                 raise IndexError(
#                     f"No entry found for epoch {epoch_idx}."
#                 )
#         else:
#             raise ValueError(
#                 f"Unrecognized value {level} for `level`. Must be in ['batch', 'epoch']."
#             )
        
#     def get_all_entries(self, key: str, level: str, epoch_idx: Optional[int] = None):
#         """ 
#         """
#         if level not in ['batch', 'epoch']:
#             raise ValueError(
#                 f"Unrecognized value {level} for `level`. Must be 'batch' or 'epoch'."
#             )
        
#         source = self.epoch_logs if level == 'epoch' else self.batch_logs

#         # If epoch index was provided for level='batch', retrieve only values 
#         # from that epoch, else return all batches from all epochs.
#         if level == 'batch' and epoch_idx is not None:
#             source = {
#                 (i_epoch, i_batch): entry
#                 for (i_epoch, i_batch), entry in source.items()
#                 if i_epoch == epoch_idx
#             }

#         try:
#             values = [entry[key] for entry in source.values()]
#         except KeyError:
#             raise KeyError(
#                 f"The key '{key}' is missing from one or more {level} entries."
#             )
        
#         return torch.tensor(values)
    
#     def compute_weighted_sum(self, key: str, level: str, weights: torch.Tensor, epoch_idx: Optional[int] = None):
#         """ 
#         """
#         values = self.get_all_entries(key=key, level=level, epoch_idx=epoch_idx)
#         tensor_utils.validate_tensor(weights, 1)
#         return torch.sum(values * weights)
    
#     def convert_to_serializable_format(self, target: Union[str, Sequence[str]]):
#         """ 
#         JSON serializable items plus dataclasses can be saved.
#         """
#         def convert(log):
#             # Convert tensors to TensorConfigs, then all dataclasses to tagged dicts.
#             log = tensor_utils.recursive_tensor_to_tensor_config(log)
#             log = serialization_utils.recursive_dataclass_to_tagged_dict(log)

#             return log
        
#         if isinstance(target, str):
#             target = [target]
        
#         for log_name in target:
#             if log_name not in ('batch_logs', 'epoch_logs'):
#                 raise ValueError(
#                     "Unrecognized value for `target`. Must be one of " 
#                     f"['batch_logs', 'epoch_logs'] but got '{log_name}'."
#                 ) 
#             setattr(self, log_name, convert(getattr(self, log_name)))
        

#     def recover_from_serializable_format(self, target: Union[str, Sequence[str]]):
#         """ 
#         """
#         def recover(log):
#             log = serialization_utils.recursive_tagged_dict_to_dataclass(log)
#             log = serialization_utils.recursive_recover(log)
#             return log

#         if isinstance(target, str):
#             target = [target]
        
#         for log_name in target:
#             if log_name not in ('batch_logs', 'epoch_logs'):
#                 raise ValueError(
#                     "Unrecognized value for `target`. Must be one of " 
#                     f"['batch_logs', 'epoch_logs'] but got '{log_name}'."
#                 ) 
#             setattr(self, log_name, recover(getattr(self, log_name)))
            
    
#     def save(self):
#         """ 
#         """
#         # TODO: User is currently responsible for calling serialization method
#         # before attempting to save. Could add try except block for more explicit exception handling.

#         batch_path = self.log_dir / f'{self.log_name}_batch_log.jsonl'
#         epoch_path = self.log_dir / f'{self.log_name}_epoch_log.jsonl'

#         with open(batch_path, 'w', newline='\n') as f:
#             for (epoch_idx, batch_idx), entry in self.batch_logs.items():
#                 entry = {**entry, 'epoch_idx': epoch_idx, 'batch_idx': batch_idx}
#                 f.write(json.dumps(entry, sort_keys=True) + '\n')

#         with open(epoch_path, 'w', newline='\n') as f:
#             for epoch_idx, entry in self.epoch_logs.items():
#                 entry = {**entry, 'epoch_idx': epoch_idx}
#                 f.write(json.dumps(entry, sort_keys=True) + '\n')

#     def reset(self, warn=True):
#         """ 
#         """
#         if warn:
#             warnings.warn(f"Resetting logs for {self.log_name}.")
#         self.batch_logs = {}
#         self.epoch_logs = {}
