from abc import ABC, abstractmethod
from pathlib import Path
import warnings

import torch

from .. import validation as validation_utils


class StopTraining(Exception):
    pass


class EarlyStopping(ABC):
    """ 
    """
    def __init__(
        self, 
        metric_name: str, 
        warmup: int = 0, 
        verbose: bool = True, 
        disabled: bool = False
    ):
        validation_utils.validate_str(metric_name)
        validation_utils.validate_nonneg_int(warmup)
        self.metric_name = metric_name
        self.warmup = warmup
        self.verbose = verbose
        self.disabled = disabled

        self.condition_reached_at_update = None
        self.num_updates = 0

    def should_stop(self, x: float) -> bool:
        """ 
        Accepts new value and returns boolean indicating whether stopping 
        condition has been reached.
        """
        validation_utils.validate_float(x)

        # Always update, even if disabled, so any best value tracking
        # implemented by child classes can still be carried out.
        self._update_rule(x)
        self.num_updates += 1

        if self.disabled:
            return False
        
        should_stop = self._stopping_rule()

        if should_stop and self.condition_reached_at_update is None:
            self.condition_reached_at_update = self.num_updates
        
        return should_stop
    
    def reset(self):
        """ 
        Child classes should implement their own reset() methods that call
        super().reset() and then clear any other internal state attributes
        based on their respective policies.
        """
        self.condition_reached_at_update = None
        self.num_updates = 0
          
    # ---------------Abstract methods defining stopping policy--------------- #
    @abstractmethod
    def _update_rule(self, x: float):
        """ 
        Takes in new value for tracked quantity and updates state so that 
        object is ready for call to self._stopping_rule().
        """
        pass

    @abstractmethod
    def _stopping_rule(self):
        """ 
        Boolean function, returns True if stopping condition reached, else False.
        """
        pass

    @abstractmethod
    def print_to_console(self, update_name='update'):
        """ 
        Deferred to allow child class to print more detailed stopping info 
        based on its policy.
        """
        pass

    
class NoImprovementStopping(EarlyStopping):
    """ 
    """
    def __init__(
        self,
        patience: int,
        mode : str,
        tol: float = 1e-4,
        **kwargs # common attributes in base class passed in here
    ):
        super().__init__(**kwargs)

        validation_utils.validate_pos_int(patience)
        validation_utils.validate_nonneg_float(tol)
        self.patience = patience
        self.mode = mode
        self.tol = tol
        if self.mode not in ['min', 'max']:
            raise ValueError(
                f"Unrecognized value {self.mode} for mode. Should be 'min' or 'max'."
            )

        self.best_value = None
        self.counter = 0
        self.recent_vals = torch.full((self.patience + 1,), torch.nan)

    def _is_improvement(self, x):
        """
        Check whether new value is improvement on previous best value.
        """
        diff = x - self.best_value
        # Only count as improvement if better than best by at least tol.
        if self.mode == 'min':
            return (diff + self.tol < 0, diff)
        else: # A check for correct values is perfomed in constructor
            return (diff - self.tol > 0, diff)

    def _update_rule(self, x: float):
        """ 
        If no improvement after warmup period, increment counter.
        """
        if self.best_value is None: # First update
            self.best_value = x
        else:
            warming_up = self.num_updates < self.warmup
            improved, diff = self._is_improvement(x)
            if improved:
                self.best_value = x

                if warming_up:
                    # Counter should be at 0 during warmup, just print diff.
                    if self.verbose:
                        print(f"Warming up. Target improved by {diff}.")
                else:
                    # After warmup period, may need to reset counter.
                    self.counter = 0
                    if self.verbose:
                        print(f"Target improved by {diff}. Setting early stopping counter to 0.")
            elif not warming_up: # Not improved and not warming up
                self.counter += 1
                if self.verbose:
                    print(f"Early stopping counter: {self.counter}/{self.patience}.")

        # Tracking of recent vals is not necessary for stopping logic but useful to print.
        self.recent_vals = self.recent_vals.roll(-1)
        self.recent_vals[-1] = x

    def _stopping_rule(self):
        """ 
        """
        if self.counter > self.patience:
            raise RuntimeError(
                f"Unexpected runtime condition, where self.counter ({self.counter}) "
                f"exceeds self.patience ({self.patience})."
            )
        return self.counter >= self.patience
    
    def print_to_console(self, update_name='update'):
        """ 
        """
        print(
            f"Early stopping condition reached after {update_name} {self.condition_reached_at_update} with tol = {self.tol}.\n"
            f"Tracked value over last {self.patience+1} {update_name}s: {self.recent_vals}.\n"
            f"Final changes prior to early stopping: {torch.diff(self.recent_vals, n=1)}."
        )

    def reset(self):
        """ 
        """
        super().reset()
        self.best_value = None
        self.counter = 0
        self.recent_vals.fill_(torch.nan)
    
        
class CheckpointMonitor:
    """ 
    """
    def __init__(self, metric_name, checkpoint_dir, mode: str, frequency: str ='best'):
        """ 
        mode : string
            ['min' | 'max']. 
        """
        if not mode in ['min', 'max']:
            raise ValueError(
                f"Unrecognized value {mode} for `mode`. Must be one of ['min', 'max']."
            )
        if not (
            frequency in ['best', 'always'] 
            or (isinstance(frequency, int) and frequency > 0)
        ):
            raise ValueError(
                f"Unrecognized value {frequency} for `frequency`. Must be "
                "either in ['best', 'always'], or a positive int."
            )
        
        self.metric_name = metric_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency
        self.mode = mode
        self.prev_best_value = None
        self.prev_best_epoch = None
        self.best_value = float('inf') if mode == 'min' else -float('inf') 
        self.best_epoch = None

    def get_checkpoint_path(self, epoch_idx, is_best=False, ext='.pt'):
        suffix = '_best' if is_best else ''
        return self.checkpoint_dir / f'{epoch_idx}{suffix}{ext}'

    def _update_if_is_best(self, value, epoch_idx):
        value = float(value)
        is_best = (
            value < self.best_value if self.mode == 'min' 
            else value > self.best_value 
        )
        if is_best:
            # Retain previous best epoch index for reference in `save` method.
            self.prev_best_value = self.best_value
            self.prev_best_epoch = self.best_epoch
            self.best_value = value
            self.best_epoch = epoch_idx
        return is_best
    
    def should_save(self, value, epoch_idx: int) -> bool:
        """ 
        Compare new value against current stored best value; if better,
        register this in the object's internal state. Then returns boolean
        2-tuple (a, b), where a is True iff saving condition satisfied, and b
        is True iff new value is improvement over current stored best value.
        """
        # Always register value and epoch index internally if is best so far.
        is_best = self._update_if_is_best(value, epoch_idx)

        # Return True if should save, else False.
        if self.frequency == 'always': 
            return (True, is_best)
        elif isinstance(self.frequency, int):
            return (epoch_idx % self.frequency == 0 or is_best, is_best)
        elif self.frequency == 'best':
            return (is_best, is_best)
        else:
            raise ValueError(
                f"Unrecognized value for `frequency`: {self.frequency}."
            )
        
    def save(self, checkpoint, epoch_idx, is_best=False):
        """ 
        Parameters
        ----------
        checkpoint : serializable object
        epoch_idx : int
        """
        # If current model is best, need to handle previously saved best models.
        if is_best:
            # Get filename of previous best model, if it exists.
            prev_best_path = (
                self.get_checkpoint_path(self.prev_best_epoch, is_best=True)
                if self.prev_best_epoch is not None else None
            )

            # Either overwrite previous best file or rename.
            if prev_best_path is not None:
                if (
                    self.frequency == 'best'
                    or (isinstance(self.frequency, int) and epoch_idx % self.frequency != 0)
                ):
                    prev_best_path.unlink() # Delete previous best file (overwrite)
                elif (
                    self.frequency == 'always'
                    or (isinstance(self.frequency, int) and epoch_idx % self.frequency == 0)
                ):
                    # Prepare new filename with '_best' suffix stripped.
                    plain_path = self.get_checkpoint_path(
                        self.prev_best_epoch, is_best=False
                    )
                    if plain_path.exists():
                        warnings.warn(
                            f"Renaming {prev_best_path} to {plain_path}, but "
                            f"{plain_path} already exists and will be overwritten!"
                        )
                    prev_best_path.rename(plain_path)
                else:
                    warnings.warn(
                        "Current model is best, and a previous best model has " 
                        f"been saved. However, self.frequency = {self.frequency} "
                        f"and epoch_idx = {epoch_idx} yielded an unrecognized "
                        "condition. The current best model will still be saved, "
                        "but there may be more than one file with suffix "
                        "'_best', which is not intended."
                    )
                
        torch.save(checkpoint, self.get_checkpoint_path(epoch_idx, is_best))

    def save_if_should_save(self, value, epoch_idx, checkpoint):
        """ 
        Convenience method to call `should_save` and then `save`, which could
        also be called manually in sequence.
        """
        should_save, is_best = self.should_save(value, epoch_idx)
        if should_save:
            self.save(checkpoint, epoch_idx, is_best=is_best)
            