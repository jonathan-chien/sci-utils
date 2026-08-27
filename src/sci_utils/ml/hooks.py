from dataclasses import dataclass
from typing import Callable
import warnings


@dataclass
class DynamicHookConfig:
    # ArgsConfig
    module_name: str
    hook_name: str
    hook_kind: str
    capture_fn: Callable
    bhv_if_already_recorded: str


class DynamicHook:
    def __init__(
        self, 
        module_name, 
        hook_kind, 
        hook_name, 
        capture_fn, 
        capture_every_n_epochs: int = 1,
        bhv_if_already_recorded: str = 'print'
    ):
        self.module_name = module_name
        self.hook_name = hook_name
        self.hook_kind = hook_kind
        self.capture_fn = capture_fn 
        self.capture_every_n_epochs = capture_every_n_epochs
        self.bhv_if_already_recorded = bhv_if_already_recorded

        # State attributes.
        self.handle = None
        self.is_registered = False
        self.current_epoch_idx = None
        self.current_batch_idx = None
        self.data = {}
        
    def already_recorded(self, epoch_idx, batch_idx):
        message = (
            f"Attempting to record data for epoch {epoch_idx}, batch {batch_idx} "
            f"but data for this epoch and batch has already been recorded."
        )
        if self.bhv_if_already_recorded == 'silent':
            pass
        elif self.bhv_if_already_recorded == 'print':
            print(message)
        elif self.bhv_if_already_recorded == 'warn':
            warnings.warn(message)
        elif self.bhv_if_already_recorded == 'raise':
            raise RuntimeError(message)
        else:
            raise ValueError(
                f"Invalid value '{self.bhv_if_already_recorded}' for " 
                "self.bhv_if_already_recorded. Should be 'silent', 'print', 'warn', or 'raise'."
            )

    def update(self, epoch_idx: int, batch_idx: int):
        self.current_epoch_idx = epoch_idx
        self.current_batch_idx = batch_idx

    def get_data(self, epoch_idx: int, batch_idx: int):
        # return self.data[epoch_idx][batch_idx]
        return self.data[(epoch_idx, batch_idx)]

    def __call__(self, module, input, output=None):
        # TODO: consider changing names of input and output to arg1 and arg2 respectively (with documentation) since sometimes input will actually be grad_input (and output will be grad_output)
        # TODO: 2026/08/19: Add mechanism to register only every n epochs.
        if not self.is_registered:
            raise RuntimeError(
                "self.is_registered=False, but this hook is firing. This indicates "
                "that the PyTorch hook state is inconsistent with internal registration state."
            )
        if self.current_epoch_idx is None or self.current_batch_idx is None:
            raise RuntimeError(
                f"self.current_epoch_idx={self.current_epoch_idx} and "
                f"self.current_batch_idx={self.current_batch_idx}. The hook "
                "may have fired before epoch and batch indices were set."
            )
        epoch_batch_ind = (self.current_epoch_idx, self.current_batch_idx)

        if self.current_epoch_idx % self.capture_every_n_epochs == 0:
            if epoch_batch_ind in self.data:
                self.already_recorded(*epoch_batch_ind)
            self.data[epoch_batch_ind] = self.capture_fn(module, input, output)

        return None # Just to be explicit

    def register(self, module):
        if self.is_registered:
            raise RuntimeError(
                "This dynamic_hook object has already been registered "
                f"(module_name: {self.module_name}; hook_name: {self.hook_name}; hook_kind: {self.hook_kind})."
            )
        self.handle = getattr(module, f'register_{self.hook_kind}_hook')(self)
        self.is_registered = True
        # return handle

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.is_registered = False


class DynamicHookDict:
    def __init__(self, model, dynamic_hook_list):
        """ 
        self.dynamic_hooks = {
            'submoduleA.subsubmodule1': {
                'hook_name_1': <DynamicHook object>
                }
            }
        }
    
        """
        self.dynamic_hooks = self.register_dynamic_hooks(model, dynamic_hook_list)

    def register_dynamic_hooks(self, model, dynamic_hook_list):
        d = {}
        model_named_module_dict = dict(model.named_modules())

        for dynamic_hook in dynamic_hook_list:
            # Check that dynamic hook's module name matches a module in the model.
            if dynamic_hook.module_name not in model_named_module_dict:
                raise RuntimeError(
                    f"dynamic_hook object has attribute `module_name` = '{dynamic_hook.module_name}' "
                    "but the model contains no such module among its named modules."
                )

            module_name, hook_name = dynamic_hook.module_name, dynamic_hook.hook_name
            
            # There may be multiple hooks associated with the same module name.
            if module_name not in d:
                d[module_name] = {}
            
            # Ensure that a previous DynamicHook object is not overwritten.
            if hook_name in d[module_name]:
                raise RuntimeError(
                    "Attempting to register a DynamicHook object with hook_name " 
                    f"{hook_name} for module_name {module_name}, but a DynamicHook "
                    "object with that hook_name has already been registered under " 
                    "this module. Hook names must be unique across all hooks associated " 
                    "with a single module."
                )
            
            # Retrieve module and register hook.
            module = model_named_module_dict[module_name]
            dynamic_hook.register(module)
            d[module_name][hook_name] = dynamic_hook

        return d

    def update(self, epoch_idx, batch_idx):
        for module_dynamic_hooks in self.dynamic_hooks.values():
            for dynamic_hook in module_dynamic_hooks.values():
                dynamic_hook.update(epoch_idx, batch_idx)

    def remove(self):
        for module_dynamic_hooks in self.dynamic_hooks.values():
            for dynamic_hook in module_dynamic_hooks.values():
                dynamic_hook.remove()

    def get_data(self, module_name, hook_name, epoch_idx, batch_idx):
        return self.dynamic_hooks[module_name][hook_name].get_data(epoch_idx, batch_idx)
    
    @staticmethod
    def hook_list_from_dict(
        hook_dict, 
        capture_every_n_epochs: int = 1, 
        bhv_if_already_recorded='raise'
    ):
        """  
        Convenience method for instantiating a list of DynamicHook objects from
        a dict with a standardized structure; this list can be then used to
        instantiate a DynamicHookDict object. This may be useful when working
        in notebooks without configs. Structure of hook_dict should be:

        hook_dict = {
            'module_name' : {
                'hook_kind': ('forward', 'full_backward'),
                'capture_fn': (None, None)
                'hook_tag': ('', '')
            }
        }

        The hook_name attribute of each DynamicHook object is constructed as
        hook_name = hook_kind + hook_tag, where the hook_tag is
        an optional string allowing multiple hooks of the same kind to be
        registered to the same module. Note that uniqueness of hooks under a
        single module is not enforced here but non-uniqueness will cause an
        exception to be raised during instantiation of a DynamicHook object.
        """
        DEFAULT_CAPTURE_FNS = {
            'forward': lambda module, input, output: output.detach(),
            'forward_pre': lambda module, input, output: input[0].detach(), # Note returning only first item here
            'full_backward': lambda module, grad_input, grad_output: grad_output[0].detach() # Note returning only first item here
        }

        dynamic_hook_list = []
        for module_name, module_hooks in hook_dict.items():
            for hook_kind, capture_fn, hook_tag in zip(module_hooks['hook_kind'], module_hooks['capture_fn'], module_hooks['hook_tag']):
                if hook_kind not in DEFAULT_CAPTURE_FNS and capture_fn is None:
                    raise RuntimeError(
                        f"Default hook functions are supported for {DEFAULT_CAPTURE_FNS.keys()}. "
                        f"hook_kind '{hook_kind}' is not supported, so a capture_fn must be supplied "
                        "but got NoneType value."
                    )
                dynamic_hook = DynamicHook(
                    module_name=module_name, 
                    hook_kind=hook_kind,
                    hook_name=hook_kind + ':' + ('0' if hook_tag is None else hook_tag),
                    capture_fn=DEFAULT_CAPTURE_FNS[hook_kind] if capture_fn is None else capture_fn,
                    capture_every_n_epochs=capture_every_n_epochs,
                    bhv_if_already_recorded=bhv_if_already_recorded
                )
                dynamic_hook_list.append(dynamic_hook)

        return dynamic_hook_list
    