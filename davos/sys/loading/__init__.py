from pathlib import Path
import importlib
import inspect

import torch
from davos.config import env_settings
from . import default_pickler


def load_trained_network(network_path, checkpoint=None):
    net, _ = load_network(env_settings().checkpoints_dir / network_path, checkpoint)
    return net


def load_network(network_dir=None, checkpoint=None, constructor_fun_name=None, constructor_module=None, strict=True, **kwargs):
    """ Load a network checkpoint file.
        Extra keyword arguments are supplied to the network constructor to replace saved ones.
    """

    net_path = Path(network_dir).expanduser() if network_dir is not None else None
    if net_path.is_file():  # net_path is the checkpoint
        checkpoints = [net_path]
    elif checkpoint is None:  # Load most recent checkpoint
        checkpoints = list(reversed(sorted(net_path.glob('*.pth.tar'))))
    elif isinstance(checkpoint, int):  # Checkpoint is the epoch number
        checkpoints = list(sorted(net_path.glob('*_ep{:04d}.pth.tar'.format(checkpoint))))
        if len(checkpoints) > 1:
            raise FileNotFoundError('Multiple matching checkpoint files found')
    else:
        raise TypeError

    if len(checkpoints) == 0:
        raise FileNotFoundError('No matching checkpoint file found')
    checkpoint_path = str(checkpoints[0])

    # Load network
    checkpoint = torch.load(checkpoint_path, map_location='cpu', pickle_module=default_pickler)

    # Construct network model
    constructor = checkpoint.get('constructor')
    if constructor is None:
        constructor = checkpoint.get('model_constructor')
    if constructor is not None:
        if constructor_fun_name is not None:
            constructor.fun_name = constructor_fun_name
        if constructor_module is not None:
            constructor.fun_module = constructor_module

        if constructor.fun_module.startswith("ltr."):
            constructor.fun_module = "davos." + constructor.fun_module[4:]

        net_fun = getattr(importlib.import_module(constructor.fun_module), constructor.fun_name)
        net_fun_args = list(inspect.signature(net_fun).parameters.keys())
        for arg, val in kwargs.items():
            # if arg in net_fun_args:
            constructor.kwds[arg] = val
            # else:
            #     print('WARNING: Keyword argument "{}" not found when loading network. It was ignored.'.format(arg))
        net = constructor.get()
    else:
        raise RuntimeError('No constructor for the given network.')

    state_dict = checkpoint.get('net')
    state_dict = checkpoint.get('model') if state_dict is None else state_dict
    net.load_state_dict(state_dict, strict=strict)
    net.constructor = constructor
    net.info = checkpoint.get('net_info')

    return net, checkpoint
