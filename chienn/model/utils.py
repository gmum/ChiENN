import torch
from torch import nn


def _build_single_embedding_layer(in_dim: int, out_dim: int, name: str):
    if name == 'linear':
        return nn.Linear(in_dim, out_dim, bias=False)
    elif name == 'identity':
        return nn.Identity()
    elif name == 'scalar':
        return nn.Linear(in_dim, 1, bias=True)
    elif name == 'self_concat':
        return lambda x: torch.cat([x, x], dim=-1)
    elif name == 'double':
        return lambda x: 2 * x
    elif hasattr(torch.nn, name):
        return getattr(torch.nn, name)()
    else:
        raise NotImplementedError(f'Layer name {name} is not implemented.')


def build_embedding_layer(in_dim: int, out_dim: int, name: str):
    sub_names = name.split('+')
    if len(sub_names) == 1:
        return _build_single_embedding_layer(in_dim, out_dim, sub_names[0])
    else:
        return nn.Sequential(*[_build_single_embedding_layer(in_dim, out_dim, sub_name) for sub_name in sub_names])
