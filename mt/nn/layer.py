import numpy as np

from mt import Tensor
from mt import random
from mt.nn.activations import relu
from mt.nn.activations import tanh
from mt.nn.module import Module
from mt.tensor.rand import rand, randn
from mt.tensor.matrix import zeros
from typing import Union


def xavier_init(shape):
    """Xavier initialization"""
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape) * scale


def init_activation(name='relu'):
    if name is None:
        return lambda x: x
    elif name == 'relu':
        return relu
    elif name == 'tanh':
        return tanh
    else:
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_feats, out_feats, activation='relu'):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.weights = rand(in_feats, out_feats, requires_grad=True)
        self.bias = zeros(out_feats, requires_grad=True)
        self.reset_param()
        self.activation = init_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x @ self.weights + self.bias)

    def reset_param(self, method='xavier'):
        if method == 'xavier':
            self.weights.values = xavier_init(self.weights.shape)
        elif method == 'kaiming':
            raise NotImplementedError(f'{method} initialization.')
        else:
            raise NotImplementedError(f'{method} initialization.')


class Embedding(Module):
    def __init__(self, num_embeddings, embed_dim):
        super().__init__()
        self.weights = randn(num_embeddings, embed_dim, requires_grad=True)

    def forward(self, idxs: Tensor) -> Tensor:
        assert idxs.__class__ == Tensor, 'indexes must be Tensor'
        return self.weights[idxs]


if __name__ == '__main__':
    embed = Embedding(10, 2)
    x = embed(Tensor([[1, 2], [1, 3]]))
    print(x)
