import numpy as np

from mt import Tensor
from mt import rand_tensor
from mt.nn.activations import relu
from mt.nn.activations import tanh
from mt.nn.module import Module


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

        self.weights = Tensor(np.zeros((in_feats, out_feats)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_feats), requires_grad=True)
        self.reset_param()
        self.activation = init_activation(activation)

    def forward(self, x):
        return self.activation(x @ self.weights + self.bias)

    def reset_param(self, method='xavier'):
        if method == 'xavier':
            self.weights.values = xavier_init(self.weights.shape)
        elif method == 'kaiming':
            raise NotImplementedError(f'{method} initialization.')
        else:
            raise NotImplementedError(f'{method} initialization.')


if __name__ == '__main__':
    x = rand_tensor((10, 10))
