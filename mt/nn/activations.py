import numpy as np

from mt import Tensor


def tanh(tensor: Tensor) -> Tensor:
    res = np.tanh(tensor.values)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad):
            return grad * (1 - res * res)

        depends_on = [dict({'tensor': tensor, 'grad_fn': grad_fn})]
    else:
        depends_on = []

    return Tensor(values=res, requires_grad=requires_grad, depends_on=depends_on)


def relu(tensor: Tensor) -> Tensor:
    res = np.maximum(tensor.values, 0)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad):
            _t = np.array(res, copy=True)
            _t[res > 0] = 1.0
            _t[res <= 0] = 0.0
            return grad * _t

        depends_on = [dict({'tensor': tensor, 'grad_fn': grad_fn})]
    else:
        depends_on = []

    return Tensor(values=res, requires_grad=requires_grad, depends_on=depends_on)
