import numpy as np

from mt import Tensor


def cross_entropy(logits: Tensor, y: Tensor):
    output = np.exp(logits.values - logits.values.max(axis=-1, keepdims=True))
    prob = output / output.sum(axis=-1, keepdims=True)

    res = -np.einsum('ij,ij->', y.values, np.log(prob), optimize=True) / y.shape[0]
    res_requires_grad = logits.requires_grad or y.requires_grad

    if logits.requires_grad:
        def grad_fn(grad):
            return grad * (prob - y.values)

        depends_on = [dict({'tensor': logits, 'grad_fn': grad_fn})]
    else:
        depends_on = []

    return Tensor(values=res, requires_grad=res_requires_grad, depends_on=depends_on)
