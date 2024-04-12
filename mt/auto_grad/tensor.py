from typing import Union

import numpy as np

__all__ = [
    'Tensor',
]

Arrayable = Union[int, float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


Tensorable = Union[int, float, list, np.ndarray, 'Tensor']


def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor:
    def __init__(self, values: Arrayable, requires_grad=False, depends_on=None):
        self.values = ensure_array(values)

        self.shape = self.values.shape
        self.grad = None

        self.requires_grad = requires_grad

        if requires_grad:
            # initial grad for a tensor requires grad
            # keep shape same with self.values
            self.zero_grad()

        if depends_on is None:
            depends_on = []
        self.depends_on = depends_on

    def backward(self, grad: np.ndarray = None):
        assert self.requires_grad, "Call backward() on a non-requires-grad tensor."

        # 1.0 for the output of an operating chain
        if grad is None:
            assert self.values.ndim == 0, 'grad can be implicitly created only for scalar outputs'
            grad = np.ones(self.shape)

        if not isinstance(grad, np.ndarray):
            grad = np.array(grad)

        # accumulate gradient
        # self.grad is the original grad;
        # grad is the new grad calculated form this tensor's computing output;
        self.grad += grad

        # propagate the gradient to its dependencies
        for dep in self.depends_on:
            # get grad for dependency tensor
            dep_grad = dep["grad_fn"](grad)

            # back propagate the grad to the dependency tensors of dependency tensor.
            dep["tensor"].backward(dep_grad)

    def reshape(self, shape: Union[int, tuple]) -> 'Tensor':
        return _reshape(self, shape)

    @property
    def T(self) -> 'Tensor':
        return _transpose(self)

    def zero_grad(self):
        self.grad = np.zeros(self.shape)

    def __repr__(self) -> str:
        return f'{self.values}'

    def __getitem__(self, idxs: 'Tensor') -> 'Tensor':
        return _slice(self, idxs)

    def __len__(self) -> int:
        return len(self.values)

    def sum(self, dim: int) -> 'Tensor':
        return _sum(self, dim)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        assert self.__class__ == other.__class__, f'{self.__class__} matmul {other.__class__}'
        return _matmul(self, other)

    def __add__(self, other: 'Tensor') -> 'Tensor':
        return _add(self, ensure_tensor(other))

    def __radd__(self, other: 'Tensor') -> 'Tensor':
        return _add(ensure_tensor(other), self)

    def __iadd__(self, other: 'Tensor') -> 'Tensor':
        self.data = self.values + ensure_tensor(other).values
        return self

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        return _sub(self, ensure_tensor(other))

    def __rsub__(self, other: 'Tensor') -> 'Tensor':
        return _sub(ensure_tensor(other), self)

    def __isub__(self, other: 'Tensor') -> 'Tensor':
        self.data = self.values - ensure_tensor(other).values
        return self

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other: 'Tensor') -> 'Tensor':
        return _mul(ensure_tensor(other), self)

    def __neg__(self) -> 'Tensor':
        return _neg(self)

    # def __pow__(self, power, modulo=None):
    #     return _pow(self, power, modulo=None)


def _transpose(tensor: Tensor) -> Tensor:
    res = tensor.values.T
    res_requires_grad = tensor.requires_grad
    res_depends_on = []

    if tensor.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad.T

        res_depends_on = [dict({'tensor': tensor, 'grad_fn': grad_fn})]

    return Tensor(values=res, requires_grad=res_requires_grad, depends_on=res_depends_on)


def _reshape(tensor: Tensor, shape: Union[int, tuple]) -> Tensor:
    res = tensor.values.reshape(shape)
    res_requires_grad = tensor.requires_grad
    res_depends_on = []

    if tensor.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad.reshape(shape)

        res_depends_on = [dict({'tensor': tensor, 'grad_fn': grad_fn})]

    return Tensor(values=res, requires_grad=res_requires_grad, depends_on=res_depends_on)


def _slice(tensor: Tensor, idxs: Tensor) -> Tensor:
    res = tensor.values[idxs.values]
    res_requires_grad = tensor.requires_grad
    res_depends_on = []

    if tensor.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(tensor.values)
            bigger_grad[idxs] = grad
            return bigger_grad

        res_depends_on = [dict({'tensor': tensor, 'grad_fn': grad_fn})]

    return Tensor(values=res, requires_grad=res_requires_grad, depends_on=res_depends_on)


def _matmul(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    # res = A @ B
    res = tensor1.values @ tensor2.values
    res_depends_on = []
    res_requires_grad = tensor1.requires_grad or tensor2.requires_grad

    if tensor1.requires_grad:
        def grad_fn_left(grad: np.ndarray) -> np.ndarray:
            # c = a @ b
            # D_c / D_a = grad @ b.T
            # D_c / D_b = a.T @ grad

            # grad.shape == c.shape
            return grad @ tensor2.values.T

        res_depends_on.append(dict({'tensor': tensor1, 'grad_fn': grad_fn_left}))

    if tensor2.requires_grad:
        def grad_fn_right(grad: np.ndarray) -> np.ndarray:
            return tensor1.values.T @ grad

        res_depends_on.append(dict({'tensor': tensor2, 'grad_fn': grad_fn_right}))

    return Tensor(values=res, requires_grad=res_requires_grad, depends_on=res_depends_on)


def _sum(tensor: Tensor, dim: int) -> Tensor:
    res = tensor.values.sum(dim)
    res_depends_on = []
    res_requires_grad = tensor.requires_grad or tensor.requires_grad

    if tensor.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # here, may exist auto-broadcasting by numpy
            return grad * np.ones_like(tensor.values)

        res_depends_on.append(dict({'tensor': tensor, 'grad_fn': grad_fn}))

    return Tensor(values=res, requires_grad=res_requires_grad, depends_on=res_depends_on)


def _add(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    # res = A + B
    res = tensor1.values + tensor2.values
    res_depends_on = []
    res_requires_grad = tensor1.requires_grad or tensor2.requires_grad

    if tensor1.requires_grad:
        def grad_fn_left(grad: np.ndarray) -> np.ndarray:
            # grad.shape == c.shape
            ndims_added = grad.ndim - tensor1.values.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(tensor1.values.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        res_depends_on.append(dict({'tensor': tensor1, 'grad_fn': grad_fn_left}))

    if tensor2.requires_grad:
        def grad_fn_right(grad: np.ndarray) -> np.ndarray:
            # grad.shape == c.shape
            ndims_added = grad.ndim - tensor2.values.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(tensor2.values.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        res_depends_on.append(dict({'tensor': tensor2, 'grad_fn': grad_fn_right}))

    return Tensor(values=res, requires_grad=res_requires_grad, depends_on=res_depends_on)


def _sub(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return _add(tensor1, _neg(tensor2))


def _mul(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    # res = A * B
    res = tensor1.values * tensor2.values
    res_depends_on = []
    res_requires_grad = tensor1.requires_grad or tensor2.requires_grad

    if tensor1.requires_grad:
        def grad_fn_left(grad: np.ndarray) -> np.ndarray:
            # grad.shape == c.shape
            grad = grad * tensor2.values
            ndims_added = grad.ndim - tensor1.values.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcast (but non-added dims)
            for i, dim in enumerate(tensor1.values.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        res_depends_on.append(dict({'tensor': tensor1, 'grad_fn': grad_fn_left}))

    if tensor2.requires_grad:
        def grad_fn_right(grad: np.ndarray) -> np.ndarray:
            # grad.shape == c.shape
            grad = grad * tensor1.values
            ndims_added = grad.ndim - tensor2.values.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcast (but non-added dims)
            for i, dim in enumerate(tensor2.values.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        res_depends_on.append(dict({'tensor': tensor2, 'grad_fn': grad_fn_right}))

    return Tensor(values=res, requires_grad=res_requires_grad, depends_on=res_depends_on)


def _neg(tensor: Tensor) -> Tensor:
    res = -tensor.values
    res_requires_grad = tensor.requires_grad
    res_depends_on = []
    if res_requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return -grad

        res_depends_on = [dict({'tensor': tensor, 'grad_fn': grad_fn})]

    return Tensor(values=res, requires_grad=res_requires_grad, depends_on=res_depends_on)


# def _pow(tensor: Tensor, power, modulo=None):
#     res = np.power(tensor.values, power)
#     res_requires_grad = tensor.requires_grad
#     res_depends_on = []
#     if res_requires_grad:
#         def grad_fn(grad):
#             return -grad
#
#         res_depends_on = [dict({'tensor': res, 'grad_fn': grad_fn})]
#
#     return Tensor(values=res, requires_grad=res_requires_grad, depends_on=res_depends_on)


if __name__ == '__main__':
    ...
