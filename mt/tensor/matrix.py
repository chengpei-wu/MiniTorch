import numpy as np
from typing import Union

from mt.tensor.tensor import Tensor


def zeros(shape: Union[int, tuple], requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def ones(shape: Union[int, tuple], requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def zeros_like(tensor: Tensor, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros_like(tensor.values), requires_grad=requires_grad)


def ones_like(tensor: Tensor, requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones_like(tensor.values), requires_grad=requires_grad)


def eye(n: int, m: int = None, k: int = 0, requires_grad: bool = False) -> Tensor:
    return Tensor(np.eye(N=n, M=m, k=k), requires_grad=requires_grad)


def identity(n: int, requires_grad: bool = False) -> Tensor:
    return Tensor(np.identity(n), requires_grad=requires_grad)


def full(shape: Union[int, tuple], fill_value: Union[int, float], requires_grad: bool = False) -> Tensor:
    return Tensor(np.full(shape=shape, fill_value=fill_value), requires_grad=requires_grad)


def full_like(tensor: Tensor, fill_value: Union[int, float], requires_grad: bool = False) -> Tensor:
    return Tensor(np.full_like(tensor.values, fill_value=fill_value), requires_grad=requires_grad)
