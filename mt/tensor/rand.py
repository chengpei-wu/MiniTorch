import numpy as np
from typing import Union

from mt.tensor.tensor import Tensor


def random(shape: Union[int, tuple], requires_grad: bool = False) -> Tensor:
    return Tensor(values=np.random.random(shape), requires_grad=requires_grad)


def randint(low: int, high: int, shape: Union[int, tuple], requires_grad: bool = False) -> Tensor:
    return Tensor(values=np.random.randint(low, high, shape), requires_grad=requires_grad)


def rand(*shape: int, requires_grad: bool = False) -> Tensor:
    return Tensor(values=np.random.rand(*shape), requires_grad=requires_grad)


def randn(*shape: int, requires_grad: bool = False) -> Tensor:
    return Tensor(values=np.random.randn(*shape), requires_grad=requires_grad)


if __name__ == '__main__':
    a = rand(1, 2, 3)
    print(a)
