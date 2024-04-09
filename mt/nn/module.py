import inspect
from typing import Iterator

from mt import Tensor

__all__ = [
    'Module',
    'ModuleList'
]


class ModuleList:
    def __init__(self, modules=None):
        if modules is None:
            self.modules = []
        else:
            self.modules = modules

    def __getitem__(self, index):
        return self.modules[index]

    def __setitem__(self, index, value):
        self.modules[index] = value

    def __len__(self):
        return len(self.modules)

    def append(self, value):
        self.modules.append(value)

    def pop(self):
        return self.modules.pop()


class Module:
    def __init__(self):
        super().__init__()
        self.weights = None
        self.bias = None

    def parameters(self):
        params = []
        for name, value in inspect.getmembers(self):
            if isinstance(value, Tensor) and value.requires_grad:
                params.append({'name': name, 'value': value})
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, ModuleList):
                for sub_module in value:
                    params.extend(sub_module.parameters())

        return params

    def zero_grad(self):
        for parameter in self.parameters():
            parameter['value'].zero_grad()

    def forward(self, x):
        return x @ self.weights + self.bias

    def __call__(self, x):
        return self.forward(x)
