"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features))) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = X @ self.weight
        if self.bias is not None:
            res = res + ops.broadcast_to(self.bias, res.shape)
        return res
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        product = 1
        for i in range(1, len(X.shape)):
            product *= X.shape[i]
        return ops.reshape(X, (X.shape[0], product))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module.forward(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n = logits.shape[0]
        k = logits.shape[-1]
        per_sample = ops.logsumexp(logits, axes=(-1,)) - \
                     ops.summation(init.one_hot(k, y) * logits, axes=(-1,))
        return ops.summation(per_sample) / n
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        row_shape = (1, self.dim)

        if self.training:
            E_x = ops.summation(x, axes=(0,)) / batch_size
            broadcast_E_x = ops.broadcast_to(ops.reshape(E_x, row_shape), x.shape)
            Var_x = ops.summation((x - broadcast_E_x) ** 2, axes=(0,)) / batch_size
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * E_x
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * Var_x
            broadcast_Var_x = ops.broadcast_to(ops.reshape(Var_x, row_shape), x.shape)
        else:
            E_x = self.running_mean
            Var_x = self.running_var
            broadcast_E_x = ops.broadcast_to(ops.reshape(E_x, row_shape), x.shape)
            broadcast_Var_x = ops.broadcast_to(ops.reshape(Var_x, row_shape), x.shape)
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, row_shape), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, row_shape), x.shape)
        broadcast_eps = init.ones(*x.shape) * self.eps
        return broadcast_weight * (x - broadcast_E_x) / ((broadcast_Var_x + broadcast_eps) ** 0.5) + broadcast_bias
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        E_x = ops.reshape(ops.summation(x, axes=(1,)) / self.dim, (batch_size, 1))
        Var_x = ops.reshape(ops.summation((x - ops.broadcast_to(E_x, x.shape)) ** 2, axes=(1,)) / self.dim, (batch_size, 1))
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        broadcast_E_x = ops.broadcast_to(E_x, x.shape)
        broadcast_Var_x = ops.broadcast_to(Var_x, x.shape)
        broadcast_eps = init.ones(*x.shape) * self.eps
        return broadcast_weight * (x - broadcast_E_x) / ((broadcast_Var_x + broadcast_eps) ** 0.5) + broadcast_bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x

        mask = init.randb(*x.shape, p=1-self.p) / (1 - self.p)
        return x * mask
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn.forward(x) + x
        ### END YOUR SOLUTION
