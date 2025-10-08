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
        weight = Parameter(init.kaiming_uniform(fan_in=in_features, fan_out=out_features, device=device, dtype=dtype))
        bias = Parameter(init.kaiming_uniform(fan_in=out_features, fan_out=1, device=device, dtype=dtype)) if bias else None
        self.weight = weight
        self.bias = bias.reshape((1, self.out_features))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        result = X @ self.weight 
        result = result + self.bias.broadcast_to(result.shape) if self.bias is not None else result
        return result
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
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
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y_onehot = init.one_hot(logits.shape[1], y, dtype='float32', device=logits.device)
        z_y = ops.summation(y_onehot * logits, axes=(1,))
        loss = ops.logsumexp(logits, axes=(1,)) - z_y
        avg_loss = ops.summation(loss, axes=(0,)) / logits.shape[0]
        # return Tensor(avg_loss, dtype='float32')
        return avg_loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = Tensor(init.zeros(dim, device=device, dtype=dtype))
        self.running_var = Tensor(init.ones(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, feature_dim = x.shape
        
        if self.training:
            mean = ops.summation(x, axes=0) / batch_size
            mean_broadcast = ops.reshape(mean, (1, feature_dim)).broadcast_to(x.shape)

            var = ops.summation((x - mean_broadcast) ** 2.0, axes=0) / batch_size
            var_broadcast = ops.reshape(var, (1, feature_dim)).broadcast_to(x.shape)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            mean_norm = mean_broadcast
            var_norm = var_broadcast
        
        else:
            mean_norm = ops.reshape(self.running_mean, (1, feature_dim)).broadcast_to(x.shape)
            var_norm = ops.reshape(self.running_var, (1, feature_dim)).broadcast_to(x.shape)
            
        x_normalized = (x - mean_norm) / (var_norm + np.float32(self.eps)) ** 0.5
        
        weight_reshaped = ops.reshape(self.weight, (1, self.dim)).broadcast_to(x.shape)
        bias_reshaped = ops.reshape(self.bias, (1, self.dim)).broadcast_to(x.shape)
        
        return weight_reshaped * x_normalized + bias_reshaped
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, feature_dim = x.shape

        mean = ops.summation(x, axes=1) / feature_dim
        mean = ops.reshape(mean, (batch_size, 1))
        mean = mean.broadcast_to(x.shape)
        
        var = ops.summation((x - mean) ** 2.0, axes=1) / feature_dim
        var = ops.reshape(var, (batch_size, 1))
        
        # Normalize: (x - mean) / sqrt(var + eps)
        x_normalized = (x - mean) / ops.broadcast_to((var + self.eps) ** 0.5, x.shape)
        
        weight_reshaped = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), (batch_size, self.dim))
        bias_reshaped = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), (batch_size, self.dim))
        
        return weight_reshaped * x_normalized + bias_reshaped
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, dtype=x.dtype)
            return x * mask / (1-self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
