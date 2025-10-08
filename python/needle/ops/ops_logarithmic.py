from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        logsumexp_op = LogSumExp(axes=(1,))
        logsumexp_value = logsumexp_op.compute(Z)
        logsumexp_value = logsumexp_value.reshape(-1, 1)
        return Z - logsumexp_value
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # The gradient of logsoftmax is: out_grad - exp(logsoftmax(a)) * sum(out_grad)
        # This is equivalent to: out_grad - softmax(a) * sum(out_grad)
        softmax_a = exp(node)
        sum_grad = summation(out_grad, axes=(1,)).reshape((-1, 1)).broadcast_to(out_grad.shape)
        return out_grad - softmax_a * sum_grad
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        sum_exp = array_api.sum(array_api.exp(Z - max_z), axis=self.axes, keepdims=True)
        result = array_api.log(sum_exp) + max_z
        if self.axes is None:
            return array_api.float32(array_api.squeeze(result))
        else:
            return array_api.squeeze(result, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        
        max_z = array_api.max(a.realize_cached_data(), axis=self.axes, keepdims=True)
        exp_z_minus_max = array_api.exp(a.realize_cached_data() - max_z)
        sum_exp = array_api.sum(exp_z_minus_max, axis=self.axes, keepdims=True)
        
        # The gradient is: out_grad * exp(z - max_z) / sum(exp(z - max_z))
        grad_components = exp_z_minus_max / sum_exp
        grad_components_tensor = Tensor(grad_components)
        
        # Broadcast out_grad to match the input shape
        # out_grad has the shape of the output (after reduction)
        # We need to broadcast it to the input shape
        if self.axes is not None:
            # Create shape for broadcasting out_grad
            broadcast_shape = list(out_grad.shape)
            for axis in sorted(self.axes):
                broadcast_shape.insert(axis, 1)
            # Reshape and broadcast out_grad to input shape
            out_grad_broadcast = broadcast_to(reshape(out_grad, broadcast_shape), a.shape)
        else:
            # If axes is None, out_grad is a scalar, broadcast to full input shape
            out_grad_broadcast = broadcast_to(out_grad, a.shape)
        
        result = out_grad_broadcast * grad_components_tensor
        return result
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)