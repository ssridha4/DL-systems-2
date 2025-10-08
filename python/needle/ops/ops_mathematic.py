"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        assert a.shape == b.shape, "Shape mismatch in addition"
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        # ensure return type is float32
        return (a + self.scalar).astype('float32')

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        assert a.shape == b.shape, "Shape mismatch in multiplication"
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        assert a.shape == b.shape, "Shape mismatch in power"
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad1 = out_grad * b * power(a, b - 1)
        grad2 = out_grad * power(a, b) * log(a)
        return grad1, grad2
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (out_grad * self.scalar * power_scalar(a, self.scalar - 1),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        assert a.shape == b.shape, "Shape mismatch in division"
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return (out_grad / b, -out_grad * a / power_scalar(b, 2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.float32(a / self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (out_grad / self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = self.axes if self.axes is not None else (-1, -2)
        return array_api.swapaxes(a, axes[0], axes[1])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        # return (out_grad.reshape(a.shape),)
        return (reshape(out_grad, a.shape),)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        in_shape = a.shape
        # Align shapes by prepending 1s to x_shape if necessary
        num_missing_dims = len(self.shape) - len(in_shape)
        shape_aligned = (1,) * num_missing_dims + in_shape

        # Axes where broadcasting occurred
        axes_to_sum = [
            axis for axis, (dx, dy) in enumerate(zip(shape_aligned, out_grad.shape))
            if dx == 1 and dy > 1
        ] + list(range(num_missing_dims))  # also sum over added dimensions

        axes_to_sum = set(axes_to_sum)
        # Sum along broadcasted axes
        res = summation(out_grad, tuple(axes_to_sum))

        # Finally, reshape to original x_shape
        return (reshape(res, in_shape),)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        shape = list(out_grad.shape)

        if self.axes is not None:
            axes = self.axes if isinstance(self.axes, (tuple, list)) else (self.axes,)
            for axis in sorted(axes):
                shape.insert(axis, 1)

        return (broadcast_to(reshape(out_grad, shape), a.shape),)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs

        # Step 1: standard batched matmul gradient (swap last 2 axes)
        grad_a = matmul(out_grad, transpose(b))
        grad_b = matmul(transpose(a), out_grad)

        # Step 2: reduce extra leading batch dimensions
        while len(grad_a.shape) > len(a.shape):
            grad_a = summation(grad_a, (0,))
        while len(grad_b.shape) > len(b.shape):
            grad_b = summation(grad_b, (0,))

        # Step 3: sum over axes where original tensor had size 1 (broadcasted)
        for i, (ga_dim, a_dim) in enumerate(zip(grad_a.shape, a.shape)):
            if a_dim == 1 and ga_dim != 1:
                grad_a = summation(grad_a, (i,))

        for i, (gb_dim, b_dim) in enumerate(zip(grad_b.shape, b.shape)):
            if b_dim == 1 and gb_dim != 1:
                grad_b = summation(grad_b, (i,))

        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1 * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad,)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (out_grad * 1/a,)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (out_grad * exp(a))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0, dtype='float32')
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION        
        mask = (node.realize_cached_data() > 0)
        return Tensor(out_grad * Tensor(mask), dtype='float32')
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


