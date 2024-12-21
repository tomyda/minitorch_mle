"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class Function:
    @classmethod
    def _backward(cls, contx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Perform the backward pass of the function."""
        return wrap_tuple(cls.backward(contx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, contx: Context, *inps: Tensor) -> Tensor:
        """Perform the forward pass of the function."""
        return cls.forward(contx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        contx = Context(not need_grad)

        c = cls._forward(contx, *raw_vals)

        back = None
        if need_grad:
            back = minitorch.History(cls, contx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(contx: Context, t1: Tensor) -> Tensor:
        """Negate a tensor."""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tensor:
        """Negate the gradient."""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(contx: Context, t1: Tensor) -> Tensor:
        """Inverse a tensor."""
        contx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tensor:
        """Inverse the gradient."""
        (t1,) = contx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(contx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Add two tensors."""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Add the gradients."""
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(contx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all elements are true along a dimension or all dimensions if dim is None."""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    @staticmethod
    def forward(contx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiplication of two tensors."""
        contx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for element-wise multiplication."""
        a, b = contx.saved_values
        return (
            grad_output.f.mul_zip(b, grad_output),
            grad_output.f.mul_zip(a, grad_output),
        )


class Sigmoid(Function):
    @staticmethod
    def forward(contx: Context, t1: Tensor) -> Tensor:
        """Compute the sigmoid function."""
        out = t1.f.sigmoid_map(t1)
        contx.save_for_backward(out)
        return out

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the sigmoid function."""
        sigma: Tensor = contx.saved_values[0]
        return sigma * (-sigma + 1.0) * grad_output


class ReLU(Function):
    @staticmethod
    def forward(contx: Context, t1: Tensor) -> Tensor:
        """Compute the ReLU function."""
        contx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the ReLU function."""
        (a,) = contx.saved_values
        return grad_output.f.relu_back_zip(a, grad_output)


class Log(Function):
    @staticmethod
    def forward(contx: Context, t1: Tensor) -> Tensor:
        """Compute the natural logarithm."""
        contx.save_for_backward(t1)
        out = t1.f.log_map(t1)
        return out

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the natural logarithm.."""
        (a,) = contx.saved_values
        return grad_output.f.log_back_zip(a, grad_output)


class Exp(Function):
    @staticmethod
    def forward(contx: Context, t1: Tensor) -> Tensor:
        """Compute the exponential function."""
        out = t1.f.exp_map(t1)
        contx.save_for_backward(out)
        return out

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the exponential function."""
        (a,) = contx.saved_values
        return grad_output.f.mul_zip(a, grad_output)


class Sum(Function):
    @staticmethod
    def forward(contx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for sum."""
        contx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for sum."""
        a_shape, dim = contx.saved_values
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(contx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise less-than comparison"""
        contx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for less-than comparison (zero gradients)"""
        a_shape, b_shape = contx.saved_values
        return zeros(a_shape), zeros(b_shape)


class EQ(Function):
    @staticmethod
    def forward(contx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise equality comparison."""
        contx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for equality comparison (zero gradients)"""
        a_shape, b_shape = contx.saved_values
        return zeros(a_shape), zeros(b_shape)


class IsClose(Function):
    @staticmethod
    def forward(contx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise closeness check."""
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(contx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Permute the dimensions of a tensor."""
        contx.save_for_backward(order)
        return a._new(a._tensor.permute(*[int(order[i]) for i in range(order.size)]))

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for permutation (reverse permutation)."""
        order: Tensor = contx.saved_values[0]
        order2: List[int] = [
            a[0]
            for a in sorted(
                enumerate([order[i] for i in range(order.size)]), key=lambda a: a[1]
            )
        ]
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class View(Function):
    @staticmethod
    def forward(contx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """View a tensor"""
        contx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = contx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(contx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(contx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix multiply forward (module 3)"""
        contx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(contx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix multiply Backward (module 3)"""
        t1, t2 = contx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference gradient check"""
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
