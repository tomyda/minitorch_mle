from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling using unfold for better efficiency.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width)
        kernel: Tuple (kh, kw) representing the kernel size for pooling

    Returns:
    -------
        A tuple containing:
            - output: Tensor of shape (batch, channel, new_height, new_width, kh * kw)
            - new_height: int
            - new_width: int

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert (
        height % kh == 0
    ), f"Height ({height}) must be divisible by kernel height ({kh})"
    assert width % kw == 0, f"Width ({width}) must be divisible by kernel width ({kw})"
    new_height = height // kh
    new_width = width // kw

    # Use unfold to extract patches
    output = (
        input.contiguous()
        .view(batch, channel, new_height, kh, new_width, kw)
        .permute(0, 1, 2, 4, 3, 5)
    )
    # Reshape to combine the kernel dimensions
    output = output.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return output, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Average pooling 2D using unfold for better efficiency.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width)
        kernel: Tuple (kh, kw) representing the kernel size for pooling

    Returns:
    -------
        Pooled tensor of shape (batch, channel, new_height, new_width)

    """
    b, c, height, width = input.shape
    nt, nh, nw = tile(input, kernel)
    return nt.mean(dim=4).view(b, c, nh, nw)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a one-hot encoded tensor along a specified dimension.

    Args:
    ----
        input: Tensor of any shape.
        dim: Integer specifying the dimension to apply argmax.

    Returns:
    -------
        Tensor: A one-hot encoded tensor with the same shape as the input,
                where positions of the maximum values along the specified
                dimension are set to 1, and all other positions are 0.

    """
    return max_reduce(input, dim) == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, i: Tensor, d: Tensor) -> Tensor:
        """Forward is max reduction"""
        ctx.save_for_backward(i, d)
        return max_reduce(i, int(d.item()))

    @staticmethod
    def backward(ctx: Context, go: Tensor) -> Tuple[Tensor, float]:
        """Backward: argmax"""
        input, dim = ctx.saved_values
        return (argmax(input, int(dim.item())) * go, dim)


def max(input: Tensor, dim: int) -> Tensor:
    """Max reduction"""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the softmax as a tensor.


    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
    ----
        input : input tensor
        dim : dimension to apply softmax

    Returns:
    -------
        softmax tensor

    """
    o = input.exp()
    so = o.sum(dim=dim)
    return o / so


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_j e^{x_j}$

    Args:
    ----
        input : Tensor of any shape
        dim : Dimension to apply log-softmax

    Returns:
    -------
        Tensor: Log of softmax tensor with the same shape as the input

    """
    oe = input.exp()
    os = oe.sum(dim=dim)
    ol = os.log()
    return input - ol


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """maxpool2d:

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor : pooled tensor

    """
    b, c, height, width = input.shape
    nt, nh, nw = tile(input, kernel)
    pooled = max(nt, 4).view(b, c, nh, nw)

    return pooled


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor based on random noise.

    Args:
    ----
        input : Tensor of any shape
        rate : Probability [0, 1) of dropping out each position
        ignore : If True, skip dropout and return the input tensor unchanged

    Returns:
    -------
        Tensor with random positions dropped out

    """
    if not ignore:
        b_tensor = rand(input.shape, input.backend) > rate
        input = b_tensor * input
    return input
