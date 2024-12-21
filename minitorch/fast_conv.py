from typing import Tuple, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compiles the given function using Numba's njit with inlining.

    Args:
    ----
        fn (Fn): The function to compile.
        **kwargs (Any): Additional keyword arguments for Numba's njit.

    Returns:
    -------
        Fn: The compiled function.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out_storage: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    in_storage: Storage,
    in_shape: Shape,
    in_strides: Strides,
    wt_storage: Storage,
    wt_shape: Shape,
    wt_strides: Strides,
    reverse: bool,
) -> None:
    """Performs a 1D convolution operation.

    Args:
    ----
        out_storage (Storage): Storage for the output tensor.
        out_shape (Shape): Shape of the output tensor (batch_size, num_out_channels, output_width).
        out_strides (Strides): Strides of the output tensor.
        out_size (int): Total number of elements in the output tensor.
        in_storage (Storage): Storage for the input tensor.
        in_shape (Shape): Shape of the input tensor (batch_size, num_in_channels, input_width).
        in_strides (Strides): Strides of the input tensor.
        wt_storage (Storage): Storage for the weight tensor.
        wt_shape (Shape): Shape of the weight tensor (num_out_channels, num_in_channels, kernel_width).
        wt_strides (Strides): Strides of the weight tensor.
        reverse (bool): If True, performs a reverse convolution.

    Returns:
    -------
        None: The result is stored in out_storage.

    """
    batch_size_out, num_out_channels, output_width = out_shape
    batch_size_in, num_in_channels, input_width = in_shape
    num_out_channels_wt, num_in_channels_wt, kernel_width = wt_shape

    # Ensure that the dimensions are compatible
    assert (
        batch_size_in == batch_size_out
        and num_in_channels == num_in_channels_wt
        and num_out_channels == num_out_channels_wt
    )

    in_s = in_strides
    wt_s = wt_strides

    for b_idx in prange(batch_size_out):
        for o_ch in prange(num_out_channels):
            for o_w in prange(output_width):
                conv_sum = 0.0
                for i_ch in prange(num_in_channels):
                    for k_w in prange(kernel_width):
                        # Calculate weight position
                        wt_idx = o_ch * wt_s[0] + i_ch * wt_s[1] + k_w * wt_s[2]

                        # Determine input width index based on 'reverse' flag
                        if reverse:
                            in_w = o_w - k_w
                        else:
                            in_w = o_w + k_w

                        # Check if the input width index is within bounds
                        if 0 <= in_w < input_width:
                            # Calculate input position
                            in_idx = b_idx * in_s[0] + i_ch * in_s[1] + in_w * in_s[2]
                            # Accumulate the convolution sum
                            conv_sum += in_storage[in_idx] * wt_storage[wt_idx]

                # Calculate output position and store the result
                out_idx = (
                    b_idx * out_strides[0]
                    + o_ch * out_strides[1]
                    + o_w * out_strides[2]
                )
                out_storage[out_idx] = conv_sum


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for the convolution operation.

        Args:
        ----
            ctx (Context): Context object containing saved tensors from the forward pass.
            grad_output (Tensor): Gradient of the loss with respect to the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to the input and weight tensors.

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    output_storage: Storage,
    output_shape: Shape,
    output_strides: Strides,
    output_size: int,
    input_storage: Storage,
    input_shape: Shape,
    input_strides: Strides,
    kernel_storage: Storage,
    kernel_shape: Shape,
    kernel_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given an input tensor of shape:
        (batch_size, in_channels, input_height, input_width)

    and a kernel tensor of shape:
        (out_channels, in_channels, kernel_height, kernel_width)

    Computes an output tensor of shape:
        (batch_size, out_channels, output_height, output_width)

    The 'reverse' parameter determines whether the convolution is reversed.

    Args:
    ----
        output_storage (Storage): Storage for the output tensor.
        output_shape (Shape): Shape of the output tensor.
        output_strides (Strides): Strides of the output tensor.
        output_size (int): Total number of elements in the output tensor.
        input_storage (Storage): Storage for the input tensor.
        input_shape (Shape): Shape of the input tensor.
        input_strides (Strides): Strides of the input tensor.
        kernel_storage (Storage): Storage for the kernel tensor.
        kernel_shape (Shape): Shape of the kernel tensor.
        kernel_strides (Strides): Strides of the kernel tensor.
        reverse (bool): If True, performs a reversed convolution.

    """
    batch_size_out, num_out_channels, out_height, out_width = output_shape
    batch_size_in, num_in_channels, in_height, in_width = input_shape
    num_out_channels_k, num_in_channels_k, k_height, k_width = kernel_shape

    # Ensure that the dimensions are compatible
    assert (
        batch_size_out == batch_size_in
        and num_in_channels == num_in_channels_k
        and num_out_channels == num_out_channels_k
    )

    # Aliases for strides
    in_s = input_strides
    k_s = kernel_strides

    # Loop over each element in the output tensor
    for out_idx in prange(output_size):
        # Convert the flat output index to multi-dimensional indices
        out_multi_idx = [0, 0, 0, 0]
        to_index(out_idx, output_shape, out_multi_idx)  # type: ignore
        b_idx, o_ch_idx, o_h_idx, o_w_idx = out_multi_idx

        # Initialize the convolution sum accumulator
        conv_sum = 0.0

        # Loop over each input channel and kernel position
        for i_ch_idx in range(num_in_channels):
            for k_h_idx in range(k_height):
                for k_w_idx in range(k_width):
                    # Compute the corresponding input positions
                    if reverse:
                        in_h_idx = o_h_idx - k_h_idx
                        in_w_idx = o_w_idx - k_w_idx
                    else:
                        in_h_idx = o_h_idx + k_h_idx
                        in_w_idx = o_w_idx + k_w_idx

                    # Check if the input indices are within valid bounds
                    if 0 <= in_h_idx < in_height and 0 <= in_w_idx < in_width:
                        # Compute flat indices for input and kernel
                        in_multi_idx = np.array(
                            [b_idx, i_ch_idx, in_h_idx, in_w_idx], dtype=np.int32
                        )
                        in_flat_idx = index_to_position(in_multi_idx, in_s)

                        k_multi_idx = np.array(
                            [o_ch_idx, i_ch_idx, k_h_idx, k_w_idx], dtype=np.int32
                        )
                        k_flat_idx = index_to_position(k_multi_idx, k_s)

                        # Accumulate the convolution result
                        conv_sum += (
                            input_storage[in_flat_idx] * kernel_storage[k_flat_idx]
                        )
        # Compute the flat index for the output position
        out_multi_idx_np = np.array(out_multi_idx, dtype=np.int32)
        out_flat_idx = index_to_position(out_multi_idx_np, output_strides)

        # Store the accumulated result in the output storage
        output_storage[out_flat_idx] = conv_sum


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for the convolution operation.

        Args:
        ----
            ctx (Context): Context object containing saved tensors from the forward pass.
            grad_output (Tensor): Gradient of the loss with respect to the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to the input and weight tensors.

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
