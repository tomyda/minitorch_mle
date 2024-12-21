# from typing import Tuple, TypeVar, Any, Callable
# import numpy as np
# from numba import cuda, prange
# from numba.cuda import jit as _jit

# from .autodiff import Context
# from .tensor import Tensor
# from .tensor_data import (
#     MAX_DIMS,
#     Index,
#     Shape,
#     Strides,
#     Storage,
#     broadcast_index,
#     index_to_position,
#     to_index,
# )
# from .tensor_functions import Function
# from .cuda_ops import CudaOps

# Fn = TypeVar("Fn")


# def device_jit(fn: Fn, **kwargs: Any) -> Fn:
#     """JIT compile a function for CUDA device execution."""
#     return _jit(device=True, **kwargs)(fn)  # type: ignore


# def jit(fn: Callable, **kwargs: Any) -> Callable:
#     """JIT compile a function for CUDA execution."""
#     return _jit(**kwargs)(fn)  # type: ignore


# # JIT compile tensor data functions for CUDA
# to_index = device_jit(to_index)
# index_to_position = device_jit(index_to_position)
# broadcast_index = device_jit(broadcast_index)

# THREADS_PER_BLOCK = 32


# @cuda.jit
# def _tensor_conv1d_cuda(
#     out_storage: Storage,
#     out_shape: Shape,
#     out_strides: Strides,
#     out_size: int,
#     in_storage: Storage,
#     in_shape: Shape,
#     in_strides: Strides,
#     wt_storage: Storage,
#     wt_shape: Shape,
#     wt_strides: Strides,
#     reverse: bool,
# ) -> None:
#     idx = cuda.grid(1)
#     if idx >= out_size:
#         return

#     batch_size_out, num_out_channels, output_width = out_shape
#     batch_size_in, num_in_channels, input_width = in_shape
#     num_out_channels_wt, num_in_channels_wt, kernel_width = wt_shape

#     # Ensure compatibility
#     if not (
#         batch_size_in == batch_size_out
#         and num_in_channels == num_in_channels_wt
#         and num_out_channels == num_out_channels_wt
#     ):
#         return

#     # Calculate multi-dimensional indices from flat index
#     out_multi_idx = [0, 0, 0]
#     to_index(idx, out_shape, out_multi_idx)
#     b_idx, o_ch, o_w = out_multi_idx

#     conv_sum = 0.0
#     for i_ch in range(num_in_channels):
#         for k_w in range(kernel_width):
#             wt_idx = o_ch * wt_strides[0] + i_ch * wt_strides[1] + k_w * wt_strides[2]
#             if reverse:
#                 in_w = o_w - k_w
#             else:
#                 in_w = o_w + k_w

#             if 0 <= in_w < input_width:
#                 in_idx = (
#                     b_idx * in_strides[0] + i_ch * in_strides[1] + in_w * in_strides[2]
#                 )
#                 conv_sum += in_storage[in_idx] * wt_storage[wt_idx]

#     out_idx = b_idx * out_strides[0] + o_ch * out_strides[1] + o_w * out_strides[2]
#     out_storage[out_idx] = conv_sum


# tensor_conv1d_cuda = cuda.jit(_tensor_conv1d_cuda)


# @cuda.jit
# def _tensor_conv2d_cuda(
#     out_storage: Storage,
#     out_shape: Shape,
#     out_strides: Strides,
#     out_size: int,
#     in_storage: Storage,
#     in_shape: Shape,
#     in_strides: Strides,
#     wt_storage: Storage,
#     wt_shape: Shape,
#     wt_strides: Strides,
#     reverse: bool,
# ) -> None:
#     idx = cuda.grid(1)
#     if idx >= out_size:
#         return

#     batch_size_out, num_out_channels, out_height, out_width = out_shape
#     batch_size_in, num_in_channels, in_height, in_width = in_shape
#     num_out_channels_wt, num_in_channels_wt, kernel_height, kernel_width = wt_shape

#     # Ensure compatibility
#     if not (
#         batch_size_out == batch_size_in
#         and num_in_channels == num_in_channels_wt
#         and num_out_channels == num_out_channels_wt
#     ):
#         return

#     # Calculate multi-dimensional indices from flat index
#     out_multi_idx = [0, 0, 0, 0]
#     to_index(idx, out_shape, out_multi_idx)
#     b_idx, o_ch, o_h, o_w = out_multi_idx

#     conv_sum = 0.0
#     for i_ch in range(num_in_channels):
#         for k_h in range(kernel_height):
#             for k_w in range(kernel_width):
#                 wt_idx = (
#                     o_ch * wt_strides[0]
#                     + i_ch * wt_strides[1]
#                     + k_h * wt_strides[2]
#                     + k_w * wt_strides[3]
#                 )
#                 if reverse:
#                     in_h = o_h - k_h
#                     in_w = o_w - k_w
#                 else:
#                     in_h = o_h + k_h
#                     in_w = o_w + k_w

#                 if 0 <= in_h < in_height and 0 <= in_w < in_width:
#                     in_idx = (
#                         b_idx * in_strides[0]
#                         + i_ch * in_strides[1]
#                         + in_h * in_strides[2]
#                         + in_w * in_strides[3]
#                     )
#                     conv_sum += in_storage[in_idx] * wt_storage[wt_idx]

#     out_idx = (
#         b_idx * out_strides[0]
#         + o_ch * out_strides[1]
#         + o_h * out_strides[2]
#         + o_w * out_strides[3]
#     )
#     out_storage[out_idx] = conv_sum


# tensor_conv2d_cuda = cuda.jit(_tensor_conv2d_cuda)


# class Conv1dCudaFun(Function):
#     @staticmethod
#     def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
#         """Compute a 1D Convolution using CUDA."""
#         ctx.save_for_backward(input, weight)
#         batch, in_channels, in_width = input.shape
#         out_channels, _, kernel_width = weight.shape

#         output_width = in_width - kernel_width + 1
#         out = input.zeros((batch, out_channels, output_width))

#         tensor_conv1d_cuda[
#             (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, THREADS_PER_BLOCK
#         ](*out.tuple(), out.size, *input.tuple(), *weight.tuple(), False)
#         return out

#     @staticmethod
#     def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
#         input, weight = ctx.saved_values
#         batch, in_channels, in_width = input.shape
#         out_channels, _, kernel_width = weight.shape

#         grad_weight = grad_output.zeros(weight.shape)
#         grad_input = input.zeros(input.shape)

#         # Gradient with respect to weights
#         tensor_conv1d_cuda[
#             (grad_weight.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
#             THREADS_PER_BLOCK,
#         ](
#             *grad_weight.tuple(),
#             grad_weight.size,
#             *input.tuple(),
#             *grad_output.tuple(),
#             True,
#         )

#         # Gradient with respect to input
#         tensor_conv1d_cuda[
#             (grad_input.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
#             THREADS_PER_BLOCK,
#         ](
#             *grad_input.tuple(),
#             grad_input.size,
#             *grad_output.tuple(),
#             *weight.tuple(),
#             True,
#         )

#         return grad_input, grad_weight


# tensor_conv1d_cuda_fun = Conv1dCudaFun.apply


# class Conv2dCudaFun(Function):
#     @staticmethod
#     def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
#         """Compute a 2D Convolution using CUDA."""
#         ctx.save_for_backward(input, weight)
#         batch, in_channels, in_height, in_width = input.shape
#         out_channels, _, kernel_height, kernel_width = weight.shape

#         out_height = in_height - kernel_height + 1
#         out_width = in_width - kernel_width + 1
#         out = input.zeros((batch, out_channels, out_height, out_width))

#         tensor_conv2d_cuda[
#             (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, THREADS_PER_BLOCK
#         ](*out.tuple(), out.size, *input.tuple(), *weight.tuple(), False)
#         return out

#     @staticmethod
#     def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
#         input, weight = ctx.saved_values
#         batch, in_channels, in_height, in_width = input.shape
#         out_channels, _, kernel_height, kernel_width = weight.shape

#         grad_weight = grad_output.zeros(weight.shape)
#         grad_input = input.zeros(input.shape)

#         # Gradient with respect to weights
#         tensor_conv2d_cuda[
#             (grad_weight.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
#             THREADS_PER_BLOCK,
#         ](
#             *grad_weight.tuple(),
#             grad_weight.size,
#             *input.tuple(),
#             *grad_output.tuple(),
#             True,
#         )

#         # Gradient with respect to input
#         tensor_conv2d_cuda[
#             (grad_input.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
#             THREADS_PER_BLOCK,
#         ](
#             *grad_input.tuple(),
#             grad_input.size,
#             *grad_output.tuple(),
#             *weight.tuple(),
#             True,
#         )

#         return grad_input, grad_weight


# tensor_conv2d_cuda_fun = Conv2dCudaFun.apply

# conv1d_cuda = Conv1dCudaFun.apply
# conv2d_cuda = Conv2dCudaFun.apply
