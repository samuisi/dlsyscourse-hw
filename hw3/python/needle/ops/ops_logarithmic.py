from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=1, keepdims=True)
        log_sum_exp = array_api.log(array_api.exp(Z - max_z).sum(axis=1, keepdims=True)) + max_z

        return Z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        sum_out_grad = reshape(summation(out_grad, axes=(1,)), (node.shape[0], 1))

        return (out_grad - exp(node) * broadcast_to(sum_out_grad, out_grad.shape),)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=self.axes, keepdims=True)
        return array_api.log(array_api.exp(Z - max_z).sum(axis=self.axes)) + max_z.squeeze(axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        
        if self.axes is None:
            axes = tuple(range(len(Z.shape)))
        elif isinstance(self.axes, int):
            axes = tuple(self.axes)
        else:
            axes = self.axes

        boardcast_shape = list(Z.shape)
        for axis in axes:
            boardcast_shape[axis] = 1
        boardcast_shape = tuple(boardcast_shape)

        node_reshaped = reshape(node, boardcast_shape)
        out_grad_shaped = reshape(out_grad, boardcast_shape)

        return (out_grad_shaped * exp(Z - broadcast_to(node_reshaped, Z.shape)),)
        
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
