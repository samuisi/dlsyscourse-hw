"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            param_id = id(param)
            grad = ndl.Tensor(
                    param.grad.cached_data + self.weight_decay * param.data.cached_data,
                    dtype=param.dtype)
            self.u[param_id] = self.momentum * self.u.get(param_id, 0) + (1-self.momentum) * grad
            param.data = param.data - self.lr * self.u[param_id]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            param_id = id(param)
            grad = ndl.Tensor(
                    param.grad.cached_data + self.weight_decay * param.data.cached_data,
                    dtype=param.dtype)
            self.m[param_id] = (self.beta1 * self.m.get(param_id, 0) + (1-self.beta1) * grad).data
            self.v[param_id] = (self.beta2 * self.v.get(param_id, 0) + (1-self.beta2) * ((grad.data) ** 2)).data
            u_hat = self.m[param_id] / (1 - (self.beta1 ** self.t))
            v_hat = self.v[param_id] / (1 - (self.beta2 ** self.t))

            param.data = param.data - self.lr * u_hat / (v_hat ** 0.5 + self.eps)

        ### END YOUR SOLUTION
