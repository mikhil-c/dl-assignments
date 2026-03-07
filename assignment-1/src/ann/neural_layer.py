"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from . import activations

class Layer:
    def __init__(self, d_prev, d_curr, weight_init):
        if weight_init == "random":
            self.W = np.random.randn(d_prev, d_curr)
        elif weight_init == "xavier":
            self.W = np.random.randn(d_prev, d_curr) * np.sqrt(1 / d_prev)

        self.b = np.zeros((1, d_curr))
        self.x = None # input from the previous layer
        self.a = None # activation values computed at the current layer
        self.local_grad = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, x_input, activation_function):
        self.x = x_input
        z = self.x @ self.W + self.b
        self.a = activations.forward(activation_function, z)
        return self.a

    def backward(self, incoming_local_grad, weight_decay, activation_function="identity", W_next=None):
        if W_next is None: # This gets executed for the last output layer
            self.local_grad = incoming_local_grad # This incoming_local_grad is the local_grad at the output layer
        else:
            self.local_grad = (incoming_local_grad @ W_next.T) * activations.backward(activation_function, self.a)
        batch_size = self.x.shape[0]
        self.grad_W = (self.x.T @ self.local_grad) / batch_size + (weight_decay * self.W)
        self.grad_b = np.sum(self.local_grad, axis=0, keepdims=True) / batch_size
        return self.local_grad
