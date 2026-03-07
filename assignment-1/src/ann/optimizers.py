"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np

class SGD:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def update(self, param, grad, id):
        return param - (self.lr * grad)


class Momentum:
    def __init__(self, learning_rate, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.v = {}

    def update(self, param, grad, id):
        if id not in self.v:
            self.v[id] = np.zeros_like(param)
        
        self.v[id] = self.beta * self.v[id] + grad
        return param - (self.lr * self.v[id])


class NAG:
    def __init__(self, learning_rate, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.v = {}

    def update(self, param, grad, id):
        if id not in self.v:
            self.v[id] = np.zeros_like(param)
        
        v_prev = self.v[id]
        self.v[id] = self.beta * v_prev + grad
        return param - (self.lr * (self.beta * self.v[id] + grad))


class RMSProp:
    def __init__(self, learning_rate, beta=0.9, eps=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.eps = eps
        self.s = {}

    def update(self, param, grad, id):
        if id not in self.s:
            self.s[id] = np.zeros_like(param)
        
        self.s[id] = self.beta * self.s[id] + (1 - self.beta) * (grad**2)
        return param - (self.lr * grad) / (np.sqrt(self.s[id]) + self.eps)
