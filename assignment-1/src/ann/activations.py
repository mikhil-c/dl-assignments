"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

def forward(activation_function, z): # forward computes the activation values
    """
        z: numpy array of shape (batch_size, dim)
    """

    if activation_function == "relu":
        return np.maximum(0, z)

    elif activation_function == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # To avoid overflows, the value of z is capped between -500 to 500 for numerical stability

    elif activation_function == "tanh":
        return np.tanh(z)

    elif activation_function == "softmax":
        shifted_z = z - np.max(z, axis=1, keepdims=True) # Shifting doesn't change the output; to avoid overflows, z is shifted so the max exp value is e^0
        exp_z = np.exp(shifted_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    elif activation_function == "identity":
        return z


def backward(activation_function, a): # backward computes the derivative values
    if activation_function == "relu":
        return (a > 0).astype(float) # z > 0 creates a matrix with entires True/False and here astype casts them to floats 

    elif activation_function == "sigmoid":
        return a * (1 - a)

    elif activation_function == "tanh":
        return 1 - a * a

    elif activation_function == "softmax":
        b, d = a.shape
        I = np.eye(d)
        return a[:, :, None] * (I - a[:, None, :])
