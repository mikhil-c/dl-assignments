"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

def forward(y_pred, y_true, loss_type, weight_decay, l2_norm):
    batch_size = y_pred.shape[0]
    if loss_type == "cross_entropy":
        loss = -np.sum(y_true * np.log(y_pred + 1e-12)) / batch_size # Adding 1e-12 just to make sure that input to log is non-zero
    elif loss_type == "mean_squared_error":
        loss = np.mean((y_pred - y_true) ** 2)

    loss += (weight_decay / 2) * l2_norm
    return loss


def backward(y_pred, y_true, loss_type):
    dim = y_pred.shape[1]
    if loss_type == "cross_entropy":
        derivate = -(y_true / (y_pred + 1e-12))
    elif loss_type == "mean_squared_error":
        derivate = 2 / dim * (y_pred - y_true)

    return derivate
