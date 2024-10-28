# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - np.dot(tx, w)
    loss = 1/2 * np.mean(e**2) / len(y)
    return loss

def compute_mae(y, tx, w):
    e = y - np.dot(tx, w)
    loss = np.mean(np.abs(e))
    return loss

def compute_mse(y, tx, w):
    e = y - np.dot(tx, w)
    loss = 1/2 * np.mean(e**2) / len(y)
    return loss

