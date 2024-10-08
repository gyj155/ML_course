import numpy as np


def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the Mean Absolute Error (MAE) at w.

    Args:
        y: shape=(N, ). The vector of true labels.
        tx: shape=(N,2). The matrix of input data, where each row corresponds to a data point and each column corresponds to a feature.
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    # 计算预测误差
    e = y - np.dot(tx, w)
    # 计算MAE的次梯度
    subgradient = -np.dot(tx.T, np.sign(e)) / len(e)
    return subgradient
