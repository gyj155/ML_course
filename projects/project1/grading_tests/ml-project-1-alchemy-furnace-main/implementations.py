import numpy as np
from batch_iteration import *

def initial_guess(tx, random_vector=False):
    '''
    Generate an initial guess of w.

    Parameters:
        - tx: (N, D) matrix
    
    Return:
        - w: (D,) array
    '''
    D = tx.shape[1]
    
    if random_vector:
        w = np.random.rand(D)
    else:
        w = np.zeros(D)
    
    return w

def train_val_split(x, y, train_size):
    '''
    Split the dataset into training and validation sets.

    Parameters:
     - x: feature array
     - y: label array
     - train_size: float, proportion of the dataset to include in the train split

     Returns:
        - x_tr: training feature set
        - y_tr: training label set
        - x_val: validation feature set
        - y_val: validation label set
    '''

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    split_index = int(train_size * len(indices))

    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    x_tr = x[train_indices]
    y_tr = y[train_indices]
    x_val = x[val_indices]
    y_val = y[val_indices]

    return x_tr, y_tr, x_val, y_val

def mse_loss(y, tx, w):
    '''
    Compute the mse loss.

    Parameters:
        - tx: (N, D) matrix
        - y: (N,) array
        - w: (D,) array
        - lambda_: regularization para.
        - regularize: boolean

    Return:
        - mse_loss
    '''
    N = tx.shape[0]

    e = y - tx @ w

    mse_loss = 1 / (2 * N) * e.T @ e

    return mse_loss

def mse_gradient(y, tx, w):
    '''
    Compute the mse gradient.

    Parameters:
        - tx: (N, D) matrix
        - y: (N,) array
        - w: (D,) array
        - lambda_: regularization para.
        - regularize: boolean

    Return:
        - grad: gradient
    '''
    N = tx.shape[0]
    e = y - tx @ w

    grad = -1 / N * tx.T @ e

    return grad

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    '''
    Gradient descent

    Parameters:
        - tx: (N, D) matrix
        - y: (N,) array
        - init_w: (D,) array, the initial guess of w
        - lambda_: regularization para.
        - regularize: boolean
        - max_iters
        - gamma: the learning rate

    Return:
        - losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        - ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    '''
    # ws = [init_w]
    # losses = []

    w = initial_w
    loss = mse_loss(y, tx, w)
    for n_iter in range(max_iters):
        grad = mse_gradient(y, tx, w)
        

        w = w - gamma * grad
        loss = mse_loss(y, tx, w)

        # ws.append(w)
        # losses.append(loss)

        print(f"GD iter: {n_iter+1} / {max_iters},  loss: {loss}")
        
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=None, shuffle=True):
    """The Stochastic Gradient Descent algorithm (SGD).

    Parameters:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        lambda_: regularization para.
        regularize: boolean
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        shuffle: a boolean

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    if batch_size is None:
        batch_size = y.shape[0]

    w = initial_w
    loss = mse_loss(y, tx, w)
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, shuffle):
            grad = mse_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
            loss = mse_loss(y, tx, w)   
        print(f"SGD iter: {n_iter+1} / {max_iters},  loss: {loss}")

    return w, loss

def least_squares(y, tx):
    '''
    Compute the least sqr. solution and mse.

    Parameters:
        - y: (N,) array
        - tx: (N, D) matrix
    
    Return:
        - w: (D,) matrix
    '''
    N = y.shape[0]

    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    mse = 1 / (2 * N) * (y - tx @ w).T @ (y - tx @ w)

    return w, mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    N = tx.shape[0]
    D = (tx.T).shape[0]

    lambda_1 = 2 * N * lambda_
    # Identity matrix of size D
    identity_matrix = np.eye(D)

    # Compute the ridge regression weights
    txTtx = tx.T @ tx + lambda_1 * identity_matrix
    w = np.linalg.inv(txTtx) @ tx.T @ y
    mse = 1 / (2 * N) * (y - tx @ w).T @ (y - tx @ w)

    return w, mse

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    sigmoid = 1 + np.exp(-t)
    sigmoid = 1 / sigmoid

    return sigmoid

def logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """

    z = tx @ w
    sigma = sigmoid(z)

    loss = -np.mean(y * np.log(sigma) + (1 - y) * np.log(1 - sigma))

    return loss

def logistic_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """

    N = y.shape[0]

    z = tx @ w
    sigma = sigmoid(z)


    grad = 1 / N * tx.T @ (sigma - y)

    return grad

def logistic_regression(y, tx, init_w, max_iters, gamma):
    '''
    Gradient descent

    Parameters:
        - tx: (N, D) matrix
        - y: (N,) array
        - init_w: (D,) array, the initial guess of w
        - lambda_: regularization para.
        - regularize: boolean
        - max_iters
        - gamma: the learning rate

    Return:
        - losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        - ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    '''

    w = init_w
    loss = logistic_loss(y, tx, w)
    for n_iter in range(max_iters):
        grad = logistic_gradient(y, tx, w)
        w = w - gamma * grad
        loss = logistic_loss(y, tx, w)
        if n_iter % 100 == 99:
            print(f"GD iter: {n_iter+1} / {max_iters},  loss: {loss}")
    return w, loss

def logistic_regression_sgd(y, tx, initial_w, batch_size, max_iters, gamma, shuffle):
    """The Stochastic Gradient Descent algorithm (SGD).

    Parameters:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        lambda_: regularization para.
        regularize: boolean
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        shuffle: a boolean

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    w = initial_w
    loss = np.inf

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, shuffle):
            grad = logistic_gradient(minibatch_y, minibatch_tx, w)
            loss = logistic_loss(minibatch_y, minibatch_tx, w)

            w = w - gamma * grad
        if n_iter % 100 == 99:
            print(f"SGD iter: {n_iter+1} / {max_iters},  loss: {loss}")

    return w, loss

def logistic_gradient_with_reg(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    """
    grad = logistic_gradient(y, tx, w)

    grad += 2 * lambda_ * w

    return grad

def reg_logistic_regression(y, tx, lambda_, init_w, max_iters, gamma):
    '''
    Gradient descent

    Parameters:
        - tx: (N, D) matrix
        - y: (N,) array
        - init_w: (D,) array, the initial guess of w
        - lambda_: regularization para.
        - regularize: boolean
        - max_iters
        - gamma: the learning rate

    Return:
        - losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        - ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    '''

    w = init_w
    loss = logistic_loss(y, tx, w)
    for n_iter in range(max_iters):
        grad = logistic_gradient_with_reg(y, tx, w, lambda_)
        w = w - gamma * grad
        loss = logistic_loss(y, tx, w)
        if n_iter % 100 == 99:
            print(f"GD with reg iter: {n_iter+1} / {max_iters},  loss: {loss}")
        
    return w, loss

def reg_logistic_regression_sgd(y, tx, lambda_, initial_w, batch_size, max_iters, gamma, shuffle):
    """The Stochastic Gradient Descent algorithm (SGD).

    Parameters:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        lambda_: regularization para.
        regularize: boolean
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        shuffle: a boolean

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    w = initial_w
    loss = np.inf
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, shuffle):
            grad = logistic_gradient_with_reg(minibatch_y, minibatch_tx, w, lambda_)
            loss = logistic_loss(minibatch_y, minibatch_tx, w)

            w = w - gamma * grad
        if n_iter % 100 == 99:
            print(f"SGD with reg iter: {n_iter+1} / {max_iters},  loss: {loss}")

    return w, loss

def reg_logistic_regression_sgd_with_momentum(y, tx, lambda_, initial_w, batch_size, max_iters, gamma, beta, shuffle):
    """The Stochastic Gradient Descent algorithm (SGD).

    Parameters:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        lambda_: regularization para.
        regularize: boolean
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        shuffle: a boolean

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    w = initial_w
    m = np.zeros_like(w)
    loss = np.inf
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, shuffle):
            grad = logistic_gradient_with_reg(minibatch_y, minibatch_tx, w, lambda_)
            loss = logistic_loss(minibatch_y, minibatch_tx, w)

            m = beta * m + (1 - beta) * grad
            w = w - gamma * m
        if n_iter % 100 == 99:
            print(f"SGD with momentum with reg iter: {n_iter+1} / {max_iters},  loss: {loss}")

    return w, loss

def reg_logistic_regression_adam(y, tx, lambda_, initial_w, batch_size, max_iters, gamma, beta_1, beta_2,  shuffle):
    """The Stochastic Gradient Descent algorithm (SGD).

    Parameters:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        lambda_: regularization para.
        regularize: boolean
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        shuffle: a boolean

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    w = initial_w
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    loss = np.inf

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, shuffle):
            grad = logistic_gradient_with_reg(minibatch_y, minibatch_tx, w, lambda_)
            loss = logistic_loss(minibatch_y, minibatch_tx, w)

            m = beta_1 * m + (1 - beta_1) * grad
            v = beta_2 * v + (1 - beta_2) * (grad ** 2)
            w = w - gamma / (np.sqrt(v) + 1e-10) * m
        if n_iter % 100 == 99:
            print(f"ADAM with reg iter: {n_iter+1} / {max_iters},  loss: {loss}")

    return w, loss












