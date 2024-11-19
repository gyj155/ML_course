import numpy as np

def batch_iter(y, tx, batch_size, shuffle, num_batches=1):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


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
        batch_size = 1

    w = initial_w
    loss = mse_loss(y, tx, w)
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, shuffle):
            grad = mse_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
            loss = mse_loss(y, tx, w)   
        print(f"SGD iter: {n_iter+1} / {max_iters},  loss: {loss}")

    return w, loss

y = np.array([0.1, 0.3, 0.5])
tx = np.array([[2.3, 3.2],
       [1. , 0.1],
       [1.4, 2.3]])
initial_w = np.array([0.5, 1. ])
# expected_loss = 0.844595
# expected_w = np.array([0.063058, 0.39208])
print(mean_squared_error_sgd(y[:1], tx[:1], initial_w, 2, 0.1))
#(array([0.0630575, 0.39208  ]), 2.761837531250002)