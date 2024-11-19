import numpy as np

def delete_NaN_features(x, threshold):
    '''
    Replace values in features where the proportion of NaN exceeds the threshold and nanstd neq 0 with 0.

    Parameters:
        - x: the set to be cleaned
        - threshold: [0, 1]

    Return:
        - x_deleted: the set after modification
    '''
    assert 0 <= threshold <= 1, "Threshold must be between 0 and 1"

    nan_percentage = np.isnan(x).mean(axis=0)
    nan_std = np.nanstd(x, axis=0)
    columns_to_modify = np.where((nan_percentage > threshold) & (nan_std != 0))[0]
    x_deleted = x.copy()
    x_deleted[:, columns_to_modify] = 0

    return x_deleted

def replace_NaN(x, avg_thres, std_thres):
    '''
    Replace NaN in each feature with 0 if nanmean < avg_thres and nanstd < std_thres and there is no 0 in the feature,
    otherwise replace NaN with nanmean of the feature.

    Parameters:
        - x: set to be processed
        - avg_thres: threshold for nanmean
        - std_thres: threshold for nanstd
    
    Return:
        - x_filled
    '''
    x_filled = x.copy()

    column_means = np.nanmean(x_filled, axis=0)
    column_std = np.nanstd(x_filled, axis=0)
    for i in range(x_filled.shape[1]):
        column = x_filled[:, i]
        if column_std[i] < std_thres and column_means[i] < avg_thres and 0 not in column:
            column[np.isnan(column)] = 0
        else:
            column[np.isnan(column)] = column_means[i]

    return x_filled

def z_score_normalize(x):
    '''
    Normalize the data set.

    Parameters:
        - x: set to be processed
    
    Return:
        - x_normalized
    '''
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)

    # Find indices of features with non-zero standard deviation
    non_zero_std_indices = np.where(x_std != 0)[0]

    # Normalize only the features with non-zero standard deviation
    x_normalized = np.zeros_like(x)
    x_normalized[:, non_zero_std_indices] = (x[:, non_zero_std_indices] - x_mean[non_zero_std_indices]) / x_std[non_zero_std_indices]

    return x_normalized

def clipping(x):
    '''
    Limit the data within three sd.

    Parameters:
        - x

    Return:
        - x_clipped
    '''
    mean = np.mean(x, axis=0)
    std_dev = np.std(x, axis=0)

    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev

    x_clipped = np.clip(x, lower_bound, upper_bound)

    return x_clipped

def data_augmentation(x_tr, y_tr, N):
    '''
    Copy multiple instances of the negative examples.

    Parameters:
        - x_tr
        - y_tr

    Returns:
        - x_resampled
        - y_resampled
    '''
    indices_0 = np.where(y_tr == 0)[0]
    indices_1 = np.where(y_tr == 1)[0]

    x_1_resampled = np.repeat(x_tr[indices_1], N, axis=0)
    y_1_resampled = np.repeat(y_tr[indices_1], N)

    x_0 = x_tr[indices_0]
    y_0 = y_tr[indices_0]

    x_resampled = np.vstack((x_1_resampled, x_0))
    y_resampled = np.concatenate((y_1_resampled, y_0))

    shuffle_indices = np.random.permutation(len(y_resampled))
    x_resampled = x_resampled[shuffle_indices]
    y_resampled = y_resampled[shuffle_indices]

    return x_resampled, y_resampled

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

def add_noise(x, noise_level):
    '''
    Produce noise for the training set.

    Parameters:
        - x: (N,D) matrix
        - noise_level: magnification of nanstd

    Returns:
        - x_noisy: (N,D) matrix
    '''

    stds = np.nanstd(x, axis=0)

    noise = np.random.normal(0, noise_level * stds, x.shape)

    x_noisy = x + noise

    return x_noisy


def tilde_x(x):
    '''
    Generate tilde_x.

    Parameters:
        - x: set to be processed

    Return:
        - tx: tx
    '''
    tx = np.insert(x, 0, 1, axis=1)
    return tx

def labeling(y):
    '''
    Transform y in {-1, 1} into y in {0, 1}.

    Parameters:
        - y: an np array containing only -1 and 1
    
    Returns:
        - y_labeled: an np array containing only 0 and 1
    '''

    y_labeled = (y + 1) / 2

    return y_labeled



