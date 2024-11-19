import numpy as np

from implementations import sigmoid

def compute_y_pred(tx, w, alpha):

    print(f"Predicting...")

    y_pred = np.where(sigmoid(tx @ w) >= alpha, 1, -1)

    return y_pred

def calculate_accuracy(y_pred, y_true):
    '''
    Calculate the accuracy.

    Parameters:
        - y_pred: (N,) array
        - y_true: (N,) array

    Return:
        - acc: accuracy
    '''

    correct_predictions = np.sum(y_pred == y_true)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions


    return accuracy

def calculate_f1_score(y_pred, y_true):
    
    # Compute TP, FP and FN.
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == -1))
    FN = np.sum((y_pred == -1) & (y_true == 1))

    # Compute precision and recall.
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    # Compute F1 Score
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

def training_summary(tx, w, alpha, y_true):
    y_pred = compute_y_pred(tx, w, alpha)
    acc = calculate_accuracy(y_pred, y_true)
    f1_score = calculate_f1_score(y_pred, y_true)

    return acc, f1_score

