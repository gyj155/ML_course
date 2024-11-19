import argparse
import logging
import os
import json

import numpy as np

from helpers import *
from implementations import *
from data_cleaner import *
from prediction import *


def options(argv=None):
    parser = argparse.ArgumentParser(description='AlchemyFurnace')

    # io settings
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='path to the input dataset')
    parser.add_argument('--output_path', type=str, default='./model', help='path to the output folder')
    parser.add_argument('--log_path', type=str, default='./log', help='path to the log folder')
    parser.add_argument('--filename', type=str, default='prediction.csv', help='file name')

    # setting for cross validation
    parser.add_argument('--train_size', type=float, default=0.8, help='proportion of the dataset to include in the train split')
    # setting for data cleaning
    parser.add_argument('--threshold', type=float, default=1, help='deleting threshold')
    parser.add_argument('--subsample', type=bool, default=False, help='subsampling')
    parser.add_argument('--avg_thres', type=float, default=10, help='average threshold')
    parser.add_argument('--std_thres', type=float, default=3, help='std threshold')

    # setting for clipping
    parser.add_argument('--clipping', type=bool, default=False, help='Clipping')

    # setting for data augmentation
    parser.add_argument('--data_augmentation', type=bool, default=True, help='data augmentation')
    parser.add_argument('--copies', type=int, default=4, help='number of copies')

    # setting for noise
    parser.add_argument('--add_noise', type=bool, default=True, help='add noise to the training set')
    parser.add_argument('--noise_level', type=float, default=0.1, help='magnification of stds')

    # setting for training
    parser.add_argument('--read_w', type=bool, default=True, help='Read saved w')
    parser.add_argument('--random_vector', type=bool, default=False, help='random initial guess')
    parser.add_argument('--max_iters', type=int, default=100000, help='max iteration times')
    parser.add_argument('--gamma', type=float, default=5e-5, help='learning rate')

    # setting for regularization
    parser.add_argument('--regularize', type=bool, default=True, help='Regularization')
    parser.add_argument('--lambda_', type=float, default=1e-5, help='lambda')

    # setting for GD
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle')
    parser.add_argument('--sgd_with_momentum', type=bool, default=False, help='SGD with Momentum')
    parser.add_argument('--beta', type=float, default=0.8, help='momentum')
    parser.add_argument('--adam', type=bool, default=False, help='adam')
    parser.add_argument('--beta_1', type=float, default=0.9, help='beta_1')
    parser.add_argument('--beta_2', type=float, default=0.999, help='beta_2')

    # setting for prediction
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha')

    args = parser.parse_args(argv)
    return args

def cross_validation(args, x, y):

    x_tr, y_tr, x_val, y_val = train_val_split(x, y, args.train_size)

    return x_tr, y_tr, x_val, y_val

def clean_data(args, x_tr, y_tr):
    print('Cleaning data...')

    x_deleted = delete_NaN_features(x_tr, args.threshold)
    x_replaced = replace_NaN(x_deleted, args.avg_thres, args.std_thres)
    x_cleaned = z_score_normalize(x_replaced)
    
    y_labeled = labeling(y_tr)

    return x_cleaned, y_labeled

def read_w(args):
    print(f"Reading w...")

    w = np.loadtxt(args.output_path + f"/w.csv", delimiter=',')
    return w

def training(args, y, x):

    if args.clipping:
        x = clipping(x)

    if args.data_augmentation:
        x, y = data_augmentation(x, y, args.copies)

    if args.add_noise:
        x = add_noise(x, args.noise_level)

    tx = tilde_x(x)

    init_w = initial_guess(tx, args.random_vector)

    if args.read_w:
        init_w = read_w(args)

    print('Training...')

    if args.adam:
        w, loss =reg_logistic_regression_adam(y, tx, args.lambda_, init_w, args.batch_size, args.max_iters, args.gamma, args.beta_1, args.beta_2, args.shuffle)
    elif args.sgd_with_momentum:
        w, loss = reg_logistic_regression_sgd_with_momentum(y, tx, args.lambda_, init_w, args.batch_size, args.max_iters, args.gamma, args.beta, args.shuffle)
    else:
        if args.regularize:
            w, loss = reg_logistic_regression_sgd(y, tx, args.lambda_, init_w, args.batch_size, args.max_iters, args.gamma, args.shuffle)
        else:
            w, loss = logistic_regression_sgd(y, tx, init_w, args.batch_size, args.max_iters, args.gamma, args.shuffle)

    print(f"Training completed.")

    return w, loss

def compute_score(args, y_true, x, w):

    tx = tilde_x(x)

    acc, f1_score = training_summary(tx, w, args.alpha, y_true)

    print(f"acc.: {acc}")
    print(f"f1_score: {f1_score}")

    return acc, f1_score


def save_data(args, loss, w):
    # Save loss and w to file
    print(f"Saving data...")
    np.savetxt(args.output_path + '/loss.csv', [loss], delimiter=',')
    np.savetxt(args.output_path + f"/w.csv", w, delimiter=',')

    # Save parameters
    with open(args.output_path + '/args.json', 'w') as f:
        json.dump(vars(args), f)

    print(f"Data saved.")

def write_log(args, acc, f1_score):
    # Configure logging
    log_file_path = os.path.join(os.getcwd(), args.log_path + '/log_file.txt')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Log argparse parameters
    print(f"Logging experiment parameters...")
    logging.info(f"Threshold: {args.threshold}")
    logging.info(f"Subsample: {args.subsample}")
    logging.info(f"Avg Threshold: {args.avg_thres}")
    logging.info(f"Std Threshold: {args.std_thres}")
    logging.info(f"Clipping: {args.clipping}")
    logging.info(f"Data Augmentation: {args.data_augmentation}")
    if args.data_augmentation:
        logging.info(f"Number of copies: {args.copies}")
    
    logging.info(f"Train Size: {args.train_size}")
    logging.info(f"Random Vector: {args.random_vector}")
    logging.info(f"Max Iterations: {args.max_iters}")
    logging.info(f"Learning Rate: {args.gamma}")
    logging.info(f"ADAM: {args.adam}")
    if args.adam:
        logging.info(f"beta_1: {args.beta_1}")
        logging.info(f"beta_2: {args.beta_2}")
    elif args.sgd_with_momentum:
        logging.info(f"SGD with momentum: {args.sgd_with_momentum}")
        logging.info(f"beta: {args.beta}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Shuffle: {args.shuffle}")
    logging.info(f"Regularization: {args.regularize}")
    if args.regularize:
        logging.info(f"Lambda: {args.lambda_}")
    logging.info(f"Add noise: {args.add_noise}")
    if args.add_noise:
        logging.info(f"noise_level: {args.noise_level}")

    # Log accuracy and f1 score
    logging.info(f"acc.: {acc}")
    logging.info(f"F1 Score: {f1_score}")

    print('Logging saved.')
    

def main(args):
    # Load data
    x_tr, x_te, y_tr, tr_ids, te_ids = load_csv_data(args.dataset_path, args.subsample)

    x_tr_tr, y_tr_tr, x_val, y_val = cross_validation(args, x_tr, y_tr)

    x_cleaned, y_labeled = clean_data(args, x_tr_tr, y_tr_tr)

    w, loss = training(args, y_labeled, x_cleaned)

    x_val_cleaned, y_val_labeled = clean_data(args, x_val, y_val)

    acc, f1_score = compute_score(args, y_val, x_val_cleaned, w)

    save_data(args, loss, w)

    write_log(args, acc, f1_score)

    x_cleaned_te, y_labeled = clean_data(args, x_te, y_tr)

    tx_te = tilde_x(x_cleaned_te)

    y_pred = compute_y_pred(tx_te, w, args.alpha)

    # Create the submission
    create_csv_submission(te_ids, y_pred, args.filename)

    print(f"Prediction completed.")

if __name__ == '__main__':
    ARGS = options()
    main(ARGS)




