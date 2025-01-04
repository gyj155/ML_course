# -*- coding: utf-8 -*-
"""Some helper functions."""
import os
import shutil
import numpy as np
from matplotlib.pyplot import imread


def load_data():
    """Load data and convert it to the metrics system."""
    path_dataset = "faithful.csv"
    data = np.loadtxt(path_dataset, delimiter=" ", skiprows=0)
    return data


def normalize_data(data):
    """normalize the data by (x - mean(x)) / std(x)."""
    mean_data = np.mean(data, axis=0)
    data = data - mean_data
    std_data = np.std(data)
    data = data / std_data
    return data


def build_dir(dir):
    """build a new dir. if it exists, remove it and build a new one."""
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def load_image(path):
    """use the scipy.misc to load the image."""
    return imread(path)


from einops import rearrange
def build_distance_matrix(data, mu):
    """builds a distance matrix.

    Args:
        data: numpy array of shape = (N, d). original data.
        mu:   numpy array of shape = (k, d). Each row corresponds to a cluster center.
    Returns:
        numpy array of shape (N, k):
            squared distances matrix,
            the value row i column j corresponds to the squared distance of datapoint i with cluster center j.
    """
    
    reshaped_data = rearrange(data, 'n d -> n 1 d')
    reshaped_mu = rearrange(mu, 'k d -> 1 k d')
    distance_matrix = np.sum((reshaped_data - reshaped_mu) ** 2, axis=2)
    return distance_matrix