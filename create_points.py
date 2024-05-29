import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


def create_points(n_samples, n_features, random_state, n_clusters):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    return X, y


