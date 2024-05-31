import numpy as np
from sklearn.datasets import make_blobs

def create_points(n_samples, n_features, n_clusters, random_state, memmap_file='data.dat'):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    
    return X, y