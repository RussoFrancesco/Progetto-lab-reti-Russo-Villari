import numpy as np
from sklearn.datasets import make_blobs

def create_points(n_samples, n_features, n_clusters, random_state, memmap_file='data.dat'):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    
    # Salva i dati in un file di memmap
    fp = np.memmap(memmap_file, dtype='float32', mode='w+', shape=(n_samples, n_features))
    fp[:] = X[:]
    del fp  # Scarica i dati su disco
    
    return memmap_file, y
