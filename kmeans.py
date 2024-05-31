import numpy as np
import gc
import ray
import time
import os
from create_points import create_points
from write_on_file import write_on_file

def choose_centroids(dataset_scaled, k):
    np.random.seed(3)
    return dataset_scaled[np.random.choice(dataset_scaled.shape[0], size=k, replace=False)]

def split(data, num_partition):
    partitions = np.array_split(data, num_partition)
    return partitions

def euclidean(data_point, centroids):
    return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

@ray.remote(scheduling_strategy="SPREAD")
def kmeans_map(points, centroids):
    map_results = []
    for point in points:
        distances = euclidean(point, centroids)
        cluster = np.argmin(distances)
        map_results.append((cluster, (point, 1)))
    return map_results

@ray.remote(scheduling_strategy="SPREAD")
def kmeans_reduce(cluster, points):
    sum_points = np.sum(points, axis=0)
    num_points = len(points)
    return cluster, sum_points, num_points

def calculate_new_centroids(reduce_results):
    new_centroids = []
    for cluster, sum_points, num_points in reduce_results:
        centroid = sum_points / num_points
        new_centroids.append(centroid)
    return np.array(new_centroids)

def kmeans():
    k = int(input("Inserisci il numero di cluster (k): "))
    n_punti = 15000000
    
    # Usa create_points per generare e salvare il dataset
    memmap_file, _ = create_points(n_samples=n_punti, n_features=3, n_clusters=k, random_state=42)

    # Carica i dati usando memmap
    dataset_scaled = np.memmap(memmap_file, dtype='float32', mode='r', shape=(n_punti, 3))

    k_max = 19
    n_MAP = 30
    n_REDUCE = k

    centroids = choose_centroids(dataset_scaled, k)
    partitions = split(dataset_scaled, n_MAP)
    del dataset_scaled
    gc.collect()
    print(f"Numero di partizioni {len(partitions)}")

    os.environ['RAY_memory_monitor_refresh_ms'] = '0'
    ray.init()
    init = time.time()
    v = 0
    tol = 1e-3

    while True:
        v += 1
        print(f"Ciclo {v}")

        map_futures = [kmeans_map.remote(partition, centroids) for partition in partitions]
        map_results = ray.get(map_futures)
        del map_futures

        reduce_inputs = [[] for _ in range(n_REDUCE)]
        for result in map_results:
            for element in result:
                cluster_id = element[0]
                point = element[1][0]
                reduce_inputs[cluster_id].append(point)
        
        del map_results

        reduce_futures = [kmeans_reduce.remote(i, reduce_inputs[i]) for i in range(len(reduce_inputs))]
        del reduce_inputs
        
        reduce_results = ray.get(reduce_futures)
        del reduce_futures

        new_centroids = calculate_new_centroids(reduce_results)
        del reduce_results

        centroid_shift = euclidean(new_centroids, centroids).mean()
        print(f"Variazione dei centroidi {centroid_shift}")

        if centroid_shift < tol:
            break
        else:
            centroids = new_centroids
        
        gc.collect()

    ray.shutdown()

    end=time.time()

    print(f"Tempo di esecuzione: {end-init} secondi")

    # Calcola le distanze finali dei cluster 
    final_labels = []
    for point in dataset_scaled:
        distances = euclidean(point, centroids)
        cluster = np.argmin(distances)
        final_labels.append(cluster)

    write_on_file("tempi.csv",n_punti,k,end-init,"Distribuito",3,42, n_MAP)
    # Visualizza i cluster
    #plot_cluster(dataset_scaled, final_labels, centroids)



# Esegui K-Means
kmeans()
