import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ray

# URL del dataset
dataset_url = 'https://archive.ics.uci.edu/static/public/519/data.csv'

# Selezionare le caratteristiche per il clustering (escludere 'ca_cervix')
features = ['serum_sodium', 'creatinine_phosphokinase']

def plot_cluster(dataset, labels, centroids):
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="*", color='red')
    plt.title('Visualizzazione dei Cluster')
    plt.show()

def choose_centroids(dataset_scaled, k):
    return dataset_scaled[np.random.choice(dataset_scaled.shape[0], size=k, replace=False)]

def prepare_data(dataset_url, features):
    dataset = pd.read_csv(dataset_url)
    dataset.dropna(inplace=True)
    clean_dataset = dataset[features]
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(clean_dataset)
    return dataset_scaled

def ecluidean(data_point, centroids):
    return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

@ray.remote
def kmeans_map(points, centroids):
    map_results = []
    for point in points:
        distances = ecluidean(point, centroids)
        cluster = np.argmin(distances)
        map_results.append((cluster, (point, 1)))
    return map_results

@ray.remote
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
    global dataset_url, features

    # Prepara i dati
    dataset_scaled = prepare_data(dataset_url, features)
    k_max = 19
    k = int(input("Inserisci il numero di cluster (k): "))

    # Inizializza i centroidi
    centroids = choose_centroids(dataset_scaled, k)

    # Inizializza Ray
    ray.init()

    while True:
        # Fase di mappatura
        map_futures = [kmeans_map.remote(partition, centroids) for partition in np.array_split(dataset_scaled, ray.cluster_resources()['CPU'])]
        map_results = ray.get(map_futures)
        
        # Shuffling dei risultati
        reduce_inputs = [[] for _ in range(k)]
        for result in map_results:
            for element in result:
                cluster_id = element[0]
                point = element[1][0]
                reduce_inputs[cluster_id].append(point)

        # Fase di riduzione
        reduce_futures = [kmeans_reduce.remote(i, reduce_inputs[i]) for i in range(len(reduce_inputs))]
        reduce_results = ray.get(reduce_futures)

        # Calcola nuovi centroidi
        new_centroids = calculate_new_centroids(reduce_results)

        # Confronta i nuovi centroidi con quelli precedenti
        if np.array_equal(new_centroids, centroids):
            break
        else:
            centroids = new_centroids

    ray.shutdown()

    # Calcola etichette finali dei cluster
    final_labels = []
    for point in dataset_scaled:
        distances = np.linalg.norm(centroids - point, axis=1)
        cluster = np.argmin(distances)
        final_labels.append(cluster)

    # Visualizza i cluster
    plot_cluster(dataset_scaled, final_labels, centroids)

# Esegui K-Means
kmeans()
