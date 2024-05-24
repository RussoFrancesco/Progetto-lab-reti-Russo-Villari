import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ray
import time

# URL del dataset
dataset_url = 'https://archive.ics.uci.edu/static/public/537/data.csv'

# Seleziono le caratteristiche per il clustering 
features = [
    'behavior_eating', 'behavior_personalHygiene'
]
def elbow_plot(data,max_k):
    means=[]
    inertias=[]

    for k in range(1,max_k):
        k_means=KMeans(n_clusters=k)
        k_means.fit(data)

        means.append(k)
        inertias.append(k_means.inertia_)
    
    fig=plt.subplots(figsize=(10,5))
    plt.plot(means,inertias,'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


def plot_cluster(dataset, labels, centroids):    
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="*", color='red')
    plt.title('Cluster')
    plt.show()

def choose_centroids(dataset_scaled, k):
    return dataset_scaled[np.random.choice(dataset_scaled.shape[0], size=k, replace=False)]

def prepare_data(dataset_url, features):
    #leggo file csv
    dataset = pd.read_csv(dataset_url)
    
    #rimuovo record con campi vuoti 
    dataset.dropna(inplace=True)
    
    #prendo solo e feature analizzate per il k means
    clean_dataset = dataset[features]
    
    #normalizzo i dati
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(clean_dataset)

    return dataset_scaled

def split(data,num_partition):
    partitions = np.array_split(data, num_partition)
    return partitions

def ecluidean(data_point, centroids):
    #calcolo distanza euclidea (centroide-punto)^2 (torna una lista con le distanze del punto da tutti i centroidi)
    return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

@ray.remote
def kmeans_map(points, centroids):
    map_results = []
    for point in points:
        #calcolo la distanza euclidea tra il punto e tutti i centroidi (torna una lista di distanze)
        distances = ecluidean(point, centroids)
        #recupero quella più piccola (torna l'indice)
        cluster = np.argmin(distances)
        #appendo alla lista di risultati
        map_results.append((cluster, (point, 1)))
    return map_results

@ray.remote
def kmeans_reduce(cluster, points):
    #print(points)
    #somma tutti i punti
    sum_points = np.sum(points, axis=0)
    #numero di punti
    num_points = len(points)
    return cluster, sum_points, num_points

def calculate_new_centroids(reduce_results):
    #ricevo: [(indice_cluster, sommatoria_punti, n_punti),...]
    new_centroids = []
    for cluster, sum_points, num_points in reduce_results:
        centroid = sum_points / num_points
        new_centroids.append(centroid)
    return np.array(new_centroids)


def kmeans():
    global dataset_url, features

    #Scarico i dati 
    dataset_scaled = prepare_data(dataset_url, features)

    k_max = 19
    n_MAP = 5
    n_REDUCE=3

    elbow_plot(dataset_scaled, k_max)
    k = int(input("Inserisci il numero di cluster (k): "))
    n_REDUCE=k

    # Scelgo i centroidi
    centroids = choose_centroids(dataset_scaled, k)

    #SPLIT
    partitions =split(dataset_scaled,n_MAP)

    ray.init()
    init=time.time()
    while True:
        #MAP
        map_futures = [kmeans_map.remote(partition, centroids) for partition in partitions]
        map_results = ray.get(map_futures)
        
        # SHUFFLING
        reduce_inputs = [[] for _ in range(k)]
        for result in map_results:
            for element in result:
                cluster_id = element[0]
                point = element[1][0]
                reduce_inputs[cluster_id].append(point)

        # REDUCE
        reduce_futures = [kmeans_reduce.remote(i, reduce_inputs[i]) for i in range(len(reduce_inputs))]
        reduce_results = ray.get(reduce_futures)
        
        #print(reduce_results)

        # Calcola nuovi centroidi
        new_centroids = calculate_new_centroids(reduce_results)

        # Confronta i nuovi centroidi con quelli precedenti
        if np.array_equal(new_centroids, centroids):
            break
        else:
            centroids = new_centroids

    end=time.time()
    ray.shutdown()

    print(f"Tempo di esecuzione: {end-init} secondi")

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
