import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import gc
import ray
import time
from create_points import create_points
from write_on_file import write_on_file
import os

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
    fig=plt.figure()
    #per grafico 3D
    ax = fig.add_subplot(projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1],dataset[:,2], alpha=0.05,c=labels, cmap='summer')   
    #plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='viridis')
    ax.scatter(centroids[:, 0], centroids[:, 1],centroids[ :, 2], alpha=1, marker="*", color='red')
    plt.title('Cluster')
    plt.show()

def choose_centroids(dataset_scaled, k):
    np.random.seed(3)
    #seleziono k centroidi randomici prendendo dei punti dal dataset (torna le righe, ossia i punti selezionati)
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
    #splitto il dataset in n sub-array
    partitions = np.array_split(data, num_partition)
    return partitions

def euclidean(data_point, centroids):
    #calcolo distanza euclidea (centroide-punto)^2 (torna una lista con le distanze del punto da tutti i centroidi)
    return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

@ray.remote(scheduling_strategy="SPREAD")
def kmeans_map(points, centroids):
    map_results = []
    for point in points:
        #calcolo la distanza euclidea tra il punto e tutti i centroidi (torna una lista di distanze)
        distances = euclidean(point, centroids)
        #recupero quella più piccola (torna l'indice)
        cluster = np.argmin(distances)
        #appendo alla lista di risultati
        map_results.append((cluster, (point, 1)))
    return map_results

@ray.remote(scheduling_strategy="SPREAD")
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
    #Scarico i dati 
    #dataset_scaled = prepare_data(dataset_url, features)
    k = int(input("Inserisci il numero di cluster (k): "))
    n_punti = 15000000
    dataset_scaled, _ = create_points(n_samples=n_punti, n_features=3, n_clusters=k, random_state=42)

    k_max = 19
    n_MAP = 30

    #elbow_plot(dataset_scaled, k_max)
    n_REDUCE=k

    # Scelgo i centroidi
    centroids = choose_centroids(dataset_scaled, k)
    

    #SPLIT
    partitions=split(dataset_scaled,n_MAP)
    del dataset_scaled
    gc.collect()
    print(f"Numero di partizioni {len(partitions)}")

    os.environ['RAY_memory_monitor_refresh_ms'] = '0'
    ray.init()
    init=time.time()
    v = 0
    tol = 1e-3
    while True:
        v += 1
        print(f"Ciclo {v}")
        #MAP
        map_futures = [kmeans_map.remote(partition, centroids) for partition in partitions]
        map_results = ray.get(map_futures)
        del map_futures

        # SHUFFLING
        reduce_inputs = [[] for _ in range(n_REDUCE)]
        for result in map_results:
            for i in range(len(result)):
                cluster_id = result[0][0]
                point = result[0][1][0]
                reduce_inputs[cluster_id].append(point)
                del result[0] 
        del map_results

        # REDUCE
        reduce_futures = [kmeans_reduce.remote(i, reduce_inputs[i]) for i in range(len(reduce_inputs))]
        del reduce_inputs
        
        reduce_results = ray.get(reduce_futures)
        del reduce_futures

        #print(reduce_results)

        # Calcola nuovi centroidi
        new_centroids = calculate_new_centroids(reduce_results)
        del reduce_results

        centroid_shift = euclidean(new_centroids, centroids).mean()
        print(f"Variazione dei centroidi {centroid_shift}")

        if centroid_shift < tol:
            break
        else:
            centroids = new_centroids
        
        #garbage collect
        gc.collect()


    end=time.time()

    print(f"Tempo di esecuzione: {end-init} secondi")

    # Calcola le distanze finali dei cluster 
    '''final_labels = []
    for point in dataset_scaled:
        distances = euclidean(point, centroids)
        cluster = np.argmin(distances)
        final_labels.append(cluster)'''

    write_on_file("tempi.csv",n_punti,k,end-init,"Distribuito",3,42, n_MAP)
    # Visualizza i cluster
    #plot_cluster(dataset_scaled, final_labels, centroids)



# Esegui K-Means
kmeans()