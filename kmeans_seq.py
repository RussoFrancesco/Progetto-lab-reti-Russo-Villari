import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import time

# URL del dataset
dataset_url = 'https://archive.ics.uci.edu/static/public/537/data.csv'

# Seleziono le caratteristiche per il clustering 
features = [
    'behavior_eating', 'behavior_personalHygiene', 'intention_aggregation'
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
    fig=plt.figure()
    #per grafico 3D
    ax = fig.add_subplot(projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1],dataset[:,2], c=labels, cmap='viridis')   
    #plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='viridis')
    ax.scatter(centroids[:, 0], centroids[:, 1],centroids[ :, 2], marker="*", color='red')
    plt.title('Cluster')
    plt.show()

def choose_centroids(dataset_scaled, k):
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


def euclidean(data_point, centroids):
    #calcolo distanza euclidea (centroide-punto)^2 (torna una lista con le distanze del punto da tutti i centroidi)
    return np.sqrt(np.sum((centroids - data_point)**2, axis=1))


def calculate_new_centroids(clusters):
    #clusters cluster0, cluster1, cluster2, ...
    #ricevo: [[points],[points], [points],...]
    new_centroids = []
    for cluster in clusters:
        sum_points = np.sum(cluster, axis=0)
        centroid = sum_points / len(cluster)
        new_centroids.append(centroid)
    return np.array(new_centroids)


def kmeans():
    '''PASSI :
    1) Determino centroidi
    2) Determino centroide con distanza minima per ogni x in X
    3) Metto x nell'insieme dei punti apparenenti al singolo centroide 
    4) Ripeto passi 2 e 3 finchè i centroidi non sono uguali su 2 iterazioni successive'''
    global dataset_url, features

    #Scarico i dati 
    dataset_scaled = prepare_data(dataset_url, features)
    
    k_max=19
    #elbow_plot(dataset_scaled, k_max)
    k = int(input("Inserisci il numero di cluster (k): "))
    

    # Scelgo i centroidi
    centroids = choose_centroids(dataset_scaled, k)

    init=time.time()

    while True:
        #Inizializziamo la lista di cluster in base al numero di cluster da calcolare
        clusters = [list() for _ in range(k)]
        #Creo un array di zero in base alle righe del dataset
        labels = np.zeros(dataset_scaled.shape[0])
        for i, point in enumerate(dataset_scaled):
            #Calcolo la distanza del punto dai centroidi
            distances = euclidean(point, centroids)
            #Recupero l'indice del centroide la cui distanza è minima
            cluster = np.argmin(distances)
            #Aggiungo il punto alla lista del cluster corrispondente
            clusters[cluster].append(point)
            #Assegno al punto i l'indice del cluster
            labels[i] = cluster

        # Calcola nuovi centroidi
        new_centroids = calculate_new_centroids(clusters)

        # Confronta i nuovi centroidi con quelli precedenti
        if np.array_equal(new_centroids, centroids):
            break
        else:
            for cluster in clusters:
                cluster.clear()
            centroids = new_centroids

    end=time.time()

    print(f"Tempo di esecuzione: {end-init} secondi")
   
    # Visualizza i cluster
    plot_cluster(dataset_scaled, labels, centroids)



# Esegui K-Means
kmeans()