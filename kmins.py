import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ray


# X=set di dati

dataset_url = 'https://archive.ics.uci.edu/static/public/537/data.csv'
# Selezionare le caratteristiche per il clustering (escludere 'ca_cervix')
features = [
    'behavior_eating'
]


def create_dataset(dataset_url,features):
    #leggo file csv
    dataset=pd.read_csv(dataset_url)
    
    #rimuovo record con campi vuoti 
    dataset.dropna(inplace=True)

    #prendo solo e feature analizzate per il k means
    clean_dataset=dataset[features]

    #normalizzo i dati
    scaler=StandardScaler()
    dataset_scaled = scaler.fit_transform(clean_dataset)

    return dataset_scaled,dataset


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

def choose_centroids(data,k):
    centroids=np.random.uniform(low=np.amin(data), high=np.amax(data), size=(k,data.shape[1]))
    return centroids


#dividi i punti in una lista di punti
def split(data,num_partition):
    partitions = np.array_split(data, num_partition)
    return partitions
    
def ecluidean(data_point, centroids):
    return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

@ray.remote
def kmeansMap(points,centroids):
    map_res=[]

    for point in points:
        #calcolo la distanza euclidea tra il punto e tutti i centroidi (torna una lista di distanze)
        distances=ecluidean(point,centroids)
        #recupero quella più piccola (torna l'indice)
        cluster=np.argmin(distances)
        #appendo alla lista di risultati
        map_res.append((cluster,(point,1)))
        del distances
    
    return map_res





@ray.remote
def kmeansReduce(cluster,points):
    #ricevo: cluster=int, points=[(x0,y0),(x1,y1),..., (xn,yn)] 
    sumX=0
    sumY=0

    for point in points:
        sumX+=point[0]
        sumY+=point[1]

    return (cluster,(sumX,sumY))

def calculate_new_centroids(reduce_results):
    new_centroids=[]
    
    for i in range(len(reduce_results)):
        #ricevo: [(indice_cluster, (sommatoria_punti, n_punti)),...]
        print(reduce_results[i])
        cluster_id=reduce_results[i][0]
        x=reduce_results[i][1][1]
        S=reduce_results[i][1][0]

        new_centroids.append(x/abs(S))
        
    return np.array(new_centroids)
    


def kmins():
    global dataset_url
    global features
    k_max=19
    n_MAP=5
    

    #leggo file csv
    dataset_scaled,_=create_dataset(dataset_url,features)
    #elbow_plot(dataset_scaled,k_max)
    k=int(input("inserisci il k: "))
    n_REDUCE=k

    centroids=choose_centroids(dataset_scaled,k)
    partitions =split(dataset_scaled,n_MAP)
    
    ray.init()
    while True:
        
        map_futures=[kmeansMap.remote(partition,centroids) for partition in partitions]
        map_results=ray.get(map_futures)
        #print(len(map_results))
        #map_results= [(k,(point,1)),...]

        #lista di input
        reduce_inputs=[list() for _ in range(n_REDUCE)]

        #SHUFFLING
        for result in map_results:
            for element in result:
                #id del cluster
                cluster_id=element[0]
                #tupla (x,1)
                point=element[1]
                #aggiungo il punto nella lista
                reduce_inputs[cluster_id].append(point)
        
        #REDUCE
        reduce_futures=[kmeansReduce.remote(i,reduce_inputs[i]) for i in range(len(reduce_inputs))]
        reduce_results=ray.get(reduce_futures)

        print(reduce_results)
        
        new_centroids=calculate_new_centroids(reduce_results)
        print(new_centroids)
        if new_centroids.all() == centroids.all():
            break
        else:
            centroids=new_centroids


    ray.shutdown()



kmins()