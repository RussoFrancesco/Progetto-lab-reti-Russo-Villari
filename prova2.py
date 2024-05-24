import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# Carica i dati
data_url = 'https://archive.ics.uci.edu/static/public/537/data.csv'
data = pd.read_csv(data_url)

# Rimuovi eventuali valori mancanti
data.dropna(inplace=True)

# Seleziona le caratteristiche per il clustering
features = [
    'behavior_eating', 'behavior_personalHygiene', 'intention_aggregation',
    'intention_commitment', 'attitude_consistency', 'attitude_spontaneity',
    'norm_significantPerson', 'norm_fulfillment', 'perception_vulnerability',
    'perception_severity', 'motivation_strength', 'motivation_willingness',
    'socialSupport_emotionality', 'socialSupport_appreciation',
    'socialSupport_instrumental', 'empowerment_knowledge',
    'empowerment_abilities', 'empowerment_desires'
]

# Separare le caratteristiche
X = data[features]

elbow_plot(X,18)

# Standardizzazione dei dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# Riduzione della dimensionalità con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Esegui KMeans con 3 cluster
kmeans = KMeans(n_clusters=4,random_state=0)
kmeans.fit(X_scaled)

# Aggiungi i cluster al dataframe originale
data['Cluster'] = kmeans.labels_

# Visualizza i cluster
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="red",marker="*")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Visualizzazione dei Cluster')
plt.show()
