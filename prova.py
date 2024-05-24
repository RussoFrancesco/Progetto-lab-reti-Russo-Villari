import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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


# Caricare i dati
data_url = 'https://archive.ics.uci.edu/static/public/537/data.csv'
data = pd.read_csv(data_url)

# Verificare i primi record per assicurarsi che il caricamento sia corretto
print(data)

#RIMUOVI CAMPI VUOTI 
data.dropna(inplace=True)

# Selezionare le caratteristiche per il clustering (escludere 'ca_cervix')
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

print(X.describe())


scaler=StandardScaler()
X_scaled = scaler.fit_transform(X)
#print(X_scaled)

X_two_features = X_scaled[:, :2]

#elbow_plot(X_two_features,10)




kmeans = KMeans(n_clusters=3)
kmeans.fit(X_two_features)


X['Cluster'] = kmeans.labels_

print(X.head())

# Visualizzare i cluster (opzionale, se il dataset è abbastanza piccolo)
plt.scatter(X_two_features[:, 0], X_two_features[:, 1], c=X['Cluster'], cmap='viridis')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('Visualizzazione dei Cluster')
plt.show()


