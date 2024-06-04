import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# I tuoi dati
data = {
    'Numero di punti': [500000, 500000, 1000000, 1000000, 5000000, 5000000, 10000000, 10000000],
    'Tempo': [38.69880962371826, 11.270809173583984, 101.11570715904236, 30.59341073036194, 476.03889989852905, 115.75412392616272, 1498.7870407104492, 300.7421875],
    'Modalità': ['Distribuito', 'Sequenziale', 'Distribuito', 'Sequenziale', 'Distribuito', 'Sequenziale', 'Distribuito', 'Sequenziale']
}

df = pd.DataFrame(data)

# Separa i dati in due DataFrames in base alla modalità
df_distr = df[df['Modalità'] == 'Distribuito']
df_seq = df[df['Modalità'] == 'Sequenziale']

# Crea un grafico a barre per la modalità Distribuito e Sequenziale
bar_width = 0.35
index = np.arange(len(df_distr))

plt.figure(figsize=(10,5))
bars1 = plt.bar(index, df_distr['Tempo'], bar_width, color='b', label='Distribuito')
bars2 = plt.bar(index + bar_width, df_seq['Tempo'], bar_width, color='r', label='Sequenziale')

for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom') # va: vertical alignment

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom')

plt.xlabel('Numero di punti')
plt.ylabel('Tempo')
plt.xticks(index + bar_width / 2, df_distr['Numero di punti'])  # posiziona le etichette sull'asse x
plt.legend()
plt.show()