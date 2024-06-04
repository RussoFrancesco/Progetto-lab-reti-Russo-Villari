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
print (df_distr)

labels = ['500000', '1000000', '5000000', '10000000']
distr = [df_distr["Tempo"][0],df_distr["Tempo"][2], df_distr["Tempo"][4], df_distr["Tempo"][6]]
seq =  [df_seq["Tempo"][1],df_seq["Tempo"][3], df_seq["Tempo"][5], df_seq["Tempo"][7]]


x = np.arange(len(labels))  # La posizione delle etichette
width = 0.35  # La larghezza delle barre

fig, ax = plt.subplots()


bars1 = ax.bar(x - width/2, distr, width, label='Distribuito')
bars2 = ax.bar(x + width/2, seq, width, label='Sequenziale')


ax.set_xlabel('Numero di Punti')
ax.set_ylabel('Tempo')
ax.set_title('Confronto tra esecuzione sequenziale e distribuita')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


ax.grid(False)


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 punti di offset verticale
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)


plt.show()