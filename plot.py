import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("tempi.csv")

df = pd.DataFrame(data)

# Separa i dati in due DataFrames in base alla modalità
df_15 = df[df["Map"] == 15]
df_30 = df[(df["Map"] == 30) & (df["Numero di cluster"] == 3)]
df_60 = df[(df["Map"] == 60)]


labels = ['500000', '1000000', '5000000', '10000000']


x = np.arange(len(labels))  # La posizione delle etichette
width = 0.3  # La larghezza delle barre

fig, ax = plt.subplots()


bars1 = ax.bar(x - width, df_15["Tempo"], width, label='15 Mappers')
bars2 = ax.bar(x, df_30["Tempo"], width, label='30 Mappers')
bars3 = ax.bar(x + width, df_60["Tempo"], width, label='60 Mappers')


ax.set_xlabel('Numero di Punti')
ax.set_ylabel('Tempo')
ax.set_title('Confronto tra numero di mappers')
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

#add_labels(bars1)
#add_labels(bars2)
#add_labels(bars3)


plt.show()
