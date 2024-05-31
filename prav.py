from create_points import create_points
import pandas as pd
import csv
import numpy as np


def split(data,num_partition):
    #splitto il dataset in n sub-array
    partitions = np.array_split(data, num_partition)
    return partitions

# Creare e scrivere i punti nel file CSV
with open('prova.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["points"])
    punti = create_points(15, 3, 3, 42)
    for punto in punti[0]:
        writer.writerow([punto.tolist()])  # Convertire il punto in una lista prima di scrivere

# Leggere i punti dal file CSV
punti = pd.read_csv('prova.csv')


# Convertire i punti letti dal CSV in array numpy
# Usare ast.literal_eval per convertire le stringhe in liste in modo sicuro
import ast

array_punti = np.array([np.array(ast.literal_eval(punto)) for punto in punti['points']])

for partition in split(array_punti, 5):
    print(len(partition))
