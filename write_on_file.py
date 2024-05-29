import csv
import os


def write_on_file(filename, n_samples, n_cluster, time, mode, seed, random_state):
    headers = False
    if not os.path.exists(filename):
        headers = True
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(["Numero di punti", "Numero di cluster", "Seed centroidi iniziali", "Seed creazione punti", "Tempo", "Modalità"])
        
        writer.writerow([n_samples, n_cluster, seed, random_state, time, mode])
