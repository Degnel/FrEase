import pickle
import matplotlib.pyplot as plt

def benchmark_display(file_path_1, file_path_2):
    """
    Affiche l'évolution des pertes d'entraînement et de test 
    pour deux fichiers de données, en utilisant matplotlib.

    Args:
        file_path_1 (str): Chemin vers le premier fichier pickle.
        file_path_2 (str): Chemin vers le deuxième fichier pickle.
    """

    # Charger les données depuis les fichiers pickle
    with open(file_path_1, 'rb') as f:
        data1 = pickle.load(f)
    with open(file_path_2, 'rb') as f:
        data2 = pickle.load(f)

    # Extraire les données pour le graphique
    temps1 = [d['time'] for d in data1]
    pertes_train1 = [d['loss'] for d in data1]
    pertes_test1 = [d['test_loss'] for d in data1]

    temps2 = [d['time'] for d in data2]
    pertes_train2 = [d['loss'] for d in data2]
    pertes_test2 = [d['test_loss'] for d in data2]

    # Créer le graphique
    plt.figure(figsize=(10, 6))

    plt.plot(temps1, pertes_train1, label=f'Train Loss ({file_path_1})', marker='o')
    plt.plot(temps1, pertes_test1, label=f'Test Loss ({file_path_1})', marker='x')
    plt.plot(temps2, pertes_train2, label=f'Train Loss ({file_path_2})', marker='o')
    plt.plot(temps2, pertes_test2, label=f'Test Loss ({file_path_2})', marker='x')

    plt.xlabel('Temps (unités)')
    plt.ylabel('Perte')
    plt.title('Comparaison des pertes d\'entraînement et de test')
    plt.legend()
    plt.grid(True)
    plt.show()