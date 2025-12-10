import numpy as np
import os
import matplotlib.pyplot as plt
from visualization import plot_features_vs_target

def load_data_from_csv(filename):
    """Carica i dati dal CSV in un array numpy"""
    data_list = []

    try: 
      with open(filename, 'r') as file:
        next(file) # Salta l'intestazione
        for line in file:
          features = line.strip().split(',')
          data_list.append([float(feature) for feature in features])
    except (IOError, OSError, FileNotFoundError) as e:
      print(f"Errore nel caricamento del file {filename}: {e}")
      return None
    except Exception as e:
      print(f"Errore imprevisto durante il caricamento: {e}")
      return None
    
    if not data_list:
      print("Il file è vuoto o non contiene dati validi")
      return None
      
    data_array = np.array(data_list)
    X = data_array[:, :-1]  # Tutte le colonne tranne l'ultima (features)
    Y = data_array[:, -1]   # Ultima colonna (target)
    return X, Y

# il taining set è stato scaricato da kagglehub e si trova nella directory dello script: https://www.kaggle.com/datasets/prokshitha/home-value-insights?resource=download
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'house_price_regression_dataset.csv')

# Leggi i nomi delle features dal CSV
with open(csv_path, 'r') as file:
    header = file.readline().strip()
    feature_names = header.split(',')[:-1]  # Tutte le colonne tranne l'ultima (target)

X, y = load_data_from_csv(csv_path)

print(X.shape)
print(y.shape)
print(X[:5])
print(y[:5])

plot_features_vs_target(X, y, feature_names=feature_names)
plt.show()