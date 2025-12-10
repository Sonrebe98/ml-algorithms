import numpy as np

# --- Parametri di Generazione ---
TRUE_W = 9.0  # Pendenza ideale
TRUE_B = 10.0 # Intercetta ideale
NUM_SAMPLES = 1000
NOISE_SCALE = 8.0 # Deviazione standard del rumore

# 1. Generazione delle Ore di Studio (X) e Rumore
X = np.random.uniform(low=1.0, high=10.0, size=NUM_SAMPLES)
noise = np.random.normal(loc=0.0, scale=NOISE_SCALE, size=NUM_SAMPLES)

# 2. Generazione dei Voti Reali (Y)
# Y = W * X + B + Rumore
Y = (TRUE_W * X + TRUE_B + noise).clip(min=0, max=100) # Assicuriamo che i voti siano tra 0 e 100

# 3. Preparazione dei Dati per il Salvataggio
# Combiniamo X e Y in una singola matrice e arrotondiamo
data_matrix = np.stack((X.round(2), Y.round(2)), axis=1)

# 4. Scrittura del File CSV usando solo Python I/O
filename = 'dati_voti.csv'
header = "Ore_Studio,Voto_Finale\n"

try:
    with open(filename, 'w') as f:
        # Scrive l'intestazione
        f.write(header)
        
        # Scrive ogni riga di dati
        for row in data_matrix:
            # Formatta la riga come stringa "valore1,valore2\n"
            line = f"{row[0]},{row[1]}\n"
            f.write(line)

    print(f"Generato dataset di {NUM_SAMPLES} campioni nel file '{filename}'.")
    print("La struttura del file è Ore_Studio,Voto_Finale.")

except IOError:
    print(f"Errore nella scrittura del file {filename}.")