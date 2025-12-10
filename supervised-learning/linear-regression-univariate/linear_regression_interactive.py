import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from linear_regression import load_data_from_csv, compute_cost, compute_gradient, gradient_descent, predict

# Carica i dati
X, y = load_data_from_csv('dati_voti.csv')
m = len(X)

# Valori iniziali dei parametri
initial_w = 0.0
initial_b = 0.0
initial_learning_rate = 0.01
initial_iterations = 5000

# Crea la figura e i subplot
fig = plt.figure(figsize=(16, 10))
plt.subplots_adjust(left=0.15, bottom=0.35, right=0.95, top=0.95)

# Crea i 3 subplot per i grafici
ax_cost = plt.subplot(2, 3, 1)
ax_params = plt.subplot(2, 3, 2)
ax_regression = plt.subplot(2, 3, 3)
ax_cost_history = plt.subplot(2, 3, (4, 6))  # Grafico più grande per il costo

# Funzione per aggiornare i grafici
def update_plots(w_init, b_init, lr, iterations):
    # Esegui il gradient descent
    final_w, final_b, history_j, history_params = gradient_descent(
        X, y, w_init, b_init, lr, iterations, m
    )
    
    # Pulisci tutti i subplot
    ax_cost.clear()
    ax_params.clear()
    ax_regression.clear()
    ax_cost_history.clear()
    
    # 1. Grafico del costo nel tempo (piccolo)
    iterations_range = range(len(history_j))
    ax_cost.plot(iterations_range, history_j, 'b-', linewidth=2)
    ax_cost.set_xlabel('Iterazione', fontsize=10)
    ax_cost.set_ylabel('Costo J(w, b)', fontsize=10)
    ax_cost.set_title('Evoluzione del Costo', fontsize=11, fontweight='bold')
    ax_cost.grid(True, alpha=0.3)
    
    # 2. Grafico dell'evoluzione dei parametri
    w_history = [params[0] for params in history_params]
    b_history = [params[1] for params in history_params]
    ax_params.plot(iterations_range, w_history, 'r-', linewidth=2, label='w (pendenza)')
    ax_params.plot(iterations_range, b_history, 'g-', linewidth=2, label='b (intercetta)')
    ax_params.set_xlabel('Iterazione', fontsize=10)
    ax_params.set_ylabel('Valore del Parametro', fontsize=10)
    ax_params.set_title('Evoluzione dei Parametri', fontsize=11, fontweight='bold')
    ax_params.legend(fontsize=9)
    ax_params.grid(True, alpha=0.3)
    
    # 3. Grafico dei dati con la retta di regressione
    ax_regression.scatter(X, y, alpha=0.5, s=20, color='blue', label='Dati reali')
    X_line = np.linspace(X.min(), X.max(), 100)
    y_line = final_w * X_line + final_b
    ax_regression.plot(X_line, y_line, 'r-', linewidth=3, 
                      label=f'y = {final_w:.2f}x + {final_b:.2f}')
    ax_regression.set_xlabel('Ore di Studio', fontsize=10)
    ax_regression.set_ylabel('Voto Finale', fontsize=10)
    ax_regression.set_title('Regressione Lineare', fontsize=11, fontweight='bold')
    ax_regression.legend(fontsize=9)
    ax_regression.grid(True, alpha=0.3)
    
    # 4. Grafico del costo (più grande, in basso)
    ax_cost_history.plot(iterations_range, history_j, 'b-', linewidth=2)
    ax_cost_history.set_xlabel('Iterazione', fontsize=12)
    ax_cost_history.set_ylabel('Costo J(w, b)', fontsize=12)
    ax_cost_history.set_title(f'Costo Finale: {history_j[-1]:.4f} | w={final_w:.4f}, b={final_b:.4f}', 
                             fontsize=12, fontweight='bold')
    ax_cost_history.grid(True, alpha=0.3)
    
    # Aggiorna la figura
    plt.draw()

# Funzione callback per gli slider
def update(val):
    w_val = slider_w.val
    b_val = slider_b.val
    lr_val = slider_lr.val
    iter_val = int(slider_iter.val)
    update_plots(w_val, b_val, lr_val, iter_val)

# Crea gli slider
ax_slider_w = plt.axes([0.15, 0.25, 0.3, 0.03])
ax_slider_b = plt.axes([0.15, 0.20, 0.3, 0.03])
ax_slider_lr = plt.axes([0.15, 0.15, 0.3, 0.03])
ax_slider_iter = plt.axes([0.15, 0.10, 0.3, 0.03])

slider_w = Slider(ax_slider_w, 'w iniziale', -5.0, 15.0, valinit=initial_w, valstep=0.1)
slider_b = Slider(ax_slider_b, 'b iniziale', -10.0, 20.0, valinit=initial_b, valstep=0.1)
slider_lr = Slider(ax_slider_lr, 'Learning Rate', 0.001, 0.1, valinit=initial_learning_rate, valstep=0.001)
slider_iter = Slider(ax_slider_iter, 'Iterazioni', 100, 10000, valinit=initial_iterations, valstep=100)

# Collega gli slider alla funzione di update
slider_w.on_changed(update)
slider_b.on_changed(update)
slider_lr.on_changed(update)
slider_iter.on_changed(update)

# Crea un pulsante di reset
ax_reset = plt.axes([0.5, 0.15, 0.1, 0.04])
button_reset = Button(ax_reset, 'Reset')
def reset(event):
    slider_w.reset()
    slider_b.reset()
    slider_lr.reset()
    slider_iter.reset()
button_reset.on_clicked(reset)

# Inizializza i grafici
update_plots(initial_w, initial_b, initial_learning_rate, initial_iterations)

plt.suptitle('Regressione Lineare Interattiva - Modifica i parametri con gli slider!', 
             fontsize=14, fontweight='bold', y=0.98)

plt.show()
