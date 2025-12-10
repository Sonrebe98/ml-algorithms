import matplotlib.pyplot as plt
import numpy as np

def plot_features_vs_target(X, y, feature_names=None, figsize=(15, 10)):
    """
    Visualizza tutte le features rispetto al target in subplots.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Matrice delle features (n_samples, n_features)
    y : numpy.ndarray
        Array del target (n_samples,)
    feature_names : list, optional
        Lista dei nomi delle features. Se None, usa nomi generici.
    figsize : tuple, optional
        Dimensione della figura (width, height)
    """
    n_features = X.shape[1]
    
    # Calcola il numero di righe e colonne per i subplots
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols  # Arrotonda per eccesso
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    # Se non sono forniti i nomi delle features, usa nomi generici
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    for i in range(n_features):
        ax = axes[i]
        ax.scatter(X[:, i], y, alpha=0.5, s=20)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Target (House Price)')
        ax.set_title(f'{feature_names[i]} vs House Price')
        ax.grid(True, alpha=0.3)
    
    # Nasconde i subplots non utilizzati
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_single_feature_vs_target(X, y, feature_idx, feature_name=None, figsize=(8, 6)):
    """
    Visualizza una singola feature rispetto al target.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Matrice delle features (n_samples, n_features)
    y : numpy.ndarray
        Array del target (n_samples,)
    feature_idx : int
        Indice della feature da visualizzare
    feature_name : str, optional
        Nome della feature. Se None, usa 'Feature {feature_idx}'
    figsize : tuple, optional
        Dimensione della figura (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if feature_name is None:
        feature_name = f'Feature {feature_idx}'
    
    ax.scatter(X[:, feature_idx], y, alpha=0.5, s=20)
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Target (House Price)')
    ax.set_title(f'{feature_name} vs House Price')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_target_distribution(y, figsize=(8, 6)):
    """
    Visualizza la distribuzione del target.
    
    Parameters:
    -----------
    y : numpy.ndarray
        Array del target (n_samples,)
    figsize : tuple, optional
        Dimensione della figura (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(y, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('House Price')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of House Prices')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_features_distribution(X, feature_names=None, figsize=(15, 10)):
    """
    Visualizza la distribuzione di tutte le features.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Matrice delle features (n_samples, n_features)
    feature_names : list, optional
        Lista dei nomi delle features. Se None, usa nomi generici.
    figsize : tuple, optional
        Dimensione della figura (width, height)
    """
    n_features = X.shape[1]
    
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    for i in range(n_features):
        ax = axes[i]
        ax.hist(X[:, i], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feature_names[i]}')
        ax.grid(True, alpha=0.3, axis='y')
    
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig
