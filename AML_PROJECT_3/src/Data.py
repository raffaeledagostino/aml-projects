import numpy as np
from scipy.stats import (norm, uniform, lognorm, dweibull, wald, laplace)
from sklearn.calibration import LabelEncoder
from torch import le
from ucimlrepo import fetch_ucirepo

def magic_gamma_data():
    """
    Load the heart disease dataset from UCI ML repository.
    
    Returns:
    --------
    tuple
        - X: Feature matrix (DataFrame)
        - y: Target variable (Series, binary: 0=no disease, 1=disease present)
    """
    # Fetch dataset from UCI repository
    magic_gamma = fetch_ucirepo(id=159)
    X = magic_gamma.data.features
    y = magic_gamma.data.targets 

    return X, y

def prepare_magic_gamma(X,y):
    if hasattr(y, 'values'):
        y = y.values.ravel()
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y
    


def get_mixed_values(u_vals, distributions):
    """
    Applica un mix pesato di inverse CDF (PPF).
    """
    mixed = np.zeros_like(u_vals)
    total_weight = 0.0
    
    for name, dist_func, kwargs, weight in distributions:
        if weight <= 1e-6: continue 

        if name == 'custom_bimodal':
            sep = kwargs.get('separation', 3.0)
            vals_left = norm.ppf(u_vals * 2) - sep
            vals_right = norm.ppf((u_vals - 0.5) * 2) + sep
            vals = np.where(u_vals < 0.5, vals_left, vals_right)
            
        elif name == 'custom_peaked':
            vals = norm.ppf(u_vals) * 0.5 
            
        else:
            vals = dist_func.ppf(u_vals, **kwargs)
        
        if np.std(vals) > 0:
            vals = (vals - np.mean(vals)) / np.std(vals)
            
        mixed += vals * weight
        total_weight += weight
        
    return mixed / total_weight if total_weight > 0 else mixed

def get_random_weights(n, dominance_factor=1.0):
    weights = np.random.dirichlet(np.ones(n) * dominance_factor)
    return weights

def generate_points(difficulty: str, n_samples=1000, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_features = 10
    n_classes = 2
    samples_per_class = n_samples // n_classes

    # --- 0. Configurazione Difficoltà ---
    if difficulty == "easy":
        rho_min, rho_max = 0.0, 0.2
        separation_factor = 4.0 
        non_gaussian_count = 0 
        n_informative = 10  # Tutte le feature servono
        
    elif difficulty == "medium":
        rho_min, rho_max = 0.3, 0.5
        separation_factor = 2.0 
        non_gaussian_count = 8 
        n_informative = 5   # 2 feature sono rumore puro
        
    elif difficulty == "hard":
        rho_min, rho_max = 0.6, 0.8
        separation_factor = 0.0 # Medie sovrapposte
        non_gaussian_count = 10 
        n_informative = 5   # Solo 5 feature su 10 portano informazione (densità diversa)
        
    else:
        raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")

    # Selezioniamo quali feature sono informative
    feature_indices = np.arange(n_features)
    # Mescoliamo gli indici ma teniamo traccia di chi è chi
    np.random.shuffle(feature_indices)
    informative_indices = set(feature_indices[:n_informative])
    
    # --- 1. Generazione Struttura di Covarianza ---
    rho = np.random.uniform(rho_min, rho_max)
    cov_matrix = np.full((n_features, n_features), rho)
    np.fill_diagonal(cov_matrix, 1.0)
    cov_matrix += np.eye(n_features) * 1e-6

    # --- 2. Generazione Dati Base (Gaussiani) ---
    mean_0 = np.zeros(n_features)
    mean_1 = np.zeros(n_features) # Partiamo da zero per tutti
    
    # Applichiamo il separation_factor SOLO alle feature informative
    # Le feature rumorose avranno media 0 vs 0 (sovrapposizione totale)
    for i in range(n_features):
        if i in informative_indices:
            mean_1[i] = separation_factor
        else:
            mean_1[i] = 0.0

    X0 = np.random.multivariate_normal(mean_0, cov_matrix, samples_per_class)
    X1 = np.random.multivariate_normal(mean_1, cov_matrix, samples_per_class)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(samples_per_class), np.ones(samples_per_class)])

    # --- 3. Definizione Pool Distribuzioni ---
    pool_class_0 = [
        ('custom_bimodal', None,   {'separation': 3.5}),
        ('uniform',        uniform,{'loc':-3, 'scale':6}),
        ('dweibull',       dweibull,{'c': 1.0}),
    ]
    
    pool_class_1 = [
        ('laplace',        laplace, {'loc': 0, 'scale': 0.8}),
        ('lognorm',        lognorm, {'s': 0.6}),
        ('custom_peaked',  None,    {}),
    ]

    # --- 4. Trasformazione Non-Gaussiana ---
    # Decidiamo quali feature trasformare in base a non_gaussian_count
    # (indipendentemente dal fatto che siano informative o no)
    transform_candidates = np.arange(n_features)
    np.random.shuffle(transform_candidates)
    indices_to_transform = transform_candidates[:non_gaussian_count]

    for idx in indices_to_transform:
        u_vals = norm.cdf(X[:, idx] - X[:, idx].mean())
        u_vals = np.clip(u_vals, 1e-6, 1 - 1e-6)
        
        mask_0 = (y == 0)
        mask_1 = (y == 1)
        
        # Generiamo pesi casuali
        weights_A = get_random_weights(len(pool_class_0), dominance_factor=0.8)
        weights_B = get_random_weights(len(pool_class_1), dominance_factor=0.8)
        
        dist_set_A = [(n, f, k, w) for (n, f, k), w in zip(pool_class_0, weights_A)]
        dist_set_B = [(n, f, k, w) for (n, f, k), w in zip(pool_class_1, weights_B)]

        # LOGICA CHIAVE:
        if idx in informative_indices:
            # FEATURE INFORMATIVA:
            # Classe 0 usa Mix A (es. Bimodale)
            # Classe 1 usa Mix B (es. Unimodale)
            # Risultato: Forme diverse -> Distinguibili
            X[mask_0, idx] = get_mixed_values(u_vals[mask_0], dist_set_A)
            X[mask_1, idx] = get_mixed_values(u_vals[mask_1], dist_set_B)
        else:
            # FEATURE RUMOROSA (Inutile):
            # Entrambe le classi usano lo STESSO Mix (es. Mix A per tutti)
            # Risultato: Forme identiche -> Indistinguibili
            # Nota: Usiamo lo stesso set (dist_set_A) per tutto il vettore
            X[:, idx] = get_mixed_values(u_vals, dist_set_A)

    # Aggiunta rumore finale
    X += np.random.normal(0, 0.1, size=X.shape)
    
    # Restituiamo anche la lista delle feature informative per debug/plot
    return X, y

# --- VISUALIZZAZIONE ---
if __name__ == "__main__":
    pass