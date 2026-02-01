from .Data import *
from .Classifiers import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
import multiprocessing
from sklearn.preprocessing import StandardScaler


def process_single_run(difficulty, seed, D_VALUES):
    """Process a single run for a given difficulty level and seed."""
    run_results = []
    
    # Generate dataset
    X, y = generate_points(difficulty, n_samples=5000, random_state=seed)
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # ---------------------------------------------------------
    # Baseline: Exact KDE (with seed for reproducibility)
    # ---------------------------------------------------------
    kde_model = KDENaiveBayes(bandwidth='silverman', random_state=seed)
    
    _, train_time_kde, train_mem_kde = measure_execution(kde_model.fit, X_train, y_train)
    y_pred_kde, infer_time_kde, infer_mem_kde = measure_execution(kde_model.predict_proba, X_test)
    y_pred_kde = y_pred_kde[:, 1]
    auc_kde = roc_auc_score(y_test, y_pred_kde)
    
    run_results.append({
        'Difficulty': difficulty,
        'Seed': seed,
        'Method': 'Exact_KDE',
        'D': 0,
        'AUC': auc_kde,
        'Train_Time_sec': train_time_kde,
        'Infer_Time_sec': infer_time_kde,
        'Total_Time_sec': train_time_kde + infer_time_kde,
        'Train_Memory_MB': train_mem_kde,
        'Infer_Memory_MB': infer_mem_kde
    })
    
    # ---------------------------------------------------------
    # Gaussian Naive Bayes
    # ---------------------------------------------------------
    gnb_model = GaussianNB()
    
    _, train_time_gnb, train_mem_gnb = measure_execution(gnb_model.fit, X_train, y_train)
    y_pred_gnb, infer_time_gnb, infer_mem_gnb = measure_execution(gnb_model.predict_proba, X_test)
    y_pred_gnb = y_pred_gnb[:, 1]
    auc_gnb = roc_auc_score(y_test, y_pred_gnb)
    
    run_results.append({
        'Difficulty': difficulty,
        'Seed': seed,
        'Method': 'Gaussian_NB',
        'D': 0,
        'AUC': auc_gnb,
        'Train_Time_sec': train_time_gnb,
        'Infer_Time_sec': infer_time_gnb,
        'Total_Time_sec': train_time_gnb + infer_time_gnb,
        'Train_Memory_MB': train_mem_gnb,
        'Infer_Memory_MB': infer_mem_gnb
    })
    
    # ---------------------------------------------------------
    # RFF-KDE with different D values (with seed for reproducibility)
    # ---------------------------------------------------------
    for D in D_VALUES:
        rff_model = RFFNaiveBayes(n_components=D, random_state=seed)
        
        # Measure training
        _, train_time_rff, train_mem_rff = measure_execution(rff_model.fit, X_train, y_train)
        
        # Measure inference
        y_pred_rff, infer_time_rff, infer_mem_rff = measure_execution(rff_model.predict_proba, X_test)
        y_pred_rff = y_pred_rff[:, 1]
        auc_rff = roc_auc_score(y_test, y_pred_rff)
        
        run_results.append({
            'Difficulty': difficulty,
            'Seed': seed,
            'Method': 'RFF_KDE',
            'D': D,
            'AUC': auc_rff,
            'Train_Time_sec': train_time_rff,
            'Infer_Time_sec': infer_time_rff,
            'Total_Time_sec': train_time_rff + infer_time_rff,
            'Train_Memory_MB': train_mem_rff,
            'Infer_Memory_MB': infer_mem_rff
        })
    
    return run_results

# ==========================================
# PARALLEL EXECUTION FUNCTION
# ==========================================
def process_single_run_real(X, y, seed, D, compute_baselines=True):
    """Process a single run for a given seed and D value.
    
    Args:
        X, y: Dataset
        seed: Random seed
        D: Number of RFF components
        compute_baselines: If True, also compute Exact_KDE and Gaussian_NB
    """
    run_results = []

    # Prepare data
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------------------------------------------------
    # Baselines (only when flag is True)
    # ---------------------------------------------------------
    if compute_baselines:
        # Exact KDE
        kde_model = KDENaiveBayes(bandwidth='silverman', random_state=seed)
        
        _, train_time_kde, train_mem_kde = measure_execution(kde_model.fit, X_train, y_train)
        y_pred_kde, infer_time_kde, infer_mem_kde = measure_execution(kde_model.predict_proba, X_test)
        y_pred_kde = y_pred_kde[:, 1]
        auc_kde = roc_auc_score(y_test, y_pred_kde)
        
        run_results.append({
            'Seed': seed,
            'Method': 'Exact_KDE',
            'D': 0,
            'AUC': auc_kde,
            'Train_Time_sec': train_time_kde,
            'Infer_Time_sec': infer_time_kde,
            'Total_Time_sec': train_time_kde + infer_time_kde,
            'Train_Memory_MB': train_mem_kde,
            'Infer_Memory_MB': infer_mem_kde
        })
        
        # Gaussian Naive Bayes
        gnb_model = GaussianNB()
        
        _, train_time_gnb, train_mem_gnb = measure_execution(gnb_model.fit, X_train, y_train)
        y_pred_gnb, infer_time_gnb, infer_mem_gnb = measure_execution(gnb_model.predict_proba, X_test)
        y_pred_gnb = y_pred_gnb[:, 1]
        auc_gnb = roc_auc_score(y_test, y_pred_gnb)
        
        run_results.append({
            'Seed': seed,
            'Method': 'Gaussian_NB',
            'D': 0,
            'AUC': auc_gnb,
            'Train_Time_sec': train_time_gnb,
            'Infer_Time_sec': infer_time_gnb,
            'Total_Time_sec': train_time_gnb + infer_time_gnb,
            'Train_Memory_MB': train_mem_gnb,
            'Infer_Memory_MB': infer_mem_gnb
        })

    # ---------------------------------------------------------
    # RFF-KDE for the specific D value
    # ---------------------------------------------------------
    rff_model = RFFNaiveBayes(n_components=int(D), random_state=seed)
    
    _, train_time_rff, train_mem_rff = measure_execution(rff_model.fit, X_train, y_train)
    y_pred_rff, infer_time_rff, infer_mem_rff = measure_execution(rff_model.predict_proba, X_test)
    y_pred_rff = y_pred_rff[:, 1]
    auc_rff = roc_auc_score(y_test, y_pred_rff)

    run_results.append({
        'Seed': seed,
        'Method': 'RFF_KDE',
        'D': int(D),
        'AUC': auc_rff,
        'Train_Time_sec': train_time_rff,
        'Infer_Time_sec': infer_time_rff,
        'Total_Time_sec': train_time_rff + infer_time_rff,
        'Train_Memory_MB': train_mem_rff,
        'Infer_Memory_MB': infer_mem_rff
    })

    return run_results




def process_size_run(size, seed, difficulty='medium'):
    """Process a single run for a given dataset size and seed."""
    run_results = []
    
    # Generate dataset
    X, y = generate_points(difficulty, n_samples=size, random_state=seed)
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Calculate D = log(n) for RFF
    D = int(np.log(size))
    
    # ---------------------------------------------------------
    # Exact KDE
    # ---------------------------------------------------------
    kde_model = KDENaiveBayes(bandwidth='silverman', random_state=seed)
    
    _, train_time_kde, train_mem_kde = measure_execution(kde_model.fit, X_train, y_train)
    y_pred_kde, infer_time_kde, infer_mem_kde = measure_execution(kde_model.predict_proba, X_test)
    y_pred_kde = y_pred_kde[:, 1]
    auc_kde = roc_auc_score(y_test, y_pred_kde)
    
    run_results.append({
        'Size': size,
        'Seed': seed,
        'Method': 'Exact_KDE',
        'D': 0,
        'AUC': auc_kde,
        'Train_Time_sec': train_time_kde,
        'Infer_Time_sec': infer_time_kde,
        'Total_Time_sec': train_time_kde + infer_time_kde,
        'Train_Memory_MB': train_mem_kde,
        'Infer_Memory_MB': infer_mem_kde
    })
    
    # ---------------------------------------------------------
    # Gaussian Naive Bayes
    # ---------------------------------------------------------
    gnb_model = GaussianNB()
    
    _, train_time_gnb, train_mem_gnb = measure_execution(gnb_model.fit, X_train, y_train)
    y_pred_gnb, infer_time_gnb, infer_mem_gnb = measure_execution(gnb_model.predict_proba, X_test)
    y_pred_gnb = y_pred_gnb[:, 1]
    auc_gnb = roc_auc_score(y_test, y_pred_gnb)
    
    run_results.append({
        'Size': size,
        'Seed': seed,
        'Method': 'Gaussian_NB',
        'D': 0,
        'AUC': auc_gnb,
        'Train_Time_sec': train_time_gnb,
        'Infer_Time_sec': infer_time_gnb,
        'Total_Time_sec': train_time_gnb + infer_time_gnb,
        'Train_Memory_MB': train_mem_gnb,
        'Infer_Memory_MB': infer_mem_gnb
    })
    
    # ---------------------------------------------------------
    # RFF-KDE with D = log(n)
    # ---------------------------------------------------------
    rff_model = RFFNaiveBayes(n_components=D, random_state=seed)
    
    _, train_time_rff, train_mem_rff = measure_execution(rff_model.fit, X_train, y_train)
    y_pred_rff, infer_time_rff, infer_mem_rff = measure_execution(rff_model.predict_proba, X_test)
    y_pred_rff = y_pred_rff[:, 1]
    auc_rff = roc_auc_score(y_test, y_pred_rff)
    
    run_results.append({
        'Size': size,
        'Seed': seed,
        'Method': 'RFF_KDE',
        'D': D,
        'AUC': auc_rff,
        'Train_Time_sec': train_time_rff,
        'Infer_Time_sec': infer_time_rff,
        'Total_Time_sec': train_time_rff + infer_time_rff,
        'Train_Memory_MB': train_mem_rff,
        'Infer_Memory_MB': infer_mem_rff
    })
    
    return run_results
