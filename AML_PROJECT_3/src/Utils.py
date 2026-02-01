from .Data import *
import pandas as pd
import seaborn as sns
from scipy.stats import iqr
import numpy as np
import time 
import tracemalloc
from sklearn.preprocessing import LabelEncoder


def hardness_example(hardness):
    X, y = generate_points(hardness)
    feat_names = [f"Feat_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_names)
    df['Class'] = y
    sns.pairplot(df, hue='Class')

def visualize_distribution(X, y):
    le_vis = LabelEncoder()
    y_encoded = le_vis.fit_transform(y)

    df = pd.DataFrame(X)
    df['Class'] = y_encoded
    sns.pairplot(df, hue='Class')

def measure_execution(func, *args, **kwargs):
    tracemalloc.start()
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, end_time - start_time, peak / (1024 * 1024)

def silverman_bandwidth(X):
    """Silverman's rule of thumb for bandwidth selection"""
    n = len(X)
    sigma = np.std(X, ddof=1)
    iqr_val = iqr(X)
        
    h = 0.9 * min(sigma, iqr_val / 1.34) * (n ** (-1/5))
    return h