import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.kernel_approximation import RBFSampler
from .Utils import *

class KDENaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth='silverman', random_state=None):
        self.bandwidth = bandwidth
        self.random_state = random_state
        self.classes_ = None
        self.models_ = {} 
        self.priors_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in self.classes_:
            X_c = X[y == c]
            if len(X_c) == 0:
                self.priors_[c] = 0.0
                continue
                
            self.priors_[c] = len(X_c) / n_samples
            self.models_[c] = {}
            
            for feature_idx in range(n_features):
                feature_data = X_c[:, feature_idx]
                
                if self.bandwidth == 'silverman':
                    bw = silverman_bandwidth(feature_data)
                    if bw <= 1e-9: bw = 1.0 
                else:
                    bw = self.bandwidth
                
                # Note: KernelDensity is deterministic and doesn't use random_state
                kde = KernelDensity(bandwidth=bw, kernel='gaussian')
                kde.fit(feature_data.reshape(-1, 1))
                self.models_[c][feature_idx] = kde
        return self

    def predict_proba(self, X):
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            if self.priors_[c] > 0:
                log_probs[:, i] = np.log(self.priors_[c])
            else:
                log_probs[:, i] = -1000.0
            
            for feature_idx in range(n_features):
                kde = self.models_[c][feature_idx]
                log_density = kde.score_samples(X[:, feature_idx].reshape(-1, 1))
                log_density = np.clip(log_density, -700, 700)
                log_probs[:, i] += log_density

        lse = logsumexp(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - lse)
        
        if np.isnan(probs).any():
            probs = np.nan_to_num(probs, nan=1.0/n_classes)
            
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

class RFFNaiveBayes(ClassifierMixin, BaseEstimator):
    """
    Naive Bayes that estimates feature densities using Random Fourier Features (RFF).
    """

    def __init__(self, n_components=100, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        self.estimators_ = {}
        self.priors_ = {}
        
        for c in self.classes_:
            X_c = X[y == c]
            
            # 1. Calculate Priors
            if len(X_c) == 0:
                self.priors_[c] = 0.0
                continue
            self.priors_[c] = len(X_c) / n_samples
            
            self.estimators_[c] = {}
            
            # 2. Fit RFF Sampler per feature (Independence Assumption)
            for feature_idx in range(n_features):
                feature_data = X_c[:, feature_idx].reshape(-1, 1)

                bw = silverman_bandwidth(feature_data)

                gamma = 1.0 / (2 * bw**2)
                
                # Initialize RFF Sampler
                # We use a distinct random state per feature to avoid correlation artifacts
                rff = RBFSampler(gamma=gamma, 
                                 n_components=self.n_components, 
                                 random_state=self.random_state + feature_idx)
                
                # Transform Training Data -> High Dim Space
                Z = rff.fit_transform(feature_data)
                
                # COMPUTE MEAN EMBEDDING (The "Distribution")
                # We only need to store this Mean Vector (size D), not the N points!
                mean_vector = np.mean(Z, axis=0)
                
                self.estimators_[c][feature_idx] = (rff, mean_vector)
                
        return self

    def predict_proba(self, X):
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        # Initialize with -inf so strict logsumexp doesn't fail on empty classes
        log_probs = np.full((n_samples, n_classes), -1000.0)
        
        for i, c in enumerate(self.classes_):
            if self.priors_[c] == 0:
                continue

            # Log Prior
            log_probs[:, i] = np.log(self.priors_[c])
            
            for feature_idx in range(n_features):
                if feature_idx not in self.estimators_[c]:
                    continue
                    
                rff, mean_vector = self.estimators_[c][feature_idx]
                
                # Transform Test Data
                Z_test = rff.transform(X[:, feature_idx].reshape(-1, 1))
                
                # ESTIMATE DENSITY via DOT PRODUCT
                # Density approx = Z_test dot Mean_Vector
                density = np.dot(Z_test, mean_vector)
                
                # RFF approximation can technically dip slightly below 0 due to noise
                # Clamp to epsilon to avoid log(negative)
                density = np.clip(density, 1e-9, None)
                
                log_probs[:, i] += np.log(density)

        # Robust Softmax
        lse = logsumexp(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - lse)
        
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
    
def main():
    pass

if __name__ == "__main__":
    main()