# Advanced Machine Learning Projects

This repository contains three projects exploring classification algorithms, kernel methods, and computational efficiency, completed as part of the MDS Advanced Machine Learning course (2025-2026).

## Projects

### 1. Heart Disease Classification
Comparative analysis of Logistic Regression, SVM, Custom Naive Bayes, and Decision Trees on the UCI Heart Disease dataset. **Best result:** LR achieved AUC 0.906 ± 0.003 with rigorous 100-fold repeated validation.

### 2. Indefinite Kernel Learning
Analysis of sigmoid kernel behavior in SVMs and evaluation of spectral correction methods (shifting, clipping, normalized clipping). **Key finding:** Eigenvalue clipping is least destructive, but corrections often degrade high-performing indefinite kernels.

### 3. Random Fourier Features for Scalable KDE
RFF-based approximation for Kernel Density Estimation in Naive Bayes, reducing inference complexity from O(N²) to O(D). **Key result:** 97% AUC retention with 10x speedup on complex distributions.
