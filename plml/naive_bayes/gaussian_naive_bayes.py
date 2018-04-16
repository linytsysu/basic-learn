#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        n_features = X.shape[1]
        unique_y = np.unique(y)
        n_classes = unique_y.shape[0]

        self.mu = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for y_i in unique_y:
            i = unique_y.searchsorted(y_i)
            X_i = X[y == y_i, :]
            self.mu[i, :] = np.mean(X_i, axis=0)
            self.var[i, :] = np.var(X_i, axis=0)
            self.priors[i] = float(len(X_i)) / len(X)

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            density = (1.0 / np.sqrt(2 * np.pi * self.var)) * np.exp(-(((X[i] - self.mu) ** 2) / (2 * self.var)))
            prob_desity = np.multiply(np.multiply.reduce(density, axis=1), self.priors)
            y_pred[i] = np.argmax(prob_desity)
        return y_pred
