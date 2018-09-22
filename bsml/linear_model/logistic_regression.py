#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class LogisticRegression:
    def __init__(self):
        pass

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(np.negative(z)))

    def fit(self, X, y, alpha=0.01, loop=1000):
        n_sample, n_feature = X.shape
        self.theta = np.zeros(n_feature)
        for _ in range(loop):
            hypothesis = self.sigmoid(np.dot(X, self.theta))
            loss = hypothesis - y
            gradient = np.dot(X.T, loss)
            self.theta = self.theta - alpha / n_sample * gradient
        return self

    def predict(self, X):
        return self.sigmoid(np.dot(X, self.theta))
