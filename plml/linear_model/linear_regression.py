#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class LinearRegression:
    def __init__(self):
        pass

    def fit(self, X, y, alpha=0.01, loop=1000):
        m, n = X.shape
        self.theta = np.zeros(n)
        for _ in range(loop):
            hypothesis = np.dot(X, self.theta)
            loss = hypothesis - y
            gradient = np.dot(X.T, loss)
            self.theta = self.theta - alpha / m * gradient
        return self

    def predict(self, X):
        return np.dot(X, self.theta)
