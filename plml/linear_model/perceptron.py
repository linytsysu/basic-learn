#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Perceptron:
    def __init__(self):
        pass

    def fit(self, X, y, alpha=0.001, loop=1000):
        n_sample, n_feature = X.shape
        times = np.zeros(n_sample)
        gram = np.dot(X, X.T)
        for _ in range(loop):
            has_error = False
            for i in range(n_sample):
                if (y[i] * np.dot(alpha * times * y, gram[:, i])) <= 0:
                    has_error = True
                    times[i] += 1
            if not has_error:
                break
        self.theta = np.dot(alpha * times * y, X)
        return self

    def predict(self, X):
        return np.dot(X, self.theta)

# class Perceptron:
#     '''
#     origin algorithm of perceptron
#     '''
#     def __init__(self):
#         pass

#     def fit(self, X, y, alpha=0.001, loop=1000):
#         n_sample, n_feature = X.shape
#         self.theta = np.zeros(n_feature)
#         for _ in range(loop):
#             has_error = False
#             for i in range(n_sample):
#                 hypothesis = np.dot(self.theta, X[i])
#                 if (hypothesis * y[i]) <= 0:
#                     has_error = True
#                     self.theta += alpha * X[i] * y[i]
#             if not has_error:
#                 break
#         return self

#     def predict(self, X):
#         return np.dot(X, self.theta)
