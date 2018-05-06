#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class KNeighborsRegressor:
    def __init__(self, n_neighbors=5, p=2):
        self.n_neighbors = n_neighbors
        self.p = p

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        m_train = self.X_train.shape[0]
        m_test = X.shape[0]
        dist = np.zeros((m_test, m_train))
        if self.p == 2:
            H = np.ones((m_train, 1)) * np.sum(np.square(X), axis=1)
            K = np.ones((m_test, 1)) * np.sum(np.square(self.X_train), axis=1)
            G = np.dot(X, self.X_train.T)
            dist = np.sqrt(H.T + K - 2 * G)
        else:
            for i in range(m_test):
                for j in range(m_train):
                    # dist[i, j] = np.linalg.norm(X[i] - self.X_train[j], ord=self.p)
                    dist[i, j] = np.sum(np.absolute(X[i] - self.X_train[j]) ** self.p) ** 1.0 / self.p

        y_pred = np.zeros(m_test)
        for i in range(m_test):
            closest_idx = np.argsort(dist[i])
            tmp = 0.0
            for idx in closest_idx[:self.n_neighbors]:
                tmp += self.y_train[idx]
            y_pred[i] = tmp / self.n_neighbors
        return y_pred
