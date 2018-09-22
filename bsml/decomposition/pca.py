#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def svd_flip(u, v, u_based_decision=True):
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, xrange(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[xrange(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


class PCA:
    def __init__(self, n_components):
        self.n_components_ = n_components

    def fit(self, X):
        self._fit(X)
        return self

    def _fit(self, X):
        self.mean_ = X.mean(axis=0)
        # Remove mean
        X = X - self.mean_
        U, S, V = np.linalg.svd(X, full_matrices=False)
        U, V = svd_flip(U, V)
        self.components_ = V[:self.n_components_]
        return U, S, V

    def transform(self, X):
        X = X - self.mean_
        # X_new = X * V
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed

    def fit_transform(self, X):
        U, S, V = self._fit(X)
        U = U[:, :self.n_components_]
        # X_new = X * V = U * S * V^T * V = U * S
        U *= S[:self.n_components_]
        return U
