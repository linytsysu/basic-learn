#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from ..utils import check_random_state

def _init_centroids(X, k, init, random_state=None):
    n_samples, n_features = X.shape
    random_state = check_random_state(random_state)
    if init == 'random':
        seeds = random_state.permutation(n_samples)[:k]
        centers = X[seeds]
    if init == 'k-means++':
        centers = np.empty((k, n_features), dtype=X.dtype)
        center_id = random_state.randint(n_samples)
        centers[0] = X[center_id]

        closest_distances = _euclidean_distances(X, centers[0])
        current_pot = closest_distances.sum()
        for i in range(1, k):
            rand_val = random_state.random_sample() * current_pot
            candidate_id = np.searchsorted(np.cumsum(closest_distances), rand_val)
            distance_to_candidate = _euclidean_distances(X, X[candidate_id])
            closest_distances = np.minimum(closest_distances, distance_to_candidate)
            current_pot = closest_distances.sum()
            centers[i] = X[candidate_id]
    return centers

def _euclidean_distances(X, y):
    n_samples = X.shape[0]
    distances = np.zeros(n_samples)
    for i in range(n_samples):
        distances[i] = np.sum(np.absolute(X[i] - y) ** 2) ** 1.0 / 2
    return distances


class KMeans():
    def __init__(self, n_clusters, init='k-means++', max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.init = init    # 'k-means++' | 'random'
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        self.centroids = _init_centroids(X, self.n_clusters, self.init)
        distances = np.zeros((n_samples, self.n_clusters))
        self.labels = np.empty(n_samples)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                for c in range(self.n_clusters):
                    distances[i][c] = np.sum(np.absolute(X[i] - self.centroids[c]) ** 2) ** 1.0 / 2
                self.labels[i] = np.argmin(distances[i])
            new_centers = np.zeros((self.n_clusters, n_features), dtype=np.float32)
            counts = np.zeros(self.n_clusters)
            for i in range(n_samples):
                new_centers[int(self.labels[i])] += X[i]
                counts[int(self.labels[i])] += 1
            new_centers /= counts[:, np.newaxis]
            center_shift = np.sqrt(np.sum((self.centroids - new_centers) ** 2, axis=1))
            self.centroids = new_centers
            if np.sum(center_shift) < self.tol:
                break
        return self

    def fit_predict(self, X):
        return self.fit(X).labels

    def predict(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        labels = np.empty(n_samples)
        for i in range(n_samples):
            for c in range(self.n_clusters):
                distances[i][c] = np.sum(np.absolute(X[i] - self.centroids[c]) ** 2) ** 1.0 / 2
            labels[i] = np.argmin(distances[i])
        return labels
