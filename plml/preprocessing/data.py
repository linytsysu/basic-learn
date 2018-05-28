#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def _handle_zeros_in_scale(scale):
    scale[scale == 0.0] = 1.0
    return scale

class MinMaxScaler:
    '''
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min
    '''
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X):
        data_min = X.min(axis=0)
        data_max = X.max(axis=0)
        data_range = data_max - data_min
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / _handle_zeros_in_scale(data_range)
        self.min_ =  self.feature_range[0] - data_min * self.scale_
        return self


    def transform(self, X):
        X *= self.scale_
        X += self.min_
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class StandardScaler:
    '''
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    '''
    def __init__(self):
        pass
    
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / _handle_zeros_in_scale(self.std_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
