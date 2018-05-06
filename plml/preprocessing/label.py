#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class LabelEncoder:
    def __init__(self):
        pass

    def fit(self, y):
        self.classes_ = np.unique(y)

    def fit_transform(self, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        return y

    def transform(self, y):
        classes = np.unique(y)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            raise ValueError("y contains new labels: %s" % str(diff))
        return np.searchsorted(self.classes_, y)
