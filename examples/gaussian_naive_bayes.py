#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np

import bsml

X, y = bsml.datasets.load_iris()

encoder = bsml.preprocessing.LabelEncoder()
y = encoder.fit_transform(y)

n_features = X.shape[1]

X_train, X_test, y_train, y_test = bsml.model_selection.train_test_split(X, y)

p_gnb = bsml.naive_bayes.GaussianNB()
p_gnb.fit(X_train, y_train)
y_pred = p_gnb.predict(X_test)

print bsml.metrics.accuracy_score(y_pred, y_test)
