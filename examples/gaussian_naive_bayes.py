#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np

import plml

X, y = plml.datasets.load_iris()

encoder = plml.preprocessing.LabelEncoder()
y = encoder.fit_transform(y)

n_features = X.shape[1]

X_train, X_test, y_train, y_test = plml.model_selection.train_test_split(X, y)

p_gnb = plml.naive_bayes.GaussianNB()
p_gnb.fit(X_train, y_train)
y_pred = p_gnb.predict(X_test)

print plml.metrics.accuracy_score(y_pred, y_test)
