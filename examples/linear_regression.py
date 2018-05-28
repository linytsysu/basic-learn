#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np

import plml

X, y = plml.datasets.load_boston()

X_train, X_test, y_train, y_test = plml.model_selection.train_test_split(X, y, test_size=0.2, random_state=999)

scaler = plml.preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)

X_train_with_bias = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_with_bias = np.c_[np.ones((len(X_test), 1)), X_test]

lr = plml.linear_model.LinearRegression()
lr.fit(X_train_with_bias, y_train, alpha=0.1, loop=1000)
y_pred = lr.predict(X_test_with_bias)

rmse = plml.metrics.root_mean_squared_error(y_test, y_pred)
print rmse
