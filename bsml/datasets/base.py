#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

def load_boston():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), './data/boston_house_prices.csv'), header=1)
    X = data.iloc[:, :-1].values
    y = data.MEDV.values
    return X, y

def load_iris():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), './data/iris.csv'), header=0)
    X = data.iloc[:, :-1].values
    y = data.Name.values
    return X, y

def load_wine():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), './data/wine.csv'), header=None)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    return X, y
