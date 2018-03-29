#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class MinMaxScaler:
    def __init__(self):
        pass

    def fix_transform(self, X):
        data_min = X.min(axis=0)
        data_max = X.max(axis=0)
        scale = data_max - data_min
        scale[scale == 0.0] = 1.0
        return (X - data_min) / scale
