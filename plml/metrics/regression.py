#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def mean_squared_error(y_true, y_pred):
    output_errors = np.average((y_true - y_pred) ** 2)
    return output_errors

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
