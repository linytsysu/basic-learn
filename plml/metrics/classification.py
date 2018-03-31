#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def accuracy_score(y_true, y_pred, normalize=True):
    score = y_true == y_pred
    if normalize:
        return np.average(score)
    else:
        return score.sum()
