#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..utils import check_random_state

def safe_indexing(X, indices):
    if hasattr(X, "iloc"):
        indices = indices if indices.flags.writeable else indices.copy()
        return X.iloc[indices]
    elif hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]


def train_test_split(X, y, test_size=0.25, random_state=None):
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    rng = check_random_state(random_state)
    permutation = rng.permutation(n_samples)
    ind_test = permutation[:n_test]
    ind_train = permutation[n_test:(n_test + n_train)]
    return safe_indexing(X, ind_train), safe_indexing(X, ind_test), safe_indexing(y, ind_train), safe_indexing(y, ind_test)
