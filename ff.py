#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ff_helpers import horz_split, trainModel, federatedForest
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm
from datetime import datetime
import numpy as np
import gc

gc.enable()
"""Federated Forest Implementation for Horizontal Federated Learning.

This script implements a federated learning approach using a federated forest for classification.

Workflow:
    Fetch df_list
    Split train_test divn
    Split X, y
    Train locally
    Aggregate models
    Evaluate
"""


def ff(df_list, max_depth=200):
    """Train and evaluate a federated forest on horizontally federated data.

    Parameters:
    - df_list (list): List of DataFrames containing data from different clients.
    - max_depth (int): Maximum depth of trees in the federated forest (default: 200).

    Returns:
    - ff_acc (float): Average accuracy of the federated forest on the test sets.
    - ff_prec (float): Average precision of the federated forest on the test sets.
    - ff_rec (float): Average recall of the federated forest on the test sets.
    - max_depth (int): Maximum depth of trees in the federated forest.
    - total_leaves (int): Total number of leaves in the federated forest.
    """
    x_train, y_train, x_test, y_test = horz_split(df_list)

    start = datetime.now()

    models = []
    for x, y in tqdm(zip(x_train, y_train), total=len(x_train)):
        models.append(trainModel(x, y, max_depth))

    fed = federatedForest(models)

    fed_acc = []
    fed_p = []
    fed_r = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        fed_y_pred = fed.predict(x)
        fed_acc.append(accuracy_score(y, fed_y_pred))
        fed_p.append(precision_score(y, fed_y_pred, average="weighted"))
        fed_r.append(recall_score(y, fed_y_pred, average="weighted"))
    ff_acc = sum(fed_acc) / len(fed_acc)
    ff_prec = sum(fed_p) / len(fed_p)
    ff_rec = sum(fed_r) / len(fed_r)

    print(
        "\033[1;33m"
        + "\nAverage training time per client :"
        f" {(datetime.now()-start)/len(df_list)} \nSize of federated forest:"
        f" {len(fed.estimators_)} trees\nMax tree depth:"
        f" {max([estimator.get_depth() for estimator in fed.estimators_])}\nTotal"
        " number of leaves in federated forest:"
        f" {np.sum([estimator.get_n_leaves() for estimator in fed.estimators_])}\n"
        + "\033[0m"
    )
    returned_max_depth = max([estimator.get_depth() for estimator in fed.estimators_])
    total_leaves = np.sum([estimator.get_n_leaves() for estimator in fed.estimators_])

    del models
    del fed

    if max_depth == 200:
        return ff_acc, ff_prec, ff_rec
    else:
        return ff_acc, ff_prec, ff_rec, returned_max_depth, total_leaves
