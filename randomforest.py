#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Created on Tue Jun 13 22:49:20 2023.

@author:
"""

from ff_helpers import horz_split, trainModel
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm
from datetime import datetime
import gc

gc.enable()


def rf(df_list: list):
    """Wrapper function for individual random forest (not federated)

    Parameters
    ----------
    df_list : list

    Returns
    -------
    rf_acc : float
        accuracy.
    rf_prec : float
        precision.
    rf_rec : float
        recall.
    """
    x_train, y_train, x_test, y_test = horz_split(df_list)

    start = datetime.now()

    models = []
    for x, y in tqdm(zip(x_train, y_train), total=len(x_train)):
        models.append(trainModel(x, y))

    fed_acc = []
    fed_p = []
    fed_r = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        y_pred = models[i].predict(x)
        fed_acc.append(accuracy_score(y, y_pred))
        fed_p.append(precision_score(y, y_pred, average="weighted"))
        fed_r.append(recall_score(y, y_pred, average="weighted"))
    rf_acc = sum(fed_acc) / len(fed_acc)
    rf_prec = sum(fed_p) / len(fed_p)
    rf_rec = sum(fed_r) / len(fed_r)

    print(
        "\033[1;33m"
        + f"\nAverage training time per client : {(datetime.now()-start)/len(df_list)}"
        + "\033[0m"
    )

    del models
    return rf_acc, rf_prec, rf_rec
