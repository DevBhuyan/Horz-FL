#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ff_helpers import (
    horz_split,
    trainModel,
    federatedForest,
    federated_ensemble
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score
)
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
import gc

gc.enable()


def ff(df_list: list,
       num_classes: int,
       non_iid: bool = False,
       max_depth: int = 200):
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

    x_train, y_train, x_test, y_test = horz_split(df_list)

    start = datetime.now()

    models = []
    for x, y in tqdm(zip(x_train, y_train), total=len(x_train)):
        models.append(trainModel(x, y, num_classes, max_depth))

    x_test_concat = pd.concat([pd.DataFrame(x)
                              for x in x_test], axis=0, ignore_index=True)
    y_test_concat = pd.concat([pd.DataFrame(y)
                              for y in y_test], axis=0, ignore_index=True)

    if not non_iid:
        fed = federatedForest(models)

        print("Evaluating....")
        x_test_concat = np.nan_to_num(x_test_concat)

        fed_y_pred = fed.predict(x_test_concat)
        ff_acc = accuracy_score(y_test_concat, fed_y_pred)
        ff_f1 = f1_score(y_test_concat, fed_y_pred, average="weighted")

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

        returned_max_depth = max([estimator.get_depth()
                                 for estimator in fed.estimators_])
        total_leaves = np.sum([estimator.get_n_leaves()
                              for estimator in fed.estimators_])

    else:
        # TODO: Run the Federated Ensembling method
        fed_y_pred = federated_ensemble(models, x_test)
        ff_acc = mean_squared_error(y_test_concat, fed_y_pred, squared=False)
        ff_f1 = r2_score(y_test_concat, fed_y_pred)

        print(
            "\033[1;33m"
            + "\nAverage training time per client :"
            f" {(datetime.now()-start)/len(df_list)}"
            + "\033[0m"
        )

    if max_depth == 200:
        return ff_acc, ff_f1
    else:
        return ff_acc, ff_f1, returned_max_depth, total_leaves
