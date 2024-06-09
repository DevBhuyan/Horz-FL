#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import math
import numpy as np
from warnings import warn
import gc

gc.enable()


N_JOBS = -1
warn(f"Using {N_JOBS if N_JOBS > 0 else 'all'} out of 12 cores for training RandomForests. Update N_JOBS in ff_helpers.py as required")


def horz_split(df_list):
    """Horizontal split of a list of DataFrames into training and testing sets.

    Parameters
    ----------
    df_list : list
        List of DataFrames to be split.

    Returns
    -------
    x_train, y_train, x_test, y_test : list
        Lists containing training and testing data.
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for df in df_list:
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        x_train.append(train.iloc[:, :-1])
        y_train.append(train.iloc[:, -1])
        x_test.append(test.iloc[:, :-1])
        y_test.append(test.iloc[:, -1])

    return x_train, y_train, x_test, y_test


def trainModel(x,
               y,
               num_classes,
               max_depth=200):
    """Train a Random Forest classifier.

    Parameters
    ----------
    x : array-like or pd.DataFrame
        Input features.
    y : array-like or pd.Series
        Target variable.
    max_depth : int, optional
        Maximum depth of the trees in the forest. Default is 200.

    Returns
    -------
    clf : RandomForestClassifier
        Trained Random Forest classifier.
    """
    if num_classes > 0:
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=max_depth,
            random_state=42,
            n_jobs=N_JOBS)
    else:
        clf = RandomForestRegressor(
            random_state=42,
            n_jobs=N_JOBS)
    clf.fit(np.nan_to_num(x), np.nan_to_num(y))

    # print(
    #     "\033[1;33m"
    #     + f"\nSize of random forest: {len(clf.estimators_)} trees\nMax tree depth:"
    #     f" {max([estimator.get_depth() for estimator in clf.estimators_])}\nTotal"
    #     " number of leaves:"
    #     f" {np.sum([estimator.get_n_leaves() for estimator in clf.estimators_])}\n\n"
    #     + "\033[0m"
    # )
    return clf


def aggregateForests(agg, model, n):
    """Aggregate multiple Random Forest models.

    Parameters
    ----------
    agg : RandomForestClassifier
        Aggregated Random Forest model.
    model : RandomForestClassifier
        Individual Random Forest model to be aggregated.
    n : int
        Desired size of the aggregated forest.

    Returns
    -------
    agg : RandomForestClassifier
        Aggregated Random Forest model.
    """
    # Aggregating individual forests
    size = model.n_estimators
    dupl = math.floor(n / size)
    for i in range(dupl):
        agg.estimators_ += model.estimators_
        agg.n_estimators = len(agg.estimators_)

    return agg


def getMaxForest(model_list):
    """Get the maximum size of Random Forests in a list.

    Parameters
    ----------
    model_list : list
        List of Random Forest models.

    Returns
    -------
    max_size : int
        Maximum size of Random Forests in the list.
    """
    sizes = [len(model.estimators_) for model in model_list]
    return max(sizes)


def federatedForest(model_list):
    """Generate an aggregated Federated Forest model.

    Parameters
    ----------
    model_list : list
        List of independently trained Random Forest models.

    Returns
    -------
    ff : RandomForestClassifier
        Federated Forest.
    """
    ff = model_list[0]
    max_size = getMaxForest(model_list)

    # Aggregate models into the federated forest
    for i, model in enumerate(model_list):
        if i > 0:
            ff = aggregateForests(ff, model_list[i], max_size)

    return ff


def federated_ensemble(models: list[RandomForestClassifier],
                       X_test):
    """Generate an aggregated Federated Ensemble prediction using majority voting.
    Parameters
    ----------
    models : list
        List of trained Random Forest models from each client.
    X_test : array-like or pd.DataFrame
        Input features for prediction.
    Returns
    -------
    ensemble_predictions : array-like
        Aggregated predictions using majority voting.
    """
    predictions_list = []
    for model, x_test in tqdm(zip(models, X_test), desc="Ensembling...", total=len(models)):
        predictions = model.predict(np.nan_to_num(x_test))
        predictions_list.extend(predictions)

    predictions_list = np.array(predictions_list).astype(int)

    # ensemble_predictions = np.apply_along_axis(
    #     lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions_list)

    # 27/05/2024 ---------------------------------------------------------------
    # max_length = max(len(pred) for pred in predictions_list)

    # padded_predictions_list = []
    # for preds in predictions_list:
    #     if len(preds) < max_length:
    #         padded_preds = np.pad(
    #             preds, (0, max_length - len(preds)), constant_values=-1)
    #     else:
    #         padded_preds = preds
    #     padded_predictions_list.append(padded_preds)

    # predictions_list = np.array(padded_predictions_list).astype(int)

    # def majority_vote(x):
    #     counts = np.bincount(x[x != -1])
    #     return np.argmax(counts) if len(counts) > 0 else -1

    # ensemble_predictions = np.apply_along_axis(
    #     majority_vote, axis=0, arr=predictions_list)

    return predictions_list
