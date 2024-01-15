#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Created on Tue Jul 4 17:55:23 2023.

@author: 
"""

import numpy as np
import pandas as pd

"""
This code is based on the paper titled "Federated Feature Selection for Horizontal Federated Learning in IoT Networks" by Zhang et. Al. This is derived from the algorithmic descriptions given in the text. The function definitions follow henceforth. This implementation is invoked in the `sota.py` file
"""


def mutual_information(fx, fy):
    """Calculate the mutual information between two features.

    Parameters:
    fx (pd.Series): First feature.
    fy (pd.Series): Second feature.

    Returns:
    mi (float): Mutual information between fx and fy.
    """
    counts_fx = fx.value_counts()
    counts_fy = fy.value_counts()
    counts_joint = pd.crosstab(fx, fy)

    prob_fx = counts_fx / len(fx)
    prob_fy = counts_fy / len(fy)
    prob_joint = counts_joint / len(fx)

    mi = 0
    for x in prob_fx.index:
        for y in prob_fy.index:
            if prob_joint.loc[x, y] != 0:
                mi += prob_joint.loc[x, y] * np.log2(
                    prob_joint.loc[x, y] / (prob_fx[x] * prob_fy[y])
                )

    return mi


def entropy(feature):
    """Calculate the entropy of a feature.

    Parameters:
    feature (pd.Series): Input feature.

    Returns:
    entropy (float): Entropy of the feature.
    """
    value_counts = feature.value_counts()
    probabilities = value_counts / len(feature)
    entropy = -sum(probabilities * np.log2(probabilities))
    return entropy


def symmetric_uncertainty(fx, fy):
    """Calculate the symmetric uncertainty between two features.

    Parameters:
    fx (pd.Series): First feature.
    fy (pd.Series): Second feature.

    Returns:
    su (float): Symmetric uncertainty between fx and fy.
    """
    mi = mutual_information(fx, fy)
    entropy_fx = entropy(fx)
    entropy_fy = entropy(fy)

    if entropy_fx + entropy_fy == 0:
        return 0

    su = (2 * mi) / (entropy_fx + entropy_fy)
    return su


def feature_relevance(feature, dataframe):
    """Calculate the relevance of a feature in the context of others.

    Parameters:
    feature (str): Feature to calculate relevance for.
    dataframe (pd.DataFrame): Input DataFrame.

    Returns:
    relevance (float): Relevance of the feature.
    """
    relevance_sum = 0
    num_features = dataframe.shape[1]

    for i in range(num_features):
        if dataframe.columns[i] != feature:
            relevance_sum += symmetric_uncertainty(
                dataframe[feature], dataframe[dataframe.columns[i]]
            )

    relevance = relevance_sum / (num_features - 1)
    return relevance


def FAR_based_outlier_detection(dataframe, threshold):
    """Perform Feature Association Rule (FAR)-based outlier detection.

    Parameters:
    dataframe (pd.DataFrame): Input DataFrame.
    threshold (float): Threshold for relevance to determine outliers.

    Returns:
    filtered_df (pd.DataFrame): DataFrame with relevant features.
    r (dict): Dictionary of relevant features and their relevance scores.
    ol (list): List of outlier features.
    """
    relevant_features = []
    ol = []
    r = {}
    y = dataframe.iloc[:, -1]

    for feature in dataframe.columns[:-1]:
        relevance = feature_relevance(feature, dataframe)

        if relevance > threshold:
            relevant_features.append(feature)
            r[feature] = relevance
        else:
            ol.append(feature)

    filtered_df = dataframe[relevant_features]
    filtered_df = filtered_df.assign(Class=y)
    return filtered_df, r, ol
