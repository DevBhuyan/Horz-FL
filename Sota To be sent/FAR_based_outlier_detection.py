#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:55:23 2023

@author: dev
"""

import numpy as np
import pandas as pd

def mutual_information(fx, fy):
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
                mi += prob_joint.loc[x, y] * np.log2(prob_joint.loc[x, y] / (prob_fx[x] * prob_fy[y]))

    return mi


def entropy(feature):
    value_counts = feature.value_counts()
    probabilities = value_counts / len(feature)
    entropy = -sum(probabilities * np.log2(probabilities))
    return entropy


def symmetric_uncertainty(fx, fy):
    mi = mutual_information(fx, fy)
    entropy_fx = entropy(fx)
    entropy_fy = entropy(fy)

    if entropy_fx + entropy_fy == 0:
        return 0

    su = (2 * mi) / (entropy_fx + entropy_fy)
    return su

def feature_relevance(feature, dataframe):
    relevance_sum = 0
    num_features = dataframe.shape[1]

    for i in range(num_features):
        if dataframe.columns[i] != feature:
            relevance_sum += symmetric_uncertainty(dataframe[feature], dataframe[dataframe.columns[i]])

    relevance = relevance_sum / (num_features - 1)
    return relevance

def FAR_based_outlier_detection(dataframe, threshold):
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
    filtered_df = filtered_df.assign(Class = y)
    return filtered_df, r, ol