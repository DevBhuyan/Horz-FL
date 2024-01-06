#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:55:23 2023

@author: dev
"""


import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from datetime import datetime

def symmetric_uncertainty(fx, fy):
    # THIS CODE IS 10X FASTER THAN ITS PREDECESSOR AND 50X FASTER THAN ORIGINAL
    start = datetime.now()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fx_arr = fx.to_numpy()
        fy_arr = fy.to_numpy()

        fx_arr = (fx_arr*100).astype(np.uint8)
        fy_arr = (fy_arr*100).astype(np.uint8)

        fx_min = np.min(fx_arr)
        fy_min = np.min(fy_arr)
        fx_arr = fx_arr - fx_min
        fy_arr = fy_arr - fy_min

        unique_fx, counts_fx = np.unique(fx_arr, return_counts=True)
        unique_fy, counts_fy = np.unique(fy_arr, return_counts=True)

        # FIXME: Takes 0.007s
        joint_counts, _, _ = np.histogram2d(fx_arr, fy_arr, bins=[unique_fx.size, unique_fy.size])
        joint_counts += 1e-9
        # print(datetime.now()-start)

        # FIXME: Takes 0.004s
        prob_fx = counts_fx / len(fx_arr)
        prob_fy = counts_fy / len(fy_arr)
        prob_joint = joint_counts / len(fx_arr)
        # print(datetime.now()-start)

        # FIXME: Takes 0.026s
        mi = np.sum(prob_joint * np.log2(prob_joint / np.outer(prob_fx, prob_fy)))
        # print(datetime.now()-start)

        entropy_fx = -np.sum(prob_fx * np.log2(prob_fx))
        entropy_fy = -np.sum(prob_fy * np.log2(prob_fy))

        if entropy_fx + entropy_fy == 0:
            return 0

        su = (2 * mi) / (entropy_fx + entropy_fy)
        # print('\n\n\n\n')
    return su


def feature_relevance(feature, dataframe):
    relevance_sum = 0
    num_features = dataframe.shape[1]

    for i in range(num_features):
        if dataframe.columns[i] != feature:
            relevance_sum += symmetric_uncertainty(dataframe[feature], dataframe[dataframe.columns[i]])
    relevance = relevance_sum / (num_features - 1)
    return relevance

def FAR_based_outlier_detection(dataframe : pd.DataFrame, 
                                threshold : float):
    '''
    FAR_based_outlier_detection function of FSHFL

    Parameters
    ----------
    dataframe : pd.DataFrame
    threshold : float

    Returns
    -------
    filtered_df : pd.DataFrame

    '''
    relevant_features = []
    ol = []
    r = {}
    y = dataframe.iloc[:, -1]


    for feature in tqdm(dataframe.columns[:-1], total=len(dataframe.columns[:-1])):
        relevance = feature_relevance(feature, dataframe)

        if relevance > threshold:
            relevant_features.append(feature)
            r[feature] = relevance
        else:
            ol.append(feature)

    filtered_df = dataframe[relevant_features]
    filtered_df = filtered_df.assign(Class = y)
    return filtered_df, r, ol
