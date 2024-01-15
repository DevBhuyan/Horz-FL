#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:22:53 2023

@author: 
"""

from FAR_based_outlier_detection import FAR_based_outlier_detection
from frhc import frhc
from horz_data_divn import horz_data_divn

def lfs(dataset, n_client, t):

    df_list = horz_data_divn('diabetes', 5)
    new_df_list = []

    for df in df_list:
        y = df.iloc[:, -1]
        filtered_df, r, _ = FAR_based_outlier_detection(df, threshold=t)
        clusters, sc = frhc(filtered_df.iloc[:, :-1])

        most_relevant_features = []

        for cluster in clusters:
            max_relevance_score = 0
            most_relevant_feature = None
            for feature in cluster:
                relevance_score = r[feature]
                if relevance_score > max_relevance_score:
                    max_relevance_score = relevance_score
                    most_relevant_feature = feature

            most_relevant_features.append(most_relevant_feature)

        cleaned_df = df[most_relevant_features]
        cleaned_df = cleaned_df.assign(Class = y)
        new_df_list.append(cleaned_df)

    return new_df_list

def gfs(df_list):
    common_features = set(df_list[0].iloc[:, :-1].columns.tolist())
    new_df_list = []

    for df in df_list[1:]:
        common_features = common_features.intersection(set(df.columns.tolist()))

    common_features_list = list(common_features)
    print('Global features: ', common_features_list)

    for df in df_list:
        y = df['Class']
        new_df = df[common_features_list]
        new_df = new_df.assign(Class = y)
        new_df_list.append(new_df)

    return new_df_list

def far(dataset, n_client = 5, t = 0.5):
    return gfs(lfs(dataset, n_client, t))
