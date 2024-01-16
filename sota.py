#!/usr/bin/env python
# -*- coding: utf-8 -*-

from FAR_based_outlier_detection import FAR_based_outlier_detection
from frhc import frhc
from horz_data_divn import horz_data_divn

"""
This code is based on the paper titled "Federated Feature Selection for Horizontal Federated Learning in IoT Networks" by Zhang et. Al. This is derived from the algorithmic descriptions given in the text. The function definitions follow henceforth. This implementation is invoked in the `sota_driver.py` file
"""


def lfs(dataset, n_client, t):
    """Local-Feature Selection component of FSHFL.

    Parameters:
    - dataset (str): The name of the dataset.
    - n_client (int): Number of clients for horizontal federated learning.
    - t (float): Threshold for FAR-based outlier detection.

    Returns:
    - new_df_list (list): List of cleaned DataFrames after local feature selection.
    """
    df_list = horz_data_divn(dataset, n_client)
    new_df_list = []

    for df in df_list:
        y = df.iloc[:, -1]

        # Apply FAR-based outlier detection
        filtered_df, r, _ = FAR_based_outlier_detection(df, threshold=t)

        # Apply FRHC clustering for feature grouping
        clusters, sc = frhc(filtered_df.iloc[:, :-1])

        most_relevant_features = []

        for cluster in clusters:
            # Find the most relevant feature in each cluster based on FAR relevance scores
            max_relevance_score = 0
            most_relevant_feature = None
            for feature in cluster:
                relevance_score = r[feature]
                if relevance_score > max_relevance_score:
                    max_relevance_score = relevance_score
                    most_relevant_feature = feature

            most_relevant_features.append(most_relevant_feature)

        cleaned_df = df[most_relevant_features]
        cleaned_df = cleaned_df.assign(Class=y)
        new_df_list.append(cleaned_df)

    return new_df_list


def gfs(df_list):
    """Global-Feature Selection component of FSHFL.

    Parameters:
    - df_list (list): List of cleaned DataFrames after local feature selection.

    Returns:
    - new_df_list (list): List of DataFrames with globally common features.
    """
    common_features = set(df_list[0].iloc[:, :-1].columns.tolist())
    new_df_list = []

    for df in df_list[1:]:
        common_features = common_features.intersection(set(df.columns.tolist()))

    common_features_list = list(common_features)

    print("Global features: ", common_features_list)

    for df in df_list:
        y = df["Class"]
        new_df = df[common_features_list]
        new_df = new_df.assign(Class=y)
        new_df_list.append(new_df)

    return new_df_list


def far(dataset, n_client=5, t=0.5):
    """Federated Feature Selection for Horizontal Federated Learning (FSHFL).

    Parameters:
    - dataset (str): The name of the dataset.
    - n_client (int): Number of clients for horizontal federated learning (default: 5).
    - t (float): Threshold for FAR-based outlier detection (default: 0.5).

    Returns:
    - new_df_list (list): List of DataFrames after global feature selection.
    """
    return gfs(lfs(dataset, n_client, t))
