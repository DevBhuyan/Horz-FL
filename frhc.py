#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from FAR_based_outlier_detection import symmetric_uncertainty
import copy

cor = symmetric_uncertainty
"""This code is based on the paper titled "Federated Feature Selection for
Horizontal Federated Learning in IoT Networks" by Zhang et.

Al. This is derived from the algorithmic descriptions given in the text. The function definitions follow henceforth. This implementation is invoked in the `sota.py` file
"""


def Gavg(Q, df):
    """Calculate the group average linkage measure and find the minimum.

    Parameters:
    Q (list): List of triples containing delta (symmetric uncertainty) and feature indices.
    df (pd.DataFrame): Input DataFrame.

    Returns:
    min_avg (float): Minimum average linkage measure.
    min_i (str): Index of the first feature in the minimum average linkage.
    min_j (str): Index of the second feature in the minimum average linkage.
    """
    min_avg = float("inf")
    min_i, min_j = None, None

    for q in Q:
        delta, i, j = q
        avg = compute_group_average(df[i], df[j])

        if avg < min_avg:
            min_avg = avg
            min_i = i
            min_j = j

    return min_avg, min_i, min_j


def compute_group_average(cluster_i, cluster_j):
    """Calculate the average linkage measure between two clusters.

    Parameters:
    cluster_i (list): Data of the first cluster.
    cluster_j (list): Data of the second cluster.

    Returns:
    average (float): Average linkage measure.
    """
    # Calculate the average linkage measure between cluster_i and cluster_j
    total_distance = 0
    count = 0

    for element_i in cluster_i:
        for element_j in cluster_j:
            distance = calculate_distance(element_i, element_j)
            total_distance += distance
            count += 1

    average = total_distance / count

    return average


def calculate_distance(element_i, element_j):
    """Calculate the Euclidean distance between two elements.

    Parameters:
    element_i (numpy.ndarray): First element.
    element_j (numpy.ndarray): Second element.

    Returns:
    distance (float): Euclidean distance.
    """
    distance = np.linalg.norm(element_i - element_j)
    return distance


def replace_columns_with_data(clusters, df):
    """Replace feature indices in clusters with actual data from the DataFrame.

    Parameters:
    clusters (list): List of clusters containing feature indices.
    df (pd.DataFrame): Input DataFrame.

    Returns:
    clusters_copy (list): List of clusters with actual data.
    """
    clusters_copy = copy.deepcopy(clusters)

    for i in range(len(clusters_copy)):
        cluster = clusters_copy[i]
        for j in range(len(cluster)):
            feature = cluster[j]
            cluster[j] = df[feature].values.tolist()

    return clusters_copy


def SCK(clusters, df):
    """Compute the Silhouette Coefficient for a set of clusters.

    Parameters:
    clusters (list): List of clusters containing data.
    df (pd.DataFrame): Input DataFrame.

    Returns:
    avg_silhouette (float): Average Silhouette Coefficient.
    """
    DL = np.array(replace_columns_with_data(clusters, df))
    silhouette_scores = []

    for cluster_idx, cluster in enumerate(DL):
        for sample in cluster:
            a = np.mean(
                [np.linalg.norm(sample - other_sample) for other_sample in cluster]
            )
            b = min(
                [
                    np.mean(
                        [
                            np.linalg.norm(sample - other_sample)
                            for other_sample in other_cluster
                        ]
                    )
                    for other_cluster in DL
                    if not np.array_equal(other_cluster, cluster)
                ]
            )
            silhouette_scores.append((b - a) / max(a, b))

    avg_silhouette = np.mean(silhouette_scores)

    return avg_silhouette


def frhc(df):
    """Federated Recursive Hierarchical Clustering (FRHC) for feature
    selection.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    clusters (list): Final clusters obtained from FRHC.
    sc (float): Silhouette Coefficient for the final set of clusters.
    """
    e = len(df.columns)
    Q = []
    clusters = df.columns.tolist()
    sc = {}

    for j in range(1, e):
        for i in range(j):
            delta = cor(df[df.columns[i]], df[df.columns[j]])
            Q.append((delta, df.columns[i], df.columns[j]))
    for t in range(e):
        min_avg, min_i, min_j = Gavg(Q, df)
        for i in clusters:
            if isinstance(i, list):
                if min_i in i:
                    new = i
                    new.append(min_j)
                    clusters.remove(min_j)
                    clusters.remove(i)
                    clusters.append(new)
                    break
                elif min_j in i:
                    new = i
                    new.append(min_i)
                    clusters.remove(min_i)
                    clusters.remove(i)
                    clusters.append(new)
                    break
            elif i == min_i or i == min_j:
                clusters.remove(min_i)
                clusters.remove(min_j)
                clusters.append([min_i, min_j])
                break
        Q = [
            q
            for q in Q
            if not (q[1] == min_j or q[2] == min_j or q[1] == min_i or q[2] == min_i)
        ]
    sc = SCK(clusters, df)
    return clusters, sc
