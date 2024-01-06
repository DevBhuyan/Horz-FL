#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 00:24:45 2023

@author: dev
"""

import numpy as np
import pandas as pd
from FAR_based_outlier_detection import symmetric_uncertainty
from sklearn.metrics import silhouette_samples, pairwise_distances
import copy

cor = symmetric_uncertainty

def Gavg(Q, df):
    min_avg = float('inf')
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
    # Calculate the average linkage measure between cluster_i and cluster_j
    total_distance = 0
    count = 0

    # Iterate over all pairs of elements in the clusters
    for element_i in cluster_i:
        for element_j in cluster_j:
            # Calculate the pairwise distance between element_i and element_j
            distance = calculate_distance(element_i, element_j)
            total_distance += distance
            count += 1

    # Compute the average linkage measure
    average = total_distance / count

    return average

def calculate_distance(element_i, element_j):
    # Assuming element_i and element_j are numerical vectors of the same length
    distance = np.linalg.norm(element_i - element_j)
    return distance

def replace_columns_with_data(clusters, df):
    # Make a deep copy of the clusters list
    clusters_copy = copy.deepcopy(clusters)

    for i in range(len(clusters_copy)):
        cluster = clusters_copy[i]
        for j in range(len(cluster)):
            feature = cluster[j]
            cluster[j] = df[feature].values.tolist()

    return clusters_copy


def SCK(clusters, df):
    DL = np.array(replace_columns_with_data(clusters, df))
    silhouette_scores = []

    for cluster_idx, cluster in enumerate(DL):
        for sample in cluster:
            a = np.mean([np.linalg.norm(sample - other_sample) for other_sample in cluster])
            b = min([np.mean([np.linalg.norm(sample - other_sample) for other_sample in other_cluster]) for other_cluster in DL if not np.array_equal(other_cluster, cluster)])
            silhouette_scores.append((b - a) / max(a, b))

    avg_silhouette = np.mean(silhouette_scores)

    return avg_silhouette


def frhc(df):
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
        Q = [q for q in Q if not (q[1] == min_j or q[2] == min_j or q[1] == min_i or q[2] == min_i)]
    # for k in range(2, e):
        # sc[k] = SCK(clusters, k, df)
    # scl = sorted(sc, key = lambda x:sc[x], reverse = True)
    sc = SCK(clusters, df)
    return clusters, sc