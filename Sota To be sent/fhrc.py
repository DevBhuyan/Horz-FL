#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 00:24:45 2023

@author: 
"""

import numpy as np
from FAR_based_outlier_detection import symmetric_uncertainty
from sklearn.metrics import silhouette_samples

cor = symmetric_uncertainty

def Gavg(Q, DL):
    min_avg = float('inf')
    min_i, min_j = None, None

    for q in Q:
        delta, i, j = q
        avg = compute_group_average(DL[i], DL[j])

        if avg < min_avg:
            min_avg = avg
            min_i = i
            min_j = j

    Q.remove([min_avg, min_i, min_j])
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


def SCK(DL, k):
    # Convert the DL to a numpy array
    data = np.array(DL)

    # Compute the silhouette scores for each sample
    silhouette_scores = silhouette_samples(data, k)

    # Calculate the average silhouette coefficient for all samples
    avg_silhouette = np.mean(silhouette_scores)

    return avg_silhouette

def frhc(DL):
    e = len(DL)  # Number of features
    Q = []  # Initialize the Q matrix

    # Step 2: Compute correlation distances and store in Q
    for i in range(e):
        for j in range(i + 1, e):
            delta = cor(DL[i], DL[j])  # Compute correlation distance
            Q.append([delta, i, j])

    # Step 5: Perform agglomerative hierarchical clustering
    for t in range(e - 1):
        delta, i, j = Gavg(Q)
        DL[i] = DL[i] + DL[j]  # Merge clusters
        DL.pop(j)  # Remove cluster j from DL

        # Update Q matrix
        Q = [q for q in Q if (q[1] != i) and (q[2] != i) and (q[1] != j) and (q[2] != j)]
        for h in range(len(Q)):
            if Q[h][1] == h:
                delta = cor(h, i)
                Q.append([delta, h, i])
            else:
                delta = cor(i, h)
                Q.append([delta, i, h])

    # Step 17: Find the optimal number of clusters (K)
    SCK_values = []
    for K in range(2, e + 1):
        SCK_value = SCK(DL)
        SCK_values.append((SCK_value, K))
    SCK_values.sort(reverse=True)

    return DL, SCK_values[0][1]  # Return clustering results (DL) and optimal K
