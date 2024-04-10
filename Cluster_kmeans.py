import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import gc

gc.enable()


def Cluster_kmeans(MI: list, MI_df: pd.DataFrame, k: int, flag1: int):
    """Perform K-Means clustering on Mutual Information data.

    Parameters
    ----------
    MI : list
        List of lists containing Mutual Information of feature vs. feature.
    MI_df : pd.DataFrame
        DataFrame containing Mutual Information values.
    k : int
        Number of clusters.
    flag1 : int
        Flag indicating whether to choose the minimum or maximum cluster.

    Returns
    -------
    cluster_label_list : list
        List of cluster labels assigned to each data point.
    cluster_center_list : list
        List of cluster centers.
    cluster_map : pd.DataFrame
        DataFrame mapping data indices to cluster labels.
    val : int
        Chosen cluster value based on the flag (minimum or maximum).
    """
    MI_Ffmi1 = np.array(MI)

    clusterer = KMeans(n_clusters=k, init="k-means++",
                       random_state=1).fit(MI_Ffmi1)

    cluster_map = pd.DataFrame()
    cluster_map["data_index"] = MI_df.index.values
    cluster_map["cluster"] = clusterer.labels_

    cluster_label_list = clusterer.labels_.tolist()
    cluster_center_list = clusterer.cluster_centers_.tolist()

    dict_ccl = {}
    for inx in range(0, len(cluster_center_list)):
        flag = 0
        for i in range(0, len(cluster_label_list)):
            if cluster_label_list[i] == inx and flag == 0:
                dict_ccl[cluster_center_list[inx][0]] = cluster_label_list[i]
                flag = 1

    # Create an inverse dictionary for mapping cluster labels to cluster centers
    inv_ccl = {v: k for k, v in dict_ccl.items()}

    # Choose the cluster value based on the flag (minimum or maximum)
    if flag1 == 0:
        val = min(inv_ccl, key=inv_ccl.get)
    else:
        val = max(inv_ccl, key=inv_ccl.get)

    silhouette_score(MI_Ffmi1, clusterer.labels_)

    sample_silhouette_values = silhouette_samples(MI_Ffmi1, clusterer.labels_)

    return cluster_label_list, cluster_center_list, cluster_map, val
