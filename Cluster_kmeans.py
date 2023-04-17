import matplotlib.cm as cm
import numpy as np
from matplotlib.pyplot import show, suptitle, subplots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd


def Cluster_kmeans(MI, MI_df, k, flag1):
    # MI_Fcmi1 = np.array(MI_Fcmi)

    MI_Ffmi1 = np.array(MI)
    # print(MI_df)
    # range_n_clusters = [2, 3, 4]

    clusterer = KMeans(n_clusters=k, init='k-means++', random_state=1).fit(MI_Ffmi1)

    # cluster_labels = clusterer.fit_predict(MI_Ffmi1)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = MI_df.index.values
    cluster_map['cluster'] = clusterer.labels_
    # print(cluster_map)
    cluster_label_list = clusterer.labels_.tolist()
    cluster_center_list = clusterer.cluster_centers_.tolist()
    # print(cluster_label_list)
    # print(cluster_center_list)
    dict_ccl = {}
    for inx in range(0, len(cluster_center_list)):
        cl_l = []
        flag = 0
        for i in range(0, len(cluster_label_list)):
            if cluster_label_list[i] == inx and flag == 0:
                dict_ccl[cluster_center_list[inx][0]] = cluster_label_list[i]
                flag = 1
    # print("dict_ccl :", dict_ccl)
    inv_ccl = {v: k for k, v in dict_ccl.items()}
    # print("inv_ccl :", inv_ccl)
    if flag1 == 0:
        val = min(inv_ccl, key=inv_ccl.get)
        # print("FFMI")
    else:
        val = max(inv_ccl, key=inv_ccl.get)
        # print("FCMI")
    # df_ccl = pd.DataFrame.from_dict(dict_ccl)
    silhouette_avg = silhouette_score(MI_Ffmi1, clusterer.labels_)
    # print("For n_clusters =", k,
          # "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(MI_Ffmi1, clusterer.labels_)
    ssv = sample_silhouette_values.tolist()
    # print("sample_silhouette_value :", ssv)
    
    return cluster_label_list, cluster_center_list, cluster_map, val
