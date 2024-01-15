from Cluster_kmeans import Cluster_kmeans
from calc_MI import calc_MI
import pandas as pd
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)


def local_fs(data_df: pd.DataFrame, n_clust_fcmi=2, n_clust_ffmi=2):
    """Perform local feature selection using clustering based on Fcmi and FFmi
    metrics.

    Parameters
    ----------
    data_df : pd.DataFrame
        Input DataFrame containing features and a target class column.
    n_clust_fcmi : int, optional
        Number of clusters for Fcmi-based clustering. Default is 2.
    n_clust_ffmi : int, optional
        Number of clusters for FFmi-based clustering. Default is 2.

    Returns
    -------
    local_feature : list
        List of locally selected features along with Fcmi and FFmi values.
    data_df : pd.DataFrame
        DataFrame with selected features.
    """
    # Calculate Mutual Information metrics
    MI_Fcmi = []
    MI_Ffmi = []
    avg_Ffmi = []
    mi = calc_MI(data_df)
    Fcmi = mi.iloc[:, -1:]
    ffmi = mi.iloc[:, :-1]
    ffmi = ffmi.iloc[:-1, :]
    cols = list(mi.columns)
    cols.pop()  # contains all features except class

    # Compute average FFmi for each feature
    for col in cols:
        avg_Ffmi.append(ffmi[col].mean())  # average ffmi

    list_fcmi = Fcmi["Class"].values.tolist()
    list_fcmi.pop()  # deleting the last value as it always comes zero
    MI = {"Fcmi": list_fcmi, "FFmi": avg_Ffmi}
    MI_df = pd.DataFrame(MI)

    # Prepare data for clustering
    for i in range(0, len(list_fcmi)):
        MI_Fcmi.append([list_fcmi[i], 0])  # preparing fcmi values for 2D clustering
    for i in range(0, len(avg_Ffmi)):
        MI_Ffmi.append([avg_Ffmi[i], 0])  # preparing ffmi values for 2D clustering

    print("FCMI:", Fcmi)
    print("Avg FFMI:", avg_Ffmi)

    # Compute clusters based on minimum ffmi/redundancy
    flag1 = 0
    cl_labels, cl_center, cluster_map, val = Cluster_kmeans(
        MI_Ffmi, MI_df, n_clust_ffmi, flag1
    )
    cluster_map1 = pd.merge(MI_df, cluster_map, right_index=True, left_index=True)
    cluster_map2 = cluster_map1.loc[cluster_map1["cluster"] == val]
    cluster_map2 = cluster_map2[(cluster_map2.T != 0).any()]
    print(" no_of feature after ffmi", cluster_map2.shape[0])

    # Compute clusters based on maximum fcmi/relevancy
    flag1 = 1
    cl_labels, cl_center, cluster_map, val = Cluster_kmeans(
        MI_Fcmi, MI_df, n_clust_fcmi, flag1
    )
    cluster_map3 = pd.merge(MI_df, cluster_map, right_index=True, left_index=True)
    cluster_map4 = cluster_map3.loc[cluster_map3["cluster"] == val]
    cluster_map4.loc[~(cluster_map4 == 0).all(axis=1)]

    print(" no_of feature after fcmi", cluster_map4.shape[0])
    data_df_col = list(data_df.columns)
    cluster_map5 = pd.concat([cluster_map2, cluster_map4])
    cluster_map5 = cluster_map5.drop_duplicates()
    col = []
    col1 = cluster_map5["data_index"].tolist()
    for i in range(0, len(col1)):
        col_inx = col1[i]
        col.append(data_df_col[col_inx])  # adding locally selected features
    cluster_map5["features"] = col
    local_feature = cluster_map5.reset_index()[
        ["features", "Fcmi", "FFmi"]
    ].values.tolist()
    print(local_feature)
    col.append(data_df_col[-1])  # last column of data_df
    data_df = data_df[col]
    print(data_df.head())

    return local_feature, data_df


def full_spec_fs(data_df, n_clust_fcmi, n_clust_ffmi):
    """Force select all features during local clustering."""
    avg_Ffmi = []
    mi = calc_MI(data_df)
    Fcmi = mi.iloc[:, -1:]
    ffmi = mi.iloc[:, :-1]
    ffmi = ffmi.iloc[:-1, :]
    cols = list(mi.columns)
    cols.pop()  # contains all features except class
    for col in cols:
        avg_Ffmi.append(ffmi[col].mean())  # average ffmi

    # While averaging we were considering the ffmi with the class also, in the last line, that last line has now been removed from the ffmi table
    list_fcmi = Fcmi["Class"].values.tolist()
    list_fcmi.pop()  # deleting the last value as it always comes zero

    out = []

    for i, ftr in enumerate(cols):
        out.append([ftr, list_fcmi[i], avg_Ffmi[i]])

    return out, data_df


def fcmi_and_affmi(data_df):
    """To inspect.

    Returns the list of FCMI and aFFMI scores for data inspection.
    """
    MI_Fcmi = []
    MI_Ffmi = []
    avg_Ffmi = []
    mi = calc_MI(data_df)
    Fcmi = mi.iloc[:, -1:]
    ffmi = mi.iloc[:, :-1]
    cols = list(mi.columns)
    cols.pop()  # contains all features except class
    for col in cols:
        avg_Ffmi.append(ffmi[col].mean())  # average ffmi
    list_fcmi = Fcmi["Class"].values.tolist()
    list_fcmi.pop()  # deleting the last value as it always comes zero
    for i in range(0, len(list_fcmi)):
        MI_Fcmi.append([list_fcmi[i], 0])  # preparing fcmi values for 2D clustering
    for i in range(0, len(avg_Ffmi)):
        MI_Ffmi.append([avg_Ffmi[i], 0])  # preparing ffmi values for 2D clustering
    print("PRINTING MI")
    print(mi)
    return list_fcmi, avg_Ffmi
