from Cluster_kmeans import Cluster_kmeans
from calc_MI import calc_MI
import pandas as pd
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def local_fs(data_df, n_clust_fcmi, n_clust_ffmi):
    MI_Fcmi = []
    MI_Ffmi = []
    avg_Ffmi = []
    mi = calc_MI(data_df)
    Fcmi = mi.iloc[:, -1:]
    ffmi = mi.iloc[:, :-1]
    ffmi = ffmi.iloc[:-1, :]
    cols = list(mi.columns)
    cols.pop()  # contains all features except class
    for col in cols:
        avg_Ffmi.append(ffmi[col].mean())  # average ffmi
    # print('Average FFMI: ', avg_Ffmi)
    
    # While averaging do we were considering the ffmi with the class also, in the last line, that last line has now been removed from the ffmi table
    list_fcmi = Fcmi['Class'].values.tolist()
    list_fcmi.pop()  # deleting the last value as it always comes zero
    # print('List_FCMI: ', list_fcmi)
    MI = {'Fcmi': list_fcmi,
          'FFmi': avg_Ffmi}
    MI_df = pd.DataFrame(MI)
    for i in range(0, len(list_fcmi)):
        MI_Fcmi.append([list_fcmi[i], 0])  # preparing fcmi values for 2D clustering
    for i in range(0, len(avg_Ffmi)):
        MI_Ffmi.append([avg_Ffmi[i], 0])  # preparing ffmi values for 2D clustering

    

    # print("FCMI:", Fcmi)
    # print("Avg FFMI:", avg_Ffmi)

    # compute clusters based on minimum ffmi/redundancy
    flag1 = 0
    cl_labels, cl_center, cluster_map, val = Cluster_kmeans(MI_Ffmi, MI_df, n_clust_ffmi, flag1)
    # print("Cluster map :", cluster_map)
    cluster_map1 = pd.merge(MI_df, cluster_map, right_index=True, left_index=True)
    # print("cluster_map1", cluster_map1)
    cluster_map2 = cluster_map1.loc[cluster_map1['cluster'] == val]
    cluster_map2 = cluster_map2[(cluster_map2.T != 0).any()]
    # print(" no_of feature after ffmi", cluster_map2.shape[0])
    # print("cluster_map2", cluster_map2)

    # Fcmi_list = cluster_map2['Fcmi'].to_list()
    # print("Fcmi_list", Fcmi_list)
    # MI_Fcmi = []
    # for i in range(0, len(Fcmi_list)):
    #     MI_Fcmi.append([Fcmi_list[i], 0])

    # compute clusters based on maximum fcmi/relevancy

    flag1 = 1
    cl_labels, cl_center, cluster_map, val = Cluster_kmeans(MI_Fcmi, MI_df, n_clust_fcmi, flag1)
    # print("Cluster map :", cluster_map)
    cluster_map3 = pd.merge(MI_df, cluster_map, right_index=True, left_index=True)
    # print("cluster_map 3 :", cluster_map3)
    cluster_map4 = cluster_map3.loc[cluster_map3['cluster'] == val]
    cluster_map4.loc[~(cluster_map4 == 0).all(axis=1)]

    # print(" no_of feature after fcmi", cluster_map4.shape[0])
    # print("cluster_map 4 :", cluster_map4)
    data_df_col = list(data_df.columns)
    cluster_map5 = pd.concat([cluster_map2, cluster_map4])
    cluster_map5 = cluster_map5.drop_duplicates()
    # print("cluster_map 5 comb :", cluster_map5)
    col = []
    col1 = cluster_map5["data_index"].tolist()
    # print("col1", col1)
    for i in range(0, len(col1)):
        col_inx = col1[i]
        col.append(data_df_col[col_inx])  # adding locally selected features
    cluster_map5['features'] = col
    # print("cluster_map5 :", cluster_map5)
    local_feature = cluster_map5.reset_index()[['features', 'Fcmi', 'FFmi']].values.tolist()
    # print(local_feature)
    col.append(data_df_col[-1])  # last column of data_df
    # print("col :", col)
    data_df = data_df[col]
    # print(data_df.head())
    
    return local_feature, data_df


def fcmi_and_affmi(data_df):
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
    list_fcmi = Fcmi['Class'].values.tolist()
    list_fcmi.pop()  # deleting the last value as it always comes zero
    for i in range(0, len(list_fcmi)):
        MI_Fcmi.append([list_fcmi[i], 0])  # preparing fcmi values for 2D clustering
    for i in range(0, len(avg_Ffmi)):
        MI_Ffmi.append([avg_Ffmi[i], 0])  # preparing ffmi values for 2D clustering
    # print('PRINTING MI')
    # print(mi)
    return list_fcmi, avg_Ffmi