import pandas as pd
from package1 import NSGA_2


def feature_modeling(feature_list):
    flat_list = []
    for sublist in feature_list:
        for item in sublist:
            flat_list.append(item)
    # print(flat_list)
    return flat_list


def global_feature_select(feature_list):
    flat_list = feature_modeling(feature_list)
    df = pd.DataFrame(flat_list, columns=['features', 'FCMI', 'FFMI'])
    df = df.groupby('features').mean().reset_index()
    print("global feature list", df)
    # df_mean = df['FFMI'].mean()
    # df['FCMI'] = df['FCMI'] - df_mean
    # df.sort_values(by=['FCMI'], inplace=True, ascending=False)
    # print(df)
    df = NSGA_2.nsga_2(df)
    list1 = df['features'].values.tolist()
    print(list1)
    # print(df1)
    # print(df_mean)
    # list1 = df.features.unique()

    return list1


def global_feature_select_single(feature_list):
    flat_list = feature_modeling(feature_list)
    df = pd.DataFrame(flat_list, columns=['features', 'FCMI', 'FFMI'])
    df = df.groupby('features').mean().reset_index()
    df_mean = df['FFMI'].mean()
    df['FCMI'] = df['FCMI'] - (df['FFMI'] / (len(df)))
    df.sort_values(by=['FCMI'], inplace=True, ascending=False)
    print(df)

    list1 = df['features'].values.tolist()
    k1 = len(list1)

    print("k1 = ", k1)
    list1 = list1[:10]
    print(list1)
    # print(df1)
    print(df_mean)
    # list1 = df.features.unique()

    return list1
