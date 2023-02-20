import pandas as pd
from NSGA_2 import nsga_2
import matplotlib.pyplot as plt

def feature_modeling(feature_list):
    flat_list = []
    for sublist in feature_list:
        for item in sublist:
            flat_list.append(item)
    # print(flat_list)
    return flat_list

def global_feature_select(dataset, feature_list, num_fronts = 1):
    flat_list = feature_modeling(feature_list)
    df = pd.DataFrame(flat_list, columns=['features', 'FCMI', 'aFFMI'])
    df = df.groupby('features').mean().reset_index()
    print("global feature list", df)
    plt.scatter(df['FCMI'], df['aFFMI'])
    plt.xlabel('FCMI')
    plt.ylabel('aFFMI')
    plt.title('Global Feature list')
    for i in range(len(df)):
        plt.text(df['FCMI'][i], df['aFFMI'][i], df['features'][i])
    plt.show()
    # df_mean = df['FFMI'].mean()
    # df['FCMI'] = df['FCMI'] - df_mean
    # df.sort_values(by=['FCMI'], inplace=True, ascending=False)
    # print(df)
    ftrs_in_fronts = nsga_2(dataset, df, num_fronts = num_fronts)
    lst = []
    if num_fronts > len(ftrs_in_fronts):
        num_fronts = len(ftrs_in_fronts)
    for front in ftrs_in_fronts[:num_fronts]:
        for i in front:
            lst.append(i)
    return lst

def global_feature_select_single(feature_list, num_ftr):
    # Prioritises features with greater FCMI Score
    flat_list = feature_modeling(feature_list)
    df = pd.DataFrame(flat_list, columns=['features', 'FCMI', 'aFFMI'])
    df = df.groupby('features').mean().reset_index()
    # df_mean = df['FFMI'].mean()
    df['FCMI'] = df['FCMI'] - (df['aFFMI'] / (len(df)))  # Significance of this line
    df.sort_values(by=['FCMI'], inplace=True, ascending=False)
    # print(df)
    list1 = df['features'].values.tolist()
    list1 = list1[:num_ftr]

    return list1
