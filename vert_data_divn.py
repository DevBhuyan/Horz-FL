import pandas as pd
from preprocessing import preprocessing_data
import os
from math import ceil
from random import randint

def divide(data_df, dataset, n_client):
    
    df_list = []
    data_df = preprocessing_data(data_df, dataset)
    data_df = data_df.sample(frac = 1)
    ftrs = len(data_df.columns) - 1
    prev = 0
    print(data_df.columns)
    for i in range(n_client):
        nxt = randint(ceil((i+0.5)*ftrs/(n_client)), ceil((i+1.5)*ftrs/(n_client)))
        if i == 0:
            lb = 0
            ub = nxt
        if i == n_client-1:
            lb = prev
            ub = ftrs
        else:
            lb = prev
            ub = nxt
        df = data_df.iloc[:, lb:ub]
        df = df.assign(Class = data_df['Class'])
        df_list.append(df)
        print(df)
        prev = nxt
        
    return df_list

def vert_data_divn(dataset, n_client):
    
    curr_dir = os.getcwd()
    
    if dataset == 'nsl':
        data_df = pd.read_csv(curr_dir + "/datasets/NSL-KDD/KDDTrain+.csv")
        df_list = divide(data_df, dataset, n_client)
        
    elif dataset == 'ac':
        data_df = pd.read_csv(curr_dir + "/datasets/annonymized-credit-card/creditcard.csv")
        df_list = divide(data_df, dataset, n_client)
        
    elif dataset == 'arcene':
        data_df = pd.read_csv(curr_dir + "/datasets/ARCENE.csv")
        df_list = divide(data_df, dataset, n_client)
            
    elif dataset == 'ionosphere':
        data_df = pd.read_csv(curr_dir + "/datasets/ionosphere.csv")
        df_list = divide(data_df, dataset, n_client)
            
    elif dataset == 'relathe':
        data_df = pd.read_csv(curr_dir + "/datasets/RELATHE.csv")
        df_list = divide(data_df, dataset, n_client)
    
    elif dataset == 'musk':
        data_df = pd.read_csv(curr_dir + "/datasets/musk_csv.csv")
        df_list = divide(data_df, dataset, n_client)
            
    elif dataset == 'TOX-171':
        data_df = pd.read_csv(curr_dir + "/datasets/TOX-171.csv")
        df_list = divide(data_df, dataset, n_client)
            
    elif dataset == 'wdbc':
        data_df = pd.read_csv(curr_dir + "/datasets/WDBC/data.csv")
        df_list = divide(data_df, dataset, n_client)
    
    elif dataset == 'vowel':
        data_df = pd.read_csv(curr_dir + "/datasets/csv_result-dataset_58_vowel.csv")
        df_list = divide(data_df, dataset, n_client)
        
    elif dataset == 'wine':
        data_df = pd.read_csv(curr_dir + "/datasets/wine.csv")
        df_list = divide(data_df, dataset, n_client)
     
    elif dataset == 'isolet':
        data_df = pd.read_csv(curr_dir + "/datasets/isolet_csv.csv")
        df_list = divide(data_df, dataset, n_client)
    
    elif dataset == 'hillvalley':
        data_df = pd.read_csv(curr_dir + "/datasets/hill-valley_csv.csv")
        df_list = divide(data_df, dataset, n_client)
        
    return df_list