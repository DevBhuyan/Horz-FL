#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from random import randint
from math import ceil
from preprocesing import preprocessing_data
import os
import gc
gc.enable()


def divide(data_df : pd.DataFrame, 
           dataset : str, 
           n_client=5):
    '''
    This function takes in a raw dataframe (full dataset), processes the data, and divides it into `n_client` number of horizontally divided dataframes with a random number of samples

    Parameters
    ----------
    data_df : pd.DataFrame
        Raw DataFrame.
    dataset : str
        code name of the dataset in this framework. See 'datasets' table in main().
    n_client : int, optional
        number of clients. The default is 5.

    Returns
    -------
    df_list : list
        list of horizontally divided dataframes.

    '''
    
    df_list = []
    data_df = data_df.sample(frac = 1)
    data_df = preprocessing_data(data_df, dataset)
    data_df = data_df.sample(frac = 1)
    data_df = data_df.astype(float)
    l = data_df.shape[0]

    prev = 0
    for i in range(n_client):
        nxt = randint(ceil((i+0.9)*l/(n_client)), ceil((i+1.1)*l/(n_client)))
        if i == 0:
            lb = 0
            ub = nxt
        if i == n_client-1:
            lb = prev
            ub = l
        if i > 0 and i < n_client-1:
            lb = prev
            ub = nxt
        df = data_df.iloc[lb:ub, :]
        df_list.append(df)
        print('number of samples at cli '+str(i)+' is '+str(ub-lb))
        prev = nxt

    return df_list


def horz_data_divn(dataset : str, 
                   n_client=5):
    '''
    Reads a dataset from storage and returns the horizontally divided dataframes

    Parameters
    ----------
    dataset : str
        code name of dataset.
    n_client : int, optional
        number of clients. The default is 5.

    Returns
    -------
    df_list : list
        list of dataframes.

    '''
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

    elif dataset == 'vehicle':
        data_df = pd.read_csv(curr_dir + "/datasets/vehicle.csv")
        df_list = divide(data_df, dataset, n_client)

    elif dataset == 'segmentation':
        data_df = pd.read_csv(curr_dir + "/datasets/segmentation.csv")
        df_list = divide(data_df, dataset, n_client)

    elif dataset == 'iot':
        data_df = pd.read_csv(curr_dir + "/datasets/iot.csv")
        df_list = divide(data_df, dataset, n_client)

    elif dataset == 'diabetes':
        data_df = pd.read_csv(curr_dir + "/datasets/diabetes.csv")
        df_list = divide(data_df, dataset, n_client)

    elif dataset == 'automobile':
        data_df = pd.read_csv(curr_dir + "/datasets/Automobile_data.csv")
        df_list = divide(data_df, dataset, n_client)

    return df_list
