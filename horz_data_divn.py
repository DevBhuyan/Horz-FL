#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from random import randint, seed
from math import ceil
from preprocesing import preprocessing_data
import os
import pickle
from warnings import warn
from tqdm import tqdm
import gc

gc.enable()


MAX_FTRS = {
    "iot": 28,
    "synthetic": 200,
    "isolet": 617,
    'vehicle': 8,
    'segmentation': 9,
    "ac": 29,
    "nsl": 38,
    "vowel": 12,
    "ionosphere": 33,
    "wdbc": 30,
    "wine": 13,
    "hillvalley": 100,
    "diabetes": 8,
    "california": 9,
    "boston": 13
}


NUM_CLASSES = {
    "iot": 18,
    "ac": 2,
    "nsl": 2,
    "synthetic": 25,
    "isolet": 26,
    'vehicle': 3,
    'segmentation': 4,
    "vowel": 6,
    "ionosphere": 2,
    "wdbc": 2,
    "wine": 3,
    "hillvalley": 1,
    "diabetes": 2,
    "california": -1,
    "boston": -1
}


CLIENT_DIST_FOR_NONIID = {
    "iot": 100,
    # "ac": 50,
    # "nsl": 50,
    # "stackoverflow": None,
    "synthetic": 100,
    "isolet": 60,
    'vehicle': 75,
    'segmentation': 100
}


def divide(data_df: pd.DataFrame,
           dataset: str,
           n_client: int = None):

    seed(42)

    if os.path.exists(f"./datasets/ds_cache/{dataset}_{n_client}_iid.pkl"):
        with open(f"./datasets/ds_cache/{dataset}_{n_client}_iid.pkl", "rb") as f:
            warn("Data loaded from cache. Delete ./datasets/ds_cache to run afresh")
            return pickle.load(f)

    df_list = []
    data_df = preprocessing_data(data_df, dataset)
    data_df = data_df.sample(frac=1)
    l = data_df.shape[0]

    prev = 0
    for i in range(n_client):
        nxt = randint(
            ceil((i + 0.9) * l / (n_client)), ceil((i + 1.1) * l / (n_client))
        )
        if i == 0:
            lb = 0
            ub = nxt
        if i == n_client - 1:
            lb = prev
            ub = l
        if i > 0 and i < n_client - 1:
            lb = prev
            ub = nxt
        df = data_df.iloc[lb:ub, :]
        df_list.append(df)
        prev = nxt

    if not os.path.exists('./datasets/ds_cache'):
        os.makedirs('./datasets/ds_cache', exist_ok=True)
    with open(f"./datasets/ds_cache/{dataset}_{n_client}_iid.pkl", "wb") as f:
        pickle.dump(df_list, f)

    return df_list


def divide_noniid(data_df: pd.DataFrame,
                  dataset: str,
                  iid_ratio: float = 0.2):

    n_client = CLIENT_DIST_FOR_NONIID[dataset]

    if iid_ratio == 1.0:
        return divide(data_df, dataset, n_client=n_client)

    if os.path.exists(f"./datasets/ds_cache/{dataset}_{n_client}_{str(iid_ratio)}.pkl"):
        with open(f"./datasets/ds_cache/{dataset}_{n_client}_{str(iid_ratio)}.pkl", "rb") as f:
            warn("Data loaded from cache. Delete ./datasets/ds_cache to run afresh")
            return pickle.load(f)

    data_df = preprocessing_data(data_df, dataset)

    labels = data_df['Class'].unique()
    nunique = len(labels)

    # Splitting the dataframe into separate labels to pop later
    dataframes_dict = {
        label: data_df[data_df['Class'] == label] for label in labels}

    num_labels = min(int(iid_ratio * nunique), nunique)

    l = data_df.shape[0]
    # num_samples is the number of samples allotted to each client
    num_samples = [randint(ceil((0.9) * l / n_client),
                           ceil((1.1) * l / n_client)) for i in range(n_client)]

    client_data = [pd.DataFrame(columns=data_df.columns)
                   for _ in range(n_client)]

    for cli in tqdm(range(n_client), total=n_client):

        # allotted_labels is the list of labels that will be allotted to the client[cli]
        allotted_labels = [labels[(cli+idx) % nunique]
                           for idx in range(num_labels)]

        samples_per_class = num_samples[cli]//len(allotted_labels)

        for allotted_label in allotted_labels:
            client_data[cli] = client_data[cli].append(
                dataframes_dict[allotted_label].iloc[:samples_per_class, :])
            dataframes_dict[allotted_label] = dataframes_dict[allotted_label].drop(
                # TIP: If not using SMOTE oversampling, then uncomment below line
                # dataframes_dict[allotted_label].index[:min(samples_per_class, dataframes_dict[allotted_label].shape[0]-1)])
                dataframes_dict[allotted_label].index[:samples_per_class])

        if cli == n_client-1:
            for df in dataframes_dict.values():
                try:
                    client_data[cli].append(df)
                except Exception() as e:
                    print(e)
                    raise Exception(
                        "Install pandas==1.4.4 or pip install -r requirements.txt")

    # TIP: Code to check veracity of function
    # for idx, client in enumerate(client_data):
        # print(f"Client_{idx} | num_samples: {client.shape[0]} | labels: {client['Class'].value_counts()}")

    if not os.path.exists('./datasets/ds_cache'):
        os.makedirs('./datasets/ds_cache', exist_ok=True)
    with open(f"./datasets/ds_cache/{dataset}_{n_client}_{str(iid_ratio)}.pkl", "wb") as f:
        pickle.dump(client_data, f)

    return client_data


def horz_data_divn(dataset: str,
                   n_client: int = 50,
                   non_iid: bool = False,
                   iid_ratio: float = 0.2) -> list:

    if dataset == "nsl":
        data_df = pd.read_csv("./datasets/NSL-KDD/KDDTrain+.csv")

    elif dataset == "ac":
        data_df = pd.read_csv(
            "./datasets/annonymized-credit-card/creditcard.csv")

    elif dataset == "arcene":
        data_df = pd.read_csv("./datasets/ARCENE.csv")

    elif dataset == "ionosphere":
        data_df = pd.read_csv("./datasets/ionosphere.csv")

    elif dataset == "relathe":
        data_df = pd.read_csv("./datasets/RELATHE.csv")

    elif dataset == "musk":
        data_df = pd.read_csv("./datasets/musk_csv.csv")

    elif dataset == "TOX-171":
        data_df = pd.read_csv("./datasets/TOX-171.csv")

    elif dataset == "wdbc":
        data_df = pd.read_csv("./datasets/WDBC/data.csv")

    elif dataset == "vowel":
        data_df = pd.read_csv(
            "./datasets/csv_result-dataset_58_vowel.csv")

    elif dataset == "wine":
        data_df = pd.read_csv("./datasets/wine.csv")

    elif dataset == "isolet":
        data_df = pd.read_csv("./datasets/isolet_csv.csv")

    elif dataset == "hillvalley":
        data_df = pd.read_csv("./datasets/hill-valley_csv.csv")

    elif dataset == "vehicle":
        data_df = pd.read_csv("./datasets/vehicle.csv")

    elif dataset == "segmentation":
        data_df = pd.read_csv("./datasets/segmentation.csv")

    elif dataset == "iot":
        data_df = pd.read_csv("./datasets/iot.csv")

    elif dataset == "diabetes":
        data_df = pd.read_csv("./datasets/diabetes.csv")

    elif dataset == "automobile":
        data_df = pd.read_csv("./datasets/Automobile_data.csv")

    elif dataset == "synthetic":
        data_df = pd.read_csv("./datasets/synthetic.csv")

    elif dataset == "california":
        data_df = pd.read_csv('./datasets/california_housing.csv')

    elif dataset == "boston":
        data_df = pd.read_csv('./datasets/boston_housing.csv')

    if non_iid:
        return divide_noniid(data_df, dataset, iid_ratio)
    else:
        return divide(data_df, dataset, n_client)
