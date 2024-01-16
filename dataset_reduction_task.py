#!/usr/bin/env python
# -*- coding: utf-8 -*-

from local_feature_select import local_fs, full_spec_fs
from global_feature_select import global_feature_select, global_feature_select_single
from horz_data_divn import horz_data_divn
import pandas as pd
from normalize import normalize
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from functools import reduce
from preprocesing import preprocessing_data
import os
import gc

gc.enable()
"""This program will use the raw datasets downloaded in the root directory and
write the condensed datasets using various FS methods into the "./condensed"
directory in csv format.

Each dataset is passed through the data_preprocessing function that will
convert all text values to numeric floats, z-score normalize the data,
and convert all Class values to integers starting from 0.

This was an experimental work, hence most of the values are hardcoded.
You may comment out lines inside dataset_list or obj_list according to
your requirements.
"""


dataset_list = [
    ["ionosphere", 33, 0.15, 0.76, 0.91, 0.97, 1],
    ["wdbc", 31, 0.23, 0.39, 0.52, 0.52, 1],
    ["hillvalley", 100, 0.25, 0.10, 0.90, 0.05, 1],
    ["vehicle", 8, 0.87, 0.87, 0.88, 0.75, 1],
    ["segmentation", 9, 0.78, 0.89, 0.89, 0.89, 1],
    ["nsl", 38, 0.84, 0.63, 0.90, 0.78, 1],
    ["vowel", 12, 0.92, 0.83, 0.50, 0.92, 1],
    ["wine", 13, 0.54, 0.77, 0.85, 0.92, 1],
    ["diabetes", 8, 0.75, 0.37, 0.88, 0.75, 1],
]

datasets = pd.DataFrame(
    dataset_list,
    columns=["dataset", "total", "single", "multi", "anova", "rfe", "nofs"],
)

lftr = []
df_list = []
obj_list = [
    "single",  # Fed-FiS
    "multi",  # Fed-MoFS
    "anova",
    "rfe",
    "nofs",  # No-Feature-Selection
]


def run_iid(dataset: str, num_ftr: int):
    """


    Parameters
    ----------
    dataset : str
        name of the dataset.
    num_ftr : int
        Number of features expected in the best performing run of given method (sourced from datasets table).

    Returns
    -------
    feature_list : list
        list of best features returned.

    """

    global lftr
    global df_list
    global obj
    global classifier
    local_feature = []

    # Create a cached local_feature for multiple runs
    if len(lftr) == 0 and obj in ["single", "multi"]:
        dlist = []
        df_list = horz_data_divn(dataset, 5)
        for cli in range(0, 5):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            if dataset == "vowel" or dataset == "vehicle":
                local, data_df = full_spec_fs(data_dfx, 2, 2)
            else:
                local, data_df = local_fs(data_dfx, 2, 2)
            local_feature.append(local)
            dlist.append(data_df)
        dlist = normalize(dlist)
        lftr = local_feature
        max([len(cli) for cli in lftr])

    if obj == "single":
        print("SINGLE-OBJECTIVE GLOBAL FTR SELECTION....")
        feature_list, num_avbl_ftrs = global_feature_select_single(lftr, num_ftr)
        print("feature list: ", feature_list)
        print("number of features: ", len(feature_list))

    elif obj == "multi":
        # Multi-Objective ftr sel
        print("MULTI-OBJECTIVE GLOBAL FTR SELECTION....")
        feature_list, num_avbl_ftrs = global_feature_select(lftr, num_ftr)
        print("feature list: ", feature_list)
        print("number of features: ", len(feature_list))

    elif obj == "anova":
        # ANOVA
        print("ANOVA....")
        f_selector = SelectKBest(score_func=f_classif, k=num_ftr)

        df_list = horz_data_divn(dset["dataset"], 5)

        feature_list = []
        for df in df_list:
            df = df.reset_index(drop=True)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            f_selector.fit_transform(X, y)
            feature_indices = f_selector.get_support(indices=True)
            feature_list.append(list(X.columns[feature_indices]))
        feature_list = list(
            reduce(set.intersection, (set(lst) for lst in feature_list))
        )
        print("feature list: ", feature_list)
        print("number of features: ", len(feature_list))

    elif obj == "rfe":
        # RFE
        print("RFE....")
        estimator = RandomForestClassifier()
        rfe = RFE(estimator, n_features_to_select=num_ftr)

        df_list = horz_data_divn(dset["dataset"], 5)
        feature_list = []
        for df in df_list:
            df = df.reset_index(drop=True)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            rfe.fit_transform(X, y)
            selected_features_indices = rfe.get_support(indices=True)
            feature_list.append(list(X.columns[selected_features_indices]))
        feature_list = list(
            reduce(set.intersection, (set(lst) for lst in feature_list))
        )
        print("feature list: ", feature_list)
        print("number of features: ", len(feature_list))

    elif obj == "nofs":
        df_list = horz_data_divn(dataset, 5)
        feature_list = list(df_list[0].columns)

    return feature_list


def main(dataset, num_ftr=None):
    global obj

    print("dataset name: ", dataset)

    dataframes_to_send = run_iid(dataset, num_ftr)

    return dataframes_to_send


if __name__ == "__main__":
    for _, dset in datasets.iterrows():  # for each dataset
        for obj in obj_list:
            lftr = []

            feature_list = main(dset["dataset"], int(dset["total"] * dset[obj]))

            dataset = dset["dataset"]
            curr_dir = os.getcwd()

            if dataset == "nsl":
                data_df = pd.read_csv(curr_dir + "/datasets/NSL-KDD/KDDTrain+.csv")

            elif dataset == "ac":
                data_df = pd.read_csv(
                    curr_dir + "/datasets/annonymized-credit-card/creditcard.csv"
                )

            elif dataset == "arcene":
                data_df = pd.read_csv(curr_dir + "/datasets/ARCENE.csv")

            elif dataset == "ionosphere":
                data_df = pd.read_csv(curr_dir + "/datasets/ionosphere.csv")

            elif dataset == "relathe":
                data_df = pd.read_csv(curr_dir + "/datasets/RELATHE.csv")

            elif dataset == "musk":
                data_df = pd.read_csv(curr_dir + "/datasets/musk_csv.csv")

            elif dataset == "TOX-171":
                data_df = pd.read_csv(curr_dir + "/datasets/TOX-171.csv")

            elif dataset == "wdbc":
                data_df = pd.read_csv(curr_dir + "/datasets/WDBC/data.csv")

            elif dataset == "vowel":
                data_df = pd.read_csv(
                    curr_dir + "/datasets/csv_result-dataset_58_vowel.csv"
                )

            elif dataset == "wine":
                data_df = pd.read_csv(curr_dir + "/datasets/wine.csv")

            elif dataset == "isolet":
                data_df = pd.read_csv(curr_dir + "/datasets/isolet_csv.csv")

            elif dataset == "hillvalley":
                data_df = pd.read_csv(curr_dir + "/datasets/hill-valley_csv.csv")

            elif dataset == "vehicle":
                data_df = pd.read_csv(curr_dir + "/datasets/vehicle.csv")

            elif dataset == "segmentation":
                data_df = pd.read_csv(curr_dir + "/datasets/segmentation.csv")

            elif dataset == "iot":
                data_df = pd.read_csv(curr_dir + "/datasets/iot.csv")

            elif dataset == "diabetes":
                data_df = pd.read_csv(curr_dir + "/datasets/diabetes.csv")

            elif dataset == "automobile":
                data_df = pd.read_csv(curr_dir + "/datasets/Automobile_data.csv")

            data_df = preprocessing_data(data_df, dataset)
            df = data_df[feature_list]
            y = data_df.iloc[:, -1]
            df = df.assign(Class=y)
            df.to_csv(
                "./datasets/condensed/" + dataset + "_condensed_using_" + obj + ".csv",
                index=False,
            )
