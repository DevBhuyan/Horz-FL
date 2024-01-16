#!/usr/bin/env python
# -*- coding: utf-8 -*-

from local_feature_select import local_fs, full_spec_fs
from global_feature_select import global_feature_select, global_feature_select_single
from horz_data_divn import horz_data_divn
from ff import ff
import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import gc

gc.enable()
"""Experimental.

This code aims to find the optimal size of trees for which the pipeline
returns maximum accuracy for selected FS methods across selected
datasets. Comment or uncomment lines from the OPTIONS to choose your
objectives
"""

# OPTIONS ----------------------------------------------------------
dataset_list = [
    ["ionosphere", 33, 0.15, 0.76, 0.91, 0.97, 1],
    ["wdbc", 31, 0.23, 0.39, 0.52, 0.52, 1],
    ["hillvalley", 100, 0.25, 0.10, 0.90, 0.05, 1],
    ["vehicle", 8, 0.87, 0.87, 0.88, 0.75, 1],
    ["segmentation", 9, 0.78, 0.89, 0.89, 0.89, 1],
    ["nsl", 38, 0.84, 0.63, 0.90, 0.78, 1],
]

datasets = pd.DataFrame(
    dataset_list,
    columns=["dataset", "total", "single", "multi", "anova", "rfe", "nofs"],
)

lftr = []
df_list = []
obj_list = [
    "nofs",
    "single",
    "multi",
    "anova",
    "rfe",
]

# END of OPTIONS
classifier = "ff"
ff_list = []


def run_iid(dataset: str, num_ftr: int, max_depth=200):
    """Function to compute max_depth of federated-forest trees vs. accuracy.

    Parameters
    ----------
    dataset : str
    num_ftr : int
    max_depth : TYPE, optional

    Returns
    -------
    str
        type of classifier.
    ff_acc : float
        accuracy.
    ff_prec : float
        precision.
    ff_rec : float
        recall.
    max_depth : int
    total_leaves : int
    """

    global lftr
    global df_list
    global obj
    global classifier
    local_feature = []

    if len(lftr) == 0 and obj in ["single", "multi"]:
        df_list = horz_data_divn(dataset, 5)
        for cli in range(0, 5):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            if dataset == "vowel" or dataset == "vehicle":
                local, data_df = full_spec_fs(data_dfx, 2, 2)
            else:
                local, data_df = local_fs(data_dfx, 2, 2)
            local_feature.append(local)
        lftr = local_feature

    if obj == "single":
        print("SINGLE-OBJECTIVE GLOBAL FTR SELECTION....")
        feature_list, num_avbl_ftrs = global_feature_select_single(lftr, num_ftr)
        print("feature list: ", feature_list)
        print("number of features: ", len(feature_list))

        dataframes_to_send = []
        for cli in range(0, 5):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            df1 = data_dfx.iloc[:, -1]
            data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
            data_dfx = data_dfx.assign(Class=df1)
            dataframes_to_send.append(data_dfx)

        if classifier == "ff":
            ff_acc, ff_prec, ff_rec, max_depth, total_leaves = ff(
                dataframes_to_send, max_depth
            )
            print(f"ff_acc: {ff_acc}, ff_prec: {ff_prec}, ff_rec: {ff_rec}")
            return "ff", ff_acc, ff_prec, ff_rec, max_depth, total_leaves

    elif obj == "multi":
        print("MULTI-OBJECTIVE GLOBAL FTR SELECTION....")
        feature_list, num_avbl_ftrs = global_feature_select(lftr, num_ftr)
        print("feature list: ", feature_list)
        print("number of features: ", len(feature_list))

        dataframes_to_send = []
        for cli in range(0, 5):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            df1 = data_dfx.iloc[:, -1]
            data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
            data_dfx = data_dfx.assign(Class=df1)
            dataframes_to_send.append(data_dfx)
        if classifier == "ff":
            ff_acc, ff_prec, ff_rec, max_depth, total_leaves = ff(
                dataframes_to_send, max_depth
            )
            print(f"ff_acc: {ff_acc}, ff_prec: {ff_prec}, ff_rec: {ff_rec}")
            return "ff", ff_acc, ff_prec, ff_rec, max_depth, total_leaves

    elif obj == "anova":
        print("ANOVA....")
        f_selector = SelectKBest(score_func=f_classif, k=num_ftr)

        df_list = horz_data_divn(dset["dataset"], 5)

        dataframes_to_send = []
        for df in df_list:
            df = df.reset_index(drop=True)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            f_selector.fit_transform(X, y)
            feature_indices = f_selector.get_support(indices=True)
            selected_feature_names = X.columns[feature_indices]
            df = pd.DataFrame(X)
            df = df[df.columns.intersection(selected_feature_names)]
            df = df.assign(Class=y)
            dataframes_to_send.append(df)

        if classifier == "ff":
            ff_acc, ff_prec, ff_rec, max_depth, total_leaves = ff(
                dataframes_to_send, max_depth
            )
            print(f"ff_acc: {ff_acc}, ff_prec: {ff_prec}, ff_rec: {ff_rec}")
            return "ff", ff_acc, ff_prec, ff_rec, max_depth, total_leaves

    elif obj == "rfe":
        print("RFE....")
        estimator = RandomForestClassifier()
        rfe = RFE(estimator, n_features_to_select=num_ftr)

        df_list = horz_data_divn(dset["dataset"], 5)
        dataframes_to_send = []
        for df in df_list:
            df = df.reset_index(drop=True)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            X = rfe.fit_transform(X, y)
            df = pd.DataFrame(X)
            df = df.assign(Class=y)
            dataframes_to_send.append(df)

        if classifier == "ff":
            ff_acc, ff_prec, ff_rec, max_depth, total_leaves = ff(
                dataframes_to_send, max_depth
            )
            print(f"ff_acc: {ff_acc}, ff_prec: {ff_prec}, ff_rec: {ff_rec}")
            return "ff", ff_acc, ff_prec, ff_rec, max_depth, total_leaves

    else:
        print("NoFS....")
        dataframes_to_send = horz_data_divn(dset["dataset"], 5)

        if classifier == "ff":
            ff_acc, ff_prec, ff_rec, max_depth, total_leaves = ff(
                dataframes_to_send, max_depth
            )
            print(f"ff_acc: {ff_acc}, ff_prec: {ff_prec}, ff_rec: {ff_rec}")
            return "ff", ff_acc, ff_prec, ff_rec, max_depth, total_leaves


def main(dataset, num_ftr):
    global obj
    global classifier
    global ff_list

    print("Dataset: ", dataset)

    name, acc, prec, rec, max_depth, total_leaves = run_iid(dataset, num_ftr)

    return name, acc, prec, rec, max_depth, total_leaves


if __name__ == "__main__":
    with open("model_size_vs_accuracy_results.csv", "w") as f:
        f.write("Dataset, Objective, Num_ftr, Max_depth, Total Leaves, Accuracy")

    f = open("model_size_vs_accuracy_results.csv", "a")
    for _, dset in datasets.iterrows():
        for obj in obj_list:
            lftr = []

            name, acc, prec, rec, max_depth, total_leaves = main(
                dset["dataset"], int(dset["total"] * dset[obj])
            )
            out = [
                dset["dataset"],
                obj,
                int(dset["total"] * dset[obj]),
                max_depth,
                total_leaves,
                acc,
            ]
            f.write("\n" + ",".join([str(i) for i in out]))

    f.close()
