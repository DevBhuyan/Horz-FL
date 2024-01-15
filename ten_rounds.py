#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from local_feature_select import local_fs, full_spec_fs
from global_feature_select import global_feature_select, global_feature_select_single
import os
from horz_data_divn import horz_data_divn
from ff import ff
from Fed_MLP import Fed_MLP
import pandas as pd
from normalize import normalize
import numpy as np
from datetime import datetime
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from randomforest import rf
import csv
import gc
import warnings

warnings.simplefilter("ignore")

gc.enable()

"""
This code will run the main pipeline for selected FS methods across selected datasets, averaging the results over `runs` number of runs. Comment or uncomment lines from the OPTIONS to choose your objectives
"""

# OPTIONS: ----------------------------------------------
dataset_list = [
    ["ionosphere", 5, 33, 1],
    ["wdbc", 5, 31, 1],
    ["vowel", 2, 12, 1],
    ["wine", 5, 13, 1],
    ["hillvalley", 5, 100, 5],
    ["vehicle", 2, 8, 1],
    ["segmentation", 2, 9, 1],
    ["nsl", 5, 38, 5],
    ["isolet", 80, 617, 80],
    ["ac", 5, 29, 5],
    ["TOX-171", 500, 5748, 500],
    ["iot", 5, 28, 4],
    ["diabetes", 2, 8, 1],
    ["automobile", 5, 19, 1],
]
datasets = pd.DataFrame(dataset_list, columns=["dataset", "lb", "ub", "step"])

lftr = []
df_list = []
max_MLP = 0.0
obj_list = ["single", "multi", "anova", "rfe"]
classifier = "mlp"  # Either of mlp or ff or randomforest
ff_list = []
mlp_list = []
runs = 10  # This variable will decide the number of times you wish to run the pipeline. The mean and standard deviation of the code will be automatically computed and the end outputs will contain the final computed values.

# END of OPTIONS


def run_iid(n_client, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, dset):
    """Run the IID (Independent and Identically Distributed) feature selection
    process.

    Parameters
    ----------
    n_client : int
        Number of clients in the federated learning system.
    n_clust_fcmi : int
        Number of clusters for Fcmi feature selection.
    n_clust_ffmi : int
        Number of clusters for FFmi feature selection.
    dataset : str
        Name of the dataset.
    num_ftr : int
        Number of features to be selected.
    dset : dict
        Dictionary containing dataset information.

    Returns
    -------
    result : tuple
        Tuple containing information about the selected feature selection method,
        accuracy, precision, recall, aggregation, precision, recall, and the number
        of features returned by the local feature selection.
    """
    global lftr
    global df_list
    global obj
    global classifier
    local_feature = []

    # The following code stores the local_features into a variable lftr so that they need not be explicitly computed for different num_ftr during a single run
    if len(lftr) == 0 and obj in ["single", "multi"]:
        dlist = []
        df_list = horz_data_divn(dataset, n_client)
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            if dataset == "vowel" or dataset == "vehicle":
                local, data_df = full_spec_fs(data_dfx, n_clust_fcmi, n_clust_ffmi)
            else:
                local, data_df = local_fs(data_dfx, n_clust_fcmi, n_clust_ffmi)
            local_feature.append(local)
            dlist.append(data_df)
        dlist = normalize(dlist)
        lftr = local_feature

    if obj == "single":
        # Single-Objective ftr sel
        print("SINGLE-OBJECTIVE GLOBAL FTR SELECTION....")
        feature_list, num_avbl_ftrs = global_feature_select_single(lftr, num_ftr)
        print("feature list: ", feature_list)
        print("number of features: ", len(feature_list))

        dataframes_to_send = []
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            df1 = data_dfx.iloc[:, -1]
            data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
            data_dfx = data_dfx.assign(Class=df1)
            dataframes_to_send.append(data_dfx)

        if classifier == "ff":
            ff_acc, ff_prec, ff_rec = ff(dataframes_to_send)
            print(f"ff_acc: {ff_acc}, ff_prec: {ff_prec}, ff_rec: {ff_rec}")
            return "ff", ff_acc, ff_prec, ff_rec
        elif classifier == "mlp":
            MLP_acc, MLP_prec, MLP_rec = Fed_MLP(dataframes_to_send)
            print(f"MLP_acc: {MLP_acc}, MLP_prec: {MLP_prec}, MLP_rec: {MLP_rec}")
            return "mlp", MLP_acc, MLP_prec, MLP_rec
        elif classifier == "randomforest":
            rf_acc, rf_prec, rf_rec = rf(dataframes_to_send)
            print(f"rf_acc: {rf_acc}, rf_prec: {rf_prec}, rf_rec: {rf_rec}")
            return (
                "randomforest",
                rf_acc,
                rf_prec,
                rf_rec,
            )

    elif obj == "multi":
        # Multi-Objective ftr sel
        print("MULTI-OBJECTIVE GLOBAL FTR SELECTION....")
        feature_list, num_avbl_ftrs = global_feature_select(lftr, num_ftr)
        print("feature list: ", feature_list)
        print("number of features: ", len(feature_list))

        dataframes_to_send = []
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            df1 = data_dfx.iloc[:, -1]
            data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
            data_dfx = data_dfx.assign(Class=df1)
            dataframes_to_send.append(data_dfx)
        if classifier == "ff":
            ff_acc, ff_prec, ff_rec = ff(dataframes_to_send)
            print(f"ff_acc: {ff_acc}, ff_prec: {ff_prec}, ff_rec: {ff_rec}")
            return "ff", ff_acc, ff_prec, ff_rec
        elif classifier == "mlp":
            MLP_acc, MLP_prec, MLP_rec = Fed_MLP(dataframes_to_send)
            print(f"MLP_acc: {MLP_acc}, MLP_prec: {MLP_prec}, MLP_rec: {MLP_rec}")
            return "mlp", MLP_acc, MLP_prec, MLP_rec
        elif classifier == "randomforest":
            rf_acc, rf_prec, rf_rec = rf(dataframes_to_send)
            print(f"rf_acc: {rf_acc}, rf_prec: {rf_prec}, rf_rec: {rf_rec}")
            return (
                "randomforest",
                rf_acc,
                rf_prec,
                rf_rec,
            )

    elif obj == "anova":
        # ANOVA
        print("ANOVA....")
        f_selector = SelectKBest(score_func=f_classif, k=num_ftr)

        df_list = horz_data_divn(dset["dataset"], n_client)

        new_list = []
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
            new_list.append(df)

        if classifier == "ff":
            ff_acc, ff_prec, ff_rec = ff(new_list)
            print(f"ff_acc: {ff_acc}, ff_prec: {ff_prec}, ff_rec: {ff_rec}")
            return "ff", ff_acc, ff_prec, ff_rec
        elif classifier == "mlp":
            MLP_acc, MLP_prec, MLP_rec = Fed_MLP(new_list)
            print(f"MLP_acc: {MLP_acc}, MLP_prec: {MLP_prec}, MLP_rec: {MLP_rec}")
            return "mlp", MLP_acc, MLP_prec, MLP_rec
        elif classifier == "randomforest":
            rf_acc, rf_prec, rf_rec = rf(new_list)
            print(f"rf_acc: {rf_acc}, rf_prec: {rf_prec}, rf_rec: {rf_rec}")
            return (
                "randomforest",
                rf_acc,
                rf_prec,
                rf_rec,
            )

    elif obj == "rfe":
        # RFE
        print("RFE....")
        estimator = RandomForestClassifier()
        rfe = RFE(estimator, n_features_to_select=num_ftr)

        df_list = horz_data_divn(dset["dataset"], n_client)
        new_list = []
        for df in df_list:
            df = df.reset_index(drop=True)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            X = rfe.fit_transform(X, y)
            df = pd.DataFrame(X)
            df = df.assign(Class=y)
            new_list.append(df)

        if classifier == "ff":
            ff_acc, ff_prec, ff_rec = ff(new_list)
            print(f"ff_acc: {ff_acc}, ff_prec: {ff_prec}, ff_rec: {ff_rec}")
            return "ff", ff_acc, ff_prec, ff_rec
        elif classifier == "mlp":
            MLP_acc, MLP_prec, MLP_rec = Fed_MLP(new_list)
            print(f"MLP_acc: {MLP_acc}, MLP_prec: {MLP_prec}, MLP_rec: {MLP_rec}")
            return "mlp", MLP_acc, MLP_prec, MLP_rec
        elif classifier == "randomforest":
            rf_acc, rf_prec, rf_rec = rf(new_list)
            print(f"rf_acc: {rf_acc}, rf_prec: {rf_prec}, rf_rec: {rf_rec}")
            return (
                "randomforest",
                rf_acc,
                rf_prec,
                rf_rec,
            )


def main(dataset, num_ftr, dset, run):
    """Execute the main feature selection process for a given dataset, number
    of features, dataset information, and run.

    Parameters
    ----------
    dataset : str
        Name of the dataset.
    num_ftr : int
        Number of features to be selected.
    dset : dict
        Dictionary containing dataset information.
    run : int
        Current run iteration.

    Returns
    -------
    result : tuple
        Tuple containing information about the selected feature selection method,
        accuracy, precision, recall, aggregation, precision, recall, and the number
        of features returned by the local feature selection.
    """
    global max_MLP
    global obj
    global classifier
    global ff_list
    global mlp_list

    print("Dataset: ", dataset)

    FCMI_clust_num = "2"
    FFMI_clust_num = "2"
    dataset = dataset
    cli_num = "5"

    curr_dir = os.getcwd()
    print(curr_dir)

    n_clust_fcmi = int(FCMI_clust_num)
    n_clust_ffmi = int(FFMI_clust_num)
    n_client = int(cli_num)

    name, acc, prec, rec = run_iid(
        n_client, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, dset
    )

    return name, acc, prec, rec


if __name__ == "__main__":
    # Iterate through each dataset in the dataset list
    for _, dset in datasets.iterrows():
        all_acc = []
        all_prec = []
        all_rec = []

        # Iterate through each feature selection objective
        for obj in obj_list:
            ds = []
            rng = list(range(dset["lb"], dset["ub"] + 1, dset["step"]))

            if (dset["ub"]) % (dset["step"]) != 0:
                rng.append(dset["ub"])
            run_accuracies = []
            run_precs = []
            run_recs = []

            # Read existing cache file or initialize row count
            try:
                with open(obj + dset["dataset"] + "cache.csv", "r") as f:
                    reader = csv.reader(f)
                    row_count = sum(1 for row in reader)
            except:
                row_count = 0

            # Iterate through each run
            for run in range((row_count) // len(rng), runs):
                lftr = []
                num_ftr_accuracies = []
                num_ftr_precs = []
                num_ftr_recs = []

                # Iterate through each number of features
                for num_ftr in rng:
                    name, acc, prec, rec = main(dset["dataset"], num_ftr, dset, run)
                    num_ftr_accuracies.append(acc)
                    num_ftr_precs.append(prec)
                    num_ftr_recs.append(rec)

                run_accuracies.append([num_ftr_accuracies])
                run_precs.append([num_ftr_precs])
                run_recs.append([num_ftr_recs])

                # Append results to the cache file
                with open(obj + dset["dataset"] + "cache.csv", "a", newline="") as f:
                    writer = csv.writer(f)

                    for i in range(len(num_ftr_accuracies)):
                        row_data = [
                            run,
                            num_ftr_accuracies[i],
                            num_ftr_precs[i],
                            num_ftr_recs[i],
                        ]
                        writer.writerow(row_data)

            # Read and process results from the cache file
            with open(obj + dset["dataset"] + "cache.csv", "r") as f:
                reader = csv.reader(f)
                num_ftr_accuracies = []
                num_ftr_precs = []
                num_ftr_recs = []
                run_accuracies = []
                run_precs = []
                run_recs = []
                prev = "0"
                for row in reader:
                    if row[0] == prev:
                        num_ftr_accuracies.append(float(row[1]))
                        num_ftr_precs.append(float(row[2]))
                        num_ftr_recs.append(float(row[3]))
                    else:
                        run_accuracies.append(num_ftr_accuracies)
                        run_precs.append(num_ftr_precs)
                        run_recs.append(num_ftr_recs)
                        num_ftr_accuracies = []
                        num_ftr_precs = []
                        num_ftr_recs = []
                        num_ftr_accuracies.append(float(row[1]))
                        num_ftr_precs.append(float(row[2]))
                        num_ftr_recs.append(float(row[3]))

                    prev = row[0]
                run_accuracies.append(num_ftr_accuracies)
                run_precs.append(num_ftr_precs)
                run_recs.append(num_ftr_recs)

            # Calculate mean and standard deviation for accuracy, precision, and recall
            run_acc = []
            racc = np.array(run_accuracies)
            run_acc.append(racc.mean(axis=0))
            run_acc.append(racc.std(axis=0))
            run_acc = np.array(run_acc)

            run_prec = []
            rprec = np.array(run_precs)
            run_prec.append(rprec.mean(axis=0))
            run_prec.append(rprec.std(axis=0))
            run_prec = np.array(run_prec)

            run_rec = []
            rrec = np.array(run_recs)
            run_rec.append(rrec.mean(axis=0))
            run_rec.append(rrec.std(axis=0))
            run_rec = np.array(run_rec)

            ds.append(rng)
            ds.append(run_acc[0].tolist())
            ds.append(run_acc[1].tolist())
            ds.append(run_prec[0].tolist())
            ds.append(run_prec[1].tolist())
            ds.append(run_rec[0].tolist())
            ds.append(run_rec[1].tolist())

            dataset_df = pd.DataFrame(ds)
            dataset_df = dataset_df.transpose()
            dataset_df.columns = [
                "num_ftr",
                "accuracy_mean",
                "accuracy_stddev",
                "precision_mean",
                "precision_stddev",
                "recall_mean",
                "recall_stddev",
            ]

            dataset_df.to_csv(
                dset["dataset"]
                + "_"
                + obj
                + "_Fed"
                + classifier
                + "_"
                + str(datetime.now())
                + ".csv"
            )
