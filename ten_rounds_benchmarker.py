#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from horz_data_divn import horz_data_divn
from ff import ff
from Fed_MLP import Fed_MLP
import pandas as pd
import numpy as np
from datetime import datetime
from randomforest import rf
import csv
import gc

gc.enable()

"""
This code will run the main pipeline for the No-FS method across selected datasets, averaging the results over `runs` number of runs. Comment or uncomment lines from the OPTIONS to choose your objectives.

Code flow follows `ten_rounds.py` except that there is only one objective in this case.
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
    ["ac", 5, 30, 1],
    ["nsl", 5, 41, 1],
    ["isolet", 80, 617, 80],
    ["TOX-171", 500, 5748, 500],
    ["iot", 5, 28, 1],
    ["diabetes", 2, 8, 1],
    ["automobile", 5, 19, 1],
]
datasets = pd.DataFrame(dataset_list, columns=["dataset", "lb", "ub", "step"])

lftr = []
df_list = []
obj = "No-FS"
classifier = "ff"
ff_list = []
mlp_list = []
runs = 10  # This variable will decide the number of times you wish to run the pipeline. The mean and standard deviation of the code will be automatically computed and the end outputs will contain the final computed values.

# END of OPTIONS


def run_iid(n_client, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, dset):
    global lftr
    global df_list
    global obj
    global classifier

    dataframes_to_send = horz_data_divn(dataset, n_client)

    for dframe in dataframes_to_send:
        print(np.unique(dframe["Class"]))

    if classifier == "ff":
        ff_acc, ff_prec, ff_rec = ff(dataframes_to_send)
        print(f"ff_acc: {ff_acc}, ff_prec: {ff_prec}, ff_rec: {ff_rec}")
        return "ff", ff_acc, ff_prec, ff_rec, a, p, r, ftrs_returned_by_lfs
    elif classifier == "mlp":
        MLP_acc, MLP_prec, MLP_rec = Fed_MLP(dataframes_to_send)
        print(f"MLP_acc: {MLP_acc}, MLP_prec: {MLP_prec}, MLP_rec: {MLP_rec}")
        return "mlp", MLP_acc, MLP_prec, MLP_rec, a, p, r, ftrs_returned_by_lfs
    elif classifier == "randomforest":
        rf_acc, rf_prec, rf_rec = rf(dataframes_to_send)
        print(f"rf_acc: {rf_acc}, rf_prec: {rf_prec}, rf_rec: {rf_rec}")
        return "randomforest", rf_acc, rf_prec, rf_rec, a, p, r, ftrs_returned_by_lfs


def main(dataset, num_ftr, dset, run):
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

    name, acc, prec, rec, a, p, r, ftrs_returned_by_lfs = run_iid(
        n_client, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, dset
    )

    return name, acc, prec, rec, a, p, r, ftrs_returned_by_lfs


if __name__ == "__main__":
    for _, dset in datasets.iterrows():  # for each dataset
        all_acc = []
        all_prec = []
        all_rec = []
        lfs_accuracies = []

        ds = []
        rng = []
        rng.append(dset["ub"])
        run_accuracies = []
        run_precs = []
        run_recs = []

        try:
            with open(obj + dset["dataset"] + "cache.csv", "r") as f:
                reader = csv.reader(f)
                row_count = sum(1 for row in reader)
        except:
            row_count = 0

        for run in range((row_count) // len(rng), runs):  # for each run
            lftr = []
            num_ftr_accuracies = []
            num_ftr_precs = []
            num_ftr_recs = []

            for num_ftr in rng:  # for each number of features
                name, acc, prec, rec, a, p, r, ftrs_returned_by_lfs = main(
                    dset["dataset"], num_ftr, dset, run
                )
                num_ftr_accuracies.append(acc)
                num_ftr_precs.append(prec)
                num_ftr_recs.append(rec)

                if a:
                    lfs_accuracies.append(
                        [dset["dataset"], a, p, r, ftrs_returned_by_lfs - 1]
                    )

            run_accuracies.append([num_ftr_accuracies])
            run_precs.append([num_ftr_precs])
            run_recs.append([num_ftr_recs])

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
