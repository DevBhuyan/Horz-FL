#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from run_iid import run_iid
from run_noniid import run_noniid
from horz_data_divn import CLIENT_DIST_FOR_NONIID
import os
import pandas as pd
import numpy as np
from datetime import datetime
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
    ["synthetic", 20, 200, 20]
]
datasets = pd.DataFrame(dataset_list, columns=["dataset", "lb", "ub", "step"])


lftr = []
df_list = []
obj_list = ["single", "multi", "anova", "rfe"]
classifier = "ff"  # Either of mlp or ff or randomforest\
non_iid = False
runs = 1  # This variable will decide the number of times you wish to run the pipeline. The mean and standard deviation of the code will be automatically computed and the end outputs will contain the final computed values.

# END of OPTIONS


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
    global obj
    global classifier

    print("Dataset: ", dataset)

    if not non_iid:

        n_clust_fcmi = 2
        n_clust_ffmi = 2
        n_client = 5

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
                    name, acc, prec, rec = main(
                        dset["dataset"], num_ftr, dset, run)
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
