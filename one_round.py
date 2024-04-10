#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from run_iid import run_iid
from run_noniid import run_noniid
import os
import pandas as pd
from helpers import get_rng
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
    ["vehicle", 2, 8, 1],
    ["segmentation", 2, 9, 1],
    # ["nsl", 5, 38, 5],
    ["isolet", 80, 617, 80],
    # ["ac", 5, 29, 5],
    ["iot", 5, 28, 4],
    ["synthetic", 20, 200, 20],
    # ["vowel", 2, 12, 1], # Cannot have a large number of clients due to small dataset size
    # ["ionosphere", 5, 33, 1],
    # ["wdbc", 5, 31, 1],
    # ["wine", 5, 13, 1], # Cannot have a large number of clients due to small dataset size
    # ["hillvalley", 5, 100, 5],
    # ["diabetes", 2, 8, 1],
    # ["automobile", 5, 19, 1]
]
DATASETS = pd.DataFrame(dataset_list, columns=["dataset", "lb", "ub", "step"])


OBJ_LIST = ["single", "multi", "anova", "rfe"]
IID_RATIOS = [
    # 0.2,
    0.5,
    0.8,
    1.0
]
CLASSIFIER = "ff"  # Either of mlp or ff or randomforest
non_iid = True
# END of OPTIONS


def main(num_ftr: int,
         dset: pd.Series,
         iid_ratio: float = 1.0):
    global non_iid
    global obj
    global CLASSIFIER

    print(num_ftr, dset, iid_ratio)
    dataset = dset["dataset"]

    print("Dataset: ", dataset)

    n_clust_fcmi = 2
    n_clust_ffmi = 2

    if not non_iid:
        n_client = 50

        name, acc, f1 = run_iid(
            n_client, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, dset, obj, CLASSIFIER)
    else:
        name, acc, f1 = run_noniid(
            n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, obj, CLASSIFIER, iid_ratio)

    return name, acc, f1


if __name__ == "__main__":
    # Iterate through each dataset in the dataset list
    for didx, dset in DATASETS.iterrows():

        # Iterate through each feature selection objective
        for obj in OBJ_LIST:

            if not non_iid:
                filename = obj + '_' + dset["dataset"] + "_iid.csv"
                rng = get_rng(dset)

                # Read existing cache file or initialize row count
                if os.path.exists(filename):
                    with open(filename, "r") as f:
                        reader = csv.reader(f)
                        row_count = sum(1 for row in reader) - 1

                    # Iterate through each number of features
                    # Append results file
                    with open(filename, "a", newline="") as f:
                        writer = csv.writer(f)

                        for num_ftr in rng[row_count:]:
                            name, acc, f1 = main(num_ftr, dset)
                            row_data = [
                                num_ftr,
                                acc,
                                f1
                            ]
                            writer.writerow(row_data)
                else:
                    with open(filename, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            ["Number of Features", "Accuracy", "F1-Score"])

                        for num_ftr in rng:
                            name, acc, f1 = main(num_ftr, dset)
                            row_data = [
                                num_ftr,
                                acc,
                                f1
                            ]
                            writer.writerow(row_data)

            else:

                for iid_ratio in IID_RATIOS:

                    filename = obj + '_' + \
                        dset["dataset"] + '_' + str(iid_ratio) + ".csv"
                    rng = get_rng(dset)

                    # Read existing cache file or initialize row count
                    if os.path.exists(filename):
                        with open(filename, "r") as f:
                            reader = csv.reader(f)
                            row_count = sum(1 for row in reader) - 1

                        # Iterate through each number of features
                        # Append results file
                        with open(filename, "a", newline="") as f:
                            writer = csv.writer(f)

                            for num_ftr in rng[row_count:]:
                                name, acc, f1 = main(
                                    num_ftr, dset, iid_ratio)
                                row_data = [
                                    num_ftr,
                                    acc,
                                    f1
                                ]
                                writer.writerow(row_data)
                                f.flush()

                    else:

                        with open(filename, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                ["Number of Features", "Accuracy", "F1-Score"])

                            for num_ftr in rng:
                                name, acc, f1 = main(
                                    num_ftr, dset, iid_ratio)
                                row_data = [
                                    num_ftr,
                                    acc,
                                    f1
                                ]
                                writer.writerow(row_data)
                                f.flush()
