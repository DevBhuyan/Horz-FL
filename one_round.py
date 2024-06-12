#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from run_iid import run_iid
from run_noniid import run_noniid
import os
import pandas as pd
from helpers import get_rng
import csv
import gc
from datetime import datetime, timedelta
from sys import exit
from art import text2art

gc.enable()

"""
This code will run the main pipeline for selected FS methods across selected datasets, averaging the results over `runs` number of runs. Comment or uncomment lines from the OPTIONS to choose your objectives
"""

# OPTIONS: ----------------------------------------------
dataset_list = [
    ["vehicle", 2, 8, 1],     # ALL DONE
    ["segmentation", 2, 9, 1],    # ALL DONE
    ["isolet", 80, 617, 80],  # rfe takes at least 1 hr on single core
    ["iot", 6, 21, 5],
    ["synthetic", 40, 160, 40],   # at least 1h41m per main call
    ["nsl", 5, 38, 5],    # two classes only
    ["vowel", 2, 12, 1],
    ["ionosphere", 5, 33, 1],
    ["wdbc", 5, 31, 1],
    ["wine", 5, 13, 1],  # Cannot have a large number of clients due to small dataset size
    ["hillvalley", 5, 100, 5],
    ["diabetes", 2, 8, 1],
    ["ac", 5, 29, 5],
    ["california", 2, 9, 1],
    ["boston", 2, 13, 1]
]
DATASETS = pd.DataFrame(dataset_list, columns=["dataset", "lb", "ub", "step"])


OBJ_LIST = [
    "single",
    "multi",
    # "anova",
    # "rfe",
    'mrmr'
]
IID_RATIOS = [
    0.2,
    0.5,
    0.8,
    1.0
]
CLASSIFIER = "ff"  # Either of mlp or ff or randomforest
non_iid = False
# END of OPTIONS


# TIP: To run time-profiling on this script, run
# python -m cProfile -o profile_data.prof one_round.py && snakeviz profile_data.prof
# OPTIONAL: pip install snakeviz


print(text2art("Fed-MoFS"))


def main(num_ftr: int,
         dset: pd.Series,
         iid_ratio: float = 1.0):
    global non_iid
    global obj
    global CLASSIFIER

    dataset = dset["dataset"]

    print(
        f"\n\n{num_ftr} features || {iid_ratio*100}% IID || {dataset} Dataset || {obj} FS")

    n_clust_fcmi = 2
    n_clust_ffmi = 2

    if not non_iid:
        n_client = 5 if dataset not in ["california", "boston"] else 100

        name, acc, f1 = run_iid(
            n_client, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, obj, CLASSIFIER)
    else:
        name, acc, f1 = run_noniid(
            n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, obj, CLASSIFIER, iid_ratio)

    return name, acc, f1


if __name__ == "__main__":

    start = datetime.now()

    # Iterate through each dataset in the dataset list
    for didx, dset in DATASETS.iterrows():

        # Iterate through each feature selection objective
        for obj in OBJ_LIST:

            if not non_iid:
                filename = f'{obj}_{dset["dataset"]}_iid.csv'
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
                            # TODO: Add a condition here to check if time elapsed since start is more than 11 hours, if yes, close all open files and stop execution
                            if datetime.now() - start >= timedelta(hours=11):
                                print("Time limit exceeded. Stopping execution")
                                exit(1)
                            name, acc, f1 = main(num_ftr, dset)
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
                        if dset["dataset"] in ["california", "boston"]:
                            writer.writerow([
                                "Number of Features",
                                "RMSE",
                                "R2"
                            ])
                        else:
                            writer.writerow([
                                "Number of Features",
                                "Accuracy",
                                "F1-Score"
                            ])
                        f.flush()

                        for num_ftr in rng:
                            # TODO: Add a condition here to check if time elapsed since start is more than 11 hours, if yes, close all open files and stop execution
                            if datetime.now() - start >= timedelta(hours=11):
                                print("Time limit exceeded. Stopping execution")
                                exit(1)
                            name, acc, f1 = main(num_ftr, dset)
                            row_data = [
                                num_ftr,
                                acc,
                                f1
                            ]
                            writer.writerow(row_data)
                            f.flush()

            else:

                for iid_ratio in IID_RATIOS:

                    filename = f'{obj}_{dset["dataset"]}_{str(iid_ratio)}.csv'
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
                                # TODO: Add a condition here to check if time elapsed since start is more than 11 hours, if yes, close all open files and stop execution
                                if datetime.now() - start >= timedelta(hours=11):
                                    print("Time limit exceeded. Stopping execution")
                                    exit(1)
                                try:
                                    name, acc, f1 = main(
                                        num_ftr, dset, iid_ratio)
                                    row_data = [
                                        num_ftr,
                                        acc,
                                        f1
                                    ]
                                    writer.writerow(row_data)
                                    f.flush()
                                except ZeroDivisionError:
                                    print(
                                        "Too few classes for non-iid division. Try at least 80 % iid_ratio")

                    else:

                        with open(filename, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                ["Number of Features",
                                 "Accuracy",
                                 "F1-Score"])
                            f.flush()

                            for num_ftr in rng:
                                # TODO: Add a condition here to check if time elapsed since start is more than 11 hours, if yes, close all open files and stop execution
                                if datetime.now() - start >= timedelta(hours=11):
                                    print("Time limit exceeded. Stopping execution")
                                    exit(1)
                                try:
                                    name, acc, f1 = main(
                                        num_ftr, dset, iid_ratio)
                                    row_data = [
                                        num_ftr,
                                        acc,
                                        f1
                                    ]
                                    writer.writerow(row_data)
                                    f.flush()
                                except ZeroDivisionError:
                                    print(
                                        "Too few classes for non-iid division. Try at least 80 % iid_ratio")
