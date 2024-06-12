#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 01:56:04 2023

@author: dev
"""

from run_iid import run_iid
import pandas as pd
import gc
gc.enable()

dataset_list = [
    ['ionosphere', 33, 0.15, 0.76, 0.91, 0.97, 1],
    ['wdbc', 31, 0.23, 0.39, 0.52, 0.52, 1],
    ['hillvalley', 100, 0.25, 0.10, 0.90, 0.05, 1],
    ['vehicle', 8, 0.87, 0.87, 0.88,  0.75, 1],
    ['segmentation', 9, 0.78, 0.89, 0.89, 0.89, 1],
    ['nsl', 38, 0.84, 0.63, 0.90, 0.78, 1]
]

datasets = pd.DataFrame(dataset_list, columns=[
                        'dataset',
                        'total',
                        'single',
                        'multi',
                        'anova',
                        'rfe',
                        'nofs'
                        ])

lftr = []
df_list = []
obj_list = [
    'nofs',
    'single',
    'multi',
    'anova',
    'rfe',
    # 'fshfl'
]
classifier = 'ff'
ff_list = []


def main(dataset, num_ftr):

    global obj
    global classifier
    global ff_list

    print('Dataset: ', dataset)

    name, acc, f1, max_depth, total_leaves = run_iid(5,
                                                     2,
                                                     2,
                                                     dataset,
                                                     num_ftr,
                                                     obj,
                                                     'ff',
                                                     max_depth=201)

    return name, acc, f1, max_depth, total_leaves


if __name__ == "__main__":

    with open('model_size_vs_accuracy_results.csv', 'w') as f:
        f.write('Dataset, Objective, Num_ftr, Max_depth, Total Leaves, Accuracy')

    f = open('model_size_vs_accuracy_results.csv', 'a')
    for _, dset in datasets.iterrows():

        for obj in obj_list:
            try:
                lftr = []

                name, acc, f1, max_depth, total_leaves = main(
                    dset['dataset'],
                    int(dset['total']*dset[obj])
                )
                out = [
                    dset['dataset'],
                    obj,
                    int(dset['total']*dset[obj]),
                    max_depth,
                    total_leaves,
                    acc
                ]
                f.write('\n' + ",".join([str(i) for i in out]))
            except:
                f.flush()

    f.close()
