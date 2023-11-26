#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 20:46:00 2023

@author: dev
"""

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
gc.enable()

# FIXME: CHANGE THE INTERVALS WHEN GOING FROM NOFS TO MULTI
# dataset_list = [
#                 # ['ionosphere', 5, 33, 1],
#                 # ['wdbc', 5, 31, 1],
#                 # ['vowel', 2, 12, 1],
#                 # ['wine', 5, 13, 1],
#                 # ['hillvalley', 5, 100, 5],
#                 ['vehicle', 2, 8, 1],
#                 # ['segmentation', 2, 9, 1],
#                 # ['nsl', 5, 38, 5],
#                 # ['isolet', 80, 617, 80],
#                 # ['ac', 5, 29, 5],
#                 # ['TOX-171', 500, 5748, 500],
#                 # ['iot', 5, 28, 4],
#                 # ['diabetes', 2, 8, 1], 
#                 # ['automobile', 5, 19, 1]
#                 ]

dataset_list = [
                ['ionosphere', 16, 20, 1],
                # ['wdbc', 5, 12, 1],
                # ['vowel', 12, 12, 1],
                # ['wine', 13, 13, 1],
                # ['hillvalley', 5, 20, 5],
                ['vehicle', 4, 7, 1],
                ['segmentation', 5, 8, 1],
                # ['nsl', 5, 38, 5],
                # ['isolet', 80, 617, 80],
                # ['ac', 5, 29, 5],
                # ['TOX-171', 500, 5748, 500],
                # ['iot', 5, 28, 4],
                # ['diabetes', 2, 8, 1],
                # ['automobile', 5, 19, 1]
                ]
datasets = pd.DataFrame(dataset_list, columns = ['dataset', 'lb', 'ub', 'step'])

lftr = []
df_list = []
max_MLP = 0.0
obj_list = [
        # 'single',
        'multi',
        # 'anova',
        # 'rfe'
        # 'nofs'
    ]
classifier = 'mlp'
ff_list = []
mlp_list = []
comm_iters = 100

def run_iid(n_client, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, dset, comm_iter):

    global lftr
    global df_list
    global obj
    global classifier
    local_feature = []

    a, p, r, f, ftrs_returned_by_lfs = (0, 0, 0, 0, 0)

    if len(lftr) == 0 and obj in ['single', 'multi']:
        dlist = []
        df_list = horz_data_divn(dataset, n_client)
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            if dataset == 'vowel' or dataset == 'vehicle':
                local, data_df = full_spec_fs(data_dfx, n_clust_fcmi, n_clust_ffmi)
            else:
                local, data_df = local_fs(data_dfx, n_clust_fcmi, n_clust_ffmi)
            local_feature.append(local)
            dlist.append(data_df)
        dlist = normalize(dlist)
        # if classifier == 'mlp':
        #     a, p, r, f = Fed_MLP(dlist)
        lftr = local_feature
        ftrs_returned_by_lfs = max([len(cli) for cli in lftr])



    if obj == 'single':
        # Single-Objective ftr sel
        print('SINGLE-OBJECTIVE GLOBAL FTR SELECTION....')
        feature_list, num_avbl_ftrs = global_feature_select_single(lftr, num_ftr)
        print('feature list: ', feature_list)
        print('number of features: ', len(feature_list))


        dataframes_to_send = []
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            df1 = data_dfx.iloc[:, -1]
            data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
            data_dfx = data_dfx.assign(Class = df1)
            dataframes_to_send.append(data_dfx)

        if classifier == 'mlp':
            MLP_acc, MLP_prec, MLP_rec = Fed_MLP(dataframes_to_send, communication_iterations=comm_iter)
            print(f'MLP_acc: {MLP_acc}, MLP_prec: {MLP_prec}, MLP_rec: {MLP_rec}')
            return 'mlp', MLP_acc, MLP_prec, MLP_rec, a, p, r, ftrs_returned_by_lfs

    elif obj == 'multi':
        # Multi-Objective ftr sel
        print('MULTI-OBJECTIVE GLOBAL FTR SELECTION....')
        feature_list, num_avbl_ftrs = global_feature_select(lftr, num_ftr)
        print('feature list: ', feature_list)
        print('number of features: ', len(feature_list))

        dataframes_to_send = []
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            df1 = data_dfx.iloc[:, -1]
            data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
            data_dfx = data_dfx.assign(Class = df1)
            dataframes_to_send.append(data_dfx)
        if classifier == 'mlp':
            MLP_acc, MLP_prec, MLP_rec = Fed_MLP(dataframes_to_send, communication_iterations=comm_iter)
            print(f'MLP_acc: {MLP_acc}, MLP_prec: {MLP_prec}, MLP_rec: {MLP_rec}')
            return 'mlp', MLP_acc, MLP_prec, MLP_rec, a, p, r, ftrs_returned_by_lfs


    elif obj == 'anova':
        # ANOVA
        print('ANOVA....')
        f_selector = SelectKBest(score_func=f_classif, k=num_ftr)

        df_list = horz_data_divn(dset['dataset'], n_client)

        new_list = []
        for df in df_list:
            df = df.reset_index(drop = True)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            X_selected = f_selector.fit_transform(X, y)
            feature_indices = f_selector.get_support(indices=True)
            selected_feature_names = X.columns[feature_indices]
            df = pd.DataFrame(X)
            df = df[df.columns.intersection(selected_feature_names)]
            df = df.assign(Class = y)
            new_list.append(df)

        if classifier == 'mlp':
            MLP_acc, MLP_prec, MLP_rec = Fed_MLP(new_list, communication_iterations=comm_iter)
            print(f'MLP_acc: {MLP_acc}, MLP_prec: {MLP_prec}, MLP_rec: {MLP_rec}')
            return 'mlp', MLP_acc, MLP_prec, MLP_rec, a, p, r, ftrs_returned_by_lfs


    elif obj == 'rfe':
        # RFE
        print('RFE....')
        estimator = RandomForestClassifier()
        rfe = RFE(estimator, n_features_to_select=num_ftr)

        df_list = horz_data_divn(dset['dataset'], n_client)
        new_list = []
        for df in df_list:
            df = df.reset_index(drop = True)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            X = rfe.fit_transform(X, y)
            df = pd.DataFrame(X)
            df = df.assign(Class = y)
            new_list.append(df)

        if classifier == 'mlp':
            MLP_acc, MLP_prec, MLP_rec = Fed_MLP(new_list, communication_iterations=comm_iter)
            print(f'MLP_acc: {MLP_acc}, MLP_prec: {MLP_prec}, MLP_rec: {MLP_rec}')
            return 'mlp', MLP_acc, MLP_prec, MLP_rec, a, p, r, ftrs_returned_by_lfs
        
    else:
        print('NoFS')
        df_list = horz_data_divn(dset['dataset'], n_client)
        
        if classifier == 'mlp':
            MLP_acc, MLP_prec, MLP_rec = Fed_MLP(df_list, communication_iterations=comm_iter)
            print(f'MLP_acc: {MLP_acc}, MLP_prec: {MLP_prec}, MLP_rec: {MLP_rec}')
            return 'mlp', MLP_acc, MLP_prec, MLP_rec, a, p, r, ftrs_returned_by_lfs


def main(dataset, num_ftr, dset, comm_iter):

    global max_MLP
    global obj
    global classifier
    global ff_list
    global mlp_list

    print('Dataset: ', dataset)

    FCMI_clust_num = '2'
    FFMI_clust_num = '2'
    dataset = dataset
    cli_num = '5'

    curr_dir = os.getcwd()
    print(curr_dir)

    n_clust_fcmi = int(FCMI_clust_num)
    n_clust_ffmi = int(FFMI_clust_num)
    n_client = int(cli_num)

    name, acc, prec, rec, a, p, r, ftrs_returned_by_lfs = run_iid(n_client, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, dset, comm_iter)

    return name, acc, prec, rec, a, p, r, ftrs_returned_by_lfs

if __name__ == "__main__":

    for _, dset in datasets.iterrows():     # for each dataset

        for obj in obj_list:
            ds = []
            rng = list(range(dset['lb'], dset['ub']+1, dset['step']))

            if (dset['ub'])%(dset['step']) != 0:
                rng.append(dset['ub'])

            row_count = 5

            try:
                with open("_".join([classifier, obj, dset['dataset'], 'cache.csv']), "r") as f:
                    reader = csv.reader(f)
                    row_count = sum(1 for row in reader)
            except:
                row_count = 5
            
            for comm_iter in range((row_count//(dset['ub']-dset['lb']+1)+5), comm_iters+1):           # for each comm_iter
                # sfavsf = input('Press any key to proceed')
                lftr = []
                num_ftr_accuracies = []
                num_ftr_precs = []
                num_ftr_recs = []

                for num_ftr in rng:                   # for each number of features

                    name, acc, prec, rec, a, p, r, ftrs_returned_by_lfs = main(dset['dataset'], num_ftr, dset, comm_iter)
                    num_ftr_accuracies.append(acc)
                    num_ftr_precs.append(prec)
                    num_ftr_recs.append(rec)

                with open("_".join([classifier, obj, dset['dataset'], 'cache.csv']), "a", newline="") as f:
                    writer = csv.writer(f)

                    for i in range(len(num_ftr_accuracies)):
                        row_data = [comm_iter, num_ftr_accuracies[i], num_ftr_precs[i], num_ftr_recs[i]]
                        writer.writerow(row_data)
                        
                del(num_ftr_accuracies)
                del(num_ftr_precs)
                del(num_ftr_recs)
