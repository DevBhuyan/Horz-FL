#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 01:13:48 2023

@author: dev
"""


from sota import far
import os
from horz_data_divn import horz_data_divn
from ff import ff
from Fed_MLP import Fed_MLP
import pandas as pd
from normalize import normalize
from sys import exit
import numpy as np
from datetime import datetime
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from randomforest import rf
import csv
import gc
gc.enable()

dataset_list = [
                ['ionosphere', 5, 33, 1],
                # ['wdbc', 5, 31, 1],
                # ['vowel', 2, 12, 1],
                # ['wine', 5, 13, 1], --------DONE
                # ['hillvalley', 5, 100, 5],
                # ['vehicle', 2, 8, 1],
                # ['segmentation', 2, 9, 1],
                # ['nsl', 5, 38, 5],
                # ['isolet', 80, 617, 80],
                # ['ac', 5, 29, 1],
                # ['TOX-171', 500, 5748, 500],
                # ['iot', 5, 28, 1],
                # ['diabetes', 2, 8, 1],
                # ['automobile', 5, 19, 1]
                ]
datasets = pd.DataFrame(dataset_list, columns = ['dataset', 'lb', 'ub', 'step'])

df_list = []
obj = 'sota'
classifier = 'randomforest'

def run_iid(n_client, n_clust_fcmi, n_clust_ffmi, dataset, dset, thresh):

    global df_list
    global obj
    global classifier

    if obj == 'sota':
        print('SOTA ALGORITHM....')
        dataframes_to_send = far(dataset, n_client, thresh)
        num_ftr_returned = len(dataframes_to_send[0].columns) - 1

        if classifier == 'ff':
            ff_acc, ff_prec, ff_rec = ff(dataframes_to_send)
            print(f'ff_acc: {ff_acc}, ff_prec: {ff_prec}, ff_rec: {ff_rec}, num_ftr_returned: {num_ftr_returned}')
            return 'ff', ff_acc, ff_prec, ff_rec, num_ftr_returned
        elif classifier == 'mlp':
            MLP_acc, MLP_prec, MLP_rec = Fed_MLP(dataframes_to_send)
            print(f'MLP_acc: {MLP_acc}, MLP_prec: {MLP_prec}, MLP_rec: {MLP_rec}, num_ftr_returned: {num_ftr_returned}')
            return 'mlp', MLP_acc, MLP_prec, MLP_rec, num_ftr_returned
        elif classifier == 'randomforest':
            rf_acc, rf_prec, rf_rec = rf(dataframes_to_send)
            print(f'rf_acc: {rf_acc}, rf_prec: {rf_prec}, rf_rec: {rf_rec}, num_ftr_returned: {num_ftr_returned}')
            return 'randomforest', rf_acc, rf_prec, rf_rec, num_ftr_returned



def main(dataset, dset, thresh):

    global obj
    global classifier


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

    name, acc, prec, rec, num_ftr_returned = run_iid(n_client, n_clust_fcmi, n_clust_ffmi, dataset, dset, thresh)

    return name, acc, prec, rec, num_ftr_returned

if __name__ == "__main__":

    for _, dset in datasets.iterrows():     # for each dataset

        ds = []

        for thresh in range(10):

            name, acc, prec, rec, num_ftr_returned = main(dset['dataset'], dset, float(thresh)/10.0)

            ds.append([num_ftr_returned, acc, prec, rec])

        dataset_df = pd.DataFrame(ds)
        dataset_df.columns = ['num_ftr', 'acc', 'prec', 'rec']

        dataset_df.to_csv(dset['dataset']+'_'+obj+'_Fed'+classifier+'_'+str(datetime.now())+'.csv')