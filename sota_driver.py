#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 01:13:48 2023

@author: dev
"""


from sota import far
import os
from ff import ff
from Fed_MLP import Fed_MLP
import pandas as pd
from datetime import datetime
from randomforest import rf
from horz_data_divn import horz_data_divn
import gc
gc.enable()

dataset_list = [
                # ['ionosphere', 5, 33, 1],
                # ['wdbc', 5, 31, 1],
                ['vowel', 2, 12, 1],
                ['wine', 5, 13, 1],
                # ['hillvalley', 5, 100, 5],
                # ['vehicle', 2, 8, 1],
                # ['segmentation', 2, 9, 1],
                # ['nsl', 5, 38, 5],
                # ['isolet', 80, 617, 80],
                # ['ac', 5, 29, 1],
                # ['iot', 5, 28, 1],
                # ['diabetes', 2, 8, 1]
                ]
datasets = pd.DataFrame(dataset_list, columns = ['dataset', 'lb', 'ub', 'step'])

df_list = []
obj = 'sota'

def run_iid(dataset : str, 
            ub : int, 
            n_client = 5):
    '''
    Driver function for FSHFL implementation

    Parameters
    ----------
    dataset : str
    ub : int
        upper bound of number of features.
    n_client : int, optional
        number of clients. The default is 5.

    Returns
    -------
    None.

    '''
    global df_list
    global obj

    df_list = horz_data_divn(dataset, n_client)

    if obj == 'sota':
        print('SOTA ALGORITHM....')
        dataframes_to_send, thresh = far(df_list)
        num_ftr_returned = len(dataframes_to_send[0].columns) - 1

        for classifier in ['ff', 'randomforest', 'mlp']:

            print()
            print("\033[1;32m" + f'Learning on features using {classifier} at {n_client} clients' + "\033[0m")

            if classifier == 'ff':
                acc, prec, rec = ff(dataframes_to_send)
                print(f'threshold: {thresh}, ff_acc: {acc}, ff_prec: {prec}, ff_rec: {rec}, num_ftr_returned: {num_ftr_returned}')

            elif classifier == 'mlp':
                acc, prec, rec = Fed_MLP(dataframes_to_send)
                print(f'threshold: {thresh}, MLP_acc: {acc}, MLP_prec: {prec}, MLP_rec: {rec}, num_ftr_returned: {num_ftr_returned}')

            elif classifier == 'randomforest':
                acc, prec, rec = rf(dataframes_to_send)
                print(f'threshold: {thresh}, rf_acc: {acc}, rf_prec: {prec}, rf_rec: {rec}, num_ftr_returned: {num_ftr_returned}')


            ds = [num_ftr_returned, acc, prec, rec]

            dataset_df = pd.DataFrame(ds)
            dataset_df = dataset_df.T
            dataset_df.columns = ['num_ftr', 'acc', 'prec', 'rec']

            dataset_df.to_csv(dset['dataset']+'_'+obj+'_Fed'+classifier+'_'+str(datetime.now())+'.csv')

    return


def main(dataset, ub):

    global obj
    global classifier

    print('Dataset: ', dataset)

    curr_dir = os.getcwd()
    print(curr_dir)

    run_iid(dataset, ub)

    return

if __name__ == "__main__":

    for _, dset in datasets.iterrows():     # for each dataset

        main(dset['dataset'], dset['ub'])
