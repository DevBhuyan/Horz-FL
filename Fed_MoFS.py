#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 02:37:27 2023

@author: dev
"""

from horz_data_divn import horz_data_divn
from local_feature_select import local_fs
from datetime import datetime
from global_feature_select import global_feature_select
from tqdm import tqdm

def fedmofs(df_list : list, 
            n_clust_fcmi : int, 
            n_clust_ffmi : int, 
            num_ftr=None):
    '''
    Wrapper function for entire Fed-MoFS algorithm

    Parameters
    ----------
    df_list : list
        list of dataframes.
    n_clust_fcmi : int
        number of fcmi clusters.
    n_clust_ffmi : int
        number of ffmi clusters.
    num_ftr : TYPE, optional
        number of features. The default is None.

    Returns
    -------
    dataframes_to_send : list
        list of reduced dataframes.

    '''
    local_feature = []

    print('Initializing Local FS....')
    start = datetime.now()
    for df in tqdm(df_list, total=len(df_list)):
        local_feature.append(local_fs(df, n_clust_fcmi, n_clust_ffmi)[0])
    print("\033[1;33m" + f'Average learning time per client : {(datetime.now()-start)/len(df_list)}' + "\033[0m")
    start = datetime.now()

    print('Initializing Global FS....')
    feature_list, num_avbl_ftrs = global_feature_select(local_feature, num_ftr)
    print("\033[1;33m" + f'Found {len(feature_list)} global features: {feature_list}' + "\033[0m")

    dataframes_to_send = []
    for df in tqdm(df_list, total=len(df_list)):
        y = df.iloc[:, -1]
        df = df[df.columns.intersection(feature_list)]
        df = df.assign(Class = y)
        dataframes_to_send.append(df)

    print("\033[1;33m" + f'Learning time for server : {(datetime.now()-start)/len(df_list)}' + "\033[0m")

    return dataframes_to_send
