#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 01:25:06 2023

@author: dev
"""

from horz_data_divn import horz_data_divn
import pandas as pd
from datetime import datetime
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

def rfe(df_list : list, 
        num_ftr : int):
    '''
    Wrapper function for RFE FS method

    Parameters
    ----------
    df_list : list
    num_ftr : int

    Returns
    -------
    new_list : TYPE
        new list of reduced dataframes.

    '''
    start = datetime.now()

    estimator = RandomForestClassifier()
    rfe = RFE(estimator, n_features_to_select=num_ftr)

    new_list = []
    for df in tqdm(df_list, total=len(df_list)):
        df = df.reset_index(drop = True)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X = rfe.fit_transform(X, y)
        df = pd.DataFrame(X)
        df = df.assign(Class = y)
        new_list.append(df)

    print("\033[1;33m" + f'Total learning time : {(datetime.now()-start)/len(df_list)}' + "\033[0m")
    return new_list
