#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 23:40:28 2023

@author: dev
"""

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from horz_data_divn import horz_data_divn
from ff import ff
import pandas as pd

n_client = 5
dataset_list = [
                # ['ionosphere', 5, 33, 1], 
                # ['musk', 5, 169, 10], 
                # ['wdbc', 5, 31, 1], 
                # ['vowel', 2, 14, 1], 
                # ['wine', 5, 13, 1], 
                ['hillvalley', 5, 100, 5],
                # ['vehicle', 2, 9, 1],
                # ['segmentation', 2, 9, 1],
                # ['ac', 5, 30, 1], 
                # ['nsl', 5, 41, 1], 
                # ['isolet', 80, 617, 80], 
                # ['TOX-171', 500, 5748, 500],
                # ['iot', 5, 28, 1],
                ['diabetes', 2, 8, 1],
                # ['automobile', 5, 19, 1]
                ]
datasets = pd.DataFrame(dataset_list, columns = ['dataset', 'lb', 'ub', 'step'])


summary = []
for _, dset in datasets.iterrows():
    rng = list(range(dset['lb'], dset['ub']+1, dset['step']))
    
    if (dset['ub'])%(dset['step']) != 0:
        rng.append(dset['ub'])
        
    a = []
    p = []
    r = []
    f = []
    for num_ftr in rng:
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
        
        ff_acc, ff_prec, ff_rec, ff_f1 = ff(new_list)
        a.append(ff_acc)
        p.append(ff_prec)
        r.append(ff_rec)
        f.append(ff_f1)
        
        print(f'ff_acc: {ff_acc}\n ff_prec: {ff_prec}\n ff_rec: {ff_rec}\n ff_f1: {ff_f1}')
    summary.append([dset['dataset'], a, p, r, f])