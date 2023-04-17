#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:59:57 2023

@author: dev
"""

def normalize(dlist):
    cols = []
    dnew = []
    for df in dlist:
        df = df.loc[:, ~df.columns.duplicated()]
        cols.append(df.columns)
    cols_sets = [set(sublist) for sublist in cols]
    common = list(set(list(set.intersection(*cols_sets))))
    for df in dlist:
        df = df.loc[:, ~df.columns.duplicated()]
        df = df[df.columns.intersection(common)]
        dnew.append(df)
    return dnew